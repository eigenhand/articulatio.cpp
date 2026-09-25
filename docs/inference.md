# How inference works

This document follows the path from a piece of text to playable audio the way
this code actually takes it, with pointers to the relevant places in the
source. It complements [`architecture.md`](architecture.md), which describes
the model architecture; this one is about the **flow**, and about the
extensions this fork adds.

---

## 1. Overview

```
Text ──► Gemma BPE ──► T5Gemma2 encoder ──► projection 1152→2048 ──┐
                                                                   │
Reference ──► Mimi encoder ──► codes [T,16] ──► audio embedding ───┤
                                                                   ▼
                                                        Qwen3 backbone
                                                                   │
                                            codebook 0  ┌──────────┘
                                                        ▼
                                              depth decoder (15 steps)
                                                        │ codebooks 1..15
                                                        ▼
                                                    vocoder
                                                        ▼
                                              24 kHz PCM, 16 bit mono
```

One backbone step produces **one frame**. A frame consists of 16 codebook
entries and becomes **1920 samples**, exactly 80 ms at 24 kHz, or
**12.5 frames per second**. That number sets the pace for everything else; it
lives in `config.h` as `samples_per_frame`.

### The four stages and their size

Measured on the Q8_0 GGUF (sum of the tensors per prefix):

| Stage | Parameters | Share | Job |
|---|---|---|---|
| Backbone (Qwen3, 28 layers, 2048) | 1.41 B | 46.9 % | produces codebook 0 per frame |
| Text encoder (T5Gemma2, 26 layers, 1152) | 1.00 B | 33.3 % | understands the text |
| Depth decoder (12 layers, 1024) | 367 M | 12.2 % | fills in codebooks 1..15 |
| Codec (encoder + decoder) | 162 M | 5.4 % | audio ↔ codes |
| Audio embedding (shared table) | 67 M | 2.2 % | codes → vectors |
| **Total** | **3.01 B** | | |

---

## 2. What the prompt looks like

The decisive part is in `build_segments()` (`src/generation.cpp`). For a
cloned voice it comes out as:

```
[S0]<reference text>  │  <reference audio codes>  │  [S0]<ins_bos>instruction<ins_eos><text>
└─────── only when there is a reference ───────┘     └────────────── always ───────────────┘
```

Three observations that are easy to miss:

1. **The reference clip sits in the prompt as audio**, not as a speaker
   vector. The model has no notion of a "speaker ID"; it simply continues what
   it hears. That is also why background noise in the reference carries over.
2. **Without a reference** ("voice design") the first block is dropped, and
   the instruction alone determines the voice. The first generated piece then
   becomes the reference for all the following ones, see `GenSession::speak`.
3. `assemble()` sends text segments through the text encoder and audio
   segments through the embedding table, and simply concatenates the results.
   Both end up in the same 2048-dimensional space.

---

## 3. A frame in detail

The loop is in `generate_chunk()`:

```cpp
for (int step = 0; step < max_new; step++) {
    if (cb0 == backbone_eos_token_id) break;

    depth_codes = depth.run(m, hiddens, cb0, ...);   // 15 steps
    frame = { cb0, depth_codes... };                 // 16 entries

    ae   = audio_embed_forward(m, frame, 1);         // frame → vector
    o_c  = backbone_run(m, st_c, ae, 1);             // one step further
    cb0  = sample_token(o_c.logits, ...);            // next codebook 0
}
```

**The depth decoder is the expensive part.** It runs 15 sequential steps per
frame (`for (int j = 1; j < nc; j++)` in `depth_decoder.cpp`), and its KV
cache is **reset** at the start of every frame (`kv.reset()`). So it carries
no context across frames; only the backbone does that.

Measured on an RTX 4070 (Vulkan, Q8_0), warm:

| Stage | per frame |
|---|---|
| Backbone | ~6.0 ms |
| **Depth decoder** | **~28 ms** |
| Vocoder | ~6.0 ms |
| Total | ~40 ms for 80 ms of audio → **~2x realtime** |

The depth decoder accounts for about 70 % of the compute time even though it
holds only 12 % of the parameters, because it runs 15 times per frame.

**Tokens 2048–2050 are suppressed during sampling**, 2051 ends the utterance.
So the backbone can only draw valid codes.

---

## 4. Why generation happens in pieces

Long text is split into pieces of about `split_chars` characters
(`generate()` → `split_text()` → `GenSession::speak()` per piece). The reason
is **not** memory, but that the model loses the thread on long text: it skips
or repeats sentences. `split_chars 0` turns the splitting off.

The price of splitting: at every seam a new generation starts, with its own
sampling. Delivery, pitch and tempo can wander. This fork offers three levels
against that:

### a) Fixed reference (default)

Every piece gets the same reference clip in its prompt. The speaker identity
holds, but the delivery can still wander. On top of that, the seed
deliberately moves along: `seed + m_piece`.

### b) `rolling_anchor`

Instead of the original clip, the **previously generated piece** becomes the
reference:

```cpp
const bool roll = (m_req.rolling_anchor && !m_req.carry_cache) || m_codes.empty();
if (ok && roll && made.n_frames > 0) { m_codes = std::move(made.codes); ... }
```

The layout stays the trained `[text][audio][text]`, and it costs nothing: the
piece's codes are already there, it is a `std::move`. The risk is cumulative
drift away from the starting voice over many pieces.

### c) `carry_cache`

The backbone's KV cache **lives on across piece boundaries**. The next piece
only appends its new text; the reference and everything spoken so far are
already in the cache:

```cpp
const bool ref_in_prompt = has_ref && !resume;   // no reference again when resuming
```

This is real continuation instead of a swapped reference. It shows in the
prompt length:

```
normal / rolling_anchor:  prompt 78   prefill 133 ms
carry_cache:              prompt 41   prefill  71 ms   ← half the prompt
```

**This is experimental.** The layout `[reference][text1][audio1][text2]` never
occurred like this in training. The cache also costs memory: `cache_seq`
positions take `28 × 2 × 128 × 8 × cache_seq × 4` bytes, so at 2048 that is
**448 MiB** (about 470 MB). With CFG (`cfg_scale` other than 1) there is a
second cache of the same size. If it does not fit, the session falls back to
a cache per piece instead of crashing.

**Every piece ends with an EOS frame in the cache** (`carry_eos`, on by
default). In training, every audio span is followed by one, the reference
included, see `assemble()`. The generation loop, however, stops at the
*sampled* EOS without feeding it in. Earlier versions of this fork therefore
attached the next text directly to the last audio frame: `[text1][audio1][text2]`
instead of `[text1][audio1][EOS][text2]`. The model then took the previous
piece as unfinished; if it had left out a sentence there, it made up for it in
the next piece, heard as a shuffled order ("first sentence, then the third,
then the second"). Now the EOS frame is run into the cache after every piece
(one step, ~6 ms).

The EOS frame on its own, though, only made swallowed sentences visible
instead of fixing them: before, they came later; now they were missing. Two
more places, both independent of the mode:

- **Piece splitting on the WebSocket (`drain`)**: every sentence becomes a
  piece of its own. Previously everything up to the last sentence end went out
  as *one* piece, and out of several sentences the model liked to drop the
  last one. Sentences under 20 letters ("Sure!", "Exactly.") are not spoken
  alone but wait for the next one: `Sure! Here are a few tips …`. Only at the
  end of the utterance does a short remainder go out alone.
- **Minimum length per piece**: EOS is blocked for the first
  `max(4, letters/2)` steps. Speech takes a bit more than one frame per
  letter, so the block only kicks in when the model wants to cut a piece off
  right away.

The cache has a fixed size, and `cache_append` does not check any bounds. That
is why a session starts over from the reference when the next piece might no
longer fit safely, and why the loop ends a piece before it would write past
the end.

---

## 5. How streaming works

The vocoder is not called once at the end but continuously. The `flush`
lambda in `generate_chunk()`:

```cpp
const int ctx = m.cfg.voc.sliding_window + 16;       // look-back for the vocoder
const int ctx_start = start > ctx ? start - ctx : 0;
audio = codec.decode(sub, sub_T);
fx.push(audio.data() + skip, count * spf);           // skip drops the lead-in
```

(`fx` is the post-processing stage from section 6, which passes the samples on
to the caller's callback.)

Two tricks are in there:

**Look-back.** Decoding starts at `start - ctx`, output only at `start`. That
gives the vocoder's causal convolutions and sliding window some context, and
no clicks appear at the chunk boundaries.

**Growing chunks.** `chunk` starts at `chunk_first` (4 frames = 320 ms) and
grows with `chunk = min(chunk + chunk/3 + 1, chunk_max)`. The first chunk
comes early, later ones get larger and therefore more efficient. Hence
~400 ms to the first sound, even though the vocoder itself only needs 6 ms per
frame.

**Across piece boundaries** the look-back is handed on as well (`m_voc_tail`
in `GenSession`). Without it the vocoder starts cold at every piece boundary,
which is measurable in the jumps between adjacent samples (the largest jump,
and how many jumps exceed 4000 and 6000):

| | max. jump | >4000 | >6000 |
|---|---|---|---|
| without hand-over | 6963 | 203 | 13 |
| with hand-over | 6104 | 25 | 1 |

---

## 6. Post-processing

`PostFx` (`src/generation.cpp`) sits between the vocoder and the caller. The
order is: **stretch first, then trim**, so the milliseconds refer to what is
heard in the end.

**`speed`** stretches time with WSOLA. Resampling would be two lines, but
would shift the pitch along with it. For every output window WSOLA searches
the input for the spot that best continues what has been output so far, and
crossfades into it; the waveform is copied locally unchanged, so the pitch
stays. Checked with a fixed seed:

| Rate | Duration | F0 | with resampling it would be |
|---|---|---|---|
| 0.8x | 4.19 s | 146.3 Hz | 117.8 Hz |
| 1.0x | 3.36 s | 147.2 Hz | 147.2 Hz |
| 1.3x | 2.58 s | 146.3 Hz | 191.4 Hz |

Usable from about 0.7 to 1.5; beyond that the processing becomes audible.

**`trim_head_ms` / `trim_tail_ms`** cut off the start and the end of every
piece. The tail has to be held back for that, since only at the end of a piece
is it known which samples are the last ones.

**`pause_ms`** puts that much silence between two pieces, none before the
first piece of a session. On the WebSocket every sentence is a piece, so this
is the pause between sentences. It goes into the stream as real samples, so
every client gets it.

---

## 7. Interfaces

The parameters are available on all three paths, except where the table shows
a dash (—). Precedence: the value in the request, else the server default,
else the default in the core.

| Parameter | CLI | HTTP form field | WebSocket `start` |
|---|---|---|---|
| Voice | `--voice` | `voice_id` | `voice_id` |
| Piece size | `--split-chars` | `split_chars` | `split_chars` |
| Rolling anchor | `--rolling-anchor` | `rolling_anchor=1` | `"rolling_anchor":1` |
| Carry the cache | `--carry-cache` | `carry_cache=1` | `"carry_cache":1` |
| EOS frame after each carried piece (on) | — | `carry_eos=0` turns it off | `"carry_eos":0` turns it off |
| Cache size | — | `cache_seq` | `"cache_seq"` |
| Speech rate | `--speed` | `speed` | `"speed"` |
| Trim start/end | `--trim-head-ms` / `--trim-tail-ms` | `trim_head_ms` / `trim_tail_ms` | `"trim_head_ms"` / `"trim_tail_ms"` |
| Pause between pieces | `--pause-ms` | `pause_ms` | `"pause_ms"` |

Server-wide defaults are set when starting `breeze-server`: `--speed`,
`--trim-head-ms`, `--trim-tail-ms`, `--pause-ms`, `--split-chars`,
`--chunk-first`, `--chunk-max`.

On the WebSocket the switches are numbers (`1` / `0`). JSON `true` / `false`
is not understood there, and the field keeps its default.

The CPU thread count applies to the whole process and only to the CPU backend:
`--threads` in the CLI, or the `BREEZE_THREADS` environment variable for the
CLI and the server alike (default 4, ggml's own default).

The WebSocket (port HTTP+1) is meant for text that arrives **bit by bit**, for
example from an LLM token stream. Send `start` once, then any number of
`text` messages, and `end` at the end. The server speaks whole sentences as
soon as they are complete; there is no need to split anything yourself.

---

## 8. Pitfalls that cost us time

- **A failed memory allocation is fatal.** If
  `ggml_backend_alloc_ctx_tensors` fails, the buffer is null and the next
  access ends the process via `GGML_ASSERT` in `ggml_view_3d`. That is why
  `BackboneState::ok()` now checks for it.
- **`\uXXXX` in JSON was dropped.** The WebSocket parser silently skipped the
  digits; any client that escapes non-ASCII (Python's `json.dumps` does by
  default) lost every non-ASCII character that way, including ä, ö, ü and
  ß. Fixed in `json_str()`, surrogate pairs included.
- **The `design` log label lies for saved voices.** It depends on
  `ref_audio`, i.e. an uploaded WAV; with `voice_id` the codes land in
  `ref_codes`, and it still says "design". The voice is cloned all the same.
- **Instructions on a cloned voice need guidance.** A delivery instruction
  ("a gravelly old pirate captain") on a saved German voice made no audible
  difference at the default `cfg_scale` 1 or at 2; at 3 it came through. The
  voice keeps its timbre either way; only the delivery changes. The German
  fine-tune weakened this considerably: instructions take effect much less
  strongly than with the base model. Voice design without a reference
  responds at 1 already. Guidance costs a second pass, and with `carry_cache`
  a second cache (section 4c).
- **HTTP answers `409 busy` while a generation is running**, whereas the
  WebSocket queues the request (`queued`). Over HTTP you need a retry.
- **CUDA is slower than Vulkan here**, see [`build.md`](build.md). Generation
  consists of thousands of small graphs, each in a fresh context; CUDA graph
  capture never kicks in.

---

## 9. License

The source code is under Apache 2.0. **The model weights are not**: they are
under the *BreezeBlue Research and Non-Commercial License*. Fine-tunes, LoRA
adapters, merged weights and quantized GGUFs count as derivative models and
fall under the same license. Generated audio is not a derivative model, but
the license's use restrictions still apply to it, e.g. no commercial use.
Fine for research and private use, not for commercial use.
See <https://huggingface.co/BreezeBlue/Breeze-TTS-2>.

Reference clips and cloned voices require the consent of the person speaking.
