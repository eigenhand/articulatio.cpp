<div align="center">

# articulatio.cpp

*A fork of [Breeze-TTS-2.cpp](https://github.com/HoppouAI/Breeze-TTS-2.cpp) for Articulatio-DE (German).*

**Real-time German text-to-speech in C++ and GGUF.**
Streams the first audio after about half a second and speaks faster than real time, with cloned
German as intelligible as real recordings.

<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-1f6feb?style=for-the-badge" alt="License"></a>
<img src="https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++17">
<img src="https://img.shields.io/badge/Vulkan-AC162C?style=for-the-badge&logo=vulkan&logoColor=white" alt="Vulkan">
<a href="https://huggingface.co/eigenhand/Articulatio-DE-GGUF"><img src="https://img.shields.io/badge/Articulatio--DE_GGUF-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Articulatio-DE GGUF weights"></a>

</div>

A C++ / GGUF reimplementation of [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2),
running on [ggml](https://github.com/ggml-org/ggml). The Vulkan backend means it works across NVIDIA,
AMD and Intel GPUs, and it falls back to CPU. Ships a CLI, a streaming HTTP and WebSocket server with
a web UI, and a plain C shared library for bindings.

With the base model it speaks English and Mandarin at 24 kHz, roughly 1.2x realtime at Q8_0 on an
RTX 3060 (upstream measurement).

## Articulatio-DE at a glance

[Articulatio-DE](https://huggingface.co/eigenhand/Articulatio-DE) is a German (and English) fine-tune
of Breeze TTS 2; this fork is built to serve it.

| | |
| --- | --- |
| Word error rate, cloned German voices | 4.2 % (the real recordings of the same sentences: 4.4 %) |
| Speed, Q8_0 with Vulkan | RTF 0.49 on an RTX 4070, 0.69 on an AMD Radeon AI PRO R9700 (below 1 is faster than real time) |
| First audio when streaming | about half a second |
| VRAM | about 4 GB for Q8_0, 3 GB for Q4_K |

Method and all measurements: [articulatio-training/docs/RESULTS.md](https://github.com/eigenhand/articulatio-training/blob/main/docs/RESULTS.md).
Chinese no longer works with Articulatio-DE, and vocal events are largely lost (see *Vocal events*).

### Quick start (German)

```bash
hf download eigenhand/Articulatio-DE-GGUF articulatio-de-q8_0.gguf --local-dir models
build/breeze-server models/articulatio-de-q8_0.gguf --port 8080 --webui --split-chars 120 --pause-ms 150
```

Open http://localhost:8080/ for the web UI. Keep `--split-chars` small: in one long piece the model
drifts after about 20 seconds, sentence by sentence it stays stable. The WebSocket API splits into
sentences by itself. For text with digits, units or abbreviations, put the German text frontend
(`tools/german-frontend`) in front of the server.

## Changes in this fork

This is a fork of [HoppouAI/Breeze-TTS-2.cpp](https://github.com/HoppouAI/Breeze-TTS-2.cpp) that serves
Articulatio-DE through the streaming server. It adds:

- **Carrying the KV cache across pieces** (`carry_cache`, experimental, off by default). Later pieces
  only append their new text to the backbone's cache instead of prefilling the reference and prompt
  again, so the model continues from its own audio. Each piece is closed in the cache with an EOS
  frame, as every audio span is in training (`carry_eos`, on by default). The cache is sized once per
  session (`cache_seq`, default 2048 positions, about 470 MB per cache). A capacity guard starts over
  from the reference when the next piece might not fit and ends a piece before it would write past
  the end, and if the allocation fails the session falls back to a cache per piece instead of
  aborting.
- **Rolling anchor** (`rolling_anchor`, off by default): each piece uses the previously generated
  piece as its reference, rather than the original reference clip.
- **One piece per sentence on the WebSocket.** Text is no longer sent as one piece up to the last
  sentence end; every sentence becomes its own piece, and sentences shorter than 20 letters are merged
  into the next one.
- **Minimum length before EOS.** EOS is blocked for the first `max(4, letters/2)` steps of a piece, so
  the model cannot swallow a short sentence by ending it right away.
- **Vocoder context across piece boundaries.** The last frames of the previous piece are decoded as a
  lead-in that is never output, so the vocoder does not start cold (and click) at every piece.
- **Post-processing:** `speed` changes the speech rate by WSOLA time stretching with the pitch
  preserved, `trim_head_ms` / `trim_tail_ms` cut the start and end of every piece, and `pause_ms`
  inserts silence between pieces (on the WebSocket, between sentences).
- **JSON `\uXXXX` decoding fix.** The WebSocket's JSON reader used to drop `\uXXXX` escapes, so clients
  that escape non-ASCII (Python's `json.dumps` does by default) lost every non-ASCII character, such
  as the German ä, ö, ü and ß. It now decodes them, surrogate pairs included.
- **German text frontend** (`tools/german-frontend`, Python): because the model reads digits and
  symbols unreliably, it writes out numbers, ordinals, dates, units, abbreviations and simple formulas
  in German and removes Markdown and emoji, as a library or as a proxy in front of the server.
- **CPU thread count override:** `BREEZE_THREADS` (or `--threads` in the CLI) replaces ggml's default
  of 4 threads on the CPU backend.
- **New flags and fields:** `breeze-cli` gains `--rolling-anchor`, `--carry-cache`, `--threads`,
  `--speed`, `--trim-head-ms`, `--trim-tail-ms` and `--pause-ms`. `breeze-server` gains `--speed`,
  `--trim-head-ms`, `--trim-tail-ms` and `--pause-ms` as server-wide defaults. HTTP requests and the
  WebSocket `start` message accept `rolling_anchor`, `carry_cache`, `carry_eos`, `cache_seq`, `speed`,
  `trim_head_ms`, `trim_tail_ms` and `pause_ms`.

The new options default to upstream behavior; the per-sentence pieces on the WebSocket, the minimum
length before EOS and the vocoder context across pieces are always on. The C API does not expose the
new options. [docs/inference.md](docs/inference.md) walks through the inference flow and these
changes, with measurements.

## Demo

https://github.com/user-attachments/assets/aaee58ac-1228-48b6-b9f9-7ac4150b0338

13 s of German in a voice the model created itself, one sentence per piece, `speed` 1.25 with the pitch
preserved and 150 ms between sentences: *"Guten Morgen! Heute ist Donnerstag, der fünfundzwanzigste
September. Draußen sind es achtzehn Grad, am Nachmittag zieht von Westen ein Gewitter auf. Vergiss also
den Regenschirm nicht, wenn du später noch zum Bahnhof fährst."*

Synthetic speech generated with Articulatio-DE (GGUF Q8_0, articulatio.cpp); the voice belongs to no real
person. Derived from Breeze TTS 2 by BreezeBlue and licensed for research and non-commercial use only: the
clip is an output of the model and falls under the
[BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE),
not under the Apache 2.0 license of this repository.

## Documentation

<div align="center">

<a href="docs/build.md"><img src="https://img.shields.io/badge/Build-2b3137?style=for-the-badge" alt="Build"></a>
<a href="docs/models.md"><img src="https://img.shields.io/badge/Models-2b3137?style=for-the-badge" alt="Models"></a>
<a href="docs/cli.md"><img src="https://img.shields.io/badge/CLI-2b3137?style=for-the-badge" alt="CLI"></a>
<a href="docs/server.md"><img src="https://img.shields.io/badge/HTTP_Server-2b3137?style=for-the-badge" alt="Server"></a>
<a href="docs/websocket.md"><img src="https://img.shields.io/badge/WebSocket-2b3137?style=for-the-badge" alt="WebSocket"></a>
<br>
<a href="docs/voices.md"><img src="https://img.shields.io/badge/Saved_Voices-2b3137?style=for-the-badge" alt="Voices"></a>
<a href="docs/voice-conversion.md"><img src="https://img.shields.io/badge/Voice_Conversion-6e2b3a?style=for-the-badge" alt="Voice conversion"></a>
<a href="docs/c-api.md"><img src="https://img.shields.io/badge/C_API-2b3137?style=for-the-badge" alt="C API"></a>
<a href="docs/ctypes.md"><img src="https://img.shields.io/badge/Python-2b3137?style=for-the-badge" alt="Python"></a>
<a href="docs/architecture.md"><img src="https://img.shields.io/badge/Architecture-2b3137?style=for-the-badge" alt="Architecture"></a>

</div>

## What it does

| Mode | You give it | You get |
| --- | --- | --- |
| **Voice design** | A text description of a voice | A voice invented to match the description |
| **Voice cloning** | A reference clip and its transcript | New speech in that voice |
| **Voice direction** | A reference clip plus an instruction | That voice, steered in tone or pace |
| **Voice conversion** | A recording to respeak, and a target voice | The same performance in a different voice |

## Build

```
git clone --recursive https://github.com/eigenhand/articulatio.cpp.git
cd articulatio.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Add `-DBREEZE_VULKAN=OFF` for a build without Vulkan (CPU, or Metal on macOS). On Apple Silicon the
Metal path is slow (about RTF 4.4 on an M3 Pro); use [articulatio-mlx](https://github.com/eigenhand/articulatio-mlx)
there instead. Outputs are `breeze-cli`, `breeze-convert`,
`breeze-server`, `breeze-quantize` and the shared library. See [docs/build.md](docs/build.md).

### Nix

If you have [Nix](https://nixos.org/) with flake support, the dev shell provides all build
dependencies (CMake, Ninja, Vulkan SDK, Python for conversion) and a few convenience commands:

To enter the dev shell, run:

```
nix develop          # or: nix develop .#cpu
```

After entering the dev shell, run the following commands to build and run the CLI/server:

```
breeze-build         # configure + build (Vulkan)
breeze-build --cpu   # configure + build (CPU only)
breeze-cli           # run the CLI
breeze-server        # run the server
```

## Weights

<div align="center">

<a href="https://huggingface.co/HoppouAI/Breeze-TTS-2.cpp"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg.svg" alt="Model on Hugging Face"></a>

</div>

**Articulatio-DE** (German): Q8_0 and Q4_K GGUFs are at
[eigenhand/Articulatio-DE-GGUF](https://huggingface.co/eigenhand/Articulatio-DE-GGUF) and run with this
fork like the base model. Only these two variants are published; the table below lists the variants of
the base model.

Prebuilt GGUFs of the base model are on the Hub at
[HoppouAI/Breeze-TTS-2.cpp](https://huggingface.co/HoppouAI/Breeze-TTS-2.cpp), or convert them
yourself. The download must include the `audio_tokenizer/` directory, which holds the vocoder.

```
pip install -r scripts/requirements.txt
python scripts/convert_hf_to_gguf.py /path/to/Breeze-TTS-2 -o breeze-tts-2-f16.gguf --dtype f16
build/breeze-quantize breeze-tts-2-f16.gguf breeze-tts-2-q4_k.gguf q4_k
```

| Variant | Size | Approximate VRAM | Notes |
| --- | --- | --- | --- |
| F16 | 5.9 GB | ~7 GB | Reference quality |
| Q8_0 | 3.3 GB | ~4 GB | **Recommended** |
| Q6_K | 2.9 GB | ~3.5 GB | |
| Q4_K | 2.4 GB | ~3 GB | Smallest safe choice |
| Q8_0-dd4, Q8_0-dd2, Q4_K-dd2 | 2.3 to 3.4 GB | | Experimental, see below |

The `-dd` variants quantize the depth decoder as well, which the others leave at higher precision.
They can be meaningfully faster on hardware where the depth decoder is the bottleneck, since it runs
15 sequential steps for every single frame of audio. The catch is that depth codes feed back into the
backbone each frame, so error compounds with length: output holds up early and then drifts muffled and
thin past roughly 45 seconds of continuous generation. Fine for short lines, not for narration.

## CLI

```
# voice design
build/breeze-cli breeze-tts-2-q8_0.gguf \
  --text "(sigh) Welcome aboard. Your journey begins now." \
  --instruction "A warm, thoughtful young woman with a clear, calm delivery." \
  --output design.wav

# voice clone
build/breeze-cli breeze-tts-2-q8_0.gguf \
  --text "It is good to hear your voice again." \
  --ref-audio reference.wav --ref-text "Exact transcript of the reference." \
  --output clone.wav

# voice direction
build/breeze-cli breeze-tts-2-q8_0.gguf \
  --text "(clears throat) We need to talk." \
  --instruction "Speak slowly with a restrained, serious tone." \
  --ref-audio reference.wav --ref-text "Exact transcript of the reference." \
  --output direction.wav
```

Encoding a reference clip is the slowest part of a clone and it is deterministic, so save it once with
`--save-voice name` and use `--voice name` from then on. Saved voices load instantly, work in the CLI,
the server and the web UI, and cut time to first audio from around 900 ms to around 280 ms. See
[docs/voices.md](docs/voices.md).

## Vocal events

Inline tags in round brackets produce non speech sounds. `(laugh)`, `(sigh)`, `(cough)` and
`(clears throat)` are the reliable ones, with `[笑]` and `[叹气]` on the Chinese side, but the tag
vocabulary is **open**: the model was trained on descriptive tags rather than a fixed token list,
so things like `(whispering)`, `(gasp)` or `(nervous chuckle)` will often work.

The catch is that at the default `--cfg-scale 1.0` the model treats a tag as a suggestion and usually
ignores anything outside the common set. **Vocal events generally need `--cfg-scale 2` to `3` to
actually fire**, because guidance is what pushes the output away from plain neutral reading. Expect
some added harshness at that range, so raise it for lines that need the event and drop back to 1.0 for
ordinary speech.

```
build/breeze-cli breeze-tts-2-q8_0.gguf \
  --text "(nervous chuckle) I am sure it is nothing to worry about." \
  --instruction "An anxious man trying to sound casual." \
  --cfg-scale 2.5 --output event.wav
```

**Articulatio-DE has largely lost vocal events:** in German, tags such as `(laugh)` rarely produce the
sound, because the fine-tune was trained on plain read speech.

## Voice conversion (experimental)

`breeze-convert` respeaks an existing recording in another voice. Unlike cloning, which reads new text,
this keeps the original performance: the timing, the phrasing, the rhythm and the emphasis all survive,
and only the speaker identity changes. This is not part of the upstream model. It falls out of the way
the codec splits semantic content from acoustic detail, so it is unique to this implementation and it is
genuinely experimental.

```
build/breeze-convert breeze-tts-2-q8_0.gguf \
  --source recording.wav --text "Transcript of the recording." \
  --ref-audio target_voice.wav --ref-text "Exact transcript of the target clip." \
  --output converted.wav
```

Worth knowing before you rely on it:

- It defaults to near greedy sampling (`--temp 0.3 --top-k 1`) on purpose. Opening sampling up lets
  source timbre leak back into the result, which measurably weakens the target identity.
- **Pitch is regenerated, not copied.** The take is re-sung in the target voice's own register rather
  than at the source's, so a converted vocal can land a fifth away from the original. Whether the tune
  survives varies by clip, so judge it by ear. `--keep-acoustic 1` or `2` copies the lowest acoustic
  codebooks straight from the source and pulls more of the original contour through, at some cost to
  how cleanly the target voice comes out. Higher values fall apart, and on ordinary speech even 1
  causes warbling, so leave it at 0 unless the source is sung.
- Runs at roughly realtime, so a three minute recording takes about three minutes.
- `--text` is optional but matters most here. Giving the source transcript lines the backbone up with
  the forced codes. The same sung clip came back as "Does it pinch I? Send my tears" textless, and as
  "drink it up, I have no fear" once the lyrics were supplied.

See [docs/voice-conversion.md](docs/voice-conversion.md).

## Server

```
build/breeze-server breeze-tts-2-q8_0.gguf --host 127.0.0.1 --port 8080 --webui
```

Open http://localhost:8080/ for the web UI. The HTTP API streams mono 24 kHz signed 16 bit little
endian PCM, matching the reference server.

```
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  --form-string "text=(clears throat) We need to talk." \
  --form-string "instruction=Speak slowly with a restrained, serious tone." \
  --form-string "voice_id=narrator" \
  --output out.pcm
```

Use `--form-string` for text fields, not `-F`. A value starting with `(` makes `curl` build a
multipart group instead of sending the text, which bites the moment you use a vocal event tag.

A WebSocket server also comes up on the HTTP port plus one. It takes text incrementally, streams audio
back as it is produced, and supports changing the delivery instruction or cancelling mid sentence,
which is what you want when driving it from a chat model. See [docs/websocket.md](docs/websocket.md).

> **Neither port has authentication or rate limiting.** Keep them bound to `127.0.0.1` unless something
> in front of them is handling that.

## Bindings

The shared library exposes a plain C ABI, so any language with an FFI can drive it.

```c
breeze_context * ctx = breeze_init("breeze-tts-2-q8_0.gguf", 1);
breeze_request req = {0};
req.text = "Hello there.";
req.cfg_scale = 1.0f;
req.seed = 42;
breeze_generate_wav(ctx, &req, "out.wav");
breeze_free(ctx);
```

Every field treats `0` as "use the model default", so zero initialising the struct is safe and stays
safe as fields are added. See [docs/c-api.md](docs/c-api.md) and [docs/ctypes.md](docs/ctypes.md).

On mingw the library links the compiler runtime statically, so `vulkan-1.dll` from the GPU driver is
the only external dependency.

## Notes

- `--cfg-scale` defaults to 1.0, matching the reference. Values above 1 run the whole pipeline twice
  and push harder toward the instruction. Past about 3 it turns harsh.
- The vocoder comes from the bundled `audio_tokenizer/`, not the Mimi codec sitting in the main
  checkpoint. The reference never uses that one at inference time.
- Streaming decodes in 2 second chunks with 72 frames of left context, which matches the vocoder
  transformer's attention window, so chunks decode the same as they would in a single pass.

## License

Source code is [Apache 2.0](LICENSE), see also [NOTICE](NOTICE). The Breeze TTS 2 model weights and fine-tunes of them (including Articulatio-DE) are governed by the BreezeBlue
Research and Non-Commercial License. You are responsible for complying with the weight license and for
obtaining consent for any reference audio or voices you use.
