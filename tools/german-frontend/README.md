# German text frontend

German text normalization for Breeze-TTS-2 in two parts:

- **`german_normalizer.py`**, a small Python library. It rewrites German text into the form a German
  fine-tune of the model was trained on: numbers, dates, units, formulas, markdown and
  abbreviations all written out as words.
- **`proxy.py`**, a drop-in proxy for `breeze-server`. It has the same HTTP and WebSocket API and
  runs every `text` and `ref_text` through the normalizer. Existing clients get German pronunciation
  without any changes.

## Why

Breeze-TTS-2 reads digits the way its base model learned them: in English, sometimes in French. The
German fine-tune this frontend was built for never saw a digit. Its training transcripts had every
number written out with [num2words](https://github.com/savoirfairelinux/num2words) (`lang="de"`), so
"3.000 m" never occurred, only "dreitausend Meter". Chat models also produce units, abbreviations,
LaTeX, markdown and emojis all the time. The model either reads them aloud ("Sternchen Sternchen"),
spells them out letter by letter, or stumbles over them.

The normalizer does the same conversion at inference time, with the same library the training data
was prepared with:

```
Am 3. Oktober um 8:30 Uhr fährt der Zug mit 120 km/h.
Am dritten Oktober um acht Uhr dreißig fährt der Zug mit einhundertzwanzig Kilometer pro Stunde.
```

## Files

| File | Purpose |
| --- | --- |
| `german_normalizer.py` | The library. Pure Python, needs only `num2words`. |
| `abbreviations.json` | Abbreviations and symbols. Editable and reloaded automatically. |
| `units.json` | About 250 units and currencies. Editable and reloaded automatically. |
| `proxy.py` | HTTP and WebSocket proxy in front of `breeze-server`. |
| `requirements.txt` | Dependencies for both the library and the proxy. |
| `tests/test_normalizer.py` | Golden tests. |

## Install

```
cd tools/german-frontend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # library only: pip install num2words==0.5.14
```

Python 3.9 or newer, which is what the proxy dependencies require. Tested with 3.12 and 3.14.

## What gets rewritten

`normalize()` runs five stages in a fixed order, then two final passes:

1. `clean_markdown`: markdown, LaTeX, emojis, smileys, arrows, list numbers.
2. `expand_units`: units after numbers.
3. `expand_math_symbols`: Unicode math and operators.
4. `expand_numbers`: digits to German words.
5. `expand_abbreviations`: abbreviations and symbols.

The two final passes turn a slash between words into a space and remove spaces before punctuation.
The order matters. Units run before numbers because "1 h" still has to become "eine Stunde", not
"eins Stunde". Math runs after units, so "85 m²" is square metres and not "m hoch zwei". The comments
in the code explain each case.

Every example below is real output.

### Numbers

| Input | Spoken as |
| --- | --- |
| `Es sind 3.000 m.` | Es sind dreitausend Meter. |
| `Der Wert liegt bei 2,5.` | Der Wert liegt bei zwei Komma fünf. |
| `Der Wert beträgt 0,05.` | Der Wert beträgt null Komma null fünf. |
| `Im Jahr 1990 fiel die Mauer.` | Im Jahr neunzehnhundertneunzig fiel die Mauer. |
| `Es dauert 3-5 Tage.` | Es dauert drei bis fünf Tage. |
| `Heute hat es -5 °C.` | Heute hat es minus fünf Grad Celsius. |
| `Der Raum misst 2 x 3 m.` | Der Raum misst zwei mal drei Meter. |
| `Die Nummer 0123456789 ist frei.` | Die Nummer null eins zwei drei vier fünf sechs sieben acht neun ist frei. |
| `Das A320 und COVID19.` | Das A dreihundertzwanzig und COVID neunzehn. |

- The period is a thousands separator and the comma is the decimal mark. Digits after the comma are
  read one at a time.
- Four-digit numbers from 1100 to 1999 are read as years.
- Numbers with a leading zero or more than nine digits are read digit by digit. These are phone
  numbers and IDs.
- `3-5` is a range, `-5` is negative, and digits glued to letters are split off.

### Dates, ordinals, times

| Input | Spoken as |
| --- | --- |
| `Am 3. Oktober ist Feiertag.` | Am dritten Oktober ist Feiertag. |
| `Seit dem 19. Jahrhundert wächst die Stadt.` | Seit dem neunzehnten Jahrhundert wächst die Stadt. |
| `Die 3. Auflage erschien 2021.` | Die dritte Auflage erschien zweitausendeinundzwanzig. |
| `Der Zug fährt um 8:30 Uhr.` | Der Zug fährt um acht Uhr dreißig. |
| `Treffpunkt ist um 1:00 Uhr.` | Treffpunkt ist um ein Uhr. |
| `Das Spiel endete 2:1.` | Das Spiel endete zwei zu eins. |

- `3.` becomes an ordinal in two places: before a month name, or after a word that fixes its
  ending. `am`/`im`/`vom`/`zum`/`dem`/`den`/... give "dritten", and `der`/`die`/`das`/`ein`/... give
  "dritte".
- `H:MM` from 0:00 to 24:59 is a time of day. Any other `a:b` is a score or a ratio ("zu").

### Money

| Input | Spoken as |
| --- | --- |
| `Das kostet 3,50 €.` | Das kostet drei Euro fünfzig. |
| `Das kostet $5.` | Das kostet fünf Dollar. |
| `Das kostet £20.` | Das kostet zwanzig Pfund. |
| `Er verdient 3 Mio. € im Jahr.` | Er verdient drei Millionen Euro im Jahr. |
| `Es kostet 1 Mio. Euro.` | Es kostet eine Million Euro. |
| `Der Etat beträgt 1,5 Mrd. Euro.` | Der Etat beträgt eins Komma fünf Milliarden Euro. |

### Units

| Input | Spoken as |
| --- | --- |
| `Tempo 50 km/h` | Tempo fünfzig Kilometer pro Stunde. |
| `Das dauert 1 h.` | Das dauert eine Stunde. |
| `Das Paket kommt in 1 h.` | Das Paket kommt in einer Stunde. |
| `Für 1 km braucht er 5 min.` | Für einen Kilometer braucht er fünf Minuten. |
| `Es wurde um 1 °C wärmer.` | Es wurde um ein Grad Celsius wärmer. |
| `Sie ist 1,80 m groß.` | Sie ist ein Meter achtzig groß. |
| `Der Akku hat 75 kWh.` | Der Akku hat fünfundsiebzig Kilowattstunden. |
| `Die Wohnung hat 85 m².` | Die Wohnung hat fünfundachtzig Quadratmeter. |
| `Die Dosis lag bei 0,1 µSv/h.` | Die Dosis lag bei null Komma eins Mikrosievert pro Stunde. |
| `Der Lärm erreichte 85 dB(A).` | Der Lärm erreichte fünfundachtzig Dezibel A. |
| `Der Luftdruck liegt bei 1013 hPa.` | Der Luftdruck liegt bei eintausenddreizehn Hektopascal. |
| `Draußen hat es 70 °F.` | Draußen hat es siebzig Grad Fahrenheit. |
| `Der Nullpunkt liegt bei 0 K.` | Der Nullpunkt liegt bei null Kelvin. |
| `Ein Auto mit 150 PS.` | Ein Auto mit einhundertfünfzig Pferdestärken. |
| `Er verbraucht 6,5 l/100 km.` | Er verbraucht sechs Komma fünf Liter auf hundert Kilometer. |
| `Die Temperatur liegt bei 20-25 °C.` | Die Temperatur liegt bei zwanzig bis fünfundzwanzig Grad Celsius. |
| `Die Sicherung hat 16A.` | Die Sicherung hat sechzehn A. |
| `Ein 4K-Monitor und 10 T-Shirts.` | Ein vier K-Monitor und zehn T-Shirts. |
| `Vitamin C bleibt Vitamin C.` | Vitamin C bleibt Vitamin C. |

- Units are replaced only directly after a number. "Vitamin C", "S-Bahn" and "4K-Monitor" stay
  untouched.
- For 1, the article follows the unit's gender and the preposition in front: "in einer Stunde",
  "für einen Kilometer", "um ein Grad".
- Units that are also ordinary letters (`A`, `K`, `T`, `N`, `d`, ...) need a space before them:
  "16 A" is read as ampere, "16A" is left alone.
- `x/y` compounds are built from two entries ("pro"), and lengths with centimetres read naturally
  ("ein Meter achtzig").

### Fractions

| Input | Spoken as |
| --- | --- |
| `Die Seiten 3/4 und 1/2.` | Die Seiten drei Viertel und ein halb. |
| `Der Laden hat 24/7 geöffnet.` | Der Laden hat vierundzwanzig sieben geöffnet. |

Only real fractions (numerator smaller than denominator, denominator at most 10) are read as
fractions. `24/7` and `9/11` stay pairs of numbers.

### Math: LaTeX and Unicode

| Input | Spoken as |
| --- | --- |
| `Der Satz des Pythagoras: $a^2+b^2=c^2$.` | Der Satz des Pythagoras: a hoch zwei plus b hoch zwei gleich c hoch zwei. |
| `$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$` | x gleich minus b plus minus Wurzel aus b hoch zwei minus vier a c, geteilt durch zwei a. |
| `$$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$` | Summe von i gleich eins bis n über i gleich n mal n plus eins, geteilt durch zwei. |
| `$\int_0^1 x^2 \, dx$` | Integral von null bis eins x hoch zwei d x. |
| `$\lim_{x \to \infty} \frac{1}{x} = 0$` | Limes für x gegen unendlich von eins durch x gleich null. |
| `E = mc²` | E gleich m c hoch zwei. |
| `√2 ist irrational.` | Wurzel aus zwei ist irrational. |
| `10⁻³ ist ein Tausendstel.` | zehn hoch minus drei ist ein Tausendstel. |
| `H₂O ist Wasser.` | H zwei O ist Wasser. |
| `2^10 = 1024` | zwei hoch zehn gleich eintausendvierundzwanzig. |
| `5 + 3 = 8 und 10 - 4 = 6.` | fünf plus drei gleich acht und zehn minus vier gleich sechs. |
| `5 * 3 = 15` | fünf mal drei gleich fünfzehn. |

- LaTeX is recognized in `$...$`, `$$...$$`, `\(...\)` and `\[...\]`. That covers fractions, roots,
  sums and products with limits, integrals, limits, Greek letters, common operators and relations,
  `\sin(x)` ("Sinus von x"), `f'(x)` ("f Strich von x") and factorials.
- A dollar sign before or after a number is money, not math.
- Operators are only read between operands, so `C++` and a dash in the middle of a sentence stay
  as they are.

### Markdown

```
## Überblick

Das Land hat **drei** Stärken:

1. Industrie
2. *Forschung*
3. Lage
```

becomes

```
Überblick. Das Land hat drei Stärken. Erstens, Industrie. Zweitens, Forschung. Drittens, Lage.
```

| Input | Spoken as |
| --- | --- |
| `**3. Wirtschaft:**` + line break + `Die Wirtschaft wächst.` | Drittens, Wirtschaft. Die Wirtschaft wächst. |
| `` `code` und [Link](https://example.org) `` | code und Link. |
| `– geschafft, vorbei` | – geschafft, vorbei. |

- Code fences are dropped. Headings, emphasis, bullets (`-`, `*`, `+`, `•`), block quotes and
  horizontal rules lose their markup.
- Links keep their text. Tables become their cells joined with commas.
- List numbers become "Erstens, Zweitens, ...". Emphasis is removed first, so `**3. Wirtschaft:**`
  is recognized as a list item too.
- Every line gets a sentence end, and line breaks become spaces. Without the sentence end, the model
  reads a heading into the next sentence. It never saw line breaks in training.
- A dash at the start of a line is punctuation, not a bullet.

### Emojis, smileys, arrows

| Input | Spoken as |
| --- | --- |
| `Das ist super😂toll!` | Das ist super toll! |
| `Es wird regnen :) Nimm einen Schirm mit.` | Es wird regnen, Nimm einen Schirm mit. |
| `Danke 👍` | Danke. |
| `Treffen um 12:30 :)` | Treffen um zwölf Uhr dreißig. |
| `Berlin → Hauptstadt` | Berlin, Hauptstadt. |

Emojis are removed. That includes flags, skin tones, joiners and keycaps. Text smileys (`:)`, `;-)`,
`xD`, `^^`, `<3`, `¯\_(ツ)_/¯`) are removed only when they stand free. Where one replaced
punctuation, a comma keeps the pause. Arrows become a comma.

### Abbreviations

| Input | Spoken as |
| --- | --- |
| `Obst, z. B. Äpfel, Birnen usw.` | Obst, zum Beispiel Äpfel, Birnen und so weiter |
| `Das ist z.B. so.` | Das ist zum Beispiel so. |
| `Im 3. Jh. v. Chr. Er zeigte ein Bild.` | Im dritten Jahrhundert vor Christus. Er zeigte ein Bild. |
| `Um 300 v. Chr. entstand die Stadt.` | Um dreihundert vor Christus entstand die Stadt. |
| `Mo. bis Fr. geöffnet.` | Montag bis Freitag geöffnet. |
| `Er hat ca. 30 % mehr.` | Er hat circa dreißig Prozent mehr. |

- If an abbreviation ends in a period and a typical sentence start follows ("Er", "Die", "Danach",
  ...), the period is kept. Otherwise the pause after "v. Chr." would be lost. A capitalized word
  alone is not enough, since every German noun is capitalized.
- Written out, "z. B." no longer looks like a sentence end to the streaming splitter.

### Slashes and prose

| Input | Spoken as |
| --- | --- |
| `Rolls-Royce/Snecma baut das Triebwerk.` | Rolls-Royce Snecma baut das Triebwerk. |
| `Kommt und/oder geht.` | Kommt und oder geht. |
| `Die Geschwindigkeit in km/h angeben.` | Die Geschwindigkeit in Kilometer pro Stunde angeben. |
| `Ich programmiere in C++.` | Ich programmiere in C plus plus. |
| `Er sagte – wie gesagt – nichts.` | Er sagte – wie gesagt – nichts. |
| `Das Ergebnis (siehe oben) war gut.` | Das Ergebnis (siehe oben) war gut. |

## Using the library

```python
from german_normalizer import normalize

normalize("Das kostet 3,50 €.")      # 'Das kostet drei Euro fünfzig.'
```

The stages are available on their own: `clean_markdown`, `expand_latex` (also called from
`clean_markdown`), `expand_units`, `expand_math_symbols`, `expand_numbers` and
`expand_abbreviations`. Only `normalize()` runs them in the right order and applies the final
slash and punctuation passes.

From the command line:

```
python german_normalizer.py "Es sind 3.000 m."        # Es sind dreitausend Meter.
echo "Das dauert 1 h." | python german_normalizer.py   # Das dauert eine Stunde.
```

### Streaming: `SentenceBuffer`

Chat models stream tokens, and Breeze speaks one sentence at a time. `SentenceBuffer.push()` collects
tokens and returns normalized complete sentences as soon as they are safe to cut, or `""` until
then. `flush()` returns the rest.

```python
from german_normalizer import SentenceBuffer

buf = SentenceBuffer()
for token in ["Das ist z.", " B. gut. Am 3", ". Oktober ist frei. ", "Danach geht es weiter."]:
    piece = buf.push(token)          # '', '', 'Das ist zum Beispiel gut. ', 'Am dritten Oktober ist frei. '
    if piece:
        speak(piece)
speak(buf.flush())                   # 'Danach geht es weiter.'
```

Cutting at every ". " would split "z. B." and "am 3. Oktober" apart. The buffer marks every
abbreviation first and only cuts at a period that does not belong to one. It also never cuts at:

- a period after a number (list numbers, ordinals),
- anything inside a `$$ ... $$` or `\[ ... \]` block,
- the last 16 characters, where half an abbreviation could still be arriving.

## The proxy

Move `breeze-server` to internal ports and let the proxy take the public ones:

```
build/breeze-server model.gguf --host 127.0.0.1 --port 8147 --webui    # WebSocket on 8148
python tools/german-frontend/proxy.py                                  # HTTP 8137, WebSocket 8138
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `PROXY_HOST` | `0.0.0.0` | Interface the proxy binds to. |
| `PROXY_HTTP_PORT` | `8137` | HTTP port. |
| `PROXY_WS_PORT` | `8138` | WebSocket port. |
| `BREEZE_UPSTREAM_HTTP` | `http://127.0.0.1:8147` | `breeze-server` HTTP base URL. |
| `BREEZE_UPSTREAM_WS` | `ws://127.0.0.1:8148` | `breeze-server` WebSocket URL. |

Only the fields `text` and `ref_text` are touched. Everything else is passed through unchanged.

**HTTP.** Every path and method is forwarded, for example `POST /v1/audio/speech`,
`/v1/audio/convert` and `/v1/voices`.

- For `multipart/form-data` and `application/x-www-form-urlencoded` requests, the form fields `text`
  and `ref_text` are normalized. So are query parameters with those names.
- Uploaded files (`ref_audio`, `source`) pass through untouched.
- The response is streamed back as it is generated, so time to first audio stays the same.
- If `breeze-server` cannot be reached, the proxy answers `502`.

```
curl --form-string "text=Es sind 3.000 m." --form-string "voice_id=my-voice" \
     http://127.0.0.1:8137/v1/audio/speech -o out.pcm         # speaks "Es sind dreitausend Meter."
```

**WebSocket.** Each connection gets its own `SentenceBuffer`.

- A `text` message is forwarded once it completes a sentence, normalized. Text that does not finish a
  sentence waits in the proxy.
- `end` and `flush` forward everything still waiting, plus their own `text`, normalized.
- `cancel` discards what is waiting and is forwarded.
- `ref_text` in `start` is normalized.
- All other messages, binary frames and every server reply are relayed unchanged. The `speaking`
  event therefore shows the text as it is actually spoken:

```json
-> {"type":"start","voice_id":"my-voice"}
-> {"type":"end","text":"Tempo 50 km/h 😂"}
<- {"type":"speaking","text":"Tempo fünfzig Kilometer pro Stunde."}
<- <binary audio>
<- {"type":"done"}
```

**`/health`** returns the health report of `breeze-server` with `ws_port` replaced by the proxy's
own WebSocket port. The web UI (with `--webui`) is served through the proxy and looks up the socket
there, so it also connects through the proxy and not past it.

The proxy logs each table load and each rewritten HTTP field:

```
[german_normalizer] units loaded: 252
[german_normalizer] abbreviations loaded: 97 abbreviations, 12 symbols
[text] v1/audio/speech text: 'Es sind 3.000 m.' -> 'Es sind dreitausend Meter.'
```

> **No authentication and no rate limiting**, just like `breeze-server`. The default `0.0.0.0`
> makes the proxy a drop-in for a server that was reachable on the local network. Set
> `PROXY_HOST=127.0.0.1` if it does not need to be, keep `breeze-server` itself on `127.0.0.1`, and
> never expose either one to the internet.

## Extending the tables

Both JSON files are checked on every call and reloaded as soon as their modification time changes.
Edits apply immediately, even in a running proxy. If a file fails to parse, the previous table is
kept and a warning is logged. To use tables stored elsewhere, set
`german_normalizer.ABBREVIATIONS_PATH` and `german_normalizer.UNITS_PATH`.

The section names (`abkuerzungen`, `zeichen`, `einheiten`) are German. They are data keys, the same
as in the tables this package was extracted from. Keep them.

### `abbreviations.json`

```json
{
  "abkuerzungen": { "z. B.": "zum Beispiel", "v. Chr.": "vor Christus" },
  "zeichen": { "%": " Prozent", "&": " und " }
}
```

- **`abkuerzungen`** (abbreviations) match whole words only: no letter directly before or after.
  - A space in the key matches any whitespace, including none, so `z. B.` also matches `z.B.`.
  - Matching is case-sensitive.
  - Longer keys are applied first.
  - For keys ending in a period, the period-before-a-sentence-start rule above applies.
- **`zeichen`** (symbols) are replaced wherever they occur. Put the spaces you need into the value.
- Abbreviations that inflect (`sog.`, `o. ä.`) can only expand to one fixed form, which does not fit
  every sentence.

### `units.json`

```json
{
  "einheiten": {
    "kWh": ["Kilowattstunde", "Kilowattstunden", "f", 0],
    "A":   ["Ampere", "Ampere", "n", 1]
  }
}
```

Each entry is `[singular, plural, gender, space required]`:

- Gender `m`/`f`/`n` picks the article for 1: "ein"/"einen"/"einem" or "eine"/"einer".
- `space required = 1` is for units that are also ordinary letters or appear in product names.
  "16 A" is read as ampere, "16A" and "4K" are left alone.
- A compound `x/y` needs no entry of its own, because it is built as "x pro y". Add one only when it
  is read differently, like `l/100 km` ("Liter auf hundert Kilometer").

Everything else (sentence starts, month names, prepositions, LaTeX commands, Greek letters) lives in
the code, near the rule that uses it.

## Tests

```
python tests/test_normalizer.py      # plain python
python -m pytest tests               # or pytest
```

The expected outputs are literals produced by the implementation this package was extracted from.
Any change in how something is spoken shows up as a failure. The tests also cover streaming and
table reloading.

## Limitations

The rules are heuristics for chat-style German. Known cases where they fall short:

- A phone number with a space ("0123 456789") is read as two numbers.
- An abbreviation at the very end of the text takes the final period with it ("usw." becomes
  "und so weiter").
- Inflecting abbreviations always use their one table form.
- `num2words` is pinned, because the exact wording depends on its version.

## License

Apache-2.0, like the rest of this repository (see [LICENSE](../../LICENSE)).

`num2words` is licensed under the LGPL-2.1 (or later). It is used as an ordinary dependency installed
from PyPI. Nothing of it is bundled, copied or modified here. The proxy dependencies are FastAPI
(MIT), Uvicorn, Starlette, HTTPX and websockets (BSD-3-Clause), and python-multipart (Apache-2.0).
