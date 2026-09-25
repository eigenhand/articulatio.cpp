"""Golden tests for german_normalizer.

Every expected output below is a literal that was produced by the reference
implementation this package was extracted from (with num2words 0.5.14), so a
failure means the spoken text changed.

    python tests/test_normalizer.py      # plain python
    python -m pytest tests               # or pytest
"""
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import german_normalizer as gn  # noqa: E402


# numbers, dates, ordinals, times, money
NUMBERS = [
    ("Es sind 3.000 m.",
     "Es sind dreitausend Meter."),
    ("Das kostet 3,50 €.",
     "Das kostet drei Euro fünfzig."),
    ("Das kostet 1,00 €.",
     "Das kostet ein Euro."),
    ("Im Jahr 1990 fiel die Mauer.",
     "Im Jahr neunzehnhundertneunzig fiel die Mauer."),
    ("Am 3. Oktober ist Feiertag.",
     "Am dritten Oktober ist Feiertag."),
    ("Der Zug fährt um 8:30 Uhr.",
     "Der Zug fährt um acht Uhr dreißig."),
    ("Treffpunkt ist um 1:00 Uhr.",
     "Treffpunkt ist um ein Uhr."),
    ("Es dauert 3-5 Tage.",
     "Es dauert drei bis fünf Tage."),
    ("Heute hat es -5 °C.",
     "Heute hat es minus fünf Grad Celsius."),
    ("Das dauert 1 h.",
     "Das dauert eine Stunde."),
    ("Das Paket kommt in 1 h.",
     "Das Paket kommt in einer Stunde."),
    ("Sie ist 1,80 m groß.",
     "Sie ist ein Meter achtzig groß."),
    ("Der Raum misst 2 x 3 m.",
     "Der Raum misst zwei mal drei Meter."),
    ("Etwa 3/4 der Befragten stimmten zu.",
     "Etwa drei Viertel der Befragten stimmten zu."),
    # only real fractions (numerator < denominator <= 10) are read as fractions
    ("Der Laden hat 24/7 geöffnet.",
     "Der Laden hat vierundzwanzig sieben geöffnet."),
    ("Das Spiel endete 2:1.",
     "Das Spiel endete zwei zu eins."),
    # leading zero or more than 9 digits: digit by digit
    ("Die Nummer 0123456789 ist frei.",
     "Die Nummer null eins zwei drei vier fünf sechs sieben acht neun ist frei."),
    # known limitation: a space splits a phone number into two numbers
    ("Rufen Sie 0123 456789 an.",
     "Rufen Sie null eins zwei drei vierhundertsechsundfünfzigtausendsiebenhundertneunundachtzig an."),
    ("Es waren 1.234.567 Besucher.",
     "Es waren eine Million zweihundertvierunddreißigtausendfünfhundertsiebenundsechzig Besucher."),
    ("Der Wert liegt bei 2,5.",
     "Der Wert liegt bei zwei Komma fünf."),
    ("Die 3. Auflage erschien 2021.",
     "Die dritte Auflage erschien zweitausendeinundzwanzig."),
    ("Seit dem 19. Jahrhundert wächst die Stadt.",
     "Seit dem neunzehnten Jahrhundert wächst die Stadt."),
    ("Er verdient 3 Mio. € im Jahr.",
     "Er verdient drei Millionen Euro im Jahr."),
    ("Der Etat beträgt 1,5 Mrd. Euro.",
     "Der Etat beträgt eins Komma fünf Milliarden Euro."),
    ("Das kostet $5.",
     "Das kostet fünf Dollar."),
    ("Es kostet 1 Mio. Euro.",
     "Es kostet eine Million Euro."),
]

# units after numbers, articles for 1, units that need a space
UNITS = [
    ("Tempo 50 km/h",
     "Tempo fünfzig Kilometer pro Stunde."),
    ("Der Akku hat 75 kWh.",
     "Der Akku hat fünfundsiebzig Kilowattstunden."),
    ("Die Wohnung hat 85 m².",
     "Die Wohnung hat fünfundachtzig Quadratmeter."),
    ("Die Dosis lag bei 0,1 µSv/h.",
     "Die Dosis lag bei null Komma eins Mikrosievert pro Stunde."),
    ("Der Lärm erreichte 85 dB(A).",
     "Der Lärm erreichte fünfundachtzig Dezibel A."),
    ("Der Luftdruck liegt bei 1013 hPa.",
     "Der Luftdruck liegt bei eintausenddreizehn Hektopascal."),
    ("Draußen hat es 70 °F.",
     "Draußen hat es siebzig Grad Fahrenheit."),
    ("Der Nullpunkt liegt bei 0 K.",
     "Der Nullpunkt liegt bei null Kelvin."),
    ("Ein Auto mit 150 PS.",
     "Ein Auto mit einhundertfünfzig Pferdestärken."),
    ("Er verbraucht 6,5 l/100 km.",
     "Er verbraucht sechs Komma fünf Liter auf hundert Kilometer."),
    ("Für 1 km braucht er 5 min.",
     "Für einen Kilometer braucht er fünf Minuten."),
    ("Es wurde um 1 °C wärmer.",
     "Es wurde um ein Grad Celsius wärmer."),
    # single-letter units need a space: 16A is not ampere
    ("Die Sicherung hat 16A.",
     "Die Sicherung hat sechzehn A."),
    # no unit in front of a hyphen
    ("Ein 4K-Monitor und 10 T-Shirts.",
     "Ein vier K-Monitor und zehn T-Shirts."),
    ("Vitamin C bleibt Vitamin C.",
     "Vitamin C bleibt Vitamin C."),
]

# LaTeX and Unicode math
MATH = [
    ("Der Satz des Pythagoras: $a^2+b^2=c^2$.",
     "Der Satz des Pythagoras: a hoch zwei plus b hoch zwei gleich c hoch zwei."),
    ("$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$",
     " x gleich minus b plus minus Wurzel aus b hoch zwei minus vier a c, geteilt durch zwei a."),
    ("$$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$$",
     " Summe von i gleich eins bis n über i gleich n mal n plus eins, geteilt durch zwei."),
    ("$\\int_0^1 x^2 \\, dx$",
     " Integral von null bis eins x hoch zwei d x."),
    ("$\\lim_{x \\to \\infty} \\frac{1}{x} = 0$",
     " Limes für x gegen unendlich von eins durch x gleich null."),
    ("E = mc²",
     "E gleich m c hoch zwei."),
    ("√2 ist irrational.",
     " Wurzel aus zwei ist irrational."),
    ("10⁻³ ist ein Tausendstel.",
     "zehn hoch minus drei ist ein Tausendstel."),
    ("H₂O ist Wasser.",
     "H zwei O ist Wasser."),
    ("2^10 = 1024",
     "zwei hoch zehn gleich eintausendvierundzwanzig."),
    ("5 + 3 = 8 und 10 - 4 = 6.",
     "fünf plus drei gleich acht und zehn minus vier gleich sechs."),
    ("5 * 3 = 15",
     "fünf mal drei gleich fünfzehn."),
    # regression: a superscript root degree used to raise ValueError ("³".isdigit() is True)
    ("Die Wurzel: $\\sqrt[³]{x}$",
     "Die Wurzel: dritte Wurzel aus x."),

]

# markdown: headings, emphasis, list numbers, bullets, tables, links
MARKDOWN = [
    ("## Überblick\n\nDas Land hat **drei** Stärken:\n\n1. Industrie\n2. *Forschung*\n3. Lage",
     "Überblick. Das Land hat drei Stärken. Erstens, Industrie. Zweitens, Forschung. Drittens, Lage."),
    ("**3. Wirtschaft:**\nDie Wirtschaft wächst.",
     "Drittens, Wirtschaft. Die Wirtschaft wächst."),
    ("- Punkt eins\n- Punkt zwei",
     "Punkt eins. Punkt zwei."),
    ("| Stadt | Einwohner |\n|---|---|\n| Berlin | 3,7 Mio. |",
     "Stadt, Einwohner. Berlin, drei Komma sieben Millionen"),
    ("`code` und [Link](https://example.org)",
     "code und Link."),
    # a dash at the start of a line is punctuation, not a bullet
    ("– geschafft, vorbei",
     "– geschafft, vorbei."),
]

# emojis, smileys, arrows
EMOJIS = [
    ("Das ist super😂toll!",
     "Das ist super toll!"),
    ("Es wird regnen :) Nimm einen Schirm mit.",
     "Es wird regnen, Nimm einen Schirm mit."),
    ("Danke 👍",
     "Danke."),
    ("Berlin → Hauptstadt",
     "Berlin, Hauptstadt."),
    ("Na gut ;-)",
     "Na gut."),
]

# abbreviations and symbols
ABBREVIATIONS = [
    # known limitation: a trailing abbreviation takes the final period with it
    ("Obst, z. B. Äpfel, Birnen usw.",
     "Obst, zum Beispiel Äpfel, Birnen und so weiter."),
    ("Das ist z.B. so.",
     "Das ist zum Beispiel so."),
    # the period stays when a typical sentence start follows ...
    ("Im 3. Jh. v. Chr. Er zeigte ein Bild.",
     "Im dritten Jahrhundert vor Christus. Er zeigte ein Bild."),
    # ... and goes when the sentence continues
    ("Um 300 v. Chr. entstand die Stadt.",
     "Um dreihundert vor Christus entstand die Stadt."),
    ("Mo. bis Fr. geöffnet.",
     "Montag bis Freitag geöffnet."),
    ("Er hat ca. 30 % mehr.",
     "Er hat circa dreißig Prozent mehr."),
    # regression: an abbreviation at the very end of the text keeps its sentence period
    ("Äpfel, Birnen usw.",
     "Äpfel, Birnen und so weiter."),

]

# slashes, C++, dashes and parentheses in prose
PROSE = [
    ("Rolls-Royce/Snecma baut das Triebwerk.",
     "Rolls-Royce Snecma baut das Triebwerk."),
    ("Kommt und/oder geht.",
     "Kommt und oder geht."),
    ("Die Geschwindigkeit in km/h angeben.",
     "Die Geschwindigkeit in Kilometer pro Stunde angeben."),
    ("Ich programmiere in C++.",
     "Ich programmiere in C plus plus."),
    ("Er sagte – wie gesagt – nichts.",
     "Er sagte – wie gesagt – nichts."),
    ("Das Ergebnis (siehe oben) war gut.",
     "Das Ergebnis (siehe oben) war gut."),
]


def _check(cases):
    failures = []
    for text, expected in cases:
        got = gn.normalize(text)
        if got != expected:
            failures.append(f"{text!r}\n  expected {expected!r}\n  got      {got!r}")
    assert not failures, "\n" + "\n".join(failures)


def test_numbers():
    _check(NUMBERS)

def test_units():
    _check(UNITS)

def test_math():
    _check(MATH)

def test_markdown():
    _check(MARKDOWN)

def test_emojis():
    _check(EMOJIS)

def test_abbreviations():
    _check(ABBREVIATIONS)

def test_prose():
    _check(PROSE)


def test_empty_input():
    assert gn.normalize("") == ""
    assert gn.SentenceBuffer().flush() == ""


def test_stages():
    # each stage on its own; normalize() chains them
    cases = [
        (gn.clean_markdown, "**Fett** und *kursiv*\n- Liste",
         "Fett und kursiv. Liste."),
        (gn.expand_units, "in 1 h und 230 V und 50 km/h",
         "in einer Stunde und 230 Volt und 50 Kilometer pro Stunde"),
        (gn.expand_math_symbols, "E = mc², √2, 10⁻³, H₂O",
         "E = m c hoch 2 ,  Wurzel aus 2, 10 hoch minus 3 , H 2 O"),
        (gn.expand_numbers, "Am 3. Oktober um 8:30 Uhr, 3.000 und 2,5",
         "Am dritten Oktober um acht Uhr dreißig, dreitausend und zwei Komma fünf"),
        (gn.expand_latex, "Es gilt $a^2+b^2=c^2$ hier.",
         "Es gilt  a hoch zwei plus b hoch zwei gleich c hoch zwei  hier."),
        (gn.expand_abbreviations, "z. B. usw. 50 % & mehr",
         "zum Beispiel und so weiter 50 Prozent und mehr"),
    ]
    for fn, text, expected in cases:
        got = fn(text)
        assert got == expected, f"{fn.__name__}({text!r})\n  expected {expected!r}\n  got      {got!r}"


def _feed(text, size):
    buf = gn.SentenceBuffer()
    out = []
    for i in range(0, len(text), size):
        piece = buf.push(text[i:i + size])
        if piece.strip():             # whitespace-only pieces carry nothing to speak
            out.append(piece)
    rest = buf.flush()
    if rest.strip():
        out.append(rest)
    return out


def test_sentence_buffer_streams():
    cases = [
        # abbreviations and ordinals are never cut, even when fed one character at a time
        ("Das ist z. B. gut. Am 3. Oktober ist frei. Danach geht es weiter. Ende.", 1, [
            "Das ist zum Beispiel gut. ",
            "Am dritten Oktober ist frei. ",
            "Danach geht es weiter. Ende.",
        ]),
        # bold list numbers stay whole (chunks of 7 characters, like a token stream)
        ("## Plan\n\n**1. Einkauf:** Brot und Milch.\n**2. Kochen:** ca. 30 min.\n\nGuten Appetit!", 7, [
            "Plan. ",
            "Erstens, Einkauf: Brot und Milch. ",
            "Zweitens, Kochen: circa dreißig Minuten. Guten Appetit!",
        ]),
        # a multi-line $$ formula block is released in one piece
        ("Die Formel lautet:\n$$\nx = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\n$$\nDamit ist alles gesagt. Fertig.", 1, [
            "Die Formel lautet. ",
            " x gleich minus b plus minus Wurzel aus b hoch zwei minus vier a c, geteilt durch zwei a. ",
            "Damit ist alles gesagt. Fertig.",
        ]),
    ]
    for text, size, expected in cases:
        got = _feed(text, size)
        assert got == expected, f"{text!r} in chunks of {size}\n  expected {expected!r}\n  got      {got!r}"


def test_sentence_buffer_holds_back_until_flush():
    buf = gn.SentenceBuffer()
    assert buf.push("Das ist z.") == ""          # could be the start of "z. B."
    assert buf.push(" B. gut. Und") == ""        # the sentence end is still too close to the tail
    assert buf.flush(" weiter.") == "Das ist zum Beispiel gut. Und weiter."
    assert buf.pending == ""


def test_tables_reload_on_change():
    tmp = Path(tempfile.mkdtemp())
    saved = gn.ABBREVIATIONS_PATH, gn.UNITS_PATH
    try:
        abbr, units = tmp / "abbreviations.json", tmp / "units.json"
        shutil.copy(saved[0], abbr)
        shutil.copy(saved[1], units)
        gn.ABBREVIATIONS_PATH, gn.UNITS_PATH = abbr, units
        assert gn.normalize("Das ist xyz. klar.") == "Das ist xyz. klar."
        assert gn.normalize("Es sind 3 fl.") == "Es sind drei fl."

        def edit(path, section, key, value):
            data = json.loads(path.read_text(encoding="utf-8"))
            data[section][key] = value
            mtime = path.stat().st_mtime
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.utime(path, (mtime + 10, mtime + 10))    # a new mtime even on coarse clocks

        edit(abbr, "abkuerzungen", "xyz.", "iks ypsilon zett")
        edit(units, "einheiten", "fl", ["Flasche", "Flaschen", "f", 1])
        assert gn.normalize("Das ist xyz. klar.") == "Das ist iks ypsilon zett klar."
        assert gn.normalize("Es sind 3 fl und 1 fl.") == "Es sind drei Flaschen und eine Flasche."

        # a broken file keeps the last good table instead of failing (and logs a warning)
        mtime = abbr.stat().st_mtime
        abbr.write_text("{ not json", encoding="utf-8")
        os.utime(abbr, (mtime + 10, mtime + 10))
        log = logging.getLogger("german_normalizer")
        level = log.level
        log.setLevel(logging.ERROR)                 # the warning is expected here
        try:
            assert gn.normalize("Das ist xyz. klar.") == "Das ist iks ypsilon zett klar."
        finally:
            log.setLevel(level)
    finally:
        gn.ABBREVIATIONS_PATH, gn.UNITS_PATH = saved
        shutil.rmtree(tmp, ignore_errors=True)
    assert gn.normalize("Das ist xyz. klar.") == "Das ist xyz. klar."   # back to the shipped tables


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}{exc}")
    print(f"{len(tests) - failed}/{len(tests)} test functions passed")
    sys.exit(1 if failed else 0)
