"""german_normalizer.py - prepare German text so that Breeze-TTS-2 reads it as German.

A German fine-tune of Breeze-TTS-2 only ever saw numbers written out as words
during training (num2words, lang="de"). Digits, units, formulas and markdown
make it fall back to the English or French readings it knows from the base
model. This module rewrites text into the form the model was trained on.

Pipeline in normalize():
    clean_markdown -> expand_units -> expand_math_symbols -> expand_numbers
    -> expand_abbreviations

  clean_markdown        LaTeX formulas, markdown, emojis, smileys, arrows,
                        list numbers
  expand_units          230 V -> 230 Volt, 1 h -> eine Stunde (units.json)
  expand_math_symbols   a², √2, 2^10, 5 + 3, 10 - 4 (after the units, so that
                        85 m² is already "Quadratmeter")
  expand_numbers        write digits out in German (num2words, as in training)
  expand_abbreviations  abbreviations and symbols (abbreviations.json)

SentenceBuffer releases a token stream sentence by sentence without tearing
abbreviations, list numbers or ordinals apart at a chunk boundary.

Both tables are reloaded automatically whenever their file changes.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from num2words import num2words

__all__ = ["normalize", "SentenceBuffer", "clean_markdown", "expand_units", "expand_numbers",
           "expand_math_symbols", "expand_latex", "expand_abbreviations",
           "ABBREVIATIONS_PATH", "UNITS_PATH", "HOLDBACK_CHARS"]

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------ abbreviations
#
# The model spells "z. B." out letter by letter or swallows it. Writing it out
# before synthesis fixes that - and a second problem along with it: to the
# sentence splitter, the period in "z. B." looks like the end of a sentence
# and cuts the utterance in the wrong place.
#
# The table lives in abbreviations.json so it can be changed without touching
# the code. It is reloaded whenever the file changes.

ABBREVIATIONS_PATH = _HERE / "abbreviations.json"
_abbrev_cache: tuple = (None, [], [])        # ((path, mtime), rules, symbols)

# Longest abbreviation plus some slack. This many characters at the end of the
# buffer are held back, because half an abbreviation could still be sitting
# there: "z." arrives in one token, " B." only in the next.
HOLDBACK_CHARS = 16


# When an abbreviation ends in a period, that period is often the end of the
# sentence as well: "im 3. Jh. v. Chr. Er zeigte" - written out, the pause was
# gone. If a typical sentence opener follows, the period is kept. A capitalised
# word alone is not enough: in German every noun is capitalised.
_SENTENCE_START = re.compile(r"\s+(?:Er|Sie|Es|Der|Die|Das|Den|Dem|Ein|Eine|Einen|Danach|Dann|"
                             r"Später|Damals|Im|In|Am|Auch|Doch|Aber|Wir|Ich|Man|Dies|Diese|Dieser|"
                             r"Dieses|So|Heute|Seit|Bis|Nach|Vor|Als|Wenn|Weil|Hier|Dort|Zudem|"
                             r"Außerdem|Deshalb|Daher|Trotzdem|Jedoch|Allerdings)\b")


def _with_sentence_period(expansion: str):
    """Replacement function: the expansion, plus the period if a sentence starts next."""
    def replace(m: re.Match) -> str:
        # also at the very end of the text: the period was the sentence end there too
        at_end = not m.string[m.end():].strip()
        return expansion + ("." if at_end or _SENTENCE_START.match(m.string, m.end()) else "")
    return replace


def _abbreviation_rules():
    """All (regex, replacement) pairs: abbreviations first, then symbols.

    Reloads abbreviations.json as soon as the file changes.
    """
    global _abbrev_cache
    try:
        key = (str(ABBREVIATIONS_PATH), ABBREVIATIONS_PATH.stat().st_mtime)
    except OSError:
        return []
    if _abbrev_cache[0] == key:
        return _abbrev_cache[1] + _abbrev_cache[2]
    try:
        data = json.loads(ABBREVIATIONS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:                           # noqa: BLE001
        log.warning("%s unreadable, keeping the previous table: %s", ABBREVIATIONS_PATH, exc)
        return _abbrev_cache[1] + _abbrev_cache[2]

    rules = []
    # Longest first: otherwise "km" would fire inside "km/h" and leave
    # "Kilometer/h" behind.
    for k, v in sorted(data.get("abkuerzungen", {}).items(), key=lambda x: -len(x[0])):
        pat = re.escape(k).replace("\\ ", r"\s*")
        # no letter before, no letter after - otherwise "Str." would also
        # match in the middle of a word
        rules.append((re.compile(r"(?<![\w\u00c0-\u024f])" + pat + r"(?![\w\u00c0-\u024f])"),
                      _with_sentence_period(v) if k.endswith(".") else v))
    symbols = [(re.compile(re.escape(k)), v) for k, v in
               sorted(data.get("zeichen", {}).items(), key=lambda x: -len(x[0]))]
    _abbrev_cache = (key, rules, symbols)
    log.info("abbreviations loaded: %d abbreviations, %d symbols", len(rules), len(symbols))
    return rules + symbols


def expand_abbreviations(text: str) -> str:
    """Write out abbreviations and symbols from abbreviations.json: z. B. -> zum Beispiel."""
    if not text:
        return text
    for rx, replacement in _abbreviation_rules():
        text = rx.sub(replacement if callable(replacement) else replacement.replace("\\", "\\\\"), text)
    return re.sub(r"[ \t]{2,}", " ", text)


# Markup that trips up speech synthesis. Chat models answer in markdown:
# horizontal rules, headings, asterisks, tables, bullets. The model either
# reads them aloud ("Sternchen Sternchen") or chokes on the blank lines,
# because they produce empty pieces.
_CODE_FENCE    = re.compile(r"```[\s\S]*?```")
_HRULE         = re.compile(r"(?m)^[ \t]*([-*_])[ \t]*(?:\1[ \t]*){2,}$")
_HEADING       = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*")
# Markdown bullets are only -, * and + (plus the bullet character itself).
# Dashes do NOT belong here: an en dash at the start of a line
# ("– geschafft, vorbei") is punctuation, not a list. They used to be listed
# here by mistake and were swallowed at the start of a line.
_BULLET        = re.compile(r"(?m)^[ \t]{0,6}[-*+\u2022][ \t]+")
_LIST_NUMBER   = re.compile(r"(?m)^[ \t]{0,6}\d{1,2}[.)][ \t]+")
_BLOCKQUOTE    = re.compile(r"(?m)^[ \t]{0,3}>[ \t]?")
_TABLE_ROW     = re.compile(r"(?m)^[ \t]*\|(.*)\|[ \t]*$")
_TABLE_RULE    = re.compile(r"(?m)^[ \t]*\|[\s:|-]+\|[ \t]*$")
_LINK          = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_EMPHASIS      = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
_BLANK_LINES   = re.compile(r"\n{2,}")
_SYMBOLS_ONLY  = re.compile(r"(?m)^[ \t]*[^\wÀ-ɏ\n]+[ \t]*$")


# The model read emojis and smileys aloud ("😂" as a word). Remove them - as a
# space, so that "super😂toll" does not become "supertoll". The blocks:
# pictographs and emoticons, dingbats/symbols (✅ ❤ ☀ ★), miscellaneous
# symbols and arrows (⭐), flags (regional indicators), plus the invisible
# joiners (variation selector FE0F, zero-width joiner 200D, keycap 20E3, flag
# tag characters).
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF"
                    "\uFE0E\uFE0F\u200D\u20E3\U000E0020-\U000E007F\u3030\u303D]+")
# text smileys only when free-standing: ":)" yes, "12:30" and "1:3" no
_SMILEY = re.compile(r"(?<![\w])(?:[:;=][-'o^]?[)(\]\[DPpOo3|/\\]+|[xX]D+|\^\^|<3"
                     r"|¯\\_\(ツ\)_/¯)(?=\s|$|[.,!?])")
# arrows ("Berlin → Hauptstadt", "A -> B") are a pause, not a word
_ARROW = re.compile("[ \t]*(?:[\u2190-\u21FF\u27F5-\u27FF]|->|=>)[ \t]*")


def _drop_symbol(m: re.Match) -> str:
    """Remove an emoji or smiley. If it stands in for punctuation
    ("regnen :) Nimm ..."), a comma is left as a pause - otherwise the sentence
    would run into the next one without a breath ("regnen Nimm")."""
    before = m.string[:m.start()].rstrip()
    after = m.string[m.end():].lstrip()
    if before and before[-1].isalnum() and after[:1].isupper():
        return ", "
    return " "


def _enumeration_word(m: re.Match) -> str:
    """List number as an enumeration adverb: 1. -> Erstens, 3) -> Drittens.
    Ordinal plus "ns": dritte -> drittens, zwanzigste -> zwanzigstens."""
    n = int(re.search(r"\d+", m.group(0)).group(0))
    if n == 0:
        return ""
    return num2words(n, lang="de", to="ordinal").capitalize() + "ns, "


def clean_markdown(text: str) -> str:
    """Turn markdown (and emojis, smileys, arrows, LaTeX) into something that can be read aloud."""
    if not text:
        return text
    text = _CODE_FENCE.sub(" ", text)
    text = expand_latex(text)                            # before emphasis: in a formula, _ and * are math
    text = re.sub(r"(?<=[\w)])\s+\*\s+(?=[\w(])", " mal ", text)   # 5 * 3, before the asterisks go
    text = text.replace("`", "")
    text = _EMOJI.sub(_drop_symbol, text)
    text = _SMILEY.sub(_drop_symbol, text)
    text = _ARROW.sub(", ", text)
    text = _TABLE_RULE.sub("", text)                     # |---|---| removed entirely
    text = _TABLE_ROW.sub(lambda m: m.group(1).strip().replace("|", ", "), text)
    text = _HRULE.sub("", text)
    text = _HEADING.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _LINK.sub(r"\1", text)
    # Emphasis BEFORE list numbers: "**3. Wirtschaft:**" starts with
    # asterisks, so the number only reaches the start of the line once they
    # are gone. The other way round, "3." stayed and was read in English.
    text = _EMPHASIS.sub(r"\2", text)
    # Any asterisks still left belong to nothing - e.g. bold text that ran
    # across a chunk boundary. A stray "**" before a line break made the
    # model swallow the next word.
    text = text.replace("*", "")
    text = _BULLET.sub("", text)
    text = _LIST_NUMBER.sub(_enumeration_word, text)     # "2. " -> "Zweitens, "
    text = _SYMBOLS_ONLY.sub("", text)                   # lines without a single letter
    text = _BLANK_LINES.sub("\n", text)                  # blank lines -> one line break
    # Lines are pauses. A heading ("Wirtschaft:") or a list item gets a
    # sentence end, otherwise the model reads it into the next sentence. Line
    # breaks themselves never occurred in training - so they become spaces.
    text = re.sub(r"(?m):[ \t]*$", ".", text)
    text = re.sub(r"(?m)([\wÀ-ɏ)\"“”'])[ \t]*$", r"\1.", text)
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    return text


# ------------------------------------------------------------ math
#
# Chat answers contain formulas: LaTeX ($a^2 + b^2 = c^2$, \frac, \sqrt) or
# Unicode (E = mc², √2, π). Without this stage the model got "a^zwei ...
# Dollar", "5 * 3" became "fünf drei" (asterisk removed) and "10 - 4" became
# "zehn bis vier". Two stages: expand_latex() early, inside clean_markdown -
# before the emphasis rule, which would otherwise eat "_" and "*" as
# markdown - and expand_math_symbols() after the units, so that "85 m²" is
# already "Quadratmeter" there and not "m hoch zwei".
_GREEK = {"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma", "delta": "Delta",
          "epsilon": "Epsilon", "varepsilon": "Epsilon", "zeta": "Zeta", "eta": "Eta",
          "theta": "Theta", "vartheta": "Theta", "iota": "Iota", "kappa": "Kappa",
          "lambda": "Lambda", "mu": "My", "nu": "Ny", "xi": "Xi", "pi": "Pi", "rho": "Rho",
          "sigma": "Sigma", "tau": "Tau", "upsilon": "Ypsilon", "phi": "Phi", "varphi": "Phi",
          "chi": "Chi", "psi": "Psi", "omega": "Omega"}
_LATEX_COMMANDS = {"cdot": " mal ", "times": " mal ", "div": " geteilt durch ", "pm": " plus minus ",
                   "mp": " minus plus ", "approx": " ungefähr ", "neq": " ungleich ", "ne": " ungleich ",
                   "leq": " kleiner gleich ", "le": " kleiner gleich ", "geq": " größer gleich ",
                   "ge": " größer gleich ", "lt": " kleiner als ", "gt": " größer als ",
                   "infty": " unendlich ", "to": " gegen ", "rightarrow": " gegen ",
                   "Rightarrow": ", daraus folgt, ", "implies": ", daraus folgt, ",
                   "iff": " genau dann, wenn ", "in": " Element von ", "sum": " Summe ",
                   "prod": " Produkt ", "int": " Integral ", "lim": " Limes ", "sin": " Sinus ",
                   "cos": " Kosinus ", "tan": " Tangens ", "log": " Logarithmus ",
                   "ln": " natürlicher Logarithmus ", "exp": " e hoch ", "partial": " partiell ",
                   "nabla": " Nabla ", "cdots": " und so weiter ", "ldots": " und so weiter ",
                   "dots": " und so weiter ", "circ": " Grad ", "degree": " Grad ", "percent": " Prozent "}
_LATEX_SPAN = re.compile(r"\$\$(.+?)\$\$|\$(?=\S)([^$\n]+?)(?<=\S)\$(?!\d)|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)
_ROOT_ORDINALS = {2: "zweite", 3: "dritte", 4: "vierte", 5: "fünfte"}


def _latex_fraction(m: re.Match) -> str:
    numerator, denominator = m.group(1).strip(), m.group(2).strip()
    # compound numerator: "..., geteilt durch 2 a" instead of "... durch 2 a"
    compound = bool(re.search(r"\s|[-+=]|\\pm", numerator))
    return f" ({numerator}){', geteilt' if compound else ''} durch ({denominator}) "


def _nth_root(degree: str, radicand: str) -> str:
    # "³".isdigit() is True but int("³") raises ValueError: map superscript digits first
    d = degree.translate(_SUPERSCRIPT_DIGITS).strip()
    word = (_ROOT_ORDINALS.get(int(d), d + '-te') if d.isdecimal() else d)
    return f" {word} Wurzel aus ({radicand}) "


def _latex_to_words(s: str) -> str:
    # limits belong to the operator: "Integral von 0 bis 1", not "Integral 0 hoch 1"
    s = re.sub(r"\\int\s*_\s*\{?([^{}\s^]+)\}?\s*\^\s*\{?([^{}\s]+)\}?", r" Integral von \1 bis \2 ", s)
    s = re.sub(r"\\(sum|prod)\s*_\s*\{([^{}=]+)=([^{}]+)\}\s*\^\s*\{?([^{}\s]+)\}?",
               lambda m: f" {'Summe' if m.group(1) == 'sum' else 'Produkt'} von {m.group(2)} gleich {m.group(3)} bis {m.group(4)} über ", s)
    s = re.sub(r"\\lim\s*_\s*\{([^{}]+?)\s*\\to\s*([^{}]+)\}", r" Limes für \1 gegen \2 von ", s)
    previous = None
    while previous != s:                 # nested: from the inside out
        previous = s
        s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", _latex_fraction, s)
        s = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}",
                   lambda m: _nth_root(m.group(1), m.group(2)), s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r" Wurzel aus (\1) ", s)
        s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname|textrm)\s*\{([^{}]*)\}", r" \1 ", s)
        s = re.sub(r"\^\s*\{([^{}]*)\}", r" hoch (\1) ", s)
        s = re.sub(r"_\s*\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\^\s*\\circ", " Grad ", s)
    s = re.sub(r"\\sqrt\s*(\w)", r" Wurzel aus \1 ", s)
    s = re.sub(r"\^\s*(-?\w)", r" hoch \1 ", s)
    s = re.sub(r"_\s*(\w)", r" \1 ", s)
    s = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)\b", " ", s)
    s = re.sub(r"\\[,;:! ]|\\q?quad\b", " ", s)
    s = re.sub(r"\\([A-Za-z]+)",
               lambda m: _LATEX_COMMANDS.get(m.group(1)) or (" " + _GREEK[m.group(1).lower()] + " "
                                                             if m.group(1).lower() in _GREEK else " " + m.group(1) + " "), s)
    s = s.replace("{", " ").replace("}", " ")
    s = re.sub(r"([A-Za-z])'", r"\1 Strich ", s)                     # f'(x)
    # f(x) -> f von x, Sinus von x. Other letters in front of a parenthesis
    # are more likely a product: n(n+1) is n mal (n plus 1)
    s = re.sub(r"\b(Sinus|Kosinus|Tangens|Logarithmus)\s*\(([^()]*)\)", r"\1 von \2", s)
    s = re.sub(r"\b([fghFGHP](?:\s+Strich)*)\s*\(([^()]*)\)", r"\1 von \2", s)
    s = re.sub(r"\b([a-zA-Z0-9])\s*\(", r"\1 mal (", s)
    s = s.replace("(", " ").replace(")", " ")
    s = re.sub(r"(?<=\w)\s*!", " Fakultät", s)
    s = re.sub(r"\s*=\s*", " gleich ", s)
    s = re.sub(r"\s*\+\s*", " plus ", s)
    s = re.sub(r"\s*[-−]\s*", " minus ", s)
    s = re.sub(r"\s*[*·]\s*", " mal ", s)
    s = re.sub(r"\s*/\s*", " durch ", s)
    s = re.sub(r"\s*<\s*", " kleiner als ", s)
    s = re.sub(r"\s*>\s*", " größer als ", s)
    # "2a", "4ac": speak number and variable separately
    s = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", s)
    s = re.sub(r"\b([a-z])([a-z])\b", r"\1 \2", s)      # ac, mc, dx -> a c, m c, d x
    # Write the digits out right here: otherwise expand_units() would read
    # "x hoch 2 d x" as the quantity "2 d" = two days
    return re.sub(r"\s{2,}", " ", expand_numbers(s)).strip()


def expand_latex(text: str) -> str:
    """Read LaTeX formulas ($...$, $$...$$, \\(...\\), \\[...\\]) as German words."""
    return _LATEX_SPAN.sub(lambda m: " " + _latex_to_words(next(g for g in m.groups() if g is not None)) + " ", text)


_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_MATH_SYMBOLS = {"√": " Wurzel aus ", "∛": " dritte Wurzel aus ", "π": " Pi ", "∞": " unendlich ",
                 "≠": " ungleich ", "÷": " geteilt durch ", "∑": " Summe ", "∫": " Integral ",
                 "∂": " partiell ", "∆": " Delta ", "Δ": " Delta ", "α": " Alpha ", "β": " Beta ",
                 "γ": " Gamma ", "δ": " Delta ", "ε": " Epsilon ", "θ": " Theta ", "λ": " Lambda ",
                 "σ": " Sigma ", "φ": " Phi ", "ω": " Omega ", "Ω": " Omega ", "ρ": " Rho ",
                 "τ": " Tau ", "η": " Eta ", "∈": " Element von ", "∀": " für alle ",
                 "∃": " es gibt "}


def expand_math_symbols(text: str) -> str:
    """Formulas without LaTeX: a², 10⁻³, H₂O, √2, 2^10, 5 + 3, 10 - 4."""
    if not text:
        return text
    # "mc²" is two quantities, not a word - unless it is a unit (cm²)
    table, _ = _units_rule()
    text = re.sub(r"\b([a-z]{2})(?=[²³])",
                  lambda m: m.group(1) if (m.group(1) + "²") in table else " ".join(m.group(1)), text)
    text = re.sub(r"(?<=[\w)])([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)",
                  lambda m: " hoch " + m.group(1).translate(_SUPERSCRIPT_DIGITS).replace("-", "minus ") + " ", text)
    text = re.sub(r"[₀₁₂₃₄₅₆₇₈₉]+", lambda m: " " + m.group(0).translate(_SUBSCRIPT_DIGITS) + " ", text)
    text = re.sub(r"(?<=[\w)])\s*\^\s*(-?[\w.,]+)", r" hoch \1", text)
    for symbol, words in _MATH_SYMBOLS.items():
        text = text.replace(symbol, words)
    # Operators only between operands: "C++" or a dash in the middle of a
    # sentence stay as they are
    text = re.sub(r"(?<=[\w)])\s*\+\s*(?=[\w(])", " plus ", text)
    text = re.sub(r"(\b\d+(?:[.,]\d+)?|\b[a-zA-Z]|\))\s+[-−]\s+(\(|\d|[a-zA-Z](?![a-zA-ZäöüÄÖÜß]))", r"\1 minus \2", text)
    text = re.sub(r"(?<=[\w)])\s+·\s+(?=[\w(])", " mal ", text)
    text = re.sub(r"(?<=[\w)])\s+<\s+(?=[\w(])", " kleiner als ", text)
    text = re.sub(r"(?<=[\w)])\s+>\s+(?=[\w(])", " größer als ", text)
    return text


# ------------------------------------------------------------ units
#
# Physical quantities, currencies, data sizes: "230 V", "5 km/h", "1 h".
# Replaced only directly after a number - in running text "A", "T", "K" or
# "S" are just letters, and "Vitamin C" stays "Vitamin C". The table lives in
# units.json (singular, plural, gender, space required) and, like the
# abbreviations, is reloaded without a restart. Runs BEFORE expand_numbers():
# there the 1 would already be "eins", here it is still known that it has to
# be "eine Stunde".
UNITS_PATH = _HERE / "units.json"
_units_cache: tuple = (None, {}, None)       # ((path, mtime), table, regex)
_CURRENCY_PREFIX = re.compile(r"([$€£¥])\s?(\d+(?:[.,]\d+)*)")


def _units_rule():
    """(table, compiled regex) - reloads units.json as soon as the file changes."""
    global _units_cache
    try:
        key = (str(UNITS_PATH), UNITS_PATH.stat().st_mtime)
    except OSError:
        return {}, None
    if _units_cache[0] == key:
        return _units_cache[1], _units_cache[2]
    try:
        table = json.loads(UNITS_PATH.read_text(encoding="utf-8")).get("einheiten", {})
    except Exception as exc:                           # noqa: BLE001
        log.warning("%s unreadable, keeping the previous table: %s", UNITS_PATH, exc)
        return _units_cache[1], _units_cache[2]
    alternatives = "|".join(re.escape(k) for k in sorted(table, key=len, reverse=True))
    rx = re.compile(r"(?<![\w.,])(-?\d+(?:[.,]\d+)*(?:\s?[-–]\s?\d+(?:[.,]\d+)*)?)"  # number, range
                    r"(\s?)(" + alternatives + r")(?:/(" + alternatives + r"))?"      # unit[/unit]
                    r"(?![\wÀ-ɏ²³])")
    _units_cache = (key, table, rx)
    log.info("units loaded: %d", len(table))
    return table, rx


# "in einer Stunde", "für einen Kilometer", "ein Meter": the article depends
# on the grammatical case, which the preposition in front usually gives away.
# Two-way prepositions are sorted by their most common use with quantities.
_DATIVE_PREPOSITIONS = {"in", "mit", "nach", "vor", "seit", "bei", "von", "aus", "zu", "binnen",
                        "innerhalb", "ab", "unter"}
_ACCUSATIVE_PREPOSITIONS = {"für", "durch", "gegen", "ohne", "um", "über", "auf"}


def _indefinite_article(gender: str, preceding: str) -> str:
    """ein/eine/einen/einem/einer for gender m/f/n, chosen by the word before the number."""
    w = re.search(r"(\w+)\s*$", preceding)
    p = w.group(1).lower() if w else ""
    if p in _DATIVE_PREPOSITIONS:
        return "einer" if gender == "f" else "einem"
    if p in _ACCUSATIVE_PREPOSITIONS:
        return {"f": "eine", "m": "einen"}.get(gender, "ein")
    return "eine" if gender == "f" else "ein"


def expand_units(text: str) -> str:
    """230 V -> 230 Volt, 1 h -> eine Stunde, 50 km/h -> 50 Kilometer pro Stunde."""
    if not text or not any(c.isdigit() for c in text):
        return text
    table, rx = _units_rule()
    if not rx:
        return text
    text = _CURRENCY_PREFIX.sub(r"\2 \1", text)          # $5 -> 5 $

    def replace(m: re.Match) -> str:
        number, space, unit, denominator = m.group(1), m.group(2), m.group(3), m.group(4)
        singular, plural, gender, needs_space = table[unit]
        # "16A", "4K-Monitor", "10 T-Shirts": no ampere, kelvin, tesla
        if needs_space and not space:
            return m.group(0)
        if len(unit) == 1 and m.string[m.end():m.end() + 1] == "-":
            return m.group(0)
        # Body height and lengths with centimetres: "1,80 m" is said
        # "ein Meter achtzig", not "eins Komma acht null Meter"
        if unit == "m" and not denominator and re.fullmatch(r"\d+,\d\d", number):
            whole, cm = (int(x) for x in number.split(","))
            front = _indefinite_article("m", m.string[:m.start()]) if whole == 1 else _cardinal(whole)
            return f"{front} Meter" + (f" {_cardinal(cm)}" if cm else "")
        one = number.lstrip("-") == "1"
        word = (singular if one else plural) + (" pro " + table[denominator][0] if denominator else "")
        if one:
            number = ("minus " if number.startswith("-") else "") + _indefinite_article(gender, m.string[:m.start()])
        return f"{number} {word}"

    return rx.sub(replace, text)


# ------------------------------------------------------------ numbers
#
# The model reads digits in English, sometimes in French: that is how it knows
# them from the base model, and the German training data only contained
# numbers written out as words (num2words, lang="de"). So do the same here.
_MONTHS = "Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember"
# after these words "3." is an ordinal - with this ending
_ORD_ENDING_TEN = {"am", "im", "vom", "zum", "zur", "beim", "dem", "den", "des",
                   "einem", "einen", "einer", "jedem", "jeden"}
_ORD_ENDING_TE = {"der", "die", "das", "ins", "ein", "eine", "jede", "jeder", "jedes"}
_ORDINAL = re.compile(r"\b(" + "|".join(sorted(_ORD_ENDING_TEN | _ORD_ENDING_TE, key=len, reverse=True))
                      + r")\s+(\d{1,3})\.(?=\s)", re.IGNORECASE)
_DATE = re.compile(r"\b(\d{1,2})\.\s*(" + _MONTHS + r")\b")
_CLOCK_TIME = re.compile(r"\b([01]?\d|2[0-4]):([0-5]\d)\b(?:\s*Uhr\b)?")
_MONEY = re.compile(r"\b(\d+),(\d{2})\s*(€|Euro|US-Dollar|Dollar|Pfund|Franken)\b")
_TIMES = re.compile(r"(?<=\d)\s?[x×]\s?(?=\d)")
_THOUSANDS = re.compile(r"\b\d{1,3}(?:\.\d{3})+\b")
_DECIMAL = re.compile(r"\b(\d+),(\d+)\b")
_RANGE = re.compile(r"\b(\d+)\s?[–-]\s?(\d+)\b")
_NEGATIVE = re.compile(r"(?<![\w])-(?=\d)")
_GLUED = re.compile(r"(?<=[A-Za-zÄÖÜäöüß])(?=\d)|(?<=\d)(?=[A-Za-zÄÖÜäöüß])")
_INTEGER = re.compile(r"\b\d+\b")


def _cardinal(n: int) -> str:
    return num2words(n, lang="de")


# Fractions only when they really are fractions: numerator smaller than the
# denominator, denominator at most 10. "24/7" and "9/11" stay pairs of numbers.
_FRACTION = re.compile(r"(?<![\w.,/])(\d{1,2})/(\d{1,2})(?![\w/])")


def _fraction(m: re.Match) -> str:
    numerator, denominator = int(m.group(1)), int(m.group(2))
    if not (0 < numerator < denominator <= 10):
        return m.group(0)
    name = "halb" if denominator == 2 else num2words(denominator, lang="de", to="ordinal").capitalize() + "l"
    return ("ein" if numerator == 1 else _cardinal(numerator)) + " " + name


def _ordinal(n: int, ending_n: bool) -> str:
    return num2words(n, lang="de", to="ordinal") + ("n" if ending_n else "")   # dritte(n)


def _integer_words(s: str) -> str:
    if (len(s) > 1 and s.startswith("0")) or len(s) > 9:
        return " ".join(_cardinal(int(c)) for c in s)      # phone numbers, IDs: digit by digit
    n = int(s)
    if len(s) == 4 and 1100 <= n <= 1999:
        return num2words(n, lang="de", to="year")          # neunzehnhundertneunzig
    return _cardinal(n)


def expand_numbers(text: str) -> str:
    """Write digits out in German. Order matters: first the forms with a
    period, comma or colon, then the bare integers."""
    if not text or not any(c.isdigit() for c in text):
        return text
    text = _ORDINAL.sub(lambda m: f"{m.group(1)} "
                        f"{_ordinal(int(m.group(2)), m.group(1).lower() in _ORD_ENDING_TEN)}", text)
    text = _DATE.sub(lambda m: f"{_ordinal(int(m.group(1)), True)} {m.group(2)}", text)
    text = _CLOCK_TIME.sub(lambda m: ("ein" if int(m.group(1)) == 1 else _cardinal(int(m.group(1))))
                           + " Uhr" + ("" if m.group(2) == "00" else " " + _cardinal(int(m.group(2)))), text)
    # whatever _CLOCK_TIME did not take as a time of day is a ratio or a score
    text = re.sub(r"(?<![\w.,:])(\d+):(\d+)(?![\d:])", r"\1 zu \2", text)
    text = _MONEY.sub(lambda m: ("ein" if m.group(1) == "1" else _cardinal(int(m.group(1))))
                      + " " + ("Euro" if m.group(3) == "€" else m.group(3))
                      + ("" if m.group(2) == "00" else f" {_cardinal(int(m.group(2)))}"), text)
    text = _TIMES.sub(" mal ", text)                                   # 2 x 3 m
    text = _FRACTION.sub(_fraction, text)                              # 3/4 -> drei Viertel
    text = re.sub(r"(?<![\w.,])1(?=\s+Uhr\b)", "ein", text)              # ein Uhr
    text = re.sub(r"(?<![\w.,])1(?=\s+(?:Million|Milliarde|Billion)\b)", "eine", text)
    text = _THOUSANDS.sub(lambda m: _cardinal(int(m.group(0).replace(".", ""))), text)
    text = _DECIMAL.sub(lambda m: f"{_integer_words(m.group(1))} Komma "
                        + " ".join(_cardinal(int(c)) for c in m.group(2)), text)
    text = _RANGE.sub(r"\1 bis \2", text)
    text = _NEGATIVE.sub("minus ", text)
    text = _GLUED.sub(" ", text)
    return _INTEGER.sub(lambda m: _integer_words(m.group(0)), text)


def normalize(text: str) -> str:
    """Prepare German text for speech: the whole pipeline, in this order.

    clean_markdown -> expand_units -> expand_math_symbols -> expand_numbers
    -> expand_abbreviations, then slashes between words and spacing before
    punctuation. The order matters; the comments at each stage say why.
    """
    t = expand_abbreviations(expand_numbers(expand_math_symbols(expand_units(clean_markdown(text)))))
    # A slash between words separates them: the model read
    # "Rolls-Royce/Snecma" as a single word. Only done here, after the units -
    # "km/h" is "Kilometer pro Stunde" by now and is not torn into "km h".
    t = re.sub(r"(?<=[^\W\d_])[ \t]*/[ \t]*(?=[^\W\d_])", " ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return re.sub(r"[ \t]+([,.;:!?])", r"\1", t)   # no space before punctuation


# periods after numbers where SentenceBuffer must not cut
_NUMBER_PERIODS = [re.compile(r"(?m)^[ \t>]*(?:\*\*|__|\*|_)?\d{1,3}[.)]"),
                   re.compile(r"\d\.(?=[ \t]+\S)")]


class SentenceBuffer:
    """Normalizes a token stream without tearing abbreviations apart at chunk boundaries.

    A short tail is always held back; it goes out once more text arrives or
    the utterance ends (flush).
    """

    def __init__(self) -> None:
        self.pending = ""

    def push(self, text: str) -> str:
        """Add text; return the normalized complete sentences, or "" if there are none yet.

        Only whole sentences are released, and a cut never lands inside an
        abbreviation. Merely holding back the last N characters is not
        enough: an abbreviation can straddle the cut. So all abbreviation
        spans are marked first, and a sentence end only counts if its period
        does not belong to one of them.

        Releasing text before a sentence end would gain nothing anyway -
        Breeze only speaks once a sentence is complete.
        """
        self.pending += text
        protected = [(m.start(), m.end())
                     for rx, _ in _abbreviation_rules()
                     for m in rx.finditer(self.pending)]
        # A period after a number is usually not a sentence end either: list
        # number ("**3. Wirtschaft**"), ordinal ("am 3. Oktober"). Cutting
        # there used to split the bold markers, and the numbers could no
        # longer be recognised as ordinals.
        protected += [(m.start(), m.end()) for rx in _NUMBER_PERIODS
                      for m in rx.finditer(self.pending)]
        # Never cut formula blocks ($$ ... $$, \[ ... \]) - they often span
        # several lines, and every line break would otherwise be a cut. A
        # block that is still open protects everything up to the end of the
        # buffer.
        for opening, closing in (("$$", "$$"), ("\\[", "\\]")):
            i = 0
            while (a := self.pending.find(opening, i)) >= 0:
                b = self.pending.find(closing, a + len(opening))
                end = len(self.pending) + 1 if b < 0 else b + len(closing)
                protected.append((a, end))
                i = end
        # this close to the end, more text could still attach
        limit = len(self.pending) - HOLDBACK_CHARS
        cut = -1
        for m in re.finditer(r"[.!?\u2026][\s]|\n", self.pending):
            if m.end() > limit:
                break
            if any(a <= m.start() < b for a, b in protected):
                continue          # the period belongs to an abbreviation, number or formula
            cut = m.end()
        if cut <= 0:
            return ""
        out, self.pending = self.pending[:cut], self.pending[cut:]
        return normalize(out)

    def flush(self, text: str = "") -> str:
        """Release everything still buffered (plus optional final text), normalized."""
        out, self.pending = self.pending + text, ""
        return normalize(out)


if __name__ == "__main__":
    # python german_normalizer.py "Es sind 3.000 m."   or   echo "..." | python german_normalizer.py
    import sys
    print(normalize(" ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read()).strip())
