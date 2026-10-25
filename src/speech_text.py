"""Turn what speech-to-text writes ("double six four nine ...") into what the agent needs ("6649 ...")."""
import re

_EN = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_HI = {"shunya": "0", "ek": "1", "do": "2", "teen": "3", "char": "4", "chaar": "4", "paanch": "5", "panch": "5", "chhe": "6",
       "chhah": "6", "che": "6", "saat": "7", "aath": "8", "nau": "9", "nav": "9"}
_DEV = {"शून्य": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पांच": "5", "पाँच": "5", "छह": "6", "छः": "6", "छे": "6",
        "सात": "7", "आठ": "8", "नौ": "9"}
DIGITS = {**_EN, **_HI, **_DEV}
MULT = {"double": 2, "dabal": 2, "triple": 3, "treble": 3, "डबल": 2, "ट्रिपल": 3}
_DEVNUM = str.maketrans("०१२३४५६७८९", "0123456789")
_SEP = re.compile(r"^[\s,\-\u2013.]+$")


def normalize_numbers(text):
    """Convert runs of 2+ spoken digits to numerals. Single words ('two coupons', 'one question') are left alone."""
    text = text.translate(_DEVNUM)
    toks = re.findall(r"[\w\u0900-\u097f]+|[^\w\u0900-\u097f]+", text)
    out, run, i = [], [], 0          # run = list of (token_index_start, token_index_end, digits)

    def flush():
        nonlocal run
        if run and sum(len(d) for _, _, d in run) >= 2:
            first, last = run[0][0], run[-1][1]
            repl = "".join(d for _, _, d in run)
            del_range.append((first, last, repl))
        run = []

    del_range = []
    n = len(toks)
    while i < n:
        t = toks[i].lower()
        if t in MULT:                # "double six"
            j = i + 1
            while j < n and _SEP.match(toks[j]):
                j += 1
            if j < n and toks[j].lower() in DIGITS:
                run.append((i, j, DIGITS[toks[j].lower()] * MULT[t]))
                i = j + 1
                continue
        if t in DIGITS:
            run.append((i, i, DIGITS[t]))
            i += 1
            continue
        if run and _SEP.match(toks[i]):
            i += 1
            continue
        flush()
        i += 1
    flush()
    for first, last, repl in reversed(del_range):
        toks[first:last + 1] = [repl]
    return "".join(toks)


_EN_WORDS = set("the is my to can how what does i a of you are for it this not have me when will would should could "
                "there was were with and or in on at be been if so but".split())


def language_hint(text):
    """Only for voice: pin the reply language when the caller is clearly English or Devanagari Hindi."""
    from .voice import HINGLISH
    if re.search(r"[\u0900-\u097f]", text):
        return "The customer is speaking Hindi: reply in Hindi using Devanagari script."
    words = re.findall(r"[a-z']+", text.lower())
    if sum(w in _EN_WORDS for w in words) >= 2 and not any(w in HINGLISH for w in words):
        return "The customer is speaking English: reply in English."
    return ""
