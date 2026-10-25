"""Voice layer: Gnani STT in, Gnani TTS out. Gnani is the ONLY speech service used."""
import os
import re
import struct
import wave
from pathlib import Path

from . import config

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
MONTHS = "January February March April May June July August September October November December".split()


def num_words(n):
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    if n < 1000: return ONES[n // 100] + " hundred" + (" " + num_words(n % 100) if n % 100 else "")
    if n < 100000: return num_words(n // 1000) + " thousand" + (" " + num_words(n % 1000) if n % 1000 else "")
    return num_words(n // 100000) + " lakh" + (" " + num_words(n % 100000) if n % 100000 else "")


def ordinal(d):
    return f"{d}{'th' if 10 <= d % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')}"


def normalize_for_speech(text):
    """Write for the ear: dates with year, order IDs in groups of four, amounts in rupees."""
    t = re.sub(r"[*_`#>]+", "", text)
    t = re.sub(r"(\d{4})-(\d{2})-(\d{2})",
               lambda m: f"{ordinal(int(m[3]))} of {MONTHS[int(m[2]) - 1]} {m[1]}", t)
    t = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS) + r")(?:\s+(\d{4}))?\b",
               lambda m: f"{ordinal(int(m[1]))} of {m[2]} {m[3] or 2026}", t)
    t = re.sub(r"\b\d{8,12}\b", lambda m: " ".join(m[0][i:i + 4] for i in range(0, len(m[0]), 4)), t)
    t = re.sub(r"(?:₹|Rs\.?\s?|INR\s?)\s?([\d,]+)",
               lambda m: num_words(int(m[1].replace(",", ""))) + " rupees", t)
    t = re.sub(r"\b(\d{1,3}(?:,\d{3})*|\d+)\s*(?:rupees)\b",
               lambda m: num_words(int(m[1].replace(",", ""))) + " rupees" if m[1].replace(",", "").isdigit() else m[0], t)
    return re.sub(r"\s+", " ", t).strip()


SCRIPTS = [("\u0900", "\u097f", "hi-IN"), ("\u0980", "\u09ff", "bn-IN"), ("\u0a00", "\u0a7f", "pa-IN"),
           ("\u0a80", "\u0aff", "gu-IN"), ("\u0b80", "\u0bff", "ta-IN"), ("\u0c00", "\u0c7f", "te-IN"),
           ("\u0c80", "\u0cff", "kn-IN"), ("\u0d00", "\u0d7f", "ml-IN")]
HINGLISH = set("hai hain kya aap mein main mujhe nahi nahin kar karo karna kijiye chahiye hoga hogi "
               "aapka aapki ke ki ka se ko ho apna dhanyavad shukriya abhi order ka".split())


def tts_language(text):
    for lo, hi, code in SCRIPTS:
        if sum(lo <= ch <= hi for ch in text) > 3:
            return code
    words = re.findall(r"[a-z]+", text.lower())
    if words and sum(w in HINGLISH for w in words) / len(words) > 0.12:
        return "hi-en"
    return "en-IN"


class Voice:
    def __init__(self):
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient
        self.stt = GnaniSTTClient()   # reads GNANI_API_KEY
        self.tts = GnaniTTSClient()

    # --- STT: REST accepts clips up to 60 s, so long WAVs are split into 55 s pieces ---
    def _split_wav(self, path, seg=55):
        try:
            with wave.open(str(path)) as w:
                n, rate = w.getnframes(), w.getframerate()
                if n / rate <= 58:
                    return [path]
                out = []
                for i, start in enumerate(range(0, n, seg * rate)):
                    w.setpos(start)
                    frames = w.readframes(min(seg * rate, n - start))
                    p = Path(str(path) + f".part{i}.wav")
                    with wave.open(str(p), "wb") as o:
                        o.setparams(w.getparams()); o.writeframes(frames)
                    out.append(p)
                return out
        except Exception:
            return [path]

    def listen(self, audio_path, language_code=None):
        codes = language_code or config.STT_LANGS
        parts, texts = self._split_wav(audio_path), []
        for p in parts:
            try:
                r = self.stt.transcribe(str(p), language_code=codes)
            except Exception:
                r = self.stt.transcribe(str(p), language_code="en-IN")  # fall back to single language
            texts.append((r.get("transcript") or "").strip())
            if p != audio_path:
                Path(p).unlink(missing_ok=True)
        return " ".join(t for t in texts if t)

    @staticmethod
    def _fix_wav_header(path):
        """Gnani streams WAV with 0xFFFFFFFF length fields; some players show that as blank/0:00. Rewrite real sizes."""
        b = Path(path).read_bytes()
        i = b.find(b"data")
        if not b.startswith(b"RIFF") or i < 0:
            return
        pcm = b[i + 8:]
        b = bytearray(b[:i + 4] + struct.pack("<I", len(pcm)) + pcm)
        b[4:8] = struct.pack("<I", len(b) - 8)
        Path(path).write_bytes(bytes(b))

    def speak(self, text, out_file):
        spoken = normalize_for_speech(text)
        lang = tts_language(spoken)
        speed = float(os.getenv("GNANI_TTS_SPEED", "0.9"))
        attempts = [(config.TTS_MODEL, lang), ("timbre-v3.0", lang), (config.TTS_MODEL, "en-IN")]
        err = None
        for model, lg in attempts:
            try:
                kw = {"speed": speed} if model == "timbre-v2.5" else {}
                self.tts.synthesize(spoken, voice=config.TTS_VOICE, model=model, language=lg,
                                    output_file=str(out_file), **kw)
                if Path(out_file).stat().st_size < 4000:
                    raise RuntimeError("TTS returned empty audio")
                self._fix_wav_header(out_file)
                print(f"   tts: model={model} lang={lg} speed={speed}")
                return spoken
            except Exception as e:  # try the next option; never fall back to a non-Gnani service
                err = e
        raise RuntimeError(f"Gnani TTS failed: {err}")


REPROMPT = {"en": "Sorry, I didn't catch that. Could you say it again?",
            "hi": "Maaf kijiye, mujhe sunai nahi diya. Kya aap dobara bol sakte hain?"}
