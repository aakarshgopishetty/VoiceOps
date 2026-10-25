#!/usr/bin/env python3
"""Record a test clip from your microphone:   python record.py q1 [seconds]   ->  clips/q1.wav
Name turns of one conversation  <conv>__<turn>  (e.g. call1__1, call1__2) so run.py treats them as one call."""
import os
import sys
import wave

import sounddevice as sd

os.makedirs("clips", exist_ok=True)
if len(sys.argv) < 2:
    sys.exit("usage: python record.py <name> [seconds]")
name, secs, rate = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 7, 16000
print("Speak now...")
audio = sd.rec(int(secs * rate), samplerate=rate, channels=1, dtype="int16")
sd.wait()
with wave.open(f"clips/{name}.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(audio.tobytes())
print(f"saved clips/{name}.wav")
