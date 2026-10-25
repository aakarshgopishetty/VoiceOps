#!/usr/bin/env python3
"""Batch run: folder of customer audio -> STT -> agent -> TTS. Saves text, audio and retrieved chunks.

  python run.py --audio_dir ./test_audio --out_dir ./outputs

Files named  <conv>__<turn>.wav  are treated as turns of ONE conversation (ordered by turn);
every other file is its own single-turn conversation.
"""
import argparse
import json
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

from src import config

AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--dataset", help="path to extracted dataset (or set ZENTRO_DATASET)")
    ap.add_argument("--stt_lang", help="override Gnani STT language_code, e.g. hi-IN or en-IN,hi-IN")
    a = ap.parse_args()
    if a.dataset:
        import os; os.environ["ZENTRO_DATASET"] = a.dataset
        config.DATASET = Path(a.dataset); config.KB_DIR = config.DATASET / "kb"
        config.DATA_DIR = config.DATASET / "data"
    config.ensure_dataset()

    from src.agent import Agent, get_retriever
    from src.voice import Voice, REPROMPT
    voice, retriever = Voice(), get_retriever()
    files = sorted(p for p in Path(a.audio_dir).iterdir() if p.suffix.lower() in AUDIO)
    if not files:
        sys.exit(f"No audio files found in {a.audio_dir}")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    convs = defaultdict(list)
    for p in files:
        m = re.match(r"^(.*)__(\d+)$", p.stem)
        convs[m[1] if m else p.stem].append((int(m[2]) if m else 0, p))

    results = []
    for cid, turns in convs.items():
        agent = Agent(channel="voice", retriever=retriever)
        for _, p in sorted(turns):
            d = out / p.stem; d.mkdir(exist_ok=True)
            rec = {"file": p.name, "conversation": cid}
            try:
                heard = voice.listen(p, a.stt_lang)
                rec["transcript"] = heard
                if not heard.strip():
                    reply = REPROMPT["en"]; rec["retrieved"] = []
                else:
                    reply = agent.respond(heard)
                    rec["retrieved"] = agent.last_chunks
                    rec["search_query"] = agent.last_query
                    rec["tool_calls"] = agent.tool_log[-6:]
                rec["reply"] = reply
                rec["spoken_text"] = voice.speak(reply, d / "reply.wav")
                rec["audio"] = str((d / "reply.wav").relative_to(out))
            except Exception as e:  # never crash the batch: always leave a record
                rec["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            (d / "transcript.txt").write_text(rec.get("transcript", ""), encoding="utf-8")
            (d / "reply.txt").write_text(rec.get("reply", ""), encoding="utf-8")
            (d / "sources.json").write_text(json.dumps(rec.get("retrieved", []), ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(rec)
            print(f"[{p.name}] heard: {rec.get('transcript','')!r}\n   reply: {rec.get('reply','')!r}")
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. {len(results)} files -> {out}/results.json")


if __name__ == "__main__":
    main()
