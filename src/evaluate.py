"""Evaluation: Step 1 (prompt only) vs Step 2 (RAG) on practice/questions.csv, and Step 3 on conversations.csv.

  python -m src.evaluate --dataset /path/to/dataset [--limit N] [--only questions|convs] [--fresh]

Progress is saved to eval_cache.json after every item, so if a daily API quota stops the run you can simply run the
same command again later and it resumes. Writes eval_results.json and eval_summary.md.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

CACHE = Path("eval_cache.json")


def _items(cell):
    return [x.strip() for x in str(cell).split(";") if x.strip() and str(cell) != "nan"]


def check_text(reply, must, must_not):
    """must_include: ' ; ' separates required items; ' / ' separates alternatives (any one suffices)."""
    r = reply.lower()
    miss = [it for it in _items(must) if not any(alt.strip().lower() in r for alt in it.split(" / "))]
    bad = [it for it in _items(must_not) if it.lower() in r]
    return (not miss and not bad), miss, bad


def judge_pair(question, reference, r1, r2, answerable):
    """One grading call for both replies (saves quota). Returns (step1_pass, step2_pass)."""
    from . import llm
    p = (f"Customer question: {question}\nReference answer: {reference}\nAnswerable from documents: {answerable}\n"
         f"Reply A: {r1}\nReply B: {r2}\n\nFor each reply, is it factually correct and consistent with the reference "
         "(same decision, same key numbers/dates)? If answerable is False, the correct behaviour is to say it doesn't have "
         "that information and offer a human; inventing an answer is wrong. Output exactly: A=PASS or FAIL, B=PASS or FAIL")
    t = llm.simple("You are a strict grader.", p, max_tokens=256).upper()
    ma, mb = re.search(r"A\s*=\s*(PASS|FAIL)", t), re.search(r"B\s*=\s*(PASS|FAIL)", t)
    return bool(ma and ma[1] == "PASS"), bool(mb and mb[1] == "PASS")


def save(cache):
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def summarize(cache):
    qs, cs = list(cache["questions"].values()), [t for v in cache["convs"].values() for t in v]
    n = len(qs)
    rec = [d["recall"] for d in qs if d["recall"] is not None]
    out = {"questions": n, "step1_pass": sum(d["judge1"] for d in qs), "step2_pass": sum(d["judge2"] for d in qs),
           "step1_keyfacts": sum(d["keys1"] for d in qs), "step2_keyfacts": sum(d["keys2"] for d in qs),
           "mean_retrieval_recall": round(sum(rec) / len(rec), 3) if rec else None, "details": qs,
           "conversations": len(cache["convs"]), "conv_turns": len(cs),
           "conv_text_ok": sum(t["text_ok"] for t in cs), "conv_tools_ok": sum(t["tools_ok"] for t in cs),
           "conv_escalation_ok": sum(t["escalation_ok"] for t in cs), "conv_details": cs}
    md = ["# Evaluation summary\n", "| Metric | Step 1 (prompt only) | Step 2 (RAG) |", "|---|---|---|",
          f"| Judge-correct answers | {out['step1_pass']}/{n} | {out['step2_pass']}/{n} |",
          f"| Must-include / must-not checks | {out['step1_keyfacts']}/{n} | {out['step2_keyfacts']}/{n} |",
          f"| Mean gold-evidence recall | - | {out['mean_retrieval_recall']} |", "",
          f"Step 3 conversations ({out['conversations']} convs, {out['conv_turns']} turns): text checks "
          f"{out['conv_text_ok']}/{out['conv_turns']}, tool-call checks {out['conv_tools_ok']}/{out['conv_turns']}, "
          f"escalation tickets {out['conv_escalation_ok']}/{out['conv_turns']}."]
    return out, "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    ap.add_argument("--dataset", default=os.environ.get("ZENTRO_DATASET"),
                    help="extracted dataset folder (default: ZENTRO_DATASET from .env / environment)")
    ap.add_argument("--limit", type=int, help="only the first N questions and ~N/3 conversations")
    ap.add_argument("--only", choices=["questions", "convs"])
    ap.add_argument("--skip-convs", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore saved progress")
    ap.add_argument("--redo-failed", action="store_true",
                    help="re-run only items that failed last time (keeps Step 1 answers); use after changing the agent")
    a = ap.parse_args()
    if not a.dataset:
        sys.exit("Dataset path missing: set ZENTRO_DATASET in .env or pass --dataset <folder>")
    os.environ["ZENTRO_DATASET"] = a.dataset
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config
    from src.agent import Agent, baseline_answer, get_retriever
    cache = {"questions": {}, "convs": {}}
    if CACHE.exists() and not a.fresh:
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    R = get_retriever()
    reuse = {}
    if a.redo_failed:
        for qid, d in list(cache["questions"].items()):
            if not (d["judge2"] and d["keys2"]):
                reuse[qid] = cache["questions"].pop(qid)["step1"]
        for cid, turns in list(cache["convs"].items()):
            if not all(t["text_ok"] and t["tools_ok"] and t["escalation_ok"] for t in turns):
                del cache["convs"][cid]
        save(cache)
    stopped = None
    try:
        if a.only != "convs":
            q = pd.read_csv(config.PRACTICE_DIR / "questions.csv", dtype=str)
            q = q.head(a.limit) if a.limit else q
            for _, row in q.iterrows():
                if row.id in cache["questions"]:
                    continue
                b = reuse.get(row.id) or baseline_answer(row.question)
                ag = Agent(retriever=R)
                r2 = ag.respond(row.question)
                got = {c["id"] for c in ag.last_chunks}
                gold = _items(row.gold_evidence)
                recall = (sum(g in got for g in gold) / len(gold)) if gold else None
                j1, j2 = judge_pair(row.question, row.reference_answer, b, r2, row.answerable)
                cache["questions"][row.id] = dict(
                    id=row.id, q=row.question, step1=b, step2=r2, judge1=j1, judge2=j2, recall=recall,
                    keys1=check_text(b, row.must_include, row.must_not_include)[0],
                    keys2=check_text(r2, row.must_include, row.must_not_include)[0],
                    gold=gold, retrieved=sorted(got),
                    tools=[(t["tool"], t["args"]) for t in ag.tool_log])
                save(cache)
                print(row.id, "S1", j1, "S2", j2, "recall", recall, flush=True)
        if a.only != "questions" and not a.skip_convs:
            c = pd.read_csv(config.PRACTICE_DIR / "conversations.csv", dtype=str)
            ids = list(c.conversation_id.unique())
            ids = ids[:max(1, a.limit // 3)] if a.limit else ids
            for cid in ids:
                if cid in cache["convs"]:
                    continue
                ag, res = Agent(retriever=R), []
                g = c[c.conversation_id == cid]
                for _, t in g.sort_values("turn", key=lambda s: s.astype(int)).iterrows():
                    before = len(ag.tool_log)
                    reply = ag.respond(t.customer_message)
                    ok, miss, bad = check_text(reply, t.must_include, t.must_not_include)
                    names = [x["tool"] for x in ag.tool_log[before:]]
                    want = re.findall(r"(lookup_order|create_ticket)", str(t.expected_tool_calls))
                    tok = all(w in names for w in want)
                    esc = (t.expected_decision != "escalate") or (ag.ticket is not None)
                    res.append(dict(conv=cid, turn=t.turn, msg=t.customer_message, reply=reply, text_ok=ok, missing=miss,
                                    forbidden=bad, tools_called=names, tools_ok=tok, escalation_ok=esc))
                    print(cid, t.turn, "text", ok, "tools", tok, "esc", esc, flush=True)
                cache["convs"][cid] = res      # a conversation is saved only when complete
                save(cache)
    except Exception as e:  # quota or network: keep what we have, tell the user how to resume
        stopped = f"{type(e).__name__}: {str(e)[:300]}"
    out, md = summarize(cache)
    Path("eval_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    Path("eval_summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    if stopped:
        print(f"\nSTOPPED EARLY: {stopped}\nProgress is saved. Run the same command again later (or after the quota resets) to resume.")


if __name__ == "__main__":
    main()
