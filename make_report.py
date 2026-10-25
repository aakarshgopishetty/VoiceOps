#!/usr/bin/env python3
"""Builds report.pdf (max 4 pages). Run after the final evaluation:
     python make_report.py ["Team name"]
Uses eval_results.json (written by `python -m src.evaluate`) for the final numbers."""
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

team = sys.argv[1] if len(sys.argv) > 1 else ""
# Measured on the first full run of the practice set, before the fixes described below.
V1 = dict(n=52, s1=17, s2=38, k1=13, k2=32, rec=0.918, turns=96, text=79, tools=94, esc=95)

p = Path("eval_results.json")
E = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

S = getSampleStyleSheet(); H = S["Heading2"]; B = S["BodyText"]; B.leading = 13
f = [Paragraph("VoiceOps 2026 - ZentroMart voice support agent" + (f" ({team})" if team else ""), S["Title"])]


def sec(title, body):
    f.extend([Paragraph(title, H), Paragraph(body, B), Spacer(1, 3)])


def table(rows):
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                           ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    f.extend([t, Spacer(1, 6)])


sec("Architecture", "Customer audio -> Gnani STT -> agent (LLM + RAG + tools) -> text normalised for the ear -> Gnani TTS -> spoken reply. "
    "Gnani is the only speech service used. The LLM is Google Gemini (any hosted LLM was allowed); retrieval, tools and the "
    "conversation logic are plain Python.")
sec("Step 1 - prompt-only LLM", "A system prompt makes the model ZentroMart's AI support assistant: it introduces itself as an AI, "
    "replies in the customer's language and script, and keeps answers to 2-3 short sentences. It has seen no documents, so it "
    "guesses policy details (return windows, refund times, warranty) and often sounds confident while being wrong.")
sec("Step 2 - grounded in the documents (RAG)", "The 21 knowledge-base files are split by [SECTION-ID], keeping doc_type, status and "
    "effective_from. Product, pincode and holiday rows are indexed too (gold evidence cites them). Retrieval is hybrid: BM25 plus "
    "character n-gram TF-IDF over an LLM-written English search query (handles Hindi, Hinglish and speech-recognition errors), a "
    "small synonym map, and document precedence (policy > agent guide > help > FAQ > marketing; the superseded returns v3 is "
    "demoted but still reachable for pre-1-August electronics). Product rows pull in the policy for their category, and the core "
    "rules (precedence, business days, escalation, verification, style) are always in context. The model answers only from the "
    "excerpts, says it has no information when the documents are silent, and every answer saves the chunk IDs it used.")
sec("Step 3 - conversation", "The agent keeps the dialogue and the verified-order state. Code, not the model, enforces identity "
    "checks: a verification value must be one the customer actually said, at most two attempts, then order details stay locked. "
    "create_ticket results are the only basis for telling a customer a human will follow up. A date_math tool does the "
    "business-day arithmetic (Mon-Sat minus holidays.csv; today = 20 October 2026) and calendar-day return windows, so the model "
    "never calculates dates itself.")
sec("Step 4 - voice", "Gnani STT (en-IN,hi-IN; clips over 60 s are split) feeds the agent; the reply is normalised for the ear "
    "(dates with the year, order IDs in groups of four, amounts as rupees) and spoken with Gnani TTS (timbre-v2.5, voice Kaveri; "
    "language chosen from the reply: English, Hinglish or the Indian script used). Empty or unclear audio gets a polite "
    "'please say that again'. run.py processes a whole folder of audio and saves transcript, reply text, reply audio and the "
    "retrieved chunks for each file.")

f.append(Paragraph("Step 1 vs Step 2 - results on the 52 practice questions", H))
table([["Metric", "Step 1 (prompt only)", "Step 2 (RAG, first run)"],
       ["Answers judged correct", f"{V1['s1']}/{V1['n']}", f"{V1['s2']}/{V1['n']}"],
       ["Must-include / must-not checks", f"{V1['k1']}/{V1['n']}", f"{V1['k2']}/{V1['n']}"],
       ["Gold-evidence recall", "-", str(V1["rec"])]])
f.append(Paragraph(f"Step 3 on the 23 practice conversations ({V1['turns']} turns), first run: text checks {V1['text']}/{V1['turns']}, "
                   f"correct tool calls {V1['tools']}/{V1['turns']}, handovers that created a ticket {V1['esc']}/{V1['turns']}.", B))
f.append(Spacer(1, 6))
if E:
    n, t = E["questions"], E["conv_turns"]
    f.append(Paragraph("Final results (after the fixes below)", H))
    table([["Metric", "Step 1 (prompt only)", "Step 2 (final)"],
           ["Answers judged correct", f"{E['step1_pass']}/{n}", f"{E['step2_pass']}/{n}"],
           ["Must-include / must-not checks", f"{E['step1_keyfacts']}/{n}", f"{E['step2_keyfacts']}/{n}"],
           ["Gold-evidence recall", "-", str(E["mean_retrieval_recall"])]])
    f.append(Paragraph(f"Conversations (final): text checks {E['conv_text_ok']}/{t}, tool calls {E['conv_tools_ok']}/{t}, "
                       f"handover tickets {E['conv_escalation_ok']}/{t}.", B))
    f.append(Spacer(1, 6))

sec("What went wrong in the first run", "Retrieval was rarely the problem (recall 0.92); the model's use of what it retrieved was. "
    "(1) Refund 'by which date' questions: it quoted the day range but never computed the date, so nine questions failed. "
    "(2) Questions the documents do not answer (midnight delivery, student discount, store pickup): it asserted 'we do not offer "
    "that' instead of saying it had no information and offering a colleague. (3) It asked for an order ID on general policy "
    "questions it could answer directly. (4) Handover rules were skipped when the caller was unverified (laptop over Rs 10,000, an "
    "order the customer never placed). (5) In conversations it did not compute the last return date or mention the pickup fee, "
    "ignored the order's payment method when giving refund timing, and did not escalate a refund that was months overdue. "
    "(6) Smaller issues: spoken-style dates and number words instead of digits, Latin-script replies to Devanagari Hindi, and no "
    "apology when refusing a voucher.")
sec("What I changed", "The prompt now requires date_math for every date, a fixed 'I don't have information about that, I can "
    "connect you to a colleague' reply when the documents are silent, direct answers to policy questions without an order ID, "
    "an immediate ticket whenever ESC-1 applies (verification not required), order-specific refund timelines from the payment "
    "method, and replies in the customer's own script with dates and numbers as digits (a separate step converts text for "
    "speech). Retrieval gained synonyms for cash on delivery, refund timing, damage and out-of-country delivery. The evaluation "
    "script saves progress and can re-run only failed items, which made it practical to iterate within free-tier API quotas.")
sec("What I would fix next", "Dense multilingual embeddings with a reranker for paraphrased and ASR-garbled questions; a second "
    "retrieval pass when the first finds nothing relevant; spoken order-ID confirmation ('did you say 4 5 0 1...?'); voice-activity "
    "detection and barge-in for real calls; testing with more accents and noisy audio; and checking the LLM's tool-call "
    "accuracy on smaller models when quota is tight.")
SimpleDocTemplate("report.pdf", pagesize=A4, leftMargin=44, rightMargin=44, topMargin=40, bottomMargin=40).build(f)
pages = Path("report.pdf").read_bytes().count(b"/Type /Page\n") + Path("report.pdf").read_bytes().count(b"/Type /Page ")
print("report.pdf written", "(final results included)" if E else "(first-run results only - run the evaluation, then rerun this)")
