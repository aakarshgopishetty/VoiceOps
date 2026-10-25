"""ZentroMart support agent: Step 1 (prompt only), Step 2 (RAG), Step 3 (conversation + tools)."""
import json
import re

from . import config, datetools, llm, refunds
from .speech_text import language_hint, normalize_numbers
from .kb import Retriever

BASE_PROMPT = """You are ZentroMart's AI customer-support assistant for an Indian online store selling groceries and electronics.
Today's date is 20 October 2026 (use this, never your own clock).
Your first reply in a conversation must say you are an AI assistant (in the customer's language).
LANGUAGE: reply in the language AND script the customer used. English -> English. Hindi written in Devanagari -> Hindi in Devanagari. Hinglish (Hindi in Latin letters) -> Hinglish in Latin letters. Same for other Indian languages. Switch if they switch.
STYLE: warm, brief, direct: 2-3 short sentences, answer first. No markdown, bullets or emojis. Write dates like "22 October 2026" and use digits for numbers and amounts (3 attempts, 7 days, ₹79); a separate step converts text for speech. When you must decline or cannot do what was asked, start with a brief apology (sorry / maaf kijiye / माफ़ कीजिए)."""

RAG_RULES = """
GROUNDING
- Answer ONLY from the <kb_context> excerpts, the verified order record, and tool results. Never invent a policy, date, fee, amount or timeline.
- Precedence when excerpts disagree: current policy > agent guide > help article > FAQ > marketing. Superseded documents apply only to the cases the current policy says they still cover (e.g. electronics delivered before 1 August 2026 keep the old window).
- Combine sources when needed (policy + product row + order record) and state the concrete answer first. Always give the actual numbers from the policy (days, hours, attempts, fees). When you offer several options (e.g. wallet or bank transfer), give the time for EACH option. Copy numeric ranges exactly as the excerpt states them (for example "5 to 7 business days" or "7 to 10 calendar days"); never shorten a range or drop a number.
- Use these English terms inside Hindi/Hinglish sentences when they apply: "quality check", "ticket", "colleague", "refuse the parcel", "reverse pickup", "RRN".

ANSWER DIRECTLY
- General policy questions need NO order ID and NO verification. Answer them straight away from the policy and the product row (category, price). Ask for an order ID only when the answer truly depends on that specific order. If you must ask, ask ONE question.
- If the customer gives a date ("delivered the day before yesterday", "ordered on 10 October", "refund started on 17 October"), use it. Sale rule: orders placed 1-15 October 2026 in clothing, footwear and bags have a 30-day return window (RET-8).
- If the customer asks about a refund without saying when it started or how they paid, ask how they paid or for the order ID (one question).

DATES (use tools, never do date arithmetic in your head)
- "By which date should I get my refund?" -> call refund_window(payment_method, start_date). It reads the REF-2 timeline from the policy and returns the earliest and latest dates; answer with its "answer" text ("between X and Y"). For instant methods say the refund arrives within 2 hours. Do not compute refund dates yourself.
- Return deadline = delivery date + window days (date_math add_calendar_days). Tell the customer the last date to return, the reverse pickup fee (or that it is waived for Plus members), and when the refund starts.
- Other date questions: date_math (business days = Mon-Sat excluding warehouse holidays).

WHEN THE DOCUMENTS DO NOT COVER IT
- If the excerpts do not say, do NOT say "we do not offer it" or guess. Say exactly this in the customer's language: "Sorry, I don't have information about that. I can connect you to a colleague who can help." (Hindi: "माफ़ कीजिए, मेरे पास इस बारे में जानकारी नहीं है। मैं आपको किसी सहयोगी से जोड़ सकती हूँ।" Hinglish: "Sorry, iske baare mein mere paas jaankari nahi hai. Main aapko ek colleague se connect kar sakti hoon.") Do not create a ticket unless the customer accepts.

ORDER DATA AND VERIFICATION
- Before sharing ANYTHING about a specific order you need its order ID and ONE verification value: last 4 digits of the registered phone OR the delivery pincode. Ask for one thing at a time.
- The system looks orders up automatically when the customer gives an order ID or a verification value. Follow the <system_note> at the top of the customer message exactly; do not call lookup_order again for the same value. Never use a value the customer did not say. Never reveal the expected value, phone, email or address.
- Verification failed once: say the details don't match our registered details and ask them to try again. Failed twice: say sorry, you cannot share details of this order because it could not be verified against the registered details, and offer general policy help or a colleague.
- Remember the order ID and the issue from earlier turns; resolve follow-ups ("how long will that take?") about the same order without asking again.
- Refund questions about a specific order: use the order's payment/refund method and say when the refund starts (Plus members: at pickup; others: after the quality check passes). The verified order includes "computed_checks" with the expected arrival date and whether the refund is overdue.
- A shipped order cannot be cancelled (CAN-2): decline politely (sorry) and give the options: refuse the parcel at the door (cash on delivery: nothing is charged; prepaid: the refund starts once the parcel reaches the warehouse, usually 7 to 10 calendar days after the last delivery attempt, REF-1), or accept the delivery and return the item within its return window if its category is returnable. Do NOT create a ticket for this.

HANDOVER (ESC-1)
- Hand over ONLY when: (1) the customer reports an item damaged, defective or wrong AND its price is above ₹10,000; (2) an order, payment or account activity the customer says they did not make; (3) legal action; (4) contacts_last_7_days >= 2 (third contact); (5) the customer asks where a refund is (or when it will arrive) AND computed_checks says the refund-overdue rule applies (also give the RRN) - do not escalate merely because the order was just verified; (6) the customer asks for a human; (7) product safety risk. Do NOT escalate merely because an order contains an expensive item, or because the customer is cancelling or asking a normal question.
- When it applies, call create_ticket IMMEDIATELY (verification is not required) with a summary of: issue, intent, order ID if known, product and price, what the customer told you, what you checked, and the reason. Never approve replacements or refunds yourself. Then say a colleague (e.g. the Priority Resolution team for items over ₹10,000, the account-security team for activity the customer did not make) will contact them.
- Politely refuse anything the policy does not allow (no compensation, goodwill vouchers or exceptions). Only say a ticket was created if create_ticket returned created=true. Never claim an action was done unless a tool result confirms it."""

VOICE_RULES = """
VOICE CHANNEL: the reply will be spoken aloud. Keep it to 2-3 short sentences in plain words. If the audio seems unclear or nonsensical, ask the customer to repeat. Write dates and numbers normally; they are converted for speech afterwards."""

TOOLS = [
    {"name": "lookup_order",
     "description": "Look up an order. Without verification_value it only opens the identity check. With the customer's stated last-4 phone digits or 6-digit delivery pincode it returns order details.",
     "input_schema": {"type": "object", "properties": {
         "order_id": {"type": "string"}, "verification_value": {"type": "string"}},
         "required": ["order_id"]}},
    {"name": "create_ticket",
     "description": "Create a human-handover ticket. Must be called before telling the customer a human will follow up.",
     "input_schema": {"type": "object", "properties": {
         "summary": {"type": "string", "description": "issue, intent, order id, what customer said, what you checked, reason for escalation"},
         "order_id": {"type": "string"}, "reason": {"type": "string"}, "intent": {"type": "string"}},
         "required": ["summary", "reason"]}},
    {"name": "refund_window",
     "description": "Earliest and latest date a refund reaches the customer, from the REF-2 timeline for their payment method. Always use this for 'by which date' refund questions.",
     "input_schema": {"type": "object", "properties": {
         "payment_method": {"type": "string", "description": "as the customer said or the order shows: UPI, credit card, debit card, net banking, wallet, cod_wallet, cod_bank"},
         "start_date": {"type": "string", "description": "date the refund was initiated, ISO YYYY-MM-DD"}},
         "required": ["payment_method", "start_date"]}},
    {"name": "date_math",
     "description": "Deterministic date arithmetic. Dates are ISO (YYYY-MM-DD); a/b default to today (2026-10-20).",
     "input_schema": {"type": "object", "properties": {
         "op": {"type": "string", "enum": ["add_business_days", "add_calendar_days", "business_days_between", "calendar_days_between"]},
         "a": {"type": "string"}, "b": {"type": "string"}, "n": {"type": "integer"}},
         "required": ["op"]}},
]

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        config.ensure_dataset()
        _retriever = Retriever()
    return _retriever


def baseline_answer(question):
    """Step 1: LLM with a system prompt only. No documents, no tools."""
    return llm.simple(BASE_PROMPT, question, max_tokens=1024, temperature=0.2)


def digits(s):
    return re.sub(r"\D", "", str(s))


class Agent:
    def __init__(self, channel="chat", retriever=None):
        config.ensure_dataset()
        import tools as mock_tools  # noqa: from the dataset folder
        self.tools = mock_tools
        self.retriever = retriever or get_retriever()
        self.channel = channel
        self.history = []          # google-genai Content messages
        self.user_text = []        # everything the customer said (to validate verification values)
        self.order = None          # verified order record (this conversation only)
        self.order_id = None
        self.failed = {}           # order_id -> failed verification count
        self.pending = None        # order id whose identity check is open
        self.turn_cache = {}
        self.tool_log = []
        self.last_chunks = []
        self.ticket = None

    # ---- retrieval -----------------------------------------------------
    def _search_query(self, text):
        mode = config.REWRITE
        if mode == "never":
            return text
        if mode == "auto" and len(self.user_text) == 1 and text.isascii():
            from .voice import HINGLISH
            if not any(w in HINGLISH for w in text.lower().split()):
                return text                 # plain English first turn: skip the extra LLM call
        ctx = " | ".join(self.user_text[-3:-1])
        prompt = ("Rewrite the customer's latest message as a short ENGLISH search query (max 25 words) "
                  "for a support policy knowledge base. Translate Hindi/Hinglish, fix speech-recognition "
                  "errors (e.g. 'zero zero one two' in a SKU -> ELE-0012 style codes), resolve follow-ups "
                  "using earlier messages, and include product names/SKUs, topic (return, refund, cancel, "
                  f"delivery, warranty, damage, coupon, plus, cod...) and customer situation.\nEarlier: {ctx}\n"
                  f"Latest: {text}\nQuery:")
        try:
            q = llm.simple("You write search queries.", prompt, max_tokens=256)
            return q or text
        except Exception:
            return text

    def retrieve(self, text):
        q = self._search_query(text)
        hits = self.retriever.search(f"{q} {text}")
        # add catalog rows for the items on the verified order
        if self.order:
            have = {c.cid for c, _ in hits}
            by = {c.cid: c for c in self.retriever.chunks}
            for it in self.order.get("items", []):
                cid = f"products.csv:{it.get('sku')}"
                if cid in by and cid not in have:
                    hits.append((by[cid], 0.0))
        self.last_chunks = [{"id": c.cid, "title": c.title, "doc_type": c.doc_type, "status": c.status,
                             "effective_from": c.effective_from, "score": round(s, 4), "text": c.text}
                            for c, s in hits]
        ctx = "\n\n".join(f"{c.header()}\n{c.text}" for c, _ in hits)
        return q, ctx

    # ---- tools ---------------------------------------------------------
    def _run_tool(self, name, args):
        if name == "date_math":
            try:
                return datetools.date_math(**args)
            except Exception as e:
                return {"error": str(e)}
        if name == "refund_window":
            return refunds.refund_window(args.get("payment_method"), args.get("start_date"))
        if name == "lookup_order":
            oid = digits(args.get("order_id", ""))
            val = args.get("verification_value")
            if val and digits(val) not in self.all_user_digits():     # the model may not invent values
                return {"error": "verification_value must be exactly what the customer said"}
            return self._lookup(oid, val)
        if name == "create_ticket":
            res = self.tools.create_ticket(summary=args.get("summary", ""), order_id=args.get("order_id") or self.order_id,
                                           reason=args.get("reason"), intent=args.get("intent"))
            if res.get("created"):
                self.ticket = res
            return res
        return {"error": "unknown tool"}

    def _lookup(self, oid, val=None):
        """The only place identity checks happen. Same (order, value) twice in one turn is evaluated once."""
        key = (oid, digits(val) if val else None)
        if key in self.turn_cache:
            return self.turn_cache[key]
        if val:
            if self.failed.get(oid, 0) >= 2:
                res = {"verified": False, "blocked": True,
                       "message": "Two failed attempts. Do not share order details; offer general help or a colleague."}
            else:
                res = self.tools.lookup_order(oid, val)
                if res.get("verified") and res.get("found"):
                    self.order, self.order_id = res, oid
                else:
                    self.failed[oid] = self.failed.get(oid, 0) + 1
                    res = {**res, "failed_attempts": self.failed[oid]}
        else:
            res = self.tools.lookup_order(oid)
        self.turn_cache[key] = res
        return res

    _MONTHS = re.compile(r"january|february|march|april|may|june|july|august|september|october|november|december", re.I)

    def _auto_verify(self, text):
        """Run order lookups in code when the customer gives an 8-digit order ID or a 4/6-digit verification value."""
        notes = []
        if self.order:
            return notes
        for run in re.findall(r"\d[\d ]*\d|\d", text):
            d = re.sub(r"\D", "", run)
            if len(d) == 8 and self.pending != d:
                res = self._lookup(d)
                self.pending = d
                self.tool_log.append({"tool": "lookup_order", "args": {"order_id": d}, "result_keys": list(res)[:8]})
                notes.append(f"Order check opened for order {d}. Ask for the last 4 digits of the registered phone number OR the delivery pincode (one question).")
            elif self.pending and len(d) in (4, 6) and not self._MONTHS.search(text):
                res = self._lookup(self.pending, d)
                self.tool_log.append({"tool": "lookup_order", "args": {"order_id": self.pending, "verification_value": d},
                                      "result_keys": list(res)[:8]})
                n = self.failed.get(self.pending, 0)
                if self.order:
                    notes.append("Verification SUCCEEDED. The verified order record is provided below.")
                elif res.get("blocked") or n >= 2:
                    notes.append("Verification FAILED for the second time. Share NO order details. Apologise, say you cannot share details of this order because it could not be verified against the registered details, and offer general policy help or a colleague.")
                else:
                    notes.append("Verification FAILED (first attempt): the value does not match our registered details. Apologise and ask the customer to try again with the last 4 digits of the registered phone number or the delivery pincode.")
                break
        return notes

    def all_user_digits(self):
        return {digits(t)[i:j] for t in self.user_text for i in range(len(digits(t)))
                for j in (i + 4, i + 6) if j <= len(digits(t))}

    # ---- main loop -----------------------------------------------------
    def respond(self, text):
        text = normalize_numbers(text)            # "double six four nine ..." -> "66494 ..."
        first_reply = not self.history
        self.user_text.append(text)
        self.turn_cache = {}
        notes = self._auto_verify(text)
        notes.insert(0, "This is your FIRST reply: introduce yourself as ZentroMart's AI assistant." if first_reply
                     else "This is NOT your first reply: do not introduce yourself again.")
        if self.channel == "voice" and language_hint(text):
            notes.append(language_hint(text))
        query, ctx = self.retrieve(text)
        order_blk = ""
        if self.order:
            safe = {k: v for k, v in self.order.items() if k not in ("customer_id",)}
            safe["computed_checks"] = refunds.order_checks(self.order)
            order_blk = f"\n\n<verified_order (this customer is verified; never read out pincode/phone)>\n{json.dumps(safe, default=str)}\n</verified_order>"
        system = BASE_PROMPT + RAG_RULES + (VOICE_RULES if self.channel == "voice" else "")
        note_blk = ("<system_note>" + " ".join(notes) + "</system_note>\n") if notes else ""
        user_msg = f"{note_blk}<kb_context>\n{ctx}\n</kb_context>{order_blk}\n\nCustomer: {text}"
        msgs = self.history + [llm.user(user_msg)]
        reply = ""
        for _ in range(8):
            r = llm.create(system, msgs, tools=TOOLS, max_tokens=2048)
            calls = llm.function_calls(r)
            if not calls:
                reply = llm.text_of(r)
                break
            msgs.append(r.candidates[0].content)
            results = []
            for fc in calls:
                args = dict(fc.args or {})
                out = self._run_tool(fc.name, args)
                self.tool_log.append({"tool": fc.name, "args": args, "result_keys": list(out)[:8]})
                results.append((fc.name, json.loads(json.dumps(out, default=str))))
            msgs.append(llm.tool_results(results))
        # keep history compact: store the plain customer text and the final reply only
        self.history.append(llm.user(text))
        self.history.append(llm.model(reply))
        self.last_query = query
        return reply
