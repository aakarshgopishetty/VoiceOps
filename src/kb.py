"""Knowledge-base loading, chunking and hybrid retrieval (BM25 + char-ngram TF-IDF [+ optional dense])."""
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config

# Precedence from KB-00.1: policy > agent guide > help > faq > marketing; superseded demoted.
PRECEDENCE = {"policy": 1.0, "agent_guide": 0.96, "help": 0.9, "faq": 0.84, "marketing": 0.65,
              "meta": 0.5, "data": 0.95}

# Customers rarely use document vocabulary ("send it back" vs "reverse pickup").
SYNONYMS = {
    "send back": "return reverse pickup", "sending back": "return reverse pickup",
    "give back": "return", "money back": "refund", "wapas": "return refund",
    "paisa": "refund", "paise": "refund", "vapas": "return refund",
    "nobody came": "reverse pickup missed", "didn't come": "reverse pickup missed",
    "pickup": "reverse pickup", "collect": "reverse pickup collection",
    "cancel": "cancellation refuse parcel warehouse refund initiated", "late": "delay delivery", "not arrived": "delivery delay",
    "hasn't arrived": "delivery delay", "broken": "damaged defective", "torn": "damaged",
    "wrong item": "wrong product", "guarantee": "warranty", "member": "plus membership",
    "compensation": "compensation goodwill", "cod": "cash on delivery",
    "wapsi": "return", "kitne din": "how many days timeline", "kab tak": "when timeline",
    "cash on delivery": "cod refund wallet bank transfer neft", "never placed": "unauthorised order account activity",
    "did not order": "unauthorised order account activity", "didn't order": "unauthorised order account activity",
    "crack": "damaged defective", "doesn't fit": "change of mind return", "not home": "delivery attempts rto", "not at home": "delivery attempts rto", "nobody home": "delivery attempts rto", "ghar pe nahi": "delivery attempts rto", "ghar par nahi": "delivery attempts rto",
    "dubai": "international delivery serviceable pincodes india", "abroad": "international delivery serviceable pincodes india",
    "phone number of": "other customer details privacy",
}


@dataclass
class Chunk:
    cid: str            # e.g. "RET-6" or "products.csv:ELE-0012"
    text: str
    doc_id: str = ""
    title: str = ""
    doc_type: str = "data"
    status: str = "current"
    effective_from: str = ""
    source: str = ""
    weight: float = 1.0
    meta: dict = field(default_factory=dict)

    def header(self):
        return (f"[{self.cid}] ({self.title}; {self.doc_type}; status={self.status}; "
                f"effective_from={self.effective_from})")


def _front_matter(text):
    meta, body = {}, text
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = m.group(2)
    return meta, body


def load_chunks():
    chunks = []
    for path in sorted(config.KB_DIR.glob("*.md")):
        meta, body = _front_matter(path.read_text(encoding="utf-8"))
        dtype = meta.get("doc_type", "data")
        status = meta.get("status", "current")
        weight = PRECEDENCE.get(dtype, 0.8) * (0.55 if status == "superseded" else 1.0)
        parts = re.split(r"(?m)^##\s+(?=\[)", body)
        for part in parts[1:]:
            m = re.match(r"\[([^\]]+)\]\s*(.*)", part)
            if not m:
                continue
            cid, rest = m.group(1), part
            chunks.append(Chunk(cid=cid, text=rest.strip(), doc_id=meta.get("doc_id", ""),
                                title=meta.get("title", path.stem), doc_type=dtype, status=status,
                                effective_from=meta.get("effective_from", ""), source=path.name,
                                weight=weight))
    # Structured reference data is also retrievable (gold evidence cites e.g. products.csv:ELE-0012).
    for row in csv.DictReader(open(config.DATA_DIR / "products.csv", encoding="utf-8")):
        txt = "; ".join(f"{k}={v}" for k, v in row.items() if v != "")
        chunks.append(Chunk(cid=f"products.csv:{row['sku']}", text=txt, title="Product catalog",
                            source="products.csv", weight=0.95,
                            meta={"cat": row.get("return_category")}))
    for row in csv.DictReader(open(config.DATA_DIR / "pincodes.csv", encoding="utf-8")):
        txt = "; ".join(f"{k}={v}" for k, v in row.items() if v != "")
        chunks.append(Chunk(cid=f"pincodes.csv:{row['pincode']}", text="pincode " + txt,
                            title="Serviceable pincodes", source="pincodes.csv", weight=0.9))
    hol = list(csv.DictReader(open(config.DATA_DIR / "holidays.csv", encoding="utf-8")))
    chunks.append(Chunk(cid="holidays.csv", title="Warehouse holidays", source="holidays.csv",
                        text="Holidays skipped in business-day counts: " +
                        "; ".join(f"{r['date']} {r['name']}" for r in hol), weight=0.9))
    return chunks


_TOK = re.compile(r"[a-z0-9\u0900-\u097f\u0b80-\u0bff\u0c00-\u0c7f]+")
_STOP = set("the a an is are was were to of and or for in on at it my me i you your do does can "
            "how what when will be with this that have has hai kya ka ki ke se ko mein ho".split())


def tokenize(text):
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


def expand(query):
    q = query.lower()
    extra = [v for k, v in SYNONYMS.items() if k in q]
    if "refund" in q or "paisa" in q:
        extra.append("refund timeline business days payment method initiated")
    return query + " " + " ".join(extra)


class Retriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or load_chunks()
        docs = [tokenize(f"{c.title} {c.cid} {c.text}") for c in self.chunks]
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.len.mean()
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.mat = self.vec.fit_transform([f"{c.title} {c.text}" for c in self.chunks])
        self.dense = None
        try:  # optional multilingual dense embeddings (pip install sentence-transformers)
            import os
            name = os.environ.get("EMBED_MODEL")
            if name:
                from sentence_transformers import SentenceTransformer
                self.st = SentenceTransformer(name)
                self.dense = self.st.encode([c.text for c in self.chunks], normalize_embeddings=True)
        except Exception:
            self.dense = None

    def _bm25(self, q, k1=1.5, b=0.75):
        qt = tokenize(q)
        s = np.zeros(len(self.chunks))
        for i, tf in enumerate(self.tf):
            for t in qt:
                if t in tf:
                    f = tf[t]
                    s[i] += self.idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * self.len[i] / self.avg))
        return s

    @staticmethod
    def _norm(x):
        m = x.max()
        return x / m if m > 0 else x

    CORE = ("KB-00.1", "KB-00.3", "ESC-1", "IDV-2", "STY-1")

    def search(self, query, k=None, k_data=3, core=True):
        """Top-k policy/guide chunks + top-k_data structured-data rows (+ always-on core rules)."""
        k = k or config.TOP_K
        scored = self._score(query)
        order = np.argsort(-scored)
        kb, data = [], []
        for i in order:
            c = self.chunks[i]
            (data if c.source.endswith(".csv") else kb).append((c, float(scored[i])))
        out = kb[:k] + data[:k_data]
        # Cross-document links: a product row needs the policy that applies to its category.
        by = {c.cid: c for c in self.chunks}
        want = []
        ql = expand(query).lower()
        prods = [c for c, _ in data[:k_data] if c.cid.startswith("products.csv:")]
        returny = any(w in ql for w in ("return", "refund", "exchange", "reverse pickup", "fee"))
        if prods and returny:
            want += ["RET-2", "RET-4"]
            cats = {c.meta.get("cat") for c in prods}
            if cats & {"grocery", "innerwear", "digital"}:
                want.append("RET-5")
            if "fashion" in cats:
                want += ["RET-6", "RET-8"]
            if "electronics" in cats:
                want.append("RET-3")
        if prods and any(w in ql for w in ("warranty", "assured", "claim", "guarantee")):
            want += ["WAR-1", "WAR-2"]
        if "return" in ql and not prods:
            want.append("RET-2")
        if "cancel" in ql:                       # cancellation answers need CAN-2 and the refund-start rules
            want += ["CAN-2", "REF-1"]
        if "cash on delivery" in ql or re.search(r"\bcod\b", ql):
            want += ["REF-2", "REF-3"]
        have = {c.cid for c, _ in out}
        out += [(by[x], 0.0) for x in dict.fromkeys(want) if x in by and x not in have]
        if core:
            have = {c.cid for c, _ in out}
            by = {c.cid: c for c in self.chunks}
            out += [(by[x], 0.0) for x in self.CORE if x in by and x not in have]
        return out

    def _score(self, query):
        q = expand(query)
        s = 0.5 * self._norm(self._bm25(q))
        s = s + 0.35 * self._norm((self.mat @ self.vec.transform([q]).T).toarray().ravel())
        if self.dense is not None:
            s = s + 0.5 * self._norm(self.dense @ self.st.encode([query], normalize_embeddings=True)[0])
        return s * np.array([c.weight for c in self.chunks])
