"""Thin Gemini wrapper (any hosted LLM is allowed by the rules; this is the only file that talks to the LLM provider)."""
import os
import re
import time

from google import genai
from google.genai import types

from . import config

_client = None


def client():
    global _client
    if _client is None:
        _client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY)
    return _client


def _schema(d):
    """Convert a JSON-schema dict into a google-genai Schema."""
    kw = {"type": d["type"].upper()}
    if "description" in d: kw["description"] = d["description"]
    if "enum" in d: kw["enum"] = d["enum"]
    if "properties" in d: kw["properties"] = {k: _schema(v) for k, v in d["properties"].items()}
    if "required" in d: kw["required"] = d["required"]
    return types.Schema(**kw)


def tool_config(tools):
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(name=t["name"], description=t["description"], parameters=_schema(t["input_schema"]))
        for t in tools])


def user(text):
    return types.Content(role="user", parts=[types.Part(text=text)])


def model(text):
    return types.Content(role="model", parts=[types.Part(text=text)])


def tool_results(pairs):
    """pairs: [(function_name, result_dict)] -> one user Content carrying the function responses."""
    return types.Content(role="user", parts=[types.Part.from_function_response(name=n, response={"result": r})
                                             for n, r in pairs])


_last = {}


def _throttle(model):
    """Free tier allows ~5 requests/min per model; stay under it (override with GEMINI_RPM)."""
    gap = 60.0 / float(os.environ.get("GEMINI_RPM", "4"))
    wait = _last.get(model, 0) + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last[model] = time.time()


def _retry_delay(msg):
    m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"retryDelay'?\W+(\d+)s", msg)
    return float(m.group(1)) if m else None


def create(system, contents, tools=None, max_tokens=4096, temperature=0.2, model=None):
    """Call Gemini with throttling. Per-minute 429 -> wait and retry. 404 / daily-quota -> next model in FALLBACK_MODELS."""
    first = model or config.LLM_MODEL
    chain = [first] + [m for m in config.FALLBACKS if m != first]
    cfg = types.GenerateContentConfig(
        system_instruction=system, temperature=temperature, max_output_tokens=max(max_tokens, 4096),
        tools=[tool_config(tools)] if tools else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    last = None
    for mdl in chain:
        for attempt in range(6):
            _throttle(mdl)
            try:
                return client().models.generate_content(model=mdl, contents=contents, config=cfg)
            except Exception as e:
                last, msg = e, str(e)
                if "API key" in msg or "API_KEY" in msg:
                    raise
                if "404" in msg[:60] or "PerDay" in msg or "RESOURCE_EXHAUSTED" in msg and "per day" in msg.lower():
                    break                      # this model is unusable right now -> next in chain
                d = _retry_delay(msg)
                if d is not None and d > 120:
                    break                      # long wait == daily/hourly cap
                time.sleep(min((d or 3 * (attempt + 1)) + 2, 90))
        else:
            continue
    raise last


def simple(system, text, max_tokens=1024, temperature=0.0):
    return text_of(create(system, [user(text)], max_tokens=max_tokens, temperature=temperature,
                          model=config.AUX_MODEL))


def function_calls(resp):
    return list(resp.function_calls or [])


def text_of(resp):
    try:
        return "".join(p.text for p in resp.candidates[0].content.parts if getattr(p, "text", None)).strip()
    except Exception:
        return ""
