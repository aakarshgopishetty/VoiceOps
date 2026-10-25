#!/usr/bin/env python3
"""Interactive text chat with the agent (for development):  python chat.py"""
from src.agent import Agent

a = Agent(channel="chat")
print("ZentroMart support (Ctrl-C to quit)")
while True:
    try:
        t = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if t:
        print("bot>", a.respond(t))
        print("     [sources]", ", ".join(c["id"] for c in a.last_chunks))
