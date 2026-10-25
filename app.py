#!/usr/bin/env python3
"""Optional demo web page (microphone in, spoken reply out):   pip install gradio   then   python app.py"""
import tempfile
from pathlib import Path

import gradio as gr

from src import config
from src.agent import Agent
from src.voice import Voice, REPROMPT

config.ensure_dataset()
voice = Voice()


def turn(audio_path, agent, history):
    agent = agent or Agent(channel="voice")
    history = history or []
    if not audio_path:
        return agent, history, None, history, ""
    heard = voice.listen(audio_path)
    reply = agent.respond(heard) if heard.strip() else REPROMPT["en"]
    out = Path(tempfile.mkdtemp()) / "reply.wav"
    voice.speak(reply, out)
    history = history + [(heard or "(unclear audio)", reply)]
    sources = ", ".join(c["id"] for c in agent.last_chunks) if heard.strip() else ""
    return agent, history, str(out), history, sources


with gr.Blocks(title="ZentroMart voice support") as demo:
    gr.Markdown("## ZentroMart support - press record, speak, stop, then submit")
    st, hist = gr.State(None), gr.State([])
    mic = gr.Audio(sources=["microphone"], type="filepath", label="Your voice")
    chat = gr.Chatbot(label="Conversation")
    speech = gr.Audio(label="Agent reply", autoplay=True)
    src = gr.Textbox(label="Document chunks used", interactive=False)
    mic.stop_recording(turn, [mic, st, hist], [st, hist, speech, chat, src])
    gr.Button("New call").click(lambda: (None, [], None, [], ""), None, [st, hist, speech, chat, src])

if __name__ == "__main__":
    demo.launch()
