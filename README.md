# VoiceOps 2026 - ZentroMart voice support agent

Repository: https://github.com/aakarshgopishetty/VoiceOps.git   <- replace with your public GitHub URL

A support agent that answers from company documents, holds a multi-turn conversation (identity check, order lookup,
ticket handover) and speaks/listens through **Gnani STT/TTS** (the only speech service used).

```
audio -> Gnani STT -> Agent (LLM + RAG + tools) -> normalise for speech -> Gnani TTS -> reply.wav
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill GNANI_API_KEY and GEMINI_API_KEY (free key from aistudio.google.com) (never commit .env)
# extract the organisers' dataset somewhere and point to it (it is NOT part of this repo):
export ZENTRO_DATASET=/path/to/Capstone_Project_Dataset_FTTT_2026
```
Any hosted LLM is allowed; `src/llm.py` is the only file that talks to the LLM provider (default: Google Gemini via `google-genai`, set `LLM_MODEL`).

## Run command (organisers)
```bash
python run.py --audio_dir ./test_audio --out_dir ./outputs
```
Per audio file it writes `outputs/<name>/{transcript.txt, reply.txt, reply.wav, sources.json}` (sources = document chunks
used) and `outputs/results.json`. Files named `<conv>__<turn>.wav` are treated as consecutive turns of one conversation;
all other files are independent single-turn calls. A failure on one file never stops the batch.

## Develop / evaluate
```bash
python chat.py                                                   # text chat
python -m src.evaluate --dataset $ZENTRO_DATASET                 # Step 1 vs Step 2 + conversations (resumable)
python make_report.py "My Team"                                  # builds report.pdf from eval_results.json
python app.py                                                    # optional demo web page (pip install gradio)
```
`evaluate` saves progress to `eval_cache.json` after every question/conversation. If an API daily quota stops it, run the
same command again later and it continues. Options: `--limit N`, `--only questions|convs`, `--fresh`.

**Quota tips (Gemini free tier):** limits are per model per day. Set `LLM_MODEL` to a model with a high daily limit,
`FALLBACK_MODELS` to a backup, `GEMINI_RPM` to the model's requests-per-minute, and `REWRITE=auto` to save calls.
Windows PowerShell: `$env:ZENTRO_DATASET = "C:\path\to\dataset"` (or put it in `.env`).

## How it works
- `src/kb.py` - chunks `kb/*.md` by `[SECTION-ID]` keeping doc_type/status/effective_from; indexes product, pincode and holiday
  rows; hybrid BM25 + char-n-gram TF-IDF (+ optional dense via `EMBED_MODEL`); precedence weighting (superseded v3 demoted);
  synonym map and product->policy links; always-on core rules.
- `src/agent.py` - Step 1 `baseline_answer`, Step 2/3 `Agent`: LLM-rewritten search query, grounded prompt, tools
  `lookup_order`, `create_ticket`, `date_math`. Verification limits are enforced in code, not just in the prompt.
- `src/datetools.py` - business days (Mon-Sat minus holidays), calendar-day windows, today = 2026-10-20.
- `src/voice.py` - Gnani STT/TTS, 60 s clip splitting, spoken-text normalisation, language selection, retry/fallback within Gnani only.
- Practice-set results are in `report.pdf` (Step 1 prompt-only vs Step 2 RAG, plus conversations). Reproduce with `python -m src.evaluate`.

## Notes
- Uses only the synthetic dataset. Do not commit keys, `.env`, virtualenvs or the dataset.
- Tickets are written by the mock tool to `<dataset>/tools/tickets_created.csv`.

## Submitting
```bash
python -m src.evaluate --fresh       # final numbers (resumable)
python make_report.py "Team name"    # report.pdf
python run.py --audio_dir clips --out_dir sample_outputs
# add demo.mp4 (max 2 min), put your GitHub URL on the "Repository:" line above, then:
python package.py team_name          # checks everything and builds team_name_voiceops.zip
```
