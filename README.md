# Karen AI

Karen is a privacy-first, voice-first personal assistant for Windows, in the spirit
of JARVIS. It runs on a local Ollama model (CPU-only laptops are fine), remembers
what you tell it, researches the web when a question needs current information,
and answers out loud.

## How a request flows

```text
voice / text ─► clean utterance ("Hey Karen, ..." → "...")
             ─► pending confirmation?  (yes / sure / nope / never mind ...)
             ─► fast path: instant commands, no model needed
             ─► local model, one streamed call that either
                  • answers directly (spoken sentence by sentence as it streams)
                  • replies "SEARCH: <query>" → web research → answer from sources
                  • returns a JSON action → permission policy → trusted skill
```

Only allowlisted skills can run. Shell commands, arbitrary executables, file
deletion and registry changes are never reachable. Sending a WhatsApp message,
clearing memories/conversation and cancelling reminders always ask for confirmation.

## Features

| Area | What Karen can do |
|---|---|
| Conversation | Local model answers with today's date/time, your name, relevant memories and the last few turns |
| Web research | Weather (wttr.in), current office holders (Wikipedia), and general search → reads the top pages → answers with sources |
| Memory | Learns facts you state ("my name is…", "I live in…", "I like…"); "remember that…", "forget that…", "what do you know about me" |
| Reminders | "remind me to call mom in 10 minutes / at 6 pm / tomorrow"; announced even after a restart |
| Apps & web | Open any Start Menu app, websites ("open YouTube", "go to github.com"), "play lo-fi on YouTube" |
| System | Time, date, battery, CPU/RAM status; volume up/down/mute, play/pause/next track, lock the PC |
| Folders & Windows | Desktop/Documents/Downloads…, Settings, Task Manager |
| WhatsApp | Open WhatsApp Web; send an exact message after confirmation |
| Voice | Push-to-talk (Ctrl+Space or TALK) that stops when you stop speaking; optional "Hey Jarvis" wake word; spoken replies you can interrupt (Esc / STOP / start talking) |

Without a local model Karen still runs: commands, reminders and memory work, and
questions are answered from web sources extractively.

## Setup

1. Install Python 3.12+ and [Ollama](https://ollama.com/).
2. Pull a model. On a CPU-only laptop a 4B model is a good balance; use a smaller
   one for speed:

   ```text
   ollama pull qwen3:4b        # default (better answers)
   ollama pull qwen3:1.7b      # faster on slower CPUs
   ```

3. Install dependencies:

   ```text
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   `requirements.txt` installs everything: Windows voices (comtypes), better web
   search (ddgs), microphone + Whisper speech recognition, and the wake word. The
   only hard requirement is `psutil`; every other feature degrades gracefully.

4. Run:

   ```text
   python -m karen              # animated HUD (default)
   python -m karen --text       # terminal chat (replies are also spoken)
   python -m karen --voice      # terminal, talk with the microphone
   python -m karen --mute       # don't speak replies
   ```

The first voice command downloads the Whisper `base.en` model once (~150 MB), and
the first wake-word use downloads its small model once. After that, everything
runs offline except web research.

## HUD shortcuts

`Enter` send • `Ctrl+Space` talk • `Esc` stop speaking / leave fullscreen •
`Ctrl+H` conversation history • `F11` fullscreen • `Ctrl+Q` quit.
VOICE and WAKE buttons toggle spoken replies and the wake word (saved to
`config/preferences.json`).

## Configuration (`config/settings.json`)

| Key | Meaning |
|---|---|
| `ollama_model` | Model name, e.g. `qwen3:4b` or `qwen3:1.7b` |
| `context_window`, `max_answer_tokens` | Smaller = faster on CPU |
| `llm_threads` | Limit Ollama CPU threads (0 = automatic) to keep the laptop responsive |
| `keep_alive` | How long the model stays loaded between questions |
| `tts_engine` | `sapi` (Windows voices), `piper` (needs `piper_model_path`), or `none` |
| `tts_voice`, `tts_rate` | Part of a Windows voice name (e.g. `Zira`), speed −10…10 |
| `whisper_model` | `tiny.en` (fastest), `base.en` (default), `small.en` (most accurate) |
| `listen_silence_seconds` | Pause length that ends a voice command |
| `wake_word_enabled`, `wake_word_model` | Wake word, default `hey_jarvis` |
| `live_information_enabled`, `web_max_pages` | Web research on/off and how many pages to read |

Unknown or invalid keys are ignored with a warning in `data/karen.log`.

## Privacy

Conversations, memories and reminders live in `data/karen.db` on your machine
(never committed to git). Web research sends only the search query. Logs are
local and size-capped.

## Tests

```text
pip install pytest
python -m pytest
```
