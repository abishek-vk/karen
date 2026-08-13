# Karen AI — Phase 1.1

Karen is a privacy-first, offline Windows AI companion. This phase implements a
text-only core: user text goes to a local Ollama model, the response is parsed as
strict intent JSON, and only explicitly allowlisted skills can execute.

## Structure

```text
Karen/
├── main.py                 # Compatibility entry point
├── karen/
│   ├── core/
│   │   ├── config.py       # Compatibility configuration API
│   │   ├── confirmation.py  # In-memory pending confirmations
│   │   ├── fast_path.py     # Deterministic common commands
│   │   ├── intent.py        # JSON parsing and validation
│   │   ├── llm.py           # Ollama-backed LLM boundary and system prompt
│   │   ├── permissions.py   # Capability-based policy checks
│   │   ├── router.py        # Registry-backed intent dispatch
│   │   └── skill_registry.py # Trusted skill registration
│   ├── skills/
│   │   ├── app_control.py  # Notepad, Calculator, Chrome allowlist
│   │   ├── base.py         # Skill result and manifest contracts
│   │   └── system.py       # Read-only OS/time/CPU/RAM information
│   ├── ai/ollama.py        # Local Ollama HTTP client
│   ├── text_app.py         # Interactive text application
│   ├── ui/opening.py       # Animated-only graphical interface
│   └── voice/              # Optional local voice adapters
│   └── ui/opening.py       # Animated-only graphical interface
├── memory/memory.json      # Disabled by default; no conversation storage
├── config/settings.json    # Local model configuration (180s CPU-safe timeout)
├── requirements.txt
└── README.md
```

All active application modules now live under the `karen/` package. The root
`main.py` remains as a compatibility entry point for development. Configuration
is managed by `karen.config`; the original `karen.core.config` import remains a
compatibility wrapper.

## Setup

1. Install Python 3.12+.
2. Install [Ollama](https://ollama.com/) locally.
3. Pull the configured model:

   ```text
   ollama pull qwen3:4b
   ```

4. Create and activate a virtual environment:

   ```text
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

5. Start Ollama, then run Karen's text interface from the project directory:

   ```text
   python -m karen
   ```

   To open the animated-only graphical interface:

   ```text
   python -m karen --ui
   ```

   For compatibility, `python main.py` starts the same text interface.

No cloud API, telemetry, browser history, clipboard, or recording is used.
Voice support is optional and lazy-loaded. Install it with `pip install -e .[voice]`
when local microphone input is desired. WhatsApp Web automation is also optional:
install `pip install -e .[browser]`, then run `playwright install chromium`. Karen
will open a persistent local browser profile so you can log in to WhatsApp Web.
Ollama is restricted by the client to a local HTTP endpoint. When the GUI opens,
Karen attempts to start `ollama serve` automatically; Qwen must already be
installed locally with `ollama pull qwen3:4b`.

## Example

```text
Karen: What can I help you with?
You: Open Notepad
Karen: Opening Notepad.
You: What operating system am I using?
Karen: Operating system: Windows-...
Time: ...
CPU usage: ...
Memory usage: ...
```

The model must return an object such as:

```json
{
  "intent": "open_application",
  "target": "notepad",
  "parameters": {},
  "requires_confirmation": false
}
```

Shell commands, arbitrary executables, file deletion, registry changes, and
system configuration changes are not exposed to the router.

## Future expansion

- Expand the deterministic fast-path parser for common commands.
- Extend the in-memory confirmation service to future sensitive skills.
- Add voice and wake-word adapters behind the same text input interface.
- Add opt-in memory with explicit user consent and schema validation.
- Add plugin registration with capability manifests and permission boundaries.
- Add Windows packaging and integration only after the core test suite is stable.
