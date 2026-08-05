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
│   │   ├── config.py       # Local Ollama settings
│   │   ├── intent.py       # JSON parsing and validation
│   │   ├── llm.py          # Ollama-backed LLM boundary and system prompt
│   │   └── router.py       # Intent-to-skill dispatch
│   ├── skills/
│   │   ├── app_control.py  # Notepad, Calculator, Chrome allowlist
│   │   └── system.py       # Read-only OS/time/CPU/RAM information
│   ├── ai/ollama.py        # Local Ollama HTTP client
│   └── text_app.py         # Interactive text application
├── memory/memory.json      # Disabled by default; no conversation storage
├── config/settings.json    # Local model configuration
├── requirements.txt
└── README.md
```

All active application modules now live under the `karen/` package. The root
`main.py` remains as a compatibility entry point for development.

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

5. Start Ollama, then run Karen from the project directory:

   ```text
   python -m karen
   ```

   For compatibility, `python main.py` starts the same text interface.

No cloud API, telemetry, browser history, clipboard, or recording is used.
Ollama is restricted by the client to a local HTTP endpoint.

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

- Add deterministic fast-path parsing for common commands so they do not wait
  for the LLM.
- Add a confirmation service before any future sensitive skill.
- Add voice and wake-word adapters behind the same text input interface.
- Add opt-in memory with explicit user consent and schema validation.
- Add plugin registration with capability manifests and permission boundaries.
- Add Windows packaging and integration only after the core test suite is stable.
