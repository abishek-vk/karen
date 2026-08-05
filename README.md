# Karen AI

Privacy-first, offline Windows AI companion. The project is being built incrementally;
Phase 1 currently provides local configuration, logging, and an Ollama client.

## Requirements

- Windows 11 (runtime target)
- Python 3.12+
- [Ollama](https://ollama.com/) installed locally
- The configured model available locally, by default `qwen3:4b`

## Development

```text
python -m venv .venv
.venv\\Scripts\\activate
pip install -e .[dev]
pytest
```

Karen connects only to `127.0.0.1` by default. The Ollama client sends prompts to
that local service and does not provide a cloud fallback. Configuration is stored
in local `data/settings.json` and `data/preferences.json` when saved; these files
are intentionally ignored by Git.
