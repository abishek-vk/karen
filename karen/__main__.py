"""Run Karen's 3D interface (default), the classic HUD, or the terminal."""

from __future__ import annotations

import argparse


def _preferred_style() -> str:
    try:
        from .app import PROJECT_DIR
        from .config import ConfigurationStore

        return ConfigurationStore(PROJECT_DIR / "config").load_settings().ui_style.strip().lower()
    except Exception:  # Settings problems are reported later by the runtime.
        return "3d"


def main() -> None:
    parser = argparse.ArgumentParser(description="Karen AI local assistant")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ui", action="store_true", help="launch the 3D interface (default)")
    mode.add_argument("--classic", action="store_true", help="launch the lightweight Tk HUD instead of 3D")
    mode.add_argument("--text", action="store_true", help="use the terminal instead of a window")
    parser.add_argument("--voice", action="store_true", help="terminal mode: talk with the microphone")
    speech = parser.add_mutually_exclusive_group()
    speech.add_argument("--speak", action="store_true", help="speak replies aloud")
    speech.add_argument("--mute", action="store_true", help="do not speak replies")
    args = parser.parse_args()
    speak = True if args.speak else False if args.mute else None

    if args.text or args.voice:
        from .text_app import main as text_main

        text_main(speak=speak, voice=args.voice)
        return
    if not args.classic and (args.ui or _preferred_style() != "classic"):
        from .ui import web_app

        if web_app.available():
            web_app.launch(speak=speak)
            return
    from .ui.opening import launch

    launch(speak=speak)


if __name__ == "__main__":
    main()
