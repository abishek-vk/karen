"""Run Karen's graphical (default) or terminal interface."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Karen AI local assistant")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ui", action="store_true", help="launch the animated HUD (default)")
    mode.add_argument("--text", action="store_true", help="use the terminal instead of the HUD")
    parser.add_argument("--voice", action="store_true", help="terminal mode: talk with the microphone")
    speech = parser.add_mutually_exclusive_group()
    speech.add_argument("--speak", action="store_true", help="speak replies aloud")
    speech.add_argument("--mute", action="store_true", help="do not speak replies")
    args = parser.parse_args()
    speak = True if args.speak else False if args.mute else None

    if args.text or args.voice:
        from .text_app import main as text_main

        text_main(speak=speak, voice=args.voice)
    else:
        from .ui.opening import launch

        launch(speak=speak)


if __name__ == "__main__":
    main()
