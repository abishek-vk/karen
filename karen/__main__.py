"""Run Karen's text or graphical interface."""

from __future__ import annotations

import argparse

from .text_app import main as text_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Karen AI local assistant")
    parser.add_argument(
        "--ui",
        action="store_true",
        help="launch the animated graphical opening page",
    )
    args = parser.parse_args()
    if args.ui:
        from .ui.opening import launch

        launch()
    else:
        text_main()


if __name__ == "__main__":
    main()
