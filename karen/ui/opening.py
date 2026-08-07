"""Cinematic, privacy-first opening screen for Karen AI.

This UI intentionally uses only Tkinter so the opening experience works before
optional GUI dependencies are installed. It is inspired by futuristic HUDs,
not a reproduction of any specific film interface.
"""

from __future__ import annotations

import math
import queue
import random
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

from karen.core.application_service import ApplicationService
from karen.text_app import build_application


try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency
    psutil = None


BACKGROUND = "#030912"
PANEL = "#071522"
CYAN = "#61e7ff"
BRIGHT_CYAN = "#c8f8ff"
BLUE = "#1c9fce"
MUTED = "#6e9aaa"
GREEN = "#76f7c5"


@dataclass(slots=True)
class Particle:
    angle: float
    radius: float
    speed: float
    size: float
    alpha_step: int


class OpeningPage(tk.Tk):
    """Animated Karen welcome window with a lightweight HUD aesthetic."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Karen AI")
        self.geometry("1180x760")
        self.minsize(900, 600)
        self.configure(bg=BACKGROUND)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        self._started = False
        self._busy = False
        self._frame = 0
        self._service: ApplicationService | None = None
        self._responses: queue.Queue[str] = queue.Queue()
        self._last_response = "Initialize Karen to begin your local session."
        self._model_status = "CHECKING LOCAL MODEL..."
        self._start_time = time.monotonic()
        self._particles = [
            Particle(
                angle=random.random() * math.tau,
                radius=random.uniform(150, 430),
                speed=random.uniform(0.0003, 0.0012),
                size=random.choice((1, 1, 1, 2)),
                alpha_step=random.randint(0, 100),
            )
            for _ in range(90)
        ]

        self.canvas = tk.Canvas(
            self,
            bg=BACKGROUND,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._draw())
        self.bind("<Escape>", lambda _event: self.destroy())

        self._build_controls()
        self.after(40, self._animate)
        self.after(60, self._poll_response)
        self.after(1000, self._refresh_ollama_status)

    def _build_controls(self) -> None:
        self._start_button = tk.Button(
            self,
            text="  INITIALIZE KAREN  ",
            command=self._initialize,
            bg="#0b2838",
            activebackground="#164c62",
            fg=BRIGHT_CYAN,
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=18,
            pady=11,
            cursor="hand2",
            font=("Consolas", 10, "bold"),
        )
        self._start_button.place(relx=0.5, rely=0.78, anchor="center")

        self._command_entry = tk.Entry(
            self,
            bg="#0a1e2c",
            fg=BRIGHT_CYAN,
            insertbackground=CYAN,
            relief="flat",
            font=("Consolas", 10),
            state=tk.DISABLED,
        )
        self._command_entry.place(relx=0.39, rely=0.88, relwidth=0.42, anchor="center")
        self._command_entry.bind("<Return>", lambda _event: self._send_command())
        self._send_button = tk.Button(
            self,
            text="SEND",
            command=self._send_command,
            bg="#0b4052",
            activebackground="#16657a",
            fg=BRIGHT_CYAN,
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=15,
            pady=7,
            cursor="hand2",
            state=tk.DISABLED,
            font=("Consolas", 9, "bold"),
        )
        self._send_button.place(relx=0.68, rely=0.88, anchor="center")

    def _initialize(self) -> None:
        """Activate commands while keeping the animated HUD as the only UI."""

        project_dir = Path(__file__).resolve().parents[2]
        llm, router = build_application(project_dir)
        self._service = ApplicationService(llm, router)
        self._started = True
        self._start_button.configure(
            text="  SYSTEMS ONLINE  ",
            state=tk.DISABLED,
            disabledforeground=GREEN,
        )
        self._command_entry.configure(state=tk.NORMAL)
        self._send_button.configure(state=tk.NORMAL)
        self._command_entry.focus_set()
        self._last_response = "Local session ready. Ask Karen a question or give a command."

    def _send_command(self) -> None:
        if self._service is None or self._busy:
            return
        text = self._command_entry.get().strip()
        if not text:
            return
        self._command_entry.delete(0, tk.END)
        self._busy = True
        self._send_button.configure(state=tk.DISABLED, text="...")
        self._last_response = "Processing locally..."
        threading.Thread(target=self._process_command, args=(text,), daemon=True).start()

    def _process_command(self, text: str) -> None:
        service = self._service
        if service is not None:
            response = service.handle_text(text)
            self._responses.put(response.message)

    def _poll_response(self) -> None:
        try:
            response = self._responses.get_nowait()
        except queue.Empty:
            self.after(60, self._poll_response)
            return
        self._last_response = response
        self._busy = False
        self._send_button.configure(state=tk.NORMAL, text="SEND")
        self._command_entry.focus_set()
        self.after(60, self._poll_response)

    def _refresh_ollama_status(self) -> None:
        """Start/check Ollama in the background without freezing the animation."""

        service = self._service
        if service is not None:
            def check() -> None:
                available = service.llm.ensure_available()
                ready = available and service.llm.has_model()
                status = (
                    "QWEN3:4B ONLINE // LOCAL"
                    if ready
                    else "OLLAMA ONLINE // INSTALL QWEN3:4B"
                    if available
                    else "FAST PATH READY // OLLAMA OFFLINE"
                )
                self.after(0, lambda: self._set_model_status(status))

            threading.Thread(target=check, daemon=True).start()
        self.after(3000, self._refresh_ollama_status)

    def _set_model_status(self, status: str) -> None:
        self._model_status = status
        self._draw()

    def _animate(self) -> None:
        self._frame += 1
        self._draw()
        self.after(40, self._animate)

    def _draw(self) -> None:
        width = max(self.canvas.winfo_width(), 900)
        height = max(self.canvas.winfo_height(), 600)
        center_x = width / 2
        center_y = height * 0.46
        elapsed = time.monotonic() - self._start_time
        pulse = (math.sin(elapsed * 2.2) + 1) / 2
        rotation = elapsed * (0.22 if not self._started else 0.65)

        self.canvas.delete("all")
        self._draw_background(width, height, elapsed)
        self._draw_particles(center_x, center_y, width, height, elapsed)
        self._draw_hud(center_x, center_y, pulse, rotation)
        self._draw_text(width, height, pulse, elapsed)
        self._draw_status(width, height)

    def _draw_background(self, width: float, height: float, elapsed: float) -> None:
        for index in range(12):
            margin = index * 38
            color = "#" + format(max(4, 14 - index), "02x") + format(max(10, 25 - index), "02x") + format(max(18, 38 - index), "02x")
            self.canvas.create_rectangle(
                margin,
                margin,
                width - margin,
                height - margin,
                outline=color,
                width=1,
            )

        scan_y = (elapsed * 95) % max(height, 1)
        self.canvas.create_line(0, scan_y, width, scan_y, fill="#0a3443", width=1)
        self.canvas.create_line(0, scan_y + 1, width, scan_y + 1, fill="#061e2b", width=1)

    def _draw_particles(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        elapsed: float,
    ) -> None:
        for particle in self._particles:
            angle = particle.angle + elapsed * particle.speed * 80
            radius = particle.radius + math.sin(elapsed + particle.alpha_step) * 7
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius * 0.56
            if 0 < x < width and 0 < y < height:
                color = CYAN if particle.alpha_step % 3 else BLUE
                self.canvas.create_oval(
                    x,
                    y,
                    x + particle.size,
                    y + particle.size,
                    fill=color,
                    outline="",
                )

    def _draw_hud(
        self,
        center_x: float,
        center_y: float,
        pulse: float,
        rotation: float,
    ) -> None:
        base = 138 + pulse * 8
        for offset, color, line_width in (
            (30, "#092b3b", 1),
            (18, "#0d536b", 1),
            (7, BLUE, 2),
            (0, CYAN, 1),
        ):
            radius = base + offset
            self.canvas.create_oval(
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
                outline=color,
                width=line_width,
            )

        for start, extent, color, width in (
            (rotation * 55, 92, BRIGHT_CYAN, 3),
            (rotation * -37 + 120, 44, BLUE, 2),
            (rotation * 23 + 240, 70, CYAN, 2),
        ):
            radius = base + 5
            self.canvas.create_arc(
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
                start=start,
                extent=extent,
                outline=color,
                width=width,
                style=tk.ARC,
            )

        inner = 62 + pulse * 5
        self.canvas.create_oval(
            center_x - inner,
            center_y - inner,
            center_x + inner,
            center_y + inner,
            fill="#061522",
            outline="#2fb4d1",
            width=2,
        )
        self.canvas.create_oval(
            center_x - 9 - pulse * 2,
            center_y - 9 - pulse * 2,
            center_x + 9 + pulse * 2,
            center_y + 9 + pulse * 2,
            fill=BRIGHT_CYAN,
            outline=CYAN,
        )

        for angle in range(0, 360, 30):
            radians = math.radians(angle) + rotation
            inner_tick = base + 38
            outer_tick = inner_tick + (10 if angle % 60 == 0 else 5)
            self.canvas.create_line(
                center_x + math.cos(radians) * inner_tick,
                center_y + math.sin(radians) * inner_tick,
                center_x + math.cos(radians) * outer_tick,
                center_y + math.sin(radians) * outer_tick,
                fill=MUTED if angle % 60 else CYAN,
                width=2 if angle % 60 == 0 else 1,
            )

    def _draw_text(self, width: float, height: float, pulse: float, elapsed: float) -> None:
        title_font = ("Consolas", 30, "bold")
        subtitle_font = ("Consolas", 10)
        label_font = ("Consolas", 8)
        center_x = width / 2

        for offset, color in ((3, "#082c3b"), (1, "#17637a"), (0, BRIGHT_CYAN)):
            self.canvas.create_text(
                center_x + offset,
                height * 0.12 + offset,
                text="K A R E N",
                fill=color,
                font=title_font,
                anchor="center",
            )
        self.canvas.create_text(
            center_x,
            height * 0.18,
            text="LOCAL INTELLIGENCE // PRIVACY FIRST",
            fill=MUTED,
            font=subtitle_font,
            anchor="center",
        )

        self.canvas.create_text(
            center_x,
            height * 0.46,
            text="K",
            fill=BRIGHT_CYAN,
            font=("Consolas", 34, "bold"),
            anchor="center",
        )
        self.canvas.create_text(
            center_x,
            height * 0.68,
            text="AWAITING YOUR COMMAND" if not self._started else self._last_response[:130],
            fill=GREEN if self._started else CYAN,
            font=subtitle_font,
            anchor="center",
        )
        self.canvas.create_text(
            center_x,
            height * 0.73,
            text=self._model_status,
            fill=GREEN if "ONLINE" in self._model_status else MUTED,
            font=label_font,
            anchor="center",
        )
        self.canvas.create_text(
            28,
            height - 28,
            text="OFFLINE MODE  •  NO TELEMETRY  •  ESC TO EXIT",
            fill="#397082",
            font=label_font,
            anchor="w",
        )
        self.canvas.create_text(
            width - 28,
            height - 28,
            text=f"CORE {int((elapsed * 10) % 100):02d}%  //  LINK STABLE",
            fill="#397082",
            font=label_font,
            anchor="e",
        )

    def _draw_status(self, width: float, height: float) -> None:
        cpu = "--"
        memory = "--"
        if psutil is not None:
            cpu = f"{psutil.cpu_percent(interval=None):.0f}%"
            memory = f"{psutil.virtual_memory().percent:.0f}%"
        self.canvas.create_text(
            30,
            34,
            text=f"SYSTEM STATUS   CPU {cpu}   RAM {memory}",
            fill=GREEN,
            font=("Consolas", 9),
            anchor="w",
        )


def launch() -> None:
    """Launch the animated opening page."""

    OpeningPage().mainloop()


if __name__ == "__main__":
    launch()
