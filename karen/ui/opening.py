"""Karen's animated HUD: the main graphical, voice-first interface.

Performance notes (CPU-only laptops):
* Canvas items are created once per window size and only *moved* each frame,
  instead of deleting and recreating ~200 items 25 times a second.
* The frame rate drops to 10 fps when idle and animation pauses while minimized.
* System stats refresh once per second; the model status check runs on one
  slow timer and never overlaps itself.
* All AI, web and audio work runs on background threads; the Tk thread only
  drains a queue of events.
"""

from __future__ import annotations

import math
import queue
import random
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Any

from karen.core.application_service import ServiceResponse
from karen.voice.contracts import VoiceError

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

_MODE_COLORS = {
    "booting": CYAN,
    "ready": GREEN,
    "listening": "#ffe08a",
    "thinking": "#f5c451",
    "searching": "#b48cff",
    "speaking": "#61e7ff",
    "error": "#ff7185",
}
_ACTIVE_MODES = frozenset({"listening", "thinking", "searching", "speaking"})
_MODE_LABELS = {
    "booting": "BOOTING",
    "ready": "STANDING BY",
    "listening": "LISTENING",
    "thinking": "THINKING",
    "searching": "SEARCHING THE WEB",
    "speaking": "SPEAKING",
    "error": "ATTENTION",
}
_MAX_DISPLAY_CHARS = 900
_PARTICLES = 48
_WAVE_BARS = 42


@dataclass(slots=True)
class Particle:
    angle: float
    radius: float
    speed: float
    size: float
    alpha_step: int


class OpeningPage(tk.Tk):
    """Animated Karen window with a lightweight HUD aesthetic."""

    def __init__(self, *, speak: bool | None = None) -> None:
        super().__init__()
        self.title("Karen AI")
        self.geometry("1180x760")
        self.minsize(900, 600)
        self.configure(bg=BACKGROUND)
        self.protocol("WM_DELETE_WINDOW", self._quit)

        self._speak_override = speak
        self._runtime: Any | None = None
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._fullscreen = True
        self.attributes("-fullscreen", True)
        self._busy = False
        self._listening = False
        self._mode = "booting"
        self._reply = ""  # Accumulates streamed chunks of the current answer.
        self._display = "Bringing systems online..."
        self._heard = ""
        self._model_status = "CHECKING LOCAL MODEL..."
        self._status_inflight = False
        self._start_time = time.monotonic()
        self._items: dict[str, Any] = {}
        self._scene_size = (0, 0)
        self._rebuild_job: str | None = None
        self._closing = False
        self._center = (0.0, 0.0)
        self._scale = 1.0
        self._particles = [
            Particle(
                angle=random.random() * math.tau,
                radius=random.uniform(150, 430),
                speed=random.uniform(0.0003, 0.0012),
                size=random.choice((1, 1, 2)),
                alpha_step=random.randint(0, 100),
            )
            for _ in range(_PARTICLES)
        ]

        self.canvas = tk.Canvas(self, bg=BACKGROUND, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.bind("<Escape>", lambda _event: self._escape())
        self.bind("<F11>", lambda _event: self._toggle_fullscreen())
        self.bind("<Control-q>", lambda _event: self._quit())
        self.bind("<Control-space>", lambda _event: self._start_listening())
        self.bind("<Control-h>", lambda _event: self._toggle_history())

        self._build_controls()
        self.after(30, self._tick)
        self.after(50, self._poll_events)
        self.after(1000, self._refresh_system_stats)
        threading.Thread(target=self._boot, name="karen-boot", daemon=True).start()

    # -- startup ----------------------------------------------------------

    def _boot(self) -> None:
        """Build services off the Tk thread so the window appears instantly."""

        try:
            from karen.app import KarenRuntime

            runtime = KarenRuntime(on_speech_state=lambda speaking: self._events.put(("speaking", speaking)))
            if self._speak_override is not None:
                runtime.speaker.enabled = self._speak_override and runtime.speaker.available
            runtime.start_background(lambda message: self._events.put(("reminder", message)))
            if runtime.preferences.voice_enabled or runtime.preferences.wake_word_enabled:
                runtime.recognizer.preload()
            self._events.put(("ready", runtime))
        except Exception as error:  # pragma: no cover - environment dependent
            self._events.put(("init_error", str(error)))

    def _on_ready(self, runtime: Any) -> None:
        self._runtime = runtime
        prefs = runtime.preferences
        for widget in (self._command_entry, self._send_button, self._voice_button, self._speak_button,
                       self._wake_button, self._stop_button, self._history_button):
            widget.configure(state=tk.NORMAL)
        self._update_toggle_labels()
        self._command_entry.focus_set()
        name = runtime.memory.user_name()
        hour = time.localtime().tm_hour
        part = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
        greeting = f"Good {part}{', ' + name if name else ''}. All systems are online."
        self._set_mode("ready")
        self._show_reply(greeting, heard="")
        self._log("KAREN", greeting)
        runtime.speaker.say(greeting)
        if prefs.wake_word_enabled:
            self._enable_wake_word()
        self._schedule_model_status(1500)

    # -- controls ---------------------------------------------------------

    def _button(self, text: str, command, *, bg: str = "#0b4052", size: int = 9) -> tk.Button:  # noqa: ANN001
        return tk.Button(
            self, text=text, command=command, bg=bg, activebackground="#16657a", fg=BRIGHT_CYAN,
            activeforeground="white", disabledforeground="#3d6575", relief="flat", bd=0, padx=12,
            pady=7, cursor="hand2", state=tk.DISABLED, font=("Consolas", size, "bold"),
        )

    def _build_controls(self) -> None:
        self._command_entry = tk.Entry(
            self, bg="#0a1e2c", fg=BRIGHT_CYAN, insertbackground=CYAN, relief="flat",
            font=("Consolas", 11), state=tk.DISABLED, disabledbackground="#07141e",
        )
        self._command_entry.bind("<Return>", lambda _event: self._send_command())
        self._send_button = self._button("SEND", self._send_command)
        self._voice_button = self._button("● TALK", self._start_listening, bg="#123449")
        self._stop_button = self._button("■ STOP", self._stop_everything, bg="#3a1b27", size=8)
        self._speak_button = self._button("VOICE: ON", self._toggle_speech, bg="#0d2a3a", size=8)
        self._wake_button = self._button("WAKE: OFF", self._toggle_wake_word, bg="#0d2a3a", size=8)
        self._history_button = self._button("HISTORY", self._toggle_history, bg="#0d2a3a", size=8)

        # One centered control bar: pixel offsets from the middle keep spacing even at any width.
        self._command_entry.place(relx=0.5, x=-150, rely=0.885, width=560, height=34, anchor="center")
        self._send_button.place(relx=0.5, x=175, rely=0.885, anchor="center")
        self._voice_button.place(relx=0.5, x=260, rely=0.885, anchor="center")
        self._stop_button.place(relx=0.5, x=345, rely=0.885, anchor="center")
        self._speak_button.place(relx=0.5, x=-110, rely=0.94, anchor="center")
        self._wake_button.place(relx=0.5, x=0, rely=0.94, anchor="center")
        self._history_button.place(relx=0.5, x=110, rely=0.94, anchor="center")

        self._history = tk.Text(
            self, bg=PANEL, fg=BRIGHT_CYAN, insertbackground=CYAN, relief="flat", wrap="word",
            font=("Consolas", 10), padx=12, pady=10, state=tk.DISABLED, highlightthickness=1,
            highlightbackground="#174456",
        )
        self._history.tag_configure("you", foreground="#ffe08a", font=("Consolas", 9, "bold"))
        self._history.tag_configure("karen", foreground=GREEN, font=("Consolas", 9, "bold"))
        self._history_visible = False

    def _update_toggle_labels(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        speaking = runtime.speaker.enabled
        self._speak_button.configure(text="VOICE: ON" if speaking else "VOICE: OFF",
                                     fg=GREEN if speaking else MUTED)
        wake = runtime.wake_word is not None
        self._wake_button.configure(text="WAKE: ON" if wake else "WAKE: OFF", fg=GREEN if wake else MUTED)

    def _toggle_speech(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        if not runtime.speaker.available:
            self._show_reply("No text-to-speech engine is available. Install comtypes: pip install comtypes")
            return
        runtime.speaker.enabled = not runtime.speaker.enabled
        if not runtime.speaker.enabled:
            runtime.speaker.stop()
        runtime.preferences.speak_responses = runtime.speaker.enabled
        runtime.save_preferences()
        self._update_toggle_labels()

    def _toggle_wake_word(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        if runtime.wake_word is None:
            self._enable_wake_word()
        else:
            runtime.stop_wake_word()
        runtime.preferences.wake_word_enabled = runtime.wake_word is not None
        runtime.save_preferences()
        self._update_toggle_labels()

    def _enable_wake_word(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        runtime.recognizer.preload()
        runtime.start_wake_word(
            lambda: self._events.put(("wake", None)),
            on_error=lambda message: self._events.put(("wake_error", message)),
        )
        self._update_toggle_labels()

    def _toggle_history(self) -> None:
        if self._history_visible:
            self._history.place_forget()
        else:
            self._history.place(relx=0.985, rely=0.08, relwidth=0.3, relheight=0.74, anchor="ne")
            self._history.see(tk.END)
        self._history_visible = not self._history_visible

    def _log(self, speaker: str, text: str) -> None:
        self._history.configure(state=tk.NORMAL)
        self._history.insert(tk.END, f"{speaker}\n", "you" if speaker == "YOU" else "karen")
        self._history.insert(tk.END, f"{text}\n\n")
        # Keep the widget small: trim the oldest lines.
        if int(self._history.index("end-1c").split(".")[0]) > 600:
            self._history.delete("1.0", "200.0")
        self._history.see(tk.END)
        self._history.configure(state=tk.DISABLED)

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        self.attributes("-fullscreen", self._fullscreen)

    def _escape(self) -> None:
        runtime = self._runtime
        if runtime is not None and (runtime.speaker.is_speaking or self._listening):
            self._stop_everything()
        elif self._fullscreen:
            self._toggle_fullscreen()

    def _quit(self) -> None:
        if self._closing:
            return
        self._closing = True
        runtime = self._runtime
        if runtime is not None:
            try:
                runtime.shutdown()
            except Exception:  # pragma: no cover - best effort on exit
                pass
        self.destroy()

    # -- requests ---------------------------------------------------------

    def _send_command(self) -> None:
        text = self._command_entry.get().strip()
        if not text or self._runtime is None or self._busy:
            return
        self._command_entry.delete(0, tk.END)
        self._submit(text)

    def _submit(self, text: str) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        runtime.speaker.stop()  # Barge-in: a new request interrupts speech.
        if runtime.wake_word is not None:
            runtime.wake_word.pause()
        self._busy = True
        self._reply = ""
        self._show_reply("", heard=text)
        self._log("YOU", text)
        self._set_mode("thinking")
        self._send_button.configure(state=tk.DISABLED)
        self._voice_button.configure(state=tk.DISABLED)
        threading.Thread(target=self._process, args=(runtime, text), name="karen-request", daemon=True).start()

    def _process(self, runtime: Any, text: str) -> None:
        def on_chunk(chunk: str) -> None:
            self._events.put(("chunk", chunk))
            runtime.speaker.feed(chunk)

        try:
            response = runtime.service.handle_text(
                text, on_chunk=on_chunk, on_status=lambda state: self._events.put(("status", state))
            )
        except Exception as error:  # pragma: no cover - defensive
            response = ServiceResponse(f"Something went wrong: {error}", "error", False)
            runtime.speaker.say(response.message)
        runtime.speaker.flush()
        self._events.put(("response", response))

    def _start_listening(self) -> None:
        runtime = self._runtime
        if runtime is None or self._busy:
            return
        runtime.speaker.stop()
        if runtime.wake_word is not None:
            runtime.wake_word.pause()
        self._busy = True
        self._listening = True
        self._set_mode("listening")
        self._show_reply("Listening... speak now.", heard="")
        self._send_button.configure(state=tk.DISABLED)
        self._voice_button.configure(state=tk.DISABLED, text="LISTENING")
        threading.Thread(target=self._listen, args=(runtime,), name="karen-listen", daemon=True).start()

    def _listen(self, runtime: Any) -> None:
        try:
            self._events.put(("transcript", runtime.recognizer.transcribe_once()))
        except VoiceError as error:
            self._events.put(("voice_error", str(error)))
        except Exception as error:  # pragma: no cover - device dependent
            self._events.put(("voice_error", f"Microphone problem: {error}"))

    def _stop_everything(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        runtime.speaker.stop()
        if self._listening:
            runtime.recognizer.cancel()
        if not self._busy:
            self._set_mode("ready")

    def _finish_request(self) -> None:
        self._busy = False
        self._listening = False
        self._send_button.configure(state=tk.NORMAL)
        self._voice_button.configure(state=tk.NORMAL, text="● TALK")
        runtime = self._runtime
        if runtime is not None and runtime.wake_word is not None:
            runtime.wake_word.resume()
        if runtime is not None and runtime.speaker.is_speaking:
            self._set_mode("speaking")
        elif self._mode != "error":
            self._set_mode("ready")
        self._command_entry.focus_set()

    # -- event queue ------------------------------------------------------

    def _poll_events(self) -> None:
        if self._closing:
            return
        try:
            for _ in range(200):
                kind, payload = self._events.get_nowait()
                self._handle_event(kind, payload)
        except queue.Empty:
            pass
        self.after(50, self._poll_events)

    def _handle_event(self, kind: str, payload: Any) -> None:
        if kind == "ready":
            self._on_ready(payload)
        elif kind == "init_error":
            self._set_mode("error")
            self._show_reply(f"Startup failed: {payload}")
        elif kind == "status":
            if self._busy:
                self._set_mode("searching" if payload == "searching" else "thinking")
        elif kind == "chunk":
            self._reply += payload
            self._show_reply(self._reply)
        elif kind == "response":
            response: ServiceResponse = payload
            self._show_reply(response.message)
            self._log("KAREN", response.message)
            if response.source == "error":
                self._set_mode("error")
                self.after(2500, lambda: self._mode == "error" and not self._busy and self._set_mode("ready"))
            self._finish_request()
        elif kind == "transcript":
            self._listening = False
            self._busy = False
            self._submit(payload)
        elif kind == "voice_error":
            self._show_reply(payload)
            self._set_mode("error")
            self._finish_request()
            self.after(2500, lambda: self._mode == "error" and not self._busy and self._set_mode("ready"))
        elif kind == "speaking":
            if not self._busy:
                self._set_mode("speaking" if payload else "ready")
        elif kind == "reminder":
            self._log("KAREN", payload)
            if not self._busy:
                self._show_reply(payload, heard="")
            if self._runtime is not None:
                self._runtime.speaker.say(payload)
        elif kind == "wake":
            if self._runtime is not None:
                self._runtime.speaker.stop()
            if self._busy:
                if self._runtime is not None and self._runtime.wake_word is not None:
                    self._runtime.wake_word.resume()
            else:
                self._start_listening()
        elif kind == "wake_error":
            if self._runtime is not None:
                self._runtime.stop_wake_word()
                self._runtime.preferences.wake_word_enabled = False
            self._update_toggle_labels()
            self._show_reply(payload)
        elif kind == "model_status":
            self._model_status, next_check_ms = payload
            self._set_text("status", self._status_line())
            self._schedule_model_status(next_check_ms)

    # -- model status -----------------------------------------------------

    def _schedule_model_status(self, delay_ms: int) -> None:
        if not self._closing:
            self.after(delay_ms, self._refresh_model_status)

    def _refresh_model_status(self) -> None:
        runtime = self._runtime
        if runtime is None or self._status_inflight:
            self._schedule_model_status(10_000)
            return
        self._status_inflight = True

        def check() -> None:
            try:
                available = runtime.llm.is_available()
                ready = available and runtime.llm.use_installed_model() is not None
            finally:
                self._status_inflight = False
            status = (
                f"{runtime.llm.model} ONLINE // LOCAL".upper()
                if ready
                else f"OLLAMA ONLINE // NO MODEL: RUN OLLAMA PULL {runtime.settings.ollama_model}".upper()
                if available
                else "COMMANDS + WEB READY // LOCAL MODEL OFFLINE"
            )
            # Check rarely once healthy; a little more often while offline.
            self._events.put(("model_status", (status, 30_000 if ready else 10_000)))

        threading.Thread(target=check, name="karen-model-status", daemon=True).start()

    def _refresh_system_stats(self) -> None:
        if self._closing:
            return
        if psutil is not None and self._items:
            text = f"SYSTEM   CPU {psutil.cpu_percent(interval=None):.0f}%   RAM {psutil.virtual_memory().percent:.0f}%"
            self._set_text("sys", text)
        self.after(1000, self._refresh_system_stats)

    # -- state helpers ----------------------------------------------------

    def _set_mode(self, mode: str) -> None:
        if mode == self._mode and self._items:
            return
        self._mode = mode
        self._apply_colors()
        self._set_text("status", self._status_line())
        self._set_text("link", f"LINK // {_MODE_LABELS.get(mode, mode.upper())}")

    def _status_line(self) -> str:
        return f"{_MODE_LABELS.get(self._mode, self._mode.upper())}  //  {self._model_status}"

    def _show_reply(self, text: str, heard: str | None = None) -> None:
        self._display = text
        display = text if len(text) <= _MAX_DISPLAY_CHARS else text[:_MAX_DISPLAY_CHARS].rsplit(" ", 1)[0] + " ...  (full reply: Ctrl+H)"
        self._set_text("reply", display)
        if heard is not None:
            self._heard = heard
            self._set_text("heard", f"YOU: {heard}" if heard else "")

    def _set_text(self, key: str, text: str) -> None:
        item = self._items.get(key)
        if item is not None and self.canvas.itemcget(item, "text") != text:
            self.canvas.itemconfigure(item, text=text)

    # -- scene ------------------------------------------------------------

    def _on_resize(self, _event: tk.Event) -> None:
        if self._rebuild_job is not None:
            self.after_cancel(self._rebuild_job)
        self._rebuild_job = self.after(120, self._build_scene)

    def _build_scene(self) -> None:
        """Create every canvas item once for the current window size."""

        self._rebuild_job = None
        width = max(self.canvas.winfo_width(), 900)
        height = max(self.canvas.winfo_height(), 600)
        self._scene_size = (width, height)
        canvas = self.canvas
        canvas.delete("all")
        items: dict[str, Any] = {}
        cx, cy = width / 2, height * 0.34
        self._scale = max(0.6, min(1.15, height / 1000))

        for index in range(10):
            margin = index * 38
            shade = "#" + format(max(4, 14 - index), "02x") + format(max(10, 25 - index), "02x") + format(max(18, 38 - index), "02x")
            canvas.create_rectangle(margin, margin, width - margin, height - margin, outline=shade)
        items["scan"] = [
            canvas.create_line(0, 0, width, 0, fill="#0a3443"),
            canvas.create_line(0, 0, width, 0, tags=("accent_line",)),
            canvas.create_line(0, 0, width, 0, fill="#061e2b"),
        ]
        items["particles"] = [
            canvas.create_oval(0, 0, 0, 0, outline="", tags=("accent_fill",) if particle.alpha_step % 3 else ())
            for particle in self._particles
        ]
        for particle, item in zip(self._particles, items["particles"]):
            if particle.alpha_step % 3 == 0:
                canvas.itemconfigure(item, fill=BLUE)
        bracket = 42
        for x, direction in ((28, 1), (width - 28, -1)):
            canvas.create_line(x, 112, x + direction * bracket, 112, tags=("accent_line",))
            canvas.create_line(x, 112, x, 152, tags=("accent_line",))
            canvas.create_line(x, height - 112, x + direction * bracket, height - 112, fill="#174456")
            canvas.create_line(x, height - 112, x, height - 152, fill="#174456")
        items["link"] = canvas.create_text(32, 128, text="", font=("Consolas", 8), anchor="w", tags=("accent_text",))
        canvas.create_text(width - 32, 128, text="LOCAL / PRIVATE", fill=MUTED, font=("Consolas", 8), anchor="e")
        items["radar"] = canvas.create_arc(0, 0, 0, 0, extent=18, width=2, style=tk.ARC, tags=("accent_line_outline",))
        items["rings"] = [
            canvas.create_oval(0, 0, 0, 0, outline=color, width=line_width, tags=tags)
            for color, line_width, tags in (
                ("#092b3b", 1, ()), ("#0d536b", 1, ()), (BLUE, 2, ()), ("", 1, ("accent_line_outline",)),
            )
        ]
        items["arcs"] = [
            canvas.create_arc(0, 0, 0, 0, extent=extent, width=line_width, style=tk.ARC, outline=color, tags=tags)
            for extent, color, line_width, tags in (
                (92, "", 3, ("accent_line_outline",)), (44, BLUE, 2, ()), (70, "", 2, ("accent_line_outline",)),
            )
        ]
        items["inner"] = canvas.create_oval(0, 0, 0, 0, fill="#061522", width=2, tags=("accent_line_outline",))
        items["core"] = canvas.create_oval(0, 0, 0, 0, outline=BRIGHT_CYAN, tags=("accent_fill",))
        items["ticks"] = [
            canvas.create_line(0, 0, 0, 0, fill=CYAN if angle % 60 == 0 else MUTED, width=2 if angle % 60 == 0 else 1)
            for angle in range(0, 360, 30)
        ]
        items["wave"] = [canvas.create_line(0, 0, 0, 0, width=2 if index % 4 == 0 else 1) for index in range(_WAVE_BARS)]

        for offset, color in ((3, "#082c3b"), (1, "#17637a"), (0, BRIGHT_CYAN)):
            canvas.create_text(cx + offset, height * 0.09 + offset, text="K A R E N", fill=color, font=("Consolas", 30, "bold"))
        items["heard"] = canvas.create_text(cx, height * 0.635, text="", fill="#ffe08a", font=("Consolas", 10), width=width * 0.7)
        items["reply"] = canvas.create_text(
            cx, height * 0.665, text="", fill=BRIGHT_CYAN, font=("Segoe UI", 12), width=min(width * 0.72, 980),
            anchor="n", justify="center",
        )
        items["status"] = canvas.create_text(cx, height * 0.61, text="", font=("Consolas", 8), tags=("accent_text",))
        items["sys"] = canvas.create_text(30, 34, text="SYSTEM   CPU --   RAM --", font=("Consolas", 9), anchor="w", tags=("accent_text",))
        canvas.create_text(
            28, height - 22, fill="#397082", font=("Consolas", 8), anchor="w",
            text="ENTER SEND  •  CTRL+SPACE TALK  •  ESC STOP  •  CTRL+H HISTORY  •  F11 FULLSCREEN  •  CTRL+Q QUIT",
        )
        self._items = items
        self._center = (cx, cy)
        self._apply_colors()
        self._set_text("status", self._status_line())
        self._set_text("link", f"LINK // {_MODE_LABELS.get(self._mode, self._mode.upper())}")
        self._show_reply(self._display, heard=self._heard)
        self._draw_frame()

    def _apply_colors(self) -> None:
        if not self._items:
            return
        color = _MODE_COLORS.get(self._mode, CYAN)
        self.canvas.itemconfigure("accent_line", fill=color)
        self.canvas.itemconfigure("accent_line_outline", outline=color)
        self.canvas.itemconfigure("accent_fill", fill=color)
        self.canvas.itemconfigure("accent_text", fill=color)
        active = self._mode in _ACTIVE_MODES
        for item in self._items.get("wave", []):
            self.canvas.itemconfigure(item, fill=color if active else "#15516a")

    # -- animation --------------------------------------------------------

    def _tick(self) -> None:
        if self._closing:
            return
        if self.state() == "iconic":
            self.after(500, self._tick)
            return
        if not self._items:
            self._build_scene()
        else:
            self._draw_frame()
        self.after(50 if self._mode in _ACTIVE_MODES else 100, self._tick)

    def _draw_frame(self) -> None:
        canvas = self.canvas
        items = self._items
        width, height = self._scene_size
        cx, cy = self._center
        elapsed = time.monotonic() - self._start_time
        pulse = (math.sin(elapsed * 2.2) + 1) / 2
        active = self._mode in _ACTIVE_MODES
        rotation = elapsed * (0.65 if active else 0.25)

        scan_y = (elapsed * 95) % height
        for offset, item in zip((0, -1, 1), items["scan"]):
            canvas.coords(item, 0, scan_y + offset, width, scan_y + offset)

        scale = self._scale
        squash = 0.56
        for particle, item in zip(self._particles, items["particles"]):
            angle = particle.angle + elapsed * particle.speed * 80
            radius = (particle.radius + math.sin(elapsed + particle.alpha_step) * 7) * scale
            x = cx + math.cos(angle) * radius
            y = cy + math.sin(angle) * radius * squash
            canvas.coords(item, x, y, x + particle.size, y + particle.size)

        base = (128 + pulse * 8 + (8 if self._mode == "listening" else 0)) * scale
        for offset, item in zip((30, 18, 7, 0), items["rings"]):
            radius = base + offset * scale
            canvas.coords(item, cx - radius, cy - radius, cx + radius, cy + radius)
        radius = base + 5
        for (speed, phase), item in zip(((55, 0), (-37, 120), (23, 240)), items["arcs"]):
            canvas.coords(item, cx - radius, cy - radius, cx + radius, cy + radius)
            canvas.itemconfigure(item, start=rotation * speed + phase)
        inner = (58 + pulse * 5) * scale
        canvas.coords(items["inner"], cx - inner, cy - inner, cx + inner, cy + inner)
        core = 9 + pulse * 2 + (4 if self._mode == "speaking" else 0)
        canvas.coords(items["core"], cx - core, cy - core, cx + core, cy + core)
        for index, item in enumerate(items["ticks"]):
            radians = math.radians(index * 30) + rotation
            inner_tick = base + 38 * scale
            outer_tick = inner_tick + (10 if index % 2 == 0 else 5) * scale
            canvas.coords(
                item,
                cx + math.cos(radians) * inner_tick, cy + math.sin(radians) * inner_tick,
                cx + math.cos(radians) * outer_tick, cy + math.sin(radians) * outer_tick,
            )
        sweep_radius = min(width, height) * 0.26
        canvas.coords(items["radar"], cx - sweep_radius, cy - sweep_radius, cx + sweep_radius, cy + sweep_radius)
        canvas.itemconfigure(items["radar"], start=(elapsed * 90) % 360)

        wave_y = height * 0.585
        for index, item in enumerate(items["wave"]):
            x = cx - 210 + index * 10
            wave = math.sin(elapsed * (8 if active else 3) + index * 0.7)
            amplitude = (16 if active else 4) * (0.45 + abs(wave) * 0.55)
            canvas.coords(item, x, wave_y - amplitude, x, wave_y + amplitude)


def launch(*, speak: bool | None = None) -> None:
    """Launch the animated HUD."""

    OpeningPage(speak=speak).mainloop()


if __name__ == "__main__":
    launch()
