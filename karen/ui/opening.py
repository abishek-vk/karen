"""Karen's cinematic HUD: the main graphical, voice-first interface.

Design: dark navy "FUI" (fictional user interface) look with an arc-reactor
core, audio-reactive rings, a boot sequence, subtitle-style replies and eased
color transitions between modes.

Performance (CPU-only laptops):
* Every canvas item is created once per window size and then only moved or
  recolored; nothing is deleted and recreated per frame.
* ~30 fps while listening/thinking/speaking, ~12 fps idle, ~2 fps while
  minimized; heavier effects are skipped when idle.
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
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from karen.core.application_service import ServiceResponse
from karen.voice.contracts import VoiceError

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency
    psutil = None

# -- theme ------------------------------------------------------------------

BACKGROUND = "#04070d"
BG_TOP = (6, 12, 22)
BG_BOTTOM = (2, 4, 9)
PANEL = "#08111c"
PANEL_BORDER = "#15293c"
GRID = "#0a1522"
TEXT = "#e8f6ff"
TEXT_DIM = "#8aa4b8"
MUTED = "#4d6578"
CYAN = "#4fd8ff"
BRIGHT_CYAN = "#c8f4ff"
GREEN = "#3ddcb0"
DANGER = "#ff5c7a"

# Kept for backwards compatibility with older imports.
BLUE = "#1c9fce"

MODE_COLORS = {
    "booting": "#4fd8ff",
    "ready": "#4fd8ff",
    "listening": "#ffb547",
    "thinking": "#9b8cff",
    "searching": "#3ddcb0",
    "speaking": "#62e3ff",
    "error": "#ff5c7a",
}
MODE_LABELS = {
    "booting": "INITIALIZING",
    "ready": "STANDING BY",
    "listening": "LISTENING",
    "thinking": "THINKING",
    "searching": "SEARCHING THE WEB",
    "speaking": "SPEAKING",
    "error": "ATTENTION",
}
ACTIVE_MODES = frozenset({"booting", "listening", "thinking", "searching", "speaking"})

FONT_TITLE = ("Bahnschrift Light", 30)
FONT_CLOCK = ("Bahnschrift Light", 26)
FONT_LABEL = ("Bahnschrift SemiLight", 9)
FONT_VALUE = ("Bahnschrift", 11)
FONT_MONO = ("Cascadia Mono", 8)
FONT_CAPTION = ("Segoe UI Variable Display", 15)
FONT_HEARD = ("Segoe UI Variable Text", 10)
FONT_BUTTON = ("Bahnschrift SemiLight", 10)
FONT_INPUT = ("Segoe UI Variable Text", 12)

_MAX_DISPLAY_CHARS = 650
_BARS = 48
_SEGMENTS = 18
_DUST = 34
_BOOT_SECONDS = 2.6
_BOOT_LINES = (
    "INITIALIZING NEURAL CORE",
    "MOUNTING MEMORY BANKS",
    "CALIBRATING VOICE SUBSYSTEM",
    "ESTABLISHING SECURE LINK",
)

# -- helpers ----------------------------------------------------------------


def _rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(value)))) for value in rgb)


def _mix(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def _rounded(canvas: tk.Canvas, x1: float, y1: float, x2: float, y2: float, r: float, **kwargs: Any) -> int:
    """A smooth rounded rectangle drawn as one polygon."""

    r = min(r, (x2 - x1) / 2, (y2 - y1) / 2)
    points = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
        x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


@dataclass(slots=True)
class Dust:
    x: float
    y: float
    speed: float
    drift: float
    size: int


@dataclass(slots=True)
class ButtonSpec:
    key: str
    label: str
    command: Callable[[], None]
    enabled: bool = False
    active: bool = False
    danger: bool = False


class OpeningPage(tk.Tk):
    """Animated Karen window with a cinematic HUD aesthetic."""

    def __init__(self, *, speak: bool | None = None) -> None:
        super().__init__()
        self.title("Karen AI")
        self.geometry("1280x800")
        self.minsize(960, 640)
        self.configure(bg=BACKGROUND)
        self.protocol("WM_DELETE_WINDOW", self._quit)

        reduced_motion, fullscreen = False, True
        try:
            from karen.app import PROJECT_DIR
            from karen.config import ConfigurationStore

            settings = ConfigurationStore(PROJECT_DIR / "config").load_settings()
            reduced_motion, fullscreen = settings.ui_reduced_motion, settings.ui_fullscreen
        except Exception:  # pragma: no cover - settings problems are reported by the runtime
            pass
        self._reduced_motion = reduced_motion
        self._fullscreen = fullscreen
        self.attributes("-fullscreen", fullscreen)

        self._speak_override = speak
        self._runtime: Any | None = None
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._busy = False
        self._listening = False
        self._mode = "booting"
        self._closing = False
        self._focused = True
        self._model_status = "CHECKING LOCAL MODEL"
        self._status_inflight = False
        self._boot_lines: list[str] = []
        self._boot_done = reduced_motion
        self._pending_ready: Any | None = None
        self._start_time = time.monotonic()
        self._last_frame = self._start_time
        self._dust_dt = 0.0

        # Subtitle reveal state.
        self._reply = ""  # Accumulates streamed chunks of the current answer.
        self._target_text = ""
        self._shown = 0.0
        self._heard = ""

        # Visual state.
        self._accent = _rgb(MODE_COLORS["booting"])
        self._accent_target = self._accent
        self._colors_dirty = True
        self._level = 0.0
        self._cpu_history: deque[float] = deque([0.0] * 40, maxlen=40)
        self._cpu = 0.0
        self._ram = 0.0
        self._next_reminder = "NONE"
        self._toast_until = 0.0
        self._toast_start = 0.0
        self._history_visible = False
        self._history_anim: tuple[float, float, float] | None = None
        self._dust = [
            Dust(random.random(), random.random(), random.uniform(0.004, 0.014), random.uniform(-0.004, 0.004),
                 random.choice((1, 1, 2)))
            for _ in range(_DUST)
        ]
        self._buttons: dict[str, ButtonSpec] = {
            spec.key: spec
            for spec in (
                ButtonSpec("talk", "TALK", self._start_listening),
                ButtonSpec("send", "SEND", self._send_command),
                ButtonSpec("stop", "STOP", self._stop_everything, danger=True),
                ButtonSpec("voice", "VOICE", self._toggle_speech),
                ButtonSpec("wake", "WAKE WORD", self._toggle_wake_word),
                ButtonSpec("history", "HISTORY", self._toggle_history),
            )
        }
        self._items: dict[str, Any] = {}
        self._last_coords: dict[int, tuple[int, ...]] = {}
        self._last_options: dict[tuple[int, str], Any] = {}
        self._scene_size = (0, 0)
        self._rebuild_job: str | None = None

        self.canvas = tk.Canvas(self, bg=BACKGROUND, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.bind("<Escape>", lambda _event: self._escape())
        self.bind("<F11>", lambda _event: self._toggle_fullscreen())
        self.bind("<Control-q>", lambda _event: self._quit())
        self.bind("<Control-space>", lambda _event: self._start_listening())
        self.bind("<Control-h>", lambda _event: self._toggle_history())
        # Focus events bubble up from child widgets, so ask Tk whether the app still has focus.
        self.bind("<FocusIn>", lambda _event: self._update_focus())
        self.bind("<FocusOut>", lambda _event: self.after(50, self._update_focus))

        self._build_widgets()
        self.after(16, self._tick)
        self.after(50, self._poll_events)
        self.after(500, self._refresh_system_stats)
        threading.Thread(target=self._boot, name="karen-boot", daemon=True).start()

    # -- startup ----------------------------------------------------------

    def _boot(self) -> None:
        """Build services off the Tk thread so the window appears instantly."""

        try:
            self._events.put(("boot", "LOADING CONFIGURATION"))
            from karen.app import KarenRuntime

            runtime = KarenRuntime(on_speech_state=lambda speaking: self._events.put(("speaking", speaking)))
            self._events.put(("boot", f"LOCAL MODEL: {runtime.settings.ollama_model.upper()}"))
            if self._speak_override is not None:
                runtime.speaker.enabled = self._speak_override and runtime.speaker.available
            runtime.start_background(lambda message: self._events.put(("reminder", message)))
            if runtime.preferences.voice_enabled or runtime.preferences.wake_word_enabled:
                runtime.recognizer.preload()
            self._events.put(("boot", "ALL SYSTEMS NOMINAL"))
            self._events.put(("ready", runtime))
        except Exception as error:  # pragma: no cover - environment dependent
            self._events.put(("init_error", str(error)))

    def _on_ready(self, runtime: Any) -> None:
        if not self._boot_done:
            self._pending_ready = runtime  # Finish the boot animation first.
            return
        self._runtime = runtime
        for key in self._buttons:
            self._buttons[key].enabled = True
        self._command_entry.configure(state=tk.NORMAL)
        self._sync_toggles()
        self._command_entry.focus_set()
        name = runtime.memory.user_name()
        hour = time.localtime().tm_hour
        part = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
        greeting = f"Good {part}{', ' + name if name else ''}. All systems are online."
        self._set_mode("ready")
        self._show_reply(greeting, heard="")
        self._log("KAREN", greeting)
        runtime.speaker.say(greeting)
        if runtime.preferences.wake_word_enabled:
            self._enable_wake_word()
        self._refresh_reminder()
        self._schedule_model_status(1200)

    # -- widgets ----------------------------------------------------------

    def _build_widgets(self) -> None:
        self._command_entry = tk.Entry(
            self, bg=PANEL, fg=TEXT, insertbackground=CYAN, relief="flat", bd=0,
            font=FONT_INPUT, state=tk.DISABLED, disabledbackground=PANEL, highlightthickness=0,
        )
        self._command_entry.bind("<Return>", lambda _event: self._send_command())
        self._command_entry.bind("<FocusIn>", lambda _event: self._set_input_focus(True))
        self._command_entry.bind("<FocusOut>", lambda _event: self._set_input_focus(False))

        self._history = tk.Text(
            self, bg=PANEL, fg=TEXT, insertbackground=CYAN, relief="flat", wrap="word",
            font=("Segoe UI Variable Text", 10), padx=16, pady=14, state=tk.DISABLED,
            highlightthickness=1, highlightbackground=PANEL_BORDER, spacing1=2, spacing3=4,
        )
        self._history.tag_configure("you", foreground="#ffb547", font=("Bahnschrift SemiLight", 9))
        self._history.tag_configure("karen", foreground=CYAN, font=("Bahnschrift SemiLight", 9))

    def _update_focus(self) -> None:
        if not self._closing:
            self._focused = self.focus_displayof() is not None

    def _set_input_focus(self, focused: bool) -> None:
        item = self._items.get("input_border")
        if item is not None:
            self.canvas.itemconfigure(item, outline=_hex(self._accent) if focused else PANEL_BORDER)

    def _sync_toggles(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        self._buttons["voice"].active = bool(runtime.speaker.enabled)
        self._buttons["wake"].active = runtime.wake_word is not None
        self._buttons["history"].active = self._history_visible
        for key in self._buttons:
            self._paint_button(key)

    # -- toggles ----------------------------------------------------------

    def _toggle_speech(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        if not runtime.speaker.available:
            self._toast("No text-to-speech engine is available. Install comtypes: pip install comtypes")
            return
        runtime.speaker.enabled = not runtime.speaker.enabled
        if not runtime.speaker.enabled:
            runtime.speaker.stop()
        runtime.preferences.speak_responses = runtime.speaker.enabled
        runtime.save_preferences()
        self._sync_toggles()
        self._toast("Voice replies on." if runtime.speaker.enabled else "Voice replies muted.")

    def _toggle_wake_word(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        if runtime.wake_word is None:
            self._enable_wake_word()
            self._toast("Wake word on. Say: Hey Jarvis.")
        else:
            runtime.stop_wake_word()
            self._toast("Wake word off.")
        runtime.preferences.wake_word_enabled = runtime.wake_word is not None
        runtime.save_preferences()
        self._sync_toggles()

    def _enable_wake_word(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        runtime.recognizer.preload()
        runtime.start_wake_word(
            lambda: self._events.put(("wake", None)),
            on_error=lambda message: self._events.put(("wake_error", message)),
        )
        self._sync_toggles()

    def _toggle_history(self) -> None:
        if self._runtime is None and not self._history_visible:
            return
        self._history_visible = not self._history_visible
        start = 1.32 if self._history_visible else 0.985
        end = 0.985 if self._history_visible else 1.32
        if self._history_visible:
            self._history.place(relx=start, rely=0.11, relwidth=0.28, relheight=0.68, anchor="ne")
            self._history.see(tk.END)
        self._history_anim = (time.monotonic(), start, end)
        self._sync_toggles()

    def _animate_history(self, now: float) -> None:
        if self._history_anim is None:
            return
        started, start, end = self._history_anim
        progress = _ease_out((now - started) / 0.28)
        self._history.place_configure(relx=start + (end - start) * progress)
        if progress >= 1:
            self._history_anim = None
            if not self._history_visible:
                self._history.place_forget()

    def _log(self, speaker: str, text: str) -> None:
        self._history.configure(state=tk.NORMAL)
        stamp = time.strftime("%H:%M")
        self._history.insert(tk.END, f"{speaker}  ·  {stamp}\n", "you" if speaker == "YOU" else "karen")
        self._history.insert(tk.END, f"{text}\n\n")
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
        elif self._history_visible:
            self._toggle_history()
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
        self._set_buttons_enabled(False)
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
        self._show_reply("Listening. Go ahead.", heard="", instant=True)
        self._set_buttons_enabled(False)
        self._buttons["stop"].enabled = True
        self._paint_button("stop")
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

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for key in ("talk", "send"):
            self._buttons[key].enabled = enabled
            self._paint_button(key)

    def _finish_request(self) -> None:
        self._busy = False
        self._listening = False
        self._set_buttons_enabled(True)
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
        self.after(40, self._poll_events)

    def _recover_from_error(self) -> None:
        self.after(2600, lambda: self._mode == "error" and not self._busy and self._set_mode("ready"))

    def _handle_event(self, kind: str, payload: Any) -> None:
        if kind == "boot":
            self._boot_lines.append(payload)
        elif kind == "ready":
            self._on_ready(payload)
        elif kind == "init_error":
            self._boot_done = True
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
                self._recover_from_error()
            self._finish_request()
        elif kind == "transcript":
            self._listening = False
            self._busy = False
            self._submit(payload)
        elif kind == "voice_error":
            self._show_reply(payload, instant=True)
            self._set_mode("error")
            self._finish_request()
            self._recover_from_error()
        elif kind == "speaking":
            if not self._busy:
                self._set_mode("speaking" if payload else "ready")
        elif kind == "reminder":
            self._log("KAREN", payload)
            self._toast(payload, seconds=9)
            if self._runtime is not None:
                self._runtime.speaker.say(payload)
            self._refresh_reminder()
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
            self._sync_toggles()
            self._toast(payload, seconds=8)
        elif kind == "model_status":
            self._model_status, next_check_ms = payload
            self._set_text("model", self._model_status)
            self._schedule_model_status(next_check_ms)

    # -- periodic data ------------------------------------------------------

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
                f"{runtime.llm.model.upper()}  ·  ONLINE"
                if ready
                else "OLLAMA ONLINE  ·  NO MODEL"
                if available
                else "OFFLINE  ·  COMMANDS + WEB"
            )
            self._events.put(("model_status", (status, 30_000 if ready else 10_000)))

        threading.Thread(target=check, name="karen-model-status", daemon=True).start()

    def _refresh_system_stats(self) -> None:
        if self._closing:
            return
        if psutil is not None:
            self._cpu = psutil.cpu_percent(interval=None)
            self._ram = psutil.virtual_memory().percent
            self._cpu_history.append(self._cpu)
            self._update_telemetry()
        self.after(1000, self._refresh_system_stats)

    def _refresh_reminder(self) -> None:
        runtime = self._runtime
        if runtime is None or self._closing:
            return
        try:
            pending = runtime.reminders.pending()
        except Exception:  # pragma: no cover - database dependent
            pending = []
        if pending:
            text, due = pending[0]
            self._next_reminder = f"{due.strftime('%a %H:%M').upper()}  ·  {text[:22].upper()}"
        else:
            self._next_reminder = "NONE"
        self._set_text("reminder", self._next_reminder)
        self.after(30_000, self._refresh_reminder)

    # -- state helpers ----------------------------------------------------

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._accent_target = _rgb(MODE_COLORS.get(mode, CYAN))
        self._set_text("status", MODE_LABELS.get(mode, mode.upper()))

    def _show_reply(self, text: str, heard: str | None = None, *, instant: bool = False) -> None:
        if len(text) > _MAX_DISPLAY_CHARS:
            text = text[:_MAX_DISPLAY_CHARS].rsplit(" ", 1)[0] + "  …  (full reply: Ctrl+H)"
        if not text.startswith(self._target_text[: int(self._shown)]):
            self._shown = 0.0
        self._target_text = text
        if instant or self._reduced_motion:
            self._shown = float(len(text))
            self._set_text("reply", text)
        if heard is not None:
            self._heard = heard
            self._set_text("heard", f"YOU  ›  {heard}" if heard else "")

    def _toast(self, text: str, *, seconds: float = 4.0) -> None:
        self._set_text("toast_text", text[:110])
        self._toast_start = time.monotonic()
        self._toast_until = self._toast_start + seconds

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
        canvas = self.canvas
        width = max(canvas.winfo_width(), 960)
        height = max(canvas.winfo_height(), 640)
        self._scene_size = (width, height)
        canvas.delete("all")
        items: dict[str, Any] = {}
        # Buttons register their items into this same dict while the scene is built.
        self._items = items
        self._last_coords = {}
        self._last_options = {}
        cx, cy = width / 2, height * 0.40
        radius = min(width, height) * 0.155
        self._center, self._radius = (cx, cy), radius

        # Background: vertical gradient, faint grid, vignette.
        bands = 28
        for index in range(bands):
            color = _hex(_mix(BG_TOP, BG_BOTTOM, index / (bands - 1)))
            canvas.create_rectangle(0, height * index / bands, width, height * (index + 1) / bands + 1,
                                    fill=color, outline="")
        step = 64
        for x in range(step, int(width), step):
            canvas.create_line(x, 0, x, height, fill=GRID)
        for y in range(step, int(height), step):
            canvas.create_line(0, y, width, y, fill=GRID)
        for index in range(6):
            margin = index * 7
            shade = _hex(_mix((0, 0, 0), BG_TOP, index / 6))
            canvas.create_rectangle(margin, margin, width - margin, height - margin, outline=shade, width=7)
        items["dust"] = [canvas.create_oval(0, 0, 0, 0, fill="#16324a", outline="") for _ in self._dust]
        items["scan"] = canvas.create_line(0, 0, width, 0, fill="#0d2233")

        # Frame corners with blinking markers.
        corner, inset = 34, 26
        for x, y, dx, dy in ((inset, inset, 1, 1), (width - inset, inset, -1, 1),
                             (inset, height - inset, 1, -1), (width - inset, height - inset, -1, -1)):
            canvas.create_line(x, y, x + dx * corner, y, fill=PANEL_BORDER, width=2, tags=("acc_dim_line",))
            canvas.create_line(x, y, x, y + dy * corner, fill=PANEL_BORDER, width=2, tags=("acc_dim_line",))
        items["blink"] = [
            canvas.create_rectangle(x - 2, y - 2, x + 2, y + 2, outline="", tags=("acc_fill",))
            for x, y in ((inset + 10, inset + 10), (width - inset - 10, inset + 10))
        ]

        # Top bar: wordmark, title, clock.
        canvas.create_text(inset + 22, 58, text="K · A · R · E · N", fill=TEXT_DIM, font=("Bahnschrift SemiLight", 10), anchor="w")
        canvas.create_text(inset + 22, 76, text="PERSONAL INTELLIGENCE  ·  LOCAL CORE", fill=MUTED, font=FONT_MONO, anchor="w")
        items["title"] = canvas.create_text(cx, height * 0.075, text="", fill=TEXT, font=FONT_TITLE)
        items["title_rule"] = canvas.create_line(cx - 120, height * 0.075 + 30, cx + 120, height * 0.075 + 30,
                                                 tags=("acc_dim_line",))
        items["clock"] = canvas.create_text(width - inset - 22, 62, text="", fill=TEXT, font=FONT_CLOCK, anchor="e")
        items["date"] = canvas.create_text(width - inset - 22, 92, text="", fill=TEXT_DIM, font=FONT_MONO, anchor="e")

        # Side panels (only when there is room).
        if width >= 1180:
            self._build_panels(canvas, items, width, height, inset)

        # Arc-reactor core.
        items["halo"] = [
            canvas.create_oval(0, 0, 0, 0, outline="", width=width_px, tags=(f"halo{index}",))
            for index, width_px in enumerate((6, 4, 3, 2))
        ]
        items["bars"] = [canvas.create_line(0, 0, 0, 0, width=3, capstyle=tk.ROUND, tags=("acc_line",)) for _ in range(_BARS)]
        items["segments"] = [
            canvas.create_arc(cx - radius * 1.12, cy - radius * 1.12, cx + radius * 1.12, cy + radius * 1.12,
                              start=index * 360 / _SEGMENTS, extent=360 / _SEGMENTS - 6, style=tk.ARC, width=3,
                              tags=("acc_outline",))
            for index in range(_SEGMENTS)
        ]
        ticks_r1, ticks_r2 = radius * 1.24, radius * 1.29
        for index in range(72):
            angle = math.radians(index * 5)
            r2 = ticks_r2 + (radius * 0.04 if index % 6 == 0 else 0)
            canvas.create_line(cx + math.cos(angle) * ticks_r1, cy + math.sin(angle) * ticks_r1,
                               cx + math.cos(angle) * r2, cy + math.sin(angle) * r2,
                               fill=PANEL_BORDER if index % 6 else MUTED)
        items["ring_outer"] = canvas.create_oval(cx - radius * 1.34, cy - radius * 1.34, cx + radius * 1.34,
                                                 cy + radius * 1.34, outline=PANEL_BORDER)
        items["arcs"] = [
            canvas.create_arc(0, 0, 0, 0, extent=extent, style=tk.ARC, width=line_width, tags=tags)
            for extent, line_width, tags in ((110, 2, ("acc_outline",)), (60, 2, ("acc_dim_outline",)),
                                             (28, 4, ("acc_outline",)))
        ]
        items["scanner"] = canvas.create_arc(0, 0, 0, 0, extent=26, style=tk.ARC, width=1, tags=("acc_outline",))
        items["inner_ring"] = canvas.create_oval(0, 0, 0, 0, width=2, tags=("acc_outline",))
        items["inner_fill"] = canvas.create_oval(0, 0, 0, 0, outline="", tags=("core_fill",))
        items["core"] = canvas.create_oval(0, 0, 0, 0, outline=BRIGHT_CYAN, width=1, tags=("acc_fill",))
        items["core_hot"] = canvas.create_oval(0, 0, 0, 0, fill="#ffffff", outline="")
        items["orbits"] = [canvas.create_oval(0, 0, 0, 0, outline="", tags=("acc_fill",)) for _ in range(3)]

        # Status pill under the core.
        pill_y = cy + radius * 1.62
        _rounded(canvas, cx - 112, pill_y - 15, cx + 112, pill_y + 15, 15, fill=PANEL, outline=PANEL_BORDER,
                 tags=("acc_dim_poly",))
        items["status_dot"] = canvas.create_oval(cx - 92, pill_y - 4, cx - 84, pill_y + 4, outline="", tags=("acc_fill",))
        items["status"] = canvas.create_text(cx + 6, pill_y, text=MODE_LABELS.get(self._mode, ""), fill=TEXT,
                                             font=("Bahnschrift SemiLight", 10))

        # Captions.
        caption_top = pill_y + 34
        items["heard"] = canvas.create_text(cx, caption_top, text="", fill="#ffcf85", font=FONT_HEARD, width=width * 0.6)
        items["reply"] = canvas.create_text(cx, caption_top + 26, text="", fill=TEXT, font=FONT_CAPTION,
                                            width=min(width * 0.6, 1100), anchor="n", justify="center")

        # Input bar and buttons.
        bar_y = height * 0.885
        bar_w = min(620, width * 0.46)
        bar_x1, bar_x2 = cx - bar_w / 2 + 34, cx + bar_w / 2 + 34
        items["input_border"] = _rounded(canvas, bar_x1, bar_y - 24, bar_x2, bar_y + 24, 24, fill=PANEL,
                                         outline=PANEL_BORDER, width=2)
        self._command_entry.place(x=bar_x1 + 22, y=bar_y, width=bar_w - 130, height=30, anchor="w")
        self._make_button("send", bar_x2 - 52, bar_y, 78, 34)
        self._make_button("talk", bar_x1 - 40, bar_y, 56, 56, circle=True)
        self._make_button("stop", bar_x2 + 54, bar_y, 72, 34)
        row_y = height * 0.948
        self._make_button("voice", cx - 150, row_y, 112, 28)
        self._make_button("wake", cx, row_y, 124, 28)
        self._make_button("history", cx + 150, row_y, 112, 28)
        canvas.create_text(inset + 6, height - 14, anchor="w", fill=MUTED, font=FONT_MONO,
                           text="ENTER SEND  ·  CTRL+SPACE TALK  ·  ESC STOP  ·  CTRL+H HISTORY  ·  F11 FULLSCREEN  ·  CTRL+Q QUIT")
        canvas.create_text(width - inset - 6, height - 14, anchor="e", fill=MUTED, font=FONT_MONO,
                           text="LOCAL  ·  PRIVATE  ·  OFFLINE-FIRST  ·  v0.2")

        # Boot log (bottom left) and toast (top center).
        items["boot"] = canvas.create_text(inset + 22, height * 0.80, text="", fill=MUTED, font=FONT_MONO,
                                           anchor="sw", justify="left")
        items["toast_box"] = _rounded(canvas, cx - 280, -60, cx + 280, -14, 14, fill=PANEL, outline=PANEL_BORDER,
                                      width=2, tags=("acc_dim_poly", "toast"))
        items["toast_text"] = canvas.create_text(cx, -37, text="", fill=TEXT, font=("Segoe UI Variable Text", 11),
                                                 width=560, tags=("toast",))

        self._items = items
        self._last_coords: dict[int, tuple[int, ...]] = {}
        self._last_options: dict[tuple[int, str], Any] = {}
        self._colors_dirty = True
        self._set_text("status", MODE_LABELS.get(self._mode, self._mode.upper()))
        self._set_text("model", self._model_status)
        self._set_text("reminder", self._next_reminder)
        self._set_text("reply", self._target_text[: int(self._shown)])
        self._set_text("heard", f"YOU  ›  {self._heard}" if self._heard else "")
        self._update_telemetry()
        for key in self._buttons:
            self._paint_button(key)
        self._draw_frame(time.monotonic(), 0.0)

    def _build_panels(self, canvas: tk.Canvas, items: dict[str, Any], width: float, height: float, inset: float) -> None:
        panel_w, top, bottom = 250, height * 0.20, height * 0.62
        left_x, right_x = inset + 22, width - inset - 22 - panel_w
        for x in (left_x, right_x):
            _rounded(canvas, x, top, x + panel_w, bottom, 14, fill=PANEL, outline=PANEL_BORDER)
            canvas.create_line(x + 18, top + 1, x + 70, top + 1, width=2, tags=("acc_line",))
        # Left: system telemetry.
        canvas.create_text(left_x + 18, top + 24, text="SYSTEM TELEMETRY", fill=TEXT_DIM, font=FONT_LABEL, anchor="w")
        rows = (("cpu", "CPU", top + 62), ("ram", "MEMORY", top + 108))
        for key, label, y in rows:
            canvas.create_text(left_x + 18, y, text=label, fill=MUTED, font=FONT_MONO, anchor="w")
            items[f"{key}_value"] = canvas.create_text(left_x + panel_w - 18, y, text="--", fill=TEXT, font=FONT_VALUE, anchor="e")
            canvas.create_rectangle(left_x + 18, y + 14, left_x + panel_w - 18, y + 18, fill="#0e1d2c", outline="")
            items[f"{key}_bar"] = canvas.create_rectangle(left_x + 18, y + 14, left_x + 19, y + 18, outline="", tags=("acc_fill",))
            items[f"{key}_x"] = (left_x + 18, left_x + panel_w - 18, y)
        graph_top, graph_bottom = top + 150, bottom - 70
        canvas.create_text(left_x + 18, graph_top - 12, text="CPU · 40 S", fill=MUTED, font=FONT_MONO, anchor="w")
        canvas.create_rectangle(left_x + 18, graph_top, left_x + panel_w - 18, graph_bottom, outline="#0e1d2c")
        items["graph"] = canvas.create_line(0, 0, 0, 0, 0, 0, width=2, smooth=True, tags=("acc_line",))
        items["graph_box"] = (left_x + 20, graph_top + 2, left_x + panel_w - 20, graph_bottom - 2)
        canvas.create_text(left_x + 18, bottom - 42, text="LOCAL MODEL", fill=MUTED, font=FONT_MONO, anchor="w")
        items["model"] = canvas.create_text(left_x + 18, bottom - 22, text="", fill=TEXT, font=("Bahnschrift SemiLight", 10), anchor="w")
        # Right: session.
        canvas.create_text(right_x + 18, top + 24, text="SESSION", fill=TEXT_DIM, font=FONT_LABEL, anchor="w")
        fields = (("mode_label", "STATE"), ("voice_label", "VOICE OUTPUT"), ("wake_label", "WAKE WORD"),
                  ("reminder", "NEXT REMINDER"), ("uptime", "UPTIME"))
        for index, (key, label) in enumerate(fields):
            y = top + 62 + index * 50
            canvas.create_text(right_x + 18, y, text=label, fill=MUTED, font=FONT_MONO, anchor="w")
            items[key] = canvas.create_text(right_x + 18, y + 20, text="--", fill=TEXT, font=("Bahnschrift SemiLight", 10), anchor="w")

    def _make_button(self, key: str, x: float, y: float, w: float, h: float, *, circle: bool = False) -> None:
        canvas = self.canvas
        tag = f"btn_{key}"
        if circle:
            shape = canvas.create_oval(x - w / 2, y - h / 2, x + w / 2, y + h / 2, width=2, tags=(tag,))
            # Microphone glyph.
            glyph = [
                _rounded(canvas, x - 6, y - 15, x + 6, y + 3, 6, outline="", tags=(tag, f"{tag}_glyph")),
                canvas.create_arc(x - 11, y - 10, x + 11, y + 9, start=200, extent=140, style=tk.ARC, width=2,
                                  tags=(tag, f"{tag}_glyph_arc")),
                canvas.create_line(x, y + 9, x, y + 15, width=2, tags=(tag, f"{tag}_glyph_line")),
            ]
            label = None
            _ = glyph
        else:
            shape = _rounded(canvas, x - w / 2, y - h / 2, x + w / 2, y + h / 2, h / 2, width=1, tags=(tag,))
            label = canvas.create_text(x, y, text=self._buttons[key].label, font=FONT_BUTTON, tags=(tag,))
        self._items[f"{tag}_shape"] = shape
        self._items[f"{tag}_label"] = label
        canvas.tag_bind(tag, "<Enter>", lambda _event: self._hover(key, True))
        canvas.tag_bind(tag, "<Leave>", lambda _event: self._hover(key, False))
        canvas.tag_bind(tag, "<ButtonRelease-1>", lambda _event: self._click(key))

    def _hover(self, key: str, inside: bool) -> None:
        spec = self._buttons[key]
        self.canvas.configure(cursor="hand2" if inside and spec.enabled else "")
        self._paint_button(key, hover=inside)

    def _click(self, key: str) -> None:
        spec = self._buttons[key]
        if spec.enabled:
            spec.command()

    def _paint_button(self, key: str, *, hover: bool = False) -> None:
        canvas = self.canvas
        tag = f"btn_{key}"
        shape = self._items.get(f"{tag}_shape")
        if shape is None:
            return
        spec = self._buttons[key]
        accent = _hex(self._accent)
        if spec.danger:
            accent = DANGER
        if not spec.enabled:
            fill, outline, text = PANEL, "#1c3043", MUTED
        elif spec.active:
            fill, outline, text = _hex(_mix(_rgb(PANEL), _rgb(accent), 0.30)), accent, TEXT
        elif hover:
            fill, outline, text = _hex(_mix(_rgb(PANEL), _rgb(accent), 0.22)), accent, TEXT
        else:
            fill, outline, text = _hex(_mix(_rgb(PANEL), _rgb(accent), 0.08)), _hex(_mix(_rgb(PANEL_BORDER), _rgb(accent), 0.6)), TEXT
        if key == "talk" and spec.enabled and not hover:
            outline = accent
        canvas.itemconfigure(shape, fill=fill, outline=outline)
        label = self._items.get(f"{tag}_label")
        if label is not None:
            suffix = ""
            if key in {"voice", "wake"}:
                suffix = "  ON" if spec.active else "  OFF"
            canvas.itemconfigure(label, fill=text, text=spec.label + suffix)
        else:
            canvas.itemconfigure(f"{tag}_glyph", fill=text)
            canvas.itemconfigure(f"{tag}_glyph_line", fill=text)
            canvas.itemconfigure(f"{tag}_glyph_arc", outline=text)

    def _update_telemetry(self) -> None:
        canvas = self.canvas
        for key, value in (("cpu", self._cpu), ("ram", self._ram)):
            x_info = self._items.get(f"{key}_x")
            if x_info is None:
                continue
            x1, x2, y = x_info
            canvas.coords(self._items[f"{key}_bar"], x1, y + 14, x1 + max(2.0, (x2 - x1) * value / 100), y + 18)
            self._set_text(f"{key}_value", f"{value:.0f}%")
        box = self._items.get("graph_box")
        if box is not None:
            gx1, gy1, gx2, gy2 = box
            values = list(self._cpu_history)
            points: list[float] = []
            for index, value in enumerate(values):
                points.extend((gx1 + (gx2 - gx1) * index / (len(values) - 1), gy2 - (gy2 - gy1) * value / 100))
            canvas.coords(self._items["graph"], *points)
        runtime = self._runtime
        self._set_text("mode_label", MODE_LABELS.get(self._mode, self._mode.upper()))
        if runtime is not None:
            self._set_text("voice_label", "ENABLED" if runtime.speaker.enabled else "MUTED")
            self._set_text("wake_label", "LISTENING FOR 'HEY JARVIS'" if runtime.wake_word is not None else "OFF")
        uptime = int(time.monotonic() - self._start_time)
        self._set_text("uptime", f"{uptime // 3600:02d}:{uptime % 3600 // 60:02d}:{uptime % 60:02d}")
        self._set_text("clock", time.strftime("%H:%M"))
        self._set_text("date", time.strftime("%A  ·  %d %B %Y").upper())

    # -- animation --------------------------------------------------------

    def _tick(self) -> None:
        if self._closing:
            return
        now = time.monotonic()
        if self.state() == "iconic":
            self._last_frame = now
            self.after(500, self._tick)
            return
        if not self._items:
            self._build_scene()
        dt = min(0.2, now - self._last_frame)
        self._last_frame = now
        self._draw_frame(now, dt)
        active = self._mode in ACTIVE_MODES or self._history_anim is not None or now < self._toast_until + 0.4
        interval = 33 if active else 110
        if not self._focused and not active:
            interval = 160
        if self._reduced_motion and not active:
            interval = 250
        self.after(interval, self._tick)

    def _move(self, item: int, *coords: float) -> None:
        """Move an item only if it actually changed: every change makes Tk repaint."""

        key = tuple(int(round(value)) for value in coords)
        if self._last_coords.get(item) != key:
            self._last_coords[item] = key
            self.canvas.coords(item, *key)

    def _cfg(self, item: int, **options: Any) -> None:
        changed = {}
        for name, value in options.items():
            if isinstance(value, float):
                value = int(round(value))
            if self._last_options.get((item, name)) != value:
                self._last_options[(item, name)] = value
                changed[name] = value
        if changed:
            self.canvas.itemconfigure(item, **changed)

    def _apply_colors(self) -> None:
        canvas = self.canvas
        accent = self._accent
        bg = _rgb(BACKGROUND)
        canvas.itemconfigure("acc_fill", fill=_hex(accent))
        canvas.itemconfigure("acc_line", fill=_hex(accent))
        canvas.itemconfigure("acc_outline", outline=_hex(accent))
        canvas.itemconfigure("acc_dim_outline", outline=_hex(_mix(bg, accent, 0.45)))
        canvas.itemconfigure("acc_dim_line", fill=_hex(_mix(_rgb(PANEL_BORDER), accent, 0.45)))
        canvas.itemconfigure("acc_dim_poly", outline=_hex(_mix(_rgb(PANEL_BORDER), accent, 0.35)))
        canvas.itemconfigure("core_fill", fill=_hex(_mix(bg, accent, 0.10)))
        for index, strength in enumerate((0.06, 0.10, 0.16, 0.24)):
            canvas.itemconfigure(f"halo{index}", outline=_hex(_mix(bg, accent, strength)))
        for key in self._buttons:
            self._paint_button(key)

    def _draw_frame(self, now: float, dt: float) -> None:
        items = self._items
        if not items:
            return
        canvas = self.canvas
        width, height = self._scene_size
        cx, cy = self._center
        radius = self._radius
        elapsed = now - self._start_time
        mode = self._mode
        active = mode in ACTIVE_MODES
        motion = 0.0 if self._reduced_motion else 1.0

        # Eased color transition between modes.
        if self._accent != self._accent_target:
            blend = 1.0 if self._reduced_motion else min(1.0, dt * 6.0)
            self._accent = _mix(self._accent, self._accent_target, blend)
            if max(abs(a - b) for a, b in zip(self._accent, self._accent_target)) < 1.5:
                self._accent = self._accent_target
            self._colors_dirty = True
        if self._colors_dirty:
            self._colors_dirty = False
            self._apply_colors()

        # Boot sequence.
        boot = 1.0 if self._boot_done else min(1.0, elapsed / _BOOT_SECONDS)
        if not self._boot_done:
            title = "K A R E N"
            self._cfg(items["title"], text=title[: int(len(title) * _ease_out(boot * 1.4))])
            scripted = [line for index, line in enumerate(_BOOT_LINES) if boot > index / len(_BOOT_LINES)]
            lines = (scripted + self._boot_lines)[-7:]
            self._cfg(items["boot"], text="\n".join(f"›  {line}" for line in lines))
            if boot >= 1.0:
                self._boot_done = True
                self._cfg(items["boot"], text="")
                if self._pending_ready is not None:
                    runtime, self._pending_ready = self._pending_ready, None
                    self._on_ready(runtime)
        elif canvas.itemcget(items["title"], "text") != "K A R E N":
            self._cfg(items["title"], text="K A R E N")
            self._cfg(items["boot"], text="")
        reveal = _ease_out(boot)

        # Audio level: real microphone while listening, synthetic voice envelope while speaking.
        target = 0.0
        runtime = self._runtime
        if mode == "listening" and runtime is not None and runtime._recognizer is not None:
            target = min(1.0, runtime._recognizer.level * 1.6)
        elif mode == "speaking":
            target = 0.35 + 0.4 * abs(math.sin(elapsed * 7.3) * math.sin(elapsed * 2.9 + 1.2)) + random.random() * 0.12
        elif mode in {"thinking", "searching", "booting"}:
            target = 0.18
        self._level += (target - self._level) * min(1.0, dt * 14)
        level = self._level
        breathe = (math.sin(elapsed * 1.6) + 1) / 2

        # Halo.
        # The halo is the most expensive item to repaint, so it only moves with real sound.
        halo_level = round(level * 10) / 10
        for index, item in enumerate(items["halo"]):
            r = radius * (1.42 + index * -0.07) * reveal + halo_level * 12
            self._move(item, cx - r, cy - r, cx + r, cy + r)

        # Rotating segmented ring; segments appear one by one while booting.
        spin = elapsed * (26 if active else 7) * motion
        visible_segments = int(_SEGMENTS * reveal + 0.999)
        if active or dt == 0.0 or not self._boot_done or int(elapsed * 9) % 2 == 0:
            for index, item in enumerate(items["segments"]):
                if index < visible_segments:
                    self._cfg(item, start=(index * 360 / _SEGMENTS + spin) % 360, state="normal")
                else:
                    self._cfg(item, state="hidden")

        # Reactive bars around the ring.
        base_r = radius * 1.42
        if active or dt == 0.0 or level > 0.02:
            for index, item in enumerate(items["bars"]):
                angle = index * math.tau / _BARS - math.pi / 2
                if mode in {"thinking", "searching"}:
                    wave = (math.sin(elapsed * 6 - index * 0.55) + 1) / 2
                    length = 3 + wave * 16 * level * 4
                else:
                    jitter = (math.sin(elapsed * 9 + index * 1.7) + math.sin(elapsed * 5.3 + index * 0.9)) / 4 + 0.5
                    length = 2 + jitter * level * 48
                length *= reveal
                x1, y1 = cx + math.cos(angle) * base_r, cy + math.sin(angle) * base_r
                x2, y2 = cx + math.cos(angle) * (base_r + length), cy + math.sin(angle) * (base_r + length)
                self._move(item, x1, y1, x2, y2)

        # Counter-rotating arcs and scanner sweep.
        arc_r = radius * 0.92
        for (speed, phase, scale), item in zip(((48, 0, 1.0), (-31, 140, 0.84), (90, 260, 1.0)), items["arcs"]):
            r = arc_r * scale * reveal
            self._move(item, cx - r, cy - r, cx + r, cy + r)
            self._cfg(item, start=(phase + elapsed * speed * (1.6 if active else 0.6) * motion) % 360)
        # The scanner sweep only runs while Karen is working (it repaints the whole ring).
        working = mode in {"listening", "thinking", "searching", "booting"}
        self._cfg(items["scanner"], state="normal" if working else "hidden")
        if working:
            scan_r = radius * 1.34
            self._move(items["scanner"], cx - scan_r, cy - scan_r, cx + scan_r, cy + scan_r)
            self._cfg(items["scanner"], start=(-elapsed * (140 if mode == "searching" else 50) * motion) % 360,
                      extent=70 if mode == "searching" else 26)

        # Inner core.
        inner = radius * 0.56 * reveal
        self._move(items["inner_ring"], cx - inner, cy - inner, cx + inner, cy + inner)
        fill_r = inner - 4
        self._move(items["inner_fill"], cx - fill_r, cy - fill_r, cx + fill_r, cy + fill_r)
        core = radius * (0.13 + 0.03 * breathe + 0.10 * level) * reveal
        self._move(items["core"], cx - core, cy - core, cx + core, cy + core)
        hot = core * 0.45
        self._move(items["core_hot"], cx - hot, cy - hot, cx + hot, cy + hot)
        for index, item in enumerate(items["orbits"]):
            angle = elapsed * (0.9 + index * 0.5) * (2.2 if active else 1.0) * motion + index * 2.1
            r = radius * (0.68 + index * 0.13) * reveal
            x, y = cx + math.cos(angle) * r, cy + math.sin(angle) * r
            size = 3 if index == 0 else 2
            self._move(item, x - size, y - size, x + size, y + size)

        # Background dust drifting upward (slowly refreshed when idle).
        self._dust_dt += dt
        if motion and (active or self._dust_dt >= 0.3):
            step, self._dust_dt = self._dust_dt, 0.0
            for particle, item in zip(self._dust, items["dust"]):
                particle.y -= particle.speed * step
                particle.x += particle.drift * step
                if particle.y < -0.02:
                    particle.y, particle.x = 1.02, random.random()
                x, y = particle.x * width, particle.y * height
                self._move(item, x, y, x + particle.size, y + particle.size)
        # A full-width scan line repaints a strip across the whole screen, so it
        # only sweeps while Karen is working.
        self._cfg(items["scan"], state="normal" if active and motion else "hidden")
        if active and motion:
            scan_y = (elapsed * 90) % height
            self._move(items["scan"], 0, scan_y, width, scan_y)

        # Blinking corner markers and status dot.
        on = int(elapsed * 1.6) % 2 == 0
        for item in items["blink"]:
            self._cfg(item, state="normal" if on else "hidden")
        self._cfg(items["status_dot"], state="normal" if (on or not active) else "hidden")

        # Subtitle reveal (typewriter that catches up smoothly with streaming text).
        target_len = len(self._target_text)
        if self._shown < target_len:
            remaining = target_len - self._shown
            self._shown = min(float(target_len), self._shown + max(1.5, remaining * 0.18, dt * 70))
            self._cfg(items["reply"], text=self._target_text[: int(self._shown)])
        elif self._shown > target_len:
            self._shown = float(target_len)
            self._cfg(items["reply"], text=self._target_text)

        # Toast slide in/out.
        if self._toast_until:
            t_in = _ease_out((now - self._toast_start) / 0.35)
            t_out = _ease_out((now - self._toast_until) / 0.35) if now > self._toast_until else 0.0
            # Slide from above the window to just under the title rule.
            final_top = height * 0.075 + 42
            offset = -70 + (final_top + 70) * (t_in - t_out)
            current = canvas.coords(items["toast_box"])
            if current:
                top = min(current[1::2])
                canvas.move("toast", 0, (offset - 60) - top + 60)
            if now > self._toast_until + 0.4:
                self._toast_until = 0.0

        self._animate_history(now)


def launch(*, speak: bool | None = None) -> None:
    """Launch the animated HUD."""

    OpeningPage(speak=speak).mainloop()


if __name__ == "__main__":
    launch()
