"""3D (WebGL) interface: Karen's services behind a tiny local web bridge.

The window is a chromeless Microsoft Edge (or Chrome) "app" window that renders
a Three.js scene on the GPU, so the 3D animation costs almost no CPU. Python
talks to it through a standard-library HTTP server bound to 127.0.0.1:

* ``GET  /events``        Server-Sent Events stream (state, streamed replies, levels)
* ``POST /api/<action>``  commands from the page (send, talk, stop, toggle, quit)

Every API call and the event stream require a random per-launch token, so
other web pages or local programs cannot drive the assistant.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import queue
import secrets
import shutil
import subprocess
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from karen.core.application_service import ServiceResponse
from karen.voice.contracts import VoiceError

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency
    psutil = None

LOGGER = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent / "web"
_MAX_BODY = 16_384


class EventHub:
    """Fan out events to every connected page (normally exactly one)."""

    def __init__(self) -> None:
        self._clients: list[queue.Queue[tuple[str, Any]]] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue[tuple[str, Any]]:
        client: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=2000)
        with self._lock:
            self._clients.append(client)
        return client

    def unsubscribe(self, client: queue.Queue[tuple[str, Any]]) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)

    def publish(self, event: str, data: Any = None) -> None:
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.put_nowait((event, data))
            except queue.Full:  # A stalled page must never block the assistant.
                pass


class KarenController:
    """UI-independent session logic shared by the web interface."""

    def __init__(self, *, speak: bool | None = None) -> None:
        self.hub = EventHub()
        self.speak_override = speak
        self.runtime: Any | None = None
        self.mode = "booting"
        self.model_status = "CHECKING LOCAL MODEL"
        self.boot_lines: list[str] = []
        self.history: list[dict[str, str]] = []
        self.next_reminder = "NONE"
        self.reduced_motion = False
        self.fullscreen = True
        self.quit_event = threading.Event()
        self._busy = False
        self._listening = False
        self._lock = threading.Lock()
        self._stopped = threading.Event()

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        threading.Thread(target=self._boot, name="karen-boot", daemon=True).start()
        threading.Thread(target=self._telemetry_loop, name="karen-telemetry", daemon=True).start()

    def _boot_line(self, line: str) -> None:
        self.boot_lines.append(line)
        self.hub.publish("boot", {"line": line})

    def _boot(self) -> None:
        try:
            self._boot_line("LOADING CONFIGURATION")
            from karen.app import KarenRuntime

            runtime = KarenRuntime(on_speech_state=self._on_speech_state)
            self.reduced_motion = runtime.settings.ui_reduced_motion
            self.fullscreen = runtime.settings.ui_fullscreen
            self._boot_line("MOUNTING MEMORY BANKS")
            if self.speak_override is not None:
                runtime.speaker.enabled = self.speak_override and runtime.speaker.available
            runtime.start_background(self._on_reminder)
            self._boot_line(f"LOCAL MODEL  {runtime.settings.ollama_model.upper()}")
            if runtime.preferences.voice_enabled or runtime.preferences.wake_word_enabled:
                runtime.recognizer.preload()
                self._boot_line("CALIBRATING VOICE SUBSYSTEM")
            self.runtime = runtime
            self._boot_line("ALL SYSTEMS NOMINAL")
            name = runtime.memory.user_name()
            hour = time.localtime().tm_hour
            part = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
            greeting = f"Good {part}{', ' + name if name else ''}. All systems are online."
            self._set_mode("ready")
            self._log("KAREN", greeting)
            self.hub.publish("ready", {"greeting": greeting, **self._toggles()})
            runtime.speaker.say(greeting)
            if runtime.preferences.wake_word_enabled:
                self._enable_wake_word()
            self._refresh_reminder()
        except Exception as error:  # pragma: no cover - environment dependent
            LOGGER.exception("Startup failed")
            self._set_mode("error")
            self.hub.publish("reply", {"text": f"Startup failed: {error}", "source": "error", "success": False})

    def shutdown(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        runtime = self.runtime
        if runtime is not None:
            try:
                runtime.shutdown()
            except Exception:  # pragma: no cover - best effort on exit
                LOGGER.exception("Shutdown failed")

    # -- state ------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "model": self.model_status,
            "boot": list(self.boot_lines),
            "ready": self.runtime is not None,
            "history": self.history[-60:],
            "reminder": self.next_reminder,
            "reduced_motion": self.reduced_motion,
            **self._toggles(),
        }

    def _toggles(self) -> dict[str, bool]:
        runtime = self.runtime
        return {
            "voice": bool(runtime and runtime.speaker.enabled),
            "voice_available": bool(runtime and runtime.speaker.available),
            "wake": bool(runtime and runtime.wake_word is not None),
        }

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.hub.publish("mode", {"mode": mode})

    def _log(self, speaker: str, text: str) -> None:
        entry = {"speaker": speaker, "text": text, "time": time.strftime("%H:%M")}
        self.history.append(entry)
        del self.history[:-200]
        self.hub.publish("log", entry)

    def _toast(self, text: str, seconds: float = 4.0) -> None:
        self.hub.publish("toast", {"text": text, "seconds": seconds})

    def _on_speech_state(self, speaking: bool) -> None:
        self.hub.publish("speaking", {"speaking": speaking})
        if not self._busy:
            self._set_mode("speaking" if speaking else "ready")

    def _on_reminder(self, message: str) -> None:
        self._log("KAREN", message)
        self._toast(message, 9)
        runtime = self.runtime
        if runtime is not None:
            runtime.speaker.say(message)
        self._refresh_reminder()

    def _refresh_reminder(self) -> None:
        runtime = self.runtime
        if runtime is None:
            return
        try:
            pending = runtime.reminders.pending()
        except Exception:  # pragma: no cover - database dependent
            pending = []
        if pending:
            text, due = pending[0]
            self.next_reminder = f"{due.strftime('%a %H:%M')} · {text[:28]}"
        else:
            self.next_reminder = "NONE"
        self.hub.publish("reminder", {"next": self.next_reminder})

    # -- actions (called from HTTP threads) ------------------------------

    def send(self, text: str) -> dict[str, Any]:
        text = " ".join(text.split())[:2000]
        runtime = self.runtime
        if not text or runtime is None:
            return {"ok": False}
        with self._lock:
            if self._busy:
                return {"ok": False, "reason": "busy"}
            self._busy = True
        runtime.speaker.stop()  # Barge-in: a new request interrupts speech.
        if runtime.wake_word is not None:
            runtime.wake_word.pause()
        self.hub.publish("heard", {"text": text})
        self._log("YOU", text)
        self._set_mode("thinking")
        threading.Thread(target=self._process, args=(runtime, text), name="karen-request", daemon=True).start()
        return {"ok": True}

    def _process(self, runtime: Any, text: str) -> None:
        def on_chunk(chunk: str) -> None:
            self.hub.publish("chunk", {"text": chunk})
            runtime.speaker.feed(chunk)

        def on_status(state: str) -> None:
            self._set_mode("searching" if state == "searching" else "thinking")

        try:
            response = runtime.service.handle_text(text, on_chunk=on_chunk, on_status=on_status)
        except Exception as error:  # pragma: no cover - defensive
            response = ServiceResponse(f"Something went wrong: {error}", "error", False)
            runtime.speaker.say(response.message)
        runtime.speaker.flush()
        self.hub.publish("reply", {"text": response.message, "source": response.source, "success": response.success})
        self._log("KAREN", response.message)
        self._finish(error=response.source == "error")

    def _finish(self, *, error: bool = False) -> None:
        runtime = self.runtime
        with self._lock:
            self._busy = False
            self._listening = False
        if runtime is not None and runtime.wake_word is not None:
            runtime.wake_word.resume()
        if error:
            self._set_mode("error")

            def recover() -> None:
                if self.mode == "error" and not self._busy:
                    self._set_mode("ready")

            threading.Timer(2.6, recover).start()
        elif runtime is not None and runtime.speaker.is_speaking:
            self._set_mode("speaking")
        else:
            self._set_mode("ready")
        self._refresh_reminder()

    def talk(self) -> dict[str, Any]:
        runtime = self.runtime
        if runtime is None:
            return {"ok": False}
        with self._lock:
            if self._busy:
                return {"ok": False, "reason": "busy"}
            self._busy = True
            self._listening = True
        runtime.speaker.stop()
        if runtime.wake_word is not None:
            runtime.wake_word.pause()
        self._set_mode("listening")
        threading.Thread(target=self._listen, args=(runtime,), name="karen-listen", daemon=True).start()
        threading.Thread(target=self._level_loop, args=(runtime,), name="karen-level", daemon=True).start()
        return {"ok": True}

    def _level_loop(self, runtime: Any) -> None:
        while self._listening and not self._stopped.is_set():
            recognizer = runtime._recognizer
            level = recognizer.level if recognizer is not None else 0.0
            self.hub.publish("level", {"value": round(level, 3)})
            time.sleep(0.05)
        self.hub.publish("level", {"value": 0.0})

    def _listen(self, runtime: Any) -> None:
        try:
            transcript = runtime.recognizer.transcribe_once()
        except VoiceError as error:
            self.hub.publish("reply", {"text": str(error), "source": "voice", "success": False})
            self._finish(error=True)
            return
        except Exception as error:  # pragma: no cover - device dependent
            self.hub.publish("reply", {"text": f"Microphone problem: {error}", "source": "voice", "success": False})
            self._finish(error=True)
            return
        with self._lock:
            self._busy = False
            self._listening = False
        self.send(transcript)

    def stop(self) -> dict[str, Any]:
        runtime = self.runtime
        if runtime is None:
            return {"ok": False}
        runtime.speaker.stop()
        if self._listening:
            runtime.recognizer.cancel()
        if not self._busy:
            self._set_mode("ready")
        return {"ok": True}

    def toggle(self, name: str) -> dict[str, Any]:
        runtime = self.runtime
        if runtime is None:
            return {"ok": False}
        if name == "voice":
            if not runtime.speaker.available:
                self._toast("No text-to-speech engine is available. Install comtypes: pip install comtypes")
                return {"ok": False}
            runtime.speaker.enabled = not runtime.speaker.enabled
            if not runtime.speaker.enabled:
                runtime.speaker.stop()
            runtime.preferences.speak_responses = runtime.speaker.enabled
            self._toast("Voice replies on." if runtime.speaker.enabled else "Voice replies muted.")
        elif name == "wake":
            if runtime.wake_word is None:
                self._enable_wake_word()
                self._toast("Wake word on. Say: Hey Jarvis.")
            else:
                runtime.stop_wake_word()
                self._toast("Wake word off.")
            runtime.preferences.wake_word_enabled = runtime.wake_word is not None
        else:
            return {"ok": False}
        runtime.save_preferences()
        self.hub.publish("toggles", self._toggles())
        return {"ok": True, **self._toggles()}

    def _enable_wake_word(self) -> None:
        runtime = self.runtime
        if runtime is None:
            return
        runtime.recognizer.preload()

        def on_detect() -> None:
            runtime.speaker.stop()
            if self._busy:
                if runtime.wake_word is not None:
                    runtime.wake_word.resume()
            else:
                self.talk()

        def on_error(message: str) -> None:
            runtime.stop_wake_word()
            runtime.preferences.wake_word_enabled = False
            self.hub.publish("toggles", self._toggles())
            self._toast(message, 8)

        runtime.start_wake_word(on_detect, on_error=on_error)
        self.hub.publish("toggles", self._toggles())

    # -- background data --------------------------------------------------

    def _telemetry_loop(self) -> None:
        next_model_check = 0.0
        while not self._stopped.wait(1.0):
            if psutil is not None:
                self.hub.publish("telemetry", {
                    "cpu": psutil.cpu_percent(interval=None),
                    "ram": psutil.virtual_memory().percent,
                })
            runtime = self.runtime
            if runtime is not None and time.monotonic() >= next_model_check:
                available = runtime.llm.is_available()
                ready = available and runtime.llm.use_installed_model() is not None
                self.model_status = (
                    f"{runtime.llm.model.upper()} · ONLINE" if ready
                    else "OLLAMA ONLINE · NO MODEL" if available
                    else "OFFLINE · COMMANDS + WEB"
                )
                self.hub.publish("model", {"status": self.model_status})
                next_model_check = time.monotonic() + (30 if ready else 10)


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        # The app window closing its keep-alive connections is normal, not an error.
        import sys

        if isinstance(sys.exc_info()[1], (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def _make_handler(controller: KarenController, token: str, port_holder: list[int]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Karen/0.2"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            LOGGER.debug("web: " + format, *args)

        def _host_ok(self) -> bool:
            host = self.headers.get("Host", "")
            return host in {f"127.0.0.1:{port_holder[0]}", f"localhost:{port_holder[0]}"}

        def _token_ok(self, query: dict[str, list[str]]) -> bool:
            supplied = self.headers.get("X-Karen-Token") or (query.get("t") or [""])[0]
            return secrets.compare_digest(supplied, token)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            if not self._host_ok():
                self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
                return
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/events":
                if not self._token_ok(query):
                    self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
                    return
                self._stream_events()
                return
            relative = "index.html" if parsed.path in {"", "/"} else parsed.path.lstrip("/")
            target = (WEB_DIR / relative).resolve()
            # Serve only real files inside the web folder (no path traversal).
            if WEB_DIR not in target.parents or not target.is_file():
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if target.suffix == ".js":
                content_type = "text/javascript"
            self._send(HTTPStatus.OK, target.read_bytes(), content_type + ("; charset=utf-8" if target.suffix in {".html", ".css", ".js"} else ""))

        def _stream_events(self) -> None:
            client = controller.hub.subscribe()
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                self._write_event("snapshot", controller.snapshot())
                while not controller.quit_event.is_set():
                    try:
                        event, data = client.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        continue
                    self._write_event(event, data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                controller.hub.unsubscribe(client)

        def _write_event(self, event: str, data: Any) -> None:
            payload = json.dumps(data if data is not None else {})
            self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            parsed = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length > _MAX_BODY:
                self.close_connection = True
                self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, b"too large", "text/plain")
                return
            # Always read the body, even for rejected requests: closing a socket with
            # unread data makes Windows reset the connection before the reply arrives.
            raw = self.rfile.read(length) if length > 0 else b""
            if not self._host_ok() or not self._token_ok(parse_qs(parsed.query)):
                self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
                return
            try:
                body = json.loads(raw or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = {}
            if not isinstance(body, dict):
                body = {}
            action = parsed.path.removeprefix("/api/")
            if action == "send":
                self._json(controller.send(str(body.get("text", ""))))
            elif action == "talk":
                self._json(controller.talk())
            elif action == "stop":
                self._json(controller.stop())
            elif action == "toggle":
                self._json(controller.toggle(str(body.get("name", ""))))
            elif action == "quit":
                self._json({"ok": True})
                controller.quit_event.set()
            else:
                self._send(HTTPStatus.NOT_FOUND, b"unknown action", "text/plain")

    return Handler


def _find_browser() -> str | None:
    candidates = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return shutil.which("msedge") or shutil.which("chrome")


def browser_args(browser: str, url: str, profile: Path, *, fullscreen: bool) -> list[str]:
    """Command line for a lean, chromeless app window with its own profile."""

    args = [
        browser, f"--app={url}", f"--user-data-dir={profile}", "--no-first-run",
        "--no-default-browser-check", "--disable-extensions", "--disable-sync",
        "--window-size=1440,900", "--autoplay-policy=no-user-gesture-required",
        # Keep the app window lean: no background services, one renderer.
        "--disable-background-networking", "--disable-component-update", "--disable-default-apps",
        "--no-pings", "--renderer-process-limit=1", "--disable-breakpad",
        "--disable-features=Translate,MediaRouter,OptimizationHints,msEdgeShopping,msEdgeCollections,"
        "msEdgeSidebarV2,EdgeShoppingAssistant,msImplicitSignin,msEdgeWorkspacesIntegration",
    ]
    if fullscreen:
        args.append("--start-fullscreen")
    return args


def available() -> bool:
    """The 3D interface needs a Chromium browser for its app window."""

    return _find_browser() is not None and (WEB_DIR / "index.html").is_file()


def launch(*, speak: bool | None = None) -> None:
    controller = KarenController(speak=speak)
    token = secrets.token_urlsafe(24)
    port_holder = [0]
    server = _QuietServer(("127.0.0.1", 0), _make_handler(controller, token, port_holder))
    port_holder[0] = server.server_address[1]
    threading.Thread(target=server.serve_forever, name="karen-web", daemon=True).start()
    controller.start()

    url = f"http://127.0.0.1:{port_holder[0]}/?t={token}"
    from karen.app import PROJECT_DIR

    profile = PROJECT_DIR / "data" / "ui_profile"
    browser = _find_browser()
    process: subprocess.Popen[bytes] | None = None
    if browser is not None:
        try:
            from karen.config import ConfigurationStore

            fullscreen = ConfigurationStore(PROJECT_DIR / "config").load_settings().ui_fullscreen
        except Exception:  # pragma: no cover - settings errors are reported by the runtime
            fullscreen = True
        args = browser_args(browser, url, profile, fullscreen=fullscreen)
        process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        webbrowser.open(url)

    try:
        while not controller.quit_event.is_set():
            if process is not None and process.poll() is not None:
                break  # The app window was closed.
            controller.quit_event.wait(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        controller.quit_event.set()
        if process is not None and process.poll() is None:
            process.terminate()
        controller.shutdown()
        server.shutdown()
