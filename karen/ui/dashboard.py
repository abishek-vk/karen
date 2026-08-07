"""First interactive Karen dashboard after the animated opening page."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import scrolledtext

from karen.core.application_service import ApplicationService

from .opening import BACKGROUND, BRIGHT_CYAN, CYAN, GREEN, MUTED, PANEL


@dataclass(frozen=True, slots=True)
class _PendingResponse:
    user_text: str
    response_text: str
    source: str
    success: bool


class DashboardPage(tk.Toplevel):
    """Lightweight conversation dashboard backed by ApplicationService."""

    def __init__(self, owner: tk.Tk, service: ApplicationService) -> None:
        super().__init__(owner)
        self.service = service
        self._responses: queue.Queue[_PendingResponse] = queue.Queue()
        self._busy = False
        self.title("Karen AI // Command Center")
        self.geometry("1180x760")
        self.minsize(900, 600)
        self.configure(bg=BACKGROUND)
        self.protocol("WM_DELETE_WINDOW", owner.destroy)
        self.bind("<Escape>", lambda _event: owner.destroy())
        self._build()
        self._poll_responses()
        self.after(250, self._check_local_ai)

    def _build(self) -> None:
        header = tk.Frame(self, bg=BACKGROUND)
        header.pack(fill="x", padx=28, pady=(22, 12))

        tk.Label(
            header,
            text="K A R E N",
            bg=BACKGROUND,
            fg=BRIGHT_CYAN,
            font=("Consolas", 22, "bold"),
        ).pack(side="left")
        self._status = tk.Label(
            header,
            text="● LOCAL CORE READY",
            bg=BACKGROUND,
            fg=GREEN,
            font=("Consolas", 10, "bold"),
        )
        self._status.pack(side="right", pady=8)

        tk.Label(
            self,
            text="COMMAND CENTER  //  PRIVATE LOCAL SESSION",
            bg=BACKGROUND,
            fg=MUTED,
            font=("Consolas", 9),
        ).pack(anchor="w", padx=30)

        self._conversation = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            bg="#04111d",
            fg="#d5f8ff",
            insertbackground=CYAN,
            relief="flat",
            bd=0,
            padx=20,
            pady=18,
            font=("Consolas", 11),
            state=tk.DISABLED,
        )
        self._conversation.pack(fill="both", expand=True, padx=28, pady=(16, 14))
        self._conversation.tag_configure("karen", foreground=BRIGHT_CYAN)
        self._conversation.tag_configure("user", foreground="#d8e9ed")
        self._conversation.tag_configure("system", foreground=MUTED)
        self._append("KAREN", "Systems initialized. Your local session is ready.", "karen")
        self._append("SYSTEM", "Fast commands bypass the LLM. Press Escape to exit.", "system")

        command_bar = tk.Frame(self, bg=PANEL)
        command_bar.pack(fill="x", padx=28, pady=(0, 24))
        self._entry = tk.Entry(
            command_bar,
            bg="#0a1e2c",
            fg=BRIGHT_CYAN,
            insertbackground=CYAN,
            relief="flat",
            font=("Consolas", 11),
        )
        self._entry.pack(side="left", fill="x", expand=True, padx=(14, 8), pady=14, ipady=8)
        self._entry.bind("<Return>", lambda _event: self._send())
        self._send_button = tk.Button(
            command_bar,
            text="SEND",
            command=self._send,
            bg="#0b4052",
            activebackground="#16657a",
            fg=BRIGHT_CYAN,
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=24,
            pady=10,
            cursor="hand2",
            font=("Consolas", 10, "bold"),
        )
        self._send_button.pack(side="right", padx=(0, 14), pady=14)
        self._entry.focus_set()

    def _append(self, speaker: str, message: str, tag: str) -> None:
        self._conversation.configure(state=tk.NORMAL)
        self._conversation.insert(tk.END, f"{speaker}\n", tag)
        self._conversation.insert(tk.END, f"{message}\n\n")
        self._conversation.see(tk.END)
        self._conversation.configure(state=tk.DISABLED)

    def _send(self) -> None:
        if self._busy:
            return
        text = self._entry.get().strip()
        if not text:
            return
        self._entry.delete(0, tk.END)
        self._append("YOU", text, "user")
        self._busy = True
        self._send_button.configure(state=tk.DISABLED, text="...")
        self._status.configure(text="● PROCESSING LOCAL REQUEST", fg=CYAN)
        threading.Thread(target=self._process, args=(text,), daemon=True).start()

    def _process(self, text: str) -> None:
        response = self.service.handle_text(text)
        self._responses.put(_PendingResponse(text, response.message, response.source, response.success))

    def _poll_responses(self) -> None:
        try:
            pending = self._responses.get_nowait()
        except queue.Empty:
            self.after(60, self._poll_responses)
            return
        self._append("KAREN", pending.response_text, "karen")
        self._status.configure(
            text=f"● {pending.source.upper().replace('_', ' ')}",
            fg=GREEN if pending.success else CYAN,
        )
        self._busy = False
        self._send_button.configure(state=tk.NORMAL, text="SEND")
        self._entry.focus_set()
        self.after(60, self._poll_responses)

    def _check_local_ai(self) -> None:
        """Refresh the status without blocking the Tk event loop."""

        def check() -> None:
            available = self.service.llm.ensure_available()
            model_ready = available and self.service.llm.has_model()
            self.after(0, lambda: self._set_ai_status(available, model_ready))

        threading.Thread(target=check, daemon=True).start()

    def _set_ai_status(self, available: bool, model_ready: bool) -> None:
        if not self.winfo_exists() or self._busy:
            return
        if available and model_ready:
            self._status.configure(text="● QWEN3:4B ONLINE // LOCAL", fg=GREEN)
        elif available:
            self._status.configure(text="● OLLAMA ONLINE // INSTALL QWEN3:4B", fg=CYAN)
        else:
            self._status.configure(text="● FAST PATH READY // STARTING OLLAMA FAILED", fg=CYAN)
