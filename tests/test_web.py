"""The 3D interface bridge must only obey its own page."""

from __future__ import annotations

import http.client
import json
import threading

import pytest

from karen.ui import web_app


@pytest.fixture()
def bridge():
    controller = web_app.KarenController(speak=False)
    port = [0]
    server = web_app._QuietServer(("127.0.0.1", 0), web_app._make_handler(controller, "secret-token", port))
    port[0] = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield port[0]
    controller.quit_event.set()
    server.shutdown()


def _request(port: int, method: str, path: str, *, token: str | None = None, host: str | None = None, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": host or f"127.0.0.1:{port}"}
    if token:
        headers["X-Karen-Token"] = token
    data = json.dumps(body or {}).encode() if method == "POST" else None
    if data is not None:
        headers["Content-Type"] = "application/json"
    connection.request(method, path, body=data, headers=headers)
    response = connection.getresponse()
    return response.status, response.read()


def test_page_and_vendor_files_are_served(bridge: int) -> None:
    assert _request(bridge, "GET", "/")[0] == 200
    assert _request(bridge, "GET", "/vendor/three.module.min.js")[0] == 200


@pytest.mark.parametrize("path", ["/../karen/app.py", "/%2e%2e/config/settings.json", "/missing.js"])
def test_files_outside_the_web_folder_are_not_served(bridge: int, path: str) -> None:
    assert _request(bridge, "GET", path)[0] == 404


def test_commands_require_the_token(bridge: int) -> None:
    assert _request(bridge, "POST", "/api/stop")[0] == 403
    assert _request(bridge, "POST", "/api/stop", token="wrong")[0] == 403
    assert _request(bridge, "POST", "/api/stop", token="secret-token")[0] == 200
    assert _request(bridge, "GET", "/events")[0] == 403


def test_other_hosts_are_rejected(bridge: int) -> None:
    # Blocks DNS-rebinding pages that point another hostname at 127.0.0.1.
    assert _request(bridge, "GET", "/", host="evil.example")[0] == 403
