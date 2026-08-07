from dataclasses import dataclass

from karen.core.application_service import ServiceResponse
from karen.voice.service import VoiceService


@dataclass
class FakeRecognizer:
    text: str = "open notepad"

    def transcribe_once(self) -> str:
        return self.text


@dataclass
class FakeSpeaker:
    spoken: list[str]

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakeApplication:
    def handle_text(self, text: str) -> ServiceResponse:
        return ServiceResponse(f"processed: {text}", "fast_path", True)


def test_voice_service_connects_transcription_to_application_and_speech() -> None:
    spoken: list[str] = []
    service = VoiceService(FakeApplication(), FakeRecognizer(), FakeSpeaker(spoken))

    result = service.listen_and_process()

    assert result.transcript == "open notepad"
    assert result.response == "processed: open notepad"
    assert spoken == ["processed: open notepad"]
    assert service.state.value == "ready"
