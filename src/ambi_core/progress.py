from __future__ import annotations

import shutil
import sys


def fmt_time(seconds: float) -> str:
    s = int(max(0, seconds) + 0.5)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s" if m else f"{s}s"


class LiveLine:
    def __init__(self, stream=None, enabled: bool | None = None):
        self.stream = stream or sys.stdout
        self.enabled = bool(getattr(self.stream, "isatty", lambda: False)()) if enabled is None else enabled
        self._w = 0

    def _columns(self) -> int:
        try:
            return max(20, shutil.get_terminal_size((100, 24)).columns - 1)
        except OSError:
            return 99

    def update(self, text: str) -> None:
        if not self.enabled:
            return
        text = text[: self._columns()]
        self.stream.write("\r" + text + " " * max(0, self._w - len(text)))
        self.stream.flush()
        self._w = len(text)

    def clear(self) -> None:
        if self.enabled and self._w:
            self.stream.write("\r" + " " * self._w + "\r")
            self.stream.flush()
        self._w = 0

    def done(self) -> None:
        if self.enabled and self._w:
            self.stream.write("\n")
            self.stream.flush()
        self._w = 0
