from __future__ import annotations

import sys
import threading
import traceback

from PySide6.QtCore import QThread, Signal

from .engine import Cancelled, EngineError


class JobThread(QThread):
    progress = Signal(str, float)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            result = self._fn(lambda text, frac: self.progress.emit(text, float(frac)), self.cancel_event)
        except Cancelled:
            self.cancelled.emit()
        except EngineError as e:
            self.failed.emit(str(e))
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            self.succeeded.emit(result)
