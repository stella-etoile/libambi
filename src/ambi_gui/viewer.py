from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

BG = QColor(27, 27, 29)
MODES = ("decoded", "original", "swipe")


def to_qimage(rgb: np.ndarray) -> QImage:
    """(H, W, 3) uint8 -> QImage that owns its pixels."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class CompareView(QWidget):
    viewChanged = Signal()
    modeChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._orig: QImage | None = None
        self._dec: QImage | None = None
        self._mode = "decoded"
        self._zoom = 1.0
        self._off = QPointF(0, 0)
        self._fit = True
        self._div = 0.5
        self._drag = None
        self._hint = "Drop a picture here to encode it,\nor an .ambi file to view it"

    def set_images(self, original: np.ndarray | None, decoded: np.ndarray | None, mode: str | None = None):
        old = self._size()
        self._orig = to_qimage(original) if original is not None else None
        self._dec = to_qimage(decoded) if decoded is not None else None
        if mode:
            self._mode = mode
        elif self._orig is not None and self._dec is None:
            self._mode = "original"
        elif self._dec is not None and self._orig is None:
            self._mode = "decoded"
        if self._size() != old:
            self._fit = True
        if self._fit:
            self._refit()
        self.update()
        self.viewChanged.emit()

    def clear(self, hint: str | None = None):
        self._orig = self._dec = None
        if hint:
            self._hint = hint
        self.update()
        self.viewChanged.emit()

    def has_both(self) -> bool:
        return self._orig is not None and self._dec is not None

    def set_mode(self, mode: str):
        if mode in MODES and mode != self._mode:
            self._mode = mode
            self.update()
            self.viewChanged.emit()

    def mode(self) -> str:
        return self._mode

    def picture_size(self):
        return self._size()

    def _size(self):
        im = self._dec or self._orig
        return (im.width(), im.height()) if im else None

    def _fit_zoom(self) -> float:
        s = self._size()
        if not s:
            return 1.0
        return max(0.01, min(4.0, (self.width() - 16) / s[0], (self.height() - 16) / s[1]))

    def _refit(self):
        s = self._size()
        if not s:
            return
        self._zoom = self._fit_zoom()
        self._off = QPointF((self.width() - s[0] * self._zoom) / 2, (self.height() - s[1] * self._zoom) / 2)

    def fit(self):
        self._fit = True
        self._refit()
        self.update()
        self.viewChanged.emit()

    def actual_size(self):
        self._set_zoom(1.0 / self.devicePixelRatioF(), QPointF(self.width() / 2, self.height() / 2))

    def zoom_percent(self) -> float:
        return self._zoom * self.devicePixelRatioF() * 100

    def _set_zoom(self, z: float, anchor: QPointF):
        s = self._size()
        if not s:
            return
        z = min(64.0, max(0.02, z))
        px = (anchor.x() - self._off.x()) / self._zoom
        py = (anchor.y() - self._off.y()) / self._zoom
        self._zoom, self._fit = z, False
        self._off = QPointF(anchor.x() - px * z, anchor.y() - py * z)
        self._clamp()
        self.update()
        self.viewChanged.emit()

    def _clamp(self):
        s = self._size()
        if not s:
            return
        w, h, keep = s[0] * self._zoom, s[1] * self._zoom, 80.0
        x = min(max(self._off.x(), keep - w), self.width() - keep)
        y = min(max(self._off.y(), keep - h), self.height() - keep)
        self._off = QPointF(x, y)

    def _rect(self) -> QRectF:
        s = self._size()
        return QRectF(self._off.x(), self._off.y(), s[0] * self._zoom, s[1] * self._zoom)

    def _divider_x(self) -> float:
        r = self._rect()
        return r.left() + r.width() * self._div

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        if not self._size():
            p.setPen(QColor(150, 150, 155))
            p.setFont(QFont(self.font().family(), 13))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._hint)
            return
        r = self._rect()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 1.0)
        mode = self._mode
        base = {"decoded": self._dec or self._orig, "original": self._orig or self._dec, "swipe": self._dec or self._orig}[mode]
        p.drawImage(r, base)
        label_l = label_r = None
        if mode == "swipe" and self.has_both():
            x = min(max(self._divider_x(), r.left()), r.right())
            p.save()
            p.setClipRect(QRectF(r.left(), r.top(), x - r.left(), r.height()))
            p.drawImage(r, self._orig)
            p.restore()
            p.setPen(QPen(QColor(255, 255, 255, 230), 2))
            top, bot = max(r.top(), 0.0), min(r.bottom(), float(self.height()))
            p.drawLine(QPointF(x, top), QPointF(x, bot))
            mid = (top + bot) / 2
            p.setBrush(QColor(255, 255, 255, 235))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(x, mid), 11, 11)
            p.setPen(QPen(QColor(30, 30, 30), 2))
            p.drawLine(QPointF(x - 5, mid), QPointF(x + 5, mid))
            p.drawLine(QPointF(x - 5, mid), QPointF(x - 2, mid - 3))
            p.drawLine(QPointF(x - 5, mid), QPointF(x - 2, mid + 3))
            p.drawLine(QPointF(x + 5, mid), QPointF(x + 2, mid - 3))
            p.drawLine(QPointF(x + 5, mid), QPointF(x + 2, mid + 3))
            label_l, label_r = "Original", "AMBI"
        elif mode == "original":
            label_l = "Original"
        elif self._dec is not None:
            label_l = "AMBI (decoded)"
        font = QFont(self.font().family(), 10)
        font.setBold(True)
        p.setFont(font)
        for text, left in ((label_l, True), (label_r, False)):
            if not text:
                continue
            w = p.fontMetrics().horizontalAdvance(text) + 14
            x0 = max(r.left(), 6.0) + 6 if left else min(r.right(), self.width() - 6.0) - w - 6
            box = QRectF(x0, max(r.top(), 0.0) + 6, w, 22)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawRoundedRect(box, 4, 4)
            p.setPen(QColor(240, 240, 240))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def resizeEvent(self, e):
        if self._fit:
            self._refit()
        else:
            self._clamp()
        super().resizeEvent(e)
        self.viewChanged.emit()

    def wheelEvent(self, e):
        if self._size():
            steps = e.angleDelta().y() / 120.0
            self._set_zoom(self._zoom * (1.25 ** steps), e.position())
        e.accept()

    def _near_divider(self, pos: QPointF) -> bool:
        if self._mode != "swipe" or not self.has_both():
            return False
        r = self._rect()
        return abs(pos.x() - self._divider_x()) <= 10 and r.top() - 12 <= pos.y() <= r.bottom() + 12

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._size():
            self.setFocus()
            self._drag = ("div",) if self._near_divider(e.position()) else ("pan", e.position())
            self.setCursor(Qt.CursorShape.SizeHorCursor if self._drag[0] == "div" else Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._drag and self._drag[0] == "pan":
            last = self._drag[1]
            self._off += pos - last
            self._drag = ("pan", pos)
            self._fit = False
            self._clamp()
            self.update()
            self.viewChanged.emit()
        elif self._drag and self._drag[0] == "div":
            r = self._rect()
            self._div = min(1.0, max(0.0, (pos.x() - r.left()) / max(1.0, r.width())))
            self.update()
        else:
            self.setCursor(Qt.CursorShape.SizeHorCursor if self._near_divider(pos) else Qt.CursorShape.OpenHandCursor
                           if self._size() else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e):
        self._drag = None
        self.setCursor(Qt.CursorShape.OpenHandCursor if self._size() else Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, e):
        if self._size():
            if self._fit:
                self.actual_size()
            else:
                self.fit()

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_F:
            self.fit()
        elif k == Qt.Key.Key_1:
            self.actual_size()
        elif k == Qt.Key.Key_Space and self.has_both():
            self._mode = "original" if self._mode != "original" else "decoded"
            self.update()
            self.modeChanged.emit(self._mode)
            self.viewChanged.emit()
        else:
            super().keyPressEvent(e)
