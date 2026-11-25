from __future__ import annotations

import sys


def apply_dark(app) -> None:
    from PySide6.QtGui import QColor, QPalette
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in ((QPalette.ColorRole.Window, "#26262a"), (QPalette.ColorRole.WindowText, "#e6e6e6"),
                      (QPalette.ColorRole.Base, "#1b1b1d"), (QPalette.ColorRole.AlternateBase, "#26262a"),
                      (QPalette.ColorRole.Text, "#e6e6e6"), (QPalette.ColorRole.Button, "#33333a"),
                      (QPalette.ColorRole.ButtonText, "#e6e6e6"), (QPalette.ColorRole.ToolTipBase, "#33333a"),
                      (QPalette.ColorRole.ToolTipText, "#e6e6e6"), (QPalette.ColorRole.Highlight, "#3d7bd9"),
                      (QPalette.ColorRole.HighlightedText, "#ffffff"), (QPalette.ColorRole.PlaceholderText, "#8a8a90")):
        pal.setColor(role, QColor(col))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor("#6d6d73"))
    app.setPalette(pal)


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print('PySide6 is not installed: pip install PySide6   (or: pip install -e ".[gui]")', file=sys.stderr)
        return 1
    from .window import MainWindow
    app = QApplication(argv)
    apply_dark(app)
    win = MainWindow(argv[1] if len(argv) > 1 else None)
    win.show()
    return app.exec()
