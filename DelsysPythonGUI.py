import os
import sys


def _ensure_qt_plugin_path():
    if os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        return
    try:
        import PySide6
    except Exception:
        return
    base = os.path.dirname(PySide6.__file__)
    candidate = os.path.join(base, "plugins", "platforms")
    if os.path.exists(os.path.join(candidate, "qwindows.dll")):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = candidate


_ensure_qt_plugin_path()

from PySide6.QtWidgets import QApplication
from UIControls.LandingScreenController import *

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet('.QLabel { font-size: 12pt;}'
                      '.QPushButton { font-size: 12pt;}'
                      '.QListWidget { font-size: 12pt;}'
                      '.QComboBox{ font-size: 12pt;}'
                      )
    controller = LandingScreenController()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
