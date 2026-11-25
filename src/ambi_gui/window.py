from __future__ import annotations

import html
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QAbstractButton, QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter, QTextBrowser,
                               QToolButton, QVBoxLayout, QWidget)

from ambi_core.color import psnr_rgb
from ambi_core.metrics import ssim_y
from .engine import (DEFAULT_LAM, IMAGE_EXTS, LAM_MAX, LAM_MIN, PRUNE_PRESETS, Engine, EncodeSettings, EngineError, default_workers,
                     lam_to_quality, load_picture, quality_to_lam, torch_status)
from .viewer import CompareView
from .worker import JobThread

LOSSLESS_EXTS = (".png", ".bmp", ".tif", ".tiff")
MUTED = "color:#9a9aa0"


def rel(nbytes: float, ref: float) -> str:
    d = 100 * (1 - nbytes / ref)
    return f"{abs(d):.1f}% smaller" if d >= 0 else f"{abs(d):.1f}% LARGER"


def fmt_bytes(n: float) -> str:
    return f"{int(n):,} bytes ({n / 1024:,.1f} KB)"


def default_dir(name: str) -> str:
    for d in (Path.cwd() / name, Path(__file__).resolve().parents[2] / name):
        if d.is_dir():
            return str(d)
    return str(Path.cwd() / name)


class MainWindow(QMainWindow):
    def __init__(self, open_path: str | None = None):
        super().__init__()
        self.setWindowTitle("AMBI")
        self.resize(1380, 860)
        self.setAcceptDrops(True)
        self.engine = Engine()
        self.qs = QSettings("AMBI", "ambi-gui")
        self.rgb: np.ndarray | None = None
        self.src_path: Path | None = None
        self.src_bytes = 0
        self.outcome = None
        self.ambi_data: bytes | None = None
        self.decoded: np.ndarray | None = None
        self.job: JobThread | None = None
        self._build()
        self._load_settings()
        self._refresh_files(self._saved["profile"], self._saved["policy"])
        self._sync()
        self._show_results(self._intro_html())
        if open_path:
            self.open_any(open_path)

    def _build(self):
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 10, 10)
        lv.setSpacing(10)

        # picture
        g = QGroupBox("Picture")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        self.btn_open_pic = QPushButton("Open picture…")
        self.btn_open_ambi = QPushButton("Open .ambi…")
        row.addWidget(self.btn_open_pic)
        row.addWidget(self.btn_open_ambi)
        v.addLayout(row)
        self.lbl_file = QLabel("Nothing loaded. You can also drop a file on the window.")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet(MUTED)
        v.addWidget(self.lbl_file)
        self.btn_compare = QPushButton("Compare the opened .ambi with its original…")
        v.addWidget(self.btn_compare)
        lv.addWidget(g)

        g = QGroupBox("Profile and policy")
        f = QFormLayout(g)
        self.cmb_profile, self.cmb_policy = QComboBox(), QComboBox()
        self.btn_profile_dir, self.btn_policy_dir = QPushButton("Folder…"), QPushButton("Folder…")
        for label, cmb, btn in (("Profile", self.cmb_profile, self.btn_profile_dir), ("Policy", self.cmb_policy, self.btn_policy_dir)):
            cmb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cmb.setMinimumContentsLength(14)
            r = QHBoxLayout()
            r.addWidget(cmb, 1)
            r.addWidget(btn)
            f.addRow(label, r)
        self.profile_dir = default_dir("profiles")
        self.policy_dir = default_dir("policies")
        lv.addWidget(g)

        g = QGroupBox("Encode for")
        gl = QVBoxLayout(g)
        self.rb_quality, self.rb_size, self.rb_psnr = QRadioButton("Quality"), QRadioButton("File size"), QRadioButton("PSNR")
        self.mode_group = QButtonGroup(self)
        for rb in (self.rb_quality, self.rb_size, self.rb_psnr):
            self.mode_group.addButton(rb)
        self.rb_quality.setChecked(True)
        gl.addWidget(self.rb_quality)
        r = QHBoxLayout()
        self.sl_quality = QSlider(Qt.Orientation.Horizontal)
        self.sl_quality.setRange(0, 100)
        self.lbl_quality = QLabel()
        self.lbl_quality.setMinimumWidth(110)
        r.addWidget(self.sl_quality, 1)
        r.addWidget(self.lbl_quality)
        gl.addLayout(r)
        hint = QLabel("left = smaller file, right = better picture")
        hint.setStyleSheet(MUTED)
        gl.addWidget(hint)
        gl.addWidget(self.rb_size)
        r = QHBoxLayout()
        self.sp_size = QDoubleSpinBox()
        self.sp_size.setRange(0.01, 10_000_000)
        self.sp_size.setDecimals(2)
        self.sp_size.setValue(10)
        self.cmb_unit = QComboBox()
        self.cmb_unit.addItems(["KB", "MB", "% of the original file"])
        self.cmb_unit.setCurrentIndex(2)
        r.addWidget(self.sp_size, 1)
        r.addWidget(self.cmb_unit, 1)
        gl.addLayout(r)
        lbl = QLabel("the best quality that fits under this size")
        lbl.setStyleSheet(MUTED)
        gl.addWidget(lbl)
        gl.addWidget(self.rb_psnr)
        r = QHBoxLayout()
        self.sp_psnr = QDoubleSpinBox()
        self.sp_psnr.setRange(15, 60)
        self.sp_psnr.setSingleStep(0.5)
        self.sp_psnr.setValue(36.0)
        self.sp_psnr.setSuffix(" dB")
        r.addWidget(self.sp_psnr)
        r.addStretch(1)
        gl.addLayout(r)
        lbl = QLabel("the smallest file that reaches this PSNR")
        lbl.setStyleSheet(MUTED)
        gl.addWidget(lbl)
        lv.addWidget(g)

        g = QGroupBox("Pruning")
        v = QVBoxLayout(g)
        self.cmb_prune = QComboBox()
        for label, k, _ in PRUNE_PRESETS:
            self.cmb_prune.addItem(label, k)
        self.cmb_prune.setCurrentIndex(2)
        v.addWidget(self.cmb_prune)
        self.lbl_prune = QLabel()
        self.lbl_prune.setWordWrap(True)
        self.lbl_prune.setStyleSheet(MUTED)
        v.addWidget(self.lbl_prune)
        note = QLabel("k = how many candidate cuts the exact search keeps per rectangle. More = less pruning = slower, smaller files.")
        note.setWordWrap(True)
        note.setStyleSheet(MUTED)
        v.addWidget(note)
        lv.addWidget(g)

        g = QGroupBox("Speed")
        f = QFormLayout(g)
        self.sp_workers = QSpinBox()
        self.sp_workers.setRange(1, 64)
        self.sp_workers.setValue(default_workers())
        self.cmb_device = QComboBox()
        self.cmb_device.addItems(["Auto", "CPU", "CUDA"])
        self.lbl_device = QLabel()
        self.lbl_device.setStyleSheet(MUTED)
        self.lbl_device.setWordWrap(True)
        f.addRow("Workers", self.sp_workers)
        f.addRow("Network on", self.cmb_device)
        f.addRow(self.lbl_device)
        have_torch, have_cuda = torch_status()
        self.lbl_device.setText("CUDA is available." if have_cuda else "No CUDA: the network runs on the CPU." if have_torch
                                else "PyTorch is not installed: only 'Exact oracle' works.")
        lv.addWidget(g)

        self.chk_compare = QCheckBox("Compare with PNG and JPEG after encoding")
        self.chk_compare.setChecked(True)
        lv.addWidget(self.chk_compare)

        row = QHBoxLayout()
        self.btn_encode = QPushButton("Encode")
        self.btn_encode.setDefault(True)
        self.btn_encode.setMinimumHeight(34)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setMinimumHeight(34)
        row.addWidget(self.btn_encode, 2)
        row.addWidget(self.btn_cancel, 1)
        lv.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setMaximumHeight(8)
        lv.addWidget(self.bar)
        self.lbl_status = QLabel("Open a picture to start.")
        self.lbl_status.setWordWrap(True)
        lv.addWidget(self.lbl_status)
        row = QHBoxLayout()
        self.btn_save_ambi = QPushButton("Save .ambi…")
        self.btn_save_png = QPushButton("Save decoded PNG…")
        row.addWidget(self.btn_save_ambi)
        row.addWidget(self.btn_save_png)
        lv.addLayout(row)
        lv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(390)
        scroll.setMaximumWidth(460)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        self.mode_btns = {}
        grp = QButtonGroup(self)
        for key, text in (("decoded", "AMBI"), ("original", "Original"), ("swipe", "Swipe")):
            b = QToolButton()
            b.setText(text)
            b.setCheckable(True)
            b.setMinimumWidth(70)
            grp.addButton(b)
            bar.addWidget(b)
            self.mode_btns[key] = b
        self.mode_btns["decoded"].setChecked(True)
        bar.addSpacing(16)
        self.btn_fit, self.btn_100 = QPushButton("Fit"), QPushButton("100%")
        bar.addWidget(self.btn_fit)
        bar.addWidget(self.btn_100)
        self.lbl_zoom = QLabel()
        self.lbl_zoom.setStyleSheet(MUTED)
        bar.addSpacing(10)
        bar.addWidget(self.lbl_zoom)
        bar.addStretch(1)
        hint = QLabel("wheel = zoom · drag = pan · space = A/B")
        hint.setStyleSheet(MUTED)
        bar.addWidget(hint)
        rv.addLayout(bar)
        self.view = CompareView()
        rv.addWidget(self.view, 1)
        self.results = QTextBrowser()
        self.results.setOpenLinks(False)
        self.results.setMinimumHeight(215)
        self.results.setMaximumHeight(300)
        rv.addWidget(self.results)

        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([410, 970])
        self.setCentralWidget(split)

        m = self.menuBar().addMenu("&File")
        for text, key, slot in (("Open picture…", QKeySequence.StandardKey.Open, self.choose_picture),
                                ("Open .ambi…", "Ctrl+Shift+O", self.choose_ambi),
                                ("Save .ambi…", QKeySequence.StandardKey.Save, self.save_ambi),
                                ("Save decoded PNG…", "Ctrl+Shift+S", self.save_png)):
            a = QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(slot)
            m.addAction(a)
        m.addSeparator()
        a = QAction("Quit", self)
        a.setShortcut("Ctrl+Q")
        a.triggered.connect(self.close)
        m.addAction(a)

        self.btn_open_pic.clicked.connect(self.choose_picture)
        self.btn_open_ambi.clicked.connect(self.choose_ambi)
        self.btn_compare.clicked.connect(self.compare_with_original)
        self.btn_profile_dir.clicked.connect(lambda: self.choose_dir("profile"))
        self.btn_policy_dir.clicked.connect(lambda: self.choose_dir("policy"))
        self.btn_encode.clicked.connect(self.encode)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_save_ambi.clicked.connect(self.save_ambi)
        self.btn_save_png.clicked.connect(self.save_png)
        self.btn_fit.clicked.connect(self.view.fit)
        self.btn_100.clicked.connect(self.view.actual_size)
        for key, b in self.mode_btns.items():
            b.clicked.connect(lambda _=False, k=key: self.view.set_mode(k))
        self.view.viewChanged.connect(self._view_changed)
        self.view.modeChanged.connect(lambda m: self.mode_btns[m].setChecked(True))
        self.mode_group.buttonToggled.connect(lambda *_: self._sync())
        self.sl_quality.valueChanged.connect(self._quality_changed)
        self.cmb_prune.currentIndexChanged.connect(lambda *_: self._sync())

    def _quality_changed(self, q: int):
        lam = quality_to_lam(q)
        self.lbl_quality.setText(f"{q}   (λ {lam:.4f})")

    def _view_changed(self):
        both = self.view.has_both()
        self.mode_btns["swipe"].setEnabled(both)
        self.mode_btns["original"].setEnabled(both or self.view.mode() == "original")
        self.mode_btns["decoded"].setEnabled(both or self.view.mode() == "decoded")
        self.mode_btns[self.view.mode()].setChecked(True)
        s = self.view.picture_size()
        self.lbl_zoom.setText(f"{s[0]}×{s[1]}   {self.view.zoom_percent():.0f}%" if s else "")

    def _k(self) -> int:
        return int(self.cmb_prune.currentData())

    def _sync(self):
        busy = self.job is not None
        k = self._k()
        self.lbl_prune.setText(PRUNE_PRESETS[self.cmb_prune.currentIndex()][2])
        self.sl_quality.setEnabled(self.rb_quality.isChecked() and not busy)
        self.sp_size.setEnabled(self.rb_size.isChecked() and not busy)
        self.cmb_unit.setEnabled(self.rb_size.isChecked() and not busy)
        self.sp_psnr.setEnabled(self.rb_psnr.isChecked() and not busy)
        self.cmb_policy.setEnabled(k >= 0 and not busy)
        self.btn_policy_dir.setEnabled(k >= 0 and not busy)
        self.sp_workers.setEnabled(k > 0 and not busy)
        self.cmb_device.setEnabled(k >= 0 and not busy)
        for w in (self.cmb_profile, self.btn_profile_dir, self.cmb_prune, self.rb_quality, self.rb_size, self.rb_psnr,
                  self.chk_compare, self.btn_open_pic, self.btn_open_ambi):
            w.setEnabled(not busy)
        self.btn_encode.setEnabled(self.rgb is not None and not busy)
        self.btn_cancel.setEnabled(busy)
        self.btn_save_ambi.setEnabled(self.ambi_data is not None and not busy)
        self.btn_save_png.setEnabled(self.decoded is not None and not busy)
        self.btn_compare.setVisible(self.ambi_data is not None and self.outcome is None)
        self.btn_compare.setEnabled(self.ambi_data is not None and self.decoded is not None and not busy)
        self._quality_changed(self.sl_quality.value())
        self._view_changed()

    def _refresh_files(self, keep_profile: str | None = None, keep_policy: str | None = None):
        keep_profile = keep_profile or self.cmb_profile.currentData()
        keep_policy = keep_policy or self.cmb_policy.currentData()
        for cmb, folder, pat, keep, prefer in ((self.cmb_profile, self.profile_dir, "*.ambiprofile", keep_profile, "clic-base-2k"),
                                               (self.cmb_policy, self.policy_dir, "*.ambipolicy", keep_policy, "clic2k_big")):
            cmb.blockSignals(True)
            cmb.clear()
            files = sorted(Path(folder).glob(pat)) if Path(folder).is_dir() else []
            for fp in files:
                cmb.addItem(fp.stem, str(fp))
            if not files:
                cmb.addItem("(none in this folder)", "")
            else:
                idx = next((i for i in range(cmb.count()) if cmb.itemData(i) == keep), None)
                if idx is None:
                    idx = next((i for i in range(cmb.count()) if prefer in cmb.itemText(i)), 0)
                cmb.setCurrentIndex(idx)
            cmb.setToolTip(str(folder))
            cmb.blockSignals(False)

    def _load_settings(self):
        q = self.qs
        self.profile_dir = q.value("profile_dir", self.profile_dir)
        self.policy_dir = q.value("policy_dir", self.policy_dir)
        self._saved = dict(profile=q.value("profile", ""), policy=q.value("policy", ""))
        self.cmb_prune.setCurrentIndex(int(q.value("prune_index", 2)))
        self.sp_workers.setValue(int(q.value("workers", default_workers())))
        self.cmb_device.setCurrentIndex(int(q.value("device_index", 0)))
        self.sl_quality.setValue(int(q.value("quality", round(lam_to_quality(DEFAULT_LAM)))))
        self.sp_size.setValue(float(q.value("size_value", 10)))
        self.cmb_unit.setCurrentIndex(int(q.value("size_unit", 2)))
        self.sp_psnr.setValue(float(q.value("psnr", 36.0)))
        self.chk_compare.setChecked(str(q.value("compare", "true")).lower() == "true")
        {"quality": self.rb_quality, "size": self.rb_size, "psnr": self.rb_psnr}.get(str(q.value("mode", "quality")), self.rb_quality).setChecked(True)
        self._last_dir = q.value("last_dir", str(Path.home()))
        g = q.value("geometry")
        if g is not None:
            self.restoreGeometry(g)

    def _save_settings(self):
        q = self.qs
        q.setValue("profile_dir", self.profile_dir)
        q.setValue("policy_dir", self.policy_dir)
        q.setValue("profile", self.cmb_profile.currentData() or "")
        q.setValue("policy", self.cmb_policy.currentData() or "")
        q.setValue("prune_index", self.cmb_prune.currentIndex())
        q.setValue("workers", self.sp_workers.value())
        q.setValue("device_index", self.cmb_device.currentIndex())
        q.setValue("quality", self.sl_quality.value())
        q.setValue("size_value", self.sp_size.value())
        q.setValue("size_unit", self.cmb_unit.currentIndex())
        q.setValue("psnr", self.sp_psnr.value())
        q.setValue("compare", self.chk_compare.isChecked())
        q.setValue("mode", self._mode_name())
        q.setValue("last_dir", self._last_dir)
        q.setValue("geometry", self.saveGeometry())

    def _mode_name(self) -> str:
        return "size" if self.rb_size.isChecked() else "psnr" if self.rb_psnr.isChecked() else "quality"

    def _status(self, text: str, busy: bool = False):
        self.lbl_status.setText(text)
        self.bar.setRange(0, 0 if busy and "·" not in text else 1000)

    def choose_dir(self, which: str):
        cur = self.profile_dir if which == "profile" else self.policy_dir
        d = QFileDialog.getExistingDirectory(self, f"Folder with {'.ambiprofile' if which == 'profile' else '.ambipolicy'} files", cur)
        if d:
            if which == "profile":
                self.profile_dir = d
            else:
                self.policy_dir = d
            self._refresh_files()
            self._sync()

    def choose_picture(self):
        exts = " ".join(f"*{e}" for e in IMAGE_EXTS)
        p, _ = QFileDialog.getOpenFileName(self, "Open a picture", self._last_dir, f"Pictures ({exts});;All files (*)")
        if p:
            self.open_any(p)

    def choose_ambi(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open an .ambi file", self._last_dir, "AMBI files (*.ambi);;All files (*)")
        if p:
            self.open_any(p)

    def open_any(self, path: str):
        if self.job is not None:
            return
        p = Path(path)
        self._last_dir = str(p.parent)
        if p.suffix.lower() == ".ambi":
            self.open_ambi(p)
        else:
            self.open_picture(p)

    def open_picture(self, p: Path):
        try:
            rgb, note = load_picture(p)
        except Exception as e:
            QMessageBox.warning(self, "AMBI", f"Could not open {p.name}:\n{e}")
            return
        self.rgb, self.src_path, self.src_bytes = rgb, p, p.stat().st_size
        self.outcome, self.ambi_data, self.decoded = None, None, None
        H, W = rgb.shape[:2]
        lossy = "" if p.suffix.lower() in LOSSLESS_EXTS else f"\nThis is already a lossy {p.suffix.lstrip('.').upper()} file."
        self.lbl_file.setText(f"{p.name}\n{W}×{H} px · {fmt_bytes(self.src_bytes)}" + (f"\n{note}" if note else "") + lossy)
        self.view.set_images(rgb, None, "original")
        self._status("Choose what to encode for, then press Encode.")
        self._show_results(self._intro_html(picture=True))
        self._sync()

    def open_ambi(self, p: Path):
        try:
            data = p.read_bytes()
        except OSError as e:
            QMessageBox.warning(self, "AMBI", f"Could not read {p.name}:\n{e}")
            return
        dirs = [self.profile_dir, str(Path(self.cmb_profile.currentData() or "").parent)]
        self._run(lambda prog, cancel: self.engine.decode(data, dirs, prog), "Decoding…",
                  lambda out: self._ambi_opened(p, data, out))

    def _ambi_opened(self, p: Path, data: bytes, out):
        self.rgb, self.src_path, self.src_bytes, self.outcome = None, p, len(data), None
        self.ambi_data, self.decoded = data, out.rgb
        i = out.info
        self.lbl_file.setText(f"{p.name}\n{i['width']}×{i['height']} px · {fmt_bytes(i['total_bytes'])}")
        self.view.set_images(None, out.rgb, "decoded")
        self._status(f"Decoded in {out.seconds:.1f} s.")
        M = f"style='{MUTED}'"
        self._show_results(
            f"<table cellspacing=4>"
            f"<tr><td {M}>File&nbsp;&nbsp;</td><td><b>{html.escape(p.name)}</b></td></tr>"
            f"<tr><td {M}>Size</td><td><b>{fmt_bytes(i['total_bytes'])}</b> · {i['bpp']:.3f} bpp · {i['width']}×{i['height']} px</td></tr>"
            f"<tr><td {M}>Profile</td><td>{html.escape(out.profile_name)} (id {i['profile_id']}) · deblock level {i['deblock_level']}</td></tr>"
            f"<tr><td {M}>Decoded</td><td>{out.seconds:.2f} s</td></tr></table>"
            f"<p style='{MUTED}'>There is no original here, so no PSNR or SSIM. Use “Compare the opened .ambi with its original…” "
            f"to load the picture it was made from.</p>")
        self._sync()

    def compare_with_original(self):
        if self.decoded is None:
            return
        exts = " ".join(f"*{e}" for e in IMAGE_EXTS)
        p, _ = QFileDialog.getOpenFileName(self, "Open the original picture", self._last_dir, f"Pictures ({exts});;All files (*)")
        if not p:
            return
        try:
            rgb, _note = load_picture(p)
        except Exception as e:
            QMessageBox.warning(self, "AMBI", f"Could not open {Path(p).name}:\n{e}")
            return
        if rgb.shape != self.decoded.shape:
            QMessageBox.warning(self, "AMBI", f"That picture is {rgb.shape[1]}×{rgb.shape[0]} but the .ambi file is "
                                f"{self.decoded.shape[1]}×{self.decoded.shape[0]}, so they cannot be compared.")
            return
        dec, orig_bytes, ambi_bytes = self.decoded, Path(p).stat().st_size, len(self.ambi_data)

        def job(prog, cancel):
            prog("Measuring…", 0.5)
            return psnr_rgb(rgb, dec), ssim_y(rgb, dec)

        def done(res):
            ps, ss = res
            self.view.set_images(rgb, dec, "swipe")
            self._status("Compared.")
            self._show_results(
                f"<table cellspacing=4><tr><td><span style='{MUTED}'>.ambi</span></td><td><b>{fmt_bytes(ambi_bytes)}</b> — "
                f"{rel(ambi_bytes, orig_bytes)} than the original file ({orig_bytes / ambi_bytes:.1f}:1) · "
                f"{rel(ambi_bytes, rgb.size)} than raw · {ambi_bytes * 8 / (rgb.shape[0] * rgb.shape[1]):.3f} bpp</td></tr>"
                f"<tr><td><span style='{MUTED}'>Quality</span></td><td>PSNR <b>{ps:.2f} dB</b> · SSIM(Y) <b>{ss:.4f}</b></td></tr>"
                f"<tr><td><span style='{MUTED}'>Original</span></td><td>{html.escape(Path(p).name)} · {fmt_bytes(orig_bytes)}</td></tr></table>")
            self._sync()

        self._run(job, "Measuring…", done)

    def _settings(self) -> EncodeSettings:
        k = self._k()
        mode = self._mode_name()
        size = self.sp_size.value()
        unit = self.cmb_unit.currentIndex()
        target = size * 1024 if unit == 0 else size * 1024 * 1024 if unit == 1 else size / 100.0 * self.src_bytes
        if k >= 0 and not self.cmb_policy.currentData():
            raise EngineError("There is no policy file in the policy folder. Pick the folder with your .ambipolicy files, "
                              "or choose 'Exact oracle' under Pruning.")
        if not self.cmb_profile.currentData():
            raise EngineError("There is no profile file in the profile folder. Pick the folder with your .ambiprofile files.")
        return EncodeSettings(profile_path=self.cmb_profile.currentData(), policy_path=self.cmb_policy.currentData() or "",
                              prune_k=k, workers=self.sp_workers.value(), device=("auto", "cpu", "cuda")[self.cmb_device.currentIndex()],
                              mode=mode, lam=quality_to_lam(self.sl_quality.value()), target_bytes=int(round(target)),
                              target_psnr=self.sp_psnr.value(), compare=self.chk_compare.isChecked())

    def encode(self):
        if self.rgb is None or self.job is not None:
            return
        try:
            s = self._settings()
        except EngineError as e:
            QMessageBox.warning(self, "AMBI", str(e))
            return
        rgb, src_bytes = self.rgb, self.src_bytes
        self._save_settings()
        self._run(lambda prog, cancel: self.engine.encode(rgb, s, prog, cancel, src_bytes), "Encoding…", self._encoded)

    def _encoded(self, o):
        self.outcome, self.ambi_data, self.decoded = o, o.data, o.decoded
        self.view.set_images(self.rgb, o.decoded, "swipe")
        self._status(f"Done in {o.seconds:.1f} s." + ("" if o.reached else " The target was not met; details below."))
        self._show_results(self._outcome_html(o))
        self._sync()

    def cancel(self):
        if self.job is not None:
            self.job.cancel()
            self.btn_cancel.setEnabled(False)
            self._status("Cancelling…", busy=True)

    def _run(self, fn, label: str, on_done):
        self._on_done = on_done
        self.job = JobThread(fn, self)
        self.job.progress.connect(self._on_progress)
        self.job.succeeded.connect(self._on_succeeded)
        self.job.failed.connect(self._on_failed)
        self.job.cancelled.connect(self._on_cancelled)
        self._status(label, busy=True)
        self.job.start()
        self._sync()

    def _on_progress(self, text: str, frac: float):
        self.lbl_status.setText(text)
        self.bar.setRange(0, 1000)
        self.bar.setValue(int(frac * 1000))

    def _on_succeeded(self, result):
        done = self._on_done
        self._job_end()
        done(result)

    def _on_failed(self, msg: str):
        self._job_end()
        self._status("Failed.")
        QMessageBox.critical(self, "AMBI", msg)

    def _on_cancelled(self):
        self._job_end()
        self._status("Cancelled.")

    def _job_end(self):
        job, self.job = self.job, None
        if job is not None:
            job.wait()
            job.deleteLater()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self._sync()

    def save_ambi(self):
        if self.ambi_data is None:
            return
        stem = self.src_path.stem if self.src_path else "picture"
        p, _ = QFileDialog.getSaveFileName(self, "Save the .ambi file", str(Path(self._last_dir) / f"{stem}.ambi"), "AMBI files (*.ambi)")
        if p:
            Path(p).write_bytes(self.ambi_data)
            self._last_dir = str(Path(p).parent)
            self._status(f"Saved {Path(p).name} ({len(self.ambi_data):,} bytes).")

    def save_png(self):
        if self.decoded is None:
            return
        stem = self.src_path.stem if self.src_path else "picture"
        p, _ = QFileDialog.getSaveFileName(self, "Save the decoded picture", str(Path(self._last_dir) / f"{stem}.decoded.png"), "PNG (*.png)")
        if p:
            Image.fromarray(self.decoded).save(p)
            self._last_dir = str(Path(p).parent)
            self._status(f"Saved {Path(p).name}.")

    def _show_results(self, body: str):
        self.results.setHtml(f"<div style='font-size:13px'>{body}</div>")

    def _intro_html(self, picture: bool = False) -> str:
        if picture:
            return (f"<p style='{MUTED}'>Pick quality, a file size or a PSNR on the left, choose how much pruning, then press Encode. "
                    f"The result and its numbers appear here, and the picture shows the original next to the decoded file "
                    f"(drag the divider).</p>")
        return (f"<p style='{MUTED}'>Open a picture to encode it, or an .ambi file to view it. Drag and drop works too.</p>")

    def _outcome_html(self, o) -> str:
        n, orig, raw = o.nbytes, o.original_file_bytes or 0, o.width * o.height * 3
        row = lambda k, v: f"<tr><td style='{MUTED}' valign=top>{k}&nbsp;&nbsp;</td><td>{v}</td></tr>"
        rows = [row("Result", f"<b>{fmt_bytes(n)}</b> — {rel(n, orig)} than the original file ({orig / n:.1f}:1) · "
                              f"{rel(n, raw)} than raw · {n * 8 / (o.width * o.height):.3f} bpp"),
                row("Quality", f"PSNR <b>{o.psnr:.2f} dB</b> · SSIM(Y) <b>{o.ssim:.4f}</b>")]
        if o.mode == "quality":
            rows.append(row("Setting", f"quality {lam_to_quality(o.lam):.0f} (λ {o.lam:.4f})"))
        else:
            steps = ", ".join(f"λ {l:.4f} → {b / 1024:,.1f} KB" + (f" / {p:.2f} dB" if p is not None else "") for l, b, p in o.probes)
            rows.append(row("Search", f"{html.escape(o.message)}<br><span style='{MUTED}'>{steps}</span>"))
        if o.original_file_bytes and self.src_path is not None and self.src_path.suffix.lower() not in LOSSLESS_EXTS:
            rows.append(row("Note", f"<span style='{MUTED}'>the original is already a lossy {self.src_path.suffix.lstrip('.').upper()} file, "
                                    f"so “smaller than the original file” compares with another lossy file</span>"))
        if o.png_bytes is not None:
            rows.append(row("vs PNG", f"PNG of the same pixels: {fmt_bytes(o.png_bytes)} → AMBI is <b>{rel(n, o.png_bytes)}</b> "
                                      f"<span style='{MUTED}'>(PNG keeps every pixel, AMBI does not)</span>"))
        if o.jpeg is not None:
            j = o.jpeg
            if j["reached"]:
                rows.append(row("vs JPEG", f"JPEG 4:2:0 at the same PSNR (≥ {o.psnr:.2f} dB, quality {j['quality']}): {fmt_bytes(j['bytes'])} → "
                                           f"AMBI is <b>{rel(n, j['bytes'])}</b>"))
            else:
                rows.append(row("vs JPEG", f"JPEG cannot reach {o.psnr:.2f} dB (quality 100 gives {j['psnr']:.2f} dB, {fmt_bytes(j['bytes'])})"))
        rows.append(row("Search effort", f"{html.escape(o.how)}" + (
            f"<br><span style='{MUTED}'>looked at {o.prune['looked']:.1f}% of the rectangles, priced {o.prune['priced']:.1f}% as leaves, "
            f"{o.prune['ms']:.0f} ms per tile</span>" if o.prune else "")))
        rows.append(row("Time", f"{o.seconds:.1f} s in all · the chosen encode {o.encode_seconds:.1f} s · decode {o.decode_seconds:.2f} s"
                                f" · deblock level {o.deblock}"))
        return "<table cellspacing=3>" + "".join(rows) + "</table>"

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.isLocalFile() for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.isLocalFile():
                self.open_any(u.toLocalFile())
                break

    def closeEvent(self, e):
        if self.job is not None:
            self.job.cancel()
            self.job.wait(15000)
        self._save_settings()
        self.engine.close()
        super().closeEvent(e)
