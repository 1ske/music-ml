from __future__ import annotations

"""
Mini DAW v3.4 – Tillbaka till Stora Ikoner (Fixad font)
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

from pathlib import Path

# AI-instrumentklassificering
from predict import predict_instrument


import numpy as np
import sounddevice as sd
import soundfile as sf

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox, 
    QSlider, QLineEdit, QHBoxLayout, QVBoxLayout,
    QMessageBox, QFrame, QScrollArea, QColorDialog, QSizePolicy, QListView
)

# Instrument → (ikon-emoji, färg)
INSTRUMENT_STYLES = {
    "cello":           ("🎻", QtGui.QColor("#8e44ad")),
    "clarinet":        ("🎺", QtGui.QColor("#2980b9")),
    "flute":           ("🪈", QtGui.QColor("#1abc9c")),
    "acoustic_guitar": ("🎸", QtGui.QColor("#e67e22")),
    "electric_guitar": ("🎸", QtGui.QColor("#d35400")),
    "organ":           ("🎹", QtGui.QColor("#9b59b6")),
    "piano":           ("🎹", QtGui.QColor("#2ecc71")),
    "saxophone":       ("🎷", QtGui.QColor("#f1c40f")),
    "trumpet":         ("🎺", QtGui.QColor("#f39c12")),
    "violin":          ("🎻", QtGui.QColor("#c0392b")),
    "voice":           ("🗣️", QtGui.QColor("#34495e")),
}


# Lägg till ffmpeg om det finns
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

# ===================== Data Modeller =====================

@dataclass
class AudioClip:
    file_path: str
    start_time: float
    duration: float
    color: QtGui.QColor
    peaks: List[tuple[float, float]] = field(default_factory=list)
    _data_cache: Optional[np.ndarray] = None
    
    def load_data(self):
        if self._data_cache is None and os.path.exists(self.file_path):
            try:
                d, sr = sf.read(self.file_path, always_2d=True)
                self._data_cache = d.astype(np.float32)
                if not self.peaks: self._generate_peaks()
            except: pass

    def _generate_peaks(self):
        if self._data_cache is None: return
        data = self._data_cache[:, 0]
        step = 2000 
        self.peaks = []
        for i in range(0, len(data), step):
            chunk = data[i:i+step]
            if len(chunk) > 0:
                self.peaks.append((float(np.min(chunk)), float(np.max(chunk))))

    def get_audio_segment(self, global_start_frame: int, frames_needed: int, global_sr: int) -> np.ndarray:
        self.load_data()
        if self._data_cache is None: return np.zeros((frames_needed, 2), dtype=np.float32)
        
        clip_start_frame = int(self.start_time * global_sr)
        relative_start = global_start_frame - clip_start_frame
        
        if relative_start >= len(self._data_cache): return np.zeros((frames_needed, 2), dtype=np.float32)
        read_start = max(0, relative_start)
        read_end = min(len(self._data_cache), relative_start + frames_needed)
        pad_front = abs(relative_start) if relative_start < 0 else 0
        
        chunk = self._data_cache[read_start:read_end]
        if len(chunk) == 0: return np.zeros((frames_needed, 2), dtype=np.float32)

        out = np.zeros((frames_needed, 2), dtype=np.float32)
        write_len = min(len(chunk), frames_needed - pad_front)
        out[pad_front : pad_front + write_len] = chunk[:write_len]
        return out

# ===================== UI Helpers =====================

class VerticalLine(QFrame):
    """En enkel vertikal avgränsningslinje"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.setFixedWidth(2)
        self.setStyleSheet("color: #444; background-color: transparent; border: none;") 

class ColorLabel(QLabel):
    """Numrerad färgstrimma - Stylesheet Fix"""
    colorChanged = pyqtSignal(QtGui.QColor)

    def __init__(self, index, color, parent=None):
        super().__init__(str(index+1), parent)
        self.setFixedWidth(35)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._color = color
        self._update_style()

    def _update_style(self):
        c = self._color.name()
        fg = "#000000" if self._color.lightness() > 128 else "#ffffff"
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {c};
                color: {fg};
                font-weight: bold;
                border: none;
                border-bottom: 1px solid #444; 
            }}
        """)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            col = QColorDialog.getColor(self._color, None, "Välj spårfärg")
            if col.isValid():
                self._color = col
                self._update_style()
                self.colorChanged.emit(col)

class WaveformCanvas(QWidget):
    timeline_scroll_request = pyqtSignal(float)
    zoom_request = pyqtSignal(float)
    cursor_moved = pyqtSignal(float)
    file_dropped = pyqtSignal(str, float) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.bg_color = QtGui.QColor(30, 33, 38) 
        
        self.pixels_per_sec = 50.0
        self.view_offset_sec = 0.0
        self.clips: List[AudioClip] = []
        self.playhead_pos = 0.0
        
        self.dragging_clip: Optional[AudioClip] = None
        self.drag_start_x = 0
        self.drag_orig_time = 0.0
        self.scrubbing = False

    def add_clip(self, clip: AudioClip):
        self.clips.append(clip)
        self.update()

    def set_view_params(self, px_per_sec, offset_sec):
        self.pixels_per_sec = px_per_sec
        self.view_offset_sec = offset_sec
        self.update()

    def time_to_x(self, t):
        return (t - self.view_offset_sec) * self.pixels_per_sec

    def x_to_time(self, x):
        return (x / self.pixels_per_sec) + self.view_offset_sec

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if path.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                drop_time = max(0.0, self.x_to_time(event.position().x()))
                self.file_dropped.emit(path, drop_time)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        
        p.fillRect(rect, self.bg_color)
        
        # --- LINJE I BOTTEN ---
        p.setPen(QtGui.QPen(QtGui.QColor(80, 80, 80), 1)) 
        p.drawLine(0, rect.height()-1, rect.width(), rect.height()-1)

        # Rutnät
        p.setPen(QtGui.QPen(QtGui.QColor(45, 48, 55), 1))
        start_sec = int(self.view_offset_sec)
        end_sec = int(self.x_to_time(rect.width())) + 1
        for s in range(start_sec, end_sec + 1):
            x = self.time_to_x(s)
            p.drawLine(int(x), 0, int(x), rect.height())

        # Klipp
        for clip in self.clips:
            cx = self.time_to_x(clip.start_time)
            cw = clip.duration * self.pixels_per_sec
            if cw < 2: cw = 2
            if cx + cw < 0 or cx > rect.width(): continue

            clip_rect = QRectF(cx, 2, cw, rect.height() - 5)
            path = QtGui.QPainterPath()
            path.addRoundedRect(clip_rect, 4, 4)
            
            fill = clip.color
            fill.setAlpha(200)
            p.fillPath(path, fill)
            p.setPen(QtGui.QPen(clip.color.lighter(130), 1))
            p.drawPath(path)
            
            # Vågform
            if clip.peaks:
                p.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20), 1))
                h_avail = rect.height() - 5
                h_half = h_avail / 2
                steps = len(clip.peaks)
                if steps > 0:
                    px_step = cw / max(1, steps)
                    draw_skip = 1 if px_step >= 0.5 else int(0.5 / px_step)
                    for i in range(0, steps, draw_skip):
                        mn, mx = clip.peaks[i]
                        px = cx + i * px_step
                        if 0 <= px <= rect.width():
                            y1 = h_half - (mx * (h_avail/2))
                            y2 = h_half - (mn * (h_avail/2))
                            p.drawLine(int(px), int(y1), int(px), int(y2))

        # Playhead
        ph_x = self.time_to_x(self.playhead_pos)
        if 0 <= ph_x <= rect.width():
            p.setPen(QtGui.QPen(QtGui.QColor(255, 60, 60), 1))
            p.drawLine(int(ph_x), 0, int(ph_x), rect.height())
            p.setBrush(QtGui.QColor(255, 60, 60))
            p.drawRect(int(ph_x)-3, 0, 6, 6)

    def wheelEvent(self, event):
        mods = QApplication.keyboardModifiers()
        if mods == Qt.KeyboardModifier.ControlModifier:
            self.zoom_request.emit(event.angleDelta().y())
        else:
            d = event.angleDelta().y() if event.angleDelta().x() == 0 else event.angleDelta().x()
            self.timeline_scroll_request.emit(-d)
        event.accept()

    def mousePressEvent(self, event):
        x = event.position().x()
        t = self.x_to_time(x)
        modifiers = QApplication.keyboardModifiers()
        
        if modifiers == Qt.KeyboardModifier.ControlModifier and event.button() == Qt.MouseButton.LeftButton:
            clicked_clip = None
            for clip in reversed(self.clips):
                cx_start = self.time_to_x(clip.start_time)
                cx_end = self.time_to_x(clip.start_time + clip.duration)
                if cx_start <= x <= cx_end:
                    clicked_clip = clip
                    break
            if clicked_clip:
                self.dragging_clip = clicked_clip
                self.drag_start_x = x
                self.drag_orig_time = clicked_clip.start_time
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            self.scrubbing = True
            self.cursor_moved.emit(max(0, t))

    def mouseMoveEvent(self, event):
        x = event.position().x()
        t = self.x_to_time(x)
        if self.dragging_clip:
            diff_sec = (x - self.drag_start_x) / self.pixels_per_sec
            self.dragging_clip.start_time = max(0.0, self.drag_orig_time + diff_sec)
            self.update()
        elif self.scrubbing:
            self.cursor_moved.emit(max(0, t))

    def mouseReleaseEvent(self, event):
        self.dragging_clip = None
        self.scrubbing = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

class TrackControls(QFrame):
    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.setFixedWidth(400)
        self.setContentsMargins(0,0,0,0) 
        
        # --- LINJE I BOTTEN ---
        self.setStyleSheet("""
            TrackControls { 
                background-color: #2b2b2b; 
                border-bottom: 1px solid #505050; 
                border-right: 1px solid #444; 
            }
            QLabel { color: #aaa; font-size: 11px; border: none; }
            QLineEdit { background: #1a1a1a; color: #ddd; border: 1px solid #444; border-radius: 3px; padding: 2px; }
        """)
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 10, 0) 
        self.layout.setSpacing(8)

        # 1. Färgstrimma
        hue = (index * 45) % 360
        start_col = QtGui.QColor.fromHsl(hue, 160, 110)
        self.color_label = ColorLabel(index, start_col)
        self.color_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.color_label)

        # 2. Ikon (TILLBAKA TILL STORA EMOJIS UTAN TEXT)
        self.icon_cb = QComboBox()
        self.icon_cb.setView(QListView()) 
        # Bara emojis igen!
        self.icon_cb.addItems(["🎻", "🎺", "🪈", "🎸", "🎹", "🎷", "🗣️"])
        self.icon_cb.setFixedSize(50, 40) # Lite bredare för att emojin ska få plats
        # Tvinga EMOJI FONT så det inte blir punkter
        self.icon_cb.setStyleSheet("""
            QComboBox { 
                font-family: "Segoe UI Emoji", "Segoe UI Symbol", "Apple Color Emoji", sans-serif; 
                font-size: 26px; 
                background: #222; color: #eee; 
                border: 1px solid #444; border-radius: 4px; padding-left: 5px; 
            }
            QComboBox::drop-down { border: none; width: 0px; }
            QListView { font-size: 26px; min-height: 200px; } 
        """)
        self.icon_cb.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.layout.addWidget(self.icon_cb)

        self.layout.addWidget(VerticalLine()) 

        # 3. Namn
        self.name_edit = QLineEdit(f"Spår {index+1}")
        self.name_edit.setFixedWidth(80)
        self.name_edit.setFocusPolicy(Qt.FocusPolicy.ClickFocus) 
        self.layout.addWidget(self.name_edit)

        self.layout.addWidget(VerticalLine()) 

        # 4. Knappar
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(2)
        
        self.btn_rec = QPushButton("R")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setFixedSize(24, 24)
        self.btn_rec.setStyleSheet("""
            QPushButton { background: #333; color: #888; border: 1px solid #444; border-radius: 3px; font-weight: bold; }
            QPushButton:checked { background: #e00; color: white; border: 1px solid #f00; }
        """)
        self.btn_rec.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.btn_mute = QPushButton("M")
        self.btn_mute.setCheckable(True)
        self.btn_mute.setFixedSize(24, 24)
        self.btn_mute.setStyleSheet("""
            QPushButton { background: #333; color: #888; border: 1px solid #444; border-radius: 3px; font-weight: bold; }
            QPushButton:checked { background: #08d; color: white; }
        """)
        self.btn_mute.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.btn_solo = QPushButton("S")
        self.btn_solo.setCheckable(True)
        self.btn_solo.setFixedSize(24, 24)
        self.btn_solo.setStyleSheet("""
            QPushButton { background: #333; color: #888; border: 1px solid #444; border-radius: 3px; font-weight: bold; }
            QPushButton:checked { background: #db0; color: black; }
        """)
        self.btn_solo.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        btn_layout.addWidget(self.btn_rec)
        btn_layout.addWidget(self.btn_mute)
        btn_layout.addWidget(self.btn_solo)
        self.layout.addLayout(btn_layout)

        self.layout.addWidget(VerticalLine()) 

        # 5. Input & Volym
        self.input_ch_spin = QComboBox()
        self.input_ch_spin.setStyleSheet("font-size: 10px; background: #1a1a1a; color: #ccc; border: 1px solid #444; min-width: 40px;")
        self.input_ch_spin.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.layout.addWidget(self.input_ch_spin)

        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(-60, 12)
        self.vol_slider.setValue(0)
        self.vol_slider.setFixedWidth(60)
        self.vol_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #111; border-radius: 2px; }
            QSlider::handle:horizontal { background: #999; width: 12px; margin: -4px 0; border-radius: 6px; }
        """)
        self.vol_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.layout.addWidget(self.vol_slider)

        # 6. Delete
        self.layout.addStretch()
        self.btn_del = QPushButton("🗑")
        self.btn_del.setFlat(True)
        self.btn_del.setFixedWidth(24)
        self.btn_del.setStyleSheet("color: #666; font-size: 14px; border: none;")
        self.btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_del.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.layout.addWidget(self.btn_del)

class TrackRow(QWidget):
    removed = pyqtSignal(object)

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.setFixedHeight(80)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.controls = TrackControls(index)
        self.controls.btn_del.clicked.connect(lambda: self.removed.emit(self))
        self.controls.color_label.colorChanged.connect(self.update_clip_colors)
        
        self.wave = WaveformCanvas()
        self.wave.file_dropped.connect(self.handle_import)
        
        self.layout.addWidget(self.controls)
        self.layout.addWidget(self.wave)

    def update_clip_colors(self, col):
        for c in self.wave.clips: c.color = col
        self.wave.update()
        
    def handle_import(self, path, time_pos):
        try:
            f = sf.SoundFile(path)
            dur = len(f) / f.samplerate
            col = self.controls.color_label._color
            clip = AudioClip(path, time_pos, dur, col)
            clip.load_data()
            self.wave.add_clip(clip)
        except Exception as e:
            QMessageBox.warning(self, "Fel", f"Kunde inte ladda fil: {e}")

    def get_color(self): return self.controls.color_label._color
    def safe_name(self): return "".join(c for c in self.controls.name_edit.text() if c.isalnum() or c in (' ', '_', '-')).strip()

# ===================== Main =====================

class MiniDAW_V3_4(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini DAW v3.4")
        self.resize(1300, 750)
        self.setStyleSheet("background-color: #1e1e1e; color: #ddd;")

        self.samplerate = 44100
        self.global_zoom = 50.0
        self.global_scroll = 0.0
        self.playhead_pos = 0.0
        
        self.is_playing = False
        self.is_recording = False
        self.start_time_sys = 0.0
        self.tracks: List[TrackRow] = []
        self.rec_files = {}
        self.rec_clips = {}

        self._init_ui()
        self._refresh_audio_devices()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._update_playback_ui)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        top_bar = QFrame()
        top_bar.setFixedHeight(50)
        top_bar.setStyleSheet("background-color: #252525; border-bottom: 1px solid #111;")
        top_layout = QHBoxLayout(top_bar)
        
        self.btn_rec = QPushButton("●")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setFixedSize(40,30)
        self.btn_rec.setStyleSheet("QPushButton { font-size: 16px; background: #300; color: #f55; border: 1px solid #500; border-radius: 4px; } QPushButton:checked { background: #f00; color: white; }")
        self.btn_rec.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(40,30)
        self.btn_play.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.btn_stop = QPushButton("■")
        self.btn_stop.setFixedSize(40,30)
        self.btn_stop.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.time_display = QLabel("00:00.00")
        self.time_display.setStyleSheet("font-family: monospace; font-size: 20px; color: #0ff; background: #111; padding: 4px 10px; border-radius: 4px; border: 1px solid #333;")

        self.btn_auto_label = QPushButton("Auto labeling")
        self.btn_auto_label.setFixedHeight(30)
        self.btn_auto_label.setStyleSheet("background: #333; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 0 10px;")
        self.btn_auto_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.combo_device = QComboBox()
        self.combo_device.setMinimumWidth(250)
        self.combo_device.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        top_layout.addWidget(self.btn_rec)
        top_layout.addWidget(self.btn_stop)
        top_layout.addWidget(self.btn_play)
        top_layout.addSpacing(20)
        top_layout.addWidget(self.time_display)
        top_layout.addSpacing(10)
        top_layout.addWidget(self.btn_auto_label)
        top_layout.addStretch()
        top_layout.addWidget(QLabel("Ljudenhet:"))
        top_layout.addWidget(self.combo_device)
        main_layout.addWidget(top_bar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background: #181818; }")
        self.tracks_container = QWidget()
        self.tracks_container.setStyleSheet("background: #1e1e1e;")
        self.tracks_layout = QVBoxLayout(self.tracks_container)
        self.tracks_layout.setSpacing(0) 
        self.tracks_layout.setContentsMargins(0,0,0,0)
        self.tracks_layout.addStretch()
        self.scroll_area.setWidget(self.tracks_container)
        main_layout.addWidget(self.scroll_area)

        bot_bar = QFrame()
        bot_bar.setFixedHeight(40)
        bot_bar.setStyleSheet("background: #252525; border-top: 1px solid #111;")
        bl = QHBoxLayout(bot_bar)
        self.btn_add_track = QPushButton("+ Lägg till spår")
        self.btn_add_track.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        bl.addWidget(self.btn_add_track)
        bl.addStretch()
        main_layout.addWidget(bot_bar)

        self.btn_add_track.clicked.connect(self.add_track)
        self.btn_rec.clicked.connect(self.toggle_record)
        self.btn_play.clicked.connect(self.start_playback)
        self.btn_stop.clicked.connect(self.stop_all)
        self.combo_device.currentIndexChanged.connect(self._device_changed)
        self.btn_auto_label.clicked.connect(self.auto_label_tracks)


        self.add_track()
        self.add_track()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            focus_widget = QApplication.focusWidget()
            if isinstance(focus_widget, QLineEdit):
                super().keyPressEvent(event)
                return
            
            if self.is_playing or self.is_recording:
                self.stop_all()
            else:
                self.start_playback()
        else:
            super().keyPressEvent(event)

    def _refresh_audio_devices(self):
        self.combo_device.blockSignals(True)
        self.combo_device.clear()
        try:
            devs = sd.query_devices()
            apis = sd.query_hostapis()
            for i, d in enumerate(devs):
                if d['max_input_channels'] > 0:
                    api = apis[d['hostapi']]['name']
                    self.combo_device.addItem(f"{d['name']} ({api})", i)
        except: pass
        self.combo_device.blockSignals(False)
        self._device_changed()

    def _device_changed(self):
        idx = self.combo_device.currentData()
        if idx is not None:
            try:
                info = sd.query_devices(idx)
                chans = info['max_input_channels']
                for t in self.tracks: self._populate_track_inputs(t, chans)
            except: pass

    def _populate_track_inputs(self, t: TrackRow, channels: int):
        old_sel = t.controls.input_ch_spin.currentData()
        t.controls.input_ch_spin.clear()
        for c in range(channels): t.controls.input_ch_spin.addItem(f"In {c+1}", c)
        if old_sel is not None and old_sel < channels: t.controls.input_ch_spin.setCurrentIndex(old_sel)

    def add_track(self):
        t = TrackRow(len(self.tracks))
        t.removed.connect(self.remove_track)
        t.wave.set_view_params(self.global_zoom, self.global_scroll)
        t.wave.timeline_scroll_request.connect(self.handle_timeline_scroll)
        t.wave.zoom_request.connect(self.handle_zoom)
        t.wave.cursor_moved.connect(self.set_playhead)
        t.wave.playhead_pos = self.playhead_pos
        
        idx = self.combo_device.currentData()
        ch = 2
        if idx is not None:
             try: ch = sd.query_devices(idx)['max_input_channels']
             except: pass
        self._populate_track_inputs(t, ch)
        self.tracks_layout.insertWidget(self.tracks_layout.count()-1, t)
        self.tracks.append(t)

    def remove_track(self, t):
        if t in self.tracks:
            self.tracks.remove(t)
            t.deleteLater()

    def handle_timeline_scroll(self, d_px):
        d_sec = d_px / self.global_zoom
        self.global_scroll = max(0, self.global_scroll + d_sec)
        self.update_all_views()

    def handle_zoom(self, delta):
        fac = 1.1 if delta > 0 else 0.9
        self.global_zoom = max(5, min(1000, self.global_zoom * fac))
        self.update_all_views()

    def update_all_views(self):
        for t in self.tracks:
            t.wave.set_view_params(self.global_zoom, self.global_scroll)
            t.wave.playhead_pos = self.playhead_pos
            t.wave.update()

    def set_playhead(self, pos):
        self.playhead_pos = pos
        m, s = divmod(pos, 60)
        self.time_display.setText(f"{int(m):02d}:{s:05.2f}")
        self.update_all_views()

    def toggle_record(self):
        if self.is_recording: self.stop_all()
        else: self.start_recording()

    def start_recording(self):
        self.stop_all()
        idx = self.combo_device.currentData()
        if idx is None: return
        active = [t for t in self.tracks if t.controls.btn_rec.isChecked()]
        if not active:
            QMessageBox.warning(self, "Info", "Aktivera 'R' på ett spår.")
            self.btn_rec.setChecked(False)
            return
        try:
            info = sd.query_devices(idx)
            self.samplerate = int(info['default_samplerate'])
            stamp = time.strftime("%H%M%S")
            os.makedirs("inspelningar", exist_ok=True)
            self.rec_files = {}
            self.rec_clips = {}
            self.rec_start_pos = self.playhead_pos
            for t in active:
                path = f"inspelningar/{t.safe_name()}_{stamp}.wav"
                f = sf.SoundFile(path, 'w', self.samplerate, 1)
                self.rec_files[t] = f
                c = AudioClip(path, self.rec_start_pos, 0, t.get_color())
                t.wave.add_clip(c)
                self.rec_clips[t] = c

            def cb(indata, frames, time, status):
                if status: print(status)
                for t, f in self.rec_files.items():
                    ch_data = t.controls.input_ch_spin.currentData()
                    ch = int(ch_data) if ch_data is not None else 0
                    if ch < indata.shape[1]:
                        mono = indata[:, ch]
                        f.write(mono)
                        mn, mx = float(np.min(mono)), float(np.max(mono))
                        self.rec_clips[t].peaks.append((mn, mx))
                        self.rec_clips[t].duration += frames / self.samplerate
            self.stream = sd.InputStream(device=idx, channels=info['max_input_channels'],
                                         samplerate=self.samplerate, callback=cb, blocksize=2048)
            self.stream.start()
            self.is_recording = True
            self.start_time_sys = time.time()
            self.timer.start()
        except Exception as e:
            QMessageBox.critical(self, "Fel", str(e))
            self.stop_all()

    def start_playback(self):
        if self.is_playing or self.is_recording: return
        playing_tracks = []
        solo = any(t.controls.btn_solo.isChecked() for t in self.tracks)
        max_t = 0
        for t in self.tracks:
            on = t.controls.btn_solo.isChecked() if solo else not t.controls.btn_mute.isChecked()
            if on and t.wave.clips:
                playing_tracks.append(t)
                for c in t.wave.clips: max_t = max(max_t, c.start_time + c.duration)
        if not playing_tracks: return
        dur = max_t - self.playhead_pos
        if dur <= 0: self.playhead_pos = 0; dur = max_t
        frames = int(dur * self.samplerate)
        if frames <= 0: return
        mix = np.zeros((frames, 2), dtype=np.float32)
        start_f = int(self.playhead_pos * self.samplerate)
        for t in playing_tracks:
            buf = np.zeros((frames, 2), dtype=np.float32)
            vol = 10 ** (t.controls.vol_slider.value()/20)
            for c in t.wave.clips:
                c_data = c.get_audio_segment(start_f, frames, self.samplerate)
                buf += c_data
            buf *= vol
            mix += buf
        peak = np.max(np.abs(mix))
        if peak > 1.0: mix /= peak
        self.is_playing = True
        self.start_time_sys = time.time()
        def cb(out, f, t, s):
            nonlocal mix
            if len(mix) < f:
                out[:len(mix)] = mix
                out[len(mix):] = 0
                raise sd.CallbackStop
            out[:] = mix[:f]
            mix = mix[f:]
        try:
            # Tvinga utströmmen att vara stereo (2 kanaler)
            self.stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=2,
                callback=cb
            )
            self.stream.start()
            self.timer.start()
        except Exception as e:
            QMessageBox.critical(self, "Fel", str(e))
            self.stop_all()

    def stop_all(self):
        if hasattr(self, 'stream') and self.stream: self.stream.stop(); self.stream.close()
        if self.is_recording:
            for f in self.rec_files.values(): f.close()
            for c in self.rec_clips.values(): c.load_data(); c._generate_peaks()
        self.is_recording = False
        self.is_playing = False
        self.btn_rec.setChecked(False)
        self.timer.stop()
        self.update_all_views()

    def _update_playback_ui(self):
        elapsed = time.time() - self.start_time_sys
        if self.is_recording:
            self.set_playhead(self.rec_start_pos + elapsed)
            for t in self.tracks:
                if t.controls.btn_rec.isChecked(): t.wave.update()
        elif self.is_playing:
            if not self.stream.active: self.stop_all()
            else:
                self.set_playhead(self.playhead_pos + elapsed)
                self.start_time_sys = time.time()

    def auto_label_tracks(self):
        """
        Kör AI på varje spår med minst ett klipp:
        - gissar instrument
        - döper om spåret
        - väljer ikon
        - sätter färg
        """
        if not self.tracks:
            QMessageBox.information(self, "Auto labeling", "Det finns inga spår att labela.")
            return

        for track in self.tracks:
            # hoppa över tomma spår
            if not track.wave.clips:
                continue

            first_clip = track.wave.clips[0]
            audio_path = first_clip.file_path

            if not os.path.exists(audio_path):
                continue

            try:
                instrument, _ = predict_instrument(
                    audio_path,
                    model_name="random_forest",
                    show_probabilities=False
                )
            except Exception as e:
                QMessageBox.warning(self, "Auto labeling",
                                    f"Kunde inte analysera {audio_path}:\n{e}")
                continue

            if not instrument:
                continue

            # 1) Sätt spårnamn (gör text lite snyggare)
            display_name = instrument.replace("_", " ").title()
            track.controls.name_edit.setText(display_name)

            # 2) Hämta stil (ikon + färg)
            icon, color = INSTRUMENT_STYLES.get(instrument, ("🔊", track.get_color()))

            # 3) Sätt ikon i comboboxen
            idx = track.controls.icon_cb.findText(icon)
            if idx >= 0:
                track.controls.icon_cb.setCurrentIndex(idx)

            # 4) Uppdatera färg på label + alla klipp
            track.controls.color_label._color = color
            track.controls.color_label._update_style()
            track.update_clip_colors(color)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MiniDAW_V3_4()
    w.show()
    sys.exit(app.exec())