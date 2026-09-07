# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""PyQt5 MVP for pixel-domain crack segmentation and V2 quantification."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
    from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QSplitter,
        QStatusBar,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit("PyQt5 is required. Install it with: conda install -n cubit_yolo pyqt=5") from exc
from ultralytics import YOLO

import quantify_cracks_test_all as v2


MODEL_PATH = Path("D:/CrackSegRuns/pruned_student_10_kd_30e/weights/best.pt")
IMGSZ = 640
DEVICE = 0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TABLE_HEADERS = (
    "区域",
    "置信度",
    "支持实例",
    "长度(px)",
    "平均宽度(px)",
    "最大宽度(px)",
    "面积(px²)",
)


def read_image(path: Path) -> np.ndarray | None:
    """Read an image through NumPy so non-ASCII Windows paths are supported."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def write_image(path: Path, image: np.ndarray) -> bool:
    """Write an image through NumPy so non-ASCII Windows paths are supported."""
    suffix = path.suffix.lower() if path.suffix.lower() in IMAGE_EXTENSIONS else ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
    except OSError:
        return False
    return True


class ResultTable(QTableWidget):
    """Table with a clean centered message when no measurements exist."""

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.rowCount() != 0:
            return
        painter = QPainter(self.viewport())
        painter.setPen(QColor("#7f8996"))
        painter.drawText(self.viewport().rect(), Qt.AlignCenter, "暂无检测结果")
        painter.end()


def create_table() -> ResultTable:
    table = ResultTable(0, len(TABLE_HEADERS))
    table.setHorizontalHeaderLabels(TABLE_HEADERS)
    table.setAlternatingRowColors(True)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectRows)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    table.horizontalHeader().setMinimumHeight(34)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(28)
    table.setMinimumHeight(120)
    table.setMaximumHeight(280)
    return table


class InferenceWorker(QObject):
    """Own the YOLO model and run all expensive work outside the GUI thread."""

    completed = pyqtSignal(object)
    failed = pyqtSignal(str, str)
    model_ready = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        self.model = YOLO(str(MODEL_PATH), task="segment")
        if getattr(self.model.model, "task", None) is None:
            self.model.model.task = "segment"
        self.model_ready.emit(str(MODEL_PATH))

    @pyqtSlot(object)
    def infer(self, job):
        kind = job["kind"]
        started = time.perf_counter()
        try:
            self._load_model()
            measurements, visualization = self._process(job)
            self.completed.emit(
                {
                    "kind": kind,
                    "token": job["token"],
                    "measurements": measurements,
                    "visualization": visualization,
                    "elapsed": time.perf_counter() - started,
                    "quantified": job["quantify"],
                    "source_name": job["source_name"],
                }
            )
        except Exception as exc:  # Surface worker errors in the UI instead of terminating the thread.
            self.failed.emit(kind, f"{type(exc).__name__}: {exc}")

    def _process(self, job):
        image = job["image"]
        height, width = image.shape[:2]
        results = self.model.predict(
            source=image,
            imgsz=IMGSZ,
            conf=job["conf"],
            device=DEVICE,
            retina_masks=True,
            save=False,
            verbose=False,
        )
        result = results[0] if results else None
        visualization = image.copy()
        if result is None or result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return ([] if job["quantify"] else None), visualization

        raw_masks = result.masks.data.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        raw_count = min(len(raw_masks), len(confidences))
        instance_masks = [
            v2.resize_mask_to_original(raw_masks[index] > 0.5, height, width) for index in range(raw_count)
        ]
        valid_confidences = [float(confidences[index]) for index in range(raw_count)]
        merged_mask = v2.merge_instance_masks(instance_masks)
        if merged_mask is None:
            return ([] if job["quantify"] else None), visualization

        regions, _ = v2.extract_connected_crack_regions(merged_mask, job["min_area"])
        measurements = [] if job["quantify"] else None
        for region_index, region in enumerate(regions, start=1):
            region_mask = region["mask"]
            color = v2.COLORS[(region_index - 1) % len(v2.COLORS)]
            if job["show_mask"]:
                v2.overlay_mask(visualization, region_mask, color)
            if job["show_box"]:
                x1, y1, x2, y2 = region["bbox"]
                cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)

            if job["quantify"]:
                geometry = v2.measure_connected_region(region_mask)
                supporting_instances, max_confidence = v2.find_supporting_instances(
                    region_mask, instance_masks, valid_confidences
                )
                if job["show_skeleton"]:
                    visualization[geometry["skeleton"]] = (255, 255, 255)
                measurements.append(
                    {
                        "image": job["source_name"],
                        "connected_crack_region_id": region_index,
                        "max_confidence": max_confidence,
                        "supporting_instances": supporting_instances,
                        "length_px": geometry["length_px"],
                        "mean_width_px": geometry["mean_width_px"],
                        "max_width_px": geometry["max_width_px"],
                        "area_px2": geometry["area_px2"],
                    }
                )
            cv2.putText(
                visualization,
                f"R{region_index}",
                (region["bbox"][0], max(18, region["bbox"][1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return measurements, visualization


class ImageLabel(QLabel):
    """Aspect-ratio-preserving image surface."""

    def __init__(self, placeholder):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 210)
        self.setStyleSheet("background:#15191f; border:1px solid #343b45; color:#7f8996;")
        self._pixmap = None

    def set_cv_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self._refresh()

    def _refresh(self):
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


class CrackGui(QMainWindow):
    submit_job = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("裂缝分割与量化（Crack Segmentation & Quantification）")
        available = QApplication.primaryScreen().availableGeometry()
        minimum_width = min(900, available.width())
        minimum_height = min(620, available.height())
        target_width = max(minimum_width, min(1280, int(available.width() * 0.88)))
        target_height = max(minimum_height, min(700, int(available.height() * 0.8)))
        self.resize(target_width, target_height)
        self.setMinimumSize(minimum_width, minimum_height)

        self.image_paths = []
        self.image_index = -1
        self.current_image = None
        self.current_result = None
        self.current_measurements = []
        self.image_results = {}
        self.inference_busy = False
        self.camera = None
        self.camera_running = False
        self.camera_paused = False
        self.camera_token = 0
        self.camera_result = None
        self.camera_measurements = []
        self.camera_submit_count = 0
        self.fps_average = 0.0

        self._build_ui()
        self._apply_style()

        self.worker_thread = QThread(self)
        self.worker = InferenceWorker()
        self.worker.moveToThread(self.worker_thread)
        self.submit_job.connect(self.worker.infer)
        self.worker.completed.connect(self._inference_completed)
        self.worker.failed.connect(self._inference_failed)
        self.worker.model_ready.connect(lambda _: self.statusBar().showMessage("模型已就绪", 3000))
        self.worker_thread.start()

        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self._read_camera)

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 10, 12, 8)
        root_layout.setSpacing(8)

        header = QWidget()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 8, 14, 8)
        title = QLabel("裂缝分割与量化系统")
        title.setObjectName("title")
        header_layout.addWidget(title)
        header_layout.addStretch()
        device_text = "CUDA" if torch.cuda.is_available() else "CPU"
        for text in ("KD Student", device_text, "3.743M Params", "5.823G MACs"):
            badge = QLabel(text)
            badge.setObjectName("badge")
            header_layout.addWidget(badge)
        root_layout.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_image_tab(), "图像模式")
        self.tabs.addTab(self._build_camera_tab(), "摄像头模式")
        root_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().showMessage("就绪")

    def _parameter_box(self):
        box = QGroupBox("参数设置")
        box.setMinimumHeight(160)
        box.setMaximumHeight(180)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 12, 10, 8)
        layout.setSpacing(4)
        form = QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        form.setColumnStretch(1, 1)
        confidence = QDoubleSpinBox()
        confidence.setRange(0.01, 1.0)
        confidence.setSingleStep(0.05)
        confidence.setValue(0.25)
        confidence.setDecimals(2)
        min_area = QSpinBox()
        min_area.setRange(1, 1_000_000)
        min_area.setValue(20)
        show_mask = QCheckBox("Mask 掩膜")
        show_mask.setChecked(True)
        show_skeleton = QCheckBox("Skeleton 骨架")
        show_skeleton.setChecked(True)
        show_box = QCheckBox("Box 边界框")
        show_box.setChecked(True)
        form.addWidget(QLabel("Confidence"), 0, 0)
        form.addWidget(confidence, 0, 1)
        form.addWidget(QLabel("Min Area"), 1, 0)
        form.addWidget(min_area, 1, 1)
        form.setRowMinimumHeight(0, 27)
        form.setRowMinimumHeight(1, 27)
        layout.addLayout(form)
        layout.addWidget(QLabel("显示图层"))
        toggles = QGridLayout()
        toggles.setVerticalSpacing(3)
        toggles.addWidget(show_mask, 0, 0)
        toggles.addWidget(show_skeleton, 1, 0)
        toggles.addWidget(show_box, 2, 0)
        toggles.setColumnStretch(0, 1)
        layout.addLayout(toggles)
        return box, (confidence, min_area, show_mask, show_skeleton, show_box)

    def _build_image_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)

        content_widget = QWidget()
        content = QHBoxLayout(content_widget)
        content.setContentsMargins(0, 0, 0, 0)
        controls = QGroupBox("控制面板")
        controls.setObjectName("controlPanel")
        controls.setFixedWidth(240)
        control_layout = QVBoxLayout(controls)
        self.open_image_button = QPushButton("打开图像")
        self.open_folder_button = QPushButton("打开文件夹")
        self.previous_button = QPushButton("上一张")
        self.next_button = QPushButton("下一张")
        self.detect_button = QPushButton("检测并量化")
        self.detect_button.setObjectName("primary")
        self.save_result_button = QPushButton("保存结果")
        self.export_csv_button = QPushButton("导出 CSV")
        for button in (
            self.previous_button,
            self.next_button,
            self.save_result_button,
            self.export_csv_button,
        ):
            button.setObjectName("secondary")
        control_layout.addWidget(self.open_image_button)
        control_layout.addWidget(self.open_folder_button)
        params, self.image_params = self._parameter_box()
        control_layout.addWidget(params)
        control_layout.addStretch()
        control_layout.addWidget(self.detect_button)
        content.addWidget(controls)

        splitter = QSplitter(Qt.Horizontal)
        self.original_box = QGroupBox("原始图像")
        self.original_box.setObjectName("imageCard")
        original_layout = QVBoxLayout(self.original_box)
        self.original_view = ImageLabel("请打开图像或文件夹")
        original_layout.addWidget(self.original_view)
        self.result_box = QGroupBox("检测与量化结果")
        self.result_box.setObjectName("imageCard")
        result_layout = QVBoxLayout(self.result_box)
        self.result_view = ImageLabel("等待检测")
        result_layout.addWidget(self.result_view)
        splitter.addWidget(self.original_box)
        splitter.addWidget(self.result_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        content.addWidget(splitter, 1)

        result_group = QGroupBox("量化结果")
        result_group_layout = QVBoxLayout(result_group)
        self.image_table = create_table()
        result_group_layout.addWidget(self.image_table)
        image_vertical_splitter = QSplitter(Qt.Vertical)
        image_vertical_splitter.setChildrenCollapsible(False)
        image_vertical_splitter.addWidget(content_widget)
        image_vertical_splitter.addWidget(result_group)
        image_vertical_splitter.setStretchFactor(0, 3)
        image_vertical_splitter.setStretchFactor(1, 2)
        image_vertical_splitter.setSizes([320, 250])
        layout.addWidget(image_vertical_splitter, 1)

        bottom = QHBoxLayout()
        bottom.addWidget(self.previous_button)
        bottom.addWidget(self.next_button)
        bottom.addStretch()
        self.image_position = QLabel("尚未选择图像")
        self.image_position.setObjectName("position")
        bottom.addWidget(self.image_position)
        bottom.addStretch()
        bottom.addWidget(self.save_result_button)
        bottom.addWidget(self.export_csv_button)
        layout.addLayout(bottom)

        self.open_image_button.clicked.connect(self._open_image)
        self.open_folder_button.clicked.connect(self._open_folder)
        self.previous_button.clicked.connect(lambda: self._navigate_image(-1))
        self.next_button.clicked.connect(lambda: self._navigate_image(1))
        self.detect_button.clicked.connect(self._detect_image)
        self.save_result_button.clicked.connect(self._save_result)
        self.export_csv_button.clicked.connect(self._export_csv)
        return tab

    def _build_camera_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)
        content_widget = QWidget()
        content = QHBoxLayout(content_widget)
        content.setContentsMargins(0, 0, 0, 0)

        controls = QGroupBox("控制面板")
        controls.setObjectName("controlPanel")
        controls.setFixedWidth(240)
        control_layout = QVBoxLayout(controls)
        camera_index_row = QHBoxLayout()
        camera_index_row.addWidget(QLabel("摄像头"))
        self.camera_index = QComboBox()
        self.camera_index.setEditable(True)
        self.camera_index.addItems([str(index) for index in range(4)])
        self.camera_index.setCurrentText("0")
        camera_index_row.addWidget(self.camera_index, 1)
        control_layout.addLayout(camera_index_row)
        self.start_camera_button = QPushButton("启动")
        self.start_camera_button.setObjectName("primary")
        self.stop_camera_button = QPushButton("停止")
        self.pause_camera_button = QPushButton("暂停")
        self.save_frame_button = QPushButton("保存当前帧")
        for button in (self.stop_camera_button, self.pause_camera_button, self.save_frame_button):
            button.setObjectName("secondary")
        camera_actions = QHBoxLayout()
        camera_actions.setSpacing(5)
        camera_actions.addWidget(self.start_camera_button)
        camera_actions.addWidget(self.stop_camera_button)
        camera_actions.addWidget(self.pause_camera_button)
        control_layout.addLayout(camera_actions)
        params, self.camera_params = self._parameter_box()
        control_layout.addWidget(params)
        note = QLabel("仅进行像素域量化\n不代表真实毫米尺寸")
        note.setObjectName("note")
        control_layout.addWidget(note)
        control_layout.addStretch()
        control_layout.addWidget(self.save_frame_button)
        content.addWidget(controls)

        camera_group = QGroupBox("实时检测结果")
        camera_group.setObjectName("imageCard")
        camera_layout = QVBoxLayout(camera_group)
        camera_header = QHBoxLayout()
        camera_header.addStretch()
        self.fps_label = QLabel("FPS  0.0")
        self.fps_label.setObjectName("badge")
        camera_header.addWidget(self.fps_label)
        camera_layout.addLayout(camera_header)
        self.camera_view = ImageLabel("摄像头未启动")
        camera_layout.addWidget(self.camera_view, 1)
        content.addWidget(camera_group, 1)

        camera_result_group = QGroupBox("实时量化结果")
        camera_result_layout = QVBoxLayout(camera_result_group)
        self.camera_table = create_table()
        camera_result_layout.addWidget(self.camera_table)
        camera_vertical_splitter = QSplitter(Qt.Vertical)
        camera_vertical_splitter.setChildrenCollapsible(False)
        camera_vertical_splitter.addWidget(content_widget)
        camera_vertical_splitter.addWidget(camera_result_group)
        camera_vertical_splitter.setStretchFactor(0, 3)
        camera_vertical_splitter.setStretchFactor(1, 2)
        camera_vertical_splitter.setSizes([320, 250])
        layout.addWidget(camera_vertical_splitter, 1)

        self.start_camera_button.clicked.connect(self._start_camera)
        self.stop_camera_button.clicked.connect(self._stop_camera)
        self.pause_camera_button.clicked.connect(self._toggle_pause)
        self.save_frame_button.clicked.connect(self._save_frame)
        return tab

    def _job_options(self, widgets):
        confidence, min_area, show_mask, show_skeleton, show_box = widgets
        return {
            "conf": confidence.value(),
            "min_area": min_area.value(),
            "show_mask": show_mask.isChecked(),
            "show_skeleton": show_skeleton.isChecked(),
            "show_box": show_box.isChecked(),
        }

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)"
        )
        if path:
            self.image_paths = [Path(path)]
            self.image_index = 0
            self._load_selected_image()

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "打开图像文件夹")
        if not folder:
            return
        root = Path(folder)
        self.image_paths = sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: str(path).lower(),
        )
        if not self.image_paths:
            QMessageBox.information(self, "未找到图像", "该文件夹中没有支持的图像文件。")
            return
        self.image_index = 0
        self._load_selected_image()

    def _navigate_image(self, offset):
        if not self.image_paths:
            return
        self.image_index = (self.image_index + offset) % len(self.image_paths)
        self._load_selected_image()

    def _load_selected_image(self):
        path = self.image_paths[self.image_index]
        image = read_image(path)
        if image is None:
            QMessageBox.warning(self, "读取失败", f"无法读取图像：\n{path}")
            return
        self.current_image = image
        self.current_result = None
        self.current_measurements = []
        self.original_view.set_cv_image(image)
        self.result_view._pixmap = None
        self.result_view.clear()
        self.result_view.setText("等待检测")
        self.original_box.setTitle(f"原始图像 · {path.name}")
        self.result_box.setTitle("检测与量化结果 · 等待检测")
        self._fill_table(self.image_table, [])
        self.image_position.setText(f"{self.image_index + 1} / {len(self.image_paths)}    {path.name}")
        self.statusBar().showMessage(f"已加载：{path.name}")

    def _detect_image(self):
        if self.current_image is None or self.inference_busy:
            return
        path = self.image_paths[self.image_index]
        job = {
            "kind": "image",
            "token": str(path),
            "source_name": path.name,
            "image": self.current_image.copy(),
            "quantify": True,
            **self._job_options(self.image_params),
        }
        self._submit(job)
        self.detect_button.setText("正在检测…")

    def _submit(self, job):
        self.inference_busy = True
        self.submit_job.emit(job)

    @pyqtSlot(object)
    def _inference_completed(self, result):
        self.inference_busy = False
        if result["kind"] == "image":
            self.detect_button.setText("检测并量化")
            if not self.image_paths or result["token"] != str(self.image_paths[self.image_index]):
                return
            self.current_result = result["visualization"]
            self.current_measurements = result["measurements"] or []
            self.result_view.set_cv_image(self.current_result)
            self._fill_table(self.image_table, self.current_measurements)
            self.image_results[result["token"]] = list(self.current_measurements)
            count = len(self.current_measurements)
            self.result_box.setTitle(f"检测与量化结果 · {count} 个区域")
            self.statusBar().showMessage(f"检测完成：{count} 个区域，用时 {result['elapsed']:.3f} 秒")
            return

        if not self.camera_running or result["token"] != self.camera_token:
            return
        self.camera_result = result["visualization"]
        self.camera_view.set_cv_image(self.camera_result)
        instant_fps = 1.0 / max(result["elapsed"], 1e-6)
        self.fps_average = instant_fps if self.fps_average == 0 else 0.8 * self.fps_average + 0.2 * instant_fps
        self.fps_label.setText(f"FPS  {self.fps_average:.1f}")
        if result["quantified"]:
            self.camera_measurements = result["measurements"] or []
            self._fill_table(self.camera_table, self.camera_measurements)

    @pyqtSlot(str, str)
    def _inference_failed(self, kind, message):
        self.inference_busy = False
        self.detect_button.setText("检测并量化")
        self.statusBar().showMessage(message)
        if kind == "camera":
            self._stop_camera()
        QMessageBox.critical(self, "Inference failed", message)

    def _fill_table(self, table, measurements):
        table.setRowCount(len(measurements))
        for row, item in enumerate(measurements):
            values = (
                item["connected_crack_region_id"],
                f"{item['max_confidence']:.3f}",
                item["supporting_instances"],
                f"{item['length_px']:.2f}",
                f"{item['mean_width_px']:.2f}",
                f"{item['max_width_px']:.2f}",
                item["area_px2"],
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                cell.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, column, cell)

    def _save_result(self):
        if self.current_result is None:
            QMessageBox.information(self, "没有结果", "请先运行检测并量化。")
            return
        default_name = f"{self.image_paths[self.image_index].stem}_quantified.png"
        path, _ = QFileDialog.getSaveFileName(self, "保存结果", default_name, "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if path and not write_image(Path(path), self.current_result):
            QMessageBox.warning(self, "保存失败", f"无法保存：\n{path}")

    def _export_csv(self):
        if not self.image_results:
            QMessageBox.information(self, "没有数据", "请至少检测一张图像。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", "crack_measurements.csv", "CSV (*.csv)")
        if not path:
            return
        fields = (
            "image",
            "connected_crack_region_id",
            "max_confidence",
            "supporting_instances",
            "length_px",
            "mean_width_px",
            "max_width_px",
            "area_px2",
        )
        try:
            with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                for measurements in self.image_results.values():
                    for item in measurements:
                        writer.writerow({field: item[field] for field in fields})
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        self.statusBar().showMessage(f"CSV 已导出：{Path(path).name}")

    def _start_camera(self):
        if self.camera_running:
            return
        try:
            index = int(self.camera_index.currentText())
        except ValueError:
            QMessageBox.warning(self, "摄像头编号无效", "摄像头编号必须是整数。")
            return
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(index)
        if not camera.isOpened():
            camera.release()
            QMessageBox.warning(self, "摄像头不可用", f"无法打开摄像头 {index}。")
            return
        self.camera = camera
        self.camera_running = True
        self.camera_paused = False
        self.camera_token += 1
        self.camera_submit_count = 0
        self.fps_average = 0.0
        self.camera_result = None
        self.camera_measurements = []
        self._fill_table(self.camera_table, [])
        self.pause_camera_button.setText("暂停")
        self.camera_timer.start(30)
        self.statusBar().showMessage(f"摄像头 {index} 已启动；每 4 个推理帧更新一次量化结果")

    def _read_camera(self):
        if not self.camera_running or self.camera_paused or self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            self._stop_camera()
            QMessageBox.warning(self, "摄像头读取失败", "摄像头已停止返回画面。")
            return
        if self.inference_busy:
            return
        self.camera_submit_count += 1
        quantify = self.camera_submit_count % 4 == 1
        job = {
            "kind": "camera",
            "token": self.camera_token,
            "source_name": f"camera_{self.camera_index.currentText()}",
            "image": frame,
            "quantify": quantify,
            **self._job_options(self.camera_params),
        }
        self._submit(job)

    def _toggle_pause(self):
        if not self.camera_running:
            return
        self.camera_paused = not self.camera_paused
        self.pause_camera_button.setText("继续" if self.camera_paused else "暂停")
        self.statusBar().showMessage("摄像头已暂停" if self.camera_paused else "摄像头已继续")

    def _stop_camera(self):
        self.camera_timer.stop()
        self.camera_running = False
        self.camera_paused = False
        self.camera_token += 1
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.pause_camera_button.setText("暂停")
        self.fps_label.setText("FPS  0.0")
        self.statusBar().showMessage("摄像头已停止")

    def _save_frame(self):
        if self.camera_result is None:
            QMessageBox.information(self, "没有可保存画面", "请启动摄像头并等待处理后的画面。")
            return
        default_name = time.strftime("crack_camera_%Y%m%d_%H%M%S.png")
        path, _ = QFileDialog.getSaveFileName(self, "保存当前帧", default_name, "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if path and not write_image(Path(path), self.camera_result):
            QMessageBox.warning(self, "保存失败", f"无法保存：\n{path}")

    def closeEvent(self, event):
        self._stop_camera()
        self.submit_job.disconnect(self.worker.infer)
        self.worker_thread.quit()
        if not self.worker_thread.wait(5000):
            QMessageBox.warning(self, "Closing", "Inference is still finishing; the window will close when it returns.")
            self.worker_thread.wait()
        event.accept()

    def _apply_style(self):
        self.setStyleSheet(
            """
            QWidget { background:#20252c; color:#e8ecf1; font-family:"Microsoft YaHei UI"; font-size:13px; }
            QWidget#header { background:#171b20; border:1px solid #343b45; border-radius:7px; }
            QLabel#title { font-size:18px; font-weight:700; color:#ffffff; }
            QLabel#badge { background:#2d3540; color:#cbd5e1; border:1px solid #46515f; border-radius:9px; padding:3px 8px; font-size:11px; font-weight:600; }
            QLabel#position { color:#aeb8c5; }
            QGroupBox { border:1px solid #343b45; border-radius:6px; margin-top:8px; padding-top:8px; font-size:13px; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
            QGroupBox#controlPanel { background:#252b33; border-color:#3a444f; }
            QGroupBox#imageCard { background:#252b33; border-color:#46515f; }
            QPushButton { background:#343c47; border:1px solid #46515f; border-radius:5px; padding:5px 9px; min-height:18px; font-size:13px; font-weight:500; }
            QPushButton:hover { background:#414b58; }
            QPushButton#primary { background:#1473e6; border-color:#3089ef; min-height:27px; font-weight:700; }
            QPushButton#secondary { background:#2b323b; padding:4px 8px; min-height:16px; font-size:12px; }
            QSpinBox, QDoubleSpinBox, QComboBox { background:#171b20; border:1px solid #46505d; padding:3px; min-height:17px; }
            QTableWidget { background:#171b20; alternate-background-color:#1d2229; gridline-color:#343b45; font-size:12px; }
            QHeaderView::section { background:#303741; padding:5px; border:0; border-right:1px solid #454e5a; font-size:12px; font-weight:600; }
            QTabWidget::pane { border:0; }
            QTabBar::tab { background:#2c333c; padding:7px 18px; margin-right:2px; font-size:13px; font-weight:600; }
            QTabBar::tab:selected { background:#1473e6; }
            QSplitter::handle:vertical { background:#343b45; height:5px; }
            QLabel#note { color:#f2c66d; padding:2px; font-size:11px; }
            QStatusBar { background:#171b20; color:#8994a3; font-size:12px; }
            """
        )


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Crack Segmentation & Quantification")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = CrackGui()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
