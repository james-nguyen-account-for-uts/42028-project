from __future__ import annotations

import platform
import tkinter.font as tkfont
from typing import Optional

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk

from Camera import CameraController, Prediction
from model import ModelInfo, discover_models, load_classes, load_model

import config


class GestureDetectStudio(tk.Tk):

  def __init__(self):
    super().__init__()
    self.title("GestureDetect Studio")
    self.geometry("1180x760")
    self.minsize(1040, 760)

    self.device = config.DEVICE
    self.available_models = discover_models()
    self.model_by_name = {
      info.display_name: info for info in self.available_models
    }
    self.active_model_info: Optional[ModelInfo] = (
      self.available_models[0] if self.available_models else None)
    self.model_var = tk.StringVar(
      value=(self.active_model_info.display_name
             if self.active_model_info else "No trained model"))
    self.classes = load_classes(self.active_model_info)
    self.model = load_model(
      self.classes,
      self.device,
      self.active_model_info)
    self.camera: Optional[CameraController] = None

    self.running = False
    self.words: list[str] = []
    self.video_image = None

    self._configure_style()
    self._build_ui()
    self.protocol("WM_DELETE_WINDOW", self.on_close)

  def _configure_style(self) -> None:
    self.configure(bg="#F4F6F8")
    style = ttk.Style(self)
    if platform.system() == "Darwin":
      style.theme_use("aqua")
    else:
      style.theme_use("clam")
    style.configure("TFrame", background="#F4F6F8")
    style.configure("Panel.TFrame", background="#FFFFFF")
    style.configure("Title.TLabel", background="#F4F6F8",
                    foreground="#111827", font=("Segoe UI", 22, "bold"))
    style.configure("Muted.TLabel", background="#F4F6F8",
                    foreground="#6B7280", font=("Segoe UI", 10))
    style.configure("PanelTitle.TLabel", background="#FFFFFF",
                    foreground="#111827", font=("Segoe UI", 15, "bold"))
    style.configure("PanelText.TLabel", background="#FFFFFF",
                    foreground="#374151", font=("Segoe UI", 10))
    style.configure("Value.TLabel", background="#FFFFFF",
                    foreground="#047857", font=("Segoe UI", 24, "bold"))
    style.configure("Blue.Horizontal.TProgressbar",
                    troughcolor="#E5E7EB", background="#2563EB")
    style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))
    style.configure("Secondary.TButton", font=("Segoe UI", 10))

  def _build_ui(self) -> None:
    root = ttk.Frame(self, padding=24)
    root.pack(fill=tk.BOTH, expand=True)

    header = ttk.Frame(root)
    header.pack(fill=tk.X)
    ttk.Label(header, text="GestureDetect Studio",
              style="Title.TLabel").pack(side=tk.LEFT)

    header_right = ttk.Frame(header)
    header_right.pack(side=tk.RIGHT, fill=tk.X)
    selector_row = ttk.Frame(header_right)
    selector_row.pack(anchor=tk.E)
    ttk.Label(
      selector_row,
      text="Model",
      style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))
    self.model_selector = ttk.Combobox(
      selector_row,
      textvariable=self.model_var,
      values=[info.display_name for info in self.available_models],
      state=("readonly" if self.available_models else "disabled"),
      width=22)
    self.model_selector.pack(side=tk.LEFT)
    self.model_selector.bind("<<ComboboxSelected>>", self.on_model_selected)

    self.status_label = ttk.Label(
      header_right,
      text=f"Idle | device: {self.device}",
      style="Muted.TLabel")
    self.status_label.pack(anchor=tk.E, pady=(4, 0))

    content = ttk.Frame(root)
    content.pack(fill=tk.BOTH, expand=True, pady=(18, 0))
    content.columnconfigure(0, weight=7, minsize=640)
    content.columnconfigure(1, weight=3, minsize=320)
    content.rowconfigure(0, weight=1)

    left = ttk.Frame(content)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
    left.columnconfigure(0, weight=1)
    left.rowconfigure(0, weight=3)
    left.rowconfigure(1, weight=2)

    self.video_canvas = tk.Canvas(
      left,
      bg="#111827",
      highlightthickness=0,
      width=760,
      height=420)
    self.video_canvas.grid(row=0, column=0, sticky="nsew")
    self.video_canvas.bind("<Configure>", self._on_video_resize)
    self.video_placeholder = self.video_canvas.create_text(
      380,
      210,
      text="Webcam Feed",
      fill="#F9FAFB",
      font=("Segoe UI", 18, "bold"))

    subtitle_panel = tk.Frame(
      left,
      bg="#FFFFFF",
      highlightbackground="#E5E7EB",
      highlightthickness=1)
    subtitle_panel.grid(row=1, column=0, sticky="nsew", pady=(16, 0))
    subtitle_panel.columnconfigure(0, weight=1)
    subtitle_panel.rowconfigure(1, weight=1)
    tk.Label(
      subtitle_panel,
      text="Subtitle output",
      bg="#FFFFFF",
      fg="#374151",
      font=("Segoe UI", 10),
      anchor=tk.W).grid(row=0, column=0, sticky="ew", padx=18, pady=(12, 0))
    self.subtitle_text = tk.Text(
      subtitle_panel,
      height=4,
      wrap=tk.WORD,
      borderwidth=0,
      bg="#FFFFFF",
      fg="#111827",
      font=("Segoe UI", 18, "bold"),
      padx=18,
      pady=8)
    self.subtitle_text.grid(row=1, column=0, sticky="nsew")
    self.subtitle_text.configure(state=tk.DISABLED)

    side = tk.Frame(
      content,
      bg="#FFFFFF",
      width=360,
      highlightbackground="#E5E7EB",
      highlightthickness=1)
    side.grid(row=0, column=1, sticky="nsew")
    side.grid_propagate(False)

    side_inner = tk.Frame(side, bg="#FFFFFF")
    side_inner.pack(fill=tk.BOTH, expand=True, padx=24, pady=20)
    side_inner.columnconfigure(0, weight=1)
    side_inner.bind("<Configure>", self._on_side_resize)

    self._make_side_label(
      side_inner, "Recognition", 0, 18, "bold", "#111827", pady=(2, 14))
    self._make_side_label(
      side_inner, "Current prediction", 1, 10, "normal", "#374151",
      pady=(0, 4))
    self.current_word_label = self._make_side_label(
      side_inner, "--", 2, 24, "bold", "#047857", pady=(4, 14), height=32)

    self._make_side_label(
      side_inner, "Current confidence", 3, 10, "normal", "#374151",
      pady=(0, 4))
    self.confidence_value = self._make_side_label(
      side_inner, "0%", 4, 10, "normal", "#374151", pady=(2, 4))
    self.confidence_bar = ttk.Progressbar(
      side_inner, style="Blue.Horizontal.TProgressbar", mode="determinate",
      maximum=100)
    self.confidence_bar.grid(row=5, column=0, sticky="ew", pady=(0, 16))

    self._make_side_label(
      side_inner, "Last output", 6, 10, "normal", "#374151", pady=(0, 4))
    self.last_output_label = self._make_side_label(
      side_inner,
      "No words yet",
      7,
      10,
      "normal",
      "#374151",
      pady=(4, 16),
      height=36,
      justify=tk.LEFT)

    ttk.Separator(side_inner).grid(row=8, column=0, sticky="ew", pady=(0, 16))
    self.word_count_label = self._make_side_label(
      side_inner, "Words: 0", 9, 10, "normal", "#374151", pady=(0, 10))
    self.model_status_label = self._make_side_label(
      side_inner,
      self._model_status_text(),
      10,
      10,
      "normal",
      "#374151",
      pady=(0, 0),
      height=34,
      justify=tk.LEFT)

    controls = tk.Frame(side_inner, bg="#FFFFFF")
    controls.grid(row=11, column=0, sticky="ew", pady=(16, 0))
    controls.columnconfigure(0, weight=1)
    self.start_button = ttk.Button(
      controls, text="Start camera", style="Primary.TButton",
      command=self.start_camera)
    self.start_button.grid(row=0, column=0, sticky="ew", pady=(0, 8), ipady=3)
    self.stop_button = ttk.Button(
      controls, text="Stop camera", style="Secondary.TButton",
      command=self.stop_camera, state=tk.DISABLED)
    self.stop_button.grid(row=1, column=0, sticky="ew", pady=(0, 8), ipady=3)
    self.clear_button = ttk.Button(
      controls, text="Clear subtitles", style="Secondary.TButton",
      command=self.clear_subtitles)
    self.clear_button.grid(row=2, column=0, sticky="ew", ipady=3)

  def _make_side_label(
      self,
      parent,
      text: str,
      row: int,
      size: int,
      weight: str,
      color: str,
      pady: tuple[int, int] = (0, 8),
      height: int = 22,
      justify: str = tk.LEFT) -> tk.Label:
    label = tk.Label(
      parent,
      text=text,
      bg="#FFFFFF",
      fg=color,
      font=("Segoe UI", size, weight),
      anchor=tk.W,
      justify=justify)
    label.grid(row=row, column=0, sticky="ew", pady=pady)
    text_lines = max(1, height // 18)
    label.configure(height=text_lines)
    label.grid_configure(ipady=max(0, (height - (18 * text_lines)) // 2))
    return label

  def _on_video_resize(self, event) -> None:
    if not self.running:
      self.video_canvas.coords(
        self.video_placeholder,
        event.width // 2,
        event.height // 2)

  def _on_side_resize(self, event) -> None:
    if not hasattr(self, "last_output_label"):
      return
    wraplength = max(event.width - 4, 160)
    self.last_output_label.configure(wraplength=wraplength)
    self.model_status_label.configure(wraplength=wraplength)

  def _model_status_text(self) -> str:
    model_name = (
      self.active_model_info.display_name
      if self.active_model_info else "No model")
    if not self.classes:
      return f"{model_name}: classes not found. Run preprocessing first."
    if self.model is None:
      return f"{model_name}: model not found. Run training first."
    model_kind = (
      self.active_model_info.kind
      if self.active_model_info else "model")
    return f"{model_name}: {model_kind} ready ({len(self.classes)} classes)"

  def _set_current_word(self, word: str) -> None:
    max_width = self.current_word_label.winfo_width()
    if max_width <= 1:
      max_width = 260
    max_width = max(max_width - 4, 160)

    font_size = 24
    while font_size > 12:
      font = tkfont.Font(family="Segoe UI", size=font_size, weight="bold")
      if font.measure(word) <= max_width:
        break
      font_size -= 1
    self.current_word_label.configure(
      text=word,
      font=("Segoe UI", font_size, "bold"))

  def start_camera(self) -> None:
    if self.running:
      return
    if self.model is None:
      messagebox.showwarning("Model unavailable", self._model_status_text())
      return

    try:
      self.camera = CameraController(
        self.model,
        self.classes,
        self.device,
        self.active_model_info)
    except Exception as exc:
      messagebox.showerror("MediaPipe error", str(exc))
      return

    if not self.camera.start():
      self.camera.close()
      self.camera = None
      messagebox.showerror("Camera error", "Could not open webcam.")
      return

    self.running = True
    self.start_button.configure(state=tk.DISABLED)
    self.stop_button.configure(state=tk.NORMAL)
    self.status_label.configure(text=f"Camera running | device: {self.device}")
    sequence_length = (
      self.active_model_info.sequence_length
      if self.active_model_info else config.SEQUENCE_LENGTH)
    self.last_output_label.configure(
      text=f"Warming up: 0/{sequence_length} frames")
    self.after(0, self._update_frame)

  def on_model_selected(self, _event=None) -> None:
    selected_name = self.model_var.get()
    selected_info = self.model_by_name.get(selected_name)
    if selected_info is None or selected_info == self.active_model_info:
      return

    if self.running:
      self.stop_camera()

    try:
      classes = load_classes(selected_info)
      model = load_model(classes, self.device, selected_info)
    except Exception as exc:
      messagebox.showerror("Model load error", str(exc))
      if self.active_model_info is not None:
        self.model_var.set(self.active_model_info.display_name)
      return

    if not classes or model is None:
      messagebox.showwarning(
        "Model unavailable",
        f"Could not load {selected_info.display_name}.")
      if self.active_model_info is not None:
        self.model_var.set(self.active_model_info.display_name)
      return

    self.active_model_info = selected_info
    self.classes = classes
    self.model = model
    self.clear_subtitles()
    self.model_status_label.configure(text=self._model_status_text())
    self.status_label.configure(text=f"Idle | device: {self.device}")

  def stop_camera(self) -> None:
    self.running = False
    if self.camera is not None:
      self.camera.close()
      self.camera = None
    self.video_canvas.delete("all")
    self.video_placeholder = self.video_canvas.create_text(
      self.video_canvas.winfo_width() // 2,
      self.video_canvas.winfo_height() // 2,
      text="Webcam Feed",
      fill="#F9FAFB",
      font=("Segoe UI", 18, "bold"))
    self.start_button.configure(state=tk.NORMAL)
    self.stop_button.configure(state=tk.DISABLED)
    self.status_label.configure(text=f"Idle | device: {self.device}")

  def _update_frame(self) -> None:
    if not self.running or self.camera is None:
      return

    frame = self.camera.read()
    if frame is None:
      self.stop_camera()
      messagebox.showerror("Camera error", "Lost webcam stream.")
      return

    prediction = self.camera.process_frame(frame)
    if prediction is not None:
      self._show_prediction(prediction)
    elif len(self.camera.sequence_buffer) < self.camera.sequence_length:
      self.last_output_label.configure(
        text=f"Warming up: {len(self.camera.sequence_buffer)}/{self.camera.sequence_length} frames")

    self._render_video(frame)
    self.after(15, self._update_frame)

  def _show_prediction(self, prediction: Prediction) -> None:
    self._set_current_word(prediction.word.upper())
    self.confidence_value.configure(text=f"{prediction.confidence:.0%}")
    self.confidence_bar["value"] = prediction.confidence * 100

    if prediction.emitted:
      self.words.append(prediction.word)
      if prediction.status == "emitted_stable":
        self.last_output_label.configure(
          text=f"{prediction.word.upper()} after stable match at {prediction.confidence:.0%}")
      else:
        self.last_output_label.configure(
          text=f"{prediction.word.upper()} at {prediction.confidence:.0%} confidence")
      self.word_count_label.configure(text=f"Words: {len(self.words)}")
      self._refresh_subtitle_text()
    elif prediction.status == "low_confidence":
      self.last_output_label.configure(
        text=f"Waiting: {prediction.word.upper()} at {prediction.confidence:.0%}")
    elif prediction.status == "cooldown":
      self.last_output_label.configure(
        text=f"Holding: {prediction.word.upper()} at {prediction.confidence:.0%}")
    elif prediction.status == "ambiguous":
      self.last_output_label.configure(
        text=f"Waiting: {prediction.word.upper()} not distinct enough")
    elif prediction.status == "stabilizing":
      self.last_output_label.configure(
        text=f"Checking: {prediction.word.upper()} at {prediction.confidence:.0%}")
    elif prediction.status == "stabilizing_low_confidence":
      self.last_output_label.configure(
        text=f"Checking stable: {prediction.word.upper()} at {prediction.confidence:.0%}")
    elif prediction.status == "no_hand":
      self.last_output_label.configure(text="Waiting: no hand landmarks")

  def _refresh_subtitle_text(self) -> None:
    self.subtitle_text.configure(state=tk.NORMAL)
    self.subtitle_text.delete("1.0", tk.END)
    self.subtitle_text.insert("1.0", " ".join(self.words))
    self.subtitle_text.configure(state=tk.DISABLED)

  def clear_subtitles(self) -> None:
    self.words.clear()
    if self.camera is not None:
      self.camera.reset_output_state()
    self.last_output_label.configure(text="No words yet")
    self.word_count_label.configure(text="Words: 0")
    self._refresh_subtitle_text()

  def _render_video(self, frame) -> None:
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)

    target_w = max(self.video_canvas.winfo_width(), 1)
    target_h = max(self.video_canvas.winfo_height(), 1)
    if target_w < 20 or target_h < 20:
      target_w, target_h = 640, 360
    image.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), "#111827")
    x = (target_w - image.width) // 2
    y = (target_h - image.height) // 2
    canvas.paste(image, (x, y))

    self.video_image = ImageTk.PhotoImage(canvas)
    self.video_canvas.delete("all")
    self.video_canvas.create_image(
      target_w // 2,
      target_h // 2,
      image=self.video_image,
      anchor=tk.CENTER)

  def on_close(self) -> None:
    self.stop_camera()
    self.destroy()
