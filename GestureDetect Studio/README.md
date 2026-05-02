# GestureDetect Studio

Tkinter desktop GUI for the WLASL gesture recognition demo.

## Run

From the project root:

```powershell
python.exe "GestureDetect Studio\app.py"
```

On macOS/Linux:

```bash
python "GestureDetect Studio/app.py"
```

## Expected Project Files

The GUI expects the preprocessing and training pipeline to have produced:

- `data/2_processed/wlasl_class_list.npy`
- `models/wlasl_lstm.pth`
- `models/hand_landmarker.task` when using Python 3.13-compatible MediaPipe.

The subtitle output is displayed in a separate bottom panel, not inside the
webcam preview.

## Code Layout

- `app.py`: application entry point only.
- `UI.py`: Tkinter layout, buttons, subtitle output, confidence display.
- `Camera.py`: webcam capture, MediaPipe landmarks, prediction loop.
- `model.py`: LSTM architecture and model/class loading.
- `paths.py`: project-root path helpers.
