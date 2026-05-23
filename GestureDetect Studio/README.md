# GestureDetect Studio

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

- `data/2_processed/landmark_v2/wlasl_landmark_class_list_v2.npy`
- `models/landmark_v2/wlasl_landmark_lstm_v2.pth`

The current GUI uses the Landmark LSTM v2 pipeline: MediaPipe extracts hand
landmarks, the app applies wrist-centered and scale-normalized landmark
features, and the trained LSTM classifier predicts one of the selected WLASL
word classes.

The model selector in the header scans `models` for loadable model folders. Each
model folder must have a matching processed-data folder with the same name under
`data/2_processed`.

The official Landmark v2 preprocessing uses the fixed class list in
`config.LANDMARK_V2_DEFAULT_CLASSES`. Run `select_landmark_v2_classes.py` only
when exploring a new vocabulary.

The subtitle output is displayed in a separate bottom panel, not inside the
webcam preview.

## Code Layout

- `app.py`: application entry point only.
- `UI.py`: Tkinter layout, buttons, subtitle output, confidence display.
- `Camera.py`: webcam capture, MediaPipe landmarks, prediction loop.
- `model.py`: LSTM architecture and model/class loading.
- `paths.py`: project-root path helpers.
