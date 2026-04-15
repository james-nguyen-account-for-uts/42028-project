import numpy as np
from scipy.interpolate import interp1d
from config import TARGET_FRAMES, LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX


def extract_xy_conf(keypoints, n_points):
  """Pulls (x, y) coordinates and ignores confidence scores."""
  kp = np.asarray(keypoints, dtype=np.float32).reshape(n_points, 3)
  coords = kp[:, :2]
  conf = kp[:, 2]
  coords[conf == 0] = np.nan
  return coords


def interpolate_nan(sequence):
  """Fills in gaps if a hand disappears for a few frames."""
  T, D = sequence.shape
  t = np.arange(T)
  for d in range(D):
    col = sequence[:, d]
    mask = ~np.isnan(col)
    if mask.sum() < 2:
      continue
    sequence[:, d] = np.interp(t, t[mask], col[mask])
  return sequence


def resample_sequence(sequence, target_len=TARGET_FRAMES):
  """Intepolate sequences to a fixed number of frames"""
  T, D = sequence.shape
  if T == target_len:
    return sequence

  x_old = np.linspace(0, 1, T)
  x_new = np.linspace(0, 1, target_len)
  f = interp1d(
    x_old, sequence, axis=0, kind='linear', fill_value='extrapolate')
  return f(x_new)


def normalize_frames(frames):
  """Center on shoulders and scales to a -1 to 1 range"""
  T, P, _ = frames.shape
  new_frames = np.copy(frames)

  for t in range(T):
    ls = frames[t, LEFT_SHOULDER_INDEX]
    rs = frames[t, RIGHT_SHOULDER_INDEX]

    # Check if shoulders exist (not NaN)
    if not np.isnan(ls).any() and not np.isnan(rs).any():
      center = (ls + rs) / 2
      scale = np.linalg.norm(ls - rs)
    else:
      # Fallback: Center on the average of all detected points in this frame
      visible_points = frames[t][~np.isnan(frames[t]).any(axis=1)]
      if len(visible_points) > 0:
        center = np.mean(visible_points, axis=0)
        scale = 1.0
      else:
        center = np.array([0, 0])
        scale = 1.0

    # Apply normalization to this specific frame
    if scale > 1e-6:
      new_frames[t] = (frames[t] - center) / scale
    else:
      new_frames[t] = frames[t] - center

  return np.nan_to_num(new_frames, nan=0.0, posinf=0.0, neginf=0.0)
