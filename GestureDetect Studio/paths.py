from __future__ import annotations

import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))


def project_path(path: str) -> Path:
  """Resolve project-relative paths from config.py."""
  candidate = Path(path)
  if candidate.is_absolute():
    return candidate
  return PROJECT_ROOT / candidate
