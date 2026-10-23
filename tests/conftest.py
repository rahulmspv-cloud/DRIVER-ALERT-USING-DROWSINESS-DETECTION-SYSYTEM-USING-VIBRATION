import sys
from pathlib import Path

# allow `import drowsiness_adas` and `import eval...` helpers when running pytest from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
