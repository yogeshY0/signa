"""Kaggle notebook code for training with --resize-only on raw BHSig+CEDAR.

Paste cells into a Kaggle notebook.
"""

# ==== Cell 1: Setup ====
"""
%cd /kaggle/working
!rm -rf sigver && git clone -b feature/bhsig-cedar-grayscale \\
  https://github.com/yogeshY0/signa.git sigver
%cd /kaggle/working/sigver
!pip install -q timm opencv-python-headless matplotlib
import os; os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
"""

# ==== Cell 2: Paths ====
"""
DATA = "/kaggle/input/bhsig-cedar-raw"   # your dataset
RUN = "raw_triplet"
CKPT = f"/kaggle/working/{RUN}/best_model.pt"
LOG = "/kaggle/working/training_log.txt"
CURVES = "/kaggle/working/curves.png"
EVAL = "/kaggle/working/eval.json"
!ls $DATA && head -3 $DATA/manifest.csv
"""

# ==== Cell 3: Train ====
"""
!python -m training.train_siamese --manifest $MF --image-root $DATA \\
  --resize-only --dataset-mix equal --references 1 \\
  --output-dir /kaggle/working/raw_triplet \\
  --device cuda --workers 2 \\
  --phase1-epochs 5 --phase2-epochs 40 --patience 8 \\
  --margin 0.3 --seed 42
"""

# ==== Cell 4: Curves ====
"""
!python -m training.analyze_log $LOG --plot $CURVES
from IPython.display import Image, display
display(Image(CURVES))
"""

# ==== Cell 5: Evaluate ====
"""
!python -m training.evaluate --checkpoint $CKPT --manifest $MF --image-root $DATA \\
  --resize-only --references 1 --device cuda --workers 2 --bootstrap 1000 \\
  --target-far 0.01 0.05
"""