import os
import subprocess
import sys
import pytest
import shutil

def test_smoke_train_runs(tmp_path):
    out_dir = tmp_path / "exported_model_test"
    cmd = [sys.executable, "train_and_export.py", "--sample", "0.01", "--model-dir", str(out_dir)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
    assert proc.returncode == 0, f"train script failed: {proc.stderr}"
    model_file = out_dir / "rf_churn_pipe.joblib"
    assert model_file.exists(), f"model not produced at {model_file}"
