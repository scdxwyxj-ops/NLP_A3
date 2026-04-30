import subprocess
import sys
from pathlib import Path


def run_eval(eval_script, predictions_path, groundtruth_path):
    result = subprocess.run(
        [
            sys.executable,
            str(eval_script),
            "--predictions",
            str(predictions_path),
            "--groundtruth",
            str(groundtruth_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()

