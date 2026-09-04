"""Forwarding shim — the real script is src/train_recoverability_model_v3.py."""
import runpy, sys

sys.exit(runpy.run_path("src/train_recoverability_model_v3.py", run_name="__main__"))
