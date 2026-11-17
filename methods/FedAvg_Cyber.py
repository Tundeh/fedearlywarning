#!/usr/bin/env python3
"""Convenience launcher for FedAvg cybersecurity experiment."""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
cmd = [sys.executable, str(ROOT/'methods'/'run_cyber.py'), '--algorithm', 'FedAvg'] + sys.argv[1:]
raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))
