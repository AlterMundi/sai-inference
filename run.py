#!/usr/bin/env python3
"""
Entry point for SAI Inference Service
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main import run_server

if __name__ == "__main__":
    run_server()