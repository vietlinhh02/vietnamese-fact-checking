"""Test script for the complete Vietnamese fact-checking pipeline."""

import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.llm_controller import create_ll