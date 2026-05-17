#!/usr/bin/env python3
"""
CLI entry point for the Phase-3 reconstruction workflow.

Usage:
    python3 analysis/run_extraction.py [--tiniest | --tinier | --makeblobs | --full]
                                       [--from-scratch] [--sign-search]
                                       [--refine] [--refine-unfreeze]
                                       [--refine-epochs N] [--refine-lr LR]

This is the new modular replacement for `analysis/test_extraction4.py`. It
delegates to `extraction_pipeline.workflow.main`. The old test_extraction4.py
remains as a thin shim around the same entry point so existing scripts and
documentation continue to work unchanged.
"""

import os
import sys

# Allow running as `python3 analysis/run_extraction.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extraction_pipeline.workflow import main


if __name__ == '__main__':
    main()
