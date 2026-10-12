#!/usr/bin/env python3
"""Convenience entry: run the AETHER invariant test-suite."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tests import test_invariants

if __name__ == "__main__":
    test_invariants.main()
