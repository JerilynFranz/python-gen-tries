#!/usr/bin/env python3
"""Pytest launcher for gentrie tests.

Usage:
  ./test_plan.py                # run all tests under tests/gentrie
  ./test_plan.py KeyToken       # pattern filter (class or test name substring / -k expression)

Environment (optional):
  GENTRIE_FAIL_FAST=1           # enable fail-fast (-x)
"""

from __future__ import annotations

import os
import sys
import pytest


def build_pytest_args(pattern: str | None) -> list[str]:
    args: list[str] = []
    # Fail fast
    if os.environ.get("GENTRIE_FAIL_FAST") == "1":
        args.append("-x")
    # Quiet by default; adjust as needed
    args.extend(["-q", "--disable-warnings"])
    # Add test path
    args.append("tests/gentrie")
    # Pattern handling: allow either simple substring or full -k expression
    if pattern:
        # If user already passed pytest expression operators, trust it
        if any(op in pattern for op in (" and ", " or ", " not ", "(", ")")):
            args.extend(["-k", pattern])
        else:
            # Simple substring: wrap to match anywhere in node id
            expr = pattern
            args.extend(["-k", expr])
    return args


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else None
    args = build_pytest_args(pattern)
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main())
