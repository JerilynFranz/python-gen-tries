#!/usr/bin/env python3
# This plan contains tests that demonstrate failures as well.
"""Testplan for testing the gentri module."""

# pylint: disable=import-error
import sys

from testplan import test_plan  # type: ignore
from testplan.testing import pyunit  # type: ignore

from tests.gentrie import test_gentri


# def before_start(env, result):
#     result.log("Executing before start hook.")
#
#
# def after_start(env, result):
#    result.log("Executing after start hook.")
#
#
# def before_stop(env, result):
#     result.log("Executing before stop hook.")
#
#
# def after_stop(env, result):
#    result.log("Executing after stop hook.")
#
#
@test_plan(name='PyUnitGentri', description='PyUnit gentri tests')
def main(plan):  # type: ignore
    plan.add(  # type: ignore
        pyunit.PyUnit(
            name="My PyUnit",
            description="PyUnit example testcase",
            testcases=[test_gentri.TestHashable,
                       test_gentri.TestGeneralizedKey,
                       test_gentri.TestGeneralizedTrie],
            # before_start=before_start,
            # after_start=after_start,
            # before_stop=before_stop,
            # after_stop=after_stop,
        )
    )


if __name__ == "__main__":
    # pylint: disable=invalid-name, no-value-for-parameter
    res = main()  # type: ignore
    sys.exit(res.exit_code)  # type: ignore
