"""Launch UFODeployPolicy with the isolated solve-close Carry runtime."""

from __future__ import annotations

import runpy

import rl_policy.carry_runtime as carry_runtime
from rl_policy.carry_solve_close_166559744_runtime import SolveCloseCarryRuntime


def main() -> None:
    # Patch only this process. The existing strong entry point and runtime files
    # are not edited or imported under a different global configuration.
    carry_runtime.CarryRuntime = SolveCloseCarryRuntime
    runpy.run_module("rl_policy.ufo_policy", run_name="__main__")


if __name__ == "__main__":
    main()
