"""Progress reporting.

A factor build makes thousands of network calls and can run for minutes, so it prints what it
is doing. Output goes to stdout with flush=True rather than through the logging module: this
is a research library, callers run it in notebooks and terminals, and a buffered handler that
shows nothing for four minutes is worse than no logging at all.

Silence the package with set_verbose(enabled=False).
"""
import sys
import time
from typing import Optional

VERBOSE = True
STEP_STARTED_AT: Optional[float] = None


def set_verbose(enabled: bool) -> None:
    """
    Turn progress reporting on or off for the whole package.

    :param enabled: True to print progress, False to stay silent.
    :return: None.
    """
    global VERBOSE
    VERBOSE = enabled


def step(message: str) -> None:
    """
    Announce the start of a named stage and start its timer.

    :param message: Stage description.
    :return: None.
    """
    global STEP_STARTED_AT
    STEP_STARTED_AT = time.time()
    if VERBOSE:
        print(f"[semantic-factors] {message}", flush=True)


def done(message: str) -> None:
    """
    Close the stage opened by the last step() call, reporting its elapsed time.

    :param message: Completion description.
    :return: None.
    """
    elapsed = 0.0 if STEP_STARTED_AT is None else time.time() - STEP_STARTED_AT
    if VERBOSE:
        print(f"[semantic-factors] {message} ({elapsed:.1f}s)", flush=True)


def info(message: str) -> None:
    """
    Print an indented detail line beneath the current stage.

    :param message: Detail description.
    :return: None.
    """
    if VERBOSE:
        print(f"[semantic-factors]   {message}", flush=True)


def warn(message: str) -> None:
    """
    Print a warning that stays visible even when progress reporting is off.

    Warnings report facts that change how a result should be read, such as a truncated
    retrieval, so they are never silenced.

    :param message: Warning description.
    :return: None.
    """
    print(f"[semantic-factors] WARNING: {message}", file=sys.stderr, flush=True)
