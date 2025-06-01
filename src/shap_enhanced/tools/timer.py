"""
Timing Utility for Code Execution
=================================

Overview
--------

This module provides a lightweight utility class `Timer` for measuring the execution time  
of arbitrary blocks of Python code using a context manager (`with` block). It is intended  
for profiling and monitoring performance during development, benchmarking, or debugging.

Features
^^^^^^^^

- **Context Manager Interface**:  
  Use `with Timer("label"):` to automatically time a code block.

- **Human-Readable Output**:  
  Prints the elapsed time in seconds with 4-digit precision.

- **Silent Mode**:  
  Set `verbose=False` to suppress output and access `.elapsed` manually.

Use Case
--------

Useful for timing:
- Model training loops.
- Data loading or preprocessing steps.
- Custom algorithm profiling or benchmarking.
"""

import time


class Timer:
    r"""
    Timing context manager for profiling code execution.

    This utility measures wall-clock time (in seconds) for any code block wrapped in a `with` statement.  
    It prints the elapsed time with a label, or stores it for later access via the `.elapsed` attribute.

    Example
    -------
    .. code-block:: python

        with Timer("Sleeping 1s..."):
            time.sleep(1)

    :param str label: Optional label to describe the timed block.
    :param bool verbose: If True, prints elapsed time on exit. Otherwise, stores in `.elapsed`.

    :ivar float elapsed: Time in seconds between context entry and exit.
    """
    def __init__(self, label: str = "", verbose: bool = True):
        """
        :param str label: Description to print with the timing result.
        :param bool verbose: Whether to print elapsed time automatically.
        """
        self.label = label
        self.verbose = verbose
        self._start = None
        self.elapsed = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        self.elapsed = end - self._start
        if self.verbose:
            print(f"[Timer] {self.label} took {self.elapsed:.4f} seconds")

if __name__ == "__main__":
    with Timer("Simulated work"):
        time.sleep(1.5)