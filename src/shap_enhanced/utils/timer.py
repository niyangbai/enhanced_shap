"""Timer utility for measuring elapsed time."""

import time

class Timer:
    """Timer utility for measuring elapsed time."""

    def __init__(self):
        self.start_time = None

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds.

        :return float: Elapsed time.
        """
        if self.start_time is None:
            raise RuntimeError("Timer was not started.")
        elapsed = time.time() - self.start_time
        self.start_time = None
        return elapsed
