"""
Logging utilities: file-backed logger with automatic backup on restart.
"""
import os


class Logger:
    """Write log lines to a file and stdout simultaneously."""

    def __init__(self, path: str) -> None:
        if os.path.exists(path):
            count = 1
            while os.path.exists(f'{path}_{count}'):
                count += 1
            os.rename(path, f'{path}_{count}')
        self._file = open(path, 'w', buffering=1)

    def print(self, line: str) -> None:
        """Write a line with newline to both the log file and stdout."""
        self._file.write(line + '\n')
        print(line)

    def log(self, line: str) -> None:
        """Write raw text to the log file only."""
        self._file.write(line)

    def close(self) -> None:
        self._file.close()


class NullLogger:
    """Discards all messages; used when logging is disabled."""

    def print(self, *args, **kwargs) -> None:
        pass

    def log(self, *args, **kwargs) -> None:
        pass
