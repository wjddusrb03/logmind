"""Log collectors for various sources."""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
from typing import Iterator, List, Optional


class FileCollector:
    """Collect logs from file(s), optionally following new lines."""

    def __init__(
        self,
        paths: List[str],
        follow: bool = False,
        poll_interval: float = 0.5,
    ):
        # Expand glob patterns
        self.files: List[str] = []
        for p in paths:
            expanded = glob.glob(p, recursive=True)
            if expanded:
                self.files.extend(expanded)
            elif os.path.isfile(p):
                self.files.append(p)

        if not self.files:
            raise FileNotFoundError(f"No log files found matching: {paths}")

        self.follow = follow
        self.poll_interval = poll_interval

    def read_all(self) -> List[str]:
        """Read all lines from all files (batch mode)."""
        lines: List[str] = []
        for path in sorted(self.files):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines.extend(f.readlines())
            except (IOError, PermissionError) as e:
                print(f"Warning: cannot read {path}: {e}", file=sys.stderr)
        return lines

    def stream(self) -> Iterator[str]:
        """Stream lines, optionally following file growth."""
        # First read existing content
        for path in sorted(self.files):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        yield line
            except (IOError, PermissionError):
                continue

        if not self.follow:
            return

        # Follow mode: watch last file for new lines
        target = self.files[-1]
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)  # end of file
            while True:
                line = f.readline()
                if line:
                    yield line
                else:
                    time.sleep(self.poll_interval)


class StdinCollector:
    """Collect logs from stdin (pipe)."""

    def stream(self) -> Iterator[str]:
        for line in sys.stdin:
            yield line

    def read_all(self) -> List[str]:
        return sys.stdin.readlines()


class DockerCollector:
    """Collect logs from Docker container."""

    def __init__(
        self,
        container: str,
        follow: bool = True,
        tail: Optional[int] = None,
    ):
        self.container = container
        self.follow = follow
        self.tail = tail

    def stream(self) -> Iterator[str]:
        cmd = ["docker", "logs"]
        if self.follow:
            cmd.append("--follow")
        if self.tail:
            cmd.extend(["--tail", str(self.tail)])
        cmd.append(self.container)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                yield line
        finally:
            proc.terminate()

    def read_all(self) -> List[str]:
        cmd = ["docker", "logs"]
        if self.tail:
            cmd.extend(["--tail", str(self.tail)])
        cmd.append(self.container)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return (result.stdout + result.stderr).splitlines(keepends=True)


class CommandCollector:
    """Collect logs from any shell command output."""

    def __init__(self, command: str, follow: bool = True):
        self.command = command
        self.follow = follow

    def stream(self) -> Iterator[str]:
        proc = subprocess.Popen(
            self.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                yield line
        finally:
            proc.terminate()

    def read_all(self) -> List[str]:
        result = subprocess.run(
            self.command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return (result.stdout + result.stderr).splitlines(keepends=True)
