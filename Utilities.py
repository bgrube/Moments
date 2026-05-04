"""Module that provides utility functions of general scope"""

from __future__ import annotations

from collections.abc import Sequence
import contextlib
from dataclasses import (
  dataclass,
  field,
)
import functools
import os
import subprocess
import time


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def printGitInfo() -> None:
  """Prints directory of this file and git hash in this directory"""
  repoDir = os.path.dirname(os.path.abspath(__file__))
  gitInfo = subprocess.check_output(["git", "describe", "--always"], cwd = repoDir).strip().decode()
  print(f"Running code in '{repoDir}', git version '{gitInfo}'")


def makeDirPath(dirPath: str) -> str:
  """Create path to directory and return directory path as given by argument"""
  try:
    os.makedirs(dirPath, exist_ok = False)
  except FileExistsError:
    pass  # directory already exists; do nothing
  except Exception:
    raise  # something went wrong
  else:
    print(f"Created directory '{dirPath}'")
  return dirPath


@dataclass
class TimeData:
  """Holds start and stop times for wall and cpu timer"""
  wallTimeStart: float
  cpuTimeStart:  float
  wallTimeStop:  float | None = None
  cpuTimeStop:   float | None = None

  def stop(self) -> None:
    """Sets stop times"""
    self.wallTimeStop = time.monotonic()
    self.cpuTimeStop  = time.process_time()

  @property
  def wallTime(self) -> float | None:
    """Returns elapsed wall time"""
    return None if self.wallTimeStop is None else self.wallTimeStop - self.wallTimeStart

  @property
  def cpuTime(self) -> float | None:
    """Returns elapsed CPU time"""
    return None if self.cpuTimeStop  is None else self.cpuTimeStop  - self.cpuTimeStart

  @property
  def summary(self) -> str | None:
    """Returns string that summarizes wall and CPU time"""
    strings = []
    if self.wallTime is not None:
      strings.append(f"wall time = {self.wallTime:.4g} sec")
    if self.cpuTime is not None:
      strings.append(f"CPU time = {self.cpuTime:.4g} sec")
    summary = ", ".join(strings)
    return summary if summary else None


@dataclass
class Timer:
  """Measures time differences"""
  _times: dict[str, TimeData] = field(default_factory = lambda: {})  # stores start and stop times for wall time and CPU time indexed by name

  def start(
    self,
    name: str,
  ) -> TimeData:
    """Creates or updates the timer associated with the given name"""
    t = TimeData(wallTimeStart = time.monotonic(), cpuTimeStart = time.process_time())
    self._times[name] = t
    return t

  def stop(
    self,
    name: str,
  ) -> TimeData | None:
    """Stops the timer associated with given name"""
    if name not in self._times:
      # gracefully ignore unknown timers
      return None
    t = self._times[name]
    t.stop()
    return t

  @contextlib.contextmanager
  def timeThis(
    self,
    name: str
  ):
    """Context manager that measures time of enclosed code block"""
    try:
      t = self.start(name)
      yield
    finally:
      t.stop()

  @property
  def summary(self) -> str | None:
    """Returns string with summary of all timers"""
    strings = []
    for name, timeData in self._times.items():
      if timeData.summary is not None:
        strings.append(f"{name}: {timeData.summary}")
    summary = "\n".join(strings)
    return summary if summary else None
