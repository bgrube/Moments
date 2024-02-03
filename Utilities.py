"""Module that provides utility functions of general scope"""

from dataclasses import (
  dataclass,
  field,
)
import functools
import os
import subprocess
import time
from typing import (
  Dict,
  Optional,
)


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def printGitInfo() -> None:
  """Prints directory of this file and git hash in this directory"""
  repoDir = os.path.dirname(os.path.abspath(__file__))
  gitInfo = subprocess.check_output(["git", "describe", "--always"], cwd = repoDir).strip().decode()
  print(f"Running code in '{repoDir}', git version '{gitInfo}'")


@dataclass
class TimeData:
  """Holds start and stop times for wall and cpu timer"""
  wallTimeStart: float
  cpuTimeStart:  float
  wallTimeStop:  Optional[float] = None
  cpuTimeStop:   Optional[float] = None

  def stop(
    self,
    wallTime: float,
    cpuTime:  float,
  ) -> None:
    """Sets stop times"""
    self.wallTimeStop = wallTime
    self.cpuTimeStop  = cpuTime

  def wallTime(self) -> Optional[float]:
    """Returns elapsed wall time"""
    return None if self.wallTimeStop is None else self.wallTimeStop - self.wallTimeStart

  def cpuTime(self) -> Optional[float]:
    """Returns elapsed CPU time"""
    return None if self.cpuTimeStop  is None else self.cpuTimeStop  - self.cpuTimeStart

  def summary(self) -> Optional[str]:
    """Returns string that summarizes wall and CPU time"""
    strings = []
    if self.wallTime() is not None:
      strings.append(f"wall time = {self.wallTime():.4g} sec")
    if self.cpuTime() is not None:
      strings.append(f"CPU time = {self.cpuTime():.4g} sec")
    summary = ", ".join(strings)
    return summary if summary else None


@dataclass
class Timer:
  """Measures time differences"""
  _times: Dict[str, TimeData] = field(default_factory = lambda: {})  # start and stop times for wall time and CPU time indexed by name

  def start(
    self,
    name: str,
  ) -> None:
    """Creates a new timer with given name"""
    self._times[name] = TimeData(wallTimeStart = time.monotonic(), cpuTimeStart  =time.process_time())

  def stop(
    self,
    name: str,
  ) -> None:
    """Stops timer with given name"""
    if name not in self._times:
      # gracefully ignore unknown timers
      return
    self._times[name].stop(time.monotonic(), time.process_time())

  def summary(self) -> Optional[str]:
    """Returns string with summary of all timers"""
    strings = []
    for name, timeData in self._times.items():
      if timeData.summary() is not None:
        strings.append(f"{name}: {timeData.summary()}")
    summary = "\n".join(strings)
    return summary if summary else None
