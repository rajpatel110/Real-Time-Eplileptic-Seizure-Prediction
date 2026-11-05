"""Summary-clock timeline for auditing cross-recording labels.

Summary order is assumed chronological, with less than one day between starts.
Clock-only reconstruction cannot prove dates or exclude missing whole days.
"""
from dataclasses import dataclass
import re

from eeg_project.dataset import Seizure, parse_summary


@dataclass(frozen=True)
class TimedRecording:
    start: int
    end: int
    events: tuple[Seizure, ...]


def clock_seconds(value):
    match = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})", value)
    if not match:
        raise ValueError(f"Invalid recording clock: {value}")
    hour, minute, second = map(int, match.groups())
    if hour > 24 or minute > 59 or second > 59:
        raise ValueError(f"Invalid recording clock: {value}")
    return hour * 3600 + minute * 60 + second


def summary_timeline(text):
    """Reconstruct minimum elapsed time consistent with ordered summary clocks."""
    annotations = parse_summary(text)
    timeline = {}
    previous_start = previous_end = None
    for section in re.split(r"(?m)^File Name:\s*", text)[1:]:
        name = section.splitlines()[0].strip()
        clocks = []
        for kind in ("Start", "End"):
            values = re.findall(rf"(?m)^File {kind} Time:\s*([^\r\n]+)", section)
            if len(values) != 1:
                raise ValueError(f"Missing or duplicate {kind} clock for {name}")
            clocks.append(clock_seconds(values[0].strip()))
        start_clock, end_clock = clocks
        duration = end_clock - start_clock
        if duration < 0:
            duration += 86400
        if not 0 < duration < 86400:
            raise ValueError(f"Invalid recording duration: {name}")
        start = start_clock % 86400
        if previous_start is not None:
            start += (previous_start // 86400) * 86400
            if start < previous_start:
                start += 86400
            if start < previous_end:
                raise ValueError(f"Overlapping or unordered recording: {name}")
        events = annotations[name].seizures
        if any(not 0 <= event.start < event.end <= duration for event in events):
            raise ValueError(f"Event outside summary duration: {name}")
        timeline[name] = TimedRecording(start, start + duration, tuple(
            Seizure(start + event.start, start + event.end) for event in events))
        previous_start, previous_end = start, start + duration
    return timeline
