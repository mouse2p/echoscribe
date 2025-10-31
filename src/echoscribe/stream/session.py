"""Session records: what a streaming run produced, serialized for later scoring.

A session is the unit a benchmark reports on: one pass over a corpus under one
configuration, with one record per utterance.  The record carries the hypothesis, the
reference, the frame count, the event count, the latency and whether an endpoint
fired -- everything a scorer and a report need, and nothing that requires the audio
to still exist.

Latency is stored as a number of *frames* rather than milliseconds.  The frame shift
is part of the run metadata, so a millisecond value would either duplicate it or
silently disagree with it; keeping frames means one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from echoscribe.errors import SchemaError, StreamError, require

#: Bumped when the serialized shape of a session changes.
SESSION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class UtteranceRecord:
    """One utterance's outcome inside a session."""

    utterance_id: str
    hypothesis: tuple[int, ...]
    reference: str
    text: str
    frames: int
    events: int
    endpoint_frames: int | None
    score: float

    def __post_init__(self) -> None:
        require(bool(self.utterance_id), "an utterance record needs an id", StreamError)
        require(self.frames >= 0, f"frames must not be negative, got {self.frames}", StreamError)
        require(self.events >= 0, f"events must not be negative, got {self.events}", StreamError)
        require(
            all(symbol >= 0 for symbol in self.hypothesis),
            f"{self.utterance_id}: a hypothesis cannot hold a negative symbol",
            StreamError,
        )
        if self.endpoint_frames is not None:
            require(
                0 <= self.endpoint_frames <= self.frames,
                f"{self.utterance_id}: the endpoint fired at frame {self.endpoint_frames} "
                f"of {self.frames}",
                StreamError,
            )

    @property
    def endpointed(self) -> bool:
        """True when an endpoint fired before the audio ran out."""
        return self.endpoint_frames is not None

    def to_dict(self) -> dict:
        """A JSON-ready mapping of the record."""
        return {
            "utterance_id": self.utterance_id,
            "hypothesis": list(self.hypothesis),
            "reference": self.reference,
            "text": self.text,
            "frames": self.frames,
            "events": self.events,
            "endpoint_frames": self.endpoint_frames,
            "score": round(float(self.score), 12),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> UtteranceRecord:
        """Rebuild a record from :meth:`to_dict`."""
        for name in ("utterance_id", "hypothesis", "reference", "text", "frames", "events"):
            require(name in payload, f"utterance record is missing {name!r}", SchemaError)
        endpoint = payload.get("endpoint_frames")
        return cls(
            utterance_id=str(payload["utterance_id"]),
            hypothesis=tuple(int(symbol) for symbol in payload["hypothesis"]),
            reference=str(payload["reference"]),
            text=str(payload["text"]),
            frames=int(payload["frames"]),
            events=int(payload["events"]),
            endpoint_frames=None if endpoint is None else int(endpoint),
            score=float(payload.get("score", 0.0)),
        )


@dataclass(frozen=True)
class Session:
    """One run over a corpus: its metadata and one record per utterance."""

    session_id: str
    metadata: dict
    utterances: tuple[UtteranceRecord, ...]

    def __post_init__(self) -> None:
        require(bool(self.session_id), "a session needs an id", StreamError)
        require(
            len({item.utterance_id for item in self.utterances}) == len(self.utterances),
            f"session {self.session_id} holds a repeated utterance id",
            StreamError,
        )

    def __len__(self) -> int:
        return len(self.utterances)

    def record(self, utterance_id: str) -> UtteranceRecord:
        """The record for one utterance."""
        for item in self.utterances:
            if item.utterance_id == utterance_id:
                return item
        raise StreamError(f"{utterance_id!r} is not part of session {self.session_id!r}")

    @property
    def total_frames(self) -> int:
        """Frames decoded across the whole session."""
        return sum(item.frames for item in self.utterances)

    @property
    def endpoint_rate(self) -> float:
        """Fraction of utterances an endpoint fired on; 0.0 for an empty session."""
        if not self.utterances:
            return 0.0
        fired = sum(1 for item in self.utterances if item.endpointed)
        return fired / len(self.utterances)

    def mean_endpoint_delay(self) -> float:
        """Mean frames between an utterance's end and its endpoint, or ``None``.

        A negative value means the endpoint fired *before* the audio ran out, which
        is the usual case and the number a latency budget cares about.
        """
        delays = [
            item.endpoint_frames - item.frames
            for item in self.utterances
            if item.endpoint_frames is not None
        ]
        return sum(delays) / len(delays) if delays else 0.0

    def summary(self) -> dict:
        """The numbers a report quotes, rounded where they are produced."""
        return {
            "session_id": self.session_id,
            "utterances": len(self.utterances),
            "frames": self.total_frames,
            "endpoint_rate": round(self.endpoint_rate, 6),
            "mean_endpoint_delay": round(self.mean_endpoint_delay(), 6),
            "mean_score": round(
                sum(item.score for item in self.utterances) / len(self.utterances), 6
            )
            if self.utterances
            else 0.0,
        }
