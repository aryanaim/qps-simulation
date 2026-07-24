"""Token buffer with staleness tracking and configurable consumption order."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Token:
    """A QPS token generated from one satellite pass."""

    token_id: str
    created_timestamp: float
    pass_id: int


class TokenBuffer:
    """Capacity-limited token buffer.

    consume_token defaults to FIFO. The random policy is available for the
    timing-side-channel mitigation tested in integration.
    """

    def __init__(
        self,
        capacity: int,
        staleness_max_hours: float,
        consumption_policy: str = "fifo",
        rng: random.Random | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if staleness_max_hours <= 0:
            raise ValueError("staleness_max_hours must be positive")
        if consumption_policy not in {"fifo", "random"}:
            raise ValueError("consumption_policy must be fifo or random")
        self.capacity = int(capacity)
        self.staleness_max_sec = float(staleness_max_hours) * 3600.0
        self.consumption_policy = consumption_policy
        self.rng = rng or random.Random()
        self._tokens: deque[Token] = deque()

    def add_tokens(self, batch: Iterable[Token], timestamp: float) -> int:
        """Add as many tokens as capacity allows and return the number added."""
        self.expire_stale(timestamp)
        added = 0
        for token in batch:
            if len(self._tokens) >= self.capacity:
                break
            self._tokens.append(token)
            added += 1
        return added

    def consume_token(self, timestamp: float) -> Token | None:
        """Return one non-expired token or None when the buffer is empty."""
        self.expire_stale(timestamp)
        if not self._tokens:
            return None
        if self.consumption_policy == "fifo":
            return self._tokens.popleft()
        index = self.rng.randrange(len(self._tokens))
        tokens = list(self._tokens)
        token = tokens.pop(index)
        self._tokens = deque(tokens)
        return token

    def expire_stale(self, timestamp: float) -> int:
        """Discard stale tokens and return the expired count."""
        before = len(self._tokens)
        cutoff = timestamp - self.staleness_max_sec
        self._tokens = deque(token for token in self._tokens if token.created_timestamp >= cutoff)
        return before - len(self._tokens)

    def level(self) -> int:
        """Return the number of usable tokens currently buffered."""
        return len(self._tokens)

    def is_degraded(self) -> bool:
        """Return True when no pseudonyms can be issued."""
        return not self._tokens

    def snapshot(self) -> list[Token]:
        """Return a copy of buffered tokens for diagnostics."""
        return list(self._tokens)
