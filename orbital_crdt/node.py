"""G-Set CRDT nodes with simple revocation semantics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GSetNode:
    """Grow-only set node used for token revocations."""

    node_id: str
    revocations: set[str] = field(default_factory=set)

    def revoke(self, token_id: str) -> None:
        """Add a token revocation locally."""
        self.revocations.add(token_id)

    def merge(self, other: "GSetNode") -> int:
        """Merge another node's revocations and return the number added."""
        before = len(self.revocations)
        self.revocations |= other.revocations
        return len(self.revocations) - before

    def export_state(self) -> set[str]:
        """Return a copy of the revocation state."""
        return set(self.revocations)

    def import_state(self, state: set[str]) -> int:
        """Merge a raw G-Set state and return the number added."""
        before = len(self.revocations)
        self.revocations |= set(state)
        return len(self.revocations) - before

    def accepts(self, token_id: str) -> bool:
        """Return True when the token is not yet known as revoked."""
        return token_id not in self.revocations

