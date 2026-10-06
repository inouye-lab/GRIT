"""Minimal checkpoint identity and inference-restoration contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pydantic import Field, StrictStr

from grit.schemas import StrictBoundaryModel
from grit.selection.cmnist import CheckpointIdentity, FrozenCheckpointSelection

StateT = TypeVar("StateT")


@dataclass(frozen=True)
class StoredCheckpoint(Generic[StateT]):
    """An identity plus opaque inference state returned by a checkpoint store."""

    identity: CheckpointIdentity
    state: StateT


class CheckpointStore(Protocol[StateT]):
    """Replaceable read boundary with no promised persistence format."""

    @property
    def store_id(self) -> str: ...

    def load(self, checkpoint_id: str) -> StoredCheckpoint[StateT]: ...


class RestorationReceipt(StrictBoundaryModel):
    receipt_id: StrictStr = Field(min_length=1)
    candidate_selection_id: StrictStr = Field(min_length=1)
    store_id: StrictStr = Field(min_length=1)
    checkpoint: CheckpointIdentity


def restore_checkpoint(
    selection: FrozenCheckpointSelection,
    store: CheckpointStore[StateT],
    restore_inference_state: Callable[[StateT], None],
) -> RestorationReceipt:
    """Restore exactly the selected inference state and return an auditable receipt."""

    stored = store.load(selection.checkpoint.checkpoint_id)
    if stored.identity != selection.checkpoint:
        raise ValueError(
            "checkpoint store returned an identity that does not match selection"
        )
    restore_inference_state(stored.state)
    return RestorationReceipt(
        receipt_id=f"restored:{store.store_id}:{stored.identity.checkpoint_id}",
        candidate_selection_id=selection.candidate_selection_id,
        store_id=store.store_id,
        checkpoint=stored.identity,
    )
