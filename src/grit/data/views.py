"""Role-scoped data views that keep final-test data behind the selection gate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, StrictStr, model_validator

from grit.schemas import (
    HELD_OUT_VALIDATION_NAMES,
    IN_DOMAIN_VALIDATION_NAMES,
    StrictBoundaryModel,
)

if TYPE_CHECKING:
    from grit.methods.checkpoints import RestorationReceipt
    from grit.selection.cmnist import (
        FrozenCandidateSelection,
        FrozenCheckpointSelection,
    )


class ExampleIdentity(StrictBoundaryModel):
    """Stable identities shared across otherwise distinct rendered examples."""

    example_id: StrictStr = Field(min_length=1)
    source_id: StrictStr = Field(min_length=1)
    view_id: StrictStr = Field(min_length=1)


class _SplitDescriptor(StrictBoundaryModel):
    dataset_id: StrictStr = Field(min_length=1)
    manifest_id: StrictStr = Field(min_length=1)
    name: StrictStr = Field(min_length=1)
    source_partition_id: StrictStr = Field(min_length=1)
    view_id: StrictStr = Field(min_length=1)


class TrainingSplitDescriptor(_SplitDescriptor):
    role: Literal["training"]

    @model_validator(mode="after")
    def _validate_cmnist_role(self) -> TrainingSplitDescriptor:
        allowed = {
            "cmnist": {
                "train_e01": "train_e01_sources",
                "train_e02": "train_e02_sources",
            },
        }
        if (
            self.dataset_id not in allowed
            or allowed[self.dataset_id].get(self.name) != self.source_partition_id
        ):
            raise ValueError(
                "training descriptors must name an approved CMNIST training split"
            )
        if self.view_id != self.name:
            raise ValueError("training view_id must equal the split name")
        return self


class ValidationSplitDescriptor(_SplitDescriptor):
    role: Literal["validation"]

    @model_validator(mode="after")
    def _validate_cmnist_role(self) -> ValidationSplitDescriptor:
        allowed = {
            "cmnist": {*IN_DOMAIN_VALIDATION_NAMES, *HELD_OUT_VALIDATION_NAMES},
        }
        if (
            self.dataset_id not in allowed
            or self.name not in allowed[self.dataset_id]
            or self.source_partition_id != "validation_sources"
        ):
            raise ValueError(
                "validation descriptors must name an approved dataset validation split"
            )
        if self.view_id != self.name:
            raise ValueError("validation view_id must equal the split name")
        return self


class FinalTestSplitDescriptor(_SplitDescriptor):
    role: Literal["final_test"]

    @model_validator(mode="after")
    def _validate_cmnist_role(self) -> FinalTestSplitDescriptor:
        expected_by_dataset = {
            "cmnist": ("cmnist", "test_ood", "test_sources", "test_ood"),
        }
        observed = (
            self.dataset_id,
            self.name,
            self.source_partition_id,
            self.view_id,
        )
        if observed != expected_by_dataset.get(self.dataset_id):
            raise ValueError(
                "final-test descriptors must name an approved dataset final split"
            )
        return self


def _validate_examples(
    descriptor: _SplitDescriptor,
    examples: tuple[ExampleIdentity, ...],
) -> None:
    if not examples:
        raise ValueError("a scoped view must contain at least one example")
    if any(example.view_id != descriptor.view_id for example in examples):
        raise ValueError("every example view_id must match its split descriptor")
    example_ids = tuple(example.example_id for example in examples)
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("example IDs must be unique within a split view")
    source_ids = tuple(example.source_id for example in examples)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source IDs must be unique within a split view")


class TrainingView(StrictBoundaryModel):
    descriptor: TrainingSplitDescriptor
    examples: tuple[ExampleIdentity, ...]

    @model_validator(mode="after")
    def _validate_contents(self) -> TrainingView:
        _validate_examples(self.descriptor, self.examples)
        return self


class ValidationView(StrictBoundaryModel):
    descriptor: ValidationSplitDescriptor
    examples: tuple[ExampleIdentity, ...]

    @model_validator(mode="after")
    def _validate_contents(self) -> ValidationView:
        _validate_examples(self.descriptor, self.examples)
        return self


class FinalTestView(StrictBoundaryModel):
    """Materialized final data, constructed only by the final-test gate."""

    descriptor: FinalTestSplitDescriptor
    examples: tuple[ExampleIdentity, ...]
    authorization_id: StrictStr = Field(min_length=1)
    run_id: StrictStr = Field(min_length=1)
    candidate_id: StrictStr = Field(min_length=1)
    scientific_config_digest: StrictStr = Field(min_length=1)
    checkpoint_id: StrictStr = Field(min_length=1)
    epoch: int = Field(ge=0)
    method_id: StrictStr = Field(min_length=1)
    seed: int

    @model_validator(mode="after")
    def _validate_contents(self) -> FinalTestView:
        _validate_examples(self.descriptor, self.examples)
        return self


class _FinalTestPayload(StrictBoundaryModel):
    handle_id: StrictStr = Field(min_length=1)
    run_id: StrictStr = Field(min_length=1)
    candidate_id: StrictStr = Field(min_length=1)
    scientific_config_digest: StrictStr = Field(min_length=1)
    descriptor: FinalTestSplitDescriptor
    examples: tuple[ExampleIdentity, ...]

    @model_validator(mode="after")
    def _validate_contents(self) -> _FinalTestPayload:
        _validate_examples(self.descriptor, self.examples)
        return self


class FinalTestHandle:
    """Opaque access token; examples are unavailable until an approved gate opens it."""

    __slots__ = ("__payload",)

    def __init__(self, payload: _FinalTestPayload) -> None:
        self.__payload = payload

    @property
    def handle_id(self) -> str:
        return self.__payload.handle_id

    def open(
        self,
        candidate: FrozenCandidateSelection,
        checkpoint: FrozenCheckpointSelection,
        restoration: RestorationReceipt,
    ) -> FinalTestView:
        """Materialize final data only for mutually matching lifecycle receipts."""

        payload = self.__payload
        if checkpoint.candidate_selection_id != candidate.frozen_selection_id:
            raise ValueError(
                "checkpoint selection does not belong to the frozen candidate"
            )
        if checkpoint.checkpoint.candidate_id != candidate.candidate_id:
            raise ValueError("checkpoint candidate does not match the frozen candidate")
        if checkpoint.selector is not candidate.selector:
            raise ValueError("checkpoint selector does not match the frozen candidate")
        if checkpoint.method_id != candidate.method_id:
            raise ValueError("checkpoint method does not match the frozen candidate")
        if (
            checkpoint.checkpoint.scientific_config_digest
            != candidate.scientific_config_digest
        ):
            raise ValueError(
                "checkpoint configuration does not match the frozen candidate"
            )
        if restoration.candidate_selection_id != candidate.frozen_selection_id:
            raise ValueError("restoration does not belong to the frozen candidate")
        if restoration.checkpoint != checkpoint.checkpoint:
            raise ValueError("restoration does not match the selected checkpoint")
        if payload.run_id != checkpoint.checkpoint.run_id:
            raise ValueError("final-test handle run does not match selected checkpoint")
        if payload.candidate_id != candidate.candidate_id:
            raise ValueError(
                "final-test handle candidate does not match frozen candidate"
            )
        if payload.scientific_config_digest != candidate.scientific_config_digest:
            raise ValueError("final-test handle configuration does not match selection")

        authorization_id = f"final-access:{payload.handle_id}:{restoration.receipt_id}"
        return FinalTestView(
            descriptor=payload.descriptor,
            examples=payload.examples,
            authorization_id=authorization_id,
            run_id=payload.run_id,
            candidate_id=payload.candidate_id,
            scientific_config_digest=payload.scientific_config_digest,
            checkpoint_id=checkpoint.checkpoint.checkpoint_id,
            epoch=checkpoint.checkpoint.epoch,
            method_id=checkpoint.method_id,
            seed=checkpoint.decision.seed,
        )


def issue_final_test_handle(
    *,
    handle_id: str,
    run_id: str,
    candidate_id: str,
    scientific_config_digest: str,
    descriptor: FinalTestSplitDescriptor,
    examples: tuple[ExampleIdentity, ...],
) -> FinalTestHandle:
    """Issue an opaque handle without exposing the final examples to ordinary code."""

    return FinalTestHandle(
        _FinalTestPayload(
            handle_id=handle_id,
            run_id=run_id,
            candidate_id=candidate_id,
            scientific_config_digest=scientific_config_digest,
            descriptor=descriptor,
            examples=examples,
        )
    )


def validate_cmnist_repeated_validation_views(
    views: tuple[ValidationView, ValidationView, ValidationView],
) -> None:
    """Validate the shared-source identity contract of CMNIST validation views."""

    if any(type(view) is not ValidationView for view in views):
        raise TypeError("repeated validation checks accept ValidationView values only")
    names = tuple(view.descriptor.name for view in views)
    if names[:2] != IN_DOMAIN_VALIDATION_NAMES or (
        names[2] not in HELD_OUT_VALIDATION_NAMES
    ):
        raise ValueError(
            "validation views must be ordered as val_e01, val_e02, then the held-out "
            f"rendering, not {names!r}"
        )
    dataset_ids = {view.descriptor.dataset_id for view in views}
    if dataset_ids != {"cmnist"}:
        raise ValueError("CMNIST validation views must share dataset_id='cmnist'")
    manifest_ids = {view.descriptor.manifest_id for view in views}
    if len(manifest_ids) != 1:
        raise ValueError("CMNIST validation views must share one manifest identity")
    source_partitions = {view.descriptor.source_partition_id for view in views}
    if len(source_partitions) != 1:
        raise ValueError("CMNIST validation views must share one source partition")
    source_orders = {
        tuple(example.source_id for example in view.examples) for view in views
    }
    if len(source_orders) != 1:
        raise ValueError("CMNIST validation views must share ordered source identities")
    example_ids = [example.example_id for view in views for example in view.examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError(
            "CMNIST validation renderings require distinct example identities"
        )
    view_ids = {view.descriptor.view_id for view in views}
    if len(view_ids) != len(views):
        raise ValueError(
            "CMNIST validation renderings require distinct view identities"
        )
