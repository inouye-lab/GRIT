"""Shared frozen-feature linear-probe methods, training, and checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import ClassVar, Literal, Protocol, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field, StrictInt, StrictStr

from grit.config import LinearProbeTrainingConfig
from grit.methods.checkpoints import CheckpointStore, StoredCheckpoint
from grit.methods.groupdro import (
    GroupDroObjective,
    group_balanced_epoch_indices,
)
from grit.methods.invariance import (
    annealed_penalty_weight,
    environment_balanced_epoch_batches,
    irmv1_objective,
    vrex_objective,
)
from grit.methods.projection import FittedLinearProjection
from grit.methods.training_state import LinearProbeState
from grit.methods.types import MethodId
from grit.schemas import SeedStage, StrictBoundaryModel
from grit.selection.cmnist import (
    CheckpointIdentity,
    ValidationMetricRecord,
    ValidationSplitName,
)
from grit.tracking import RunTracker

__all__ = ["LinearProbeState"]

# The epoch callback receives the stored state, its identity, and the epoch's mean
# training objective.
EpochValidation: TypeAlias = Callable[
    ["LinearProbeAlgorithm", CheckpointIdentity, float], None
]


class LinearProbeAlgorithm:
    """Own a fixed-class linear model, Adam, and one bounded update operation."""

    # The optimizer this algorithm builds; the candidate's training config must agree.
    optimizer_name: ClassVar[Literal["adam"]] = "adam"

    def __init__(
        self,
        config: LinearProbeTrainingConfig,
        *,
        model_seed: int,
        projection: FittedLinearProjection | None,
        num_classes: int = 2,
        pair_differences: torch.Tensor | None = None,
        consistency_weight: float = 0.0,
        feature_dim: int = 512,
    ) -> None:
        if num_classes <= 1:
            raise ValueError("linear probe requires at least two output classes")
        if config.optimizer != self.optimizer_name:
            raise ValueError(
                f"{type(self).__name__} trains with {self.optimizer_name}, "
                f"not {config.optimizer}"
            )
        if feature_dim <= 0:
            raise ValueError("linear probe feature dimension must be positive")
        self._feature_dim = feature_dim
        self._model = torch.nn.Linear(feature_dim, num_classes, bias=True, device="cpu")
        generator = torch.Generator(device="cpu").manual_seed(model_seed)
        bound = 1.0 / math.sqrt(feature_dim)
        weight = (
            torch.rand(
                (num_classes, feature_dim), generator=generator, dtype=torch.float32
            )
            .mul(2 * bound)
            .sub(bound)
        )
        bias = (
            torch.rand((num_classes,), generator=generator, dtype=torch.float32)
            .mul(2 * bound)
            .sub(bound)
        )
        with torch.no_grad():
            self._model.weight.copy_(weight)
            self._model.bias.copy_(bias)
        self._learning_rate = float(config.learning_rate)
        self._weight_decay = float(config.weight_decay)
        self._optimizer = self._new_optimizer()
        self._projection_basis = (
            None
            if projection is None
            else projection.basis.detach().cpu().to(torch.float32)
        )
        if not math.isfinite(consistency_weight) or consistency_weight < 0.0:
            raise ValueError("consistency weight must be finite and non-negative")
        if pair_differences is None and consistency_weight != 0.0:
            raise ValueError(
                "prediction consistency requires training-pair differences"
            )
        if pair_differences is not None:
            if (
                pair_differences.ndim != 2
                or pair_differences.shape[0] == 0
                or pair_differences.shape[1] != feature_dim
                or not bool(torch.isfinite(pair_differences).all())
            ):
                raise ValueError(
                    "pair differences must be a finite non-empty [N, features] matrix"
                )
            if projection is not None:
                raise ValueError(
                    "projection and consistency are exclusive interventions"
                )
        self._consistency_pair_differences = (
            None
            if pair_differences is None
            else pair_differences.detach().cpu().to(torch.float32).clone()
        )
        self._consistency_weight = consistency_weight
        self.diagnostic_history: dict[int, dict[str, float]] = {}

    def _new_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam(
            self._model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )

    def reset_optimizer(self) -> None:
        """Discard Adam's moments, keeping the model and the candidate's settings.

        Fishr and RDM reset once when their penalty activates, because Adam does not
        take kindly to the jump in gradient magnitude at that update.
        """

        self._optimizer = self._new_optimizer()

    @property
    def num_classes(self) -> int:
        return int(self._model.out_features)

    def prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        """The classifier's actual input; methods that need it must not re-derive it."""

        return self._prepare(features)

    def update(self, features: torch.Tensor, targets: torch.Tensor) -> float:
        """Perform the algorithm-owned mutation for one trainer-provided batch."""

        return self.update_with_objective(
            features, targets, lambda losses: losses.mean()
        )

    def update_with_objective(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        objective: Callable[[torch.Tensor], torch.Tensor],
        *,
        consistency_scale: float = 1.0,
    ) -> float:
        """Update from an algorithm-specific reduction of per-example losses."""

        def logits_objective(
            logits: torch.Tensor, prepared_targets: torch.Tensor
        ) -> torch.Tensor:
            per_example = torch.nn.functional.cross_entropy(
                logits, prepared_targets, reduction="none"
            )
            return objective(per_example)

        return self.update_with_logits_objective(
            features,
            targets,
            logits_objective,
            consistency_scale=consistency_scale,
        )

    def update_with_logits_objective(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        consistency_scale: float = 1.0,
    ) -> float:
        """Update from logits, then add consistency outside supervised statistics.

        ``consistency_scale`` applies the base objective's complete-loss rescaling;
        IRMv1 and V-REx pass ``1 / max(1, current_penalty_weight)``.
        """

        self._model.train()
        prepared = self._prepare(features)
        prepared_targets = targets.detach().cpu().to(torch.int64)
        self._optimizer.zero_grad(set_to_none=True)
        logits = self._model(prepared)
        loss = objective(logits, prepared_targets)
        if (
            self._consistency_pair_differences is not None
            and self._consistency_weight != 0.0
        ):
            loss = loss + (
                consistency_scale
                * self._consistency_weight
                * prediction_consistency_penalty(
                    self._model.weight, self._consistency_pair_differences
                )
            )
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("linear-probe objective must return one finite scalar")
        backward = cast(CallableWithoutArguments, loss.backward)
        step = cast(CallableWithoutArguments, self._optimizer.step)
        backward()
        step()
        return float(loss.detach().item())

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Run side-effect-free evaluation through the same input transform."""

        was_training = self._model.training
        self._model.eval()
        try:
            with torch.inference_mode():
                return self._model(self._prepare(features)).argmax(dim=1)
        finally:
            self._model.train(was_training)

    def mean_loss(self, features: torch.Tensor, targets: torch.Tensor) -> float:
        """Side-effect-free mean cross-entropy of the current inference state."""

        was_training = self._model.training
        self._model.eval()
        try:
            with torch.inference_mode():
                logits = self._model(self._prepare(features))
                loss = torch.nn.functional.cross_entropy(
                    logits, targets.detach().cpu().to(torch.int64)
                )
                return float(loss.item())
        finally:
            self._model.train(was_training)

    def training_diagnostics(self) -> dict[str, float]:
        """The unweighted prediction-consistency penalty on the training pairs."""

        differences = self._consistency_pair_differences
        if differences is None:
            return {}
        with torch.no_grad():
            return pair_penalty_diagnostics(differences, self._model.weight)

    def capture_inference_state(self) -> LinearProbeState:
        return LinearProbeState(
            weight=self._model.weight.detach().cpu().clone(),
            bias=self._model.bias.detach().cpu().clone(),
            projection_basis=(
                torch.empty((self._feature_dim, 0), dtype=torch.float32)
                if self._projection_basis is None
                else self._projection_basis.clone()
            ),
        )

    def restore_inference_state(self, state: LinearProbeState) -> None:
        if (
            state.weight.shape != self._model.weight.shape
            or state.bias.shape != self._model.bias.shape
        ):
            raise ValueError("linear checkpoint state has incompatible shapes")
        if state.projection_basis is not None:
            basis = state.projection_basis.detach().cpu().to(torch.float32)
            if (
                basis.ndim != 2
                or basis.shape[0] != self._feature_dim
                or basis.shape[1] > self._feature_dim
                or not bool(torch.isfinite(basis).all())
            ):
                raise ValueError("checkpoint projection basis has incompatible shape")
            if not torch.allclose(
                basis.T @ basis,
                torch.eye(int(basis.shape[1]), dtype=torch.float32),
                atol=1e-4,
                rtol=1e-4,
            ):
                raise ValueError("checkpoint projection basis must be orthonormal")
            self._projection_basis = basis.clone()
        with torch.no_grad():
            self._model.weight.copy_(state.weight.to(torch.float32))
            self._model.bias.copy_(state.bias.to(torch.float32))

    def _prepare(self, features: torch.Tensor) -> torch.Tensor:
        prepared = features.detach().cpu().to(torch.float32)
        if prepared.ndim != 2 or int(prepared.shape[1]) != self._feature_dim:
            raise ValueError(
                f"linear probe requires feature shape [N, {self._feature_dim}]"
            )
        if self._projection_basis is not None:
            if not bool(torch.isfinite(prepared).all()):
                raise ValueError("projection inputs must be finite")
            if self._projection_basis.shape[1]:
                basis = self._projection_basis
                prepared = prepared - (prepared @ basis) @ basis.transpose(0, 1)
        return prepared


def prediction_consistency_penalty(
    weight: torch.Tensor, pair_differences: torch.Tensor
) -> torch.Tensor:
    """Mean over pairs, sum over output coordinates; classifier bias cancels."""

    if pair_differences.ndim != 2 or pair_differences.shape[0] == 0:
        raise ValueError("prediction consistency requires a non-empty pair matrix")
    return (pair_differences @ weight.transpose(0, 1)).square().sum(dim=1).mean()


def pair_penalty_diagnostics(
    response: torch.Tensor, head_weight: torch.Tensor
) -> dict[str, float]:
    """The unweighted prediction-consistency penalty at one head input."""

    return {
        "pairs/prediction_consistency": float(
            prediction_consistency_penalty(head_weight, response)
        )
    }


class LinearProbeTrainingMethod(Protocol):
    """One method's batching, objective, and projection for a linear-probe run."""

    @property
    def method_id(self) -> MethodId: ...

    @property
    def projection(self) -> FittedLinearProjection | None: ...

    @property
    def projection_rank(self) -> int | None: ...

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]: ...

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float: ...

    def build_algorithm(
        self,
        config: LinearProbeTrainingConfig,
        *,
        model_seed: int,
        num_classes: int,
    ) -> LinearProbeAlgorithm: ...

    def checkpoint_state(self, algorithm: LinearProbeAlgorithm) -> LinearProbeState:
        """The state saved and validated at an epoch boundary."""
        ...


class _HasProjection(Protocol):
    @property
    def projection(self) -> FittedLinearProjection | None: ...


class LinearProbeMethodDefaults:
    """Plain linear probe over the method's projection; live weights are checkpoints."""

    def build_algorithm(
        self: _HasProjection,
        config: LinearProbeTrainingConfig,
        *,
        model_seed: int,
        num_classes: int,
    ) -> LinearProbeAlgorithm:
        return LinearProbeAlgorithm(
            config,
            model_seed=model_seed,
            projection=self.projection,
            num_classes=num_classes,
        )

    def checkpoint_state(self, algorithm: LinearProbeAlgorithm) -> LinearProbeState:
        return algorithm.capture_inference_state()


class FeatureTableLike(Protocol):
    """Minimal frozen-feature table consumed by shared training/evaluation."""

    @property
    def name(self) -> str: ...

    @property
    def source_ids(self) -> tuple[str, ...]: ...

    @property
    def features(self) -> torch.Tensor: ...

    @property
    def targets(self) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class OrdinaryLinearProbeMethod(LinearProbeMethodDefaults):
    """ERM-style minibatches, optionally preceded by a frozen GRIT projection."""

    method_id: MethodId
    projection: FittedLinearProjection | None
    projection_rank: int | None

    def __post_init__(self) -> None:
        if self.method_id not in ("erm", "grit"):
            raise ValueError("ordinary linear-probe updates support only ERM and GRIT")
        projected = self.projection is not None and self.projection_rank is not None
        if self.method_id == "erm" and (
            self.projection is not None or self.projection_rank is not None
        ):
            raise ValueError("ERM cannot use a projection")
        if self.method_id == "grit" and not projected:
            raise ValueError("GRIT requires a fitted projection and rank")

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        return random_epoch_batches(row_count, batch_size, generator)

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        return algorithm.update(features[rows], targets[rows])


@dataclass(slots=True)
class GroupDroLinearProbeMethod(LinearProbeMethodDefaults):
    """Group-balanced epochs and the stateful GroupDRO robust objective."""

    group_ids: torch.Tensor
    group_count: int
    step_size: float
    method_id: Literal["groupdro"] = field(init=False, default="groupdro")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)
    _objective: GroupDroObjective = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.group_ids = self.group_ids.detach().cpu().to(torch.int64)
        self._objective = GroupDroObjective(
            group_count=self.group_count,
            step_size=self.step_size,
        )

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        require_aligned_ids(self.group_ids, row_count, "group")
        permutation = group_balanced_epoch_indices(
            self.group_ids,
            group_count=self.group_count,
            generator=generator,
        )
        return batch_indices(permutation, batch_size)

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        batch_groups = self.group_ids[rows]
        return algorithm.update_with_objective(
            features[rows],
            targets[rows],
            lambda losses: self._objective(losses, batch_groups),
        )


@dataclass(slots=True)
class RexLinearProbeMethod(LinearProbeMethodDefaults):
    """Environment-balanced epochs and an annealed V-REx objective."""

    environment_ids: torch.Tensor
    environment_count: int
    penalty_weight: float
    penalty_anneal_updates: int
    method_id: Literal["rex"] = field(init=False, default="rex")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)
    _update_count: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self.environment_ids = self.environment_ids.detach().cpu().to(torch.int64)

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        require_aligned_ids(self.environment_ids, row_count, "environment")
        return environment_balanced_epoch_batches(
            self.environment_ids,
            environment_count=self.environment_count,
            batch_size=batch_size,
            generator=generator,
        )

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        penalty_weight = annealed_penalty_weight(
            self.penalty_weight,
            anneal_updates=self.penalty_anneal_updates,
            update_count=self._update_count,
        )
        batch_environments = self.environment_ids[rows]
        loss = algorithm.update_with_objective(
            features[rows],
            targets[rows],
            partial(
                vrex_objective,
                environment_ids=batch_environments,
                environment_count=self.environment_count,
                penalty_weight=penalty_weight,
            ),
            consistency_scale=1.0 / max(1.0, penalty_weight),
        )
        self._update_count += 1
        return loss


@dataclass(slots=True)
class IrmLinearProbeMethod(LinearProbeMethodDefaults):
    """Environment-balanced epochs and an annealed IRMv1 objective."""

    environment_ids: torch.Tensor
    environment_count: int
    penalty_weight: float
    penalty_anneal_updates: int
    method_id: Literal["irm"] = field(init=False, default="irm")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)
    _update_count: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self.environment_ids = self.environment_ids.detach().cpu().to(torch.int64)

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        require_aligned_ids(self.environment_ids, row_count, "environment")
        return environment_balanced_epoch_batches(
            self.environment_ids,
            environment_count=self.environment_count,
            batch_size=batch_size,
            generator=generator,
        )

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        penalty_weight = annealed_penalty_weight(
            self.penalty_weight,
            anneal_updates=self.penalty_anneal_updates,
            update_count=self._update_count,
        )
        batch_environments = self.environment_ids[rows]
        loss = algorithm.update_with_logits_objective(
            features[rows],
            targets[rows],
            partial(
                irmv1_objective,
                environment_ids=batch_environments,
                environment_count=self.environment_count,
                penalty_weight=penalty_weight,
            ),
            consistency_scale=1.0 / max(1.0, penalty_weight),
        )
        self._update_count += 1
        return loss


def random_epoch_batches(
    row_count: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    return batch_indices(torch.randperm(row_count, generator=generator), batch_size)


def batch_indices(
    permutation: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, ...]:
    if batch_size <= 0:
        raise ValueError("linear-probe batch size must be positive")
    return tuple(
        permutation[start : start + batch_size]
        for start in range(0, len(permutation), batch_size)
    )


def require_aligned_ids(ids: torch.Tensor, row_count: int, kind: str) -> None:
    if ids.ndim != 1 or int(ids.shape[0]) != row_count:
        raise ValueError(f"training-{kind} IDs must align with training rows")


class InMemoryLinearCheckpointStore(CheckpointStore[LinearProbeState]):
    """Epoch-candidate store; only the selected state is persisted by the runner."""

    def __init__(self, store_id: str) -> None:
        self._store_id = store_id
        self._states: dict[str, StoredCheckpoint[LinearProbeState]] = {}

    @property
    def store_id(self) -> str:
        return self._store_id

    def save(self, identity: CheckpointIdentity, state: LinearProbeState) -> None:
        if identity.checkpoint_id in self._states:
            raise ValueError("checkpoint identity was already stored")
        self._states[identity.checkpoint_id] = StoredCheckpoint(identity, state)

    def load(self, checkpoint_id: str) -> StoredCheckpoint[LinearProbeState]:
        try:
            return self._states[checkpoint_id]
        except KeyError as error:
            raise KeyError(f"checkpoint is unavailable: {checkpoint_id}") from error


@dataclass(frozen=True, slots=True)
class TrainedLinearProbeRun:
    run_id: str
    candidate_id: str
    method_id: MethodId
    seed_stage: SeedStage
    seed: int
    projection_rank: int | None
    validation_metrics: tuple[ValidationMetricRecord, ...]
    store: InMemoryLinearCheckpointStore
    algorithm: LinearProbeAlgorithm
    epoch_losses: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TrainedLinearProbeCore:
    """Dataset-neutral output of the shared linear-probe epoch lifecycle."""

    store: InMemoryLinearCheckpointStore
    algorithm: LinearProbeAlgorithm
    epoch_losses: tuple[float, ...]


def train_linear_probe_epochs(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    config: LinearProbeTrainingConfig,
    *,
    run_id: str,
    candidate_id: str,
    scientific_config_digest: str,
    seed: int,
    method: LinearProbeTrainingMethod,
    validate_epoch: EpochValidation,
    num_classes: int = 2,
) -> TrainedLinearProbeCore:
    """Train one method and hand each saved epoch to dataset-specific validation.

    ``validate_epoch`` also receives the epoch's mean training objective so a caller
    can mirror it without recomputing anything.
    """

    features = train_features.detach().cpu().to(torch.float32)
    targets = train_targets.detach().cpu().to(torch.int64)
    if features.ndim != 2 or int(features.shape[0]) == 0:
        raise ValueError("linear probe training features must be a non-empty matrix")
    if targets.ndim != 1 or int(targets.shape[0]) != int(features.shape[0]):
        raise ValueError("linear probe training targets must align with features")
    torch.use_deterministic_algorithms(True)
    algorithm = method.build_algorithm(config, model_seed=seed, num_classes=num_classes)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    store = InMemoryLinearCheckpointStore(store_id=f"memory:{run_id}")
    epoch_losses: list[float] = []
    row_count = int(features.shape[0])
    for epoch in range(1, config.max_epochs + 1):
        row_batches = method.epoch_batches(
            row_count=row_count,
            batch_size=int(config.batch_size),
            generator=generator,
        )
        if not row_batches:
            raise ValueError("linear-probe method produced an empty epoch")
        batch_losses = tuple(
            method.update(algorithm, features, targets, rows) for rows in row_batches
        )
        train_objective = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(train_objective)
        identity = CheckpointIdentity(
            checkpoint_id=f"checkpoint:{run_id}:epoch:{epoch}",
            candidate_id=candidate_id,
            run_id=run_id,
            scientific_config_digest=scientific_config_digest,
            epoch=epoch,
        )
        # Validate exactly the stored checkpoint state, then hand the live state back.
        diagnostics = algorithm.training_diagnostics()
        if diagnostics:
            algorithm.diagnostic_history[epoch] = diagnostics
        state = method.checkpoint_state(algorithm)
        store.save(identity, state)
        live = algorithm.capture_inference_state()
        algorithm.restore_inference_state(state)
        validate_epoch(algorithm, identity, train_objective)
        algorithm.restore_inference_state(live)
    return TrainedLinearProbeCore(
        store=store,
        algorithm=algorithm,
        epoch_losses=tuple(epoch_losses),
    )


def train_linear_probe(
    training_tables: tuple[FeatureTableLike, FeatureTableLike],
    validation_tables: tuple[FeatureTableLike, FeatureTableLike, FeatureTableLike],
    config: LinearProbeTrainingConfig,
    *,
    run_id: str,
    candidate_id: str,
    scientific_config_digest: str,
    seed_stage: SeedStage,
    seed: int,
    method: LinearProbeTrainingMethod,
    num_classes: int = 2,
    tracker: RunTracker | None = None,
) -> TrainedLinearProbeRun:
    """Run trainer-owned epoch/batch iteration and emit validation every epoch.

    ``tracker`` only mirrors measurements that were computed anyway.
    """

    train_features = torch.cat([table.features for table in training_tables], dim=0).to(
        torch.float32
    )
    train_targets = torch.cat([table.targets for table in training_tables], dim=0).to(
        torch.int64
    )
    metrics: list[ValidationMetricRecord] = []

    def validate_epoch(
        algorithm: LinearProbeAlgorithm,
        identity: CheckpointIdentity,
        train_objective: float,
    ) -> None:
        records = _validation_metrics(
            algorithm,
            validation_tables,
            run_id=run_id,
            candidate_id=candidate_id,
            method_id=method.method_id,
            scientific_config_digest=scientific_config_digest,
            seed_stage=seed_stage,
            seed=seed,
            checkpoint_id=identity.checkpoint_id,
            epoch=identity.epoch,
            projection_rank=method.projection_rank,
        )
        metrics.extend(records)
        if tracker is None:
            return
        # `train_objective` is not comparable across methods: every penalty is already
        # inside it, so it is never labeled cross-entropy.
        values = {"train_objective": train_objective}
        values.update(algorithm.training_diagnostics())
        values.update(
            {
                f"validation/{record.split_name}_accuracy": float(record.value)
                for record in records
            }
        )
        tracker.log_epoch(identity.epoch, values)

    core = train_linear_probe_epochs(
        train_features,
        train_targets,
        config,
        run_id=run_id,
        candidate_id=candidate_id,
        scientific_config_digest=scientific_config_digest,
        seed=seed,
        method=method,
        validate_epoch=validate_epoch,
        num_classes=num_classes,
    )
    return TrainedLinearProbeRun(
        run_id=run_id,
        candidate_id=candidate_id,
        method_id=method.method_id,
        seed_stage=seed_stage,
        seed=seed,
        projection_rank=method.projection_rank,
        validation_metrics=tuple(metrics),
        store=core.store,
        algorithm=core.algorithm,
        epoch_losses=core.epoch_losses,
    )


def evaluate_accuracy(
    algorithm: LinearProbeAlgorithm, table: FeatureTableLike
) -> float:
    predictions = algorithm.predict(table.features)
    return float((predictions == table.targets).to(torch.float64).mean().item())


def _validation_metrics(
    algorithm: LinearProbeAlgorithm,
    tables: tuple[FeatureTableLike, FeatureTableLike, FeatureTableLike],
    *,
    run_id: str,
    candidate_id: str,
    method_id: MethodId,
    scientific_config_digest: str,
    seed_stage: SeedStage,
    seed: int,
    checkpoint_id: str,
    epoch: int,
    projection_rank: int | None,
) -> tuple[ValidationMetricRecord, ...]:
    return tuple(
        ValidationMetricRecord(
            record_id=f"metric:{run_id}:{checkpoint_id}:{table.name}",
            run_id=run_id,
            candidate_id=candidate_id,
            method_id=method_id,
            scientific_config_digest=scientific_config_digest,
            checkpoint_id=checkpoint_id,
            epoch=epoch,
            seed=seed,
            value=evaluate_accuracy(algorithm, table),
            sample_count=len(table.source_ids),
            metric_kind="validation",
            seed_stage=seed_stage,
            split_name=cast(ValidationSplitName, table.name),
            metric_name="accuracy",
            projection_rank=projection_rank,
        )
        for table in tables
    )


class LinearCheckpointFile(StrictBoundaryModel):
    relative_path: StrictStr
    digest: StrictStr
    shape: tuple[StrictInt, ...]
    dtype: Literal["float32"]


class PersistedLinearCheckpointManifest(StrictBoundaryModel):
    schema_version: Literal["grit.linear-checkpoint/v1"]
    store_id: StrictStr
    checkpoint: CheckpointIdentity
    weight: LinearCheckpointFile
    bias: LinearCheckpointFile
    projection_basis: LinearCheckpointFile | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class _TensorToNumpy(Protocol):
    def __call__(self, *, force: bool = False) -> NDArray[np.float32]: ...


class _TorchFromNumpy(Protocol):
    def __call__(self, array: NDArray[np.float32]) -> torch.Tensor: ...


_torch_from_numpy = cast(_TorchFromNumpy, torch.from_numpy)


class CallableWithoutArguments(Protocol):
    def __call__(self) -> object: ...


class PersistedLinearCheckpointStore(CheckpointStore[LinearProbeState]):
    """Read one selected inference checkpoint from its narrow on-disk format."""

    def __init__(self, root: Path) -> None:
        manifest_path = root / "manifest.json"
        self._root = root
        self._manifest = PersistedLinearCheckpointManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )

    @property
    def store_id(self) -> str:
        return self._manifest.store_id

    @property
    def manifest(self) -> PersistedLinearCheckpointManifest:
        return self._manifest

    def load(self, checkpoint_id: str) -> StoredCheckpoint[LinearProbeState]:
        if checkpoint_id != self._manifest.checkpoint.checkpoint_id:
            raise KeyError(f"checkpoint is unavailable: {checkpoint_id}")
        weight = _load_checkpoint_array(self._root, self._manifest.weight)
        bias = _load_checkpoint_array(self._root, self._manifest.bias)
        return StoredCheckpoint(
            self._manifest.checkpoint,
            LinearProbeState(
                weight=weight,
                bias=bias,
                projection_basis=(
                    None
                    if self._manifest.projection_basis is None
                    else _load_checkpoint_array(
                        self._root, self._manifest.projection_basis
                    )
                ),
            ),
        )


def persist_training_diagnostics(
    algorithm: LinearProbeAlgorithm, run_root: Path
) -> None:
    """Keep pilot scaling evidence even when the optional tracking mirror is off."""
    if algorithm.diagnostic_history:
        (run_root / "representation-diagnostics.json").write_text(
            json.dumps(algorithm.diagnostic_history, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )


def persist_selected_linear_checkpoint(
    stored: StoredCheckpoint[LinearProbeState],
    output_dir: Path,
) -> PersistedLinearCheckpointManifest:
    """Persist only the selected inference state; no resume guarantee is implied."""

    output_dir.mkdir(parents=True, exist_ok=False)
    weight = _write_checkpoint_array(output_dir, "weight.npy", stored.state.weight)
    bias = _write_checkpoint_array(output_dir, "bias.npy", stored.state.bias)
    manifest = PersistedLinearCheckpointManifest(
        schema_version="grit.linear-checkpoint/v1",
        store_id=f"linear-checkpoint:{stored.identity.checkpoint_id}",
        checkpoint=stored.identity,
        weight=weight,
        bias=bias,
        projection_basis=(
            None
            if stored.state.projection_basis is None
            else _write_checkpoint_array(
                output_dir, "projection_basis.npy", stored.state.projection_basis
            )
        ),
    )
    (output_dir / "manifest.json").write_text(
        manifest.canonical_json() + "\n", encoding="utf-8"
    )
    return manifest


def _write_checkpoint_array(
    root: Path,
    name: str,
    tensor: torch.Tensor,
) -> LinearCheckpointFile:
    path = root / name
    contiguous = tensor.detach().cpu().to(torch.float32).contiguous()
    to_numpy = cast(_TensorToNumpy, contiguous.numpy)
    array = to_numpy()
    np.save(path, array, allow_pickle=False)
    return LinearCheckpointFile(
        relative_path=name,
        digest=_file_digest(path),
        shape=tuple(int(value) for value in contiguous.shape),
        dtype="float32",
    )


def _load_checkpoint_array(root: Path, file: LinearCheckpointFile) -> torch.Tensor:
    path = root / file.relative_path
    if _file_digest(path) != file.digest:
        raise ValueError("persisted linear checkpoint digest mismatch")
    loaded = np.load(path, allow_pickle=False)
    loaded_shape = cast(tuple[int, ...], loaded.shape)
    if loaded_shape != file.shape:
        raise ValueError("persisted linear checkpoint shape mismatch")
    if str(loaded.dtype) != file.dtype:
        raise ValueError("persisted linear checkpoint dtype mismatch")
    copied = cast(NDArray[np.float32], np.array(loaded, copy=True))
    return _torch_from_numpy(copied)


def _file_digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
