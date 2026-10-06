"""UrbanCars collection-geometry runner (App. G): checks, optimizer choice, v1 search.

Three config kinds share one fit worker:

- ``check`` (validation only): C4 energy calibration and count table, C2 reference
  geometry, C1 ERM shortcut reliance (validation-tuned, 10 seeds), C3 the large-pair
  reference projection, and the bank-0 one-seed pilot. Writes ``check-report.json``.
- ``optimizer`` (validation only): on bank 0 at k = 256, each method's LR x WD
  setting maximizing the equal-weight average over the four v1 collections of the
  best candidate's primary validation score. Writes ``optimizer.json``, which also
  holds the full-grid optimizer check. The chosen settings are then written into
  the main config by hand.
- ``main``: per selection unit (method, collection, bank) and selector,
  3 tuning seeds, the top 3 confirmed with 2 more, then 10 final seeds; matched-rank
  rows fix the rank. ``winners.json`` is frozen first and test is opened only by
  ``--unseal-test <hash>``.

Shared definitions are in ``grit.geometry``. The check labels C1-C4 are listed in
``results/README.md``.
"""

# Experiment JSON records are plain dictionaries validated by construction.
# pyright: reportAny=false, reportExplicitAny=false
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from statistics import fmean, pstdev
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field, StrictInt, model_validator

from grit.config import LinearProbeTrainingConfig, SeedSets
from grit.data.collection_geometry import Real
from grit.data.urbancars import UrbanCarsDataset, load_urbancars_dataset
from grit.features.urbancars import (
    load_urbancars_test_table,
    load_urbancars_tuning_tables,
)
from grit.geometry import (
    GROUP_COUNT,
    SELECTORS,
    FloatArray,
    Selector,
    captured_energy,
    classifier_contrast,
    fit_projection,
    group_metrics,
    label_energy,
    minority_count,
    o_positions,
    paired_interval,
    realized_share,
    removed_basis,
    residual_sensitivity,
    selector_score,
    to_array,
    to_tensor,
    top_eigenvectors,
    training_group_weights,
)
from grit.jsonio import read_json, write_json
from grit.methods.interventions import ComposedLinearProbeMethod
from grit.methods.projection import FittedLinearProjection
from grit.methods.training import (
    LinearProbeAlgorithm,
    LinearProbeTrainingMethod,
    OrdinaryLinearProbeMethod,
    train_linear_probe_epochs,
)
from grit.methods.training_state import LinearProbeState
from grit.paths import expand_config_path
from grit.schemas import StrictBoundaryModel, canonical_digest_value
from grit.search.execution import (
    SEQUENTIAL_EXECUTION,
    ExecutionOptions,
    execution_scope,
    run_parallel,
)
from grit.search.plan import current_code_provenance, load_search_yaml
from grit.selection.cmnist import CheckpointIdentity

PositiveInt = Annotated[StrictInt, Field(gt=0)]
# `erm_matched` is plain ERM trained with GRIT's frozen optimizer: a labelled
# matched-optimizer control beside the planned (robustly selected) ERM row.
Method: TypeAlias = Literal["erm", "erm_matched", "grit", "consistency", "reference"]
Stage: TypeAlias = Literal["tuning", "confirmation", "final", "c1", "c3", "pilot"]
ENFORCEMENT_TOLERANCE = 1e-6
# C1 feasibility: each validation shift accuracy at least 3 points below in-dist.
C1_THRESHOLD = 0.03
# C4: Count-50 is a control when E_B / E_O lies outside this range.
COUNT_50_RANGE = (0.67, 1.5)
# V_F for residual sensitivity: the top eigenvectors of the reference Sigma_F
# holding this fraction of its energy.
FACTOR_ENERGY = 0.9


class CollectionRule(StrictBoundaryModel):
    """O's energy share ``share`` or a count split ``count_fraction``."""

    name: Annotated[str, Field(pattern=r"^[a-z0-9_]+$")]
    share: Annotated[Real, Field(ge=0.0, le=1.0)] | None = None
    count_fraction: Annotated[Real, Field(ge=0.0, le=1.0)] | None = None

    @model_validator(mode="after")
    def _one_rule(self) -> CollectionRule:
        if (self.share is None) == (self.count_fraction is None):
            raise ValueError(f"{self.name}: set exactly one of share, count_fraction")
        return self


class OptimizerSetting(StrictBoundaryModel):
    learning_rate: Annotated[Real, Field(gt=0.0)]
    weight_decay: Annotated[Real, Field(ge=0.0)]


class UrbanCarsSearchConfig(StrictBoundaryModel):
    dataset: Literal["urbancars"]
    experiment_name: Annotated[str, Field(min_length=1)]
    cache_root: str
    output_root: str
    kind: Literal["check", "optimizer", "main"]
    seeds: SeedSets
    budget: PositiveInt = 256
    banks: tuple[Annotated[StrictInt, Field(ge=0, le=2)], ...] = (0, 1, 2)
    collections: tuple[CollectionRule, ...]
    # Main only: C4's decision whether Count-50 runs as a control.
    include_count_50: bool | None = None
    methods: tuple[Method, ...] = ("erm", "grit", "consistency", "reference")
    ranks: tuple[PositiveInt, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    matched_ranks: tuple[PositiveInt, ...] = (1, 2, 4, 8, 16)
    consistency_strengths: tuple[Annotated[Real, Field(gt=0.0)], ...] = (
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
        1000.0,
        1e4,
    )
    # Check and optimizer kinds: the LR x WD grid.
    learning_rates: tuple[Annotated[Real, Field(gt=0.0)], ...] = (
        1e-4,
        3e-4,
        1e-3,
        3e-3,
    )
    weight_decays: tuple[Annotated[Real, Field(ge=0.0)], ...] = (0.0, 1e-4)
    # Main only: the frozen shared optimizer per method (from optimizer.json).
    optimizers: dict[Method, OptimizerSetting] = {}
    batch_size: PositiveInt = 256
    max_epochs: PositiveInt = 100

    @model_validator(mode="after")
    def _consistent(self) -> UrbanCarsSearchConfig:
        names = [c.name for c in self.collections]
        if len(set(names)) != len(names):
            raise ValueError("collection names must be unique")
        if self.kind == "main":
            missing = [m for m in self.methods if m not in self.optimizers]
            if missing:
                raise ValueError(
                    f"main needs the frozen optimizer of {missing}: copy them from "
                    "optimizer.json"
                )
            if self.include_count_50 is None:
                raise ValueError("main needs include_count_50 from the C4 check")
        if self.kind == "optimizer" and self.banks != (0,):
            raise ValueError("the optimizer choice uses bank 0 only")
        return self

    def active_collections(self) -> tuple[CollectionRule, ...]:
        if self.kind == "main" and not self.include_count_50:
            return tuple(c for c in self.collections if c.name != "count_50")
        return self.collections

    def scientific(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"cache_root", "output_root"})


def load_config(path: Path) -> UrbanCarsSearchConfig:
    return UrbanCarsSearchConfig.model_validate_json(json.dumps(load_search_yaml(path)))


# Candidates.


@dataclass(frozen=True)
class Candidate:
    method: Method
    learning_rate: float
    weight_decay: float
    collection: str | None = None
    bank: int | None = None
    rank: int | None = None
    strength: float | None = None

    @property
    def variant(self) -> str:
        if self.collection is None:
            return self.method
        return f"{self.method}@{self.collection}-b{self.bank}"

    @property
    def candidate_id(self) -> str:
        parts = [self.variant]
        if self.rank is not None:
            parts.append(f"r{self.rank}")
        if self.strength is not None:
            parts.append(f"a{self.strength:g}")
        parts.append(f"lr{self.learning_rate:g}-wd{self.weight_decay:g}")
        return ":".join(parts)


# Data, built once per process.


def _pack(correct: NDArray[np.bool_]) -> str:
    return np.packbits(correct).tobytes().hex()


def _unpack(bits: str, n: int) -> NDArray[np.bool_]:
    return np.unpackbits(np.frombuffer(bytes.fromhex(bits), dtype=np.uint8))[:n].astype(
        np.bool_
    )


@dataclass(frozen=True)
class Collection:
    name: str
    bank: int
    k: int
    n_o: int
    differences: FloatArray

    @property
    def key(self) -> str:
        return f"{self.name}-b{self.bank}"


class Data:
    """Features, labels, groups, pair differences and reference geometry."""

    def __init__(self, config: UrbanCarsSearchConfig, *, test: bool = False) -> None:
        self.config = config
        root = Path(expand_config_path(config.cache_root))
        self.dataset: UrbanCarsDataset = load_urbancars_dataset(root)
        tables = load_urbancars_tuning_tables(root / "features", self.dataset)
        by_id = {r["image_id"]: r for r in self.dataset.manifest["images"]}
        construction = self.dataset.manifest["construction"]
        self.weights = training_group_weights(
            construction["b_ratio"], construction["o_ratio"]
        )

        def split(
            name: str,
        ) -> tuple[torch.Tensor, NDArray[np.int64], NDArray[np.int64]]:
            ids, features = tables[name] if name in tables else (None, None)
            assert ids is not None and features is not None
            labels = np.array([by_id[i]["label"] for i in ids], dtype=np.int64)
            groups = np.array([by_id[i]["group"] for i in ids], dtype=np.int64)
            return features, labels, groups

        self.train_x, self.train_y, _ = split("train")
        self.val_x, self.val_y, self.val_groups = split("validation")
        self.val_per_group = np.bincount(self.val_groups, minlength=GROUP_COUNT)
        train_ids = tables["train"][0]
        train_rows = {i: n for n, i in enumerate(train_ids)}
        train_f = to_array(self.train_x).astype(np.float64)
        partner = {
            f: dict(
                zip(
                    tables[f"partner_{f}"][0],
                    to_array(tables[f"partner_{f}"][1]).astype(np.float64),
                    strict=True,
                )
            )
            for f in ("b", "o")
        }

        def delta(anchor: str, factor: str) -> FloatArray:
            return (
                train_f[train_rows[anchor]]
                - partner[factor][f"partner-{factor}/{anchor}"]
            )

        self.delta = delta
        pools = self.dataset.manifest["anchor_pools"]
        self.pools = pools
        calibration = [p["anchor"] for p in pools["calibration"]]
        self.energy_b = float(
            np.mean([np.square(delta(a, "b")).sum() for a in calibration])
        )
        self.energy_o = float(
            np.mean([np.square(delta(a, "o")).sum() for a in calibration])
        )
        reference = [p["anchor"] for p in pools["reference"]]
        self.reference_b = np.stack([delta(a, "b") for a in reference])
        self.reference_o = np.stack([delta(a, "o") for a in reference])
        self.second_moments = {
            "b": self.reference_b.T @ self.reference_b / len(reference),
            "o": self.reference_o.T @ self.reference_o / len(reference),
        }
        self.factor_bases = {
            f: _energy_basis(m) for f, m in self.second_moments.items()
        }
        val_f = to_array(self.val_x).astype(np.float64)
        self.label_direction = val_f[self.val_y == 1].mean(0) - val_f[
            self.val_y == 0
        ].mean(0)
        self.digest = self.dataset.digest
        self.test: tuple[torch.Tensor, NDArray[np.int64], NDArray[np.int64]] | None = (
            None
        )
        if test:
            ids, features = load_urbancars_test_table(root / "features", self.dataset)
            self.test = (
                features,
                np.array([by_id[i]["label"] for i in ids], dtype=np.int64),
                np.array([by_id[i]["group"] for i in ids], dtype=np.int64),
            )

    def n_o(self, rule: CollectionRule, k: int) -> int:
        if rule.count_fraction is not None:
            return int(round(k * rule.count_fraction))
        assert rule.share is not None
        return minority_count(k, rule.share, self.energy_b, self.energy_o)

    def collection(self, rule: CollectionRule, bank: int) -> Collection | None:
        """The coupled construction on bank ``bank``; None when unreachable."""

        k = self.config.budget
        positions = self.pools[f"bank{bank}"][:k]
        if len(positions) < k:
            raise ValueError(f"bank {bank} has fewer than {k} positions")
        n_o = self.n_o(rule, k)
        if (rule.share or 0.0) > 0.0 and n_o == 0:
            return None
        draws = np.array([p["u"] for p in positions])
        is_o = o_positions(draws, k, n_o)
        differences = np.stack(
            [
                self.delta(p["anchor"], "o" if o else "b")
                for p, o in zip(positions, is_o, strict=True)
            ]
        )
        return Collection(rule.name, bank, k, n_o, differences)

    def is_o(self, collection: Collection) -> NDArray[np.bool_]:
        draws = np.array(
            [p["u"] for p in self.pools[f"bank{collection.bank}"][: collection.k]]
        )
        return o_positions(draws, collection.k, collection.n_o)

    def reference_differences(self) -> FloatArray:
        return np.concatenate([self.reference_b, self.reference_o])


def _energy_basis(second_moment: FloatArray) -> FloatArray:
    values, vectors = np.linalg.eigh(second_moment)
    values, vectors = values[::-1], vectors[:, ::-1]
    share = np.cumsum(values) / values.sum()
    rank = int(np.searchsorted(share, FACTOR_ENERGY) + 1)
    return vectors[:, :rank]


def effective_ranks(second_moment: FloatArray) -> dict[str, float]:
    values = np.clip(np.linalg.eigvalsh(second_moment)[::-1], 0.0, None)
    share = np.cumsum(values) / values.sum()
    return {
        "participation_ratio": float(values.sum() ** 2 / np.square(values).sum()),
        "rank_90": int(np.searchsorted(share, 0.9) + 1),
        "rank_99": int(np.searchsorted(share, 0.99) + 1),
    }


# Candidate grids.


def _optimizer_grid(config: UrbanCarsSearchConfig) -> list[tuple[float, float]]:
    return list(product(config.learning_rates, config.weight_decays))


def _optimizer_for(
    config: UrbanCarsSearchConfig, method: Method
) -> list[tuple[float, float]]:
    if config.kind == "main":
        setting = config.optimizers[method]
        return [(setting.learning_rate, setting.weight_decay)]
    return _optimizer_grid(config)


def candidates(config: UrbanCarsSearchConfig, data: Data) -> list[Candidate]:
    """Main and optimizer grids; GRIT ranks capped and deduplicated per collection."""

    result: list[Candidate] = []
    for method in config.methods:
        for lr, wd in _optimizer_for(config, method):
            if method in ("erm", "erm_matched"):
                result.append(Candidate(method, lr, wd))
            elif method == "reference":
                cap = _numerical_rank(data.reference_differences())
                for rank in sorted({min(r, cap) for r in config.ranks}):
                    result.append(Candidate("reference", lr, wd, rank=rank))
            else:
                for bank in config.banks:
                    for rule in config.active_collections():
                        c = data.collection(rule, bank)
                        if c is None:
                            continue
                        if method == "grit":
                            for rank in effective_ranks_of(c, config):
                                result.append(
                                    Candidate(
                                        "grit", lr, wd, rule.name, bank, rank=rank
                                    )
                                )
                        else:
                            for alpha in config.consistency_strengths:
                                result.append(
                                    Candidate(
                                        "consistency",
                                        lr,
                                        wd,
                                        rule.name,
                                        bank,
                                        strength=alpha,
                                    )
                                )
    return result


def _numerical_rank(differences: FloatArray) -> int:
    projection = fit_projection(
        differences, np.zeros_like(differences), 1, digest="rank"
    )
    return projection.diagnostics.numerical_rank


def effective_ranks_of(c: Collection, config: UrbanCarsSearchConfig) -> list[int]:
    cap = _numerical_rank(c.differences)
    return sorted({min(r, cap) for r in (*config.ranks, *config.matched_ranks)})


# Fits.


@dataclass(frozen=True)
class Task:
    candidate: Candidate
    stage: Stage
    seed: int


def task_path(root: Path, task: Task) -> Path:
    return (
        root / "runs" / task.stage / task.candidate.candidate_id / f"{task.seed}.json"
    )


def _projection(
    data: Data, candidate: Candidate
) -> tuple[FittedLinearProjection | None, Collection | None]:
    """The fitted projection (GRIT, reference) and the candidate's collection."""

    collection: Collection | None = None
    if candidate.collection is not None:
        rule = next(
            r for r in data.config.collections if r.name == candidate.collection
        )
        collection = data.collection(rule, candidate.bank or 0)
        assert collection is not None
    if candidate.method == "reference":
        differences = data.reference_differences()
    elif candidate.method == "grit":
        assert collection is not None
        differences = collection.differences
    else:
        return None, collection
    assert candidate.rank is not None
    projection = fit_projection(
        differences, np.zeros_like(differences), candidate.rank, digest=data.digest
    )
    if projection.diagnostics.effective_rank != candidate.rank:
        raise ValueError(f"{candidate.candidate_id}: rank exceeds the numerical rank")
    return projection, collection


def build_method(
    data: Data, candidate: Candidate
) -> tuple[LinearProbeTrainingMethod, FittedLinearProjection | None, Collection | None]:
    base = OrdinaryLinearProbeMethod("erm", projection=None, projection_rank=None)
    projection, collection = _projection(data, candidate)
    if candidate.method in ("erm", "erm_matched"):
        return base, None, None
    if candidate.method in ("grit", "reference"):
        assert projection is not None
        return (
            OrdinaryLinearProbeMethod("grit", projection, candidate.rank),
            projection,
            collection,
        )
    assert collection is not None and candidate.strength is not None
    trace = float(np.square(collection.differences).sum(axis=1).mean())
    return (
        ComposedLinearProbeMethod(
            base,
            "erm_consistency",
            pair_differences=to_tensor(collection.differences).to(torch.float32),
            consistency_weight=candidate.strength / trace,
        ),
        None,
        collection,
    )


def _counts(
    algorithm: LinearProbeAlgorithm,
    x: torch.Tensor,
    y: NDArray[np.int64],
    groups: NDArray[np.int64],
) -> tuple[list[int], NDArray[np.bool_]]:
    correct = to_array(algorithm.predict(x)) == y
    return [int(correct[groups == g].sum()) for g in range(GROUP_COUNT)], correct


def _metrics(
    counts: Sequence[int], per_group: NDArray[np.int64], weights: FloatArray
) -> dict[str, float]:
    return group_metrics(
        [c / n for c, n in zip(counts, per_group, strict=True)], weights
    )


def _weights_digest(state: LinearProbeState) -> str:
    digest = hashlib.sha256()
    for tensor in (state.weight, state.bias):
        digest.update(to_array(tensor.to(torch.float32)).tobytes())
    return f"sha256:{digest.hexdigest()}"


def fit(data: Data, task: Task) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Train one task; return its record and, for final tasks, test diagnostics."""

    config, candidate = data.config, task.candidate
    method, projection, collection = build_method(data, candidate)
    validation_counts: list[list[int]] = []
    test_counts: list[list[int]] = []
    states: dict[int, LinearProbeState] = {}
    test_correct: dict[int, NDArray[np.bool_]] = {}
    final = task.stage == "final"
    if final and data.test is None:
        raise ValueError("final tasks need the unsealed test table")

    def observe(
        algorithm: LinearProbeAlgorithm, checkpoint: CheckpointIdentity, _loss: float
    ) -> None:
        counts, _ = _counts(algorithm, data.val_x, data.val_y, data.val_groups)
        validation_counts.append(counts)
        states[checkpoint.epoch] = algorithm.capture_inference_state()
        if final:
            assert data.test is not None
            counts, correct = _counts(algorithm, *data.test)
            test_counts.append(counts)
            test_correct[checkpoint.epoch] = correct

    run_id = (
        f"urbancars:{config.experiment_name}:{candidate.candidate_id}"
        f":{task.stage}:{task.seed}"
    )
    trained = train_linear_probe_epochs(
        data.train_x,
        torch.from_numpy(data.train_y),  # pyright: ignore[reportUnknownMemberType]
        LinearProbeTrainingConfig(
            optimizer="adam",
            batch_size=config.batch_size,
            learning_rate=candidate.learning_rate,
            weight_decay=candidate.weight_decay,
            max_epochs=config.max_epochs,
        ),
        run_id=run_id,
        candidate_id=candidate.candidate_id,
        scientific_config_digest=canonical_digest_value(asdict(candidate)),
        seed=task.seed,
        method=method,
        validate_epoch=observe,
        num_classes=2,
    )
    removed = None if projection is None else removed_basis(projection)
    estimated = (
        {}
        if collection is None
        else {
            r: top_eigenvectors(collection.differences, r) for r in config.matched_ranks
        }
    )
    record: dict[str, Any] = {
        "candidate": asdict(candidate),
        "candidate_id": candidate.candidate_id,
        "variant": candidate.variant,
        "stage": task.stage,
        "seed": task.seed,
        "dataset_digest": data.digest,
        "validation_counts": validation_counts,
        "train_objective": list(trained.epoch_losses),
        "selected": {},
    }
    if collection is not None:
        record["collection"] = {
            "n_o": collection.n_o,
            "k": collection.k,
            "realized_share": realized_share(
                collection.differences, data.is_o(collection)
            ),
        }
    if projection is not None and removed is not None:
        record["effective_rank"] = projection.diagnostics.effective_rank
        record["numerical_rank"] = projection.diagnostics.numerical_rank
        record["geometry"] = {
            "captured_b": captured_energy(removed, data.second_moments["b"]),
            "captured_o": captured_energy(removed, data.second_moments["o"]),
            "label_energy": label_energy(removed, data.label_direction),
        }
    for selector in SELECTORS:
        epoch, score = select_epoch(
            validation_counts, data.val_per_group, data.weights, selector
        )
        state = states[epoch]
        contrast = classifier_contrast(state.weight)
        residual = residual_sensitivity(
            contrast,
            removed=removed,
            factor_bases=data.factor_bases,
            estimated=estimated,
        )
        if removed is not None:
            own = residual_sensitivity(
                contrast, removed=removed, factor_bases={}, estimated={0: removed}
            )["enforcement_r0"]
            assert own is not None
            if own > ENFORCEMENT_TOLERANCE * max(1.0, residual["w_eff_norm"] or 0.0):
                raise AssertionError(
                    f"integrity: {candidate.candidate_id} enforcement {own:.3e}"
                )
            residual["enforcement_self"] = own
        entry: dict[str, Any] = {
            "epoch": epoch,
            "score": score,
            "validation": _metrics(
                validation_counts[epoch - 1], data.val_per_group, data.weights
            ),
            "weights_sha256": _weights_digest(state),
            "residual": residual,
        }
        if final:
            assert data.test is not None
            per_group = np.bincount(data.test[2], minlength=GROUP_COUNT)
            entry["test"] = _metrics(test_counts[epoch - 1], per_group, data.weights)
            entry["test_correct"] = _pack(test_correct[epoch])
        record["selected"][selector] = entry
    diagnostics = {"test_counts_by_epoch": test_counts} if final else None
    return record, diagnostics


def select_epoch(
    counts: list[list[int]],
    per_group: NDArray[np.int64],
    weights: FloatArray,
    selector: Selector,
) -> tuple[int, float]:
    best_epoch, best = 0, -math.inf
    for index, epoch_counts in enumerate(counts):
        score = selector_score(_metrics(epoch_counts, per_group, weights), selector)
        if score > best:
            best_epoch, best = index + 1, score
    return best_epoch, best


@dataclass(frozen=True)
class _Worker:
    config: UrbanCarsSearchConfig
    root: Path
    binding_digest: str
    test: bool = False

    def __call__(self) -> Callable[[Task], None]:
        data = Data(self.config, test=self.test)
        code = current_code_provenance().model_dump(mode="json")

        def run(task: Task) -> None:
            record, diagnostics = fit(data, task)
            record["binding_digest"] = self.binding_digest
            record["code"] = code
            path = task_path(self.root, task)
            write_json(path, record)
            if diagnostics is not None:
                write_json(path.with_suffix(".diagnostics.json"), diagnostics)
            print(
                f"{task.stage}: {task.candidate.candidate_id} seed={task.seed}",
                flush=True,
            )

        return run


def _execute(
    tasks: list[Task],
    factory: Callable[[], Callable[[Task], None]],
    options: ExecutionOptions,
) -> None:
    if not tasks:
        return
    if options.workers > 1 and len(tasks) > 1:
        _ = run_parallel(tasks, factory, options)
    else:
        run = factory()
        for task in tasks:
            run(task)


# Selection: configuration by the mean over seeds of the per-seed selected score.


def _tie_key(candidate: Candidate) -> tuple[float, str]:
    return (float(candidate.rank or candidate.strength or 0.0), candidate.candidate_id)


def rank_candidates(
    records: Sequence[dict[str, Any]], by_id: dict[str, Candidate], selector: Selector
) -> list[tuple[str, float]]:
    scores: dict[str, list[float]] = {}
    for record in records:
        scores.setdefault(record["candidate_id"], []).append(
            record["selected"][selector]["score"]
        )
    ordered = sorted(scores, key=lambda c: _tie_key(by_id[c]))
    return sorted(((c, fmean(scores[c])) for c in ordered), key=lambda item: -item[1])


class Runner:
    def __init__(self, path: Path, options: ExecutionOptions) -> None:
        self.config = load_config(path)
        self.root = Path(expand_config_path(self.config.output_root))
        self.options = options
        self.data = Data(self.config)
        self.binding = {
            "config": self.config.scientific(),
            "dataset_digest": self.data.digest,
        }
        self.digest = canonical_digest_value(self.binding)
        self.grid = candidates(self.config, self.data)
        self.by_id = {c.candidate_id: c for c in self.grid}

    def worker(self, *, test: bool = False) -> _Worker:
        return _Worker(self.config, self.root, self.digest, test)

    def run(self, tasks: list[Task], *, test: bool = False) -> list[dict[str, Any]]:
        missing = [t for t in tasks if not task_path(self.root, t).exists()]
        _execute(missing, self.worker(test=test), self.options)
        records: list[dict[str, Any]] = []
        for task in tasks:
            record = read_json(task_path(self.root, task))
            if record["binding_digest"] != self.digest:
                raise ValueError(
                    f"{task_path(self.root, task)} was fitted for other inputs"
                )
            records.append(record)
        return records

    def variants(self) -> dict[str, list[Candidate]]:
        result: dict[str, list[Candidate]] = {}
        for c in self.grid:
            result.setdefault(c.variant, []).append(c)
        return result


# Main: tune, confirm, freeze; unseal for final.


def freeze(runner: Runner) -> dict[str, Any]:
    seeds = runner.config.seeds
    tuning = runner.run(
        [Task(c, "tuning", s) for c in runner.grid for s in seeds.tuning]
    )
    finalists: dict[str, dict[str, list[str]]] = {}
    for variant, members in runner.variants().items():
        ids = {c.candidate_id for c in members}
        records = [r for r in tuning if r["candidate_id"] in ids]
        finalists[variant] = {
            s: [c for c, _ in rank_candidates(records, runner.by_id, s)[:3]]
            for s in SELECTORS
        }
    union = sorted(
        {c for per in finalists.values() for ids in per.values() for c in ids}
    )
    confirmation = runner.run(
        [
            Task(runner.by_id[c], "confirmation", s)
            for c in union
            for s in seeds.confirmation
        ]
    )
    winners: dict[str, dict[str, str]] = {}
    for variant, per in finalists.items():
        winners[variant] = {}
        for selector in SELECTORS:
            ids = per[selector]
            records = [r for r in (*tuning, *confirmation) if r["candidate_id"] in ids]
            winners[variant][selector] = rank_candidates(
                records, runner.by_id, selector
            )[0][0]
    matched: dict[str, dict[str, str]] = {}
    for c in runner.grid:
        if c.method == "grit" and c.rank is not None:
            data_c = runner.data.collection(
                next(r for r in runner.config.collections if r.name == c.collection),
                c.bank or 0,
            )
            assert data_c is not None
            cap = _numerical_rank(data_c.differences)
            for requested in runner.config.matched_ranks:
                if min(requested, cap) == c.rank:
                    matched.setdefault(c.variant, {})[str(requested)] = c.candidate_id
    frozen = {"finalists": finalists, "winners": winners, "matched": matched}
    value = canonical_digest_value(frozen)
    path = runner.root / "winners.json"
    if path.exists():
        if read_json(path)["winners_sha256"] != value:
            raise ValueError("recomputed winners differ from the frozen artifact")
    else:
        write_json(
            path, {"binding_digest": runner.digest, "winners_sha256": value, **frozen}
        )
    return {"winners_sha256": value, **frozen}


def final_tasks(runner: Runner, frozen: dict[str, Any]) -> list[Task]:
    ids = {c for per in frozen["winners"].values() for c in per.values()}
    ids |= {c for per in frozen["matched"].values() for c in per.values()}
    return [
        Task(runner.by_id[c], "final", s)
        for c in sorted(ids)
        for s in runner.config.seeds.final
    ]


def _bootstrap(
    correct_a: list[NDArray[np.bool_]],
    correct_b: list[NDArray[np.bool_]],
    groups: NDArray[np.int64],
    weights: FloatArray,
    metric: str,
    resamples: int = 2000,
) -> dict[str, float]:
    """Resample test examples within groups; the difference of model-mean metrics."""

    rng = np.random.default_rng(0)
    a = np.stack(correct_a).astype(np.float64)
    b = np.stack(correct_b).astype(np.float64)
    acc_a = np.empty((resamples, GROUP_COUNT, a.shape[0]))
    acc_b = np.empty((resamples, GROUP_COUNT, b.shape[0]))
    for g in range(GROUP_COUNT):
        members = np.flatnonzero(groups == g)
        idx = rng.choice(members, size=(resamples, len(members)), replace=True)
        acc_a[:, g] = a[:, idx].mean(axis=2).T
        acc_b[:, g] = b[:, idx].mean(axis=2).T

    def value(acc: NDArray[np.float64]) -> NDArray[np.float64]:
        within = WITHIN_POSITIONS[metric]
        if metric == "worst_group":
            return acc.min(axis=1).mean(axis=1)
        if metric == "in_distribution":
            return np.einsum("g,rgm->rm", weights, acc).mean(axis=1)
        return (0.5 * (acc[:, within] + acc[:, 4 + within])).mean(axis=1)

    diff = value(acc_a) - value(acc_b)
    return {
        "ci_low": float(np.quantile(diff, 0.025)),
        "ci_high": float(np.quantile(diff, 0.975)),
    }


WITHIN_POSITIONS = {
    "aligned": 0,
    "b_shift": 1,
    "o_shift": 2,
    "both_shift": 3,
    "worst_group": 0,
    "in_distribution": 0,
}
METRICS = ("b_shift", "o_shift", "both_shift", "worst_group", "in_distribution")


def summarize(
    runner: Runner, frozen: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Rows per unit and selector; contrasts paired by seed within bank, averaged
    over banks (95% t over the 10 final seeds); the range of bank means; and a
    bootstrap over test examples within groups for population claims."""

    assert runner.data.test is not None
    test_groups = runner.data.test[2]
    n_test = len(test_groups)
    by_task = {(r["candidate_id"], r["seed"]): r for r in records}
    seeds = runner.config.seeds.final

    def model(candidate_id: str, selector: str) -> dict[int, dict[str, Any]]:
        return {s: by_task[(candidate_id, s)]["selected"][selector] for s in seeds}

    rows: dict[str, Any] = {}

    def add_row(row_id: str, candidate_id: str, selector: str) -> None:
        per_seed = model(candidate_id, selector)
        record = by_task[(candidate_id, seeds[0])]
        rows[row_id] = {
            "candidate_id": candidate_id,
            "test": {m: [per_seed[s]["test"][m] for s in seeds] for m in METRICS},
            "validation": {
                m: [per_seed[s]["validation"][m] for s in seeds] for m in METRICS
            },
            "epochs": [per_seed[s]["epoch"] for s in seeds],
            "effective_rank": record.get("effective_rank"),
            "geometry": record.get("geometry"),
            "collection": record.get("collection"),
            "residual": {
                k: _mean_optional([per_seed[s]["residual"].get(k) for s in seeds])
                for k in per_seed[seeds[0]]["residual"]
            },
        }

    for variant, per in frozen["winners"].items():
        for selector, candidate_id in per.items():
            add_row(f"{variant}|{selector}", candidate_id, selector)
    for variant, per in frozen["matched"].items():
        for rank, candidate_id in per.items():
            for selector in SELECTORS:
                add_row(f"{variant}|matched-r{rank}|{selector}", candidate_id, selector)

    def bank_averaged(
        unit: str, suffix: str, selector: str, metric: str
    ) -> list[float] | None:
        """Per final seed, the mean over banks (units without a bank: as is)."""
        if "@" not in unit:
            row = rows.get(f"{unit}{suffix}|{selector}")
            return None if row is None else row["test"][metric]
        per_bank = [
            rows.get(f"{unit}-b{b}{suffix}|{selector}") for b in runner.config.banks
        ]
        if any(r is None for r in per_bank):
            return None
        return [
            fmean(r["test"][metric][i] for r in per_bank if r)
            for i in range(len(seeds))
        ]

    def correct_list(
        unit: str, suffix: str, selector: str
    ) -> list[NDArray[np.bool_]] | None:
        units = (
            [unit] if "@" not in unit else [f"{unit}-b{b}" for b in runner.config.banks]
        )
        out: list[NDArray[np.bool_]] = []
        for u in units:
            row = rows.get(f"{u}{suffix}|{selector}")
            if row is None:
                return None
            out += [
                _unpack(model(row["candidate_id"], selector)[s]["test_correct"], n_test)
                for s in seeds
            ]
        return out

    collections = [c.name for c in runner.config.active_collections()]
    pairs: list[tuple[str, str]] = []
    for method in ("grit", "consistency"):
        for c in collections:
            pairs.append((f"{method}@{c}", "erm"))
    for a, b in (
        ("balanced", "b_only"),
        ("balanced", "o_only"),
        ("uneven", "b_only"),
        ("balanced", "uneven"),
    ):
        pairs.append((f"grit@{a}", f"grit@{b}"))
    for c in collections:
        pairs.append((f"grit@{c}", f"consistency@{c}"))
    if "erm_matched" in runner.config.methods:
        for c in collections:
            pairs.append((f"grit@{c}", "erm_matched"))
    contrasts: dict[str, Any] = {}
    for selector in SELECTORS:
        for a, b in pairs:
            for metric in METRICS:
                va, vb = (
                    bank_averaged(a, "", selector, metric),
                    bank_averaged(b, "", selector, metric),
                )
                if va is None or vb is None:
                    continue
                stats: dict[str, Any] = dict(
                    paired_interval([x - y for x, y in zip(va, vb, strict=True)])
                )
                ca, cb = correct_list(a, "", selector), correct_list(b, "", selector)
                if ca is not None and cb is not None:
                    stats["bootstrap"] = _bootstrap(
                        ca, cb, test_groups, runner.data.weights, metric
                    )
                contrasts[f"{a}-minus-{b}|{metric}|{selector}"] = stats
    bank_means = {
        row_id: {m: fmean(row["test"][m]) for m in METRICS}
        for row_id, row in rows.items()
    }
    summary = {
        "experiment": runner.config.experiment_name,
        "config": runner.config.scientific(),
        "dataset_digest": runner.data.digest,
        "winners_sha256": frozen["winners_sha256"],
        "calibration_energies": {"b": runner.data.energy_b, "o": runner.data.energy_o},
        "rows": rows,
        "bank_means": bank_means,
        "contrasts": contrasts,
        "code": current_code_provenance().model_dump(mode="json"),
    }
    write_json(runner.root / "summary.json", summary)
    return summary


def _mean_optional(values: list[Any]) -> float | None:
    present = [v for v in values if isinstance(v, int | float)]
    return fmean(present) if present else None


# Optimizer choice and checks.


def choose_optimizer(runner: Runner) -> dict[str, Any]:
    config = runner.config
    seeds = config.seeds.tuning
    records = runner.run([Task(c, "tuning", s) for c in runner.grid for s in seeds])
    by_setting: dict[tuple[str, float, float], dict[str, list[dict[str, Any]]]] = {}
    for r in records:
        c = runner.by_id[r["candidate_id"]]
        by_setting.setdefault(
            (c.method, c.learning_rate, c.weight_decay), {}
        ).setdefault(c.collection or "-", []).append(r)
    table: list[dict[str, Any]] = []
    for (method, lr, wd), per_collection in sorted(by_setting.items()):
        row: dict[str, Any] = {
            "method": method,
            "learning_rate": lr,
            "weight_decay": wd,
        }
        for selector in SELECTORS:
            best = {
                name: rank_candidates(recs, runner.by_id, selector)[0]
                for name, recs in per_collection.items()
            }
            row[selector] = {
                name: {"candidate_id": c, "score": s} for name, (c, s) in best.items()
            }
            row[f"{selector}_average"] = fmean(s for _, s in best.values())
        table.append(row)
    chosen: dict[str, Any] = {}
    confirmation: dict[str, Any] = {}
    for method in config.methods:
        rows = [r for r in table if r["method"] == method]
        if method == "erm":
            # ERM has one candidate per setting, so its choice is exposed to seed
            # noise: confirm its three best learning rates on the confirmation
            # seeds and freeze the best five-seed mean (the main search's rule).
            best_per_lr: dict[float, dict[str, Any]] = {}
            for r in sorted(
                rows, key=lambda r: (-r["worst_group_average"], r["weight_decay"])
            ):
                best_per_lr.setdefault(r["learning_rate"], r)
            top = sorted(
                best_per_lr.values(),
                key=lambda r: (-r["worst_group_average"], r["learning_rate"]),
            )[:3]
            ids = [r["worst_group"]["-"]["candidate_id"] for r in top]
            extra = runner.run(
                [
                    Task(runner.by_id[c], "confirmation", s)
                    for c in ids
                    for s in config.seeds.confirmation
                ]
            )
            both = [r for r in records if r["candidate_id"] in ids] + extra
            for c in ids:
                mine = [r for r in both if r["candidate_id"] == c]
                confirmation[c] = {
                    "tuning_mean": fmean(
                        r["selected"]["worst_group"]["score"]
                        for r in mine
                        if r["stage"] == "tuning"
                    ),
                    "five_seed_mean": fmean(
                        r["selected"]["worst_group"]["score"] for r in mine
                    ),
                    "selected_epochs": [
                        r["selected"]["worst_group"]["epoch"] for r in mine
                    ],
                    "validation_in_distribution": [
                        r["selected"]["worst_group"]["validation"]["in_distribution"]
                        for r in mine
                    ],
                }
            winner_id = rank_candidates(both, runner.by_id, "worst_group")[0][0]
            winner = runner.by_id[winner_id]
            chosen[method] = {
                "learning_rate": winner.learning_rate,
                "weight_decay": winner.weight_decay,
                "worst_group_average": confirmation[winner_id]["five_seed_mean"],
                "confirmed": True,
            }
            continue
        best = max(
            rows,
            key=lambda r: (
                r["worst_group_average"],
                -r["learning_rate"],
                -r["weight_decay"],
            ),
        )
        chosen[method] = {
            "learning_rate": best["learning_rate"],
            "weight_decay": best["weight_decay"],
            "worst_group_average": best["worst_group_average"],
        }
    result = {
        "experiment": config.experiment_name,
        "rule": "per method, the LR x WD setting maximizing the equal-weight average "
        "over the collections of each collection's best candidate's mean primary "
        "validation score on bank 0 (tuning seeds); ties to the smaller learning rate",
        "chosen": chosen,
        "erm_confirmation": confirmation,
        "table": table,
        "code": current_code_provenance().model_dump(mode="json"),
    }
    write_json(runner.root / "optimizer.json", result)
    return result


def _val_summary(
    records: list[dict[str, Any]], selector: str
) -> dict[str, dict[str, float]]:
    return {
        m: {
            "mean": fmean(r["selected"][selector]["validation"][m] for r in records),
            "sd": pstdev([r["selected"][selector]["validation"][m] for r in records]),
        }
        for m in METRICS
    }


def run_checks(runner: Runner) -> dict[str, Any]:
    """Validation only: C4, C2, C1, C3 and the pilot."""

    config, data = runner.config, runner.data
    ratio = data.energy_b / data.energy_o
    count_table = {
        str(k): {rule.name: data.n_o(rule, k) for rule in config.collections}
        for k in (64, 256, 1024)
    }
    c4 = {
        "energy_b": data.energy_b,
        "energy_o": data.energy_o,
        "ratio": ratio,
        "include_count_50": not (COUNT_50_RANGE[0] <= ratio <= COUNT_50_RANGE[1]),
        "count_table": count_table,
    }
    geometry: dict[str, Any] = {}
    tops: dict[str, FloatArray] = {}
    for factor, moment in data.second_moments.items():
        values = np.linalg.eigvalsh(moment)[::-1]
        geometry[factor] = {
            "top_eigenvalues": [float(v) for v in values[:50]],
            "trace": float(values.sum()),
            **effective_ranks(moment),
            "label_energy_by_rank": {
                str(r): label_energy(
                    np.linalg.eigh(moment)[1][:, ::-1][:, :r], data.label_direction
                )
                for r in config.matched_ranks
            },
        }
        tops[factor] = data.factor_bases[factor]
    cosines = np.linalg.svd(tops["b"].T @ tops["o"], compute_uv=False)
    geometry["principal_angle_cosines"] = [float(c) for c in cosines[:20]]
    geometry["factor_basis_ranks"] = {f: int(b.shape[1]) for f, b in tops.items()}

    erm = [c for c in runner.grid if c.method == "erm"]
    tuning = runner.run(
        [Task(c, "tuning", s) for c in erm for s in config.seeds.tuning]
    )
    best_id = rank_candidates(tuning, runner.by_id, "worst_group")[0][0]
    best = runner.by_id[best_id]
    c1_seeds = list(config.seeds.final)
    c1_records = runner.run([Task(best, "c1", s) for s in c1_seeds])
    c1: dict[str, Any] = {
        "optimizer": {
            "learning_rate": best.learning_rate,
            "weight_decay": best.weight_decay,
        }
    }
    for selector in SELECTORS:
        s = _val_summary(c1_records, selector)
        c1[selector] = {
            **s,
            "b_shift_drop": s["in_distribution"]["mean"] - s["b_shift"]["mean"],
            "o_shift_drop": s["in_distribution"]["mean"] - s["o_shift"]["mean"],
        }
    final_epoch = [
        _metrics(r["validation_counts"][-1], data.val_per_group, data.weights)
        for r in c1_records
    ]
    c1["final_epoch"] = {m: fmean(x[m] for x in final_epoch) for m in METRICS}
    primary = c1["worst_group"]
    c1["passes"] = (
        primary["b_shift_drop"] >= C1_THRESHOLD
        and primary["o_shift_drop"] >= C1_THRESHOLD
    )
    c1["rule"] = (
        "at the primary-selected checkpoint, validation B- and O-shift accuracy "
        "each at least 3 points below in-distribution"
    )

    lr, wd = best.learning_rate, best.weight_decay
    cap = _numerical_rank(data.reference_differences())
    reference = [
        Candidate("reference", lr, wd, rank=r)
        for r in sorted({min(x, cap) for x in config.ranks})
    ]
    for c in reference:
        runner.by_id.setdefault(c.candidate_id, c)
    c3_records = runner.run(
        [Task(c, "c3", s) for c in reference for s in config.seeds.tuning]
    )
    c3 = {
        c.candidate_id: _val_summary(
            [r for r in c3_records if r["candidate_id"] == c.candidate_id],
            "worst_group",
        )
        for c in reference
    }

    pilot_candidates: list[Candidate] = []
    for rule in config.collections:
        collection = data.collection(rule, 0)
        if collection is None:
            continue
        pilot_candidates += [
            Candidate("grit", lr, wd, rule.name, 0, rank=r)
            for r in effective_ranks_of(collection, config)
        ]
        pilot_candidates += [
            Candidate("consistency", lr, wd, rule.name, 0, strength=a)
            for a in config.consistency_strengths
        ]
    for c in pilot_candidates:
        runner.by_id.setdefault(c.candidate_id, c)
    pilot_records = runner.run(
        [Task(c, "pilot", config.seeds.tuning[0]) for c in pilot_candidates]
    )
    pilot = {
        r["candidate_id"]: {
            "worst_group_selected": r["selected"]["worst_group"]["validation"],
            "geometry": r.get("geometry"),
            "collection": r.get("collection"),
        }
        for r in pilot_records
    }
    report = {
        "experiment": config.experiment_name,
        "construction": data.dataset.manifest["construction"],
        "C4_energy_calibration": c4,
        "C2_geometry": geometry,
        "C1_shortcut_reliance": c1,
        "C3_reference_projection": c3,
        "pilot": pilot,
        "code": current_code_provenance().model_dump(mode="json"),
    }
    write_json(runner.root / "check-report.json", report)
    return report


# Entry points.


def pin_inputs(
    root: Path, binding: dict[str, Any], code: dict[str, Any], *, write: bool
) -> None:
    path = root / "inputs.json"
    if path.exists():
        pinned = read_json(path)["binding"]
        if pinned != binding:
            changed = sorted(
                k
                for k in pinned.keys() | binding.keys()
                if pinned.get(k) != binding.get(k)
            )
            raise ValueError(
                "config or dataset manifest changed; use a new output root "
                f"(differs: {', '.join(changed)})"
            )
        return
    if write:
        write_json(path, {"binding": binding, "code": code})


def urbancars_status(path: Path) -> dict[str, Any]:
    config = load_config(path)
    root = Path(expand_config_path(config.output_root))
    stages = (
        sorted(p.name for p in (root / "runs").iterdir())
        if (root / "runs").is_dir()
        else []
    )
    return {
        "dataset": "urbancars",
        "experiment": config.experiment_name,
        "kind": config.kind,
        "output_root": str(root),
        "tasks_by_stage": {
            s: len(list((root / "runs" / s).glob("*/*[0-9].json"))) for s in stages
        },
        "winners_sha256": read_json(root / "winners.json")["winners_sha256"]
        if (root / "winners.json").exists()
        else None,
        "done": any(
            (root / f).exists()
            for f in ("summary.json", "optimizer.json", "check-report.json")
        ),
    }


def run_urbancars(
    path: Path,
    *,
    execution: ExecutionOptions = SEQUENTIAL_EXECUTION,
    unseal_test: str | None = None,
) -> dict[str, Any]:
    options = ExecutionOptions(execution.workers, execution.threads_per_worker or 1)
    runner = Runner(path, options)
    if unseal_test is not None and runner.config.kind != "main":
        raise ValueError("only a main config can be unsealed")
    with execution_scope(runner.root, options):
        pin_inputs(
            runner.root,
            runner.binding,
            current_code_provenance().model_dump(mode="json"),
            write=True,
        )
        if runner.config.kind == "check":
            run_checks(runner)
            return urbancars_status(path)
        if runner.config.kind == "optimizer":
            choose_optimizer(runner)
            return urbancars_status(path)
        frozen = freeze(runner)
        if unseal_test is None:
            print(
                f"winners frozen: {frozen['winners_sha256']}\nvalidation only so far; "
                f"to evaluate test run with --unseal-test {frozen['winners_sha256']}",
                file=sys.stderr,
            )
            return urbancars_status(path)
        if unseal_test != frozen["winners_sha256"]:
            raise ValueError("--unseal-test does not match the frozen winners' hash")
        runner.data = Data(runner.config, test=True)
        records = runner.run(final_tasks(runner, frozen), test=True)
        summarize(runner, frozen, records)
        return urbancars_status(path)
