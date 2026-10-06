"""Tracking-mirror tests: no scientific effect, no test access, no hard failure."""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from grit.config import (
    ComposedAlgorithmConfig,
    FishrAlgorithmConfig,
    LinearProbeTrainingConfig,
)
from grit.features.cmnist import FeatureTable, TableRole
from grit.methods.training import OrdinaryLinearProbeMethod, train_linear_probe
from grit.schemas import SeedStage
from grit.search.plan import (
    SearchCandidate,
    SearchPlan,
    current_code_provenance,
)
from grit.search.scheduler import SearchRunTask
from grit.tracking import (
    MirrorRunTracker,
    NullRunTracker,
    TrackedRun,
    TrackingSettings,
    TrackingValue,
    WandbBackend,
    log_summary_table,
    start_summary_tracker,
    start_task_tracker,
    task_tracker,
    tracking_settings_from_environment,
)


class _RecordingRun:
    """One backend run that keeps what a real mirror would have transmitted."""

    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []
        self.summary: dict[str, object] = {}
        self.tables: list[tuple[str, tuple[str, ...], tuple[tuple[object, ...], ...]]]
        self.tables = []
        self.finished: list[int | None] = []

    def log(self, values: Mapping[str, object]) -> None:
        self.logged.append(dict(values))

    def update_summary(self, values: Mapping[str, object]) -> None:
        self.summary.update(values)

    def log_table(
        self,
        name: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[TrackingValue]],
    ) -> None:
        self.tables.append((name, tuple(columns), tuple(tuple(row) for row in rows)))

    def finish(self, exit_code: int | None = None) -> None:
        self.finished.append(exit_code)


class _FailingRun:
    """Every tracking call fails; nothing may escape the tracker."""

    def log(self, values: Mapping[str, object]) -> None:
        del values
        raise RuntimeError("mirror is unavailable")

    def update_summary(self, values: Mapping[str, object]) -> None:
        del values
        raise RuntimeError("mirror is unavailable")

    def log_table(
        self,
        name: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[TrackingValue]],
    ) -> None:
        del name, columns, rows
        raise RuntimeError("mirror is unavailable")

    def finish(self, exit_code: int | None = None) -> None:
        del exit_code
        raise RuntimeError("mirror is unavailable")


class _RecordingBackend:
    def __init__(self) -> None:
        self.started: list[dict[str, object]] = []
        self.runs: list[_RecordingRun] = []

    def start(
        self,
        *,
        project: str,
        entity: str | None,
        mode: str,
        directory: Path,
        group: str,
        name: str,
        job_type: str,
        config: Mapping[str, TrackingValue],
    ) -> TrackedRun:
        self.started.append(
            {
                "project": project,
                "entity": entity,
                "mode": mode,
                "directory": directory,
                "group": group,
                "name": name,
                "job_type": job_type,
                "config": dict(config),
            }
        )
        run = _RecordingRun()
        self.runs.append(run)
        return run


def _table(name: str, role: TableRole, rows: int, offset: int) -> FeatureTable:
    generator = torch.Generator(device="cpu").manual_seed(500 + offset)
    features = torch.randn((rows, 512), generator=generator)
    targets = (features[:, 0] + 0.25 * features[:, 1] > 0).to(torch.int64)
    return FeatureTable(
        name=name,
        role=role,
        source_ids=tuple(f"source:{name}:{index}" for index in range(rows)),
        features=features,
        digits=targets * 5,
        clean_labels=targets,
        targets=targets,
        colors=targets.clone(),
    )


def _tables() -> tuple[
    tuple[FeatureTable, FeatureTable],
    tuple[FeatureTable, FeatureTable, FeatureTable],
]:
    return (
        (
            _table("train_e01", "training", 16, 1),
            _table("train_e02", "training", 16, 2),
        ),
        (
            _table("val_e01", "validation", 8, 3),
            _table("val_e02", "validation", 8, 4),
            _table("val_e05", "validation", 8, 5),
        ),
    )


def _training_config() -> LinearProbeTrainingConfig:
    return LinearProbeTrainingConfig(
        optimizer="adam",
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0.0,
        max_epochs=3,
    )


def _train(tracker: object | None):
    training, validation = _tables()
    return train_linear_probe(
        training,
        validation,
        _training_config(),
        run_id="run:tracking",
        candidate_id="candidate:erm",
        scientific_config_digest="sha256:erm",
        seed_stage=SeedStage.TUNING,
        seed=301,
        method=OrdinaryLinearProbeMethod("erm", None, None),
        tracker=cast(MirrorRunTracker | None, tracker),
    )


def _candidate() -> SearchCandidate:
    return SearchCandidate(
        candidate_id="candidate:erm:abcdef0123456789",
        scientific_config_digest="sha256:candidate",
        method_id="erm",
        learning_rate=0.001,
        weight_decay=0.0,
        requested_rank=None,
    )


def _fake_plan(
    selectors: tuple[str, ...] = ("primary_robust", "secondary_source"),
) -> SearchPlan:
    """A plan-shaped stand-in: tracking only reads these fields."""

    return cast(
        SearchPlan,
        SimpleNamespace(
            dataset="cmnist",
            selectors=selectors,
            protocol_id="cmnist/v1",
            experiment_name="cmnist-primary",
            experiment_variant="primary_unnormalized",
            normalization="none",
            canonical_digest=lambda: "sha256:0123456789abcdef",
            code=SimpleNamespace(git_revision="abc123", git_dirty=False),
            resolved_config=SimpleNamespace(
                config=SimpleNamespace(batch_size=64, max_epochs=5, pair_count=256),
                lineage=SimpleNamespace(
                    dataset_manifest_digest="sha256:dataset",
                    feature_cache_manifest_digest="sha256:features",
                    pair_manifest_digest="sha256:pairs",
                    adjusted_weight_spec_digest=None,
                    held_out_validation_split="val_e05",
                ),
            ),
        ),
    )


def _fake_task(
    *, stage: SeedStage = SeedStage.TUNING, selector: str | None = None
) -> SearchRunTask:
    return cast(
        SearchRunTask,
        SimpleNamespace(
            task_id="task:0123456789ab",
            candidate=_candidate(),
            stage=stage,
            seed=301,
            selector=selector,
            relative_directory="runs/tuning/erm/candidate/301",
        ),
    )


def _enabled(tmp_path: Path) -> TrackingSettings:
    return TrackingSettings(
        project="grit-test",
        entity="lab",
        mode="offline",
        directory=tmp_path / "wandb",
    )


def test_tracking_is_off_unless_the_environment_names_a_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PROJECT_SCRATCH", tmp_path.as_posix())
    for name in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_MODE"):
        monkeypatch.delenv(name, raising=False)
    default = tracking_settings_from_environment()
    assert not default.enabled
    assert default.directory == tmp_path / "wandb"

    monkeypatch.setenv("WANDB_PROJECT", "grit")
    monkeypatch.setenv("WANDB_ENTITY", "lab")
    enabled = tracking_settings_from_environment()
    assert enabled.enabled
    assert (enabled.project, enabled.entity, enabled.mode) == ("grit", "lab", "online")

    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert not tracking_settings_from_environment().enabled

    backend = _RecordingBackend()
    tracker = start_task_tracker(
        _fake_plan(),
        _fake_task(),
        settings=tracking_settings_from_environment(),
        backend=backend,
    )
    assert isinstance(tracker, NullRunTracker)
    assert backend.started == []


def test_enabled_task_run_carries_identity_without_touching_the_experiment(
    tmp_path: Path,
) -> None:
    backend = _RecordingBackend()
    tracker = start_task_tracker(
        _fake_plan(),
        _fake_task(stage=SeedStage.FINAL, selector="primary_robust"),
        settings=_enabled(tmp_path),
        backend=backend,
    )
    assert len(backend.started) == 1
    started = backend.started[0]
    assert started["project"] == "grit-test"
    assert started["entity"] == "lab"
    assert started["job_type"] == "final"
    assert started["group"] == "cmnist-primary:0123456789ab"
    assert started["name"] == "erm/abcdef012345/primary_robust/final/seed-301"
    assert started["directory"] == tmp_path / "wandb"
    config = cast(dict[str, object], started["config"])
    for key in (
        "dataset",
        "plan_digest",
        "task_id",
        "attempt_id",
        "seed",
        "selector",
        "learning_rate",
        "weight_decay",
        "git_revision",
        "dataset_manifest_digest",
        "feature_cache_manifest_digest",
        "pair_manifest_digest",
    ):
        assert key in config
    assert config["selector"] == "primary_robust"
    assert config["execution/workers"] == 1
    assert config["execution/threads_per_worker"] == torch.get_num_threads()
    assert config["execution/start_method"] == "direct"
    assert config["track"] == "ordinary"
    tracker.finish()
    assert backend.runs[0].finished == [0]


def test_repeated_attempts_of_one_task_are_separate_runs(tmp_path: Path) -> None:
    backend = _RecordingBackend()
    task = _fake_task()
    for _ in range(2):
        start_task_tracker(
            _fake_plan(), task, settings=_enabled(tmp_path), backend=backend
        ).finish()
    attempts = [
        cast(dict[str, object], started["config"])["attempt_id"]
        for started in backend.started
    ]
    task_ids = {
        cast(dict[str, object], started["config"])["task_id"]
        for started in backend.started
    }
    assert task_ids == {"task:0123456789ab"}
    assert attempts[0] != attempts[1]


def test_mirroring_does_not_change_training_selection_or_open_test_data(
    tmp_path: Path,
) -> None:
    untracked = _train(None)
    backend = _RecordingBackend()
    tracker = start_task_tracker(
        _fake_plan(), _fake_task(), settings=_enabled(tmp_path), backend=backend
    )
    tracked = _train(tracker)
    tracker.finish()

    assert tracked.validation_metrics == untracked.validation_metrics
    assert tracked.epoch_losses == untracked.epoch_losses
    tracked_state = tracked.algorithm.capture_inference_state()
    untracked_state = untracked.algorithm.capture_inference_state()
    assert torch.equal(tracked_state.weight, untracked_state.weight)
    assert torch.equal(tracked_state.bias, untracked_state.bias)

    logged = backend.runs[0].logged
    assert [entry["epoch"] for entry in logged] == [1, 2, 3]
    keys = {key for entry in logged for key in entry}
    assert keys == {
        "epoch",
        "train_objective",
        "validation/val_e01_accuracy",
        "validation/val_e02_accuracy",
        "validation/val_e05_accuracy",
    }
    # The ordinary track has no test record to mirror, by construction.
    assert not any("test" in key for key in keys)
    for entry, expected in zip(logged, untracked.epoch_losses, strict=True):
        assert entry["train_objective"] == expected


def test_a_failing_mirror_warns_and_leaves_the_run_intact() -> None:
    tracker = MirrorRunTracker(cast(TrackedRun, _FailingRun()))
    with pytest.warns(RuntimeWarning):
        trained = _train(tracker)
    with pytest.warns(RuntimeWarning):
        tracker.record_selection({"selected/primary_robust/epoch": 1})
    with pytest.warns(RuntimeWarning):
        tracker.record_table("summary", ("a",), ((1,),))
    with pytest.warns(RuntimeWarning):
        tracker.finish()
    assert trained.validation_metrics == _train(None).validation_metrics


def test_a_backend_that_cannot_start_degrades_to_no_tracking(
    tmp_path: Path,
) -> None:
    class _BrokenBackend:
        def start(self, **kwargs: object) -> TrackedRun:
            del kwargs
            raise RuntimeError("no credentials")

    with pytest.warns(RuntimeWarning):
        tracker = start_task_tracker(
            _fake_plan(),
            _fake_task(),
            settings=_enabled(tmp_path),
            backend=cast(_RecordingBackend, _BrokenBackend()),
        )
    assert isinstance(tracker, NullRunTracker)
    tracker.finish()


def test_summary_mirror_reports_one_labeled_table(tmp_path: Path) -> None:
    backend = _RecordingBackend()
    rows: list[list[TrackingValue]] = [["erm", "primary_robust", 0.5]]
    log_summary_table(
        _fake_plan(),
        track="ordinary",
        table_name="cmnist_final_summary",
        columns=("method", "selector", "mean"),
        rows=rows,
        settings=_enabled(tmp_path),
        backend=backend,
    )
    assert [started["job_type"] for started in backend.started] == ["summary"]
    assert [started["name"] for started in backend.started] == ["summary/ordinary"]
    assert [run.tables[0][0] for run in backend.runs] == ["cmnist_final_summary"]
    assert all(run.finished == [0] for run in backend.runs)
    disabled = start_summary_tracker(
        _fake_plan(),
        track="ordinary",
        settings=TrackingSettings(
            project=None, entity=None, mode="online", directory=tmp_path
        ),
        backend=backend,
    )
    assert isinstance(disabled, NullRunTracker)


# Startup failures that mean "this machine forbids it", not "the mirror is broken":
# W&B's own service-start errors, and the socket, permission, and connection refusals a
# restricted sandbox raises underneath them.
_BLOCKED_ENVIRONMENT_SIGNATURES = (
    "servicestart",
    "permissionerror",
    "permission denied",
    "connectionrefusederror",
    "connection refused",
    "socket",
    "errno 13",
    "errno 111",
    "errno 101",
    "network is unreachable",
    "operation not permitted",
)


def _is_blocked_environment(reported: str) -> bool:
    lowered = reported.lower()
    return any(item in lowered for item in _BLOCKED_ENVIRONMENT_SIGNATURES)


@pytest.mark.skipif(
    importlib.util.find_spec("wandb") is None, reason="wandb is not installed"
)
def test_offline_mode_writes_a_usable_local_run_without_an_account(
    tmp_path: Path,
) -> None:
    settings = TrackingSettings(
        project="grit-offline-test",
        entity=None,
        mode="offline",
        directory=tmp_path / "wandb",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracker = start_task_tracker(
            _fake_plan(),
            _fake_task(),
            settings=settings,
            backend=WandbBackend(),
        )
    if isinstance(tracker, NullRunTracker):
        # W&B starts a local service process and socket even offline, which some
        # sandboxes forbid. That is an environment limitation, not a defect; any other
        # startup failure is a real regression and must fail here.
        reported = " ".join(str(item.message) for item in caught)
        if not _is_blocked_environment(reported):
            pytest.fail(f"W&B offline start failed unexpectedly: {reported}")
        pytest.skip(f"this environment cannot start W&B locally: {reported}")
    tracker.log_epoch(1, {"train_objective": 0.5, "validation/val_e01_accuracy": 0.75})
    tracker.record_selection({"selected/primary_robust/epoch": 1})
    tracker.finish()
    assert tuple((tmp_path / "wandb").rglob("*.wandb"))


def _fishr_algorithm() -> FishrAlgorithmConfig:
    return FishrAlgorithmConfig(
        kind="fishr",
        environment_names=("train_e01", "train_e02"),
        penalty_weight=100.0,
        penalty_anneal_updates=500,
        ema=0.95,
        penalty="classifier_gradient_variance_distance",
        sampling="environment_balanced_without_replacement",
    )


def test_run_config_separates_the_executing_revision_from_the_plan(
    tmp_path: Path,
) -> None:
    backend = _RecordingBackend()
    start_task_tracker(
        _fake_plan(), _fake_task(), settings=_enabled(tmp_path), backend=backend
    ).finish()
    config = cast(dict[str, object], backend.started[0]["config"])
    executing = current_code_provenance()
    assert config["plan_git_revision"] == "abc123"
    assert config["plan_git_dirty"] is False
    assert config["git_revision"] == executing.git_revision
    assert config["git_dirty"] == executing.git_dirty


def test_run_config_includes_settings_a_candidate_holds_fixed(
    tmp_path: Path,
) -> None:
    backend = _RecordingBackend()
    start_task_tracker(
        _fake_plan(),
        _fake_task(),
        algorithm=_fishr_algorithm(),
        settings=_enabled(tmp_path),
        backend=backend,
    ).finish()
    config = cast(dict[str, object], backend.started[0]["config"])
    assert config["algorithm/kind"] == "fishr"
    assert config["algorithm/ema"] == 0.95
    assert config["algorithm/penalty_anneal_updates"] == 500
    assert config["algorithm/penalty_weight"] == 100.0
    assert config["algorithm/environment_names"] == "train_e01,train_e02"


def test_a_failed_attempt_does_not_leave_a_finished_run(tmp_path: Path) -> None:
    backend = _RecordingBackend()
    with pytest.raises(RuntimeError, match="training failed"):
        with task_tracker(
            _fake_plan(), _fake_task(), settings=_enabled(tmp_path), backend=backend
        ):
            raise RuntimeError("training failed")
    assert backend.runs[0].finished == [1]

    with task_tracker(
        _fake_plan(), _fake_task(), settings=_enabled(tmp_path), backend=backend
    ):
        pass
    assert backend.runs[1].finished == [0]


def test_composed_tracking_distinguishes_base_and_pair_coefficients(
    tmp_path: Path,
) -> None:
    backend = _RecordingBackend()
    candidate = _candidate().model_copy(
        update={
            "method_id": "fishr_consistency",
            "penalty_weight": 100.0,
            "consistency_weight": 0.1,
        }
    )
    task = cast(
        SearchRunTask,
        SimpleNamespace(**{**vars(_fake_task()), "candidate": candidate}),
    )
    start_task_tracker(
        _fake_plan(),
        task,
        algorithm=ComposedAlgorithmConfig(
            kind="composed",
            base_objective=_fishr_algorithm(),
            pair_intervention="consistency",
            consistency_weight=0.1,
            objective_version="prediction-pairs/v1",
        ),
        settings=_enabled(tmp_path),
        backend=backend,
    ).finish()
    config = cast(dict[str, object], backend.started[0]["config"])
    assert config["base_objective"] == "fishr"
    assert config["pair_intervention"] == "consistency"
    assert config["algorithm/base_objective/penalty_weight"] == 100.0
    assert config["algorithm/base_objective/penalty_anneal_updates"] == 500
    assert config["algorithm/consistency_weight"] == 0.1
    assert config["track"] == "ordinary"


def test_pair_diagnostics_mirror_computed_training_values(
    tmp_path: Path,
) -> None:
    from grit.methods.interventions import ComposedLinearProbeMethod

    training, validation = _tables()
    backend = _RecordingBackend()
    tracker = start_task_tracker(
        _fake_plan(),
        _fake_task(),
        settings=_enabled(tmp_path),
        backend=backend,
    )
    run = train_linear_probe(
        training,
        validation,
        _training_config(),
        run_id="run:consistency",
        candidate_id="candidate:consistency",
        scientific_config_digest="sha256:consistency",
        seed_stage=SeedStage.TUNING,
        seed=301,
        method=ComposedLinearProbeMethod(
            OrdinaryLinearProbeMethod("erm", None, None),
            "erm_consistency",
            pair_differences=training[0].features[:4] - training[1].features[:4],
            consistency_weight=0.1,
        ),
        tracker=tracker,
    )
    tracker.finish()
    for entry, epoch in zip(backend.runs[0].logged, (1, 2, 3), strict=True):
        values = run.algorithm.diagnostic_history[epoch]
        assert tuple(values) == ("pairs/prediction_consistency",)
        assert all(entry[name] == value for name, value in values.items())
        assert not any("test" in name for name in entry)


def test_training_metadata_names_what_each_base_objective_is_told() -> None:
    from grit.config import (
        ComposedAlgorithmConfig,
        ErmAlgorithmConfig,
        GroupDroAlgorithmConfig,
        LisaAlgorithmConfig,
        RexAlgorithmConfig,
    )
    from grit.tracking import training_metadata

    def groupdro(definition: str) -> GroupDroAlgorithmConfig:
        return GroupDroAlgorithmConfig.model_validate(
            {
                "kind": "groupdro",
                "group_definition": definition,
                "adversarial_step_size": 0.01,
                "sampling": "inverse_group_frequency_with_replacement",
                "generalization_adjustment": 0.0,
                "normalize_loss": False,
            }
        )

    lisa = LisaAlgorithmConfig(
        kind="lisa",
        group_definition="target_environment",
        selection_prob=0.5,
        mixing="beta_2_2",
        sampling="uniform_single_group_batches",
    )
    rex = RexAlgorithmConfig(
        kind="rex",
        environment_names=("train_e01", "train_e02"),
        penalty_weight=1.0,
        penalty_anneal_updates=0,
        risk_variance="population",
        sampling="environment_balanced_without_replacement",
        loss_rescaling="divide_by_penalty_weight_above_one",
    )
    composed = ComposedAlgorithmConfig(
        kind="composed",
        base_objective=groupdro("training_environment"),
        pair_intervention="consistency",
        consistency_weight=0.1,
    )
    assert training_metadata("cmnist", ErmAlgorithmConfig(kind="erm")) == "none"
    assert training_metadata("cmnist", None) == "none"
    assert training_metadata("cmnist", groupdro("target_color")) == "label_x_color"
    assert training_metadata("cmnist", composed) == "source_environment"
    label_source = composed.model_copy(
        update={"base_objective": groupdro("target_environment")}
    )
    assert training_metadata("cmnist", label_source) == "label_x_source_environment"
    assert training_metadata("waterbirds_cf", composed) == "background"
    assert training_metadata("cmnist", lisa) == "label_x_source_environment"
    assert training_metadata("cmnist", rex) == "source_environment"
    assert (
        training_metadata("waterbirds_cf", groupdro("target_background"))
        == "label_x_background"
    )
