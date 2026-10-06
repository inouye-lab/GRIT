"""Real linear-probe update, checkpoint, and rank-zero-equivalence tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from grit.config import (
    GroupDroAlgorithmConfig,
    IrmAlgorithmConfig,
    LinearProbeTrainingConfig,
    RexAlgorithmConfig,
)
from grit.features.cmnist import FeatureTable, TableRole
from grit.methods.checkpoints import restore_checkpoint
from grit.methods.groupdro import (
    GroupDroObjective,
    cmnist_group_ids,
    group_balanced_epoch_indices,
)
from grit.methods.invariance import (
    annealed_penalty_weight,
    environment_balanced_epoch_batches,
    environment_mean_losses,
    irmv1_objective,
    vrex_objective,
)
from grit.methods.projection import fit_linear_projection
from grit.methods.training import (
    GroupDroLinearProbeMethod,
    IrmLinearProbeMethod,
    LinearProbeState,
    OrdinaryLinearProbeMethod,
    PersistedLinearCheckpointStore,
    RexLinearProbeMethod,
    persist_selected_linear_checkpoint,
    train_linear_probe,
)
from grit.schemas import CmnistSelector, SeedStage
from grit.selection.cmnist import (
    FrozenCheckpointSelection,
    select_checkpoint,
)


def _table(name: str, role: TableRole, rows: int, offset: int) -> FeatureTable:
    generator = torch.Generator(device="cpu").manual_seed(100 + offset)
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


def _training_config() -> LinearProbeTrainingConfig:
    return LinearProbeTrainingConfig(
        optimizer="adam",
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0.0,
        max_epochs=3,
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


def _groupdro_training_tables() -> tuple[FeatureTable, FeatureTable]:
    tables: list[FeatureTable] = []
    for name, offset in (("train_e01", 21), ("train_e02", 22)):
        generator = torch.Generator(device="cpu").manual_seed(100 + offset)
        features = torch.randn((16, 512), generator=generator)
        targets = torch.tensor([0, 0, 1, 1] * 4, dtype=torch.int64)
        colors = torch.tensor([0, 1, 0, 1] * 4, dtype=torch.int64)
        tables.append(
            FeatureTable(
                name=name,
                role="training",
                source_ids=tuple(f"source:{name}:{index}" for index in range(16)),
                features=features,
                digits=targets * 5,
                clean_labels=targets,
                targets=targets,
                colors=colors,
            )
        )
    return tables[0], tables[1]


def _groupdro_config() -> GroupDroAlgorithmConfig:
    return GroupDroAlgorithmConfig(
        kind="groupdro",
        group_definition="target_color",
        adversarial_step_size=0.01,
        sampling="inverse_group_frequency_with_replacement",
        generalization_adjustment=0.0,
        normalize_loss=False,
    )


def _groupdro_method(
    tables: tuple[FeatureTable, FeatureTable],
) -> GroupDroLinearProbeMethod:
    return GroupDroLinearProbeMethod(
        group_ids=cmnist_group_ids(
            torch.cat([table.targets for table in tables]),
            torch.cat([table.colors for table in tables]),
        ),
        group_count=4,
        step_size=float(_groupdro_config().adversarial_step_size),
    )


def _rex_config() -> RexAlgorithmConfig:
    return RexAlgorithmConfig(
        kind="rex",
        environment_names=("train_e01", "train_e02"),
        penalty_weight=10.0,
        penalty_anneal_updates=1,
        risk_variance="population",
        sampling="environment_balanced_without_replacement",
        loss_rescaling="divide_by_penalty_weight_above_one",
    )


def _irm_config() -> IrmAlgorithmConfig:
    return IrmAlgorithmConfig(
        kind="irm",
        environment_names=("train_e01", "train_e02"),
        penalty_weight=100.0,
        penalty_anneal_updates=1,
        penalty="irmv1_dummy_classifier_scale",
        sampling="environment_balanced_without_replacement",
        loss_rescaling="divide_by_penalty_weight_above_one",
    )


def test_real_trainer_selects_persists_and_restores_checkpoint(tmp_path: Path) -> None:
    training, validation = _tables()
    run = train_linear_probe(
        training,
        validation,
        _training_config(),
        run_id="run:erm:301",
        candidate_id="candidate:erm",
        scientific_config_digest="sha256:erm",
        seed_stage=SeedStage.FINAL,
        seed=301,
        method=OrdinaryLinearProbeMethod("erm", None, None),
    )
    assert len(run.validation_metrics) == 3 * 3
    decision = select_checkpoint(
        run.validation_metrics,
        CmnistSelector.PRIMARY_ROBUST,
    )
    frozen = FrozenCheckpointSelection(
        frozen_checkpoint_id="frozen:checkpoint",
        candidate_selection_id="frozen:candidate",
        selector=CmnistSelector.PRIMARY_ROBUST,
        method_id="erm",
        checkpoint=decision.checkpoint,
        decision=decision,
    )
    selected = run.store.load(decision.checkpoint.checkpoint_id)
    checkpoint_root = tmp_path / "selected"
    manifest = persist_selected_linear_checkpoint(selected, checkpoint_root)
    assert manifest.checkpoint == decision.checkpoint

    zero_state = LinearProbeState(
        weight=torch.zeros((2, 512)),
        bias=torch.zeros(2),
    )
    run.algorithm.restore_inference_state(zero_state)
    receipt = restore_checkpoint(
        frozen,
        PersistedLinearCheckpointStore(checkpoint_root),
        run.algorithm.restore_inference_state,
    )
    assert receipt.checkpoint == decision.checkpoint
    assert torch.equal(
        run.algorithm.capture_inference_state().weight,
        selected.state.weight,
    )


def test_rank_zero_grit_and_erm_are_exactly_equivalent() -> None:
    training, validation = _tables()
    pair_left = torch.randn((8, 512), generator=torch.Generator().manual_seed(9))
    pair_right = torch.randn((8, 512), generator=torch.Generator().manual_seed(10))
    identity = fit_linear_projection(
        pair_left,
        pair_right,
        requested_rank=0,
        pair_manifest_digest="sha256:test-pairs",
        feature_cache_manifest_digest="sha256:test-features",
    )
    erm = train_linear_probe(
        training,
        validation,
        _training_config(),
        run_id="run:erm",
        candidate_id="candidate:erm",
        scientific_config_digest="sha256:shared",
        seed_stage=SeedStage.TUNING,
        seed=101,
        method=OrdinaryLinearProbeMethod("erm", None, None),
    )
    grit = train_linear_probe(
        training,
        validation,
        _training_config(),
        run_id="run:grit",
        candidate_id="candidate:grit",
        scientific_config_digest="sha256:shared",
        seed_stage=SeedStage.TUNING,
        seed=101,
        method=OrdinaryLinearProbeMethod("grit", identity, 0),
    )
    erm_state = erm.algorithm.capture_inference_state()
    grit_state = grit.algorithm.capture_inference_state()
    assert torch.equal(erm_state.weight, grit_state.weight)
    assert torch.equal(erm_state.bias, grit_state.bias)
    assert erm.epoch_losses == grit.epoch_losses
    assert tuple(metric.value for metric in erm.validation_metrics) == tuple(
        metric.value for metric in grit.validation_metrics
    )


def test_groupdro_objective_updates_adversarial_probabilities() -> None:
    objective = GroupDroObjective(group_count=4, step_size=0.1)
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
    value = objective(losses, torch.tensor([0, 1, 2, 3]))
    expected_probabilities = torch.softmax(0.1 * losses.detach(), dim=0)
    assert torch.allclose(objective.probabilities, expected_probabilities)
    assert torch.allclose(value, losses.dot(expected_probabilities))


def test_groupdro_sampler_uses_inverse_group_frequency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[torch.Tensor, int, bool, torch.Generator | None]] = []

    def fake_multinomial(
        weights: torch.Tensor,
        num_samples: int,
        replacement: bool,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        captured.append((weights.clone(), num_samples, replacement, generator))
        return torch.arange(num_samples)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    generator = torch.Generator(device="cpu").manual_seed(9)
    rows = group_balanced_epoch_indices(
        torch.tensor([0, 0, 1, 2, 3]),
        group_count=4,
        generator=generator,
    )
    assert torch.equal(rows, torch.arange(5))
    assert len(captured) == 1
    weights, sample_count, replacement, observed_generator = captured[0]
    assert torch.equal(weights, torch.tensor([2.5, 2.5, 5.0, 5.0, 5.0]))
    assert sample_count == 5
    assert replacement
    assert observed_generator is generator


def test_cmnist_groupdro_training_is_deterministic() -> None:
    training = _groupdro_training_tables()
    _, validation = _tables()
    runs = tuple(
        train_linear_probe(
            training,
            validation,
            _training_config(),
            run_id=f"run:groupdro:{index}",
            candidate_id="candidate:groupdro",
            scientific_config_digest="sha256:groupdro",
            seed_stage=SeedStage.TUNING,
            seed=101,
            method=_groupdro_method(training),
        )
        for index in range(2)
    )
    first, second = runs
    assert first.epoch_losses == second.epoch_losses
    assert torch.equal(
        first.algorithm.capture_inference_state().weight,
        second.algorithm.capture_inference_state().weight,
    )
    assert {metric.method_id for metric in first.validation_metrics} == {"groupdro"}
    groups = cmnist_group_ids(
        torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])
    )
    assert torch.equal(groups, torch.tensor([0, 1, 2, 3]))


def test_vrex_objective_uses_environment_risks() -> None:
    losses = torch.tensor([1.0, 3.0, 5.0, 9.0])
    environment_ids = torch.tensor([0, 0, 1, 1])
    risks = environment_mean_losses(
        losses, environment_ids, environment_count=2
    )
    objective = vrex_objective(
        losses,
        environment_ids,
        environment_count=2,
        penalty_weight=2.0,
    )
    assert torch.equal(risks, torch.tensor([2.0, 7.0]))
    # mean risk = 4.5, population variance = 6.25, then reference rescaling / 2.
    assert objective.item() == 8.5


def test_invariant_penalty_switches_at_fixed_update_boundary() -> None:
    assert annealed_penalty_weight(100.0, anneal_updates=100, update_count=99) == 1.0
    assert (
        annealed_penalty_weight(100.0, anneal_updates=100, update_count=100)
        == 100.0
    )


def test_environment_balanced_batches_are_deterministic_and_exhaustive() -> None:
    environment_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    batches = tuple(
        environment_balanced_epoch_batches(
            environment_ids,
            environment_count=2,
            batch_size=4,
            generator=torch.Generator(device="cpu").manual_seed(17),
        )
        for _ in range(2)
    )
    assert all(torch.equal(left, right) for left, right in zip(*batches, strict=True))
    rows = torch.cat(batches[0])
    assert torch.equal(rows.sort().values, torch.arange(8))
    assert all(
        torch.equal(
            torch.bincount(environment_ids[batch], minlength=2),
            torch.tensor([2, 2]),
        )
        for batch in batches[0]
    )


def test_environment_balanced_batches_support_multiple_source_environments() -> None:
    environment_ids = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    batches = environment_balanced_epoch_batches(
        environment_ids,
        environment_count=3,
        batch_size=6,
        generator=torch.Generator(device="cpu").manual_seed(17),
    )
    assert torch.equal(torch.cat(batches).sort().values, torch.arange(12))
    assert all(
        torch.equal(
            torch.bincount(environment_ids[batch], minlength=3),
            torch.tensor([2, 2, 2]),
        )
        for batch in batches
    )


def test_cmnist_rex_training_is_deterministic() -> None:
    training, validation = _tables()
    environment_ids = torch.cat(
        (
            torch.zeros(16, dtype=torch.int64),
            torch.ones(16, dtype=torch.int64),
        )
    )
    runs = tuple(
        train_linear_probe(
            training,
            validation,
            _training_config(),
            run_id=f"run:rex:{index}",
            candidate_id="candidate:rex",
            scientific_config_digest="sha256:rex",
            seed_stage=SeedStage.TUNING,
            seed=101,
            method=RexLinearProbeMethod(
                environment_ids=environment_ids,
                environment_count=2,
                penalty_weight=float(_rex_config().penalty_weight),
                penalty_anneal_updates=_rex_config().penalty_anneal_updates,
            ),
        )
        for index in range(2)
    )
    first, second = runs
    assert first.epoch_losses == second.epoch_losses
    assert torch.equal(
        first.algorithm.capture_inference_state().weight,
        second.algorithm.capture_inference_state().weight,
    )
    assert {metric.method_id for metric in first.validation_metrics} == {"rex"}


def test_irmv1_objective_uses_one_dummy_scale_penalty_per_environment() -> None:
    log_three = math.log(3.0)
    logits = torch.tensor(((log_three, 0.0), (0.0, log_three)))
    targets = torch.tensor((0, 0), dtype=torch.int64)
    environment_ids = torch.tensor((0, 1), dtype=torch.int64)
    objective = irmv1_objective(
        logits,
        targets,
        environment_ids,
        environment_count=2,
        penalty_weight=2.0,
    )
    mean_risk = (math.log(4.0 / 3.0) + math.log(4.0)) / 2.0
    mean_squared_gradient = 5.0 * log_three**2 / 16.0
    expected = mean_risk / 2.0 + mean_squared_gradient
    assert math.isclose(
        float(objective.detach().item()), expected, rel_tol=0.0, abs_tol=1e-6
    )


def test_cmnist_irm_training_is_deterministic() -> None:
    training, validation = _tables()
    environment_ids = torch.cat(
        (
            torch.zeros(16, dtype=torch.int64),
            torch.ones(16, dtype=torch.int64),
        )
    )
    runs = tuple(
        train_linear_probe(
            training,
            validation,
            _training_config(),
            run_id=f"run:irm:{index}",
            candidate_id="candidate:irm",
            scientific_config_digest="sha256:irm",
            seed_stage=SeedStage.TUNING,
            seed=101,
            method=IrmLinearProbeMethod(
                environment_ids=environment_ids,
                environment_count=2,
                penalty_weight=float(_irm_config().penalty_weight),
                penalty_anneal_updates=_irm_config().penalty_anneal_updates,
            ),
        )
        for index in range(2)
    )
    first, second = runs
    assert first.epoch_losses == second.epoch_losses
    assert torch.equal(
        first.algorithm.capture_inference_state().weight,
        second.algorithm.capture_inference_state().weight,
    )
    assert {metric.method_id for metric in first.validation_metrics} == {"irm"}
