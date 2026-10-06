"""Objective composition, training controls, and complete predictor restoration."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import pytest
import torch
import torch.nn.functional as functional

from grit.config import LinearProbeTrainingConfig
from grit.methods.baselines import (
    FishrLinearProbeMethod,
    LisaLinearProbeMethod,
    RdmLinearProbeMethod,
    SpectralDecouplingLinearProbeMethod,
)
from grit.methods.checkpoints import StoredCheckpoint
from grit.methods.interventions import (
    ComposedLinearProbeMethod,
    prediction_consistency_penalty,
)
from grit.methods.invariance import irmv1_objective, vrex_objective
from grit.methods.lisa import mix_features_and_targets, plan_lisa_epoch
from grit.methods.projection import FittedLinearProjection, fit_linear_projection
from grit.methods.rdm import rdm_objective
from grit.methods.training import (
    GroupDroLinearProbeMethod,
    IrmLinearProbeMethod,
    LinearProbeAlgorithm,
    LinearProbeState,
    LinearProbeTrainingMethod,
    OrdinaryLinearProbeMethod,
    PersistedLinearCheckpointStore,
    RexLinearProbeMethod,
    persist_selected_linear_checkpoint,
)
from grit.methods.types import MethodId
from grit.selection.cmnist import CheckpointIdentity

Base = Literal["erm", "rex", "irm", "fishr"]


def _config() -> LinearProbeTrainingConfig:
    return LinearProbeTrainingConfig(
        optimizer="adam",
        batch_size=8,
        learning_rate=0.001,
        weight_decay=0.0,
        max_epochs=2,
    )


def _data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    return (
        torch.randn((8, 512), generator=generator),
        torch.tensor([0, 1, 1, 0, 1, 0, 1, 0]),
        torch.randn((3, 512), generator=generator),
    )


def _base(kind: Base) -> LinearProbeTrainingMethod:
    ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    if kind == "erm":
        return OrdinaryLinearProbeMethod("erm", None, None)
    if kind == "rex":
        return RexLinearProbeMethod(ids, 2, 10.0, 1)
    if kind == "irm":
        return IrmLinearProbeMethod(ids, 2, 10.0, 1)
    return FishrLinearProbeMethod(ids, 2, 10.0, 1, 0.95)


def _projection(rank: int) -> FittedLinearProjection:
    differences = torch.zeros((2, 512))
    differences[0, 0] = 1.0
    differences[1, 1] = 1.0
    return fit_linear_projection(
        differences,
        torch.zeros_like(differences),
        requested_rank=rank,
        pair_manifest_digest="sha256:pairs",
        feature_cache_manifest_digest="sha256:cache",
    )


def test_consistency_matches_paired_logits_and_gradient() -> None:
    generator = torch.Generator().manual_seed(4)
    left = torch.randn((5, 7), generator=generator, dtype=torch.float64)
    right = torch.randn((5, 7), generator=generator, dtype=torch.float64)
    weight = torch.randn(
        (3, 7), generator=generator, dtype=torch.float64
    ).requires_grad_()
    bias = torch.randn((3,), generator=generator, dtype=torch.float64).requires_grad_()
    differences = left - right
    penalty = prediction_consistency_penalty(weight, differences)
    explicit = (
        (functional.linear(left, weight, bias) - functional.linear(right, weight, bias))
        .square()
        .sum(dim=1)
        .mean()
    )
    torch.testing.assert_close(penalty, explicit)
    gradient = torch.autograd.grad(penalty, weight)[0]
    explicit_weight, explicit_bias = torch.autograd.grad(explicit, (weight, bias))
    torch.testing.assert_close(gradient, explicit_weight)
    torch.testing.assert_close(gradient, 2.0 * weight @ differences.T @ differences / 5)
    torch.testing.assert_close(explicit_bias, torch.zeros_like(bias))


@pytest.mark.parametrize("kind", ["erm", "rex", "irm", "fishr"])
@pytest.mark.parametrize("control", ["identity", "zero_consistency"])
def test_control_interventions_reproduce_base_updates(kind: Base, control: str) -> None:
    features, targets, differences = _data()
    plain = _base(kind)
    if control == "identity":
        method_id = "grit" if kind == "erm" else f"{kind}_grit"
        composed = ComposedLinearProbeMethod(
            _base(kind),
            cast(MethodId, method_id),
            projection=_projection(0),
            projection_rank=0,
        )
    else:
        composed = ComposedLinearProbeMethod(
            _base(kind),
            cast(MethodId, f"{kind}_consistency"),
            pair_differences=differences,
            consistency_weight=0.0,
        )
    vanilla = plain.build_algorithm(_config(), model_seed=3, num_classes=2)
    intervened = composed.build_algorithm(_config(), model_seed=3, num_classes=2)
    # Includes the annealing transition, Fishr's optimizer reset, and EMA history.
    for _ in range(3):
        rows = torch.arange(8)
        assert plain.update(vanilla, features, targets, rows) == composed.update(
            intervened, features, targets, rows
        )
    expected = vanilla.capture_inference_state()
    actual = intervened.capture_inference_state()
    assert torch.equal(expected.weight, actual.weight)
    assert torch.equal(expected.bias, actual.bias)


@pytest.mark.parametrize("kind", ["rex", "irm"])
def test_rescaled_objective_scales_complete_consistency_term(kind: Base) -> None:
    features, targets, differences = _data()
    method = ComposedLinearProbeMethod(
        _base(kind),
        cast(MethodId, f"{kind}_consistency"),
        pair_differences=differences,
        consistency_weight=2.0,
    )
    algorithm = method.build_algorithm(_config(), model_seed=5, num_classes=2)
    for expected_weight in (1.0, 10.0):
        state = algorithm.capture_inference_state()
        logits = functional.linear(features, state.weight, state.bias)
        ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        if kind == "rex":
            expected = vrex_objective(
                functional.cross_entropy(logits, targets, reduction="none"),
                ids,
                environment_count=2,
                penalty_weight=expected_weight,
            )
        else:
            expected = irmv1_objective(
                logits,
                targets,
                ids,
                environment_count=2,
                penalty_weight=expected_weight,
            )
        expected += (
            2.0
            / expected_weight
            * prediction_consistency_penalty(state.weight, differences)
        )
        actual = method.update(algorithm, features, targets, torch.arange(8))
        assert math.isclose(actual, float(expected.item()), rel_tol=1e-6)


def test_fishr_projection_uses_projected_supervised_gradient_features() -> None:
    features, targets, _ = _data()
    projection = _projection(2)
    composed = ComposedLinearProbeMethod(_base("fishr"), "fishr_grit", projection, 2)
    projected = composed.build_algorithm(_config(), model_seed=4, num_classes=2)
    plain_method = _base("fishr")
    plain = plain_method.build_algorithm(_config(), model_seed=4, num_classes=2)
    for _ in range(3):
        assert composed.update(projected, features, targets, torch.arange(8)) == (
            plain_method.update(
                plain, projection.transform(features), targets, torch.arange(8)
            )
        )
    assert torch.equal(
        projected.capture_inference_state().weight,
        plain.capture_inference_state().weight,
    )


def test_fishr_consistency_is_active_in_warmup_and_excluded_from_statistics() -> None:
    features, targets, differences = _data()
    base = cast(FishrLinearProbeMethod, _base("fishr"))
    plain_method = cast(FishrLinearProbeMethod, _base("fishr"))
    composed = ComposedLinearProbeMethod(
        base,
        "fishr_consistency",
        pair_differences=differences,
        consistency_weight=2.0,
    )
    algorithm = composed.build_algorithm(_config(), model_seed=4, num_classes=2)
    vanilla = plain_method.build_algorithm(_config(), model_seed=4, num_classes=2)
    penalty = prediction_consistency_penalty(
        algorithm.capture_inference_state().weight, differences
    )
    ordinary_loss = plain_method.update(vanilla, features, targets, torch.arange(8))
    combined = composed.update(algorithm, features, targets, torch.arange(8))
    assert math.isclose(
        combined, ordinary_loss + 2.0 * float(penalty.item()), rel_tol=1e-6
    )
    for environment in range(2):
        actual, expected = (
            base.ema.state(environment),
            plain_method.ema.state(environment),
        )
        assert actual is not None and expected is not None
        assert torch.equal(actual, expected)


def test_persisted_predictor_restores_projection_without_external_fit(
    tmp_path: Path,
) -> None:
    features, targets, _ = _data()
    method = ComposedLinearProbeMethod(_base("erm"), "grit", _projection(2), 2)
    algorithm = method.build_algorithm(_config(), model_seed=4, num_classes=2)
    method.update(algorithm, features, targets, torch.arange(8))
    identity = CheckpointIdentity(
        checkpoint_id="checkpoint:projected",
        candidate_id="candidate:projected",
        run_id="run:projected",
        scientific_config_digest="sha256:config",
        epoch=1,
    )
    root = tmp_path / "checkpoint"
    persist_selected_linear_checkpoint(
        StoredCheckpoint(identity, method.checkpoint_state(algorithm)), root
    )
    restored = LinearProbeAlgorithm(_config(), model_seed=999, projection=None)
    restored.restore_inference_state(
        PersistedLinearCheckpointStore(root).load(identity.checkpoint_id).state
    )
    assert torch.equal(algorithm.predict(features), restored.predict(features))
    assert algorithm.mean_loss(features, targets) == restored.mean_loss(
        features, targets
    )
    perturbed = features.clone()
    perturbed[:, :2] += 100.0
    assert torch.equal(restored.predict(features), restored.predict(perturbed))
    assert restored.mean_loss(features, targets) == restored.mean_loss(
        perturbed, targets
    )


def test_modern_vanilla_state_clears_projection_and_legacy_state_retains_it() -> None:
    features, targets, _ = _data()
    vanilla = LinearProbeAlgorithm(_config(), model_seed=4, projection=None)
    projected = LinearProbeAlgorithm(_config(), model_seed=9, projection=_projection(2))
    state = vanilla.capture_inference_state()
    projected.restore_inference_state(LinearProbeState(state.weight, state.bias))
    assert torch.equal(
        projected.prepare_features(features), _projection(2).transform(features)
    )
    projected.restore_inference_state(state)
    assert torch.equal(projected.prepare_features(features), features)
    assert projected.mean_loss(features, targets) == vanilla.mean_loss(
        features, targets
    )


# GroupDRO, SD, RDM and LISA keep their own samplers and schedules; the pair term is
# added after their supervised reduction.
NewBase = Literal["groupdro", "sd", "rdm", "lisa"]
NEW_BASES: tuple[NewBase, ...] = ("groupdro", "sd", "rdm", "lisa")
_ENVIRONMENTS = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
# With the targets of `_data`, every target/color group holds two rows.
_GROUPS = _data()[1] * 2 + torch.tensor([0, 0, 1, 1, 0, 1, 1, 0])
_GROUPDRO_STEP = 0.1
_SD_WEIGHT = 0.1
_RDM_WEIGHT = 1.0
_RDM_VARIANCE = 0.004


def _new_base(kind: NewBase) -> LinearProbeTrainingMethod:
    if kind == "groupdro":
        return GroupDroLinearProbeMethod(_GROUPS, 4, _GROUPDRO_STEP)
    if kind == "sd":
        return SpectralDecouplingLinearProbeMethod(_SD_WEIGHT)
    if kind == "rdm":
        # The penalty and the optimizer reset start at the second update.
        return RdmLinearProbeMethod(_ENVIRONMENTS, 2, _RDM_WEIGHT, 1, _RDM_VARIANCE)
    return LisaLinearProbeMethod(_GROUPS, 4, 0.5)


def _composed(
    kind: NewBase, intervention: str, strength: float = 2.0
) -> ComposedLinearProbeMethod:
    method_id = cast(MethodId, f"{kind}_{intervention}")
    differences = _data()[2]
    if intervention == "grit":
        return ComposedLinearProbeMethod(_new_base(kind), method_id, _projection(2), 2)
    assert intervention == "consistency"
    return ComposedLinearProbeMethod(
        _new_base(kind),
        method_id,
        pair_differences=differences,
        consistency_weight=strength,
    )


def _epoch_rows(
    method: LinearProbeTrainingMethod, generator: torch.Generator
) -> tuple[torch.Tensor, ...]:
    return method.epoch_batches(row_count=8, batch_size=4, generator=generator)


def _run_epochs(
    method: LinearProbeTrainingMethod,
    algorithm: LinearProbeAlgorithm,
    epochs: int = 3,
) -> list[float]:
    features, targets, _ = _data()
    generator = torch.Generator().manual_seed(11)
    return [
        method.update(algorithm, features, targets, rows)
        for _ in range(epochs)
        for rows in _epoch_rows(method, generator)
    ]


@pytest.mark.parametrize("kind", NEW_BASES)
@pytest.mark.parametrize("control", ["identity", "zero_consistency"])
def test_new_base_controls_reproduce_direct_base_updates(
    kind: NewBase, control: str
) -> None:
    plain = _new_base(kind)
    if control == "identity":
        composed = ComposedLinearProbeMethod(
            _new_base(kind),
            cast(MethodId, f"{kind}_grit"),
            projection=_projection(0),
            projection_rank=0,
        )
    else:
        composed = _composed(kind, "consistency", strength=0.0)
    vanilla = plain.build_algorithm(_config(), model_seed=3, num_classes=2)
    intervened = composed.build_algorithm(_config(), model_seed=3, num_classes=2)
    # Three epochs cross RDM's activation and reset and GroupDRO's state updates.
    assert _run_epochs(plain, vanilla) == _run_epochs(composed, intervened)
    expected = vanilla.capture_inference_state()
    actual = intervened.capture_inference_state()
    assert torch.equal(expected.weight, actual.weight)
    assert torch.equal(expected.bias, actual.bias)
    if kind == "groupdro":
        assert isinstance(plain, GroupDroLinearProbeMethod)
        assert isinstance(composed.base, GroupDroLinearProbeMethod)
        assert torch.equal(
            plain._objective.probabilities,  # pyright: ignore[reportPrivateUsage]
            composed.base._objective.probabilities,  # pyright: ignore[reportPrivateUsage]
        )


def _explicit_base_loss(
    kind: NewBase,
    method: ComposedLinearProbeMethod,
    logits_of: Callable[[torch.Tensor], torch.Tensor],
    rows: torch.Tensor,
    update_index: int,
) -> torch.Tensor:
    """Each base objective written out from its protocol, on supervised terms only."""

    features, targets, _ = _data()
    if kind == "lisa":
        base = cast(LisaLinearProbeMethod, method.base)
        plan = base._planned[0]  # pyright: ignore[reportPrivateUsage]
        assert torch.equal(plan.rows, rows)
        weights = plan.mixing_weights.unsqueeze(1)
        mixed = (
            weights * features[plan.rows] + (1 - weights) * features[plan.partner_rows]
        )
        soft = weights * functional.one_hot(targets[plan.rows], 2) + (
            1 - weights
        ) * functional.one_hot(targets[plan.partner_rows], 2)
        return -(functional.log_softmax(logits_of(mixed), dim=1) * soft).sum(1).mean()
    logits = logits_of(features[rows])
    if kind == "sd":
        return functional.cross_entropy(logits, targets[rows]) + (
            _SD_WEIGHT * logits.square().mean()
        )
    per_example = functional.cross_entropy(logits, targets[rows], reduction="none")
    if kind == "groupdro":
        base = cast(GroupDroLinearProbeMethod, method.base)
        groups = _GROUPS[rows]
        group_losses = torch.stack(
            [
                per_example[groups == group].mean()
                if bool((groups == group).any())
                else per_example.new_zeros(())
                for group in range(4)
            ]
        )
        prior = base._objective.probabilities  # pyright: ignore[reportPrivateUsage]
        adversary = prior * torch.exp(_GROUPDRO_STEP * group_losses.detach())
        return group_losses.dot(adversary / adversary.sum())
    active = update_index >= 1
    return rdm_objective(
        per_example,
        _ENVIRONMENTS[rows],
        environment_count=2,
        penalty_weight=_RDM_WEIGHT if active else 0.0,
        variance_weight=_RDM_VARIANCE if active else 0.0,
    )


@pytest.mark.parametrize("kind", NEW_BASES)
def test_new_base_pair_term_is_added_outside_explicit_base_objective(
    kind: NewBase,
) -> None:
    method = _composed(kind, "consistency")
    algorithm = method.build_algorithm(_config(), model_seed=6, num_classes=2)
    features, targets, differences = _data()
    layers = (algorithm._model,)  # pyright: ignore[reportPrivateUsage]
    generator = torch.Generator().manual_seed(11)
    update_index = 0
    for _ in range(2):
        for rows in _epoch_rows(method, generator):
            parameters = [
                tensor.detach().clone().requires_grad_()
                for layer in layers
                for tensor in (layer.weight, layer.bias)
            ]

            def logits_of(
                inputs: torch.Tensor, layer_parameters: list[torch.Tensor] = parameters
            ) -> torch.Tensor:
                for index in range(0, len(layer_parameters), 2):
                    inputs = functional.linear(
                        inputs, layer_parameters[index], layer_parameters[index + 1]
                    )
                return inputs

            # Squared paired logits.
            pair_term = (differences @ parameters[0].T).square().sum(dim=1).mean()
            expected = (
                _explicit_base_loss(kind, method, logits_of, rows, update_index)
                + 2.0 * pair_term
            )
            gradients = torch.autograd.grad(expected, parameters)
            actual = method.update(algorithm, features, targets, rows)
            assert math.isclose(actual, float(expected.detach()), rel_tol=1e-5)
            observed = [
                tensor.grad for layer in layers for tensor in (layer.weight, layer.bias)
            ]
            for gradient, reference in zip(observed, gradients, strict=True):
                assert gradient is not None
                torch.testing.assert_close(gradient, reference)
            update_index += 1


def test_groupdro_adversary_sees_supervised_group_risks_only() -> None:
    features, targets, _ = _data()
    rows = torch.arange(8)
    probabilities: dict[str, torch.Tensor] = {}
    for label, method in (
        ("vanilla", _new_base("groupdro")),
        ("consistency", _composed("groupdro", "consistency", strength=10.0)),
    ):
        algorithm = method.build_algorithm(_config(), model_seed=7, num_classes=2)
        method.update(algorithm, features, targets, rows)
        base = method.base if isinstance(method, ComposedLinearProbeMethod) else method
        assert isinstance(base, GroupDroLinearProbeMethod)
        probabilities[label] = base._objective.probabilities  # pyright: ignore[reportPrivateUsage]
    # Identical starting logits give identical adversary updates, pairs or not.
    assert not torch.equal(probabilities["vanilla"], torch.full((4,), 0.25))
    assert torch.equal(probabilities["vanilla"], probabilities["consistency"])


def test_rdm_pairs_act_in_warmup_and_activation_resets_all_trainable_layers() -> None:
    method = _composed("rdm", "consistency")
    algorithm = method.build_algorithm(_config(), model_seed=8, num_classes=2)
    features, targets, _ = _data()
    generator = torch.Generator().manual_seed(11)
    optimizers: list[torch.optim.Optimizer] = []
    for _ in range(2):
        for rows in _epoch_rows(method, generator):
            method.update(algorithm, features, targets, rows)
            optimizers.append(algorithm._optimizer)  # pyright: ignore[reportPrivateUsage]
    # Update 1 activates the penalty with one reset; the schedule is not restarted.
    assert [optimizer is optimizers[0] for optimizer in optimizers] == [
        True,
        False,
        False,
        False,
    ]
    assert all(optimizer is optimizers[1] for optimizer in optimizers[1:])
    layers = (algorithm._model,)  # pyright: ignore[reportPrivateUsage]
    assert {
        id(parameter)
        for group in optimizers[-1].param_groups
        for parameter in group["params"]
    } == {id(parameter) for layer in layers for parameter in layer.parameters()}
    many = RdmLinearProbeMethod(
        torch.tensor([0] * 16 + [1] * 16), 2, _RDM_WEIGHT, 0, _RDM_VARIANCE
    )
    composed = ComposedLinearProbeMethod(
        many, "rdm_consistency", pair_differences=_data()[2], consistency_weight=1.0
    )
    with pytest.raises(ValueError, match="at least 2 examples per environment"):
        composed.epoch_batches(
            row_count=32, batch_size=6, generator=torch.Generator().manual_seed(2)
        )


def test_lisa_projects_mixtures_and_mixes_labels_with_feature_weights() -> None:
    features, targets, _ = _data()
    projection = _projection(2)
    generator = torch.Generator().manual_seed(11)
    plan = plan_lisa_epoch(
        _GROUPS, group_count=4, batch_size=4, selection_prob=0.5, generator=generator
    )[0]
    mixed, soft = mix_features_and_targets(features, targets, plan, num_classes=2)
    weights = plan.mixing_weights.unsqueeze(1)
    torch.testing.assert_close(
        projection.transform(mixed),
        weights * projection.transform(features[plan.rows])
        + (1 - weights) * projection.transform(features[plan.partner_rows]),
    )
    torch.testing.assert_close(
        soft[:, 1],
        plan.mixing_weights * targets[plan.rows]
        + (1 - plan.mixing_weights) * targets[plan.partner_rows],
    )
    composed = _composed("lisa", "grit")
    projected = composed.build_algorithm(_config(), model_seed=4, num_classes=2)
    plain_method = _new_base("lisa")
    plain = plain_method.build_algorithm(_config(), model_seed=4, num_classes=2)
    # Same plans, since LISA sampling depends only on groups and the generator.
    composed_losses = _run_epochs(composed, projected, epochs=2)
    generator = torch.Generator().manual_seed(11)
    plain_losses = [
        plain_method.update(plain, projection.transform(features), targets, rows)
        for _ in range(2)
        for rows in _epoch_rows(plain_method, generator)
    ]
    for actual, expected in zip(composed_losses, plain_losses, strict=True):
        assert math.isclose(actual, expected, rel_tol=1e-5)
    torch.testing.assert_close(
        projected.capture_inference_state().weight,
        plain.capture_inference_state().weight,
    )
