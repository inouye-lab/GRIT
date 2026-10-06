"""Strict configuration and role-scoped data boundary tests."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from grit.config import (
    CmnistArtifactLineageConfig,
    DisabledPairsConfig,
    ErmAlgorithmConfig,
    LinearProjectionConfig,
    OraclePairsConfig,
    OrdinaryExperimentConfig,
    OrdinarySelectionConfig,
    parse_experiment_config_json,
)
from grit.data.views import (
    ExampleIdentity,
    FinalTestHandle,
    FinalTestSplitDescriptor,
    TrainingSplitDescriptor,
    TrainingView,
    ValidationSplitDescriptor,
    ValidationView,
    issue_final_test_handle,
    validate_cmnist_repeated_validation_views,
)
from grit.schemas import CmnistSelector
from tests.contract_fixtures import (
    dataset_config,
    feature_config,
    ordinary_erm_config,
    ordinary_grit_config,
    runtime_config,
    seed_sets,
    training_config,
)


def _examples(view_id: str) -> tuple[ExampleIdentity, ...]:
    return tuple(
        ExampleIdentity(
            example_id=f"{view_id}:example:{index}",
            source_id=f"source:{index}",
            view_id=view_id,
        )
        for index in range(3)
    )


def _validation_view(name: str) -> ValidationView:
    return ValidationView(
        descriptor=ValidationSplitDescriptor(
            dataset_id="cmnist",
            manifest_id="manifest:synthetic",
            name=name,
            role="validation",
            source_partition_id="validation_sources",
            view_id=name,
        ),
        examples=_examples(name),
    )


def test_configuration_round_trip_digest_and_strict_nested_fields() -> None:
    for config in (ordinary_erm_config(), ordinary_grit_config()):
        payload = config.canonical_json()
        reparsed = parse_experiment_config_json(payload)
        assert reparsed == config
        assert reparsed.canonical_json() == payload
        assert reparsed.canonical_digest() == config.canonical_digest()

    payload_with_unknown = (
        ordinary_erm_config()
        .canonical_json()
        .replace(
            '"encoder_id":"openai-clip-vit-b32"',
            '"encoder_id":"openai-clip-vit-b32","unexpected":true',
        )
    )
    with pytest.raises(ValidationError, match="unexpected"):
        parse_experiment_config_json(payload_with_unknown)


def test_scientific_candidate_identity_excludes_selector_branch_only() -> None:
    primary = ordinary_erm_config(CmnistSelector.PRIMARY_ROBUST)
    secondary = ordinary_erm_config(CmnistSelector.SECONDARY_SOURCE)
    assert primary.canonical_digest() != secondary.canonical_digest()
    assert primary.scientific_config_digest() == secondary.scientific_config_digest()


def test_reportable_config_requires_pinned_official_clip() -> None:
    payload = ordinary_erm_config().model_dump(mode="python")
    payload["reportable"] = True
    payload["artifact_lineage"] = CmnistArtifactLineageConfig(
        dataset_manifest_digest="sha256:dataset",
        feature_cache_manifest_digest="sha256:features",
        pair_manifest_digest=None,
    )
    reportable = OrdinaryExperimentConfig.model_validate(payload)
    assert reportable.reportable is True

    wrong_revision = reportable.model_dump(mode="python")
    wrong_revision["representation"]["encoder_revision"] = "unverified-revision"
    with pytest.raises(ValidationError, match="pinned official CLIP identity"):
        OrdinaryExperimentConfig.model_validate(wrong_revision)


def test_unsupported_schema_and_lossy_values_are_rejected() -> None:
    unsupported = (
        ordinary_erm_config()
        .canonical_json()
        .replace("grit.experiment/v1", "grit.experiment/v999")
    )
    with pytest.raises(ValidationError, match="schema_version"):
        parse_experiment_config_json(unsupported)

    with pytest.raises(ValidationError, match="tuning"):
        type(seed_sets()).model_validate(
            {
                "tuning": ("101",),
                "confirmation": (201,),
                "final": (301,),
            }
        )

    invalid_counts = seed_sets().model_copy(update={"confirmation": (201,)})
    with pytest.raises(ValidationError, match="confirmation seed set must contain 2"):
        type(seed_sets()).model_validate(invalid_counts.model_dump(mode="python"))


def test_erm_with_projection_is_rejected() -> None:
    with pytest.raises(ValidationError, match="projection.kind='disabled'"):
        OrdinaryExperimentConfig(
            schema_version="grit.experiment/v1",
            run_kind="ordinary",
            experiment_name="invalid-erm-projection",
            protocol_id="cmnist/v1",
            reportable=False,
            dataset=dataset_config(),
            representation=feature_config(),
            pairs=DisabledPairsConfig(kind="disabled"),
            projection=LinearProjectionConfig(
                kind="linear_pair_difference",
                requested_rank=1,
                center_differences=False,
                relative_singular_value_tolerance=1e-12,
            ),
            algorithm=ErmAlgorithmConfig(kind="erm"),
            training=training_config(),
            runtime=runtime_config(),
            seed_sets=seed_sets(),
            selection=OrdinarySelectionConfig(selector=CmnistSelector.PRIMARY_ROBUST),
        )

    with pytest.raises(ValidationError, match="less than or equal to 24"):
        LinearProjectionConfig(
            kind="linear_pair_difference",
            requested_rank=25,
            center_differences=False,
            relative_singular_value_tolerance=1e-12,
        )


def test_oracle_pairs_accept_only_approved_training_sources() -> None:
    with pytest.raises(ValidationError, match="cmnist-clean-oracle-pairs-v1"):
        OraclePairsConfig.model_validate(
            {
                "kind": "oracle",
                "construction_id": "invalid-pair-construction",
                "source_partition_ids": (
                    "train_e01_sources",
                    "train_e02_sources",
                ),
                "pair_count": 256,
                "pair_seed": 0,
                "orientation": "red_minus_green",
            }
        )
    with pytest.raises(ValidationError, match="approved training source partitions"):
        OraclePairsConfig(
            kind="oracle",
            construction_id="cmnist-clean-oracle-pairs-v1",
            source_partition_ids=("validation_sources", "test_sources"),
            pair_count=256,
            pair_seed=0,
            orientation="red_minus_green",
        )


def test_ordinary_configuration_rejects_test_selection_fields() -> None:
    ordinary = ordinary_erm_config().canonical_json()
    mixed = ordinary.replace(
        '"algorithm"',
        '"diagnostic_selection":{"selector":"test_ood_accuracy",'
        '"test_oracle":true},"algorithm"',
        1,
    )
    with pytest.raises(ValidationError, match="diagnostic_selection"):
        parse_experiment_config_json(mixed)

    wrong_discriminant = ordinary.replace(
        '"run_kind":"ordinary"',
        '"run_kind":"cmnist_test_oracle_diagnostic"',
    )
    with pytest.raises(ValidationError, match="run_kind"):
        parse_experiment_config_json(wrong_discriminant)


def test_scoped_views_reject_role_substitution_and_final_handle_is_opaque() -> None:
    training_descriptor = TrainingSplitDescriptor(
        dataset_id="cmnist",
        manifest_id="manifest:synthetic",
        name="train_e01",
        role="training",
        source_partition_id="train_e01_sources",
        view_id="train_e01",
    )
    training = TrainingView(
        descriptor=training_descriptor,
        examples=_examples("train_e01"),
    )
    assert training.descriptor.role == "training"

    with pytest.raises(ValidationError, match="validation"):
        ValidationView.model_validate(
            {
                "descriptor": training_descriptor,
                "examples": training.examples,
            }
        )

    with pytest.raises(ValidationError, match="approved CMNIST training split"):
        TrainingSplitDescriptor(
            dataset_id="cmnist",
            manifest_id="manifest:synthetic",
            name="test_ood",
            role="training",
            source_partition_id="test_sources",
            view_id="test_ood",
        )

    final_descriptor = FinalTestSplitDescriptor(
        dataset_id="cmnist",
        manifest_id="manifest:synthetic",
        name="test_ood",
        role="final_test",
        source_partition_id="test_sources",
        view_id="test_ood",
    )
    handle = issue_final_test_handle(
        handle_id="handle:final",
        run_id="run:final",
        candidate_id="candidate:a",
        scientific_config_digest=ordinary_erm_config().scientific_config_digest(),
        descriptor=final_descriptor,
        examples=_examples("test_ood"),
    )
    assert isinstance(handle, FinalTestHandle)
    assert not hasattr(handle, "examples")
    with pytest.raises(ValidationError):
        ValidationView.model_validate(handle)


def test_repeated_cmnist_validation_views_preserve_ordered_sources() -> None:
    views = (
        _validation_view("val_e01"),
        _validation_view("val_e02"),
        _validation_view("val_e05"),
    )
    validate_cmnist_repeated_validation_views(views)
    assert tuple(example.source_id for example in views[0].examples) == tuple(
        example.source_id for example in views[2].examples
    )
    assert views[0].examples[0].example_id != views[2].examples[0].example_id

    mismatched = ValidationView(
        descriptor=views[2].descriptor,
        examples=(
            ExampleIdentity(
                example_id="val_e05:example:0",
                source_id="different-source",
                view_id="val_e05",
            ),
            *views[2].examples[1:],
        ),
    )
    with pytest.raises(ValueError, match="ordered source identities"):
        validate_cmnist_repeated_validation_views((views[0], views[1], mismatched))

    mismatched_manifest = ValidationView(
        descriptor=views[2].descriptor.model_copy(
            update={"manifest_id": "manifest:different"}
        ),
        examples=views[2].examples,
    )
    with pytest.raises(ValueError, match="manifest identity"):
        validate_cmnist_repeated_validation_views(
            (views[0], views[1], mismatched_manifest)
        )

    repeated_example_ids = cast(
        tuple[ValidationView, ValidationView, ValidationView],
        tuple(
            ValidationView(
                descriptor=view.descriptor,
                examples=tuple(
                    example.model_copy(update={"example_id": f"shared:{index}"})
                    for index, example in enumerate(view.examples)
                ),
            )
            for view in views
        ),
    )
    with pytest.raises(ValueError, match="distinct example identities"):
        validate_cmnist_repeated_validation_views(repeated_example_ids)

    invalid_role = cast(
        ValidationView,
        cast(
            object,
            TrainingView(
                descriptor=TrainingSplitDescriptor(
                    dataset_id="cmnist",
                    manifest_id="manifest:synthetic",
                    name="train_e01",
                    role="training",
                    source_partition_id="train_e01_sources",
                    view_id="train_e01",
                ),
                examples=_examples("train_e01"),
            ),
        ),
    )
    with pytest.raises(TypeError, match="ValidationView"):
        validate_cmnist_repeated_validation_views((invalid_role, views[1], views[2]))
