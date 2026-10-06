"""Minimal strict experiment configuration boundaries."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias, cast

from pydantic import (
    Field,
    FiniteFloat,
    PositiveInt,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    model_validator,
)

from grit.methods.types import MethodId, consumes_pairs
from grit.schemas import (
    HELD_OUT_VALIDATION_NAMES,
    IN_DOMAIN_VALIDATION_NAMES,
    CmnistSelector,
    CmnistValidationName,
    HeldOutValidationName,
    StrictBoundaryModel,
    canonical_digest_value,
)

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]
Probability: TypeAlias = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]

OPENAI_CLIP_REVISION = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
OPENAI_CLIP_WEIGHTS_IDENTITY = (
    "sha256:40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
)
OPENAI_CLIP_PREPROCESSING_ID = f"openai-clip-vit-b32-preprocess@{OPENAI_CLIP_REVISION}"


class CmnistSourceCounts(StrictBoundaryModel):
    train_e01: PositiveInt
    train_e02: PositiveInt
    validation: PositiveInt
    test: PositiveInt


class CmnistDatasetConfig(StrictBoundaryModel):
    """Dataset identity bound to the approved deterministic partition algorithm."""

    dataset_id: Literal["cmnist"]
    construction_method_id: Literal["cmnist-stratified-hash-v1"]
    construction_seed: StrictInt
    label_flip_prob: Probability
    source_counts: CmnistSourceCounts
    training_split_names: tuple[Literal["train_e01", "train_e02"], ...]
    validation_split_names: tuple[CmnistValidationName, ...]
    final_test_split_name: Literal["test_ood"]

    @model_validator(mode="after")
    def _validate_required_splits(self) -> CmnistDatasetConfig:
        expected_training = ("train_e01", "train_e02")
        if self.training_split_names != expected_training:
            raise ValueError(f"training_split_names must be {expected_training!r}")
        names = self.validation_split_names
        if (
            len(names) != 3
            or names[:2] != IN_DOMAIN_VALIDATION_NAMES
            or names[2] not in HELD_OUT_VALIDATION_NAMES
        ):
            raise ValueError(
                "validation_split_names must be val_e01, val_e02, and one held-out "
                f"rendering from {HELD_OUT_VALIDATION_NAMES!r}"
            )
        return self

    @property
    def held_out_validation_split(self) -> HeldOutValidationName:
        return self.validation_split_names[2]  # pyright: ignore[reportReturnType]


class FrozenFeatureConfig(StrictBoundaryModel):
    """The only representation admitted by the initial contract boundary."""

    kind: Literal["frozen_features"]
    encoder_id: Literal["openai-clip-vit-b32", "synthetic-fake-512"]
    encoder_revision: NonEmptyStr
    weights_identity: NonEmptyStr
    preprocessing_identity: NonEmptyStr
    feature_dimension: Literal[512]
    normalization: Literal["none", "l2"]


class DisabledPairsConfig(StrictBoundaryModel):
    kind: Literal["disabled"]


class OraclePairsConfig(StrictBoundaryModel):
    """Configuration identity only; this does not implement pair construction."""

    kind: Literal["oracle"]
    construction_id: Literal["cmnist-clean-oracle-pairs-v1"]
    source_partition_ids: tuple[NonEmptyStr, ...]
    pair_count: PositiveInt
    pair_seed: StrictInt
    orientation: Literal["red_minus_green"]

    @model_validator(mode="after")
    def _validate_sources(self) -> OraclePairsConfig:
        expected_sources = ("train_e01_sources", "train_e02_sources")
        if self.source_partition_ids != expected_sources:
            raise ValueError(
                "initial CMNIST oracle pairs require the approved training source "
                f"partitions {expected_sources!r}"
            )
        return self


PairsConfig: TypeAlias = Annotated[
    DisabledPairsConfig | OraclePairsConfig,
    Field(discriminator="kind"),
]


class DisabledProjectionConfig(StrictBoundaryModel):
    kind: Literal["disabled"]


class LinearProjectionConfig(StrictBoundaryModel):
    """Projection settings only; no projection mathematics happens here."""

    kind: Literal["linear_pair_difference"]
    requested_rank: Annotated[StrictInt, Field(ge=0, le=24)]
    center_differences: Literal[False]
    relative_singular_value_tolerance: Annotated[StrictFloat, Field(gt=0.0)]


ProjectionConfig: TypeAlias = Annotated[
    DisabledProjectionConfig | LinearProjectionConfig,
    Field(discriminator="kind"),
]


class ErmAlgorithmConfig(StrictBoundaryModel):
    kind: Literal["erm"]


class GritAlgorithmConfig(StrictBoundaryModel):
    kind: Literal["grit"]


# CMNIST binds the two rendered training sources; Waterbirds binds the training
# background. The bindings are named so a config states what its environments are.
EnvironmentNames: TypeAlias = (
    tuple[Literal["train_e01"], Literal["train_e02"]]
    | tuple[Literal["background_land"], Literal["background_water"]]
)
# `target_*` groups are label x attribute. GroupDRO's `training_environment` groups
# are the training environments alone; `target_environment` groups (GroupDRO) and
# domains (LISA) are the training environments crossed with the label.
GroupDefinition: TypeAlias = Literal[
    "target_color", "target_background", "training_environment", "target_environment"
]


class GroupDroAlgorithmConfig(StrictBoundaryModel):
    kind: Literal["groupdro"]
    group_definition: GroupDefinition
    adversarial_step_size: Annotated[StrictFloat, Field(gt=0.0)]
    sampling: Literal["inverse_group_frequency_with_replacement"]
    generalization_adjustment: Annotated[StrictFloat, Field(ge=0.0, le=0.0)]
    normalize_loss: Literal[False]


class RexAlgorithmConfig(StrictBoundaryModel):
    kind: Literal["rex"]
    environment_names: EnvironmentNames
    penalty_weight: Annotated[StrictFloat, Field(gt=0.0)]
    penalty_anneal_updates: NonNegativeInt
    risk_variance: Literal["population"]
    sampling: Literal["environment_balanced_without_replacement"]
    loss_rescaling: Literal["divide_by_penalty_weight_above_one"]


class IrmAlgorithmConfig(StrictBoundaryModel):
    kind: Literal["irm"]
    environment_names: EnvironmentNames
    penalty_weight: Annotated[StrictFloat, Field(gt=0.0)]
    penalty_anneal_updates: NonNegativeInt
    penalty: Literal["irmv1_dummy_classifier_scale"]
    sampling: Literal["environment_balanced_without_replacement"]
    loss_rescaling: Literal["divide_by_penalty_weight_above_one"]


class LisaAlgorithmConfig(StrictBoundaryModel):
    kind: Literal["lisa"]
    group_definition: GroupDefinition
    selection_prob: Probability
    mixing: Literal["beta_2_2"]
    sampling: Literal["uniform_single_group_batches"]


class SdAlgorithmConfig(StrictBoundaryModel):
    """Spectral Decoupling: no environments, no warm-up, ERM sampling."""

    kind: Literal["sd"]
    penalty_weight: Annotated[StrictFloat, Field(gt=0.0)]
    penalty: Literal["mean_squared_logits"]
    sampling: Literal["uniform_without_replacement"]


class FishrAlgorithmConfig(StrictBoundaryModel):
    kind: Literal["fishr"]
    environment_names: EnvironmentNames
    penalty_weight: Annotated[StrictFloat, Field(gt=0.0)]
    penalty_anneal_updates: NonNegativeInt
    ema: Annotated[StrictFloat, Field(ge=0.0, lt=1.0)]
    penalty: Literal["classifier_gradient_variance_distance"]
    sampling: Literal["environment_balanced_without_replacement"]


class RdmAlgorithmConfig(StrictBoundaryModel):
    kind: Literal["rdm"]
    environment_names: EnvironmentNames
    penalty_weight: Annotated[StrictFloat, Field(gt=0.0)]
    penalty_anneal_updates: NonNegativeInt
    variance_weight: Annotated[StrictFloat, Field(ge=0.0)]
    penalty: Literal["worst_environment_versus_pooled_risk_mmd"]
    sampling: Literal["environment_balanced_without_replacement"]


class ComposedAlgorithmConfig(StrictBoundaryModel):
    """A supervised objective with an independent predictor/intervention choice."""

    kind: Literal["composed"]
    base_objective: Annotated[
        ErmAlgorithmConfig
        | RexAlgorithmConfig
        | IrmAlgorithmConfig
        | FishrAlgorithmConfig
        | GroupDroAlgorithmConfig
        | SdAlgorithmConfig
        | RdmAlgorithmConfig
        | LisaAlgorithmConfig,
        Field(discriminator="kind"),
    ]
    pair_intervention: Literal["grit", "consistency"]
    consistency_weight: Annotated[StrictFloat, Field(ge=0.0)] | None = None
    objective_version: Literal["prediction-pairs/v1"] = "prediction-pairs/v1"

    @model_validator(mode="after")
    def _validate_intervention(self) -> ComposedAlgorithmConfig:
        if (self.pair_intervention == "consistency") != (
            self.consistency_weight is not None
        ):
            raise ValueError("only consistency requires consistency_weight")
        if self.base_objective.kind == "erm" and self.pair_intervention == "grit":
            raise ValueError("use the existing grit algorithm for ERM plus GRIT")
        return self


AlgorithmConfig: TypeAlias = Annotated[
    ErmAlgorithmConfig
    | GritAlgorithmConfig
    | GroupDroAlgorithmConfig
    | RexAlgorithmConfig
    | IrmAlgorithmConfig
    | LisaAlgorithmConfig
    | SdAlgorithmConfig
    | FishrAlgorithmConfig
    | RdmAlgorithmConfig
    | ComposedAlgorithmConfig,
    Field(discriminator="kind"),
]


def algorithm_method_id(algorithm: AlgorithmConfig) -> MethodId:
    if isinstance(algorithm, ComposedAlgorithmConfig):
        return cast(
            MethodId, f"{algorithm.base_objective.kind}_{algorithm.pair_intervention}"
        )
    return algorithm.kind


class LinearProbeTrainingConfig(StrictBoundaryModel):
    optimizer: Literal["adam"]
    batch_size: PositiveInt
    learning_rate: Annotated[StrictFloat, Field(gt=0.0)]
    weight_decay: Annotated[StrictFloat, Field(ge=0.0)]
    max_epochs: PositiveInt


class CpuRuntimeConfig(StrictBoundaryModel):
    device: Literal["cpu"]
    deterministic_algorithms: Literal[True]


class SeedSets(StrictBoundaryModel):
    """Prespecified search stages; scheduling happens in the search layer."""

    tuning: tuple[StrictInt, ...]
    confirmation: tuple[StrictInt, ...]
    final: tuple[StrictInt, ...]

    @model_validator(mode="after")
    def _validate_seed_sets(self) -> SeedSets:
        named = {
            "tuning": self.tuning,
            "confirmation": self.confirmation,
            "final": self.final,
        }
        expected_counts = {"tuning": 3, "confirmation": 2, "final": 10}
        for name, values in named.items():
            if not values:
                raise ValueError(f"{name} seed set must not be empty")
            if len(values) != expected_counts[name]:
                raise ValueError(
                    f"{name} seed set must contain {expected_counts[name]} seeds"
                )
            if len(set(values)) != len(values):
                raise ValueError(f"{name} seed set contains duplicates")
        if set(self.tuning) & set(self.confirmation):
            raise ValueError("tuning and confirmation seeds must be disjoint")
        if set(self.tuning) & set(self.final):
            raise ValueError("tuning and final seeds must be disjoint")
        if set(self.confirmation) & set(self.final):
            raise ValueError("confirmation and final seeds must be disjoint")
        return self


class OrdinarySelectionConfig(StrictBoundaryModel):
    selector: CmnistSelector


class CmnistArtifactLineageConfig(StrictBoundaryModel):
    """Prepared-artifact identity required by reportable CMNIST runs."""

    dataset_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    pair_manifest_digest: NonEmptyStr | None


class _CommonCmnistExperimentConfig(StrictBoundaryModel):
    schema_version: Literal["grit.experiment/v1"]
    experiment_name: NonEmptyStr
    protocol_id: Literal["cmnist/v1"]
    reportable: StrictBool
    dataset: CmnistDatasetConfig
    representation: FrozenFeatureConfig
    pairs: PairsConfig
    projection: ProjectionConfig
    algorithm: AlgorithmConfig
    training: LinearProbeTrainingConfig
    runtime: CpuRuntimeConfig
    seed_sets: SeedSets
    artifact_lineage: CmnistArtifactLineageConfig | None = None

    def scientific_config_digest(self) -> str:
        """Identify trainable candidate fields independently of selector and seeds."""

        candidate_fields = self.model_dump(
            mode="json",
            exclude={
                "experiment_name",
                "run_kind",
                "seed_sets",
                "selection",
            },
        )
        return canonical_digest_value(candidate_fields)

    @model_validator(mode="after")
    def _validate_algorithm_components(self) -> _CommonCmnistExperimentConfig:
        uses_pairs = consumes_pairs(algorithm_method_id(self.algorithm))
        if not uses_pairs:
            if not isinstance(self.pairs, DisabledPairsConfig):
                raise ValueError(
                    f"{self.algorithm.kind} requires pairs.kind='disabled'"
                )
        elif not isinstance(self.pairs, OraclePairsConfig):
            raise ValueError(f"{self.algorithm.kind} requires pairs.kind='oracle'")
        # A composed objective keeps its base's training-only metadata bindings.
        base = (
            self.algorithm.base_objective
            if isinstance(self.algorithm, ComposedAlgorithmConfig)
            else self.algorithm
        )
        if isinstance(base, GroupDroAlgorithmConfig) and base.group_definition not in (
            "target_color",
            "training_environment",
            "target_environment",
        ):
            raise ValueError(
                "CMNIST GroupDRO uses target-color, training-environment or "
                "target-environment groups"
            )
        if isinstance(base, LisaAlgorithmConfig) and base.group_definition not in (
            "target_color",
            "target_environment",
        ):
            raise ValueError(
                "CMNIST LISA uses target-color groups or target-environment domains"
            )
        if isinstance(
            base,
            RexAlgorithmConfig
            | IrmAlgorithmConfig
            | FishrAlgorithmConfig
            | RdmAlgorithmConfig,
        ) and base.environment_names != ("train_e01", "train_e02"):
            raise ValueError("CMNIST invariant methods use the training sources")
        if isinstance(self.algorithm, GritAlgorithmConfig) or (
            isinstance(self.algorithm, ComposedAlgorithmConfig)
            and self.algorithm.pair_intervention == "grit"
        ):
            if not isinstance(self.projection, LinearProjectionConfig):
                raise ValueError(
                    "initial GRIT requires projection.kind='linear_pair_difference'"
                )
        elif not isinstance(self.projection, DisabledProjectionConfig):
            raise ValueError(
                f"{self.algorithm.kind} requires projection.kind='disabled'"
            )
        if self.reportable:
            if self.artifact_lineage is None:
                raise ValueError("reportable CMNIST requires prepared-artifact lineage")
            if not uses_pairs:
                if self.artifact_lineage.pair_manifest_digest is not None:
                    raise ValueError(
                        f"reportable CMNIST {self.algorithm.kind} cannot bind "
                        "oracle pairs"
                    )
            elif self.artifact_lineage.pair_manifest_digest is None:
                raise ValueError(
                    f"reportable CMNIST {self.algorithm.kind} requires oracle-pair "
                    "lineage"
                )
            counts = self.dataset.source_counts
            if (counts.train_e01, counts.train_e02, counts.validation, counts.test) != (
                25_000,
                25_000,
                10_000,
                10_000,
            ):
                raise ValueError("reportable CMNIST requires production source counts")
            if float(self.dataset.label_flip_prob) != 0.25:
                raise ValueError("reportable CMNIST requires label_flip_prob=0.25")
            if self.representation.encoder_id != "openai-clip-vit-b32":
                raise ValueError("reportable CMNIST requires official OpenAI CLIP")
            representation_identity = (
                self.representation.encoder_revision,
                self.representation.weights_identity,
                self.representation.preprocessing_identity,
            )
            if representation_identity != (
                OPENAI_CLIP_REVISION,
                OPENAI_CLIP_WEIGHTS_IDENTITY,
                OPENAI_CLIP_PREPROCESSING_ID,
            ):
                raise ValueError(
                    "reportable CMNIST requires the pinned official CLIP identity"
                )
        return self


class OrdinaryExperimentConfig(_CommonCmnistExperimentConfig):
    run_kind: Literal["ordinary"]
    selection: OrdinarySelectionConfig


ExperimentConfig: TypeAlias = OrdinaryExperimentConfig

_EXPERIMENT_CONFIG_ADAPTER: TypeAdapter[ExperimentConfig] = TypeAdapter(
    ExperimentConfig
)


def parse_experiment_config_json(payload: str) -> ExperimentConfig:
    """Parse canonical or authored JSON through the discriminated strict root."""

    return _EXPERIMENT_CONFIG_ADAPTER.validate_json(payload)
