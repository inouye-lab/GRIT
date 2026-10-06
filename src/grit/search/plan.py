"""Production local-search configuration and canonical planning boundaries."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeAlias, cast

from pydantic import (
    Field,
    PositiveInt,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    model_validator,
)

from grit.config import (
    OPENAI_CLIP_PREPROCESSING_ID,
    OPENAI_CLIP_REVISION,
    OPENAI_CLIP_WEIGHTS_IDENTITY,
    AlgorithmConfig,
    CmnistArtifactLineageConfig,
    CmnistDatasetConfig,
    CmnistSourceCounts,
    ComposedAlgorithmConfig,
    CpuRuntimeConfig,
    DisabledPairsConfig,
    DisabledProjectionConfig,
    EnvironmentNames,
    ErmAlgorithmConfig,
    FishrAlgorithmConfig,
    FrozenFeatureConfig,
    GritAlgorithmConfig,
    GroupDefinition,
    GroupDroAlgorithmConfig,
    IrmAlgorithmConfig,
    LinearProbeTrainingConfig,
    LinearProjectionConfig,
    LisaAlgorithmConfig,
    OraclePairsConfig,
    OrdinaryExperimentConfig,
    OrdinarySelectionConfig,
    RdmAlgorithmConfig,
    RexAlgorithmConfig,
    SdAlgorithmConfig,
    SeedSets,
)
from grit.data.cmnist import (
    CmnistDatasetManifest,
    CmnistOraclePairManifest,
)
from grit.data.waterbirds import (
    WaterbirdsDatasetManifest,
    mint_waterbirds_adjusted_weight_spec,
)
from grit.data.waterbirds_pairs import WaterbirdsOraclePairManifest
from grit.features.cmnist import CmnistFeatureCacheManifest, EncoderIdentity
from grit.features.waterbirds import WaterbirdsFeatureCacheManifest
from grit.methods.types import (
    IMPLEMENTED_METHODS,
    METHOD_LABELS,
    MethodId,
    base_objective,
    consumes_pairs,
    is_composed,
    pair_intervention,
)
from grit.paths import REPO_ROOT, expand_config_path
from grit.results import CodeProvenance, EnvironmentProvenance
from grit.schemas import (
    DEFAULT_HELD_OUT_VALIDATION_NAME,
    CmnistSelector,
    HeldOutValidationName,
    StrictBoundaryModel,
    canonical_digest_value,
)
from grit.search.waterbirds_contracts import WaterbirdsCandidateConfig

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
Normalization: TypeAlias = Literal["none", "l2"]


# Defaults used by the checked-in configs and tests. Any grid may be configured.
APPROVED_LEARNING_RATES: tuple[float, float, float, float] = (
    0.0001,
    0.0003,
    0.001,
    0.003,
)
APPROVED_WEIGHT_DECAYS: tuple[float, float, float, float] = (
    0.0,
    0.00001,
    0.0001,
    0.001,
)
APPROVED_RANKS: tuple[int, ...] = tuple(range(2, 25))
APPROVED_GROUPDRO_STEP_SIZES: tuple[float, float, float] = (0.001, 0.01, 0.1)
APPROVED_REX_PENALTY_WEIGHTS: tuple[float, float, float, float] = (
    10.0,
    100.0,
    1_000.0,
    10_000.0,
)
APPROVED_IRM_PENALTY_WEIGHTS: tuple[float, float, float, float] = (
    100.0,
    1_000.0,
    10_000.0,
    100_000.0,
)
REX_PENALTY_ANNEAL_UPDATES = 100
IRM_PENALTY_ANNEAL_UPDATES = 190
APPROVED_LISA_SELECTION_PROBS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
APPROVED_SD_PENALTY_WEIGHTS: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
APPROVED_FISHR_PENALTY_WEIGHTS: tuple[float, ...] = (10.0, 100.0, 1_000.0, 10_000.0)
APPROVED_RDM_PENALTY_WEIGHTS: tuple[float, ...] = (0.1, 1.0, 5.0, 10.0)
# The reference's default update thresholds, not a fraction of this study's training.
FISHR_PENALTY_ANNEAL_UPDATES = 1500
RDM_PENALTY_ANNEAL_UPDATES = 1500
FISHR_EMA = 0.95
RDM_VARIANCE_WEIGHT = 0.004


def training_optimizer(method: MethodId) -> Literal["adam"]:
    """Every method trains with Adam."""

    del method
    return "adam"


class _YamlModule(Protocol):
    def safe_load(self, payload: str) -> object: ...


_yaml = cast(_YamlModule, importlib.import_module("yaml"))


class SearchArtifactPaths(StrictBoundaryModel):
    dataset_manifest: NonEmptyStr
    feature_cache_manifest: NonEmptyStr
    oracle_pair_manifest: NonEmptyStr


class SearchSeedConfig(StrictBoundaryModel):
    construction: StrictInt
    pairs: StrictInt
    stages: SeedSets


class SearchSpaceConfig(StrictBoundaryModel):
    """The grid is whatever the YAML says; the plan records what actually ran."""

    methods: Annotated[tuple[MethodId, ...], Field(min_length=1)]
    learning_rates: Annotated[tuple[StrictFloat, ...], Field(min_length=1)]
    weight_decays: Annotated[tuple[StrictFloat, ...], Field(min_length=1)]
    consistency_weights: tuple[StrictFloat, ...] = Field(
        default=(), exclude_if=lambda values: not values
    )
    projection_ranks: tuple[StrictInt, ...] = Field(
        default=(), exclude_if=lambda values: not values
    )
    groupdro_step_sizes: tuple[StrictFloat, ...] = Field(
        default=(), exclude_if=lambda values: not values
    )
    # Unset keeps each dataset's label x attribute groups, so existing configs and
    # candidate identities are unchanged. `training_environment` groups by source
    # alone; `target_environment` by label x source (CMNIST only).
    groupdro_group_definition: (
        Literal["training_environment", "target_environment"] | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    rex_penalty_weights: tuple[StrictFloat, ...] = Field(
        default=(), exclude_if=lambda values: not values
    )
    rex_penalty_anneal_updates: Annotated[StrictInt, Field(ge=0)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    irm_penalty_weights: tuple[StrictFloat, ...] = Field(
        default=(), exclude_if=lambda values: not values
    )
    irm_penalty_anneal_updates: Annotated[StrictInt, Field(ge=0)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    lisa_selection_probs: tuple[StrictFloat, ...] = Field(
        default=(), exclude_if=lambda values: not values
    )
    lisa_group_definition: Literal["target_environment"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    sd_penalty_weights: tuple[StrictFloat, ...] = Field(
        default=(), exclude_if=lambda values: not values
    )
    fishr_penalty_weights: tuple[StrictFloat, ...] = Field(
        default=(), exclude_if=lambda values: not values
    )
    fishr_penalty_anneal_updates: Annotated[StrictInt, Field(ge=0)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    fishr_ema: Annotated[StrictFloat, Field(ge=0.0, lt=1.0)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    rdm_penalty_weights: tuple[StrictFloat, ...] = Field(
        default=(), exclude_if=lambda values: not values
    )
    rdm_penalty_anneal_updates: Annotated[StrictInt, Field(ge=0)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    rdm_variance_weight: Annotated[StrictFloat, Field(ge=0.0)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def _validate_space(self) -> SearchSpaceConfig:
        bases = {base_objective(method) for method in self.methods}
        if any(value < 0 for value in self.consistency_weights):
            raise ValueError("consistency_weights must be non-negative")
        if any(pair_intervention(m) == "consistency" for m in self.methods) != bool(
            self.consistency_weights
        ):
            raise ValueError(
                "consistency_weights required exactly for consistency variants"
            )
        if len(set(self.consistency_weights)) != len(self.consistency_weights):
            raise ValueError("consistency_weights contains duplicate values")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("production search methods must be unique")
        if tuple(sorted(self.methods, key=IMPLEMENTED_METHODS.index)) != self.methods:
            raise ValueError("production search methods must use canonical order")
        for name, values in (
            ("learning_rates", self.learning_rates),
            ("weight_decays", self.weight_decays),
            ("projection_ranks", self.projection_ranks),
            ("groupdro_step_sizes", self.groupdro_step_sizes),
            ("rex_penalty_weights", self.rex_penalty_weights),
            ("irm_penalty_weights", self.irm_penalty_weights),
            ("lisa_selection_probs", self.lisa_selection_probs),
            ("sd_penalty_weights", self.sd_penalty_weights),
            ("fishr_penalty_weights", self.fishr_penalty_weights),
            ("rdm_penalty_weights", self.rdm_penalty_weights),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicate values")
        if any(not 0.0 <= value <= 1.0 for value in self.lisa_selection_probs):
            raise ValueError("lisa_selection_probs must lie in [0, 1]")
        for name, weights in (
            ("sd_penalty_weights", self.sd_penalty_weights),
            ("fishr_penalty_weights", self.fishr_penalty_weights),
            ("rdm_penalty_weights", self.rdm_penalty_weights),
        ):
            if any(value <= 0 for value in weights):
                raise ValueError(f"{name} must be positive")
        selected_grids: tuple[
            tuple[MethodId, str, tuple[float, ...] | tuple[int, ...]], ...
        ] = (
            ("lisa", "lisa_selection_probs", self.lisa_selection_probs),
            ("sd", "sd_penalty_weights", self.sd_penalty_weights),
            ("fishr", "fishr_penalty_weights", self.fishr_penalty_weights),
            ("rdm", "rdm_penalty_weights", self.rdm_penalty_weights),
        )
        for method, name, values in selected_grids:
            if (method in bases) != bool(values):
                raise ValueError(
                    f"{name} must be non-empty exactly when {METHOD_LABELS[method]} "
                    "is selected"
                )
        if any(value <= 0 for value in self.learning_rates):
            raise ValueError("learning_rates must be positive")
        if any(value < 0 for value in self.weight_decays):
            raise ValueError("weight_decays must be non-negative")
        if any(value < 0 for value in self.projection_ranks):
            raise ValueError("projection_ranks must be non-negative")
        if any(value <= 0 for value in self.groupdro_step_sizes):
            raise ValueError("groupdro_step_sizes must be positive")
        if any(value <= 0 for value in self.rex_penalty_weights):
            raise ValueError("rex_penalty_weights must be positive")
        if any(value <= 0 for value in self.irm_penalty_weights):
            raise ValueError("irm_penalty_weights must be positive")
        if (any(pair_intervention(m) == "grit" for m in self.methods)) != bool(
            self.projection_ranks
        ):
            raise ValueError(
                "projection_ranks must be non-empty exactly when GRIT is selected"
            )
        if ("groupdro" in bases) != bool(self.groupdro_step_sizes):
            raise ValueError(
                "groupdro_step_sizes must be non-empty exactly when GroupDRO is "
                "selected"
            )
        if self.groupdro_group_definition is not None and "groupdro" not in bases:
            raise ValueError("groupdro_group_definition requires GroupDRO")
        if self.lisa_group_definition is not None and "lisa" not in bases:
            raise ValueError("lisa_group_definition requires LISA")
        rex_selected = "rex" in bases
        if rex_selected != bool(self.rex_penalty_weights):
            raise ValueError(
                "rex_penalty_weights must be non-empty exactly when REx is selected"
            )
        if rex_selected != (self.rex_penalty_anneal_updates is not None):
            raise ValueError(
                "rex_penalty_anneal_updates must be set exactly when REx is selected"
            )
        irm_selected = "irm" in bases
        if irm_selected != bool(self.irm_penalty_weights):
            raise ValueError(
                "irm_penalty_weights must be non-empty exactly when IRM is selected"
            )
        if irm_selected != (self.irm_penalty_anneal_updates is not None):
            raise ValueError(
                "irm_penalty_anneal_updates must be set exactly when IRM is selected"
            )
        # Fishr's EMA decay and RDM's variance weight are fixed per search rather than
        # searched, but they change training, so a run states them explicitly.
        fixed_settings: tuple[tuple[MethodId, str, object], ...] = (
            (
                "fishr",
                "fishr_penalty_anneal_updates",
                self.fishr_penalty_anneal_updates,
            ),
            ("fishr", "fishr_ema", self.fishr_ema),
            ("rdm", "rdm_penalty_anneal_updates", self.rdm_penalty_anneal_updates),
            ("rdm", "rdm_variance_weight", self.rdm_variance_weight),
        )
        for method, name, value in fixed_settings:
            if (method in bases) != (value is not None):
                raise ValueError(
                    f"{name} must be set exactly when {METHOD_LABELS[method]} is "
                    "selected"
                )
        return self


class SearchRuntimeConfig(StrictBoundaryModel):
    device: Literal["cpu"]
    deterministic_algorithms: Literal[True]
    workers: Literal[1]


class _CommonProductionSearchConfig(StrictBoundaryModel):
    schema_version: Literal["grit.production-search/v1"]
    experiment_name: NonEmptyStr
    experiment_variant: Literal["primary_unnormalized", "l2_normalized_sensitivity"]
    normalization: Normalization
    artifacts: SearchArtifactPaths
    seeds: SearchSeedConfig
    search_space: SearchSpaceConfig
    output_root: NonEmptyStr
    runtime: SearchRuntimeConfig
    relative_singular_value_tolerance: Annotated[StrictFloat, Field(gt=0.0)]

    @model_validator(mode="after")
    def _validate_variant(self) -> _CommonProductionSearchConfig:
        expected: dict[str, Normalization] = {
            "primary_unnormalized": "none",
            "l2_normalized_sensitivity": "l2",
        }
        if self.normalization != expected[self.experiment_variant]:
            raise ValueError("experiment variant and normalization are inconsistent")
        return self


class CmnistProductionSearchConfig(_CommonProductionSearchConfig):
    """`selectors` is the primary robust and secondary source-only pair."""

    dataset: Literal["cmnist"]
    protocol_id: Literal["cmnist/v1"]
    pair_count: PositiveInt
    selectors: tuple[Literal["primary_robust"], Literal["secondary_source"]]
    batch_size: PositiveInt
    max_epochs: PositiveInt

    @model_validator(mode="after")
    def _validate_grid_fits(self) -> CmnistProductionSearchConfig:
        _require_runnable_grid(self.search_space, self.pair_count)
        return self


class WaterbirdsProductionSearchConfig(_CommonProductionSearchConfig):
    """`selectors` is the validation worst-group selector."""

    dataset: Literal["waterbirds_cf"]
    protocol_id: Literal["waterbirds_cf/v1"]
    pair_count: PositiveInt
    selectors: tuple[Literal["waterbirds_validation_worst_group"]]
    batch_size: PositiveInt
    max_epochs: PositiveInt

    @model_validator(mode="after")
    def _validate_grid_fits(self) -> WaterbirdsProductionSearchConfig:
        if self.search_space.lisa_group_definition is not None:
            raise ValueError(
                "Waterbirds LISA domains are backgrounds, which are its environments"
            )
        if self.search_space.groupdro_group_definition == "target_environment":
            raise ValueError(
                "Waterbirds label x environment groups are its label x background "
                "groups; leave groupdro_group_definition unset"
            )
        _require_runnable_grid(self.search_space, self.pair_count)
        return self


FEATURE_DIMENSION = 512
FINALIST_COUNT = 3


def _require_runnable_grid(space: SearchSpaceConfig, pair_count: int) -> None:
    """Catch at plan time what would otherwise fail hours later."""

    max_rank = min(pair_count, FEATURE_DIMENSION)
    too_large = [rank for rank in space.projection_ranks if rank > max_rank]
    if too_large:
        raise ValueError(
            f"projection_ranks {too_large} exceed min(pair_count, feature_dim) = "
            f"{max_rank}"
        )
    per_optimizer = len(space.learning_rates) * len(space.weight_decays)
    counts: dict[MethodId, int] = {
        method: per_optimizer * _method_setting_count(space, method)
        for method in space.methods
    }
    if min(counts.values()) < FINALIST_COUNT:
        formatted = " and ".join(
            f"{count} {METHOD_LABELS[method]}" for method, count in counts.items()
        )
        raise ValueError(
            f"selection keeps the top {FINALIST_COUNT} candidates per method, but the "
            f"grid yields {formatted} candidates"
        )


def _method_setting_count(space: SearchSpaceConfig, method: MethodId) -> int:
    return len(_method_settings_grid(space, method))


ProductionSearchConfig: TypeAlias = Annotated[
    CmnistProductionSearchConfig | WaterbirdsProductionSearchConfig,
    Field(discriminator="dataset"),
]
_SEARCH_CONFIG_ADAPTER: TypeAdapter[ProductionSearchConfig] = TypeAdapter(
    ProductionSearchConfig
)


class VerifiedInputArtifact(StrictBoundaryModel):
    kind: Literal["dataset_manifest", "feature_manifest", "pair_manifest"]
    path: NonEmptyStr
    digest: NonEmptyStr
    schema_version: NonEmptyStr


class SearchLineage(StrictBoundaryModel):
    """Input-artifact identity. The held-out validation name is omitted when it is
    the protocol default so pre-existing plans and task IDs are unchanged."""

    dataset_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    pair_manifest_digest: NonEmptyStr
    normalization: Normalization
    adjusted_weight_spec_digest: NonEmptyStr | None
    held_out_validation_split: HeldOutValidationName = Field(
        default=DEFAULT_HELD_OUT_VALIDATION_NAME,
        exclude_if=lambda value: value == DEFAULT_HELD_OUT_VALIDATION_NAME,
    )


class ResolvedProductionSearchConfig(StrictBoundaryModel):
    schema_version: Literal["grit.resolved-production-search/v1"]
    authored_config_digest: NonEmptyStr
    authored_config_path: NonEmptyStr
    output_root: NonEmptyStr
    config: ProductionSearchConfig
    lineage: SearchLineage
    input_artifacts: tuple[
        VerifiedInputArtifact,
        VerifiedInputArtifact,
        VerifiedInputArtifact,
    ]

    @model_validator(mode="after")
    def _validate_resolution(self) -> ResolvedProductionSearchConfig:
        if self.authored_config_digest != self.config.canonical_digest():
            raise ValueError("resolved search authored-config digest is inconsistent")
        if self.config.normalization != self.lineage.normalization:
            raise ValueError("resolved search normalization lineage is inconsistent")
        expected_kinds = (
            "dataset_manifest",
            "feature_manifest",
            "pair_manifest",
        )
        if tuple(item.kind for item in self.input_artifacts) != expected_kinds:
            raise ValueError("resolved search inputs must use canonical artifact order")
        if len({item.path for item in self.input_artifacts}) != 3:
            raise ValueError("resolved search input manifest paths must be unique")
        by_kind = {item.kind: item for item in self.input_artifacts}
        if (
            by_kind["dataset_manifest"].digest != self.lineage.dataset_manifest_digest
            or by_kind["feature_manifest"].digest
            != self.lineage.feature_cache_manifest_digest
            or by_kind["pair_manifest"].digest != self.lineage.pair_manifest_digest
        ):
            raise ValueError("resolved search input digests do not match lineage")
        if isinstance(self.config, CmnistProductionSearchConfig):
            expected_versions = (
                "grit.cmnist-dataset/v1",
                "grit.cmnist-features/v2",
                "grit.cmnist-oracle-pairs/v2",
            )
            if self.lineage.adjusted_weight_spec_digest is not None:
                raise ValueError("CMNIST search cannot carry Waterbirds weights")
        else:
            expected_versions = (
                "grit.waterbirds-cf-dataset/v2",
                "grit.waterbirds-features/v2",
                "grit.waterbirds-oracle-pairs/v2",
            )
            if self.lineage.adjusted_weight_spec_digest is None:
                raise ValueError("Waterbirds search requires adjusted-weight lineage")
        if tuple(item.schema_version for item in self.input_artifacts) != (
            expected_versions
        ):
            raise ValueError("resolved search input schema versions are inconsistent")
        resolved_paths = (
            self.authored_config_path,
            self.output_root,
            *(item.path for item in self.input_artifacts),
        )
        if any(not Path(value).is_absolute() for value in resolved_paths):
            raise ValueError("resolved search paths must be absolute")
        return self


@dataclass(frozen=True, slots=True)
class MethodSettings:
    """Method-specific candidate settings; exactly the fields a method needs are set."""

    consistency_weight: float | None = None
    requested_rank: int | None = None
    groupdro_step_size: float | None = None
    penalty_weight: float | None = None
    lisa_selection_prob: float | None = None

    def present_fields(self) -> frozenset[str]:
        return frozenset(
            name for name in _SETTING_FIELDS if getattr(self, name) is not None
        )


_SETTING_FIELDS: tuple[str, ...] = (
    "consistency_weight",
    "requested_rank",
    "groupdro_step_size",
    "penalty_weight",
    "lisa_selection_prob",
)
REQUIRED_SETTINGS: dict[MethodId, frozenset[str]] = {
    "erm": frozenset(),
    "grit": frozenset({"requested_rank"}),
    "groupdro": frozenset({"groupdro_step_size"}),
    "rex": frozenset({"penalty_weight"}),
    "irm": frozenset({"penalty_weight"}),
    "lisa": frozenset({"lisa_selection_prob"}),
    "sd": frozenset({"penalty_weight"}),
    "fishr": frozenset({"penalty_weight"}),
    "rdm": frozenset({"penalty_weight"}),
}
for _method in IMPLEMENTED_METHODS:
    if is_composed(_method):
        _intervention_settings = {
            "grit": {"requested_rank"},
            "consistency": {"consistency_weight"},
        }
        REQUIRED_SETTINGS[_method] = (
            REQUIRED_SETTINGS[base_objective(_method)]
            | _intervention_settings[pair_intervention(_method)]
        )


class SearchCandidate(StrictBoundaryModel):
    consistency_weight: Annotated[StrictFloat, Field(ge=0.0)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    candidate_id: NonEmptyStr
    scientific_config_digest: NonEmptyStr
    method_id: MethodId
    learning_rate: StrictFloat
    weight_decay: StrictFloat
    requested_rank: Annotated[StrictInt, Field(ge=0, le=24)] | None
    groupdro_step_size: Annotated[StrictFloat, Field(gt=0.0)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    penalty_weight: Annotated[StrictFloat, Field(gt=0.0)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    lisa_selection_prob: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    def settings(self) -> MethodSettings:
        return MethodSettings(**{name: getattr(self, name) for name in _SETTING_FIELDS})

    @model_validator(mode="after")
    def _validate_method_settings(self) -> SearchCandidate:
        expected = REQUIRED_SETTINGS[self.method_id]
        if self.settings().present_fields() != expected:
            raise ValueError(
                f"{METHOD_LABELS[self.method_id]} candidates require exactly "
                f"{sorted(expected)} as method-specific settings"
            )
        return self


class ExpectedRunCounts(StrictBoundaryModel):
    tuning: PositiveInt
    confirmation_minimum: PositiveInt
    confirmation_maximum: PositiveInt
    final: PositiveInt


class OutputSchemaVersion(StrictBoundaryModel):
    artifact_kind: NonEmptyStr
    schema_version: NonEmptyStr


class SearchPlan(StrictBoundaryModel):
    schema_version: Literal["grit.search-plan/v1"]
    protocol_id: Literal["cmnist/v1", "waterbirds_cf/v1"]
    dataset: Literal["cmnist", "waterbirds_cf"]
    experiment_name: NonEmptyStr
    experiment_variant: Literal["primary_unnormalized", "l2_normalized_sensitivity"]
    normalization: Normalization
    resolved_config: ResolvedProductionSearchConfig
    methods: Annotated[tuple[MethodId, ...], Field(min_length=1)]
    selectors: tuple[NonEmptyStr, ...]
    candidates: tuple[SearchCandidate, ...]
    seeds: SearchSeedConfig
    expected_run_counts: ExpectedRunCounts
    code: CodeProvenance
    environment: EnvironmentProvenance
    output_schemas: tuple[OutputSchemaVersion, ...]

    @model_validator(mode="after")
    def _validate_plan(self) -> SearchPlan:
        config = self.resolved_config.config
        if self.dataset != self.resolved_config.config.dataset:
            raise ValueError("search plan dataset does not match resolved config")
        if self.protocol_id != self.resolved_config.config.protocol_id:
            raise ValueError("search plan protocol does not match resolved config")
        if self.normalization != self.resolved_config.lineage.normalization:
            raise ValueError("search plan normalization lineage is inconsistent")
        if (
            self.experiment_name != config.experiment_name
            or self.experiment_variant != config.experiment_variant
            or self.normalization != config.normalization
        ):
            raise ValueError("search plan experiment identity is inconsistent")
        if self.methods != config.search_space.methods:
            raise ValueError("search plan methods do not match resolved config")
        if self.selectors != tuple(str(selector) for selector in config.selectors):
            raise ValueError("search plan selectors do not match resolved config")
        if self.seeds != config.seeds:
            raise ValueError("search plan seeds do not match resolved config")
        if self.candidates != _candidate_grid(self.resolved_config):
            raise ValueError("search plan candidate grid is inconsistent")
        if self.expected_run_counts != _expected_run_counts(config):
            raise ValueError("search plan run counts are inconsistent")
        if self.output_schemas != _expected_output_schemas(config):
            raise ValueError("search plan output schema inventory is inconsistent")
        return self


def load_production_search_config(path: Path) -> ProductionSearchConfig:
    """Parse an authored production YAML through the strict discriminated boundary."""

    authored = load_search_yaml(path)
    return _SEARCH_CONFIG_ADAPTER.validate_json(json.dumps(authored, allow_nan=False))


def load_search_yaml(path: Path) -> object:
    """Shared YAML number/path normalization for dataset search adapters."""
    authored = _normalize_numbers(
        _expand_environment(_yaml.safe_load(path.read_text(encoding="utf-8")))
    )
    return authored


_FLOAT_FIELDS = frozenset(
    {
        "consistency_weights",
        "learning_rates",
        "weight_decays",
        "groupdro_step_sizes",
        "rex_penalty_weights",
        "irm_penalty_weights",
        "lisa_selection_probs",
        "sd_penalty_weights",
        "fishr_penalty_weights",
        "fishr_ema",
        "rdm_penalty_weights",
        "rdm_variance_weight",
        "relative_singular_value_tolerance",
    }
)
_EXPONENT_FLOAT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def _normalize_numbers(value: object) -> object:
    """Let YAML authors write `0` or `1e-5` for float fields.

    YAML 1.1 parses `1e-5` as a string and `0` as an int; the strict schema wants
    floats. This coerces only the float-valued fields and exponent-form strings.
    """

    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in cast(dict[str, object], value).items():
            if key in _FLOAT_FIELDS:
                item = _coerce_float(item)
            result[key] = _normalize_numbers(item)
        return result
    if isinstance(value, list):
        return [_normalize_numbers(item) for item in cast(list[object], value)]
    if isinstance(value, str) and _EXPONENT_FLOAT.match(value):
        return float(value)
    return value


def _coerce_float(item: object) -> object:
    if isinstance(item, list):
        return [_coerce_float(element) for element in cast(list[object], item)]
    if isinstance(item, bool):
        return item
    if isinstance(item, int | str):
        try:
            return float(item)
        except ValueError:
            return item
    return item


def _expand_environment(value: object) -> object:
    """Expand `${VAR}` references in YAML strings so configs are machine-portable."""

    if isinstance(value, str):
        return expand_config_path(value)
    if isinstance(value, dict):
        return {
            str(key): _expand_environment(item)
            for key, item in cast(dict[object, object], value).items()
        }
    if isinstance(value, list):
        return [_expand_environment(item) for item in cast(list[object], value)]
    return value


def resolve_production_search_config(
    config: ProductionSearchConfig,
    *,
    config_path: Path,
) -> ResolvedProductionSearchConfig:
    """Parse and cross-check every manifest without opening feature arrays."""

    validated = _SEARCH_CONFIG_ADAPTER.validate_json(config.canonical_json())
    base = config_path.resolve().parent
    dataset_path = _resolve_path(base, validated.artifacts.dataset_manifest)
    feature_path = _resolve_path(base, validated.artifacts.feature_cache_manifest)
    pair_path = _resolve_path(base, validated.artifacts.oracle_pair_manifest)
    if isinstance(validated, CmnistProductionSearchConfig):
        lineage, artifacts = _verify_cmnist_artifacts(
            validated, dataset_path, feature_path, pair_path
        )
    else:
        lineage, artifacts = _verify_waterbirds_artifacts(
            validated, dataset_path, feature_path, pair_path
        )
    output_root = _resolve_path(base, validated.output_root)
    _require_safe_output_root(
        output_root,
        input_manifest_paths=(dataset_path, feature_path, pair_path),
    )
    return ResolvedProductionSearchConfig(
        schema_version="grit.resolved-production-search/v1",
        authored_config_digest=validated.canonical_digest(),
        authored_config_path=config_path.resolve().as_posix(),
        output_root=output_root.as_posix(),
        config=validated,
        lineage=lineage,
        input_artifacts=artifacts,
    )


def build_search_plan(resolved: ResolvedProductionSearchConfig) -> SearchPlan:
    """Expand one verified config into the complete deterministic candidate plan."""

    return _build_search_plan(resolved, _code_provenance())


def _build_search_plan(
    resolved: ResolvedProductionSearchConfig,
    code_provenance: CodeProvenance,
) -> SearchPlan:
    checked = ResolvedProductionSearchConfig.model_validate_json(
        resolved.canonical_json()
    )
    candidates = _candidate_grid(checked)
    config = checked.config
    return SearchPlan(
        schema_version="grit.search-plan/v1",
        protocol_id=config.protocol_id,
        dataset=config.dataset,
        experiment_name=config.experiment_name,
        experiment_variant=config.experiment_variant,
        normalization=config.normalization,
        resolved_config=checked,
        methods=config.search_space.methods,
        selectors=tuple(str(selector) for selector in config.selectors),
        candidates=candidates,
        seeds=config.seeds,
        expected_run_counts=_expected_run_counts(config),
        code=CodeProvenance.model_validate(code_provenance),
        environment=_environment_provenance(),
        output_schemas=_expected_output_schemas(config),
    )


def _expected_run_counts(config: ProductionSearchConfig) -> ExpectedRunCounts:
    candidate_count = _candidate_count_for_config(config)
    method_count = len(config.search_space.methods)
    selector_count = len(config.selectors)
    confirmation_seed_count = len(config.seeds.stages.confirmation)
    final_seed_count = len(config.seeds.stages.final)
    return ExpectedRunCounts(
        tuning=candidate_count * len(config.seeds.stages.tuning),
        confirmation_minimum=(method_count * FINALIST_COUNT * confirmation_seed_count),
        confirmation_maximum=(
            method_count * selector_count * FINALIST_COUNT * confirmation_seed_count
        ),
        final=method_count * selector_count * final_seed_count,
    )


def _expected_output_schemas(
    config: ProductionSearchConfig,
) -> tuple[OutputSchemaVersion, ...]:
    if isinstance(config, CmnistProductionSearchConfig):
        dataset_specific = (
            OutputSchemaVersion(
                artifact_kind="stage_run",
                schema_version="grit.cmnist-search-stage-run/v1",
            ),
            OutputSchemaVersion(
                artifact_kind="final_result", schema_version="grit.run-result/v1"
            ),
            OutputSchemaVersion(
                artifact_kind="tuning_finalists",
                schema_version="grit.cmnist-tuning-finalists/v1",
            ),
            OutputSchemaVersion(
                artifact_kind="finalist_union",
                schema_version="grit.cmnist-finalist-union/v1",
            ),
            OutputSchemaVersion(
                artifact_kind="frozen_candidate",
                schema_version="grit.cmnist-frozen-candidate/v1",
            ),
            OutputSchemaVersion(
                artifact_kind="production_summary",
                schema_version="grit.cmnist-production-summary/v1",
            ),
            *(
                (
                    OutputSchemaVersion(
                        artifact_kind="paired_summary",
                        schema_version="grit.cmnist-paired-summary/v1",
                    ),
                )
                if "erm" in config.search_space.methods
                and "grit" in config.search_space.methods
                else ()
            ),
        )
    else:
        dataset_specific = (
            OutputSchemaVersion(
                artifact_kind="stage_run",
                schema_version="grit.waterbirds-search-stage-run/v1",
            ),
            OutputSchemaVersion(
                artifact_kind="final_result",
                schema_version="grit.waterbirds-run-result/v3",
            ),
            OutputSchemaVersion(
                artifact_kind="tuning_finalists",
                schema_version="grit.waterbirds-tuning-finalists/v1",
            ),
            OutputSchemaVersion(
                artifact_kind="frozen_candidate",
                schema_version="grit.waterbirds-frozen-candidate/v1",
            ),
            OutputSchemaVersion(
                artifact_kind="production_summary",
                schema_version="grit.waterbirds-production-summary/v2",
            ),
            *(
                (
                    OutputSchemaVersion(
                        artifact_kind="paired_summary",
                        schema_version="grit.waterbirds-paired-summary/v1",
                    ),
                )
                if "erm" in config.search_space.methods
                and "grit" in config.search_space.methods
                else ()
            ),
        )
    return (
        OutputSchemaVersion(
            artifact_kind="authored_config",
            schema_version="grit.production-search/v1",
        ),
        OutputSchemaVersion(
            artifact_kind="resolved_config",
            schema_version="grit.resolved-production-search/v1",
        ),
        OutputSchemaVersion(
            artifact_kind="search_plan", schema_version="grit.search-plan/v1"
        ),
        *dataset_specific,
        *(
            (
                OutputSchemaVersion(
                    artifact_kind="projection_diagnostics",
                    schema_version="grit.linear-projection/v2",
                ),
            )
            if any(pair_intervention(m) == "grit" for m in config.search_space.methods)
            else ()
        ),
        OutputSchemaVersion(
            artifact_kind="selected_checkpoint",
            schema_version="grit.linear-checkpoint/v1",
        ),
        OutputSchemaVersion(
            artifact_kind="experiment_index",
            schema_version="grit.experiment-index/v1",
        ),
    )


def _candidate_count_for_config(config: ProductionSearchConfig) -> int:
    per_optimizer_grid = len(config.search_space.learning_rates) * len(
        config.search_space.weight_decays
    )
    candidate_count = 0
    for method in config.search_space.methods:
        candidate_count += per_optimizer_grid * _method_setting_count(
            config.search_space, method
        )
    return candidate_count


def write_search_plan(
    config: ProductionSearchConfig,
    *,
    config_path: Path,
) -> SearchPlan:
    """Validate and atomically persist authored, resolved, and plan artifacts."""

    code = _code_provenance()
    resolved = resolve_production_search_config(config, config_path=config_path)
    plan = _build_search_plan(resolved, code)
    output_root = Path(resolved.output_root)
    authored_target = output_root / "authored-config.yaml"
    resolved_target = output_root / "resolved-config.json"
    plan_target = output_root / "search-plan.json"
    _require_compatible_output_directory(
        output_root,
        planning_targets=(authored_target, resolved_target, plan_target),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    existing = tuple(
        target.exists() for target in (authored_target, resolved_target, plan_target)
    )
    if any(existing):
        if not all(existing):
            raise ValueError("search planning outputs are only partially present")
        stored_config = load_production_search_config(authored_target)
        stored_resolved = ResolvedProductionSearchConfig.model_validate_json(
            resolved_target.read_text(encoding="utf-8")
        )
        stored_plan = SearchPlan.model_validate_json(
            plan_target.read_text(encoding="utf-8")
        )
        if stored_config != config or stored_resolved != resolved:
            raise ValueError("existing search plan is incompatible with configuration")
        if stored_plan.candidates != plan.candidates:
            raise ValueError("existing search plan is incompatible with configuration")
        if stored_plan.code != plan.code:
            print(
                "note: continuing a plan created at commit "
                f"{stored_plan.code.git_revision[:12]} "
                f"(dirty={stored_plan.code.git_dirty}); current code is "
                f"{plan.code.git_revision[:12]} (dirty={plan.code.git_dirty})",
                file=sys.stderr,
            )
        return stored_plan
    _atomic_write_text(authored_target, config_path.read_text(encoding="utf-8"))
    _atomic_write_text(resolved_target, resolved.canonical_json() + "\n")
    _atomic_write_text(plan_target, plan.canonical_json() + "\n")
    return plan


def _candidate_grid(
    resolved: ResolvedProductionSearchConfig,
) -> tuple[SearchCandidate, ...]:
    config = resolved.config
    lineage = resolved.lineage
    candidates: list[SearchCandidate] = []
    for method in config.search_space.methods:
        settings = _method_settings_grid(config.search_space, method)
        for learning_rate in sorted(
            float(value) for value in config.search_space.learning_rates
        ):
            for weight_decay in sorted(
                float(value) for value in config.search_space.weight_decays
            ):
                for setting in settings:
                    scientific = _candidate_scientific_digest(
                        config,
                        lineage,
                        method,
                        learning_rate,
                        weight_decay,
                        setting,
                    )
                    candidate_id = _candidate_id(config.dataset, method, scientific)
                    candidates.append(
                        SearchCandidate(
                            candidate_id=candidate_id,
                            scientific_config_digest=scientific,
                            method_id=method,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            requested_rank=setting.requested_rank,
                            consistency_weight=setting.consistency_weight,
                            groupdro_step_size=setting.groupdro_step_size,
                            penalty_weight=setting.penalty_weight,
                            lisa_selection_prob=setting.lisa_selection_prob,
                        )
                    )
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    digests = [candidate.scientific_config_digest for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate-ID collision while expanding search grid")
    if len(digests) != len(set(digests)):
        raise ValueError("duplicate scientific configuration in search grid")
    return tuple(sorted(candidates, key=_candidate_order_key))


def _method_settings_grid(
    space: SearchSpaceConfig, method: MethodId
) -> tuple[MethodSettings, ...]:
    if is_composed(method):
        base_settings = _method_settings_grid(space, base_objective(method))
        values = (
            space.projection_ranks
            if pair_intervention(method) == "grit"
            else space.consistency_weights
        )
        field_name = (
            "requested_rank"
            if pair_intervention(method) == "grit"
            else "consistency_weight"
        )
        return tuple(
            replace(setting, **{field_name: value})
            for setting in base_settings
            for value in sorted(values)
        )
    if method == "erm":
        return (MethodSettings(),)
    if method == "grit":
        return tuple(
            MethodSettings(requested_rank=int(rank))
            for rank in sorted(space.projection_ranks)
        )
    if method == "groupdro":
        return tuple(
            MethodSettings(groupdro_step_size=float(step))
            for step in sorted(space.groupdro_step_sizes)
        )
    if method == "rex":
        return tuple(
            MethodSettings(penalty_weight=float(weight))
            for weight in sorted(space.rex_penalty_weights)
        )
    if method == "irm":
        return tuple(
            MethodSettings(penalty_weight=float(weight))
            for weight in sorted(space.irm_penalty_weights)
        )
    if method == "lisa":
        return tuple(
            MethodSettings(lisa_selection_prob=float(prob))
            for prob in sorted(space.lisa_selection_probs)
        )
    if method in ("sd", "fishr", "rdm"):
        weights = {
            "sd": space.sd_penalty_weights,
            "fishr": space.fishr_penalty_weights,
            "rdm": space.rdm_penalty_weights,
        }[method]
        return tuple(
            MethodSettings(penalty_weight=float(weight)) for weight in sorted(weights)
        )
    raise AssertionError(f"candidate grid is missing method {method}")


@dataclass(frozen=True, slots=True)
class CmnistCandidateComponents:
    pairs: DisabledPairsConfig | OraclePairsConfig
    projection: DisabledProjectionConfig | LinearProjectionConfig
    algorithm: AlgorithmConfig
    pair_manifest_digest: str | None


def cmnist_candidate_components(
    config: CmnistProductionSearchConfig,
    lineage: SearchLineage,
    method: MethodId,
    setting: MethodSettings,
) -> CmnistCandidateComponents:
    """The one place that turns a planned CMNIST candidate into algorithm configs."""

    if setting.present_fields() != REQUIRED_SETTINGS[method]:
        raise AssertionError(f"planned CMNIST {method} candidate has wrong settings")
    disabled_pairs = DisabledPairsConfig(kind="disabled")
    disabled_projection = DisabledProjectionConfig(kind="disabled")
    oracle_pairs = OraclePairsConfig(
        kind="oracle",
        construction_id="cmnist-clean-oracle-pairs-v1",
        source_partition_ids=("train_e01_sources", "train_e02_sources"),
        pair_count=config.pair_count,
        pair_seed=config.seeds.pairs,
        orientation="red_minus_green",
    )
    algorithm = candidate_algorithm_config(
        config,
        method,
        setting,
        environment_names=("train_e01", "train_e02"),
        group_definition="target_color",
    )
    if pair_intervention(method) == "grit":
        rank = setting.requested_rank
        if rank is None:
            raise AssertionError("planned CMNIST GRIT candidate lacks a rank")
        return CmnistCandidateComponents(
            oracle_pairs,
            LinearProjectionConfig(
                kind="linear_pair_difference",
                requested_rank=rank,
                center_differences=False,
                relative_singular_value_tolerance=(
                    config.relative_singular_value_tolerance
                ),
            ),
            algorithm,
            lineage.pair_manifest_digest,
        )
    if consumes_pairs(method):
        return CmnistCandidateComponents(
            oracle_pairs, disabled_projection, algorithm, lineage.pair_manifest_digest
        )
    return CmnistCandidateComponents(
        disabled_pairs, disabled_projection, algorithm, None
    )


def candidate_algorithm_config(
    config: _CommonProductionSearchConfig,
    method: MethodId,
    setting: MethodSettings,
    *,
    environment_names: EnvironmentNames,
    group_definition: GroupDefinition,
) -> AlgorithmConfig:
    """The one place that turns a planned setting into a method's algorithm config.

    The dataset supplies only what the method definitions bind to: environment names
    for the invariance methods and the default group definition for GroupDRO and
    LISA. A search may replace the default with source-environment groups.
    """

    if setting.present_fields() != REQUIRED_SETTINGS[method]:
        raise AssertionError(f"planned {method} candidate has wrong settings")
    if is_composed(method):
        base = candidate_algorithm_config(
            config,
            base_objective(method),
            replace(setting, requested_rank=None, consistency_weight=None),
            environment_names=environment_names,
            group_definition=group_definition,
        )
        if not isinstance(
            base,
            ErmAlgorithmConfig
            | RexAlgorithmConfig
            | IrmAlgorithmConfig
            | FishrAlgorithmConfig
            | GroupDroAlgorithmConfig
            | SdAlgorithmConfig
            | RdmAlgorithmConfig
            | LisaAlgorithmConfig,
        ):
            raise AssertionError("unsupported composed base objective")
        intervention = pair_intervention(method)
        if intervention == "vanilla":
            raise AssertionError("composed method needs an intervention")
        return ComposedAlgorithmConfig(
            kind="composed",
            base_objective=base,
            pair_intervention=intervention,
            consistency_weight=setting.consistency_weight,
            objective_version="prediction-pairs/v1",
        )
    if method == "erm":
        return ErmAlgorithmConfig(kind="erm")
    if method == "grit":
        return GritAlgorithmConfig(kind="grit")
    if method == "groupdro":
        step = setting.groupdro_step_size
        if step is None:
            raise AssertionError("planned GroupDRO candidate lacks a step size")
        return GroupDroAlgorithmConfig(
            kind="groupdro",
            group_definition=(
                config.search_space.groupdro_group_definition or group_definition
            ),
            adversarial_step_size=step,
            sampling="inverse_group_frequency_with_replacement",
            generalization_adjustment=0.0,
            normalize_loss=False,
        )
    if method in ("rex", "irm"):
        weight = setting.penalty_weight
        anneal = (
            config.search_space.rex_penalty_anneal_updates
            if method == "rex"
            else config.search_space.irm_penalty_anneal_updates
        )
        if weight is None or anneal is None:
            raise AssertionError(f"planned {method} candidate lacks settings")
        if method == "rex":
            return RexAlgorithmConfig(
                kind="rex",
                environment_names=environment_names,
                penalty_weight=weight,
                penalty_anneal_updates=anneal,
                risk_variance="population",
                sampling="environment_balanced_without_replacement",
                loss_rescaling="divide_by_penalty_weight_above_one",
            )
        return IrmAlgorithmConfig(
            kind="irm",
            environment_names=environment_names,
            penalty_weight=weight,
            penalty_anneal_updates=anneal,
            penalty="irmv1_dummy_classifier_scale",
            sampling="environment_balanced_without_replacement",
            loss_rescaling="divide_by_penalty_weight_above_one",
        )
    if method == "lisa":
        prob = setting.lisa_selection_prob
        if prob is None:
            raise AssertionError("planned LISA candidate lacks a probability")
        return LisaAlgorithmConfig(
            kind="lisa",
            group_definition=config.search_space.lisa_group_definition
            or group_definition,
            selection_prob=prob,
            mixing="beta_2_2",
            sampling="uniform_single_group_batches",
        )
    if method in ("sd", "fishr", "rdm"):
        weight = setting.penalty_weight
        if weight is None:
            raise AssertionError(f"planned {method} candidate lacks a penalty weight")
        if method == "sd":
            return SdAlgorithmConfig(
                kind="sd",
                penalty_weight=weight,
                penalty="mean_squared_logits",
                sampling="uniform_without_replacement",
            )
        space = config.search_space
        if method == "fishr":
            anneal = space.fishr_penalty_anneal_updates
            ema = space.fishr_ema
            if anneal is None or ema is None:
                raise AssertionError("planned Fishr candidate lacks fixed settings")
            return FishrAlgorithmConfig(
                kind="fishr",
                environment_names=environment_names,
                penalty_weight=weight,
                penalty_anneal_updates=anneal,
                ema=ema,
                penalty="classifier_gradient_variance_distance",
                sampling="environment_balanced_without_replacement",
            )
        anneal = space.rdm_penalty_anneal_updates
        variance_weight = space.rdm_variance_weight
        if anneal is None or variance_weight is None:
            raise AssertionError("planned RDM candidate lacks fixed settings")
        return RdmAlgorithmConfig(
            kind="rdm",
            environment_names=environment_names,
            penalty_weight=weight,
            penalty_anneal_updates=anneal,
            variance_weight=variance_weight,
            penalty="worst_environment_versus_pooled_risk_mmd",
            sampling="environment_balanced_without_replacement",
        )
    raise AssertionError(f"algorithm config materializer is missing method {method}")


def waterbirds_candidate_algorithm(
    config: WaterbirdsProductionSearchConfig,
    method: MethodId,
    setting: MethodSettings,
) -> AlgorithmConfig:
    """Waterbirds bindings: background environments, label/background groups."""

    return candidate_algorithm_config(
        config,
        method,
        setting,
        environment_names=("background_land", "background_water"),
        group_definition="target_background",
    )


def _candidate_scientific_digest(
    config: ProductionSearchConfig,
    lineage: SearchLineage,
    method: MethodId,
    learning_rate: float,
    weight_decay: float,
    setting: MethodSettings,
) -> str:
    training = LinearProbeTrainingConfig(
        optimizer=training_optimizer(method),
        batch_size=config.batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=config.max_epochs,
    )
    requested_rank = setting.requested_rank
    if isinstance(config, CmnistProductionSearchConfig):
        components = cmnist_candidate_components(config, lineage, method, setting)
        pairs = components.pairs
        projection = components.projection
        algorithm = components.algorithm
        pair_digest = components.pair_manifest_digest
        candidate = OrdinaryExperimentConfig(
            schema_version="grit.experiment/v1",
            run_kind="ordinary",
            experiment_name=config.experiment_name,
            protocol_id="cmnist/v1",
            reportable=True,
            dataset=CmnistDatasetConfig(
                dataset_id="cmnist",
                construction_method_id="cmnist-stratified-hash-v1",
                construction_seed=config.seeds.construction,
                label_flip_prob=0.25,
                source_counts=CmnistSourceCounts(
                    train_e01=25_000,
                    train_e02=25_000,
                    validation=10_000,
                    test=10_000,
                ),
                training_split_names=("train_e01", "train_e02"),
                validation_split_names=(
                    "val_e01",
                    "val_e02",
                    lineage.held_out_validation_split,
                ),
                final_test_split_name="test_ood",
            ),
            representation=FrozenFeatureConfig(
                kind="frozen_features",
                encoder_id="openai-clip-vit-b32",
                encoder_revision=OPENAI_CLIP_REVISION,
                weights_identity=OPENAI_CLIP_WEIGHTS_IDENTITY,
                preprocessing_identity=OPENAI_CLIP_PREPROCESSING_ID,
                feature_dimension=512,
                normalization=config.normalization,
            ),
            pairs=pairs,
            projection=projection,
            algorithm=algorithm,
            training=training,
            runtime=CpuRuntimeConfig(device="cpu", deterministic_algorithms=True),
            seed_sets=config.seeds.stages,
            artifact_lineage=CmnistArtifactLineageConfig(
                dataset_manifest_digest=lineage.dataset_manifest_digest,
                feature_cache_manifest_digest=lineage.feature_cache_manifest_digest,
                pair_manifest_digest=pair_digest,
            ),
            selection=OrdinarySelectionConfig(selector=CmnistSelector.PRIMARY_ROBUST),
        )
        return candidate.scientific_config_digest()
    projection_digest = None
    pair_digest = None
    tolerance: float | None = None
    if pair_intervention(method) == "grit":
        if requested_rank is None:
            raise AssertionError("planned Waterbirds GRIT candidate lacks a rank")
        projection_digest = "pending:derived-after-plan"
        tolerance = config.relative_singular_value_tolerance
    if consumes_pairs(method):
        pair_digest = lineage.pair_manifest_digest
    candidate = WaterbirdsCandidateConfig(
        schema_version="grit.waterbirds-candidate/v3",
        protocol_id="waterbirds_cf/v1",
        non_reportable=False,
        method_id=method,
        selector="waterbirds_validation_worst_group",
        dataset_profile="production",
        dataset_manifest_digest=lineage.dataset_manifest_digest,
        feature_cache_manifest_digest=lineage.feature_cache_manifest_digest,
        normalization=config.normalization,
        adjusted_weight_spec_digest=_required_weight_digest(lineage),
        pair_manifest_digest=pair_digest,
        projection_diagnostics_digest=projection_digest,
        projection_rank=requested_rank,
        relative_singular_value_tolerance=tolerance,
        algorithm=waterbirds_candidate_algorithm(config, method, setting),
        training=training,
        seed_sets=config.seeds.stages,
    )
    return candidate.scientific_config_digest()


def _required_weight_digest(lineage: SearchLineage) -> str:
    if lineage.adjusted_weight_spec_digest is None:
        raise ValueError("Waterbirds search requires adjusted-weight lineage")
    return lineage.adjusted_weight_spec_digest


def _candidate_id(dataset: str, method: MethodId, digest: str) -> str:
    return f"candidate:{dataset}:{method}:{digest.removeprefix('sha256:')[:24]}"


def _candidate_order_key(
    candidate: SearchCandidate,
) -> tuple[float, ...]:
    def number(value: float | int | None) -> float:
        return -1.0 if value is None else float(value)

    return (
        IMPLEMENTED_METHODS.index(candidate.method_id),
        float(candidate.learning_rate),
        float(candidate.weight_decay),
        -1 if candidate.requested_rank is None else candidate.requested_rank,
        number(candidate.groupdro_step_size),
        number(candidate.penalty_weight),
        number(candidate.consistency_weight),
        number(candidate.lisa_selection_prob),
    )


def _verify_cmnist_artifacts(
    config: CmnistProductionSearchConfig,
    dataset_path: Path,
    feature_path: Path,
    pair_path: Path,
) -> tuple[
    SearchLineage,
    tuple[VerifiedInputArtifact, VerifiedInputArtifact, VerifiedInputArtifact],
]:
    dataset = CmnistDatasetManifest.model_validate_json(_read_manifest(dataset_path))
    feature = CmnistFeatureCacheManifest.model_validate_json(
        _read_manifest(feature_path)
    )
    pairs = CmnistOraclePairManifest.model_validate_json(_read_manifest(pair_path))
    targets = dataset.partition_manifest.targets
    if (targets.train_e01, targets.train_e02, targets.validation, targets.test) != (
        25_000,
        25_000,
        10_000,
        10_000,
    ) or float(dataset.label_flip_prob) != 0.25:
        raise ValueError("production CMNIST search requires the canonical dataset")
    partitions = {item.name: item for item in dataset.partition_manifest.partitions}
    training_universe = (
        set(partitions["train_e01_sources"].source_indices)
        | set(partitions["train_e02_sources"].source_indices)
        | set(partitions["validation_sources"].source_indices)
    )
    if training_universe != set(range(60_000)) or set(
        partitions["test_sources"].source_indices
    ) != set(range(10_000)):
        raise ValueError("production CMNIST requires the official source universes")
    if dataset.construction_seed != config.seeds.construction:
        raise ValueError("CMNIST construction seed does not match the dataset manifest")
    dataset_digest = dataset.canonical_digest()
    pair_digest = pairs.canonical_digest()
    if (
        pairs.dataset_manifest_digest != dataset_digest
        or pairs.pair_seed != config.seeds.pairs
    ):
        raise ValueError("CMNIST oracle-pair lineage is inconsistent")
    if pairs.realized_count < config.pair_count:
        raise ValueError(
            f"config asks for {config.pair_count} oracle pairs but the prepared bank "
            f"holds {pairs.realized_count}; rerun scripts/prepare_cmnist.py "
            f"--pair-count {config.pair_count}"
        )
    if (
        feature.source_manifest_digest != dataset_digest
        or feature.pair_manifest_digest != pair_digest
        or feature.normalization != config.normalization
    ):
        raise ValueError(
            "CMNIST feature-cache lineage or normalization is inconsistent"
        )
    _require_official_clip(feature.encoder)
    _validate_cmnist_feature_manifest(dataset, pairs, feature, feature_path.parent)
    return (
        SearchLineage(
            dataset_manifest_digest=dataset_digest,
            feature_cache_manifest_digest=feature.canonical_digest(),
            pair_manifest_digest=pair_digest,
            normalization=config.normalization,
            adjusted_weight_spec_digest=None,
            held_out_validation_split=cast(
                HeldOutValidationName, dataset.held_out_validation_name
            ),
        ),
        _artifact_records(
            dataset_path, dataset, feature_path, feature, pair_path, pairs
        ),
    )


def _verify_waterbirds_artifacts(
    config: WaterbirdsProductionSearchConfig,
    dataset_path: Path,
    feature_path: Path,
    pair_path: Path,
) -> tuple[
    SearchLineage,
    tuple[VerifiedInputArtifact, VerifiedInputArtifact, VerifiedInputArtifact],
]:
    dataset = WaterbirdsDatasetManifest.model_validate_json(
        _read_manifest(dataset_path)
    )
    feature = WaterbirdsFeatureCacheManifest.model_validate_json(
        _read_manifest(feature_path)
    )
    pairs = WaterbirdsOraclePairManifest.model_validate_json(_read_manifest(pair_path))
    if dataset.profile_kind != "production" or dataset.non_reportable:
        raise ValueError("production Waterbirds search rejects fixture datasets")
    if dataset.construction_seed != config.seeds.construction:
        raise ValueError(
            "Waterbirds construction seed does not match the dataset manifest"
        )
    dataset_digest = dataset.canonical_digest()
    pair_digest = pairs.canonical_digest()
    if (
        pairs.dataset_manifest_digest != dataset_digest
        or pairs.profile_kind != "production"
        or pairs.non_reportable
        or pairs.pair_count != config.pair_count
    ):
        raise ValueError("Waterbirds oracle-pair lineage or count is inconsistent")
    if (
        feature.dataset_manifest_digest != dataset_digest
        or feature.non_reportable
        or feature.normalization != config.normalization
    ):
        raise ValueError(
            "Waterbirds feature-cache lineage or normalization is inconsistent"
        )
    _require_official_clip(feature.encoder)
    _validate_waterbirds_feature_manifest(dataset, feature, feature_path.parent)
    if pairs.records != tuple(
        sorted(dataset.relationships, key=lambda item: item.pair_id)
    ):
        raise ValueError(
            "Waterbirds pair records do not match the dataset relationships"
        )
    weights = mint_waterbirds_adjusted_weight_spec(dataset)
    return (
        SearchLineage(
            dataset_manifest_digest=dataset_digest,
            feature_cache_manifest_digest=feature.canonical_digest(),
            pair_manifest_digest=pair_digest,
            normalization=config.normalization,
            adjusted_weight_spec_digest=weights.canonical_digest(),
        ),
        _artifact_records(
            dataset_path, dataset, feature_path, feature, pair_path, pairs
        ),
    )


def _validate_cmnist_feature_manifest(
    dataset: CmnistDatasetManifest,
    pairs: CmnistOraclePairManifest,
    feature: CmnistFeatureCacheManifest,
    feature_root: Path,
) -> None:
    expected_counts = (
        25_000,
        25_000,
        10_000,
        10_000,
        10_000,
        10_000,
        pairs.realized_count,
        pairs.realized_count,
    )
    if tuple(table.row_count for table in feature.tables) != expected_counts:
        raise ValueError(
            "production CMNIST feature-cache table counts are inconsistent"
        )
    partitions = {
        partition.name: partition for partition in dataset.partition_manifest.partitions
    }
    expected_environment_sources: dict[str, tuple[str, ...]] = {}
    for spec, environment in zip(
        dataset.environment_specs(), dataset.environments, strict=True
    ):
        partition = partitions[spec.source_partition_id]
        source_ids = tuple(
            f"mnist:{partition.official_split}:{index}"
            for index in partition.source_indices
        )
        if (
            environment.name != spec.name
            or environment.role != spec.role
            or environment.source_partition_id != spec.source_partition_id
            or float(environment.color_flip_prob) != float(spec.color_flip_prob)
            or environment.count != len(source_ids)
            or environment.source_membership_digest
            != canonical_digest_value(source_ids)
        ):
            raise ValueError("CMNIST dataset environment lineage is inconsistent")
        expected_environment_sources[spec.name] = source_ids
    for table in feature.tables[:6]:
        if table.source_ids != expected_environment_sources[table.table_name]:
            raise ValueError("CMNIST feature-table source lineage is inconsistent")
    pair_sources = tuple(record.source_id for record in pairs.records)
    if feature.tables[6].source_ids != pair_sources:
        raise ValueError("CMNIST feature pair rows do not match the pair manifest")
    training_indices = {
        index
        for partition in dataset.partition_manifest.partitions[:2]
        for index in partition.source_indices
    }
    if any(
        record.official_source_index not in training_indices
        or record.source_id != f"mnist:train:{record.official_source_index}"
        for record in pairs.records
    ):
        raise ValueError("CMNIST oracle pairs are outside the training partitions")
    _verify_referenced_files(
        feature_root,
        tuple(
            (file.relative_path, file.sha256)
            for table in feature.tables
            for file in table.files
        ),
    )


def _validate_waterbirds_feature_manifest(
    dataset: WaterbirdsDatasetManifest,
    feature: WaterbirdsFeatureCacheManifest,
    feature_root: Path,
) -> None:
    expected = tuple(
        (
            record.record_id,
            record.split_role,
            record.bird_label,
            record.background,
            record.group_id,
            record.image_sha256,
        )
        for record in dataset.records
    )
    observed = tuple(
        (
            record.record_id,
            record.split_role,
            record.bird_label,
            record.background,
            record.group_id,
            record.image_sha256,
        )
        for record in feature.records
    )
    if observed != expected:
        raise ValueError("Waterbirds feature records do not match the dataset manifest")
    _verify_referenced_files(
        feature_root,
        ((feature.feature_file.relative_path, feature.feature_file.sha256),),
    )


def _require_official_clip(identity: EncoderIdentity) -> None:
    observed = (
        identity.implementation,
        identity.implementation_revision,
        identity.model_name,
        identity.weights_identity,
        identity.preprocessing_identity,
        identity.raw_output_dimension,
    )
    expected = (
        "openai/CLIP",
        OPENAI_CLIP_REVISION,
        "ViT-B/32",
        OPENAI_CLIP_WEIGHTS_IDENTITY,
        OPENAI_CLIP_PREPROCESSING_ID,
        512,
    )
    if observed != expected:
        raise ValueError("production search requires pinned official OpenAI CLIP")


def _artifact_records(
    dataset_path: Path,
    dataset: CmnistDatasetManifest | WaterbirdsDatasetManifest,
    feature_path: Path,
    feature: CmnistFeatureCacheManifest | WaterbirdsFeatureCacheManifest,
    pair_path: Path,
    pairs: CmnistOraclePairManifest | WaterbirdsOraclePairManifest,
) -> tuple[VerifiedInputArtifact, VerifiedInputArtifact, VerifiedInputArtifact]:
    return (
        VerifiedInputArtifact(
            kind="dataset_manifest",
            path=dataset_path.as_posix(),
            digest=dataset.canonical_digest(),
            schema_version=dataset.schema_version,
        ),
        VerifiedInputArtifact(
            kind="feature_manifest",
            path=feature_path.as_posix(),
            digest=feature.canonical_digest(),
            schema_version=feature.schema_version,
        ),
        VerifiedInputArtifact(
            kind="pair_manifest",
            path=pair_path.as_posix(),
            digest=pairs.canonical_digest(),
            schema_version=pairs.schema_version,
        ),
    )


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_safe_output_root(
    output_root: Path,
    *,
    input_manifest_paths: tuple[Path, Path, Path],
) -> None:
    """Require a dedicated output tree disjoint from source and prepared inputs."""

    resolved_output = output_root.resolve()
    if resolved_output == Path(resolved_output.anchor):
        raise ValueError("production output_root cannot be a filesystem root")
    try:
        repository_root: Path | None = _git_repository_root()
    except ValueError:
        repository_root = None
    if repository_root is not None and (
        resolved_output == repository_root
        or repository_root.is_relative_to(resolved_output)
    ):
        raise ValueError("production output_root cannot be the repository root")
    for manifest_path in input_manifest_paths:
        prepared_root = manifest_path.resolve().parent
        if resolved_output == prepared_root or resolved_output.is_relative_to(
            prepared_root
        ):
            raise ValueError(
                "production output_root cannot be a prepared-artifact directory "
                "or one of its descendants"
            )
        if manifest_path.resolve().is_relative_to(resolved_output):
            raise ValueError(
                "production output_root cannot contain prepared input artifacts"
            )
    if repository_root is None:
        return
    try:
        relative = resolved_output.relative_to(repository_root)
    except ValueError:
        return
    # Source archives have no Git metadata. A dedicated scratch directory inside
    # the archive is safe, even though `git check-ignore` cannot inspect it.
    if not (repository_root / ".git").exists():
        return
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", relative.as_posix()],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ignored.returncode != 0:
        raise ValueError(
            "production output_root inside the repository must be Git-ignored"
        )


def _require_compatible_output_directory(
    output_root: Path,
    *,
    planning_targets: tuple[Path, Path, Path],
) -> None:
    if not output_root.exists():
        return
    if not output_root.is_dir():
        raise ValueError("production output_root must be a directory")
    entries = tuple(output_root.iterdir())
    existing_planning = tuple(path.is_file() for path in planning_targets)
    if not entries:
        return
    if not all(existing_planning):
        if any(existing_planning):
            raise ValueError("search planning outputs are only partially present")
        raise ValueError(
            "production output_root must be empty or contain a complete compatible plan"
        )


def _read_manifest(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required search manifest is missing: {path}")
    return path.read_text(encoding="utf-8")


def _verify_referenced_files(
    root: Path,
    files: tuple[tuple[str, str], ...],
) -> None:
    paths = tuple(relative for relative, _ in files)
    if len(paths) != len(set(paths)):
        raise ValueError("feature manifest contains duplicate artifact paths")
    for relative, expected_digest in files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"referenced feature artifact is missing: {relative}"
            )
        if _file_sha256(path) != expected_digest:
            raise ValueError(
                f"referenced feature artifact digest is inconsistent: {relative}"
            )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _atomic_write_text(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def current_code_provenance() -> CodeProvenance:
    """Commit and dirty flag for the code executing right now."""

    return _code_provenance()


def current_environment_provenance() -> EnvironmentProvenance:
    return _environment_provenance()


def _code_provenance() -> CodeProvenance:
    """Record the commit and dirty state; never refuse to run because of them."""

    if not (REPO_ROOT / ".git").exists():
        return CodeProvenance(git_revision="no-git-checkout", git_dirty=False)
    try:
        repository_root = _git_repository_root()
        revision = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_output = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, ValueError):
        return CodeProvenance(git_revision="unavailable", git_dirty=True)
    dirty = bool(dirty_output)
    if dirty:
        print(
            "warning: worktree has uncommitted changes; the plan records "
            f"{revision[:12]} as dirty",
            file=sys.stderr,
        )
    return CodeProvenance(git_revision=revision or "unavailable", git_dirty=dirty)


def _git_repository_root() -> Path:
    if not (REPO_ROOT / ".git").exists():
        return REPO_ROOT
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("production search requires a Git worktree") from error
    if not root:
        raise ValueError("production search could not resolve the Git repository root")
    return Path(root).resolve()


def _environment_provenance() -> EnvironmentProvenance:
    lock_path = REPO_ROOT / "uv.lock"
    lock_digest = (
        canonical_digest_value(lock_path.read_text(encoding="utf-8"))
        if lock_path.is_file()
        else "unavailable"
    )
    return EnvironmentProvenance(
        python_version=platform.python_version(),
        lock_digest=lock_digest,
        device="cpu",
    )
