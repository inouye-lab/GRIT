"""Method identities implemented by the production experiment lifecycle."""

from typing import Final, Literal, TypeAlias, cast

MethodId: TypeAlias = Literal[
    "erm",
    "grit",
    "groupdro",
    "rex",
    "irm",
    "lisa",
    "sd",
    "fishr",
    "rdm",
    "erm_consistency",
    "rex_grit",
    "rex_consistency",
    "irm_grit",
    "irm_consistency",
    "fishr_grit",
    "fishr_consistency",
    "groupdro_grit",
    "groupdro_consistency",
    "sd_grit",
    "sd_consistency",
    "rdm_grit",
    "rdm_consistency",
    "lisa_grit",
    "lisa_consistency",
]

# This order is part of deterministic candidate planning and presentation.
IMPLEMENTED_METHODS: Final[tuple[MethodId, ...]] = (
    "erm",
    "grit",
    "groupdro",
    "rex",
    "irm",
    "lisa",
    "sd",
    "fishr",
    "rdm",
    "erm_consistency",
    "rex_grit",
    "rex_consistency",
    "irm_grit",
    "irm_consistency",
    "fishr_grit",
    "fishr_consistency",
    "groupdro_grit",
    "groupdro_consistency",
    "sd_grit",
    "sd_consistency",
    "rdm_grit",
    "rdm_consistency",
    "lisa_grit",
    "lisa_consistency",
)

METHOD_LABELS: Final[dict[MethodId, str]] = {
    "erm": "ERM",
    "grit": "GRIT",
    "groupdro": "GroupDRO",
    "rex": "REx",
    "irm": "IRM",
    "lisa": "LISA",
    "sd": "SD",
    "fishr": "Fishr",
    "rdm": "RDM",
    "erm_consistency": "ERM + prediction consistency",
    "rex_grit": "V-REx + GRIT",
    "rex_consistency": "V-REx + prediction consistency",
    "irm_grit": "IRMv1 + GRIT",
    "irm_consistency": "IRMv1 + prediction consistency",
    "fishr_grit": "Fishr + GRIT",
    "fishr_consistency": "Fishr + prediction consistency",
    "groupdro_grit": "GroupDRO + GRIT",
    "groupdro_consistency": "GroupDRO + prediction consistency",
    "sd_grit": "SD + GRIT",
    "sd_consistency": "SD + prediction consistency",
    "rdm_grit": "RDM + GRIT",
    "rdm_consistency": "RDM + prediction consistency",
    "lisa_grit": "LISA + GRIT",
    "lisa_consistency": "LISA + prediction consistency",
}

# Every method the Waterbirds runner binds to the trainer.
WATERBIRDS_METHODS: Final[tuple[MethodId, ...]] = IMPLEMENTED_METHODS


def is_composed(method: MethodId) -> bool:
    """A base objective plus a pair intervention (legacy GRIT is ERM plus one)."""
    return "_" in method


def base_objective(method: MethodId) -> MethodId:
    """Legacy GRIT is ERM with projection; combination IDs are presentation keys."""
    if method == "grit":
        return "erm"
    if not is_composed(method):
        return method
    return cast(MethodId, method.split("_")[0])


PairIntervention: TypeAlias = Literal["vanilla", "grit", "consistency"]


def pair_intervention(method: MethodId) -> PairIntervention:
    if not is_composed(method) and method != "grit":
        return "vanilla"
    if method == "grit" or method.endswith("_grit"):
        return "grit"
    if method.endswith("_consistency"):
        return "consistency"
    return "vanilla"


def consumes_pairs(method: MethodId) -> bool:
    return pair_intervention(method) in ("grit", "consistency")
