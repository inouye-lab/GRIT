"""Strict, canonical boundary-schema primitives for the rewrite."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

# The third CMNIST validation rendering is a held-out color-flip rate chosen at
# preparation time. Its environment name encodes the rate in tenths.
HeldOutValidationName: TypeAlias = Literal[
    "val_e03", "val_e04", "val_e05", "val_e06", "val_e07"
]
HELD_OUT_VALIDATION_NAMES: tuple[HeldOutValidationName, ...] = (
    "val_e03",
    "val_e04",
    "val_e05",
    "val_e06",
    "val_e07",
)
DEFAULT_HELD_OUT_VALIDATION_NAME: HeldOutValidationName = "val_e05"
CmnistValidationName: TypeAlias = Literal[
    "val_e01", "val_e02", "val_e03", "val_e04", "val_e05", "val_e06", "val_e07"
]
IN_DOMAIN_VALIDATION_NAMES: tuple[Literal["val_e01"], Literal["val_e02"]] = (
    "val_e01",
    "val_e02",
)


def held_out_validation_name(color_flip_prob: float) -> HeldOutValidationName:
    """Map an approved held-out flip rate (a tenth from 0.3 to 0.7) to its name."""

    tenths = round(color_flip_prob * 10)
    if abs(color_flip_prob * 10 - tenths) > 1e-9:
        raise ValueError("held-out validation flip rate must be a multiple of 0.1")
    name = f"val_e{tenths:02d}"
    if name not in HELD_OUT_VALIDATION_NAMES:
        raise ValueError(
            "held-out validation flip rate must lie between 0.3 and 0.7 inclusive"
        )
    return name  # pyright: ignore[reportReturnType]


def held_out_validation_flip_prob(name: str) -> float:
    if name not in HELD_OUT_VALIDATION_NAMES:
        raise ValueError(f"unknown held-out validation split {name!r}")
    return int(name.removeprefix("val_e")) / 10


class CmnistSelector(str, Enum):
    """The two prespecified validation selectors."""

    PRIMARY_ROBUST = "primary_robust"
    SECONDARY_SOURCE = "secondary_source"


ORDINARY_CMNIST_SELECTORS = (
    CmnistSelector.PRIMARY_ROBUST,
    CmnistSelector.SECONDARY_SOURCE,
)


class SeedStage(str, Enum):
    """The model-selection stage associated with a training seed."""

    TUNING = "tuning"
    CONFIRMATION = "confirmation"
    FINAL = "final"


class StrictBoundaryModel(BaseModel):
    """Base for immutable, strict, canonical JSON boundary models."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    def canonical_json(self) -> str:
        """Return the authoritative compact JSON representation."""

        return canonical_json_value(self.model_dump(mode="json"))

    def canonical_digest(self) -> str:
        """Return a version-explicit digest of the canonical JSON bytes."""

        payload = self.canonical_json().encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def canonical_json_value(value: object) -> str:
    """Serialize an already JSON-compatible value using canonical settings."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_digest_value(value: object) -> str:
    """Digest an already JSON-compatible value using the canonical representation."""

    payload = canonical_json_value(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
