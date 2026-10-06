"""Build every pair method's inputs: canonical endpoints and pair labels only."""

from __future__ import annotations

from typing import Protocol

import torch

from grit.data.waterbirds_pairs import WaterbirdsOraclePairSet
from grit.features.waterbirds import (
    WaterbirdsFeatureCache,
    WaterbirdsTuningFeatureCache,
    waterbirds_oracle_pair_features,
)
from grit.methods.pair_inputs import PairInputs


class _PairTable(Protocol):
    @property
    def features(self) -> torch.Tensor: ...

    @property
    def targets(self) -> torch.Tensor: ...


def cmnist_pair_inputs(red: _PairTable, green: _PairTable, count: int) -> PairInputs:
    """The first ``count`` red-minus-green pairs, labelled by their noisy targets.

    Every pair source is a training source, so its noisy target is already visible to
    ERM; the clean label, digit and color never leave the feature tables.
    """

    labels = red.targets[:count]
    if not torch.equal(labels, green.targets[:count]):
        raise ValueError("CMNIST pair endpoints must share their noisy target")
    return PairInputs(
        left=red.features[:count], right=green.features[:count], labels=labels
    )


def waterbirds_pair_inputs(
    cache: WaterbirdsFeatureCache | WaterbirdsTuningFeatureCache,
    pairs: WaterbirdsOraclePairSet,
) -> PairInputs:
    """All land-minus-water pairs in manifest order, labelled by ``bird_label``.

    The endpoint lookup already checks that the relation's label equals both
    endpoints' supervised labels, so the label adds no information.
    """

    land, water = waterbirds_oracle_pair_features(cache, pairs)
    labels = torch.tensor(
        [relation.bird_label for relation in pairs.manifest.records],
        dtype=torch.int64,
    )
    return PairInputs(left=land, right=water, labels=labels)
