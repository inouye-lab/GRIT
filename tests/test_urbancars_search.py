"""UrbanCars runner on a synthetic prepared artifact: checks, optimizer, sealing."""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from grit.data import urbancars as uc
from grit.features.urbancars import CACHE_SCHEMA, write_feature_table
from grit.geometry import WITHIN_LABEL, group_index
from grit.schemas import canonical_digest_value
from grit.search import urbancars as runner

SEEDS = {
    "tuning": [11, 12, 13],
    "confirmation": [21, 22],
    "final": list(range(31, 41)),
}


def _artifact(root: Path) -> None:
    """Features z = 1.2 y s + q_B m_B + q_O m_O + noise in 512-D, 95%-aligned cues."""

    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.standard_normal((512, 512)))[0]
    stable, m_b, m_o = basis[:, 0], basis[:, 1] * 2.5, basis[:, 2] * 2.0
    images: list[dict[str, Any]] = []
    features: dict[str, list[tuple[str, np.ndarray[Any, Any], int]]] = {}

    def feature(y: int, qb: int, qo: int) -> np.ndarray[Any, Any]:
        sy = 2 * y - 1
        noise = 0.4 * rng.standard_normal(512)
        return 1.2 * sy * stable + qb * m_b + qo * m_o + noise

    def add(
        table: str, split: str, image_id: str, y: int, b_al: bool, o_al: bool
    ) -> None:
        sy = 2 * y - 1
        qb, qo = (sy if b_al else -sy), (sy if o_al else -sy)
        images.append(
            {
                "image_id": image_id,
                "role": "standard",
                "split": split,
                "label": y,
                "group": group_index(y, b_al, o_al),
            }
        )
        features.setdefault(table, []).append((image_id, feature(y, qb, qo), y))

    for i in range(400):
        y = i % 2
        add("train", "train", f"train/{i}", y, rng.random() < 0.95, rng.random() < 0.95)
    for split, table in (("val", "validation"), ("test", "test")):
        for i in range(80):
            y, w = divmod(i % 8, 4)
            b_al, o_al = WITHIN_LABEL[w]
            add(table, split, f"{split}/{i}", y, b_al, o_al)
    train = features["train"]
    anchors = [image_id for image_id, _, _ in train[:96]]
    for f in ("b", "o"):
        rows: list[tuple[str, np.ndarray[Any, Any], int]] = []
        for image_id, z, y in train[:96]:
            direction = m_b if f == "b" else m_o
            # The partner flips one cue: z' = z - 2 q m (sign from the anchor).
            q = 1.0 if float(z @ direction) > 0 else -1.0
            rows.append((f"partner-{f}/{image_id}", z - 2 * q * direction, y))
        features[f"partner_{f}"] = rows
    pools = {
        "calibration": anchors[:16],
        "reference": anchors[16:48],
        "bank0": anchors[48:64],
        "bank1": anchors[64:80],
        "bank2": anchors[80:96],
    }
    groups = {r["image_id"]: r["group"] for r in images}
    manifest: dict[str, Any] = {
        "schema_version": uc.MANIFEST_SCHEMA,
        "construction": uc.CONSTRUCTIONS["standard"].model_dump(mode="json"),
        "images": images,
        "anchor_pools": {
            name: [
                {"anchor": a, "group": groups[a], "u": float(rng.random())} for a in ids
            ]
            for name, ids in pools.items()
        },
    }
    manifest["digest"] = canonical_digest_value(manifest)
    root.mkdir(parents=True)
    (root / "dataset-manifest.json").write_text(json.dumps(manifest))
    roles = {
        "train": "training",
        "validation": "validation",
        "test": "final_test",
        "partner_b": "pair_projection",
        "partner_o": "pair_projection",
    }
    tables = [
        write_feature_table(
            root / "features",
            name,
            roles[name],  # pyright: ignore[reportArgumentType]
            tuple(r[0] for r in rows),
            tuple(r[2] for r in rows),
            torch.tensor(np.stack([r[1] for r in rows]), dtype=torch.float32),
            "none",
        )
        for name, rows in features.items()
    ]
    (root / "features" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": CACHE_SCHEMA,
                "dataset_manifest_digest": manifest["digest"],
                "tables": tables,
            }
        )
    )


def _config(tmp_path: Path, kind: str, **changes: object) -> Path:
    config: dict[str, object] = {
        "dataset": "urbancars",
        "experiment_name": f"test-{kind}",
        "cache_root": str(tmp_path / "artifact"),
        "output_root": str(tmp_path / f"out-{kind}"),
        "kind": kind,
        "seeds": SEEDS,
        "budget": 8,
        "collections": [
            {"name": "b_only", "share": 0.0},
            {"name": "balanced", "share": 0.5},
            {"name": "o_only", "share": 1.0},
            {"name": "count_50", "count_fraction": 0.5},
        ],
        "ranks": [1, 2, 4],
        "matched_ranks": [1, 2],
        "consistency_strengths": [1.0, 10.0],
        "learning_rates": [1e-2],
        "weight_decays": [0.0, 1e-4],
        "max_epochs": 3,
        "batch_size": 128,
        **changes,
    }
    path = tmp_path / f"{kind}.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    _artifact(tmp_path / "artifact")
    return tmp_path


def test_main_requires_frozen_optimizer_and_count_50_decision(artifact: Path) -> None:
    with pytest.raises(ValueError, match="frozen optimizer"):
        runner.load_config(_config(artifact, "main"))
    optimizers = {
        m: {"learning_rate": 1e-2, "weight_decay": 0.0}
        for m in ("erm", "grit", "consistency", "reference")
    }
    with pytest.raises(ValueError, match="include_count_50"):
        runner.load_config(_config(artifact, "main", optimizers=optimizers))


def test_checks_and_optimizer_are_validation_only(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def sealed(*_: object) -> None:
        raise AssertionError("test table opened")

    monkeypatch.setattr(runner, "load_urbancars_test_table", sealed)
    runner.run_urbancars(_config(artifact, "check"))
    report = json.loads((artifact / "out-check" / "check-report.json").read_text())
    assert set(report) >= {
        "C1_shortcut_reliance",
        "C2_geometry",
        "C3_reference_projection",
        "C4_energy_calibration",
        "pilot",
    }
    assert report["C1_shortcut_reliance"]["passes"] in (True, False)
    assert report["C4_energy_calibration"]["count_table"]["256"]["o_only"] == 256
    runner.run_urbancars(_config(artifact, "optimizer", banks=[0]))
    chosen = json.loads((artifact / "out-optimizer" / "optimizer.json").read_text())[
        "chosen"
    ]
    assert set(chosen) == {"erm", "grit", "consistency", "reference"}


def test_main_freezes_winners_and_unseals_only_with_their_hash(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    optimizers = {
        m: {"learning_rate": 1e-2, "weight_decay": 0.0}
        for m in ("erm", "erm_matched", "grit", "consistency", "reference")
    }
    path = _config(
        artifact,
        "main",
        methods=["erm", "erm_matched", "grit", "consistency", "reference"],
        optimizers=optimizers,
        include_count_50=False,
    )
    original = runner.load_urbancars_test_table

    def sealed(*_: object) -> None:
        raise AssertionError("test table opened before unsealing")

    monkeypatch.setattr(runner, "load_urbancars_test_table", sealed)
    status = runner.run_urbancars(path)
    digest = status["winners_sha256"]
    assert digest is not None and "final" not in status["tasks_by_stage"]
    with pytest.raises(ValueError, match="does not match"):
        runner.run_urbancars(path, unseal_test="sha256:wrong")
    monkeypatch.setattr(runner, "load_urbancars_test_table", original)
    runner.run_urbancars(path, unseal_test=digest)
    summary = json.loads((artifact / "out-main" / "summary.json").read_text())
    assert "grit@balanced-b0|worst_group" in summary["rows"]
    assert "grit@balanced-b1|matched-r2|train_weighted" in summary["rows"]
    contrast = summary["contrasts"]["grit@balanced-minus-erm|worst_group|worst_group"]
    assert contrast["n"] == 10 and "bootstrap" in contrast
    assert (
        "grit@balanced-minus-erm_matched|worst_group|worst_group"
        in summary["contrasts"]
    )
    assert not any("count_50" in row for row in summary["rows"])
