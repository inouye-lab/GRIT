"""A feature-engineered two-cue MNIST check with exact known geometry.

This is a semi-synthetic fixed-feature task, not a pretrained-feature benchmark.
The binary task is digit < 5. A float RGB canvas has an unchanged grayscale digit,
a colored frame B, a colored header O, and a strip of 30 independent tiny color
patches. Fixed regional extraction gives 64 pooled digit coordinates, two cue
coordinates and 30 nuisance coordinates; a random orthogonal change of basis then
hides their coordinate axes. Encoding colors with tanh and decoding with atanh
avoids clipping the Gaussian cues. PNG previews quantize colors for display only.

For efficiency, feature construction evaluates this same regional extraction in
batches without materializing every canvas. ``extract_canvas`` and the tests check
the equality. Only training pixels determine centering, scaling and the diagnostic
label direction. Raw test files may be downloaded up front, but test sampling and
features are deferred to ``test_digits`` after model selection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torchvision.datasets import MNIST

from grit.data.collection_geometry import Geometry, Split
from grit.geometry import WITHIN_LABEL, FloatArray, group_ids, to_array, to_tensor

DIMENSION = 96
TRAIN_SIZE = 3000
VALIDATION_IMAGES = 1000
TEST_IMAGES = 1000
CUE_MEAN = 2.5
CUE_SD = 0.3
NUISANCE_SD = 0.2
ALIGNMENT = 0.95
CONTENT_NORM = 5.0

# The official torchvision MNIST archive identities; downloaded files are checked
# against these MD5s and additionally recorded with SHA-256 in the run metadata.
MNIST_RESOURCES = (
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
)


def _rng(seed: int, replicate: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, replicate, 81, stream]))


def _hash_array(array: NDArray[Any]) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass(frozen=True)
class DigitData:
    train_images: NDArray[np.uint8]
    train_digits: NDArray[np.int64]
    test_images: NDArray[np.uint8]
    test_digits: NDArray[np.int64]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DigitProblem:
    train: Split
    validation: Split
    geometry: Geometry
    label_direction: FloatArray
    rotation: FloatArray
    center: FloatArray
    scale: float
    train_ids: NDArray[np.int64]
    validation_ids: NDArray[np.int64]
    seed: int
    replicate: int
    metadata: dict[str, Any]

    @property
    def u(self) -> FloatArray:
        return np.concatenate([self.geometry.u_b, self.geometry.u_o], axis=1)


def prepare_digits(root: Path) -> DigitData:
    """Download/reuse torchvision MNIST, verify input identities, return raw data.

    Pass ``code/scratch/collection-geometry/mnist`` as root. No evaluation metrics
    are computed here and no values from test images enter preprocessing.
    """
    if tuple(MNIST.resources) != MNIST_RESOURCES:
        raise RuntimeError("torchvision MNIST resources differ from the pinned inputs")
    training = MNIST(root=str(root), train=True, download=True)
    testing = MNIST(root=str(root), train=False, download=True)
    artifacts: dict[str, Any] = {}
    for name, expected_md5 in MNIST_RESOURCES:
        path = root / "MNIST" / "raw" / name
        payload = path.read_bytes()
        actual_md5 = hashlib.md5(payload).hexdigest()
        if actual_md5 != expected_md5:
            raise RuntimeError(f"MNIST archive checksum mismatch: {name}")
        artifacts[name] = {
            "md5": actual_md5,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return DigitData(
        train_images=to_array(training.data).astype(np.uint8),
        train_digits=to_array(training.targets).astype(np.int64),
        test_images=to_array(testing.data).astype(np.uint8),
        test_digits=to_array(testing.targets).astype(np.int64),
        metadata={
            "name": "MNIST",
            "provider": "torchvision.datasets.MNIST",
            "mirrors": list(MNIST.mirrors),
            "artifacts": artifacts,
            "task": "binary label 1 for digit < 5, else label 0",
        },
    )


def _pooled(images: NDArray[np.uint8]) -> FloatArray:
    tensor = to_tensor(images.astype(np.float64) / 255.0)[:, None]
    pooled = torch.nn.functional.adaptive_avg_pool2d(tensor, (8, 8))
    return to_array(pooled).reshape(len(images), 64).astype(np.float64)


def _encode(values: FloatArray) -> FloatArray:
    return (1.0 + np.tanh(values / 4.0)) / 2.0


def _decode(red: FloatArray) -> FloatArray:
    return 4.0 * np.arctanh(2.0 * red - 1.0)


def _regional_features(
    images: NDArray[np.uint8], b: FloatArray, o: FloatArray, nuisance: FloatArray
) -> FloatArray:
    """Exact batched equivalent of rendering float canvases then extracting them."""
    colors = np.column_stack([b, o, nuisance])
    return np.column_stack([_pooled(images), _decode(_encode(colors))])


def render_digit(
    image: NDArray[np.uint8], b: float, o: float, nuisance: FloatArray | None = None
) -> FloatArray:
    """Render one *float* RGB canvas (40x48); no quantization or clipping."""
    canvas = np.full((40, 48, 3), 0.08, dtype=np.float64)
    for value, region in [
        (b, (slice(4, 36), slice(8, 40))),
        (o, (slice(0, 4), slice(8, 40))),
    ]:
        red = float(_encode(np.array([value], dtype=np.float64))[0])
        canvas[region] = [red, 1.0 - red, 0.5]
    canvas[6:34, 10:38] = image[:, :, None].astype(np.float64) / 255.0
    values = np.zeros(30) if nuisance is None else nuisance
    red = _encode(values)
    canvas[38:40, :30, 0] = red
    canvas[38:40, :30, 1] = 1.0 - red
    canvas[38:40, :30, 2] = 0.5
    return canvas


def extract_canvas(canvas: FloatArray) -> FloatArray:
    """The fixed, nonlinear image-to-96-feature map before training normalization."""
    digit = to_tensor(canvas[6:34, 10:38, 0])[None, None]
    pooled = to_array(torch.nn.functional.adaptive_avg_pool2d(digit, (8, 8)))
    b = _decode(np.array([canvas[4:6, 8:40, 0].mean()], dtype=np.float64))
    o = _decode(np.array([canvas[0:4, 8:40, 0].mean()], dtype=np.float64))
    nuisance = _decode(canvas[38:40, :30, 0].mean(axis=0))
    return np.concatenate([pooled.ravel(), b, o, nuisance])


def _stratified_ids(
    digits: NDArray[np.int64], n: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    if n % 2:
        raise ValueError("image count must be even for exact label balance")
    labels = digits < 5
    ids = np.concatenate(
        [
            rng.permutation(np.flatnonzero(labels == label))[: n // 2]
            for label in (False, True)
        ]
    ).astype(np.int64)
    return rng.permutation(ids)


def _split(
    images: NDArray[np.uint8],
    digits: NDArray[np.int64],
    ids: NDArray[np.int64],
    rotation: FloatArray,
    center: FloatArray,
    scale: float,
    rng: np.random.Generator,
    *,
    balanced: bool,
) -> Split:
    n = len(ids)
    original_labels = (digits[ids] < 5).astype(np.int64)
    nuisance = NUISANCE_SD * rng.standard_normal((n, 30))
    # All four target views share their digit and nuisance strip. Each target
    # distribution is a relabeling of the exact same held-out content identities.
    if balanced:
        repeated_ids = np.tile(ids, 4)
        labels = np.tile(original_labels, 4)
        b_aligned = np.repeat([x[0] for x in WITHIN_LABEL], n)
        o_aligned = np.repeat([x[1] for x in WITHIN_LABEL], n)
        nuisance = np.tile(nuisance, (4, 1))
        jitter_b = np.tile(CUE_SD * rng.standard_normal(n), 4)
        jitter_o = np.tile(CUE_SD * rng.standard_normal(n), 4)
    else:
        repeated_ids = ids
        labels = original_labels
        b_aligned = rng.random(n) < ALIGNMENT
        o_aligned = rng.random(n) < ALIGNMENT
        jitter_b = CUE_SD * rng.standard_normal(n)
        jitter_o = CUE_SD * rng.standard_normal(n)
    signed = 2.0 * labels - 1.0
    q_b = np.where(b_aligned, signed, -signed)
    q_o = np.where(o_aligned, signed, -signed)
    b = CUE_MEAN * q_b + jitter_b
    o = CUE_MEAN * q_o + jitter_o
    regions = _regional_features(images[repeated_ids], b, o, nuisance)
    regions[:, :64] = (regions[:, :64] - center) * scale
    return Split(
        features=np.matmul(regions, rotation.transpose()),
        labels=labels,
        groups=group_ids(labels, b_aligned, o_aligned),
        q_b=q_b,
        q_o=q_o,
        b=b[:, None],
        o=o[:, None],
    )


def make_digits(seed: int, replicate: int, data: DigitData) -> DigitProblem:
    """Build disjoint training/validation content, known geometry and fixed map."""
    ids = _stratified_ids(
        data.train_digits, TRAIN_SIZE + VALIDATION_IMAGES, _rng(seed, replicate, 0)
    )
    # Split each binary class before reordering to retain exactly balanced labels.
    selected_labels = data.train_digits[ids] < 5
    train_ids = np.concatenate(
        [ids[selected_labels == label][: TRAIN_SIZE // 2] for label in (False, True)]
    )
    validation_ids = np.concatenate(
        [ids[selected_labels == label][TRAIN_SIZE // 2 :] for label in (False, True)]
    )
    train_ids = _rng(seed, replicate, 1).permutation(train_ids)
    validation_ids = _rng(seed, replicate, 2).permutation(validation_ids)
    if np.intersect1d(train_ids, validation_ids).size:
        raise AssertionError("training/validation MNIST identity overlap")
    q, r = np.linalg.qr(
        _rng(seed, replicate, 3).standard_normal((DIMENSION, DIMENSION))
    )
    rotation = q * np.sign(np.diag(r))
    pooled = _pooled(data.train_images[train_ids])
    center = pooled.mean(axis=0)
    scale = CONTENT_NORM / np.sqrt(np.square(pooled - center).sum(axis=1).mean())
    train = _split(
        data.train_images,
        data.train_digits,
        train_ids,
        rotation,
        center,
        scale,
        _rng(seed, replicate, 4),
        balanced=False,
    )
    validation = _split(
        data.train_images,
        data.train_digits,
        validation_ids,
        rotation,
        center,
        scale,
        _rng(seed, replicate, 5),
        balanced=True,
    )
    stable_direction = np.zeros(DIMENSION)
    stable_direction[:64] = scale * (
        pooled[train.labels == 1].mean(axis=0) - pooled[train.labels == 0].mean(axis=0)
    )
    geometry = Geometry(
        stable=rotation[:, np.r_[np.arange(64), np.arange(66, DIMENSION)]],
        u_b=rotation[:, 64:65],
        u_o=rotation[:, 65:66],
        noise_basis=rotation[:, 66:],
        noise_sd=np.full(30, NUISANCE_SD),
        m_b=np.array([CUE_MEAN]),
        m_o=np.array([CUE_MEAN]),
    )
    return DigitProblem(
        train=train,
        validation=validation,
        geometry=geometry,
        label_direction=rotation @ stable_direction,
        rotation=rotation,
        center=center,
        scale=float(scale),
        train_ids=train_ids,
        validation_ids=validation_ids,
        seed=seed,
        replicate=replicate,
        metadata={
            "train_ids": train_ids.tolist(),
            "validation_ids": validation_ids.tolist(),
            "train_ids_sha256": _hash_array(train_ids),
            "validation_ids_sha256": _hash_array(validation_ids),
            "train_features_sha256": _hash_array(train.features),
            "validation_features_sha256": _hash_array(validation.features),
            "train_images": TRAIN_SIZE,
            "validation_images": VALIDATION_IMAGES,
            "validation_views": 4 * VALIDATION_IMAGES,
            "content_scale": float(scale),
            "feature_dimension": DIMENSION,
            "content_dimension": 64,
            "stable_dimension": 94,
            "unstable_rank": 2,
            "nuisance_dimension": 30,
            "cue_mean": CUE_MEAN,
            "cue_sd": CUE_SD,
            "training_alignment": ALIGNMENT,
            "target_content_reuse": "four cue views per held-out identity",
            "label_direction_source": "training-only centered pooled class mean",
            "preprocessing": "adaptive 8x8 mean pool; train mean center; RMS norm 5",
            "renderer": "float RGB 40x48 with disjoint digit, frame, header, strip",
            "cue_encoding": "red=(1+tanh(value/4))/2; green=1-red; blue=0.5",
            "cue_extraction": "value=4*atanh(2*mean(red)-1)",
        },
    )


def test_digits(problem: DigitProblem, data: DigitData) -> Split:
    """Generate held-out test views only after the caller freezes selection.

    This also registers the exact test IDs in the problem metadata. Official
    train/test pools differ; row IDs are always scoped to their official split.
    """
    ids = _stratified_ids(
        data.test_digits, TEST_IMAGES, _rng(problem.seed, problem.replicate, 6)
    )
    split = _split(
        data.test_images,
        data.test_digits,
        ids,
        problem.rotation,
        problem.center,
        problem.scale,
        _rng(problem.seed, problem.replicate, 7),
        balanced=True,
    )
    problem.metadata["test_ids"] = ids.tolist()
    problem.metadata["test_ids_sha256"] = _hash_array(ids)
    problem.metadata["test_features_sha256"] = _hash_array(split.features)
    problem.metadata["test_images"] = TEST_IMAGES
    problem.metadata["test_views"] = 4 * TEST_IMAGES
    return split


def preview_digits(problem: DigitProblem, data: DigitData, path: Path) -> None:
    """Show the actual float renderer used by the analytic regional feature map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    identity = int(problem.train_ids[0])
    image = data.train_images[identity]
    figure, axes = plt.subplots(1, 4, figsize=(9, 2.7), constrained_layout=True)
    for axis, (b, o) in zip(axes, [(1, 1), (-1, 1), (1, -1), (-1, -1)], strict=True):
        canvas = render_digit(image, CUE_MEAN * b, CUE_MEAN * o)
        axis.imshow(canvas, interpolation="nearest")
        axis.set_title(f"Frame B: {b:+d}; header O: {o:+d}", fontsize=10)
        axis.axis("off")
    figure.suptitle(
        "Same MNIST digit; independently changed frame and header colors\n"
        "Fixed regional features preserve exact two-direction cue geometry",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
