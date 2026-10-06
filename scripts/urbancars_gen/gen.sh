#!/usr/bin/env bash
# UrbanCars generation environment: car masks, component export, upstream reference run.
#   bash scripts/urbancars_gen/gen.sh setup     # clones, venv, MaskFormer weights link
#   bash scripts/urbancars_gen/gen.sh masks     # MaskFormer car masks (GPU)
#   bash scripts/urbancars_gen/gen.sh fallback  # masks for listed cars left without one
#   bash scripts/urbancars_gen/gen.sh export    # components.json + LVIS masks
#   bash scripts/urbancars_gen/gen.sh upstream  # unmodified upstream generator (G0b);
#                                               # extra args pass through (a fallback adds
#                                               # --co_occur_obj_ratio 0.99)
# Sources live in $PROJECT_SCRATCH/data/urbancars (the upstream data/ layout), written
# by `uv run scripts/prepare_urbancars.py download`.
set -euo pipefail
SRC="${URBANCARS_SOURCES:-$PROJECT_SCRATCH/data/urbancars}"
WAM="${URBANCARS_UPSTREAM:-$PROJECT_SCRATCH/data/Whac-A-Mole}"
VENV="${URBANCARS_GEN_VENV:-$PROJECT_SCRATCH/venv-urbancars-gen}"
WAM_COMMIT=c91b097043246bc726fccd70f9860873e28d6122
MASKFORMER_COMMIT=da3e60d85fdeedcb31476b5edd7d328826ce56cc
DETECTRON2_COMMIT=a2f4a8771ab77e8411c26b27f24f9489a28a2453
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$VENV/bin/python"
run() { (cd "$WAM" && PYTHONPATH="$WAM" "$PY" "$@"); }

case "${1:-}" in
setup)
  [ -d "$WAM/.git" ] || git clone https://github.com/facebookresearch/Whac-A-Mole.git "$WAM"
  git -C "$WAM" checkout -q "$WAM_COMMIT"
  MF="$WAM/create_datasets/urbancars/maskformer"
  # Upstream clones MaskFormer over SSH; HTTPS here.
  [ -d "$MF/.git" ] || git clone https://github.com/facebookresearch/MaskFormer.git "$MF"
  git -C "$MF" checkout -q "$MASKFORMER_COMMIT"
  ln -sfn "$SRC" "$WAM/data"
  mkdir -p "$WAM/exp/weights"
  ln -sfn "$SRC/weights/maskformer_panoptic_swin_large.pkl" "$WAM/exp/weights/maskformer_panoptic_swin_large.pkl"
  [ -x "$PY" ] || uv venv --python 3.10 "$VENV"
  # torch must match the system CUDA toolkit that builds detectron2 (11.7 here).
  uv pip install --python "$PY" torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
  uv pip install --python "$PY" -r "$HERE/requirements.txt"
  uv pip install --python "$PY" --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@$DETECTRON2_COMMIT"
  "$PY" -c "import detectron2, lvis, skimage, scipy, torch; print('ok', torch.__version__, torch.cuda.is_available())"
  ;;
masks)
  mkdir -p "$SRC/stanford_cars/mask/cars_train" "$SRC/stanford_cars/mask/cars_test"
  run create_datasets/urbancars/predict_car_mask.py --split train
  run create_datasets/urbancars/predict_car_mask.py --split test
  ;;
fallback)
  run "$HERE/fallback_masks.py" --data "$SRC"
  ;;
export)
  run "$HERE/export_components.py" --data "$SRC" --segmenter maskformer-swin-large
  ;;
upstream)
  shift
  for split in train val test; do run create_datasets/urbancars/gen_urbancars.py --split "$split" "$@"; done
  ;;
*)
  echo "usage: $0 setup|masks|fallback|export|upstream" >&2
  exit 2
  ;;
esac
