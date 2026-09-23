Hyper-GCN (ICCV 2025) -> UR-Fall / BlazePose 33-joint experiment
================================================================

1) Clone the official repository from your fall_detection root:

   mkdir -p third_party
   git clone https://github.com/6UOOON9/Hyper-GCN.git third_party/Hyper-GCN

2) Copy blazepose.py from this package to:

   third_party/Hyper-GCN/graph/blazepose.py

3) Copy hypergcn.py from this package to:

   src/experiments/hypergcn.py

4) Keep your existing files available:

   src/experiments/lstm.py
   src/experiments/experiment_mlp.py
   data/urfall-cam0-falls.csv
   data/keypoints/  (or data/keypoints_normalized/)

5) Run the raw-coordinate experiment:

   uv run python src/experiments/hypergcn.py \
     --data-root data/keypoints \
     --result-root results/hypergcn \
     --hypergcn-root third_party/Hyper-GCN \
     --device cuda

6) Normalized-coordinate experiment:

   uv run python src/experiments/hypergcn.py \
     --data-root data/keypoints_normalized \
     --result-root results/hypergcn_normalized \
     --hypergcn-root third_party/Hyper-GCN \
     --device cuda

Notes
-----
- Model body: imported from the authors' official model/hypergcn_base.py.
- BlazePose graph: only a dataset adapter; official graph/tools.py is reused.
- Input: 30 x 33 x 2 (x/y), reshaped to [B, 2, 30, 33, 1].
- Classes: ADL=0, Fall=1.
- Split: video-grouped StratifiedGroupKFold, default 5 folds.
- UR-Fall fall-window labels: posture 0 ignored, majority vote of -1 vs 1.
- Training mirrors official Hyper-GCN choices: SGD momentum=.9, Nesterov,
  wd=.0004, lr=.05, warmup=5, steps=110/120, label smoothing=.1,
  plus the official DivergenceLoss. Class weights and early stopping are added
  for the strongly imbalanced, small UR-Fall setting and fair comparison with
  the existing project experiments.
