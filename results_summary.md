# Fall Detection 实验结果汇总

- 排序指标：`f1`
- 结果粒度：`window`

## 全部结果

| rank | result | scope | accuracy | precision | recall | f1 | roc_auc | tp | fp | fn | tn | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | results/ur_blazepose/stgcn | overall | 0.9889 | 0.9245 | 0.9608 | 0.9423 | 0.9982 | 49 | 4 | 2 | 486 | metrics.txt |
| 2 | results/ur_blazepose/rf | overall | 0.9834 | 0.9375 | 0.8824 | 0.9091 | 0.9974 |  |  |  |  | metrics.txt |
| 3 | results/ur_blazepose/transformer_keypoints_ur | window | 0.9797 | 0.9545 | 0.8235 | 0.8842 | 0.9727 | 42 | 2 | 9 | 488 | seed_metrics.csv |
| 4 | results/ur_blazepose/transformer_keypoints_ur/seed_42 | window | 0.9797 | 0.9545 | 0.8235 | 0.8842 | 0.9727 | 42 | 2 | 9 | 488 | metrics.txt |
| 5 | results/ur_blazepose/mlp | overall | 0.9704 | 0.7778 | 0.9608 | 0.8596 | 0.9818 | 49 | 14 | 2 | 476 | metrics.txt |
| 6 | results/combined_ur_vitpose_le2i_vitpose/transformer_combined_ur_vitpose_le2i_vitpose | window | 0.9652 | 0.8501 | 0.8155 | 0.8324 | 0.9665 | 601 | 106 | 136 | 6103 | seed_metrics.csv |
| 7 | results/combined_ur_vitpose_le2i_vitpose/transformer_combined_ur_vitpose_le2i_vitpose/seed_42 | window | 0.9652 | 0.8501 | 0.8155 | 0.8324 | 0.9665 | 601 | 106 | 136 | 6103 | metrics.txt |
| 8 | results/combined_ur_vitpose_le2i_vitpose/stgcn_xyc_coco17_lite_recall90 | overall | 0.9616 | 0.7752 | 0.8982 | 0.8322 | 0.9844 | 662 | 192 | 75 | 6017 | metrics.txt |
| 9 | results/ur_blazepose/lstm | overall | 0.9649 | 0.7759 | 0.8824 | 0.8257 | 0.9469 | 45 | 13 | 6 | 477 | metrics.txt |
| 10 | results/ur_vitpose/stgcn_xyc_coco17_lite | overall | 0.9554 | 0.7696 | 0.8698 | 0.8166 | 0.9796 | 167 | 50 | 25 | 1440 | metrics.txt |
| 11 | results/combined_ur_vitpose_le2i_vitpose/lstm | overall | 0.9621 | 0.8591 | 0.7693 | 0.8117 | 0.9776 | 567 | 93 | 170 | 6116 | metrics.txt |
| 12 | results/combined_ur_vitpose_le2i_vitpose/rf | overall | 0.9433 | 0.6903 | 0.8440 | 0.7595 | 0.9754 |  |  |  |  | metrics.txt |
| 13 | results/combined_ur_le2i_blazepose/stgcn | overall | 0.9354 | 0.6992 | 0.7959 | 0.7444 | 0.9577 | 581 | 250 | 149 | 5193 | metrics.txt |
| 14 | results/combined_ur_le2i_blazepose/rf | overall | 0.9334 | 0.6874 | 0.8014 | 0.7400 | 0.9680 |  |  |  |  | metrics.txt |
| 15 | results/le2i_blazepose/rf | overall | 0.9546 | 0.6962 | 0.7760 | 0.7339 | 0.9705 |  |  |  |  | metrics.txt |
| 16 | results/combined_ur_vitpose_le2i_vitpose/stgcn | overall | 0.9351 | 0.6525 | 0.8304 | 0.7307 | 0.9630 | 612 | 326 | 125 | 5883 | metrics.txt |
| 17 | results/le2i_blazepose/stgcn | overall | 0.9505 | 0.6520 | 0.8293 | 0.7300 | 0.9725 | 311 | 166 | 64 | 4108 | metrics.txt |
| 18 | results/combined_ur_le2i_blazepose/transformer_combined_ur_le2i_blazepose | window | 0.9373 | 0.7526 | 0.7000 | 0.7253 | 0.9544 | 511 | 168 | 219 | 5275 | seed_metrics.csv |
| 19 | results/combined_ur_le2i_blazepose/transformer_combined_ur_le2i_blazepose/seed_42 | window | 0.9373 | 0.7526 | 0.7000 | 0.7253 | 0.9544 | 511 | 168 | 219 | 5275 | metrics.txt |
| 20 | results/combined_ur_le2i_blazepose/lstm | overall | 0.9256 | 0.6507 | 0.8014 | 0.7182 | 0.9572 | 585 | 314 | 145 | 5129 | metrics.txt |
| 21 | results/combined_ur_vitpose_le2i_vitpose/mlp | overall | 0.9276 | 0.6121 | 0.8670 | 0.7176 | 0.9675 | 639 | 405 | 98 | 5804 | metrics.txt |
| 22 | results/blazepose_normalized/lstm/lstm_normalized | overall | 0.9446 | 0.7265 | 0.7010 | 0.7135 | 0.8900 | 619 | 233 | 264 | 7847 | metrics.txt |
| 23 | results/combined_ur_le2i_blazepose/mlp | overall | 0.9184 | 0.6110 | 0.8521 | 0.7117 | 0.9652 | 622 | 396 | 108 | 5047 | metrics.txt |
| 24 | results/le2i_blazepose/mlp | overall | 0.9408 | 0.5929 | 0.8507 | 0.6988 | 0.9693 | 319 | 219 | 56 | 4055 | metrics.txt |
| 25 | results/combined_blazepose_normalized_le2i_blazepose_normalized/rf_normalized | overall | 0.9260 | 0.6872 | 0.6863 | 0.6868 | 0.9481 |  |  |  |  | metrics.txt |
| 26 | results/combined_blazepose_normalized_le2i_blazepose_normalized/transformer_combined_blazepose_normalized_le2i_blazepose_normalized | window | 0.9276 | 0.7096 | 0.6562 | 0.6819 | 0.9152 | 479 | 196 | 251 | 5247 | seed_metrics.csv |
| 27 | results/combined_blazepose_normalized_le2i_blazepose_normalized/transformer_combined_blazepose_normalized_le2i_blazepose_normalized/seed_42 | window | 0.9276 | 0.7096 | 0.6562 | 0.6819 | 0.9152 | 479 | 196 | 251 | 5247 | metrics.txt |
| 28 | results/combined_blazepose_normalized_le2i_blazepose_normalized/lstm_normalized | overall | 0.9269 | 0.7067 | 0.6534 | 0.6790 | 0.9387 | 477 | 198 | 253 | 5245 | metrics.txt |
| 29 | results/blazepose_normalized/stgcn/stgcn_normalized | overall | 0.9342 | 0.6570 | 0.6942 | 0.6751 | 0.9571 | 613 | 320 | 270 | 7760 | metrics.txt |
| 30 | results/combined_blazepose_normalized_le2i_blazepose_normalized/stgcn_normalized | overall | 0.9090 | 0.5861 | 0.7836 | 0.6706 | 0.9392 | 572 | 404 | 158 | 5039 | metrics.txt |
| 31 | results/combined_blazepose_normalized_le2i_blazepose_normalized/mlp_normalized | overall | 0.9065 | 0.5772 | 0.7836 | 0.6647 | 0.9299 | 572 | 419 | 158 | 5024 | metrics.txt |
| 32 | results/blazepose_normalized/mlp/mlp_normalized | overall | 0.9279 | 0.6167 | 0.7089 | 0.6596 | 0.8781 | 626 | 389 | 257 | 7691 | metrics.txt |
| 33 | results/le2i_blazepose/lstm | overall | 0.9157 | 0.4857 | 0.7680 | 0.5950 | 0.9396 | 288 | 305 | 87 | 3969 | metrics.txt |

## 各结果根目录最佳结果

| root | result | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| results | results/ur_blazepose/stgcn | 0.9889 | 0.9245 | 0.9608 | 0.9423 | 0.9982 |
