# Fall Detection 实验结果汇总

- 排序指标：`f1`
- 结果粒度：`all`

## 全部结果

| rank | result | scope | accuracy | precision | recall | f1 | roc_auc | tp | fp | fn | tn | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | results/ur_blazepose/stgcn | overall | 0.9889 | 0.9245 | 0.9608 | 0.9423 | 0.9982 | 49 | 4 | 2 | 486 | metrics.txt |
| 2 | results/vitpose/rf | overall | 0.9891 | 0.9772 | 0.9014 | 0.9378 | 0.9979 |  |  |  |  | metrics.txt |
| 3 | results/ur_origin/stgcn | overall | 0.9861 | 0.9084 | 0.9547 | 0.9310 | 0.9976 | 843 | 85 | 40 | 7995 | metrics.txt |
| 4 | results/vitpose/transformer_vitpose | video | 0.9286 | 0.9032 | 0.9333 | 0.9180 | 0.9467 | 28 | 3 | 2 | 37 | seed_metrics.csv |
| 5 | results/vitpose/lstm | overall | 0.9843 | 0.9528 | 0.8715 | 0.9104 | 0.9899 | 787 | 39 | 116 | 8924 | metrics.txt |
| 6 | results/ur_blazepose/rf | overall | 0.9834 | 0.9375 | 0.8824 | 0.9091 | 0.9974 |  |  |  |  | metrics.txt |
| 7 | results/ur_blazepose/transformer_keypoints_ur | window | 0.9797 | 0.9545 | 0.8235 | 0.8842 | 0.9727 | 42 | 2 | 9 | 488 | seed_metrics.csv |
| 8 | results/ur_blazepose/transformer_keypoints_ur/seed_42 | window | 0.9797 | 0.9545 | 0.8235 | 0.8842 | 0.9727 | 42 | 2 | 9 | 488 | metrics.txt |
| 9 | results/ur_origin/lstm | overall | 0.9776 | 0.9231 | 0.8426 | 0.8810 | 0.9633 | 744 | 62 | 139 | 8018 | metrics.txt |
| 10 | results/vitpose/mlp | overall | 0.9772 | 0.8942 | 0.8516 | 0.8724 | 0.9880 | 769 | 91 | 134 | 8872 | metrics.txt |
| 11 | results/ur_blazepose/mlp | overall | 0.9704 | 0.7778 | 0.9608 | 0.8596 | 0.9818 | 49 | 14 | 2 | 476 | metrics.txt |
| 12 | results/ur_blazepose/lstm | overall | 0.9649 | 0.7759 | 0.8824 | 0.8257 | 0.9469 | 45 | 13 | 6 | 477 | metrics.txt |
| 13 | results/vitpose/transformer_vitpose | window | 0.9700 | 0.8936 | 0.7630 | 0.8232 | 0.9429 | 689 | 82 | 214 | 8881 | seed_metrics.csv |
| 14 | results/vitpose/transformer_vitpose/seed_42 | window | 0.9700 | 0.8936 | 0.7630 | 0.8232 | 0.9429 | 689 | 82 | 214 | 8881 | metrics.txt |
| 15 | results/ur_origin/mlp | overall | 0.9530 | 0.7077 | 0.8913 | 0.7890 | 0.9763 | 787 | 325 | 96 | 7755 | metrics.txt |
| 16 | results/vitpose/stgcn | overall | 0.9463 | 0.6451 | 0.9181 | 0.7578 | 0.9884 | 829 | 456 | 74 | 8507 | metrics.txt |
| 17 | results/combined_ur_le2i_blazepose/mlp | overall | 0.9297 | 0.6806 | 0.8355 | 0.7502 | 0.9659 | 569 | 267 | 112 | 4443 | metrics.txt |
| 18 | results/combined_ur_le2i_blazepose/rf | overall | 0.9284 | 0.6828 | 0.8091 | 0.7406 | 0.9678 |  |  |  |  | metrics.txt |
| 19 | results/combined_ur_le2i_blazepose/stgcn | overall | 0.9243 | 0.6567 | 0.8399 | 0.7371 | 0.9605 | 572 | 299 | 109 | 4411 | metrics.txt |
| 20 | results/le2i_blazepose/rf | overall | 0.9546 | 0.6962 | 0.7760 | 0.7339 | 0.9705 |  |  |  |  | metrics.txt |
| 21 | results/le2i_blazepose/stgcn | overall | 0.9505 | 0.6520 | 0.8293 | 0.7300 | 0.9725 | 311 | 166 | 64 | 4108 | metrics.txt |
| 22 | results/blazepose_normalized/lstm/lstm_normalized | overall | 0.9446 | 0.7265 | 0.7010 | 0.7135 | 0.8900 | 619 | 233 | 264 | 7847 | metrics.txt |
| 23 | results/combined_ur_le2i_blazepose/lstm | overall | 0.9141 | 0.6267 | 0.7915 | 0.6995 | 0.9503 | 539 | 321 | 142 | 4389 | metrics.txt |
| 24 | results/le2i_blazepose/mlp | overall | 0.9408 | 0.5929 | 0.8507 | 0.6988 | 0.9693 | 319 | 219 | 56 | 4055 | metrics.txt |
| 25 | results/ur_blazepose/transformer_keypoints_ur | video | 0.8000 | 1 | 0.5333 | 0.6957 | 0.9792 | 16 | 0.0000 | 14 | 40 | seed_metrics.csv |
| 26 | results/blazepose_normalized/stgcn/stgcn_normalized | overall | 0.9342 | 0.6570 | 0.6942 | 0.6751 | 0.9571 | 613 | 320 | 270 | 7760 | metrics.txt |
| 27 | results/blazepose_normalized/mlp/mlp_normalized | overall | 0.9279 | 0.6167 | 0.7089 | 0.6596 | 0.8781 | 626 | 389 | 257 | 7691 | metrics.txt |
| 28 | results/le2i_blazepose/lstm | overall | 0.9157 | 0.4857 | 0.7680 | 0.5950 | 0.9396 | 288 | 305 | 87 | 3969 | metrics.txt |

## 各结果根目录最佳结果

| root | result | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| results | results/ur_blazepose/stgcn | 0.9889 | 0.9245 | 0.9608 | 0.9423 | 0.9982 |
