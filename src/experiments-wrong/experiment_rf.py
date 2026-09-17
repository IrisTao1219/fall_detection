from pathlib import Path
import argparse
import csv
import json

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# 1. 实验配置
#
# 后续跑不同实验，主要改这一部分
# ============================================================

# 由命令行 --data-root 自动确定：
#   data/keypoints            -> EXPERIMENT_NAME="rf"
#   data/keypoints_normalized -> EXPERIMENT_NAME="rf_normalized"
EXPERIMENT_NAME = None


# ============================================================
# 数据
# ============================================================

# 由命令行 --data-root 传入
DATA_ROOT = None

# 原始 BlazePose：
# keypoints -> [T, 33, 4]
COORDINATE_FIELD = "keypoints"

# 当前帧是否存在 Pose
VALID_MASK_FIELD = "valid_mask"

# x = 0
# y = 1
# z = 2
# visibility = 3

COORDINATE_CHANNELS = (0, 1)

COORDINATE_NAMES = ("x", "y")


# ============================================================
# Visibility
# ============================================================

USE_VISIBILITY_FILTER = True

VISIBILITY_FIELD = "keypoints"

VISIBILITY_CHANNEL = 3

VISIBILITY_THRESHOLD = 0.3


# ============================================================
# Sliding Window
# ============================================================

# 所有视频都是 30 FPS
EXPECTED_FPS = 30.0

# 论文方法：
# swl = fps × seconds
WINDOW_SECONDS = 1.0

# 每次向右移动 1 帧
WINDOW_STRIDE = 1

# 至少有多少帧成功检测到人体，
# 才保留这个窗口
#
# =1 表示只过滤完全没有人体关键点的窗口
MIN_VALID_FRAMES = 1


# ============================================================
# 缺失关键点处理
#
# interpolate：
# 在当前窗口内部沿时间轴插值
#
# zero：
# 直接填 0
# ============================================================

MISSING_VALUE_STRATEGY = "interpolate"


# ============================================================
# Label
# ============================================================

LABEL_MAP = {
    "adl": 0,
    "fall": 1,
}

ID_TO_LABEL = {
    0: "adl",
    1: "fall",
}

POSITIVE_LABEL = 1


# ============================================================
# Cross Validation
# ============================================================

MAX_FOLDS = 5

RANDOM_STATE = 42


# ============================================================
# Random Forest
# ============================================================

N_ESTIMATORS = 500

MAX_FEATURES = "sqrt"

CLASS_WEIGHT = "balanced"

MAX_DEPTH = None

MIN_SAMPLES_SPLIT = 2

MIN_SAMPLES_LEAF = 1

# 共享服务器建议 4 或 8
N_JOBS = 8


# ============================================================
# 输出
# ============================================================

RESULT_ROOT = Path("results")

# 运行时由 --data-root 自动确定：
#   results/rf/
#   results/rf_normalized/
RESULT_DIR = None

SAVE_FINAL_MODEL = True

SAVE_FEATURE_IMPORTANCE = True


# ============================================================
# 2. 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Random Forest fall-detection experiment "
            "with 30-frame sliding windows."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "NPZ 数据目录，例如 data/keypoints "
            "或 data/keypoints_normalized"
        ),
    )

    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results"),
        help="结果根目录，默认 results",
    )

    return parser.parse_args()


def configure_experiment(args):
    """
    仅根据 --data-root 区分 raw / normalized 实验：

        data/keypoints
            -> results/rf/

        data/keypoints_normalized
            -> results/rf_normalized/
    """
    global DATA_ROOT
    global RESULT_ROOT
    global RESULT_DIR
    global EXPERIMENT_NAME

    DATA_ROOT = args.data_root
    RESULT_ROOT = args.result_root

    dataset_name = DATA_ROOT.resolve().name.lower()

    if "normalized" in dataset_name:
        EXPERIMENT_NAME = "rf_normalized"
    else:
        EXPERIMENT_NAME = "rf"

    RESULT_DIR = RESULT_ROOT / EXPERIMENT_NAME


# ============================================================
# 2. 缺失关键点处理
# ============================================================

def fill_missing_temporally(window):
    """
    window:
        [W, 33, C]

    对每个关节、每个坐标维度，
    沿时间轴插值。

    如果整个窗口该坐标都不存在：
        填 0

    如果只有一个有效值：
        使用该值填满窗口
    """

    window = window.astype(
        np.float32
    ).copy()

    if MISSING_VALUE_STRATEGY == "zero":

        return np.nan_to_num(
            window,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    if MISSING_VALUE_STRATEGY != "interpolate":

        raise ValueError(
            "MISSING_VALUE_STRATEGY "
            "只能是 interpolate 或 zero"
        )

    num_frames = window.shape[0]

    time_index = np.arange(
        num_frames
    )

    for joint in range(
        window.shape[1]
    ):

        for channel in range(
            window.shape[2]
        ):

            values = window[
                :,
                joint,
                channel
            ]

            valid = np.isfinite(
                values
            )

            valid_count = int(
                valid.sum()
            )

            # 整个窗口这个特征都缺失
            if valid_count == 0:

                window[
                    :,
                    joint,
                    channel
                ] = 0.0

                continue

            # 只有一个值
            if valid_count == 1:

                window[
                    :,
                    joint,
                    channel
                ] = values[
                    valid
                ][0]

                continue

            # 时间插值
            missing = ~valid

            values[
                missing
            ] = np.interp(
                time_index[missing],
                time_index[valid],
                values[valid]
            )

            window[
                :,
                joint,
                channel
            ] = values

    return window


# ============================================================
# 3. 获取一个视频的坐标序列
# ============================================================

def prepare_coordinates(data):
    """
    返回：

        coords:
            [T, 33, 2]

        valid_mask:
            [T]
    """

    keypoints = data[
        COORDINATE_FIELD
    ]

    valid_mask = data[
        VALID_MASK_FIELD
    ].astype(bool)

    # --------------------------------------------------------
    # x / y
    # --------------------------------------------------------

    coords = keypoints[
        :,
        :,
        COORDINATE_CHANNELS
    ].astype(
        np.float32
    ).copy()

    # --------------------------------------------------------
    # 整帧无 Pose
    # --------------------------------------------------------

    coords[
        ~valid_mask
    ] = np.nan

    # --------------------------------------------------------
    # visibility filter
    # --------------------------------------------------------

    if USE_VISIBILITY_FILTER:

        visibility_source = data[
            VISIBILITY_FIELD
        ]

        visibility = visibility_source[
            :,
            :,
            VISIBILITY_CHANNEL
        ]

        invalid_landmarks = (
            ~np.isfinite(
                visibility
            )
            |
            (
                visibility
                <
                VISIBILITY_THRESHOLD
            )
        )

        coords[
            invalid_landmarks
        ] = np.nan

    return (
        coords,
        valid_mask
    )


# ============================================================
# 4. 单个视频生成 Sliding Windows
# ============================================================

def generate_windows(
    npz_path
):
    """
    一个视频：

        [T, 33, 2]

    ↓

    多个：

        [30, 33, 2]

    ↓

    每个 flatten 成：

        1980维
    """

    samples = []
    labels = []
    groups = []
    metadata = []

    with np.load(
        npz_path,
        allow_pickle=False
    ) as data:

        label_name = str(
            data["label"].item()
        )

        video_id = str(
            data["video_id"].item()
        )

        if label_name not in LABEL_MAP:

            print(
                f"[跳过] 未知 label："
                f"{label_name}"
            )

            return (
                samples,
                labels,
                groups,
                metadata,
                None
            )

        label = LABEL_MAP[
            label_name
        ]

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        if "fps" in data.files:

            fps = float(
                data["fps"].item()
            )

        else:

            fps = EXPECTED_FPS

        if not np.isclose(
            fps,
            EXPECTED_FPS
        ):

            raise ValueError(
                f"{video_id} FPS={fps}，"
                f"与预期 {EXPECTED_FPS} 不一致"
            )

        window_size = int(
            round(
                fps
                * WINDOW_SECONDS
            )
        )

        coords, valid_mask = (
            prepare_coordinates(
                data
            )
        )

        num_frames = (
            coords.shape[0]
        )

        # ----------------------------------------------------
        # 帧号
        # ----------------------------------------------------

        if "frame_indices" in data.files:

            frame_indices = data[
                "frame_indices"
            ]

        else:

            frame_indices = np.arange(
                1,
                num_frames + 1
            )

        # ----------------------------------------------------
        # timestamp
        # ----------------------------------------------------

        if "timestamps" in data.files:

            timestamps = data[
                "timestamps"
            ]

        else:

            timestamps = (
                np.arange(
                    num_frames
                )
                / fps
            )

        # 视频不足一个窗口
        if num_frames < window_size:

            print(
                f"[跳过] {video_id}: "
                f"{num_frames} 帧 < "
                f"{window_size} 帧"
            )

            return (
                samples,
                labels,
                groups,
                metadata,
                window_size
            )

        # ====================================================
        # Sliding Window
        # ====================================================

        for start in range(
            0,
            num_frames - window_size + 1,
            WINDOW_STRIDE
        ):

            end = (
                start
                + window_size
            )

            # -----------------------------------------------
            # 至少要有一定数量有效 Pose 帧
            # -----------------------------------------------

            window_valid_mask = (
                valid_mask[
                    start:end
                ]
            )

            if (
                int(
                    window_valid_mask.sum()
                )
                <
                MIN_VALID_FRAMES
            ):

                continue

            # -----------------------------------------------
            # 当前窗口
            # -----------------------------------------------

            window = coords[
                start:end
            ].copy()

            # -----------------------------------------------
            # 缺失点处理
            # -----------------------------------------------

            window = (
                fill_missing_temporally(
                    window
                )
            )

            # -----------------------------------------------
            # Flatten
            #
            # [30, 33, 2]
            #
            # ->
            #
            # [1980]
            # -----------------------------------------------

            feature = window.reshape(
                -1
            )

            feature = np.nan_to_num(
                feature,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            ).astype(
                np.float32
            )

            samples.append(
                feature
            )

            labels.append(
                label
            )

            # 非常重要：
            # 所有来自同一视频的窗口
            # group 都相同
            groups.append(
                video_id
            )

            metadata.append({

                "video_id":
                    video_id,

                "label":
                    label_name,

                "window_start":
                    start,

                "window_end":
                    end - 1,

                "start_frame":
                    int(
                        frame_indices[
                            start
                        ]
                    ),

                "end_frame":
                    int(
                        frame_indices[
                            end - 1
                        ]
                    ),

                "start_time":
                    float(
                        timestamps[
                            start
                        ]
                    ),

                "end_time":
                    float(
                        timestamps[
                            end - 1
                        ]
                    ),
            })

    return (
        samples,
        labels,
        groups,
        metadata,
        window_size
    )


# ============================================================
# 5. 加载整个数据集
# ============================================================

def load_dataset():

    npz_files = sorted(
        DATA_ROOT.rglob(
            "*.npz"
        )
    )

    if not npz_files:

        raise RuntimeError(
            f"{DATA_ROOT} 中没有 npz"
        )

    print(
        f"发现 {len(npz_files)} 个视频"
    )

    X = []
    y = []
    groups = []
    metadata = []

    expected_window_size = None

    for npz_path in npz_files:

        (
            video_X,
            video_y,
            video_groups,
            video_metadata,
            window_size
        ) = generate_windows(
            npz_path
        )

        if window_size is not None:

            if expected_window_size is None:

                expected_window_size = (
                    window_size
                )

            elif (
                expected_window_size
                != window_size
            ):

                raise RuntimeError(
                    "不同视频 window_size 不一致"
                )

        X.extend(
            video_X
        )

        y.extend(
            video_y
        )

        groups.extend(
            video_groups
        )

        metadata.extend(
            video_metadata
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    groups = np.asarray(
        groups
    )

    if len(X) == 0:

        raise RuntimeError(
            "没有生成任何滑动窗口"
        )

    unique_videos = np.unique(
        groups
    )

    print()
    print(
        f"视频数量："
        f"{len(unique_videos)}"
    )

    print(
        f"窗口数量："
        f"{len(X)}"
    )

    print(
        f"Window size："
        f"{expected_window_size}"
    )

    print(
        f"Feature dimension："
        f"{X.shape[1]}"
    )

    print(
        f"ADL windows："
        f"{np.sum(y == 0)}"
    )

    print(
        f"Fall windows："
        f"{np.sum(y == 1)}"
    )

    return (
        X,
        y,
        groups,
        metadata,
        expected_window_size
    )


# ============================================================
# 6. RF
# ============================================================

def create_model(
    random_state
):

    return RandomForestClassifier(

        n_estimators=N_ESTIMATORS,

        max_features=MAX_FEATURES,

        max_depth=MAX_DEPTH,

        min_samples_split=(
            MIN_SAMPLES_SPLIT
        ),

        min_samples_leaf=(
            MIN_SAMPLES_LEAF
        ),

        class_weight=CLASS_WEIGHT,

        n_jobs=N_JOBS,

        random_state=random_state
    )


# ============================================================
# 7. 确定 Fold 数
#
# 注意：
# 这里统计的是视频数量，
# 不是窗口数量
# ============================================================

def determine_n_splits(
    y,
    groups
):

    unique_groups = np.unique(
        groups
    )

    group_labels = {}

    for group in unique_groups:

        labels = np.unique(
            y[
                groups == group
            ]
        )

        if len(labels) != 1:

            raise RuntimeError(
                f"视频 {group} "
                f"出现多个 label"
            )

        group_labels[
            group
        ] = int(
            labels[0]
        )

    group_class_counts = {
        label: 0
        for label in LABEL_MAP.values()
    }

    for label in (
        group_labels.values()
    ):

        group_class_counts[
            label
        ] += 1

    min_group_count = min(
        group_class_counts.values()
    )

    n_splits = min(
        MAX_FOLDS,
        min_group_count
    )

    if n_splits < 2:

        raise RuntimeError(
            "每类至少需要两个视频"
        )

    return n_splits


# ============================================================
# 8. 保存 Fold Metrics
# ============================================================

def save_fold_metrics(
    rows
):

    path = (
        RESULT_DIR
        / "fold_metrics.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "fold",
            "train_videos",
            "test_videos",
            "train_windows",
            "test_windows",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ])

        writer.writerows(
            rows
        )


# ============================================================
# 9. 保存 Window Prediction
# ============================================================

def save_predictions(
    metadata,
    y,
    pred,
    prob,
    fold_ids
):

    path = (
        RESULT_DIR
        / "window_predictions.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "video_id",
            "fold",
            "window_start",
            "window_end",
            "start_frame",
            "end_frame",
            "start_time",
            "end_time",
            "true_label",
            "predicted_label",
            "fall_probability",
            "correct",
        ])

        for i, info in enumerate(
            metadata
        ):

            writer.writerow([

                info[
                    "video_id"
                ],

                int(
                    fold_ids[i]
                ),

                info[
                    "window_start"
                ],

                info[
                    "window_end"
                ],

                info[
                    "start_frame"
                ],

                info[
                    "end_frame"
                ],

                f"{info['start_time']:.6f}",

                f"{info['end_time']:.6f}",

                ID_TO_LABEL[
                    int(y[i])
                ],

                ID_TO_LABEL[
                    int(pred[i])
                ],

                f"{prob[i]:.6f}",

                int(
                    y[i]
                    == pred[i]
                ),
            ])


# ============================================================
# 10. 保存 Feature Importance
# ============================================================

def save_feature_importance(
    model,
    window_size
):

    if not SAVE_FEATURE_IMPORTANCE:
        return

    importance = (
        model.feature_importances_
    )

    names = []

    for frame_offset in range(
        window_size
    ):

        for joint in range(33):

            for coordinate_name in (
                COORDINATE_NAMES
            ):

                names.append(
                    f"t{frame_offset:02d}_"
                    f"kp{joint:02d}_"
                    f"{coordinate_name}"
                )

    indices = np.argsort(
        importance
    )[::-1]

    path = (
        RESULT_DIR
        / "feature_importance.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "rank",
            "feature",
            "importance",
        ])

        for rank, index in enumerate(
            indices,
            start=1
        ):

            writer.writerow([
                rank,
                names[index],
                f"{importance[index]:.8f}",
            ])


# ============================================================
# 11. Cross Validation
# ============================================================

def cross_validate(
    X,
    y,
    groups,
    metadata
):

    n_splits = (
        determine_n_splits(
            y,
            groups
        )
    )

    print()
    print(
        f"{n_splits}-fold "
        f"StratifiedGroupKFold"
    )

    cv = StratifiedGroupKFold(

        n_splits=n_splits,

        shuffle=True,

        random_state=RANDOM_STATE
    )

    all_pred = np.zeros(
        len(y),
        dtype=np.int64
    )

    all_prob = np.zeros(
        len(y),
        dtype=np.float64
    )

    fold_ids = np.zeros(
        len(y),
        dtype=np.int32
    )

    fold_results = []

    # ========================================================
    # Fold
    # ========================================================

    for fold, (
        train_index,
        test_index
    ) in enumerate(
        cv.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        train_groups = set(
            groups[
                train_index
            ]
        )

        test_groups = set(
            groups[
                test_index
            ]
        )

        # ----------------------------------------------------
        # 防止视频泄漏
        # ----------------------------------------------------

        overlap = (
            train_groups
            & test_groups
        )

        if overlap:

            raise RuntimeError(
                "发现视频泄漏："
                f"{overlap}"
            )

        X_train = X[
            train_index
        ]

        y_train = y[
            train_index
        ]

        X_test = X[
            test_index
        ]

        y_test = y[
            test_index
        ]

        model = create_model(
            RANDOM_STATE + fold
        )

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        probability_matrix = (
            model.predict_proba(
                X_test
            )
        )

        positive_index = (
            list(
                model.classes_
            ).index(
                POSITIVE_LABEL
            )
        )

        probability = (
            probability_matrix[
                :,
                positive_index
            ]
        )

        all_pred[
            test_index
        ] = prediction

        all_prob[
            test_index
        ] = probability

        fold_ids[
            test_index
        ] = fold

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        precision = precision_score(
            y_test,
            prediction,
            pos_label=POSITIVE_LABEL,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            prediction,
            pos_label=POSITIVE_LABEL,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            prediction,
            pos_label=POSITIVE_LABEL,
            zero_division=0
        )

        try:

            auc = roc_auc_score(
                y_test,
                probability
            )

        except ValueError:

            auc = np.nan

        fold_results.append([

            fold,

            len(
                train_groups
            ),

            len(
                test_groups
            ),

            len(
                train_index
            ),

            len(
                test_index
            ),

            accuracy,
            precision,
            recall,
            f1,
            auc,
        ])

        print()
        print(
            f"Fold {fold}/{n_splits}"
        )

        print(
            f"Train videos : "
            f"{len(train_groups)}"
        )

        print(
            f"Test videos  : "
            f"{len(test_groups)}"
        )

        print(
            f"Train windows: "
            f"{len(train_index)}"
        )

        print(
            f"Test windows : "
            f"{len(test_index)}"
        )

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1       : {f1:.4f}"
        )

        print(
            f"ROC-AUC  : {auc:.4f}"
        )

    # ========================================================
    # Overall
    # ========================================================

    accuracy = accuracy_score(
        y,
        all_pred
    )

    precision = precision_score(
        y,
        all_pred,
        pos_label=POSITIVE_LABEL,
        zero_division=0
    )

    recall = recall_score(
        y,
        all_pred,
        pos_label=POSITIVE_LABEL,
        zero_division=0
    )

    f1 = f1_score(
        y,
        all_pred,
        pos_label=POSITIVE_LABEL,
        zero_division=0
    )

    auc = roc_auc_score(
        y,
        all_prob
    )

    cm = confusion_matrix(
        y,
        all_pred
    )

    report = classification_report(
        y,
        all_pred,
        target_names=[
            "ADL",
            "Fall",
        ],
        digits=4
    )

    print()
    print(
        "=" * 60
    )

    print(
        "Sliding Window RF Overall"
    )

    print(
        "=" * 60
    )

    print()
    print(report)

    print(
        "Confusion Matrix:"
    )

    print(cm)

    print()

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )

    # ========================================================
    # 保存
    # ========================================================

    save_fold_metrics(
        fold_results
    )

    save_predictions(
        metadata,
        y,
        all_pred,
        all_prob,
        fold_ids
    )

    # confusion matrix
    np.savetxt(
        RESULT_DIR
        / "confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d"
    )

    # metrics
    with open(
        RESULT_DIR
        / "metrics.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Experiment: "
            f"{EXPERIMENT_NAME}\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"Videos: "
            f"{len(np.unique(groups))}\n"
        )

        f.write(
            f"Windows: "
            f"{len(X)}\n"
        )

        f.write(
            f"Features: "
            f"{X.shape[1]}\n"
        )

        f.write(
            f"Window seconds: "
            f"{WINDOW_SECONDS}\n"
        )

        f.write(
            f"Window stride: "
            f"{WINDOW_STRIDE}\n"
        )

        f.write(
            f"Folds: "
            f"{n_splits}\n\n"
        )

        # ============================================================
        # 每一折结果
        # ============================================================

        f.write(
            "Per-Fold Results\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        for row in fold_results:
        
            (
                fold,
                train_videos,
                test_videos,
                train_windows,
                test_windows,
                fold_accuracy,
                fold_precision,
                fold_recall,
                fold_f1,
                fold_auc,
            ) = row

            f.write(
                f"\nFold {fold}/{n_splits}\n"
            )

            f.write(
                f"Train videos : "
                f"{train_videos}\n"
            )

            f.write(
                f"Test videos  : "
                f"{test_videos}\n"
            )

            f.write(
                f"Train windows: "
                f"{train_windows}\n"
            )

            f.write(
                f"Test windows : "
                f"{test_windows}\n"
            )

            f.write(
                f"Accuracy : "
                f"{fold_accuracy:.4f}\n"
            )

            f.write(
                f"Precision: "
                f"{fold_precision:.4f}\n"
            )

            f.write(
                f"Recall   : "
                f"{fold_recall:.4f}\n"
            )

            f.write(
                f"F1       : "
                f"{fold_f1:.4f}\n"
            )

            f.write(
                f"ROC-AUC  : "
                f"{fold_auc:.4f}\n"
            )

        f.write(
            "\n"
            + "=" * 60
            + "\n"
        )

        f.write(
            "Overall Results\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        f.write(
            f"Accuracy : "
            f"{accuracy:.4f}\n"
        )

        f.write(
            f"Precision: "
            f"{precision:.4f}\n"
        )

        f.write(
            f"Recall   : "
            f"{recall:.4f}\n"
        )

        f.write(
            f"F1       : "
            f"{f1:.4f}\n"
        )

        f.write(
            f"ROC-AUC  : "
            f"{auc:.4f}\n\n"
        )

        f.write(
            "Classification Report\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        f.write(
            report
        )

        f.write(
            "\nConfusion Matrix\n"
        )

        f.write(
            str(cm)
        )

    return (
        n_splits,
        all_pred,
        all_prob
    )


# ============================================================
# 12. 保存 Config
# ============================================================

def save_config(
    X,
    groups,
    window_size,
    n_splits
):

    config = {

        "experiment_name":
            EXPERIMENT_NAME,

        "data_root":
            str(DATA_ROOT),

        "result_dir":
            str(RESULT_DIR),

        "normalized_input":
            bool(
                "normalized"
                in DATA_ROOT.resolve().name.lower()
            ),

        "coordinate_field":
            COORDINATE_FIELD,

        "coordinate_channels":
            list(
                COORDINATE_CHANNELS
            ),

        "visibility_filter":
            USE_VISIBILITY_FILTER,

        "visibility_threshold":
            VISIBILITY_THRESHOLD,

        "fps":
            EXPECTED_FPS,

        "window_seconds":
            WINDOW_SECONDS,

        "window_size_frames":
            window_size,

        "window_stride":
            WINDOW_STRIDE,

        "missing_value_strategy":
            MISSING_VALUE_STRATEGY,

        "number_of_videos":
            int(
                len(
                    np.unique(
                        groups
                    )
                )
            ),

        "number_of_windows":
            int(
                len(X)
            ),

        "feature_dimension":
            int(
                X.shape[1]
            ),

        "folds":
            n_splits,

        "random_state":
            RANDOM_STATE,

        "n_estimators":
            N_ESTIMATORS,

        "max_features":
            MAX_FEATURES,

        "class_weight":
            CLASS_WEIGHT,

        "n_jobs":
            N_JOBS,
    }

    with open(
        RESULT_DIR
        / "config.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# 13. 最终模型
# ============================================================

def train_final_model(
    X,
    y,
    window_size
):

    print()
    print(
        "使用全部窗口训练最终 RF..."
    )

    model = create_model(
        RANDOM_STATE
    )

    model.fit(
        X,
        y
    )

    if SAVE_FINAL_MODEL:

        path = (
            RESULT_DIR
            / f"{EXPERIMENT_NAME}.joblib"
        )

        joblib.dump(
            model,
            path
        )

        print(
            f"模型保存：{path}"
        )

    save_feature_importance(
        model,
        window_size
    )


# ============================================================
# 14. Main
# ============================================================

def main():

    args = parse_args()
    configure_experiment(args)

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Data root: {DATA_ROOT}"
    )

    print(
        f"Experiment: {EXPERIMENT_NAME}"
    )

    print(
        f"Output dir: {RESULT_DIR}"
    )

    (
        X,
        y,
        groups,
        metadata,
        window_size
    ) = load_dataset()

    (
        n_splits,
        _,
        _
    ) = cross_validate(
        X,
        y,
        groups,
        metadata
    )

    save_config(
        X,
        groups,
        window_size,
        n_splits
    )

    train_final_model(
        X,
        y,
        window_size
    )

    print()
    print(
        "=" * 60
    )

    print(
        f"实验完成："
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"结果目录："
        f"{RESULT_DIR}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()