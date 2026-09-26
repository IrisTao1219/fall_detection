from pathlib import Path
import argparse
import csv
import json
import re

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

try:
    from . import common as experiment_common
    from .common import (
        canonical_sequence_name as common_canonical_sequence_name,
        classification_report_text,
        compute_metrics,
        get_window_label_from_urfall as common_get_window_label_from_urfall,
        fill_missing_temporally as common_fill_missing_temporally,
        interpolate_1d,
        load_fall_frame_labels as common_load_fall_frame_labels,
        preprocess_keypoints as common_preprocess_keypoints,
        save_confusion_matrix_csv,
        video_type_labels,
    )
except ImportError:
    import common as experiment_common
    from common import (
        canonical_sequence_name as common_canonical_sequence_name,
        classification_report_text,
        compute_metrics,
        get_window_label_from_urfall as common_get_window_label_from_urfall,
        fill_missing_temporally as common_fill_missing_temporally,
        interpolate_1d,
        load_fall_frame_labels as common_load_fall_frame_labels,
        preprocess_keypoints as common_preprocess_keypoints,
        save_confusion_matrix_csv,
        video_type_labels,
    )


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
WINDOWS_CACHE = None

# UR-Fall 官方逐帧姿态标注（只包含 fall 序列）
# CSV 前三列：sequence name, frame number, label
# label: -1=not lying, 0=falling transition, 1=lying
FALL_ANNOTATION_CSV = Path("data/urfall-cam0-falls.csv")

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
        "--windows-cache",
        type=Path,
        required=True,
        help="prepare_windows.py 生成的统一窗口缓存",
    )

    parser.add_argument(
        "--output-root",
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
    global WINDOWS_CACHE
    global RESULT_ROOT
    global RESULT_DIR
    global EXPERIMENT_NAME

    DATA_ROOT = args.data_root
    WINDOWS_CACHE = args.windows_cache
    RESULT_ROOT = args.output_root

    dataset_name = DATA_ROOT.resolve().name.lower()

    if "normalized" in dataset_name:
        EXPERIMENT_NAME = "rf_normalized"
    else:
        EXPERIMENT_NAME = "rf"

    RESULT_DIR = RESULT_ROOT / EXPERIMENT_NAME


# ============================================================
# UR-Fall 逐帧标签
# ============================================================

def canonical_sequence_name(value):
    """
    把各种可能的视频 ID / 文件名统一成：
        fall-01
        adl-01

    例如：
        fall-01-cam0-d -> fall-01
        fall-01-cam0-rgb -> fall-01
        fall-01 -> fall-01
    """
    return common_canonical_sequence_name(value)


def load_fall_frame_labels(csv_path):
    """
    读取 UR-Fall 官方 urfall-cam0-falls.csv。

    返回：
        {
            "fall-01": {
                1: -1,
                2: -1,
                ...
            },
            ...
        }

    官方 CSV 没有必须依赖的表头，因此这里只读取每行前三列：
        sequence name, frame number, label

    同时兼容逗号、分号、Tab 或空白分隔。
    """
    return common_load_fall_frame_labels(csv_path)


def get_window_label_from_urfall(frame_labels):
    """
    根据 UR-Fall 官方逐帧 posture label 生成二分类窗口标签。

    官方定义：
        -1 = person is not lying
         0 = temporary pose / falling
         1 = person is lying on the ground

    UR-Fall 原始分类不使用 0 帧，因此这里：
        - 忽略 0
        - -1 映射为 ADL / normal (0)
        -  1 映射为 Fall (1)
        - 剩余有效帧多数投票
        - 没有有效帧或恰好平票时，返回 None，跳过该窗口
    """
    return common_get_window_label_from_urfall(frame_labels)


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

    mode = "interp" if MISSING_VALUE_STRATEGY == "interpolate" else MISSING_VALUE_STRATEGY
    return common_fill_missing_temporally(window, mode)


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

    keypoints = data[COORDINATE_FIELD]
    valid_mask = data[VALID_MASK_FIELD].astype(bool)
    visibility_threshold = VISIBILITY_THRESHOLD if USE_VISIBILITY_FILTER else -np.inf
    coords = common_preprocess_keypoints(
        keypoints=keypoints,
        valid_mask=valid_mask,
        visibility_threshold=visibility_threshold,
        missing_mode="mask",
    )

    return (
        coords,
        valid_mask
    )


# ============================================================
# 4. 单个视频生成 Sliding Windows
# ============================================================


def load_dataset():
    x_seq, y, groups, records, cache_config = experiment_common.load_windows_cache(
        WINDOWS_CACHE
    )
    expected_window_size = int(x_seq.shape[1])
    X = x_seq.reshape(len(x_seq), -1).astype(np.float32)
    metadata = []
    for index, row in records.reset_index(drop=True).iterrows():
        metadata.append({
            "video_id": row.get("video_id", groups[index]),
            "window_start": row.get("window_start", index),
            "window_end": row.get("window_end", index),
            "start_frame": int(row.get("start_frame", 0)),
            "end_frame": int(row.get("end_frame", 0)),
            "start_time": float(row.get("start_time", 0.0)),
            "end_time": float(row.get("end_time", 0.0)),
        })

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
    """
    窗口级标注后，同一个 fall 视频里可以同时存在：
        normal windows (0)
        fall windows   (1)

    因此不能再要求“一个视频只有一个 label”。

    这里统计每个类别至少出现在多少个独立视频中，
    再据此决定 StratifiedGroupKFold 的最大折数。
    """
    unique_groups = np.unique(
        groups
    )

    group_class_counts = {
        label: 0
        for label in LABEL_MAP.values()
    }

    for group in unique_groups:

        group_y = y[
            groups == group
        ]

        for label in group_class_counts:

            if np.any(
                group_y == label
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
            "每个类别至少需要出现在两个独立视频中。"
            f" 当前统计：{group_class_counts}"
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

        fold_metrics = compute_metrics(y_test, prediction, probability)
        accuracy = fold_metrics["accuracy"]
        precision = fold_metrics["precision"]
        recall = fold_metrics["recall"]
        f1 = fold_metrics["f1"]
        auc = fold_metrics["roc_auc"]

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

    overall_metrics = compute_metrics(y, all_pred, all_prob)
    accuracy = overall_metrics["accuracy"]
    precision = overall_metrics["precision"]
    recall = overall_metrics["recall"]
    f1 = overall_metrics["f1"]
    auc = overall_metrics["roc_auc"]
    cm = save_confusion_matrix_csv(RESULT_DIR, y, all_pred)
    report = classification_report_text(y, all_pred)

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

    fold_dicts = [
        {
            "fold": int(row[0]),
            "train_videos": int(row[1]),
            "test_videos": int(row[2]),
            "train_windows": int(row[3]),
            "test_windows": int(row[4]),
            "accuracy": float(row[5]),
            "precision": float(row[6]),
            "recall": float(row[7]),
            "f1": float(row[8]),
            "roc_auc": float(row[9]),
        }
        for row in fold_results
    ]
    experiment_common.save_experiment_metrics_text(
        output=RESULT_DIR,
        model="rf",
        header=[
            f"data_root: {DATA_ROOT}",
            f"experiment: {EXPERIMENT_NAME}",
            f"videos: {len(np.unique(groups))}",
            f"windows: {len(X)}",
            f"features: {X.shape[1]}",
            f"window_seconds: {WINDOW_SECONDS}",
            f"window_stride: {WINDOW_STRIDE}",
            f"folds: {n_splits}",
        ],
        fold_rows=fold_dicts,
        overall=overall_metrics,
        y_true=y,
        y_pred=all_pred,
        confusion=cm,
        metric_keys=("accuracy", "precision", "recall", "f1", "roc_auc"),
    )
    experiment_common.save_experiment_metrics_json(
        output=RESULT_DIR,
        model="rf",
        data_root=DATA_ROOT,
        overall=overall_metrics,
        fold_rows=fold_dicts,
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

        "fall_annotation_csv":
            str(FALL_ANNOTATION_CSV),

        "window_label_strategy":
            "UR-Fall frame labels: ignore 0, majority vote -1 vs 1",

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
