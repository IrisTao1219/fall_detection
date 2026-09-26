#!/usr/bin/env python3
"""汇总打印项目中已有实验结果。

默认扫描 results、results_normalized、results-old 三个目录，优先读取：
  1. seed_metrics.csv：Transformer 这类多 seed 实验
  2. metrics.json：带均值/标准差的实验摘要
  3. metrics.txt：解析脚本保存的总体指标
  4. predictions.csv：没有汇总文件时，才由折外预测重新计算总体指标

示例：
    python src/print.py
    python src/print.py --format markdown
    python src/print.py --format markdown --output results_summary.md
    python src/print.py --roots results results_normalized --sort-by f1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
COUNT_COLUMNS = ["tn", "fp", "fn", "tp"]
DISPLAY_COLUMNS = [
    "rank",
    "result",
    "scope",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "tp",
    "fp",
    "fn",
    "tn",
    "source",
]


def safe_float(value: Any) -> float:
    """把字符串、空值和 NaN 统一转换成 float。"""
    if value is None:
        return float("nan")
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def safe_roc_auc(y_true: List[int], y_prob: List[float]) -> float:
    """用秩统计公式计算 ROC-AUC。"""
    positives = sum(1 for value in y_true if value == 1)
    negatives = sum(1 for value in y_true if value == 0)
    if positives == 0 or negatives == 0:
        return float("nan")

    pairs = sorted(zip(y_prob, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(1 for _, label in pairs[index:end] if label == 1)
        index = end

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, Any]:
    """从预测数组计算二分类指标。"""
    tn = fp = fn = tp = 0
    for true_value, pred_value in zip(y_true, y_pred):
        if true_value == 0 and pred_value == 0:
            tn += 1
        elif true_value == 0 and pred_value == 1:
            fp += 1
        elif true_value == 1 and pred_value == 0:
            fn += 1
        elif true_value == 1 and pred_value == 1:
            tp += 1

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def result_name(path: Path, project_root: Path) -> str:
    """把结果目录转换成易读的相对名称。"""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def metric_row(
    result_dir: Path,
    project_root: Path,
    scope: str,
    metrics: Dict[str, Any],
    source: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """构造统一的结果行。"""
    row: Dict[str, Any] = {
        "result": result_name(result_dir, project_root),
        "scope": scope,
        "source": source,
    }
    if seed is not None:
        row["seed"] = seed
    for key in METRIC_COLUMNS + COUNT_COLUMNS:
        row[key] = metrics.get(key, float("nan"))
    return row


def read_seed_metrics(result_dir: Path, project_root: Path) -> List[Dict[str, Any]]:
    """读取 Transformer 风格的 seed_metrics.csv。"""
    path = result_dir / "seed_metrics.csv"
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            seed = int(float(raw["seed"])) if raw.get("seed") else None
            window_metrics = {
                key: safe_float(raw.get(f"window_{key}"))
                for key in METRIC_COLUMNS + COUNT_COLUMNS
            }
            rows.append(
                metric_row(
                    result_dir,
                    project_root,
                    "window",
                    window_metrics,
                    "seed_metrics.csv",
                    seed,
                )
            )

            if any(f"video_{key}" in raw for key in METRIC_COLUMNS):
                video_metrics = {
                    key: safe_float(raw.get(f"video_{key}"))
                    for key in METRIC_COLUMNS + COUNT_COLUMNS
                }
                rows.append(
                    metric_row(
                        result_dir,
                        project_root,
                        "video",
                        video_metrics,
                        "seed_metrics.csv",
                        seed,
                    )
                )
    return rows


def read_metrics_json(result_dir: Path, project_root: Path) -> List[Dict[str, Any]]:
    """读取 metrics.json 中的 seed_metric_mean。"""
    path = result_dir / "metrics.json"
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    mean_metrics = data.get("seed_metric_mean")
    if not isinstance(mean_metrics, dict):
        return []

    rows = []
    for prefix, scope in (("window_", "window_mean"), ("video_", "video_mean")):
        if not any(key.startswith(prefix) for key in mean_metrics):
            continue
        metrics = {
            key: safe_float(mean_metrics.get(f"{prefix}{key}"))
            for key in METRIC_COLUMNS + COUNT_COLUMNS
        }
        rows.append(metric_row(result_dir, project_root, scope, metrics, "metrics.json"))
    return rows


def read_predictions(result_dir: Path, project_root: Path) -> List[Dict[str, Any]]:
    """优先从 predictions.csv 重新计算总体指标。"""
    path = result_dir / "predictions.csv"
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []

    fieldnames = set(rows[0].keys())
    label_column = "label" if "label" in fieldnames else "y_true"
    required = {label_column, "y_pred", "fall_probability"}
    if not required.issubset(fieldnames):
        return []

    y_true = [int(float(row[label_column])) for row in rows]
    y_pred = [int(float(row["y_pred"])) for row in rows]
    y_prob = [safe_float(row["fall_probability"]) for row in rows]
    metrics = compute_metrics(y_true, y_pred, y_prob)
    return [metric_row(result_dir, project_root, "window", metrics, "predictions.csv")]


def parse_metrics_block(text: str, heading_pattern: str) -> Dict[str, Any]:
    """从 metrics.txt 的指定块中解析 key: value 指标。"""
    match = re.search(heading_pattern, text, flags=re.IGNORECASE)
    if not match:
        return {}

    metrics: Dict[str, Any] = {}
    started = False
    for line in text[match.end() :].splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) == {"="}:
            if not started:
                continue
            if metrics:
                break
            continue
        if re.match(
            r"Classification report:|Confusion matrix|Fold mean|VIDEO METRICS|WINDOW METRICS|\[Fold",
            stripped,
            flags=re.IGNORECASE,
        ):
            break
        if ":" not in line:
            if started and metrics:
                break
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in METRIC_COLUMNS + COUNT_COLUMNS:
            started = True
            metrics[key] = safe_float(value)
    return metrics


def read_metrics_txt(result_dir: Path, project_root: Path) -> List[Dict[str, Any]]:
    """兜底读取 metrics.txt 中的总体指标。"""
    path = result_dir / "metrics.txt"
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    candidates = [
        ("window", r"WINDOW METRICS\s*\n"),
        ("overall", r"OVERALL OUT-OF-FOLD METRICS\s*\n"),
    ]

    rows = []
    for scope, pattern in candidates:
        metrics = parse_metrics_block(text, pattern)
        if metrics:
            rows.append(metric_row(result_dir, project_root, scope, metrics, "metrics.txt"))
            break
    return rows


def collect_result_dirs(roots: Iterable[Path]) -> List[Path]:
    """找出包含结果文件的目录。"""
    result_dirs = set()
    filenames = {
        "seed_metrics.csv",
        "metrics.json",
        "predictions.csv",
        "metrics.txt",
        "fold_metrics.csv",
    }
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name in filenames:
                result_dirs.add(path.parent)
    return sorted(result_dirs)


def read_result_dir(result_dir: Path, project_root: Path) -> List[Dict[str, Any]]:
    """按优先级读取单个结果目录。"""
    for reader in (read_seed_metrics, read_metrics_json, read_metrics_txt, read_predictions):
        rows = reader(result_dir, project_root)
        if rows:
            return rows
    return []


def collect_results(roots: Iterable[Path], project_root: Path) -> List[Dict[str, Any]]:
    """汇总所有结果目录。"""
    rows: List[Dict[str, Any]] = []
    for result_dir in collect_result_dirs(roots):
        rows.extend(read_result_dir(result_dir, project_root))
    return rows


def format_number(value: Any) -> str:
    """表格显示用的数值格式。"""
    number = safe_float(value)
    if math.isnan(number):
        return ""
    if abs(number - round(number)) < 1e-9 and abs(number) >= 1:
        return str(int(round(number)))
    return f"{number:.4f}"


def prepare_display(rows: List[Dict[str, Any]], sort_by: str, scope: str) -> List[Dict[str, Any]]:
    """排序、过滤并补充 rank。"""
    if scope != "all":
        rows = [
            row for row in rows
            if scope.lower() in str(row.get("scope", "")).lower()
        ]
    rows = sorted(
        rows,
        key=lambda row: safe_float(row.get(sort_by)),
        reverse=True,
    )
    display_rows = []
    for rank, row in enumerate(rows, start=1):
        copied = dict(row)
        copied["rank"] = rank
        display_rows.append(copied)
    return display_rows


def stringify_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """把结果行转换成表格字符串。"""
    output = []
    for row in rows:
        copied = {column: str(row.get(column, "")) for column in DISPLAY_COLUMNS}
        for key in METRIC_COLUMNS + COUNT_COLUMNS:
            copied[key] = format_number(row.get(key))
        output.append(copied)
    return output


def print_text_table(rows: List[Dict[str, Any]]) -> None:
    """打印适合终端阅读的固定宽度表格。"""
    if not rows:
        print("没有找到可汇总的实验结果。")
        return

    output = stringify_rows(rows)
    widths = {
        column: max(len(column), *(len(row[column]) for row in output))
        for column in DISPLAY_COLUMNS
    }
    print(" ".join(column.ljust(widths[column]) for column in DISPLAY_COLUMNS))
    print(" ".join("-" * widths[column] for column in DISPLAY_COLUMNS))
    for row in output:
        print(" ".join(row[column].ljust(widths[column]) for column in DISPLAY_COLUMNS))


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    """生成 Markdown 表格。"""
    if not rows:
        return "没有找到可汇总的实验结果。\n"

    output = stringify_rows(rows)
    lines = [
        "| " + " | ".join(DISPLAY_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in DISPLAY_COLUMNS) + " |",
    ]
    for row in output:
        lines.append("| " + " | ".join(row[column] for column in DISPLAY_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def print_markdown_table(rows: List[Dict[str, Any]]) -> None:
    """打印 Markdown 表格。"""
    print(markdown_table(rows), end="")


def print_csv_table(rows: List[Dict[str, Any]]) -> None:
    """打印 CSV，便于重定向到文件。"""
    output = stringify_rows(rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=DISPLAY_COLUMNS)
    writer.writeheader()
    writer.writerows(output)


def print_best_by_root(rows: List[Dict[str, Any]], sort_by: str) -> None:
    """按 results/results_normalized/results-old 分组打印最佳窗口级结果。"""
    if not rows:
        return
    window_rows = [
        row for row in rows
        if re.search("window|overall", str(row.get("scope", "")), flags=re.IGNORECASE)
    ]
    if not window_rows:
        return

    print("\n各结果根目录最佳结果：")
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in window_rows:
        root_name = str(row["result"]).split("/")[0]
        grouped.setdefault(root_name, []).append(row)

    best_rows = []
    for root_name, group in grouped.items():
        best = max(group, key=lambda row: safe_float(row.get(sort_by)))
        best_rows.append(
            {
                "root": root_name,
                "result": best["result"],
                "accuracy": format_number(best["accuracy"]),
                "precision": format_number(best["precision"]),
                "recall": format_number(best["recall"]),
                "f1": format_number(best["f1"]),
                "roc_auc": format_number(best["roc_auc"]),
            }
        )
    best_rows = sorted(best_rows, key=lambda row: safe_float(row[sort_by]), reverse=True)
    columns = ["root", "result"] + METRIC_COLUMNS
    widths = {column: max(len(column), *(len(str(row[column])) for row in best_rows)) for column in columns}
    print(" ".join(column.ljust(widths[column]) for column in columns))
    print(" ".join("-" * widths[column] for column in columns))
    for row in best_rows:
        print(" ".join(str(row[column]).ljust(widths[column]) for column in columns))


def best_by_root_markdown(rows: List[Dict[str, Any]], sort_by: str) -> str:
    """生成各结果根目录最佳结果的 Markdown 表格。"""
    if not rows:
        return ""
    window_rows = [
        row for row in rows
        if re.search("window|overall", str(row.get("scope", "")), flags=re.IGNORECASE)
    ]
    if not window_rows:
        return ""

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in window_rows:
        root_name = str(row["result"]).split("/")[0]
        grouped.setdefault(root_name, []).append(row)

    best_rows = []
    for root_name, group in grouped.items():
        best = max(group, key=lambda row: safe_float(row.get(sort_by)))
        best_rows.append(
            {
                "root": root_name,
                "result": best["result"],
                "accuracy": format_number(best["accuracy"]),
                "precision": format_number(best["precision"]),
                "recall": format_number(best["recall"]),
                "f1": format_number(best["f1"]),
                "roc_auc": format_number(best["roc_auc"]),
            }
        )
    best_rows = sorted(best_rows, key=lambda row: safe_float(row[sort_by]), reverse=True)
    columns = ["root", "result"] + METRIC_COLUMNS
    lines = [
        "## 各结果根目录最佳结果",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in best_rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="打印 fall_detection 实验结果汇总")
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=[Path("results"), Path("results_normalized"), Path("results-old")],
        help="要扫描的结果根目录",
    )
    parser.add_argument(
        "--sort-by",
        choices=METRIC_COLUMNS,
        default="f1",
        help="结果排序指标",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "window", "video", "overall"),
        default="all",
        help="打印哪些粒度的结果",
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "csv"),
        default="text",
        help="输出格式",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="把结果写入文件。markdown 格式未指定时默认写 results_summary.md",
    )
    parser.add_argument(
        "--no-best",
        action="store_true",
        help="不打印各结果根目录最佳结果",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path.cwd()
    frame = collect_results(args.roots, project_root)
    display = prepare_display(frame, args.sort_by, args.scope)

    if args.format == "markdown":
        content_lines = [
            "# Fall Detection 实验结果汇总",
            "",
            f"- 排序指标：`{args.sort_by}`",
            f"- 结果粒度：`{args.scope}`",
            "",
            "## 全部结果",
            "",
            markdown_table(display).rstrip(),
            "",
        ]
        if not args.no_best:
            best_section = best_by_root_markdown(frame, args.sort_by).rstrip()
            if best_section:
                content_lines.extend([best_section, ""])
        content = "\n".join(content_lines).rstrip() + "\n"
        output_path = args.output or Path("results_summary.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print(f"Saved Markdown summary: {output_path}")
    elif args.format == "csv":
        print_csv_table(display)
    else:
        print_text_table(display)

    if not args.no_best and args.format == "text":
        print_best_by_root(frame, args.sort_by)


if __name__ == "__main__":
    main()
