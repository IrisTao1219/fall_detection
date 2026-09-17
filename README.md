# 基于人体姿态估计与多类型分类模型的视频跌倒检测

本课程项目用普通摄像头视频完成跌倒（`fall`）与日常活动（`adl`）二分类：先用 MediaPipe BlazePose 提取 33 个人体关键点，再比较不同分类模型对连续姿态的识别能力。研究目标与实验设计见上级目录的[《实验方案》](../实验方案.md)。

当前实验重点比较使用骨架关键点时，不同分类器对跌倒视频的识别效果。RF 和 LSTM 的结果已保存；MLP 程序已完成，尚待运行。后续实验按[课程报告](../基于人体姿态估计的跌倒检测课程报告.docx)第 4.7 节的复现路线逐步开展。

**当前进度：**已实现 URFD 图像帧的姿态提取、坐标归一化，并保存 RF 和 LSTM 各两组（原始坐标、归一化坐标）实验结果。MLP 实验程序已写好，尚无运行结果。报告提出的窗口长度、统一预处理对照及 ST-GCN/Transformer 扩展尚未完成。

## 环境准备

- Python 3.11（`pyproject.toml` 要求 Python ≥3.11，仓库的 `.python-version` 为 3.11）
- 推荐使用 uv 管理环境；依赖定义见 `pyproject.toml`

在项目根目录执行：

```bash
uv sync
```

下文命令均从项目根目录运行。若不使用 uv，也可以用 Python 3.11 创建虚拟环境并安装 `pyproject.toml` 中的依赖。

## 数据准备

当前代码针对 URFD 数据集的 `cam0-rgb` 图像帧，按 **30 FPS** 处理。将解压后的 PNG 帧放到以下目录；每段视频的帧可以直接位于视频目录，也可以再嵌套一层同名目录：

```text
data/raw/
├── adl/
│   └── adl-01-cam0-rgb/
│       └── adl-01-cam0-rgb-001.png
└── fall/
    └── fall-01-cam0-rgb/
        └── fall-01-cam0-rgb-001.png
```

帧文件名末尾须带数字帧号，例如 `-001.png`。类别由 `adl`、`fall` 两级目录名决定。原始数据与生成的关键点文件未纳入仓库。

`data/download_urfd_adl.sh` 和 `data/download_urfd_falls.sh` 是下载脚本草稿
（完成之后改了下文件目录结构，脚本可能路径有点问题需要自己修正）

## 运行实验

### 1. 提取姿态关键点

```bash
uv run python src/blazepose.py
```

脚本对每张图片独立运行 BlazePose Heavy，输出 `data/keypoints/{adl,fall}/<video_id>.npz`，并在 `data/pose/visualization/` 保存少量抽样图片。每个 NPZ 包含：

| 字段 | 含义 |
| --- | --- |
| `keypoints` | `[帧数, 33, 4]`；最后一维为 `x, y, z, visibility`，未检测到人体的帧为 NaN |
| `valid_mask` | 每帧是否检测到人体 |
| `frame_indices`, `timestamps`, `fps` | 原始帧号、秒级时间戳和帧率 |
| `label`, `video_id` | 视频类别与编号 |

### 2. 归一化坐标

```bash
uv run python src/normalize_keypoints.py
```

脚本读取 `data/keypoints/`，将髋部中心作为坐标原点、肩宽作为优先尺度，输出同结构文件至 `data/keypoints_normalized/`。无法可靠归一化的帧会标记为无效。

### 3. 训练并评估随机森林

```bash
uv run python src/experiment_rf.py --data-root data/keypoints
uv run python src/experiment_rf.py --data-root data/keypoints_normalized
```

脚本依据 `--data-root` 分别写入 `results/rf/` 和 `results/rf_normalized/`。可用 `--result-root` 修改结果根目录；其他实验参数在脚本顶部配置。

每个窗口包含 30 帧（1 秒），步长为 1 帧；使用 33 个关键点的 `x/y` 坐标，展开后为 1980 维。低可见度或缺失坐标在窗口内按时间插值。分类标签为 `adl=0`、`fall=1`。评估采用按视频分组的分层交叉验证，同一视频的窗口不会同时进入训练集和测试集。交叉验证完成后，脚本再用全部窗口训练并保存最终模型。

结果目录包含 `config.json`、`metrics.txt`、`fold_metrics.csv`、`confusion_matrix.csv`、`window_predictions.csv`、`feature_importance.csv` 和最终模型 `.joblib`。`metrics.txt` 中的总体指标来自各折测试窗口的汇总预测；它们是**窗口级**指标，不是独立视频或实时场景的检测指标。

### 4. 训练并评估 LSTM

```bash
uv run python src/experiment_lstm.py --data-root data/keypoints
uv run python src/experiment_lstm.py --data-root data/keypoints_normalized
```

两组结果分别写入 `results/lstm/` 和 `results/lstm_normalized/`。已保存的配置使用 33 个关节的 `x/y`（每帧 66 维）、30 帧窗口、1 帧步长、低可见度或缺失坐标填零、10 层单向 LSTM（隐藏维度 80）、30 个训练轮次。按视频分组做 5 折评估，每折仅用训练视频拟合标准化参数。目前保存的结果包含 `config.json`、`metrics.txt`、`fold_metrics.csv`、`confusion_matrix.csv`、`predictions.csv` 和每折训练损失历史；脚本运行时也会生成模型检查点，但当前结果目录中没有这些权重文件。

### 5. MLP 实验程序（尚未运行）

`src/experiment_mlp.py` 可作为将连续 30 帧关键点展平后分类的神经网络基线：

```bash
uv run python src/experiment_mlp.py --data-root data/keypoints_normalized
```

原始坐标实验可把数据目录改为 `data/keypoints`。默认使用全部 33 个关节的 `x/y`、30 帧窗口、1 帧步长、缺失坐标时间插值和两层 MLP（256、64 个隐藏单元）。五折按视频分组，训练折内再按视频划分验证集用于早停；标准化参数仅由内部训练视频计算。输出分别保存在 `results/mlp_normalized/` 或 `results/mlp/`，包含窗口级和视频级折外预测、指标、训练曲线及各折模型。视频级结果按同一视频各窗口的跌倒概率平均后计算。

尚未发现 `results/mlp/` 或 `results/mlp_normalized/`，所以没有可报告的 MLP 实测指标。

## 已完成的实验与结果

下表取自各结果目录的 `metrics.txt`，均为 **5 折折外窗口预测汇总指标**，其中 `fall` 为正类。四组实验都使用 70 段视频、30 帧窗口、1 帧步长和按视频分组的交叉验证；表中数值不是视频级指标。

| 模型 | 输入 | 窗口数 | 准确率 | 跌倒精确率 | 跌倒召回率 | 跌倒 F1 | ROC-AUC | 结果目录 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RF | 原始 `x/y` | 8993 | 0.9158 | 0.8327 | 0.8014 | 0.8168 | 0.9507 | [`results/rf/`](results/rf/) |
| RF | 归一化 `x/y` | 8993 | 0.8589 | 0.7419 | 0.6090 | 0.6689 | 0.9093 | [`results/rf_normalized/`](results/rf_normalized/) |
| LSTM | 原始 `x/y` | 9906 | 0.9232 | 0.8667 | 0.7586 | 0.8090 | 0.9238 | [`results/lstm/`](results/lstm/) |
| LSTM | 归一化 `x/y` | 9906 | 0.8913 | 0.7507 | 0.7384 | 0.7445 | 0.8731 | [`results/lstm_normalized/`](results/lstm_normalized/) |

RF 使用窗口内时间插值，并过滤质量不足的窗口；LSTM 对缺失坐标填零，保留了更多窗口。因此 RF 与 LSTM 的样本集合不同，表中的模型间差值只能作为初步观察，不能归因于模型架构。两种输入下，原始坐标实验的 F1 都高于对应的归一化坐标实验，但当前实验不足以判定归一化普遍有害。

现有 NPZ 的 `fall` 或 `adl` 标签来自视频所属目录，程序把它赋给该视频的所有窗口。这适用于视频类别实验，但跌倒视频中尚未发生跌倒的片段也可能被标为 `fall`。因此这四组结果是**初步窗口级基线**，不能当作跌倒事件检出率或实时报警性能。如果要严格比较模型架构的优劣，需要统一窗口筛选、缺失值处理和评价规则。

## 后续工作

以下安排依据[课程报告](../基于人体姿态估计的跌倒检测课程报告.docx)第 4.7 节“面向后续复现的验证方案”和第 5.2 节“展望”。当前以 UR-Fall 的 `fall`、`adl` 目录区分**视频类别**即可；姿态提取和坐标归一化也已完成，无需把它们列为待做实验。已有 RF、LSTM 代码均按视频分组进行五折评估，这一做法应继续保留，避免同一视频的相邻窗口同时进入训练集和测试集。

| 阶段 | 实验 | 当前状态与下一步 |
| --- | --- | --- |
| 1. 分类器基线 | 固定 BlazePose，比较 RF、MLP、LSTM | RF、LSTM 已有原始与归一化坐标结果；运行已完成的 MLP 程序，补齐其两组指标。整理 Accuracy、跌倒 Recall、F1 和混淆矩阵。 |
| 2. 单因素预处理对照 | 每次只改变归一化、缺失点处理或窗口长度中的一个因素 | 原始与归一化坐标对照已初步完成；RF 与 LSTM 的缺失点处理和窗口筛选不同，需在相同模型、相同视频与评价规则下补做插值和窗口长度对照，才能判断单一因素的作用。 |
| 3. 模型结构扩展 | 增加 ST-GCN **或** Transformer | 尚未实现。若开展，固定输入与评估规则，并记录训练时间、参数量和推理延迟，与较简单的基线比较。 |

报告还建议每种配置使用多个随机种子，保存训练日志和划分信息，并报告结果的离散程度。现有结果只记录了单个种子，因此多种子重复实验仍待完成。遮挡或关键点噪声测试、跨人员或跨场景验证，以及减少关节点或压缩模型等轻量化实验属于进一步扩展，可在上述三阶段后按时间安排。

现有目录标签足以支持“这段视频是否属于跌倒视频”的分类。若后续要研究**跌倒发生的具体时刻**、事件级检出率或报警延迟，才需要补充事件时间标注，并定义窗口预测如何触发报警；这不影响当前视频类别实验。

## 目录说明

```text
data/                 # 原始帧和生成的关键点数据（不提交到 Git）
src/blazepose.py      # 逐帧姿态提取
src/normalize_keypoints.py  # 姿态坐标归一化
src/experiment_rf.py  # 滑动窗口、交叉验证和随机森林训练
src/experiment_lstm.py  # LSTM 五折实验
src/experiment_mlp.py   # MLP 五折实验程序，尚无实测结果
results/              # 已保存的实验配置、指标、预测及部分模型文件
models/               # 其他已保存模型文件
```
