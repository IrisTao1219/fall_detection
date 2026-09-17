# 基于人体姿态估计与多类型分类模型的视频跌倒检测

本课程项目用普通摄像头视频完成跌倒（`fall`）与日常活动（`adl`）二分类：先用 MediaPipe BlazePose 提取 33 个人体关键点，再比较不同分类模型对连续姿态的识别能力。研究目标与实验设计见上级目录的[《实验方案》](../实验方案.md)。

当前实验重点比较使用骨架关键点时，不同分类器对跌倒视频的识别效果。RF、LSTM、MLP 和旧版三层 ST-GCN 均已完成原始坐标与归一化坐标两组实验；十层 ST-GCN 程序已写好，尚未运行。后续实验按[课程报告](../基于人体姿态估计的跌倒检测课程报告.docx)第 4.7 节的复现路线逐步开展。

**当前进度：**已实现 URFD 图像帧的姿态提取和坐标归一化，保存了四种分类模型共八组五折实验结果，其中 ST-GCN 结果来自旧版三层网络。新版十层网络、窗口长度和缺失点处理对照，以及多个随机种子的重复实验尚未完成运行。

## 环境准备

- Python 3.11（`pyproject.toml` 要求 Python ≥3.11，仓库的 `.python-version` 为 3.11）
- 推荐使用 uv 管理环境；依赖定义见 `pyproject.toml`

在项目根目录执行（当前锁文件将 PyTorch 指向 CUDA 12.8 软件包，适用于相应平台；macOS 需要调整 PyTorch 依赖来源）：

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

LSTM 结果写入 `results/lstm/` 和 `results/lstm_normalized/`。保存的两组配置使用 33 个关节的 `x/y`（每帧 66 维）、30 帧窗口、1 帧步长、窗口内插值、跳过完全没有检测到人体的窗口、10 层单向 LSTM（隐藏维度 80）和 30 个训练轮次。按视频分组做 5 折评估，每折仅用训练视频拟合标准化参数。结果目录保存了 `config.json`、`metrics.txt`、`fold_metrics.csv`、`confusion_matrix.csv`、`predictions.csv` 和 `histories/` 中的训练损失历史；当前仓库没有保存每折模型权重。

### 5. 训练并评估 MLP

`src/experiment_mlp.py` 可作为将连续 30 帧关键点展平后分类的神经网络基线：

```bash
uv run python src/experiment_mlp.py --data-root data/keypoints_normalized
```

原始坐标实验可把数据目录改为 `data/keypoints`。默认使用全部 33 个关节的 `x/y`、30 帧窗口、1 帧步长、缺失坐标时间插值和两层 MLP（256、64 个隐藏单元）。五折按视频分组，训练折内再按视频划分验证集用于早停；标准化参数仅由内部训练视频计算。输出分别保存在 `results/mlp_normalized/` 或 `results/mlp/`，与 LSTM 使用相同的结果结构：根目录有 `config.json`、`metrics.txt`、`fold_metrics.csv`、`confusion_matrix.csv`、`predictions.csv`；程序将每折 `.pt` 模型写入 `models/`，每折训练历史 CSV 写入 `histories/`。总体指标同样来自折外窗口预测。

两组 MLP 结果已保存，具体指标见下表。当前仓库没有保存每折模型权重。

### 6. 训练并评估十层 ST-GCN

```bash
uv run python src/experiment_stgcn.py --data-root data/keypoints
uv run python src/experiment_stgcn.py --data-root data/keypoints_normalized
```

ST-GCN 复用 LSTM 的关键点读取与窗口切分：33 个关节的 `x/y`、30 帧窗口、1 帧步长、窗口内缺失坐标插值，并跳过完全没有检测到人体的窗口。模型按 BlazePose 关节连接构图，使用十个时空图卷积模块：前四层 64 通道、中间三层 128 通道、后三层 256 通道；时间卷积核大小为 9，每层包含残差连接和 Dropout，预测时用 Softmax 得到类别概率。外层按视频分组做五折评估，训练折内再按视频划分验证集用于早停；标准化参数只由内部训练视频计算。当前只支持已有的 BlazePose 数据，尚未接入其他姿态提取器或数据集。

新版原始和归一化坐标实验将分别写入 `results/stgcn_10layer/` 和 `results/stgcn_10layer_normalized/`，与旧版三层结果分开。目录结构与 LSTM 相同：程序将每折模型写入 `models/`，每折训练历史写入 `histories/`；`config.json`、`metrics.txt`、`fold_metrics.csv`、`confusion_matrix.csv` 和 `predictions.csv` 位于结果目录根部。程序还保存视频级预测和指标、划分记录及 `metrics.json`。新版目前没有实测结果。

## 已完成的实验与结果

下表取自各结果目录的 `metrics.txt`，均为 **5 折折外窗口预测汇总指标**，其中 `fall` 为正类。八组结果都覆盖 70 段视频、8993 个 30 帧窗口，步长为 1 帧，并按视频分组评估。各组保存的配置均采用窗口内插值。ST-GCN 两行属于旧版三层网络，不能代表新版十层网络；表中数值不是视频级指标。

| 模型 | 输入 | 窗口数 | 准确率 | 跌倒精确率 | 跌倒召回率 | 跌倒 F1 | ROC-AUC | 结果目录 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RF | 原始 `x/y` | 8993 | 0.9158 | 0.8327 | 0.8014 | 0.8168 | 0.9507 | [`results/rf/`](results/rf/) |
| RF | 归一化 `x/y` | 8993 | 0.8589 | 0.7419 | 0.6090 | 0.6689 | 0.9093 | [`results/rf_normalized/`](results/rf_normalized/) |
| LSTM | 原始 `x/y` | 8993 | 0.9237 | 0.8436 | 0.8276 | 0.8355 | 0.9522 | [`results/lstm/`](results/lstm/) |
| LSTM | 归一化 `x/y` | 8993 | 0.8621 | 0.7161 | 0.6808 | 0.6980 | 0.8661 | [`results/lstm_normalized/`](results/lstm_normalized/) |
| MLP | 原始 `x/y` | 8993 | 0.8453 | 0.6440 | 0.7587 | 0.6966 | 0.8736 | [`results/mlp/`](results/mlp/) |
| MLP | 归一化 `x/y` | 8993 | 0.8602 | 0.7033 | 0.6969 | 0.7001 | 0.8313 | [`results/mlp_normalized/`](results/mlp_normalized/) |
| ST-GCN（旧版三层） | 原始 `x/y` | 8993 | 0.8949 | 0.7458 | 0.8361 | 0.7884 | 0.9475 | [`results/stgcn/`](results/stgcn/) |
| ST-GCN（旧版三层） | 归一化 `x/y` | 8993 | 0.7752 | 0.5179 | 0.5696 | 0.5425 | 0.7395 | [`results/stgcn_normalized/`](results/stgcn_normalized/) |

八组结果使用相同的视频与窗口位置；RF 预测文件的 `window_start/window_end` 是从零开始的窗口位置，`start_frame/end_frame` 对应原始帧号。原始坐标下，LSTM 的窗口级 F1 最高（0.8355），ST-GCN 的跌倒召回率最高（0.8361）；归一化坐标下，MLP 的 F1 最高（0.7001）。除 MLP 外，同一模型的原始坐标 F1 高于归一化坐标。当前只有一个随机种子，且各模型训练过程不同，这些差值不宜直接归因于某一种结构或预处理。

ST-GCN 还输出视频级结果：原始坐标 F1 为 0.9667（70 段视频中 68 段分类正确），归一化坐标 F1 为 0.8000（59 段正确）。视频级概率由同一视频的窗口概率平均得到，不能与上表窗口级指标直接比较。

现有 NPZ 的 `fall` 或 `adl` 标签来自视频所属目录，程序把它赋给该视频的所有窗口。这适用于视频类别实验，但跌倒视频中尚未发生跌倒的片段也可能被标为 `fall`。因此这些结果是**视频类别的初步基线**，不能当作跌倒事件检出率或实时报警性能。

## 后续工作

以下安排依据[课程报告](../基于人体姿态估计的跌倒检测课程报告.docx)第 4.7 节“面向后续复现的验证方案”和第 5.2 节“展望”。当前以 UR-Fall 的 `fall`、`adl` 目录区分**视频类别**即可；姿态提取和坐标归一化也已完成。现有四种模型均按视频分组进行五折评估，避免同一视频的相邻窗口同时进入训练集和测试集。

| 阶段 | 实验 | 当前状态与下一步 |
| --- | --- | --- |
| 1. 分类器基线 | 固定 BlazePose，比较 RF、MLP、LSTM | 三种模型的原始与归一化坐标结果均已保存；下一步可整理多随机种子重复结果。 |
| 2. 单因素预处理对照 | 每次只改变归一化、缺失点处理或窗口长度中的一个因素 | 原始与归一化坐标对照已完成单种子实验；缺失点处理和窗口长度对照尚未完成。 |
| 3. 模型结构扩展 | 增加 ST-GCN **或** Transformer | 旧版三层 ST-GCN 两组结果已保存；新版十层程序待运行，Transformer 尚未实现。可在统一重复实验后比较模型表现与训练成本。 |

报告还建议每种配置使用多个随机种子，保存训练日志和划分信息，并报告结果的离散程度。现有结果只记录了单个种子，因此多种子重复实验仍待完成。遮挡或关键点噪声测试、跨人员或跨场景验证，以及减少关节点或压缩模型等轻量化实验属于进一步扩展，可在上述三阶段后按时间安排。

现有目录标签足以支持“这段视频是否属于跌倒视频”的分类。若后续要研究**跌倒发生的具体时刻**、事件级检出率或报警延迟，才需要补充事件时间标注，并定义窗口预测如何触发报警；这不影响当前视频类别实验。

## 目录说明

```text
data/                 # 原始帧和生成的关键点数据（不提交到 Git）
src/blazepose.py      # 逐帧姿态提取
src/normalize_keypoints.py  # 姿态坐标归一化
src/experiment_rf.py  # 滑动窗口、交叉验证和随机森林训练
src/experiment_lstm.py  # LSTM 五折实验
src/experiment_mlp.py   # MLP 五折实验
src/experiment_stgcn.py  # 十层 ST-GCN 五折实验；现存 ST-GCN 结果来自旧版三层程序
results/              # 已保存的实验配置、指标、预测及部分模型文件
models/               # 其他已保存模型文件
```
