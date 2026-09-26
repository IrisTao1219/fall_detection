# 基于人体姿态估计与多类型分类模型的视频跌倒检测

本课程项目用普通摄像头视频完成跌倒（`fall`）与日常活动（`adl`）二分类：先用 MediaPipe BlazePose 提取 33 个人体关键点，再比较不同分类模型对连续姿态的识别能力。研究目标与实验设计见上级目录的[《实验方案》](../实验方案.md)。

当前实验重点比较使用骨架关键点时，不同分类器对跌倒窗口的识别效果。已将跌倒视频的窗口标签改为依据 UR-Fall CSV 中逐帧标注计算，保存了四种模型各两组、共八组结果。旧版整段视频共用同一标签的结果保存在 `results-old/`。后续实验按[课程报告](../基于人体姿态估计的跌倒检测课程报告.docx)第 4.7 节的复现路线逐步开展。

**当前进度：**已实现 URFD 图像帧的姿态提取和坐标归一化，保存了 RF、LSTM、MLP、十层 ST-GCN 各两组标签正确对齐的五折结果。旧标签规则的八组结果保留在 `results-old/`。窗口长度和缺失点处理对照尚未完成运行。

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

### 一键运行当前实验

推荐优先使用 `src/run_all.sh` 统一生成窗口缓存并依次运行 RF、MLP、LSTM 和 ST-GCN。当前默认 keypoints profile 是 BlazePose + UR-Fall，结果写入 `results/ur/{rf,mlp,lstm,stgcn}`：

```bash
bash src/run_all.sh
```

也可以显式指定 keypoints profile：

```bash
bash src/run_all.sh --keypoints ur
bash src/run_all.sh --keypoints ur-origin
bash src/run_all.sh --keypoints blazepose-normalized
```

新增的 BlazePose + Le2i keypoints profile 使用 `data/keypoints_le2i/*.npz` 中的逐帧 `frame_labels` 生成窗口标签，结果写入 `results/le2i_blazepose/{rf,mlp,lstm,stgcn}`：

```bash
bash src/run_all.sh --keypoints le2i-blazepose
```

常用覆盖参数：

```bash
bash src/run_all.sh --keypoints le2i-blazepose --device cuda:0
bash src/run_all.sh --keypoints le2i-blazepose --window-size 25 --stride 5
```

`run_all.sh` 会先调用 `src/prepare_windows_ur.py` 生成共享窗口缓存。窗口脚本会根据 keypoints NPZ 自动选择标签来源：NPZ 中存在 `frame_labels` 时使用逐帧标签；否则按 UR-Fall keypoints 读取官方 `data/urfall-cam0-falls.csv`。

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

### Le2i：抽帧、提取 BlazePose、运行实验

Le2i 原始视频和标注放在 `data/raw_le2i/` 后，先将视频抽成 PNG 并生成每段视频的 `labels.csv`：

```bash
uv run python src/parse_le2i_annotations.py
```

再对抽帧后的 `data/raw_le2i/pngs/` 运行 BlazePose，输出 `data/keypoints_le2i/<video_id>.npz`：

```bash
uv run python src/blazepose_le2i.py
```

Le2i 的 NPZ 中 `label` 字段为 `mixed`，真正用于训练的是逐帧 `frame_labels`：

| 字段 | 含义 |
| --- | --- |
| `keypoints` | `[帧数, 33, 4]`；最后一维为 `x, y, z, visibility` |
| `valid_mask` | 每帧是否检测到人体 |
| `frame_indices`, `timestamps`, `fps` | 原始帧号、秒级时间戳和帧率；Le2i 按 25 FPS 处理 |
| `frame_labels` | `[帧数]`；`0=non-fall`，`1=fall` |
| `fall_start`, `fall_end`, `bboxes` | 原始标注中的跌倒区间和目标框信息 |
| `label`, `video_id` | `label` 固定为 `mixed`，`video_id` 为 Le2i 视频编号 |

然后运行全部模型：

```bash
bash src/run_all.sh --keypoints le2i-blazepose
```

默认 Le2i 窗口为 25 帧、步长 5 帧。窗口标签取中心帧的 `frame_labels`，同一原始视频的窗口不会同时出现在训练折和测试折中。

### 2. 归一化坐标

```bash
uv run python src/normalize_keypoints.py
```

脚本读取 data/keypoints/，以每帧髋部中心作为坐标原点，并使用整段视频中可靠肩宽的稳健统计值作为统一尺度进行归一化，从而减少逐帧尺度波动带来的时序噪声。归一化过程保持原有有效帧和关键点结构，不额外根据普通关键点 visibility 制造缺失值。

### 3. 训练并评估随机森林

```bash
uv run python src/experiments/rf.py --data-root data/keypoints
uv run python src/experiments/rf.py --data-root data/keypoints_normalized
```

脚本依据 `--data-root` 分别写入 `results/rf/` 和 `results/rf_normalized/`。可用 `--result-root` 修改结果根目录；其他实验参数在脚本顶部配置。

每个窗口包含 30 帧（1 秒），步长为 1 帧；使用 33 个关键点的 `x/y` 坐标，展开后为 1980 维。低可见度或缺失坐标在窗口内按时间插值。`adl` 视频的窗口标为 0；`fall` 视频的窗口依据 `data/urfall-cam0-falls.csv` 的逐帧标注计算，规则见下文。评估采用按视频分组的分层交叉验证，同一视频的窗口不会同时进入训练集和测试集。交叉验证完成后，脚本再用全部窗口训练并保存最终模型。

结果目录包含 `config.json`、`metrics.txt`、`fold_metrics.csv`、`confusion_matrix.csv`、`window_predictions.csv`、`feature_importance.csv` 和最终模型 `.joblib`。`metrics.txt` 中的总体指标来自各折测试窗口的汇总预测；它们是**窗口级**指标，不是独立视频或实时场景的检测指标。

### 4. 训练并评估 LSTM

```bash
uv run python src/experiments/lstm.py --data-root data/keypoints
uv run python src/experiments/lstm.py --data-root data/keypoints_normalized
```

LSTM 结果写入 `results/lstm/` 和 `results/lstm_normalized/`。保存的两组配置使用 33 个关节的 `x/y`（每帧 66 维）、30 帧窗口、1 帧步长、窗口内插值、跳过完全没有检测到人体的窗口、10 层单向 LSTM（隐藏维度 80）和 30 个训练轮次。按视频分组做 5 折评估，每折仅用训练视频拟合标准化参数。结果目录保存了 `config.json`、`metrics.txt`、`fold_metrics.csv`、`confusion_matrix.csv`、`predictions.csv` 和 `histories/` 中的训练损失历史；当前仓库没有保存每折模型权重。

### 5. 训练并评估 MLP

`src/experiments/mlp.py` 可作为将连续 30 帧关键点展平后分类的神经网络基线：

```bash
uv run python src/experiments/mlp.py --data-root data/keypoints_normalized
```

原始坐标实验可把数据目录改为 `data/keypoints`。默认使用全部 33 个关节的 `x/y`、30 帧窗口、1 帧步长、缺失坐标时间插值和两层 MLP（256、64 个隐藏单元）。五折按视频分组，训练折内再按视频划分验证集用于早停；标准化参数仅由内部训练视频计算。输出分别保存在 `results/mlp_normalized/` 或 `results/mlp/`，与 LSTM 使用相同的结果结构：根目录有 `config.json`、`metrics.txt`、`fold_metrics.csv`、`confusion_matrix.csv`、`predictions.csv`；程序将每折 `.pt` 模型写入 `models/`，每折训练历史 CSV 写入 `histories/`。总体指标同样来自折外窗口预测。

MLP 的原始与归一化坐标结果均已按 CSV 窗口标签保存，具体指标见下表。当前仓库没有保存每折模型权重。

### 6. 训练并评估十层 ST-GCN

```bash
uv run python src/experiments/stgcn.py --data-root data/keypoints
uv run python src/experiments/stgcn.py --data-root data/keypoints_normalized
```

ST-GCN 复用 LSTM 的关键点读取与窗口切分：33 个关节的 `x/y`、30 帧窗口、1 帧步长、窗口内缺失坐标插值，并跳过完全没有检测到人体的窗口。模型按 BlazePose 关节连接构图，使用十个时空图卷积模块：前四层 64 通道、中间三层 128 通道、后三层 256 通道；时间卷积核大小为 9，每层包含残差连接和 Dropout，预测时用 Softmax 得到类别概率。外层按视频分组做五折评估，训练折内再按视频划分验证集用于早停；标准化参数只由内部训练视频计算。当前只支持已有的 BlazePose 数据，尚未接入其他姿态提取器或数据集。

原始和归一化坐标实验分别写入 `results/stgcn/` 和 `results/stgcn_normalized/`。目录结构与 LSTM 相同：程序将每折模型写入 `models/`，每折训练历史写入 `histories/`；`config.json`、`metrics.txt`、`fold_metrics.csv`、`confusion_matrix.csv` 和 `predictions.csv` 位于结果目录根部。程序还保存视频级预测和指标、划分记录及 `metrics.json`。两组十层网络结果均已保存。

## 已完成的实验与结果

### 新结果：CSV 逐窗口标签

本次改动是**标签的计算单位**。旧实验按视频所属的 `fall` 或 `adl` 目录取一个标签，再赋给该视频的全部滑动窗口；这会把跌倒视频中跌倒发生前的正常活动也标成 `fall`。新实验对 `fall` 视频读取 [`data/urfall-cam0-falls.csv`](data/urfall-cam0-falls.csv) 的前三列（视频名、帧号、姿态标签），按每个 30 帧窗口实际覆盖的帧号取标注。CSV 中 `-1` 表示非躺倒、`0` 表示跌倒过渡、`1` 表示躺倒；计算窗口标签时忽略 `0`，在余下的 `-1` 与 `1` 中按多数票分别记为 `adl=0` 或 `fall=1`。有效标签为空或票数持平的窗口被跳过；`adl` 视频的窗口均标为 0。**因此这里的正类依照 CSV 对“躺倒”的标注，不等同于把跌倒过渡阶段也算作正类。**

下面取自 `results/` 与 `results_normalized/` 各目录的 `metrics.txt`，是按视频分组的 5 折**折外窗口预测汇总指标**，并非视频级或事件级指标。八组结果各有 8963 个窗口，其中 883 个为正类；MLP 和 ST-GCN 的窗口位置与真实标签均已分别同对应的 LSTM 结果逐项核对一致。ST-GCN 的当前结果已使用修正后的原始帧号映射。

| 模型 | 输入 | 窗口数 | 准确率 | 跌倒精确率 | 跌倒召回率 | 跌倒 F1 | ROC-AUC | 结果目录 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RF | 原始 `x/y` | 8963 | 0.9709 | 0.8494 | 0.8562 | 0.8528 | 0.9891 | [`results/rf/`](results/rf/) |
| RF | 归一化 `x/y` | 8963 | 0.9487 | 0.8395 | 0.5923 | 0.6946 | 0.9660 | [`results_normalized/rf/rf_normalized/`](results_normalized/rf/rf_normalized/) |
| LSTM | 原始 `x/y` | 8963 | 0.9776 | 0.9231 | 0.8426 | 0.8810 | 0.9633 | [`results/lstm/`](results/lstm/) |
| LSTM | 归一化 `x/y` | 8963 | 0.9445 | 0.7265 | 0.7010 | 0.7135 | 0.8900 | [`results_normalized/lstm/lstm_normalized/`](results_normalized/lstm/lstm_normalized/) |
| MLP | 原始 `x/y` | 8963 | 0.9530 | 0.7077 | 0.8913 | 0.7890 | 0.9763 | [`results/mlp/`](results/mlp/) |
| MLP | 归一化 `x/y` | 8963 | 0.9279 | 0.6167 | 0.7089 | 0.6596 | 0.8781 | [`results_normalized/mlp/mlp_normalized/`](results_normalized/mlp/mlp_normalized/) |
| ST-GCN（十层） | 原始 `x/y` | 8963 | 0.9861 | 0.9084 | 0.9547 | 0.9310 | 0.9976 | [`results/stgcn/`](results/stgcn/) |
| ST-GCN（十层） | 归一化 `x/y` | 8963 | 0.9342 | 0.6570 | 0.6942 | 0.6751 | 0.9571 | [`results_normalized/stgcn/stgcn_normalized/`](results_normalized/stgcn/stgcn_normalized/) |

在原始坐标的四种模型中，ST-GCN 的窗口级 F1 最高（0.9310），跌倒召回率也最高（0.9547）。归一化坐标下，LSTM 的 F1 最高（0.7135）。

### 旧结果：整段视频共用一个标签

以下历史结果保存在 `results-old/`，不覆盖。它们按视频目录将 `fall` 视频的**全部**窗口标为 1，包括跌倒前的正常活动。八组结果各覆盖 70 段视频、8993 个窗口。表中指标从 `results-old/` 的现存 `metrics.txt` 读取；ST-GCN 数值以这些文件为准。

| 模型 | 输入 | 窗口数 | 准确率 | 跌倒精确率 | 跌倒召回率 | 跌倒 F1 | ROC-AUC | 结果目录 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RF | 原始 `x/y` | 8993 | 0.9158 | 0.8327 | 0.8014 | 0.8168 | 0.9507 | [`results-old/rf/`](results-old/rf/) |
| RF | 归一化 `x/y` | 8993 | 0.8589 | 0.7419 | 0.6090 | 0.6689 | 0.9093 | [`results-old/rf_normalized/`](results-old/rf_normalized/) |
| LSTM | 原始 `x/y` | 8993 | 0.9237 | 0.8436 | 0.8276 | 0.8355 | 0.9522 | [`results-old/lstm/`](results-old/lstm/) |
| LSTM | 归一化 `x/y` | 8993 | 0.8621 | 0.7161 | 0.6808 | 0.6980 | 0.8661 | [`results-old/lstm_normalized/`](results-old/lstm_normalized/) |
| MLP | 原始 `x/y` | 8993 | 0.8453 | 0.6440 | 0.7587 | 0.6966 | 0.8736 | [`results-old/mlp/`](results-old/mlp/) |
| MLP | 归一化 `x/y` | 8993 | 0.8602 | 0.7033 | 0.6969 | 0.7001 | 0.8313 | [`results-old/mlp_normalized/`](results-old/mlp_normalized/) |
| ST-GCN（十层） | 原始 `x/y` | 8993 | 0.9004 | 0.7528 | 0.8551 | 0.8007 | 0.9591 | [`results-old/stgcn/`](results-old/stgcn/) |
| ST-GCN（十层） | 归一化 `x/y` | 8993 | 0.7535 | 0.4766 | 0.5425 | 0.5074 | 0.7520 | [`results-old/stgcn_normalized/`](results-old/stgcn_normalized/) |

旧结果只能作为按视频类别标注的历史基线，不能当作跌倒事件检出率或实时报警性能。

## 后续工作

以下安排依据[课程报告](../基于人体姿态估计的跌倒检测课程报告.docx)第 4.7 节“面向后续复现的验证方案”和第 5.2 节“展望”。姿态提取和坐标归一化已完成；当前窗口分类实验对 `fall` 视频使用 CSV 逐帧标注生成窗口标签。已运行的模型均按视频分组进行五折评估，避免同一视频的相邻窗口同时进入训练集和测试集。

| 阶段 | 实验 | 当前状态与下一步 |
| --- | --- | --- |
| 1. 分类器基线 | 固定 BlazePose，比较 RF、MLP、LSTM | 三种模型的原始与归一化坐标结果均已按新标签保存。 |
| 2. 单因素预处理对照 | 每次只改变归一化、缺失点处理或窗口长度中的一个因素 | 原始与归一化坐标对照已完成；缺失点处理和窗口长度对照尚未完成。 |
| 3. 模型结构扩展 | 增加 ST-GCN **或** Transformer | 十层 ST-GCN 的原始与归一化坐标两组结果已保存；Transformer 尚未实现。后续可比较模型表现与训练成本。 |

遮挡或关键点噪声测试、跨人员或跨场景验证，以及减少关节点或压缩模型等轻量化实验属于进一步扩展，可在上述三阶段后按时间安排。

当前 CSV 逐窗口标签支持按姿态状态评估窗口分类。若后续要研究**跌倒发生的具体时刻**、事件级检出率或报警延迟，仍需明确过渡阶段的事件定义，并设计连续窗口触发报警的规则。

## 目录说明

```text
data/                 # 原始帧和生成的关键点数据（不提交到 Git）
src/blazepose.py      # 逐帧姿态提取
src/normalize_keypoints.py  # 姿态坐标归一化
src/experiments/rf.py  # 滑动窗口、交叉验证和随机森林训练
src/experiments/lstm.py  # LSTM 五折实验
src/experiments/mlp.py   # MLP 五折实验
src/experiments/stgcn.py  # 十层 ST-GCN 五折实验
results/              # CSV 逐窗口标签结果（原始坐标为主）
results_normalized/   # 新保存的归一化坐标实验结果
results-old/          # 保留的旧视频级标签结果
models/               # 其他已保存模型文件
```
