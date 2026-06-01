# DiffSinger v3 + SFC Breathness 完整使用手册

本文档提供从环境配置到模型导出的完整操作流程。所有命令均在项目根目录 `Diffsinger-v3-sfcb/` 下执行。

---

## 第 1 步：环境配置

### 1.1 创建并激活 conda 环境

```bash
# 如果尚未创建环境，请先创建：
# conda create -n diffsinger python=3.10 -y

# 激活环境
conda activate diffsinger
```

### 1.2 安装 PyTorch

根据你的 CUDA 版本手动安装 PyTorch。推荐 PyTorch >= 2.0。

```bash
# 示例：CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 示例：CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 更多版本参见：https://pytorch.org/get-started/locally/
```

### 1.3 安装其他依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含以下主要依赖：lightning、librosa、numpy、omegaconf、praat-parselmouth、pyworld、tensorboard、tensorboardX、click、dask、einops、h5py、loguru、matplotlib、onnx、onnxsim、pydantic、PyYAML、resampy、scipy、soundfile、sympy、torchmetrics、tqdm。

### 1.4 验证环境

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 第 2 步：准备原始数据

### 2.1 目录结构

在项目根目录下创建 `raw/` 目录，按以下结构组织你的训练数据：

```
raw/
├── wavs/
│   ├── singer_01.wav
│   ├── singer_02.wav
│   ├── singer_03.wav
│   └── ...
└── transcriptions.csv
```

### 2.2 音频文件要求

- 格式：WAV
- 采样率：建议 44100 Hz（44.1 kHz），单声道
- 内容：干声（无伴奏、无混响）的人声演唱录音
- 命名：使用有意义的名称，建议包含歌手 ID 和序号

### 2.3 transcriptions.csv 格式

`transcriptions.csv` 是一个 CSV 文件，每行对应一个音频文件的标注信息。包含以下列：

| 列名 | 含义 | 示例 |
|-----|------|------|
| `name` | 对应的 WAV 文件名（不含 `.wav` 后缀） | `singer_01` |
| `ph_seq` | 音素序列（以空格分隔） | `s il i ng ch uang m ing y ue g uang` |
| `ph_dur` | 每段音符对应的音素数量（以空格分隔） | `2 3 3 3 3 3` |
| `ph_num` | 音符数量（一段歌曲中的音符总数） | `6` |
| `note_seq` | 音符序列，MIDI 音高（以空格分隔） | `60 62 64 65 64 62` |
| `note_dur` | 音符时长序列，以秒为单位（以空格分隔） | `0.5 0.5 0.5 0.5 0.5 0.5` |

> **提示**：推荐使用 OpenVPI 的 [MakeDiffSinger](https://github.com/openvpi/MakeDiffSinger) 工具来自动生成符合 DiffSinger 格式的标注文件。

### 2.4 数据质量建议

- 每个 WAV 文件时长建议在 1~15 秒之间，过短不足以提取有效特征，过长影响训练效率
- 确保发声清晰、无明显背景噪声
- 如需甜蜜气声效果，数据集中应包含足够比例的气声样本（建议 >= 30%）
- **SFCB 专用特征：`binarizer.features.aperiodic_mel.enabled: true`** — 预处理时从 VR 分离后的噪声波形提取真实非周期 mel 谱，作为 AperiodicCVAE 的 ground truth
- 可使用 `breathy_pipeline.py filter` 命令评估数据集中气声样本的比例

---

## 第 3 步：配置训练参数

### 3.1 复制配置模板

```bash
# 复制声学模型模板
cp configs/templates/config_acoustic.yaml configs/my_experiment.yaml
```

### 3.2 编辑配置文件

使用文本编辑器（VS Code、Notepad++ 等）打开 `configs/my_experiment.yaml`，修改以下关键配置项：

#### 3.2.1 数据源配置 (data)

```yaml
data:
  sources:
    - name: my_dataset          # 数据集名称
      raw_data_dir: raw/        # 原始数据目录路径
      language: zh              # 语言：zh / en / ja 等
      speakers:                 # 说话人配置
        - name: singer_01       # 说话人名称（对应 transcription 中的 speaker）
```

#### 3.2.2 说话人配置 (speaker)

确保 `speakers` 列表中的说话人名称与实际标注数据中的说话人保持一致。对于多说话人模型，列出所有说话人。

#### 3.2.3 测试集前缀 (test_prefixes)

```yaml
binarizer:
  test_prefixes:
    - singer_01     # 将以此说话人数据的前缀作为验证集
```

#### 3.2.4 SFC Breathness 配置（默认启用）

在基础配置 `acoustic_v3.yaml` 和 `base_v3.yaml` 中，`sweet_breathy` 相关模块已默认启用。如需关闭气声增强：

```yaml
model:
  sweet_breathy:
    enabled: false         # 设为 false 即回退到原生 DiffSinger v3 模式
```

完整的气声参数微调请参考 `configs/sweet_breathy/acoustic_breathy_override.yaml`，其中包含所有气声模块的门控开关、损失权重和 warm-up 策略。

### 3.3 气声风格预设

在推理阶段，支持三种气声风格预设（见 `configs/sweet_breathy/inference_preset.yaml`）：

| 预设 | breathiness | voicing | tension | 听感特征 |
|------|:----------:|:-------:|:-------:|---------|
| light | +0.10 | +0.05 | -0.05 | 轻柔气感，谐波清晰 |
| sweet | +0.25 | 0.0 | -0.15 | 均衡甜蜜气声（默认） |
| heavy | +0.35 | -0.10 | -0.25 | 浓厚气声质感 |

> 对于 C5 及以上高音（MIDI >= 72），系统会自动降低气声比例并增加张力，避免高音区出现"虚弱/漏气"听感。

---

## 第 4 步：数据预处理（Binarize）

### 4.1 执行预处理

```bash
python scripts/binarize.py acoustic --config configs/my_experiment.yaml
```

这一步将会：

1. 读取 `raw/` 目录下的所有 WAV 音频和 `transcriptions.csv` 标注
2. 对音频进行重采样、音量归一化
3. 提取 mel 频谱、基频（F0）、非周期性参数等声学特征
4. 生成二进制格式的训练数据，保存在配置中指定的 `binary_data_dir` 目录
5. 建立音素映射、说话人映射、语言映射等词汇表

### 4.2 覆盖率检查选项

`--coverage-check-option` 参数控制音素覆盖率检查的行为：

| 选项 | 行为 |
|------|------|
| `strict`（默认） | 如果标注中存在未在词典中定义的音素，预处理会报错终止 |
| `bypass` | 跳过覆盖率检查，即使存在未知音素也继续预处理 |
| `compat` | 自动从词汇表中移除未覆盖的音素 |

```bash
# 跳过覆盖率检查（如果你确定未知音素不参与训练）
python scripts/binarize.py acoustic --config configs/my_experiment.yaml --coverage-check-option bypass
```

### 4.3 覆写配置参数

可以通过 `--override` 参数在命令行临时修改配置，而不必改动 YAML 文件：

```bash
python scripts/binarize.py acoustic --config configs/my_experiment.yaml \
    --override "data.sources.0.raw_data_dir=custom_raw/" \
    --override "binarizer.test_prefixes=[singer_02]"
```

---

## 第 5 步：训练模型

### 5.1 启动训练（声学模型）

```bash
python scripts/train.py acoustic --config configs/my_experiment.yaml --exp-name my_acoustic_exp
```

### 5.2 启动训练（方差模型）

声学模型训练完成后，可继续训练方差模型以增强可控性：

```bash
python scripts/train.py variance --config configs/my_experiment.yaml --exp-name my_variance_exp
```

### 5.3 训练参数说明

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径（必填） |
| `--exp-name` | 实验名称，checkpoint 将保存在 `experiments/<exp_name>/` 下（必填） |
| `--work-dir` | 工作目录，默认为 `experiments/` |
| `--log-dir` | 日志目录，默认与 checkpoint 目录相同 |
| `--restart` | 忽略已有 checkpoint，从头开始训练 |
| `--resume-from` | 从指定 checkpoint 文件恢复训练 |
| `--override` | 临时覆写配置参数 |

### 5.4 各版本训练步数对比

| 版本 | Acoustic 满训 | Acoustic 最低可捞 | Variance 满训 | Variance 最低可捞 |
|------|:--:|:--:|:--:|:--:|
| **v2** | ~200k | ~140k | ~150k | ~100k |
| **v2.5** | 160k | 100k | 160k | 80k |
| **v3** | 160k | 80k | 160k | 70k |
| **v3-sfcb** | **120k** | **90k** | **120k** | **70k** |

> v3-sfcb 90k 步即可捞模型，120k 满训。训练启动时自动打印 `data_stats`（含 augmentation 后的有效训练时长）。

### 5.5 训练日志与 Checkpoint

- **TensorBoard 日志**：保存在 `<work_dir>/<exp_name>/lightning_logs/` 目录，可用以下命令启动可视化：

  ```bash
  tensorboard --logdir experiments/my_acoustic_exp/lightning_logs/
  ```

- **模型 Checkpoint**：保存在 `experiments/my_acoustic_exp/` 目录，文件名格式为 `model-<tag>-steps=<N>-epochs=<N>.ckpt`。训练过程中会自动保存周期性 checkpoint（可配置频率和最多个数）以及基于验证指标的最优 checkpoint。

- **配置文件**：训练启动时，自动复制 `config.yaml` 和带时间戳的 `hparams-*.yaml` 到 checkpoint 目录，便于事后追溯。

### 5.6 断点续训

如果训练意外中断，再次运行相同的训练命令即可自动恢复：

```bash
python scripts/train.py acoustic --config configs/my_experiment.yaml --exp-name my_acoustic_exp
```

如存在多个最新 checkpoint，需手动指定：

```bash
python scripts/train.py acoustic --config configs/my_experiment.yaml --exp-name my_acoustic_exp \
    --resume-from experiments/my_acoustic_exp/model-best-steps=100000-epochs=50.ckpt
```

### 5.7 SFC Breathness 指标解读

训练时进度条会显示所有 loss 指标。SFC 相关指标共 **12+** 个，分 3 类：

#### A. 基础 v3 指标

| 指标 | 正常范围 | 含义 |
|------|:------:|------|
| `diff_spec_loss` | 0.2~3.0 | Reflow 速度场 L2 loss，主模型核心指标 |
| `aux_spec_loss` | 0.02~0.08 | ConvNeXt aux decoder L1 loss |

#### B. SFC 气声损失（BrathyAwareLoss ×5）

**全部受 warmup 控制（step 0→64000 线性增长）。**

| 指标 | warmup 期（0~64k） | 全量期（64k+） | 含义 |
|------|:------:|:------:|------|
| `brathy_total` | 从 0 线性增长 | 0.5~8 | 气声损失加权总和 |
| `brathy/mel_l1_loss` | 同上 | 0.2~2 | 加权 mel L1 |
| `brathy/high_freq_loss` | 同上 | 0.05~0.5 | 高频段额外惩罚 |
| `brathy/spectral_tilt_loss` | 同上 | 0.05~0.3 | H1-H2 频谱倾斜约束 |
| `brathy/breathiness_loss` | **恒为 0** | 等 var 训练后激活 | breathiness pred vs GT（acoustic 训练中 pred==gt） |
| `brathy/voicing_loss` | **恒为 0** | 等 var 训练后激活 | voicing pred vs GT（同上） |

#### C. SFC 新模块损失（DDSP / CVAE / Perceptual / Glottal）

所有 SFC 模块从训练开始即激活，各自通过 `loss_scale` 和内部的 per-module warmup 控制权重。

| 指标 | 激活时机 | loss_scale | 含义 |
|------|:--:|:--:|------|
| `glottal_breathiness_loss` | 立即 | 0.5 | Adapter breathiness vs GT L1 |
| `ddsp_diffusion_loss` | 立即 | 1.0 | DDSP excitation+fusion mel vs GT |
| `period_singer_loss` | 立即 | 0.3 | CVAE aperiodic mel 重建 + KL |
| `perceptual_loss` | 立即 | 0.3 | H1H2/HNR/AirEnergy/NoiseMod/Fant（通过 per-module warmup 控制子分量激活节奏） |

#### 正常训练阶段时间线

```
步数 0~1k:    diff_spec 2.9→0.3 快速下降
                aux_spec 0.03~0.04 休眠期（LayerScale 1e-6，输出全零）

步数 1k~3k:   ConvNeXt LayerScale 逐层唤醒
                aux_spec 0.03→0.5→0.77 多次爬坡（6 层分波醒来）

步数 3k~5k:   ★ FS2 encoder 相变 ★
                transformer 3-4 层醒来 → cond 彻底重构
                aux_spec 0.77→0.23 crash → diff_spec 0.28→1.27

步数 5k~30k:  diff_spec 缓慢回落至 0.12~0.25
                aux_spec 0.02~0.04 稳定，cond 质量成熟

步数 30k~:    SFC 子模块持续收敛，per-module warmup 逐步释放
                显存 ~18GB

步数 90k:     即可捞模型
步数 120k:    满训
```

#### 异常情况诊断

| 症状 | 可能原因 | 解决 |
|------|------|------|
| `aux_spec_loss` 1k~3k 步多段爬坡 | ConvNeXt 6 层 LayerScale=1e-6 逐层唤醒（**正常**） | 爬完后稳定至 0.02~0.06 |
| `diff_spec_loss` 3k→5k 步 0.3→1.3 | **FS2 encoder 相变**—transformer 高层醒来，cond 彻底重构（**正常**） | 5k~15k 步自然回落至 0.5 以下 |
| `aux_spec` 与 `diff_spec` 同时 crash | **互校信号**—aux crash 验证新 cond 更好，diff 暂时涨是因 backbone 需重新适配 | 继续跑，不是过拟合 |
| `aux_spec_loss` < 0.005 持续 | aux decoder 过拟合 | 检查 `dropout_rate=0.4` + `EMA=0.999` |
| 所有指标同时翻倍 | 梯度爆炸 | 停训，确认 `gradient_clip_val=3.0` + `lr=0.0002` |
| `brathy/*` 恒为 0 | warmup 期正常；全量后仍为 0 需检查 `sweet_breathy.enabled` |
| `breathiness_loss`/`voicing_loss` 全程 0 | **正常**—等 variance 模型训练后才有值 |
| 验证音频尖锐/全损 | 参见过拟合诊断 [§5.9](#59-梯度爆炸诊断流程) |
| `period_singer_loss` NaN | VR 分离长度不匹配 | 已用 `aligned_t=min(T_cond,T_ap,T_rd)` 修复 |

### 5.8 多说话人训练参数推荐

多说话人（≥3 人）训练时梯度竞争加剧，比单说话人更容易在 4k~8k 步出现训练不稳→梯度爆炸。推荐以下配置：

```yaml
training:
  optimizer:
    cls: torch.optim.AdamW
    kwargs:
      lr: 0.0004           
      betas: [0.9, 0.98]
      weight_decay: 0
    # 差异化学习率（多模块训练必须）
    aux_lr_scale: 1.0      # 1.0 标准速度（原 0.5 让 ConvNeXt 爬坡缓慢，拖累 reflow 扩散起点）
    ddsp_lr_scale: 1.0     # DDSPGlottalSource 标准速度
    cvae_lr_scale: 1.0     # AperiodicCVAE 标准速度
    glottal_lr_scale: 2.0  # GlottalPredictor 提权（LF 模型小）

  trainer:
    gradient_clip_val: 3.0 # 单说话人 1.0，多说话人升到 3.0
    accumulate_grad_batches: 4  # 7 路 loss 梯度竞争 ×4 平滑

  weight_averaging:
    ema_enabled: true      # EMA 平滑权重
    ema_decay: 0.999

  loss:
    sweet_breathy:
      period_singer_loss_scale: 0.3   # CVAE loss 压降至 30%
      perceptual_loss_scale: 0.3      # 感知 loss 压降至 30%
      warmup_full_step: 64000

model:
  spec_decoder:
    aux_decoder_kwargs:
      dropout_rate: 0.4    # 单说话人 0.1，多说话人升到 0.4
    backbone_kwargs:
      dropout_rate: 0.1    # 单说话人 0.0，多说话人开到 0.1

  sweet_breathy:
    ddsp_gru_dropout: 0.2      # DDSP GRU 正则化
    ddsp_output_dropout: 0.1   # DDSP 输出正则化
    cvae_dropout: 0.2          # CVAE encoder/decoder 正则化
```

不建议在 `--override` 传 lr，建议直接改 `data/config_acoustic.yaml` 里的数值。

### 5.9 梯度爆炸诊断流程

当验证音频出现尖锐噪音、全损音质或完全听不出人声时：

```bash
# 1. 看 aux_spec_loss 和 brathy_total 是否同时飙升
curl -s 'http://127.0.0.1:6006/data/plugin/scalars/scalars?tag=training/aux_spec_loss&run=experiments/<name>/lightning_logs/latest' | python3 -c "
import json,sys
d=json.load(sys.stdin)
for p in d[-20:]: print(f'step {int(p[1]):>8d} aux={p[2]:.4f}')
"

# 2. aux 从 <0.1 跳到 >4 且不回落 → 停训
pkill -f 'scripts/train.py'

# 3. 修改配置（参见 §5.8）后从零重新训练
#    不要用 pretraining_from — optimizer 状态已被污染
python scripts/train.py acoustic --config data/config_acoustic.yaml --exp-name fix
```

关键判据：**音频听感 > loss 数值**。如果 diff_wav 能听清人声，loss 数值高也继续训练。反之音频全损，立即停。

### 5.10 训练常见数值异常

| 现象 | 步数 | 原因 | 是否正常 |
|------|:--:|------|:--:|
| `aux_spec_loss` 200 步起持续上升 | 0~5k | ConvNeXt LayerScale=1e-6 冷启动爬坡 | ✅ 正常（爬完稳定至 0.02~0.06） |
| `diff_spec_loss` 600 步 spike 1.2→4 | ~600 | FS2 encoder 冷启动方向突变 | ✅ 正常（一次性，后续持续下降） |
| `brathy_total` = 0 | 0~20k | warmup 权重 < 31% | ✅ 正常 |
| `breathiness_loss` = `voicing_loss` = 0 | 全程 | acoustic 训练中 pred==gt | ✅ 正常（等 var 训练） |
| `ddsp_loss` / `glottal_b` / `perceptual` tag 不存在 | 0~30k | 阶段 1 gate 关闭 | ✅ 正常（30k 步后阶段 2 自动出现） |
| `period_singer_loss` NaN | 随机 | VR 噪声波形长度偏差 | 已修复（`aligned_t=min` 兜底） |
| CUDA RNN weight compaction warning | 0 | DDSP GRU 权重分片 | 已修复（`flatten_parameters()`） |


---

## 第 6 步：导出 ONNX 模型（可选）

### 6.1 环境要求

**导出 ONNX 模型需要 PyTorch 1.13.x 环境**。这是因为 ONNX 导出依赖于特定版本的 PyTorch 算子支持。

```bash
# 创建独立的 ONNX 导出环境（推荐）
conda create -n diffsinger-export python=3.10 -y
conda activate diffsinger-export

# 安装 PyTorch 1.13
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu116

# 安装其他依赖
pip install -r requirements.txt
pip install -r requirements-onnx.txt
```

### 6.2 导出声学模型

```bash
python scripts/export.py acoustic --exp my_acoustic_exp
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--exp` | 实验名称或前缀（必填，自动匹配 `experiments/` 下的目录） |
| `--ckpt` | 指定 checkpoint 步数（不填则使用最新的） |
| `--out` | 输出目录，默认 `artifacts/<exp>/` |
| `--freeze_gender` | 固化性别参数（-1.0 ~ 1.0，默认 0.0） |
| `--freeze_velocity` | 固化为默认速度（去除随机时间伸缩） |
| `--export_spk` | 导出指定说话人或混合（多说话人模型） |
| `--freeze_spk` | 固化单个说话人或混合（多说话人模型） |

示例：

```bash
# 导出最新 checkpoint，固化男声音色（gender=1.0）
python scripts/export.py acoustic --exp my_acoustic_exp --freeze_gender 1.0

# 导出指定步数的 checkpoint
python scripts/export.py acoustic --exp my_acoustic_exp --ckpt 120000
```

### 6.3 导出方差模型

```bash
python scripts/export.py variance --exp my_variance_exp
```

### 6.4 导出 NSF-HiFiGAN 声码器

```bash
python scripts/export.py nsf_hifigan --config checkpoints/nsf_hifigan_44.1k_hop512_128bin_2024.02/config.json
```

导出的 ONNX 模型可用于 OpenUTAU for DiffSinger、DiffScope 等下游工具。

#### 常见故障排查

| 症状 | 原因 | 解决 |
|------|------|------|
| ONNX 导出时 `strict=True` 报 missing glottal keys | checkpoint 是纯 v3 训练的 | 正常，导出器会自动检测并跳过 SFC 模块 |
| ONNX 模型不含 `glottal_blend` 输入 | checkpoint 无 glottal 权重 | 导出器打印 `Glottal weights detected: False` |
| ONNX 模型含 `glottal_blend` 输入 | SFC 权重存在，三图合并成功 | 导出器打印 `Glottal weights detected: True`，`dsconfig.yaml` 含 `use_glottal_blend: True` |
| OpenUTAU 报 `Unexpected input: breathiness, languages, tension, voicing` | OpenUTAU 0.1.565.0 不支持这些输入 | 导出器已自动 freeze 为零常量，更新 `acoustic_exporter.py` 即可 |
| ONNX 导出时 `ValueError: Output f0 is not present in g1` | ONNX compose bug | 导出器已绕过（手动图合并） |
| 导出的 ONNX 无 glottal 但有 dsconfig 标志 | 导出器版本过旧 | 更新 `deployment/exporters/acoustic_exporter.py` 和 `deployment/modules/glottal_export.py` |

---

## 第 7 步：breathy_pipeline.py 工具使用

`breathy_pipeline.py` 是 SFC Breathness 的核心数据处理工具，提供五个子命令，覆盖从数据筛选到质量评估的完整工作流。

所有命令需在 `conda activate diffsinger` 环境下运行。

### 7.1 filter —— 筛选气声数据

扫描 WAV 目录，逐文件提取 H1-H2、HNR、空气能量比、频谱倾斜四项声学特征，自动分类并输出质量索引 JSON。

```bash
python breathy_pipeline.py filter --data_dir ./raw_wavs --output_json ./breathy_index.json
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | WAV 文件所在目录（必填） | - |
| `--output_json` | 输出的 JSON 索引文件路径 | `breathy_index.json` |
| `--min_duration` | 最小音频时长（秒），短于此值跳过 | `1.0` |
| `--sr` | 目标采样率 | `44100` |

输出 JSON 中包含每个文件的分类标签（`sweet_breathy` / `light_breathy` / `modal` / `over_breathy`）、置信度分数、声学特征值和条件命中情况，以及总体 sweet_breathy 占比。

### 7.2 extract —— 提取声学特征

批量提取 WAV 文件的四维声学特征（H1-H2 dB、HNR dB、空气能量比、频谱倾斜），输出为 JSON 或 CSV。

```bash
python breathy_pipeline.py extract --data_dir ./wavs --output_dir ./features --format json
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | WAV 文件所在目录（必填） | - |
| `--output_dir` | 输出目录 | `./features` |
| `--format` | 输出格式：`json` 或 `csv` | `json` |
| `--sr` | 目标采样率 | `44100` |

### 7.3 inject —— 注入气声/发声/张力曲线

向 `.ds` 二进制文件中注入 breathiness / voicing / tension 控制曲线。

```bash
# 从 JSON 文件加载曲线数据
python breathy_pipeline.py inject --ds_path original.ds --output breathy.ds --breathiness_json curves.json

# 使用单一浮点值创建均匀曲线
python breathy_pipeline.py inject --ds_path original.ds --output breathy.ds \
    --breathiness_value 0.25 --voicing_value 0.0 --tension_value -0.15
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--ds_path` | 原始 .ds 文件路径（必填） |
| `--output` | 输出 .ds 文件路径（必填） |
| `--breathiness_json` | 包含 breathiness/voicing/tension 曲线数据的 JSON 文件 |
| `--breathiness_value` | 单一 breathiness 值（0.0 ~ 1.0） |
| `--voicing_value` | 单一 voicing 值（0.0 ~ 1.0） |
| `--tension_value` | 单一 tension 值（0.0 ~ 1.0） |

### 7.4 eval —— 评估模型输出

对比模型合成音频与参考音频的气声特征，输出误差指标和等级评分。

```bash
python breathy_pipeline.py eval --pred output.wav --ref reference.wav
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pred` | 模型合成音频路径（必填） | - |
| `--ref` | 参考音频路径（必填） | - |
| `--sr` | 目标采样率 | `44100` |

输出包括：预测/参考的分类标签、四项声学特征的绝对误差、总体误差和等级评分。

等级评分标准：

| 等级 | 总体误差 | 评价 |
|:----:|---------|------|
| A | < 0.15 | 气声质量优秀 |
| B | < 0.30 | 气声质量良好 |
| C | < 0.50 | 气声质量一般，需要改进 |
| D | >= 0.50 | 气声质量差，需要大幅改进 |

### 7.5 augment —— 气声数据增强

对 WAV 文件应用气声曲线增强，包括动态波动、幅度缩放、幅度偏移三种策略。

```bash
python breathy_pipeline.py augment --data_dir ./wavs --augment_type scale --output_dir ./augmented
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | WAV 文件所在目录（必填） | - |
| `--augment_type` | 增强类型：`dynamic` / `scale` / `shift` | `scale` |
| `--output_dir` | 输出目录 | `./augmented` |
| `--sr` | 目标采样率 | `44100` |
| `--seed` | 随机种子 | `42` |

增强类型说明：

| 类型 | 原理 | 曲线裁剪范围 |
|------|------|:----------:|
| `dynamic` | 叠加低频随机波动到幅度包络 | [0.85, 1.15] |
| `scale` | 随机幅度缩放 0.85x ~ 1.15x | [0.3, 0.85] |
| `shift` | 随机音量偏移 +-0.5 dB | [0.3, 0.85] |

### 7.6 Python API

`breathy_pipeline.py` 同时提供了可在代码中直接调用的 Python API：

```python
from breathy_pipeline import build_breathy_index, evaluate_model_breathy_output

# 构建气声质量索引
index = build_breathy_index("./raw_wavs", min_duration=1.0)
print(f"Sweet breathy ratio: {index['sweet_breathy_ratio']:.1%}")

# 评估模型气声输出
result = evaluate_model_breathy_output("output.wav", "reference.wav")
print(f"Grade: {result['grade']}, Overall error: {result['overall_error']}")
```

---

## 第 8 步：Glottal Blend 参数

### 8.1 参数说明

Glottal Blend 是 v3-sfcb 新增的帧级可编辑参数，直接控制声门模型对扩散条件的影响量。

| 值 | 效果 |
|:--:|------|
| **0**（默认） | 无 glottal 干预，原生嗓音 |
| **+0.5** | 半量气声注入 |
| **+1** | 全量气声注入（训练中的最大喘度） |
| **-0.5** | 半量扣除 |
| **-1** | 反向扣除 glottal（超干净/紧闭） |
| **-1→+1 范围** | 纯线性乘法 (无 clamp) | condition + glottal_cond × blend |

与现有 `breathiness` 参数完全独立，两者效果叠加。

### 8.2 .ds 文件格式

在参数段中添加两行（`timestep` 建议与 `f0_timestep` 一致）：

```
glottal_blend: 0.0 0.0 0.3 0.6 0.8 0.5 0.2 0.0
glottal_blend_timestep: 0.01
```

每条参数值以空格分隔，帧数 = `值个数 × timestep / hop_size`。不需要写该参数时，模型行为与原来完全一致。

### 8.3 效果预期

与现有 breathiness 配合使用可实现精细气声控制：

| breathiness | glottal_blend | 效果 |
|:--:|:--:|------|
| 0（variance 默认） | +0.8 | 仅声门空气感，不放大人声参数 |
| 0.6 | +0.5 | 声门+人声参数双重气声 |
| 1.0 | -0.3 | 抵消部分声门气声，保持可控 |

#### 实测验证 (38k step checkpoint)
| 场景 | 差异 | 说明 |
|------|:--:|------|
| blend=0 vs blend=+1 | ~0.6 (mean abs diff) | glottal 注入有效 |
| blend=+1 vs blend=-1 | ~1.2 (完美 2×) | 全范围对称 |
| blend=None vs blend=+1 | ~0.0 | 默认满注入 |

> **默认行为**：不传 `glottal_blend` 时模型默认满格注入（等同 +1）。如需回退到纯 v3 行为，传入全 0 曲线。

---

## 常见问题 FAQ

### Q1: 训练时 GPU 内存不足怎么办？

**A:** 可以尝试以下措施：
- 减小 `batch_size`（在配置文件的 `training` 部分调整）
- 启用梯度累积（增大 `accumulate_grad_batches`）
- 使用混合精度训练（设置 `precision: 16-mixed`）
- 降低音频片段长度（减小 `binarizer.max_input_length`）

### Q2: 预处理时报音素覆盖错误？

**A:** 这表示你的标注数据中包含不在词典中的音素。解决方案：
- 使用 `--coverage-check-option bypass` 跳过检查
- 使用 `--coverage-check-option compat` 自动移除未知音素
- 补充 `dictionaries/` 目录下的词典文件

### Q3: 如何确认气声增强是否生效？

**A:** 有以下几种方式：
- 训练日志中检查 `sweet_breathy` 相关的 loss 项是否在下降
- 使用 `breathy_pipeline.py eval` 对比合成输出与参考气声样本
- 检查 `configs/my_experiment.yaml` 或基础配置中 `model.sweet_breathy.enabled` 是否为 `true`
- 推理时尝试不同 `breathiness` 值，对比听感差异

### Q4: 导出的 ONNX 模型加载失败？

**A:** ONNX 导出严格要求 PyTorch 1.13.x 环境，请确认：
- 使用的是 `scripts/export.py` 而非手动导出
- PyTorch 版本确为 1.13.x（通过 `python -c "import torch; print(torch.__version__)"` 检查）
- 已安装 `requirements-onnx.txt` 中的依赖
- 如果 glottal 模型导出报错，确认 `deployment/modules/glottal_export.py` 和 `deployment/exporters/acoustic_exporter.py` 为最新版本（三图合并）

### Q5: 训练过程中 loss 不下降或出现 NaN？

**A:** 分情况处理：
- **brathy/* 指标从 0 开始，前 64k 步一直很低** — 这是正常行为，warmup 线性释放中
- **breathiness/voicing loss 恒为 0** — **正常**，等 variance 模型训练后才激活
- **ddsp / glottal_b / period_singer / perceptual tag 不存在** — 所有 SFC 模块从训练开始即激活，如缺失请检查 config_acoustic.yaml 中对应模块是否 `enabled: true`
- **显存逐步增加** — **正常**，SFC 子模块 warmup 逐步释放时显存从 ~14GB 逐步升至 ~18GB
- **diff_spec 可能小幅反弹** — **正常**，SFC warmup ramp 接入
- **aux_spec_loss 200 步起持续上升，多段爬坡** — ConvNeXt 6 层 LayerScale=1e-6 逐层唤醒，3-5k 步 crash 回落
- **diff_spec_loss ~3k 步 0.3→1.3 反跳** — **FS2 encoder 相变**，transformer 高层醒来 cond 彻底重构，5k→15k 步自然回落
- **aux crash 与 diff fix 同时发生** — **互校信号**，aux 验证新 cond 更好，backbone 需重新适配（不是过拟合）
- **aux_spec_loss 飙升至 > 3 且音频尖锐/全损** — 过拟合→梯度悬崖，参见 [§5.9](#59-梯度爆炸诊断流程)
- **period_singer_loss NaN** — VR 分离长度不匹配（已用 `aligned_t=min` 修复）
- **所有指标一起连跳数倍** — 停训练，检查配置是否用了多说话人推荐参数（[§5.8](#58-多说话人训练参数推荐)），然后从零重训

### Q6: 如何制作自己的数据集？

**A:** 推荐使用 OpenVPI 的 [MakeDiffSinger](https://github.com/openvpi/MakeDiffSinger) 工具链，它提供：
- 自动音素对齐与标注生成
- 音频切分与质量检查
- 符合 DiffSinger 格式的 transcription 导出
- 支持多种语言和多种标注方案

### Q7: 气声效果不明显怎么办？

**A:**
- 检查训练数据中气声样本比例是否足够（建议 >= 30%），可使用 `breathy_pipeline.py filter` 评估
- 在 `acoustic_breathy_override.yaml` 中增大 `breathiness_weight` 和 `h1h2_weight`
- 推理时尝试 `heavy` 预设或手动提高 `breathiness` 参数
- 确保 warm-up 阶段已完成（默认在 64k 步时全量 brathy loss 生效，建议 100k 步后气声达到最佳效果）
- 考虑使用 `breathy_pipeline.py augment` 对训练数据进行气声增强

### Q8: 支持哪些语言？

**A:** 理论上支持任意语言，只需提供对应语言的发音词典（`dictionaries/` 目录）。项目默认包含 OpenCpop 扩展词典（中文）。对于英文、日文等语言，需自行准备音素词典。

### Q9: checkpoint 文件太大，如何只保留最优的？

**A:** 在配置文件的 `training.trainer.checkpoints` 部分设置：
- `save_top_k` 控制保留的 checkpoint 数量
- `weights_only: true` 可仅保存模型权重（减小体积，但无法用于断点续训）

### Q10: 推理时如何切换不同气声风格？

**A:** 使用 `configs/sweet_breathy/inference_preset.yaml` 中定义的预设：
- `light`：适合需要清晰谐波但带轻微气感的场景
- `sweet`：默认均衡预设，适合大多数甜蜜曲风
- `heavy`：适合需要浓厚气声质感的抒情曲风
- 也可在推理时手动指定 `breathiness`、`voicing`、`tension` 三个参数实现精细控制


---

## 训练稳定性修复（2026-05-27）

### 56k 步爆炸根因

训练至 56k 步时 aux_spec 从 0.014 飙升至 0.93，DDSP 从 0.17 飙升至 76.4。根因追踪：

1. **brathy warmup 达 87%**（56k/64k）→ aux_out 分布大幅偏移
2. **DDSP 使用 MSE loss** → 梯度 `2×error` 随误差线性增长
3. **perceptual loss 同步 warmup** → 8 路梯度在 ConvNeXt 参数上方向冲突
4. **DDSP 先爆**（56.7k DDSP=6.68）→ aux 后被拖垮（56.8k aux=0.73）

### 三项修复

#### 1. DDSP loss: MSE → Huber (SmoothL1Loss, β=1.0)

```
MSE:        gradient = 2 × error     → 误差 40→梯度 80  (爆炸)
Huber(β=1): gradient = error (|x|≤1) → 小误差L2精度
            gradient = ±1   (|x|>1)  → 大误差硬上限  (永不爆炸)
```

DDSP 梯度物理上限 ±1，任意 aux_out 偏移都无法引爆。

#### 2. FusionAdapter — 梯度解耦

```
修复前:
  aux_out → ConvNeXt ──┬── aux_spec      (L1)
                        ├── brathy ×5     (L1)
                        ├── DDSP ×3       (MSE 🔴)
                        ├── CVAE merged   (L1)
                        └── perceptual ×4 (L1)
  8 路梯度在 ConvNeXt 参数上方向冲突

修复后:
  aux_out ──→ ConvNeXt ──┬── aux_spec      (L1)  2 路同向:
                         └── brathy ×5     (L1)  "输出 GT mel"

  aux_out.detach() → FusionAdapter(656K) ──┬── DDSP ×3       (Huber)
                                           ├── CVAE merged   (L1)   6 路独立:
                                           └── perceptual ×4 (L1)   "做好融合组件"
```

FusionAdapter 是轻量残差 Conv1d 模块（128→512→128），初始近恒等变换（res_scale=0.1），完全由 DDSP/CVAE/Perceptual 训练，梯度不回流 ConvNeXt。

#### 3. aux_decoder_grad: 0.1（恢复 v3 基线正则化）

Huber 封死爆炸路径后，0.1 的编码器正则化安全可用。

### SFCB 功能确认

所有 SFCB 子模块训练不受影响：

| 模块 | 训练路径 | 状态 |
|------|------|:--:|
| GlottalPredictor | `cond.detach()` → glottal | ✅ |
| DDSPGlottalSource | `cond.detach()` → excitation | ✅ |
| SourceFilterFusion | 非参模块 | ✅ |
| AperiodicCVAE | 正常 L1 (period_singer) | ✅ |
| BrathyAwareLoss ×5 | 完整梯度 → ConvNeXt | ✅ |
| PerceptualLoss ×4 | 评估 FusionAdapter | ✅ |
| glottal_blend [-1,1] | 注入路径未改动 | ✅ |
| FusionAdapter 🆕 | DDSP/CVAE/Perceptual 共同训练 | ✅ |

---

## 训练稳定性修复（2026-05-30）

### 40k 步爆炸根因：perceptual 子分量双重累加

训练至 40k 步时，`aux_spec_loss` 从 0.015 暴涨至 0.49（33x），`perceptual_mel_l1` 从 0.98 归零，`ddsp_diffusion_loss` 从 0.66 飙至 7.89。

**根因代码**（`training/acoustic_module.py` L424-450）：

```python
# p_total 已经是 h1h2 + air + hnr + noise_mod + fant 的总和
p_total = perceptual_losses.pop('total', ...)
losses['perceptual_loss'] = p_total * wu * perc_scale   # ← 约 5.85

# 然后又把每个子分量单独加进去！
for k, v in perceptual_losses.items():
    v_safe = self.adaloss_norm.normalize(f'p_{k}', v)  # ← 每个 ~0.4-0.8
    losses[f'perceptual/{k}'] = v_safe                   # ← 合计 ~2.8
```

`training/pl_module_base.py` L302：`total_loss = sum(losses.values())` 将所有值累加 → **p_total 和子分量各被加一次 → 2x 梯度**。

**为什么 40k 才触发**：30k 前各 warmup coef 很小（0~0.4），双倍效应不到 1.0。40k 时 h1h2+air 都到 0.8，hnr 0.6，noise 0.4，fant 0.2，**五个 warmup 同时生效** → 双倍效应暴涨到 ~2.8 → 梯度爆炸 → norm/denorm 层 running stats 被打穿 → mel 输出全零 → 声码器收到近零 mel → 嗡嗡声。

### 修复

**`training/acoustic_module.py` L450** — 子分量脱离计算图：

```python
# 前:  losses[f'perceptual/{k}'] = v_safe
# 后:  losses[f'perceptual/{k}'] = v_safe.detach()
```

子分量只记录不参与梯度，`perceptual_loss`（`p_total * warmup * scale`）仍是完整感知损失。

| 衡量 | 修复前 | 修复后 |
|------|:--:|:--:|
| perceptual 对 total_loss 梯度贡献 | ~8.65（双倍） | ~4.4（仅 p_total） |
| TensorBoard 子分量可见 | ✅ | ✅（detach 不影响 log 值） |
| 梯度流向 | p_total + 5 个子分量 | 仅 p_total |

### 补充修复（6 个文件，同步上传）

| 文件 | 问题 | 修复 |
|------|------|------|
| `utils/__init__.py` | spec_decoder 键名映射缺失（ONNX 权重静默丢失） | L138-139: `spec_decoder.decoder→diffusion`, `spec_decoder.aux_decoder→aux_decoder` |
| `modules/aux_decoder/convnext.py` | ConvNeXtDecoder.outconv 无激活，aux 输出含离群值 | `torch.tanh(x)` 约束 [-1,1] |
| `deployment/modules/rectified_flow.py` | aux 路径二次 norm_spec | 跳过 norm_spec |
| `deployment/modules/toplevel.py` | glottal 通路未正确接入 ONNX 合并 | 独立 ONNX + 三图合并 |
| `deployment/exporters/acoustic_exporter.py` | glottal_blend 未作为外部输入 | 第 7 个外部输入 |
| `data/config_acoustic.yaml` | 无 weight_decay | `weight_decay: 1.0e-5` |

### 注意事项

- `.detach()` 修复后 **resume 从旧 checkpoint 会有 Adam 动量惯性残留**（aux_spec_loss 短暂上升 1.6x）。建议从零重新训练。
- 若必须 resume，可在 resume 后重置 optimizer state 清空历史动量。
- 本次修复的 aco_testf6 从零训练，预计 120k 步满训。
