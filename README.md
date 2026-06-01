# DiffSinger v3 — LF Glottal Conditioner

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2105.02446)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/openvpi/DiffSinger/blob/main/LICENSE)

基于 [DiffSinger (OpenVPI)](https://github.com/openvpi/DiffSinger) v3 架构，引入 **LF 声门条件器 (Glottal Conditioner)**，通过物理可微声门脉冲模型为歌唱合成注入可控的气声 (breathiness) 特征。

## 核心特性

- **物理可微声门模型**: 集成 Liljencrants-Fant (LF) 声门脉冲模型作为可微分层，实现从物理参数到频谱的端到端训练
- **物理约束硬编码**: Fant 1997 的 OQ-Rd 约束 (`OQ ≤ 0.11·Rd + 0.55`) 通过 `torch.min()` 实现，确保预测参数在物理可行域内
- **双重气声控制**: 方差模型的 breathiness（数据驱动）+ Glottal Blend（物理驱动）独立叠加，推理时可通过帧级参数 `glottal_blend ∈ [-1, 1]` 精细调节
- **差异化学习率**: 声门预测器 2× 基础 LR，LynxNet/FS2 1×，解决小模块与大模块的梯度竞争
- **多层 Warmup**: 全局 warmup + 每个损失子组件独立 warmup + glottal_cond 注入 warmup（100步线性渐增）
- **AdaLossNorm**: EMA 归一化消除不同损失间 16× 的量级差异
- **ONNX 兼容**: `GlottalConditionerONNX` 封装，供 OpenUTAU 等部署环境使用

## 架构概览

```
cond [B,T,256] + f0 [B,T]
        │
        ▼
┌─────────────────────────────┐
│   GlottalParamPredictor     │  2层 Transformer + 4独立预测头
│   (cond+f0 → Rd,OQ,AQ,na)  │  Rd∈[0.3,2.7], OQ 受 Fant 约束
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│ GlottalToDiffSingerAdapter  │  可学习 sigmoid 映射
│ (Rd,OQ,AQ,na → breathiness)│  breathiness ∈ [0,1]
└──────────┬──────────────────┘
           ▼
     breathiness [B,T]
           │
           ▼
   Linear(1 → 256) + warmup
           │
           ▼
     glottal_cond [B,T,256]
           │
           ▼
     cond = cond + glottal_cond
           │
           ▼
     DiffusionDecoder → mel
```

## 模块说明

| 模块 | 文件 | 说明 |
|------|------|------|
| LF 声门条件器 | `modules/sweet_breathy/lf_glottal_conditioner.py` | GlottalParamPredictor + GlottalToDiffSingerAdapter + LFGlottalModel |
| 声学模型 | `modules/toplevel.py` | DiffSingerAcoustic，含 glottal forward 路径和 warmup 机制 |
| 气声损失 | `modules/losses/breathy_loss.py` | BrathyAwareLoss 多组件气声感知损失 |
| 声学特征 | `lib/feature/glottal.py` | H1-H2, HNR, 空气能量比, 频谱倾斜提取 |
| ONNX 导出 | `deployment/modules/glottal_export.py` | 可导出的声门条件器封装 |
| 训练模块 | `training/acoustic_module.py` | 分组 LR、AdaLossNorm、glottal_breathiness_loss |

## 训练配置

### 优化器

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 基础 LR | 6×10⁻⁴ |
| Betas | [0.9, 0.98] |
| Weight Decay | 0 |
| Glottal LR | 2× 基础 LR |
| LR 调度 | StepLR, step=50000, γ=0.5 |
| 总步数 | 120,000 |
| 精度 | FP16 混合精度 |
| 梯度裁剪 | 3.0 |

### 损失路径

| 损失 | 权重 | Warmup |
|------|------|--------|
| `diff_spec_loss` (Rectified Flow) | L2 | — |
| `aux_spec_loss` (浅扩散) | 0.2× | — |
| `glottal_breathiness_loss` | 0.5× | [0, 5000] |
| `brathy_total` (BrathyAwareLoss) | 动态 | [20000, 64000] |

### 关键配置文件

- `data/config_acoustic.yaml` — 完整训练配置（模型、优化器、损失、数据）
- `configs/acoustic_v3.yaml` — v3 模型配置模板
- `configs/base_v3.yaml` — v3 基础默认值

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

### 安装

```bash
pip install -r requirements.txt
```

### 预处理

```bash
python scripts/binarize.py acoustic --config data/config_acoustic.yaml
```

### 训练

```bash
python scripts/train.py acoustic --config data/config_acoustic.yaml --exp-name v3_glottal
```

### 推理

```bash
python scripts/inference.py acoustic --config data/config_acoustic.yaml \
    --exp-name v3_glottal --restored-step 120000 \
    --text "歌词文本" --notes "音符序列"
```

### 测试

```bash
pytest tests/test_lf_glottal.py tests/test_glottal.py tests/test_glottal_export.py -v
```

## 设计文档

| 文档 | 内容 |
|------|------|
| `docs/specs/2026-05-25-glottal-blend-design.md` | Glottal Blend 帧级参数设计规范 |
| `docs/superpowers/specs/2026-05-26-phased-training-design.md` | 分阶段训练设计（解决多模块梯度冲突） |
| `docs/GettingStarted.md` | 入门指南 |
| `docs/BestPractices.md` | 最佳实践 |
| `docs/ConfigurationSchemas.md` | 配置参数 Schema |

## References

### Original Paper & Implementation

- Paper: [DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446)
- Implementation: [MoonInTheRiver/DiffSinger](https://github.com/MoonInTheRiver/DiffSinger)
- OpenVPI maintained version: [openvpi/DiffSinger](https://github.com/openvpi/DiffSinger)

### Glottal Model & Voice Quality

- LF Model: [Fant, G. (1997). "The voice source in connected speech"](https://www.sciencedirect.com/science/article/pii/S0167639397000219)
- Rd parameter: [Fant, G. & Liljencrants, J. (1985). "A four-parameter model of glottal flow"](https://www.speech.kth.se/prod/publications/files/qpsr/1985/1985_26_4_001-013.pdf)

### Generative Models & Algorithms

- Rectified Flow (RF): [paper](https://arxiv.org/abs/2209.03003), [implementation](https://github.com/gnobitab/RectifiedFlow)
- LynxNet backbone: custom lightweight U-Net for shallow diffusion

## Disclaimer

Any organization or individual is prohibited from using any functionalities included in this repository to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

## License

This forked DiffSinger repository is licensed under the [Apache 2.0 License](LICENSE).
