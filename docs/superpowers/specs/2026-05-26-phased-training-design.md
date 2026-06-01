# 分阶段训练：根治多模块梯度冲突与 loss 反向增长

**日期**: 2026-05-26  
**状态**: 设计中

---

## 问题

SFCB 引入 DDSP + CVAE + Perceptual 三个新模块后，7 路 loss 在当前 cond 上同时回传。cond 质量早期很差（diff_spec ~3+），多路梯度冲突导致：

1. diff_spec_loss 反向增长（0.2 → 1.4）
2. 音频爆鸣持续到 3k+ 步
3. aux_spec_loss 爬坡峰值过高

之前在 4k~8k 步尝试过"基础模型先跑、再加模块"，但当时没有 detach DDSP/CVAE cond，基础模型本身也在 4k~8k 过拟合爆炸。现在 detach 已修复 cond 污染问题，分阶段训练不再被过拟合阻挡。

## 方案：两阶段训练

| | 阶段 1 (0-30k) | 阶段 2 (30k+) |
|---|:---:|:---:|
| diff_spec_loss | ✅ | ✅ |
| aux_spec_loss | ✅ | ✅ |
| brathy_total | ✅ warmup 0→64k | ✅ |
| glottal_b_loss | ❌ | ✅ |
| ddsp_loss | ❌ | ✅ (cond.detach) |
| period_singer_loss | ❌ | ✅ (cond.detach) |
| perceptual_loss | ❌ | ✅ 分段 warmup |
| glottal cond 注入 | ✅ (不参与 loss) | ✅ |
| DDSP/CVAE forward | ❌ (省显存) | ✅ |

- 阶段 1 等同纯 v3 + brathy mel loss + glottal cond 注入（无梯度回传），30k 步时 cond 质量已成熟
- 阶段 2 激活全部 SFC 模块，DDSP/CVAE 吃 `cond.detach()`，感知 loss 分段 warmup 从 30k 起算

## 配置新增 key

```yaml
training.loss.sweet_breathy:
  phase1_end_step: 30000        # 阶段 1 结束步
  phase2_start_step: 30000      # 阶段 2 开始步
  warmup_full_step: 64000       # brathy warmup（不变）
```

Perceptual 子 warmup 在阶段 2 起点重新计算：
- 30k→60k: H1H2 + AirEnergy
- 60k→100k: HNR + NoiseMod
- 100k→140k: Fant

## 文件改动

| 文件 | 改动 |
|------|------|
| `lib/config/schema.py` | `SweetBreathyLossConfig` + `phase1_end_step`, `phase2_start_step` |
| `data/config_acoustic.yaml` | 写死 phase 值 |
| `modules/toplevel.py` | `forward()` + `phase` 参数；phase=1 跳过 DDSP/CVAE |
| `training/acoustic_module.py` | `_get_phase()` + 传入 model + loss 注册按 phase gate |

## 预期效果

- diff_spec 阶段 1 稳定下降到 0.3-0.7（30k 步）
- 阶段 2 切换时 diff 可能小幅反弹后继续下降
- 纯 v3 阶段显存省 4-5GB，阶段 2 恢复到完整显存
- 音频 10k 步能听，30k 步接近原版音质
- 最终（140k+）达到原版精度 + 完整 SFCB 气声增强
