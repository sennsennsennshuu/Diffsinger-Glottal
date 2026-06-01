# Glottal Blend — 声门气声偏置帧级参数设计规范

日期：2026-05-25  
状态：待审查

---

## 1. 概述

为 DiffSinger v3-sfcb 的 ONNX 推理管线新增一个**帧级可编辑参数 Glottal Blend**，允许用户在 OpenUTAU 中以类似 Pitch Bend 曲线的方式逐帧控制气声 vs 发声平衡，范围 [-1, 1]。

该参数与现有方差模型参数（breathiness/voicing/tension）**完全独立**，两者可叠加使用，赋予用户双重控制气声效果的能力。

---

## 2. 目标

| 目标 | 说明 |
|------|------|
| **帧级曲线参数** | 用户可在 OpenUTAU 编辑器中以控制点方式逐帧绘制曲线 |
| **气声 vs 发声平衡** | 0=原生嗓音，+1=全量气声注入，-1=反向扣除→超干净 |
| **与现有参数独立** | 不和 breathiness/voicing/tension 冲突，用户可同时使用两者 |
| **不影响现有训练** | ONNX 导出路径的改动不接触训练代码 |
| **模型体积增量 < 0** | 不加新权重，仅加一个标量输入 |

---

## 3. 设计

### 3.1 数据流

```
推理时：
  cond + f0 → GlottalPredictor → {Rd,OQ,AQ,noise} → Adapter → breathiness
                                                                ↓
                                                            glottal_cond [B,T,256]
                                                                ↓ × glottal_blend [B,T]
                                                          cond = cond + glottal_cond × glottal_blend
                                                                ↓
                                                          DiffusionDecoder → mel
```

### 3.2 参数规格

| 属性 | 值 |
|------|------|
| 名称 | Glottal Blend |
| 类型 | `float32` 帧级曲线 |
| 张量形状 | `[B, T]` 或 `[1, T]` |
| 值域 | [-1, 1] |
| 默认值 | 0（无 glottal 干预） |
| 语义 | +1=最大化气声注入，0=原生，-1=反向扣除→紧闭 |

### 3.3 改动文件

| 文件 | 改动 | 影响训练 |
|------|------|:--:|
| `deployment/modules/toplevel.py` | `forward_fs2_aux()` 加可选参数 `glottal_blend` | ❌ 不动训练 |
| `deployment/modules/toplevel.py` | 新增 `_compute_glottal_cond()` 方法 | ❌ 同上 |
| `inference/ds_acoustic.py` | `preprocess_input()` 读取 glottal_blend 参数 | ❌ 同上 |
| `lib/config/schema.py` | `InferenceConfig` 加 `glottal_blend` 允许值域 | ❌ 同上 |

### 3.4 关键代码

#### `deployment/modules/toplevel.py`

```python
class DiffSingerAcousticONNX(DiffSingerAcoustic):
    def __init__(self, vocab_size, out_dims, cross_lingual_token_idx=None,
                 use_sweet_breathy: bool = True, condition_dim: int = 256):
        super().__init__(vocab_size, out_dims)
        del self.fs2
        del self.diffusion
        self.fs2 = FastSpeech2AcousticONNX(...)
        # ... 现有 diffusion 替换逻辑 ...

        # 显式创建 glottal 模块（因为 legacy __init__ 跳过了它们）
        if use_sweet_breathy:
            from modules.sweet_breathy.lf_glottal_conditioner import (
                GlottalParamPredictor, GlottalToDiffSingerAdapter,
            )
            self.glottal_predictor = GlottalParamPredictor(d_cond=condition_dim)
            self.glottal_adapter = GlottalToDiffSingerAdapter()
        else:
            self.glottal_predictor = None
            self.glottal_adapter = None
        self._glottal_proj = nn.Linear(1, condition_dim)

    def _compute_glottal_cond(self, cond: Tensor, f0: Tensor) -> Tensor | None:
        """Compute glottal condition residual. Returns None if glottal is disabled."""
        if self.glottal_predictor is None:
            return None
        lf_params = self.glottal_predictor(cond, f0)
        glottal_cond = torch.stack([
            self.glottal_adapter(
                lf_params['Rd'], lf_params['OQ'],
                lf_params['AQ'], lf_params['noise_amp']
            )[0],
        ], dim=-1)  # [B, T, 1]
        return self._glottal_proj.to(cond.device)(glottal_cond)  # [B, T, H]

    def forward_fs2_aux(self, ..., glottal_blend: Tensor = None):
        # ... 现有 condition 构建逻辑 ...
        if glottal_blend is not None:
            glottal_cond = self._compute_glottal_cond(cond, f0)
            if glottal_cond is not None:
                cond = cond + glottal_cond * glottal_blend.unsqueeze(-1)
        return ...
```

#### `inference/ds_acoustic.py`

```python
def preprocess_input(self, ...):
    batch = { ... }
    # 新增：从参数列表读取 glottal_blend
    if 'glottal_blend' in param_names:
        glottal_blend = param_values[param_names.index('glottal_blend')]
        batch['glottal_blend'] = torch.FloatTensor(glottal_blend).unsqueeze(0)
    else:
        batch['glottal_blend'] = None
    return batch
```

---

## 4. 与现有参数的关系

| 参数 | 来源 | 通路 | 与 Glottal Blend 关系 |
|------|------|------|------|
| **breathiness** | 方差模型预测 / 手动 | variance embedding → cond 残差 | 独立叠加 |
| **Glottal Blend** | 用户手动曲线 | GlottalPredictor→Adapter→cond 残差 | **新增，独立** |
| **voicing** | 方差模型预测 | variance embedding → cond 残差 | 独立 |
| **tension** | 方差模型预测 | variance embedding → cond 残差 | 独立 |

两条通路分别向 cond 注入不同的残差信号，diffusion decoder 会自然地融合两者的影响。

---

## 5. 非目标

- **不替换**现有 breathiness/voicing/tension 参数
- **不新增**训练阶段的 forward/backward 逻辑
- **不增加**模型参数量（只加一个 ONNX 可选输入）
- **不改变** GlottalPredictor 或 Adapter 的结构

---

## 6. 测试策略

| 测试项 | 方法 |
|------|------|
| `_compute_glottal_cond` 输出形状 | 单元测试：输入 `[B,T,256]` cond + `[B,T]` f0，输出 `[B,T,256]` |
| `glottal_blend=0` 等价原生 | 对比 blend=0 与完全不传 blend 的 mel 输出 |
| `glottal_blend=+1` 气声增强 | 对比 mel 频谱高频差异 |
| `glottal_blend=-1` 闭合嗓音 | 对比 mel 频谱低频差异 |
| ONNX 导出不报错 | `scripts/export.py` 加 `--extra-param glottal_blend` |

---

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| `glottal_blend` 外推到训练时未见的范围导致失真 | 文档建议用户先从小范围 (±0.5) 开始试 |
| GlottalPredictor 在 ONNX 中未导出 | 在 __init__ 中完整创建 glottal_predictor/adapter/_glottal_proj |
| OpenUTAU 后端不支持新参数名 | 与 OpenUTAU 社区确认参数注册方式 |
