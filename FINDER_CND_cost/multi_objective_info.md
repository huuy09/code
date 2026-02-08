# FINDER_CND_cost 多目标优化说明

## 概述
FINDER_CND_cost 模型实现了 CN (Component Number Density) 和 ND (Max Component) 两个目标的线性组合优化。

## Reward 计算公式

```cpp
reward = -(alpha × CN_normalized + (1-alpha) × ND_normalized) × weight_ratio
```

### 各项说明：

1. **CN_normalized**: 归一化的CND score
   - `CN_normalized = getRemainingCNDScore() / [N×(N-1)/2]`
   - 范围: [0, 1]
   - 衡量所有连通分量的"配对连通性"总和

2. **ND_normalized**: 归一化的最大连通分量
   - `ND_normalized = getMaxConnectedNodesNum() / N`
   - 范围: [0, 1]
   - 衡量最大连通分量的相对大小

3. **weight_ratio**: 节点权重比例
   - `weight_ratio = node_weight / total_weight`
   - 体现攻击代价

4. **alpha**: 权重参数（可调）
   - `alpha = 0.5`: 等权重组合（默认）
   - `alpha = 0.0`: 纯ND目标（等价于FINDER_ND_cost）
   - `alpha = 1.0`: 纯CN目标（等价于FINDER_CN_cost）
   - `alpha ∈ (0,1)`: 多目标平衡

## 归一化的重要性

两个指标都归一化到[0,1]范围，确保：
- **尺度一致**: CN和ND的数值范围相同，避免某个目标主导
- **可解释性**: Combined metric也在[0,1]范围，表示"剩余网络连通性"
- **稳定性**: 不同规模的图使用相同的组合权重alpha

## 调整 alpha 参数

修改 `src/lib/mvc_env.cpp` 第 307 行：

```cpp
double alpha = 0.5;  // 修改此值来调整CN和ND的权重
```

### 推荐设置：
- **平衡型** (alpha=0.5): 同时优化全局连通性和局部碎片化
- **全局优先** (alpha=0.3): 更关注最大连通分量（类似关键基础设施保护）
- **局部优先** (alpha=0.7): 更关注网络碎片化程度（类似疫情传播阻断）

## 与单目标模型的对比

| 模型 | CN权重 | ND权重 | 特点 |
|------|--------|--------|------|
| FINDER_CN_cost | 1.0 | 0.0 | 优化全网络碎片化 |
| FINDER_ND_cost | 0.0 | 1.0 | 优化最大连通分量 |
| FINDER_CND_cost | 0.5 | 0.5 | 平衡两个目标 |

## 训练建议

1. **先用alpha=0.5训练**: 获得平衡的基线模型
2. **实验不同alpha**: 尝试0.3, 0.5, 0.7等值
3. **任务导向**: 根据实际应用场景选择最优alpha
4. **性能评估**: 分别计算LCC和PC指标，观察权衡效果

## 代码位置

- **Reward计算**: `src/lib/mvc_env.cpp` 第 289-312 行
- **CN计算**: `src/lib/mvc_env.cpp` 第 390-424 行 (getRemainingCNDScore)
- **ND计算**: `src/lib/mvc_env.cpp` 第 367-388 行 (getMaxConnectedNodesNum)
