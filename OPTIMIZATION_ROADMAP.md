# 中国象棋 AI 优化路线图

## 项目现状

### 已完成 ✅
- 人类棋谱监督学习 (99,813 局)
- 分批训练支持 (内存友好)
- 网络 + 搜索整合
- Web GUI 对弈界面
- UCCI 协议支持

### 当前棋力
- 纯网络: 业余 3-4 段
- 网络+搜索: 业余 5-6 段

---

## Phase 1: 基础强化 (1-2周)

### P0-1: 动作空间精简 (8100 → 2086)
**收益**: 训练效率提升 30%+

```python
# 精简有效动作：
# - 直线移动（车/炮/将/兵）
# - 马的8方向
# - 象的4方向  
# - 士的4方向
```

### P0-2: Pikafish 蒸馏
**收益**: 标签质量大幅提升

```bash
# 用 Pikafish 生成软标签
python generate_soft_labels.py --engine pikafish --depth 18

# 蒸馏训练
python train_distill.py --soft_labels
```

### P0-3: Elo 评分系统
**收益**: 量化棋力进步

```python
# 自动对弈评估
python evaluate_model.py --vs baseline --games 50
# 输出 Elo 评分和胜率
```

---

## Phase 2: 性能优化 (2-3周)

### P1-1: NN Cache
- LRU 缓存神经网络输出
- 避免重复局面重复计算
- 预计命中率 30-50%

### P1-2: 并行 MCTS
- 多线程模拟
- 虚拟损失避免重复路径
- 4线程提升 3x 搜索效率

### P1-3: 批量推理优化
- GPU 批量推理
- 动态 batching

---

## Phase 3: 工程化 (2-4周)

### P2-1: 分布式训练
- Master/Slave 架构
- 多机并行自对弈
- 自动模型更新

### P2-2: 完整 UCCI
- 对接各种象棋 GUI
- 支持 ponder/info 输出
- 时间管理

---

## Phase 4: 进阶探索 (长期)

| 方向 | 说明 | 难度 |
|---|---|---|
| Transformer | 替换 CNN 骨干 | ⭐⭐⭐ |
| NNUE | 增量评估网络 | ⭐⭐⭐⭐ |
| C++ 重写 | 极致性能 | ⭐⭐⭐⭐⭐ |

---

## 优先级排序

| 优先级 | 任务 | 工时 | 收益 |
|---|---|---|---|
| **P0** | 动作空间精简 | 3天 | 训练效率+30% |
| **P0** | Pikafish蒸馏 | 5天 | 标签质量++ |
| **P0** | Elo系统 | 2天 | 量化进步 |
| P1 | NN Cache | 2天 | 推理效率+ |
| P1 | 并行MCTS | 5天 | 搜索效率+ |
| P2 | 分布式 | 7天 | 大规模训练 |

---

## 下一步行动

```bash
# 1. 完成当前监督训练
python train_batch.py

# 2. 测试当前模型
python chess_server.py --net checkpoints_human/best_model.pt

# 3. 后续优化按 Phase 逐步推进
```
