# 中国象棋 AI 完整训练指南

## 快速开始

### 本地分批训练（推荐）

```bash
# 内存友好的分批训练（每批 3000 局）
python train_batch.py --epochs_per_batch 3

# 训练全部 99,813 局，约 2-3 小时
```

### GPU 云训练

```bash
# 1. 租用 AutoDL（~15元）
# 2. 上传项目 + pgn_data/
# 3. 运行：
python train_human_supervised.py --epochs 15
```

---

## 训练流程

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  阶段1: 监督学习  │ →  │  阶段2: 搜索整合  │ →  │  阶段3: 蒸馏闭环  │
│  (人类棋谱)       │    │  (网络+搜索)      │    │  (自我提升)       │
│  ~2小时          │    │  即时生效        │    │  可选             │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        ↓                      ↓                      ↓
   准确率 ~35%            业余5-6段              可超越人类
```

---

## 完整命令清单

```bash
# 阶段1: 监督学习
python train_batch.py                    # 本地分批（推荐）
python train_human_supervised.py         # GPU 一次性加载

# 阶段2: 测试 AI
python chess_server.py --net checkpoints_human/best_model.pt
# 浏览器 http://localhost:8080

# 阶段3: 蒸馏闭环（可选）
python train_distill.py --games 100 --depth 3
```

---

## 预期效果

| 阶段 | 准确率 | 棋力 | 时间 |
|---|---|---|---|
| 监督完成 | ~35% | 业余3-4段 | 2小时 |
| +搜索整合 | - | 业余5-6段 | 即时 |
| +蒸馏闭环 | ~40% | 接近专业 | +5小时 |

---

## GPU 云服务器配置

### 推荐平台
- **[AutoDL](https://www.autodl.com)** - 国内首选，~2元/小时

### 推荐配置
```
GPU: RTX 3090 / A10
显存: 24GB
内存: 64GB+
存储: 50GB
预计费用: ~15-20元
```

### 上传步骤
```bash
# 打包（排除大文件）
zip -r xiangqi.zip . -x "*.pt" -x "__pycache__/*"

# 上传文件：
# - xiangqi.zip
# - pgn_data/dpxq-99813games.pgns (100MB)
```

---

## 常见问题

### 内存不足？
```bash
python train_batch.py --games_per_batch 2000  # 减少每批大小
```

### 训练太慢？
```bash
python -c "import torch; print(torch.cuda.is_available())"
# 应显示 True
```

---

## 下一步优化（可选）

详见 `OPTIMIZATION_ROADMAP.md`
