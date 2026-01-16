# 中国象棋 AI 🎮

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.9+-yellow.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.9+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Docker-Supported-blue.svg" alt="Docker">
</p>

<p align="center">
  <b>业余5-6段棋力</b> · PyTorch 训练 · Web 对弈界面 · Docker 一键部署
</p>

---

## ✨ 特点

- 🎯 **99,813 局大师棋谱训练** - 学习人类顶尖走法
- 🧠 **神经网络 + Alpha-Beta 搜索** - 混合架构，兼顾速度与棋力
- 🎮 **即开即用的 Web 界面** - 浏览器直接对弈
- 🐳 **Docker 支持** - 一行命令启动服务
- 📊 **完整训练流水线** - 从棋谱到模型一键完成

---

## 🚀 快速开始

### 方式一：使用 Docker (推荐)

如果您安装了 Docker，这是最简单的启动方式：

```bash
# 构建并启动服务
docker-compose up --build
```

启动后，打开浏览器访问 [http://localhost:8080](http://localhost:8080) 即可开始对弈。

### 方式二：手动安装

1. **克隆代码**
   ```bash
   git clone https://github.com/your-username/xiangqi-engine.git
   cd xiangqi-engine
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **启动服务器**
   ```bash
   python chess_server.py
   # 打开浏览器访问 http://localhost:8080
   ```

### 使用预训练模型

```bash
# Docker 方式
docker-compose run chess-engine python chess_server.py --net checkpoints_human/best_model.pt

# 手动方式
python chess_server.py --net checkpoints_human/best_model.pt
```

---

## 🏋️ 训练自己的模型

### 下载棋谱

从 [东萍象棋](http://www.dpxq.com/) 下载大师棋谱，放入 `pgn_data/` 目录。

### 开始训练

```bash
# 分批训练（推荐，内存友好）
python train_batch.py

# 或 GPU 一次性加载（需要 64GB+ 内存）
python train_human_supervised.py --epochs 15
```

### 训练效果

| 训练数据 | 准确率 | 棋力 |
|---|---|---|
| 10,000 局 | ~26% | 业余中级 |
| 99,813 局 | ~35% | 业余高段 |
| +搜索整合 | - | 业余5-6段 |

---

## 💻 开发指南

### 环境准备

建议使用 Python 3.9+ 环境。

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -r requirements.txt
```

### 项目结构

```
xiangqi-engine/
├── src/                    # 核心代码
│   ├── board.py            # 棋盘表示
│   ├── move.py             # 着法生成
│   ├── search_v2.py        # Alpha-Beta 搜索
│   └── policy_value_net.py # 神经网络
├── train_batch.py          # 分批训练脚本
├── train_human_supervised.py # 监督学习
├── chess_server.py         # Web 服务器
├── chess_gui_connected.html # Web 界面
├── Dockerfile              # Docker 构建文件
├── docker-compose.yml      # Docker Compose 配置
└── ucci.py                 # UCCI 协议支持
```

---

## 🛠️ 技术栈

- **深度学习**: PyTorch
- **搜索算法**: Alpha-Beta + 置换表 + Killer 启发
- **神经网络**: CNN Policy-Value 网络
- **Web 服务**: Python HTTP Server
- **部署**: Docker

---

## 📈 路线图

- [x] 人类棋谱监督学习
- [x] Web 对弈界面
- [x] UCCI 协议支持
- [x] Docker 容器化部署
- [ ] Elo 评分系统
- [ ] Pikafish 蒸馏
- [ ] 分布式训练

---

## 🤝 贡献

欢迎 PR 和 Issue！

---

## 📄 许可证

MIT License
