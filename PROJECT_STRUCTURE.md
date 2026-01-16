# 项目结构说明

## 推荐的发布结构

```
xiangqi-ai/                      # 发布仓库（干净版本）
├── README.md                    # 项目介绍
├── LICENSE                      # MIT 许可证
├── requirements.txt             # 依赖
├── .gitignore
│
├── xiangqi/                     # 核心包
│   ├── __init__.py
│   ├── board.py                 # 棋盘
│   ├── move.py                  # 着法
│   ├── fen.py                   # FEN 解析
│   ├── search.py                # 搜索引擎
│   ├── network.py               # 神经网络
│   ├── action.py                # 动作编码
│   └── evaluate.py              # 评估函数
│
├── scripts/                     # 脚本
│   ├── train.py                 # 训练脚本
│   ├── play.py                  # 对弈脚本
│   └── evaluate.py              # 评估脚本
│
├── web/                         # Web 界面
│   ├── server.py                # 后端
│   └── index.html               # 前端
│
├── models/                      # 预训练模型（发布时可选）
│   └── .gitkeep
│
├── data/                        # 数据目录（不含实际数据）
│   └── README.md                # 说明如何下载数据
│
└── docs/                        # 文档
    ├── training.md              # 训练指南
    └── api.md                   # API 文档
```

---

## 当前开发结构 → 发布结构映射

| 当前文件 | 发布位置 | 说明 |
|---|---|---|
| `src/*.py` | `xiangqi/*.py` | 核心代码 |
| `train_batch.py` | `scripts/train.py` | 训练脚本 |
| `chess_server.py` | `web/server.py` | Web 后端 |
| `chess_gui_connected.html` | `web/index.html` | Web 前端 |
| `ucci.py` | `scripts/ucci.py` | UCCI 协议 |
| `pgn_data/` | 不发布 | 用户自行下载 |
| `checkpoints_*/` | 不发布 | 用户自行训练 |
| `backup_*/` | 不发布 | 开发备份 |

---

## 快速创建发布版本

```bash
# 1. 创建发布目录
mkdir xiangqi-ai-release
cd xiangqi-ai-release

# 2. 复制核心文件
cp -r ../xiangqi-engine/src ./xiangqi
cp ../xiangqi-engine/train_batch.py ./scripts/train.py
cp ../xiangqi-engine/chess_server.py ./web/server.py
cp ../xiangqi-engine/chess_gui_connected.html ./web/index.html
cp ../xiangqi-engine/README.md .
cp ../xiangqi-engine/requirements.txt .

# 3. 初始化 Git
git init
git add .
git commit -m "Initial release v0.1"
```
