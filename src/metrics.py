"""
专业级训练指标监控系统
参考 DeepMind AlphaZero 的指标设计
"""
import os
import json
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt


class TrainingMetrics:
    """
    训练指标收集与可视化
    
    跟踪以下核心指标：
    1. 规则学习曲线 - 各类违规的下降趋势
    2. 棋局质量 - 游戏长度、吃子效率
    3. 胜率趋势 - Agent 的实际表现
    4. 网络训练 - Loss 曲线
    """
    
    def __init__(self, save_dir: str = "metrics"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 时间序列数据
        self.history = {
            # 核心指标
            "episode": [],
            "reward": [],
            "win_rate": [],           # 胜率
            "game_length": [],        # 棋局长度（有效步数）
            
            # 规则学习指标
            "valid_ratio": [],        # 合法率
            "no_piece_rate": [],      # 空位违规率
            "wrong_color_rate": [],   # 颜色违规率
            "invalid_pattern_rate": [],  # 走法违规率
            
            # 棋局质量指标
            "capture_count": [],      # 平均吃子数
            "capture_efficiency": [], # 吃子效率（吃掉的 / 被吃的）
            
            # 网络训练指标
            "loss": [],
            "epsilon": [],
            
            # 时间戳
            "timestamp": [],
        }
        
        # 滑动窗口统计
        self.window_size = 100
        self.recent_wins = deque(maxlen=self.window_size)
        self.recent_rewards = deque(maxlen=self.window_size)
        self.recent_lengths = deque(maxlen=self.window_size)
        self.recent_captures = deque(maxlen=self.window_size)
        
        # 违规类型统计（滑动窗口）
        self.violation_windows = {
            "NO_PIECE": deque(maxlen=self.window_size),
            "WRONG_COLOR": deque(maxlen=self.window_size),
            "SAME_COLOR": deque(maxlen=self.window_size),
            "INVALID_PATTERN": deque(maxlen=self.window_size),
            "BLOCKED": deque(maxlen=self.window_size),
        }
        
        # 累计统计
        self.total_episodes = 0
        self.total_wins = {"red": 0, "black": 0, "draw": 0}
        self.best_valid_ratio = 0
        self.best_win_rate = 0
    
    def record_episode(
        self,
        episode: int,
        reward: float,
        winner: int,
        valid_moves: int,
        invalid_moves: int,
        violations: Dict[str, int],
        captures_made: int = 0,
        captures_lost: int = 0,
        loss: float = 0,
        epsilon: float = 1.0,
    ):
        """
        记录一局的指标
        
        Args:
            episode: 当前 episode 编号
            reward: 本局总奖励
            winner: 胜者 (1=红, -1=黑, 0=和)
            valid_moves: 合法移动次数
            invalid_moves: 违规次数
            violations: 各类违规的次数 {"NO_PIECE": 5, ...}
            captures_made: 吃掉对方的棋子数
            captures_lost: 被对方吃掉的棋子数
            loss: 网络训练损失
            epsilon: 当前探索率
        """
        self.total_episodes = episode
        
        # 胜率统计
        if winner == 1:
            self.total_wins["red"] += 1
            self.recent_wins.append(1)
        elif winner == -1:
            self.total_wins["black"] += 1
            self.recent_wins.append(0)
        else:
            self.total_wins["draw"] += 1
            self.recent_wins.append(0.5)
        
        # 滑动窗口更新
        self.recent_rewards.append(reward)
        self.recent_lengths.append(valid_moves)
        self.recent_captures.append(captures_made)
        
        # 违规统计
        total_moves = valid_moves + invalid_moves
        for vtype, window in self.violation_windows.items():
            count = violations.get(vtype, 0)
            rate = count / max(total_moves, 1)
            window.append(rate)
        
        # 计算指标
        valid_ratio = valid_moves / max(total_moves, 1)
        capture_eff = captures_made / max(captures_lost, 1)
        
        # 记录历史
        self.history["episode"].append(episode)
        self.history["reward"].append(np.mean(self.recent_rewards))
        self.history["win_rate"].append(np.mean(self.recent_wins) if self.recent_wins else 0)
        self.history["game_length"].append(np.mean(self.recent_lengths))
        self.history["valid_ratio"].append(valid_ratio)
        self.history["capture_count"].append(np.mean(self.recent_captures))
        self.history["capture_efficiency"].append(capture_eff)
        self.history["loss"].append(loss)
        self.history["epsilon"].append(epsilon)
        self.history["timestamp"].append(datetime.now().isoformat())
        
        # 违规率
        for vtype, window in self.violation_windows.items():
            key = f"{vtype.lower()}_rate"
            if key in self.history:
                self.history[key].append(np.mean(window) if window else 0)
        
        # 更新最佳记录
        self.best_valid_ratio = max(self.best_valid_ratio, valid_ratio)
        if self.recent_wins:
            self.best_win_rate = max(self.best_win_rate, np.mean(self.recent_wins))
    
    def get_summary(self) -> str:
        """获取当前状态摘要"""
        if not self.history["episode"]:
            return "No data yet"
        
        latest_idx = -1
        return (
            f"Ep {self.history['episode'][latest_idx]:5d} | "
            f"WinRate: {self.history['win_rate'][latest_idx]*100:5.1f}% | "
            f"ValidR: {self.history['valid_ratio'][latest_idx]*100:5.1f}% | "
            f"Length: {self.history['game_length'][latest_idx]:5.1f} | "
            f"Capture: {self.history['capture_count'][latest_idx]:4.1f} | "
            f"Loss: {self.history['loss'][latest_idx]:.4f}"
        )
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        绘制训练曲线
        生成多个子图展示不同指标
        """
        if len(self.history["episode"]) < 2:
            print("Not enough data to plot")
            return
        
        episodes = self.history["episode"]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Training Metrics Dashboard", fontsize=14)
        
        # 1. 胜率曲线
        ax = axes[0, 0]
        ax.plot(episodes, [w*100 for w in self.history["win_rate"]], 'b-', linewidth=1)
        ax.set_title("Win Rate")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Win Rate (%)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # 2. 合法率曲线（核心规则学习指标）
        ax = axes[0, 1]
        ax.plot(episodes, [v*100 for v in self.history["valid_ratio"]], 'g-', linewidth=1)
        ax.set_title("Valid Move Ratio (Rule Learning)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Valid Ratio (%)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # 3. 棋局长度
        ax = axes[0, 2]
        ax.plot(episodes, self.history["game_length"], 'orange', linewidth=1)
        ax.set_title("Game Length")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Valid Moves per Game")
        ax.grid(True, alpha=0.3)
        
        # 4. 违规类型分解
        ax = axes[1, 0]
        if self.history.get("no_piece_rate"):
            ax.plot(episodes, [r*100 for r in self.history["no_piece_rate"]], 
                   label="No Piece", alpha=0.7)
        if self.history.get("wrong_color_rate"):
            ax.plot(episodes, [r*100 for r in self.history["wrong_color_rate"]], 
                   label="Wrong Color", alpha=0.7)
        if self.history.get("invalid_pattern_rate"):
            ax.plot(episodes, [r*100 for r in self.history["invalid_pattern_rate"]], 
                   label="Invalid Pattern", alpha=0.7)
        ax.set_title("Violation Types Breakdown")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rate (%)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 5. 奖励曲线
        ax = axes[1, 1]
        ax.plot(episodes, self.history["reward"], 'purple', linewidth=1)
        ax.set_title("Average Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        
        # 6. Loss 曲线
        ax = axes[1, 2]
        ax.plot(episodes, self.history["loss"], 'red', linewidth=1)
        ax.set_title("Training Loss")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = os.path.join(self.save_dir, "training_curves.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Training curves saved to {save_path}")
    
    def save_history(self, path: Optional[str] = None):
        """保存训练历史到 JSON"""
        if path is None:
            path = os.path.join(self.save_dir, "training_history.json")
        
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to {path}")
    
    def load_history(self, path: str):
        """加载训练历史"""
        with open(path, 'r') as f:
            self.history = json.load(f)
        print(f"History loaded from {path}")


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Training Metrics Test\n")
    
    metrics = TrainingMetrics(save_dir="test_metrics")
    
    # 模拟一些训练数据
    import random
    for ep in range(1, 201):
        # 模拟逐渐改善的指标
        valid_ratio = min(0.05 + ep * 0.001, 0.5)
        valid_moves = int(random.gauss(5 + ep * 0.1, 2))
        invalid_moves = int(valid_moves / valid_ratio - valid_moves)
        
        violations = {
            "NO_PIECE": random.randint(0, max(1, int(20 - ep * 0.05))),
            "WRONG_COLOR": random.randint(0, max(1, int(15 - ep * 0.04))),
            "INVALID_PATTERN": random.randint(0, max(1, int(25 - ep * 0.08))),
        }
        
        metrics.record_episode(
            episode=ep,
            reward=random.gauss(-20 + ep * 0.05, 3),
            winner=random.choice([1, -1, 0]),
            valid_moves=valid_moves,
            invalid_moves=invalid_moves,
            violations=violations,
            captures_made=random.randint(0, 3),
            captures_lost=random.randint(0, 3),
            loss=max(0.001, 0.1 - ep * 0.0003 + random.gauss(0, 0.01)),
            epsilon=max(0.1, 1.0 - ep * 0.004),
        )
        
        if ep % 50 == 0:
            print(metrics.get_summary())
    
    # 绘制曲线
    metrics.plot_training_curves()
    metrics.save_history()
    
    print("\nTest completed! Check test_metrics/ folder.")
