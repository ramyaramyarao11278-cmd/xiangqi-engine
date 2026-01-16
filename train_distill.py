"""
统一蒸馏式闭环训练脚本
完全按照 A 方案：根节点每个候选着法做 depth-1 搜索打分

核心要点：
1. 统一视角：value/π 都是"当前执子方视角"
2. 统一动作空间：8100 = from*90+to
3. soft π：softmax(scores/T) 替代 one-hot
4. z：终局胜负作为 value 标签
5. 分布损失：KL/交叉熵替代 CE(one-hot)
"""
import os
import sys
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.board import Board
from src.move import Move, MoveGenerator
from src.search_v2 import SearchEngineV2
from src.action import ACTION_SIZE, encode_move, decode_to_move, get_legal_mask
from src.policy_value_net import SimplePolicyValueNet


@dataclass
class TrainingSample:
    """训练样本（稀疏存储）"""
    state: np.ndarray         # (15, 10, 9)
    pi_actions: np.ndarray    # (K,) int32 - 着法索引
    pi_probs: np.ndarray      # (K,) float32 - 对应概率
    z: float                  # 终局胜负（当前方视角）


class DistillationTrainer:
    """
    蒸馏式闭环训练器
    
    流程：
    1. 自对弈生成数据（搜索作为老师）
    2. 从 root_scores 构造 soft π
    3. 从终局胜负构造 z
    4. 训练网络（分布损失）
    """
    
    def __init__(
        self,
        save_dir: str = "checkpoints_distill",
        search_depth: int = 3,
        search_time_ms: int = 800,
        top_k: int = 16,
        temperature: float = 2.5,
        score_scale: float = 200.0,
        explore_moves: int = 12,  # 前 N 步用采样探索
        batch_size: int = 64,
        lr: float = 1e-3,
        use_cuda: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # 搜索引擎（老师）
        self.search_engine = SearchEngineV2(tt_size_mb=32)
        self.search_depth = search_depth
        self.search_time_ms = search_time_ms
        
        # 网络（学生）
        self.net = SimplePolicyValueNet(action_size=ACTION_SIZE).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        # 训练配置
        self.top_k = top_k
        self.temperature = temperature
        self.score_scale = score_scale
        self.explore_moves = explore_moves
        self.batch_size = batch_size
        
        # 数据缓冲
        self.replay_buffer: List[TrainingSample] = []
        self.buffer_size = 50000
    
    def get_state(self, board: Board) -> np.ndarray:
        """棋盘转网络输入（15通道）"""
        state = np.zeros((15, 10, 9), dtype=np.float32)
        for row in range(10):
            for col in range(9):
                piece = board.board[row, col]
                if piece > 0:
                    state[piece - 1, row, col] = 1.0
        # 第 15 通道：当前玩家
        if board.current_player == 1:
            state[14, :, :] = 1.0
        return state
    
    def soft_pi(self, root_scores: List[Tuple[Move, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 root_scores 构造 soft π 分布
        
        Args:
            root_scores: [(move, score_side)] 当前执子方视角分数
            
        Returns:
            actions: (K,) int32
            probs: (K,) float32，归一化概率
        """
        if not root_scores:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        
        top = root_scores[:self.top_k]
        actions = np.array([encode_move(m) for m, _ in top], dtype=np.int32)
        scores = np.array([s for _, s in top], dtype=np.float32)
        
        # 数值稳定的 softmax
        scores = np.clip(scores, -100000, 100000)
        x = (scores / self.score_scale) / max(self.temperature, 1e-6)
        x = x - x.max()
        p = np.exp(x)
        p = p / (p.sum() + 1e-12)
        
        return actions, p.astype(np.float32)
    
    def play_game(self, game_id: int = 0) -> List[TrainingSample]:
        """
        自对弈一局
        
        Returns:
            样本列表（回填 z 后）
        """
        board = Board()
        trajectory = []  # [(state, pi_actions, pi_probs, player)]
        move_count = 0
        
        while not board.is_game_over() and move_count < 150:
            state = self.get_state(board)
            player = board.current_player
            
            # 搜索获取 root_scores
            root_scores = self.search_engine.search_top_k(
                board,
                depth=self.search_depth,
                time_limit_ms=self.search_time_ms,
                top_k=self.top_k,
            )
            
            if not root_scores:
                break
            
            # 构造 soft π
            pi_actions, pi_probs = self.soft_pi(root_scores)
            
            if len(pi_actions) == 0:
                break
            
            # 选择着法
            if move_count < self.explore_moves:
                # 探索阶段：按概率采样
                idx = np.random.choice(len(pi_actions), p=pi_probs)
            else:
                # 收敛阶段：选最优
                idx = 0
            
            action = pi_actions[idx]
            move = decode_to_move(action)
            
            # 记录样本
            trajectory.append((state, pi_actions, pi_probs, player))
            
            # 执行着法
            board.make_move(move)
            move_count += 1
        
        # 判断结果
        red_king = board.find_king(True)
        black_king = board.find_king(False)
        
        if red_king is None:
            winner = -1  # 黑胜
        elif black_king is None:
            winner = 1   # 红胜
        else:
            winner = 0   # 和棋
        
        # 回填 z
        samples = []
        for state, pi_actions, pi_probs, player in trajectory:
            if winner == 0:
                z = 0.0
            elif winner == player:
                z = 1.0  # 当前方赢
            else:
                z = -1.0  # 当前方输
            
            samples.append(TrainingSample(
                state=state,
                pi_actions=pi_actions,
                pi_probs=pi_probs,
                z=z,
            ))
        
        result_str = {1: "红胜", -1: "黑胜", 0: "和棋"}[winner]
        print(f"  Game {game_id}: {move_count} moves, {result_str}, {len(samples)} samples")
        
        return samples
    
    def train_batch(self) -> Tuple[float, float]:
        """训练一个 batch"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # 准备数据
        states = torch.FloatTensor(np.array([s.state for s in batch])).to(self.device)
        z_targets = torch.FloatTensor([s.z for s in batch]).to(self.device)
        
        # 前向传播
        self.net.train()
        policy_logits, value_pred = self.net(states)
        
        # Policy 损失（稀疏分布 KL）
        log_probs = F.log_softmax(policy_logits, dim=1)
        
        policy_loss = 0.0
        for i, sample in enumerate(batch):
            actions = torch.LongTensor(sample.pi_actions).to(self.device)
            probs = torch.FloatTensor(sample.pi_probs).to(self.device)
            
            picked = log_probs[i, actions]
            policy_loss -= (probs * picked).sum()
        
        policy_loss = policy_loss / len(batch)
        
        # Value 损失（MSE with z）
        value_loss = F.mse_loss(value_pred.squeeze(), z_targets)
        
        # 总损失
        total_loss = policy_loss + value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def train(
        self,
        num_games: int = 50,
        train_steps_per_game: int = 20,
        save_interval: int = 10,
    ):
        """运行训练"""
        print("="*60)
        print("Distillation Training (A-Plan)")
        print("="*60)
        print(f"Search depth: {self.search_depth}")
        print(f"Top-K: {self.top_k}")
        print(f"Temperature: {self.temperature}")
        print(f"Explore moves: {self.explore_moves}")
        print("-"*60)
        
        start_time = time.time()
        
        for game_id in range(num_games):
            # 自对弈
            samples = self.play_game(game_id + 1)
            self.replay_buffer.extend(samples)
            
            # 限制缓冲大小
            if len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer = self.replay_buffer[-self.buffer_size:]
            
            # 训练
            p_losses = []
            v_losses = []
            for _ in range(train_steps_per_game):
                p_loss, v_loss = self.train_batch()
                if p_loss > 0:
                    p_losses.append(p_loss)
                    v_losses.append(v_loss)
            
            if p_losses:
                avg_p = sum(p_losses) / len(p_losses)
                avg_v = sum(v_losses) / len(v_losses)
                print(f"    Train: P_Loss={avg_p:.4f}, V_Loss={avg_v:.4f}")
            
            # 保存
            if (game_id + 1) % save_interval == 0:
                path = self.save_dir / f"distill_game{game_id+1}.pt"
                torch.save({
                    'net': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'game': game_id + 1,
                    'buffer_size': len(self.replay_buffer),
                }, path)
                print(f"  -> Saved: {path.name}")
        
        total_time = time.time() - start_time
        print("-"*60)
        print(f"Training complete!")
        print(f"  Games: {num_games}")
        print(f"  Samples: {len(self.replay_buffer)}")
        print(f"  Time: {total_time/60:.1f} min")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=30)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--time', type=int, default=600)
    parser.add_argument('--topk', type=int, default=12)
    parser.add_argument('--temp', type=float, default=2.5)
    parser.add_argument('--explore', type=int, default=10)
    
    args = parser.parse_args()
    
    trainer = DistillationTrainer(
        save_dir="checkpoints_distill",
        search_depth=args.depth,
        search_time_ms=args.time,
        top_k=args.topk,
        temperature=args.temp,
        explore_moves=args.explore,
    )
    
    trainer.train(num_games=args.games)


if __name__ == "__main__":
    main()
