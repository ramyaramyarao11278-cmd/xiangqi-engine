"""
AlphaZero 风格训练 v2
关键改进：
1. Policy: Soft Label（Top-K 分数分布）而非 One-Hot
2. Value: 终局胜负 z ∈ {+1,0,-1} 而非搜索分数
3. 完整自对弈生成数据
4. 合法着法 Mask
"""
import os
import sys
import time
import random
import numpy as np
from collections import deque
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.board import Board
from src.move import Move, MoveGenerator
from src.search_v2 import SearchEngineV2
from src.policy_value_net import SimplePolicyValueNet, MoveEncoder


@dataclass
class SoftLabelSample:
    """Soft Label 训练样本"""
    state: np.ndarray                          # 棋盘状态 (15, 10, 9)
    policy_distribution: np.ndarray            # Soft label (8100,)
    legal_mask: np.ndarray                     # 合法着法 mask (8100,)
    value_target: float                        # 终局胜负 [-1, 1]


class AlphaZeroStyleTrainer:
    """
    AlphaZero 风格训练器
    
    核心改进：
    - Policy 目标：搜索分数的 softmax 分布（soft label）
    - Value 目标：终局胜负 z
    - 合法着法 mask
    - 完整自对弈
    """
    
    def __init__(
        self,
        save_dir: str = "checkpoints_az",
        search_depth: int = 3,
        search_time_ms: int = 1500,
        top_k: int = 10,           # 用于构造 soft label 的 top-k 着法
        temperature: float = 2.0,  # softmax 温度
        batch_size: int = 64,
        lr: float = 1e-3,
        use_cuda: bool = True,
    ):
        self.save_dir = save_dir
        self.search_depth = search_depth
        self.search_time_ms = search_time_ms
        self.top_k = top_k
        self.temperature = temperature
        self.batch_size = batch_size
        
        # 设备
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # 网络
        self.net = SimplePolicyValueNet(action_size=8100).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
        # 搜索引擎
        self.search_engine = SearchEngineV2(tt_size_mb=16)
        
        # 着法编码
        self.move_encoder = MoveEncoder()
        
        # 数据缓冲
        self.replay_buffer: List[SoftLabelSample] = []
        self.buffer_size = 50000
        
        # 统计
        self.total_games = 0
        self.total_samples = 0
        
        os.makedirs(save_dir, exist_ok=True)
    
    def get_state(self, board: Board) -> np.ndarray:
        """棋盘→网络输入"""
        state = np.zeros((15, 10, 9), dtype=np.float32)
        for row in range(10):
            for col in range(9):
                piece = board.board[row, col]
                if piece > 0:
                    state[piece - 1, row, col] = 1.0
        if board.current_player == 1:
            state[14, :, :] = 1.0
        return state
    
    def get_legal_mask(self, board: Board) -> np.ndarray:
        """获取合法着法 mask"""
        mask = np.zeros(8100, dtype=np.float32)
        
        gen = MoveGenerator(board)
        is_red = board.current_player == 1
        moves = gen.generate_moves(is_red)
        
        for move in moves:
            board.make_move(move)
            if not board.is_in_check(is_red):
                action = self.move_encoder.encode_move(
                    move.from_row, move.from_col,
                    move.to_row, move.to_col
                )
                mask[action] = 1.0
            board.unmake_move(move)
        
        return mask
    
    def create_soft_label(
        self,
        move_scores: List[Tuple[Move, int]],
        legal_mask: np.ndarray,
    ) -> np.ndarray:
        """
        从搜索结果创建 Soft Label
        
        使用 softmax(scores / T) 作为分布
        """
        if not move_scores:
            return np.zeros(8100, dtype=np.float32)
        
        # 提取分数并归一化
        actions = []
        scores = []
        
        for move, score in move_scores:
            action = self.move_encoder.encode_move(
                move.from_row, move.from_col,
                move.to_row, move.to_col
            )
            actions.append(action)
            scores.append(score)
        
        scores = np.array(scores, dtype=np.float32)
        
        # 分数中心化（避免数值问题）
        scores = scores - np.max(scores)
        
        # Softmax with temperature
        probs = np.exp(scores / (self.temperature * 100))  # 缩放
        probs = probs / (probs.sum() + 1e-8)
        
        # 构造完整分布
        distribution = np.zeros(8100, dtype=np.float32)
        for action, prob in zip(actions, probs):
            distribution[action] = prob
        
        # 确保只在合法着法上有概率
        distribution = distribution * legal_mask
        total = distribution.sum()
        if total > 0:
            distribution = distribution / total
        
        return distribution
    
    def play_self_game(self) -> List[SoftLabelSample]:
        """
        完整自对弈一局
        
        记录每一步的：
        - 状态
        - Soft label (搜索分布)
        - 合法 mask
        
        最后用终局胜负作为所有样本的 value 目标
        """
        board = Board()
        game_history = []  # (state, soft_label, legal_mask, player)
        
        max_moves = 150
        
        for move_num in range(max_moves):
            if board.is_game_over():
                break
            
            # 获取当前状态
            state = self.get_state(board)
            legal_mask = self.get_legal_mask(board)
            current_player = board.current_player
            
            # 获取 Top-K 着法
            top_k_moves = self.search_engine.search_top_k(
                board,
                depth=self.search_depth,
                time_limit_ms=self.search_time_ms,
                top_k=self.top_k,
            )
            
            if not top_k_moves:
                break
            
            # 创建 soft label
            soft_label = self.create_soft_label(top_k_moves, legal_mask)
            
            # 记录（value 后面补）
            game_history.append((state, soft_label, legal_mask, current_player))
            
            # 选择着法（按概率采样，增加多样性）
            if move_num < 15 and random.random() < 0.3:
                # 前15步有30%概率采样探索
                probs = soft_label / (soft_label.sum() + 1e-8)
                action = np.random.choice(8100, p=probs)
                from_row, from_col, to_row, to_col = self.move_encoder.decode_move(action)
                best_move = Move(from_row, from_col, to_row, to_col, 0)
            else:
                # 否则选最佳
                best_move = top_k_moves[0][0]
            
            board.make_move(best_move)
        
        # 确定终局胜负
        red_king = board.find_king(True)
        black_king = board.find_king(False)
        
        if red_king is None:
            winner = -1  # 黑胜
        elif black_king is None:
            winner = 1   # 红胜
        else:
            winner = 0   # 和棋
        
        # 生成样本，value = 从当时玩家视角的胜负
        samples = []
        for state, soft_label, legal_mask, player in game_history:
            if winner == 0:
                value_target = 0.0
            elif winner == player:
                value_target = 1.0
            else:
                value_target = -1.0
            
            samples.append(SoftLabelSample(state, soft_label, legal_mask, value_target))
        
        self.total_games += 1
        return samples
    
    def train_batch(self) -> Tuple[float, float]:
        """训练一个 batch"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        states = torch.FloatTensor(np.array([s.state for s in batch])).to(self.device)
        policy_targets = torch.FloatTensor(np.array([s.policy_distribution for s in batch])).to(self.device)
        legal_masks = torch.FloatTensor(np.array([s.legal_mask for s in batch])).to(self.device)
        value_targets = torch.FloatTensor([s.value_target for s in batch]).to(self.device)
        
        self.net.train()
        policy_logits, values = self.net(states)
        
        # 合法着法 mask：非法位置设为极小值
        masked_logits = policy_logits + (legal_masks - 1) * 1e9
        
        # Policy Loss: KL 散度 / 交叉熵
        log_probs = F.log_softmax(masked_logits, dim=1)
        policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
        
        # Value Loss: MSE
        value_loss = F.mse_loss(values.squeeze(), value_targets)
        
        # 总损失
        total_loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def train(
        self,
        num_games: int = 50,
        train_steps_per_game: int = 20,
    ):
        """主训练循环"""
        print("=" * 70)
        print("AlphaZero-Style Training v2")
        print("- Policy: Soft Label (Top-K softmax)")
        print("- Value: Terminal Game Outcome")
        print("- Legal Mask: Enabled")
        print("=" * 70)
        
        start_time = time.time()
        
        for game_num in range(1, num_games + 1):
            game_start = time.time()
            
            # 自对弈
            print(f"\nGame {game_num}/{num_games}: Self-play...", end=" ", flush=True)
            samples = self.play_self_game()
            
            # 添加到缓冲
            self.replay_buffer.extend(samples)
            if len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer = self.replay_buffer[-self.buffer_size:]
            
            self.total_samples += len(samples)
            
            # 训练
            policy_losses = []
            value_losses = []
            
            for _ in range(train_steps_per_game):
                p_loss, v_loss = self.train_batch()
                if p_loss > 0:
                    policy_losses.append(p_loss)
                    value_losses.append(v_loss)
            
            avg_p = np.mean(policy_losses) if policy_losses else 0
            avg_v = np.mean(value_losses) if value_losses else 0
            
            game_time = time.time() - game_start
            
            print(f"{len(samples)} samples | "
                  f"Buffer: {len(self.replay_buffer)} | "
                  f"P_Loss: {avg_p:.4f} | V_Loss: {avg_v:.4f} | "
                  f"Time: {game_time:.1f}s")
            
            # 保存
            if game_num % 10 == 0:
                self.save_checkpoint(game_num)
        
        total_time = time.time() - start_time
        print("-" * 70)
        print(f"Training completed! Games: {num_games}, Time: {total_time/60:.1f} min")
        self.save_checkpoint(num_games)
    
    def save_checkpoint(self, game_num: int):
        path = os.path.join(self.save_dir, f"az_net_game{game_num}.pt")
        torch.save({
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'game_num': game_num,
            'total_samples': self.total_samples,
        }, path)
        print(f"  Saved: {path}")


def main():
    trainer = AlphaZeroStyleTrainer(
        save_dir="checkpoints_az",
        search_depth=2,       # 较浅搜索加快自对弈
        search_time_ms=800,
        top_k=8,              # Top-8 构造 soft label
        temperature=2.0,
        batch_size=32,
        lr=1e-3,
        use_cuda=True,
    )
    
    trainer.train(
        num_games=30,
        train_steps_per_game=15,
    )


if __name__ == "__main__":
    main()
