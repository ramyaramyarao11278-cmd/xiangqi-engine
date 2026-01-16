"""
搜索监督训练
使用搜索引擎的结果训练 Policy-Value 网络

训练信号：
- Policy: 搜索得到的最佳着法
- Value: 对局结果 (+1/-1) 或搜索分数（归一化）
"""
import os
import sys
import time
import random
import numpy as np
from collections import deque
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass

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
class TrainingSample:
    """训练样本"""
    state: np.ndarray       # 棋盘状态
    policy_target: int      # 最佳着法索引
    value_target: float     # 局面价值 [-1, 1]


class SearchSupervisedTrainer:
    """
    搜索监督训练器
    
    流程：
    1. 自我对弈生成局面
    2. 用搜索引擎分析每个局面，得到最佳着法
    3. 用搜索结果训练网络
    """
    
    def __init__(
        self,
        save_dir: str = "checkpoints_pv",
        search_depth: int = 4,
        search_time_ms: int = 2000,
        batch_size: int = 64,
        lr: float = 1e-3,
        use_cuda: bool = True,
    ):
        self.save_dir = save_dir
        self.search_depth = search_depth
        self.search_time_ms = search_time_ms
        self.batch_size = batch_size
        
        # 设备
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 网络
        self.net = SimplePolicyValueNet(action_size=8100).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
        # 搜索引擎
        self.search_engine = SearchEngineV2(tt_size_mb=16)
        
        # 着法编码器
        self.move_encoder = MoveEncoder()
        
        # 数据缓冲
        self.replay_buffer: List[TrainingSample] = []
        self.buffer_size = 50000
        
        # 统计
        self.total_games = 0
        self.total_samples = 0
        
        os.makedirs(save_dir, exist_ok=True)
    
    def get_state(self, board: Board) -> np.ndarray:
        """将棋盘转换为网络输入"""
        state = np.zeros((15, 10, 9), dtype=np.float32)
        
        for row in range(10):
            for col in range(9):
                piece = board.board[row, col]
                if piece > 0:
                    state[piece - 1, row, col] = 1.0
        
        # 当前玩家
        if board.current_player == 1:
            state[14, :, :] = 1.0
        
        return state
    
    def generate_game_data(self) -> List[TrainingSample]:
        """
        自我对弈一局，生成训练数据
        
        使用搜索引擎下棋，记录每一步的：
        - 状态
        - 搜索最佳着法
        - 最终胜负
        """
        board = Board()
        game_history = []  # (state, best_move_action, player)
        
        max_moves = 150
        
        for _ in range(max_moves):
            if board.is_game_over():
                break
            
            # 获取当前状态
            state = self.get_state(board)
            current_player = board.current_player
            
            # 搜索最佳着法
            best_move, score = self.search_engine.search(
                board, 
                depth=self.search_depth, 
                time_limit_ms=self.search_time_ms
            )
            
            if best_move is None:
                break
            
            # 编码着法
            action = self.move_encoder.encode_move(
                best_move.from_row, best_move.from_col,
                best_move.to_row, best_move.to_col
            )
            
            # 记录
            game_history.append((state, action, current_player))
            
            # 执行着法
            board.make_move(best_move)
        
        # 确定胜负
        red_king = board.find_king(True)
        black_king = board.find_king(False)
        
        if red_king is None:
            winner = -1  # 黑胜
        elif black_king is None:
            winner = 1   # 红胜
        else:
            winner = 0   # 和棋
        
        # 生成训练样本
        samples = []
        for state, action, player in game_history:
            # 价值目标：从当时玩家视角
            if winner == 0:
                value_target = 0.0
            elif winner == player:
                value_target = 1.0
            else:
                value_target = -1.0
            
            samples.append(TrainingSample(state, action, value_target))
        
        return samples
    
    def generate_position_data(self, num_positions: int = 100) -> List[TrainingSample]:
        """
        从随机局面生成训练数据（更快）
        
        不需要完整对局，直接从随机局面搜索
        """
        samples = []
        board = Board()
        
        for _ in range(num_positions):
            board.reset()
            
            # 随机走几步
            random_moves = random.randint(5, 30)
            for _ in range(random_moves):
                gen = MoveGenerator(board)
                moves = gen.generate_moves(board.current_player == 1)
                
                # 过滤自将
                legal_moves = []
                for move in moves:
                    board.make_move(move)
                    if not board.is_in_check(board.current_player == -1):
                        legal_moves.append(move)
                    board.unmake_move(move)
                
                if not legal_moves:
                    break
                
                move = random.choice(legal_moves)
                board.make_move(move)
                
                if board.is_game_over():
                    break
            
            if board.is_game_over():
                continue
            
            # 搜索分析当前局面
            state = self.get_state(board)
            best_move, score = self.search_engine.search(
                board,
                depth=self.search_depth,
                time_limit_ms=self.search_time_ms
            )
            
            if best_move is None:
                continue
            
            action = self.move_encoder.encode_move(
                best_move.from_row, best_move.from_col,
                best_move.to_row, best_move.to_col
            )
            
            # 用搜索分数作为价值目标（归一化到 [-1, 1]）
            value_target = np.tanh(score / 500)
            
            # 如果是黑方走棋，翻转价值
            if board.current_player == -1:
                value_target = -value_target
            
            samples.append(TrainingSample(state, action, value_target))
        
        return samples
    
    def train_batch(self) -> Tuple[float, float]:
        """训练一个 batch"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # 采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        states = torch.FloatTensor(np.array([s.state for s in batch])).to(self.device)
        policy_targets = torch.LongTensor([s.policy_target for s in batch]).to(self.device)
        value_targets = torch.FloatTensor([s.value_target for s in batch]).to(self.device)
        
        # 前向传播
        self.net.train()
        policy_logits, values = self.net(states)
        
        # 策略损失（交叉熵）
        policy_loss = F.cross_entropy(policy_logits, policy_targets)
        
        # 价值损失（MSE）
        value_loss = F.mse_loss(values.squeeze(), value_targets)
        
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
        num_iterations: int = 100,
        positions_per_iter: int = 50,
        train_steps_per_iter: int = 20,
    ):
        """
        主训练循环
        
        Args:
            num_iterations: 迭代次数
            positions_per_iter: 每次迭代生成的局面数
            train_steps_per_iter: 每次迭代训练的步数
        """
        print("=" * 70)
        print("Search-Supervised Training for Policy-Value Network")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Search depth: {self.search_depth}")
        print(f"Batch size: {self.batch_size}")
        print("-" * 70)
        
        start_time = time.time()
        
        for iteration in range(1, num_iterations + 1):
            iter_start = time.time()
            
            # 生成数据
            print(f"\nIteration {iteration}/{num_iterations}: Generating {positions_per_iter} positions...")
            samples = self.generate_position_data(positions_per_iter)
            
            # 添加到缓冲区
            self.replay_buffer.extend(samples)
            if len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer = self.replay_buffer[-self.buffer_size:]
            
            self.total_samples += len(samples)
            
            # 训练
            policy_losses = []
            value_losses = []
            
            for _ in range(train_steps_per_iter):
                p_loss, v_loss = self.train_batch()
                if p_loss > 0:
                    policy_losses.append(p_loss)
                    value_losses.append(v_loss)
            
            avg_p_loss = np.mean(policy_losses) if policy_losses else 0
            avg_v_loss = np.mean(value_losses) if value_losses else 0
            
            iter_time = time.time() - iter_start
            total_time = time.time() - start_time
            
            print(f"  Samples: {len(samples)} | Buffer: {len(self.replay_buffer)} | "
                  f"P_Loss: {avg_p_loss:.4f} | V_Loss: {avg_v_loss:.4f} | "
                  f"Time: {iter_time:.1f}s")
            
            # 保存
            if iteration % 10 == 0:
                self.save_checkpoint(iteration)
        
        print("-" * 70)
        print(f"Training completed! Total time: {total_time/60:.1f} min")
        self.save_checkpoint(num_iterations)
    
    def save_checkpoint(self, iteration: int):
        """保存模型"""
        path = os.path.join(self.save_dir, f"pv_net_iter{iteration}.pt")
        torch.save({
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iteration': iteration,
            'total_samples': self.total_samples,
        }, path)
        print(f"  Saved: {path}")
    
    def load_checkpoint(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded: {path}")


def main():
    trainer = SearchSupervisedTrainer(
        save_dir="checkpoints_pv",
        search_depth=3,  # 较浅搜索，加快生成
        search_time_ms=1000,
        batch_size=32,
        lr=1e-3,
        use_cuda=True,
    )
    
    trainer.train(
        num_iterations=50,
        positions_per_iter=30,
        train_steps_per_iter=10,
    )


if __name__ == "__main__":
    main()
