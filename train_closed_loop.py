"""
完整闭环训练脚本
集成：自对弈 → 训练 → 门控评估 → 对手池

这是 AlphaZero 风格训练的完整实现
"""
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.board import Board
from src.move import Move, MoveGenerator
from src.search_v3 import SearchEngineV3
from src.policy_value_net import SimplePolicyValueNet, MoveEncoder
from gating import OpponentPool, GatingEvaluator


class ClosedLoopTrainer:
    """
    完整闭环训练器
    
    流程：
    1. 从对手池采样对手
    2. 自对弈生成数据（Soft Label + 终局胜负）
    3. 训练网络
    4. 门控评估（候选 vs 冠军）
    5. 通过则入池，否则丢弃
    """
    
    def __init__(
        self,
        save_dir: str = "closed_loop_training",
        pool_size: int = 10,
        search_depth: int = 2,
        search_time_ms: int = 800,
        top_k: int = 8,
        temperature: float = 2.0,
        batch_size: int = 32,
        lr: float = 1e-3,
        gate_games: int = 10,
        gate_threshold: float = 0.55,
        use_cuda: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # 网络
        self.net = SimplePolicyValueNet(action_size=8100).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        # 搜索配置
        self.search_depth = search_depth
        self.search_time_ms = search_time_ms
        self.top_k = top_k
        self.temperature = temperature
        self.batch_size = batch_size
        
        # 对手池
        self.pool = OpponentPool(
            pool_dir=str(self.save_dir / "opponent_pool"),
            max_size=pool_size,
        )
        
        # 门控评估
        self.gating = GatingEvaluator(
            search_depth=search_depth,
            search_time_ms=search_time_ms,
            win_threshold=gate_threshold,
        )
        self.gate_games = gate_games
        
        # 数据
        self.move_encoder = MoveEncoder()
        self.replay_buffer = []
        self.buffer_size = 50000
        
        # 统计
        self.generation = 0
        self.total_games = 0
        self.promotions = 0
    
    def get_state(self, board: Board) -> np.ndarray:
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
    
    def create_search_engine(self, net_path: str = None) -> SearchEngineV3:
        """创建搜索引擎（可选加载网络）"""
        return SearchEngineV3(
            tt_size_mb=16,
            net_path=net_path,
            policy_weight=1000.0,
            use_cuda=(self.device.type == 'cuda'),
        )
    
    def self_play_game(self, opponent_path: str = None):
        """
        自对弈一局
        
        Args:
            opponent_path: 对手模型路径，None 则自我对弈
        """
        # 创建引擎
        current_engine = self.create_search_engine()
        # 临时保存当前网络
        temp_path = str(self.save_dir / "temp_current.pt")
        torch.save({'net': self.net.state_dict()}, temp_path)
        current_engine._load_network(temp_path)
        
        if opponent_path:
            opponent_engine = self.create_search_engine(opponent_path)
        else:
            opponent_engine = current_engine
        
        board = Board()
        game_history = []
        
        # 随机决定谁先手
        current_is_red = random.choice([True, False])
        
        for move_num in range(150):
            if board.is_game_over():
                break
            
            state = self.get_state(board)
            legal_mask = self.get_legal_mask(board)
            player = board.current_player
            
            # 选择引擎
            if (player == 1 and current_is_red) or (player == -1 and not current_is_red):
                engine = current_engine
            else:
                engine = opponent_engine
            
            # 搜索
            top_k = engine.search_top_k(
                board,
                depth=self.search_depth,
                time_limit_ms=self.search_time_ms,
                top_k=self.top_k,
            ) if hasattr(engine, 'search_top_k') else None
            
            if not top_k:
                best_move, _ = engine.search(board, depth=self.search_depth, time_limit_ms=self.search_time_ms)
                if best_move is None:
                    break
                board.make_move(best_move)
                continue
            
            # 创建 soft label
            soft_label = self._create_soft_label(top_k, legal_mask)
            game_history.append((state, soft_label, legal_mask, player))
            
            # 选择着法
            best_move = top_k[0][0]
            board.make_move(best_move)
        
        # 确定胜负
        red_king = board.find_king(True)
        black_king = board.find_king(False)
        
        if red_king is None:
            winner = -1
        elif black_king is None:
            winner = 1
        else:
            winner = 0
        
        # 生成样本
        samples = []
        for state, soft_label, legal_mask, player in game_history:
            if winner == 0:
                value_target = 0.0
            elif winner == player:
                value_target = 1.0
            else:
                value_target = -1.0
            samples.append({
                'state': state,
                'policy': soft_label,
                'mask': legal_mask,
                'value': value_target,
            })
        
        return samples
    
    def _create_soft_label(self, move_scores, legal_mask):
        if not move_scores:
            return np.zeros(8100, dtype=np.float32)
        
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
        scores = scores - np.max(scores)
        probs = np.exp(scores / (self.temperature * 100))
        probs = probs / (probs.sum() + 1e-8)
        
        distribution = np.zeros(8100, dtype=np.float32)
        for action, prob in zip(actions, probs):
            distribution[action] = prob
        
        distribution = distribution * legal_mask
        total = distribution.sum()
        if total > 0:
            distribution = distribution / total
        
        return distribution
    
    def train_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        states = torch.FloatTensor(np.array([s['state'] for s in batch])).to(self.device)
        policies = torch.FloatTensor(np.array([s['policy'] for s in batch])).to(self.device)
        masks = torch.FloatTensor(np.array([s['mask'] for s in batch])).to(self.device)
        values = torch.FloatTensor([s['value'] for s in batch]).to(self.device)
        
        self.net.train()
        policy_logits, value_pred = self.net(states)
        
        masked_logits = policy_logits + (masks - 1) * 1e9
        log_probs = torch.nn.functional.log_softmax(masked_logits, dim=1)
        policy_loss = -(policies * log_probs).sum(dim=1).mean()
        
        value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(), values)
        
        total_loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def run_generation(self, games_per_gen: int = 5, train_steps: int = 20):
        """运行一代训练"""
        print(f"\n{'='*60}")
        print(f"Generation {self.generation}")
        print(f"{'='*60}")
        
        # 采样对手
        opponent_path = self.pool.sample_opponent()
        if opponent_path:
            print(f"Opponent: {Path(opponent_path).name}")
        else:
            print("Opponent: Self")
        
        # 自对弈
        gen_samples = []
        for g in range(games_per_gen):
            print(f"  Game {g+1}/{games_per_gen}...", end=" ", flush=True)
            samples = self.self_play_game(opponent_path)
            gen_samples.extend(samples)
            self.total_games += 1
            print(f"{len(samples)} samples")
        
        # 添加到缓冲
        self.replay_buffer.extend(gen_samples)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = self.replay_buffer[-self.buffer_size:]
        
        # 训练
        print(f"Training ({train_steps} steps)...")
        for _ in range(train_steps):
            p_loss, v_loss = self.train_batch()
        print(f"  P_Loss: {p_loss:.4f}, V_Loss: {v_loss:.4f}")
        
        # 保存候选
        candidate_path = str(self.save_dir / f"candidate_gen{self.generation}.pt")
        torch.save({
            'net': self.net.state_dict(),
            'generation': self.generation,
        }, candidate_path)
        
        # 门控
        champion_path = self.pool.get_champion()
        if champion_path is None:
            print("First model, adding to pool")
            self.pool.add_model(candidate_path, self.generation)
            self.promotions += 1
            passed = True
        else:
            passed, _ = self.gating.evaluate(
                candidate_path, champion_path, self.gate_games
            )
            if passed:
                self.pool.add_model(candidate_path, self.generation)
                self.promotions += 1
        
        self.generation += 1
        return passed
    
    def train(self, num_generations: int = 10, games_per_gen: int = 5):
        """运行完整训练"""
        print("="*60)
        print("Closed-Loop Training")
        print("="*60)
        print(f"Generations: {num_generations}")
        print(f"Games/Gen: {games_per_gen}")
        print("-"*60)
        
        start_time = time.time()
        
        for gen in range(num_generations):
            self.run_generation(games_per_gen)
        
        total_time = time.time() - start_time
        print("-"*60)
        print(f"Training complete!")
        print(f"  Generations: {self.generation}")
        print(f"  Games: {self.total_games}")
        print(f"  Promotions: {self.promotions}")
        print(f"  Time: {total_time/60:.1f} min")


def main():
    trainer = ClosedLoopTrainer(
        save_dir="closed_loop_v1",
        pool_size=5,
        search_depth=2,
        search_time_ms=500,
        top_k=6,
        batch_size=32,
        gate_games=6,
        gate_threshold=0.50,  # 降低门槛方便测试
        use_cuda=True,
    )
    
    trainer.train(
        num_generations=3,  # 测试用，实际可增加
        games_per_gen=3,
    )


if __name__ == "__main__":
    main()
