"""
版本门控 & 对手池
用于闭环训练的稳定性保障

核心机制：
- 对手池：保存最近 N 个模型，自对弈时随机选对手
- 版本门控：新模型必须击败当前冠军才能晋级
"""
import os
import random
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np

from src.board import Board
from src.move import Move, MoveGenerator
from src.search_v3 import SearchEngineV3


@dataclass
class EvaluationResult:
    """评估结果"""
    wins: int
    losses: int
    draws: int
    
    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws
    
    @property
    def win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total_games


class OpponentPool:
    """
    对手池
    
    保存最近 N 个模型 checkpoint，用于自对弈时随机选择对手
    """
    
    def __init__(
        self,
        pool_dir: str = "opponent_pool",
        max_size: int = 10,
    ):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
    
    def add_model(self, model_path: str, generation: int) -> str:
        """添加模型到对手池"""
        src = Path(model_path)
        dst = self.pool_dir / f"gen_{generation:04d}.pt"
        
        # 复制模型
        import shutil
        shutil.copy2(src, dst)
        
        # 清理旧模型
        self._cleanup()
        
        print(f"[Pool] Added: {dst.name}")
        return str(dst)
    
    def _cleanup(self):
        """保持池大小不超过 max_size"""
        models = sorted(self.pool_dir.glob("gen_*.pt"))
        while len(models) > self.max_size:
            old = models.pop(0)
            old.unlink()
            print(f"[Pool] Removed: {old.name}")
    
    def sample_opponent(self) -> Optional[str]:
        """随机采样一个对手"""
        models = list(self.pool_dir.glob("gen_*.pt"))
        if not models:
            return None
        return str(random.choice(models))
    
    def get_champion(self) -> Optional[str]:
        """获取最新的冠军模型"""
        models = sorted(self.pool_dir.glob("gen_*.pt"))
        if not models:
            return None
        return str(models[-1])
    
    def list_models(self) -> List[str]:
        """列出所有模型"""
        return [str(p) for p in sorted(self.pool_dir.glob("gen_*.pt"))]


class GatingEvaluator:
    """
    版本门控评估器
    
    新模型必须在对战中击败当前冠军才能晋级
    """
    
    def __init__(
        self,
        search_depth: int = 3,
        search_time_ms: int = 1000,
        win_threshold: float = 0.55,  # 胜率阈值
    ):
        self.search_depth = search_depth
        self.search_time_ms = search_time_ms
        self.win_threshold = win_threshold
    
    def evaluate(
        self,
        candidate_path: str,
        champion_path: str,
        num_games: int = 20,
    ) -> Tuple[bool, EvaluationResult]:
        """
        评估候选模型
        
        Args:
            candidate_path: 候选模型路径
            champion_path: 当前冠军模型路径
            num_games: 对战局数
            
        Returns:
            (是否通过门控, 评估结果)
        """
        print(f"\n[Gating] Evaluating: candidate vs champion")
        print(f"  Candidate: {candidate_path}")
        print(f"  Champion: {champion_path}")
        print(f"  Games: {num_games}")
        
        # 加载引擎
        candidate_engine = SearchEngineV3(
            tt_size_mb=16,
            net_path=candidate_path,
            policy_weight=1000.0,
        )
        champion_engine = SearchEngineV3(
            tt_size_mb=16,
            net_path=champion_path,
            policy_weight=1000.0,
        )
        
        wins = 0
        losses = 0
        draws = 0
        
        for game_idx in range(num_games):
            # 交替先手
            candidate_is_red = (game_idx % 2 == 0)
            
            result = self._play_game(
                red_engine=candidate_engine if candidate_is_red else champion_engine,
                black_engine=champion_engine if candidate_is_red else candidate_engine,
            )
            
            # 从 candidate 视角统计
            if result == 0:
                draws += 1
                outcome = "draw"
            elif (result == 1 and candidate_is_red) or (result == -1 and not candidate_is_red):
                wins += 1
                outcome = "win"
            else:
                losses += 1
                outcome = "loss"
            
            print(f"  Game {game_idx + 1}/{num_games}: {outcome} "
                  f"(W:{wins} L:{losses} D:{draws})")
        
        result = EvaluationResult(wins, losses, draws)
        passed = result.win_rate >= self.win_threshold
        
        print(f"\n[Gating] Result: WinRate={result.win_rate:.1%}, Threshold={self.win_threshold:.1%}")
        print(f"[Gating] {'PASSED' if passed else 'FAILED'}")
        
        return passed, result
    
    def _play_game(
        self,
        red_engine: SearchEngineV3,
        black_engine: SearchEngineV3,
        max_moves: int = 150,
    ) -> int:
        """
        对弈一局
        
        Returns:
            1 = 红胜, -1 = 黑胜, 0 = 和棋
        """
        board = Board()
        
        for _ in range(max_moves):
            if board.is_game_over():
                break
            
            current_engine = red_engine if board.current_player == 1 else black_engine
            
            best_move, _ = current_engine.search(
                board,
                depth=self.search_depth,
                time_limit_ms=self.search_time_ms,
            )
            
            if best_move is None:
                break
            
            board.make_move(best_move)
        
        # 判断结果
        red_king = board.find_king(True)
        black_king = board.find_king(False)
        
        if red_king is None:
            return -1  # 黑胜
        elif black_king is None:
            return 1   # 红胜
        else:
            return 0   # 和棋


class TrainingPipeline:
    """
    完整训练流水线
    
    集成：自对弈 + 训练 + 门控 + 对手池
    """
    
    def __init__(
        self,
        save_dir: str = "training_pipeline",
        pool_size: int = 10,
        gate_games: int = 20,
        gate_threshold: float = 0.55,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.pool = OpponentPool(
            pool_dir=str(self.save_dir / "opponent_pool"),
            max_size=pool_size,
        )
        
        self.gating = GatingEvaluator(
            win_threshold=gate_threshold,
        )
        
        self.current_generation = 0
    
    def run_iteration(
        self,
        trainer,  # AlphaZeroStyleTrainer 实例
        num_games: int = 10,
    ) -> bool:
        """
        运行一次迭代
        
        Returns:
            新模型是否通过门控
        """
        print(f"\n{'='*60}")
        print(f"Generation {self.current_generation}")
        print(f"{'='*60}")
        
        # 1. 从对手池采样对手（如果有）
        opponent_path = self.pool.sample_opponent()
        if opponent_path:
            print(f"[Iter] Opponent: {opponent_path}")
        else:
            print("[Iter] No opponent in pool, using self-play")
        
        # 2. 自对弈 + 训练
        trainer.train(num_games=num_games, train_steps_per_game=15)
        
        # 3. 保存候选模型
        candidate_path = str(self.save_dir / f"candidate_gen{self.current_generation}.pt")
        torch.save({
            'net': trainer.net.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'generation': self.current_generation,
        }, candidate_path)
        print(f"[Iter] Saved candidate: {candidate_path}")
        
        # 4. 门控评估
        champion_path = self.pool.get_champion()
        
        if champion_path is None:
            # 第一个模型直接入池
            print("[Iter] First model, adding to pool directly")
            self.pool.add_model(candidate_path, self.current_generation)
            passed = True
        else:
            passed, _ = self.gating.evaluate(
                candidate_path=candidate_path,
                champion_path=champion_path,
                num_games=20,
            )
            
            if passed:
                self.pool.add_model(candidate_path, self.current_generation)
        
        self.current_generation += 1
        return passed


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Gating & Opponent Pool Test\n")
    
    # 测试对手池
    pool = OpponentPool(pool_dir="test_pool", max_size=3)
    print(f"Pool models: {pool.list_models()}")
    
    # 测试门控（需要有训练好的模型）
    model_path = "checkpoints_az/az_net_game30.pt"
    if os.path.exists(model_path):
        print(f"\nTesting with model: {model_path}")
        
        evaluator = GatingEvaluator(
            search_depth=2,
            search_time_ms=500,
        )
        
        # 自我对战测试
        passed, result = evaluator.evaluate(
            candidate_path=model_path,
            champion_path=model_path,
            num_games=4,
        )
        
        print(f"\nSelf-play result: {result}")
        print(f"Win rate: {result.win_rate:.1%}")
