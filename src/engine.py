"""
引擎 API 封装
用于实际使用：输入局面 -> 输出推荐着法
"""
import os
from typing import List, Dict, Optional
import numpy as np

from .board import Board
from .move import Move, MoveGenerator
from .fen import parse_fen, to_fen
from .environment import XiangqiEnv
from .agent import DQNAgent


class XiangqiEngine:
    """
    中国象棋 AI 引擎
    
    使用方式：
    ```
    engine = XiangqiEngine.load("model.pt")
    moves = engine.analyze("fen_string", top_n=5)
    ```
    """
    
    def __init__(self):
        self.env = XiangqiEnv()
        self.agent = None
    
    @classmethod
    def load(cls, model_path: str) -> 'XiangqiEngine':
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            引擎实例
        """
        engine = cls()
        engine.agent = DQNAgent(use_cuda=False)  # 推理时可用 CPU
        
        if os.path.exists(model_path):
            engine.agent.load(model_path)
        else:
            print(f"Warning: Model not found at {model_path}, using untrained network")
        
        return engine
    
    def analyze(self, fen: str, top_n: int = 5) -> List[Dict]:
        """
        分析给定局面，返回推荐着法
        
        Args:
            fen: 局面的 FEN 字符串
            top_n: 返回的候选着法数量
            
        Returns:
            推荐着法列表，每项包含：
            - move: 着法对象
            - uci: UCI 格式字符串
            - chinese: 中文着法描述
            - score: 评分 (0-1)
        """
        # 解析 FEN
        board = Board()
        if not parse_fen(fen, board):
            return [{"error": "Invalid FEN"}]
        
        self.env.board = board
        self.env.move_gen = MoveGenerator(board)
        
        # 获取状态
        state = self.env.get_state()
        
        # 获取所有合法着法
        is_red = board.current_player == 1
        legal_moves = self.env.move_gen.generate_moves(is_red)
        
        if len(legal_moves) == 0:
            return [{"error": "No legal moves"}]
        
        # 获取 Q 值
        legal_actions = [self.env.move_to_action(m) for m in legal_moves]
        q_values = self.agent.policy_net.predict(state)
        
        # 为每个合法着法评分
        move_scores = []
        for move, action in zip(legal_moves, legal_actions):
            score = q_values[action]
            move_scores.append((move, score))
        
        # 排序并取 top_n
        move_scores.sort(key=lambda x: x[1], reverse=True)
        top_moves = move_scores[:top_n]
        
        # 归一化分数到 0-1
        if len(top_moves) > 0:
            scores = [s for _, s in top_moves]
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s != min_s else 1
        
        results = []
        for move, score in top_moves:
            normalized = (score - min_s) / range_s if range_s > 0 else 0.5
            results.append({
                "move": move,
                "uci": move.to_uci(),
                "chinese": self._move_to_chinese(move, board),
                "score": float(normalized),
                "raw_score": float(score),
            })
        
        return results
    
    def get_best_move(self, fen: str) -> Optional[Dict]:
        """获取最佳着法"""
        moves = self.analyze(fen, top_n=1)
        return moves[0] if moves and "error" not in moves[0] else None
    
    def _move_to_chinese(self, move: Move, board: Board) -> str:
        """
        将着法转换为中文描述
        简化版本，实际需要更复杂的逻辑
        """
        piece = board.board[move.from_row, move.from_col]
        
        piece_names = {
            1: "帅", 2: "仕", 3: "相", 4: "马", 5: "车", 6: "炮", 7: "兵",
            8: "将", 9: "士", 10: "象", 11: "马", 12: "车", 13: "炮", 14: "卒",
        }
        
        name = piece_names.get(piece, "?")
        
        # 简化的列名转换
        col_names = "九八七六五四三二一" if board.is_red_piece(piece) else "１２３４５６７８９"
        from_col = col_names[move.from_col] if move.from_col < 9 else "?"
        
        # 移动方向
        if move.to_row < move.from_row:
            direction = "进" if board.is_red_piece(piece) else "退"
        elif move.to_row > move.from_row:
            direction = "退" if board.is_red_piece(piece) else "进"
        else:
            direction = "平"
        
        to_col = col_names[move.to_col] if move.to_col < 9 else "?"
        
        return f"{name}{from_col}{direction}{to_col}"


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Engine API Test\n")
    
    # 创建未训练的引擎（仅测试 API）
    engine = XiangqiEngine()
    engine.agent = DQNAgent(use_cuda=False)
    
    # 分析初始局面
    fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
    
    print(f"Analyzing: {fen}\n")
    
    moves = engine.analyze(fen, top_n=5)
    
    print("Top 5 moves:")
    for i, m in enumerate(moves):
        if "error" in m:
            print(f"  Error: {m['error']}")
        else:
            print(f"  {i+1}. {m['chinese']} ({m['uci']}) - Score: {m['score']:.3f}")
