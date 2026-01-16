"""
Zobrist 哈希
用于置换表的快速局面哈希
"""
import random
import numpy as np
from typing import Tuple


# 固定随机种子以保证可重复性
random.seed(42)


class ZobristHash:
    """
    Zobrist 哈希生成器
    
    为棋盘上每个位置的每种棋子生成随机数，
    通过 XOR 运算快速计算局面哈希值
    """
    
    def __init__(self):
        # 为每个位置(10x9)的每种棋子(15种，包括空)生成随机数
        self.piece_keys = np.zeros((15, 10, 9), dtype=np.uint64)
        
        for piece in range(15):
            for row in range(10):
                for col in range(9):
                    self.piece_keys[piece, row, col] = random.getrandbits(64)
        
        # 当前玩家的随机数（用于区分红方/黑方走棋）
        self.side_key = random.getrandbits(64)
    
    def compute_hash(self, board) -> int:
        """
        计算完整局面的哈希值
        
        Args:
            board: Board 对象
            
        Returns:
            64位哈希值
        """
        h = 0
        
        for row in range(10):
            for col in range(9):
                piece = board.board[row, col]
                if piece != 0:
                    h ^= self.piece_keys[piece, row, col]
        
        # 如果是黑方走棋，XOR side_key
        if board.current_player == -1:
            h ^= self.side_key
        
        return h
    
    def update_hash(
        self,
        old_hash: int,
        piece: int,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        captured: int = 0
    ) -> int:
        """
        增量更新哈希值（用于 make_move 后快速更新）
        
        Args:
            old_hash: 之前的哈希值
            piece: 移动的棋子
            from_pos: 起点
            to_pos: 终点
            captured: 被吃的棋子（0表示无）
            
        Returns:
            新的哈希值
        """
        h = old_hash
        
        # 移除起点的棋子
        h ^= self.piece_keys[piece, from_pos[0], from_pos[1]]
        
        # 如果有被吃的棋子，移除它
        if captured != 0:
            h ^= self.piece_keys[captured, to_pos[0], to_pos[1]]
        
        # 添加终点的棋子
        h ^= self.piece_keys[piece, to_pos[0], to_pos[1]]
        
        # 切换玩家
        h ^= self.side_key
        
        return h


# 全局 Zobrist 实例
ZOBRIST = ZobristHash()
