"""
Alpha-Beta 搜索引擎
中国象棋核心搜索算法
"""
from typing import List, Tuple, Optional
import time

from .board import Board, Piece
from .move import Move, MoveGenerator
from .evaluate import evaluate, MATE_SCORE


# 无穷大
INF = 999999


class SearchEngine:
    """
    Alpha-Beta 搜索引擎
    
    特性：
    - Alpha-Beta 剪枝
    - 迭代加深 (Iterative Deepening)
    - 基础走子排序 (MVV-LVA)
    """
    
    def __init__(self):
        self.nodes_searched = 0
        self.best_move = None
        self.search_depth = 0
        self.start_time = 0
        self.time_limit = 0
    
    def search(
        self,
        board: Board,
        depth: int = 4,
        time_limit_ms: int = 5000,
    ) -> Tuple[Optional[Move], int]:
        """
        搜索最佳着法
        
        Args:
            board: 棋盘
            depth: 最大搜索深度
            time_limit_ms: 时间限制（毫秒）
            
        Returns:
            (最佳着法, 评估分数)
        """
        self.nodes_searched = 0
        self.best_move = None
        self.start_time = time.time()
        self.time_limit = time_limit_ms / 1000
        
        best_score = -INF
        
        # 迭代加深
        for d in range(1, depth + 1):
            self.search_depth = d
            
            try:
                score = self._alpha_beta(
                    board, d, -INF, INF,
                    maximizing=(board.current_player == 1)
                )
                best_score = score
            except TimeoutError:
                break
            
            # 输出每层的最佳结果
            elapsed = time.time() - self.start_time
            print(f"Depth {d}: score={best_score:+5d}, "
                  f"move={self.best_move}, "
                  f"nodes={self.nodes_searched}, "
                  f"time={elapsed:.2f}s")
        
        return self.best_move, best_score
    
    def _alpha_beta(
        self,
        board: Board,
        depth: int,
        alpha: int,
        beta: int,
        maximizing: bool,
    ) -> int:
        """
        Alpha-Beta 搜索核心
        
        Args:
            board: 棋盘
            depth: 剩余深度
            alpha: Alpha 值
            beta: Beta 值
            maximizing: 是否最大化（红方）
            
        Returns:
            评估分数
        """
        self.nodes_searched += 1
        
        # 时间检查
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError()
        
        # 终止条件
        if depth == 0 or board.is_game_over():
            return evaluate(board)
        
        # 生成合法着法
        moves = self._generate_legal_moves(board)
        
        if len(moves) == 0:
            # 无子可走 = 被困毙
            if board.is_in_check(board.current_player == 1):
                return -MATE_SCORE if maximizing else MATE_SCORE
            return 0  # 和棋
        
        # 走子排序（MVV-LVA）
        moves = self._order_moves(board, moves)
        
        best_move_at_root = None
        
        if maximizing:
            max_eval = -INF
            for move in moves:
                board.make_move(move)
                
                # 检查走后自将（非法）
                if board.is_in_check(is_red=True):
                    board.unmake_move(move)
                    continue
                
                eval_score = self._alpha_beta(board, depth - 1, alpha, beta, False)
                board.unmake_move(move)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move_at_root = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            
            # 在根节点记录最佳着法
            if depth == self.search_depth and best_move_at_root:
                self.best_move = best_move_at_root
            
            return max_eval
        else:
            min_eval = INF
            for move in moves:
                board.make_move(move)
                
                # 检查走后自将（非法）
                if board.is_in_check(is_red=False):
                    board.unmake_move(move)
                    continue
                
                eval_score = self._alpha_beta(board, depth - 1, alpha, beta, True)
                board.unmake_move(move)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move_at_root = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            if depth == self.search_depth and best_move_at_root:
                self.best_move = best_move_at_root
            
            return min_eval
    
    def _generate_legal_moves(self, board: Board) -> List[Move]:
        """生成所有伪合法着法（self-check 在搜索中检查）"""
        gen = MoveGenerator(board)
        is_red = board.current_player == 1
        return gen.generate_moves(is_red)
    
    def _order_moves(self, board: Board, moves: List[Move]) -> List[Move]:
        """
        走子排序（MVV-LVA: Most Valuable Victim - Least Valuable Attacker）
        优先搜索吃子着法，且优先用小子吃大子
        """
        def move_score(move: Move) -> int:
            target = board.board[move.to_row, move.to_col]
            attacker = board.board[move.from_row, move.from_col]
            
            if target == 0:
                return 0
            
            # MVV-LVA: 被吃子价值 * 10 - 攻击子价值
            from .evaluate import PIECE_VALUES
            victim_value = PIECE_VALUES.get(target, 0)
            attacker_value = PIECE_VALUES.get(attacker, 0)
            return victim_value * 10 - attacker_value
        
        return sorted(moves, key=move_score, reverse=True)


def perft(board: Board, depth: int) -> int:
    """
    Perft 测试：计算指定深度的节点数
    用于验证走法生成的正确性
    
    Args:
        board: 棋盘
        depth: 深度
        
    Returns:
        节点数
    """
    if depth == 0:
        return 1
    
    nodes = 0
    gen = MoveGenerator(board)
    is_red = board.current_player == 1
    moves = gen.generate_moves(is_red)
    
    for move in moves:
        board.make_move(move)
        
        # 检查自将（非法着法）
        if not board.is_in_check(is_red):
            nodes += perft(board, depth - 1)
        
        board.unmake_move(move)
    
    return nodes


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Search Engine Test\n")
    
    board = Board()
    board.render()
    
    # Perft 测试
    print("\nPerft test:")
    for d in range(1, 4):
        nodes = perft(board, d)
        print(f"  Depth {d}: {nodes} nodes")
    
    # 搜索测试
    print("\nSearch test (depth=4):")
    engine = SearchEngine()
    best_move, score = engine.search(board, depth=4, time_limit_ms=10000)
    
    print(f"\nBest move: {best_move}")
    print(f"Score: {score}")
    print(f"Nodes searched: {engine.nodes_searched}")
