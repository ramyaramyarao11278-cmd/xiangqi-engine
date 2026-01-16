"""
增强版 Alpha-Beta 搜索引擎
包含：置换表 (TT) + Quiescence Search + Killer Moves
"""
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import IntEnum
import time

from .board import Board, Piece
from .move import Move, MoveGenerator
from .evaluate_v2 import evaluate, MATE_SCORE, PIECE_VALUES
from .zobrist import ZOBRIST


INF = 999999


class TTFlag(IntEnum):
    """置换表条目类型"""
    EXACT = 0    # 精确值
    ALPHA = 1    # 上界 (fail-low)
    BETA = 2     # 下界 (fail-high)


@dataclass
class TTEntry:
    """置换表条目"""
    hash_key: int
    depth: int
    score: int
    flag: TTFlag
    best_move: Optional[Move]


class TranspositionTable:
    """置换表"""
    
    def __init__(self, size_mb: int = 64):
        # 每个条目约 40 字节，计算条目数
        entry_size = 40
        self.size = (size_mb * 1024 * 1024) // entry_size
        self.table: Dict[int, TTEntry] = {}
        self.hits = 0
        self.stores = 0
    
    def probe(self, hash_key: int) -> Optional[TTEntry]:
        """查询置换表"""
        entry = self.table.get(hash_key % self.size)
        if entry and entry.hash_key == hash_key:
            self.hits += 1
            return entry
        return None
    
    def store(self, hash_key: int, depth: int, score: int, flag: TTFlag, best_move: Optional[Move]):
        """存储到置换表"""
        self.stores += 1
        self.table[hash_key % self.size] = TTEntry(hash_key, depth, score, flag, best_move)
    
    def clear(self):
        """清空置换表"""
        self.table.clear()
        self.hits = 0
        self.stores = 0


class SearchEngineV2:
    """
    增强版搜索引擎
    
    特性：
    - Alpha-Beta + 置换表
    - Quiescence Search (静态搜索)
    - Killer Moves
    - MVV-LVA 排序
    - 迭代加深
    """
    
    def __init__(self, tt_size_mb: int = 64):
        self.tt = TranspositionTable(tt_size_mb)
        self.nodes_searched = 0
        self.qnodes = 0
        self.best_move = None
        self.search_depth = 0
        self.start_time = 0
        self.time_limit = 0
        
        # Killer moves: 每层保存 2 个
        self.killer_moves = [[None, None] for _ in range(64)]
        
        # History heuristic
        self.history = {}
    
    def search(
        self,
        board: Board,
        depth: int = 6,
        time_limit_ms: int = 10000,
    ) -> Tuple[Optional[Move], int]:
        """搜索最佳着法"""
        self.nodes_searched = 0
        self.qnodes = 0
        self.best_move = None
        self.start_time = time.time()
        self.time_limit = time_limit_ms / 1000
        self.killer_moves = [[None, None] for _ in range(64)]
        self.history.clear()
        
        # 计算初始哈希
        hash_key = ZOBRIST.compute_hash(board)
        
        best_score = -INF
        
        # 迭代加深
        for d in range(1, depth + 1):
            self.search_depth = d
            
            try:
                score = self._alpha_beta(
                    board, hash_key, d, -INF, INF,
                    maximizing=(board.current_player == 1),
                    ply=0
                )
                best_score = score
            except TimeoutError:
                break
            
            elapsed = time.time() - self.start_time
            nps = self.nodes_searched / elapsed if elapsed > 0 else 0
            print(f"Depth {d}: score={best_score:+5d}, "
                  f"move={self.best_move}, "
                  f"nodes={self.nodes_searched:,} (q:{self.qnodes:,}), "
                  f"tt_hits={self.tt.hits:,}, "
                  f"nps={nps:,.0f}, "
                  f"time={elapsed:.2f}s")
        
        return self.best_move, best_score
    
    def search_top_k(
        self,
        board: Board,
        depth: int = 4,
        time_limit_ms: int = 5000,
        top_k: int = 5,
    ) -> List[Tuple[Move, int]]:
        """
        搜索 Top-K 着法及其分数
        
        用于生成 soft label 训练数据
        
        Args:
            board: 棋盘
            depth: 搜索深度
            time_limit_ms: 时间限制
            top_k: 返回前 K 个着法
            
        Returns:
            [(move1, score1), (move2, score2), ...] 按分数降序
        """
        self.nodes_searched = 0
        self.qnodes = 0
        self.start_time = time.time()
        self.time_limit = time_limit_ms / 1000
        self.killer_moves = [[None, None] for _ in range(64)]
        self.history.clear()
        
        hash_key = ZOBRIST.compute_hash(board)
        
        # 生成所有合法着法
        moves = self._generate_legal_moves(board)
        
        # 过滤自将
        legal_moves = []
        is_red = board.current_player == 1
        for move in moves:
            board.make_move(move)
            if not board.is_in_check(is_red):
                legal_moves.append(move)
            board.unmake_move(move)
        
        if not legal_moves:
            return []
        
        # 对每个合法着法进行窗口搜索
        move_scores = []
        maximizing = board.current_player == 1
        
        for move in legal_moves:
            captured = board.board[move.to_row, move.to_col]
            piece = board.board[move.from_row, move.from_col]
            new_hash = ZOBRIST.update_hash(
                hash_key, piece,
                (move.from_row, move.from_col),
                (move.to_row, move.to_col),
                captured
            )
            
            board.make_move(move)
            
            try:
                # 搜索这个着法后的局面
                if depth > 1:
                    score = self._alpha_beta(
                        board, new_hash, depth - 1, -INF, INF,
                        not maximizing, ply=1
                    )
                else:
                    score = evaluate(board)
                
                # 从当前方视角
                if not maximizing:
                    score = -score
                    
            except TimeoutError:
                score = evaluate(board)
                if not maximizing:
                    score = -score
            
            board.unmake_move(move)
            
            # 分数转换为"当前执子方视角"（越大越好）
            # alpha_beta 返回的是红方视角分数
            # 如果当前是红方走，score 保持不变
            # 如果当前是黑方走，score 取反
            score_side = score if maximizing else -score
            move_scores.append((move, score_side))
        
        # 按当前执子方视角分数排序（高分优先）
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        return move_scores[:top_k]
    
    def _alpha_beta(
        self,
        board: Board,
        hash_key: int,
        depth: int,
        alpha: int,
        beta: int,
        maximizing: bool,
        ply: int,
    ) -> int:
        """Alpha-Beta 搜索核心"""
        self.nodes_searched += 1
        
        # 时间检查
        if self.nodes_searched % 1000 == 0:
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError()
        
        # 置换表查询
        tt_entry = self.tt.probe(hash_key)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TTFlag.EXACT:
                return tt_entry.score
            elif tt_entry.flag == TTFlag.ALPHA and tt_entry.score <= alpha:
                return alpha
            elif tt_entry.flag == TTFlag.BETA and tt_entry.score >= beta:
                return beta
        
        # 叶节点：进入静态搜索
        if depth <= 0:
            return self._quiescence(board, alpha, beta, maximizing)
        
        # 游戏结束检查
        if board.is_game_over():
            return -MATE_SCORE + ply if maximizing else MATE_SCORE - ply
        
        # 生成着法
        moves = self._generate_legal_moves(board)
        if len(moves) == 0:
            if board.is_in_check(board.current_player == 1):
                return -MATE_SCORE + ply  # 被将死
            return 0  # 和棋
        
        # 着法排序
        moves = self._order_moves(board, moves, ply, tt_entry)
        
        best_move_here = None
        original_alpha = alpha
        
        if maximizing:
            max_eval = -INF
            for move in moves:
                # 执行着法
                captured = board.board[move.to_row, move.to_col]
                piece = board.board[move.from_row, move.from_col]
                new_hash = ZOBRIST.update_hash(
                    hash_key, piece,
                    (move.from_row, move.from_col),
                    (move.to_row, move.to_col),
                    captured
                )
                
                board.make_move(move)
                
                # 检查自将
                if board.is_in_check(is_red=True):
                    board.unmake_move(move)
                    continue
                
                eval_score = self._alpha_beta(
                    board, new_hash, depth - 1, alpha, beta, False, ply + 1
                )
                board.unmake_move(move)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move_here = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    # Beta cutoff - 记录 killer move
                    if captured == 0:  # 非吃子着法
                        self._store_killer(move, ply)
                        self._update_history(move, depth)
                    break
            
            # 存储到置换表
            if max_eval <= original_alpha:
                flag = TTFlag.ALPHA
            elif max_eval >= beta:
                flag = TTFlag.BETA
            else:
                flag = TTFlag.EXACT
            self.tt.store(hash_key, depth, max_eval, flag, best_move_here)
            
            # 根节点记录最佳着法
            if ply == 0 and best_move_here:
                self.best_move = best_move_here
            
            return max_eval
        else:
            min_eval = INF
            for move in moves:
                captured = board.board[move.to_row, move.to_col]
                piece = board.board[move.from_row, move.from_col]
                new_hash = ZOBRIST.update_hash(
                    hash_key, piece,
                    (move.from_row, move.from_col),
                    (move.to_row, move.to_col),
                    captured
                )
                
                board.make_move(move)
                
                if board.is_in_check(is_red=False):
                    board.unmake_move(move)
                    continue
                
                eval_score = self._alpha_beta(
                    board, new_hash, depth - 1, alpha, beta, True, ply + 1
                )
                board.unmake_move(move)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move_here = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    if captured == 0:
                        self._store_killer(move, ply)
                        self._update_history(move, depth)
                    break
            
            if min_eval <= original_alpha:
                flag = TTFlag.ALPHA
            elif min_eval >= beta:
                flag = TTFlag.BETA
            else:
                flag = TTFlag.EXACT
            self.tt.store(hash_key, depth, min_eval, flag, best_move_here)
            
            if ply == 0 and best_move_here:
                self.best_move = best_move_here
            
            return min_eval
    
    def _quiescence(self, board: Board, alpha: int, beta: int, maximizing: bool) -> int:
        """
        静态搜索（Quiescence Search）
        只搜索吃子着法，直到局面"安静"
        """
        self.qnodes += 1
        
        # 站立评估（standing pat）
        stand_pat = evaluate(board)
        
        if maximizing:
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if stand_pat < beta:
                beta = stand_pat
        
        # 只生成吃子着法
        moves = self._generate_captures(board)
        moves = self._order_moves_simple(board, moves)
        
        for move in moves:
            board.make_move(move)
            
            # 检查自将
            is_red = board.current_player == -1  # 刚走完，检查走之前的方
            if board.is_in_check(is_red):
                board.unmake_move(move)
                continue
            
            score = self._quiescence(board, alpha, beta, not maximizing)
            board.unmake_move(move)
            
            if maximizing:
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            else:
                if score <= alpha:
                    return alpha
                if score < beta:
                    beta = score
        
        return alpha if maximizing else beta
    
    def _generate_legal_moves(self, board: Board) -> List[Move]:
        """生成所有伪合法着法"""
        gen = MoveGenerator(board)
        is_red = board.current_player == 1
        return gen.generate_moves(is_red)
    
    def _generate_captures(self, board: Board) -> List[Move]:
        """只生成吃子着法"""
        all_moves = self._generate_legal_moves(board)
        return [m for m in all_moves if board.board[m.to_row, m.to_col] != 0]
    
    def _order_moves(self, board: Board, moves: List[Move], ply: int, tt_entry: Optional[TTEntry]) -> List[Move]:
        """着法排序"""
        def move_score(move: Move) -> int:
            score = 0
            
            # TT 最佳着法优先
            if tt_entry and tt_entry.best_move:
                if (move.from_row == tt_entry.best_move.from_row and
                    move.from_col == tt_entry.best_move.from_col and
                    move.to_row == tt_entry.best_move.to_row and
                    move.to_col == tt_entry.best_move.to_col):
                    return 100000
            
            target = board.board[move.to_row, move.to_col]
            attacker = board.board[move.from_row, move.from_col]
            
            # MVV-LVA
            if target != 0:
                victim_value = PIECE_VALUES.get(target, 0)
                attacker_value = PIECE_VALUES.get(attacker, 0)
                score += 10000 + victim_value * 10 - attacker_value
            
            # Killer moves
            if ply < len(self.killer_moves):
                for km in self.killer_moves[ply]:
                    if km and self._moves_equal(move, km):
                        score += 5000
                        break
            
            # History heuristic
            key = (move.from_row, move.from_col, move.to_row, move.to_col)
            score += self.history.get(key, 0)
            
            return score
        
        return sorted(moves, key=move_score, reverse=True)
    
    def _order_moves_simple(self, board: Board, moves: List[Move]) -> List[Move]:
        """简化排序（用于 quiescence）"""
        def score(move: Move) -> int:
            target = board.board[move.to_row, move.to_col]
            return PIECE_VALUES.get(target, 0)
        return sorted(moves, key=score, reverse=True)
    
    def _store_killer(self, move: Move, ply: int):
        """存储 killer move"""
        if ply >= len(self.killer_moves):
            return
        if not self._moves_equal(move, self.killer_moves[ply][0]):
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = move
    
    def _update_history(self, move: Move, depth: int):
        """更新 history heuristic"""
        key = (move.from_row, move.from_col, move.to_row, move.to_col)
        self.history[key] = self.history.get(key, 0) + depth * depth
    
    def _moves_equal(self, m1: Move, m2: Optional[Move]) -> bool:
        """比较两个着法是否相同"""
        if m2 is None:
            return False
        return (m1.from_row == m2.from_row and m1.from_col == m2.from_col and
                m1.to_row == m2.to_row and m1.to_col == m2.to_col)


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Enhanced Search Engine Test\n")
    
    board = Board()
    board.render()
    
    print("\nSearch (depth=6, time=15s):")
    engine = SearchEngineV2(tt_size_mb=32)
    best_move, score = engine.search(board, depth=6, time_limit_ms=15000)
    
    print(f"\nResult:")
    print(f"  Best move: {best_move}")
    print(f"  Score: {score:+d}")
    print(f"  Nodes: {engine.nodes_searched:,} (q:{engine.qnodes:,})")
    print(f"  TT hits: {engine.tt.hits:,}")
