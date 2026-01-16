"""
增强版搜索引擎 v3
集成 Policy-Value 网络：
- Policy: 用于 move ordering
- Value: 用于叶节点评估（可选融合）
"""
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import IntEnum
import time
import numpy as np

import torch
import torch.nn.functional as F

from .board import Board, Piece
from .move import Move, MoveGenerator
from .evaluate_v2 import evaluate, MATE_SCORE, PIECE_VALUES
from .zobrist import ZOBRIST
from .policy_value_net import SimplePolicyValueNet, MoveEncoder


INF = 999999


class TTFlag(IntEnum):
    EXACT = 0
    ALPHA = 1
    BETA = 2


@dataclass
class TTEntry:
    hash_key: int
    depth: int
    score: int
    flag: TTFlag
    best_move: Optional[Move]


class TranspositionTable:
    def __init__(self, size_mb: int = 64):
        entry_size = 40
        self.size = (size_mb * 1024 * 1024) // entry_size
        self.table: Dict[int, TTEntry] = {}
        self.hits = 0
        self.stores = 0
    
    def probe(self, hash_key: int) -> Optional[TTEntry]:
        entry = self.table.get(hash_key % self.size)
        if entry and entry.hash_key == hash_key:
            self.hits += 1
            return entry
        return None
    
    def store(self, hash_key: int, depth: int, score: int, flag: TTFlag, best_move: Optional[Move]):
        self.stores += 1
        self.table[hash_key % self.size] = TTEntry(hash_key, depth, score, flag, best_move)
    
    def clear(self):
        self.table.clear()
        self.hits = 0
        self.stores = 0


class SearchEngineV3:
    """
    增强版搜索引擎 v3
    
    新增特性：
    - Policy 网络参与 move ordering
    - Value 网络辅助叶节点评估（可选）
    """
    
    def __init__(
        self,
        tt_size_mb: int = 64,
        net_path: Optional[str] = None,
        use_cuda: bool = True,
        policy_weight: float = 1000.0,  # Policy 在 move ordering 中的权重
        use_net_value: bool = False,    # 是否使用网络 value
        value_blend: float = 0.3,       # 网络 value 的混合比例
    ):
        self.tt = TranspositionTable(tt_size_mb)
        self.nodes_searched = 0
        self.qnodes = 0
        self.best_move = None
        self.search_depth = 0
        self.start_time = 0
        self.time_limit = 0
        
        self.killer_moves = [[None, None] for _ in range(64)]
        self.history = {}
        
        # 网络相关
        self.policy_weight = policy_weight
        self.use_net_value = use_net_value
        self.value_blend = value_blend
        self.move_encoder = MoveEncoder()
        
        # 初始化网络
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.net = None
        
        if net_path:
            self._load_network(net_path)
    
    def _load_network(self, path: str):
        """加载 Policy-Value 网络"""
        try:
            self.net = SimplePolicyValueNet(action_size=8100).to(self.device)
            checkpoint = torch.load(path, map_location=self.device)
            self.net.load_state_dict(checkpoint['net'])
            self.net.eval()
            print(f"[SearchV3] Loaded network: {path}")
        except Exception as e:
            print(f"[SearchV3] Failed to load network: {e}")
            self.net = None
    
    def _get_state(self, board: Board) -> np.ndarray:
        """棋盘转网络输入"""
        state = np.zeros((15, 10, 9), dtype=np.float32)
        for row in range(10):
            for col in range(9):
                piece = board.board[row, col]
                if piece > 0:
                    state[piece - 1, row, col] = 1.0
        if board.current_player == 1:
            state[14, :, :] = 1.0
        return state
    
    def _get_policy_priors(self, board: Board) -> Optional[np.ndarray]:
        """获取网络的 policy 先验"""
        if self.net is None:
            return None
        
        state = self._get_state(board)
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy_logits, _ = self.net(x)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        
        return policy
    
    def _get_net_value(self, board: Board) -> Optional[float]:
        """获取网络的 value 评估"""
        if self.net is None:
            return None
        
        state = self._get_state(board)
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, value = self.net(x)
            v = value.item()
        
        # 网络输出是当前方视角 [-1, 1]
        # 转换为红方视角（与手工 eval 一致）
        if board.current_player == -1:
            v = -v
        
        # 缩放到与手工 eval 相同量级
        return v * 500
    
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
        
        hash_key = ZOBRIST.compute_hash(board)
        
        # 预计算 policy priors（根节点只计算一次）
        self.root_policy = self._get_policy_priors(board)
        
        best_score = -INF
        
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
            net_status = "NET" if self.net else "NO-NET"
            print(f"Depth {d} [{net_status}]: score={best_score:+5d}, "
                  f"move={self.best_move}, "
                  f"nodes={self.nodes_searched:,} (q:{self.qnodes:,}), "
                  f"time={elapsed:.2f}s")
        
        return self.best_move, best_score
    
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
        
        if self.nodes_searched % 1000 == 0:
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError()
        
        tt_entry = self.tt.probe(hash_key)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TTFlag.EXACT:
                return tt_entry.score
            elif tt_entry.flag == TTFlag.ALPHA and tt_entry.score <= alpha:
                return alpha
            elif tt_entry.flag == TTFlag.BETA and tt_entry.score >= beta:
                return beta
        
        if depth <= 0:
            return self._quiescence(board, alpha, beta, maximizing)
        
        if board.is_game_over():
            return -MATE_SCORE + ply if maximizing else MATE_SCORE - ply
        
        moves = self._generate_legal_moves(board)
        if len(moves) == 0:
            if board.is_in_check(board.current_player == 1):
                return -MATE_SCORE + ply
            return 0
        
        # 使用 policy priors 进行 move ordering
        moves = self._order_moves_with_policy(board, moves, ply, tt_entry)
        
        best_move_here = None
        original_alpha = alpha
        
        if maximizing:
            max_eval = -INF
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
                    if captured == 0:
                        self._store_killer(move, ply)
                        self._update_history(move, depth)
                    break
            
            if max_eval <= original_alpha:
                flag = TTFlag.ALPHA
            elif max_eval >= beta:
                flag = TTFlag.BETA
            else:
                flag = TTFlag.EXACT
            self.tt.store(hash_key, depth, max_eval, flag, best_move_here)
            
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
        """静态搜索"""
        self.qnodes += 1
        
        # 基础评估
        hand_eval = evaluate(board)
        
        # 可选：融合网络 value
        if self.use_net_value:
            net_val = self._get_net_value(board)
            if net_val is not None:
                stand_pat = int((1 - self.value_blend) * hand_eval + self.value_blend * net_val)
            else:
                stand_pat = hand_eval
        else:
            stand_pat = hand_eval
        
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
        
        moves = self._generate_captures(board)
        moves = self._order_moves_simple(board, moves)
        
        for move in moves:
            board.make_move(move)
            
            is_red = board.current_player == -1
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
        gen = MoveGenerator(board)
        is_red = board.current_player == 1
        return gen.generate_moves(is_red)
    
    def _generate_captures(self, board: Board) -> List[Move]:
        all_moves = self._generate_legal_moves(board)
        return [m for m in all_moves if board.board[m.to_row, m.to_col] != 0]
    
    def _order_moves_with_policy(
        self,
        board: Board,
        moves: List[Move],
        ply: int,
        tt_entry: Optional[TTEntry]
    ) -> List[Move]:
        """使用 Policy 网络进行 move ordering"""
        
        # 获取 policy priors（非根节点每次都计算）
        if ply == 0:
            policy = self.root_policy
        else:
            policy = self._get_policy_priors(board) if self.net else None
        
        def move_score(move: Move) -> float:
            score = 0.0
            
            # TT 最佳着法优先
            if tt_entry and tt_entry.best_move:
                if (move.from_row == tt_entry.best_move.from_row and
                    move.from_col == tt_entry.best_move.from_col and
                    move.to_row == tt_entry.best_move.to_row and
                    move.to_col == tt_entry.best_move.to_col):
                    return 1000000.0
            
            target = board.board[move.to_row, move.to_col]
            attacker = board.board[move.from_row, move.from_col]
            
            # MVV-LVA
            if target != 0:
                victim_value = PIECE_VALUES.get(target, 0)
                attacker_value = PIECE_VALUES.get(attacker, 0)
                score += 10000.0 + victim_value * 10 - attacker_value
            
            # Policy prior（关键改进！）
            if policy is not None:
                action = self.move_encoder.encode_move(
                    move.from_row, move.from_col,
                    move.to_row, move.to_col
                )
                policy_prob = policy[action]
                # 使用 log 避免小概率被忽略
                score += self.policy_weight * np.log(policy_prob + 1e-8)
            
            # Killer moves
            if ply < len(self.killer_moves):
                for km in self.killer_moves[ply]:
                    if km and self._moves_equal(move, km):
                        score += 5000.0
                        break
            
            # History heuristic
            key = (move.from_row, move.from_col, move.to_row, move.to_col)
            score += self.history.get(key, 0)
            
            return score
        
        return sorted(moves, key=move_score, reverse=True)
    
    def _order_moves_simple(self, board: Board, moves: List[Move]) -> List[Move]:
        def score(move: Move) -> int:
            target = board.board[move.to_row, move.to_col]
            return PIECE_VALUES.get(target, 0)
        return sorted(moves, key=score, reverse=True)
    
    def _store_killer(self, move: Move, ply: int):
        if ply >= len(self.killer_moves):
            return
        if not self._moves_equal(move, self.killer_moves[ply][0]):
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = move
    
    def _update_history(self, move: Move, depth: int):
        key = (move.from_row, move.from_col, move.to_row, move.to_col)
        self.history[key] = self.history.get(key, 0) + depth * depth
    
    def _moves_equal(self, m1: Move, m2: Optional[Move]) -> bool:
        if m2 is None:
            return False
        return (m1.from_row == m2.from_row and m1.from_col == m2.from_col and
                m1.to_row == m2.to_row and m1.to_col == m2.to_col)


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Search Engine V3 (with Network) Test\n")
    
    board = Board()
    board.render()
    
    # 测试无网络
    print("\n[1] Without network:")
    engine = SearchEngineV3(tt_size_mb=16)
    best_move, score = engine.search(board, depth=3, time_limit_ms=5000)
    print(f"Best: {best_move}, Score: {score}")
    
    # 测试有网络
    print("\n[2] With network:")
    engine_net = SearchEngineV3(
        tt_size_mb=16,
        net_path="checkpoints_az/az_net_game30.pt",
        policy_weight=1000.0,
    )
    best_move2, score2 = engine_net.search(board, depth=3, time_limit_ms=5000)
    print(f"Best: {best_move2}, Score: {score2}")
