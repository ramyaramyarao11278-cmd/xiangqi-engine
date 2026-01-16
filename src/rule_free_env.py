"""
无规则约束学习环境 (Rule-Free Learning)

核心思想：
- 不告诉 Agent 任何规则
- Agent 可以尝试任何移动 (from_pos, to_pos)
- 违规移动给负奖励信号
- Agent 通过试错自己学会规则

这种方式更接近人类学习象棋的方式：
1. 尝试一个移动
2. 被告知"不行，这样走不对"（负信号）
3. 逐渐悟出规则
"""
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import IntEnum

from .board import Board, Piece
from .move import Move, MoveGenerator


class MoveResult(IntEnum):
    """移动结果类型"""
    VALID = 0           # 合法移动
    NO_PIECE = 1        # 起点无棋子
    WRONG_COLOR = 2     # 移动对方棋子
    SAME_COLOR = 3      # 吃自己棋子
    INVALID_PATTERN = 4 # 不符合棋子走法
    BLOCKED = 5         # 被阻挡（蹩马腿等）
    OUT_OF_BOARD = 6    # 超出棋盘
    SELF_CHECK = 7      # 送将


@dataclass
class MoveSignal:
    """移动信号"""
    result: MoveResult
    reward: float
    message: str


class RuleFreeEnv:
    """
    无规则约束的象棋学习环境
    
    特点：
    - Agent 可以尝试任何 (from, to) 移动
    - 违规移动返回不同类型的负信号
    - 通过奖惩信号让 Agent 自己学会规则
    """
    
    # 违规惩罚（可调整）
    PENALTIES = {
        MoveResult.NO_PIECE: -0.5,        # 起点没棋子
        MoveResult.WRONG_COLOR: -0.5,     # 移动对方棋子
        MoveResult.SAME_COLOR: -0.8,      # 吃自己人
        MoveResult.INVALID_PATTERN: -0.3, # 走法不对
        MoveResult.BLOCKED: -0.2,         # 被堵住
        MoveResult.OUT_OF_BOARD: -1.0,    # 走出棋盘
        MoveResult.SELF_CHECK: -0.7,      # 送将
    }
    
    # 正向奖励
    REWARDS = {
        "valid_move": 0.05,       # 合法移动（小奖励）
        "capture": 0.3,           # 吃子
        "check": 0.2,             # 将军
        "checkmate": 10.0,        # 将死
        "win": 10.0,              # 获胜
        "lose": -10.0,            # 失败
    }
    
    # 棋子价值（吃子奖励系数）
    PIECE_VALUES = {
        Piece.R_KING: 100, Piece.B_KING: 100,
        Piece.R_ROOK: 9, Piece.B_ROOK: 9,
        Piece.R_CANNON: 4.5, Piece.B_CANNON: 4.5,
        Piece.R_KNIGHT: 4, Piece.B_KNIGHT: 4,
        Piece.R_BISHOP: 2, Piece.B_BISHOP: 2,
        Piece.R_ADVISOR: 2, Piece.B_ADVISOR: 2,
        Piece.R_PAWN: 1, Piece.B_PAWN: 1,
    }
    
    def __init__(self, max_moves: int = 300, max_invalid_moves: int = 50):
        """
        Args:
            max_moves: 最大有效步数
            max_invalid_moves: 连续违规次数上限（超过则结束）
        """
        self.board = Board()
        self.move_gen = MoveGenerator(self.board)
        self.max_moves = max_moves
        self.max_invalid_moves = max_invalid_moves
        
        # 统计
        self.valid_move_count = 0
        self.invalid_move_count = 0
        self.consecutive_invalid = 0  # 连续违规次数
        self.violation_history = []   # 违规历史（用于分析）
        
        self.done = False
        self.winner = 0
    
    def reset(self) -> np.ndarray:
        """重置游戏"""
        self.board.reset()
        self.move_gen = MoveGenerator(self.board)
        
        self.valid_move_count = 0
        self.invalid_move_count = 0
        self.consecutive_invalid = 0
        self.violation_history = []
        
        self.done = False
        self.winner = 0
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        获取状态
        
        包含：
        - 14 通道：各类棋子位置
        - 1 通道：当前玩家
        - 1 通道：违规计数（归一化）
        """
        state = np.zeros((16, 10, 9), dtype=np.float32)
        
        # 棋子位置
        for row in range(10):
            for col in range(9):
                piece = self.board.board[row, col]
                if piece > 0:
                    state[piece - 1, row, col] = 1.0
        
        # 当前玩家
        if self.board.current_player == 1:
            state[14, :, :] = 1.0
        
        # 违规计数信号（让 Agent 知道自己犯了多少错）
        violation_ratio = min(self.invalid_move_count / 100, 1.0)
        state[15, :, :] = violation_ratio
        
        return state
    
    def step(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步移动（不检查是否合法，而是通过信号反馈）
        
        Args:
            from_pos: (row, col) 起点
            to_pos: (row, col) 终点
            
        Returns:
            next_state: 下一状态
            reward: 奖励信号
            done: 是否结束
            info: 包含移动结果信号
        """
        if self.done:
            return self.get_state(), 0, True, {"error": "Game ended"}
        
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # 验证移动并获取信号
        signal = self._validate_and_execute(from_row, from_col, to_row, to_col)
        
        info = {
            "signal": signal.result,
            "signal_name": signal.result.name,
            "message": signal.message,
            "valid_moves": self.valid_move_count,
            "invalid_moves": self.invalid_move_count,
            "consecutive_invalid": self.consecutive_invalid,
        }
        
        # 检查游戏结束条件
        self._check_game_end()
        
        if self.done:
            info["winner"] = self.winner
        
        return self.get_state(), signal.reward, self.done, info
    
    def step_by_action(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        通过动作索引执行移动
        
        action = from_idx * 90 + to_idx
        from_idx = row * 9 + col
        """
        from_idx = action // 90
        to_idx = action % 90
        
        from_pos = (from_idx // 9, from_idx % 9)
        to_pos = (to_idx // 9, to_idx % 9)
        
        return self.step(from_pos, to_pos)
    
    def _validate_and_execute(self, from_row, from_col, to_row, to_col) -> MoveSignal:
        """验证移动并执行（或返回违规信号）"""
        
        # 1. 检查是否在棋盘内
        if not (0 <= from_row < 10 and 0 <= from_col < 9 and
                0 <= to_row < 10 and 0 <= to_col < 9):
            self._record_violation(MoveResult.OUT_OF_BOARD)
            return MoveSignal(
                MoveResult.OUT_OF_BOARD,
                self.PENALTIES[MoveResult.OUT_OF_BOARD],
                "Move out of board"
            )
        
        piece = self.board.board[from_row, from_col]
        
        # 2. 检查起点是否有棋子
        if piece == 0:
            self._record_violation(MoveResult.NO_PIECE)
            return MoveSignal(
                MoveResult.NO_PIECE,
                self.PENALTIES[MoveResult.NO_PIECE],
                "No piece at starting position"
            )
        
        # 3. 检查是否移动己方棋子
        is_red_piece = 1 <= piece <= 7
        is_red_turn = self.board.current_player == 1
        
        if is_red_piece != is_red_turn:
            self._record_violation(MoveResult.WRONG_COLOR)
            return MoveSignal(
                MoveResult.WRONG_COLOR,
                self.PENALTIES[MoveResult.WRONG_COLOR],
                "Moving opponent's piece"
            )
        
        target = self.board.board[to_row, to_col]
        
        # 4. 检查是否吃自己人
        if target != 0:
            is_target_red = 1 <= target <= 7
            if is_target_red == is_red_turn:
                self._record_violation(MoveResult.SAME_COLOR)
                return MoveSignal(
                    MoveResult.SAME_COLOR,
                    self.PENALTIES[MoveResult.SAME_COLOR],
                    "Capturing own piece"
                )
        
        # 5. 检查是否符合棋子走法规则
        if not self._is_valid_piece_move(piece, from_row, from_col, to_row, to_col):
            self._record_violation(MoveResult.INVALID_PATTERN)
            return MoveSignal(
                MoveResult.INVALID_PATTERN,
                self.PENALTIES[MoveResult.INVALID_PATTERN],
                "Invalid move pattern for this piece"
            )
        
        # === 移动合法！执行移动 ===
        self._execute_move(from_row, from_col, to_row, to_col, piece, target)
        
        # 计算奖励
        reward = self.REWARDS["valid_move"]
        
        # 吃子奖励
        if target != 0:
            capture_value = self.PIECE_VALUES.get(target, 1) / 10
            reward += self.REWARDS["capture"] + capture_value
        
        # 将军奖励
        if self._is_checking():
            reward += self.REWARDS["check"]
        
        return MoveSignal(MoveResult.VALID, reward, "Valid move")
    
    def _is_valid_piece_move(self, piece, fr, fc, tr, tc) -> bool:
        """检查棋子走法是否符合规则（简化版本）"""
        piece_type = piece if piece <= 7 else piece - 7
        dr, dc = tr - fr, tc - fc
        
        if piece_type == 1:  # 将/帅
            # 只能在九宫内，一步一格
            if abs(dr) + abs(dc) != 1:
                return False
            is_red = piece <= 7
            if is_red:
                return 7 <= tr <= 9 and 3 <= tc <= 5
            else:
                return 0 <= tr <= 2 and 3 <= tc <= 5
        
        elif piece_type == 2:  # 士/仕
            # 斜走一格，只能在九宫
            if abs(dr) != 1 or abs(dc) != 1:
                return False
            is_red = piece <= 7
            if is_red:
                return 7 <= tr <= 9 and 3 <= tc <= 5
            else:
                return 0 <= tr <= 2 and 3 <= tc <= 5
        
        elif piece_type == 3:  # 象/相
            # 田字，不能过河，检查象眼
            if abs(dr) != 2 or abs(dc) != 2:
                return False
            eye_r, eye_c = fr + dr // 2, fc + dc // 2
            if self.board.board[eye_r, eye_c] != 0:  # 塞象眼
                return False
            is_red = piece <= 7
            if is_red:
                return tr >= 5  # 不过河
            else:
                return tr <= 4
        
        elif piece_type == 4:  # 马
            # 日字，检查马腿
            if not ((abs(dr) == 2 and abs(dc) == 1) or (abs(dr) == 1 and abs(dc) == 2)):
                return False
            # 检查蹩马腿
            if abs(dr) == 2:
                leg_r = fr + (1 if dr > 0 else -1)
                if self.board.board[leg_r, fc] != 0:
                    return False
            else:
                leg_c = fc + (1 if dc > 0 else -1)
                if self.board.board[fr, leg_c] != 0:
                    return False
            return True
        
        elif piece_type == 5:  # 车
            # 直线移动
            if dr != 0 and dc != 0:
                return False
            # 检查路径上无障碍
            return self._path_clear(fr, fc, tr, tc)
        
        elif piece_type == 6:  # 炮
            # 直线移动
            if dr != 0 and dc != 0:
                return False
            target = self.board.board[tr, tc]
            pieces_between = self._count_pieces_between(fr, fc, tr, tc)
            if target == 0:
                return pieces_between == 0  # 移动时路径无障碍
            else:
                return pieces_between == 1  # 吃子需要隔一个
        
        elif piece_type == 7:  # 兵/卒
            is_red = piece <= 7
            # 只能前进或横移（过河后）
            if is_red:
                if dr > 0:  # 红兵不能后退
                    return False
                if dr == 0:  # 横移
                    return fr <= 4 and abs(dc) == 1  # 必须过河
                return dr == -1 and dc == 0  # 前进一步
            else:
                if dr < 0:  # 黑卒不能后退
                    return False
                if dr == 0:
                    return fr >= 5 and abs(dc) == 1
                return dr == 1 and dc == 0
        
        return False
    
    def _path_clear(self, fr, fc, tr, tc) -> bool:
        """检查路径上是否无障碍"""
        return self._count_pieces_between(fr, fc, tr, tc) == 0
    
    def _count_pieces_between(self, fr, fc, tr, tc) -> int:
        """统计路径上的棋子数"""
        count = 0
        if fr == tr:  # 水平
            step = 1 if tc > fc else -1
            for c in range(fc + step, tc, step):
                if self.board.board[fr, c] != 0:
                    count += 1
        else:  # 垂直
            step = 1 if tr > fr else -1
            for r in range(fr + step, tr, step):
                if self.board.board[r, fc] != 0:
                    count += 1
        return count
    
    def _execute_move(self, fr, fc, tr, tc, piece, captured):
        """执行移动"""
        self.board.board[fr, fc] = 0
        self.board.board[tr, tc] = piece
        self.board.current_player *= -1
        self.board.move_count += 1
        
        self.valid_move_count += 1
        self.consecutive_invalid = 0  # 重置连续违规
    
    def _record_violation(self, result: MoveResult):
        """记录违规"""
        self.invalid_move_count += 1
        self.consecutive_invalid += 1
        self.violation_history.append(result)
    
    def _is_checking(self) -> bool:
        """检查是否正在将军"""
        # 简化：暂不实现
        return False
    
    def _check_game_end(self):
        """检查游戏是否结束"""
        # 连续违规太多
        if self.consecutive_invalid >= self.max_invalid_moves:
            self.done = True
            self.winner = -self.board.current_player  # 当前方输
            return
        
        # 有效步数用完
        if self.valid_move_count >= self.max_moves:
            self.done = True
            self.winner = 0  # 和棋
            return
        
        # 将帅被吃
        red_king = self.board.find_king(is_red=True)
        black_king = self.board.find_king(is_red=False)
        
        if red_king is None:
            self.done = True
            self.winner = -1  # 黑胜
        elif black_king is None:
            self.done = True
            self.winner = 1   # 红胜
    
    def render(self):
        """渲染"""
        self.board.render()
        print(f"Valid moves: {self.valid_move_count} | "
              f"Invalid moves: {self.invalid_move_count} | "
              f"Consecutive invalid: {self.consecutive_invalid}")
    
    def get_violation_stats(self) -> Dict:
        """获取违规统计"""
        stats = {r.name: 0 for r in MoveResult if r != MoveResult.VALID}
        for v in self.violation_history:
            stats[v.name] += 1
        return stats


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Rule-Free Environment Test\n")
    
    env = RuleFreeEnv()
    state = env.reset()
    
    print(f"State shape: {state.shape}")  # (16, 10, 9)
    env.render()
    
    # 测试一些移动
    print("\n--- Testing moves ---\n")
    
    test_moves = [
        ((9, 1), (7, 2)),  # 马二进三（合法）
        ((5, 5), (4, 5)),  # 空位置移动（违规）
        ((0, 0), (1, 0)),  # 移动黑方棋子（违规）
        ((9, 4), (9, 3)),  # 帅平四（合法）
        ((9, 4), (7, 4)),  # 帅进二（违规-走法不对）
        ((7, 1), (7, 4)),  # 炮二平五（合法）
    ]
    
    for from_pos, to_pos in test_moves:
        state, reward, done, info = env.step(from_pos, to_pos)
        print(f"Move {from_pos} -> {to_pos}:")
        print(f"  Signal: {info['signal_name']}, Reward: {reward:.2f}")
        print(f"  Message: {info['message']}")
        print()
    
    env.render()
    print("\nViolation stats:", env.get_violation_stats())
