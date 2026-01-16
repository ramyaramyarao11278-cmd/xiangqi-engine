"""
着法表示与生成器
实现所有棋子的合法移动规则
"""
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# 从 board 模块导入（运行时）
# from .board import Board, Piece


@dataclass
class Move:
    """着法表示"""
    from_row: int
    from_col: int
    to_row: int
    to_col: int
    piece: int = 0           # 移动的棋子
    captured: int = 0        # 被吃的棋子（0 表示无）
    
    def to_uci(self) -> str:
        """转换为 UCI 格式（如 'a0a1'）"""
        from_col_char = chr(ord('a') + self.from_col)
        to_col_char = chr(ord('a') + self.to_col)
        return f"{from_col_char}{self.from_row}{to_col_char}{self.to_row}"
    
    @classmethod
    def from_uci(cls, uci: str) -> 'Move':
        """从 UCI 格式解析"""
        from_col = ord(uci[0]) - ord('a')
        from_row = int(uci[1])
        to_col = ord(uci[2]) - ord('a')
        to_row = int(uci[3])
        return cls(from_row, from_col, to_row, to_col)
    
    def __str__(self):
        return f"({self.from_row},{self.from_col})->({self.to_row},{self.to_col})"


class MoveGenerator:
    """着法生成器"""
    
    # 棋子类型常量
    EMPTY = 0
    R_KING, R_ADVISOR, R_BISHOP, R_KNIGHT, R_ROOK, R_CANNON, R_PAWN = 1, 2, 3, 4, 5, 6, 7
    B_KING, B_ADVISOR, B_BISHOP, B_KNIGHT, B_ROOK, B_CANNON, B_PAWN = 8, 9, 10, 11, 12, 13, 14
    
    # 马的移动：(腿位置偏移, 目标位置偏移)
    KNIGHT_MOVES = [
        ((-1, 0), (-2, -1)), ((-1, 0), (-2, 1)),   # 上
        ((1, 0), (2, -1)), ((1, 0), (2, 1)),       # 下
        ((0, -1), (-1, -2)), ((0, -1), (1, -2)),   # 左
        ((0, 1), (-1, 2)), ((0, 1), (1, 2)),       # 右
    ]
    
    # 象的移动：(眼位置偏移, 目标位置偏移)
    BISHOP_MOVES = [
        ((-1, -1), (-2, -2)), ((-1, 1), (-2, 2)),
        ((1, -1), (2, -2)), ((1, 1), (2, 2)),
    ]
    
    # 士的移动偏移
    ADVISOR_MOVES = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # 将帅的移动偏移
    KING_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def __init__(self, board):
        self.board = board
    
    def is_red(self, piece: int) -> bool:
        return 1 <= piece <= 7
    
    def is_black(self, piece: int) -> bool:
        return 8 <= piece <= 14
    
    def is_enemy(self, piece: int, is_red_player: bool) -> bool:
        if is_red_player:
            return self.is_black(piece)
        return self.is_red(piece)
    
    def in_board(self, row: int, col: int) -> bool:
        return 0 <= row < 10 and 0 <= col < 9
    
    def in_palace(self, row: int, col: int, is_red: bool) -> bool:
        """检查是否在九宫格内"""
        if is_red:
            return 7 <= row <= 9 and 3 <= col <= 5
        else:
            return 0 <= row <= 2 and 3 <= col <= 5
    
    def in_own_half(self, row: int, is_red: bool) -> bool:
        """检查是否在己方半场"""
        if is_red:
            return row >= 5
        else:
            return row <= 4
    
    def generate_moves(self, is_red_player: bool) -> List[Move]:
        """生成所有合法着法"""
        moves = []
        board = self.board.board
        
        for row in range(10):
            for col in range(9):
                piece = board[row, col]
                if piece == 0:
                    continue
                
                # 检查是否为当前方棋子
                if is_red_player and not self.is_red(piece):
                    continue
                if not is_red_player and not self.is_black(piece):
                    continue
                
                # 根据棋子类型生成着法
                piece_type = piece if piece <= 7 else piece - 7
                
                if piece_type == 1:  # 将/帅
                    moves.extend(self._gen_king_moves(row, col, is_red_player))
                elif piece_type == 2:  # 士/仕
                    moves.extend(self._gen_advisor_moves(row, col, is_red_player))
                elif piece_type == 3:  # 象/相
                    moves.extend(self._gen_bishop_moves(row, col, is_red_player))
                elif piece_type == 4:  # 马
                    moves.extend(self._gen_knight_moves(row, col, is_red_player))
                elif piece_type == 5:  # 车
                    moves.extend(self._gen_rook_moves(row, col, is_red_player))
                elif piece_type == 6:  # 炮
                    moves.extend(self._gen_cannon_moves(row, col, is_red_player))
                elif piece_type == 7:  # 兵/卒
                    moves.extend(self._gen_pawn_moves(row, col, is_red_player))
        
        return moves
    
    def _gen_king_moves(self, row: int, col: int, is_red: bool) -> List[Move]:
        """生成将/帅着法"""
        moves = []
        board = self.board.board
        piece = board[row, col]
        
        for dr, dc in self.KING_MOVES:
            nr, nc = row + dr, col + dc
            if self.in_palace(nr, nc, is_red):
                target = board[nr, nc]
                if target == 0 or self.is_enemy(target, is_red):
                    moves.append(Move(row, col, nr, nc, piece, target))
        
        # 将帅对面（飞将）
        enemy_king = self.B_KING if is_red else self.R_KING
        direction = -1 if is_red else 1
        nr = row + direction
        while 0 <= nr < 10:
            target = board[nr, col]
            if target == enemy_king:
                moves.append(Move(row, col, nr, col, piece, target))
                break
            elif target != 0:
                break
            nr += direction
        
        return moves
    
    def _gen_advisor_moves(self, row: int, col: int, is_red: bool) -> List[Move]:
        """生成士/仕着法"""
        moves = []
        board = self.board.board
        piece = board[row, col]
        
        for dr, dc in self.ADVISOR_MOVES:
            nr, nc = row + dr, col + dc
            if self.in_palace(nr, nc, is_red):
                target = board[nr, nc]
                if target == 0 or self.is_enemy(target, is_red):
                    moves.append(Move(row, col, nr, nc, piece, target))
        
        return moves
    
    def _gen_bishop_moves(self, row: int, col: int, is_red: bool) -> List[Move]:
        """生成象/相着法（田字，不能过河，塞象眼）"""
        moves = []
        board = self.board.board
        piece = board[row, col]
        
        for (er, ec), (dr, dc) in self.BISHOP_MOVES:
            eye_r, eye_c = row + er, col + ec
            nr, nc = row + dr, col + dc
            
            # 检查象眼是否被堵
            if not self.in_board(eye_r, eye_c) or board[eye_r, eye_c] != 0:
                continue
            
            # 检查目标位置
            if not self.in_board(nr, nc):
                continue
            if not self.in_own_half(nr, is_red):  # 不能过河
                continue
            
            target = board[nr, nc]
            if target == 0 or self.is_enemy(target, is_red):
                moves.append(Move(row, col, nr, nc, piece, target))
        
        return moves
    
    def _gen_knight_moves(self, row: int, col: int, is_red: bool) -> List[Move]:
        """生成马着法（日字，蹩马腿）"""
        moves = []
        board = self.board.board
        piece = board[row, col]
        
        for (lr, lc), (dr, dc) in self.KNIGHT_MOVES:
            leg_r, leg_c = row + lr, col + lc
            nr, nc = row + dr, col + dc
            
            # 检查马腿是否被堵
            if not self.in_board(leg_r, leg_c) or board[leg_r, leg_c] != 0:
                continue
            
            # 检查目标位置
            if not self.in_board(nr, nc):
                continue
            
            target = board[nr, nc]
            if target == 0 or self.is_enemy(target, is_red):
                moves.append(Move(row, col, nr, nc, piece, target))
        
        return moves
    
    def _gen_rook_moves(self, row: int, col: int, is_red: bool) -> List[Move]:
        """生成车着法（直线移动）"""
        moves = []
        board = self.board.board
        piece = board[row, col]
        
        # 四个方向
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            while self.in_board(nr, nc):
                target = board[nr, nc]
                if target == 0:
                    moves.append(Move(row, col, nr, nc, piece, 0))
                elif self.is_enemy(target, is_red):
                    moves.append(Move(row, col, nr, nc, piece, target))
                    break
                else:
                    break
                nr += dr
                nc += dc
        
        return moves
    
    def _gen_cannon_moves(self, row: int, col: int, is_red: bool) -> List[Move]:
        """生成炮着法（直线移动，隔子吃子）"""
        moves = []
        board = self.board.board
        piece = board[row, col]
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            jumped = False
            
            while self.in_board(nr, nc):
                target = board[nr, nc]
                
                if not jumped:
                    if target == 0:
                        moves.append(Move(row, col, nr, nc, piece, 0))
                    else:
                        jumped = True  # 找到炮架
                else:
                    if target != 0:
                        if self.is_enemy(target, is_red):
                            moves.append(Move(row, col, nr, nc, piece, target))
                        break
                
                nr += dr
                nc += dc
        
        return moves
    
    def _gen_pawn_moves(self, row: int, col: int, is_red: bool) -> List[Move]:
        """生成兵/卒着法"""
        moves = []
        board = self.board.board
        piece = board[row, col]
        
        # 前进方向
        forward = -1 if is_red else 1
        crossed = not self.in_own_half(row, is_red)  # 是否过河
        
        # 前进
        nr, nc = row + forward, col
        if self.in_board(nr, nc):
            target = board[nr, nc]
            if target == 0 or self.is_enemy(target, is_red):
                moves.append(Move(row, col, nr, nc, piece, target))
        
        # 过河后可以左右移动
        if crossed:
            for dc in [-1, 1]:
                nr, nc = row, col + dc
                if self.in_board(nr, nc):
                    target = board[nr, nc]
                    if target == 0 or self.is_enemy(target, is_red):
                        moves.append(Move(row, col, nr, nc, piece, target))
        
        return moves


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 导入棋盘
    from board import Board
    
    print("Move Generator Test\n")
    
    board = Board()
    board.render()
    
    gen = MoveGenerator(board)
    
    # 生成红方所有着法
    red_moves = gen.generate_moves(is_red_player=True)
    print(f"\nRed has {len(red_moves)} legal moves:")
    for m in red_moves[:10]:  # 只显示前 10 个
        print(f"  {m} (UCI: {m.to_uci()})")
    print(f"  ... and {len(red_moves) - 10} more")
    
    # 生成黑方所有着法
    black_moves = gen.generate_moves(is_red_player=False)
    print(f"\nBlack has {len(black_moves)} legal moves")
