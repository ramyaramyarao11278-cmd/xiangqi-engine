"""
中国象棋棋盘和棋子定义
"""
from enum import IntEnum
from typing import Optional, Tuple, List
import numpy as np


class Piece(IntEnum):
    """棋子类型编码"""
    EMPTY = 0
    
    # 红方棋子 (正数)
    R_KING = 1      # 帅
    R_ADVISOR = 2   # 仕
    R_BISHOP = 3    # 相
    R_KNIGHT = 4    # 马
    R_ROOK = 5      # 车
    R_CANNON = 6    # 炮
    R_PAWN = 7      # 兵
    
    # 黑方棋子 (负数表示，但这里用 8-14)
    B_KING = 8      # 将
    B_ADVISOR = 9   # 士
    B_BISHOP = 10   # 象
    B_KNIGHT = 11   # 马
    B_ROOK = 12     # 车
    B_CANNON = 13   # 炮
    B_PAWN = 14     # 卒


# 棋子显示字符
PIECE_CHARS = {
    Piece.EMPTY: '．',
    Piece.R_KING: '帅', Piece.R_ADVISOR: '仕', Piece.R_BISHOP: '相',
    Piece.R_KNIGHT: '马', Piece.R_ROOK: '车', Piece.R_CANNON: '炮', Piece.R_PAWN: '兵',
    Piece.B_KING: '将', Piece.B_ADVISOR: '士', Piece.B_BISHOP: '象',
    Piece.B_KNIGHT: '馬', Piece.B_ROOK: '車', Piece.B_CANNON: '砲', Piece.B_PAWN: '卒',
}

# ASCII 版本（Windows 兼容）
PIECE_ASCII = {
    Piece.EMPTY: ' . ',
    Piece.R_KING: ' K ', Piece.R_ADVISOR: ' A ', Piece.R_BISHOP: ' B ',
    Piece.R_KNIGHT: ' N ', Piece.R_ROOK: ' R ', Piece.R_CANNON: ' C ', Piece.R_PAWN: ' P ',
    Piece.B_KING: ' k ', Piece.B_ADVISOR: ' a ', Piece.B_BISHOP: ' b ',
    Piece.B_KNIGHT: ' n ', Piece.B_ROOK: ' r ', Piece.B_CANNON: ' c ', Piece.B_PAWN: ' p ',
}


class Board:
    """
    中国象棋棋盘
    
    坐标系统：
    - 行 (row): 0-9，0 为黑方底线，9 为红方底线
    - 列 (col): 0-8，从左到右
    
    棋盘布局：
        0 1 2 3 4 5 6 7 8
      0 車馬象士将士象馬車  <- 黑方
      1 ．．．．．．．．．
      2 ．砲．．．．．砲．
      3 卒．卒．卒．卒．卒
      4 ．．．．．．．．．  <- 楚河
      5 ．．．．．．．．．  <- 汉界
      6 兵．兵．兵．兵．兵
      7 ．炮．．．．．炮．
      8 ．．．．．．．．．
      9 车马相仕帅仕相马车  <- 红方
    """
    
    ROWS = 10
    COLS = 9
    
    # 初始棋盘布局
    INIT_BOARD = [
        [12, 11, 10, 9, 8, 9, 10, 11, 12],  # 黑方底线
        [0,  0,  0,  0, 0, 0,  0,  0,  0],
        [0, 13,  0,  0, 0, 0,  0, 13,  0],  # 黑砲
        [14, 0, 14,  0, 14, 0, 14,  0, 14], # 黑卒
        [0,  0,  0,  0, 0, 0,  0,  0,  0],
        [0,  0,  0,  0, 0, 0,  0,  0,  0],
        [7,  0,  7,  0, 7, 0,  7,  0,  7],  # 红兵
        [0,  6,  0,  0, 0, 0,  0,  6,  0],  # 红炮
        [0,  0,  0,  0, 0, 0,  0,  0,  0],
        [5,  4,  3,  2, 1, 2,  3,  4,  5],  # 红方底线
    ]
    
    def __init__(self):
        """初始化棋盘"""
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = 1  # 1: 红方, -1: 黑方
        self.move_count = 0
        self.reset()
    
    def reset(self):
        """重置为初始局面"""
        self.board = np.array(self.INIT_BOARD, dtype=np.int8)
        self.current_player = 1  # 红方先行
        self.move_count = 0
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """获取当前状态（用于 RL）"""
        return self.board.copy()
    
    def get_piece(self, row: int, col: int) -> int:
        """获取指定位置的棋子"""
        if 0 <= row < self.ROWS and 0 <= col < self.COLS:
            return self.board[row, col]
        return -1
    
    def set_piece(self, row: int, col: int, piece: int):
        """设置指定位置的棋子"""
        if 0 <= row < self.ROWS and 0 <= col < self.COLS:
            self.board[row, col] = piece
    
    def is_red_piece(self, piece: int) -> bool:
        """判断是否为红方棋子"""
        return 1 <= piece <= 7
    
    def is_black_piece(self, piece: int) -> bool:
        """判断是否为黑方棋子"""
        return 8 <= piece <= 14
    
    def is_own_piece(self, piece: int) -> bool:
        """判断是否为当前方棋子"""
        if self.current_player == 1:
            return self.is_red_piece(piece)
        else:
            return self.is_black_piece(piece)
    
    def is_enemy_piece(self, piece: int) -> bool:
        """判断是否为对方棋子"""
        if self.current_player == 1:
            return self.is_black_piece(piece)
        else:
            return self.is_red_piece(piece)
    
    def find_king(self, is_red: bool) -> Optional[Tuple[int, int]]:
        """找到将/帅的位置"""
        target = Piece.R_KING if is_red else Piece.B_KING
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if self.board[row, col] == target:
                    return (row, col)
        return None
    
    def render(self, use_ascii=True):
        """打印棋盘"""
        chars = PIECE_ASCII if use_ascii else PIECE_CHARS
        
        print("    0   1   2   3   4   5   6   7   8")
        print("  +" + "---+" * 9)
        
        for row in range(self.ROWS):
            line = f"{row} |"
            for col in range(self.COLS):
                piece = self.board[row, col]
                line += chars.get(piece, ' ? ') + "|"
            print(line)
            
            if row == 4:  # 楚河汉界
                print("  |" + "===+===+===+===+===+===+===+===+===|")
            else:
                print("  +" + "---+" * 9)
        
        player = "Red" if self.current_player == 1 else "Black"
        print(f"\nCurrent Player: {player} | Move: {self.move_count}")
    
    def clone(self) -> 'Board':
        """创建棋盘副本"""
        new_board = Board()
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.move_count = self.move_count
        return new_board
    
    # ========== 搜索引擎核心方法 ==========
    
    def make_move(self, move) -> int:
        """
        执行着法（可逆）
        
        Args:
            move: Move 对象，需要有 from_row, from_col, to_row, to_col
            
        Returns:
            被吃掉的棋子（用于 unmake）
        """
        captured = self.board[move.to_row, move.to_col]
        piece = self.board[move.from_row, move.from_col]
        
        # 记录到 move 对象
        move.captured = captured
        move.piece = piece
        
        # 执行移动
        self.board[move.to_row, move.to_col] = piece
        self.board[move.from_row, move.from_col] = 0
        
        # 切换玩家
        self.current_player *= -1
        self.move_count += 1
        
        return captured
    
    def unmake_move(self, move):
        """
        撤销着法
        
        Args:
            move: Move 对象（必须是之前 make_move 执行过的）
        """
        # 恢复棋子位置
        self.board[move.from_row, move.from_col] = move.piece
        self.board[move.to_row, move.to_col] = move.captured
        
        # 恢复玩家
        self.current_player *= -1
        self.move_count -= 1
    
    def is_in_check(self, is_red: bool) -> bool:
        """
        检查指定方是否被将军
        
        使用更稳定的方式：检查敌方所有着法是否能吃到将/帅
        
        Args:
            is_red: True=检查红方是否被将，False=检查黑方
            
        Returns:
            True 如果被将军
        """
        king_pos = self.find_king(is_red)
        if king_pos is None:
            return True  # 将帅不存在，视为被将死
        
        king_row, king_col = king_pos
        
        # 1. 检查将帅对面（飞将）
        enemy_king_pos = self.find_king(not is_red)
        if enemy_king_pos and enemy_king_pos[1] == king_col:
            min_row = min(king_row, enemy_king_pos[0])
            max_row = max(king_row, enemy_king_pos[0])
            blocked = False
            for r in range(min_row + 1, max_row):
                if self.board[r, king_col] != 0:
                    blocked = True
                    break
            if not blocked:
                return True  # 将帅照面
        
        # 2. 检查是否被敌方棋子攻击
        # 使用逐一检测各类棋子的方式，更可靠
        return self._is_attacked_by_enemy(king_row, king_col, not is_red)
    
    def _is_attacked(self, row: int, col: int, by_red: bool) -> bool:
        """
        检查指定位置是否被指定方攻击
        
        Args:
            row, col: 目标位置
            by_red: True=检查红方是否攻击此位置
        """
        # 检查车的攻击
        rook_type = Piece.R_ROOK if by_red else Piece.B_ROOK
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            while 0 <= r < 10 and 0 <= c < 9:
                piece = self.board[r, c]
                if piece != 0:
                    if piece == rook_type:
                        return True
                    break
                r += dr
                c += dc
        
        # 检查炮的攻击
        cannon_type = Piece.R_CANNON if by_red else Piece.B_CANNON
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            jumped = False
            while 0 <= r < 10 and 0 <= c < 9:
                piece = self.board[r, c]
                if piece != 0:
                    if not jumped:
                        jumped = True
                    else:
                        if piece == cannon_type:
                            return True
                        break
                r += dr
                c += dc
        
        # 检查马的攻击
        knight_type = Piece.R_KNIGHT if by_red else Piece.B_KNIGHT
        knight_moves = [
            ((-1, 0), (-2, -1)), ((-1, 0), (-2, 1)),
            ((1, 0), (2, -1)), ((1, 0), (2, 1)),
            ((0, -1), (-1, -2)), ((0, -1), (1, -2)),
            ((0, 1), (-1, 2)), ((0, 1), (1, 2)),
        ]
        for (lr, lc), (dr, dc) in knight_moves:
            nr, nc = row + dr, col + dc
            leg_r, leg_c = row + lr, col + lc
            if 0 <= nr < 10 and 0 <= nc < 9:
                if self.board[nr, nc] == knight_type:
                    # 检查马腿（从攻击者视角）
                    attack_leg_r = nr + (-dr // 2 if abs(dr) == 2 else 0)
                    attack_leg_c = nc + (-dc // 2 if abs(dc) == 2 else 0)
                    if self.board[attack_leg_r, attack_leg_c] == 0:
                        return True
        
        # 检查兵/卒的攻击
        pawn_type = Piece.R_PAWN if by_red else Piece.B_PAWN
        if by_red:
            # 红兵向上攻击
            attack_positions = [(row + 1, col), (row, col - 1), (row, col + 1)]
        else:
            # 黑卒向下攻击
            attack_positions = [(row - 1, col), (row, col - 1), (row, col + 1)]
        
        for ar, ac in attack_positions:
            if 0 <= ar < 10 and 0 <= ac < 9:
                if self.board[ar, ac] == pawn_type:
                    # 验证兵是否能攻击此位置
                    if by_red:
                        # 红兵只能攻击自己上方和左右（过河后）
                        if ar == row + 1:  # 在自己下方
                            return True
                        elif ar == row and ar <= 4:  # 过河后横攻
                            return True
                    else:
                        if ar == row - 1:
                            return True
                        elif ar == row and ar >= 5:
                            return True
        
        return False
    
    def _is_attacked_by_enemy(self, row: int, col: int, enemy_is_red: bool) -> bool:
        """
        检查位置是否被敌方攻击（_is_attacked 的别名，参数更清晰）
        
        Args:
            row, col: 目标位置
            enemy_is_red: True=检查红方是否攻击，False=检查黑方
        """
        return self._is_attacked(row, col, enemy_is_red)
    
    def is_game_over(self) -> bool:
        """检查游戏是否结束"""
        red_king = self.find_king(True)
        black_king = self.find_king(False)
        return red_king is None or black_king is None


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Chinese Chess Board Test\n")
    
    board = Board()
    board.render(use_ascii=True)
    
    # 测试查找将帅
    red_king = board.find_king(is_red=True)
    black_king = board.find_king(is_red=False)
    print(f"\nRed King (Shuai): {red_king}")
    print(f"Black King (Jiang): {black_king}")
