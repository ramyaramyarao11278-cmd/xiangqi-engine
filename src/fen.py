"""
FEN 格式解析与导出
FEN (Forsyth-Edwards Notation) 用于描述棋局局面
"""
from typing import Optional
from .board import Board, Piece


# FEN 字符到棋子的映射
FEN_TO_PIECE = {
    'K': Piece.R_KING, 'A': Piece.R_ADVISOR, 'B': Piece.R_BISHOP,
    'N': Piece.R_KNIGHT, 'R': Piece.R_ROOK, 'C': Piece.R_CANNON, 'P': Piece.R_PAWN,
    'k': Piece.B_KING, 'a': Piece.B_ADVISOR, 'b': Piece.B_BISHOP,
    'n': Piece.B_KNIGHT, 'r': Piece.B_ROOK, 'c': Piece.B_CANNON, 'p': Piece.B_PAWN,
}

# 棋子到 FEN 字符的映射
PIECE_TO_FEN = {v: k for k, v in FEN_TO_PIECE.items()}
PIECE_TO_FEN[Piece.EMPTY] = '.'


def parse_fen(fen: str, board: Board) -> bool:
    """
    解析 FEN 字符串并设置棋盘
    
    FEN 格式示例（初始局面）：
    rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w
    
    Args:
        fen: FEN 字符串
        board: 要设置的棋盘对象
        
    Returns:
        是否解析成功
    """
    try:
        parts = fen.strip().split(' ')
        board_fen = parts[0]
        
        # 解析棋盘部分
        rows = board_fen.split('/')
        if len(rows) != 10:
            return False
        
        # 清空棋盘
        board.board.fill(0)
        
        for row_idx, row_str in enumerate(rows):
            col_idx = 0
            for char in row_str:
                if char.isdigit():
                    col_idx += int(char)
                elif char in FEN_TO_PIECE:
                    board.board[row_idx, col_idx] = FEN_TO_PIECE[char]
                    col_idx += 1
                else:
                    return False
            
            if col_idx != 9:
                return False
        
        # 解析当前玩家
        if len(parts) > 1:
            board.current_player = 1 if parts[1].lower() == 'w' else -1
        
        return True
        
    except Exception:
        return False


def to_fen(board: Board) -> str:
    """
    将棋盘转换为 FEN 字符串
    
    Args:
        board: 棋盘对象
        
    Returns:
        FEN 字符串
    """
    rows = []
    
    for row_idx in range(10):
        row_str = ""
        empty_count = 0
        
        for col_idx in range(9):
            piece = board.board[row_idx, col_idx]
            
            if piece == 0:
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += PIECE_TO_FEN.get(piece, '?')
        
        if empty_count > 0:
            row_str += str(empty_count)
        
        rows.append(row_str)
    
    board_fen = '/'.join(rows)
    player = 'w' if board.current_player == 1 else 'b'
    
    return f"{board_fen} {player}"


# 别名
board_to_fen = to_fen


# 常用开局 FEN
STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("FEN Parser Test\n")
    
    board = Board()
    
    # 测试导出初始局面
    fen = to_fen(board)
    print(f"Initial FEN: {fen}")
    print(f"Expected:    {STARTING_FEN}")
    print(f"Match: {fen == STARTING_FEN}")
    
    # 测试解析
    print("\n--- Testing FEN parsing ---")
    test_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
    
    board2 = Board()
    board2.board.fill(0)  # 清空
    
    if parse_fen(test_fen, board2):
        print("Parse successful!")
        board2.render()
    else:
        print("Parse failed!")
    
    # 测试中局局面
    print("\n--- Testing midgame position ---")
    midgame_fen = "r1bakab1r/9/1cn4c1/p1p1p3p/6p2/2P6/P3P1P1P/1C4N1C/9/RNBAKAB1R b"
    
    board3 = Board()
    if parse_fen(midgame_fen, board3):
        print(f"Loaded: {midgame_fen}")
        board3.render()
    else:
        print("Parse failed!")
