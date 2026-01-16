"""
中国象棋评估函数
用于搜索引擎的局面评估
"""
from .board import Board, Piece


# 子力价值（相对兵=100）
PIECE_VALUES = {
    Piece.R_KING: 10000, Piece.B_KING: 10000,
    Piece.R_ROOK: 900, Piece.B_ROOK: 900,
    Piece.R_CANNON: 450, Piece.B_CANNON: 450,
    Piece.R_KNIGHT: 400, Piece.B_KNIGHT: 400,
    Piece.R_BISHOP: 200, Piece.B_BISHOP: 200,
    Piece.R_ADVISOR: 200, Piece.B_ADVISOR: 200,
    Piece.R_PAWN: 100, Piece.B_PAWN: 100,
}

# 位置加成表 (Piece-Square Tables)
# 红方视角，0行是黑方底线，9行是红方底线
# 值越大表示该位置越好

# 车的位置加成（中心和七路强）
ROOK_PST = [
    [0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0],
    [5,  5,  5,  5,  5,  5,  5,  5,  5],
    [0, 10,  5,  5,  5,  5,  5, 10,  0],
    [0,  0,  5, 10, 10, 10,  5,  0,  0],
    [0,  0,  5,  5,  5,  5,  5,  0,  0],
]

# 马的位置加成（中心强，边角弱）
KNIGHT_PST = [
    [ 0, -5,  0,  0,  0,  0,  0, -5,  0],
    [ 0,  0,  5,  5,  5,  5,  5,  0,  0],
    [ 0,  5, 10, 10, 10, 10, 10,  5,  0],
    [ 0,  5, 10, 15, 15, 15, 10,  5,  0],
    [ 0,  5, 10, 15, 15, 15, 10,  5,  0],
    [ 0,  5, 10, 15, 15, 15, 10,  5,  0],
    [ 0,  5, 10, 10, 10, 10, 10,  5,  0],
    [ 0,  0,  5,  5,  5,  5,  5,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0, -5,  0,  0,  0,  0,  0, -5,  0],
]

# 炮的位置加成
CANNON_PST = [
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 5,  5,  5,  5, 10,  5,  5,  5,  5],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  5,  5,  5,  5,  5,  5,  5,  0],
    [10, 15, 15, 15, 15, 15, 15, 15, 10],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
]

# 兵的位置加成（过河后价值增加）
PAWN_PST = [
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [20, 30, 40, 50, 60, 50, 40, 30, 20],  # 过河线
    [30, 40, 50, 60, 70, 60, 50, 40, 30],
    [40, 50, 60, 70, 80, 70, 60, 50, 40],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],  # 红兵起始位置
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
]


def evaluate(board: Board) -> int:
    """
    评估局面（红方视角）
    
    正值对红方有利，负值对黑方有利
    
    Args:
        board: 棋盘对象
        
    Returns:
        评估分数（厘兵）
    """
    score = 0
    
    # 遍历棋盘
    for row in range(10):
        for col in range(9):
            piece = board.board[row, col]
            if piece == 0:
                continue
            
            # 基础子力价值
            value = PIECE_VALUES.get(piece, 0)
            
            # 位置加成
            pst_bonus = get_pst_bonus(piece, row, col)
            
            # 红方加分，黑方减分
            if board.is_red_piece(piece):
                score += value + pst_bonus
            else:
                score -= value + pst_bonus
    
    # 将帅安全评估
    score += king_safety(board)
    
    # 机动性评估（简化：不计算）
    
    return score


def get_pst_bonus(piece: int, row: int, col: int) -> int:
    """获取位置加成"""
    piece_type = piece if piece <= 7 else piece - 7
    
    # 黑方需要翻转行号
    if piece > 7:
        row = 9 - row
    
    if piece_type == 5:  # 车
        return ROOK_PST[row][col]
    elif piece_type == 4:  # 马
        return KNIGHT_PST[row][col]
    elif piece_type == 6:  # 炮
        return CANNON_PST[row][col]
    elif piece_type == 7:  # 兵
        return PAWN_PST[row][col]
    
    return 0


def king_safety(board: Board) -> int:
    """
    评估将帅安全性
    
    考虑因素：
    - 是否被将军
    - 九宫周围的防守
    """
    score = 0
    
    # 红方帅安全
    if board.is_in_check(is_red=True):
        score -= 50  # 被将军惩罚
    
    # 黑方将安全
    if board.is_in_check(is_red=False):
        score += 50  # 对方被将军加分
    
    return score


# 评估分数常量
MATE_SCORE = 100000  # 将杀分数
DRAW_SCORE = 0


def is_mate(board: Board) -> int:
    """
    检查是否将死
    
    Returns:
        1 = 红方胜, -1 = 黑方胜, 0 = 未结束
    """
    red_king = board.find_king(True)
    black_king = board.find_king(False)
    
    if red_king is None:
        return -1  # 红方被将死
    if black_king is None:
        return 1  # 黑方被将死
    
    return 0


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Evaluation Function Test\n")
    
    board = Board()
    board.render()
    
    score = evaluate(board)
    print(f"\nInitial position score: {score}")
    print("(Should be ~0, slight advantage to first mover)")
