"""
增强版评估函数 v2
更多特征，更精准的局面评估
"""
from .board import Board, Piece
from typing import Tuple, List


# ========== 子力价值 ==========
# 单位：厘兵（1兵 = 100）

PIECE_VALUES = {
    Piece.R_KING: 10000, Piece.B_KING: 10000,
    Piece.R_ROOK: 1000, Piece.B_ROOK: 1000,
    Piece.R_CANNON: 500, Piece.B_CANNON: 500,
    Piece.R_KNIGHT: 450, Piece.B_KNIGHT: 450,
    Piece.R_BISHOP: 200, Piece.B_BISHOP: 200,
    Piece.R_ADVISOR: 200, Piece.B_ADVISOR: 200,
    Piece.R_PAWN: 100, Piece.B_PAWN: 100,
}


# ========== 位置加成表 (PST) ==========
# 红方视角，行0=黑方底线，行9=红方底线

# 车：控制线力强，七路/三路价值高
ROOK_PST = [
    [10, 10, 10, 15, 15, 15, 10, 10, 10],
    [20, 25, 25, 30, 30, 30, 25, 25, 20],
    [10, 20, 20, 25, 25, 25, 20, 20, 10],
    [ 5, 10, 15, 20, 20, 20, 15, 10,  5],
    [ 0,  5, 10, 15, 15, 15, 10,  5,  0],
    [ 0,  5, 10, 15, 15, 15, 10,  5,  0],
    [ 0,  5,  5, 10, 15, 10,  5,  5,  0],
    [ 0,  0,  5, 10, 10, 10,  5,  0,  0],
    [ 0,  0,  0,  5,  5,  5,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
]

# 马：中心强，边角弱，蹩腿惩罚在搜索中体现
KNIGHT_PST = [
    [ 0, -5,  0,  5,  5,  5,  0, -5,  0],
    [ 0,  5, 15, 20, 20, 20, 15,  5,  0],
    [ 5, 15, 25, 30, 30, 30, 25, 15,  5],
    [ 5, 15, 25, 35, 35, 35, 25, 15,  5],
    [ 5, 15, 25, 35, 35, 35, 25, 15,  5],
    [ 5, 10, 20, 30, 30, 30, 20, 10,  5],
    [ 0,  5, 15, 20, 20, 20, 15,  5,  0],
    [ 0,  0, 10, 15, 15, 15, 10,  0,  0],
    [ 0,  0,  0,  5,  5,  5,  0,  0,  0],
    [ 0, -10, 0,  0,  0,  0,  0,-10,  0],
]

# 炮：开局中路强，残局炮价值下降
CANNON_PST = [
    [ 0,  0,  5, 10, 20, 10,  5,  0,  0],
    [ 0,  0,  5, 15, 20, 15,  5,  0,  0],
    [ 0,  5, 10, 20, 25, 20, 10,  5,  0],
    [ 0,  5, 10, 15, 20, 15, 10,  5,  0],
    [ 0,  5,  5, 10, 15, 10,  5,  5,  0],
    [ 0,  5,  5, 10, 15, 10,  5,  5,  0],
    [ 0,  5, 10, 15, 20, 15, 10,  5,  0],
    [10, 15, 20, 25, 30, 25, 20, 15, 10],
    [ 0,  0,  0,  0,  5,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
]

# 兵：过河前价值低，过河后大幅增加
PAWN_PST = [
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 50, 60, 70, 80, 90, 80, 70, 60, 50],  # 河界
    [ 60, 70, 80, 90,100, 90, 80, 70, 60],
    [ 70, 80, 90,100,110,100, 90, 80, 70],
    [  0,  5,  0,  5, 10,  5,  0,  5,  0],  # 起始位置
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
]

# 士：靠近将帅更好
ADVISOR_PST = [
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0, 10, 15, 10,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0, 10, 15, 10,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  5, 10,  5,  0,  0,  0],
]

# 相：中相强
BISHOP_PST = [
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0, 10,  0, 15,  0, 10,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  5,  0,  0,  0,  0,  0,  0,  0,  5],
    [  5,  0,  0,  0,  0,  0,  0,  0,  5],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0, 10,  0, 15,  0, 10,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0],
]


def get_pst_value(piece: int, row: int, col: int) -> int:
    """获取位置加成值"""
    piece_type = piece if piece <= 7 else piece - 7
    
    # 黑方翻转
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
    elif piece_type == 2:  # 士
        return ADVISOR_PST[row][col]
    elif piece_type == 3:  # 相
        return BISHOP_PST[row][col]
    
    return 0


# ========== 评估函数 ==========

MATE_SCORE = 100000
DRAW_SCORE = 0


def evaluate_v2(board: Board) -> int:
    """
    增强版评估函数（红方视角）
    
    评估因素：
    1. 子力价值
    2. 位置加成 (PST)
    3. 将帅安全
    4. 机动性（着法数量）
    5. 车的控制力
    6. 兵形结构
    """
    score = 0
    
    # 收集棋子信息
    red_pieces = []
    black_pieces = []
    
    for row in range(10):
        for col in range(9):
            piece = board.board[row, col]
            if piece == 0:
                continue
            
            if board.is_red_piece(piece):
                red_pieces.append((piece, row, col))
            else:
                black_pieces.append((piece, row, col))
    
    # 1. 子力价值 + PST
    for piece, row, col in red_pieces:
        score += PIECE_VALUES[piece]
        score += get_pst_value(piece, row, col)
    
    for piece, row, col in black_pieces:
        score -= PIECE_VALUES[piece]
        score -= get_pst_value(piece, row, col)
    
    # 2. 将帅安全
    score += evaluate_king_safety(board)
    
    # 3. 车的控制力
    score += evaluate_rook_power(board, red_pieces, black_pieces)
    
    # 4. 兵形结构
    score += evaluate_pawn_structure(board, red_pieces, black_pieces)
    
    # 5. 残局调整
    score += endgame_adjustment(board, red_pieces, black_pieces, score)
    
    return score


def evaluate_king_safety(board: Board) -> int:
    """评估将帅安全"""
    score = 0
    
    # 红方帅
    red_king_pos = board.find_king(True)
    if red_king_pos:
        # 被将军惩罚
        if board.is_in_check(True):
            score -= 80
        
        # 帅的位置：中心比角落好
        row, col = red_king_pos
        if col == 4:  # 中路
            score += 10
    
    # 黑方将
    black_king_pos = board.find_king(False)
    if black_king_pos:
        if board.is_in_check(False):
            score += 80
        
        row, col = black_king_pos
        if col == 4:
            score -= 10
    
    return score


def evaluate_rook_power(board: Board, red_pieces: List, black_pieces: List) -> int:
    """评估车的控制力"""
    score = 0
    
    for piece, row, col in red_pieces:
        if piece == Piece.R_ROOK:
            # 车在第七路（河界）
            if row <= 4:
                score += 20
            # 车在对方半场
            if row <= 2:
                score += 30
            # 霸王车（在对方底线）
            if row == 0:
                score += 40
    
    for piece, row, col in black_pieces:
        if piece == Piece.B_ROOK:
            if row >= 5:
                score -= 20
            if row >= 7:
                score -= 30
            if row == 9:
                score -= 40
    
    return score


def evaluate_pawn_structure(board: Board, red_pieces: List, black_pieces: List) -> int:
    """评估兵形结构"""
    score = 0
    
    red_pawns = [(r, c) for p, r, c in red_pieces if p == Piece.R_PAWN]
    black_pawns = [(r, c) for p, r, c in black_pieces if p == Piece.B_PAWN]
    
    # 过河兵数量
    red_crossed = sum(1 for r, c in red_pawns if r <= 4)
    black_crossed = sum(1 for r, c in black_pawns if r >= 5)
    score += red_crossed * 15
    score -= black_crossed * 15
    
    # 连兵（同一列相邻的兵）
    red_cols = [c for r, c in red_pawns]
    black_cols = [c for r, c in black_pawns]
    
    # 底兵（到达对方底线）
    red_bottom = sum(1 for r, c in red_pawns if r == 0)
    black_bottom = sum(1 for r, c in black_pawns if r == 9)
    score += red_bottom * 50
    score -= black_bottom * 50
    
    return score


def endgame_adjustment(board: Board, red_pieces: List, black_pieces: List, current_score: int) -> int:
    """残局调整"""
    adjustment = 0
    
    # 计算子力总值（不含将帅）
    red_material = sum(PIECE_VALUES[p] for p, r, c in red_pieces if p != Piece.R_KING)
    black_material = sum(PIECE_VALUES[p] for p, r, c in black_pieces if p != Piece.B_KING)
    total_material = red_material + black_material
    
    # 残局门槛（双方子力和 < 3000）
    if total_material < 3000:
        # 残局：优势方的将帅要靠近对方
        red_king = board.find_king(True)
        black_king = board.find_king(False)
        
        if red_king and black_king:
            king_distance = abs(red_king[0] - black_king[0]) + abs(red_king[1] - black_king[1])
            
            if current_score > 200:  # 红方优势
                adjustment += (14 - king_distance) * 5  # 鼓励靠近
            elif current_score < -200:  # 黑方优势
                adjustment -= (14 - king_distance) * 5
    
    # 炮在残局价值下降
    if total_material < 2000:
        red_cannons = sum(1 for p, r, c in red_pieces if p == Piece.R_CANNON)
        black_cannons = sum(1 for p, r, c in black_pieces if p == Piece.B_CANNON)
        adjustment -= red_cannons * 50  # 红炮贬值
        adjustment += black_cannons * 50  # 黑炮贬值
    
    return adjustment


# 兼容接口
def evaluate(board: Board) -> int:
    """评估函数（兼容旧接口）"""
    return evaluate_v2(board)


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Enhanced Evaluation v2 Test\n")
    
    board = Board()
    board.render()
    
    score = evaluate_v2(board)
    print(f"\nInitial position score: {score}")
    
    # 测试：红方动一步
    from .move import Move
    move = Move(6, 4, 5, 4, Piece.R_PAWN)  # 兵五进一
    board.make_move(move)
    
    score2 = evaluate_v2(board)
    print(f"After e6-e5: {score2} (diff: {score2 - score:+d})")
