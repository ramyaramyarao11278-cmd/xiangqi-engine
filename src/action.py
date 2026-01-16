"""
统一动作空间编码
全项目唯一的编码/解码来源

动作空间：8100 = 90(起点) × 90(终点)
编码：action = from_idx * 90 + to_idx
其中：from_idx = from_row * 9 + from_col
      to_idx = to_row * 9 + to_col
"""
import numpy as np
from typing import Tuple, List
from .move import Move, MoveGenerator
from .board import Board


ACTION_SIZE = 8100  # 统一动作空间大小


def encode_move(move: Move) -> int:
    """将着法编码为动作索引"""
    from_idx = move.from_row * 9 + move.from_col
    to_idx = move.to_row * 9 + move.to_col
    return from_idx * 90 + to_idx


def encode_coords(from_row: int, from_col: int, to_row: int, to_col: int) -> int:
    """将坐标编码为动作索引"""
    from_idx = from_row * 9 + from_col
    to_idx = to_row * 9 + to_col
    return from_idx * 90 + to_idx


def decode_action(action: int) -> Tuple[int, int, int, int]:
    """将动作索引解码为坐标 (from_row, from_col, to_row, to_col)"""
    from_idx = action // 90
    to_idx = action % 90
    return (from_idx // 9, from_idx % 9, to_idx // 9, to_idx % 9)


def decode_to_move(action: int) -> Move:
    """将动作索引解码为 Move 对象"""
    from_row, from_col, to_row, to_col = decode_action(action)
    return Move(from_row, from_col, to_row, to_col, 0)


def get_legal_mask(board: Board) -> np.ndarray:
    """
    获取当前局面的合法着法掩码
    
    Returns:
        mask: shape (8100,), 合法着法位置为 1.0，其余为 0.0
    """
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    
    gen = MoveGenerator(board)
    is_red = board.current_player == 1
    moves = gen.generate_moves(is_red)
    
    for move in moves:
        # 检查不自将
        board.make_move(move)
        if not board.is_in_check(is_red):
            action = encode_move(move)
            mask[action] = 1.0
        board.unmake_move(move)
    
    return mask


def get_legal_actions(board: Board) -> List[int]:
    """获取当前局面的所有合法动作索引"""
    actions = []
    
    gen = MoveGenerator(board)
    is_red = board.current_player == 1
    moves = gen.generate_moves(is_red)
    
    for move in moves:
        board.make_move(move)
        if not board.is_in_check(is_red):
            actions.append(encode_move(move))
        board.unmake_move(move)
    
    return actions


# ========== 测试 ==========
if __name__ == "__main__":
    print("Action Space Test")
    print(f"ACTION_SIZE = {ACTION_SIZE}")
    
    # 测试编码/解码
    move = Move(6, 4, 5, 4, 0)  # 兵五进一
    action = encode_move(move)
    decoded = decode_action(action)
    print(f"Move: {move}")
    print(f"Encoded: {action}")
    print(f"Decoded: {decoded}")
    
    # 测试合法掩码
    board = Board()
    mask = get_legal_mask(board)
    legal_count = int(mask.sum())
    print(f"Legal moves in starting position: {legal_count}")
