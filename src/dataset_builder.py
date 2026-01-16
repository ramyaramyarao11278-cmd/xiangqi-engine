"""
数据集构建器
将 ICCS 棋谱转换为训练样本 (state, action, z)

核心功能：
1. ICCS 坐标转换
2. 合法性校验（使用 MoveGenerator + 自将过滤）
3. 终局胜负回填（当前执子方视角）
4. 左右镜像数据增强
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .board import Board
from .move import Move, MoveGenerator
from .fen import parse_fen, STARTING_FEN
from .action import ACTION_SIZE, encode_move, encode_coords
from .iccs_pgn import ICCSGame, parse_file, parse_iccs_games


# ICCS 列名映射
ICCS_FILES = "ABCDEFGHI"


@dataclass
class SupervisedSample:
    """监督学习样本"""
    state: np.ndarray      # (15, 10, 9)
    action: int            # 0..8099
    z: float               # 终局胜负（当前执子方视角）


def iccs_to_coords(iccs: str) -> Tuple[int, int, int, int]:
    """
    ICCS 着法转坐标
    
    ICCS 坐标系：
    - 列: A-I (左到右) -> col 0-8
    - 行: 0-9 (红方底线0，黑方底线9)
    
    我们的棋盘坐标系：
    - 行: 0-9 (黑方底线0，红方底线9)
    - 所以需要翻转: board_row = 9 - iccs_row
    
    Args:
        iccs: "C3-C4" 或 "c3c4" 格式
        
    Returns:
        (from_row, from_col, to_row, to_col)
    """
    iccs = iccs.upper().replace("-", "")
    
    fc = ICCS_FILES.index(iccs[0])  # from_col
    fr = 9 - int(iccs[1])           # from_row (翻转)
    tc = ICCS_FILES.index(iccs[2])  # to_col
    tr = 9 - int(iccs[3])           # to_row (翻转)
    
    return fr, fc, tr, tc


def iccs_to_move(iccs: str) -> Move:
    """ICCS 着法转 Move 对象"""
    fr, fc, tr, tc = iccs_to_coords(iccs)
    return Move(fr, fc, tr, tc, 0)


def encode_state(board: Board) -> np.ndarray:
    """
    棋盘编码为网络输入
    
    15 通道：
    - 0-13: 14种棋子的 one-hot
    - 14: 当前执子方（红方=1）
    """
    state = np.zeros((15, 10, 9), dtype=np.float32)
    
    for row in range(10):
        for col in range(9):
            piece = int(board.board[row, col])
            if piece > 0:
                state[piece - 1, row, col] = 1.0
    
    # 当前执子方通道
    if board.current_player == 1:
        state[14, :, :] = 1.0
    
    return state


def is_legal_move(board: Board, move: Move) -> Tuple[bool, str]:
    """
    检查着法是否合法
    
    1. 是否在伪合法着法列表中
    2. 是否导致自将
    
    Returns:
        (is_legal, reason) - reason 只在非法时有意义
    """
    gen = MoveGenerator(board)
    is_red = board.current_player == 1
    pseudo_moves = gen.generate_moves(is_red)
    
    # 检查是否在伪合法列表中
    found = False
    for m in pseudo_moves:
        if (m.from_row == move.from_row and m.from_col == move.from_col and
            m.to_row == move.to_row and m.to_col == move.to_col):
            found = True
            break
    
    if not found:
        return False, "pseudo_legal_mismatch"
    
    # 检查自将
    board.make_move(move)
    in_check = board.is_in_check(is_red)
    board.unmake_move(move)
    
    if in_check:
        return False, "self_check"
    
    return True, "ok"


def result_to_winner(result: str) -> int:
    """结果字符串转胜负（1=红胜, -1=黑胜, 0=和）"""
    if "1-0" in result:
        return 1
    elif "0-1" in result:
        return -1
    else:
        return 0


def build_samples_from_game(
    game: ICCSGame,
    augment_mirror: bool = True,
) -> Tuple[Optional[List[SupervisedSample]], str]:
    """
    从单盘棋谱构建训练样本
    
    Args:
        game: ICCSGame 对象
        augment_mirror: 是否添加左右镜像增强
        
    Returns:
        (样本列表, 失败原因) - 失败时样本列表为 None
    """
    board = Board()
    
    # 如果有 FEN，从 FEN 开始
    fen = game.headers.get("FEN")
    if fen:
        if not parse_fen(fen, board):
            return None, "fen_parse_error"
    
    winner = result_to_winner(game.result)
    
    trajectory = []  # [(state, action, player)]
    
    for iccs in game.moves:
        try:
            move = iccs_to_move(iccs)
        except:
            return None, "iccs_parse_error"
        
        # 合法性校验
        is_legal, reason = is_legal_move(board, move)
        if not is_legal:
            return None, reason
        
        # 记录样本
        state = encode_state(board)
        action = encode_move(move)
        player = board.current_player
        
        trajectory.append((state, action, player))
        
        # 执行着法
        board.make_move(move)
    
    # 回填 z（当前执子方视角）
    samples = []
    for state, action, player in trajectory:
        if winner == 0:
            z = 0.0
        elif winner == player:
            z = 1.0   # 当前方赢
        else:
            z = -1.0  # 当前方输
        
        samples.append(SupervisedSample(state=state, action=action, z=z))
    
    # 左右镜像增强
    if augment_mirror and samples:
        mirrored = []
        for s in samples:
            m_state = mirror_state(s.state)
            m_action = mirror_action(s.action)
            mirrored.append(SupervisedSample(state=m_state, action=m_action, z=s.z))
        samples.extend(mirrored)
    
    return samples, "ok"


def mirror_state(state: np.ndarray) -> np.ndarray:
    """左右镜像状态"""
    # 沿列维度翻转
    return np.flip(state, axis=2).copy()


def mirror_action(action: int) -> int:
    """左右镜像动作"""
    from_idx = action // 90
    to_idx = action % 90
    
    from_row, from_col = from_idx // 9, from_idx % 9
    to_row, to_col = to_idx // 9, to_idx % 9
    
    # 镜像列
    from_col = 8 - from_col
    to_col = 8 - to_col
    
    return encode_coords(from_row, from_col, to_row, to_col)


def build_dataset(
    pgn_dir: str,
    max_games: int = None,
    augment_mirror: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从目录构建完整数据集
    
    Args:
        pgn_dir: 棋谱目录
        max_games: 最大棋谱数
        augment_mirror: 是否镜像增强
        
    Returns:
        states: (N, 15, 10, 9)
        actions: (N,)
        zs: (N,)
    """
    pgn_path = Path(pgn_dir)
    
    all_samples = []
    games_processed = 0
    games_failed = 0
    fail_reasons = {}  # 统计失败原因
    
    # 查找所有 PGN 文件（包括 .pgns 扩展名）
    pgn_files = (list(pgn_path.glob("*.pgn")) + 
                 list(pgn_path.glob("*.PGN")) +
                 list(pgn_path.glob("*.pgns")))
    
    for pgn_file in pgn_files:
        print(f"Processing: {pgn_file.name}")
        games = parse_file(str(pgn_file))
        
        for game in games:
            if max_games and games_processed >= max_games:
                break
            
            samples, reason = build_samples_from_game(game, augment_mirror)
            
            if samples:
                all_samples.extend(samples)
                games_processed += 1
            else:
                games_failed += 1
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            
            if games_processed % 1000 == 0 and games_processed > 0:
                print(f"  Processed: {games_processed}, Failed: {games_failed}, "
                      f"Samples: {len(all_samples)}")
        
        if max_games and games_processed >= max_games:
            break
    
    print(f"\nDataset built:")
    print(f"  Games processed: {games_processed}")
    print(f"  Games failed: {games_failed} ({100*games_failed/(games_processed+games_failed+0.001):.1f}%)")
    if fail_reasons:
        print(f"  Fail reasons:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count}")
    print(f"  Total samples: {len(all_samples)}")
    
    if not all_samples:
        return np.array([]), np.array([]), np.array([])
    
    states = np.stack([s.state for s in all_samples])
    actions = np.array([s.action for s in all_samples], dtype=np.int32)
    zs = np.array([s.z for s in all_samples], dtype=np.float32)
    
    return states, actions, zs


# ========== 测试 ==========
if __name__ == "__main__":
    print("Dataset Builder Test\n")
    
    # 测试 ICCS 转换
    print("ICCS conversion test:")
    iccs = "C3-C4"
    coords = iccs_to_coords(iccs)
    print(f"  {iccs} -> {coords}")
    
    move = iccs_to_move(iccs)
    print(f"  Move: {move}")
    
    action = encode_move(move)
    print(f"  Action: {action}")
    
    # 测试镜像
    print("\nMirror test:")
    m_action = mirror_action(action)
    print(f"  Original action: {action}")
    print(f"  Mirrored action: {m_action}")
