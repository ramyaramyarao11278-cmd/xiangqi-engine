"""
ICCS 格式棋谱解析器
支持标准 ICCS PGN 格式（如世界象棋联合会棋谱）
"""
import re
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ICCSGame:
    """解析后的棋谱"""
    headers: Dict[str, str]   # 标签（Event, Result, FEN 等）
    moves: List[str]          # ICCS 着法列表 ["C3-C4", "C9-E7", ...]
    result: str               # "1-0" / "0-1" / "1/2-1/2"


# ICCS 着法正则（匹配 C3-C4 或 c3c4 格式）
_MOVE_RE = re.compile(r'\b([A-Ia-i])([0-9])[- ]?([A-Ia-i])([0-9])\b')


def parse_iccs_games(text: str) -> List[ICCSGame]:
    """
    解析多盘 ICCS 格式棋谱
    
    Args:
        text: 棋谱文本（可包含多盘）
        
    Returns:
        ICCSGame 列表
    """
    games: List[ICCSGame] = []
    headers: Dict[str, str] = {}
    moves: List[str] = []
    result: Optional[str] = None
    
    def flush():
        nonlocal headers, moves, result
        if headers and moves:
            games.append(ICCSGame(
                headers=headers.copy(),
                moves=moves.copy(),
                result=result or "1/2-1/2"
            ))
        headers.clear()
        moves.clear()
        result = None
    
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        
        # 解析标签：[Key "Value"]
        if line.startswith("[") and line.endswith("]"):
            # 新棋盘开始，刷新上一盘
            if line.startswith("[Event") and moves:
                flush()
            
            m = re.match(r'^\[(\w+)\s+"(.*)"\]$', line)
            if m:
                key, val = m.group(1), m.group(2)
                headers[key] = val
                if key == "Result":
                    result = val
            continue
        
        # 独立结果行
        if line in ("1-0", "0-1", "1/2-1/2"):
            result = line
            continue
        
        # 提取 ICCS 着法
        for fc, fr, tc, tr in _MOVE_RE.findall(line):
            # 统一大写
            move_str = f"{fc.upper()}{fr}-{tc.upper()}{tr}"
            moves.append(move_str)
    
    # 最后一盘
    flush()
    
    return games


def parse_file(filepath: str) -> List[ICCSGame]:
    """解析棋谱文件"""
    # 尝试不同编码
    for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
        try:
            with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
                return parse_iccs_games(f.read())
        except:
            continue
    return []


# ========== 测试 ==========
if __name__ == "__main__":
    sample = '''
[Game "Chinese Chess"]
[Event "1998年全国象棋个人赛"]
[Result "1-0"]
[FEN "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"]
[Format "ICCS"]
1. C3-C4 C9-E7
2. B2-D2 G6-G5
3. B0-C2 B9-C7
1-0
'''
    
    games = parse_iccs_games(sample)
    print(f"Parsed {len(games)} games")
    for g in games:
        print(f"  Result: {g.result}, Moves: {len(g.moves)}")
        print(f"  First 5 moves: {g.moves[:5]}")
