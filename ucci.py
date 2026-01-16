"""
UCI/UCCI 协议支持
用于对接象棋 GUI（如 ChessOnline、象棋巫师等）

UCCI (Universal Chinese Chess Interface) 是 UCI 的中国象棋版本
"""
import sys
import time
import threading
from typing import Optional

from src.board import Board
from src.fen import parse_fen, board_to_fen
from src.move import Move
from src.search_v3 import SearchEngineV3


class UCCIEngine:
    """
    UCCI 协议引擎
    
    支持的命令：
    - ucci: 握手
    - isready: 准备检查
    - position fen <fen> [moves <move1> <move2> ...]
    - go [depth <d>] [time <ms>]
    - quit: 退出
    """
    
    ENGINE_NAME = "XiangqiEngine"
    ENGINE_AUTHOR = "RL Project"
    
    def __init__(
        self,
        net_path: Optional[str] = None,
        default_depth: int = 6,
        default_time_ms: int = 5000,
    ):
        self.board = Board()
        self.default_depth = default_depth
        self.default_time_ms = default_time_ms
        
        # 搜索引擎
        self.engine = SearchEngineV3(
            tt_size_mb=64,
            net_path=net_path,
            policy_weight=1000.0,
            use_cuda=True,
        )
        
        self.running = True
        self.searching = False
        self.search_thread: Optional[threading.Thread] = None
    
    def run(self):
        """主循环"""
        while self.running:
            try:
                line = input().strip()
                if line:
                    self._process_command(line)
            except EOFError:
                break
            except Exception as e:
                self._send(f"info string Error: {e}")
    
    def _send(self, message: str):
        """发送消息"""
        print(message, flush=True)
    
    def _process_command(self, line: str):
        """处理命令"""
        parts = line.split()
        if not parts:
            return
        
        cmd = parts[0].lower()
        
        if cmd == "ucci":
            self._cmd_ucci()
        elif cmd == "isready":
            self._cmd_isready()
        elif cmd == "position":
            self._cmd_position(parts[1:])
        elif cmd == "go":
            self._cmd_go(parts[1:])
        elif cmd == "stop":
            self._cmd_stop()
        elif cmd == "quit":
            self._cmd_quit()
        elif cmd == "d":
            self._cmd_display()
        else:
            self._send(f"info string Unknown command: {cmd}")
    
    def _cmd_ucci(self):
        """UCCI 握手"""
        self._send(f"id name {self.ENGINE_NAME}")
        self._send(f"id author {self.ENGINE_AUTHOR}")
        self._send("option name Hash type spin default 64 min 1 max 1024")
        self._send("option name Depth type spin default 6 min 1 max 20")
        self._send("ucciok")
    
    def _cmd_isready(self):
        """准备检查"""
        self._send("readyok")
    
    def _cmd_position(self, args):
        """设置局面"""
        if not args:
            return
        
        idx = 0
        
        if args[0] == "startpos":
            self.board.reset()
            idx = 1
        elif args[0] == "fen":
            # 收集 FEN 字符串
            fen_parts = []
            idx = 1
            while idx < len(args) and args[idx] != "moves":
                fen_parts.append(args[idx])
                idx += 1
            
            fen = " ".join(fen_parts)
            try:
                parse_fen(fen, self.board)
            except Exception as e:
                self._send(f"info string FEN parse error: {e}")
                return
        
        # 处理 moves
        if idx < len(args) and args[idx] == "moves":
            idx += 1
            while idx < len(args):
                move_str = args[idx]
                move = self._parse_move(move_str)
                if move:
                    self.board.make_move(move)
                idx += 1
    
    def _cmd_go(self, args):
        """开始搜索"""
        depth = self.default_depth
        time_ms = self.default_time_ms
        
        i = 0
        while i < len(args):
            if args[i] == "depth" and i + 1 < len(args):
                depth = int(args[i + 1])
                i += 2
            elif args[i] in ("movetime", "time") and i + 1 < len(args):
                time_ms = int(args[i + 1])
                i += 2
            else:
                i += 1
        
        # 在单独线程中搜索
        self.searching = True
        self.search_thread = threading.Thread(
            target=self._search_thread,
            args=(depth, time_ms)
        )
        self.search_thread.start()
    
    def _search_thread(self, depth: int, time_ms: int):
        """搜索线程"""
        try:
            best_move, score = self.engine.search(
                self.board,
                depth=depth,
                time_limit_ms=time_ms,
            )
            
            if best_move and self.searching:
                move_str = self._format_move(best_move)
                self._send(f"info depth {depth} score cp {score}")
                self._send(f"bestmove {move_str}")
            else:
                self._send("bestmove none")
        except Exception as e:
            self._send(f"info string Search error: {e}")
            self._send("bestmove none")
        finally:
            self.searching = False
    
    def _cmd_stop(self):
        """停止搜索"""
        self.searching = False
    
    def _cmd_quit(self):
        """退出"""
        self.running = False
        self.searching = False
    
    def _cmd_display(self):
        """显示棋盘（调试用）"""
        self._send("info string Current position:")
        fen = board_to_fen(self.board)
        self._send(f"info string FEN: {fen}")
    
    def _parse_move(self, move_str: str) -> Optional[Move]:
        """
        解析着法字符串
        格式: a0a1 (ICCS 格式)
        """
        if len(move_str) < 4:
            return None
        
        try:
            # ICCS: a0a1 表示 a0 -> a1
            from_col = ord(move_str[0].lower()) - ord('a')
            from_row = 9 - int(move_str[1])
            to_col = ord(move_str[2].lower()) - ord('a')
            to_row = 9 - int(move_str[3])
            
            return Move(from_row, from_col, to_row, to_col, 0)
        except:
            return None
    
    def _format_move(self, move: Move) -> str:
        """格式化着法为 ICCS"""
        from_col = chr(ord('a') + move.from_col)
        from_row = 9 - move.from_row
        to_col = chr(ord('a') + move.to_col)
        to_row = 9 - move.to_row
        
        return f"{from_col}{from_row}{to_col}{to_row}"


def main():
    """命令行启动"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Xiangqi UCCI Engine")
    parser.add_argument("--net", type=str, default=None, help="Network checkpoint path")
    parser.add_argument("--depth", type=int, default=6, help="Default search depth")
    parser.add_argument("--time", type=int, default=5000, help="Default time per move (ms)")
    
    args = parser.parse_args()
    
    engine = UCCIEngine(
        net_path=args.net,
        default_depth=args.depth,
        default_time_ms=args.time,
    )
    
    engine.run()


if __name__ == "__main__":
    main()
