"""
中国象棋 AI API 服务器
提供 HTTP API 让前端 GUI 调用搜索引擎
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.board import Board
from src.move import Move, MoveGenerator
from src.search_v3 import SearchEngineV3
from src.fen import parse_fen, to_fen


class ChessAPIHandler(SimpleHTTPRequestHandler):
    """API 请求处理器"""
    
    engine = None
    board = None
    
    @classmethod
    def init_engine(cls, net_path=None):
        cls.engine = SearchEngineV3(
            tt_size_mb=32,
            net_path=net_path,
            policy_weight=1000.0,
            use_cuda=True,
        )
        cls.board = Board()
    
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        
        if path == '/api/best_move':
            self.handle_best_move(parsed)
        elif path == '/api/reset':
            self.handle_reset()
        elif path == '/api/make_move':
            self.handle_make_move(parsed)
        elif path == '/api/status':
            self.handle_status()
        elif path.endswith('.html') or path == '/':
            self.serve_html()
        else:
            super().do_GET()
    
    def serve_html(self):
        """提供 HTML 文件"""
        try:
            html_path = os.path.join(os.path.dirname(__file__), 'chess_gui_connected.html')
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except Exception as e:
            self.send_error(500, str(e))
    
    def handle_best_move(self, parsed):
        """处理获取最佳着法请求"""
        try:
            params = urllib.parse.parse_qs(parsed.query)
            depth = int(params.get('depth', [4])[0])
            time_ms = int(params.get('time', [3000])[0])
            
            # 从前端传来的棋盘状态更新
            if 'board' in params:
                board_json = params['board'][0]
                board_data = json.loads(board_json)
                self.board.board[:] = board_data
            
            if 'player' in params:
                self.board.current_player = int(params['player'][0])
            
            # 搜索
            best_move, score = self.engine.search(
                self.board,
                depth=depth,
                time_limit_ms=time_ms,
            )
            
            if best_move:
                result = {
                    'success': True,
                    'from_row': best_move.from_row,
                    'from_col': best_move.from_col,
                    'to_row': best_move.to_row,
                    'to_col': best_move.to_col,
                    'score': score,
                }
            else:
                result = {'success': False, 'error': 'No move found'}
            
            self.send_json(result)
        except Exception as e:
            self.send_json({'success': False, 'error': str(e)})
    
    def handle_reset(self):
        """重置棋盘"""
        self.board = Board()
        self.send_json({'success': True})
    
    def handle_make_move(self, parsed):
        """执行着法"""
        try:
            params = urllib.parse.parse_qs(parsed.query)
            from_row = int(params['fr'][0])
            from_col = int(params['fc'][0])
            to_row = int(params['tr'][0])
            to_col = int(params['tc'][0])
            
            move = Move(from_row, from_col, to_row, to_col, 0)
            self.board.make_move(move)
            
            self.send_json({'success': True})
        except Exception as e:
            self.send_json({'success': False, 'error': str(e)})
    
    def handle_status(self):
        """获取状态"""
        fen = to_fen(self.board)
        self.send_json({
            'success': True,
            'fen': fen,
            'player': self.board.current_player,
        })
    
    def send_json(self, data):
        """发送 JSON 响应"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def log_message(self, format, *args):
        """简化日志"""
        if '/api/' in args[0]:
            print(f"[API] {args[0]}")


def run_server(port=8080, net_path=None):
    """启动服务器"""
    ChessAPIHandler.init_engine(net_path)
    
    # Modified to listen on 0.0.0.0 for Docker compatibility
    server = HTTPServer(('0.0.0.0', port), ChessAPIHandler)
    print(f"="*50)
    print(f"Chess AI Server started!")
    print(f"  URL: http://localhost:{port}")
    print(f"  Net: {net_path or 'None'}")
    print(f"="*50)
    print("Open http://localhost:{} in your browser".format(port))
    print("Press Ctrl+C to stop\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--net', type=str, default=None)
    
    args = parser.parse_args()
    
    run_server(args.port, args.net)
