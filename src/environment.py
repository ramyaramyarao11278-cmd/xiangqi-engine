"""
中国象棋游戏环境（用于强化学习）
整合棋盘、着法生成、规则判定
"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass

from .board import Board, Piece
from .move import Move, MoveGenerator


@dataclass
class GameResult:
    """游戏结果"""
    winner: int  # 1: 红胜, -1: 黑胜, 0: 和棋
    reason: str  # 胜负原因


class XiangqiEnv:
    """
    中国象棋强化学习环境
    
    遵循 Gym 风格的接口：
    - reset() -> state
    - step(action) -> (next_state, reward, done, info)
    """
    
    # 棋子价值（用于计算奖励）
    PIECE_VALUES = {
        Piece.R_KING: 10000, Piece.B_KING: 10000,
        Piece.R_ROOK: 900, Piece.B_ROOK: 900,
        Piece.R_CANNON: 450, Piece.B_CANNON: 450,
        Piece.R_KNIGHT: 400, Piece.B_KNIGHT: 400,
        Piece.R_BISHOP: 200, Piece.B_BISHOP: 200,
        Piece.R_ADVISOR: 200, Piece.B_ADVISOR: 200,
        Piece.R_PAWN: 100, Piece.B_PAWN: 100,
    }
    
    def __init__(self, max_moves: int = 200):
        """
        Args:
            max_moves: 最大回合数（超过则和棋）
        """
        self.board = Board()
        self.move_gen = MoveGenerator(self.board)
        self.max_moves = max_moves
        self.history: List[Move] = []
        self.done = False
        self.result: Optional[GameResult] = None
    
    def reset(self) -> np.ndarray:
        """重置游戏，返回初始状态"""
        self.board.reset()
        self.move_gen = MoveGenerator(self.board)
        self.history = []
        self.done = False
        self.result = None
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        获取当前状态（用于神经网络输入）
        
        返回形状: (15, 10, 9)
        - 14 个通道表示不同棋子
        - 1 个通道表示当前玩家
        """
        state = np.zeros((15, 10, 9), dtype=np.float32)
        
        for row in range(10):
            for col in range(9):
                piece = self.board.board[row, col]
                if piece > 0:
                    state[piece - 1, row, col] = 1.0
        
        # 当前玩家通道
        if self.board.current_player == 1:
            state[14, :, :] = 1.0
        
        return state
    
    def get_legal_moves(self) -> List[Move]:
        """获取当前方的所有合法着法"""
        is_red = self.board.current_player == 1
        return self.move_gen.generate_moves(is_red)
    
    def step(self, move: Move) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行着法
        
        Args:
            move: 着法对象
            
        Returns:
            next_state: 下一状态
            reward: 即时奖励（红方视角）
            done: 是否结束
            info: 附加信息
        """
        if self.done:
            return self.get_state(), 0.0, True, {"error": "Game already ended"}
        
        # 记录被吃的棋子
        captured = self.board.board[move.to_row, move.to_col]
        move.captured = captured
        
        # 执行着法
        piece = self.board.board[move.from_row, move.from_col]
        move.piece = piece
        self.board.board[move.from_row, move.from_col] = 0
        self.board.board[move.to_row, move.to_col] = piece
        
        # 记录历史
        self.history.append(move)
        self.board.move_count += 1
        
        # 计算奖励（红方视角）
        reward = self._calculate_reward(move, captured)
        
        # 切换玩家
        self.board.current_player *= -1
        self.move_gen = MoveGenerator(self.board)
        
        # 检查游戏结束
        self._check_game_end()
        
        info = {
            "move": move,
            "captured": captured,
            "move_count": self.board.move_count,
        }
        
        if self.done:
            info["result"] = self.result
        
        return self.get_state(), reward, self.done, info
    
    def step_by_index(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        通过动作索引执行着法
        
        Args:
            action_index: 合法着法列表中的索引
        """
        legal_moves = self.get_legal_moves()
        if action_index >= len(legal_moves):
            return self.get_state(), -1.0, True, {"error": "Invalid action index"}
        
        return self.step(legal_moves[action_index])
    
    def _calculate_reward(self, move: Move, captured: int) -> float:
        """
        计算即时奖励（红方视角）
        
        奖励设计：
        - 吃子：+/- 棋子价值
        - 胜利：+1000
        - 失败：-1000
        - 每步：-0.01（鼓励快速结束）
        """
        reward = -0.01  # 基础步惩罚
        
        # 吃子奖励
        if captured > 0:
            value = self.PIECE_VALUES.get(captured, 0) / 1000  # 归一化
            # 红方吃黑子得正分，黑方吃红子红方得负分
            if self.board.current_player == 1:  # 红方刚走
                reward += value
            else:
                reward -= value
        
        return reward
    
    def _check_game_end(self):
        """检查游戏是否结束"""
        # 检查将帅是否存在
        red_king = self.board.find_king(is_red=True)
        black_king = self.board.find_king(is_red=False)
        
        if red_king is None:
            self.done = True
            self.result = GameResult(-1, "Red king captured")
            return
        
        if black_king is None:
            self.done = True
            self.result = GameResult(1, "Black king captured")
            return
        
        # 检查当前方是否无子可动（困毙）
        legal_moves = self.get_legal_moves()
        if len(legal_moves) == 0:
            winner = -self.board.current_player  # 当前方输
            self.done = True
            self.result = GameResult(winner, "No legal moves (stalemate)")
            return
        
        # 检查回合数
        if self.board.move_count >= self.max_moves:
            self.done = True
            self.result = GameResult(0, "Max moves reached")
            return
    
    def is_in_check(self, is_red: bool) -> bool:
        """检查指定方是否被将军"""
        king_pos = self.board.find_king(is_red)
        if king_pos is None:
            return True
        
        # 检查对方是否有着法能吃到将/帅
        enemy_moves = self.move_gen.generate_moves(not is_red)
        for move in enemy_moves:
            if (move.to_row, move.to_col) == king_pos:
                return True
        
        return False
    
    def render(self):
        """渲染当前局面"""
        self.board.render()
        
        if self.done and self.result:
            if self.result.winner == 1:
                print("Result: RED WINS!")
            elif self.result.winner == -1:
                print("Result: BLACK WINS!")
            else:
                print("Result: DRAW")
            print(f"Reason: {self.result.reason}")
    
    def get_action_space_size(self) -> int:
        """
        获取动作空间大小
        中国象棋所有可能的着法数 = 起点(90) × 终点(90) = 8100
        （实际有效着法远小于此）
        """
        return 90 * 90
    
    def move_to_action(self, move: Move) -> int:
        """将着法转换为动作索引"""
        from_idx = move.from_row * 9 + move.from_col
        to_idx = move.to_row * 9 + move.to_col
        return from_idx * 90 + to_idx
    
    def action_to_move(self, action: int) -> Move:
        """将动作索引转换为着法"""
        from_idx = action // 90
        to_idx = action % 90
        return Move(
            from_row=from_idx // 9,
            from_col=from_idx % 9,
            to_row=to_idx // 9,
            to_col=to_idx % 9
        )


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Xiangqi Environment Test\n")
    
    env = XiangqiEnv()
    state = env.reset()
    
    print(f"State shape: {state.shape}")
    env.render()
    
    # 模拟几步随机走子
    print("\nSimulating 10 random moves...\n")
    
    for i in range(10):
        legal_moves = env.get_legal_moves()
        if len(legal_moves) == 0 or env.done:
            break
        
        # 随机选择一个着法
        move = np.random.choice(legal_moves)
        next_state, reward, done, info = env.step(move)
        
        player = "Red" if env.board.current_player == -1 else "Black"
        print(f"Move {i+1}: {move} | Reward: {reward:.3f}")
        
        if done:
            print(f"\nGame Over!")
            break
    
    env.render()
