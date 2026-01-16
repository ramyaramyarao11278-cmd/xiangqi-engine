"""
监督学习训练脚本
使用人类高手棋谱训练 Policy-Value 网络

数据来源：GitHub 上的大师棋谱 PGN 文件
"""
import os
import sys
import re
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.board import Board
from src.move import Move, MoveGenerator
from src.fen import parse_fen, STARTING_FEN
from src.policy_value_net import SimplePolicyValueNet, MoveEncoder


@dataclass
class GameRecord:
    """棋谱记录"""
    moves: List[str]  # ICCS 格式着法列表
    result: int       # 1=红胜, -1=黑胜, 0=和


class PGNParser:
    """PGN 棋谱解析器"""
    
    def __init__(self):
        self.move_pattern = re.compile(r'([a-i])(\d)([a-i])(\d)')
    
    def parse_file(self, filepath: str) -> List[GameRecord]:
        """解析 PGN 文件"""
        games = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
                content = f.read()
        
        # 分割多局棋谱
        game_texts = re.split(r'\n\s*\n(?=\[Event)', content)
        
        for text in game_texts:
            game = self._parse_game(text)
            if game and len(game.moves) >= 10:  # 至少 10 步
                games.append(game)
        
        return games
    
    def _parse_game(self, text: str) -> Optional[GameRecord]:
        """解析单局棋谱"""
        # 提取结果
        result_match = re.search(r'\[Result\s+"([^"]+)"\]', text)
        if result_match:
            result_str = result_match.group(1)
            if '1-0' in result_str:
                result = 1
            elif '0-1' in result_str:
                result = -1
            else:
                result = 0
        else:
            result = 0
        
        # 提取着法（ICCS 格式）
        moves = []
        
        # 移除标签部分
        text_without_tags = re.sub(r'\[.*?\]', '', text)
        
        # 查找 ICCS 格式着法 (如 b2e2, h7e7)
        for match in self.move_pattern.finditer(text_without_tags.lower()):
            move_str = match.group(0)
            moves.append(move_str)
        
        if not moves:
            return None
        
        return GameRecord(moves=moves, result=result)
    
    def iccs_to_move(self, iccs: str) -> Optional[Move]:
        """ICCS 格式转 Move 对象"""
        if len(iccs) < 4:
            return None
        
        try:
            from_col = ord(iccs[0]) - ord('a')
            from_row = 9 - int(iccs[1])
            to_col = ord(iccs[2]) - ord('a')
            to_row = 9 - int(iccs[3])
            
            if 0 <= from_row < 10 and 0 <= from_col < 9:
                if 0 <= to_row < 10 and 0 <= to_col < 9:
                    return Move(from_row, from_col, to_row, to_col, 0)
        except:
            pass
        
        return None


class ChessDataset(Dataset):
    """象棋训练数据集"""
    
    def __init__(self, samples: List[Tuple[np.ndarray, int, float]]):
        """
        samples: [(state, action, value), ...]
        """
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        state, action, value = self.samples[idx]
        return (
            torch.FloatTensor(state),
            torch.LongTensor([action]),
            torch.FloatTensor([value])
        )


class SupervisedTrainer:
    """监督学习训练器"""
    
    def __init__(
        self,
        pgn_dir: str = "pgn_data",
        save_dir: str = "checkpoints_supervised",
        batch_size: int = 256,
        lr: float = 1e-3,
        use_cuda: bool = True,
    ):
        self.pgn_dir = Path(pgn_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        self.net = SimplePolicyValueNet(action_size=8100).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        self.batch_size = batch_size
        self.move_encoder = MoveEncoder()
        self.parser = PGNParser()
    
    def get_state(self, board: Board) -> np.ndarray:
        """棋盘转网络输入"""
        state = np.zeros((15, 10, 9), dtype=np.float32)
        for row in range(10):
            for col in range(9):
                piece = board.board[row, col]
                if piece > 0:
                    state[piece - 1, row, col] = 1.0
        if board.current_player == 1:
            state[14, :, :] = 1.0
        return state
    
    def load_games(self, max_games: int = None) -> List[GameRecord]:
        """加载棋谱"""
        all_games = []
        
        pgn_files = list(self.pgn_dir.glob("*.pgn")) + list(self.pgn_dir.glob("*.PGN"))
        
        for pgn_file in pgn_files:
            print(f"Loading: {pgn_file.name}")
            games = self.parser.parse_file(str(pgn_file))
            all_games.extend(games)
            
            if max_games and len(all_games) >= max_games:
                all_games = all_games[:max_games]
                break
        
        print(f"Total games loaded: {len(all_games)}")
        return all_games
    
    def games_to_samples(self, games: List[GameRecord]) -> List[Tuple[np.ndarray, int, float]]:
        """棋谱转训练样本"""
        samples = []
        
        for game_idx, game in enumerate(games):
            board = Board()
            
            for move_idx, iccs in enumerate(game.moves):
                move = self.parser.iccs_to_move(iccs)
                if move is None:
                    break
                
                # 验证着法合法性
                piece = board.board[move.from_row, move.from_col]
                if piece == 0:
                    break
                
                # 记录样本
                state = self.get_state(board)
                action = self.move_encoder.encode_move(
                    move.from_row, move.from_col,
                    move.to_row, move.to_col
                )
                
                # Value 目标：从当前方视角的最终结果
                if game.result == 0:
                    value = 0.0
                elif (game.result == 1 and board.current_player == 1) or \
                     (game.result == -1 and board.current_player == -1):
                    value = 1.0
                else:
                    value = -1.0
                
                samples.append((state, action, value))
                
                # 执行着法
                try:
                    board.make_move(move)
                except:
                    break
            
            if (game_idx + 1) % 1000 == 0:
                print(f"  Processed {game_idx + 1} games, {len(samples)} samples")
        
        print(f"Total samples: {len(samples)}")
        return samples
    
    def train(self, epochs: int = 10, max_games: int = None):
        """训练"""
        print("="*60)
        print("Supervised Learning Training")
        print("="*60)
        
        # 加载数据
        games = self.load_games(max_games)
        if len(games) == 0:
            print("No games found! Please download PGN files to pgn_data/")
            return
        
        samples = self.games_to_samples(games)
        
        # 划分训练/验证集
        random.shuffle(samples)
        split = int(len(samples) * 0.9)
        train_samples = samples[:split]
        val_samples = samples[split:]
        
        train_loader = DataLoader(
            ChessDataset(train_samples),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            ChessDataset(val_samples),
            batch_size=self.batch_size,
        )
        
        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")
        print("-"*60)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练
            self.net.train()
            train_p_loss = 0
            train_v_loss = 0
            train_batches = 0
            
            for states, actions, values in train_loader:
                states = states.to(self.device)
                actions = actions.squeeze(1).to(self.device)
                values = values.squeeze(1).to(self.device)
                
                policy_logits, value_pred = self.net(states)
                
                p_loss = F.cross_entropy(policy_logits, actions)
                v_loss = F.mse_loss(value_pred.squeeze(), values)
                total_loss = p_loss + v_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                
                train_p_loss += p_loss.item()
                train_v_loss += v_loss.item()
                train_batches += 1
            
            # 验证
            self.net.eval()
            val_p_loss = 0
            val_v_loss = 0
            val_batches = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for states, actions, values in val_loader:
                    states = states.to(self.device)
                    actions = actions.squeeze(1).to(self.device)
                    values = values.squeeze(1).to(self.device)
                    
                    policy_logits, value_pred = self.net(states)
                    
                    p_loss = F.cross_entropy(policy_logits, actions)
                    v_loss = F.mse_loss(value_pred.squeeze(), values)
                    
                    val_p_loss += p_loss.item()
                    val_v_loss += v_loss.item()
                    val_batches += 1
                    
                    # 准确率
                    pred = policy_logits.argmax(dim=1)
                    correct += (pred == actions).sum().item()
                    total += actions.size(0)
            
            train_p = train_p_loss / train_batches
            train_v = train_v_loss / train_batches
            val_p = val_p_loss / val_batches
            val_v = val_v_loss / val_batches
            accuracy = correct / total * 100
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train[P:{train_p:.4f} V:{train_v:.4f}] "
                  f"Val[P:{val_p:.4f} V:{val_v:.4f}] "
                  f"Acc:{accuracy:.1f}%")
            
            # 保存最佳模型
            val_total = val_p + val_v
            if val_total < best_val_loss:
                best_val_loss = val_total
                torch.save({
                    'net': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_total,
                    'accuracy': accuracy,
                }, self.save_dir / 'best_model.pt')
                print(f"  -> Saved best model (acc={accuracy:.1f}%)")
        
        # 保存最终模型
        torch.save({
            'net': self.net.state_dict(),
            'epoch': epochs,
        }, self.save_dir / 'final_model.pt')
        
        print("-"*60)
        print("Training complete!")
        print(f"Best model saved to: {self.save_dir / 'best_model.pt'}")


def download_sample_pgn():
    """下载示例棋谱（如果没有）"""
    pgn_dir = Path("pgn_data")
    pgn_dir.mkdir(exist_ok=True)
    
    sample_file = pgn_dir / "sample_games.pgn"
    if sample_file.exists():
        print("Sample PGN already exists")
        return
    
    # 创建一些示例棋谱用于测试
    sample_pgn = '''[Event "Sample Game 1"]
[Result "1-0"]

1. b2e2 h9g7 2. h2e2 b9c7 3. h0g2 a9a8 4. i0h0 c6c5 
5. h0h4 a8d8 6. e2e6 d8d4 7. h4h9 g7h9 8. g2h4 1-0

[Event "Sample Game 2"]
[Result "0-1"]

1. h2e2 b9c7 2. b2d2 a9b9 3. h0g2 h9g7 4. i0h0 i9h9
5. g3g4 g6g5 6. h0h6 c9e7 7. g2f4 h7h0 0-1

[Event "Sample Game 3"]
[Result "1/2-1/2"]

1. b2e2 h9g7 2. h2e2 b9c7 3. g3g4 c6c5 4. h0g2 a9a8
5. i0h0 c7b5 6. g2f4 b5c3 1/2-1/2
'''
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_pgn)
    
    print(f"Created sample PGN: {sample_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn_dir', type=str, default='pgn_data')
    parser.add_argument('--save_dir', type=str, default='checkpoints_supervised')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_games', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    
    # 确保有棋谱
    download_sample_pgn()
    
    trainer = SupervisedTrainer(
        pgn_dir=args.pgn_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    
    trainer.train(epochs=args.epochs, max_games=args.max_games)


if __name__ == "__main__":
    main()
