"""
分批训练脚本
解决内存不足问题：分批加载棋谱，逐批训练

特点：
1. 每批加载 N 局棋谱训练
2. 累积训练，模型持续改进
3. 内存使用稳定在 10GB 以内
"""
import os
import sys
import time
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.action import ACTION_SIZE
from src.policy_value_net import SimplePolicyValueNet
from src.iccs_pgn import parse_file
from src.dataset_builder import build_samples_from_game, SupervisedSample


class ChessDataset(Dataset):
    def __init__(self, samples: List[SupervisedSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.FloatTensor(s.state),
            torch.LongTensor([s.action]),
            torch.FloatTensor([s.z])
        )


class BatchTrainer:
    """分批训练器"""
    
    def __init__(
        self,
        pgn_file: str,
        save_dir: str = "checkpoints_human",
        batch_size: int = 256,
        lr: float = 1e-3,
        games_per_batch: int = 3000,  # 每批处理局数
        use_cuda: bool = True,
    ):
        self.pgn_file = Path(pgn_file)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # 网络
        self.net = SimplePolicyValueNet(action_size=ACTION_SIZE).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        # 尝试加载已有模型
        model_path = self.save_dir / 'best_model.pt'
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(checkpoint['net'])
            print(f"Loaded existing model: {model_path}")
        
        self.batch_size = batch_size
        self.games_per_batch = games_per_batch
    
    def load_batch(self, games, start_idx: int) -> List[SupervisedSample]:
        """加载一批棋谱"""
        samples = []
        end_idx = min(start_idx + self.games_per_batch, len(games))
        
        for i in range(start_idx, end_idx):
            result, reason = build_samples_from_game(games[i], augment_mirror=True)
            if result:
                samples.extend(result)
        
        return samples
    
    def train_batch(self, samples: List[SupervisedSample], epochs: int = 3):
        """训练一批数据"""
        random.shuffle(samples)
        split = int(len(samples) * 0.9)
        
        train_loader = DataLoader(
            ChessDataset(samples[:split]),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            ChessDataset(samples[split:]),
            batch_size=self.batch_size,
        )
        
        for epoch in range(epochs):
            # 训练
            self.net.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for states, actions, zs in train_loader:
                states = states.to(self.device)
                actions = actions.squeeze(1).to(self.device)
                zs = zs.squeeze(1).to(self.device)
                
                policy_logits, value_pred = self.net(states)
                
                p_loss = F.cross_entropy(policy_logits, actions, label_smoothing=0.1)
                v_loss = F.mse_loss(value_pred.squeeze(), zs)
                total_loss = p_loss + v_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += p_loss.item() * len(actions)
                train_correct += (policy_logits.argmax(1) == actions).sum().item()
                train_total += len(actions)
            
            # 验证
            self.net.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for states, actions, zs in val_loader:
                    states = states.to(self.device)
                    actions = actions.squeeze(1).to(self.device)
                    
                    policy_logits, _ = self.net(states)
                    val_correct += (policy_logits.argmax(1) == actions).sum().item()
                    val_total += len(actions)
            
            train_acc = train_correct / train_total * 100
            val_acc = val_correct / val_total * 100 if val_total > 0 else 0
            
            print(f"    Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")
        
        return val_acc
    
    def train(self, total_games: int = None, epochs_per_batch: int = 3):
        """分批训练"""
        print("="*60)
        print("Batch Training")
        print("="*60)
        print(f"Games per batch: {self.games_per_batch}")
        print(f"Epochs per batch: {epochs_per_batch}")
        print("-"*60)
        
        # 加载所有棋谱元数据
        print(f"Loading games from: {self.pgn_file}")
        games = parse_file(str(self.pgn_file))
        total = len(games)
        print(f"Total games: {total}")
        
        if total_games:
            games = games[:total_games]
            total = len(games)
        
        num_batches = (total + self.games_per_batch - 1) // self.games_per_batch
        print(f"Number of batches: {num_batches}")
        print("-"*60)
        
        best_acc = 0
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            start = batch_idx * self.games_per_batch
            
            print(f"\n[Batch {batch_idx+1}/{num_batches}] Games {start+1}-{min(start+self.games_per_batch, total)}")
            
            # 加载这批数据
            samples = self.load_batch(games, start)
            print(f"  Samples: {len(samples)}")
            
            if len(samples) < 100:
                print("  Skipping (too few samples)")
                continue
            
            # 训练
            val_acc = self.train_batch(samples, epochs_per_batch)
            
            # 保存
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'net': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'batch': batch_idx + 1,
                    'val_acc': val_acc,
                }, self.save_dir / 'best_model.pt')
                print(f"  -> Saved best model (acc={val_acc:.1f}%)")
            
            # 释放内存
            del samples
        
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print(f"Training complete!")
        print(f"  Best accuracy: {best_acc:.1f}%")
        print(f"  Time: {total_time/60:.1f} min")
        print(f"  Model: {self.save_dir / 'best_model.pt'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn', type=str, default='pgn_data/dpxq-99813games.pgns')
    parser.add_argument('--save_dir', type=str, default='checkpoints_human')
    parser.add_argument('--games_per_batch', type=int, default=3000)
    parser.add_argument('--epochs_per_batch', type=int, default=3)
    parser.add_argument('--max_games', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    
    args = parser.parse_args()
    
    trainer = BatchTrainer(
        pgn_file=args.pgn,
        save_dir=args.save_dir,
        games_per_batch=args.games_per_batch,
        batch_size=args.batch_size,
    )
    
    trainer.train(
        total_games=args.max_games,
        epochs_per_batch=args.epochs_per_batch,
    )


if __name__ == "__main__":
    main()
