"""
人类棋谱监督学习训练脚本

特点：
1. 使用 ICCS 格式棋谱作为数据源
2. Policy: one-hot CE + 合法 mask + label smoothing
3. Value: MSE(z) 终局胜负（当前执子方视角）
4. 左右镜像数据增强
5. 训练完可无缝接回蒸馏闭环
"""
import os
import sys
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.action import ACTION_SIZE, get_legal_mask
from src.board import Board
from src.policy_value_net import SimplePolicyValueNet
from src.dataset_builder import build_dataset


class ChessDataset(Dataset):
    """象棋训练数据集"""
    
    def __init__(self, states: np.ndarray, actions: np.ndarray, zs: np.ndarray):
        self.states = states
        self.actions = actions
        self.zs = zs
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.states[idx]),
            torch.LongTensor([self.actions[idx]]),
            torch.FloatTensor([self.zs[idx]])
        )


class HumanSupervisedTrainer:
    """人类棋谱监督训练器"""
    
    def __init__(
        self,
        pgn_dir: str = "pgn_data",
        save_dir: str = "checkpoints_human",
        batch_size: int = 256,
        lr: float = 1e-3,
        label_smoothing: float = 0.1,
        use_legal_mask: bool = True,
        use_cuda: bool = True,
    ):
        self.pgn_dir = Path(pgn_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # 网络
        self.net = SimplePolicyValueNet(action_size=ACTION_SIZE).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        # 配置
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.use_legal_mask = use_legal_mask
    
    def compute_policy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        smoothing: float = 0.1,
    ) -> torch.Tensor:
        """
        计算 Policy 损失（带 label smoothing）
        
        Args:
            logits: (B, 8100)
            targets: (B,) 目标动作索引
            smoothing: label smoothing 系数
        """
        if smoothing > 0:
            # Label smoothing
            n_classes = logits.size(1)
            one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
            smooth_labels = one_hot * (1 - smoothing) + smoothing / n_classes
            
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(smooth_labels * log_probs).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(logits, targets)
        
        return loss
    
    def train_epoch(self, loader: DataLoader) -> Tuple[float, float, float]:
        """训练一个 epoch"""
        self.net.train()
        
        total_p_loss = 0
        total_v_loss = 0
        total_correct = 0
        total_samples = 0
        
        for states, actions, zs in loader:
            states = states.to(self.device)
            actions = actions.squeeze(1).to(self.device)
            zs = zs.squeeze(1).to(self.device)
            
            # 前向传播
            policy_logits, value_pred = self.net(states)
            
            # Policy 损失
            p_loss = self.compute_policy_loss(
                policy_logits, actions, self.label_smoothing
            )
            
            # Value 损失
            v_loss = F.mse_loss(value_pred.squeeze(), zs)
            
            # 总损失
            total_loss = p_loss + v_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            
            # 统计
            total_p_loss += p_loss.item() * len(actions)
            total_v_loss += v_loss.item() * len(actions)
            
            pred = policy_logits.argmax(dim=1)
            total_correct += (pred == actions).sum().item()
            total_samples += len(actions)
        
        avg_p = total_p_loss / total_samples
        avg_v = total_v_loss / total_samples
        accuracy = total_correct / total_samples * 100
        
        return avg_p, avg_v, accuracy
    
    def evaluate(self, loader: DataLoader) -> Tuple[float, float, float]:
        """验证"""
        self.net.eval()
        
        total_p_loss = 0
        total_v_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for states, actions, zs in loader:
                states = states.to(self.device)
                actions = actions.squeeze(1).to(self.device)
                zs = zs.squeeze(1).to(self.device)
                
                policy_logits, value_pred = self.net(states)
                
                p_loss = F.cross_entropy(policy_logits, actions)
                v_loss = F.mse_loss(value_pred.squeeze(), zs)
                
                total_p_loss += p_loss.item() * len(actions)
                total_v_loss += v_loss.item() * len(actions)
                
                pred = policy_logits.argmax(dim=1)
                total_correct += (pred == actions).sum().item()
                total_samples += len(actions)
        
        avg_p = total_p_loss / total_samples
        avg_v = total_v_loss / total_samples
        accuracy = total_correct / total_samples * 100
        
        return avg_p, avg_v, accuracy
    
    def train(
        self,
        epochs: int = 20,
        max_games: int = None,
        augment_mirror: bool = True,
    ):
        """运行训练"""
        print("="*60)
        print("Human Supervised Training")
        print("="*60)
        print(f"Label smoothing: {self.label_smoothing}")
        print(f"Augment mirror: {augment_mirror}")
        print("-"*60)
        
        # 构建数据集
        print("\nBuilding dataset...")
        states, actions, zs = build_dataset(
            str(self.pgn_dir),
            max_games=max_games,
            augment_mirror=augment_mirror,
        )
        
        if len(actions) == 0:
            print("No samples! Check PGN files in", self.pgn_dir)
            return
        
        # 划分训练/验证集
        indices = list(range(len(actions)))
        random.shuffle(indices)
        split = int(len(indices) * 0.9)
        
        train_idx = indices[:split]
        val_idx = indices[split:]
        
        train_dataset = ChessDataset(
            states[train_idx], actions[train_idx], zs[train_idx]
        )
        val_dataset = ChessDataset(
            states[val_idx], actions[val_idx], zs[val_idx]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        print(f"\nTrain samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print("-"*60)
        
        best_val_acc = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练
            train_p, train_v, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_p, val_v, val_acc = self.evaluate(val_loader)
            
            # 学习率调整
            self.scheduler.step()
            
            print(f"Epoch {epoch+1:2d}/{epochs}: "
                  f"Train[P:{train_p:.4f} V:{train_v:.4f} Acc:{train_acc:.1f}%] "
                  f"Val[P:{val_p:.4f} V:{val_v:.4f} Acc:{val_acc:.1f}%]")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'net': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'val_acc': val_acc,
                }, self.save_dir / 'best_model.pt')
                print(f"  -> Saved best model (acc={val_acc:.1f}%)")
        
        # 保存最终模型
        torch.save({
            'net': self.net.state_dict(),
            'epoch': epochs,
        }, self.save_dir / 'final_model.pt')
        
        total_time = time.time() - start_time
        print("-"*60)
        print(f"Training complete!")
        print(f"  Best validation accuracy: {best_val_acc:.1f}%")
        print(f"  Time: {total_time/60:.1f} min")
        print(f"  Model saved to: {self.save_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn_dir', type=str, default='pgn_data')
    parser.add_argument('--save_dir', type=str, default='checkpoints_human')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--max_games', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--no_mirror', action='store_true')
    
    args = parser.parse_args()
    
    trainer = HumanSupervisedTrainer(
        pgn_dir=args.pgn_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        label_smoothing=args.smoothing,
    )
    
    trainer.train(
        epochs=args.epochs,
        max_games=args.max_games,
        augment_mirror=not args.no_mirror,
    )


if __name__ == "__main__":
    main()
