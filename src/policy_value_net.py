"""
Policy-Value 网络
用于辅助搜索引擎：
- Policy: 走子先验概率（加速搜索）
- Value: 局面评估（替换/融合手工评估）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.relu(x)


class PolicyValueNet(nn.Module):
    """
    策略-价值网络 (AlphaZero 风格)
    
    输入: (batch, 15, 10, 9) - 棋盘状态
      - 14通道: 各类棋子的位置 (one-hot)
      - 1通道: 当前玩家
    
    输出:
      - policy: (batch, 8100) 着法概率分布
        8100 = 90起点 × 90终点
      - value: (batch, 1) 局面评估 [-1, 1] (当前执子方视角)
    """
    
    # 动作空间：统一使用 (from, to) 编码
    # 8100 = 90 × 90，合法着法约 44-50 个，其余用 mask 屏蔽
    ACTION_SIZE = 8100  # 统一动作空间
    
    def __init__(self, num_res_blocks: int = 6, channels: int = 128):
        super().__init__()
        
        # 输入层
        self.conv_input = nn.Conv2d(15, channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # 残差塔
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_res_blocks)
        ])
        
        # 策略头
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 10 * 9, self.ACTION_SIZE)
        
        # 价值头
        self.value_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * 10 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (batch, 15, 10, 9) 棋盘状态
            
        Returns:
            policy_logits: (batch, ACTION_SIZE) 策略 logits
            value: (batch, 1) 价值 [-1, 1]
        """
        # 主干
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # 策略头
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)
        
        # 价值头
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy_logits, value
    
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        单个状态预测
        
        Args:
            state: (15, 10, 9) 棋盘状态
            
        Returns:
            policy: (ACTION_SIZE,) 策略概率
            value: float 价值
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            
            policy_logits, value = self(x)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()
        
        return policy, value


class MoveEncoder:
    """
    着法编码器
    将 (from_row, from_col, to_row, to_col) 映射到动作索引
    """
    
    def __init__(self):
        # 构建合法着法到索引的映射
        # 简化版：使用 from_pos * 90 + to_pos
        self.action_size = 8100  # 90 * 90
    
    def encode_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> int:
        """将着法编码为动作索引"""
        from_idx = from_row * 9 + from_col
        to_idx = to_row * 9 + to_col
        return from_idx * 90 + to_idx
    
    def decode_move(self, action: int) -> Tuple[int, int, int, int]:
        """将动作索引解码为着法"""
        from_idx = action // 90
        to_idx = action % 90
        return (from_idx // 9, from_idx % 9, to_idx // 9, to_idx % 9)


class SimplePolicyValueNet(nn.Module):
    """
    简化版策略-价值网络（用于快速实验）
    
    使用全连接层而非卷积
    """
    
    def __init__(self, action_size: int = 8100):
        super().__init__()
        
        input_size = 15 * 10 * 9  # 展平的棋盘
        hidden_size = 512
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        
        shared = self.shared(x)
        policy = self.policy_head(shared)
        value = self.value_head(shared)
        
        return policy, value
    
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).view(1, -1)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            
            policy_logits, value = self(x)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()
        
        return policy, value


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Policy-Value Network Test\n")
    
    # 测试 ResNet 版本
    print("Testing PolicyValueNet (ResNet)...")
    net = PolicyValueNet(num_res_blocks=4, channels=64)
    
    x = torch.randn(4, 15, 10, 9)
    policy, value = net(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Policy shape: {policy.shape}")  # (4, 2086)
    print(f"Value shape: {value.shape}")    # (4, 1)
    
    params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {params:,}")
    
    # 测试简化版
    print("\nTesting SimplePolicyValueNet...")
    simple_net = SimplePolicyValueNet()
    
    state = np.random.randn(15, 10, 9).astype(np.float32)
    policy, value = simple_net.predict(state)
    
    print(f"Policy sum: {policy.sum():.4f}")
    print(f"Value: {value:.4f}")
    
    simple_params = sum(p.numel() for p in simple_net.parameters())
    print(f"Parameters: {simple_params:,}")
