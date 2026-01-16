"""
神经网络模型
用于策略评估和着法选择
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class XiangqiNet(nn.Module):
    """
    中国象棋神经网络
    
    输入: (batch, 15, 10, 9) - 15 通道的棋盘状态
    输出:
      - policy: (batch, 2086) - 所有合法着法的概率分布
      - value: (batch, 1) - 局面评估值 [-1, 1]
    
    注：2086 = 90个起点 × 最多约23个可能终点（简化动作空间）
    实际我们使用 90×90=8100 作为完整动作空间
    """
    
    def __init__(self, action_size: int = 8100):
        super().__init__()
        
        self.action_size = action_size
        
        # 卷积层：提取棋盘特征
        self.conv1 = nn.Conv2d(15, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResBlock(128) for _ in range(4)
        ])
        
        # 展平后的特征大小
        self.flat_size = 128 * 10 * 9  # 11520
        
        # 策略头 (Policy Head)
        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * 10 * 9, action_size)
        
        # 价值头 (Value Head)
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * 10 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch, 15, 10, 9) 棋盘状态
            
        Returns:
            policy: (batch, action_size) 动作概率（logits）
            value: (batch, 1) 局面评估
        """
        # 主干网络
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 残差块
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 策略头
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy = self.policy_fc(p)
        
        # 价值头
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value
    
    def predict(self, state: np.ndarray) -> tuple:
        """
        单个状态预测（用于推理）
        
        Args:
            state: (15, 10, 9) 棋盘状态
            
        Returns:
            policy: (action_size,) 动作概率
            value: float 局面评估
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


class ResBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.relu(x)


class SimpleDQN(nn.Module):
    """
    简化版 DQN（用于快速验证）
    
    输入: 展平的棋盘状态
    输出: Q 值
    """
    
    def __init__(self, state_size: int = 15 * 10 * 9, action_size: int = 8100):
        super().__init__()
        
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 15, 10, 9) 或 (batch, state_size)
        """
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """获取 Q 值"""
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0)
            if x.dim() == 4:
                x = x.view(x.size(0), -1)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            q_values = self(x).squeeze(0).cpu().numpy()
        return q_values


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Neural Network Test\n")
    
    # 测试 XiangqiNet
    print("Testing XiangqiNet...")
    net = XiangqiNet()
    
    # 假设输入
    batch_size = 4
    x = torch.randn(batch_size, 15, 10, 9)
    
    policy, value = net(x)
    print(f"Input shape: {x.shape}")
    print(f"Policy shape: {policy.shape}")  # (batch, 8100)
    print(f"Value shape: {value.shape}")    # (batch, 1)
    
    # 单个预测
    state = np.random.randn(15, 10, 9).astype(np.float32)
    policy_probs, val = net.predict(state)
    print(f"\nSingle prediction:")
    print(f"Policy sum: {policy_probs.sum():.4f}")  # 应该接近 1
    print(f"Value: {val:.4f}")
    
    # 测试 SimpleDQN
    print("\n--- Testing SimpleDQN ---")
    dqn = SimpleDQN()
    q_values = dqn.predict(state)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Max Q: {q_values.max():.4f}, Min Q: {q_values.min():.4f}")
    
    # 参数统计
    total_params = sum(p.numel() for p in net.parameters())
    print(f"\nXiangqiNet total parameters: {total_params:,}")
    
    dqn_params = sum(p.numel() for p in dqn.parameters())
    print(f"SimpleDQN total parameters: {dqn_params:,}")
