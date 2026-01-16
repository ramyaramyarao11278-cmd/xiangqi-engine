"""
DQN Agent 实现
包含经验回放、目标网络等核心组件
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .network import SimpleDQN, XiangqiNet
from .move import Move


@dataclass
class Experience:
    """经验元组"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    legal_actions: List[int]  # 当前状态的合法动作


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, exp: Experience):
        self.buffer.append(exp)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN 智能体
    
    特点：
    - Experience Replay
    - Target Network
    - ε-greedy 探索
    - 合法动作掩码
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, ...] = (15, 10, 9),
        action_size: int = 8100,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        use_cuda: bool = True,
    ):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step = 0
        
        # 设备
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 计算状态展平后的大小
        flat_state_size = np.prod(state_shape)
        
        # 网络
        self.policy_net = SimpleDQN(state_size=flat_state_size, action_size=action_size).to(self.device)
        self.target_net = SimpleDQN(state_size=flat_state_size, action_size=action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def get_action(self, state: np.ndarray, legal_actions: List[int]) -> int:
        """
        ε-greedy 选择动作
        
        Args:
            state: 当前状态
            legal_actions: 合法动作索引列表
            
        Returns:
            选择的动作索引
        """
        if len(legal_actions) == 0:
            return -1
        
        if random.random() < self.epsilon:
            # 探索：随机选择合法动作
            return random.choice(legal_actions)
        else:
            # 利用：选择 Q 值最高的合法动作
            self.policy_net.eval()
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()
            
            # 只考虑合法动作
            legal_q = {a: q_values[a] for a in legal_actions}
            return max(legal_q, key=legal_q.get)
    
    def get_best_action(self, state: np.ndarray, legal_actions: List[int]) -> int:
        """获取最优动作（不探索）"""
        if len(legal_actions) == 0:
            return -1
        
        self.policy_net.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()
        
        legal_q = {a: q_values[a] for a in legal_actions}
        return max(legal_q, key=legal_q.get)
    
    def store_experience(self, state, action, reward, next_state, done, legal_actions):
        """存储经验"""
        exp = Experience(state, action, reward, next_state, done, legal_actions)
        self.replay_buffer.push(exp)
    
    def train(self) -> Optional[float]:
        """
        训练一步
        
        Returns:
            loss 值，如果缓冲区样本不足则返回 None
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样
        batch = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # 计算当前 Q 值
        self.policy_net.train()
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标 Q 值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = self.loss_fn(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 衰减 ε
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
        print(f"Model loaded from {path}")


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("DQN Agent Test\n")
    
    # 创建 Agent
    agent = DQNAgent(use_cuda=False)
    print(f"Epsilon: {agent.epsilon}")
    
    # 模拟一些经验
    print("\nSimulating experiences...")
    for i in range(100):
        state = np.random.randn(15, 10, 9).astype(np.float32)
        next_state = np.random.randn(15, 10, 9).astype(np.float32)
        legal_actions = list(range(10))  # 模拟 10 个合法动作
        action = agent.get_action(state, legal_actions)
        reward = random.uniform(-1, 1)
        done = random.random() < 0.1
        
        agent.store_experience(state, action, reward, next_state, done, legal_actions)
    
    print(f"Buffer size: {len(agent.replay_buffer)}")
    
    # 训练测试
    print("\nTraining test...")
    for i in range(10):
        loss = agent.train()
        if loss is not None:
            print(f"Step {i+1}: Loss = {loss:.4f}, Epsilon = {agent.epsilon:.4f}")
    
    print("\nDQN Agent test passed!")
