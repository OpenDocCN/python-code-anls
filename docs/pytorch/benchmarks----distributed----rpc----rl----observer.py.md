# `.\pytorch\benchmarks\distributed\rpc\rl\observer.py`

```
import random
import time

from agent import AgentBase  # 导入AgentBase类

import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import rpc_sync


class ObserverBase:
    def __init__(self):
        r"""
        Inits observer class
        """
        self.id = rpc.get_worker_info().id  # 获取当前观察者的工作信息ID

    def set_state(self, state_size, batch):
        r"""
        Further initializes observer to be aware of rpc environment
        Args:
            state_size (list): List of integers denoting dimensions of state
            batch (bool): Whether agent will be using batch select action
        """
        self.state_size = state_size  # 设置观察者状态尺寸
        self.select_action = (
            AgentBase.select_action_batch  # 如果batch为True，则使用AgentBase中的批量选择动作函数
            if batch
            else AgentBase.select_action_non_batch  # 否则使用非批量选择动作函数
        )

    def reset(self):
        r"""
        Resets state randomly
        """
        state = torch.rand(self.state_size)  # 生成随机状态
        return state

    def step(self, action):
        r"""
        Generates random state and reward
        Args:
            action (int): Int received from agent representing action to take on state
        """
        state = torch.rand(self.state_size)  # 生成随机状态
        reward = random.randint(0, 1)  # 生成随机奖励，0或1

        return state, reward

    def run_ob_episode(self, agent_rref, n_steps):
        r"""
        Runs single observer episode where for n_steps, an action is selected
        from the agent based on current state and state is updated
        Args:
            agent_rref (RRef): Remote Reference to the agent
            n_steps (int): Number of times to select an action to transform state per episode
        """
        state, ep_reward = self.reset(), None  # 重置状态，并初始化剧集奖励为None
        rewards = torch.zeros(n_steps)  # 创建大小为n_steps的零张量存储奖励
        observer_latencies = []  # 存储观察者延迟时间列表
        observer_throughput = []  # 存储观察者吞吐量列表

        for st in range(n_steps):
            ob_latency_start = time.time()  # 记录观察者操作开始时间
            action = rpc_sync(
                agent_rref.owner(),  # 获取agent_rref的所有者
                self.select_action,  # 调用选择动作方法
                args=(agent_rref, self.id, state),  # 方法参数
            )

            ob_latency = time.time() - ob_latency_start  # 计算操作延迟时间
            observer_latencies.append(ob_latency)  # 将延迟时间添加到列表
            observer_throughput.append(1 / ob_latency)  # 计算并添加吞吐量

            state, reward = self.step(action)  # 执行步骤并更新状态及奖励
            rewards[st] = reward  # 将奖励存储在对应位置

        return [rewards, ep_reward, observer_latencies, observer_throughput]  # 返回结果列表
```