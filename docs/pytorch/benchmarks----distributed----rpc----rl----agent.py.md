# `.\pytorch\benchmarks\distributed\rpc\rl\agent.py`

```py
# 导入运算符模块
import operator
# 导入多线程模块
import threading
# 导入时间模块
import time
# 导入 functools 模块中的 reduce 函数
from functools import reduce

# 导入 PyTorch 模块
import torch
# 导入 PyTorch 分布式 RPC 模块
import torch.distributed.rpc as rpc
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 导入 PyTorch 的函数接口模块
import torch.nn.functional as F
# 导入 PyTorch 优化器模块
import torch.optim as optim
# 导入 PyTorch 分布模块中的分类分布类
from torch.distributions import Categorical

# 观察者名称模板
OBSERVER_NAME = "observer{}"

# 策略类，继承自 nn.Module
class Policy(nn.Module):
    def __init__(self, in_features, nlayers, out_features):
        r"""
        初始化策略类
        Args:
            in_features (int): 模型输入特征数
            nlayers (int): 模型中的层数
            out_features (int): 模型输出的特征数
        """
        super().__init__()

        # 策略模型定义
        self.model = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(in_features, out_features),
            *[nn.Linear(out_features, out_features) for _ in range(nlayers)],
        )
        self.dim = 0

    def forward(self, x):
        # 模型前向传播，计算动作分数，并应用 softmax 激活函数
        action_scores = self.model(x)
        return F.softmax(action_scores, dim=self.dim)


# 代理基类
class AgentBase:
    def __init__(self):
        r"""
        初始化代理基类
        """
        # 获取当前工作节点的 ID
        self.id = rpc.get_worker_info().id
        self.running_reward = 0
        self.eps = 1e-7

        # 奖励字典
        self.rewards = {}

        # 未来动作的异步结果对象
        self.future_actions = torch.futures.Future()
        # 线程锁对象
        self.lock = threading.Lock()

        # 代理的延迟时间起始和结束
        self.agent_latency_start = None
        self.agent_latency_end = None
        # 代理延迟时间和吞吐量统计
        self.agent_latency = []
        self.agent_throughput = []

    def reset_metrics(self):
        r"""
        重置所有基准指标的值
        """
        self.agent_latency_start = None
        self.agent_latency_end = None
        self.agent_latency = []
        self.agent_throughput = []

    def set_world(self, batch_size, state_size, nlayers, out_features, batch=True):
        r"""
        进一步初始化代理以适应 RPC 环境
        Args:
            batch_size (int): 要处理的观察者请求的批次大小
            state_size (list): 表示状态维度的整数列表
            nlayers (int): 模型中的层数
            out_features (int): 模型输出的特征数
            batch (bool): 是否批处理观察者请求
        """
        self.batch = batch
        # 创建策略模型
        self.policy = Policy(reduce(operator.mul, state_size), nlayers, out_features)
        # 使用 Adam 优化器初始化
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)

        self.batch_size = batch_size
        # 根据批次大小为每个观察者请求分配奖励列表
        for rank in range(batch_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(rank + 2))
            self.rewards[ob_info.id] = []

        # 如果不是批处理模式，则初始化保存的对数概率字典
        self.saved_log_probs = (
            [] if self.batch else {k: [] for k in range(self.batch_size)}
        )

        # 初始化挂起的状态数和状态尺寸
        self.pending_states = self.batch_size
        self.state_size = state_size
        # 初始化状态张量
        self.states = torch.zeros(self.batch_size, *state_size)

    @staticmethod
    @rpc.functions.async_execution
    def select_action_batch(agent_rref, observer_id, state):
        r"""
        Receives state from an observer to select action for.  Queues the observers's request
        for an action until queue size equals batch size named during Agent initiation, at which point
        actions are selected for all pending observer requests and communicated back to observers
        Args:
            agent_rref (RRef): RRef of this agent
            observer_id (int): Observer id of observer calling this function
            state (Tensor): Tensor representing current state held by observer
        """
        # 获取本地代理对象的引用
        self = agent_rref.local_value()
        # 调整观察者ID，以匹配内部索引
        observer_id -= 2

        # 将观察者的状态复制到自身状态列表中的对应位置
        self.states[observer_id].copy_(state)
        
        # 等待未来的动作选择结果
        future_action = self.future_actions.then(
            lambda future_actions: future_actions.wait()[observer_id].item()
        )

        # 使用锁确保多线程安全操作
        with self.lock:
            # 如果待处理状态达到批处理大小，则记录代理开始处理时间
            if self.pending_states == self.batch_size:
                self.agent_latency_start = time.time()
            # 减少待处理状态计数
            self.pending_states -= 1
            # 如果所有观察者请求都处理完毕，则重置待处理状态计数
            if self.pending_states == 0:
                self.pending_states = self.batch_size
                # 计算当前状态下动作的概率分布
                probs = self.policy(self.states)
                m = Categorical(probs)
                # 从概率分布中采样动作
                actions = m.sample()
                # 记录动作对应的对数概率到保存的日志概率列表中
                self.saved_log_probs.append(m.log_prob(actions).t())
                # 设置未来动作的结果，并重置未来动作的状态
                future_actions = self.future_actions
                self.future_actions = torch.futures.Future()
                future_actions.set_result(actions)

                # 记录代理处理结束时间
                self.agent_latency_end = time.time()

                # 计算批处理延迟时间
                batch_latency = self.agent_latency_end - self.agent_latency_start
                # 将批处理延迟时间记录到代理延迟列表中
                self.agent_latency.append(batch_latency)
                # 计算并记录代理的吞吐量
                self.agent_throughput.append(self.batch_size / batch_latency)

        # 返回未来动作的结果
        return future_action

    @staticmethod
    def select_action_non_batch(agent_rref, observer_id, state):
        r"""
        Select actions based on observer state and communicates back to observer
        Args:
            agent_rref (RRef): RRef of this agent
            observer_id (int): Observer id of observer calling this function
            state (Tensor): Tensor representing current state held by observer
        """
        # 获取本地代理对象的引用
        self = agent_rref.local_value()
        # 调整观察者ID，以匹配内部索引
        observer_id -= 2
        # 记录非批处理动作选择的开始时间
        agent_latency_start = time.time()

        # 将观察者的状态转换为浮点张量并增加维度
        state = state.float().unsqueeze(0)
        # 计算给定状态下动作的概率分布
        probs = self.policy(state)
        m = Categorical(probs)
        # 从概率分布中采样一个动作
        action = m.sample()
        # 将动作对应的对数概率记录到保存的日志概率列表中
        self.saved_log_probs[observer_id].append(m.log_prob(action))

        # 记录非批处理动作选择的结束时间
        agent_latency_end = time.time()
        # 计算非批处理动作选择的延迟时间
        non_batch_latency = agent_latency_end - agent_latency_start
        # 将延迟时间记录到代理延迟列表中
        self.agent_latency.append(non_batch_latency)
        # 计算并记录代理的吞吐量
        self.agent_throughput.append(1 / non_batch_latency)

        # 返回选择的动作的值
        return action.item()
    # 定义一个方法，用于结束一个 episode（一段时间内的模型训练或交互）
    def finish_episode(self, rets):
        r"""
        结束当前 episode
        Args:
            rets (list): 包含在 episode 运行期间由选择动作调用生成的奖励列表
        """
        # 返回代理的延迟和吞吐量作为结果
        return self.agent_latency, self.agent_throughput
```