# `.\pytorch\torch\testing\_internal\distributed\rpc\examples\reinforcement_learning_rpc_test.py`

```py
# 忽略类型检查错误
# 如果你需要修改这个文件以使测试通过，请同时对 https://github.com/pytorch/examples/blob/master/distributed/rpc/rl/main.py
# 和 https://pytorch.org/tutorials/intermediate/rpc_tutorial.html 进行相同的编辑

import numpy as np  # 导入 NumPy 库，用于数值计算
from itertools import count  # 导入 count 函数，用于生成无限迭代器

import torch  # 导入 PyTorch 深度学习库
import torch.distributed.rpc as rpc  # 导入 PyTorch 分布式 RPC 模块
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络函数接口
import torch.optim as optim  # 导入优化器模块
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote  # 导入分布式 RPC 相关函数
from torch.distributions import Categorical  # 导入概率分布函数

from torch.testing._internal.dist_utils import dist_init, worker_name  # 导入测试工具函数
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture  # 导入 RPC 测试固件

TOTAL_EPISODE_STEP = 5000  # 总的步数
GAMMA = 0.1  # 折扣因子
SEED = 543  # 随机种子

def _call_method(method, rref, *args, **kwargs):
    r"""
    一个帮助函数，用于在给定的 RRef 上调用方法
    """
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    r"""
    一个帮助函数，用于在 RRef 所属的远程端运行方法，并使用 RPC 获取结果
    """
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


class Policy(nn.Module):
    r"""
    从强化学习示例中借用 ``Policy`` 类。复制代码使这两个示例保持独立。
    参见 https://github.com/pytorch/examples/tree/master/reinforcement_learning
    """
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(4, 128)  # 输入维度为4，输出维度为128的线性层
        self.dropout = nn.Dropout(p=0.6)  # 丢弃概率为0.6的 Dropout 层
        self.affine2 = nn.Linear(128, 2)  # 输入维度为128，输出维度为2的线性层

        self.saved_log_probs = []  # 存储的对数概率列表
        self.rewards = []  # 奖励列表

    def forward(self, x):
        x = self.affine1(x)  # 第一线性层
        x = self.dropout(x)  # Dropout 层
        x = F.relu(x)  # ReLU 激活函数
        action_scores = self.affine2(x)  # 第二线性层
        return F.softmax(action_scores, dim=1)  # 在第二维上进行 softmax 操作


class DummyEnv:
    r"""
    一个虚拟环境，实现了 OpenAI gym 接口的必需子集。仅用于在不依赖 gym 的情况下运行此文件中的测试。
    它设计为在一定的最大迭代次数内运行，每一步返回随机状态和奖励。
    """
    def __init__(self, state_dim=4, num_iters=10, reward_threshold=475.0):
        self.state_dim = state_dim  # 状态维度
        self.num_iters = num_iters  # 迭代次数
        self.iter = 0  # 当前迭代步数
        self.reward_threshold = reward_threshold  # 奖励阈值

    def seed(self, manual_seed):
        torch.manual_seed(manual_seed)  # 设置随机种子

    def reset(self):
        self.iter = 0  # 重置迭代步数
        return torch.randn(self.state_dim)  # 返回随机生成的状态

    def step(self, action):
        self.iter += 1  # 增加迭代步数
        state = torch.randn(self.state_dim)  # 随机生成状态
        reward = torch.rand(1).item() * self.reward_threshold  # 随机生成奖励
        done = self.iter >= self.num_iters  # 判断是否完成迭代
        info = {}  # 空字典
        return state, reward, done, info  # 返回状态、奖励、完成标志和信息


class Observer:
    r"""
    观察者具有对其自己环境的独占访问权限。每个观察者
    """
    Captures the state from its environment and sends the state to the agent to
    select an action. Then, the observer applies the action to its environment
    and reports the reward to the agent.
    """
    # 定义 Observer 类，用于监控环境状态并与代理进行交互
    def __init__(self):
        # 获取当前工作进程的 ID
        self.id = rpc.get_worker_info().id
        # 创建虚拟环境对象 DummyEnv
        self.env = DummyEnv()
        # 设定环境的随机种子为 SEED
        self.env.seed(SEED)

    # 运行一个包含 n_steps 步的 episode
    def run_episode(self, agent_rref, n_steps):
        """
        Run one episode of n_steps.
        Arguments:
            agent_rref (RRef): an RRef referencing the agent object.
            n_steps (int): number of steps in this episode
        """
        # 重置环境，并初始化 episode 的总奖励为 0
        state, ep_reward = self.env.reset(), 0

        # 循环执行 n_steps 步
        for step in range(n_steps):
            # 向代理发送当前状态 state，以便代理选择一个动作
            action = _remote_method(Agent.select_action, agent_rref, self.id, state)

            # 将动作应用于环境，并获取奖励和是否结束的标志 done
            state, reward, done, _ = self.env.step(action)

            # 报告奖励给代理，用于训练目的
            _remote_method(Agent.report_reward, agent_rref, self.id, reward)

            # 如果环境标志 done 为 True，则跳出循环，结束该 episode
            if done:
                break
class Agent:
    def __init__(self, world_size):
        # 初始化观察者的远程引用列表
        self.ob_rrefs = []
        # 将代理对象自身包装成远程引用
        self.agent_rref = RRef(self)
        # 存储每个观察者的奖励历史
        self.rewards = {}
        # 存储每个观察者的动作概率对数值
        self.saved_log_probs = {}
        # 创建策略网络
        self.policy = Policy()
        # 初始化优化器，使用Adam优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        # 计算浮点数类型的最小正数
        self.eps = np.finfo(np.float32).eps.item()
        # 运行奖励的初始值设为0
        self.running_reward = 0
        # 获取虚拟环境的奖励阈值
        self.reward_threshold = DummyEnv().reward_threshold
        # 根据世界大小依次获取观察者信息并远程连接观察者
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(worker_name(ob_rank))
            # 将每个观察者的远程引用存储到列表中
            self.ob_rrefs.append(remote(ob_info, Observer))
            # 初始化每个观察者的奖励历史列表
            self.rewards[ob_info.id] = []
            # 初始化每个观察者的动作概率对数值列表
            self.saved_log_probs[ob_info.id] = []

    def select_action(self, ob_id, state):
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/master/reinforcement_learning
        The main difference is that instead of keeping all probs in one list,
        the agent keeps probs in a dictionary, one key per observer.

        NB: no need to enforce thread-safety here as GIL will serialize
        executions.
        """
        # 使用策略网络预测动作概率
        probs = self.policy(state.unsqueeze(0))
        # 根据概率创建一个类别分布对象
        m = Categorical(probs)
        # 从类别分布中采样一个动作
        action = m.sample()
        # 将采样动作的对数概率存储到对应观察者的日志中
        self.saved_log_probs[ob_id].append(m.log_prob(action))
        return action.item()

    def report_reward(self, ob_id, reward):
        r"""
        Observers call this function to report rewards.
        """
        # 记录观察者报告的奖励值
        self.rewards[ob_id].append(reward)

    def run_episode(self, n_steps=0):
        r"""
        Run one episode. The agent will tell each observer to run n_steps.
        """
        futs = []
        for ob_rref in self.ob_rrefs:
            # 异步RPC调用，通知每个观察者运行指定步数的一个回合
            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    _call_method,
                    args=(Observer.run_episode, ob_rref, self.agent_rref, n_steps)
                )
            )

        # 等待直到所有观察者完成本轮回合
        for fut in futs:
            fut.wait()
    def finish_episode(self):
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/master/reinforcement_learning
        The main difference is that it joins all probs and rewards from
        different observers into one list, and uses the minimum observer rewards
        as the reward of the current episode.
        """

        # 初始化变量 R 为 0，probs 和 rewards 为空列表
        R, probs, rewards = 0, [], []

        # 遍历每个观察者的奖励列表，并将其对应的概率和奖励添加到 probs 和 rewards 列表中
        for ob_id in self.rewards:
            probs.extend(self.saved_log_probs[ob_id])
            rewards.extend(self.rewards[ob_id])

        # 计算所有观察者最小奖励作为当前 episode 的奖励
        min_reward = min(sum(self.rewards[ob_id]) for ob_id in self.rewards)
        self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward

        # 清空保存的概率和奖励列表，以备下一次使用
        for ob_id in self.rewards:
            self.rewards[ob_id] = []
            self.saved_log_probs[ob_id] = []

        # 初始化 policy_loss 和 returns 为空列表
        policy_loss, returns = [], []

        # 计算每个时间步的返回值 returns
        for r in rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)

        # 将 returns 转换为 PyTorch 张量，并进行标准化处理
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        # 计算策略损失 policy_loss
        for log_prob, R in zip(probs, returns):
            policy_loss.append(-log_prob * R)

        # 梯度置零，执行反向传播，更新神经网络参数
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # 返回最小观察者奖励作为当前 episode 的奖励
        return min_reward
def run_agent(agent, n_steps):
    # 使用count迭代器生成无限循环，每次迭代代表一个新的episode
    for i_episode in count(1):
        # 调用agent的run_episode方法执行指定步数的训练
        agent.run_episode(n_steps=n_steps)
        # 获取最后一个episode的奖励值
        last_reward = agent.finish_episode()

        # 如果当前的运行奖励值超过了设定的奖励阈值，则任务被视为解决
        if agent.running_reward > agent.reward_threshold:
            # 打印解决信息并结束循环
            print(f"Solved! Running reward is now {agent.running_reward}!")
            break


class ReinforcementLearningRpcTest(RpcAgentTestFixture):
    @dist_init(setup_rpc=False)
    def test_rl_rpc(self):
        if self.rank == 0:
            # Rank 0 是agent角色
            # 初始化RPC，连接到RPC框架
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
            # 创建agent对象，传入整个世界的大小
            agent = Agent(self.world_size)
            # 运行agent的训练函数，总步数按照世界大小来划分
            run_agent(agent, n_steps=int(TOTAL_EPISODE_STEP / (self.world_size - 1)))

            # 确保训练已经运行，对任务是否学习的结果不做过多关注，
            # 因为测试的目的在于检查API调用是否正确
            self.assertGreater(agent.running_reward, 0.0)
        else:
            # 其他rank是观察者，被动等待来自agent的指令
            # 初始化RPC，连接到RPC框架
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
        # 关闭RPC连接
        rpc.shutdown()
```