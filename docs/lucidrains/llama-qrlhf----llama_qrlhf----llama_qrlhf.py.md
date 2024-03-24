# `.\lucidrains\llama-qrlhf\llama_qrlhf\llama_qrlhf.py`

```
import torch
from torch.nn import Module
from torch.utils.data import Dataset
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from einops import rearrange, repeat

from ema_pytorch import EMA

from beartype import beartype
from beartype.typing import Optional

from torchtyping import TensorType

from accelerate import Accelerator

# helper functions

# 检查变量是否存在
def exists(v):
    return v is not None

# tensor helpers

# 从输入的张量中选择指定索引的值
def batch_select_indices(t, indices):
    indices = rearrange(indices, '... -> ... 1')
    selected = t.gather(-1, indices)
    return rearrange(selected, '... 1 -> ...')

# Q functions

# 基于自回归的 Q 学习
def autoregressive_q_learn(
    model:          Module,
    ema_model:      Module,
    states:         TensorType['b', 'n', int],     # 包含提示和生成序列的整个序列
    prompt_len:     TensorType['b', int],          # 前导提示序列的长度
    next_states:    TensorType['b', int],          # 选择的动作成为下一个状态
    rewards:        TensorType['b', 'n', float],   # 奖励可以在最后给出，也可以在中间给出
    eos_id:         Optional[int] = None,          # 从 <eos> 标记 id 计算完成状态
    discount_gamma: float = 0.998                  # 奖励折扣因子，鼓励生成答案的简洁性
) -> TensorType[()]:
    """
    einops

    b - batch
    n - sequence len
    """
    seq_len, device = states.shape[-1], states.device

    # 因为希腊字母的 Unicode 看起来很好

    γ = discount_gamma

    # 获取每个动作的预测 Q 值

    q_pred_all_actions = model(states)
    q_pred = batch_select_indices(q_pred_all_actions, actions)

    # 将下一个状态附加到当前状态，以获取目标 Q

    q_target_input = pack([states[:, 1:], next_state], 'b *')

    # 获取目标 Q

    q_target = ema_model(q_target_input)
    q_target = q_target_all_actions.max(dim = -1).values

    # 第一个完成标志之后的任何内容都将被视为终止状态

    if exists(eos_id):
        done = states == eos_id
        dones = dones.cumsum(dim = -1) > 0
        dones = F.pad(dones, (1, -1), value = False)

        not_terminal = (~dones).float()

        # 奖励不应在终止步骤及之后给出

        rewards = rewards * not_terminal
        q_target = q_target.masked_fill(dones, 0.)

    # 论文的主要贡献是以下逻辑
    # 第 4.1 节 - 公式 1

    # 在没有给出奖励的情况下，时间 t 的 Q 预测是 t + 1 的 max(Q target)

    losses_without_rewards = F.mse_loss(q_pred, q_target, reduction = 'none')

    # 处理给出奖励的时间步骤。���典的贝尔曼方程

    q_target_with_rewards = rewards + γ * q_target

    losses_with_rewards = F.mse_loss(q_pred, q_target_with_rewards, reduction = 'none')

    # 最终损失

    losses = torch.where(
        rewards > 0.,
        losses_with_reward,
        losses_without_rewards
    )

    # 执行掩码平均值
    # 仅考虑从提示的最后一个标记开始的 'q logits' 作为 '动作'

    is_action_mask = torch.arange(seq_len, device = device) > rearrange(prompt_len - 1, 'b -> b 1')
    losses = losses[is_action_mask]

    return losses.mean()

# 保守正则化损失
def conservative_regularization_loss(
    q_values:           TensorType['b', 'n', 'a', float],
    states_and_actions: TensorType['b', 'n', int],
    action_mask:        TensorType['b', 'n', bool],
    reward_min:         float = 0.
) -> TensorType[()]:
    batch, seq_len, num_actions, device = *q_values.shape, q_values.device
    non_dataset_actions = torch.arange(num_actions, device = device) == rearrange(states_and_actions, '... -> ... 1')

    q_values = q_values[~non_dataset_actions]
    q_values = rearrange(q_values, '(b n a) -> b n a', b = batch, n = seq_len)
    # 从Q值中选择动作掩码对应的值
    q_values = q_values[action_mask]

    # 创建一个包含指定值的张量，用于计算奖励的最小值
    reward_min = torch.full((), reward_min, device=device) * seq_len

    # 使用均方误差损失函数计算Q值和奖励最小值之间的损失
    return F.mse_loss(q_values, reward_min)
# 主要类

# 定义 QRLHF 类，继承自 Module 类
class QRLHF(Module):
    # 初始化方法，接受模型、数据集、加速参数和指数移动平均参数
    @beartype
    def __init__(
        self,
        model:   Module,  # 模型对象
        dataset: Dataset,  # 数据集对象
        accelerate_kwargs: dict = dict(),  # 加速参数，默认为空字典
        ema_kwargs: dict = dict(  # 指数移动平均参数，默认包含 beta=0.99
            beta = 0.99
        )
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 将传入的模型赋值给 lm 属性
        self.lm = model
        # 使用传入的模型创建 EMA 对象，并赋值给 lm_target 属性
        self.lm_target = EMA(model, **ema_kwargs)

    # 前向传播方法，抛出未实现错误
    def forward(self):
        raise NotImplementedError
```