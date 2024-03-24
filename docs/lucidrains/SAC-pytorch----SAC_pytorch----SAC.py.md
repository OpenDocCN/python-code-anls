# `.\lucidrains\SAC-pytorch\SAC_pytorch\SAC.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn, einsum, Tensor
from torch import nn, einsum, Tensor
# 从 torch.nn 库中导入 Module, ModuleList
from torch.nn import Module, ModuleList

# 导入 beartype 库
from beartype import beartype
# 从 beartype.typing 中导入 Tuple, List, Optional, Union
from beartype.typing import Tuple, List, Optional, Union

# 导入 einx 库中的 get_at 函数
from einx import get_at
# 导入 einops 库中的 rearrange, repeat, reduce, pack, unpack 函数
from einops import rearrange, repeat, reduce, pack, unpack

# 导入 ema_pytorch 库中的 EMA 类
from ema_pytorch import EMA

# 定义函数 exists，判断变量是否存在
def exists(v):
    return v is not None

# 定义函数 cast_tuple，将输入转换为元组
def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# 定义 MLP 函数，创建简单的多层感知器网络
@beartype
def MLP(
    dim,
    dim_out,
    dim_hiddens: Union[int, Tuple[int, ...]],
    layernorm = False,
    dropout = 0.,
    activation = nn.ReLU
):
    """
    simple mlp for Q and value networks

    following Figure 1 in https://arxiv.org/pdf/2110.02034.pdf for placement of dropouts and layernorm
    however, be aware that Levine in his lecture has ablations that show layernorm alone (without dropout) is sufficient for regularization
    """

    dim_hiddens = cast_tuple(dim_hiddens)

    layers = []

    curr_dim = dim

    for dim_hidden in dim_hiddens:
        layers.append(nn.Linear(curr_dim, dim_hidden))

        layers.append(nn.Dropout(dropout))

        if layernorm:
            layers.append(nn.LayerNorm(dim_hidden))

        layers.append(activation())

        curr_dim = dim_hidden

    # final layer out

    layers.append(nn.Linear(curr_dim, dim_out))

    return nn.Sequential(*layers)

# 定义 Actor 类，用于创建 Actor 神经网络模型
class Actor(Module):
    def __init__(
        self,
        *,
        dim_state,
        num_cont_actions,
        dim_hiddens: Tuple[int, ...] = tuple(),
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps

        self.to_cont_actions = MLP(
            dim_state,
            dim_hiddens = dim_hiddens,
            dim_out = num_cont_actions * 2
        )

    def forward(
        self,
        state,
        sample = False
    ):
        """
        einops notation
        n - num actions
        ms - mu sigma
        """

        out = self.to_cont_actions(state)
        mu, sigma = rearrange(out, '... (n ms) -> ms ... n', ms = 2)

        sigma = sigma.sigmoid().clamp(min = self.eps)

        if not sample:
            return mu, sigma

        return mu + sigma * torch.randn_like(sigma)

# 定义 Critic 类，用于创建 Critic 神经网络模型
class Critic(Module):
    @beartype
    def __init__(
        self,
        *,
        dim_state,
        num_continuous_actions,
        dim_hiddens: Tuple[int, ...] = tuple(),
        layernorm = False,
        dropout = 0.
    ):
        super().__init__()

        self.to_q = MLP(
            dim_state + num_continuous_actions,
            dim_out = 1,
            dim_hiddens = dim_hiddens,
            layernorm = layernorm,
            dropout = dropout
        )

    def forward(
        self,
        state,
        actions
    ):
        state_actions, _ = pack([state, actions], 'b *')

        q_values = self.to_q(state_actions)
        q_values = rearrange('b 1 -> b')

        return q_values

# 定义 ValueNetwork 类，用于创建值网络模型
class ValueNetwork(Module):
    @beartype
    def __init__(
        self,
        *,
        dim_state,
        dim_hiddens: Tuple[int, ...] = tuple()
    ):
        super().__init__()

        self.to_values = MLP(
            dim_state,
            dim_out= 1,
            dim_hiddens = dim_hiddens
        )

    def forward(
        self,
        states
    ):
        values = self.to_values(states)
        values = rearrange(values, 'b 1 -> b')
        return values

# 定义 SAC 类，用于创建 SAC 神经网络模型
class SAC(Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return x
```