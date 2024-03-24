# `.\lucidrains\st-moe-pytorch\st_moe_pytorch\st_moe_pytorch.py`

```
# 导入必要的库
from functools import partial
from collections import namedtuple
from typing import Optional, Tuple, Union

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum
import torch.nn.functional as F

# 导入额外的库
from beartype import beartype
from einops import rearrange, repeat, reduce, pack, unpack
from colt5_attention import topk as maybe_differentiable_topk
import torch.distributed as dist
from st_moe_pytorch.distributed import (
    AllGather,
    split_by_rank,
    gather_sizes,
    pad_dim_to,
    has_only_one_value
)

# 常量定义
MIN_EXPERT_CAPACITY = 4
MixtureOfExpertsReturn = namedtuple('MixtureOfExpertsReturn', [
    'outputs',
    'total_aux_loss',
    'balance_loss',
    'router_z_loss'
])

# 辅助函数定义

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, default):
    if exists(val):
        return val
    return default() if callable(default) else default

# 判断一个数是否可以被另一个数整除
def divisible_by(num, den):
    return (num % den) == 0

# 将一个数均匀分成多个部分
def chunk_num(num, chunks):
    num_per_chunk, remainder = divmod(num, chunks)
    out = []
    for i in range(chunks):
        n = num_per_chunk
        out.append(n + int(i < remainder))
    return out

# 将一个张量按照指定模式打包
def pack_one(t, pattern):
    return pack([t], pattern)

# 将一个打包的张量按照指定模式解包
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 将元素转换为元组
def cast_tuple(el, len = 1):
    return el if isinstance(el, tuple) else ((el,) * len)

# 创建一个序列模块
def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# 与张量相关的辅助函数

# 计算张量的累积和（不包括当前元素）
def cumsum_exclusive(t, dim = -3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

# 计算张量的对数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 生成古贝尔噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 安全的独热编码函数，避免索引超出范围
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]

# RMS归一化

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# 专家类
# 最佳表现是在门控后使用乘法偏置的ff geglu

class GEGLU(Module):
    def __init__(
        self,
        dim,
        mult_bias = True
    ):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else 1.

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x * self.mult_bias

class Expert(Module):
    def __init__(
        self,
        dim,
        hidden_mult = 4,
        mult_bias = True,
        prenorm = False
    ):
        super().__init__()
        dim_hidden = int(dim * hidden_mult * 2 / 3)

        self.net = Sequential(
            RMSNorm(dim) if prenorm else None,
            nn.Linear(dim, dim_hidden * 2),
            GEGLU(dim_hidden, mult_bias = mult_bias),
            nn.Linear(dim_hidden, dim)
        )

        self.apply(self.init_)

    def init_(self, module):
        if isinstance(module, nn.Linear):
            dim = module.weight.shape[0]
            std = dim ** -0.5

            module.weight.data.uniform_(-std, std)
            module.bias.data.uniform_(-std, std)

    def forward(self, x):
        return self.net(x)

class Experts(nn.Module):
    def __init__(
        self,
        experts,
        is_distributed = None,
        allow_var_seq_len = False # 是否处理可变序列长度
    # 初始化函数，设置专家数量和专家模块列表
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 获取专家数量并初始化专家模块列表
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)

        # 分布式相关设置

        # 是否处于分布式环境
        self.is_distributed = is_distributed
        # 如果未指定是否分布式，则根据当前环境判断
        if not exists(self.is_distributed):
            self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        # 创建 AllGather 对象
        self.all_gather = AllGather()

        # 是否允许变长序列长度
        self.allow_var_seq_len = allow_var_seq_len

        # 设备跟踪器，需要手动将未使用的专家移动到 CPU 上

        # 注册缓冲区，用于跟踪设备
        self.register_buffer('dummy', torch.ones(1), persistent = False)

    # 设备属性，返回 dummy 的设备
    @property
    def device(self):
        return self.dummy.device

    # 将除了指定专家之外的所有专家移动到 CPU
    def all_experts_to_cpu_besides(self, selection):
        # 根据选择的专家索引或切片获取专家列表
        if isinstance(selection, int):
            experts = [self.experts[selection]]
        if isinstance(selection, slice):
            experts = self.experts[selection]
        else:
            experts = selection

        # 将专家列表转换为集合
        experts_set = set(experts)

        # 遍历所有专家，根据是否在选择的专家列表中决定设备
        for expert in self.experts:
            device = self.device if expert in experts_set else 'cpu'
            expert.to(device)

    # 前向传播函数
    def forward(
        self,
        x,
        is_distributed = None
# 定义一个名为 TopNGating 的类，继承自 Module 类
class TopNGating(Module):

    # 初始化方法，接受多个参数
    @beartype
    def __init__(
        self,
        dim,  # 维度
        num_gates,  # 门的数量
        eps = 1e-9,  # 微小值
        top_n = 2,  # 顶部 N 个
        threshold_train: Union[float, Tuple[float, ...]] = 0.2,  # 训练阈值
        threshold_eval: Union[float, Tuple[float, ...]] = 0.2,  # 评估阈值
        capacity_factor_train = 1.25,  # 训练容量因子
        capacity_factor_eval = 2.,  # 评估容量因子
        straight_through_dispatch_tensor = True,  # 直通分发张量
        differentiable_topk = False,  # 可微分的 topk
        differentiable_topk_fused = True  # 融合的可微分 topk
    ):
        super().__init__()  # 调用父类的初始化方法
        self.eps = eps  # 将 eps 赋值给实例变量
        self.num_gates = num_gates  # 将 num_gates 赋值给实例变量
        self.to_gates = nn.Linear(dim, num_gates, bias = False)  # 创建一个线性层

        self.differentiable_topk = differentiable_topk  # 将 differentiable_topk 赋值给实例变量

        # 部分函数应用，使用 maybe_differentiable_topk 函数
        self.topk = partial(
            maybe_differentiable_topk,
            non_differentiable = not differentiable_topk,
            fused = differentiable_topk_fused  # 默认情况下使用 Triton 融合坐标下降
        )

        assert top_n >= 2, 'must be 2 or more experts'  # 断言，确保 top_n 大于等于 2
        self.top_n = top_n  # 将 top_n 赋值给实例变量
        top_n_minus_1 = top_n - 1  # 计算 top_n 减 1

        threshold_train = cast_tuple(threshold_train, top_n_minus_1)  # 将 threshold_train 转换为元组
        threshold_eval = cast_tuple(threshold_eval, top_n_minus_1)  # 将 threshold_eval 转换为元组

        assert len(threshold_train) == len(threshold_eval) == top_n_minus_1  # 断言，确保长度相等

        # 将 threshold_train 和 threshold_eval 转换为张量，并注册为缓冲区
        self.register_buffer('threshold_train', torch.tensor([eps, *threshold_train]))
        self.register_buffer('threshold_eval', torch.tensor([eps, *threshold_eval]))

        self.capacity_factor_train = capacity_factor_train  # 将 capacity_factor_train 赋值给实例变量
        self.capacity_factor_eval = capacity_factor_eval  # 将 capacity_factor_eval 赋值给实例变量

        self.straight_through_dispatch_tensor = straight_through_dispatch_tensor  # 将 straight_through_dispatch_tensor 赋值给实例变量
        # 将零值注册为缓冲区
        self.register_buffer('zero', torch.zeros((1,)), persistent = False)

    # 前向传播方法
    def forward(
        self,
        x,  # 输入张量
        noise_gates = False,  # 是否添加噪音到门
        noise_mult = 1.  # 噪音倍数



# 定义一个名为 MoE 的类，继承自 Module 类
class MoE(Module):

    # 初始化方法，接受多个参数
    @beartype
    def __init__(self,
        dim,  # 维度
        num_experts = 16,  # 专家数量
        expert_hidden_mult = 4,  # 专家隐藏倍数
        threshold_train = 0.2,  # 训练阈值
        threshold_eval = 0.2,  # 评估阈值
        capacity_factor_train = 1.25,  # 训练容量因子
        capacity_factor_eval = 2.,  # 评估容量因子
        gating_top_n = 2,  # 门的顶部 N 个
        balance_loss_coef = 1e-2,  # 平衡损失系数
        router_z_loss_coef = 1e-3,  # 路由器 z 损失系数
        experts: Optional[Module] = None,  # 专家模块
        straight_through_dispatch_tensor = True,  # 直通分发张量
        differentiable_topk = False,  # 可微分的 topk
        differentiable_topk_fused = True,  # 融合的可微分 topk
        is_distributed = None,  # 是否分布式
        allow_var_seq_len = False  # 是否允许可变序列长度
    ):
        super().__init__()  # 调用父类的初始化方法
        self.dim = dim  # 将 dim 赋值给实例变量
        self.num_experts = num_experts  # 将 num_experts 赋值给实例变量

        # 创建一个 TopNGating 实例
        self.gate = TopNGating(
            dim,
            top_n = gating_top_n,
            num_gates = num_experts,
            straight_through_dispatch_tensor = straight_through_dispatch_tensor,
            differentiable_topk = differentiable_topk,
            threshold_train = threshold_train,
            threshold_eval = threshold_eval,
            capacity_factor_train = capacity_factor_train,
            capacity_factor_eval = capacity_factor_eval
        )

        # 如果 experts 为 None，则创建一个专家列表
        experts = default(experts, lambda: [Expert(dim = dim, hidden_mult = expert_hidden_mult) for _ in range(num_experts)])

        # 创建一个 Experts 实例
        self.experts = Experts(
            experts,
            is_distributed = is_distributed,
            allow_var_seq_len = allow_var_seq_len
        )

        self.balance_loss_coef = balance_loss_coef  # 将 balance_loss_coef 赋值给实例变量
        self.router_z_loss_coef = router_z_loss_coef  # 将 router_z_loss_coef 赋值给实例变量

    # 前向传播方法
    def forward(
        self,
        x,  # 输入张量
        noise_gates = False,  # 是否添加噪音到门
        noise_mult = 1.  # 噪音倍数
        dispatch_tensor, combine_tensor, balance_loss, router_z_loss = self.gate(x, noise_gates = noise_gates, noise_mult = noise_mult)
        # 调用gate方法，获取dispatch_tensor、combine_tensor、balance_loss和router_z_loss

        # dispatch
        expert_inputs = einsum('b n d, b n e c -> b e c d', x, dispatch_tensor)
        # 使用einsum函数将输入x和dispatch_tensor进行张量乘法，得到expert_inputs

        # feed the expert inputs through the experts.
        expert_outputs = self.experts(expert_inputs)
        # 将expert_inputs传递给experts方法，得到expert_outputs

        # combine
        output = einsum('b e c d, b n e c -> b n d', expert_outputs, combine_tensor)
        # 使用einsum函数将expert_outputs和combine_tensor进行张量乘法，得到output

        # losses
        weighted_balance_loss = balance_loss * self.balance_loss_coef
        weighted_router_z_loss = router_z_loss * self.router_z_loss_coef
        # 计算加权的balance_loss和router_z_loss

        # combine the losses
        total_aux_loss = weighted_balance_loss + weighted_router_z_loss
        # 将加权的balance_loss和router_z_loss相加得到总的辅助损失

        return MixtureOfExpertsReturn(output, total_aux_loss, balance_loss, router_z_loss)
        # 返回MixtureOfExpertsReturn对象，包含output、total_aux_loss、balance_loss和router_z_loss
# 定义一个稀疏的 Mixture of Experts（MoE）块
# 特别是，他们发现在前后添加一个前馈网络可以极大地稳定训练并改善结果

class SparseMoEBlock(Module):

    @beartype
    def __init__(
        self,
        moe: MoE,
        *,
        add_ff_before = False,
        add_ff_after = True
    ):
        super().__init__()
        dim = moe.dim

        # 初始化 MoE 模块和 RMSNorm 模块
        self.moe = moe
        self.moe_prenorm = RMSNorm(dim)

        # 根据参数决定是否添加前馈网络
        self.ff_before = Expert(dim, prenorm = True) if add_ff_before else None
        self.ff_after = Expert(dim, prenorm = True) if add_ff_after else None

    def forward(
        self,
        x,
        noise_gates = False,
        noise_mult = 1.
    ):

        # 前馈网络之前的处理

        if exists(self.ff_before):
            x = self.ff_before(x) + x

        # 专家混合层

        residual = x

        # 调用 MoE 模块进行前向传播
        moe_out, total_aux_loss, balance_loss, router_z_loss = self.moe(self.moe_prenorm(x), noise_gates = noise_gates, noise_mult = noise_mult)

        x = moe_out + residual

        # 前馈网络之后的处理

        if exists(self.ff_after):
            x = self.ff_after(x) + x

        # 返回 MoE 模块的输出结果和相关损失
        return MixtureOfExpertsReturn(x, total_aux_loss, balance_loss, router_z_loss)
```