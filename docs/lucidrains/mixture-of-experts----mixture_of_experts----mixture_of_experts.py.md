# `.\lucidrains\mixture-of-experts\mixture_of_experts\mixture_of_experts.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch 库中导入 nn.functional 模块，并使用别名 F
import torch.nn.functional as F

# 导入 math 库
import math
# 从 inspect 库中导入 isfunction 函数

# 常量定义
MIN_EXPERT_CAPACITY = 4

# 辅助函数

# 默认值函数，如果 val 为 None，则返回 default_val
def default(val, default_val):
    # 如果 default_val 是函数，则调用该函数，否则直接返回 default_val
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

# 将元素 el 转换为元组
def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

# 与张量相关的辅助函数

# 获取张量 t 中最大的值和对应的索引
def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

# 计算张量 t 在指定维度上的累积和，不包括当前位置的值
def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

# 安全的 one-hot 编码函数，避免索引超出范围
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

# 初始化张量 t，使用均匀分布
def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

# 激活函数

# GELU 激活函数类
class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))

# 如果 nn 模块中存在 GELU 函数，则使用该函数，否则使用自定义的 GELU_ 激活函数
GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# 专家类

class Experts(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = GELU):
        super().__init__()

        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.act(hidden)
        out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out

# 下面的代码几乎完全从官方的 tensorflow 版本转录而来，相关论文也是基于此版本编写
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# 门控网络

class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

# 普通的专家混合模型

class MoE(nn.Module):
    # 初始化函数，设置模型参数和属性
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        # 调用父类的初始化函数
        super().__init__()

        # 设置模型的专家数量
        self.num_experts = num_experts

        # 设置门控参数
        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        # 创建门控对象
        self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
        # 创建专家对象
        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        # 设置损失系数
        self.loss_coef = loss_coef

    # 前向传播函数
    def forward(self, inputs, **kwargs):
        # 获取输入的形状信息
        b, n, d, e = *inputs.shape, self.num_experts
        # 获取门控输出和损失
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        # 将输入数据分发给专家
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # 将专家输入数据传递给专家模型
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        # 将专家输出数据合并
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        # 返回输出和损失乘以损失系数
        return output, loss * self.loss_coef
# 定义一个名为 HeirarchicalMoE 的类，表示两级层次混合专家模型
class HeirarchicalMoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = (4, 4),  # 设置专家数量，默认为 (4, 4)
        hidden_dim = None,  # 隐藏层维度，默认为 None
        activation = nn.ReLU,  # 激活函数，默认为 ReLU
        second_policy_train = 'random',  # 第二级门控策略（训练阶段），默认为 'random'
        second_policy_eval = 'random',  # 第二级门控策略（评估阶段），默认为 'random'
        second_threshold_train = 0.2,  # 第二级门控阈值（训练阶段），默认为 0.2
        second_threshold_eval = 0.2,  # 第二级门控阈值（评估阶段），默认为 0.2
        capacity_factor_train = 1.25,  # 容量因子（训练阶段），默认为 1.25
        capacity_factor_eval = 2.,  # 容量因子（评估阶段），默认为 2.0
        loss_coef = 1e-2,  # 损失系数，默认为 0.01
        experts = None):  # 专家模型，默认为 None
        super().__init__()

        assert len(num_experts) == 2, 'only 2 levels of heirarchy for experts allowed for now'  # 断言，只允许两级专家层次
        num_experts_outer, num_experts_inner = num_experts
        self.num_experts_outer = num_experts_outer
        self.num_experts_inner = num_experts_inner

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}

        # 创建外层门控模块和内层门控模块
        self.gate_outer = Top2Gating(dim, num_gates = num_experts_outer, **gating_kwargs)
        self.gate_inner = Top2Gating(dim, num_gates = num_experts_inner, outer_expert_dims = (num_experts_outer,), **gating_kwargs)

        # 创建专家模型
        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, eo, ei = *inputs.shape, self.num_experts_outer, self.num_experts_inner
        dispatch_tensor_outer, combine_tensor_outer, loss_outer = self.gate_outer(inputs)
        expert_inputs_outer = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor_outer)

        # 构建“重要性”张量，用于第二级门控
        importance = combine_tensor_outer.permute(2, 0, 3, 1).sum(dim=-1)
        importance = 0.5 * ((importance > 0.5).float() + (importance > 0.).float())

        dispatch_tensor_inner, combine_tensor_inner, loss_inner = self.gate_inner(expert_inputs_outer, importance = importance)
        expert_inputs = torch.einsum('ebnd,ebnfc->efbcd', expert_inputs_outer, dispatch_tensor_inner)

        # 通过专家模型处理专家输入
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(eo, ei, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        # 合并专家输出
        expert_outputs_outer = torch.einsum('efbcd,ebnfc->ebnd', expert_outputs, combine_tensor_inner)
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs_outer, combine_tensor_outer)
        return output, (loss_outer + loss_inner) * self.loss_coef
```