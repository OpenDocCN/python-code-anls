# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\magvit2_pytorch.py`

```py
# 导入所需的模块和库
import copy  # 导入复制对象的库
from pathlib import Path  # 导入路径操作库
from math import log2, ceil, sqrt  # 导入数学相关函数
from functools import wraps, partial  # 导入函数式编程工具

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
from torch.cuda.amp import autocast  # 导入自动混合精度训练
from torch import nn, einsum, Tensor  # 导入神经网络模块、爱因斯坦求和和张量类
from torch.nn import Module, ModuleList  # 导入 PyTorch 模块和模块列表
from torch.autograd import grad as torch_grad  # 导入自动梯度计算

import torchvision  # 导入计算机视觉库
from torchvision.models import VGG16_Weights  # 导入 VGG16 模型权重

from collections import namedtuple  # 导入命名元组

# from vector_quantize_pytorch import LFQ, FSQ  # 注释掉的导入，表示不再使用的模块
from .regularizers.finite_scalar_quantization import FSQ  # 从正则化模块导入有限标量量化
from .regularizers.lookup_free_quantization import LFQ  # 从正则化模块导入查找自由量化

from einops import rearrange, repeat, reduce, pack, unpack  # 导入 einops 的操作函数
from einops.layers.torch import Rearrange  # 导入 PyTorch 特有的 einops 重排列层

from beartype import beartype  # 导入类型检查库
from beartype.typing import Union, Tuple, Optional, List  # 导入类型注解

from magvit2_pytorch.attend import Attend  # 从 magvit2 导入注意力模块
from magvit2_pytorch.version import __version__  # 导入当前版本信息

from gateloop_transformer import SimpleGateLoopLayer  # 从 gateloop_transformer 导入简单门控循环层

from taylor_series_linear_attention import TaylorSeriesLinearAttn  # 从泰勒级数线性注意力导入相应模块

from kornia.filters import filter3d  # 从 Kornia 导入三维滤波器

import pickle  # 导入序列化和反序列化模块

# 辅助函数


def exists(v):  # 检查变量是否存在
    return v is not None  # 如果变量不为 None，返回 True


def default(v, d):  # 返回默认值
    return v if exists(v) else d  # 如果 v 存在则返回 v，否则返回 d


def safe_get_index(it, ind, default=None):  # 安全获取索引
    if ind < len(it):  # 检查索引是否在范围内
        return it[ind]  # 返回指定索引的元素
    return default  # 如果索引超出范围，则返回默认值


def pair(t):  # 将输入转换为元组
    return t if isinstance(t, tuple) else (t, t)  # 如果 t 是元组则返回，否则返回重复的元组


def identity(t, *args, **kwargs):  # 身份函数
    return t  # 返回原输入


def divisible_by(num, den):  # 检查 num 是否能被 den 整除
    return (num % den) == 0  # 返回除法余数是否为零


def pack_one(t, pattern):  # 打包一个张量
    return pack([t], pattern)  # 将张量打包成指定模式


def unpack_one(t, ps, pattern):  # 解包一个张量
    return unpack(t, ps, pattern)[0]  # 将张量解包并返回第一个元素


def append_dims(t, ndims: int):  # 向张量追加维度
    return t.reshape(*t.shape, *((1,) * ndims))  # 调整张量形状，追加指定数量的维度


def is_odd(n):  # 检查数字是否为奇数
    return not divisible_by(n, 2)  # 使用可整除函数检查


def maybe_del_attr_(o, attr):  # 有条件地删除对象属性
    if hasattr(o, attr):  # 检查对象是否具有该属性
        delattr(o, attr)  # 删除该属性


def cast_tuple(t, length=1):  # 将输入转换为元组
    return t if isinstance(t, tuple) else ((t,) * length)  # 如果 t 是元组则返回，否则创建指定长度的元组


# 张量辅助函数


def l2norm(t):  # 计算 L2 范数
    return F.normalize(t, dim=-1, p=2)  # 在最后一个维度上归一化


def pad_at_dim(t, pad, dim=-1, value=0.0):  # 在指定维度上填充张量
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)  # 计算从右边开始的维度索引
    zeros = (0, 0) * dims_from_right  # 创建零填充的元组
    return F.pad(t, (*zeros, *pad), value=value)  # 填充张量并返回


def pick_video_frame(video, frame_indices):  # 从视频中选择帧
    batch, device = video.shape[0], video.device  # 获取批次大小和设备
    video = rearrange(video, "b c f ... -> b f c ...")  # 调整视频张量的形状
    batch_indices = torch.arange(batch, device=device)  # 创建批次索引
    batch_indices = rearrange(batch_indices, "b -> b 1")  # 调整批次索引的形状
    images = video[batch_indices, frame_indices]  # 根据索引选择图像
    images = rearrange(images, "b 1 c ... -> b c ...")  # 调整图像张量的形状
    return images  # 返回选择的图像


# GAN 相关函数


def gradient_penalty(images, output):  # 计算梯度惩罚
    batch_size = images.shape[0]  # 获取批次大小

    gradients = torch_grad(  # 计算输出关于输入的梯度
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),  # 设置梯度输出
        create_graph=True,  # 创建计算图以便后续计算
        retain_graph=True,  # 保留计算图以便多次使用
        only_inputs=True,  # 仅对输入计算梯度
    )[0]  # 只取第一个返回值

    gradients = rearrange(gradients, "b ... -> b (...)")  # 调整梯度张量的形状
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # 计算并返回平均梯度惩罚


def leaky_relu(p=0.1):  # 创建带泄漏的 ReLU 激活函数
    return nn.LeakyReLU(p)  # 返回带泄漏的 ReLU 实例
# 定义一个函数来计算铰链判别损失
def hinge_discr_loss(fake, real):
    # 计算并返回铰链损失的均值
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


# 定义一个函数来计算铰链生成损失
def hinge_gen_loss(fake):
    # 返回生成样本的负均值作为损失
    return -fake.mean()


# 自动混合精度上下文装饰器，禁用混合精度
@autocast(enabled=False)
# 类型检查装饰器
@beartype
# 定义一个函数计算损失相对于某一层的梯度
def grad_layer_wrt_loss(loss: Tensor, layer: nn.Parameter):
    # 使用反向传播计算损失相对于层参数的梯度，并返回其张量
    return torch_grad(outputs=loss, inputs=layer, grad_outputs=torch.ones_like(loss), retain_graph=True)[0].detach()


# 装饰器帮助函数


# 定义一个装饰器，用于移除 VGG 属性
def remove_vgg(fn):
    # 装饰器内部函数
    @wraps(fn)
    def inner(self, *args, **kwargs):
        # 检查对象是否有 VGG 属性
        has_vgg = hasattr(self, "vgg")
        if has_vgg:
            # 保存 VGG 属性并删除它
            vgg = self.vgg
            delattr(self, "vgg")

        # 调用原始函数并获取输出
        out = fn(self, *args, **kwargs)

        # 如果有 VGG 属性，则将其恢复
        if has_vgg:
            self.vgg = vgg

        # 返回函数输出
        return out

    return inner


# 帮助类


# 定义一个顺序模块的构造函数
def Sequential(*modules):
    # 过滤出有效的模块
    modules = [*filter(exists, modules)]

    # 如果没有有效模块，则返回身份映射
    if len(modules) == 0:
        return nn.Identity()

    # 返回一个顺序容器
    return nn.Sequential(*modules)


# 定义残差模块类
class Residual(Module):
    # 类型检查装饰器
    @beartype
    def __init__(self, fn: Module):
        # 调用父类构造函数
        super().__init__()
        # 保存传入的模块函数
        self.fn = fn

    # 前向传播函数
    def forward(self, x, **kwargs):
        # 返回残差输出，即函数输出加上输入
        return self.fn(x, **kwargs) + x


# 定义一个用于张量操作的类，将张量转换为时间序列格式
class ToTimeSequence(Module):
    # 类型检查装饰器
    @beartype
    def __init__(self, fn: Module):
        # 调用父类构造函数
        super().__init__()
        # 保存传入的模块函数
        self.fn = fn

    # 前向传播函数
    def forward(self, x, **kwargs):
        # 重新排列输入张量的维度
        x = rearrange(x, "b c f ... -> b ... f c")
        # 将张量打包以便处理
        x, ps = pack_one(x, "* n c")

        # 通过模块函数处理张量
        o = self.fn(x, **kwargs)

        # 解包处理后的张量
        o = unpack_one(o, ps, "* n c")
        # 重新排列输出张量的维度
        return rearrange(o, "b ... f c -> b c f ...")


# 定义一个 squeeze-excite 模块类
class SqueezeExcite(Module):
    # 全局上下文网络 - 类似注意力机制的 squeeze-excite 变体
    def __init__(self, dim, *, dim_out=None, dim_hidden_min=16, init_bias=-10):
        # 调用父类构造函数
        super().__init__()
        # 如果未指定输出维度，则默认为输入维度
        dim_out = default(dim_out, dim)

        # 创建卷积层用于计算键
        self.to_k = nn.Conv2d(dim, 1, 1)
        # 计算隐藏层维度
        dim_hidden = max(dim_hidden_min, dim_out // 2)

        # 定义网络结构
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_hidden, 1), nn.LeakyReLU(0.1), nn.Conv2d(dim_hidden, dim_out, 1), nn.Sigmoid()
        )

        # 初始化网络的权重和偏置
        nn.init.zeros_(self.net[-2].weight)
        nn.init.constant_(self.net[-2].bias, init_bias)

    # 前向传播函数
    def forward(self, x):
        # 保存原始输入和批量大小
        orig_input, batch = x, x.shape[0]
        # 检查输入是否为视频格式
        is_video = x.ndim == 5

        # 如果是视频格式，重新排列输入张量的维度
        if is_video:
            x = rearrange(x, "b c f h w -> (b f) c h w")

        # 计算上下文信息
        context = self.to_k(x)

        # 重新排列上下文信息，并应用 softmax
        context = rearrange(context, "b c h w -> b c (h w)").softmax(dim=-1)
        # 展平输入张量
        spatial_flattened_input = rearrange(x, "b c h w -> b c (h w)")

        # 计算输出
        out = einsum("b i n, b c n -> b c i", context, spatial_flattened_input)
        # 重新排列输出
        out = rearrange(out, "... -> ... 1")
        # 通过网络计算门控值
        gates = self.net(out)

        # 如果是视频格式，重新排列门控值的维度
        if is_video:
            gates = rearrange(gates, "(b f) c h w -> b c f h w", b=batch)

        # 返回门控值与原始输入的乘积
        return gates * orig_input


# 定义一个 token shifting 模块类
class TokenShift(Module):
    # 类型检查装饰器
    @beartype
    def __init__(self, fn: Module):
        # 调用父类构造函数
        super().__init__()
        # 保存传入的模块函数
        self.fn = fn
    # 定义前向传播函数，接受输入 x 和其他可选参数
        def forward(self, x, **kwargs):
            # 将输入 x 在第 1 维上分成两部分，分别赋值给 x 和 x_shift
            x, x_shift = x.chunk(2, dim=1)
            # 在时间维度上对 x_shift 进行填充，使其适应后续操作
            x_shift = pad_at_dim(x_shift, (1, -1), dim=2)  # shift time dimension
            # 将 x 和填充后的 x_shift 在第 1 维上拼接
            x = torch.cat((x, x_shift), dim=1)
            # 调用 self.fn 函数处理拼接后的 x，并传入其他参数
            return self.fn(x, **kwargs)
# rmsnorm

# 定义 RMSNorm 类，继承自 Module 类
class RMSNorm(Module):
    # 初始化函数，接受多个参数
    def __init__(self, dim, channel_first=False, images=False, bias=False):
        # 调用父类的初始化方法
        super().__init__()
        # 根据是否为图像确定可广播维度
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        # 根据通道顺序和维度定义形状
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        # 保存通道顺序
        self.channel_first = channel_first
        # 计算缩放因子
        self.scale = dim**0.5
        # 定义可学习的参数 gamma，初始化为全1
        self.gamma = nn.Parameter(torch.ones(shape))
        # 定义偏置参数，若 bias 为真则初始化为全0，否则设为0.0
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    # 前向传播函数
    def forward(self, x):
        # 归一化输入 x，应用缩放因子和 gamma，然后加上偏置
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


# 定义自适应 RMSNorm 类，继承自 Module 类
class AdaptiveRMSNorm(Module):
    # 初始化函数，接受多个参数
    def __init__(self, dim, *, dim_cond, channel_first=False, images=False, bias=False):
        # 调用父类的初始化方法
        super().__init__()
        # 根据是否为图像确定可广播维度
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        # 根据通道顺序和维度定义形状
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        # 保存条件维度和通道顺序
        self.dim_cond = dim_cond
        self.channel_first = channel_first
        # 计算缩放因子
        self.scale = dim**0.5

        # 定义线性层用于计算 gamma
        self.to_gamma = nn.Linear(dim_cond, dim)
        # 若需要偏置，则定义相应的线性层
        self.to_bias = nn.Linear(dim_cond, dim) if bias else None

        # 初始化 gamma 的权重为零，偏置为一
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)

        # 若需要偏置，则初始化偏置层的权重和偏置为零
        if bias:
            nn.init.zeros_(self.to_bias.weight)
            nn.init.zeros_(self.to_bias.bias)

    # 前向传播函数，带有条件输入
    @beartype
    def forward(self, x: Tensor, *, cond: Tensor):
        # 获取输入的批大小
        batch = x.shape[0]
        # 确保条件张量的形状与批大小匹配
        assert cond.shape == (batch, self.dim_cond)

        # 计算 gamma 值
        gamma = self.to_gamma(cond)

        # 初始化偏置为 0
        bias = 0.0
        # 若存在偏置层，则计算偏置
        if exists(self.to_bias):
            bias = self.to_bias(cond)

        # 若通道顺序为先，则扩展 gamma 的维度
        if self.channel_first:
            gamma = append_dims(gamma, x.ndim - 2)

            # 若存在偏置层，则扩展偏置的维度
            if exists(self.to_bias):
                bias = append_dims(bias, x.ndim - 2)

        # 归一化输入 x，应用缩放因子和 gamma，然后加上偏置
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * gamma + bias


# attention

# 定义 Attention 类，继承自 Module 类
class Attention(Module):
    # 初始化函数，接受多个参数
    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_cond: Optional[int] = None,
        causal=False,
        dim_head=32,
        heads=8,
        flash=False,
        dropout=0.0,
        num_memory_kv=4,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 计算内部维度
        dim_inner = dim_head * heads

        # 检查是否需要条件维度
        self.need_cond = exists(dim_cond)

        # 根据是否需要条件维度选择归一化方式
        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond=dim_cond)
        else:
            self.norm = RMSNorm(dim)

        # 定义线性层以计算查询、键、值
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False), 
            # 重排张量维度
            Rearrange("b n (qkv h d) -> qkv b h n d", qkv=3, h=heads)
        )

        # 确保记忆键值对数量大于零
        assert num_memory_kv > 0
        # 定义可学习的参数用于存储记忆键值对
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_memory_kv, dim_head))

        # 定义注意力机制
        self.attend = Attend(causal=causal, dropout=dropout, flash=flash)

        # 定义输出层
        self.to_out = nn.Sequential(
            # 重排张量维度
            Rearrange("b h n d -> b n (h d)"), 
            nn.Linear(dim_inner, dim, bias=False)
        )

    # 继续定义其他方法...
    # 前向传播函数，接受输入张量 x 和可选的掩码与条件张量
    def forward(self, x, mask: Optional[Tensor] = None, cond: Optional[Tensor] = None):
        # 根据是否需要条件，构建条件参数字典
        maybe_cond_kwargs = dict(cond=cond) if self.need_cond else dict()
    
        # 对输入 x 进行归一化处理，可能包含条件参数
        x = self.norm(x, **maybe_cond_kwargs)
    
        # 将输入 x 转换为查询（q）、键（k）和值（v）三种张量
        q, k, v = self.to_qkv(x)
    
        # 将记忆中的键（mk）和值（mv）重复以匹配批次大小，并保持原有形状
        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=q.shape[0]), self.mem_kv)
        # 将新的键张量和记忆中的键张量沿最后一个维度拼接
        k = torch.cat((mk, k), dim=-2)
        # 将新的值张量和记忆中的值张量沿最后一个维度拼接
        v = torch.cat((mv, v), dim=-2)
    
        # 根据查询、键和值以及掩码计算注意力输出
        out = self.attend(q, k, v, mask=mask)
        # 将注意力输出转换为最终输出格式
        return self.to_out(out)
# 定义一个线性注意力类，继承自 Module
class LinearAttention(Module):
    """
    使用特定的线性注意力，参考 https://arxiv.org/abs/2106.09681
    """

    @beartype
    # 初始化方法，接收多个参数
    def __init__(self, *, dim, dim_cond: Optional[int] = None, dim_head=8, heads=8, dropout=0.0):
        # 调用父类的初始化方法
        super().__init__()
        # 计算内部维度
        dim_inner = dim_head * heads

        # 检查条件维度是否存在
        self.need_cond = exists(dim_cond)

        # 如果需要条件，则使用自适应 RMSNorm
        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond=dim_cond)
        # 否则使用 RMSNorm
        else:
            self.norm = RMSNorm(dim)

        # 创建 TaylorSeriesLinearAttn 对象
        self.attn = TaylorSeriesLinearAttn(dim=dim, dim_head=dim_head, heads=heads)

    # 前向传播方法
    def forward(self, x, cond: Optional[Tensor] = None):
        # 根据是否需要条件来设置可选参数
        maybe_cond_kwargs = dict(cond=cond) if self.need_cond else dict()

        # 通过规范化处理输入数据
        x = self.norm(x, **maybe_cond_kwargs)

        # 返回注意力计算的结果
        return self.attn(x)


# 定义一个线性空间注意力类，继承自 LinearAttention
class LinearSpaceAttention(LinearAttention):
    # 前向传播方法
    def forward(self, x, *args, **kwargs):
        # 重新排列张量维度
        x = rearrange(x, "b c ... h w -> b ... h w c")
        # 将张量打包成一个新的格式
        x, batch_ps = pack_one(x, "* h w c")
        # 再次打包张量以准备进行注意力计算
        x, seq_ps = pack_one(x, "b * c")

        # 调用父类的前向传播方法
        x = super().forward(x, *args, **kwargs)

        # 解包张量以恢复原始格式
        x = unpack_one(x, seq_ps, "b * c")
        x = unpack_one(x, batch_ps, "* h w c")
        # 重新排列输出张量维度
        return rearrange(x, "b ... h w c -> b c ... h w")


# 定义空间注意力类，继承自 Attention
class SpaceAttention(Attention):
    # 前向传播方法
    def forward(self, x, *args, **kwargs):
        # 重新排列张量维度
        x = rearrange(x, "b c t h w -> b t h w c")
        # 将张量打包以便处理
        x, batch_ps = pack_one(x, "* h w c")
        # 再次打包以准备进行注意力计算
        x, seq_ps = pack_one(x, "b * c")

        # 调用父类的前向传播方法
        x = super().forward(x, *args, **kwargs)

        # 解包张量以恢复原始格式
        x = unpack_one(x, seq_ps, "b * c")
        x = unpack_one(x, batch_ps, "* h w c")
        # 重新排列输出张量维度
        return rearrange(x, "b t h w c -> b c t h w")


# 定义时间注意力类，继承自 Attention
class TimeAttention(Attention):
    # 前向传播方法
    def forward(self, x, *args, **kwargs):
        # 重新排列张量维度
        x = rearrange(x, "b c t h w -> b h w t c")
        # 将张量打包以便处理
        x, batch_ps = pack_one(x, "* t c")

        # 调用父类的前向传播方法
        x = super().forward(x, *args, **kwargs)

        # 解包张量以恢复原始格式
        x = unpack_one(x, batch_ps, "* t c")
        # 重新排列输出张量维度
        return rearrange(x, "b h w t c -> b c t h w")


# 定义 GEGLU 类，继承自 Module
class GEGLU(Module):
    # 前向传播方法
    def forward(self, x):
        # 将输入张量分成两部分：x 和 gate
        x, gate = x.chunk(2, dim=1)
        # 返回激活函数 gelu 的结果乘以 x
        return F.gelu(gate) * x


# 定义前馈神经网络类，继承自 Module
class FeedForward(Module):
    @beartype
    # 初始化方法，接收多个参数
    def __init__(self, dim, *, dim_cond: Optional[int] = None, mult=4, images=False):
        # 调用父类的初始化方法
        super().__init__()
        # 根据 images 参数选择卷积类型
        conv_klass = nn.Conv2d if images else nn.Conv3d

        # 根据条件维度选择 RMSNorm 类型
        rmsnorm_klass = RMSNorm if not exists(dim_cond) else partial(AdaptiveRMSNorm, dim_cond=dim_cond)

        # 创建适应性规范化类的部分函数
        maybe_adaptive_norm_klass = partial(rmsnorm_klass, channel_first=True, images=images)

        # 计算内部维度
        dim_inner = int(dim * mult * 2 / 3)

        # 创建规范化层
        self.norm = maybe_adaptive_norm_klass(dim)

        # 构建前馈神经网络，包括卷积层和 GEGLU 激活
        self.net = Sequential(conv_klass(dim, dim_inner * 2, 1), GEGLU(), conv_klass(dim_inner, dim, 1))

    @beartype
    # 前向传播方法
    def forward(self, x: Tensor, *, cond: Optional[Tensor] = None):
        # 根据条件是否存在设置可选参数
        maybe_cond_kwargs = dict(cond=cond) if exists(cond) else dict()

        # 通过规范化处理输入数据
        x = self.norm(x, **maybe_cond_kwargs)
        # 返回前馈网络的结果
        return self.net(x)


# 注释: 使用反锯齿下采样（blurpool Zhang 等人的方法）构建的判别器
# 定义一个模糊处理模块，继承自 Module 类
class Blur(Module):
    # 初始化方法
    def __init__(self):
        # 调用父类构造函数
        super().__init__()
        # 创建一个一维张量 f，包含模糊核的值
        f = torch.Tensor([1, 2, 1])
        # 注册一个缓冲区以存储模糊核
        self.register_buffer("f", f)

    # 前向传播方法
    def forward(self, x, space_only=False, time_only=False):
        # 确保不同时使用空间模糊和时间模糊
        assert not (space_only and time_only)

        # 获取模糊核
        f = self.f

        # 如果只进行空间模糊
        if space_only:
            # 计算外积以生成二维模糊核
            f = einsum("i, j -> i j", f, f)
            # 调整维度为 1x1xN
            f = rearrange(f, "... -> 1 1 ...")
        # 如果只进行时间模糊
        elif time_only:
            # 调整维度为 1xNx1x1
            f = rearrange(f, "f -> 1 f 1 1")
        else:
            # 如果同时进行空间和时间模糊
            # 计算三维外积以生成三维模糊核
            f = einsum("i, j, k -> i j k", f, f, f)
            # 调整维度为 1xNxN
            f = rearrange(f, "... -> 1 ...")

        # 检查输入是否为图像格式（4维）
        is_images = x.ndim == 4

        # 如果是图像格式，则调整维度以适应后续处理
        if is_images:
            x = rearrange(x, "b c h w -> b c 1 h w")

        # 使用三维滤波器处理输入
        out = filter3d(x, f, normalized=True)

        # 如果是图像格式，则恢复到原始维度
        if is_images:
            out = rearrange(out, "b c 1 h w -> b c h w")

        # 返回处理后的输出
        return out


# 定义一个判别器块，继承自 Module 类
class DiscriminatorBlock(Module):
    # 初始化方法，接受输入通道数、过滤器数量等参数
    def __init__(self, input_channels, filters, downsample=True, antialiased_downsample=True):
        # 调用父类构造函数
        super().__init__()
        # 定义残差卷积层，决定是否下采样
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        # 定义一个序列网络，包括两个卷积层和激活函数
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu(),
        )

        # 根据条件选择是否使用模糊处理
        self.maybe_blur = Blur() if antialiased_downsample else None

        # 定义下采样操作，如果需要下采样
        self.downsample = (
            nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2), nn.Conv2d(filters * 4, filters, 1)
            )
            if downsample
            else None
        )

    # 前向传播方法
    def forward(self, x):
        # 计算残差
        res = self.conv_res(x)

        # 通过网络处理输入
        x = self.net(x)

        # 如果有下采样操作
        if exists(self.downsample):
            # 如果有模糊处理
            if exists(self.maybe_blur):
                # 进行空间模糊处理
                x = self.maybe_blur(x, space_only=True)

            # 执行下采样操作
            x = self.downsample(x)

        # 将处理后的输出与残差相加并进行归一化
        x = (x + res) * (2**-0.5)
        # 返回处理后的输出
        return x


# 定义一个判别器类，继承自 Module 类
class Discriminator(Module):
    # 类型注释的方法
    @beartype
    # 初始化方法，接受多个参数
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels=3,
        max_dim=512,
        attn_heads=8,
        attn_dim_head=32,
        linear_attn_dim_head=8,
        linear_attn_heads=16,
        ff_mult=4,
        antialiased_downsample=False,
    ):
        # 调用父类的构造函数
        super().__init__()
        # 将输入的图像大小转换为元组形式
        image_size = pair(image_size)
        # 获取图像最小分辨率
        min_image_resolution = min(image_size)

        # 计算网络层数，最小分辨率减少2的对数取整
        num_layers = int(log2(min_image_resolution) - 2)

        # 初始化块列表
        blocks = []

        # 定义每层的维度，第一层为通道数，后续层根据指数增长
        layer_dims = [channels] + [(dim * 4) * (2**i) for i in range(num_layers + 1)]
        # 确保每层的维度不超过最大维度
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        # 创建输入和输出维度的元组
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        # 重新初始化块列表和注意力块列表
        blocks = []
        attn_blocks = []

        # 设置图像分辨率为最小分辨率
        image_resolution = min_image_resolution

        # 遍历每对输入和输出通道
        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            # 计算当前层数
            num_layer = ind + 1
            # 判断当前层是否为最后一层
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            # 创建判别器块，包含输入和输出通道，是否下采样的标志
            block = DiscriminatorBlock(
                in_chan, out_chan, downsample=is_not_last, antialiased_downsample=antialiased_downsample
            )

            # 创建注意力块，包含残差连接和前馈层
            attn_block = Sequential(
                Residual(LinearSpaceAttention(dim=out_chan, heads=linear_attn_heads, dim_head=linear_attn_dim_head)),
                Residual(FeedForward(dim=out_chan, mult=ff_mult, images=True)),
            )

            # 将块和注意力块添加到块列表中
            blocks.append(ModuleList([block, attn_block]))

            # 每次迭代将图像分辨率减半
            image_resolution //= 2

        # 将所有块转换为模块列表
        self.blocks = ModuleList(blocks)

        # 获取最后一层的维度
        dim_last = layer_dims[-1]

        # 计算下采样因子
        downsample_factor = 2**num_layers
        # 计算最后特征图的大小
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        # 计算潜在维度
        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        # 创建最终输出的层，包含卷积、激活、重排列和线性层
        self.to_logits = Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1),
            leaky_relu(),
            Rearrange("b ... -> b (...)"),
            nn.Linear(latent_dim, 1),
            Rearrange("b 1 -> b"),
        )

    # 定义前向传播方法
    def forward(self, x):
        # 遍历每个块和注意力块进行前向传播
        for block, attn_block in self.blocks:
            x = block(x)  # 通过判别器块
            x = attn_block(x)  # 通过注意力块

        # 返回最后的 logits 结果
        return self.to_logits(x)
# 可调节卷积，来自 Karras 等人的 Stylegan2
# 用于对潜在变量进行条件化


class Conv3DMod(Module):
    @beartype
    # 初始化函数，设置卷积的维度、核大小等参数
    def __init__(
        self, dim, *, spatial_kernel, time_kernel, causal=True, dim_out=None, demod=True, eps=1e-8, pad_mode="zeros"
    ):
        super().__init__()  # 调用父类的初始化函数
        dim_out = default(dim_out, dim)  # 如果未指定 dim_out，则默认与 dim 相同

        self.eps = eps  # 设置一个小常数用于数值稳定性

        # 确保空间核和时间核都是奇数
        assert is_odd(spatial_kernel) and is_odd(time_kernel)

        self.spatial_kernel = spatial_kernel  # 保存空间核的大小
        self.time_kernel = time_kernel  # 保存时间核的大小

        # 根据是否为因果卷积计算时间填充
        time_padding = (time_kernel - 1, 0) if causal else ((time_kernel // 2,) * 2)

        self.pad_mode = pad_mode  # 设置填充模式
        # 计算总的填充大小，包含空间和时间的填充
        self.padding = (*((spatial_kernel // 2,) * 4), *time_padding)
        # 初始化卷积核权重为随机值，并作为可学习参数
        self.weights = nn.Parameter(torch.randn((dim_out, dim, time_kernel, spatial_kernel, spatial_kernel)))

        self.demod = demod  # 是否进行去调制的标志

        # 使用 Kaiming 正态分布初始化卷积权重
        nn.init.kaiming_normal_(self.weights, a=0, mode="fan_in", nonlinearity="selu")

    @beartype
    # 前向传播函数，定义数据的流动
    def forward(self, fmap, cond: Tensor):
        """
        符号说明

        b - 批量
        n - 卷积
        o - 输出
        i - 输入
        k - 核
        """

        b = fmap.shape[0]  # 获取批量大小

        # 准备用于调制的权重

        weights = self.weights  # 获取当前权重

        # 执行调制和去调制，参考 stylegan2 的实现

        cond = rearrange(cond, "b i -> b 1 i 1 1 1")  # 调整条件张量的形状以适配权重

        weights = weights * (cond + 1)  # 对权重进行调制

        if self.demod:  # 如果需要去调制
            # 计算权重的逆归一化因子
            inv_norm = reduce(weights**2, "b o i k0 k1 k2 -> b o 1 1 1 1", "sum").clamp(min=self.eps).rsqrt()
            weights = weights * inv_norm  # 对权重进行去调制

        # 调整 fmap 的形状以适配卷积操作
        fmap = rearrange(fmap, "b c t h w -> 1 (b c) t h w")

        # 调整权重的形状
        weights = rearrange(weights, "b o ... -> (b o) ...")

        # 对 fmap 进行填充
        fmap = F.pad(fmap, self.padding, mode=self.pad_mode)
        # 进行 3D 卷积操作
        fmap = F.conv3d(fmap, weights, groups=b)

        # 调整输出的形状为 (b, o, ...)
        return rearrange(fmap, "1 (b o) ... -> b o ...", b=b)


# 进行步幅卷积以降采样


class SpatialDownsample2x(Module):
    # 初始化函数，设置降采样的维度和卷积参数
    def __init__(self, dim, dim_out=None, kernel_size=3, antialias=False):
        super().__init__()  # 调用父类的初始化函数
        dim_out = default(dim_out, dim)  # 如果未指定 dim_out，则默认与 dim 相同
        # 根据是否启用抗混叠设置可能的模糊操作
        self.maybe_blur = Blur() if antialias else identity
        # 初始化 2D 卷积，步幅为 2，填充为核大小的一半
        self.conv = nn.Conv2d(dim, dim_out, kernel_size, stride=2, padding=kernel_size // 2)

    # 前向传播函数
    def forward(self, x):
        # 进行模糊处理（如果需要）
        x = self.maybe_blur(x, space_only=True)

        # 调整输入的形状
        x = rearrange(x, "b c t h w -> b t c h w")
        x, ps = pack_one(x, "* c h w")  # 将数据打包以便处理

        out = self.conv(x)  # 进行卷积操作

        out = unpack_one(out, ps, "* c h w")  # 解包数据
        # 调整输出的形状为 (b, c, t, h, w)
        out = rearrange(out, "b t c h w -> b c t h w")
        return out


# 时间维度的降采样


class TimeDownsample2x(Module):
    # 初始化函数，设置降采样的维度和卷积参数
    def __init__(self, dim, dim_out=None, kernel_size=3, antialias=False):
        super().__init__()  # 调用父类的初始化函数
        dim_out = default(dim_out, dim)  # 如果未指定 dim_out，则默认与 dim 相同
        # 根据是否启用抗混叠设置可能的模糊操作
        self.maybe_blur = Blur() if antialias else identity
        self.time_causal_padding = (kernel_size - 1, 0)  # 设置时间因果填充
        # 初始化 1D 卷积，步幅为 2
        self.conv = nn.Conv1d(dim, dim_out, kernel_size, stride=2)
    # 前向传播方法，接收输入 x
        def forward(self, x):
            # 根据时间维度可能模糊处理输入 x
            x = self.maybe_blur(x, time_only=True)
    
            # 重排张量维度，从 (batch, channels, time, height, width) 到 (batch, height, width, channels, time)
            x = rearrange(x, "b c t h w -> b h w c t")
            # 将重排后的张量打包，返回新张量和打包信息 ps
            x, ps = pack_one(x, "* c t")
    
            # 对张量 x 进行填充，添加时间因果填充
            x = F.pad(x, self.time_causal_padding)
            # 使用卷积层处理填充后的张量
            out = self.conv(x)
    
            # 解包卷积输出，恢复到原始张量形状
            out = unpack_one(out, ps, "* c t")
            # 再次重排维度，从 (batch, height, width, channels, time) 到 (batch, channels, time, height, width)
            out = rearrange(out, "b h w c t -> b c t h w")
            # 返回最终输出
            return out
# 深度到空间的上采样


class SpatialUpsample2x(Module):
    # 初始化上采样模块，指定输入和输出通道
    def __init__(self, dim, dim_out=None):
        # 调用父类构造函数
        super().__init__()
        # 如果未指定输出维度，则将其设置为输入维度
        dim_out = default(dim_out, dim)
        # 创建一个卷积层，将输入通道扩展为输出通道的四倍
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        # 定义网络结构，包括卷积、激活和重排列
        self.net = nn.Sequential(conv, nn.SiLU(), Rearrange("b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2))

        # 初始化卷积层的权重
        self.init_conv_(conv)

    # 初始化卷积层的权重
    def init_conv_(self, conv):
        # 获取卷积层的输出通道、输入通道、高度和宽度
        o, i, h, w = conv.weight.shape
        # 创建一个新的权重张量，形状为输出通道的四分之一
        conv_weight = torch.empty(o // 4, i, h, w)
        # 使用 He 均匀初始化卷积权重
        nn.init.kaiming_uniform_(conv_weight)
        # 扩展权重张量，使输出通道数量恢复到四倍
        conv_weight = repeat(conv_weight, "o ... -> (o 4) ...")

        # 将新的权重复制到卷积层
        conv.weight.data.copy_(conv_weight)
        # 将卷积层的偏置初始化为零
        nn.init.zeros_(conv.bias.data)

    # 前向传播函数
    def forward(self, x):
        # 重排列输入张量，使维度顺序为 (batch, time, channel, height, width)
        x = rearrange(x, "b c t h w -> b t c h w")
        # 打包张量，将其维度压缩
        x, ps = pack_one(x, "* c h w")

        # 通过网络进行前向传播
        out = self.net(x)

        # 解包输出张量，恢复维度
        out = unpack_one(out, ps, "* c h w")
        # 重排列输出张量
        out = rearrange(out, "b t c h w -> b c t h w")
        return out


class TimeUpsample2x(Module):
    # 初始化时间上采样模块，指定输入和输出通道
    def __init__(self, dim, dim_out=None):
        # 调用父类构造函数
        super().__init__()
        # 如果未指定输出维度，则将其设置为输入维度
        dim_out = default(dim_out, dim)
        # 创建一个一维卷积层，将输入通道扩展为输出通道的两倍
        conv = nn.Conv1d(dim, dim_out * 2, 1)

        # 定义网络结构，包括卷积、激活和重排列
        self.net = nn.Sequential(conv, nn.SiLU(), Rearrange("b (c p) t -> b c (t p)", p=2))

        # 初始化卷积层的权重
        self.init_conv_(conv)

    # 初始化卷积层的权重
    def init_conv_(self, conv):
        # 获取卷积层的输出通道、输入通道和时间维度
        o, i, t = conv.weight.shape
        # 创建一个新的权重张量，形状为输出通道的二分之一
        conv_weight = torch.empty(o // 2, i, t)
        # 使用 He 均匀初始化卷积权重
        nn.init.kaiming_uniform_(conv_weight)
        # 扩展权重张量，使输出通道数量恢复到两倍
        conv_weight = repeat(conv_weight, "o ... -> (o 2) ...")

        # 将新的权重复制到卷积层
        conv.weight.data.copy_(conv_weight)
        # 将卷积层的偏置初始化为零
        nn.init.zeros_(conv.bias.data)

    # 前向传播函数
    def forward(self, x):
        # 重排列输入张量，使维度顺序为 (batch, height, width, channel, time)
        x = rearrange(x, "b c t h w -> b h w c t")
        # 打包张量，将其维度压缩
        x, ps = pack_one(x, "* c t")

        # 通过网络进行前向传播
        out = self.net(x)

        # 解包输出张量，恢复维度
        out = unpack_one(out, ps, "* c t")
        # 重排列输出张量
        out = rearrange(out, "b h w c t -> b c t h w")
        return out


# 自编码器 - 这里只提供最佳变体，使用因果卷积 3D


# 创建一个带有填充的卷积层，保持输入和输出维度相同
def SameConv2d(dim_in, dim_out, kernel_size):
    # 将核大小转换为元组，如果不是的话
    kernel_size = cast_tuple(kernel_size, 2)
    # 计算填充，以保持卷积后尺寸不变
    padding = [k // 2 for k in kernel_size]
    # 返回具有指定参数的卷积层
    return nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding)


class CausalConv3d(Module):
    # 定义因果卷积 3D 的构造函数
    @beartype
    def __init__(
        # 输入通道、输出通道、核大小及填充模式
        self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], pad_mode="constant", **kwargs
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 将 kernel_size 转换为包含 3 个元素的元组
        kernel_size = cast_tuple(kernel_size, 3)

        # 解包 kernel_size 为时间、 altura 和宽度的大小
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        # 确保高度和宽度的内核大小都是奇数
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        # 从关键字参数中弹出膨胀和步幅的值，默认值为 1
        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)

        # 设置填充模式
        self.pad_mode = pad_mode
        # 计算时间维度的填充大小
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        # 计算高度和宽度的填充大小
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        # 保存时间填充大小
        self.time_pad = time_pad
        # 设置时间因果填充，包含宽度、高度和时间的填充大小
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        # 设置步幅为 (步幅, 1, 1) 的元组
        stride = (stride, 1, 1)
        # 设置膨胀为 (膨胀, 1, 1) 的元组
        dilation = (dilation, 1, 1)
        # 创建 3D 卷积层
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        # 根据输入 x 的形状和时间填充确定填充模式
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"

        # 对输入 x 进行填充
        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        # 返回经过卷积层处理后的输出
        return self.conv(x)
# 装饰器，用于类型检查
@beartype
# 定义残差单元，包含卷积操作和激活函数
def ResidualUnit(dim, kernel_size: Union[int, Tuple[int, int, int]], pad_mode: str = "constant"):
    # 创建一个顺序模型，包含一系列层
    net = Sequential(
        # 使用因果卷积进行3D卷积，指定输入和输出通道、卷积核大小及填充方式
        CausalConv3d(dim, dim, kernel_size, pad_mode=pad_mode),
        # 应用ELU激活函数
        nn.ELU(),
        # 进行1x1x1卷积
        nn.Conv3d(dim, dim, 1),
        # 再次应用ELU激活函数
        nn.ELU(),
        # 使用Squeeze and Excitation模块
        SqueezeExcite(dim),
    )

    # 返回残差模块
    return Residual(net)


# 装饰器，用于类型检查
@beartype
# 定义带条件输入的残差单元模块
class ResidualUnitMod(Module):
    # 初始化方法，定义模块参数
    def __init__(
        self, dim, kernel_size: Union[int, Tuple[int, int, int]], *, dim_cond, pad_mode: str = "constant", demod=True
    ):
        # 调用父类构造函数
        super().__init__()
        # 将卷积核大小转换为元组，确保为三维
        kernel_size = cast_tuple(kernel_size, 3)
        # 解包卷积核的时间、高度和宽度
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        # 断言高度和宽度的卷积核大小相同
        assert height_kernel_size == width_kernel_size

        # 定义条件输入的线性层
        self.to_cond = nn.Linear(dim_cond, dim)

        # 定义条件卷积层
        self.conv = Conv3DMod(
            dim=dim,
            spatial_kernel=height_kernel_size,
            time_kernel=time_kernel_size,
            causal=True,
            demod=demod,
            pad_mode=pad_mode,
        )

        # 定义输出卷积层
        self.conv_out = nn.Conv3d(dim, dim, 1)

    # 装饰器，用于类型检查
    @beartype
    # 前向传播方法，定义模块的计算流程
    def forward(
        self,
        x,
        cond: Tensor,
    ):
        # 保存输入以便于后续相加
        res = x
        # 将条件输入通过线性层转换
        cond = self.to_cond(cond)

        # 使用条件卷积处理输入
        x = self.conv(x, cond=cond)
        # 应用ELU激活函数
        x = F.elu(x)
        # 通过输出卷积层处理
        x = self.conv_out(x)
        # 再次应用ELU激活函数
        x = F.elu(x)
        # 返回残差连接的结果
        return x + res


# 定义因果卷积转置模块
class CausalConvTranspose3d(Module):
    # 初始化方法，定义模块参数
    def __init__(self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], *, time_stride, **kwargs):
        # 调用父类构造函数
        super().__init__()
        # 将卷积核大小转换为元组，确保为三维
        kernel_size = cast_tuple(kernel_size, 3)

        # 解包卷积核的时间、高度和宽度
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        # 断言高度和宽度的卷积核大小为奇数
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        # 设置上采样因子
        self.upsample_factor = time_stride

        # 计算高度和宽度的填充大小
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        # 定义步幅和填充
        stride = (time_stride, 1, 1)
        padding = (0, height_pad, width_pad)

        # 定义转置卷积层
        self.conv = nn.ConvTranspose3d(chan_in, chan_out, kernel_size, stride, padding=padding, **kwargs)

    # 前向传播方法，定义模块的计算流程
    def forward(self, x):
        # 确保输入为5维
        assert x.ndim == 5
        # 获取时间维度的大小
        t = x.shape[2]

        # 通过转置卷积层处理输入
        out = self.conv(x)

        # 切片以匹配上采样后的时间维度
        out = out[..., : (t * self.upsample_factor), :, :]
        # 返回处理后的输出
        return out


# 定义损失分解的命名元组
LossBreakdown = namedtuple(
    "LossBreakdown",
    [
        # 重构损失
        "recon_loss",
        # 辅助损失
        "lfq_aux_loss",
        # 量化器损失分解
        "quantizer_loss_breakdown",
        # 感知损失
        "perceptual_loss",
        # 对抗生成损失
        "adversarial_gen_loss",
        # 自适应对抗权重
        "adaptive_adversarial_weight",
        # 多尺度生成损失
        "multiscale_gen_losses",
        # 多尺度生成自适应权重
        "multiscale_gen_adaptive_weights",
    ],
)

# 定义鉴别器损失分解的命名元组
DiscrLossBreakdown = namedtuple("DiscrLossBreakdown", ["discr_loss", "multiscale_discr_losses", "gradient_penalty"])


# 定义视频分词器模块
class VideoTokenizer(Module):
    # 装饰器，用于类型检查
    @beartype
    # 初始化方法，用于创建类的实例，接受多个参数
        def __init__(
            self,
            *,  # 使用关键字参数
            image_size,  # 输入图像的尺寸
            layers: Tuple[Union[str, Tuple[str, int]], ...] = ("residual", "residual", "residual"),  # 网络层的类型与配置
            residual_conv_kernel_size=3,  # 残差卷积核的大小
            num_codebooks=1,  # 代码本的数量
            codebook_size: Optional[int] = None,  # 代码本的大小（可选）
            channels=3,  # 输入图像的通道数（如RGB）
            init_dim=64,  # 初始化维度
            max_dim=float("inf"),  # 最大维度，默认为无穷大
            dim_cond=None,  # 条件维度（可选）
            dim_cond_expansion_factor=4.0,  # 条件维度扩展因子
            input_conv_kernel_size: Tuple[int, int, int] = (7, 7, 7),  # 输入卷积核的大小
            output_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),  # 输出卷积核的大小
            pad_mode: str = "constant",  # 填充模式，默认为常数填充
            lfq_entropy_loss_weight=0.1,  # LFQ熵损失权重
            lfq_commitment_loss_weight=1.0,  # LFQ承诺损失权重
            lfq_diversity_gamma=2.5,  # LFQ多样性超参数
            quantizer_aux_loss_weight=1.0,  # 量化辅助损失权重
            lfq_activation=nn.Identity(),  # LFQ激活函数，默认为恒等函数
            use_fsq=False,  # 是否使用FSQ
            fsq_levels: Optional[List[int]] = None,  # FSQ的级别（可选）
            attn_dim_head=32,  # 注意力维度头大小
            attn_heads=8,  # 注意力头的数量
            attn_dropout=0.0,  # 注意力的丢弃率
            linear_attn_dim_head=8,  # 线性注意力维度头大小
            linear_attn_heads=16,  # 线性注意力头的数量
            vgg: Optional[Module] = None,  # VGG模型（可选）
            vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,  # VGG权重
            perceptual_loss_weight=1e-1,  # 感知损失权重
            discr_kwargs: Optional[dict] = None,  # 判别器参数（可选）
            multiscale_discrs: Tuple[Module, ...] = tuple(),  # 多尺度判别器
            use_gan=True,  # 是否使用GAN
            adversarial_loss_weight=1.0,  # 对抗损失权重
            grad_penalty_loss_weight=10.0,  # 梯度惩罚损失权重
            multiscale_adversarial_loss_weight=1.0,  # 多尺度对抗损失权重
            flash_attn=True,  # 是否使用闪存注意力
            separate_first_frame_encoding=False,  # 是否分开第一帧编码
        @property
        def device(self):  # 属性方法，返回设备信息
            return self.zero.device
    
        @classmethod
        def init_and_load_from(cls, path, strict=True):  # 类方法，用于初始化并从指定路径加载模型
            path = Path(path)  # 将路径转换为Path对象
            assert path.exists()  # 确保路径存在
            pkg = torch.load(str(path), map_location="cpu")  # 从指定路径加载模型，映射到CPU
    
            assert "config" in pkg, "model configs were not found in this saved checkpoint"  # 确保配置存在
    
            config = pickle.loads(pkg["config"])  # 反序列化配置
            tokenizer = cls(**config)  # 使用配置创建类的实例
            tokenizer.load(path, strict=strict)  # 加载模型权重
            return tokenizer  # 返回初始化的tokenizer实例
    
        def parameters(self):  # 返回模型的所有可训练参数
            return [
                *self.conv_in.parameters(),  # 输入卷积层的参数
                *self.conv_in_first_frame.parameters(),  # 第一帧输入卷积层的参数
                *self.conv_out_first_frame.parameters(),  # 第一帧输出卷积层的参数
                *self.conv_out.parameters(),  # 输出卷积层的参数
                *self.encoder_layers.parameters(),  # 编码层的参数
                *self.decoder_layers.parameters(),  # 解码层的参数
                *self.encoder_cond_in.parameters(),  # 编码条件输入的参数
                *self.decoder_cond_in.parameters(),  # 解码条件输入的参数
                *self.quantizers.parameters(),  # 量化器的参数
            ]
    
        def discr_parameters(self):  # 返回判别器的参数
            return self.discr.parameters()  # 获取判别器的可训练参数
    
        def copy_for_eval(self):  # 创建用于评估的模型副本
            device = self.device  # 获取当前设备
            vae_copy = copy.deepcopy(self.cpu())  # 深拷贝模型并转到CPU
    
            maybe_del_attr_(vae_copy, "discr")  # 删除判别器属性（如果存在）
            maybe_del_attr_(vae_copy, "vgg")  # 删除VGG属性（如果存在）
            maybe_del_attr_(vae_copy, "multiscale_discrs")  # 删除多尺度判别器属性（如果存在）
    
            vae_copy.eval()  # 设置模型为评估模式
            return vae_copy.to(device)  # 将模型移动到原设备并返回
    
        @remove_vgg  # 装饰器，用于去掉VGG的相关内容
        def state_dict(self, *args, **kwargs):  # 返回模型的状态字典
            return super().state_dict(*args, **kwargs)  # 调用父类方法获取状态字典
    
        @remove_vgg  # 装饰器，用于去掉VGG的相关内容
        def load_state_dict(self, *args, **kwargs):  # 加载状态字典
            return super().load_state_dict(*args, **kwargs)  # 调用父类方法加载状态字典
    # 保存模型参数到指定路径
    def save(self, path, overwrite=True):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 如果 overwrite 为 False 且路径已存在，则抛出异常
        assert overwrite or not path.exists(), f"{str(path)} already exists"

        # 创建包含模型参数、版本和配置的字典
        pkg = dict(model_state_dict=self.state_dict(), version=__version__, config=self._configs)

        # 将字典保存到指定路径
        torch.save(pkg, str(path))

    # 从指定路径加载模型参数
    def load(self, path, strict=True):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 如果路径不存在，则抛出异常
        assert path.exists()

        # 加载保存的模型参数
        pkg = torch.load(str(path))
        state_dict = pkg.get("model_state_dict")
        version = pkg.get("version")

        # 断言模型参数存在
        assert exists(state_dict)

        # 如果存在版本信息，则打印加载的版本信息
        if exists(version):
            print(f"loading checkpointed tokenizer from version {version}")

        # 加载模型参数到当前模型
        self.load_state_dict(state_dict, strict=strict)

    # 编码视频
    @beartype
    def encode(self, video: Tensor, quantize=False, cond: Optional[Tensor] = None, video_contains_first_frame=True):
        # 是否分开编码第一帧
        encode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        # 是否对视频进行填充
        if video_contains_first_frame:
            video_len = video.shape[2]
            video = pad_at_dim(video, (self.time_padding, 0), value=0.0, dim=2)
            video_packed_shape = [torch.Size([self.time_padding]), torch.Size([]), torch.Size([video_len - 1])]

        # 如果需要条件编码，则对条件进行处理
        assert (not self.has_cond) or exists(
            cond
        ), "`cond` must be passed into tokenizer forward method since conditionable layers were specified"

        if exists(cond):
            assert cond.shape == (video.shape[0], self.dim_cond)
            cond = self.encoder_cond_in(cond)
            cond_kwargs = dict(cond=cond)

        # 初始卷积
        if encode_first_frame_separately:
            pad, first_frame, video = unpack(video, video_packed_shape, "b c * h w")
            first_frame = self.conv_in_first_frame(first_frame)

        video = self.conv_in(video)

        if encode_first_frame_separately:
            video, _ = pack([first_frame, video], "b c * h w")
            video = pad_at_dim(video, (self.time_padding, 0), dim=2)

        # 编码器层
        for fn, has_cond in zip(self.encoder_layers, self.has_cond_across_layers):
            layer_kwargs = dict()

            if has_cond:
                layer_kwargs = cond_kwargs

            video = fn(video, **layer_kwargs)

        # 是否进行量化
        maybe_quantize = identity if not quantize else self.quantizers

        return maybe_quantize(video)

    @beartype
    # 从编码索引解码，将编码转换为原始数据
    def decode_from_code_indices(self, codes: Tensor, cond: Optional[Tensor] = None, video_contains_first_frame=True):
        # 断言编码的数据类型为 long 或 int32
        assert codes.dtype in (torch.long, torch.int32)

        # 如果编码的维度为2，则重新排列成视频编码的形状
        if codes.ndim == 2:
            video_code_len = codes.shape[-1]
            assert divisible_by(
                video_code_len, self.fmap_size**2
            ), f"flattened video ids must have a length ({video_code_len}) that is divisible by the fmap size ({self.fmap_size}) squared ({self.fmap_size ** 2})"
            codes = rearrange(codes, "b (f h w) -> b f h w", h=self.fmap_size, w=self.fmap_size)

        # 将索引编码转换为量化编码
        quantized = self.quantizers.indices_to_codes(codes)

        # 调用解码方法，返回解码后的视频
        return self.decode(quantized, cond=cond, video_contains_first_frame=video_contains_first_frame)

    # 解码方法
    @beartype
    def decode(self, quantized: Tensor, cond: Optional[Tensor] = None, video_contains_first_frame=True):
        # 如果需要单独解码第一帧，则设置为True
        decode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        batch = quantized.shape[0]

        # 如果需要条件编码，则进行条件编码
        assert (not self.has_cond) or exists(
            cond
        ), "`cond` must be passed into tokenizer forward method since conditionable layers were specified"
        if exists(cond):
            assert cond.shape == (batch, self.dim_cond)
            cond = self.decoder_cond_in(cond)
            cond_kwargs = dict(cond=cond)

        # 解码层
        x = quantized
        for fn, has_cond in zip(self.decoder_layers, reversed(self.has_cond_across_layers)):
            layer_kwargs = dict()
            if has_cond:
                layer_kwargs = cond_kwargs
            x = fn(x, **layer_kwargs)

        # 转换为像素
        if decode_first_frame_separately:
            left_pad, xff, x = (
                x[:, :, : self.time_padding],
                x[:, :, self.time_padding],
                x[:, :, (self.time_padding + 1) :],
            )
            out = self.conv_out(x)
            outff = self.conv_out_first_frame(xff)
            video, _ = pack([outff, out], "b c * h w")
        else:
            video = self.conv_out(x)
            # 如果视频有填充，则移除填充
            if video_contains_first_frame:
                video = video[:, :, self.time_padding :]

        return video

    # 无梯度的标记方法，用于标记不需要梯度的操作
    @torch.no_grad()
    def tokenize(self, video):
        self.eval()
        return self.forward(video, return_codes=True)

    # 前向传播方法
    @beartype
    def forward(
        self,
        video_or_images: Tensor,
        cond: Optional[Tensor] = None,
        return_loss=False,
        return_codes=False,
        return_recon=False,
        return_discr_loss=False,
        return_recon_loss_only=False,
        apply_gradient_penalty=True,
        video_contains_first_frame=True,
        adversarial_loss_weight=None,
        multiscale_adversarial_loss_weight=None,
# 主类定义
class MagViT2(Module):
    # 构造函数，初始化类
    def __init__(self):
        # 调用父类的构造函数
        super().__init__()

    # 前向传播函数，处理输入数据
    def forward(self, x):
        # 返回输入数据 x
        return x
```