# `.\cogvideo-finetune\sat\vae_modules\attention.py`

```py
# 导入数学库
import math
# 从 inspect 模块导入 isfunction 函数，用于检查对象是否为函数
from inspect import isfunction
# 导入 Any 和 Optional 类型注解
from typing import Any, Optional

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能性操作
import torch.nn.functional as F
# 从 einops 库导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 导入版本控制工具
from packaging import version
# 导入 PyTorch 的神经网络模块
from torch import nn

# 检查 PyTorch 版本是否大于或等于 2.0.0
if version.parse(torch.__version__) >= version.parse("2.0.0"):
    # 设置 SDP_IS_AVAILABLE 为 True，表示 SDP 后端可用
    SDP_IS_AVAILABLE = True
    # 从 CUDA 后端导入 SDPBackend 和 sdp_kernel
    from torch.backends.cuda import SDPBackend, sdp_kernel

    # 定义后端配置映射
    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,  # 启用数学模式
            "enable_flash": False,  # 禁用闪电模式
            "enable_mem_efficient": False,  # 禁用内存高效模式
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,  # 禁用数学模式
            "enable_flash": True,  # 启用闪电模式
            "enable_mem_efficient": False,  # 禁用内存高效模式
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,  # 禁用数学模式
            "enable_flash": False,  # 禁用闪电模式
            "enable_mem_efficient": True,  # 启用内存高效模式
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},  # 默认情况下启用所有模式
    }
else:
    # 从上下文管理模块导入 nullcontext
    from contextlib import nullcontext

    # 设置 SDP_IS_AVAILABLE 为 False，表示 SDP 后端不可用
    SDP_IS_AVAILABLE = False
    # 将 sdp_kernel 设置为 nullcontext
    sdp_kernel = nullcontext
    # 打印警告信息，提示用户升级 PyTorch 版本
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. In fact, "
        f"you are using PyTorch {torch.__version__}. You might want to consider upgrading."
    )

# 尝试导入 xformers 和 xformers.ops 模块
try:
    import xformers
    import xformers.ops

    # 如果导入成功，设置 XFORMERS_IS_AVAILABLE 为 True
    XFORMERS_IS_AVAILABLE = True
# 如果导入失败，设置 XFORMERS_IS_AVAILABLE 为 False，并打印提示信息
except:
    XFORMERS_IS_AVAILABLE = False
    print("no module 'xformers'. Processing without...")

# 从 modules.utils 模块导入 checkpoint 函数
from modules.utils import checkpoint


# 定义一个检查值是否存在的函数
def exists(val):
    # 返回值是否不为 None
    return val is not None


# 定义一个去重函数
def uniq(arr):
    # 将数组转换为字典以去重，然后返回字典的键
    return {el: True for el in arr}.keys()


# 定义一个返回默认值的函数
def default(val, d):
    # 如果 val 存在，返回 val
    if exists(val):
        return val
    # 否则返回 d 的结果，如果 d 是函数则调用它
    return d() if isfunction(d) else d


# 定义一个返回最大负值的函数
def max_neg_value(t):
    # 返回指定数据类型的最大负值
    return -torch.finfo(t.dtype).max


# 定义一个初始化张量的函数
def init_(tensor):
    # 获取张量的最后一个维度
    dim = tensor.shape[-1]
    # 计算标准差
    std = 1 / math.sqrt(dim)
    # 在[-std, std]范围内均匀初始化张量
    tensor.uniform_(-std, std)
    # 返回初始化后的张量
    return tensor


# 定义一个前馈神经网络类
class GEGLU(nn.Module):
    # 初始化方法
    def __init__(self, dim_in, dim_out):
        super().__init__()  # 调用父类的初始化方法
        # 创建线性变换层，输出维度为输入维度的两倍
        self.proj = nn.Linear(dim_in, dim_out * 2)

    # 前向传播方法
    def forward(self, x):
        # 将输入 x 通过线性层投影并拆分为两部分
        x, gate = self.proj(x).chunk(2, dim=-1)
        # 返回 x 和经过 GELU 激活的 gate 的乘积
        return x * F.gelu(gate)


# 定义一个前馈神经网络类
class FeedForward(nn.Module):
    # 初始化方法
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()  # 调用父类的初始化方法
        # 计算内部维度
        inner_dim = int(dim * mult)
        # 如果 dim_out 为空，设置为 dim
        dim_out = default(dim_out, dim)
        # 根据 glu 参数决定使用哪种输入变换
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        # 创建包含输入变换、dropout 和线性变换的序列模型
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    # 前向传播方法
    def forward(self, x):
        # 返回网络的输出
        return self.net(x)


# 定义一个将模块参数归零的函数
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    # 遍历模块的所有参数
    for p in module.parameters():
        # 断开与计算图的联系并将参数归零
        p.detach().zero_()
    # 返回修改后的模块
    return module


# 定义一个归一化函数
def Normalize(in_channels):
    # 返回 GroupNorm 实例，用于对输入通道进行归一化
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# 定义一个线性注意力类
class LinearAttention(nn.Module):
    # 初始化方法，设置维度、头数和每个头的维度
        def __init__(self, dim, heads=4, dim_head=32):
            # 调用父类的初始化方法
            super().__init__()
            # 存储头数
            self.heads = heads
            # 计算隐藏层的维度，即头数与每个头的维度的乘积
            hidden_dim = dim_head * heads
            # 创建一个卷积层，将输入通道数转换为三倍的隐藏维度，用于生成查询、键和值
            self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
            # 创建一个卷积层，将隐藏维度转换回原始的输入通道数
            self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    
        # 前向传播方法，处理输入数据
        def forward(self, x):
            # 获取输入的批次大小、通道数、高度和宽度
            b, c, h, w = x.shape
            # 通过卷积层生成查询、键和值的组合
            qkv = self.to_qkv(x)
            # 重新排列 qkv 数据，使其分开为 q、k 和 v，并按头数分组
            q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
            # 对键进行 softmax 操作，计算注意力权重
            k = k.softmax(dim=-1)
            # 使用爱因斯坦求和约定计算上下文，即加权后的值
            context = torch.einsum("bhdn,bhen->bhde", k, v)
            # 根据查询和上下文计算输出
            out = torch.einsum("bhde,bhdn->bhen", context, q)
            # 重新排列输出数据，恢复到原始形状
            out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
            # 返回最终的输出结果
            return self.to_out(out)
# 定义一个空间自注意力类，继承自 nn.Module
class SpatialSelfAttention(nn.Module):
    # 初始化方法，接受输入通道数
    def __init__(self, in_channels):
        # 调用父类构造函数
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels

        # 创建归一化层
        self.norm = Normalize(in_channels)
        # 创建查询卷积层
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建键卷积层
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建值卷积层
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建输出投影卷积层
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法
    def forward(self, x):
        # 初始化 h_ 为输入 x
        h_ = x
        # 对输入进行归一化
        h_ = self.norm(h_)
        # 计算查询
        q = self.q(h_)
        # 计算键
        k = self.k(h_)
        # 计算值
        v = self.v(h_)

        # 计算注意力
        b, c, h, w = q.shape  # 获取批量大小、通道数、高度和宽度
        # 重新排列查询以便进行矩阵乘法
        q = rearrange(q, "b c h w -> b (h w) c")
        # 重新排列键以便进行矩阵乘法
        k = rearrange(k, "b c h w -> b c (h w)")
        # 计算注意力权重
        w_ = torch.einsum("bij,bjk->bik", q, k)

        # 缩放注意力权重
        w_ = w_ * (int(c) ** (-0.5))
        # 应用 softmax 得到注意力分布
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # 处理值
        v = rearrange(v, "b c h w -> b c (h w)")  # 重新排列值
        w_ = rearrange(w_, "b i j -> b j i")  # 重新排列注意力权重
        # 根据注意力权重对值进行加权
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        # 重新排列以匹配输出形状
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        # 通过输出卷积层进行投影
        h_ = self.proj_out(h_)

        # 返回原始输入与输出的和
        return x + h_


# 定义一个交叉注意力类，继承自 nn.Module
class CrossAttention(nn.Module):
    # 初始化方法，接受多个参数
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        # 调用父类构造函数
        super().__init__()
        # 计算内部维度
        inner_dim = dim_head * heads
        # 设置上下文维度的默认值
        context_dim = default(context_dim, query_dim)

        # 计算缩放因子
        self.scale = dim_head**-0.5
        # 保存头的数量
        self.heads = heads

        # 创建查询线性层
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # 创建键线性层
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # 创建值线性层
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 创建输出线性层及 dropout
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        # 保存后端
        self.backend = backend

    # 前向传播方法
    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads  # 获取头的数量，用于后续的多头注意力机制

        if additional_tokens is not None:  # 检查是否有额外的标记
            # 获取输出序列开头的掩蔽标记数量
            n_tokens_to_mask = additional_tokens.shape[1]  
            # 将额外的标记添加到输入序列前面
            x = torch.cat([additional_tokens, x], dim=1)  

        # 将输入 x 转换为查询向量 q
        q = self.to_q(x)  
        # 默认上下文为 x
        context = default(context, x)  
        # 将上下文转换为键向量 k
        k = self.to_k(context)  
        # 将上下文转换为值向量 v
        v = self.to_v(context)  

        if n_times_crossframe_attn_in_self:  # 检查是否需要进行跨帧注意力机制
            # 按照文献中的方法重新编程跨帧注意力
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0  # 确保输入长度是跨帧次数的整数倍
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self  # 计算每个跨帧的大小
            # 根据跨帧次数重复键向量
            k = repeat(k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)  
            # 根据跨帧次数重复值向量
            v = repeat(v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)  

        # 将 q, k, v 重新排列为适合多头注意力的形状
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))  

        # old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale  # 计算查询和键的相似度
        del q, k  # 删除不再需要的 q 和 k

        if exists(mask):  # 检查是否存在掩蔽
            mask = rearrange(mask, 'b ... -> b (...)')  # 重新排列掩蔽形状
            max_neg_value = -torch.finfo(sim.dtype).max  # 获取最大负值，用于掩蔽
            mask = repeat(mask, 'b j -> (b h) () j', h=h)  # 将掩蔽重复到多头
            sim.masked_fill_(~mask, max_neg_value)  # 将不需要的部分填充为最大负值

        # 计算注意力分布
        sim = sim.softmax(dim=-1)  

        out = einsum('b i j, b j d -> b i d', sim, v)  # 计算最终输出
        """
        # new
        with sdp_kernel(**BACKEND_MAP[self.backend]):  # 使用指定后端的 SDP 核心
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)  # 调试信息，打印 q, k, v 的形状
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # 计算缩放点积注意力，默认缩放因子为 dim_head ** -0.5

        del q, k, v  # 删除 q, k, v，释放内存
        out = rearrange(out, "b h n d -> b n (h d)", h=h)  # 将输出重新排列为合适的形状

        if additional_tokens is not None:  # 如果存在额外的标记
            # 移除额外的标记
            out = out[:, n_tokens_to_mask:]  
        return self.to_out(out)  # 将输出转换为最终输出格式并返回
# 定义一个内存高效的交叉注意力模块，继承自 nn.Module
class MemoryEfficientCrossAttention(nn.Module):
    # 初始化方法，设置参数并打印相关信息
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs):
        # 调用父类的初始化方法
        super().__init__()
        # 打印类名及查询维度、上下文维度、头数和维度信息
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}."
        )
        # 计算每个头的内部维度
        inner_dim = dim_head * heads
        # 如果上下文维度为 None，则默认为查询维度
        context_dim = default(context_dim, query_dim)

        # 保存头数和每个头的维度
        self.heads = heads
        self.dim_head = dim_head

        # 定义查询、键、值的线性变换，无偏置项
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 定义最终输出的线性变换和 dropout 层
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        # 初始化注意力操作为 None
        self.attention_op: Optional[Any] = None

    # 前向传播方法，接收输入数据和可选参数
    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        # 检查是否有额外的令牌
        if additional_tokens is not None:
            # 获取输出序列开头被遮蔽的令牌数量
            n_tokens_to_mask = additional_tokens.shape[1]
            # 将额外的令牌与输入张量在列上拼接
            x = torch.cat([additional_tokens, x], dim=1)
        # 将输入张量转换为查询张量
        q = self.to_q(x)
        # 默认上下文为输入张量
        context = default(context, x)
        # 将上下文转换为键张量
        k = self.to_k(context)
        # 将上下文转换为值张量
        v = self.to_v(context)

        # 检查是否需要在自注意力中进行跨帧注意力
        if n_times_crossframe_attn_in_self:
            # 按照文献重新编程跨帧注意力
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # k的形状将被重复n_times_crossframe_attn_in_self次
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            # v的形状将被重复n_times_crossframe_attn_in_self次
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        # 获取查询张量的批次大小和其他维度
        b, _, _ = q.shape
        # 对q, k, v进行处理以匹配注意力机制的要求
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # 实际计算注意力机制
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        # TODO: 将这个直接用于注意力操作作为偏差
        if exists(mask):
            # 如果存在掩码，抛出未实现异常
            raise NotImplementedError
        # 调整输出张量的形状以符合后续处理
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        # 如果有额外的令牌，去除它们
        if additional_tokens is not None:
            out = out[:, n_tokens_to_mask:]
        # 将输出张量转换为最终输出
        return self.to_out(out)
# 基础变换器块，继承自 nn.Module
class BasicTransformerBlock(nn.Module):
    # 定义注意力模式的映射
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # 标准注意力
        "softmax-xformers": MemoryEfficientCrossAttention,  # 节省内存的注意力
    }

    # 初始化方法，接收多个参数
    def __init__(
        self,
        dim,  # 特征维度
        n_heads,  # 注意力头数量
        d_head,  # 每个注意力头的维度
        dropout=0.0,  # dropout比率
        context_dim=None,  # 上下文维度
        gated_ff=True,  # 是否使用门控前馈网络
        checkpoint=True,  # 是否使用检查点
        disable_self_attn=False,  # 是否禁用自注意力
        attn_mode="softmax",  # 注意力模式
        sdp_backend=None,  # 后端配置
    ):
        # 调用父类初始化方法
        super().__init__()
        # 确保指定的注意力模式有效
        assert attn_mode in self.ATTENTION_MODES
        # 如果选择的模式不可用，则回退到默认模式
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            print(
                f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
            )
            attn_mode = "softmax"
        # 如果选择的是标准模式，但不再支持，则进行适当处理
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print("We do not support vanilla attention anymore, as it is too expensive. Sorry.")
            # 确保 xformers 可用
            if not XFORMERS_IS_AVAILABLE:
                assert False, "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                print("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"
        # 根据选择的模式获取注意力类
        attn_cls = self.ATTENTION_MODES[attn_mode]
        # 检查 PyTorch 版本以确保后端有效
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        # 设置自注意力禁用标志
        self.disable_self_attn = disable_self_attn
        # 创建第一个注意力层
        self.attn1 = attn_cls(
            query_dim=dim,  # 查询维度
            heads=n_heads,  # 注意力头数量
            dim_head=d_head,  # 每个头的维度
            dropout=dropout,  # dropout比率
            context_dim=context_dim if self.disable_self_attn else None,  # 上下文维度，若禁用自注意力则为None
            backend=sdp_backend,  # 后端配置
        )  # 如果未禁用自注意力，则为自注意力层
        # 创建前馈网络
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # 创建第二个注意力层
        self.attn2 = attn_cls(
            query_dim=dim,  # 查询维度
            context_dim=context_dim,  # 上下文维度
            heads=n_heads,  # 注意力头数量
            dim_head=d_head,  # 每个头的维度
            dropout=dropout,  # dropout比率
            backend=sdp_backend,  # 后端配置
        )  # 如果上下文为None，则为自注意力层
        # 创建层归一化层
        self.norm1 = nn.LayerNorm(dim)  # 第一层归一化
        self.norm2 = nn.LayerNorm(dim)  # 第二层归一化
        self.norm3 = nn.LayerNorm(dim)  # 第三层归一化
        # 设置检查点标志
        self.checkpoint = checkpoint
        # 如果启用检查点，则打印提示
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")
    # 前向传播方法，接收输入和可选的上下文、附加标记及其他参数
    def forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        # 初始化参数字典，包含输入 x
        kwargs = {"x": x}
    
        # 如果上下文不为 None，将其添加到参数字典
        if context is not None:
            kwargs.update({"context": context})
    
        # 如果附加标记不为 None，将其添加到参数字典
        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})
    
        # 如果跨帧自注意力次数大于 0，将其添加到参数字典
        if n_times_crossframe_attn_in_self:
            kwargs.update({"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self})
    
        # 调用检查点函数，执行前向传播，返回计算结果
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
    
    # 定义实际的前向传播逻辑
    def _forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        # 对输入 x 进行归一化处理并通过第一个注意力层，结合上下文和附加标记
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self if not self.disable_self_attn else 0,
            )
            + x  # 将原始输入加到输出中，形成残差连接
        )
        # 对处理后的 x 进行归一化处理，并通过第二个注意力层
        x = self.attn2(self.norm2(x), context=context, additional_tokens=additional_tokens) + x  # 残差连接
        # 通过前馈层处理，完成归一化后加到输入上
        x = self.ff(self.norm3(x)) + x  # 残差连接
        # 返回最终的输出
        return x
# 定义一个单层基本变换器块，继承自 nn.Module
class BasicTransformerSingleLayerBlock(nn.Module):
    # 定义注意力模式的字典，映射名称到对应的注意力类
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # 标准注意力
        "softmax-xformers": MemoryEfficientCrossAttention,  # 在 A100 上速度不如上面的版本
        # (待办：可能依赖于 head_dim，检查，降级为针对 dim!=[16,32,64,128] 的半优化内核)
    }

    # 初始化函数，定义构造函数的参数
    def __init__(
        self,
        dim,  # 输入特征维度
        n_heads,  # 注意力头的数量
        d_head,  # 每个注意力头的维度
        dropout=0.0,  # dropout 概率
        context_dim=None,  # 上下文维度（可选）
        gated_ff=True,  # 是否使用门控前馈网络
        checkpoint=True,  # 是否使用检查点以节省内存
        attn_mode="softmax",  # 使用的注意力模式
    ):
        # 调用父类的构造函数
        super().__init__()
        # 确保给定的注意力模式在允许的模式中
        assert attn_mode in self.ATTENTION_MODES
        # 根据模式获取对应的注意力类
        attn_cls = self.ATTENTION_MODES[attn_mode]
        # 初始化注意力层
        self.attn1 = attn_cls(
            query_dim=dim,  # 查询维度
            heads=n_heads,  # 头数量
            dim_head=d_head,  # 每个头的维度
            dropout=dropout,  # dropout 概率
            context_dim=context_dim,  # 上下文维度
        )
        # 初始化前馈网络
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # 初始化层归一化
        self.norm1 = nn.LayerNorm(dim)
        # 初始化第二个层归一化
        self.norm2 = nn.LayerNorm(dim)
        # 保存检查点标志
        self.checkpoint = checkpoint

    # 前向传播函数
    def forward(self, x, context=None):
        # 使用检查点机制调用 _forward 方法
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    # 实际的前向传播逻辑
    def _forward(self, x, context=None):
        # 先应用注意力层，结合残差连接
        x = self.attn1(self.norm1(x), context=context) + x
        # 应用前馈网络，再结合残差连接
        x = self.ff(self.norm2(x)) + x
        # 返回输出
        return x


# 定义用于图像数据的变换器块，继承自 nn.Module
class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    # 初始化函数，定义构造函数的参数
    def __init__(
        self,
        in_channels,  # 输入通道数
        n_heads,  # 注意力头的数量
        d_head,  # 每个头的维度
        depth=1,  # 堆叠的层数
        dropout=0.0,  # dropout 概率
        context_dim=None,  # 上下文维度（可选）
        disable_self_attn=False,  # 是否禁用自注意力
        use_linear=False,  # 是否使用线性层以提高效率
        attn_type="softmax",  # 使用的注意力类型
        use_checkpoint=True,  # 是否使用检查点以节省内存
        # sdp_backend=SDPBackend.FLASH_ATTENTION  # 备用的 sdp 后端
        sdp_backend=None,  # 指定 sdp 后端（默认为 None）
    ):
        # 调用父类的构造函数，初始化父类属性
        super().__init__()
        # 打印当前类的名称、深度、输入通道数和头的数量
        print(f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads")
        # 从 omegaconf 库导入 ListConfig 类
        from omegaconf import ListConfig

        # 如果 context_dim 存在且不是列表或 ListConfig 类型，则将其转换为列表
        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        # 如果 context_dim 存在且是列表类型
        if exists(context_dim) and isinstance(context_dim, list):
            # 如果给定的深度与 context_dim 的长度不匹配
            if depth != len(context_dim):
                # 打印警告信息，提示深度与上下文维度不匹配，并重设上下文维度
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # 断言所有上下文维度相同，以便自动匹配深度
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                # 将上下文维度设置为深度乘以第一个上下文维度
                context_dim = depth * [context_dim[0]]
        # 如果上下文维度为 None，创建一个包含 None 的列表，长度为深度
        elif context_dim is None:
            context_dim = [None] * depth
        # 设置输入通道数的属性
        self.in_channels = in_channels
        # 计算内部维度，等于头的数量乘以每个头的维度
        inner_dim = n_heads * d_head
        # 创建归一化层
        self.norm = Normalize(in_channels)
        # 根据是否使用线性层选择不同的输入投影层
        if not use_linear:
            # 使用卷积层进行输入投影
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            # 使用线性层进行输入投影
            self.proj_in = nn.Linear(in_channels, inner_dim)

        # 创建变换器块的模块列表
        self.transformer_blocks = nn.ModuleList(
            [
                # 对于每一层深度，创建基本变换器块
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                )
                for d in range(depth)  # 遍历深度范围
            ]
        )
        # 根据是否使用线性层选择不同的输出投影层
        if not use_linear:
            # 使用零初始化的卷积层作为输出投影层
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            # 使用零初始化的线性层作为输出投影层
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        # 设置是否使用线性层的属性
        self.use_linear = use_linear
    # 前向传播函数，接受输入 x 和可选的上下文参数
        def forward(self, x, context=None):
            # 如果没有给定上下文，交叉注意力默认使用自注意力
            if not isinstance(context, list):
                context = [context]  # 确保上下文为列表形式
            # 获取输入 x 的形状信息
            b, c, h, w = x.shape
            # 保存原始输入以便后续使用
            x_in = x
            # 对输入 x 进行归一化处理
            x = self.norm(x)
            # 如果不使用线性变换，则进行输入投影
            if not self.use_linear:
                x = self.proj_in(x)
            # 重新排列 x 的形状，以适应后续处理
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()
            # 如果使用线性变换，则进行输入投影
            if self.use_linear:
                x = self.proj_in(x)
            # 遍历每个变换块
            for i, block in enumerate(self.transformer_blocks):
                # 如果不是第一个块且上下文只有一个，则使用相同的上下文
                if i > 0 and len(context) == 1:
                    i = 0  # 每个块使用相同的上下文
                # 将 x 和对应的上下文传入当前块
                x = block(x, context=context[i])
            # 如果使用线性变换，则进行输出投影
            if self.use_linear:
                x = self.proj_out(x)
            # 重新排列 x 的形状，恢复到原始输入的形状
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
            # 如果不使用线性变换，则进行输出投影
            if not self.use_linear:
                x = self.proj_out(x)
            # 返回输出与原始输入的和
            return x + x_in
```