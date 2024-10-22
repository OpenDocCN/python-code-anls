# `.\cogview3-finetune\sat\sgm\modules\attention.py`

```
# 导入数学库
import math
# 从 inspect 模块导入 isfunction 函数，用于检查对象是否为函数
from inspect import isfunction
# 导入 Any 和 Optional 类型
from typing import Any, Optional

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能性模块
import torch.nn.functional as F
# 从 einops 库导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 导入版本管理工具
from packaging import version
# 导入 PyTorch 的神经网络模块
from torch import nn

# 检查 PyTorch 版本是否大于或等于 2.0.0
if version.parse(torch.__version__) >= version.parse("2.0.0"):
    # 设置 SDP_IS_AVAILABLE 为 True，表示 SDP 后端可用
    SDP_IS_AVAILABLE = True
    # 从 PyTorch 导入 SDPBackend 和 sdp_kernel
    from torch.backends.cuda import SDPBackend, sdp_kernel

    # 定义后端映射字典，根据不同的后端配置相应的选项
    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    # 从上下文管理库导入 nullcontext
    from contextlib import nullcontext

    # 设置 SDP_IS_AVAILABLE 为 False，表示 SDP 后端不可用
    SDP_IS_AVAILABLE = False
    # 将 sdp_kernel 设置为 nullcontext
    sdp_kernel = nullcontext
    # 打印提示信息，告知用户当前 PyTorch 版本不支持 SDP 后端
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. In fact, "
        f"you are using PyTorch {torch.__version__}. You might want to consider upgrading."
    )

# 尝试导入 xformers 和 xformers.ops
try:
    import xformers
    import xformers.ops

    # 如果导入成功，设置 XFORMERS_IS_AVAILABLE 为 True
    XFORMERS_IS_AVAILABLE = True
# 如果导入失败，设置 XFORMERS_IS_AVAILABLE 为 False，并打印提示信息
except:
    XFORMERS_IS_AVAILABLE = False
    print("no module 'xformers'. Processing without...")

# 从 diffusionmodules.util 模块导入 checkpoint 函数
from .diffusionmodules.util import checkpoint


# 定义 exists 函数，检查输入值是否存在
def exists(val):
    return val is not None


# 定义 uniq 函数，返回数组中唯一元素的键
def uniq(arr):
    return {el: True for el in arr}.keys()


# 定义 default 函数，如果 val 存在则返回它，否则返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# 定义 max_neg_value 函数，返回给定张量类型的最大负值
def max_neg_value(t):
    return -torch.finfo(t.dtype).max


# 定义 init_ 函数，初始化张量
def init_(tensor):
    # 获取张量的最后一维的大小
    dim = tensor.shape[-1]
    # 计算标准差
    std = 1 / math.sqrt(dim)
    # 在区间 [-std, std] 内均匀初始化张量
    tensor.uniform_(-std, std)
    return tensor


# 定义 GEGLU 类，继承自 nn.Module
class GEGLU(nn.Module):
    # 初始化方法，设置输入和输出维度
    def __init__(self, dim_in, dim_out):
        super().__init__()
        # 创建一个线性投影层，将输入维度映射到两倍的输出维度
        self.proj = nn.Linear(dim_in, dim_out * 2)

    # 前向传播方法
    def forward(self, x):
        # 将输入通过投影层，分割为 x 和 gate
        x, gate = self.proj(x).chunk(2, dim=-1)
        # 返回 x 与 gate 的 GELU 激活后的乘积
        return x * F.gelu(gate)


# 定义 FeedForward 类，继承自 nn.Module
class FeedForward(nn.Module):
    # 初始化方法，设置维度、输出维度、乘法因子、是否使用 GEGLU 和 dropout 率
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        # 计算内部维度
        inner_dim = int(dim * mult)
        # 如果 dim_out 未定义，使用 dim 作为默认值
        dim_out = default(dim_out, dim)
        # 根据是否使用 GEGLU 创建输入投影层
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        # 定义网络结构
        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    # 前向传播方法
    def forward(self, x):
        # 通过网络结构处理输入
        return self.net(x)


# 定义 zero_module 函数，清零模块的参数并返回该模块
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    # 遍历模块的所有参数
    for p in module.parameters():
        # 将参数的梯度断开并清零
        p.detach().zero_()
    return module


# 定义 Normalize 函数，接收输入通道数
def Normalize(in_channels):
    # 返回一个 GroupNorm 实例，用于对输入进行分组归一化
        return torch.nn.GroupNorm(
            # 设置分组数量为 32
            num_groups=32, 
            # 设置输入通道数量
            num_channels=in_channels, 
            # 设置一个小的 epsilon 值以避免除零错误
            eps=1e-6, 
            # 设定 affine 为 True，以便进行可学习的仿射变换
            affine=True
        )
# 定义线性注意力机制的类，继承自 nn.Module
class LinearAttention(nn.Module):
    # 初始化方法，设置参数维度和头数
    def __init__(self, dim, heads=4, dim_head=32):
        # 调用父类构造函数
        super().__init__()
        # 设置头数
        self.heads = heads
        # 计算隐藏维度
        hidden_dim = dim_head * heads
        # 定义输入到 QKV 的卷积层，输出通道为 hidden_dim 的三倍
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        # 定义输出卷积层，输出通道为原始维度
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    # 前向传播方法
    def forward(self, x):
        # 获取输入的批次大小、通道数、高度和宽度
        b, c, h, w = x.shape
        # 将输入通过 QKV 卷积层
        qkv = self.to_qkv(x)
        # 将 QKV 的输出重排以分开 Q、K、V
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        # 对 K 进行 softmax 归一化
        k = k.softmax(dim=-1)
        # 计算上下文向量，通过爱因斯坦求和约定
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        # 使用上下文向量和 Q 计算输出
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        # 重排输出以恢复到原始形状
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        # 返回最终输出
        return self.to_out(out)


# 定义空间自注意力机制的类，继承自 nn.Module
class SpatialSelfAttention(nn.Module):
    # 初始化方法，设置输入通道
    def __init__(self, in_channels):
        # 调用父类构造函数
        super().__init__()
        # 存储输入通道数
        self.in_channels = in_channels

        # 创建归一化层
        self.norm = Normalize(in_channels)
        # 创建 Q、K、V 的卷积层，输出通道与输入通道相同
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        # 创建输出的卷积层
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    # 前向传播方法
    def forward(self, x):
        # 初始化 h_ 为输入 x
        h_ = x
        # 对 h_ 进行归一化处理
        h_ = self.norm(h_)
        # 计算 Q、K、V
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 计算注意力权重
        b, c, h, w = q.shape
        # 将 Q 重排为 (batch_size, h*w, c) 的形状
        q = rearrange(q, "b c h w -> b (h w) c")
        # 将 K 重排为 (batch_size, c, h*w) 的形状
        k = rearrange(k, "b c h w -> b c (h w)")
        # 计算 Q 和 K 的点积以获得权重
        w_ = torch.einsum("bij,bjk->bik", q, k)

        # 对权重进行缩放
        w_ = w_ * (int(c) ** (-0.5))
        # 对权重进行 softmax 归一化
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # 处理 V
        v = rearrange(v, "b c h w -> b c (h w)")
        # 重排权重 w_ 为 (batch_size, h*w, h*w) 的形状
        w_ = rearrange(w_, "b i j -> b j i")
        # 计算 h_，通过 V 和权重相乘
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        # 将 h_ 重排回原始形状
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        # 通过 proj_out 进行最终的线性变换
        h_ = self.proj_out(h_)

        # 返回输入与处理后的 h_ 的和
        return x + h_


# 定义交叉注意力机制的类，继承自 nn.Module
class CrossAttention(nn.Module):
    # 初始化方法，设置查询、上下文维度和其他参数
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
        # 如果没有提供上下文维度，默认使用查询维度
        context_dim = default(context_dim, query_dim)

        # 设置缩放因子
        self.scale = dim_head**-0.5
        # 设置头数
        self.heads = heads

        # 定义 Q、K、V 的线性变换
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 定义输出层，包括线性变换和 dropout
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        # 存储后端
        self.backend = backend
    # 定义前向传播函数，接收输入数据及相关参数
        def forward(
            self,
            x,
            context=None,
            mask=None,
            additional_tokens=None,
            n_times_crossframe_attn_in_self=0,
        ):
            # 获取注意力头的数量
            h = self.heads
    
            # 如果有额外的 tokens
            if additional_tokens is not None:
                # 获取输出序列开始时的掩码 token 数量
                n_tokens_to_mask = additional_tokens.shape[1]
                # 将额外的 token 添加到输入数据前
                x = torch.cat([additional_tokens, x], dim=1)
    
            # 通过线性变换生成查询向量
            q = self.to_q(x)
            # 使用默认值或输入作为上下文
            context = default(context, x)
            # 通过线性变换生成键向量
            k = self.to_k(context)
            # 通过线性变换生成值向量
            v = self.to_v(context)
    
            # 如果需要进行跨帧注意力
            if n_times_crossframe_attn_in_self:
                # 验证输入批次大小可以被跨帧次数整除
                assert x.shape[0] % n_times_crossframe_attn_in_self == 0
                # 计算每次跨帧的批次大小
                n_cp = x.shape[0] // n_times_crossframe_attn_in_self
                # 重复键向量以适应跨帧注意力
                k = repeat(
                    k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
                )
                # 重复值向量以适应跨帧注意力
                v = repeat(
                    v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
                )
    
            # 将查询、键、值向量重排为适合多头注意力的形状
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
    
            ## old
            """
            # 计算查询与键之间的相似度，并进行缩放
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
            # 删除查询和键以节省内存
            del q, k
    
            # 如果存在掩码
            if exists(mask):
                # 将掩码重排为适合的形状
                mask = rearrange(mask, 'b ... -> b (...)')
                # 获取相似度的最大负值
                max_neg_value = -torch.finfo(sim.dtype).max
                # 重复掩码以适应多头
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                # 使用掩码填充相似度矩阵
                sim.masked_fill_(~mask, max_neg_value)
    
            # 应用 softmax 计算注意力权重
            sim = sim.softmax(dim=-1)
    
            # 使用注意力权重加权值向量，生成输出
            out = einsum('b i j, b j d -> b i d', sim, v)
            """
            ## new
            # 使用指定的后端进行缩放点积注意力计算
            with sdp_kernel(**BACKEND_MAP[self.backend]):
                # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask
                )  # 默认缩放因子为 dim_head ** -0.5
    
            # 删除查询、键和值向量以释放内存
            del q, k, v
            # 将输出重排为适合最终输出的形状
            out = rearrange(out, "b h n d -> b n (h d)", h=h)
    
            # 如果有额外的 tokens
            if additional_tokens is not None:
                # 移除额外的 token
                out = out[:, n_tokens_to_mask:]
            # 返回最终的输出结果
            return self.to_out(out)
# 定义一个内存高效的交叉注意力模块，继承自 nn.Module
class MemoryEfficientCrossAttention(nn.Module):
    # 初始化方法，接受查询维度、上下文维度、头数、每个头的维度和丢弃率等参数
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 计算每个头的内部维度，等于头数乘以每个头的维度
        inner_dim = dim_head * heads
        # 如果上下文维度未提供，则将其设置为查询维度
        context_dim = default(context_dim, query_dim)

        # 保存头数和每个头的维度到实例变量
        self.heads = heads
        self.dim_head = dim_head

        # 创建线性层，将查询输入转换为内部维度，不使用偏置
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # 创建线性层，将上下文输入转换为内部维度，不使用偏置
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # 创建线性层，将上下文输入转换为内部维度，不使用偏置
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 创建一个顺序容器，包含将内部维度转换回查询维度的线性层和丢弃层
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        # 初始化注意力操作为 None
        self.attention_op: Optional[Any] = None

    # 定义前向传播方法，接受输入、上下文、掩码和其他可选参数
    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        # 检查是否提供了额外的令牌
        if additional_tokens is not None:
            # 获取输出序列开头的被遮掩令牌数量
            n_tokens_to_mask = additional_tokens.shape[1]
            # 将额外的令牌与当前输入合并
            x = torch.cat([additional_tokens, x], dim=1)
        # 将输入转换为查询向量
        q = self.to_q(x)
        # 使用默认值或输入作为上下文
        context = default(context, x)
        # 将上下文转换为键向量
        k = self.to_k(context)
        # 将上下文转换为值向量
        v = self.to_v(context)

        # 检查是否需要进行跨帧注意力的重新编程
        if n_times_crossframe_attn_in_self:
            # 进行跨帧注意力的重新编程，参考 https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # k的维度处理，使用每n_times_crossframe_attn_in_self帧的一个
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            # v的维度处理，使用每n_times_crossframe_attn_in_self帧的一个
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        # 获取批次大小和特征维度
        b, _, _ = q.shape
        # 对 q, k, v 进行维度调整和重塑
        q, k, v = map(
            lambda t: t.unsqueeze(3)  # 在最后一维添加一个新维度
            .reshape(b, t.shape[1], self.heads, self.dim_head)  # 重塑为(batch, seq_len, heads, dim_head)
            .permute(0, 2, 1, 3)  # 重新排列维度为(batch, heads, seq_len, dim_head)
            .reshape(b * self.heads, t.shape[1], self.dim_head)  # 再次重塑为(batch * heads, seq_len, dim_head)
            .contiguous(),  # 确保内存连续性
            (q, k, v),  # 对 q, k, v 进行相同处理
        )

        # 实际计算注意力，这个过程是最不可或缺的
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        # TODO: 将这个直接用作注意力操作中的偏置
        if exists(mask):
            # 如果存在遮掩，抛出未实现异常
            raise NotImplementedError
        # 对输出进行维度调整，适配最终输出形状
        out = (
            out.unsqueeze(0)  # 在最前面添加一个新维度
            .reshape(b, self.heads, out.shape[1], self.dim_head)  # 重塑为(batch, heads, seq_len, dim_head)
            .permute(0, 2, 1, 3)  # 重新排列维度为(batch, seq_len, heads, dim_head)
            .reshape(b, out.shape[1], self.heads * self.dim_head)  # 再次重塑为(batch, seq_len, heads * dim_head)
        )
        # 如果有额外的令牌，则移除它们
        if additional_tokens is not None:
            out = out[:, n_tokens_to_mask:]  # 切除被遮掩的令牌部分
        # 将输出转换为最终的输出格式
        return self.to_out(out)
# 定义基本的变换器模块类，继承自 nn.Module
class BasicTransformerBlock(nn.Module):
    # 定义可用的注意力模式，映射到对应的类
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # 普通注意力
        "softmax-xformers": MemoryEfficientCrossAttention,  # 记忆高效注意力
    }

    # 初始化方法，设置模型的参数
    def __init__(
        self,
        dim,  # 输入的维度
        n_heads,  # 注意力头的数量
        d_head,  # 每个注意力头的维度
        dropout=0.0,  # dropout 比例
        context_dim=None,  # 上下文的维度
        gated_ff=True,  # 是否使用门控前馈网络
        checkpoint=True,  # 是否启用检查点
        disable_self_attn=False,  # 是否禁用自注意力
        attn_mode="softmax",  # 注意力模式，默认为 softmax
        sdp_backend=None,  # 后端配置
    ):
        # 调用父类初始化方法
        super().__init__()
        # 检查给定的注意力模式是否有效
        assert attn_mode in self.ATTENTION_MODES
        # 如果使用的注意力模式不支持，回退到默认模式
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            print(
                f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
            )
            attn_mode = "softmax"  # 回退到 softmax 模式
        # 如果使用普通注意力且不支持，给出提示并调整模式
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print(
                "We do not support vanilla attention anymore, as it is too expensive. Sorry."
            )
            # 确保已安装 xformers
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                print("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"  # 使用 xformers 模式
        # 根据最终的注意力模式选择对应的类
        attn_cls = self.ATTENTION_MODES[attn_mode]
        # 检查 PyTorch 版本是否支持指定的后端
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None  # 对于旧版本，后端必须为 None
        self.disable_self_attn = disable_self_attn  # 保存禁用自注意力的设置
        # 初始化第一个注意力层
        self.attn1 = attn_cls(
            query_dim=dim,  # 查询的维度
            heads=n_heads,  # 注意力头数量
            dim_head=d_head,  # 每个头的维度
            dropout=dropout,  # dropout 比例
            context_dim=context_dim if self.disable_self_attn else None,  # 上下文维度
            backend=sdp_backend,  # 后端配置
        )  # 如果未禁用自注意力，则为自注意力层
        # 初始化前馈网络
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)  # 前馈网络配置
        # 初始化第二个注意力层
        self.attn2 = attn_cls(
            query_dim=dim,  # 查询的维度
            context_dim=context_dim,  # 上下文维度
            heads=n_heads,  # 注意力头数量
            dim_head=d_head,  # 每个头的维度
            dropout=dropout,  # dropout 比例
            backend=sdp_backend,  # 后端配置
        )  # 如果上下文维度为 None，则为自注意力层
        # 初始化层归一化层
        self.norm1 = nn.LayerNorm(dim)  # 第一个归一化层
        self.norm2 = nn.LayerNorm(dim)  # 第二个归一化层
        self.norm3 = nn.LayerNorm(dim)  # 第三个归一化层
        self.checkpoint = checkpoint  # 保存检查点设置
        # 如果启用检查点，输出相关信息（代码暂时注释掉）
        # if self.checkpoint:
        #     print(f"{self.__class__.__name__} is using checkpointing")

    # 前向传播方法，定义输入和上下文的处理
    def forward(
        self, x,  # 输入数据
        context=None,  # 上下文数据
        additional_tokens=None,  # 额外的 token
        n_times_crossframe_attn_in_self=0  # 跨帧自注意力的次数
    ):
        # 创建一个字典 kwargs，初始包含键 "x" 和参数 x 的值
        kwargs = {"x": x}

        # 如果 context 参数不为 None，则将其添加到 kwargs 字典中
        if context is not None:
            kwargs.update({"context": context})

        # 如果 additional_tokens 参数不为 None，则将其添加到 kwargs 字典中
        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        # 如果 n_times_crossframe_attn_in_self 为真，则将其添加到 kwargs 字典中
        if n_times_crossframe_attn_in_self:
            kwargs.update(
                {"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self}
            )

        # 返回调用 checkpoint 函数，传入 _forward 方法和相关参数
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(
        # 定义 _forward 方法，接收 x、context、additional_tokens 和 n_times_crossframe_attn_in_self 参数
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        # 对 x 进行规范化，然后通过 attn1 方法进行自注意力计算，并根据条件选择 context 和其他参数
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn
                else 0,
            )
            + x  # 将 self.attn1 的输出与原始 x 相加
        )
        # 继续对 x 进行规范化，通过 attn2 方法进行自注意力计算
        x = (
            self.attn2(
                self.norm2(x), context=context, additional_tokens=additional_tokens
            )
            + x  # 将 self.attn2 的输出与当前 x 相加
        )
        # 对 x 进行规范化，然后通过前馈网络 ff 处理，再与原始 x 相加
        x = self.ff(self.norm3(x)) + x
        # 返回处理后的 x
        return x
# 定义基本的单层变换器块类，继承自 nn.Module
class BasicTransformerSingleLayerBlock(nn.Module):
    # 定义不同的注意力模式及其对应的类
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # 标准注意力
        "softmax-xformers": MemoryEfficientCrossAttention  # 针对 A100s 的优化版本，速度可能略慢
        # (todo 可能依赖于 head_dim，需检查，对于 dim!=[16,32,64,128] 时退回到半优化内核)
    }

    # 初始化方法，设置基本参数
    def __init__(
        self,
        dim,  # 特征维度
        n_heads,  # 注意力头数
        d_head,  # 每个注意力头的维度
        dropout=0.0,  # 丢弃率
        context_dim=None,  # 上下文维度（可选）
        gated_ff=True,  # 是否使用门控前馈网络
        checkpoint=True,  # 是否启用检查点
        attn_mode="softmax",  # 注意力模式
    ):
        # 调用父类构造函数
        super().__init__()
        # 确保所选的注意力模式在定义的模式中
        assert attn_mode in self.ATTENTION_MODES
        # 获取对应的注意力类
        attn_cls = self.ATTENTION_MODES[attn_mode]
        # 初始化注意力层
        self.attn1 = attn_cls(
            query_dim=dim,  # 查询维度
            heads=n_heads,  # 头数
            dim_head=d_head,  # 每头的维度
            dropout=dropout,  # 丢弃率
            context_dim=context_dim,  # 上下文维度
        )
        # 初始化前馈网络
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # 初始化层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # 设置检查点标志
        self.checkpoint = checkpoint

    # 前向传播方法
    def forward(self, x, context=None):
        # 使用检查点机制来进行前向传播
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    # 实际的前向传播实现
    def _forward(self, x, context=None):
        # 通过注意力层进行处理并添加残差连接
        x = self.attn1(self.norm1(x), context=context) + x
        # 通过前馈网络处理并添加残差连接
        x = self.ff(self.norm2(x)) + x
        # 返回处理后的结果
        return x


# 定义空间变换器类，继承自 nn.Module
class SpatialTransformer(nn.Module):
    """
    适用于图像数据的变换器块。
    首先，将输入（即嵌入）投影并重塑为 b, t, d 形状。
    然后应用标准的变换器操作。
    最后，重塑为图像。
    新增：使用线性层提高效率，而不是 1x1 卷积。
    """

    # 初始化方法，设置变换器块的基本参数
    def __init__(
        self,
        in_channels,  # 输入通道数
        n_heads,  # 注意力头数
        d_head,  # 每个头的维度
        depth=1,  # 变换器块的深度
        dropout=0.0,  # 丢弃率
        context_dim=None,  # 上下文维度（可选）
        disable_self_attn=False,  # 是否禁用自注意力
        use_linear=False,  # 是否使用线性层
        attn_type="softmax",  # 注意力类型
        use_checkpoint=True,  # 是否使用检查点
        # sdp_backend=SDPBackend.FLASH_ATTENTION  # 可选的 SDP 后端
        sdp_backend=None,  # SDP 后端设置（默认值为 None）
    # 初始化类，调用父类构造函数
        ):
            super().__init__()
            # 打印构造信息（已注释）
            # print(
            #     f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads"
            # )
            # 从 omegaconf 导入 ListConfig 类
            from omegaconf import ListConfig
    
            # 如果 context_dim 存在且不是列表或 ListConfig 类型
            if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
                # 将 context_dim 转换为列表
                context_dim = [context_dim]
            # 如果 context_dim 存在且是列表
            if exists(context_dim) and isinstance(context_dim, list):
                # 检查 depth 是否与 context_dim 的长度匹配
                if depth != len(context_dim):
                    # 打印警告信息（已注释）
                    # print(
                    #     f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    #     f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                    # )
                    # 确保所有 context_dim 元素相同
                    assert all(
                        map(lambda x: x == context_dim[0], context_dim)
                    ), "need homogenous context_dim to match depth automatically"
                    # 如果不一致，设置 context_dim 为相同值的列表
                    context_dim = depth * [context_dim[0]]
            # 如果 context_dim 为 None
            elif context_dim is None:
                # 创建与 depth 长度相同的 None 列表
                context_dim = [None] * depth
            # 保存输入通道数
            self.in_channels = in_channels
            # 计算内部维度
            inner_dim = n_heads * d_head
            # 归一化层
            self.norm = Normalize(in_channels)
            # 如果不使用线性层
            if not use_linear:
                # 使用卷积层进行输入投影
                self.proj_in = nn.Conv2d(
                    in_channels, inner_dim, kernel_size=1, stride=1, padding=0
                )
            else:
                # 使用线性层进行输入投影
                self.proj_in = nn.Linear(in_channels, inner_dim)
    
            # 创建变压器模块列表
            self.transformer_blocks = nn.ModuleList(
                [
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
                    # 遍历深度范围，生成多个变压器块
                    for d in range(depth)
                ]
            )
            # 如果不使用线性层
            if not use_linear:
                # 使用零初始化卷积层进行输出投影
                self.proj_out = zero_module(
                    nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
                )
            else:
                # 使用零初始化线性层进行输出投影（已注释）
                # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
                self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
            # 保存是否使用线性层的标志
            self.use_linear = use_linear
    # 定义前向传播函数，接收输入 x 和可选的上下文 context
    def forward(self, x, context=None):
        # 注意：如果没有提供上下文，交叉注意力默认为自注意力
        if not isinstance(context, list):
            # 将上下文包装为列表，方便后续处理
            context = [context]
        # 获取输入张量的形状：批量大小 b，通道数 c，高 h，宽 w
        b, c, h, w = x.shape
        # 保存输入张量的原始值以便后续使用
        x_in = x
        # 对输入进行归一化处理
        x = self.norm(x)
        # 如果不使用线性变换，则进行投影变换
        if not self.use_linear:
            x = self.proj_in(x)
        # 重新排列张量的维度，将其从 (b, c, h, w) 变为 (b, h*w, c)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        # 如果使用线性变换，则再次进行投影变换
        if self.use_linear:
            x = self.proj_in(x)
        # 遍历所有的变换块
        for i, block in enumerate(self.transformer_blocks):
            # 如果不是第一个块且上下文长度为1，则使用同一个上下文
            if i > 0 and len(context) == 1:
                i = 0  # 每个块使用相同的上下文
            # 将输入传入当前变换块，并使用相应的上下文
            x = block(x, context=context[i])
        # 如果使用线性变换，则进行输出投影变换
        if self.use_linear:
            x = self.proj_out(x)
        # 重新排列张量的维度，将其从 (b, h*w, c) 变为 (b, c, h, w)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        # 如果不使用线性变换，则进行输出投影变换
        if not self.use_linear:
            x = self.proj_out(x)
        # 返回处理后的张量与原始输入的和
        return x + x_in
```