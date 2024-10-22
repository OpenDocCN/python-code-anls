# `.\cogvideo-finetune\sat\sgm\modules\attention.py`

```py
# 导入数学库
import math
# 从 inspect 模块导入 isfunction 函数，用于检查对象是否为函数
from inspect import isfunction
# 导入 Any 和 Optional 类型注解
from typing import Any, Optional

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能模块
import torch.nn.functional as F
# 从 einops 导入 rearrange 和 repeat 函数，用于张量操作
from einops import rearrange, repeat
# 导入版本管理模块
from packaging import version
# 导入 PyTorch 的神经网络模块
from torch import nn

# 检查 PyTorch 版本是否大于等于 2.0.0
if version.parse(torch.__version__) >= version.parse("2.0.0"):
    # 如果版本合适，设置 SDP 可用为 True
    SDP_IS_AVAILABLE = True
    # 从 CUDA 后端导入 SDPBackend 和 sdp_kernel
    from torch.backends.cuda import SDPBackend, sdp_kernel

    # 定义后端配置映射
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
# 如果版本低于 2.0.0
else:
    # 从上下文管理器导入 nullcontext
    from contextlib import nullcontext

    # 设置 SDP 可用为 False
    SDP_IS_AVAILABLE = False
    # 设置 sdp_kernel 为 nullcontext
    sdp_kernel = nullcontext
    # 打印提示信息，建议升级 PyTorch
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. In fact, "
        f"you are using PyTorch {torch.__version__}. You might want to consider upgrading."
    )

# 尝试导入 xformers 和其操作模块
try:
    import xformers
    import xformers.ops

    # 如果导入成功，设置 XFORMERS 可用为 True
    XFORMERS_IS_AVAILABLE = True
# 如果导入失败
except:
    # 设置 XFORMERS 可用为 False
    XFORMERS_IS_AVAILABLE = False
    # 打印提示信息，说明处理将不使用 xformers
    print("no module 'xformers'. Processing without...")

# 从本地模块导入 checkpoint 函数
from .diffusionmodules.util import checkpoint


# 定义 exists 函数，检查值是否存在
def exists(val):
    return val is not None


# 定义 uniq 函数，返回数组中的唯一元素
def uniq(arr):
    return {el: True for el in arr}.keys()


# 定义 default 函数，返回值或默认值
def default(val, d):
    if exists(val):
        return val
    # 如果默认值是函数，调用它并返回结果，否则返回默认值
    return d() if isfunction(d) else d


# 定义 max_neg_value 函数，返回张量类型的最大负值
def max_neg_value(t):
    return -torch.finfo(t.dtype).max


# 定义 init_ 函数，用于初始化张量
def init_(tensor):
    dim = tensor.shape[-1]  # 获取张量的最后一维大小
    std = 1 / math.sqrt(dim)  # 计算标准差
    tensor.uniform_(-std, std)  # 用均匀分布初始化张量
    return tensor


# 定义 GEGLU 类，继承自 nn.Module
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()  # 调用父类构造函数
        # 定义一个线性层，将输入维度映射到输出维度的两倍
        self.proj = nn.Linear(dim_in, dim_out * 2)

    # 定义前向传播方法
    def forward(self, x):
        # 将输入通过线性层，并将输出分为两个部分
        x, gate = self.proj(x).chunk(2, dim=-1)
        # 返回 x 与 gate 的 GELU 激活函数的乘积
        return x * F.gelu(gate)


# 定义 FeedForward 类，继承自 nn.Module
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()  # 调用父类构造函数
        inner_dim = int(dim * mult)  # 计算内部维度
        dim_out = default(dim_out, dim)  # 获取输出维度，若未提供则使用输入维度
        # 根据是否使用 GLU 选择不同的输入项目
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        # 定义完整的网络结构
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    # 定义前向传播方法
    def forward(self, x):
        return self.net(x)  # 返回网络的输出


# 定义 zero_module 函数，将模块的参数归零并返回模块
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    # 遍历模块的参数并将其归零
    for p in module.parameters():
        p.detach().zero_()
    return module  # 返回归零后的模块


# 定义 Normalize 函数，返回一个 GroupNorm 层
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# 定义 LinearAttention 类，继承自 nn.Module
class LinearAttention(nn.Module):
    # 初始化方法，设置维度、头数和每个头的维度
        def __init__(self, dim, heads=4, dim_head=32):
            # 调用父类初始化方法
            super().__init__()
            # 保存头数
            self.heads = heads
            # 计算隐藏层维度，等于每个头的维度乘以头数
            hidden_dim = dim_head * heads
            # 定义一个卷积层，将输入维度转换为三倍的隐藏层维度，不使用偏置
            self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
            # 定义输出卷积层，将隐藏层维度转换为原始输入维度
            self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    
        # 前向传播方法
        def forward(self, x):
            # 获取输入的批量大小、通道数、高度和宽度
            b, c, h, w = x.shape
            # 通过卷积层转换输入，得到查询、键、值
            qkv = self.to_qkv(x)
            # 重排张量，使得查询、键、值分别分开并调整维度
            q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
            # 对键进行softmax处理，归一化
            k = k.softmax(dim=-1)
            # 计算上下文，使用爱因斯坦求和约定对键和值进行操作
            context = torch.einsum("bhdn,bhen->bhde", k, v)
            # 结合上下文和查询计算输出
            out = torch.einsum("bhde,bhdn->bhen", context, q)
            # 重排输出以适应原始维度
            out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
            # 通过输出卷积层得到最终结果
            return self.to_out(out)
# 定义一个空间自注意力类，继承自 nn.Module
class SpatialSelfAttention(nn.Module):
    # 初始化函数，接收输入通道数
    def __init__(self, in_channels):
        # 调用父类的初始化函数
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

    # 前向传播函数
    def forward(self, x):
        # 将输入赋值给 h_
        h_ = x
        # 对输入进行归一化处理
        h_ = self.norm(h_)
        # 计算查询
        q = self.q(h_)
        # 计算键
        k = self.k(h_)
        # 计算值
        v = self.v(h_)

        # 计算注意力
        b, c, h, w = q.shape
        # 重排查询张量
        q = rearrange(q, "b c h w -> b (h w) c")
        # 重排键张量
        k = rearrange(k, "b c h w -> b c (h w)")
        # 计算注意力权重
        w_ = torch.einsum("bij,bjk->bik", q, k)

        # 缩放权重
        w_ = w_ * (int(c) ** (-0.5))
        # 对权重应用 softmax
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # 关注值
        # 重排值张量
        v = rearrange(v, "b c h w -> b c (h w)")
        # 重排权重张量
        w_ = rearrange(w_, "b i j -> b j i")
        # 计算最终输出
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        # 重排输出张量
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        # 应用输出投影层
        h_ = self.proj_out(h_)

        # 返回原始输入与输出的和
        return x + h_


# 定义一个交叉注意力类，继承自 nn.Module
class CrossAttention(nn.Module):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 计算内部维度
        inner_dim = dim_head * heads
        # 确定上下文维度
        context_dim = default(context_dim, query_dim)

        # 设置缩放因子
        self.scale = dim_head**-0.5
        # 保存头的数量
        self.heads = heads

        # 创建查询线性层
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # 创建键线性层
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # 创建值线性层
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 创建输出层，包含线性层和 dropout
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        # 保存后端配置
        self.backend = backend

    # 前向传播函数
    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    # 函数体开始
        ):
            # 获取当前对象的头部数量
            h = self.heads
    
            # 检查是否有附加令牌
            if additional_tokens is not None:
                # 获取输出序列开始部分的掩码令牌数量
                n_tokens_to_mask = additional_tokens.shape[1]
                # 将附加令牌与输入 x 拼接在一起
                x = torch.cat([additional_tokens, x], dim=1)
    
            # 将输入 x 转换为查询向量
            q = self.to_q(x)
            # 使用默认值确保上下文不为空
            context = default(context, x)
            # 将上下文转换为键向量
            k = self.to_k(context)
            # 将上下文转换为值向量
            v = self.to_v(context)
    
            # 检查是否需要跨帧注意力机制
            if n_times_crossframe_attn_in_self:
                # 按照论文中的方法重新编程跨帧注意力
                assert x.shape[0] % n_times_crossframe_attn_in_self == 0
                # 计算每个交叉帧的数量
                n_cp = x.shape[0] // n_times_crossframe_attn_in_self
                # 重复键向量以适应新的形状
                k = repeat(k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)
                # 重复值向量以适应新的形状
                v = repeat(v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)
    
            # 重新排列查询、键和值向量
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
    
            # 旧的注意力机制代码
            """
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
            del q, k
    
            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)
    
            # 注意力机制
            sim = sim.softmax(dim=-1)
    
            out = einsum('b i j, b j d -> b i d', sim, v)
            """
            # 新的注意力机制实现
            with sdp_kernel(**BACKEND_MAP[self.backend]):
                # 打印当前后端信息及 q/k/v 的形状
                # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
                # 计算缩放点积注意力
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # scale 是 dim_head ** -0.5
    
            # 删除查询、键和值向量以释放内存
            del q, k, v
            # 重新排列输出
            out = rearrange(out, "b h n d -> b n (h d)", h=h)
    
            # 如果有附加令牌，则移除它们
            if additional_tokens is not None:
                out = out[:, n_tokens_to_mask:]
            # 返回最终输出
            return self.to_out(out)
# 定义一个内存高效的交叉注意力层，继承自 nn.Module
class MemoryEfficientCrossAttention(nn.Module):
    # 初始化方法，设置各个参数
    # query_dim: 查询向量的维度
    # context_dim: 上下文向量的维度，默认为 None
    # heads: 注意力头的数量，默认为 8
    # dim_head: 每个注意力头的维度，默认为 64
    # dropout: dropout 概率，默认为 0.0
    # kwargs: 其他额外的关键字参数
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs):
        # 调用父类的初始化方法
        super().__init__()
        # 打印模型的设置情况，包括查询维度、上下文维度、头数和每个头的维度
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}."
        )
        # 计算内部维度，即每个头的维度乘以头的数量
        inner_dim = dim_head * heads
        # 如果 context_dim 为 None，使用 query_dim 作为上下文维度
        context_dim = default(context_dim, query_dim)

        # 保存头的数量和每个头的维度
        self.heads = heads
        self.dim_head = dim_head

        # 定义将查询向量映射到内部维度的线性层，不使用偏置
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # 定义将上下文向量映射到内部维度的线性层，不使用偏置
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # 定义将上下文向量映射到内部维度的线性层，不使用偏置
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 定义输出层，由线性层和 dropout 组成
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        # 初始化注意力操作为 None，稍后可能会被赋值
        self.attention_op: Optional[Any] = None

    # 定义前向传播方法
    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        # 检查是否存在附加的令牌
        if additional_tokens is not None:
            # 获取输出序列开头被掩码的令牌数量
            n_tokens_to_mask = additional_tokens.shape[1]
            # 将附加令牌与输入张量拼接
            x = torch.cat([additional_tokens, x], dim=1)
        # 将输入张量转换为查询向量
        q = self.to_q(x)
        # 默认上下文为输入张量
        context = default(context, x)
        # 将上下文转换为键向量
        k = self.to_k(context)
        # 将上下文转换为值向量
        v = self.to_v(context)

        # 如果需要进行跨帧注意力的次数
        if n_times_crossframe_attn_in_self:
            # 重新编程跨帧注意力，参考文献 https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # 计算每个批次中的重复次数
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            # 以跨帧的步长重复键向量
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            # 以跨帧的步长重复值向量
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        # 获取查询向量的形状信息
        b, _, _ = q.shape
        # 对查询、键和值向量进行转换和重塑
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

        # TODO: 将此直接用于注意力操作，作为偏差
        if exists(mask):
            raise NotImplementedError
        # 对输出进行重塑以适应后续处理
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        # 如果存在附加的令牌，移除它们
        if additional_tokens is not None:
            out = out[:, n_tokens_to_mask:]
        # 返回最终输出
        return self.to_out(out)
# 定义一个基础的变换器块，继承自 nn.Module
class BasicTransformerBlock(nn.Module):
    # 定义可用的注意力模式及其对应的类
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # 标准注意力
        "softmax-xformers": MemoryEfficientCrossAttention,  # 高效的注意力实现
    }

    # 初始化方法，设置基本参数
    def __init__(
        self,
        dim,  # 输入维度
        n_heads,  # 注意力头数
        d_head,  # 每个注意力头的维度
        dropout=0.0,  # dropout 概率
        context_dim=None,  # 上下文维度，可选
        gated_ff=True,  # 是否使用门控前馈网络
        checkpoint=True,  # 是否使用检查点
        disable_self_attn=False,  # 是否禁用自注意力
        attn_mode="softmax",  # 选择的注意力模式
        sdp_backend=None,  # 后端设置
    ):
        # 调用父类构造函数
        super().__init__()
        # 确保选择的注意力模式在可用模式中
        assert attn_mode in self.ATTENTION_MODES
        # 如果选择的模式不是软最大并且 xformers 不可用，回退到标准注意力
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            print(
                f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
            )
            attn_mode = "softmax"  # 回退到软最大模式
        # 如果选择的是软最大且不支持，则给出警告并回退
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print("We do not support vanilla attention anymore, as it is too expensive. Sorry.")
            # 如果 xformers 不可用，抛出错误
            if not XFORMERS_IS_AVAILABLE:
                assert False, "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                print("Falling back to xformers efficient attention.")  # 回退到 xformers 的高效注意力
                attn_mode = "softmax-xformers"
        # 根据选择的模式获取注意力类
        attn_cls = self.ATTENTION_MODES[attn_mode]
        # 检查 PyTorch 版本，如果是 2.0 或更高版本，检查 sdp_backend
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None  # 在低版本中 sdp_backend 必须为 None
        self.disable_self_attn = disable_self_attn  # 设置自注意力禁用标志
        # 创建第一个注意力层，如果禁用自注意力，则 context_dim 为 None
        self.attn1 = attn_cls(
            query_dim=dim,  # 查询的维度
            heads=n_heads,  # 注意力头数
            dim_head=d_head,  # 每个头的维度
            dropout=dropout,  # dropout 概率
            context_dim=context_dim if self.disable_self_attn else None,  # 上下文维度
            backend=sdp_backend,  # 后端设置
        )  # 如果禁用自注意力，则为自注意力
        # 创建前馈网络
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # 创建第二个注意力层
        self.attn2 = attn_cls(
            query_dim=dim,  # 查询的维度
            context_dim=context_dim,  # 上下文维度
            heads=n_heads,  # 注意力头数
            dim_head=d_head,  # 每个头的维度
            dropout=dropout,  # dropout 概率
            backend=sdp_backend,  # 后端设置
        )  # 如果上下文为 None 则为自注意力
        # 创建三个 LayerNorm 层
        self.norm1 = nn.LayerNorm(dim)  # 第一个归一化层
        self.norm2 = nn.LayerNorm(dim)  # 第二个归一化层
        self.norm3 = nn.LayerNorm(dim)  # 第三个归一化层
        self.checkpoint = checkpoint  # 设置检查点标志
        # 如果启用检查点，打印信息
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")  # 输出当前类使用检查点的提示
    # 定义前向传播方法，接收输入数据及可选的上下文和附加标记
    def forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        # 将输入数据封装到字典中，方便后续处理
        kwargs = {"x": x}
    
        # 如果提供了上下文，则将其添加到字典中
        if context is not None:
            kwargs.update({"context": context})
    
        # 如果提供了附加标记，则将其添加到字典中
        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})
    
        # 如果提供了跨帧自注意力的次数，则将其添加到字典中
        if n_times_crossframe_attn_in_self:
            kwargs.update({"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self})
    
        # 使用检查点机制进行前向传播，并返回结果
        # return mixed_checkpoint(self._forward, kwargs, self.parameters(), self.checkpoint)
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
    
    # 定义实际的前向传播实现，处理输入数据及其他参数
    def _forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        # 通过第一个注意力层处理输入数据，应用归一化，并考虑上下文和其他参数
        x = (
            self.attn1(
                self.norm1(x),  # 对输入进行归一化处理
                context=context if self.disable_self_attn else None,  # 根据条件决定是否使用上下文
                additional_tokens=additional_tokens,  # 传递附加标记
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self if not self.disable_self_attn else 0,  # 根据条件设置参数
            )
            + x  # 将注意力层的输出与原输入相加，进行残差连接
        )
        # 通过第二个注意力层处理数据，并应用上下文和附加标记
        x = self.attn2(self.norm2(x), context=context, additional_tokens=additional_tokens) + x  # 残差连接
        # 通过前馈网络处理数据，并应用归一化
        x = self.ff(self.norm3(x)) + x  # 残差连接
        # 返回最终的处理结果
        return x
# 定义一个基本的单层变换器块，继承自 nn.Module
class BasicTransformerSingleLayerBlock(nn.Module):
    # 定义支持的注意力模式及其对应的实现类
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # 普通的注意力机制
        "softmax-xformers": MemoryEfficientCrossAttention,  # 在 A100 上不如上述版本快
        # (todo 可能依赖于 head_dim，待检查，对于 dim!=[16,32,64,128] 回退到半优化内核)
    }

    # 初始化函数，设置变换器的参数
    def __init__(
        self,
        dim,  # 特征维度
        n_heads,  # 注意力头的数量
        d_head,  # 每个注意力头的维度
        dropout=0.0,  # Dropout 概率
        context_dim=None,  # 上下文的维度（可选）
        gated_ff=True,  # 是否使用门控前馈网络
        checkpoint=True,  # 是否使用检查点以节省内存
        attn_mode="softmax",  # 使用的注意力模式
    ):
        # 调用父类构造函数
        super().__init__()
        # 确保传入的注意力模式是有效的
        assert attn_mode in self.ATTENTION_MODES
        # 根据注意力模式选择对应的注意力类
        attn_cls = self.ATTENTION_MODES[attn_mode]
        # 初始化注意力层
        self.attn1 = attn_cls(
            query_dim=dim,  # 查询的维度
            heads=n_heads,  # 注意力头数量
            dim_head=d_head,  # 每个头的维度
            dropout=dropout,  # Dropout 概率
            context_dim=context_dim,  # 上下文维度
        )
        # 初始化前馈网络
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # 初始化层归一化
        self.norm1 = nn.LayerNorm(dim)  # 第一个归一化层
        self.norm2 = nn.LayerNorm(dim)  # 第二个归一化层
        # 保存检查点标志
        self.checkpoint = checkpoint

    # 前向传播函数
    def forward(self, x, context=None):
        # 使用检查点机制调用 _forward 函数
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    # 内部前向传播实现
    def _forward(self, x, context=None):
        # 应用注意力层并添加残差连接
        x = self.attn1(self.norm1(x), context=context) + x
        # 应用前馈网络并添加残差连接
        x = self.ff(self.norm2(x)) + x
        # 返回处理后的数据
        return x


# 定义一个空间变换器类，继承自 nn.Module
class SpatialTransformer(nn.Module):
    """
    用于图像数据的变换器块。
    首先，对输入进行投影（即嵌入）
    然后重塑为 b, t, d。
    接着应用标准的变换器操作。
    最后，重塑为图像。
    NEW: 使用线性层以提高效率，而不是 1x1 的卷积层。
    """

    # 初始化函数，设置空间变换器的参数
    def __init__(
        self,
        in_channels,  # 输入通道数
        n_heads,  # 注意力头的数量
        d_head,  # 每个注意力头的维度
        depth=1,  # 变换器的深度
        dropout=0.0,  # Dropout 概率
        context_dim=None,  # 上下文维度（可选）
        disable_self_attn=False,  # 是否禁用自注意力
        use_linear=False,  # 是否使用线性层
        attn_type="softmax",  # 注意力类型
        use_checkpoint=True,  # 是否使用检查点
        # sdp_backend=SDPBackend.FLASH_ATTENTION  # (注释掉的选项) SDP 后端
        sdp_backend=None,  # SDP 后端（可选）
    ):
        # 调用父类的构造函数
        super().__init__()
        # 打印当前类的名称、深度、输入通道数和头数
        print(f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads")
        # 从 omegaconf 导入 ListConfig 类
        from omegaconf import ListConfig

        # 如果 context_dim 存在且不是列表或 ListConfig 类型
        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            # 将 context_dim 转换为包含一个元素的列表
            context_dim = [context_dim]
        # 如果 context_dim 存在且是列表类型
        if exists(context_dim) and isinstance(context_dim, list):
            # 检查 context_dim 的长度是否与深度匹配
            if depth != len(context_dim):
                # 打印警告信息，指出深度与 context_dim 长度不匹配
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # 确保所有 context_dim 元素相同
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                # 将 context_dim 设置为深度数量的列表，元素为 context_dim 的第一个元素
                context_dim = depth * [context_dim[0]]
        # 如果 context_dim 为 None
        elif context_dim is None:
            # 将 context_dim 设置为深度数量的 None 列表
            context_dim = [None] * depth
        # 将输入通道数赋值给实例变量
        self.in_channels = in_channels
        # 计算内部维度
        inner_dim = n_heads * d_head
        # 创建归一化层
        self.norm = Normalize(in_channels)
        # 如果不使用线性投影
        if not use_linear:
            # 创建 2D 卷积层作为输入投影
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            # 创建线性层作为输入投影
            self.proj_in = nn.Linear(in_channels, inner_dim)

        # 创建变换器模块列表
        self.transformer_blocks = nn.ModuleList(
            [
                # 为每个深度创建 BasicTransformerBlock
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
                for d in range(depth)  # 遍历深度
            ]
        )
        # 如果不使用线性投影
        if not use_linear:
            # 创建 2D 卷积层作为输出投影，并将其设为零模块
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            # 创建线性层作为输出投影，并将其设为零模块
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        # 将使用线性标志赋值给实例变量
        self.use_linear = use_linear
    # 定义前向传播函数，接受输入 x 和可选的上下文 context
        def forward(self, x, context=None):
            # 注意：如果未提供上下文，则交叉注意力默认为自注意力
            if not isinstance(context, list):
                # 如果上下文不是列表，则将其包装为单元素列表
                context = [context]
            # 获取输入张量 x 的形状参数 b（批量大小）、c（通道数）、h（高度）、w（宽度）
            b, c, h, w = x.shape
            # 保存输入张量以便后续使用
            x_in = x
            # 对输入 x 进行归一化处理
            x = self.norm(x)
            # 如果不使用线性层，则对输入进行投影
            if not self.use_linear:
                x = self.proj_in(x)
            # 重新排列 x 的形状，将其从 (b, c, h, w) 转换为 (b, h*w, c)
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()
            # 如果使用线性层，则对 x 进行投影
            if self.use_linear:
                x = self.proj_in(x)
            # 遍历每个 transformer 块
            for i, block in enumerate(self.transformer_blocks):
                # 如果不是第一个块且上下文只有一个，重置索引为 0，以便在每个块中使用相同的上下文
                if i > 0 and len(context) == 1:
                    i = 0  # 使用相同的上下文
                # 通过 transformer 块处理 x 和相应的上下文
                x = block(x, context=context[i])
            # 如果使用线性层，则对 x 进行最终投影
            if self.use_linear:
                x = self.proj_out(x)
            # 重新排列 x 的形状，将其转换回 (b, c, h, w)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
            # 如果不使用线性层，则对输出进行投影
            if not self.use_linear:
                x = self.proj_out(x)
            # 返回处理后的 x 与原输入 x_in 之和
            return x + x_in
```