# `.\lucidrains\gateloop-transformer\gateloop_transformer\gateloop_transformer.py`

```
from functools import partial  # 导入 functools 模块中的 partial 函数

import torch  # 导入 torch 库
from torch.nn import Module, ModuleList  # 从 torch.nn 模块中导入 Module 和 ModuleList 类
from torch import nn, einsum, Tensor  # 从 torch 模块中导入 nn、einsum 和 Tensor
from torch.utils.checkpoint import checkpoint  # 从 torch.utils.checkpoint 模块导入 checkpoint 函数
import torch.nn.functional as F  # 导入 torch.nn.functional 模块并重命名为 F

from einops import rearrange  # 导入 einops 库中的 rearrange 函数
from einops.layers.torch import Rearrange  # 从 einops.layers.torch 模块中导入 Rearrange 类

from rotary_embedding_torch import RotaryEmbedding  # 导入 rotary_embedding_torch 库中的 RotaryEmbedding 类

from gateloop_transformer.associative_scan import associative_scan  # 从 gateloop_transformer.associative_scan 模块中导入 associative_scan 函数

# helpers

def exists(v):  # 定义 exists 函数，用于判断变量是否存在
    return v is not None  # 返回变量是否不为 None

def default(v, d):  # 定义 default 函数，用于返回变量或默认值
    return v if exists(v) else d  # 如果变量存在则返回变量，否则返回默认值

def Sequential(*modules):  # 定义 Sequential 函数，用于创建序列模块
    modules = list(filter(exists, modules))  # 过滤掉不存在的模块
    num_modules = len(modules)  # 获取模块数量

    if num_modules == 0:  # 如果模块数量为 0
        return nn.Identity()  # 返回一个恒等映射的模块
    elif num_modules == 1:  # 如果模块数量为 1
        return modules[0]  # 返回该模块

    return nn.Sequential(*modules)  # 返回包含所有模块的序列模块

# rms norm

class RMSNorm(Module):  # 定义 RMSNorm 类，用于实现 RMS 归一化
    def __init__(self, dim):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.scale = dim ** 0.5  # 计算缩放因子
        self.gamma = nn.Parameter(torch.ones(dim))  # 创建可学习参数 gamma

    def forward(self, x):  # 前向传播方法
        return F.normalize(x, dim=-1) * self.scale * self.gamma  # 对输入进行归一化并乘以缩放因子和 gamma

# norm wrappers

class PreNorm(Module):  # 定义 PreNorm 类，用于实现预归一化
    def __init__(self, dim, fn: Module):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.fn = fn  # 保存传入的模块
        self.norm = RMSNorm(dim)  # 创建 RMSNorm 归一化模块

    def forward(self, x, **kwargs):  # 前向传播方法
        return self.fn(self.norm(x), **kwargs) + x  # 对输入进行归一化后，再应��传入的模块并加上原始输入

class PostNorm(Module):  # 定义 PostNorm 类，用于实现后归一化
    def __init__(self, dim, fn: Module):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.fn = fn  # 保存传入的模块
        self.norm = nn.LayerNorm(dim)  # 创建 LayerNorm 归一化模块

    def forward(self, x, **kwargs):  # 前向传播方法
        return self.norm(self.fn(x, **kwargs) + x)  # 应用传入的模块后，再对结果进行归一化并加上原始输入

# feedforward

def FeedForward(dim, mult=4):  # 定义 FeedForward 函数，用于创建前馈神经网络
    dim_inner = dim * mult  # 计算内部维度
    return nn.Sequential(  # 返回一个序列模块
        nn.Linear(dim, dim_inner),  # 线性变换层
        nn.GELU(),  # GELU 激活函数
        nn.Linear(dim_inner, dim)  # 线性变换层
    )

# attention

class CausalFullAttention(Module):  # 定义 CausalFullAttention 类，用于实现自回归注意力机制
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        rotary_emb=False,
        add_swish_gating=False,
        data_dependent_rel_pos=False,
        frac_gradient_data_dependent_rel_pos=0.5,
        softmax_normalize=None
    ):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        dim_inner = dim_head * heads  # 计算内部维度
        self.softmax_normalize = default(softmax_normalize, not data_dependent_rel_pos)  # 设置 softmax 归一化参数

        self.scale = dim_head ** -0.5  # 计算缩放因子

        self.rotary_emb = RotaryEmbedding(dim_head) if rotary_emb else None  # 创建旋转嵌入对象（如果需要）

        self.to_qkv = nn.Sequential(  # 创建 Q、K、V 投影模块
            nn.Linear(dim, dim_inner * 3, bias=False),  # 线性变换层
            Rearrange('b n (qkv h d) -> qkv b h n d', h=heads, qkv=3)  # 重排张量维度
        )

        self.data_dependent_rel_pos = data_dependent_rel_pos  # 是否使用数据相关的相对位置编码
        self.frac_gradient_data_dependent_rel_pos = frac_gradient_data_dependent_rel_pos  # 数据相关的相对位置编码的梯度比例

        if data_dependent_rel_pos:  # 如果使用数据相关的相对位置编码
            self.to_a = nn.Sequential(  # 创建相对位置编码模块
                nn.Linear(dim, dim_inner, bias=False),  # 线性变换层
                Rearrange('b n (h d c) -> b h n d c', h=heads, c=2)  # 重排张量维度
            )

        self.to_gates = None  # 初始化门控模块为 None

        if add_swish_gating:  # 如果添加 Swish 门控
            self.to_gates = nn.Sequential(  # 创建门控模块
                nn.Linear(dim, dim_inner, bias=False),  # 线性变换层
                nn.SiLU(),  # Swish 激活函数
                Rearrange('b n (h d) -> b h n d', h=heads)  # 重排张量维度
            )

        self.to_out = nn.Sequential(  # 创建输出模块
            Rearrange('b h n d -> b n (h d)'),  # 重排张量维度
            nn.Linear(dim_inner, dim)  # 线性变换层
        )

    def forward(
        self,
        x,
        ablate_complex=False,
        ablate_state_transition=False
        ):
        # 将输入 x 转换为查询 q、键 k、值 v
        q, k, v = self.to_qkv(x)

        # 如果存在旋转嵌入，则对查询和键进行旋转
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # 缩放查询
        q = q * self.scale

        # 如果启用数据相关的相对位置编码，并且不禁用状态转换
        if self.data_dependent_rel_pos and not ablate_state_transition:
            # 获取数据相关的相对位置投影
            frac_gradient = self.frac_gradient_data_dependent_rel_pos

            # 计算相对位置投影
            a = self.to_a(x)

            # 允许数据相关的相对位置投影变化更慢
            a = a * frac_gradient + a.detach() * (1 - frac_gradient)

            # 将 a 转换为复数形式
            a = torch.view_as_complex(a)

            # 如果禁用复数计算
            if ablate_complex:
                a = a.real + 0.j

            # 计算幅度和相位
            magnitude, phase = a.abs(), a.angle()
            a = torch.polar(magnitude.sigmoid(), phase)

            # 重排形状
            a = rearrange(a, '... -> ... 1')
            a_cumprod = a.cumprod(dim=-2)

            # 对实部进行截断
            a_cumprod_real = a_cumprod.real.clamp(min=1e-10)
            a_cumprod_real_inverse = 1. / a_cumprod_real

            # 重排形状
            q, k = map(lambda t: rearrange(t, '... (d c) -> ... d c', c=2), (q, k))

            # 更新查询和键
            q = q * a_cumprod_real
            k = k * a_cumprod_real_inverse

            # 重排形状
            q, k = map(lambda t: rearrange(t, '... d c -> ... (d c)'), (q, k))

        # 计算相似度
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[2:]
        # 创建因果掩码
        causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)

        # 如果启用 softmax 归一化
        if self.softmax_normalize:
            # 对相似度矩阵进行掩码处理
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
            # 计算注意力权重
            attn = sim.softmax(dim=-1)
        else:
            # 对相似度矩阵进行掩码处理
            attn = sim.masked_fill(causal_mask, 0.)

        # 计算输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 如果存在门控机制
        if exists(self.to_gates):
            # 应用门控机制
            out = out * self.to_gates(x)

        # 返回输出结果
        return self.to_out(out)
# 定义一个函数，实现带有“gateloop操作符”的数据门控线性注意力
def gate_loop_operator(q, k, v, a):
    """
    the pseudocode in section 3.2 of the paper
    """

    # 计算 k 和 v 的张量积
    kv = einsum('b n d, b n e -> b n d e', k, v)
    # 将结果转换为复数张量
    kv = kv + 0.j

    # 定义一个二元操作符函数
    def binary_operator(a, b):
        a_i, kv_i = a
        a_j, kv_j = b
        return a_j * a_i, a_j * kv_i + kv_j

    # 对二元操作符进行关联扫描
    _, kv = associative_scan(binary_operator, (a, kv))

    # 计算最终输出
    return einsum('b n d, b n d e -> b n e', q, kv.real)

# GateLoopedAttention 类，继承自 Module 类
class GateLoopedAttention(Module):
    def __init__(
        self,
        dim,
        heads = None,
        dim_inner = None,
        checkpoint_gate_looped_attn = True,
        add_swish_gating = True,
        sub_ln = False,
        frac_gradient_state_transition = 0.9
    ):
        super().__init__()
        self.frac_gradient_state_transition = frac_gradient_state_transition
        self.checkpoint_gate_looped_attn = checkpoint_gate_looped_attn

        dim_inner = default(dim_inner, dim)
        heads = default(heads, dim_inner)

        # 检查维度是否符合要求
        assert (dim_inner % heads) == 0, f'dimension for gate looped attention {dim_inner} must be divisible by number of gate loop heads {heads}'

        # 将输入张量按照头数进行分割
        self.split_heads = Rearrange('b n (h d) -> (b h) n d', h = heads)

        # 线性变换，将输入转换为 Q、K、V
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        # 线性变换，将输入转换为注意力权重
        self.to_a = nn.Sequential(
            nn.Linear(dim, heads * 2),
            Rearrange('b n (h c) -> (b h) n 1 1 c', h = heads, c = 2)
        )

        # 合并头部
        self.merge_heads = Rearrange('(b h) n d -> b n (h d)', h = heads)

        # 可选的 LayerNorm
        self.maybe_sub_ln = nn.LayerNorm(dim_inner) if sub_ln else nn.Identity()

        self.to_gates = None

        # 添加 Swish 激活门控
        if add_swish_gating:
            self.to_gates = nn.Sequential(
                nn.Linear(dim, dim_inner, bias = False),
                nn.SiLU()
            )

        # 输出线性变换
        self.to_out = nn.Linear(dim_inner, dim, bias = False) if dim_inner != dim or add_swish_gating else nn.Identity()

    # 前向传播函数
    def forward(
        self,
        x,
        ablate_complex = False,
        ablate_state_transition = False
    ):
        frac_gradient = self.frac_gradient_state_transition

        # 将输入 x 转换为 Q、K、V
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(self.split_heads, (q, k, v))

        # 获取注意力权重
        a = self.to_a(x)
        a = a * frac_gradient + a.detach() * (1 - frac_gradient)

        # 将注意力权重转换为复数张量
        a = torch.view_as_complex(a)

        # 如果 ablate_complex 为 True，则将注意力权重转换为实部
        if ablate_complex:
            a = a.real + 0.j

        # 如果 ablate_state_transition 为 True，则将注意力权重设置为全 1
        if ablate_state_transition:
            a = torch.ones_like(a.real) + 0.j
        else:
            # 对状态转换的激活函数
            # 使用 sigmoid 函数处理幅度，使用恒等函数处理相位
            magnitude, phase = a.abs(), a.angle()
            a = torch.polar(magnitude.sigmoid(), phase)

        # 检查是否需要反向传播
        need_backwards = any([t.requires_grad for t in (q, k, v, a)])

        # 使用 partial 函数创建一个带有检查点的函数
        fn = partial(checkpoint, gate_loop_operator) if need_backwards and self.checkpoint_gate_looped_attn else gate_loop_operator

        # 计算输出
        out = fn(q, k, v, a)

        out = self.merge_heads(out)

        out = self.maybe_sub_ln(out)

        # 如果存在门控，则将门控应用到输出上
        if exists(self.to_gates):
            out = self.to_gates(x) * out

        return self.to_out(out)

# Transformer 类，继承自 Module 类
class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        checkpoint_gate_looped_attn = True,
        use_gate_looped_attn = True,
        gate_loop_heads = None,
        attn_add_swish_gating = True,
        dim_gate_looped_attn = None,
        attn_softmax_normalize = None,
        data_dependent_rel_pos = False,
        frac_gradient_state_transition = 0.9,
        ablate_complex = False,
        ablate_state_transition = False,
        rotary_emb = False,
        post_ln_norm = False,
        sub_ln = False
    # 初始化函数，设置模型的参数
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置是否削弱复杂性和状态转换的参数
        self.ablate_complex = ablate_complex
        self.ablate_state_transition = ablate_state_transition

        # 创建一个词嵌入层
        self.token_emb = nn.Embedding(num_tokens, dim)

        # 创建一个模块列表用于存储每个层的注意力和前馈网络
        layers = ModuleList([])

        # 根据是否后层归一化选择层包装器
        layer_wrapper = PreNorm if not post_ln_norm else PostNorm

        # 循环创建指定深度的层
        for _ in range(depth):

            # 根据是否使用门控循环注意力选择空间混合器类型
            if use_gate_looped_attn:
                spatial_mixer = GateLoopedAttention(
                    dim = dim,
                    heads = gate_loop_heads,
                    dim_inner = dim_gate_looped_attn,
                    add_swish_gating = attn_add_swish_gating,
                    sub_ln = sub_ln,
                    checkpoint_gate_looped_attn = checkpoint_gate_looped_attn,
                    frac_gradient_state_transition = frac_gradient_state_transition
                )
            else:
                spatial_mixer = CausalFullAttention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    rotary_emb = rotary_emb,
                    add_swish_gating = attn_add_swish_gating,
                    softmax_normalize = attn_softmax_normalize,
                    data_dependent_rel_pos = data_dependent_rel_pos,
                    frac_gradient_data_dependent_rel_pos = frac_gradient_state_transition
                )

            # 创建通道混合器
            channelwise_mixer = FeedForward(
                dim = dim,
                mult = ff_mult
            )

            # 将空间混合器和通道混合器添加到层列表中
            layers.append(ModuleList([
                layer_wrapper(dim, spatial_mixer),
                layer_wrapper(dim, channelwise_mixer)
            ]))

        # 将层列表转换为模块列表
        self.layers = ModuleList(layers)

        # 创建输出层，包括 RMS 归一化和线性层
        self.to_logits = Sequential(
            RMSNorm(dim) if not post_ln_norm else None,
            nn.Linear(dim, num_tokens, bias = False)
        )

    # 前向传播函数
    def forward(
        self,
        x,
        return_loss = False,
        ablate_complex = None,
        ablate_state_transition = None
    ):
        # 设置是否削弱复杂性和状态转换的参数
        ablate_complex = default(ablate_complex, self.ablate_complex)
        ablate_state_transition = default(ablate_state_transition, self.ablate_state_transition)

        # 如果需要返回损失，则提取标签
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # 对输入进行词嵌入
        x = self.token_emb(x)

        # 遍历每个层的注意力和前馈网络
        for attn, ff in self.layers:
            # 使用注意力层
            x = attn(
                x,
                ablate_complex = ablate_complex,
                ablate_state_transition = ablate_state_transition
            )

            # 使用前馈网络
            x = ff(x)

        # 获取最终输出
        logits = self.to_logits(x)

        # 如果不需要返回损失，则直接返回输出
        if not return_loss:
            return logits

        # 重新排列输出并计算交叉熵损失
        logits = rearrange(logits, 'b n c -> b c n')
        return F.cross_entropy(logits, labels)
```