# `.\lucidrains\rvq-vae-gpt\rvq_vae_gpt\rvq_vae_gpt.py`

```
# 导入 torch 库
import torch
# 导入 torch.nn.functional 模块并重命名为 F
import torch.nn.functional as F
# 从 torch 模块中导入 nn、einsum
from torch import nn, einsum

# 从 einops 库中导入 rearrange、repeat、pack、unpack
from einops import rearrange, repeat, pack, unpack
# 从 einops.layers.torch 模块中导入 Rearrange
from einops.layers.torch import Rearrange

# 导入自定义的 local_attention 模块中的 LocalMHA 类
from local_attention import LocalMHA
# 导入自定义的 vector_quantize_pytorch 模块中的 VectorQuantize、ResidualVQ 类
from vector_quantize_pytorch import VectorQuantize, ResidualVQ

# 从 beartype 库中导入 beartype、Tuple、Optional、Union
from beartype import beartype
from beartype.typing import Tuple, Optional, Union

# 从 pathlib 模块中导入 Path 类
from pathlib import Path
# 导入 pickle 库
import pickle

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 获取迭代器的第一个元素
def first(it):
    return it[0]

# 返回第一个存在的值
def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

# 判断一个数是否可以被另一个数整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 将输入转换为元组
def cast_tuple(t, len = 1):
    return ((t,) * len) if not isinstance(t, tuple) else t

# token shift - RWKV 中使用

# 将输入张量按照最后一个维度分割成两部分，并进行位移
def shift_tokens(t):
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1), value = 0.)
    return torch.cat((t, t_shift), dim = -1)

# 前馈网络

# GEGLU 激活函数
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

# 创建前馈网络模块
def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )

# 最佳的上采样和下采样方式

# 上采样模块
class Upsample(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        factor = 2
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        linear = nn.Linear(dim, dim_out * factor)

        self.net = nn.Sequential(
            linear,
            nn.SiLU(),
            Rearrange('b n (p d) -> b (n p) d', p = factor)
        )

        self.factor = factor
        self.init_(linear)

    # 初始化线性层的权重和偏置
    def init_(self, linear):
        o, i = linear.weight.shape

        linear_weight = torch.empty(o // self.factor, i)
        nn.init.kaiming_uniform_(linear_weight)

        linear_weight = repeat(linear_weight, 'o ... -> (o r) ...', r = self.factor)

        linear_weight.data.copy_(linear_weight)
        nn.init.zeros_(linear.bias.data)

    def forward(self, x):
        return self.net(x)

# 下采样模块
def Downsample(
    dim,
    dim_out = None,
    factor = 2
):
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b (n p) d -> b n (p d)', p = factor),
        nn.Linear(dim * factor, dim_out)
    )

# 本地注意力

# 本地 Transformer 模块
class LocalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        dim_head,
        window_size
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LocalMHA(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    qk_rmsnorm = True,
                    window_size = window_size,
                    use_rotary_pos_emb = True,
                    use_xpos = True,
                    causal = True
                ),
                FeedForward(dim = dim)
            ]))

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(shift_tokens(x)) + x
            x = ff(shift_tokens(x)) + x

        return x

# 模块

# 文本 VQ-VAE 模型
@beartype
class TextVQVAE(nn.Module): # 或者基因组，最终，将 num_tokens 设置为 4
    def __init__(
        self,
        *,
        num_tokens,
        dim: Union[int, Tuple[int, ...]],
        depth: Union[int, Tuple[int, ...]],
        strides: Union[int, Tuple[int, ...]],
        codebook_size = 1024,
        local_attn_window_size = 32,
        local_attn_heads = 8,
        local_attn_dim_head = 64,
        num_codebooks = 4,
        vq_decay = 0.9,
        rvq_quantize_dropout = True
    # 初始化函数，继承父类的初始化方法
    def __init__(
        self,
        vq_decay,
        strides,
        dim,
        depth,
        local_attn_window_size,
        num_tokens,
        local_attn_heads,
        local_attn_dim_head,
        num_codebooks,
        codebook_size,
        rvq_quantize_dropout
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 获取当前函数的局部变量
        config = locals()
        # 移除不需要的变量
        config.pop('self')
        config.pop('__class__')
        # 将配置信息保存到实例变量中
        self._config = config

        # 断言 vq_decay 的取值范围
        assert 0 < vq_decay <= 1.

        # 将 strides 转换为元组
        strides = cast_tuple(strides)
        num_layers = len(strides)

        # 将 dim、depth、local_attn_window_size 转换为元组
        dim = cast_tuple(dim, num_layers)
        depth = cast_tuple(depth, num_layers)
        local_attn_window_size = cast_tuple(local_attn_window_size, num_layers)

        # 断言各参数长度一致
        assert num_layers == len(depth) == len(local_attn_window_size) == len(dim)

        # 获取初始维度和 VQ 维度
        init_dim, vq_dim = dim[0], dim[-1]

        # 构建维度列表和维度对
        dims = [first(dim), *dim]
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        # 创建 token embedding 层
        self.token_emb = nn.Embedding(num_tokens, init_dim)

        # 计算总步长
        self.total_strides = torch.tensor(list(strides)).cumprod(dim = -1)[-1].item()

        # 初始化 encoder
        self.encoder = nn.ModuleList([])

        # 构建每一层的参数元组
        layer_params = tuple(zip(
            strides,
            depth,
            local_attn_window_size,
            dim_pairs
        ))

        # 初始化初始 transformer
        self.init_transformer = LocalTransformer(
            dim = init_dim,
            depth = first(depth),
            heads = local_attn_heads,
            dim_head = local_attn_dim_head,
            window_size = first(local_attn_window_size)
        )

        # 初始化最终 transformer
        self.final_transformer = LocalTransformer(
            dim = init_dim,
            depth = first(depth),
            heads = local_attn_heads,
            dim_head = local_attn_dim_head,
            window_size = first(local_attn_window_size)
        )

        # 遍历每一层参数，构建 encoder
        for layer_stride, layer_depth, layer_local_attn_window_size, (dim_in, dim_out) in layer_params:
            self.encoder.append(nn.ModuleList([
                Downsample(dim = dim_in, dim_out = dim_out, factor = layer_stride),
                LocalTransformer(
                    dim = dim_out,
                    depth = layer_depth,
                    heads = local_attn_heads,
                    dim_head = local_attn_dim_head,
                    window_size = layer_local_attn_window_size
                )
            ]))

        # 初始化 encoder_norm
        self.encoder_norm = nn.LayerNorm(vq_dim)

        # 初始化 VQ
        self.vq = ResidualVQ(
            dim = vq_dim,
            num_quantizers = num_codebooks,
            codebook_size = codebook_size,
            decay = vq_decay,
            quantize_dropout = num_codebooks > 1 and rvq_quantize_dropout,
            commitment_weight = 0.,   # the weight on the commitment loss
            kmeans_init = True,
            kmeans_iters = 10
        )

        # 初始化 decoder
        self.decoder = nn.ModuleList([])

        # 遍历每一层参数，构建 decoder
        for layer_stride, layer_depth, layer_local_attn_window_size, (dim_in, dim_out) in reversed(layer_params):
            self.decoder.append(nn.ModuleList([
                Upsample(dim = dim_out, dim_out = dim_in, factor = layer_stride),
                LocalTransformer(
                    dim = dim_out,
                    depth = layer_depth,
                    heads = local_attn_heads,
                    dim_head = local_attn_dim_head,
                    window_size = layer_local_attn_window_size
                )
            ]))

        # 初始化 to_logits
        self.to_logits = nn.Sequential(
            nn.LayerNorm(init_dim),
            nn.Linear(init_dim, num_tokens)
        )

    # 保存模型
    def save(self, path):
        path = Path(path)
        pkg = dict(
            model = self.state_dict(),
            config = pickle.dumps(self._config)
        )
        torch.save(pkg, str(path))

    # 加载模型
    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))
        self.load_state_dict(pkg['model'])

    # 初始化并加载模型
    @classmethod
    def init_and_load(cls, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))
        model = cls(**pickle.loads(pkg['config']))
        model.load(path)
        return model

    # 获取设备信息
    @property
    def device(self):
        return next(self.parameters()).device
    # 编码器，将输入的ids转换为tokens
    def encode(self, ids):
        # 使用token_emb方法将ids转换为tokens
        tokens = self.token_emb(ids)

        # 使用init_transformer方法对tokens进行初始化转换
        tokens = self.init_transformer(tokens)

        # 遍历编码器中的每个层，进行下采样和局部注意力操作
        for downsample, local_attn in self.encoder:
            tokens = downsample(tokens)
            tokens = local_attn(tokens)

        # 对编码后的tokens进行编码器归一化
        return self.encoder_norm(tokens)

    # 解码器，将codes解码为logits
    def decode(self, codes):
        # 将codes赋值给tokens
        tokens = codes

        # 遍历解码器中的每个层，进行局部注意力和上采样操作
        for upsample, local_attn in self.decoder:
            tokens = local_attn(tokens)
            tokens = upsample(tokens)

        # 对解码后的tokens进行最终转换
        tokens = self.final_transformer(tokens)

        # 将tokens转换为logits
        logits = self.to_logits(tokens)
        return logits

    # 从codebook_ids解码得到logits
    @torch.no_grad()
    def decode_from_codebook_ids(self, codebook_ids):
        # 使用vq对象的get_codes_from_indices方法将codebook_ids转换为codes
        codes = self.vq.get_codes_from_indices(codebook_ids)
        # 调用decode方法解码codes得到logits
        return self.decode(codes)

    # 整体前向传播过程
    def forward(
        self,
        ids,
        return_codebook_indices = False,
        return_reconstruction = False,
        return_loss_breakdown = False
    ):
        # 获取ids的batch和seq长度
        batch, seq = ids.shape
        # 断言seq能够被total_strides整除
        assert divisible_by(seq, self.total_strides)

        # 将ids移动到设备上
        ids = ids.to(self.device)

        # 对ids进行编码得到tokens
        tokens = self.encode(ids)

        # 对tokens进行向量量化操作，返回更新后的tokens、indices和loss
        tokens, indices, _ = self.vq(tokens)

        # 如果需要返回codebook_indices，则直接返回indices
        if return_codebook_indices:
            return indices

        # 对tokens进行解码得到logits
        logits = self.decode(tokens)

        # 将logits重新排列为 'b c n' 的形式
        logits = rearrange(logits, 'b n c -> b c n')

        # 计算交叉熵损失
        loss = F.cross_entropy(
            logits,
            ids
        )

        # 如果需要返���重构结果，则返回loss和logits的argmax值
        if return_reconstruction:
            return loss, logits.argmax(dim = 1)

        # 返回loss
        return loss
# 定义一个名为Transformer的类，表示层次结构的变换器
class Transformer(nn.Module):
    pass
```