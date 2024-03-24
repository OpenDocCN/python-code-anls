# `.\lucidrains\mirasol-pytorch\mirasol_pytorch\mirasol_pytorch.py`

```py
# 导入所需的模块和函数
import operator
from functools import partial
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum
from torch.nn import Module, ModuleList

# 导入 beartype 模块和相关类型
from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any

# 导入 einops 相关函数和层
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# 导入 x_transformers 相关模块和类
from x_transformers import (
    Encoder,
    Decoder,
    TransformerWrapper,
    AutoregressiveWrapper
)

# 导入 x_transformers 中的 RotaryEmbedding 类
from x_transformers.x_transformers import RotaryEmbedding

# 导入 mirasol_pytorch 中的分布式函数
from mirasol_pytorch.distributed import all_gather, get_is_distributed

# 辅助函数

# 判断变量是否存在
def exists(v):
    return v is not None

# 返回参数中第一个存在的值
def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

# 判断一个数是否可以被另一个数整除
def divisible_by(num, den):
    return (num % den) == 0

# 判断参数中只有一个为 True
def only_one_true(*bools):
    return sum(*[map(int, bools)]) == 1

# 将张量打包成指定模式
def pack_one(t, pattern):
    return pack([t], pattern)

# 将打包的张量解包成指定模式
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 张量操作函数

# 计算张量的 L2 范数
def l2norm(t):
    return F.normalize(t, dim = -1)

# 计算张量之间的余弦相似度损失
def cosine_sim_loss(x, y):
    x, y = map(l2norm, (x, y))
    return 1. - einsum('b n d, b n d -> b n', x, y).mean()

# 生成位置编码的正弦和余弦值
def posemb_sincos_nd(
    t: Tensor,
    temperature: int = 10000,
    dtype = torch.float32
):
    b, *dims, feat_dim, device = *t.shape, t.device
    seq_len = torch.tensor(dims).cumprod(dim = -1)[-1].item()

    arange = partial(torch.arange, device = device)

    num_dims = len(dims)
    two_times_num_dims = 2 * num_dims # 2 because sin and cos of same position

    rounded_feat_dim = feat_dim // num_dims * num_dims
    feat_dim_remainder = feat_dim % num_dims

    omega = arange(rounded_feat_dim // two_times_num_dims) / (rounded_feat_dim // two_times_num_dims - 1)
    omega = 1.0 / (temperature ** omega)
    meshed = torch.meshgrid(*[*map(arange, dims)], indexing = 'ij')

    pos = torch.cat(tuple(m.flatten()[..., None] for m in meshed), dim = 0)
    pos = pos * omega[None, :]

    pos = torch.cat((pos.sin(), pos.cos()))

    pos = rearrange(pos, '(n f) d -> n (f d)', n = seq_len)
    pos = pos.type(dtype)

    return F.pad(pos, (0, feat_dim_remainder))

# 生成具有一定概率的掩码张量
def mask_with_prob(
    shape: Tuple[int, ...],
    prob: float,
    device = None
) -> Tensor:
    length = shape[-1]
    num_mask = int(prob * length)
    randperm = torch.randn(shape, device = device).argsort(dim = -1)
    return randperm >= num_mask

# 主类

# 定义 Losses 命名元组，包含不同类型的损失
Losses = namedtuple('Losses', [
    'text_autoregressive',
    'av_autoregressive',
    'av_recon',
    'text_av_sim_reg'
])

# Mirasol 类，继承自 Module 类
class Mirasol(Module):

    @beartype
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        *,
        dim,
        num_text_tokens,
        video_image_size,
        video_frames_per_timechunk,
        audio_freq_dim,
        audio_time_dim_per_timechunk,
        audio_patch_size: Tuple[int, int],                          # 音频补丁大小 (频率, 时间)
        video_patch_size: Tuple[int, int],                          # 视频补丁大小 (空间, 时间)
        video_recon_patch_size: Optional[Tuple[int, int]] = None,   # 视频重建补丁大小 (空间, 时间) - 用于重建损失的较小视频
        video_recon_interpolate_mode = 'nearest',
        audio_encoder: Union[Module, Dict[str, Any]],
        video_encoder: Union[Module, Dict[str, Any]],
        num_audio_video_register_tokens = 8,                        # 音频视频注册令牌数量 https://arxiv.org/abs/2309.16588
        audio_video_mask_prob = 0.15,                         # 在论文中，他们使用了被屏蔽的令牌，但从伯克利遗忘-因果-掩码论文中，一个简单的键值掩码应该足够
        text_max_seq_len = 2048,
        text_forgetful_causal_mask_prob = 0.1,                      # https://arxiv.org/abs/2210.13432
        encoder_depth = 6,
        decoder_depth = 6,
        combiner_depth = 2,
        combiner_output_num_tokens = 3,
        video_channels = 3,
        attn_dim_head = 64,
        attn_heads = 8,
        flash_attn = True,
        attn_layers_kwargs: dict = dict(),
        combiner: Optional[Module] = None,
        combiner_kwargs: dict = dict(),
        autoregressive_wrapper_kwargs: dict = dict(
            pad_value = 0,
            ignore_index = -100
        ),
        av_autoregressive_loss_weight = 1.,
        av_reconstruction_loss_weight = 1.,
        sim_reg_loss_weight = 0.
    
    # 返回设备信息
    @property
    def device(self):
        return next(self.parameters()).device

    # 生成函数，用于生成序列
    @torch.no_grad()
    def generate(
        self,
        *,
        seq_len: int,
        prompt: Optional[Tensor] = None,
        **kwargs
    ):
        was_training = self.training
        self.eval()

        assert 'generate' not in kwargs
        assert 'generate_seq_len' not in kwargs

        # 调用前向传播函数生成序列
        out = self.forward(
            text = prompt,
            generate = True,
            generate_seq_len = seq_len,
            **kwargs
        )

        self.train(was_training)
        return out

    # 前向传播函数，接收输入并返回输出
    @beartype
    def forward(
        self,
        *,
        audio: Optional[Tensor] = None,
        video: Optional[Tensor] = None,
        encoded_audio: Optional[Tensor] = None,
        encoded_video: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
        return_loss = True,
        return_loss_breakdown = False,
        generate = False,
        generate_seq_len = None
```