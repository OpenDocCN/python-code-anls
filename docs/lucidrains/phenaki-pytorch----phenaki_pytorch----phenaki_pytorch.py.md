# `.\lucidrains\phenaki-pytorch\phenaki_pytorch\phenaki_pytorch.py`

```
# 导入数学库
import math
# 导入 functools 库
import functools
# 从 contextlib 库中导入 nullcontext
from contextlib import nullcontext
# 从 functools 库中导入 partial 和 wraps
from functools import partial, wraps

# 从 typing 模块中导入 Optional, List, Union
from typing import Optional, List, Union
# 从 beartype 库中导入 beartype
from beartype import beartype

# 导入 torch 库
import torch
# 从 torch.nn.functional 中导入 F
import torch.nn.functional as F
# 从 torch 中导入 nn, einsum
from torch import nn, einsum

# 从 einops 库中导入 rearrange, repeat, pack, unpack
from einops import rearrange, repeat, pack, unpack
# 从 einops.layers.torch 中导入 Rearrange
from einops.layers.torch import Rearrange

# 从 phenaki_pytorch.t5 中导入 t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME
from phenaki_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

# 从 phenaki_pytorch.cvivit 中导入 CViViT
from phenaki_pytorch.cvivit import CViViT
# 从 phenaki_pytorch.attention 中导入 Attention, Transformer, ContinuousPositionBias

# helpers

# 定义函数 exists，判断值是否存在
def exists(val):
    return val is not None

# 定义函数 default，返回值或默认值
def default(val, d):
    return val if exists(val) else d

# 定义函数 cast_tuple，将值转换为元组
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else (val,) * length

# 定义函数 reduce_mult，对数组中的元素进行累乘
def reduce_mult(arr):
    return functools.reduce(lambda x, y: x * y, arr)

# 定义函数 divisible_by，判断两个数是否整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# tensor helpers

# 定义函数 get_mask_subset_with_prob，根据概率获取掩码子集
def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device

    num_tokens = mask.sum(dim = -1)
    num_pads = seq_len - num_tokens
    num_masked = (prob * num_tokens).round().clamp(min = 1)

    randperm_indices = torch.rand((batch, seq_len), device = device).argsort(dim = -1)
    randperm_indices -= rearrange(num_pads, 'b -> b 1')
    randperm_indices.masked_fill_(randperm_indices < 0, seq_len) # set to max out of bounds, so never chosen

    mask_subset = randperm_indices < rearrange(num_masked, 'b -> b 1')
    return mask_subset

# decorators

# 定义装饰器 eval_decorator，用于在评估模型时切换模型状态
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# classifier free guidance functions

# 定义函数 uniform，生成指定形状的均匀分布张量
def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

# 定义函数 prob_mask_like，生成概率掩码张量
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# tensor helper functions

# 定义函数 log，计算张量的对数
def log(t, eps = 1e-10):
    return torch.log(t + eps)

# sampling helpers

# 定义函数 gumbel_noise，生成古贝尔噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 定义函数 gumbel_sample，使用古贝尔噪声进行采样
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# 定义函数 top_k，根据阈值获取前 k 个概率最大的位置
def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# mask git

# 定义 MaskGit 类
class MaskGit(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        gradient_shrink_alpha = 0.1,
        heads = 8,
        dim_head = 64,
        unconditional = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        **kwargs
    # 初始化函数，设置模型的维度、mask_id、是否无条件生成等参数
    ):
        super().__init__()
        self.dim = dim

        self.mask_id = num_tokens
        self.unconditional = unconditional

        # 创建 token embedding 层，num_tokens + 1 个 token，最后一个用作 mask_id
        self.token_emb = nn.Embedding(num_tokens + 1, dim)

        self.max_seq_len = max_seq_len
        # 创建位置编码 embedding 层
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # 设置梯度缩放参数
        self.gradient_shrink_alpha = gradient_shrink_alpha

        # 创建连续位置偏置
        self.continuous_pos_bias = ContinuousPositionBias(dim = dim_head, heads = heads, num_dims = 3)

        # 创建 Transformer 模型
        self.transformer = Transformer(
            dim = dim,
            attn_num_null_kv = 2,
            has_cross_attn = not self.unconditional,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            **kwargs
        )

        # 创建输出层，将 dim 维度映射到 num_tokens
        self.to_logits = nn.Linear(dim, num_tokens)

    # 带条件缩放的前向传播函数
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,
        **kwargs
    ):
        # 调用前向传播函数，cond_drop_prob 为 0
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        # 调用前向传播函数，cond_drop_prob 为 1
        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    # 前向传播函数
    def forward(
        self,
        x,
        cond_drop_prob = 0.,
        text_mask = None,
        video_mask = None,
        video_patch_shape = None,
        return_embeds = False,
        **kwargs
    ):
        assert x.ndim in {2, 4}, 'video token ids must be of shape (batch, seq) or (batch, frame, height, width)'

        if x.ndim == 4:
            video_patch_shape = x.shape[1:]
            x = rearrange(x, 'b ... -> b (...)')

        b, n, device = *x.shape, x.device

        # 如果 text_mask 不存在，则创建全为 True 的 mask
        if not exists(text_mask):
            text_mask = torch.ones((b, n), device = device, dtype = torch.bool)

        assert exists(video_patch_shape), 'video patch shape must be given'

        # 计算相对位置偏置
        rel_pos_bias = self.continuous_pos_bias(*video_patch_shape, device = device)

        # 如果 cond_drop_prob 大于 0，则生成保留 mask
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        video_shape = (b, *video_patch_shape)

        # 对输入进行 token embedding
        x = self.token_emb(x)

        # 断言视频 token 序列长度不超过 max_seq_len
        assert n <= self.max_seq_len, f'the video token sequence length you are passing in ({n}) is greater than the `max_seq_len` ({self.max_seq_len}) set on your `MaskGit`'
        x = self.pos_emb(torch.arange(n, device = device)) + x

        # 梯度缩放
        x = x * self.gradient_shrink_alpha + x.detach() * (1 - self.gradient_shrink_alpha)

        # Transformer 模型的前向传播
        x = self.transformer(
            x,
            video_shape = video_shape,
            attn_bias = rel_pos_bias,
            self_attn_mask = video_mask,
            cross_attn_context_mask = text_mask,
            **kwargs
        )

        # 如果需要返回嵌入向量，则直接返回
        if return_embeds:
            return x

        return self.to_logits(x)
# 定义 TokenCritic 类，继承自 nn.Module
class TokenCritic(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 维度
        num_tokens,  # token 数量
        max_seq_len,  # 最大序列长度
        has_cross_attn = False,  # 是否有跨注意力
        attn_dropout = 0.,  # 注意力丢弃率
        ff_dropout = 0.,  # FeedForward 层丢弃率
        **kwargs
    ):
        super().__init__()
        self.has_cross_attn = has_cross_attn

        self.mask_id = num_tokens  # 定义 mask_id 为 num_tokens

        self.token_emb = nn.Embedding(num_tokens + 1, dim)  # 创建 token 的嵌入层，最后一个 token 用作 mask_id
        self.pos_emb = nn.Embedding(max_seq_len, dim)  # 创建位置嵌入层

        self.transformer = Transformer(
            dim = dim,
            peg = True,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            has_cross_attn = has_cross_attn,
            **kwargs
        )  # 创建 Transformer 模型

        self.to_logits = nn.Sequential(
            nn.Linear(dim, 1),  # 线性层
            Rearrange('... 1 -> ...')  # 重排维度
        )  # 创建输出 logits 的序列

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,  # 条件缩放
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)  # 调用 forward 方法获取 logits

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)  # 调用 forward 方法获取 null_logits
        return null_logits + (logits - null_logits) * cond_scale  # 返回根据条件缩放计算后的结果

    def forward(
        self,
        x,
        text_mask = None,
        cond_drop_prob = None,
        context = None,
        video_mask = None,
        video_patch_shape = None,
        **kwargs
    ):
        if exists(video_patch_shape):
            video_shape = (x.shape[0], *video_patch_shape)
        else:
            video_shape = x.shape

        x = rearrange(x, 'b ... -> b (...)')  # 重排输入数据的维度
        b, n, device = *x.shape, x.device

        if not exists(text_mask):
            text_mask = torch.ones((b, n), device = device, dtype = torch.bool)  # 如果不存在文本 mask，则创建全为 True 的 mask

        if exists(context) and cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)  # 根据条件概率创建 mask
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask  # ���新文本 mask

        x = self.token_emb(x)  # 对输入数据进行 token 嵌入
        x = self.pos_emb(torch.arange(n, device = device)) + x  # 添加位置嵌入

        x = self.transformer(
            x,
            video_shape = video_shape,
            context = context,
            self_attn_mask = video_mask,
            cross_attn_context_mask = text_mask,
            **kwargs
        )  # 调用 Transformer 模型进行计算

        return self.to_logits(x)  # 返回 logits

# 定义 SelfCritic 类，继承自 nn.Module，受 Nijkamp 等人启发
@beartype
class SelfCritic(nn.Module):
    def __init__(
        self,
        maskgit: MaskGit  # 接收 MaskGit 类型参数
    ):
        super().__init__()
        self.maskgit = maskgit

        self.to_pred = nn.Sequential(
            nn.Linear(maskgit.dim, 1),  # 线性层
            Rearrange('... 1 -> ...')  # 重排维度
        )  # 创建输出预测的序列

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,  # 条件缩放
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)  # 调用 forward 方法获取 logits

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)  # 调用 forward 方法获取 null_logits
        return null_logits + (logits - null_logits) * cond_scale  # 返回根据条件缩放计算后的结果

    def forward(self, x, *args, **kwargs):
        embeds = self.maskgit(x, *args, return_embeds = True, **kwargs)  # 调用 maskgit 方法获取嵌入
        return self.to_pred(embeds)  # 返回预测结果

# 定义 Phenaki 类，继承自 nn.Module
@beartype
class Phenaki(nn.Module):
    def __init__(
        self,
        *,
        maskgit: MaskGit,  # MaskGit 类型参数
        cvivit: CViViT,  # CViViT 类型参数
        critic: Optional[Union[TokenCritic, SelfCritic]] = None,  # 可选的 TokenCritic 或 SelfCritic 类型参数
        steps = 18,  # 步数
        t5_name = DEFAULT_T5_NAME,  # T5 模型名称
        sample_temperature = 0.,  # 采样温度
        text_embed_dim = None,  # 文本嵌入维度
        cond_drop_prob = 0.25,  # 条件丢弃概率
        max_text_len = 128,  # 最大文本长度
        self_token_critic = False,  # 是否使用自身 TokenCritic
        critic_loss_weight = 1.,  # TokenCritic 权重
        critic_noise_anneal_schedule = 'decay',  # TokenCritic 噪声退火计划
        critic_train_sample_temperature = 1.  # TokenCritic 训练采样温度
    # 初始化函数，继承父类的初始化方法
    def __init__(self):
        super().__init__()

        # 复制cvivit用于评估
        self.cvivit = cvivit.copy_for_eval()

        # 设置maskgit属性
        self.maskgit = maskgit
        self.unconditional = maskgit.unconditional

        # 设置mask_id属性
        self.mask_id = maskgit.mask_id

        # 断言条件，确保self_token_critic和critic不存在，或者critic存在
        assert not (self_token_critic and exists(critic))

        # 如果self_token_critic为真，则创建SelfCritic对象
        if self_token_critic:
            critic = SelfCritic(maskgit)

        # 如果critic存在，则将其设置为评估模式
        if exists(critic):
            critic = critic.eval()

        # 断言条件，确保critic不存在或者self_token_critic为真，或者maskgit.unconditional为假且critic具有交叉注意力
        assert not exists(critic) or self_token_critic or (not maskgit.unconditional) == critic.has_cross_attn

        # 设置critic相关属性
        self.critic = critic
        self.critic_noise_anneal_schedule = critic_noise_anneal_schedule
        self.critic_loss_weight = critic_loss_weight
        self.critic_train_sample_temperature = critic_train_sample_temperature

        # 设置步数和采样温度
        self.steps = steps
        self.sample_temperature = sample_temperature

        # 文本条件
        text_embed_dim = default(text_embed_dim, get_encoded_dim(t5_name))
        self.encode_texts = partial(t5_encode_text, name = t5_name)
        self.text_embed_dim = text_embed_dim
        self.max_text_len = max_text_len

        # 断言条件，确保cond_drop_prob大于0
        assert cond_drop_prob > 0.
        # 设置cond_drop_prob属性，用于transformers的分类器自由引导
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb

    # 采样图像函数
    def sample_images(
        self,
        *,
        texts: Union[List[str], str] = None,
        batch_size = 1,
        cond_scale = 3.,
        starting_temperature = 0.9,
        noise_K = 1.
    ):
        # 生成单帧视频
        single_framed_video = self.sample(
            texts = texts,
            num_frames = 1,
            cond_scale = cond_scale,
            starting_temperature = starting_temperature,
            noise_K = noise_K
        )

        # 重新排列视频维度
        return rearrange(single_framed_video, '... c 1 h w')

    # 采样函数
    @eval_decorator
    @torch.no_grad()
    def sample(
        self,
        *,
        num_frames,
        texts: Union[List[str], str] = None,
        prime_frames = None,
        batch_size = 1,
        cond_scale = 3.,
        starting_temperature = 0.9,
        noise_K = 1. # 用于token-critic论文第3.2节中critic分数的噪声超参数，需要找到正确的值
    def forward(
        self,
        videos = None,
        *,
        texts: Optional[List[str]] = None,
        video_codebook_ids = None,
        video_frame_mask = None,
        text_embeds = None,
        cond_drop_prob = None,
        only_train_generator = False,
        only_train_critic = False
# 定义一个名为 make_video 的函数，用于生成视频

@beartype
# 使用 beartype 装饰器对函数参数进行类型检查
def make_video(
    phenaki: Phenaki,  # 接受 Phenaki 对象作为参数
    texts: List[str],  # 接受一个字符串列表作为参数
    num_frames,  # 接受一个整数作为参数，表示帧数
    prime_lengths  # 接受一个整数或整数元组作为参数，表示前置长度
):
    num_scenes = len(texts)  # 获取文本列表的长度，即场景数
    num_frames = cast_tuple(num_frames, num_scenes)  # 将 num_frames 转换为元组，长度与场景数相同

    prime_lengths = cast_tuple(prime_lengths, num_scenes - 1)  # 将 prime_lengths 转换为元组，长度为场景数减一
    prime_lengths = (*prime_lengths, 0)  # 在 prime_lengths 元组末尾添加一个 0，表示最后一个场景无需前置长度

    entire_video = []  # 初始化整个视频列表
    video_prime = None  # 初始化视频前置
    scenes = []  # 初始化场景列表

    # 遍历文本、帧数、前置长度三个参数的元素，生成视频
    for text, scene_num_frames, next_scene_prime_length in zip(texts, num_frames, prime_lengths):
        # 从 Phenaki 对象中生成视频，传入文本、视频前置、场景帧数
        video = phenaki.sample(texts=text, prime_frames=video_prime, num_frames=scene_num_frames)
        scenes.append(video)  # 将生成的视频添加到场景列表中

        video_prime = video[:, :, -next_scene_prime_length:]  # 更新视频前置为当前视频的最后 next_scene_prime_length 帧

    # 将所有场景的视频拼接在一起，沿着第二维度拼接，返回拼接后的视频和场景列表
    return torch.cat(scenes, dim=2), scenes
```