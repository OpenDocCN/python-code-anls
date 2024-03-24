# `.\lucidrains\spear-tts-pytorch\spear_tts_pytorch\spear_tts_pytorch.py`

```py
# 导入数学库
import math
# 从路径库中导入路径类
from pathlib import Path
# 从 functools 库中导入 partial 函数
from functools import partial
# 从 random 库中导入 random 函数
from random import random

# 导入 torch 库
import torch
# 从 torch.nn.functional 中导入 F
import torch.nn.functional as F
# 从 torch.nn.utils.rnn 中导入 pad_sequence
from torch.nn.utils.rnn import pad_sequence
# 从 torch 中导入 Tensor, nn, einsum, IntTensor, LongTensor
from torch import Tensor, nn, einsum, IntTensor, LongTensor

# 从 torch.nn 中导入 Module, ModuleList
from torch.nn import Module, ModuleList

# 从 torch.utils.data 中导入 Dataset
from torch.utils.data import Dataset

# 从 einops 中导入 rearrange, repeat, pack, reduce
from einops import rearrange, repeat, pack, reduce
# 从 einops.layers.torch 中导入 Rearrange
from einops.layers.torch import Rearrange

# 从 audiolm_pytorch 中导入 FairseqVQWav2Vec, HubertWithKmeans
from audiolm_pytorch import FairseqVQWav2Vec, HubertWithKmeans
# 从 audiolm_pytorch.data 中导入 get_dataloader
from audiolm_pytorch.data import get_dataloader

# 从 rotary_embedding_torch 中导入 RotaryEmbedding
from rotary_embedding_torch import RotaryEmbedding

# 从 beartype 中导入 beartype
from beartype import beartype
# 从 beartype.door 中导入 is_bearable
from beartype.door import is_bearable
# 从 beartype.typing 中导入 Optional, Union, Callable, Literal, Tuple, List
from beartype.typing import Optional, Union, Callable, Literal, Tuple, List

# 从 x_clip.tokenizer 中导入 tokenizer
from x_clip.tokenizer import tokenizer

# 从 spear_tts_pytorch 中导入 Attend, all_gather
from spear_tts_pytorch.attend import Attend
from spear_tts_pytorch.distributed import all_gather

# 从 tqdm 中导入 tqdm
from tqdm import tqdm

# 定义 FloatTensor 类型为 Union 类型，包含 torch.FloatTensor 和 torch.cuda.FloatTensor
FloatTensor = Union[
    torch.FloatTensor,
    torch.cuda.FloatTensor
]

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 判断张量是否为空
def empty(t: Tensor):
    return t.numel() == 0

# 对张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# 设置 EOS 标识符的位置
def set_eos_id(t: Tensor, eos_id: int, pad_id: int):
    eos_indices = ((t == pad_id).cumsum(dim = -1) == 0).sum(dim = -1, keepdim = True).long()

    batch_range = torch.arange(t.shape[0], device = t.device, dtype = torch.long)
    batch_range = rearrange(batch_range, '... -> ... 1')

    t = F.pad(t, (0, 1), value = pad_id)
    t[batch_range, eos_indices] = eos_id
    return t

# 对批次中的唯一连续值进行填充
def batch_unique_consecutive(t, pad_value = 0.):
    unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim = 0)]
    return pad_sequence(unique_arr, batch_first = True, padding_value = pad_value)

# 在 EOS 之后进行掩码处理
def mask_after_eos(target, eos_id, pad_id):
    mask = (target == eos_id).cumsum(dim = -1) > 0
    mask = F.pad(mask, (1, -1), value = False)
    return target.masked_fill(mask, pad_id)

# 安全除法
def safe_div(num, den, eps = 1e-10):
    return num / max(den, eps)

# 查找第一个为真的索引
def find_first_true_index(bool_tensor, dim = -1):
    return (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)

# 冻结和解冻辅助函数

# 设置模块参数是否需要梯度
def set_requires_grad_(module: Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad

# 冻结模块参数
def freeze(module: Module):
    set_requires_grad_(module, False)

# 解冻模块参数
def unfreeze(module: Module):
    set_requires_grad_(module, True)

# 采样辅助函数

# 评估装饰器
def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# 对数函数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# Gumbel 采样
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# Top-p 采样
def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = F.pad(cum_probs > thres, (1, -1), value = 0)
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    sorted_logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)
    return sorted_logits

# Top-k 采样
def top_k(logits, thres = 0.1, k = None):
    if not exists(k):
        k = math.ceil(thres * logits.shape[-1])
    val, ind = torch.topk(logits, k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

# 残差包装器

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# RMSNorm

class RMSNorm(nn.Module):
    # 初始化函数，接受一个维度参数
    def __init__(self, dim):
        # 调用父类的初始化函数
        super().__init__()
        # 计算缩放因子为维度的平方根
        self.scale = dim ** 0.5
        # 创建一个可学习的参数 gamma，维度为输入维度
        self.gamma = nn.Parameter(torch.ones(dim))
    
    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 对输入 x 进行归一化操作，dim=-1 表示对最后一个维度进行归一化
        return F.normalize(x, dim=-1) * self.scale * self.gamma
# 定义 GEGLU 类，用于实现 GEGLU 激活函数
class GEGLU(nn.Module):
    # GEGLU 类的前向传播函数
    def forward(self, x):
        # 将输入张量 x 按照最后一个维度分成两部分
        x, gate = x.chunk(2, dim = -1)
        # 对 gate 部分应用 GELU 激活函数，并与 x 相乘
        return F.gelu(gate) * x

# 定义 FeedForward 函数，用于创建前馈神经网络层
def FeedForward(dim, mult = 4, dropout = 0.):
    # 计算内部维度
    dim_inner = int(dim * mult * 2 / 3)
    # 返回一个包含多个层的神经网络模型
    return nn.Sequential(
        RMSNorm(dim),  # 使用 RMSNorm 进行归一化
        nn.Linear(dim, dim_inner * 2),  # 线性变换层
        GEGLU(),  # 使用 GEGLU 激活函数
        nn.Dropout(dropout),  # Dropout 层
        nn.Linear(dim_inner, dim)  # 线性变换层
    )

# 定义 Attention 类，用于实现注意力机制
class Attention(nn.Module):
    # Attention 类的初始化函数
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        kv_heads = None,
        causal = False,
        dim_context = None,
        dropout = 0.,
        rotary_emb: Optional[RotaryEmbedding] = None,
        flash = False,
        add_null_kv = False
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.heads = heads
        self.kv_heads = default(kv_heads, heads)
        assert (self.heads % self.kv_heads) == 0, 'number of key value heads must be divisible by query heads'

        self.scale = dim_head ** -0.5
        dim_query_inner = heads * dim_head
        dim_kv_inner = self.kv_heads * dim_head

        self.rotary_emb = rotary_emb

        self.attend = Attend(
            causal = causal,
            flash = flash,
            dropout = dropout
        )

        self.norm = RMSNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)

        # 将输入转换为查询向量
        self.to_q = nn.Sequential(
            nn.Linear(dim, dim_query_inner, bias = False),
            Rearrange('b n (h d) -> b h n d', h = self.heads)
        )

        # 将上下文转换为键值对
        self.to_kv = nn.Sequential(
            nn.Linear(dim_context, dim_kv_inner * 2, bias = False),
            Rearrange('b n (kv h d) -> kv b h n d', kv = 2, h = self.kv_heads)
        )

        # 将输出转换为指定维度
        self.to_out = nn.Linear(dim_query_inner, dim, bias = False)

        self.add_null_kv = add_null_kv
        if add_null_kv:
            self.null_kv = nn.Parameter(torch.randn(2, self.kv_heads, 1, dim_head))

    # Attention 类的前向传播函数
    def forward(
        self,
        x,
        context = None,
        mask = None,
        cache = None,
        return_cached_key_values = False
    ):
        has_context = exists(context)
        b = x.shape[0]

        x = self.norm(x)

        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context))

        if exists(cache):
            ck, cv = cache.unbind(dim = 1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        new_cache = torch.stack((k, v), dim = 1)

        if exists(self.rotary_emb):
            assert not has_context
            q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        if self.add_null_kv:
            assert not exists(self.rotary_emb)
            nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b = b), self.null_kv)
            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        if not return_cached_key_values:
            return out

        return out, new_cache

# 定义 Transformer 类，用于实现 Transformer 模型
class Transformer(nn.Module):
    # Transformer 类的初始化函数
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        kv_heads = None,
        causal = False,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        cross_attend = False,
        attn_flash = False
    ):
        # 调用父类的构造函数
        super().__init__()

        # 创建旋转嵌入对象
        rotary_emb = RotaryEmbedding(dim_head)

        # 初始化神经网络层列表
        self.layers = nn.ModuleList([])

        # 循环创建指定数量的层
        for _ in range(depth):
            # 每一层包含注意力机制、交叉注意力机制（可选）、前馈神经网络
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, kv_heads = kv_heads, dropout = attn_dropout, rotary_emb = rotary_emb, flash = attn_flash),
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = attn_flash, add_null_kv = True) if cross_attend else None,
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        # 创建最终的归一化层
        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        context = None,
        context_mask = None,
        cache = None,
        return_cache = False,
        return_hiddens = False,
        early_exit_at_layer = None,
        seq_start_pos = None
    ):
        # 检查是否存在上下文信息
        has_context = exists(context)

        # 如果存在序列起始位置信息，则生成对应的掩码
        if exists(seq_start_pos):
            assert not exists(mask)
            seq_len = x.shape[-2]
            seq_arange = torch.arange(seq_len, device = x.device, dtype = torch.long)
            mask = seq_arange >= seq_start_pos[..., None]

        # 如果存在缓存信息，则截取输入序列
        if exists(cache):
            cached_length, seq_len = cache.shape[-2], x.shape[-2]
            assert seq_len > cached_length
            x = x[:, cached_length:]

        # 初始化新的缓存列表和隐藏层列表
        new_cache = []
        hiddens = []

        # 如果存在缓存信息，则创建迭代器
        if exists(cache):
            iter_cache = iter(cache.unbind(dim = 1))
        else:
            iter_cache = iter([])

        # 遍历每一层
        for ind, (self_attn, maybe_cross_attn, ff) in enumerate(self.layers):
            layer = ind + 1

            # 计算自注意力机制输出，并更新缓存
            residual = x
            attn_out, key_values = self_attn(x, mask = mask, cache = next(iter_cache, None), return_cached_key_values = True)
            x = attn_out + residual
            new_cache.append(key_values)

            # ��果存在交叉注意力机制，则应用
            if exists(maybe_cross_attn):
                assert has_context
                x = maybe_cross_attn(x, context = context, mask = context_mask) + x

            # 应用前馈神经网络
            x = ff(x) + x
            hiddens.append(x)

            # 如果设置了提前退出层，则在该层结束循环
            if exists(early_exit_at_layer) and early_exit_at_layer == layer:
                break

        # 如果设置了提前退出层，则返回结果或缓存
        if exists(early_exit_at_layer):
            if return_cache:
                return x, torch.stack(new_cache, dim = 1)
            return x

        # 对最终输出进行归一化
        out = self.final_norm(x)

        # 如果需要返回隐藏层信息，则返回结果和隐藏层列表
        if return_hiddens:
            assert not return_cache
            return out, torch.stack(hiddens)

        # 如果不需要返回缓存信息，则返回结果
        if not return_cache:
            return out

        # 返回结果和缓存信息
        return out, torch.stack(new_cache, dim = 1)
# 定义 SpeechOrTextLiteral 类型，可以是'speech'或'text'中的一个
SpeechOrTextLiteral = Union[
    Literal['speech'],
    Literal['text']
]

# 定义 SemanticModelType 类型，可以是 FairseqVQWav2Vec 或 HubertWithKmeans 中的一个
SemanticModelType = Union[
    FairseqVQWav2Vec,
    HubertWithKmeans
]

# 定义 TextToSemantic 类，继承自 Module 类
class TextToSemantic(Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        dim,
        *,
        source_depth,
        target_depth,
        num_text_token_ids = None,
        tokenizer_encode: Optional[Callable] = None,
        use_openai_tokenizer = False,
        wav2vec: Optional[SemanticModelType] = None,
        num_semantic_token_ids = None,
        dim_head = 64,
        heads = 8,
        target_kv_heads = None,  # for grouped query attention, saving memory on decoder inference
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        semantic_pad_id = -1,
        text_pad_id = 0,
        autoset_semantic_eos_id = True,
        autoset_text_eos_id = True,
        attn_flash = False,
        cond_drop_prob = 0.,
        target_early_exit_layer = None,
        detach_early_exit_embed = False,
        align_reg_loss_weight = 0.1,
        align_reg_use_logsumexp_pool = True,
        align_reg_logsumexp_pool_temp = 0.1
    @property
    def device(self):
        # 返回第一个参数的设备
        return next(self.parameters()).device

    # 加载函数
    def load(self, path, strict = True):
        # 返回 pkg，以便如果此函数从 Trainer 函数调用中调用，则 Trainer 也可以访问从检查点加载的包
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(pkg['model'], strict = strict)
        return pkg

    # 一组冻结/解冻工具
    # 然后依赖 get_optimizer 来过滤不需要梯度的参数，使其暴露给优化器

    # 解冻所有参数
    def unfreeze_all(self):
        unfreeze(self)

    # 冻结编码器
    def freeze_encoder(self):
        freeze(self.source_transformer)

    # 冻结编码器到某一层
    def freeze_encoder_below_layer(self, layer: int):
        """
        用于在伪标记数据集上对文本到语义的最终训练
        他们将编码器部分冻结到某一层
        """
        unfreeze(self.source_transformer)

        for ind, module in enumerate(self.source_transformer.layers):
            current_layer = ind + 1

            if current_layer <= layer:
                freeze(module)

    # 冻结解码器
    def freeze_decoder(self):
        freeze(self.target_transformer)

    # 冻结语音嵌入
    def freeze_speech_emb(self):
        freeze(self.token_emb['speech'])
        self.start_token['speech'].requires_grad = False

    # 冻结文本嵌入
    def freeze_text_emb(self):
        freeze(self.token_emb['text'])
        self.start_token['text'].requires_grad = False

    # 采样函数

    @torch.no_grad()
    @eval_decorator
    @beartype
    def generate(
        self,
        source: Union[List[str], Tensor],
        *,
        source_type: SpeechOrTextLiteral,
        target_type: SpeechOrTextLiteral,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_fn_kwargs: dict = dict(),
        source_mask: Optional[Tensor] = None,
        max_length = 2048,
        beam_search_decode = False,
        spec_decode = False,
        spec_decode_gamma = 5,
        spec_decode_lenience = 1.,
        beam_size = 4,
        return_source = False,
        return_target_mask = False,
        cond_scale = 1.
    @beartype
    def forward(
        self,
        source: Union[List[str], Tensor],
        target: Union[List[str], Tensor],
        *,
        source_type: SpeechOrTextLiteral,
        target_type: SpeechOrTextLiteral,
        source_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
        return_loss = False,
        return_logits = False,
        cond_drop_prob: Optional[float] = None,
        should_sim_regularize = True,
        return_early_exit_loss = False
# 预训练模块

# 获取掩码子集概率函数
def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    # 计算每个位置需要mask的数量，根据mask的和与概率相乘，并限制最小值为min_mask
    num_to_mask = (mask.sum(dim=-1, keepdim=True) * prob).clamp(min=min_mask)
    # 生成一个指定大小的随机张量，用于存储logits
    logits = torch.rand((batch, seq), device=device)
    # 根据mask将logits中的非mask位置填充为-1
    logits = logits.masked_fill(~mask, -1)

    # 对logits进行排序，返回排序后的索引
    randperm = logits.argsort(dim=-1).float()

    # 计算每个样本中需要填充的数量
    num_padding = (~mask).sum(dim=-1, keepdim=True)
    # 将randperm中的索引减去需要填充的数量，以保证填充的位置不会被选中
    randperm -= num_padding

    # 生成一个布尔张量，表示哪些位置需要被选中
    subset_mask = randperm < num_to_mask
    # 将subset_mask中非mask位置填充为False
    subset_mask.masked_fill_(~mask, False)
    # 返回subset_mask
    return subset_mask
# 定义一个包装器类，用于语音到语义预训练任务
class SpeechSpeechPretrainWrapper(nn.Module):
    # 初始化方法
    @beartype
    def __init__(
        self,
        model: TextToSemantic,  # 语义模型
        wav2vec: Optional[SemanticModelType] = None,  # 可选的语音模型
        deletion_prob: float = 0.6,  # 删除概率
        reconstruct_seq: bool = False,  # 是否重构序列
        mask_id = None  # 掩码 ID
    ):
        super().__init__()

        self.model = model  # 保存语义模型
        self.wav2vec = default(wav2vec, model.wav2vec)  # 保存语音模型，默认为语义模型的 wav2vec

        self.deletion_prob = deletion_prob  # 保存删除概率
        self.reconstruct_seq = reconstruct_seq  # 是否重构序列
        self.mask_id = mask_id  # 掩码 ID

    # 前向传播方法
    def forward(
        self,
        x,  # 输入数据
        return_early_exit_loss = False  # 是否返回早期退出损失
    ):
        is_raw_audio = x.dtype == torch.float  # 判断输入数据是否为原始音频

        if is_raw_audio:
            assert exists(self.wav2vec)  # 断言语音模型存在
            
            with torch.no_grad():
                self.wav2vec.eval()  # 设置语音模型为评估模式
                x = self.wav2vec(x, flatten = False)  # 对输入数据进行处理

        batch = x.shape[0]  # 获取批次大小

        mask = torch.ones_like(x, dtype = torch.bool, device = self.model.device)  # 创建与输入数据相同形状的掩码

        if exists(self.mask_id):
            assert self.reconstruct_seq, 'reconstruct_seq must be true if mask id is provided'  # 如果提供了掩码 ID，则重构序列必须为真
            
            mask = mask.masked_fill(x == self.model.semantic_pad_id, False)  # 根据语义填充 ID 进行掩码
            delete_mask = get_mask_subset_prob(mask, self.deletion_prob)  # 获取删除掩码

            source = x.masked_fill(delete_mask, self.mask_id)  # 根据删除掩码和掩码 ID 生成源数据
        else:
            delete_mask = get_mask_subset_prob(mask, self.deletion_prob)  # 获取删除掩码

            source = rearrange(x[~delete_mask], '(b n) -> b n', b = batch)  # 重新排列数据

        if self.reconstruct_seq:
            target = x  # 目标数据为输入数据
        else:
            target = rearrange(x[delete_mask], '(b n) -> b n', b = batch)  # 目标数据为删除后的数据

        loss, logits = self.model(
            source, target,  # 输入源数据和目标数据
            source_type = 'speech',  # 源数据类型为语音
            target_type = 'speech',  # 目标数据类型为语音
            return_loss = True,  # 返回损失
            return_logits = True,  # 返回 logits
            return_early_exit_loss = return_early_exit_loss,  # 是否返回早期退出损失
        )

        return loss, logits

# 包装器类，用于反向翻译任务
class SemanticToTextWrapper(nn.Module):
    # 初始化方法
    @beartype
    def __init__(
        self,
        model: TextToSemantic  # 语义模型
    ):
        super().__init__()

        self.model = model  # 保存语义模型

    # 前向传播方法
    def forward(
        self,
        semantic_token_ids,  # 语义标记 ID
        grapheme_token_ids,  # 字形标记 ID
    ):
        source = semantic_token_ids  # 源数据为语义标记 ID
        target = grapheme_token_ids  # 目标数据为字形标记 ID

        loss, logits = self.model(
            source, target,  # 输入源数据和目标数据
            source_type = 'speech',  # 源数据类型为语音
            target_type = 'text',  # 目标数据类型为文本
            return_loss = True,  # 返回损失
            return_logits = True  # 返回 logits
        )

        return loss, logits

# 包装器类，用于文本到语义任务
class TextToSemanticWrapper(nn.Module):
    # 初始化方法
    @beartype
    def __init__(
        self,
        model: TextToSemantic  # 语义模型
    ):
        super().__init__()

        self.model = model  # 保存语义模型

    # 前向传播方法
    def forward(
        self,
        grapheme_token_ids,  # 字形标记 ID
        semantic_token_ids,  # 语义标记 ID
        return_early_exit_loss = True  # 是否返回早期退出损失
    ):
        source = grapheme_token_ids  # 源数据为字形标记 ID
        target = semantic_token_ids  # 目标数据为语义标记 ID

        loss, logits = self.model(
            source, target,  # 输入源数据和目标数据
            source_type = 'text',  # 源数据类型为文本
            target_type = 'speech',  # 目标数据类型为语音
            return_loss = True,  # 返回损失
            return_logits = True,  # 返回 logits
            return_early_exit_loss = return_early_exit_loss  # 是否返回早期退出损失
        )

        return loss, logits

# 包装器类，用于生成伪标记的音频到文本数据集
class SemanticToTextDatasetGenerator(nn.Module):
    # 初始化方法
    @beartype
    def __init__(
        self,
        model,  # 模型
        *,
        dataset: Dataset,  # 数据集
        folder = './generated-audio-text-pairs',  # 文件夹路径
        batch_size = 4,  # 批次大小
        delimiter_id: int = -1,  # 分隔符 ID
        audio_pad_id = None,  # 音频填充 ID
        text_pad_id = 0  # 文本填充 ID
    # 初始化函数，设置模型、数据集、数据加载器等参数
    def __init__(
        self,
        model,
        dataset,
        batch_size,
        delimiter_id,
        audio_pad_id,
        text_pad_id,
        folder
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置模型
        self.model = model

        # 设置数据集
        self.dataset = dataset
        # 根据数据集和批量大小创建数据加载器
        self.dl = get_dataloader(dataset, batch_size=batch_size)
        # 设置分隔符的 ID
        self.delimiter_id = delimiter_id

        # 设置音频填充符的 ID
        self.audio_pad_id = audio_pad_id
        # 设置文本填充符的 ID
        self.text_pad_id = text_pad_id

        # 将文件夹路径转换为 Path 对象，并创建文件夹（如果不存在）
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=True, parents=True)

    # 前向传播函数，生成文本数据
    def forward(
        self,
        max_length=2048,
        beam_search_decode=True,
        **generate_kwargs
    ):
        # 创建包含分隔符 ID 的张量
        delimiter = torch.tensor([self.delimiter_id], device=self.model.device)

        # 计数器，用于生成文件名
        counter = 0

        # 遍历数据加载器中的音频数据
        for audio, in self.dl:
            # 生成音频语义 ID 和文本 ID
            audio_semantic_ids, text_ids = self.model.generate(
                source=audio,
                source_type='speech',
                target_type='text',
                return_source=True,
                max_length=max_length,
                beam_search_decode=beam_search_decode,
                **generate_kwargs
            )

            # 遍历音频语义 ID 和文本 ID
            for audio_semantic_id, text_id in zip(audio_semantic_ids, text_ids):

                # 如果音频填充符存在，则创建音频填充掩码并去除填充符
                if exists(self.audio_pad_id):
                    audio_pad_mask = audio_semantic_id == self.audio_pad_id
                    audio_semantic_id = audio_semantic_id[~audio_pad_mask]

                # 如果文本填充符存在，则创建文本填充掩码并去除填充符
                if exists(self.text_pad_id):
                    text_pad_mask = text_id == self.text_pad_id
                    text_id = text_id[~text_pad_mask]

                # 将音频语义 ID、分隔符和文本 ID 打包成一行数据
                row, _ = pack([audio_semantic_id, delimiter, text_id], '*')
                # 构建保存路径
                path = str(self.folder / f'{counter}.pt')

                # 保存数据到指定路径
                torch.save(row, path)
                # 更新计数器
                counter += 1
```