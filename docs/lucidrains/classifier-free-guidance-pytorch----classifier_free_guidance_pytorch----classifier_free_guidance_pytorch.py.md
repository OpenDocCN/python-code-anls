# `.\lucidrains\classifier-free-guidance-pytorch\classifier_free_guidance_pytorch\classifier_free_guidance_pytorch.py`

```
# 导入必要的模块
from collections import namedtuple
from functools import wraps, partial, cache

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, pack, unpack

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Callable, Tuple, Optional, List, Literal, Union, Dict, Any

from inspect import signature

from classifier_free_guidance_pytorch.t5 import T5Adapter
from classifier_free_guidance_pytorch.open_clip import OpenClipAdapter
from classifier_free_guidance_pytorch.attend import Attend
from classifier_free_guidance_pytorch.bge import BGEAdapter

# 常量定义

COND_DROP_KEY_NAME = 'cond_drop_prob'

TEXTS_KEY_NAME = 'texts'
TEXT_EMBEDS_KEY_NAME = 'text_embeds'
TEXT_CONDITIONER_NAME = 'text_conditioner'
CONDITION_FUNCTION_KEY_NAME = 'cond_fns'

# 定义命名元组
TextCondReturn = namedtuple('TextCondReturn', [
    'embed',
    'mask'
])

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 判断列表是否为空
def is_empty(l):
    return len(l) == 0

# 返回第一个存在的值
def default(*values):
    for value in values:
        if exists(value):
            return value
    return None

# 将值转换为元组
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# 将单个值打包成元组
def pack_one(x, pattern):
    return pack([x], pattern)

# 从元组中解包单个值
def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

# 张量辅助函数

# 根据概率生成掩码张量
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# 使用自动文本条件的分类器自由引导

# 装饰器函数，用于处理函数的参数和自动文本条件
@beartype
def classifier_free_guidance(
    fn: Callable,
    cond_drop_prob_keyname = COND_DROP_KEY_NAME,
    texts_key_name = TEXTS_KEY_NAME,
    text_embeds_key_name = TEXT_EMBEDS_KEY_NAME,
    cond_fns_keyname = CONDITION_FUNCTION_KEY_NAME,
    text_conditioner_name = TEXT_CONDITIONER_NAME
):
    # 获取函数的参数信息
    fn_params = signature(fn).parameters

    # 判断是否需要自动处理文本条件
    auto_handle_text_condition = texts_key_name not in fn_params and text_embeds_key_name not in fn_params

    # 内部函数，用于实际执行分类器自由引导
    @wraps(fn)
    def inner(
        self,
        *args,
        cond_scale: float = 1.,
        rescale_phi: float = 0.,
        cfg_routed_kwargs: Dict[str, Tuple[Any, Any]] = dict(),   # 用于传递参数到前向和无效前向调用的字典（用于处理在使用 CFG 进行变换解码时的缓存）
        **kwargs
        @wraps(fn)
        # 定义一个装饰器函数，用于包装原始函数
        def fn_maybe_with_text(self, *args, **kwargs):
            # 在可能包含文本的情况下，对原始函数进行包装
            if auto_handle_text_condition:
                # 如果自动处理文本条件为真
                texts = kwargs.pop('texts', None)
                text_embeds = kwargs.pop('text_embeds', None)

                assert not (exists(texts) and exists(text_embeds))
                # 断言不存在同时有texts和text_embeds

                raw_text_cond = cond_fns = None

                text_conditioner = getattr(self, text_conditioner_name, None)
                # 获取文本条件器对象

                cond_drop_prob = kwargs.pop(cond_drop_prob_keyname, None)

                assert not exists(cond_drop_prob) or 0. <= cond_drop_prob <= 1.
                # 断言不存在cond_drop_prob或者其值在0到1之间

                # 自动将文本转换为条件函数
                if exists(texts) ^ exists(text_embeds):

                    assert is_bearable(texts, Optional[List[str]]), f'keyword `{texts_key_name}` must be a list of strings'
                    # 断言texts是可接受的类型，必须是字符串列表

                    assert exists(text_conditioner) and is_bearable(text_conditioner, Conditioner), 'text_conditioner must be set on your network with the correct hidden dimensions to be conditioned on'
                    # 断言存在text_conditioner并且其类型是Conditioner

                    text_condition_input = dict(texts = texts) if exists(texts) else dict(text_embeds = text_embeds)

                    cond_fns, raw_text_cond = text_conditioner(**text_condition_input, cond_drop_prob = cond_drop_prob)
                    # 调用文本条件器生成条件函数和原始文本条件

                elif isinstance(text_conditioner, NullConditioner):
                    assert cond_drop_prob == 0., 'null conditioner has nothing to dropout'
                    # 断言cond_drop_prob为0，空条件器没有需要丢弃的内容

                    cond_fns, raw_text_cond = text_conditioner()
                    # 调用空条件器

                if 'cond_fns' in fn_params:
                    kwargs.update(cond_fns = cond_fns)

                if 'raw_text_cond' in fn_params:
                    kwargs.update(raw_text_cond = raw_text_cond)

            return fn(self, *args, **kwargs)
            # 返回原始函数的结果

        # 主分类器自由引导逻辑

        if self.training:
            assert cond_scale == 1, 'you cannot do condition scaling when in training mode'
            # 断言在训练模式下不能进行条件缩放

            return fn_maybe_with_text(self, *args, **kwargs)
            # 返回可能包含文本的函数结果

        assert cond_scale >= 1, 'invalid conditioning scale, must be greater or equal to 1'
        # 断言条件缩放必须大于等于1

        kwargs_without_cond_dropout = {**kwargs, cond_drop_prob_keyname: 0.}
        kwargs_with_cond_dropout = {**kwargs, cond_drop_prob_keyname: 1.}
        # 创建不带条件丢弃和带条件丢弃的参数字典

        # 处理要路由到前向和空前向的参数，以便处理两次调用的缓存
        fn_kwargs = {k: v[0] for k, v in cfg_routed_kwargs.items()}
        null_fn_kwargs = {k: v[1] for k, v in cfg_routed_kwargs.items()}
        # 创建非空前向和空前向的参数字典

        # 非空前向
        outputs = fn_maybe_with_text(self, *args, **fn_kwargs, **kwargs_without_cond_dropout)
        # 调用可能包含文本的函数

        if cond_scale == 1:
            return outputs
            # 如果条件缩放为1，则直接返回结果

        logits, *rest = cast_tuple(outputs)
        # 将输出结果拆分为logits和其余部分

        # 空前向
        null_outputs = fn_maybe_with_text(self, *args, **null_fn_kwargs, **kwargs_with_cond_dropout)
        # 调用可能包含文本的函数

        null_logits, *null_rest = cast_tuple(null_outputs)
        # 将空前向的输出结果拆分为null_logits和其余部分

        zipped_rest = tuple(zip(rest, null_rest))
        # 将非空前向和空前向的其余部分进行压缩

        scaled_logits = null_logits + (logits - null_logits) * cond_scale
        # 计算缩放后的logits

        if rescale_phi <= 0:
            logit_output = scaled_logits
        else:
            # 提议的方法，用于防止分类器自由引导过度饱和
            # 与imagen的解决方案不同，适用于像素空间和潜在空间

            dims = tuple(range(1, logits.ndim - 1))
            rescaled_logits = scaled_logits * (logits.std(dim = dims, keepdim = True) / scaled_logits.std(dim = dims, keepdim= True))
            logit_output = rescaled_logits * rescale_phi + scaled_logits * (1. - rescale_phi)
            # 计算最终输出logits

        if is_empty(zipped_rest):
            return logit_output
            # 如果压缩后的结果为空，则直接返回logit_output

        return (logit_output, *zipped_rest)
        # 返回最终结果
    return inner
# class decorator

# 装饰器函数，用于添加分类器自由引导的类装饰器
@beartype
def classifier_free_guidance_class_decorator(
    orig_class,
    cond_drop_prob_keyname = COND_DROP_KEY_NAME,
    texts_key_name = TEXTS_KEY_NAME,
    text_embeds_key_name = TEXT_EMBEDS_KEY_NAME,
    cond_fns_keyname = CONDITION_FUNCTION_KEY_NAME,
    text_conditioner_name = TEXT_CONDITIONER_NAME
):
    assert issubclass(orig_class, nn.Module)

    # decorate init

    # 保存原始类的初始化方法
    orig_init = orig_class.__init__

    # 装饰原始类的初始化方法
    @wraps(orig_init)
    @beartype
    def __init__(
        self,
        *args,
        text_condition_type: Union[
            Literal['film'],
            Literal['attention'],
            Literal['null'],
            Literal['raw'],
        ] = 'film',
        text_condition_model_types: Tuple[str, ...] = ('t5',),
        text_condition_hidden_dims: Tuple[int, ...],
        text_condition_cond_drop_prob: float,
        **kwargs
    ):
        # 调用原始类的初始化方法
        orig_init(self, *args, **kwargs)

        # 根据文本条件类型选择相应的条件器类
        if text_condition_type == 'film':
            condition_klass = TextConditioner
        elif text_condition_type == 'attention':
            condition_klass = AttentionTextConditioner
        elif text_condition_type == 'raw':
            condition_klass = TextEmbeddingReturner
        else:
            condition_klass = NullConditioner

        # 初始化文本条件器
        self.text_conditioner = condition_klass(
            model_types = text_condition_model_types,
            hidden_dims = text_condition_hidden_dims,
            cond_drop_prob = text_condition_cond_drop_prob
        )

    orig_class.__init__ = __init__

    # decorate forward

    # 装饰原始类的前向传播方法
    decorated_forward = classifier_free_guidance(
        orig_class.forward,
        cond_drop_prob_keyname = cond_drop_prob_keyname,
        texts_key_name = texts_key_name,
        text_embeds_key_name = text_embeds_key_name,
        cond_fns_keyname = cond_fns_keyname,
        text_conditioner_name = text_conditioner_name
    )

    orig_class.forward = decorated_forward

    # forward `embed_texts` to the `text_conditioner.embed_texts`

    # 定义嵌入文本的方法，将其转发到文本条件器的嵌入文本方法
    @beartype
    def embed_texts(self, texts: List[str]):
        return self.text_conditioner.embed_texts(texts)

    # 定义属性，缓存最大条件文本长度
    @property
    @cache
    def max_cond_text_len(self):
        total_cond_text_len = sum([text_model.max_text_len for text_model in self.text_conditioner.text_models])
        return total_cond_text_len

    # 如果原始类没有最大条件文本长度属性，则添加
    if not hasattr(orig_class, 'max_cond_text_len'):
        orig_class.max_cond_text_len = max_cond_text_len

    # 如果原始类没有嵌入文本方法，则添加
    if not hasattr(orig_class, 'embed_texts'):
        orig_class.embed_texts = embed_texts

    # 标记类已被装饰
    orig_class.__decorated_with_cfg = True
    return orig_class

# attention

# 定义注意力模块类
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dim_context = None,
        norm_context = False,
        num_null_kv = 0,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attend = Attend(flash = flash)        

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(2, num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        mask = None
        ):
        # 获取输入张量 x 的第一个维度大小
        b = x.shape[0]

        # 如果上下文存在，则对上下文进行归一化处理
        if exists(context):
            context = self.context_norm(context)

        # 如果上下文不存在，则使用默认的 x 作为上下文输入
        kv_input = default(context, x)

        # 对输入张量 x 进行归一化处理
        x = self.norm(x)

        # 将输入张量 x 分别转换为查询 q，键 k，值 v
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        # 如果存在空键值对数量大于 0
        if self.num_null_kv > 0:
            # 重复空键值对，使其与输入张量 x 的第一个维度大小相匹配
            null_k, null_v = repeat(self.null_kv, 'kv n d -> kv b n d', b = b).unbind(dim = 0)
            # 将空键值对与原始键 k 和值 v 进行拼接
            k = torch.cat((null_k, k), dim = -2)
            v = torch.cat((null_v, v), dim = -2)

        # 如果存在掩码 mask
        if exists(mask):
            # 在掩码 mask 上添加指定数量的填充值
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)
            # 重新排列掩码 mask 的维度
            mask = rearrange(mask, 'b j -> b 1 1 j')

        # 重新排列查询 q 的维度
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # 进行注意力计算
        out = self.attend(q, k, v, mask = mask)

        # 重新排列输出 out 的维度
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 返回最终输出
        return self.to_out(out)
# dimension adapters

# 重新排列通道为最后一个维度的函数装饰器
def rearrange_channel_last(fn):
    @wraps(fn)
    def inner(hiddens):
        hiddens, ps = pack_one(hiddens, 'b * d')
        conditioned = fn(hiddens)
        return unpack_one(conditioned, ps, 'b * d')
    return inner

# 重新排列通道为第一个维度的函数装饰器
def rearrange_channel_first(fn):
    """ will adapt shape of (batch, feature, ...) for conditioning """

    @wraps(fn)
    def inner(hiddens):
        hiddens, ps = pack_one(hiddens, 'b d *')
        hiddens = rearrange(hiddens, 'b d n -> b n d')
        conditioned =  fn(hiddens)
        conditioned = rearrange(conditioned, 'b n d -> b d n')
        return unpack_one(conditioned, ps, 'b d *')

    return inner

# conditioning modules

# FiLM 模块
class FiLM(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2)
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, conditions, hiddens):
        scale, shift = self.net(conditions).chunk(2, dim = -1)
        assert scale.shape[-1] == hiddens.shape[-1], f'unexpected hidden dimesion {hiddens.shape[-1]} used for conditioning'
        scale, shift = map(lambda t: rearrange(t, 'b d -> b 1 d'), (scale, shift))
        return hiddens * (scale + 1) + shift

# 交叉注意力模块
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        heads = 8,
        dim_head = 64,
        flash = False
    ):
        super().__init__()
        self.attn = Attention(
            dim = hidden_dim,
            dim_context = dim,
            norm_context = True,
            num_null_kv = 1,
            dim_head = dim_head,
            heads = heads,
            flash = flash
        )

    def forward(
        self,
        condition,
        hiddens,
        mask = None
    ):
        return self.attn(hiddens, condition, mask = mask) + hiddens

# film text conditioning

# 条件配置字典
CONDITION_CONFIG = dict(
    t5 = T5Adapter,
    clip = OpenClipAdapter,
    bge = BGEAdapter
)

# 模型类型列表
MODEL_TYPES = CONDITION_CONFIG.keys()

# 条件器基类
class Conditioner(nn.Module):
    pass

# 空条件器
class Identity(nn.Module):
    def forward(self, t, *args, **kwargs):
        return t

# 空条件器类，继承自 Conditioner
@beartype
class NullConditioner(Conditioner):
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...],
        **kwargs
    ):
        super().__init__()
        num_null_conditioners = len(hidden_dims)
        self.cond_fns = tuple(Identity() for _ in range(num_null_conditioners))

        self.register_buffer('_device_param', torch.tensor(0), persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def embed_texts(self, texts: List[str]):
        assert False, 'null conditioner cannot embed text'

    def forward(self, *args, **kwarg):
        return self.cond_fns, None

# 带有 FiLM 的文本条件器
@beartype
class TextConditioner(Conditioner):
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...],
        model_types = 't5',
        model_names = None,
        cond_drop_prob = 0.,
        hiddens_channel_first = True,
        text_embed_stem_dim_mult = 2
    ):
        # 调用父类的构造函数
        super().__init__()
        # 将 model_types 转换为元组
        model_types = cast_tuple(model_types)
        # 将 model_names 转换为元组，并确保其长度与 model_types 相同
        model_names = cast_tuple(model_names, length = len(model_types))

        # 断言 model_types 和 model_names 的长度相同
        assert len(model_types) == len(model_names)
        # 断言 model_types 中的每个元素都在 MODEL_TYPES 中
        assert all([model_type in MODEL_TYPES for model_type in model_types])

        # 初始化一个空列表 text_models
        text_models = []

        # 遍历 model_types 和 model_names，根据 model_type 创建对应的模型，并添加到 text_models 中
        for model_type, model_name in zip(model_types, model_names):
            klass = CONDITION_CONFIG.get(model_type)
            model = klass(model_name)
            text_models.append(model)

        # 将 text_models 赋值给 self.text_models
        self.text_models = text_models
        # 获取每个模型的潜在维度，存储在 latent_dims 中
        self.latent_dims = [model.dim_latent for model in text_models]

        # 初始化一个空的 nn.ModuleList，用于存储条件器
        self.conditioners = nn.ModuleList([])

        # 将 hidden_dims、num_condition_fns、hiddens_channel_first、cond_drop_prob 等属性赋值
        self.hidden_dims = hidden_dims
        self.num_condition_fns = len(hidden_dims)
        self.hiddens_channel_first = cast_tuple(hiddens_channel_first, self.num_condition_fns) # 是否将待条件化的隐藏层放在通道维度的第一位

        # 断言 hiddens_channel_first 的长度与 num_condition_fns 相同
        assert len(self.hiddens_channel_first) == self.num_condition_fns

        # 将 cond_drop_prob 赋值给 self.cond_drop_prob

        # 计算总的潜在维度
        total_latent_dim = sum(self.latent_dims)

        # 计算 MLP 的输出维度
        mlp_stem_output_dim = total_latent_dim * text_embed_stem_dim_mult

        # 定义文本嵌入的 MLP 结构
        self.text_embed_stem_mlp = nn.Sequential(
            nn.Linear(total_latent_dim, mlp_stem_output_dim),
            nn.SiLU()
        )

        # 根据 hidden_dims 创建条件器，并添加到 self.conditioners 中
        for hidden_dim in hidden_dims:
            self.conditioners.append(FiLM(mlp_stem_output_dim, hidden_dim))

        # 初始化一个随机参数 null_text_embed
        self.null_text_embed = nn.Parameter(torch.randn(total_latent_dim))

        # 注册一个缓冲区 _device_param
        self.register_buffer('_device_param', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        # 返回第一个缓冲区的设备
        return next(self.buffers()).device

    def embed_texts(self, texts: List[str]):
        # 获取设备信息
        device = self.device

        # 初始化一个空列表 text_embeds，用于存储文本嵌入结果
        text_embeds = []
        # 遍历每个文本模型，将文本嵌入结果添加到 text_embeds 中
        for text_model in self.text_models:
            text_embed = text_model.embed_text(texts)
            text_embeds.append(text_embed.to(device))

        # 沿着最后一个维度拼接文本嵌入结果
        return torch.cat(text_embeds, dim = -1)

    def forward(
        self,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_drop_prob = None,
        repeat_batch = 1,               # 用于机器人变压器边缘情况
    ) -> Tuple[
        Tuple[Callable, ...],
        TextCondReturn
    ]:

        # 断言 texts 和 text_embeds 只有一个存在
        assert exists(texts) ^ exists(text_embeds)

        # 如果处于训练状态，则使用默认的 cond_drop_prob，否则需要显式设置
        if self.training:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            assert exists(cond_drop_prob), '当不处于训练状态时，必须显式设置 cond_drop_prob'

        # 根据 texts 或 text_embeds 的存在情况确定 batch 大小
        if exists(texts):
            batch = len(texts)
        elif exists(text_embeds):
            batch = text_embeds.shape[0]

        # 如果 text_embeds 不存在，则调用 embed_texts 方法生成
        if not exists(text_embeds):
            text_embeds = self.embed_texts(texts)

        # 如果 cond_drop_prob 大于 0，则生成一个掩码，用于对文本嵌入进行条件化
        if cond_drop_prob > 0.:
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device = self.device)
            null_text_embeds = rearrange(self.null_text_embed, 'd -> 1 d')

            text_embeds = torch.where(
                prob_keep_mask,
                text_embeds,
                null_text_embeds
            )

        # 对文本嵌入进行 MLP 处理
        text_embeds = self.text_embed_stem_mlp(text_embeds)

        # 准备条件函数
        repeat_batch = cast_tuple(repeat_batch, self.num_condition_fns)

        cond_fns = []

        # 遍历条件器，生成条件函数
        for cond, cond_hiddens_channel_first, cond_repeat_batch in zip(self.conditioners, self.hiddens_channel_first, repeat_batch):
            cond_text_embeds = repeat(text_embeds, 'b ... -> (b r) ...', r = cond_repeat_batch)
            cond_fn = partial(cond, cond_text_embeds)

            wrapper_fn = rearrange_channel_first if cond_hiddens_channel_first else rearrange_channel_last

            cond_fns.append(wrapper_fn(cond_fn))

        # 返回条件函数和文本条件返回值
        return tuple(cond_fns), TextCondReturn(text_embeds, None)
# 定义一个名为 AttentionTextConditioner 的类，继承自 Conditioner 类
@beartype
class AttentionTextConditioner(Conditioner):
    # 初始化函数，接受一系列参数
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...],  # 隐藏层维度的元组
        model_types = 't5',  # 模型类型，默认为 't5'
        model_names = None,  # 模型名称，默认为 None
        cond_drop_prob = 0.,  # 条件丢弃概率，默认为 0
        hiddens_channel_first = True,  # 是否隐藏层优先，默认为 True
        dim_latent = None,  # 潜在维度，默认为 None
        attn_dim_head = 64,  # 注意力头维度，默认为 64
        attn_heads = 8,  # 注意力头数，默认为 8
        flash = True  # 是否闪烁，默认为 True
    ):
        super().__init__()  # 调用父类的初始化函数
        model_types = cast_tuple(model_types)  # 将模型类型转换为元组
        model_names = cast_tuple(model_names, length = len(model_types))  # 将模型名称转换为元组，长度与模型类型相同

        assert len(model_types) == len(model_names)  # 断言模型类型和模型名称长度相同
        assert all([model_type in MODEL_TYPES for model_type in model_types])  # 断言所有模型类型在 MODEL_TYPES 中

        text_models = []  # 初始化文本模型列表

        # 遍历模型类型和模型名称，创建文本模型并添加到列表中
        for model_type, model_name in zip(model_types, model_names):
            klass = CONDITION_CONFIG.get(model_type)
            model = klass(model_name)
            text_models.append(model)

        self.text_models = text_models  # 将文本模型列表赋值给类属性

        self.to_latent_dims = nn.ModuleList([])  # 初始化线性层列表

        dim_latent = default(dim_latent, max([model.dim_latent for model in text_models]))  # 计算潜在维度

        self.dim_latent = dim_latent  # 将潜在维度赋值给类属性

        # 遍历文本模型，为每个模型添加线性层
        for model in text_models:
            self.to_latent_dims.append(nn.Linear(model.dim_latent, dim_latent))

        self.conditioners = nn.ModuleList([])  # 初始化条件器列表

        self.hidden_dims = hidden_dims  # 隐藏层维度赋值给类属性
        self.num_condition_fns = len(hidden_dims)  # 隐藏层维度数量赋值给类属性
        self.hiddens_channel_first = cast_tuple(hiddens_channel_first, self.num_condition_fns)  # 是否隐藏层优先赋值给类属性

        assert len(self.hiddens_channel_first) == self.num_condition_fns  # 断言隐藏层优先长度与隐藏层维度数量相同

        self.cond_drop_prob = cond_drop_prob  # 条件丢弃概率赋值给类属性

        # 遍历隐藏层维度，为每个维度添加交叉注意力模块
        for hidden_dim in hidden_dims:
            self.conditioners.append(CrossAttention(dim_latent, hidden_dim, flash = flash))

        self.register_buffer('_device_param', torch.tensor(0), persistent = False)  # 注册缓冲区

    @property
    def device(self):
        return next(self.buffers()).device  # 返回设备信息

    # 嵌入文本函数，接受文本列表，返回文本嵌入向量
    def embed_texts(self, texts: List[str]):
        device = self.device  # 获取设备信息

        text_embeds = []  # 初始化文本嵌入列表

        # 遍历文本模型和线性层，为每个文本嵌入向量添加嵌入
        for text_model, to_latent in zip(self.text_models, self.to_latent_dims):
            text_embed = text_model.embed_text(texts, return_text_encodings = True)  # 嵌入文本并返回文本编码

            text_embed = text_embed.to(device)  # 将文本嵌入向量移动到设备

            mask = (text_embed != 0).any(dim = -1)  # 创建掩码

            text_embed = to_latent(text_embed)  # 使用线性层转换文本嵌入向量
            text_embed = text_embed.masked_fill(~mask[..., None], 0.)  # 根据掩码填充文本嵌入向量

            text_embeds.append(text_embed)  # 将处理后的文本嵌入向量添加到列表中

        return torch.cat(text_embeds, dim = -2)  # 沿指定维度连接文本嵌入向量

    # 前向传播函数，接受文本列表、文本嵌入向量等参数，返回元组
    def forward(
        self,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_drop_prob = None,
        repeat_batch = 1,  # 用于机器人变压器边缘情况
    ) -> Tuple[
        Tuple[Callable, ...],
        TextCondReturn
        # 检查是否存在文本或文本嵌入
        assert exists(texts) or exists(text_embeds)

        # 如果存在文本嵌入和文本，则文本嵌入优先
        if exists(text_embeds) and exists(texts):
            texts = None

        # 如果处于训练状态，则使用默认的条件丢弃概率，否则需要显式设置条件丢弃概率
        if self.training:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'

        # 根据文本或文本嵌入的存在情况确定批次大小
        if exists(texts):
            batch = len(texts)
        elif exists(text_embeds):
            batch = text_embeds.shape[0]

        # 如果不存在文本嵌入，则使用模型的 embed_texts 方法生成文本嵌入
        if not exists(text_embeds):
            text_embeds = self.embed_texts(texts)

        # 创建一个掩码，标记非零元素的位置
        mask = (text_embeds != 0).any(dim=-1)

        # 如果条件丢弃概率大于0，则生成一个概率保留掩码
        if cond_drop_prob > 0.:
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device=self.device)
            mask = mask & prob_keep_mask

        # 准备条件函数
        repeat_batch = cast_tuple(repeat_batch, self.num_condition_fns)
        cond_fns = []

        # 遍历条件器，生成条件函数列表
        for cond, cond_hiddens_channel_first, cond_repeat_batch in zip(self.conditioners, self.hiddens_channel_first, repeat_batch):
            cond_text_embeds = repeat(text_embeds, 'b ... -> (b r) ...', r=cond_repeat_batch)
            cond_mask = repeat(mask, 'b ... -> (b r) ...', r=cond_repeat_batch) if exists(mask) else None

            cond_fn = partial(cond, cond_text_embeds, mask=cond_mask)

            wrapper_fn = rearrange_channel_first if cond_hiddens_channel_first else rearrange_channel_last

            cond_fns.append(wrapper_fn(cond_fn))

        # 返回条件函数列表和文本条件返回对象
        return tuple(cond_fns), TextCondReturn(text_embeds, mask)
# 返回原始文本嵌入

# 定义一个文本嵌入返回器类，继承自 Conditioner 类
@beartype
class TextEmbeddingReturner(Conditioner):
    # 初始化函数
    def __init__(
        self,
        *,
        dim_latent = None,  # 潜在维度，默认为 None
        hidden_dims: Tuple[int, ...] = tuple(),  # 隐藏维度，默认为空元组
        model_types = 't5',  # 模型类型，默认为 't5'
        model_names = None,  # 模型名称，默认为 None
        cond_drop_prob = 0.,  # 条件丢弃概率，默认为 0.
    ):
        super().__init__()  # 调用父类的初始化函数
        model_types = cast_tuple(model_types)  # 将模型类型转换为元组
        model_names = cast_tuple(model_names, length = len(model_types))  # 将模型名称转换为元组，长度与模型类型相同

        assert len(model_types) == len(model_names)  # 断言模型类型和模型名称长度相同
        assert all([model_type in MODEL_TYPES for model_type in model_types])  # 断言所有模型类型在 MODEL_TYPES 中

        text_models = []  # 初始化文本模型列表

        # 遍历模型类型和模型名称，创建模型对象并添加到文本模型列表中
        for model_type, model_name in zip(model_types, model_names):
            klass = CONDITION_CONFIG.get(model_type)
            model = klass(model_name)
            text_models.append(model)

        self.text_models = text_models  # 将文本模型列表赋值给实例变量

        self.to_latent_dims = nn.ModuleList([])  # 初始化潜在维度列表

        dim_latent = default(dim_latent, max([model.dim_latent for model in text_models]))  # 获取最大的模型潜在维度作为潜在维度

        self.dim_latent = dim_latent  # 将潜在维度赋值给实例变量

        # 遍历文本模型，为每个模型创建线性层并添加到潜在维度列表中
        for model in text_models:
            self.to_latent_dims.append(nn.Linear(model.dim_latent, dim_latent))

        self.conditioners = nn.ModuleList([])  # 初始化条件器列表

        self.cond_drop_prob = cond_drop_prob  # 将条件丢弃概率赋值给实例变量

        # 遍历隐藏维度，为每个维度创建恒等映射并添加到条件器列表中
        for hidden_dim in hidden_dims:
            self.conditioners.append(nn.Identity())

        self.register_buffer('_device_param', torch.tensor(0), persistent = False)  # 注册缓冲区

    @property
    def device(self):
        return next(self.buffers()).device  # 返回缓冲区的设备

    # 嵌入文本函数
    def embed_texts(self, texts: List[str]):
        device = self.device  # 获取设备

        text_embeds = []  # 初始化文本嵌入列表

        # 遍历文本模型和潜在维度列表，为每个文本模型嵌入文本并处理
        for text_model, to_latent in zip(self.text_models, self.to_latent_dims):
            text_embed = text_model.embed_text(texts, return_text_encodings = True)  # 嵌入文本并返回文本编码

            text_embed = text_embed.to(device)  # 将文本嵌入移到设备上

            mask = (text_embed != 0).any(dim = -1)  # 创建掩码，标记非零值

            text_embed = to_latent(text_embed)  # 使用线性层进行潜在维度转换
            text_embed = text_embed.masked_fill(~mask[..., None], 0.)  # 根据掩码填充文本嵌入

            text_embeds.append(text_embed)  # 将处理后的文本嵌入添加到列表中

        return torch.cat(text_embeds, dim = -2)  # 沿指定维度拼接文本嵌入

    # 前向传播函数
    def forward(
        self,
        texts: Optional[List[str]] = None,  # 文本列表，默认为 None
        text_embeds: Optional[Tensor] = None,  # 文本嵌入张量，默认为 None
        cond_drop_prob = None  # 条件丢弃概率，默认为 None
    ) -> Tuple[
        Tuple[Callable, ...],  # 返回条件器元组
        TextCondReturn  # 返回文本条件返回对象
    ]:

        assert exists(texts) ^ exists(text_embeds)  # 断言文本列表和文本嵌入张量只能有一个存在

        if self.training:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)  # 如果在训练模式下，使用默认的条件丢弃概率
        else:
            assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'  # 如果不在训练模式下，条件丢弃概率必须显式设置

        if exists(texts):
            batch = len(texts)  # 获取文本列表的长度

        elif exists(text_embeds):
            batch = text_embeds.shape[0]  # 获取文本嵌入张量的批次大小

        if not exists(text_embeds):
            text_embeds = self.embed_texts(texts)  # 如果文本嵌入不存在，则调用嵌入文本函数

        mask = (text_embeds != 0).any(dim = -1)  # 创建掩码，标记非零值

        if cond_drop_prob > 0.:
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device = self.device)  # 创建概率掩码
            mask = mask & prob_keep_mask  # 更新掩码

        return tuple(self.conditioners), TextCondReturn(text_embeds, mask)  # 返回条件器元组和文本条件返回对象
```