# `so-vits-svc\vencoder\whisper\model.py`

```
# 从 dataclasses 模块中导入 dataclass 装饰器
# 从 typing 模块中导入 Dict, Iterable, Optional 类型
import numpy as np
# 从 torch 模块中导入 Tensor, nn 类
from torch import Tensor, nn
# 从当前目录下的 decoding 模块中导入 decode 和 detect_language 函数
from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function

# 定义一个名为 ModelDimensions 的数据类，包含多个属性
@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int

# 定义一个名为 LayerNorm 的类，继承自 nn.LayerNorm
class LayerNorm(nn.LayerNorm):
    # 重写 forward 方法
    def forward(self, x: Tensor) -> Tensor:
        # 调用父类的 forward 方法，将输入转换为 float 类型后再转换为原始类型
        return super().forward(x.float()).type(x.dtype)

# 定义一个名为 Linear 的类，继承自 nn.Linear
class Linear(nn.Linear):
    # 重写 forward 方法
    def forward(self, x: Tensor) -> Tensor:
        # 调用 F.linear 函数进行线性变换
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )

# 定义一个名为 Conv1d 的类，继承自 nn.Conv1d
class Conv1d(nn.Conv1d):
    # 重写 _conv_forward 方法
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        # 调用父类的 _conv_forward 方法，将输入和权重转换为输入的数据类型
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

# 定义一个名为 sinusoids 的函数，用于生成位置嵌入的正弦波
def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    # 断言通道数为偶数
    assert channels % 2 == 0
    # 计算对数时间尺度增量
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    # 计算逆时间尺度
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    # 计算缩放时间
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    # 拼接正弦和余弦波形成位置嵌入
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

# 定义一个名为 MultiHeadAttention 的类，继承自 nn.Module
class MultiHeadAttention(nn.Module):
    # 初始化方法
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        # 定义查询、键、值和输出的线性变换层
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
    # 定义一个前向传播函数，接受输入张量 x，可选的辅助输入张量 xa，可选的掩码张量 mask，可选的键值缓存字典 kv_cache
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        # 使用 self.query 函数对输入张量 x 进行查询操作，得到查询张量 q
        q = self.query(x)

        # 如果键值缓存为空或者辅助输入 xa 为空或者 self.key 不在键值缓存中
        if kv_cache is None or xa is None or self.key not in kv_cache:
            # 如果安装了钩子（即 kv_cache 不为空），则在缓存的键值张量之前添加钩子；否则，像往常一样执行自注意力或交叉注意力的键值投影。
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # 对于交叉注意力，计算一次键和值，并在后续调用中重复使用。
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        # 使用 self.qkv_attention 函数对查询张量 q、键张量 k、值张量 v 进行注意力计算，得到加权值张量 wv 和注意力分数张量 qk
        wv, qk = self.qkv_attention(q, k, v, mask)
        # 返回输出张量和注意力分数张量
        return self.out(wv), qk

    # 定义一个查询-键-值注意力函数，接受查询张量 q、键张量 k、值张量 v，可选的掩码张量 mask
    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        # 获取查询张量 q 的批次大小、上下文大小、状态大小
        n_batch, n_ctx, n_state = q.shape
        # 计算缩放因子
        scale = (n_state // self.n_head) ** -0.25
        # 对查询张量 q 进行形状变换和维度置换，并乘以缩放因子，得到新的查询张量 q
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        # 对键张量 k 进行形状变换和维度置换，并乘以缩放因子，得到新的键张量 k
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        # 对值张量 v 进行形状变换和维度置换，得到新的值张量 v
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        # 计算查询张量 q 和键张量 k 的点积，得到注意力分数张量 qk
        qk = q @ k
        # 如果存在掩码张量，则将其加到注意力分数张量 qk 上
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        # 将注意力分数张量 qk 转换为浮点型
        qk = qk.float()

        # 对注意力分数张量 qk 进行 softmax 操作，得到权重张量 w，并转换为与查询张量 q 相同的数据类型
        w = F.softmax(qk, dim=-1).to(q.dtype)
        # 返回加权值张量和注意力分数张量
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()
# 定义一个残差注意力块的类，继承自 nn.Module
class ResidualAttentionBlock(nn.Module):
    # 初始化函数，接受输入参数 n_state（状态数量）、n_head（头数量）、cross_attention（是否跨注意力）
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        # 调用父类的初始化函数
        super().__init__()

        # 创建多头注意力层对象
        self.attn = MultiHeadAttention(n_state, n_head)
        # 创建 LayerNorm 层对象
        self.attn_ln = LayerNorm(n_state)

        # 如果需要跨注意力
        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        # 计算 MLP 层的输入维度
        n_mlp = n_state * 4
        # 创建包含线性层和 GELU 激活函数的序列
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        # 创建 LayerNorm 层对象
        self.mlp_ln = LayerNorm(n_state)

    # 前向传播函数，接受输入参数 x（张量）、xa（可选的张量）、mask（可选的张量）、kv_cache（可选的字典）
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        # 使用多头注意力层处理输入张量 x，并将结果与 x 相加
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        # 如果存在跨注意力层
        if self.cross_attn:
            # 使用跨注意力层处理输入张量 x，并将结果与 x 相加
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        # 使用 MLP 层处理输入张量 x，并将结果与 x 相加
        x = x + self.mlp(self.mlp_ln(x))
        # 返回处理后的张量 x
        return x


# 定义一个音频编码器类，继承自 nn.Module
class AudioEncoder(nn.Module):
    # 初始化函数，接受输入参数 n_mels（梅尔频率数量）、n_ctx（上下文数量）、n_state（状态数量）、n_head（头数量）、n_layer（层数量）
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一维卷积层对象
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        # 注册缓冲区，存储位置嵌入
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        # 创建残差注意力块的列表
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        # 创建 LayerNorm 层对象
        self.ln_post = LayerNorm(n_state)
    # 定义一个前向传播函数，接受一个形状为(batch_size, n_mels, n_ctx)的torch.Tensor作为输入
    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        # 使用GELU激活函数对输入进行卷积操作
        x = F.gelu(self.conv1(x))
        # 使用GELU激活函数对输入进行卷积操作
        x = F.gelu(self.conv2(x))
        # 将张量的维度进行转置
        x = x.permute(0, 2, 1)

        # 获取输入张量的长度
        len_x = x.shape[1]
        # 获取位置嵌入张量的长度
        len_e = self.positional_embedding.shape[0]
        # 断言输入张量的长度不大于位置嵌入张量的长度，否则抛出错误
        assert len_x <= len_e, "incorrect audio shape"
        # 从位置嵌入张量中取出与输入张量长度相对应的部分
        pos_e = self.positional_embedding[:len_x, :]
        # 将输入张量与位置嵌入相加，并转换为与输入张量相同的数据类型
        x = (x + pos_e).to(x.dtype)

        # 遍历所有的块并对输入张量进行处理
        for block in self.blocks:
            x = block(x)

        # 对处理后的张量进行Layer Normalization
        x = self.ln_post(x)
        # 返回处理后的张量
        return x
class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        # 创建一个词嵌入层，将词汇映射到状态空间
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        # 创建一个可学习的位置编码张量
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        # 创建多个残差注意力块，用于构建模型的层
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        # 创建一个层归一化层
        self.ln = LayerNorm(n_state)

        # 创建一个上三角矩阵作为遮挡，用于在自注意力中屏蔽未来信息
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        # 如果有键值缓存，计算偏移量
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        # 对输入的文本进行词嵌入和位置编码
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        # 通过多个残差注意力块进行前向传播
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        # 对输出进行层归一化
        x = self.ln(x)
        # 计算输出的logits
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        # 创建一个音频编码器
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        # 创建一个文本解码器
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
    # 将音频特征嵌入到编码器中，返回嵌入后的结果
    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)
    
    # 根据 tokens 和音频特征计算 logits，返回计算结果
    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)
    
    # 前向传播函数，根据音频特征和 tokens 返回计算结果的字典
    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))
    
    # 返回模型参数所在的设备
    @property
    def device(self):
        return next(self.parameters()).device
    
    # 返回模型是否支持多语言
    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865
    # 定义一个方法，用于安装键值缓存的钩子，可选地接受一个缓存字典作为参数
    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        # 如果缓存不为空，则创建缓存的副本，否则创建一个空字典
        cache = {**cache} if cache is not None else {}
        # 初始化一个空的钩子列表
        hooks = []

        # 定义一个函数，用于保存中间结果到缓存中
        def save_to_cache(module, _, output):
            # 如果模块不在缓存中，或者输出的形状大于解码器的位置嵌入形状，则直接保存输出
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            # 否则，将输出与缓存中的数据拼接起来，并且将结果分离出来
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        # 定义一个函数，用于安装钩子
        def install_hooks(layer: nn.Module):
            # 如果层是多头注意力层，则分别为其键和值模块安装前向钩子
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        # 对解码器应用安装钩子的函数
        self.decoder.apply(install_hooks)
        # 返回缓存和钩子列表
        return cache, hooks

    # 将 detect_language_function 赋值给 detect_language
    detect_language = detect_language_function
    # 将 decode_function 赋值给 decode
    decode = decode_function
```