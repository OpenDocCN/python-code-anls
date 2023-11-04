# SO-VITS-SVC源码解析 17

# `vencoder/whisper/model.py`

这段代码定义了一个名为 "ModelDimensions" 的类，该类用于表示模型中音频和文本特征的维度。这个类使用了两个额外的类 "ModelDecoding" 和 "ModelDetectLanguage"，从它们那里获取了音频和文本数据的解码和检测功能。

具体来说，这段代码的作用是定义一个包含模型中所有音频和文本特征的维度，以及如何从原始数据中提取这些特征。这个类包含了一些内部类，用于获取和处理音频和文本数据，同时也引入了两个外部函数 "decode_function" 和 "detect_language_function"，用于解码和检测语言。

这个类的实现基于以下假设：

1. 每个音频样本被表示为一个包含多个时间步的音频信号，每个时间步包含了一个包含噪声的样本。
2. 每个文本样本被表示为一个包含多个单词的文本序列，每个单词都包含了一个词汇表中的词汇。
3. 每个音频和文本特征都已经被进行预处理，包括去除噪声、分段和检测语言。

通过这个类的定义，我们可以使用 "ModelDimensions" 来创建一个模型，该模型可以使用音频和文本数据进行语音识别或文本分类等任务。


```py
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function


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


```

这段代码定义了一个名为 "LayerNorm" 的类，继承自 PyTorch 中的 nn.LayerNorm 类。这个类在 forward 方法中会对输入的 Tensor 进行转换，将其输入到 forward 方法中，然后返回 Tensor 类型。

这个 LayerNorm 类的作用是提供一个轻量级的， 在训练和推理时具有相同的行为的层。这个类中包含了一些方法， 用于在模型中进行常见的层操作， 例如对输入数据进行归一化， 以及对输入数据进行加权线性变换。

另外， 还定义了两个其他的类：Linear 和 Conv1d。Linear 类包含一个前向传播函数 linear, 对输入数据进行线性变换，并可以对输入数据和权重参数加上一个初始化电平项。Conv1d 类是一个卷积层， 可以执行一个 1D 的前向传播。在创建这个类时， 会自动创建一个包含 64 个卷积核的 4D 数据张量， 并且在 forward 方法中， 对输入数据和卷积核进行类似的操作。


```py
class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


```

This is a implementation of a simple multi-head self-attention model using ```py深度简洁的语法`高度抽象的语法`。具体来说，这个模型接收一个输入序列`x`（注意，这个序列中可能会有边缘信息，需要在后续处理中去除），然后将其中的`x`转换为`x`[:, 0]`，即`x`的第一个位置。接下来，这个模型会从两个缓存中选择一个或两个（具体是`kv_cache`还是`xa`）作为当前任务的key，然后使用这两个key来计算一个或多个与`x`相关的value。如果当前任务的key和value在缓存中没有，那么就需要根据`x`的第一个位置进行一些预处理，然后使用这些预处理后的值来进行计算。

最终，这个模型会输出一个包含`x`的序列，同时输出`kv_cache`和`xa`的引用。注意，由于这个模型中没有真正意义上的注意力机制，因此对于任何与`x`的交互，都需要在后续的处理中动态地计算这些信息。


```
def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


```py



这段代码定义了一个名为ResidualAttentionBlock的类，继承自PyTorch中的nn.Module类。这个类的目的是在神经网络中增加残差注意力机制，以提高模型的性能。

在__init__方法中，首先调用父类的构造函数，然后定义了ResidualAttentionBlock中的两个主要组件：MultiHeadAttention和AttentionLayer。MultiHeadAttention是一个多头注意力机制，可以对输入序列中的每个位置进行注意力加权。AttentionLayer是一个残差单元，用于对输入序列中的每个位置进行加权求和，并产生一个上下文向量，这个上下文向量可以用于计算预测的输出。

另外，还定义了一个cross_attn变量，表示是否使用跨注意力机制。如果cross_attn为True，则使用一个与AttentionLayer相同的MultiHeadAttention，但不需要对注意力进行加权求和。

在forward方法中，首先将输入x和xa（如果有）传递给AttentionLayer，并获取其输出结果。然后，根据cross_attn的值，选择是否使用跨注意力机制，并获取相应的MultiHeadAttention输出。将这两个输出结果进行加法运算，并使用MLP层进行前馈，最后返回加权求和后的结果。

通过这种方式，可以有效地提高模型的性能，尤其是在需要对长文本等序列数据进行处理时。


```
class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


```py

这段代码定义了一个名为 `AudioEncoder` 的类，继承自 PyTorch 的 `nn.Module` 类。这个类的主要目的是对输入的音频信号进行预处理和特征提取，以便在后面进行进一步的处理。

在 `__init__` 方法中，定义了模型的参数，包括：

* `n_mels`：音频信号的 Mel 频数，也就是每个时间步的 Mel 值数量。
* `n_ctx`：上下文长度，也就是在哪个时间步结束时停止编码。
* `n_state`：状态长度，也就是编码器中的状态空间大小。
* `n_head`：头数，也就是编码器中的注意力头数。
* `n_layer`：层数，也就是编码器中的层数。

在 `forward` 方法中，实现了对输入的音频信号进行预处理和特征提取，并返回编码后的结果。具体实现包括：

* 提取两个 3D 的卷积层，用于提取 Mel 特征。
* 提取一个 3D 的卷积层，用于提取 context 特征。
* 注册了一个 `positional_embedding` 缓冲区，用于记录每个时间步的 Mel 位置信息。
* 定义了一个 ResidualAttentionBlock，用于在编码器中实现注意力机制。
* 引入了一个 LayerNorm，用于对解码器的输出进行归一化处理。
* 对输入的音频信号进行编码，并返回编码后的结果。

总的来说，这段代码定义了一个用于对音频信号进行编码和解码的模型，可以为模型的输入提供一些有用的预处理和特征提取。


```
class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        len_x = x.shape[1]
        len_e = self.positional_embedding.shape[0]
        assert len_x <= len_e, "incorrect audio shape"
        pos_e = self.positional_embedding[:len_x, :]
        x = (x + pos_e).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


```py

This code defines a class called TextDecoder which is derived from the nn.Module class. It has an initializer that takes in the number of vocabulary elements, the number of context elements, the number of state elements, the number of attention heads, and the number of layers.

The class has a `nn.ModuleList` ofResidualAttentionBlock instances that are stacked in a卒池结构。 The `forward` method takes in two tensors, `x` and `xa`, which represent the input text tokens and the encoded audio features, respectively. It returns the output logits.

The `mask` variable is used to store the binary mask for each audio feature channel, which is used to determine the gradient of the loss with respect to each audio feature channel.

The code also imports the LayerNorm utility function and the torch.triu function, which is used to create triangular smiles.


```
class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits


```py

This is a class that implements the MultiHeadAttention module from the Transformers library. It is designed to process multimedia input, such as speech or text, using multi-head self-attention mechanisms.

The class has an `install_kv_cache_hooks` method for managing the intermediate key-value tensors, which are used for efficient computation during training. This method also installs a hook for the key-value projection modules, so that the intermediate tensors can be saved and reused later during training.

The `is_multilingual` property is used to determine whether the module is designed for a multilingual task or not. In this case, it assumes that the module is multilingual, which means that it can handle input with different languages.

The `MultiHeadAttention` class also inherits from the `nn.Module` class and has the `device` property to specify the device on which the module should run.

The class has an `attention` property, which is a subclass of `Attention`, and implements the multi-head attention mechanism. This allows the module to efficiently capture different attention relationships between the input elements.

Overall, this class is a useful tool for implementing multi-head self-attention mechanisms for multimedia input in PyTorch.


```
class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

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
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    decode = decode_function

```py

# `vencoder/whisper/tokenizer.py`

蜡笔头字符串（key）对应的ISO 639-1国家/地区代码，可以用于进行字符串的国际化。在这个示例中，mk代表马其顿，br代表 Breton，eu代表乌克兰，is代表冰岛，hy代表亚美尼亚，ne代表尼泊尔，mn代表蒙古，bs代表波斯尼亚和黑塞哥维那，kk代表哈萨克斯坦，sq代表阿尔巴尼亚，sw代表斯瓦希里，gl代表加利福尼亚，mn代表马其顿，pa代表巴基斯坦，si代表印度，km代表喀麦隆，sn代表塞内加利亚，yo代表保加利亚，so代表苏丹，af代表南非，oc代表葡萄牙，ka代表哈萨克斯坦，be代表白俄罗斯，tg代表格鲁吉亚，sd代表斯洛文尼亚，gu代表孟加拉国，am代表阿塞拜疆，yi代表亚美尼亚，lo代表拉脱维亚，uz代表乌兹别克斯坦，fo代表法拉茨，ht代表保加利亚，ps代表波兰，yt代表乌克兰，lo代表科威特，中东国家则以aws、as、at代表。



```
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import GPT2TokenizerFast

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

```py

这段代码定义了一个名为“TO_LANGUAGE_CODE”的常量，它是一个字典，其中包含了一些语言的代码名称和相应的语言名称的映射。

具体来说，这个字典包含以下内容：

- 如果您使用“language”键来查找一个已知的语言，它将返回相应的语言代码。
- 如果您使用“language”键来查找一个未知的语言，它将返回npm缓存中相应的语言代码，如果找不到，它将返回“ca”作为默认值。

例如，如果您尝试使用“haitian”键来查找黎语，它将返回“ht”。


```
# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}


```py

Additionally, you may want to consider adding some additional processing to further refine the tokenization. For example, you could:

* Remove寡轻
* Tokenize the text into individual words, and then select the first token id from the words
* Handle text with multiple spaces or other special characters.


```
@dataclass(frozen=True)
class Tokenizer:
    """A thin wrapper around `GPT2TokenizerFast` providing quick access to special tokens"""

    tokenizer: "GPT2TokenizerFast"
    language: Optional[str]
    sot_sequence: Tuple[int]

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids: Union[int, List[int], np.ndarray, torch.Tensor], **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, tokens) -> str:
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        outputs = [[]]
        for token in tokens:
            if token >= self.timestamp_begin:
                timestamp = f"<|{(token - self.timestamp_begin) * 0.02:.2f}|>"
                outputs.append(timestamp)
                outputs.append([])
            else:
                outputs[-1].append(token)
        outputs = [s if isinstance(s, str) else self.tokenizer.decode(s) for s in outputs]
        return "".join(outputs)

    @property
    @lru_cache()
    def eot(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    @lru_cache()
    def sot(self) -> int:
        return self._get_single_token_id("<|startoftranscript|>")

    @property
    @lru_cache()
    def sot_lm(self) -> int:
        return self._get_single_token_id("<|startoflm|>")

    @property
    @lru_cache()
    def sot_prev(self) -> int:
        return self._get_single_token_id("<|startofprev|>")

    @property
    @lru_cache()
    def no_speech(self) -> int:
        return self._get_single_token_id("<|nospeech|>")

    @property
    @lru_cache()
    def no_timestamps(self) -> int:
        return self._get_single_token_id("<|notimestamps|>")

    @property
    @lru_cache()
    def timestamp_begin(self) -> int:
        return self.tokenizer.all_special_ids[-1] + 1

    @property
    @lru_cache()
    def language_token(self) -> int:
        """Returns the token id corresponding to the value of the `language` field"""
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")

        additional_tokens = dict(
            zip(
                self.tokenizer.additional_special_tokens,
                self.tokenizer.additional_special_tokens_ids,
            )
        )
        candidate = f"<|{self.language}|>"
        if candidate in additional_tokens:
            return additional_tokens[candidate]

        raise KeyError(f"Language {self.language} not found in tokenizer.")

    @property
    @lru_cache()
    def all_language_tokens(self) -> Tuple[int]:
        result = []
        for token, token_id in zip(
            self.tokenizer.additional_special_tokens,
            self.tokenizer.additional_special_tokens_ids,
        ):
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)

    @property
    @lru_cache()
    def all_language_codes(self) -> Tuple[str]:
        return tuple(self.decode([l]).strip("<|>") for l in self.all_language_tokens)

    @property
    @lru_cache()
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    @property
    @lru_cache()
    def non_speech_tokens(self) -> Tuple[int]:
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list("\"#()*+/:;<=>@[\\]^_`{|}~「」『』")
        symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.tokenizer.encode(" -")[0], self.tokenizer.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [self.tokenizer.encode(symbol), self.tokenizer.encode(" " + symbol)]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))

    def _get_single_token_id(self, text) -> int:
        tokens = self.tokenizer.encode(text)
        assert len(tokens) == 1, f"{text} is not encoded as a single token"
        return tokens[0]


```py

这段代码定义了一个名为 `build_tokenizer` 的函数，它使用 `GPT2TokenizerFast` 类的一个实例来从预训练的 GPT2 模型中加载词条。它还从环境中删除了一个名为 `TOKENIZERS_PARALLELISM` 的参数，这个参数在 GPT2 的官方文档中没有被使用过，因此它的作用似乎是用于提高训练速度。

接下来，它读取一个名为 `assets` 的工作目录中的 `name.txt` 文件，并将其中的文本读取出来。然后，它遍历 `LANGUAGES` 字典中的所有语言，并将每一种语言的文本添加到 `specials` 列表中。最后，它将 `specials` 列表中的所有特殊标记符（如 `<br>`、`<br/>` 和 `<br>`）添加到已经加载好的预训练模型中。

最终，它返回已经加载好的预训练模型。


```
@lru_cache(maxsize=None)
def build_tokenizer(name: str = "gpt2"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    path = os.path.join(os.path.dirname(__file__), "assets", name)
    tokenizer = GPT2TokenizerFast.from_pretrained(path)

    specials = [
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ]

    tokenizer.add_special_tokens(dict(additional_special_tokens=specials))
    return tokenizer


```py

这段代码定义了一个名为 `get_tokenizer` 的函数，用于从不同的语言和任务中获取语言模型。它使用 `lru_cache` 装饰器来确保缓存不会超过某个设置的最大值。

具体来说，这段代码定义了一个 `Tokenizer` 类，它包含一个从 `LANGUAGES` 键中获取预定义语言列表的函数。如果用户指定了语言，则该函数将使用该语言的预定义 ID。如果用户没有指定任何语言，则该函数将使用默认的预定义 ID。

对于每个使用 `get_tokenizer` 的函数调用，函数首先检查指定的语言是否为当前预定义语言之一。如果是，则函数将返回一个基于该语言的 `Tokenizer` 实例。否则，函数将返回一个基于 `gpt2` 模型的 `Tokenizer` 实例，并将其加载到内存中。

此外，如果指定了 `multilingual` 参数为 `True`，则函数将在其内部创建一个名为 `multilingual` 的类。该类使用 `maxsize` 参数来确保缓存不会超过某个设置的最大值，并使用 `build_tokenizer` 函数来创建一个 `Tokenizer` 实例，类似于上面定义的 `Tokenizer` 类。

最后，函数使用 `TOKENIZER` 类来获取从 `LANGUAGES` 和 `multilingual` 中获取的 tokenizer，将其缓存到 `lru_cache` 缓存中，并返回该缓存实例。


```
@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool,
    *,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
    language: Optional[str] = None,
) -> Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        tokenizer_name = "multilingual"
        task = task or "transcribe"
        language = language or "en"
    else:
        tokenizer_name = "gpt2"
        task = None
        language = None

    tokenizer = build_tokenizer(name=tokenizer_name)
    all_special_ids: List[int] = tokenizer.all_special_ids
    sot: int = all_special_ids[1]
    translate: int = all_special_ids[-6]
    transcribe: int = all_special_ids[-5]

    langs = tuple(LANGUAGES.keys())
    sot_sequence = [sot]
    if language is not None:
        sot_sequence.append(sot + 1 + langs.index(language))
    if task is not None:
        sot_sequence.append(transcribe if task == "transcribe" else translate)

    return Tokenizer(tokenizer=tokenizer, language=language, sot_sequence=tuple(sot_sequence))

```py

# `vencoder/whisper/utils.py`

这段代码使用了多个 Python 标准库模块，其中包括了 `json`、`os`、`sys`、`zlib` 和 `typing`。接下来，我将逐步解释这段代码的作用。

1. `import json`: 引入了 JSON 数据格式模块，以便能够读取和写入 JSON 文件。
2. `import os`: 引入了操作系统模块，以便能够与操作系统交互并获取文件信息。
3. `import sys`: 引入了通用模块，以便能够获取关于系统的基本信息。
4. `import zlib`: 引入了 Zlib 模块，以便能够进行 compress 和 decompress 文件操作。
5. `from typing import Callable, TextIO`: 引入了 `typing` 模块的 `Callable` 类型和 `TextIO` 类型，以便能够实现一些数据操作。

接下来，我将详细解释 `make_safe` 函数的作用。

`make_safe` 函数接收一个字符串参数，并返回一个经过 "make_safe" 处理后的字符串。这个函数的作用是：

1. 如果系统默认编码不是 UTF-8，则将字符串转换为 UTF-8 编码，并返回。
2. 如果系统默认编码已经是 UTF-8，则不做任何处理，并返回原始字符串。

注意：在实际应用中，应该避免使用 `make_safe` 函数来修改原始字符串，因为它可能导致意外的结果。如果需要对原始字符串进行修改，应该使用 `CPython` 的 `str.encode` 和 `str.decode` 方法来实现。


```
import json
import os
import sys
import zlib
from typing import Callable, TextIO

system_encoding = sys.getdefaultencoding()

if system_encoding != "utf-8":
    def make_safe(string):
        # replaces any character not representable using the system default encoding with an '?',
        # avoiding UnicodeEncodeError (https://github.com/openai/whisper/discussions/729).
        return string.encode(system_encoding, errors="replace").decode(system_encoding)
else:
    def make_safe(string):
        # utf-8 can encode any Unicode code point, so no need to do the round-trip encoding
        return string


```py



这段代码定义了三个函数，具体解释如下：

1. `exact_div`函数：

```python
def exact_div(x, y):
   assert x % y == 0
   return x // y
```py

这个函数的作用是判断 `x` 是否可以被 `y` 整除，如果可以，那么返回 `x` 除以 `y` 的商，否则引发出一个异常。

2. `str2bool`函数：

```python
def str2bool(string):
   str2val = {"True": True, "False": False}
   if string in str2val:
       return str2val[string]
   else:
       raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")
```py

这个函数的作用是将一个字符串 `string` 转换为布尔值，即将 `string` 所表示的意义与 `True` 或 `False` 对应。如果 `string` 在 `str2val` 字典中，则返回对应的布尔值，否则引发出一个 `ValueError`。

3. `optional_int`函数：

```python
def optional_int(string):
   return None if string == "None" else int(string)
```py

这个函数的作用是接收一个 `string` 参数，并返回一个 optional 的整数。如果 `string` 是 `None`，则返回 `None`，否则将其转换为整数并返回。


```
def exact_div(x, y):
    assert x % y == 0
    return x // y


def str2bool(string):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string):
    return None if string == "None" else int(string)


```py



这些函数的作用如下：

1. `optional_float(string)`：如果给定的字符串是 "None"，则返回 None，否则将其转换为浮点数并返回。

2. `compression_ratio(text)`：将输入文本中的字节数除以使用 Zlib 压缩后的大约字节数，并返回这个比例。

3. `format_timestamp(seconds, always_include_hours=False, decimal_marker='.')：将输入的浮点数 `seconds` 转换为时间格式，包括小时和分钟，如果 `always_include_hours` 为 True，则将小时标记也一起输出。在输出时，使用了 Decimal 标记来保留两位小数。


```
def optional_float(string):
    return None if string == "None" else float(string)


def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"


```py

这段代码定义了一个名为 `ResultWriter` 的类，旨在将结果(如JSON或文本)写入到指定目录下。

在 `__init__` 方法中，指定了要写入结果的目录 `output_dir`，并使用了 `os` 模块中的 `join` 方法将结果文件名和目录连接起来。

在 `__call__` 方法中，使用了 `os.path.basename` 函数获取音频文件的名称，然后将其与扩展名一起组成要写入的结果文件名。接着使用 `os.path.join` 函数将结果文件目录与结果文件名连接起来。然后使用 `with` 语句打开文件，并使用 `self.write_result` 方法将结果写入文件中。如果 `self.write_result` 方法尚未实现，将会抛出 `NotImplementedError`。

`self.write_result` 方法使用 `with` 语句打开文件并写入结果。然而，这个方法在当前的实现中并没有做任何实际的工作，因此需要手动实现。


```
class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(self, result: dict, audio_path: str):
        audio_basename = os.path.basename(audio_path)
        output_path = os.path.join(self.output_dir, audio_basename + "." + self.extension)

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f)

    def write_result(self, result: dict, file: TextIO):
        raise NotImplementedError


```py

这段代码定义了两个类，一个是`WriteTXT`，另一个是`WriteVTT`。这两个类都是`ResultWriter`的子类，继承自`WriteTXT`和`WriteVTT`。

`WriteTXT`类有一个`extension`成员变量，其值为“txt”。这个成员变量指定了这个类将来的文件扩展名。

`write_result`方法有一个`result`和一个`file`参数。`result`是一个字典，包含一个或多个`segments`键，每个键都是一个字典，包含一个或多个`text`键，实际上是文本的一部分。`file`是一个`TextIO`对象，可以是文件、字符串IO等。

`write_result`方法的主要作用是输出结果。在`write_result`方法中，首先通过遍历`result`的`segments`键，提取出每个文本部分，然后将其打印到`file`对象上。

`WriteVTT`类有一个与`WriteTXT`类相同的`extension`成员变量，但值为“vtt”。这个成员变量指定了这个类将来的文件扩展名。

`write_result`方法与`WriteTXT`类的方法类似，不同之处在于`WriteVTT`类使用了一个不同的格式来输出结果。具体来说，`write_result`方法在打印输出时，使用了`format_timestamp`函数来格式化时间戳。这个函数将`segment`的`start`和`end`值转换为一个时间戳格式，并返回一个字符串。然后，在打印时，将这个时间戳格式化为一个带有`-->`的字符串。

这两个类的`write_result`方法的具体实现可能会因具体的需求而有所不同。


```
class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(self, result: dict, file: TextIO):
        for segment in result["segments"]:
            print(segment['text'].strip(), file=file, flush=True)


class WriteVTT(ResultWriter):
    extension: str = "vtt"

    def write_result(self, result: dict, file: TextIO):
        print("WEBVTT\n", file=file)
        for segment in result["segments"]:
            print(
                f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
                f"{segment['text'].strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


```py

这段代码定义了一个名为 "WriteSRT" 的类，其继承自 "ResultWriter" 类(可能需要导入相关的类)。

该类的扩展名为 "srt"，表示它将写入SRT格式的文本。

该类有一个名为 "write_result" 的方法，该方法接受两个参数，一个是结果字典 "result"，另一个是写入文件的文件对象 "file"。

在 "write_result" 方法中，代码使用两个嵌套的循环来遍历 "result" 字典中的所有 "segment" 键。对于每个 "segment" 键，代码打印出相关的SRT行，并将其写入到 "file" 对象中。

具体来说，代码首先格式化地表示每个SRT行的起始和结束时间，并将其打印出来。然后，代码用 `-->` 连接每个SRT行，以便在结果中正确显示它们。最后，代码在循环内将 `segment` 行的文本内容设置为 "-->" 并将其写入到结果中。

由于 `write_result` 方法使用了 `FileIO` 和 `print` 函数，因此它可以在向文件中写入SRT格式的文本时使用。


```
class WriteSRT(ResultWriter):
    extension: str = "srt"

    def write_result(self, result: dict, file: TextIO):
        for i, segment in enumerate(result["segments"], start=1):
            # write srt lines
            print(
                f"{i}\n"
                f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
                f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
                f"{segment['text'].strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


```py

这段代码定义了一个名为 WriteTSV 的类，该类继承自 ResultWriter 类，用于将文本结果写入到 TSV 格式（ tab-separated values ）文件中。

WriteTSV 类包含一个 extension 成员变量，用于指定文件扩展名，这里设置为 "tsv"。

在该类中，write_result 方法用于将给定的 result 字典中的文本内容写入到指定的文件中。在该方法中，首先打印出每个 segment 的起始和结束时间，以及相应的文本内容，使用三个空格分隔。然后，将每个 segment 的起始和结束时间转换为毫秒，以便在文件中正确对齐。

例如，如果 result 对象包含如下所示的起始和结束时间：

```json
{
   "segments": [
       {
           "start": 100,
           "end": 300,
           "text": "this is a sample text"
       },
       {
           "start": 500,
           "end": 700,
           "text": "another sample text"
       }
   ]
}
```py

那么，write_result 方法将输出以下内容：

```
start 100 end 300 text this is a sample text
start 500 end 700 text another sample text
```py

通过这种方法，WriteTSV 类可以将 result 对象中的文本内容写入到 TSV 文件中，每一行都使用 tab 分隔，每行内容由空格分隔。


```
class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """
    extension: str = "tsv"

    def write_result(self, result: dict, file: TextIO):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment['start']), file=file, end="\t")
            print(round(1000 * segment['end']), file=file, end="\t")
            print(segment['text'].strip().replace("\t", " "), file=file, flush=True)


```py

该代码定义了一个名为 "WriteJSON" 的类，继承自 "ResultWriter" 类。

该类的扩展名为 "json"，表示它将使用 JSON 格式来输出结果。

该类有一个名为 "write_result" 的方法，用于将传入的结果数据写入到指定的文件中。在这个方法中，使用了 "json.dump" 函数将结果数据转换为 JSON 格式并写入到文件中。

该类还定义了一个名为 "get_writer" 的函数，它接受两个参数：输出格式和输出目录。它是一个字典，其中包含所有可以写作器类的实例。如果输出格式为 "all"，则会返回所有writer类的实例。

总的来说，这个类是一个通用的结果写入器，可以用来将各种类型的数据输出到不同的文件中。


```
class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(self, result: dict, file: TextIO):
        json.dump(result, file)


def get_writer(output_format: str, output_dir: str) -> Callable[[dict, TextIO], None]:
    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
    }

    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(result: dict, file: TextIO):
            for writer in all_writers:
                writer(result, file)

        return write_all

    return writers[output_format](output_dir)


```py

# `vencoder/whisper/__init__.py`

我需要更具体的上下文来回答你的问题。可以请你提供更多背景信息或者完整的代码，这样我才能够更好地解释代码的作用。


```

```