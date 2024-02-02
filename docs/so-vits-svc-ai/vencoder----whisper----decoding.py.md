# `so-vits-svc\vencoder\whisper\decoding.py`

```py
# 导入必要的模块和类
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from .audio import CHUNK_LENGTH  # 从自定义模块中导入 CHUNK_LENGTH 常量
from .tokenizer import Tokenizer, get_tokenizer  # 从自定义模块中导入 Tokenizer 类和 get_tokenizer 函数
from .utils import compression_ratio  # 从自定义模块中导入 compression_ratio 函数

# 如果是类型检查阶段
if TYPE_CHECKING:
    from .model import Whisper  # 从自定义模块中导入 Whisper 类

# 使用 torch.no_grad() 修饰的函数，表示在该函数中不进行梯度计算
@torch.no_grad()
def detect_language(model: "Whisper", mel: Tensor, tokenizer: Tokenizer = None) -> Tuple[Tensor, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : Tensor, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    # 如果未指定 tokenizer，则使用默认的 tokenizer
    if tokenizer is None:
        tokenizer = get_tokenizer(model.is_multilingual)
    # 如果 tokenizer 的语言属性为 None，或者语言标记不在起始标记序列中，则抛出数值错误
    if tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence:
        raise ValueError("This model doesn't have language tokens so it can't perform lang id")

    # 判断输入的 mel 是否为单个样本，如果是，则扩展维度
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    # 如果 mel 的形状不符合模型要求，则通过编码器对 mel 进行编码
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # 使用单个标记 startoftranscript 进行前向传播
    n_audio = mel.shape[0]
    x = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # 创建形状为 [n_audio, 1] 的张量
    logits = model.logits(x, mel)[:, 0]  # 计算模型的输出 logits
    # 收集检测到的语言；抑制所有非语言标记
    # 创建一个全为 True 的布尔张量，形状与logits的最后一个维度相同
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    # 将所有语言标记的索引设置为 False
    mask[list(tokenizer.all_language_tokens)] = False
    # 将被屏蔽的标记对应的logits值设置为负无穷
    logits[:, mask] = -np.inf
    # 获取每个音频片段的最可能语言标记
    language_tokens = logits.argmax(dim=-1)
    # 计算每个标记的概率
    language_token_probs = logits.softmax(dim=-1).cpu()
    # 将每个音频片段的语言标记概率转换为字典形式
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]
    
    # 如果只有一个音频片段，则返回单个语言标记和概率
    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]
    
    # 返回语言标记和概率
    return language_tokens, language_probs
# 定义一个名为 DecodingOptions 的数据类，使用 @dataclass 装饰器，且不可变
@dataclass(frozen=True)
class DecodingOptions:
    # 任务类型，默认为 "transcribe"，可以是 "transcribe" 或 "translate"
    task: str = "transcribe"
    # 音频所在的语言，如果为 None，则使用检测到的语言
    language: Optional[str] = None

    # 与采样相关的选项
    temperature: float = 0.0
    # 最大要采样的标记数
    sample_len: Optional[int] = None
    # 在 t > 0 时要收集的独立样本数
    best_of: Optional[int] = None
    # 在 t == 0 时的 beam search 中的 beam 数
    beam_size: Optional[int] = None
    # beam search 中的耐心程度（https://arxiv.org/abs/2204.05424）
    patience: Optional[float] = None

    # 用于排名生成结果的选项（beam 或 best-of-N 样本）
    length_penalty: Optional[float] = None
    # Google NMT 中的 "alpha"，如果为 None，则默认为长度规范化

    # 提示、前缀和标记抑制
    prompt: Optional[Union[str, List[int]]] = None
    # 用于前一个上下文的文本或标记
    prefix: Optional[Union[str, List[int]]] = None
    # 是否抑制空白输出
    suppress_blank: bool = True

    # 要抑制的标记 id 列表（或逗号分隔的标记 id）
    # "-1" 将抑制在 `tokenizer.non_speech_tokens()` 中定义的一组符号
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"

    # 时间戳采样选项
    without_timestamps: bool = False
    # 不能晚于此初始时间戳
    max_initial_timestamp: Optional[float] = 1.0

    # 实现细节
    fp16: bool = True  # 大多数计算使用 fp16


# 定义一个名为 DecodingResult 的数据类，使用 @dataclass 装饰器，且不可变
@dataclass(frozen=True)
class DecodingResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    # 定义一个浮点型变量 temperature，并初始化为 NaN
    temperature: float = np.nan
    # 定义一个浮点型变量 compression_ratio，并初始化为 NaN
    compression_ratio: float = np.nan
# 定义一个推断类，包含logits、rearrange_kv_cache和cleanup_caching方法
class Inference:
    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        """Perform a forward pass on the decoder and return per-token logits"""
        # 执行解码器的前向传播，并返回每个标记的logits
        raise NotImplementedError

    def rearrange_kv_cache(self, source_indices) -> None:
        """Update the key-value cache according to the updated beams"""
        # 根据更新的beam更新键值缓存
        raise NotImplementedError

    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        # 在解码完成后清理任何资源或钩子
        pass


# 继承Inference类，定义PyTorchInference类
class PyTorchInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        # 初始化方法，接受模型和初始标记长度作为参数
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        # 计算logits
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()

        if tokens.shape[-1] > self.initial_token_length:
            # 只需要使用最后一个标记，除了第一次前向传播
            tokens = tokens[:, -1:]

        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)

    def cleanup_caching(self):
        # 清理缓存
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        # 重新排列键值缓存
        for module, tensor in self.kv_cache.items():
            # 更新键值缓存以包含所选序列
            self.kv_cache[module] = tensor[source_indices].detach()


# 定义一个序列排名类，包含rank方法
class SequenceRanker:
    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        # 给定一组样本和它们的累积对数概率，返回每组样本的索引，以选择为最终结果
        raise NotImplementedError


# 继承SequenceRanker类，定义MaximumLikelihoodRanker类
class MaximumLikelihoodRanker(SequenceRanker):
    """
    """
    # 选择具有最高对数概率的样本，使用简单的长度规范化或Google NMT论文中的长度惩罚进行惩罚
    """

    # 初始化方法，接受长度惩罚参数
    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    # 对tokens中的序列进行排名，根据sum_logprobs中的对数概率
    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]):
        # 计算得分的函数
        def scores(logprobs, lengths):
            result = []
            # 遍历对数概率和长度，计算得分
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    # 如果没有长度惩罚，使用长度作为惩罚
                    penalty = length
                else:
                    # 根据Google NMT论文计算长度惩罚
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # 获取具有最高得分的序列
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]
# 定义一个 TokenDecoder 类，用于解码序列
class TokenDecoder:
    # 重置解码新序列时的状态变量
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    # 更新解码器状态，选择下一个标记的方式基于当前的轨迹和logits
    def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor) -> Tuple[Tensor, bool]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    # 完成搜索并返回最终的候选序列
    def finalize(
        self, tokens: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Sequence[Sequence[Tensor]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : Tensor, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : Tensor, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[Tensor]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


# 继承 TokenDecoder 类，实现 GreedyDecoder 类
class GreedyDecoder(TokenDecoder):
    # 初始化函数，接受温度和结束标记作为参数
    def __init__(self, temperature: float, eot: int):
        # 将温度和结束标记保存为对象的属性
        self.temperature = temperature
        self.eot = eot

    # 更新函数，接受 tokens、logits 和 sum_logprobs 作为参数，返回更新后的 tokens 和是否完成的标志
    def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor) -> Tuple[Tensor, bool]:
        # 获取温度
        temperature = self.temperature
        # 如果温度为0，选择概率最大的 token 作为下一个 token
        if temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        # 如果温度不为0，根据温度从 logits 中采样下一个 token
        else:
            next_tokens = Categorical(logits=logits / temperature).sample()

        # 计算 log softmax，并获取当前 token 的 log 概率
        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        # 更新总的 log 概率
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        # 将已经结束的序列的 token 设置为结束标记
        next_tokens[tokens[:, -1] == self.eot] = self.eot
        # 将新的 token 添加到 tokens 中
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        # 检查是否所有序列都已经完成
        completed = (tokens[:, -1] == self.eot).all()
        # 返回更新后的 tokens 和是否完成的标志
        return tokens, completed

    # 完成函数，接受 tokens 和 sum_logprobs 作为参数，返回添加结束标记后的 tokens 和总的 log 概率
    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        # 确保每个序列最后都有一个结束标记
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        # 返回添加结束标记后的 tokens 和总的 log 概率
        return tokens, sum_logprobs.tolist()
# 定义一个 BeamSearchDecoder 类，继承自 TokenDecoder 类
class BeamSearchDecoder(TokenDecoder):
    # 初始化方法，接受 beam_size（束搜索大小）、eot（结束标记）、inference（推断）、patience（耐心）参数
    def __init__(self, beam_size: int, eot: int, inference: Inference, patience: Optional[float] = None):
        # 设置束搜索大小
        self.beam_size = beam_size
        # 设置结束标记
        self.eot = eot
        # 设置推断对象
        self.inference = inference
        # 设置耐心，如果没有指定则默认为1.0
        self.patience = patience or 1.0
        # 计算最大候选项数量
        self.max_candidates: int = round(beam_size * self.patience)
        # 初始化已完成的序列为空
        self.finished_sequences = None

        # 断言最大候选项数量大于0
        assert self.max_candidates > 0, f"Invalid beam size ({beam_size}) or patience ({patience})"

    # 重置方法，将已完成的序列设置为None
    def reset(self):
        self.finished_sequences = None

    # 完成方法，接受前导标记和总对数概率作为参数
    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        # 将总对数概率转移到CPU上
        sum_logprobs = sum_logprobs.cpu()
        # 遍历已完成的序列
        for i, sequences in enumerate(self.finished_sequences):
            # 当完成的序列数量小于束搜索大小时
            if len(sequences) < self.beam_size:
                # 对总对数概率进行排序，选择最大的未完成序列
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    # 将前导标记和结束标记组成序列，并计算总对数概率
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    # 当完成的序列数量达到束搜索大小时，跳出循环
                    if len(sequences) >= self.beam_size:
                        break

        # 将已完成的序列转换为张量列表
        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()] for sequences in self.finished_sequences
        ]
        # 将总对数概率转换为浮点数列表
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        # 返回序列张量列表和总对数概率列表
        return tokens, sum_logprobs


# 定义一个 LogitFilter 类
class LogitFilter:
    # 定义一个方法，用于对logits进行过滤或掩码操作，直接在原地修改logits的数值

    # 参数：
    # logits: Tensor, shape = (n_batch, vocab_size)
    #     当前步骤的概率分布的每个token的logits

    # tokens: Tensor, shape = (n_batch, current_sequence_length)
    #     到目前为止上下文中的所有token，包括前缀和sot_sequence token

    def apply(self, logits: Tensor, tokens: Tensor) -> None:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        # 抛出一个未实现的错误，提示子类需要实现这个方法
        raise NotImplementedError
# 定义一个名为SuppressBlank的类，继承自LogitFilter
class SuppressBlank(LogitFilter):
    # 初始化方法，接受tokenizer对象和sample_begin整数作为参数
    def __init__(self, tokenizer: Tokenizer, sample_begin: int):
        # 将传入的tokenizer赋值给实例变量tokenizer
        self.tokenizer = tokenizer
        # 将传入的sample_begin赋值给实例变量sample_begin

    # 应用方法，接受logits和tokens张量作为参数
    def apply(self, logits: Tensor, tokens: Tensor):
        # 如果tokens的第二个维度长度等于sample_begin
        if tokens.shape[1] == self.sample_begin:
            # 将logits中对应空格和结束标记的位置设为负无穷
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf


# 定义一个名为SuppressTokens的类，继承自LogitFilter
class SuppressTokens(LogitFilter):
    # 初始化方法，接受suppress_tokens序列作为参数
    def __init__(self, suppress_tokens: Sequence[int]):
        # 将传入的suppress_tokens转换为列表并赋值给实例变量suppress_tokens
        self.suppress_tokens = list(suppress_tokens)

    # 应用方法，接受logits和tokens张量作为参数
    def apply(self, logits: Tensor, tokens: Tensor):
        # 将logits中对应suppress_tokens位置设为负无穷
        logits[:, self.suppress_tokens] = -np.inf


# 定义一个名为ApplyTimestampRules的类，继承自LogitFilter
class ApplyTimestampRules(LogitFilter):
    # 初始化方法，接受tokenizer对象、sample_begin整数和max_initial_timestamp_index可选整数作为参数
    def __init__(
        self, tokenizer: Tokenizer, sample_begin: int, max_initial_timestamp_index: Optional[int]
    ):
        # 将传入的tokenizer赋值给实例变量tokenizer
        self.tokenizer = tokenizer
        # 将传入的sample_begin赋值给实例变量sample_begin
        self.sample_begin = sample_begin
        # 将传入的max_initial_timestamp_index赋值给实例变量max_initial_timestamp_index
        self.max_initial_timestamp_index = max_initial_timestamp_index
    # 对给定的logits和tokens进行处理
    def apply(self, logits: Tensor, tokens: Tensor):
        # 如果存在不需要时间戳的特殊标记，则将对应位置的logits设为负无穷
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # 时间戳必须成对出现，除非直接在EOT之前；相应位置的logits需要进行屏蔽
        for k in range(tokens.shape[0]):
            # 获取当前样本的token序列
            seq = [t for t in tokens[k, self.sample_begin :].tolist()]
            # 判断最后一个token是否为时间戳
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            # 判断倒数第二个token是否为时间戳
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # 必须是非时间戳
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # 不能是普通文本token
                    logits[k, : self.tokenizer.eot] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            # 抑制在开头生成非时间戳token
            logits[:, : self.tokenizer.timestamp_begin] = -np.inf

            # 应用`max_initial_timestamp`选项
            if self.max_initial_timestamp_index is not None:
                last_allowed = self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                logits[:, last_allowed + 1 :] = -np.inf

        # 如果时间戳的概率之和超过其他任何token，则采样时间戳
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            # 计算时间戳的log概率之和
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(dim=-1)
            # 获取普通文本token的最大log概率
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            # 如果时间戳的log概率之和大于普通文本token的最大log概率，则将对应位置的logits设为负无穷
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf
class DecodingTask:
    # 定义类属性
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    # 验证解码选项的有效性，并返回验证后的选项
    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        # 如果同时给定了 beam_size 和 best_of，则抛出数值错误
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        # 如果温度为 0，且同时给定了 best_of，则抛出数值错误
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        # 如果给定了耐心值，但未给定 beam_size，则抛出数值错误
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        # 如果给定了长度惩罚值，但其值不在 0 到 1 之间，则抛出数值错误
        if options.length_penalty is not None and not (0 <= options.length_penalty <= 1):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        # 返回验证后的选项
        return options

    # 获取初始标记的方法，并返回标记的元组
    def _get_initial_tokens(self) -> Tuple[int]:
        # 将起始标记转换为列表
        tokens = list(self.sot_sequence)
        prefix = self.options.prefix
        prompt = self.options.prompt

        # 如果存在前缀，则将其转换为标记并添加到 tokens 中
        if prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip()) if isinstance(prefix, str) else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        # 如果存在提示，则将其转换为标记并添加到 tokens 中
        if prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip()) if isinstance(prompt, str) else prompt
            )
            tokens = [self.tokenizer.sot_prev] + prompt_tokens[-(self.n_ctx // 2 - 1) :] + tokens

        # 返回标记的元组
        return tuple(tokens)
    # 获取需要被抑制的标记列表
    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        # 如果抑制标记是字符串，则将其转换为整数列表
        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        # 如果列表中包含-1，则移除并添加非语音标记
        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        # 如果抑制标记为空或者为None，则将其解释为空列表
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        # 添加特殊标记到抑制标记列表中
        suppress_tokens.extend(
            [self.tokenizer.sot, self.tokenizer.sot_prev, self.tokenizer.sot_lm]
        )
        # 如果存在无语音标记，则添加到抑制标记列表中
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        # 返回排序去重后的抑制标记元组
        return tuple(sorted(set(suppress_tokens)))

    # 获取音频特征
    def _get_audio_features(self, mel: Tensor):
        # 如果选项中指定使用fp16，则将mel转换为半精度
        if self.options.fp16:
            mel = mel.half()

        # 如果mel的形状为指定的音频上下文和音频状态的形状，则直接使用mel作为音频特征
        if mel.shape[-2:] == (self.model.dims.n_audio_ctx, self.model.dims.n_audio_state):
            # encoded audio features are given; skip audio encoding
            print("encoded audio features are given; skip audio encoding")
            audio_features = mel
        else:
            # 否则使用模型的编码器对mel进行编码得到音频特征
            print(mel.shape)
            print("===============================")
            audio_features = self.model.encoder(mel)

        # 如果音频特征的数据类型不符合指定的数据类型，则返回类型错误
        if audio_features.dtype != (torch.float16 if self.options.fp16 else torch.float32):
            return TypeError(f"audio_features has an incorrect dtype: {audio_features.dtype}")

        # 返回音频特征
        return audio_features
    # 检测语言，根据音频特征和标记返回语言列表和语言概率
    def _detect_language(self, audio_features: Tensor, tokens: Tensor):
        # 初始化语言列表，每个样本对应一个语言
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        # 如果语言选项为空或任务为语言识别，则进行语言检测
        if self.options.language is None or self.options.task == "lang_id":
            # 调用模型进行语言检测，返回语言标记和语言概率
            lang_tokens, lang_probs = self.model.detect_language(audio_features, self.tokenizer)
            # 根据概率选择最可能的语言作为样本的语言
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            # 如果语言选项为空，则将语言标记写入标记张量的指定位置
            if self.options.language is None:
                tokens[:, self.sot_index + 1] = lang_tokens  # write language tokens

        # 返回语言列表和语言概率
        return languages, lang_probs

    # 主循环，根据音频特征和标记进行推理
    def _main_loop(self, audio_features: Tensor, tokens: Tensor):
        # 断言音频特征和标记的样本数相同
        assert audio_features.shape[0] == tokens.shape[0]
        n_batch = tokens.shape[0]
        # 初始化对数概率和无语音概率
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        try:
            # 循环进行推理
            for i in range(self.sample_len):
                # 获取推理结果的logits
                logits = self.inference.logits(tokens, audio_features)

                # 如果是第一次循环且标记器有无语音标记，则保存无语音概率
                if i == 0 and self.tokenizer.no_speech is not None:  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # 只考虑最后一个标记的logits
                logits = logits[:, -1]

                # 应用logit过滤器，例如抑制或对logits施加惩罚
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # 使用选定的下一个标记扩展标记张量
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                # 如果已完成或标记张量长度超过指定长度，则跳出循环
                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            # 清理推理缓存
            self.inference.cleanup_caching()

        # 返回标记张量、对数概率和无语音概率
        return tokens, sum_logprobs, no_speech_probs

    # 禁用梯度计算
    @torch.no_grad()
# 使用 torch.no_grad() 上下文管理器，确保在此函数中不会进行梯度计算
@torch.no_grad()
# 定义 decode 函数，用于对 30 秒音频片段的 Mel 频谱图进行解码
def decode(model: "Whisper", mel: Tensor, options: DecodingOptions = DecodingOptions()) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    # 检查输入的 Mel 频谱图是单个还是多个
    single = mel.ndim == 2
    # 如果是单个 Mel 频谱图，则在第 0 维度上增加一个维度
    if single:
        mel = mel.unsqueeze(0)
    # 创建 DecodingTask 实例，并运行解码任务
    result = DecodingTask(model, options).run(mel)
    
    # 如果输入的是单个 Mel 频谱图，则从结果列表中取出第一个结果
    if single:
        result = result[0]

    # 返回解码结果
    return result
```