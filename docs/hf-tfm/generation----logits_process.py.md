# `.\generation\logits_process.py`

```
# 设置代码文件的编码格式为 UTF-8
# 版权声明，指明该代码的版权归 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证的要求，否则不得使用此文件
# 可以在以下链接获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不附带任何形式的明示或暗示担保或条件
# 请查看许可证了解详细信息

# 导入所需的模块和函数
import inspect
import math
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

# 导入 numpy 和 torch 模块
import numpy as np
import torch

# 从相对路径导入 utils 模块中的 add_start_docstrings 函数
from ..utils import add_start_docstrings
# 从 logging 模块中导入 get_logger 函数
from ..utils.logging import get_logger

# 获取当前模块的 logger 对象
logger = get_logger(__name__)

# 定义一个原始文档字符串，用于记录 logits 处理函数的输入和返回说明
LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。[什么是输入 ID?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            语言建模头的预测分数。当不使用 beam search 时，这些可以是每个词汇表的 logits；
            当使用 beam search 时，这些可以是每个词汇表标记的对数 softmax
        
    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: 处理后的预测分数。
"""

class LogitsProcessor:
    """所有生成过程中可以应用的 logits 处理器的抽象基类。"""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 抽象方法，需要被继承此类的类实现具体逻辑
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsWarper:
    """所有多项式采样生成过程中可以应用的 logits 转换器的抽象基类。"""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 抽象方法，需要被继承此类的类实现具体逻辑
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    """
    可用于创建一个 [`LogitsProcessor`] 或 [`LogitsWarper`] 列表，以便随后处理输入张量 `scores`。
    此类继承自列表，并添加了一个特定的 *__call__* 方法来对输入应用每个 [`LogitsProcessor`] 或 [`LogitsWarper`]。
    """
    # 定义一个特殊方法 `__call__`，使得对象可以像函数一样被调用
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        # 遍历对象中所有的处理器
        for processor in self:
            # 获取处理器的 __call__ 方法的参数签名
            function_args = inspect.signature(processor.__call__).parameters
            # 如果处理器的 __call__ 方法参数个数大于2
            if len(function_args) > 2:
                # 检查所有除了前两个参数（self 和 input_ids）外的参数是否在 kwargs 中
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    # 如果有未传递的参数，则抛出 ValueError 异常
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                # 调用处理器的 __call__ 方法，传递 input_ids, scores 和 kwargs
                scores = processor(input_ids, scores, **kwargs)
            else:
                # 调用处理器的 __call__ 方法，传递 input_ids 和 scores
                scores = processor(input_ids, scores)

        # 返回处理后的预测分数
        return scores
# 定义一个新的 logits 处理器类，继承自 LogitsProcessor
class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0. Note that, for decoder-only models
    like most LLMs, the length includes the prompt.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer("A number:", return_tensors="pt")
    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one

    >>> # setting `min_length` to a value smaller than the uncontrolled output length has no impact
    >>> gen_out = model.generate(**inputs, min_length=3)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one

    >>> # setting a larger `min_length` will force the model to generate beyond its natural ending point, which is not
    >>> # necessarily incorrect
    >>> gen_out = model.generate(**inputs, min_length=10)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one thousand, nine hundred and ninety-four
    ```

    """

    # 初始化方法，接受最小长度和 EOS 标记 ID
    def __init__(self, min_length: int, eos_token_id: Union[int, List[int]]):
        # 检查 min_length 必须为非负整数
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a non-negative integer, but is {min_length}")

        # 如果 eos_token_id 是单个整数，则转换为列表形式
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        # 检查 eos_token_id 必须为正整数列表
        if not all(isinstance(i, int) for i in eos_token_id) or any(i < 0 for i in eos_token_id):
            logger.warning(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

        # 初始化对象的属性
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    # 调用方法，处理输入的 logits 和分数，并返回处理后的分数
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 获取当前输入的长度
        cur_len = input_ids.shape[-1]
        # 如果当前长度小于最小长度
        if cur_len < self.min_length:
            # 将所有 EOS 标记的分数设为负无穷
            for i in self.eos_token_id:
                scores[:, i] = -float("inf")
        # 返回处理后的分数
        return scores


# 定义另一个新的 logits 处理器类，继承自 LogitsProcessor
class MinNewTokensLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.
    Contrarily to [`MinLengthLogitsProcessor`], this processor ignores the prompt.
    ```

    # 注释继续在下一个代码块中
    Args:
        prompt_length_to_skip (`int`):
            要跳过的输入标记长度。与 `generate` 一起使用时，不是有效的参数，因为它会自动分配输入长度。
        min_new_tokens (`int`):
            下面这个得分为 `-float("Inf")` 的条件最小 *新* 标记长度。
        eos_token_id (`Union[int, List[int]]`):
            *结束序列* 标记的 ID。可选择使用列表设置多个 *结束序列* 标记。

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer(["A number:"], return_tensors="pt")
    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one

    >>> # 设置 `min_new_tokens` 将强制模型生成超出其自然结束点，这不一定是错误的
    >>> gen_out = model.generate(**inputs, min_new_tokens=2)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one thousand
    ```
    """

    def __init__(self, prompt_length_to_skip: int, min_new_tokens: int, eos_token_id: Union[int, List[int]]):
        # 验证并设置 `prompt_length_to_skip` 和 `min_new_tokens` 参数
        for arg_name, arg_value in [
            ("prompt_length_to_skip", prompt_length_to_skip),
            ("min_new_tokens", min_new_tokens),
        ]:
            if not isinstance(arg_value, int) or arg_value < 0:
                raise ValueError(f"`{arg_name}` 必须是正整数，但其值为 {arg_value}")

        # 验证并设置 `eos_token_id` 参数，确保其为正整数列表
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if not all(isinstance(i, int) for i in eos_token_id) or any(i < 0 for i in eos_token_id):
            logger.warning(f"`eos_token_id` 必须是正整数列表，但其值为 {eos_token_id}")

        # 初始化对象的属性
        self.prompt_length_to_skip = prompt_length_to_skip
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 计算新生成标记的长度
        new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip
        # 如果生成的新标记长度小于设定的最小值，将相应的 `eos_token_id` 的得分设为 `-float("inf")`
        if new_tokens_length < self.min_new_tokens:
            for i in self.eos_token_id:
                scores[:, i] = -float("inf")

        return scores
# TemperatureLogitsWarper 类，继承自 LogitsWarper
# 用于温度（指数缩放输出概率分布），有效地控制预测标记的随机性
# 常与 TopPLogitsWarper 和 TopKLogitsWarper 一起使用

class TemperatureLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] for temperature (exponential scaling output probability distribution), which effectively means
    that it can control the randomness of the predicted tokens. Often used together with [`TopPLogitsWarper`] and
    [`TopKLogitsWarper`].

    <Tip>

    Make sure that `do_sample=True` is included in the `generate` arguments otherwise the temperature value won't have
    any effect.

    </Tip>

    Args:
        temperature (`float`):
            Strictly positive float value used to modulate the logits distribution. A value smaller than `1` decreases
            randomness (and vice versa), with `0` being equivalent to shifting all probability mass to the most likely
            token.

    Examples:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(0)  # for reproducibility

    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> model.config.pad_token_id = model.config.eos_token_id
    >>> inputs = tokenizer(["Hugging Face Company is"], return_tensors="pt")

    >>> # With temperature=1.0, the default, we consistently get random outputs due to random sampling.
    >>> generate_kwargs = {"max_new_tokens": 10, "do_sample": True, "temperature": 1.0, "num_return_sequences": 2}
    >>> outputs = model.generate(**inputs, **generate_kwargs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ['Hugging Face Company is a joint venture between GEO Group, one of',
    'Hugging Face Company is not an exact science – but what we believe does']

    >>> # However, with temperature close to 0, it approximates greedy decoding strategies (invariant)
    >>> generate_kwargs["temperature"] = 0.0001
    >>> outputs = model.generate(**inputs, **generate_kwargs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ['Hugging Face Company is a company that has been around for over 20 years',
    'Hugging Face Company is a company that has been around for over 20 years']
    ```
    """

    def __init__(self, temperature: float):
        # 检查温度参数是否为有效的浮点数且大于0
        if not isinstance(temperature, float) or not (temperature > 0):
            # 如果温度不是有效的正浮点数，抛出值错误异常
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
            # 如果温度为0，提醒用户可以设置 `do_sample=False` 来实现贪婪解码策略
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."
            raise ValueError(except_msg)

        # 设置实例的温度属性
        self.temperature = temperature

    # 添加文档字符串，参考 LOGITS_PROCESSOR_INPUTS_DOCSTRING
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    # 定义类的特殊方法 __call__，使得对象可以像函数一样被调用
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 将分数 scores 除以温度 temperature，用于调整输出的分布
        scores = scores / self.temperature
        # 返回调整后的分数
        return scores
class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that prevents the repetition of previous tokens through a penalty. This penalty is applied at
    most once per token. Note that, for decoder-only models like most LLMs, the considered tokens include the prompt.

    In the original [paper](https://arxiv.org/pdf/1909.05858.pdf), the authors suggest the use of a penalty of around
    1.2 to achieve a good balance between truthful generation and lack of repetition. To penalize and reduce
    repetition, use `penalty` values above 1.0, where a higher value penalizes more strongly. To reward and encourage
    repetition, use `penalty` values between 0.0 and 1.0, where a lower value rewards more strongly.

    Args:
        penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 penalizes previously generated
            tokens. Between 0.0 and 1.0 rewards previously generated tokens.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> # Initializing the model and tokenizer for it
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    >>> inputs = tokenizer(["I'm not going to"], return_tensors="pt")

    >>> # This shows a normal generate without any specific parameters
    >>> summary_ids = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    I'm not going to be able to do that. I'm going to be able to do that

    >>> # This generates a penalty for repeated tokens
    >>> penalized_ids = model.generate(**inputs, repetition_penalty=1.1)
    >>> print(tokenizer.batch_decode(penalized_ids, skip_special_tokens=True)[0])
    I'm not going to be able to do that. I'll just have to go out and play
    ```
    """

    def __init__(self, penalty: float):
        # 检查 penalty 是否为正的浮点数，否则抛出错误
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 从 scores 中选择对应 input_ids 的分数
        score = torch.gather(scores, 1, input_ids)

        # 如果 score < 0，则乘以 penalty 以减少 token 的概率
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        # 将修正后的分数重新写入 scores 中对应的位置
        scores.scatter_(1, input_ids, score)
        return scores


class EncoderRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that works similarly to [`RepetitionPenaltyLogitsProcessor`], but with an *inverse* penalty
    that is applied to the tokens present in the prompt. In other words, a penalty above 1.0 increases the odds of
    selecting tokens that were present in the prompt.
    def __init__(self, penalty: float, encoder_input_ids: torch.LongTensor):
        # 检查 penalty 是否为 float 类型且大于 0，否则抛出数值错误异常
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        # 计算实际的惩罚值，即将 1 除以 penalty
        self.penalty = 1 / penalty
        # 将输入的 encoder_input_ids 赋值给实例变量
        self.encoder_input_ids = encoder_input_ids

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 从 scores 中按列索引提取与 encoder_input_ids 相对应的分数
        score = torch.gather(scores, 1, self.encoder_input_ids)

        # 如果分数小于 0，则乘以 penalty 值以增加 token 的概率
        # 如果分数大于等于 0，则除以 penalty 值以降低 token 的概率
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        # 将处理后的 score 根据 encoder_input_ids 的索引位置更新到 scores 中
        scores.scatter_(1, self.encoder_input_ids, score)
        # 返回更新后的 scores
        return scores
class TopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off. Often
    used together with [`TemperatureLogitsWarper`] and [`TopKLogitsWarper`].

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(0)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 0, 2, 2. 2, 2, 2, 2

    >>> # With `top_p` sampling, the output gets restricted to high-probability tokens.
    >>> # Pro tip: In practice, LLMs use `top_p` in the 0.9-0.95 range.
    >>> outputs = model.generate(**inputs, do_sample=True, top_p=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        # 初始化 TopPLogitsWarper 对象，设置 top-p 概率截断参数
        top_p = float(top_p)
        # 检查 top_p 参数是否在有效范围 (0, 1) 内，否则引发 ValueError 异常
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        # 检查 min_tokens_to_keep 参数是否为正整数，否则引发 ValueError 异常
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        # 设置对象的属性
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    # 添加文档字符串作为类的一部分，描述输入参数
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    # 定义一个调用函数，接受输入的token IDs和对应的分数，返回处理后的分数
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 对分数进行升序排序，并返回排序后的分数和索引
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        # 对排序后的分数进行 softmax 处理并计算累积概率
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # 移除累积概率超过 top_p 阈值的token（累积概率为0的token保留）
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # 至少保留 min_tokens_to_keep 个token
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # 将排序后的移除指标张量按照排序后的索引分散到原始索引位置
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        # 使用 filter_value 替换需要移除的token对应的分数
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        # 返回处理后的分数张量
        return scores
class TopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements. Often used together
    with [`TemperatureLogitsWarper`] and [`TopPLogitsWarper`].

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(0)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: A, B, C, D", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: A, B, C, D, G, H, I. A, M

    >>> # With `top_k` sampling, the output gets restricted the k most likely tokens.
    >>> # Pro tip: In practice, LLMs use `top_k` in the 5-50 range.
    >>> outputs = model.generate(**inputs, do_sample=True, top_k=2)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: A, B, C, D, E, F, G, H, I
    ```
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        # 检查并初始化 `top_k` 参数，确保其为正整数
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        # 将 `top_k` 设为不小于 `min_tokens_to_keep` 的值，设置过滤值 `filter_value`
        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 确保 `top_k` 不超过 `scores` 的最后一维大小，以避免越界
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        # 移除概率小于 `top-k` 中最后一个概率值的所有 token
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TypicalLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs typical decoding. Inspired on how humans use language, it prioritizes tokens whose
    log probability is close to the entropy of the token probability distribution. This means that the most likely
    tokens may be discarded in the process.

    See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information.
    # 初始化函数，用于创建一个新的实例对象
    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        # 将输入参数 mass 转换为 float 类型
        mass = float(mass)
        # 检查 mass 参数是否在有效范围 (0, 1) 内，如果不是则引发 ValueError 异常
        if not (mass > 0 and mass < 1):
            raise ValueError(f"`typical_p` has to be a float > 0 and < 1, but is {mass}")
        # 检查 min_tokens_to_keep 是否为正整数，如果不是则引发 ValueError 异常
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        # 设置对象的 filter_value 属性为传入的 filter_value 参数值
        self.filter_value = filter_value
        # 设置对象的 mass 属性为处理后的 mass 参数值
        self.mass = mass
        # 设置对象的 min_tokens_to_keep 属性为处理后的 min_tokens_to_keep 参数值
        self.min_tokens_to_keep = min_tokens_to_keep
    # 定义一个调用方法，接收输入的token ID张量和得分张量，并返回处理后的得分张量
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 计算熵（entropy）
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # 移位并排序
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # 根据累积概率阈值移除部分token
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind.clamp_(max=sorted_scores.shape[-1] - 1)
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        # 使用指定的值过滤掉需要移除的token的得分
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
# 定义一个名为 EpsilonLogitsWarper 的类，继承自 LogitsWarper 类
class EpsilonLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs epsilon-sampling, i.e. restricting to tokens with `prob >= epsilon`. Takes the
    largest min_tokens_to_keep tokens if no tokens satisfy this constraint. See [Truncation Sampling as Language Model
    Desmoothing](https://arxiv.org/abs/2210.15191) for more information.

    Args:
        epsilon (`float`):
            If set to > 0, only the most tokens with probabilities `epsilon` or higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:
    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(0)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 0, 2, 2. 2, 2, 2, 2

    >>> # With epsilon sampling, the output gets restricted to high-probability tokens. Note that this is similar to
    >>> # Top P sampling, which restricts tokens based on their cumulative probability.
    >>> # Pro tip: The paper recomends using `epsilon_cutoff` values between 3e-4 and 9e-4
    >>> outputs = model.generate(**inputs, do_sample=True, epsilon_cutoff=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    # 初始化方法，设置 epsilon-sampling 的参数
    def __init__(self, epsilon: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        # 将 epsilon 强制转换为 float 类型
        epsilon = float(epsilon)
        # 如果 epsilon 不在有效范围 (0, 1) 内，抛出异常
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"`epsilon_cutoff` has to be a float > 0 and < 1, but is {epsilon}")

        # 将 min_tokens_to_keep 强制转换为 int 类型
        min_tokens_to_keep = int(min_tokens_to_keep)
        # 如果 min_tokens_to_keep 不大于等于 1，抛出异常
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

        # 初始化对象的属性
        self.epsilon = epsilon
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    # 添加 LogitsProcessor 的输入文档字符串
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    # 定义一个调用方法，接收输入的张量 input_ids 和分数张量 scores，并返回一个分数张量
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 使用 softmax 函数计算分数张量在最后一个维度上的概率分布
        probabilities = scores.softmax(dim=-1)
        # 创建一个布尔张量，指示哪些索引的概率低于阈值 self.epsilon
        indices_to_remove = probabilities < self.epsilon

        # 确保保留至少 self.min_tokens_to_keep 个最高概率的单词
        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # 进行安全性检查，取最小值
        # 使用 torch.topk 函数获取最高分数的前 top_k 个分数，并与 indices_to_remove 合并
        indices_to_remove = indices_to_remove & (scores < torch.topk(scores, top_k)[0][..., -1, None])

        # 使用指定的 self.filter_value 替换 scores 张量中 indices_to_remove 为 True 的元素
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        # 返回处理后的分数张量
        return scores
class EtaLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs eta-sampling, a technique to filter out tokens with probabilities below a dynamic
    cutoff value, `eta`, which is calculated based on a combination of the hyperparameter `epsilon` and the entropy of
    the token probabilities, i.e. `eta := min(epsilon, sqrt(epsilon * e^-entropy(probabilities)))`. Takes the largest
    min_tokens_to_keep tokens if no tokens satisfy this constraint. It addresses the issue of poor quality in long
    samples of text generated by neural language models leading to more coherent and fluent text. See [Truncation
    Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more information. Note: `do_sample`
    must be set to `True` for this `LogitsWarper` to work.


    Args:
        epsilon (`float`):
            A float value in the range (0, 1). Hyperparameter used to calculate the dynamic cutoff value, `eta`. The
            suggested values from the paper ranges from 3e-4 to 4e-3 depending on the size of the model.
        filter_value (`float`, *optional*, defaults to -inf):
            All values that are found to be below the dynamic cutoff value, `eta`, are set to this float value. This
            parameter is useful when logits need to be modified for very low probability tokens that should be excluded
            from generation entirely.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Specifies the minimum number of tokens that must be kept for generation, regardless of their probabilities.
            For example, if `min_tokens_to_keep` is set to 1, at least one token will always be kept for generation,
            even if all tokens have probabilities below the cutoff `eta`.

    Examples:
    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(0)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 0, 2, 2. 2, 2, 2, 2

    >>> # With eta sampling, the output gets restricted to high-probability tokens. You can see it as a dynamic form of
    >>> # epsilon sampling that adapts its cutoff probability based on the entropy (high entropy = lower cutoff).
    >>> # Pro tip: The paper recomends using `eta_cutoff` values between 3e-4 to 4e-3
    >>> outputs = model.generate(**inputs, do_sample=True, eta_cutoff=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """
    def __init__(self, epsilon: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        # 将 epsilon 转换为浮点数并进行验证
        epsilon = float(epsilon)
        # 检查 epsilon 的取值范围是否在 (0, 1) 之间，否则引发 ValueError 异常
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"`eta_cutoff` has to be a float > 0 and < 1, but is {epsilon}")

        # 将 min_tokens_to_keep 转换为整数并进行验证
        min_tokens_to_keep = int(min_tokens_to_keep)
        # 检查 min_tokens_to_keep 是否大于等于 1，否则引发 ValueError 异常
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

        # 初始化对象的属性
        self.epsilon = torch.tensor(epsilon)
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 计算自适应阈值 eta
        probabilities = scores.softmax(dim=-1)  # 计算概率分布
        entropy = torch.distributions.Categorical(logits=scores).entropy()  # 计算熵
        eta = torch.min(self.epsilon, torch.sqrt(self.epsilon) * torch.exp(-entropy))[..., None]  # 计算 eta

        # 确定需要移除的索引
        indices_to_remove = probabilities < eta

        # 保留概率最高的 min_tokens_to_keep 个词
        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # 安全检查，确保 top_k 不超过 scores 的最后一个维度大小
        indices_to_remove = indices_to_remove & (scores < torch.topk(scores, top_k)[0][..., -1, None])

        # 根据 indices_to_remove 进行掩码操作，用 filter_value 替换需要移除的位置的分数
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
# 定义一个函数 `_get_ngrams`，用于生成给定大小的 n-gram 并保存在字典中
def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    """
    Assume ngram_size=2 and prev_input_ids=tensor([[40, 2883, 2712, 4346]]). The output of generated ngrams look like
    this {(40,): [2883], (2883,): [2712], (2712,): [4346]}.

    Args:
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        prev_input_ids (`torch.Tensor`):
           Generated token ids for the current hypothesis.
        num_hypos (`int`):
            The number of hypotheses for which n-grams need to be generated.

    Returns:
        generated_ngrams (`dict`):
            Dictionary of generated ngrams.
    """
    # 初始化一个空的字典列表，每个假设 (索引) 对应一个字典，数量为 num_hypos
    generated_ngrams = [{} for _ in range(num_hypos)]
    # 遍历每个假设
    for idx in range(num_hypos):
        # 将当前假设的生成的 token 转换为列表
        gen_tokens = prev_input_ids[idx].tolist()
        # 获取当前假设的生成 ngram 字典
        generated_ngram = generated_ngrams[idx]
        # 遍历当前假设生成的 token 列表，生成大小为 ngram_size 的 ngram
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            # 将生成的 ngram 加入到生成的 ngram 字典中
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


# 定义一个函数 `_get_generated_ngrams`，用于确定基于先前生成的 ngram 的当前假设的禁用 token
def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    """
    Determines the banned tokens for the current hypothesis based on previously generated n-grams.

    Args:
        banned_ngrams (`dict`):
            A dictionary containing previously generated n-grams for each hypothesis.
        prev_input_ids (`torch.Tensor`):
            Generated token ids for the current hypothesis.
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        cur_len (`int`):
            The current length of the token sequences for which the n-grams are being checked.

    Returns:
        List of tokens that are banned.
    """
    # 计算当前需要检查的 ngram 的起始索引
    start_idx = cur_len + 1 - ngram_size
    # 获取当前假设生成的 ngram 的索引元组
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    # 返回先前生成的 ngrams 中与当前 ngram 索引匹配的禁用 tokens 列表
    return banned_ngrams.get(ngram_idx, [])


# 定义一个函数 `_calc_banned_ngram_tokens`，用于计算禁用的 ngram token
def _calc_banned_ngram_tokens(
    ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
) -> List[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    # 如果当前生成的 token 数量小于 ngram_size，则返回空的禁用 tokens 列表
    if cur_len + 1 < ngram_size:
        return [[] for _ in range(num_hypos)]
    # 生成当前假设的 ngrams
    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
    # 获取每个假设的禁用 tokens 列表
    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens
class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    N-grams are groups of "n" consecutive words, characters, or tokens taken from a sequence of text. Given the
    sentence: "She runs fast", the bi-grams (n=2) would be ("she", "runs") and ("runs", "fast"). In text generation,
    avoiding repetitions of word sequences provides a more diverse output. This [`LogitsProcessor`] enforces no
    repetition of n-grams by setting the scores of banned tokens to negative infinity which eliminates those tokens
    from consideration when further processing the scores. Note that, for decoder-only models like most LLMs, the
    prompt is also considered to obtain the n-grams.
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

    <Tip>

    Use n-gram penalties with care. For instance, penalizing 2-grams (bigrams) in an article about the city of New York
    might lead to undesirable outcomes where the city's name appears only once in the entire text.
    [Reference](https://huggingface.co/blog/how-to-generate)

    </Tip>

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    >>> inputs = tokenizer(["Today I"], return_tensors="pt")

    >>> output = model.generate(**inputs)
    >>> print(tokenizer.decode(output[0], skip_special_tokens=True))
    Today I’m not sure if I’m going to be able to do it.

    >>> # Now let's add ngram size using `no_repeat_ngram_size`. This stops the repetitions ("I’m") in the output.
    >>> output = model.generate(**inputs, no_repeat_ngram_size=2)
    >>> print(tokenizer.decode(output[0], skip_special_tokens=True))
    Today I’m not sure if I can get a better understanding of the nature of this issue
    ```
    """

    def __init__(self, ngram_size: int):
        # 检查并初始化 ngram_size，确保其为正整数
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 获取当前 batch 的假设数量
        num_batch_hypotheses = scores.shape[0]
        # 获取当前输入序列的长度
        cur_len = input_ids.shape[-1]
        # 计算当前 batch 每个假设中不允许出现的 n-gram tokens
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)
        # 将不允许出现的 token 的分数设为负无穷，以便在后续处理中排除这些 token
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores


class EncoderNoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that works similarly to [`NoRepeatNGramLogitsProcessor`], but applied exclusively to prevent
    """
    Initializes an instance of the ultimate n-gram blocker.

    Args:
        encoder_ngram_size (`int`):
            Size of the n-grams that should not be repeated in the decoder.
        encoder_input_ids (`torch.LongTensor`):
            Tensor containing input IDs for the encoder.

    """

    def __init__(self, encoder_ngram_size: int, encoder_input_ids: torch.LongTensor):
        # Check if encoder_ngram_size is a positive integer
        if not isinstance(encoder_ngram_size, int) or encoder_ngram_size <= 0:
            raise ValueError(
                f"`encoder_ngram_size` has to be a strictly positive integer, but is {encoder_ngram_size}"
            )
        # Store the n-gram size
        self.ngram_size = encoder_ngram_size
        
        # Ensure encoder_input_ids is 2-dimensional
        if len(encoder_input_ids.shape) == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        
        # Store batch size
        self.batch_size = encoder_input_ids.shape[0]
        
        # Generate n-grams from the encoder input IDs
        self.generated_ngrams = _get_ngrams(encoder_ngram_size, encoder_input_ids, self.batch_size)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Calculate number of hypotheses
        num_hypos = scores.shape[0]
        
        # Calculate number of beams per hypothesis
        num_beams = num_hypos // self.batch_size
        
        # Current length of input_ids
        cur_len = input_ids.shape[-1]
        
        # List of banned tokens for each hypothesis
        banned_batch_tokens = [
            _get_generated_ngrams(
                self.generated_ngrams[hypo_idx // num_beams], input_ids[hypo_idx], self.ngram_size, cur_len
            )
            for hypo_idx in range(num_hypos)
        ]
        
        # Apply -inf score to banned tokens in scores tensor
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")
        
        return scores
class SequenceBiasLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that applies an additive bias on sequences. The bias is applied to the last token of a sequence
    when the next generated token can complete it. Consequently, to take the most of biasing sequences with more than
    one token, consider using beam methods (to gracefully work around partially completed sequences that have a
    negative bias) and applying the bias to their prefixes (to ensure the bias is applied earlier).

    <Tip>

    In order to get the token ids of the sequences that you want to bias, make sure to set `add_prefix_space=True` when
    initializing the tokenizer, and use `tokenizer(bad_words, add_special_tokens=False).input_ids`. The
    `add_prefix_space` argument is only supported for some slow tokenizers, as fast tokenizers' prefixing behaviours
    come from `pre tokenizers`. Read more [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

    </Tip>

    Args:
        sequence_bias (`Dict[Tuple[int], float]`):
            Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
            sequence being selected, while negative biases do the opposite. If a sequence has a length of 1, its bias
            will always be applied. Otherwise, the bias will only be applied if the sequence in question is about to be
            completed (in the token selection step after this processor is applied).

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["The full name of Donald is Donald"], return_tensors="pt")

    >>> summary_ids = model.generate(inputs["input_ids"], max_new_tokens=4)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald J. Trump Jr

    >>> # Now let's control generation through a bias. Please note that the tokenizer is initialized differently!
    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("openai-community/gpt2", add_prefix_space=True)


    >>> def get_tokens_as_tuple(word):
    ...     return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])


    >>> # If we add a negative bias without beam search, it may become "stuck" in a prefix without good continuations
    >>> sequence_bias = {get_tokens_as_tuple("Trump"): -10.0}
    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald J. Donald,

    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)

    """

    def __init__(self, sequence_bias):
        """
        Initialize the SequenceBiasLogitsProcessor with a sequence bias dictionary.

        Args:
            sequence_bias (`Dict[Tuple[int], float]`): A dictionary mapping sequences of tokens to their bias values.
        """
        super().__init__()
        self.sequence_bias = sequence_bias

    def __call__(self, input_ids, scores):
        """
        Apply the sequence bias to the logits.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            scores (torch.Tensor): Logits (scores) for each token.

        Returns:
            torch.Tensor: Modified logits after applying sequence bias.
        """
        # Determine the sequence length
        seq_len = input_ids.size(1)
        # Get the last token's token IDs
        last_token_ids = input_ids[:, -1].tolist()

        # Check if the last token is in the sequence_bias dictionary
        if tuple(last_token_ids) in self.sequence_bias:
            # Apply bias to the last token's logits
            scores[:, -1] += self.sequence_bias[tuple(last_token_ids)]

        return scores
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald Rumsfeld,

    >>> # We can also add a positive bias to nudge the model towards specific tokens or continuations
    >>> sequence_bias = {get_tokens_as_tuple("Donald Duck"): 10.0}
    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald Duck.
    ```
    """

    # 初始化函数，接收一个序列偏置的字典作为参数
    def __init__(self, sequence_bias: Dict[Tuple[int], float]):
        self.sequence_bias = sequence_bias  # 初始化序列偏置
        self._validate_arguments()  # 调用内部方法验证参数

        # 下面的变量在第一次调用时才会被填充（为了向后兼容性，词汇大小将在第一次使用中推断出来，因此在这里不进行初始化）
        self.length_1_bias = None  # 长度为1的偏置变量
        self.prepared_bias_variables = False  # 准备好的偏置变量标志位

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    # 调用方法，接收输入的input_ids和scores，返回经过处理后的scores
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 1 - 准备偏置张量。这仅在第一次调用logit处理器时需要。
        if not self.prepared_bias_variables:
            self._prepare_bias_variables(scores)

        # 2 - 准备一个空的偏置张量以添加
        bias = torch.zeros_like(scores)

        # 3 - 包含长度为1时的偏置
        bias += self.length_1_bias

        # 4 - 包含长度大于1时的偏置，确定可以完成的偏置序列
        for sequence_ids, sequence_bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:  # 序列长度为1，已应用偏置
                continue
            if len(sequence_ids) > input_ids.shape[1]:  # 序列比上下文长，忽略
                continue
            prefix_length = len(sequence_ids) - 1
            last_token = sequence_ids[-1]
            matching_rows = torch.eq(
                input_ids[:, -prefix_length:],
                torch.tensor(sequence_ids[:-1], dtype=input_ids.dtype, device=input_ids.device),
            ).prod(dim=1)
            bias[:, last_token] += torch.where(
                matching_rows.bool(),
                torch.tensor(sequence_bias, device=input_ids.device),
                torch.tensor(0.0, device=input_ids.device),
            )

        # 5 - 将偏置应用于得分
        scores = scores + bias
        return scores
    # 准备偏置变量，根据模型得分张量的形状确定词汇表大小
    def _prepare_bias_variables(self, scores: torch.FloatTensor):
        vocabulary_size = scores.shape[-1]

        # 检查偏置的标记是否超出范围
        invalid_biases = []
        for sequence_ids in self.sequence_bias:
            for token_id in sequence_ids:
                if token_id >= vocabulary_size:
                    invalid_biases.append(token_id)
        if len(invalid_biases) > 0:
            raise ValueError(
                f"The model vocabulary size is {vocabulary_size}, but the following tokens were being biased: "
                f"{invalid_biases}"
            )

        # 预计算要应用的偏置张量。长度为1的序列单独处理，因为可以使用更简单的逻辑应用。
        self.length_1_bias = torch.zeros((vocabulary_size,), dtype=torch.float).to(scores.device)
        for sequence_ids, bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:
                self.length_1_bias[sequence_ids[-1]] = bias

        # 标记已准备好偏置变量
        self.prepared_bias_variables = True

    # 验证参数是否合法
    def _validate_arguments(self):
        sequence_bias = self.sequence_bias
        # 检查 `sequence_bias` 是否是非空字典
        if not isinstance(sequence_bias, dict) or len(sequence_bias) == 0:
            raise ValueError(f"`sequence_bias` has to be a non-empty dictionary, but is {sequence_bias}.")
        # 检查 `sequence_bias` 的键是否是元组
        if any(not isinstance(sequence_ids, tuple) for sequence_ids in sequence_bias.keys()):
            raise ValueError(f"`sequence_bias` has to be a dict with tuples as keys, but is {sequence_bias}.")
        # 检查 `sequence_bias` 的键是否为非空的正整数元组
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in sequence_ids)
            or len(sequence_ids) == 0
            for sequence_ids in sequence_bias.keys()
        ):
            raise ValueError(
                f"Each key in `sequence_bias` has to be a non-empty tuple of positive integers, but is "
                f"{sequence_bias}."
            )
        # 检查 `sequence_bias` 的值是否都是浮点数
        if any(not isinstance(bias, float) for bias in sequence_bias.values()):
            raise ValueError(f"`sequence_bias` has to be a dict with floats as values, but is {sequence_bias}.")
"""
[`LogitsProcessor`] that enforces that specified sequences will never be selected.

<Tip>

In order to get the token ids of the words that should not appear in the generated text, make sure to set
`add_prefix_space=True` when initializing the tokenizer, and use `tokenizer(bad_words,
add_special_tokens=False).input_ids`. The `add_prefix_space` argument is only supported for some slow tokenizers,
as fast tokenizers' prefixing behaviours come from `pre tokenizers`. Read more
[here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

</Tip>

Args:
    bad_words_ids (`List[List[int]]`):
        List of list of token ids that are not allowed to be generated.
    eos_token_id (`Union[int, List[int]]`):
        The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

Examples:


>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
>>> inputs = tokenizer(["In a word, the cake is a"], return_tensors="pt")

>>> output_ids = model.generate(inputs["input_ids"], max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
>>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
In a word, the cake is a bit of a mess.

>>> # Now let's take the bad words out. Please note that the tokenizer is initialized differently
>>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("openai-community/gpt2", add_prefix_space=True)


>>> def get_tokens_as_list(word_list):
...     "Converts a sequence of words into a list of tokens"
...     tokens_list = []
...     for word in word_list:
...         tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
...         tokens_list.append(tokenized_word)
...     return tokens_list


>>> bad_words_ids = get_tokens_as_list(word_list=["mess"])
>>> output_ids = model.generate(
...     inputs["input_ids"], max_new_tokens=5, bad_words_ids=bad_words_ids, pad_token_id=tokenizer.eos_token_id
... )
>>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
In a word, the cake is a bit of a surprise.

"""
    # 初始化函数，接收两个参数：bad_words_ids 是包含不良词汇列表的列表，eos_token_id 是结束标记的整数或整数列表
    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: Union[int, List[int]]):
        # 将参数 bad_words_ids 存储在对象属性中
        self.bad_word_ids = bad_words_ids
        # 调用内部方法验证参数的有效性
        self._validate_arguments()

        # 过滤掉 bad_words_ids 中包含的 EOS 标记
        if eos_token_id is None:
            eos_token_id = []
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        # 使用 lambda 函数过滤 bad_words_ids，确保不包含任何 EOS 标记的序列
        bad_words_ids = list(
            filter(lambda bad_token_seq: all(bad_token_seq != [i] for i in eos_token_id), bad_words_ids)
        )

        # 将禁止序列设置为负无穷的偏置字典
        sequence_bias = {tuple(sequence): float("-inf") for sequence in bad_words_ids}
        # 调用父类初始化方法，传递序列偏置字典作为参数
        super().__init__(sequence_bias=sequence_bias)

    # 内部方法，验证 bad_words_ids 参数的有效性
    def _validate_arguments(self):
        # 将对象属性 bad_word_ids 赋值给局部变量 bad_words_ids
        bad_words_ids = self.bad_word_ids
        # 检查 bad_words_ids 是否为非空列表
        if not isinstance(bad_words_ids, list) or len(bad_words_ids) == 0:
            raise ValueError(f"`bad_words_ids` has to be a non-empty list, but is {bad_words_ids}.")
        # 检查 bad_words_ids 中的每个元素是否为列表
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids):
            raise ValueError(f"`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.")
        # 检查 bad_words_ids 中每个列表的元素是否为正整数
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in bad_word_ids)
            for bad_word_ids in bad_words_ids
        ):
            raise ValueError(
                f"Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}."
            )
class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned constrained
    generation. See [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) for more information.

    Args:
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the
            next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID
            `batch_id`.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer("Alice and Bob", return_tensors="pt")

    >>> # By default, it continues generating according to the model's logits
    >>> outputs = model.generate(**inputs, max_new_tokens=5)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    Alice and Bob are friends

    >>> # We can contrain it with `prefix_allowed_tokens_fn` to force a certain behavior based on a prefix.
    >>> # For instance, we can force an entire entity to be generated when its beginning is detected.
    >>> entity =  tokenizer(" Bob Marley", return_tensors="pt").input_ids[0]  # 3 tokens
    >>> def prefix_allowed_tokens_fn(batch_id, input_ids):
    ...     '''
    ...     Attempts to generate 'Bob Marley' when 'Bob' is detected.
    ...     In this case, `batch_id` is not used, but you can set rules for each batch member.
    ...     '''
    ...     if input_ids[-1] == entity[0]:
    ...         return entity[1]
    ...     elif input_ids[-2] == entity[0] and input_ids[-1] == entity[1]:
    ...         return entity[2]
    ...     return list(range(tokenizer.vocab_size))  # If no match, allow all tokens

    >>> outputs = model.generate(**inputs, max_new_tokens=5, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    Alice and Bob Marley
    ```

    """

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
        # 初始化函数，接受两个参数：prefix_allowed_tokens_fn 控制允许的生成标记，num_beams 控制束搜索的数量
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    # 定义一个方法，接受输入的torch.LongTensor类型的input_ids和torch.FloatTensor类型的scores，并返回一个torch.FloatTensor类型的结果
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 创建一个与scores形状相同的张量，填充为负无穷大，用作掩码
        mask = torch.full_like(scores, -math.inf)
        
        # 遍历input_ids，按照_beam_num划分batch_id和beam_sent
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            # 遍历每个beam_sent中的beam_id和sent
            for beam_id, sent in enumerate(beam_sent):
                # 调用_prefix_allowed_tokens_fn方法获取允许的前缀标记
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                # 如果prefix_allowed_tokens列表为空，抛出ValueError异常
                if len(prefix_allowed_tokens) == 0:
                    raise ValueError(
                        f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                        f"This means that the constraint is unsatisfiable. Please check your implementation"
                        f"of `prefix_allowed_tokens_fn` "
                    )
                # 将mask中指定位置（batch_id * self._num_beams + beam_id行）的允许标记位置设为0
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        # 返回scores与mask相加后的结果
        return scores + mask
# 定义一个继承自 LogitsProcessor 的类，用于实现多样化的 Beam Search 算法。
class HammingDiversityLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces diverse beam search.

    Note that this logits processor is only effective for [`PreTrainedModel.group_beam_search`]. See [Diverse Beam
    Search: Decoding Diverse Solutions from Neural Sequence Models](https://arxiv.org/pdf/1610.02424.pdf) for more
    details.

    Traditional beam search often generates very similar sequences across different beams.
    `HammingDiversityLogitsProcessor` addresses this by penalizing beams that generate tokens already chosen by other
    beams in the same time step.

    Args:
        diversity_penalty (`float`):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. A higher `diversity_penalty` will enforce greater diversity among the beams. Adjusting
            this value can help strike a balance between diversity and natural likelihood.
        num_beams (`int`):
            Number of beams for beam search. 1 means no beam search.
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    >>> import torch

    >>> # Initialize the model and tokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

    >>> # A long text about the solar system
    >>> text = (
    ...     "The Solar System is a gravitationally bound system comprising the Sun and the objects that orbit it, "
    ...     "either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight "
    ...     "planets, with the remainder being smaller objects, such as the five dwarf planets and small Solar System "
    ...     "bodies. The Solar System formed 4.6 billion years ago from the gravitational collapse of a giant "
    ...     "interstellar molecular cloud."
    ... )
    >>> inputs = tokenizer("summarize: " + text, return_tensors="pt")

    >>> # Generate diverse summary
    >>> outputs_diverse = model.generate(
    ...     **inputs,
    ...     num_beam_groups=2,
    ...     diversity_penalty=10.0,
    ...     max_length=100,
    ...     num_beams=4,
    ...     num_return_sequences=2,
    ... )
    >>> summaries_diverse = tokenizer.batch_decode(outputs_diverse, skip_special_tokens=True)

    >>> # Generate non-diverse summary
    >>> outputs_non_diverse = model.generate(
    ...     **inputs,
    ...     max_length=100,
    ...     num_beams=4,
    ...     num_return_sequences=2,
    ... )
    >>> summary_non_diverse = tokenizer.batch_decode(outputs_non_diverse, skip_special_tokens=True)
    # 初始化方法，用于设置多样性惩罚、束搜索数和束搜索组数的初始值
    def __init__(self, diversity_penalty: float, num_beams: int, num_beam_groups: int):
        # 检查并确保 diversity_penalty 是大于0的浮点数
        if not isinstance(diversity_penalty, float) or (not diversity_penalty > 0.0):
            raise ValueError("`diversity_penalty` should be a float strictly larger than 0.")
        self._diversity_penalty = diversity_penalty  # 设置多样性惩罚参数

        # 检查并确保 num_beams 是大于1的整数
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams  # 设置束搜索数

        # 检查并确保 num_beam_groups 是大于1的整数，且不超过 num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        if num_beam_groups > num_beams:
            raise ValueError("`beam_groups` has to be smaller or equal to `num_beams`.")
        self._num_sub_beams = num_beams // num_beam_groups  # 计算并设置每个束搜索组的子束搜索数

    # 对象被调用时执行的方法，用于执行束搜索过程
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        current_tokens: torch.LongTensor,
        beam_group_idx: int,
    ) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            current_tokens (`torch.LongTensor` of shape `(batch_size)`):
                Indices of input sequence tokens in the vocabulary, corresponding to the tokens selected by the other
                beam groups in the current generation step.
            beam_group_idx (`int`):
                The index of the beam group currently being processed.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.
        """
        # hamming diversity: penalise using same token in current group which was used in previous groups at
        # the same time step
        batch_size = current_tokens.shape[0] // self._num_beams  # 计算批次大小
        group_start_idx = beam_group_idx * self._num_sub_beams  # 计算当前处理的 beam 组的起始索引
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)  # 计算当前处理的 beam 组的结束索引，确保不超过总数
        group_size = group_end_idx - group_start_idx  # 计算当前处理的 beam 组的大小
        vocab_size = scores.shape[-1]  # 获取词汇表大小

        if group_start_idx == 0:
            return scores  # 如果是第一个组，直接返回原始预测分数

        for batch_idx in range(batch_size):
            # predicted tokens of last time step of previous groups
            previous_group_tokens = current_tokens[
                batch_idx * self._num_beams : batch_idx * self._num_beams + group_start_idx
            ]  # 获取前面组在当前时间步的预测 token

            token_frequency = torch.bincount(previous_group_tokens, minlength=vocab_size).to(scores.device)
            # 计算前面组使用的 token 频率，并转移到与 scores 设备一致的张量上

            scores[batch_idx * group_size : (batch_idx + 1) * group_size] -= self._diversity_penalty * token_frequency
            # 根据多样性惩罚系数，减少当前组的预测分数

        return scores
class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the first generated token. Used with encoder-decoder
    models.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    >>> inputs = tokenizer("Translate from English to German: I love cats.", return_tensors="pt")

    >>> # By default, it continues generating according to the model's logits
    >>> outputs = model.generate(**inputs, max_new_tokens=10)
    >>> print(tokenizer.batch_decode(outputs)[0])
    <pad> Ich liebe Kitty.</s>

    >>> # We can use `forced_bos_token_id` to force the start of generation with an encoder-decoder model
    >>> # (including forcing it to end straight away with an EOS token)
    >>> outputs = model.generate(**inputs, max_new_tokens=10, forced_bos_token_id=tokenizer.eos_token_id)
    >>> print(tokenizer.batch_decode(outputs)[0])
    <pad></s>
    ```
    """

    def __init__(self, bos_token_id: int):
        # 初始化方法，设置强制起始 token 的 ID
        self.bos_token_id = bos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 获取当前生成序列的长度
        cur_len = input_ids.shape[-1]
        # 如果当前长度为1，即刚开始生成
        if cur_len == 1:
            # 获取 logits 的可能 token 数量
            num_tokens = scores.shape[1]
            # 将除了指定的强制起始 token 之外的 logits 设置为负无穷大，确保不会被生成
            scores[:, [i for i in range(num_tokens) if i != self.bos_token_id]] = -float("inf")
            # 将强制起始 token 的 logits 设置为0，确保它被生成
            scores[:, self.bos_token_id] = 0
        return scores


class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`Union[int, List[int]]`):
            The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a
            list to set multiple *end-of-sequence* tokens.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2, 3", return_tensors="pt")

    >>> # By default, it continues generating according to the model's logits
    >>> outputs = model.generate(**inputs, max_new_tokens=10)
    >>> print(tokenizer.batch_decode(outputs)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8

    >>> # `forced_eos_token_id` ensures the generation ends with a EOS token
    ```
    """

    def __init__(self, max_length: int, eos_token_id: Union[int, List[int]]):
        # 初始化方法，设置强制结束 token 的 ID 或 IDs
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 获取当前生成序列的长度
        cur_len = input_ids.shape[-1]
        # 如果达到最大长度，强制设置生成序列的最后 token(s)
        if cur_len == self.max_length:
            if isinstance(self.eos_token_id, int):
                # 如果是单个 EOS token ID，将除了它之外的 logits 设置为负无穷大
                scores[:, [i for i in range(scores.shape[1]) if i != self.eos_token_id]] = -float("inf")
                # 将 EOS token 的 logits 设置为0，确保它被生成
                scores[:, self.eos_token_id] = 0
            else:
                # 如果是多个 EOS token IDs，将除了它们之外的 logits 设置为负无穷大
                for eos_id in self.eos_token_id:
                    scores[:, [i for i in range(scores.shape[1]) if i != eos_id]] = -float("inf")
                # 将所有 EOS tokens 的 logits 设置为0，确保它们中的任意一个被生成
                for eos_id in self.eos_token_id:
                    scores[:, eos_id] = 0
        return scores
    # 使用模型生成文本输出，限制生成的新标记数目为10个，强制结束标记使用给定的 eos_token_id
    outputs = model.generate(**inputs, max_new_tokens=10, forced_eos_token_id=tokenizer.eos_token_id)
    
    # 解码生成的输出序列并打印第一个结果
    print(tokenizer.batch_decode(outputs)[0])
class InfNanRemoveLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that removes all `nan` and `inf` values to avoid the generation method to fail. Note that using
    the logits processor should only be used if necessary since it can slow down the generation method.

    This logits processor has no `generate` example, as there shouldn't be a correct combination of flags that warrants
    its use.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # set all nan values to 0.0
        scores[scores != scores] = 0.0  # 将所有的NaN值设置为0.0

        # set all +/-inf values to max/min possible value
        scores[scores == float("inf")] = torch.finfo(scores.dtype).max  # 将所有的正无穷值设置为数据类型的最大值
        scores[scores == float("-inf")] = torch.finfo(scores.dtype).min  # 将所有的负无穷值设置为数据类型的最小值

        return scores
    """
    该类的构造函数初始化对象的属性，并计算长度调整的起始点和衰减因子。

    def __init__(
        self,
        exponential_decay_length_penalty: Tuple[int, float],  # 接收一个元组，包含衰减长度和衰减因子
        eos_token_id: Union[int, List[int]],  # 接收结束标记的 ID，可以是单个整数或整数列表
        input_ids_seq_length: int,  # 输入的序列长度
    ):
        # 计算调整起始点，考虑输入序列的长度
        self.regulation_start = exponential_decay_length_penalty[0] + input_ids_seq_length
        # 设置衰减因子
        self.regulation_factor = exponential_decay_length_penalty[1]
        # 如果结束标记是整数，则转换为列表
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        # 存储结束标记的 ID
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 获取当前输入序列的长度
        cur_len = input_ids.shape[-1]
        # 如果当前长度超过了调整起始点
        if cur_len > self.regulation_start:
            # 对每个结束标记执行以下操作
            for i in self.eos_token_id:
                # 计算惩罚的索引，基于当前长度和调整起始点
                penalty_idx = cur_len - self.regulation_start
                # 支持负对数，计算绝对值的惩罚，并添加到原始对数中
                scores[:, i] = scores[:, i] + torch.abs(scores[:, i]) * (pow(self.regulation_factor, penalty_idx) - 1)
        # 返回调整后的分数
        return scores
    """
class LogitNormalization(LogitsProcessor, LogitsWarper):
    r"""
    [`LogitsWarper`] and [`LogitsProcessor`] for normalizing the scores using log-softmax. It's important to normalize
    the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in
    this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
    the scores are normalized when comparing the hypotheses.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    >>> import torch

    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2, 3", return_tensors="pt")

    >>> # By default, the scores are not normalized -- the sum of their exponentials is NOT a normalized probability
    >>> # distribution, summing to 1
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    >>> print(torch.sum(torch.exp(outputs.scores[-1])))
    tensor(816.3250)

    >>> # Normalizing them may have a positive impact on beam methods, or when using the scores on your application
    >>> outputs = model.generate(**inputs, renormalize_logits=True, return_dict_in_generate=True, output_scores=True)
    >>> print(torch.sum(torch.exp(outputs.scores[-1])))
    tensor(1.0000)
    ```
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    # 定义一个类方法，继承自 LogitsProcessor 类，并添加了文档字符串描述输入
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 对 scores 执行 log_softmax 操作，使得 scores 在最后一个维度上进行 log-softmax 归一化
        scores = scores.log_softmax(dim=-1)
        # 返回处理后的 scores
        return scores


class SuppressTokensAtBeginLogitsProcessor(LogitsProcessor):
    r"""
    [`SuppressTokensAtBeginLogitsProcessor`] supresses a list of tokens as soon as the `generate` function starts
    generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` are
    not generated at the begining. Originally created for
    [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

    Examples:

    ```python
    >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> # Whisper has `begin_suppress_tokens` set by default (= `[220, 50256]`). 50256 is the EOS token, so this means
    >>> # it can't generate and EOS token in the first iteration, but it can in the others.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    ```

    """
    >>> print(outputs.scores[1][0, 50256])  # 1 (and not 0) is the first freely generated token
    tensor(-inf)
    >>> print(outputs.scores[-1][0, 50256])  # in other places we can see some probability mass for EOS
    tensor(29.9010)

    >>> # If we disable `begin_suppress_tokens`, we can generate EOS in the first iteration.
    >>> outputs = model.generate(
    ...     **inputs, return_dict_in_generate=True, output_scores=True, begin_suppress_tokens=None
    ... )
    >>> print(outputs.scores[1][0, 50256])
    tensor(11.2027)
    ```

    """
    
    # 初始化函数，接收两个参数：begin_suppress_tokens（起始抑制令牌列表）和begin_index（起始索引）
    def __init__(self, begin_suppress_tokens, begin_index):
        # 将传入的begin_suppress_tokens转换为列表并赋值给实例变量self.begin_suppress_tokens
        self.begin_suppress_tokens = list(begin_suppress_tokens)
        # 将传入的begin_index赋值给实例变量self.begin_index
        self.begin_index = begin_index

    # 设置起始索引的方法，更新实例变量self.begin_index
    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    # 装饰器函数，添加了LOGITS_PROCESSOR_INPUTS_DOCSTRING的文档字符串，声明了输入和输出类型
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 如果输入的input_ids在第二维（列数）上的大小等于实例变量self.begin_index
        if input_ids.shape[1] == self.begin_index:
            # 则将scores张量中所有行的第self.begin_suppress_tokens列设为负无穷
            scores[:, self.begin_suppress_tokens] = -float("inf")

        # 返回修改后的scores张量
        return scores
class SuppressTokensLogitsProcessor(LogitsProcessor):
    r"""
    This processor can be used to suppress a list of tokens. The processor will set their log probs to `-inf` so
    that they are not generated. Originally created for
    [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

    Examples:

    ```python
    >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> # Whisper has a long list of suppressed tokens. For instance, in this case, the token 1 is suppressed by default.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    >>> print(outputs.scores[1][0, 1])  # 1 (and not 0) is the first freely generated token
    tensor(-inf)

    >>> # If we disable `suppress_tokens`, we can generate it.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, suppress_tokens=None)
    >>> print(outputs.scores[1][0, 1])
    tensor(5.7738)
    ```
    """

    def __init__(self, suppress_tokens):
        # 初始化函数，接受一个需要抑制的 token 列表
        self.suppress_tokens = list(suppress_tokens)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 在 scores 的特定列中，将抑制的 token 对应的概率设为负无穷
        scores[:, self.suppress_tokens] = -float("inf")
        # 返回处理后的 scores
        return scores
    # 验证所有的 scores 中除了索引为 50362 的位置外，其他位置是否都是负无穷大
    all(outputs.scores[0][0, i] == float("-inf") for i in range(processor.tokenizer.vocab_size) if i != 50362)
    True

    >>> # 打印索引为 50362 的 scores，确认其值为 0
    >>> print(outputs.scores[0][0, 50362])
    tensor(0.)

    >>> # 如果禁用了 `forced_decoder_ids`，我们停止看到上述效果
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, forced_decoder_ids=None)
    >>> # 验证所有的 scores 中除了索引为 50362 的位置外，其他位置是否都是负无穷大
    >>> print(
    ...     all(outputs.scores[0][0, i] == float("-inf") for i in range(processor.tokenizer.vocab_size) if i != 50362)
    ... )
    False
    >>> # 打印索引为 50362 的 scores，确认其新的值为 19.3140
    >>> print(outputs.scores[0][0, 50362])
    tensor(19.3140)
    ```

    """

    def __init__(self, force_token_map: List[List[int]], _has_warned: Optional[bool] = False):
        # 初始化 ForceTokensLogitsProcessor 类，接收一个强制令牌映射 force_token_map 和一个是否警告的标志 _has_warned
        self.force_token_map = dict(force_token_map)
        if not _has_warned:
            # 如果 _has_warned 为 False，发出警告，提醒在 v4.40 版本中移除该处理器
            warnings.warn(
                "This `ForceTokensLogitsProcessor` has been deprecated and will be removed in v4.40. Should you need to provide prompt ids for generation, specify `input_ids` to the generate method for decoder-only models, or `decoder_input_ids` for encoder-decoder models.",
                FutureWarning,
            )

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 根据传入的 input_ids 和 scores 处理 logits
        generation_idx = input_ids.shape[-1]  # 获取生成的索引
        current_token = self.force_token_map.get(generation_idx, None)  # 获取当前索引对应的强制令牌
        if current_token is not None:
            # 如果当前令牌不为 None，则将所有 scores 设置为负无穷大，并将当前令牌的 score 设置为 0
            scores[:, :] = -float("inf")
            scores[:, current_token] = 0
        return scores
class WhisperTimeStampLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that modifies the logits for the generation of timestamps in the transcription. When the input
    tokens are at a specific threshold, the processor sets the scores to negative infinity. The processor makes sure
    that timestamp tokens appear in pairs, by masking out the logits that would break this pairing pattern. This is
    done to maintain the consistency and structure of generated timestamps. It also ensures that when the predicted
    probability of sampling any of the timestamp token is greater than any individual non-timestamp token, those
    non-timestamp logits are set to negative infinity. This is done to ensure the generation of timestamps over other
    potential tokens.


    See [the paper](https://arxiv.org/abs/2212.04356) for more information.

    Args:
        generate_config (`GenerateConfig`):
            The generate config used to generate the output. The following parameters are required:
                eos_token_id (`int`, *optional*, defaults to 50257):
                    The id of the *end-of-sequence* token.
                no_timestamps_token_id (`int`, *optional*, defaults to 50363):
                    The id of the `"<|notimestamps|>"` token.
                max_initial_timestamp_index (`int`, *optional*, defaults to 1):
                    Used to set the maximum value of the initial timestamp. This is used to prevent the model from
                    predicting timestamps that are too far in the future.
        begin_index (`Optional`, *optional*): Token index of the first token that is generated by the model.
        _detect_timestamp_from_logprob (`bool`, *optional*): Whether timestamps can be predicted from logprobs over all timestamps.

    Examples:
    ``` python
    >>> import torch
    >>> from transformers import AutoProcessor, WhisperForConditionalGeneration, GenerationConfig
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[3]["audio"]["array"], return_tensors="pt")
    >>> input_features = inputs.input_features

    >>> #Displaying timestamps
    >>> generated_ids = model.generate(inputs=input_features, return_timestamps=True)
    >>> transcription = processor.batch_decode(generated_ids, decode_with_timestamps=True)[0]
    >>> print("Transcription:", transcription)
    Transcription: <|startoftranscript|><|0.00|> He has grave doubts whether Sir Frederick Layton's work is really Greek after all, and can<|6.44|><|6.44|> discover in it but little of rocky Ithaca.<|9.44|><|endoftext|>


    >>> #No timestamps & change EOS:
    ```
    """
    # 初始化函数，接受生成配置、可选的起始索引和检测时间戳的标志位
    def __init__(
        self, generate_config, begin_index: Optional[int] = None, _detect_timestamp_from_logprob: Optional[bool] = None
    ):  # support for the kwargs
        # 设置不带时间戳的特殊 token ID
        self.no_timestamps_token_id = generate_config.no_timestamps_token_id
        # 计算时间戳起始的 token ID
        self.timestamp_begin = generate_config.no_timestamps_token_id + 1
        # 设置终止生成的 token ID，可以从生成配置的 EOS 或 BOS token ID 中获取
        self.eos_token_id = generate_config.eos_token_id or generate_config.bos_token_id

        # 用于测试的变量，控制是否通过对数概率检测时间戳
        self._detect_timestamp_from_logprob = (
            _detect_timestamp_from_logprob
            if _detect_timestamp_from_logprob is not None
            else getattr(generate_config, "_detect_timestamp_from_logprob", True)
        )

        # 计算开始索引，考虑到强制解码器 ID 的数量
        num_forced_ids = (
            len(generate_config.forced_decoder_ids) if generate_config.forced_decoder_ids is not None else 0
        )
        self.begin_index = begin_index or (num_forced_ids + 1)

        # 最大初始时间戳索引，从生成配置中获取，默认为 None
        self.max_initial_timestamp_index = getattr(generate_config, "max_initial_timestamp_index", None)
        # TODO(Patrick): 确保官方模型将 max_initial_timestamp_index 设置为 50
        # self.max_initial_timestamp_index = 50

    # 设置起始索引的方法
    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    # 添加文档字符串，描述输入的 logits 处理器的输入
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # suppress <|notimestamps|> which is handled by without_timestamps
        # 将不带时间戳的标记 <|notimestamps|> 的分数设为负无穷，这些标记由 without_timestamps 处理
        scores[:, self.no_timestamps_token_id] = -float("inf")

        # timestamps have to appear in pairs, except directly before eos_token; mask logits accordingly
        # 时间戳必须成对出现，除非直接位于 eos_token 前面；相应地屏蔽对数几率
        for k in range(input_ids.shape[0]):
            sampled_tokens = input_ids[k, self.begin_index :]
            seq = list(sampled_tokens.tolist())

            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:
                    # has to be non-timestamp
                    # 必须是非时间戳
                    scores[k, self.timestamp_begin :] = -float("inf")
                else:
                    # cannot be normal text tokens
                    # 不能是正常文本标记
                    scores[k, : self.eos_token_id] = -float("inf")

            timestamps = sampled_tokens[sampled_tokens.ge(self.timestamp_begin)]
            if timestamps.numel() > 0:
                # `timestamps` shouldn't decrease; forbid timestamp tokens smaller than the last
                # `timestamps` 不应减少；禁止小于最后一个时间戳标记的时间戳标记
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    # Avoid to emit <|0.00|> again
                    # 避免再次生成 <|0.00|>
                    timestamp_last = timestamps[-1] + 1

                scores[k, self.timestamp_begin : timestamp_last] = -float("inf")

        # apply the `max_initial_timestamp` option
        # 应用 `max_initial_timestamp` 选项
        if input_ids.shape[1] == self.begin_index:
            scores[:, : self.timestamp_begin] = -float("inf")

            if self.max_initial_timestamp_index is not None:
                last_allowed = self.timestamp_begin + self.max_initial_timestamp_index
                scores[:, last_allowed + 1 :] = -float("inf")

        # if sum of probability over timestamps is above any other token, sample timestamp
        # 如果时间戳的概率和高于其他任何标记，则采样时间戳
        logprobs = torch.nn.functional.log_softmax(scores.float(), dim=-1)
        for k in range(input_ids.shape[0]):
            timestamp_logprob = logprobs[k, self.timestamp_begin :].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob and self._detect_timestamp_from_logprob:
                scores[k, : self.timestamp_begin] = -float("inf")

        return scores
class WhisperNoSpeechDetection(LogitsProcessor):
    r"""This processor can be used to detect silence when using Whisper. It should take as input unprocessed logits to follow the original implementation"""

    def __init__(self, no_speech_token: int, begin_index: int, scores_is_logprobs: bool = False):
        self.no_speech_token = no_speech_token
        # 原始实现中，<start-of-transcription> 标记的偏移量，等于第一个生成的标记的位置索引
        self.start_of_trans_offset = begin_index

        # `self.begin_index` 是一个实时变化的值
        self.begin_index = begin_index
        self._no_speech_prob = [0.0]
        self.is_scores_logprobs = scores_is_logprobs

        # 动态覆盖的属性
        self.model = None
        self.inputs = None

    def set_model(self, model):
        self.model = model

    def set_inputs(self, inputs):
        # 准备用于生成的输入，并将其与原始输入合并
        self.inputs = {**self.model.prepare_inputs_for_generation(**inputs), **inputs}
        self.inputs["input_features"] = self.inputs.pop("inputs")

    @property
    def no_speech_prob(self):
        return self._no_speech_prob

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] == self.begin_index:
            if self.start_of_trans_offset > 1:
                with torch.no_grad():
                    logits = self.model(**self.inputs).logits

                no_speech_index = self.begin_index - self.start_of_trans_offset
                no_speech_scores = logits[:, no_speech_index]
            else:
                no_speech_scores = scores

            if self.is_scores_logprobs:
                probs = no_speech_scores.exp()
            else:
                probs = no_speech_scores.float().softmax(dim=-1)

            self._no_speech_prob = probs[:, self.no_speech_token]

        return scores


class ClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for classifier free guidance (CFG). The scores are split over the batch dimension,
    where the first half correspond to the conditional logits (predicted from the input prompt) and the second half
    correspond to the unconditional logits (predicted from an empty or 'null' prompt). The processor computes a
    weighted average across the conditional and unconditional logits, parameterised by the `guidance_scale`.

    See [the paper](https://arxiv.org/abs/2306.05284) for more information.

    <Tip warning={true}>

    This logits processor is exclusively compatible with
    [MusicGen](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen)

    </Tip>
    def __init__(self, guidance_scale):
        # 初始化方法，接受一个参数 guidance_scale，用于设置分类器自由引导（CFG）的比例尺。CFG 通过设置 `guidance_scale > 1` 启用。
        # 较高的 guidance_scale 鼓励模型生成与输入提示更紧密相关的样本，但通常会导致质量较差的生成结果。
        if guidance_scale > 1:
            # 如果 guidance_scale 大于 1，则将其赋值给实例变量 self.guidance_scale
            self.guidance_scale = guidance_scale
        else:
            # 如果 guidance_scale 不大于 1，则抛出 ValueError 异常，提示需要 guidance_scale 大于 1 才能使用分类器自由引导处理器。
            raise ValueError(
                "Require guidance scale >1 to use the classifier free guidance processor, got guidance scale "
                f"{guidance_scale}."
            )

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 简单检查确保 logits 分数（条件和非条件）与输入的 input_ids（仅条件）具有兼容的批次大小。
        if scores.shape[0] != 2 * input_ids.shape[0]:
            # 如果 logits 的批次大小不是 input_ids 批次大小的两倍，则抛出 ValueError 异常。
            raise ValueError(
                f"Logits should have twice the batch size of the input ids, the first half of batches corresponding to "
                f"the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got "
                f"batch size {scores.shape[0]} for the logits and {input_ids.shape[0]} for the input ids."
            )
        # 计算非引导批次大小
        unguided_bsz = scores.shape[0] // 2
        # 将 scores 按照非引导批次大小分割成条件 logits 和非条件 logits
        cond_logits, uncond_logits = scores.split(unguided_bsz, dim=0)
        # 应用 guidance_scale 对 scores 进行加权处理，增强生成的条件性输出
        scores = uncond_logits + (cond_logits - uncond_logits) * self.guidance_scale
        # 返回处理后的 scores
        return scores
class AlternatingCodebooksLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing alternated generation between the two codebooks of Bark.

    <Tip warning={true}>
    
    This logits processor is exclusively compatible with
    [Bark](https://huggingface.co/docs/transformers/en/model_doc/bark)'s fine submodel. See the model documentation
    for examples.
    
    </Tip>

    Args:
        input_start_len (`int`):
            The length of the initial input sequence.
        semantic_vocab_size (`int`):
            Vocabulary size of the semantic part, i.e number of tokens associated to the semantic vocabulary.
        codebook_size (`int`):
            Number of tokens associated to the codebook.
    """

    def __init__(self, input_start_len: int, semantic_vocab_size: int, codebook_size: int):
        if not isinstance(input_start_len, int) or input_start_len < 0:
            raise ValueError(f"`input_starting_length` has to be a non-negative integer, but is {input_start_len}")

        # 初始化函数，验证并设置输入的起始长度、语义词汇表大小和码书大小
        self.input_start_len = input_start_len
        self.semantic_vocab_size = semantic_vocab_size
        self.codebook_size = codebook_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 获取当前输入序列的长度
        curr_len = input_ids.shape[-1]

        # 判断当前序列长度决定使用哪个码书：偶数长度使用第一个码书，奇数长度使用第二个码书
        is_first_codebook = ((curr_len - self.input_start_len) % 2) == 0

        if is_first_codebook:
            # 如果是第一个码书，将第一个码书的部分置为负无穷，表示不考虑这些部分的生成
            scores[:, : self.semantic_vocab_size] = -float("inf")
            scores[:, self.semantic_vocab_size + self.codebook_size :] = -float("inf")
        else:
            # 如果是第二个码书，将第二个码书的部分置为负无穷，表示不考虑这些部分的生成
            scores[:, : self.semantic_vocab_size + self.codebook_size] = -float("inf")

        # 返回处理后的得分张量
        return scores


class UnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""
    Logits processor for Classifier-Free Guidance (CFG). The processors computes a weighted average across scores
    from prompt conditional and prompt unconditional (or negative) logits, parameterized by the `guidance_scale`.
    The unconditional scores are computed internally by prompting `model` with the `unconditional_ids` branch.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.
    """
    Args:
        guidance_scale (`float`):
            CFG的引导比例，用于分类器自由引导。通过设置 `guidance_scale != 1` 来启用CFG。较高的引导比例鼓励模型生成与输入提示更紧密相关的样本，通常会以较差的质量为代价。小于1的值具有相反的效果，同时使得提供的负提示（如果有的话）作为正提示。
        model (`PreTrainedModel`):
            计算无条件分数的模型。假定与计算条件分数的模型相同。这两个模型必须使用相同的分词器。
        unconditional_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            无条件分支中输入序列标记在词汇表中的索引。如果未设置，则默认为提示的最后一个标记。
        unconditional_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            用于无条件_ids的注意力掩码。
        use_cache (`bool`, *optional*, defaults to `True`):
            是否在负提示前向传递过程中缓存键/值对。

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["Today, a dragon flew over Paris, France,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=1.5)
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Today, a dragon flew over Paris, France, killing at least 50 people and injuring more than 100'

    >>> # with a negative prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=2, negative_prompt_ids=neg_inputs["input_ids"])
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Today, a dragon flew over Paris, France, killing at least 130 people. French media reported that'

    >>> # with a positive prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=0, negative_prompt_ids=neg_inputs["input_ids"])
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    "Today, a dragon flew over Paris, France, and I'm very happy to be here. I"
    ```
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }


        # 初始化方法，设置对象的初始属性
        self.guidance_scale = guidance_scale  # 设置引导尺度
        self.model = model  # 设置模型
        # 设置无条件生成的上下文信息，包括输入id、注意力掩码、是否使用缓存、过去的键值对和第一次通行标志
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    def get_unconditional_logits(self, input_ids):
        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, -1:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["input_ids"], dtype=torch.long
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            attention_mask = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            if not self.unconditional_context["use_cache"]:
                input_ids = torch.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask


        # 根据上下文信息进行无条件生成的logits计算
        if self.unconditional_context["first_pass"]:
            # 如果是第一次通行，则根据输入的最后一个token设置初始输入id和注意力掩码
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, -1:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["input_ids"], dtype=torch.long
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            # 如果不是第一次通行，则根据是否使用缓存来更新输入id和注意力掩码
            attention_mask = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            if not self.unconditional_context["use_cache"]:
                input_ids = torch.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask

        # 调用模型生成输出，传入当前的输入id、注意力掩码、是否使用缓存以及过去的键值对
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits


    def __call__(self, input_ids, scores):
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores

        logits = self.get_unconditional_logits(input_ids)

        # 计算无条件logits的对数softmax
        unconditional_logits = torch.nn.functional.log_softmax(logits[:, -1], dim=-1)
        # 根据引导尺度调整得分的对数softmax并加上无条件生成的对数softmax
        out = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        return out
class BarkEosPrioritizerLogitsProcessor(LogitsProcessor):
    r"""This processor ensures that the EOS token is selected if its probability is greater than the `min_eos_p`.

    <Tip warning={true}>

    This logits processor is exclusively compatible with
    [Bark](https://huggingface.co/docs/transformers/en/model_doc/bark). See the model documentation for examples.

    </Tip>

    Args:
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        min_eos_p (`float`, *optional*):
            Minimum end of speech threshold.
    """

    def __init__(self, eos_token_id: Union[int, List[int]], min_eos_p: float):
        # Convert eos_token_id to a list if it's provided as an integer
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id
        # Validate min_eos_p is a positive float if provided
        if min_eos_p is not None and min_eos_p <= 0:
            raise ValueError(f"`min_eos_p` has to be a positive float, but is {min_eos_p}")
        self.min_eos_p = min_eos_p

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Check if min_eos_p is set
        if self.min_eos_p:
            # Compute softmax probabilities across the last dimension of scores tensor
            probs = torch.nn.functional.softmax(scores.float(), dim=-1)
            # Initialize a tensor with -inf values except for the eos_token_id
            early_stop_scores = torch.ones_like(scores) * -float("inf")
            early_stop_scores[:, self.eos_token_id] = scores[:, self.eos_token_id]
            
            # Determine if any EOS token's probability exceeds min_eos_p
            do_early_stop = probs[:, self.eos_token_id] > self.min_eos_p
            do_early_stop = torch.any(do_early_stop, dim=1, keepdim=True)
            # Conditionally replace scores with early_stop_scores where needed
            scores = torch.where(do_early_stop, early_stop_scores, scores)

        return scores
```