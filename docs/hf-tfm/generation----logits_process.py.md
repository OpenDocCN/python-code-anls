# `.\generation\logits_process.py`

```
# 设置文件编码为utf-8
# 版权声明
# 基于Apache License, Version 2.0授权使用该文件
# 可以在获得许可证的情况下使用该文件
# 可以在以下网址获取许可证信息：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"现状"分发软件，不附带任何保证或条件，无论是明示的还是暗示的。
# 请查看特定语言规定的许可证，以获取权限和限制信息。

# 导入模块
import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger

# 获取日志记录器
logger = get_logger(__name__)

# LogitsProcessor的输入文档字符串
LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

# LogitsProcessor类
class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""
    # 装饰器，添加函数说明文档
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    # 定义__call__方法
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 抛出未实现错误
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


# LogitsWarper类
class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""
    # 装饰器，添加函数说明文档
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    # 定义__call__方法
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 抛出未实现错误
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


# LogitsProcessorList类
class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process a
    `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """
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

        # 对于每个处理器，执行以下操作
        for processor in self:
            # 获取processor.__call__函数的参数列表
            function_args = inspect.signature(processor.__call__).parameters
            # 如果函数参数数量大于2
            if len(function_args) > 2:
                # 判断kwargs是否包含所有参数
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    # 抛出异常
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                # 调用processor的__call__方法，传递input_ids, scores和kwargs
                scores = processor(input_ids, scores, **kwargs)
            else:
                # 调用processor的__call__方法，传递input_ids和scores
                scores = processor(input_ids, scores)

        return scores
# 定义一个继承自 LogitsProcessor 的类 MinLengthLogitsProcessor，用于确保生成的序列长度不低于指定值，
# 通过将 EOS（终止符）的概率设为0来实现。需要注意，对于大多数只有解码器的语言模型来说，长度包括提示部分。

class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0. Note that, for decoder-only models
    like most LLMs, the length includes the prompt.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
            最小长度，低于此长度时 `eos_token_id` 的得分将被设为 `-float("Inf")`。
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            *终止序列* 符号的 id。可以选择使用列表设置多个 *终止序列* 符号。

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

    def __init__(self, min_length: int, eos_token_id: Union[int, List[int]]):
        # 检查 min_length 是否为非负整数
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a non-negative integer, but is {min_length}")

        # 将 eos_token_id 转换为列表形式，如果输入是单个整数也要能处理
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        # 检查 eos_token_id 是否为正整数的列表
        if not all(isinstance(i, int) for i in eos_token_id) or any(i < 0 for i in eos_token_id):
            logger.warning(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

        # 设置对象的属性 min_length 和 eos_token_id
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 获取当前输入序列的长度
        cur_len = input_ids.shape[-1]
        # 如果当前长度小于最小长度，将对应的 eos_token_id 的概率设置为负无穷
        if cur_len < self.min_length:
            for i in self.eos_token_id:
                scores[:, i] = -float("inf")
        # 返回修改后的 scores
        return scores


# 定义一个继承自 LogitsProcessor 的类 MinNewTokensLengthLogitsProcessor，
# 用于通过将 EOS（终止序列）的概率设为0来确保新生成的令牌长度不低于指定值。
# 与 MinLengthLogitsProcessor 不同的是，此处理器忽略了提示。
    Args:
        prompt_length_to_skip (`int`):
            # 设定需要跳过的输入标记长度。在使用 `generate` 方法时，此参数无效，因为它将自动分配输入长度。
        min_new_tokens (`int`):
            # 设置最小的*新*标记长度，低于此长度时，`eos_token_id` 的得分将被设为`-float("Inf")`。
        eos_token_id (`Union[int, List[int]]`):
            # *结束序列*标记的 id。可选择使用列表设置多个*结束序列*标记。

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer(["A number:"], return_tensors="pt")
    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one

    >>> # setting `min_new_tokens` will force the model to generate beyond its natural ending point, which is not
    >>> # necessarily incorrect
    >>> gen_out = model.generate(**inputs, min_new_tokens=2)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one thousand
    ```
    """

    def __init__(self, prompt_length_to_skip: int, min_new_tokens: int, eos_token_id: Union[int, List[int]]):
        for arg_name, arg_value in [
            ("prompt_length_to_skip", prompt_length_to_skip),
            ("min_new_tokens", min_new_tokens),
        ]:
            # 检查输入参数是否是正整数，如果不是则抛出数值错误
            if not isinstance(arg_value, int) or arg_value < 0:
                raise ValueError(f"`{arg_name}` has to be a positive integer, but is {arg_value}")

        # 如果 `eos_token_id` 是整数，则将其转换为列表
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        # 如果 `eos_token_id` 不是都是整数或有负数，则记录警告
        if not all(isinstance(i, int) for i in eos_token_id) or any(i < 0 for i in eos_token_id):
            logger.warning(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

        # 记录输入的参数值
        self.prompt_length_to_skip = prompt_length_to_skip
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 计算新标记的长度
        new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip
        # 如果新标记长度小于最小新标记长度，则将所有结束序列标记的得分设为负无穷
        if new_tokens_length < self.min_new_tokens:
            for i in self.eos_token_id:
                scores[:, i] = -float("inf")

        # 返回更新后的得分
        return scores
# 定义一个继承自LogitsWarper的TemperatureLogitsWarper类，用于在输出概率分布上进行温度调节，从而控制预测的随机性，通常和TopPLogitsWarper以及TopKLogitsWarper一起使用

# 提示
# 确保在generate参数中包含do_sample=True，否则temperature值将不会产生任何效果

# 参数:
# temperature (float):
# 用于调节logits分布的严格正值浮点数。小于1的值会减少随机性（反之亦然），0相当于将所有概率集中在最可能的token上

# 示例:
# 创建一个AutoTokenizer的实例tokenizer，并加载"gpt2"预训练模型
# 创建一个AutoModelForCausalLM的实例model，并加载"gpt2"预训练模型
# 将model的pad_token_id设置为model的eos_token_id
# 利用tokenizer将句子["Hugging Face Company is"]编码为tensor
# 使用temperature=1.0时，由于随机采样，每次都会得到随机输出
# 设置generate_kwargs参数为{"max_new_tokens": 10, "do_sample": True, "temperature": 1.0, "num_return_sequences": 2}
# 利用model.generate生成文本，然后使用tokenizer批量解码并打印结果
# 将generate_kwargs中的temperature设为0.0001，得到近似贪婪解码策略的输出文本

class TemperatureLogitsWarper(LogitsWarper):
    
    # 初始化函数，接受一个temperature参数
    def __init__(self, temperature: float):
        # 检查temperature是否为正值浮点数，若不是则抛出ValueError
        if not isinstance(temperature, float) or not (temperature > 0):
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."
            raise ValueError(except_msg)
        # 将temperature值赋给self.temperature

    # 调用函数，接受input_ids和scores两个参数，返回经过温度调节后的scores值
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 将scores除以temperature，返回结果
        scores = scores / self.temperature
        return scores
# 定义一个处理 logits 的类，用于防止重复生成先前的 token，通过添加惩罚来实现。该惩罚最多每个 token 应用一次。需要注意，在仅有解码器的模型（如大多数 LLMs）中，考虑的 token 包括提示。
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
    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
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

    # 初始化方法，接受一个浮点数参数作为重复惩罚的值
    def __init__(self, penalty: float):
        # 如果惩罚不是浮点数或不大于 0，则引发 ValueError 异常
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        # 设置类属性 penalty 为传入的参数值
        self.penalty = penalty

    # 重载了 __call__ 方法，用于处理输入的 input_ids 和 scores，并返回修改后的 scores
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 从 scores 中根据 input_ids 中的索引获取对应的分数
        score = torch.gather(scores, 1, input_ids)

        # 如果分数小于 0，则将重复惩罚乘以分数以降低 token 的概率
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        # 将修改后的分数 scores 根据 input_ids 中的索引重新赋值
        scores.scatter_(1, input_ids, score)
        # 返回修改后的 scores
        return scores


# 定义一个处理器类，与 RepetitionPenaltyLogitsProcessor 类似，但是对提示中存在的 token 应用 *逆* 惩罚。换句话说，大于 1.0 的惩罚增加了选择存在于提示中的 token 的几率。
class EncoderRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that works similarly to [`RepetitionPenaltyLogitsProcessor`], but with an *inverse* penalty
    that is applied to the tokens present in the prompt. In other words, a penalty above 1.0 increases the odds of
    selecting tokens that were present in the prompt.
    """
    It was designed to avoid hallucination in input-grounded tasks, like summarization. Although originally intended
    for encoder-decoder models, it can also be used with decoder-only models like LLMs.

    Args:
        penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 rewards prompt tokens. Between 0.0
            and 1.0 penalizes prompt tokens.
        encoder_input_ids (`torch.LongTensor`):
            The encoder_input_ids that should be repeated within the decoder ids.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer(["Alice and Bob. The third member's name was"], return_tensors="pt")
    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    Alice and Bob. The third member's name was not mentioned.

    >>> # With the `encoder_repetition_penalty` argument we can trigger this logits processor in `generate`, which can
    >>> # promote the use of prompt tokens ("Bob" in this example)
    >>> gen_out = model.generate(**inputs, encoder_repetition_penalty=1.2)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    Alice and Bob. The third member's name was Bob. The third member's name was Bob.
    ```
    """

    # 初始化函数，接收惩罚参数和编码器输入id
    def __init__(self, penalty: float, encoder_input_ids: torch.LongTensor):
        # 检查惩罚参数是否为正的浮点数
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        # 计算惩罚值
        self.penalty = 1 / penalty
        self.encoder_input_ids = encoder_input_ids

    # 实现__call__方法，用于处理logits
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 从scores中聚合编码器输入id的分数
        score = torch.gather(scores, 1, self.encoder_input_ids)

        # 如果分数小于0，则乘以惩罚值来增加令牌概率，否则除以惩罚值
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        # 将修改后的分数重新分配到scores中对应位置
        scores.scatter_(1, self.encoder_input_ids, score)
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
    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

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

    # 初始化方法，设置top_p参数、filter_value参数和min_tokens_to_keep参数
    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        # 将top_p转换为浮点数
        top_p = float(top_p)
        # 检查top_p是否在0到1之间
        if top_p < 0 or top_p > 1.0:
            # 如果不在范围内，抛出异常
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        # 检查min_tokens_to_keep是否为正整数
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            # 如果不是正整数，抛出异常
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        # 设置实例变量top_p为传入的top_p值
        self.top_p = top_p
        # 设置实例变量filter_value为传入的filter_value值
        self.filter_value = filter_value
        # 设置实例变量min_tokens_to_keep为传入的min_tokens_to_keep值
        self.min_tokens_to_keep = min_tokens_to_keep

    # 添加文档字符串
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
        # 定义 __call__ 方法，接受输入的 token id 和对应的分数，返回处理后的分数
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            # 按分数升序对分数和对应的索引进行排序
            sorted_logits, sorted_indices = torch.sort(scores, descending=False)
            # 对排序后的分数进行 softmax 处理，并计算累积概率
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            # 删除累积概率超过阈值的 token（概率为 0 的 token 保留）
            sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
            # 至少保留 min_tokens_to_keep 个 token
            sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

            # 将排序后的张量散射到原始索引
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            # 使用 filter_value 替换需要删除的 token 的分数
            scores = scores.masked_fill(indices_to_remove, self.filter_value)
            # 返回处理后的分数
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
    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

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
        # 检查`top_k`是否为正整数，若不是则引发值错误
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        # 将top_k设置为min(top_k, min_tokens_to_keep)
        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 为了安全，取最小值top_k和scores最后一个维度的大小
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        # 将概率小于top-k的概率的tokens移除
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TypicalLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs typical decoding. Inspired on how humans use language, it prioritizes tokens whose
    log probability is close to the entropy of the token probability distribution. This means that the most likely
    tokens may be discarded in the process.

    See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information.
    # 定义一个类，用于处理模型的输出结果
    class LogitsProcessor:
        def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
            # 初始化类的属性，设置了默认值
            mass = float(mass)
            # 如果 typical_p 的值不在 0 到 1 之间，抛出数值错误
            if not (mass > 0 and mass < 1):
                raise ValueError(f"`typical_p` has to be a float > 0 and < 1, but is {mass}")
            # 如果 min_tokens_to_keep 不是整数或小于 1，抛出数值错误
            if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
                raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

            # 设置类的属性
            self.filter_value = filter_value
            self.mass = mass
            self.min_tokens_to_keep = min_tokens_to_keep

        @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 计算输入的分数张量的softmax对数后的结果
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        # 计算softmax
        p = torch.exp(normalized)
        # 计算熵
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # 移位并排序
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # 移除累积质量超过阈值的标记
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind.clamp_(max=sorted_scores.shape[-1] - 1)
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        # 基于条件将分数张量中的值替换为过滤值
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
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
    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

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

    def __init__(self, epsilon: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        epsilon = float(epsilon)  # 转换epsilon为浮点数
        if epsilon <= 0 or epsilon >= 1:  # 检查epsilon的取值范围是否合法
            raise ValueError(f"`epsilon_cutoff` has to be a float > 0 and < 1, but is {epsilon}") # 如果不合法，抛出异常

        min_tokens_to_keep = int(min_tokens_to_keep)  # 转换min_tokens_to_keep为整数
        if min_tokens_to_keep < 1:  # 检查min_tokens_to_keep的取值范围是否合法
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )  # 如果不合法���抛出异常

        self.epsilon = epsilon  # 设定epsilon
        self.filter_value = filter_value  # 设定filter_value
        self.min_tokens_to_keep = min_tokens_to_keep  # 设定min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)  # 添加文档字符串
    # 通过调用实例对象，将输入的输入标识和分数张量转换成张量
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 将分数张量按照最后一个维度进行 softmax 归一化，得到概率值
        probabilities = scores.softmax(dim=-1)
        # 找到概率值低于阈值 epsilon 的索引
        indices_to_remove = probabilities < self.epsilon

        # 保留概率值最高的前 min_tokens_to_keep 个单词的索引
        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # 安全检查
        # 找到概率值低于阈值 epsilon 且排在前 min_tokens_to_keep 的索引
        indices_to_remove = indices_to_remove & (scores < torch.topk(scores, top_k)[0][..., -1, None])

        # 将低概率值的索引对应的分数替换为过滤值 filter_value
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
    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

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
    # 定义类的初始化方法，设置epsilon、filter_value和min_tokens_to_keep的默认值并进行类型检查
    def __init__(self, epsilon: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        # 将epsilon转换为float类型
        epsilon = float(epsilon)
        # 检查epsilon的取值范围是否合法
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"`eta_cutoff` has to be a float > 0 and < 1, but is {epsilon}")

        # 将min_tokens_to_keep转换为int类型
        min_tokens_to_keep = int(min_tokens_to_keep)
        # 检查min_tokens_to_keep是否是严格正整数
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

        # 初始化对象的epsilon、filter_value和min_tokens_to_keep属性
        self.epsilon = torch.tensor(epsilon)
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    # 添加文档说明的装饰器，处理LOGITS_PROCESSOR_INPUTS_DOCSTRING
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    # 定义类的调用方法，接收input_ids和scores两个Tensor，并返回处理后的scores
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 计算自适应截断值
        probabilities = scores.softmax(dim=-1)
        entropy = torch.distributions.Categorical(logits=scores).entropy()
        eta = torch.min(self.epsilon, torch.sqrt(self.epsilon) * torch.exp(-entropy))[..., None]
        indices_to_remove = probabilities < eta

        # 保留概率最高的top_k个单词
        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # 安全检查
        indices_to_remove = indices_to_remove & (scores < torch.topk(scores, top_k)[0][..., -1, None])

        # 根据indices_to_remove对scores进行过滤处理
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
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
    # Initialize an empty list of dictionaries, one for each hypothesis (index) in the range of num_hypos
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        # Loop through each n-gram of size ngram_size in the list of tokens (gen_tokens)
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


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
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(
    ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
) -> List[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
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
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:  # 检查ngram_size的类型和取值
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size  # 设置ngram_size属性

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]  # 计算批量假设的数量
        cur_len = input_ids.shape[-1]  # 计算输入ids的当前长度
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)  # 计算被禁止的n-元组的标记
        for i, banned_tokens in enumerate(banned_batch_tokens):  # 遍历所有被禁止的标记
            scores[i, banned_tokens] = -float("inf")  # 将禁止的标记对应的分数设为负无穷

        return scores  # 返回处理后的分数
    class NoRepeatNGramsLogitsProcessor(LogitsProcessor):
        """
        This logits processor prevents the repetition of n-grams present in the prompt.
        It was designed to promote chattiness in a language model, by preventing the generation of n-grams present in
        previous conversation rounds.
    
        Args:
            encoder_ngram_size (`int`):
                All ngrams of size `ngram_size` can only occur within the encoder input ids.
            encoder_input_ids (`int`):
                The encoder_input_ids that should not be repeated within the decoder ids.
    
        Examples:
    
        ```py
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    
        >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
        >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    
        >>> inputs = tokenizer("Alice: I love cats. What do you love?\nBob:", return_tensors="pt")
    
        >>> # With greedy decoding, we see Bob repeating Alice's opinion. If Bob was a chatbot, it would be a poor one.
        >>> outputs = model.generate(**inputs)
        >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
        Alice: I love cats. What do you love?
        Bob: I love cats. What do you
    
        >>> # With this logits processor, we can prevent Bob from repeating Alice's opinion.
        >>> outputs = model.generate(**inputs, encoder_no_repeat_ngram_size=2)
        >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
        Alice: I love cats. What do you love?
        Bob: My cats are very cute.
        ```
        """
    
        def __init__(self, encoder_ngram_size: int, encoder_input_ids: torch.LongTensor):
            # Check if encoder_ngram_size is valid
            if not isinstance(encoder_ngram_size, int) or encoder_ngram_size <= 0:
                raise ValueError(
                    f"`encoder_ngram_size` has to be a strictly positive integer, but is {encoder_ngram_size}"
                )
            # Set the ngram_size attribute
            self.ngram_size = encoder_ngram_size
            # If encoder_input_ids is 1D, reshape it to be 2D
            if len(encoder_input_ids.shape) == 1:
                encoder_input_ids = encoder_input_ids.unsqueeze(0)
            # Set the batch_size attribute
            self.batch_size = encoder_input_ids.shape[0]
            # Generate n-grams from the encoder_input_ids
            self.generated_ngrams = _get_ngrams(encoder_ngram_size, encoder_input_ids, self.batch_size)
    
        @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            # Get the number of hypotheses and number of beams
            num_hypos = scores.shape[0]
            num_beams = num_hypos // self.batch_size
            # Get the current length of the input_ids
            cur_len = input_ids.shape[-1]
            # Get the banned tokens for each hypothesis
            banned_batch_tokens = [
                _get_generated_ngrams(
                    self.generated_ngrams[hypo_idx // num_beams], input_ids[hypo_idx], self.ngram_size, cur_len
                )
                for hypo_idx in range(num_hypos)
            ]
    
            # Set scores of banned tokens to -inf
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

    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> inputs = tokenizer(["The full name of Donald is Donald"], return_tensors="pt")

    >>> summary_ids = model.generate(inputs["input_ids"], max_new_tokens=4)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald J. Trump Jr

    >>> # Now let's control generation through a bias. Please note that the tokenizer is initialized differently!
    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)


    >>> def get_tokens_as_tuple(word):
    ...     return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])


    >>> # If we add a negative bias without beam search, it may become "stuck" in a prefix without good continuations
    >>> sequence_bias = {get_tokens_as_tuple("Trump"): -10.0}
    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald J. Donald,

    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    # 创建一个LogitsProcessor类
    """
    注释：
    创建一个LogitsProcessor类，用于对生成的logits进行处理。此类的作用是为生成的logits添加一个正面的偏置，以促使模型生成特定的词语或延续。
    """
    
        def __init__(self, sequence_bias: Dict[Tuple[int], float]):
            # 定义构造方法
            """
            注释：
            初始化方法，接受一个sequence_bias的字典参数，该字典的键是由整数组成的元组，值是偏置的浮点数。
            还初始化了两个变量，length_1_bias用于存储长度为1的偏置，prepared_bias_variables用于表示是否已经准备好了偏置变量。
            """
            self.sequence_bias = sequence_bias
            self._validate_arguments()
    
            # Bias variables that will be populated on the first call (for retrocompatibility purposes, the vocabulary size
            # is infered in the first usage, which inhibits initializing here)
            self.length_1_bias = None
            self.prepared_bias_variables = False
    
        @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            # 实现call方法
            """
            注释：
            实现__call__方法，该方法接受input_ids和scores两个参数，并返回处理后的scores。
            __call__方法的作用是对传入的input_ids进行处理，并根据处理结果调整scores，最后返回调整后的scores。
            """
            # 1 - Prepares the bias tensors. This is only needed the first time the logit processor is called.
            if not self.prepared_bias_variables:
                self._prepare_bias_variables(scores)
    
            # 2 - prepares an empty bias to add
            bias = torch.zeros_like(scores)
    
            # 3 - include the bias from length = 1
            bias += self.length_1_bias
    
            # 4 - include the bias from length > 1, after determining which biased sequences may be completed.
            for sequence_ids, sequence_bias in self.sequence_bias.items():
                # 检查序列的长度是否为1，如果是则直接应用偏置
                if len(sequence_ids) == 1:  
                    continue
                # 检查序列的长度是否大于上下文长度，如果是则忽略
                if len(sequence_ids) > input_ids.shape[1]:  
                    continue
                # 计算序列的前缀长度和最后一个标记
                prefix_length = len(sequence_ids) - 1
                last_token = sequence_ids[-1]
                # 判断输入的标记和序列的前缀是否匹配，返回匹配的行数
                matching_rows = torch.eq(
                    input_ids[:, -prefix_length:],
                    torch.tensor(sequence_ids[:-1], dtype=input_ids.dtype, device=input_ids.device),
                ).prod(dim=1)
                # 根据匹配行数的结果，将偏置值加到相应的位置上
                bias[:, last_token] += torch.where(
                    matching_rows.bool(),
                    torch.tensor(sequence_bias, device=input_ids.device),
                    torch.tensor(0.0, device=input_ids.device),
                )
    
            # 5 - apply the bias to the scores
            scores = scores + bias
            # 返回处理后的scores
            return scores
    # 准备偏置变量，根据给定的分数张量
    def _prepare_bias_variables(self, scores: torch.FloatTensor):
        # 获取词汇表大小
        vocabulary_size = scores.shape[-1]

        # 检查偏置的标记是否超出边界
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

        # 预先计算要应用的偏置张量。长度为1的序列被单独保留，因为可以使用更简单的逻辑来应用它们。
        self.length_1_bias = torch.zeros((vocabulary_size,), dtype=torch.float).to(scores.device)
        for sequence_ids, bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:
                self.length_1_bias[sequence_ids[-1]] = bias

        self.prepared_bias_variables = True

    # 验证参数是否有效
    def _validate_arguments(self):
        sequence_bias = self.sequence_bias
        if not isinstance(sequence_bias, dict) or len(sequence_bias) == 0:
            raise ValueError(f"`sequence_bias` has to be a non-empty dictionary, but is {sequence_bias}.")
        if any(not isinstance(sequence_ids, tuple) for sequence_ids in sequence_bias.keys()):
            raise ValueError(f"`sequence_bias` has to be a dict with tuples as keys, but is {sequence_bias}.")
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in sequence_ids)
            or len(sequence_ids) == 0
            for sequence_ids in sequence_bias.keys()
        ):
            raise ValueError(
                f"Each key in `sequence_bias` has to be a non-empty tuple of positive integers, but is "
                f"{sequence_bias}."
            )
        if any(not isinstance(bias, float) for bias in sequence_bias.values()):
            raise ValueError(f"`sequence_bias` has to be a dict with floats as values, but is {sequence_bias}.")
# 定义 NoBadWordsLogitsProcessor 类，继承自 SequenceBiasLogitsProcessor 类
class NoBadWordsLogitsProcessor(SequenceBiasLogitsProcessor):
    """
    [`LogitsProcessor`] that enforces that specified sequences will never be selected.
    
    【`LogitsProcessor`] 强制某些指定的序列不会被选中。
    
    <Tip>
    
    In order to get the token ids of the words that should not appear in the generated text, make sure to set
    `add_prefix_space=True` when initializing the tokenizer, and use `tokenizer(bad_words,
    add_special_tokens=False).input_ids`. The `add_prefix_space` argument is only supported for some slow tokenizers,
    as fast tokenizers' prefixing behaviours come from `pre tokenizers`. Read more
    [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).
    
    为了获取在生成的文本中不应出现的单词的标记 ID，确保在初始化分词器时将`add_prefix_space=True`设定为真，并使用`tokenizer(bad_words,
    add_special_tokens=False).input_ids`。 `add_prefix_space`参数仅适用于某些慢分词器，因为快速分词器的前缀行为来自`pre tokenizers`。 了解更多信息
    [这里](https://huggingface.co/docs/tokenizers/api/pre-tokenizers)。
    
    </Tip>
    
    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated.
            
            不允许生成的标记 ID 的列表的列表。
            
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            
            “序列结束”标记的 ID。 可选地，使用列表来设置多个“序列结束”标记。
            
    Examples:
    
    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> inputs = tokenizer(["In a word, the cake is a"], return_tensors="pt")

    >>> output_ids = model.generate(inputs["input_ids"], max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    >>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
    In a word, the cake is a bit of a mess.

    >>> # Now let's take the bad words out. Please note that the tokenizer is initialized differently
    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)


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
    ```
    """
    # 初始化方法，接受两个参数：bad_words_ids和eos_token_id，分别代表不良词汇列表和结束标记的id
    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: Union[int, List[int]]):
        # 将参数bad_words_ids赋值给实例变量bad_word_ids，并验证参数的合法性
        self.bad_word_ids = bad_words_ids
        self._validate_arguments()

        # 从bad_words_ids中过滤掉EOS标记
        if eos_token_id is None:
            eos_token_id = []
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        # 使用lambda表达式过滤掉包含EOS标记的不良词汇序列
        bad_words_ids = list(
            filter(lambda bad_token_seq: all(bad_token_seq != [i] for i in eos_token_id), bad_words_ids)
        )

        # 禁止某个序列等价于将其偏置设置为负无穷
        sequence_bias = {tuple(sequence): float("-inf") for sequence in bad_words_ids}
        # 调用父类的初始化方法，传入序列偏置
        super().__init__(sequence_bias=sequence_bias)

    # 验证参数的合法性
    def _validate_arguments(self):
        # 获取不良词汇列表
        bad_words_ids = self.bad_word_ids
        # 检查不良词汇列表是否是非空列表
        if not isinstance(bad_words_ids, list) or len(bad_words_ids) == 0:
            raise ValueError(f"`bad_words_ids` has to be a non-empty list, but is {bad_words_ids}.")
        # 检查不良词汇列表中的每个元素是否都是列表
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids):
            raise ValueError(f"`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.")
        # 检查不良词汇列表中的每个元素是否都是正整数
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in bad_word_ids)
            for bad_word_ids in bad_words_ids
        ):
            raise ValueError(
                f"Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}."
            )
# 自定义 LogitsProcessor 类，用于强制执行约束生成，并且适用于前缀条件下的约束生成
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

    # 初始化函数，接收两个参数：prefix_allowed_tokens_fn 和 num_beams
    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn  # 存储传入的 prefix_allowed_tokens_fn 参数
        self._num_beams = num_beams  # 存储传入的 num_beams 参数

    # 继承父类的函数
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 创建一个与scores形状相同的张量，填充为负无穷大
        mask = torch.full_like(scores, -math.inf)
        # 将输入的input_ids重塑为形状为(-1, self._num_beams, input_ids.shape[-1])的张量，并遍历
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            # 遍历每一个beam
            for beam_id, sent in enumerate(beam_sent):
                # 获取前缀允许的标记
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                # 如果前缀允许的标记列表为空，则抛出异常
                if len(prefix_allowed_tokens) == 0:
                    raise ValueError(
                        f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                        f"This means that the constraint is unsatisfiable. Please check your implementation"
                        f"of `prefix_allowed_tokens_fn` "
                    )
                # 将mask中batch_id * self._num_beams + beam_id索引对应的位置上的标记设为0
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        # 返回scores和mask相加的结果
        return scores + mask
# 定义一个继承自 LogitsProcessor 的类 HammingDiversityLogitsProcessor
class HammingDiversityLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces diverse beam search.

    # 说明这个 LogitsProcessor 只对 [`PreTrainedModel.group_beam_search`] 有效。参考 [Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models](https://arxiv.org/pdf/1610.02424.pdf) 了解更多细节。

    # 传统的 beam search 经常生成非常相似的序列。`HammingDiversityLogitsProcessor` 通过惩罚在同一时间步生成与其他组中的任何 beam 已选择的标记相同的排序来解决这个问题。

    # 参数：
    #     diversity_penalty (`float`): 如果一个 beam 生成一个在特定时间与其他组的任何 beam 相同的标记，Beam 分数将减去这个值。较高的 `diversity_penalty` 会在 beam 之间施加更大的差异性。调整此值可以帮助在差异性和自然可能性之间取得平衡。
    #     num_beams (`int`): beam search 的数量。1 表示没有 beam search。
    #     num_beam_groups (`int`): 为了确保不同组的 beam 之间的差异性，将 `num_beams` 分成的组数。参考 [this paper](https://arxiv.org/pdf/1610.02424.pdf) 了解更多细节。

    # 示例：
    #     下面介绍了如何使用这个 LogitsProcessor。
    #     首先从 transformers 模块中导入所需的类和函数
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    >>> import torch

    #     初始化模型和分词器
    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    #     一个关于太阳系的长文本
    >>> text = (
    ...     "The Solar System is a gravitationally bound system comprising the Sun and the objects that orbit it, "
    ...     "either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight "
    ...     "planets, with the remainder being smaller objects, such as the five dwarf planets and small Solar System "
    ...     "bodies. The Solar System formed 4.6 billion years ago from the gravitational collapse of a giant "
    ...     "interstellar molecular cloud."
    ... )
    >>> inputs = tokenizer("summarize: " + text, return_tensors="pt")

    #     生成多样化的摘要
    >>> outputs_diverse = model.generate(
    ...     **inputs,
    ...     num_beam_groups=2,
    ...     diversity_penalty=10.0,
    ...     max_length=100,
    ...     num_beams=4,
    ...     num_return_sequences=2,
    ... )
    >>> summaries_diverse = tokenizer.batch_decode(outputs_diverse, skip_special_tokens=True)

    #     ���成非多样化的摘要
    >>> outputs_non_diverse = model.generate(
    ...     **inputs,
    ...     max_length=100,
    ...     num_beams=4,
    ...     num_return_sequences=2,
    ... )
    >>> summary_non_diverse = tokenizer.batch_decode(outputs_non_diverse, skip_special_tokens=True)
    >>> # 根据 `diversity_penalty`，生成的 beam 更加多样化
    >>> # 打印非多样化摘要
    >>> print(summary_non_diverse)
    ['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
    'the Solar System formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.']

    >>> # 打印多样化摘要
    >>> print(summaries_diverse)
    ['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
    'the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets. the rest of the objects are smaller objects, such as the five dwarf planets and small solar system bodies.']
    ```

    def __init__(self, diversity_penalty: float, num_beams: int, num_beam_groups: int):
        # 检查并设置多样性惩罚项
        if not isinstance(diversity_penalty, float) or (not diversity_penalty > 0.0):
            raise ValueError("`diversity_penalty` should be a float strictly larger than 0.")
        self._diversity_penalty = diversity_penalty
        # 检查并设置束搜索数量
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        # 检查并设置束搜索组数量
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        if num_beam_groups > num_beams:
            raise ValueError("`beam_groups` has to be smaller or equal to `num_beams`.")
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        current_tokens: torch.LongTensor,
        beam_group_idx: int,
    ) -> torch.FloatTensor:
        # 函数签名，指定输入和输出的数据类型
        r"""
        Args:
            # 输入参数说明
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                # 输入序列标记的索引，在词汇表中的位置
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                # 语言建模头的预测分数。当不使用束搜索时，这些可以是每个词汇表的对数或在使用束搜索时每个词汇表标记的对数 softmax
            current_tokens (`torch.LongTensor` of shape `(batch_size)`):
                # 输入序列标记的索引，在词汇表中的位置，对应于当前生成步骤中其他束组选择的标记
            beam_group_idx (`int`):
                # 当前处理的束组的索引

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                # 处理后的预测分数
        """
        # hamming多样性：惩罚当前组中与前序组在同一时间步骤中使用的相同标记
        # 获取当前组大小和词汇表大小
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]

        # 如果组起始索引为0，则直接返回分数
        if group_start_idx == 0:
            return scores

        # 遍历批次，计算多样性惩罚
        for batch_idx in range(batch_size):
            # 获取上一组最后一个时间步骤的预测标记
            previous_group_tokens = current_tokens[
                batch_idx * self._num_beams : batch_idx * self._num_beams + group_start_idx
            ]
            # 计算标记频率
            token_frequency = torch.bincount(previous_group_tokens, minlength=vocab_size).to(scores.device)
            # 分数减去多样性惩罚乘以标记频率
            scores[batch_idx * group_size : (batch_idx + 1) * group_size] -= self._diversity_penalty * token_frequency

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
        self.bos_token_id = bos_token_id  # 初始化强制的第一个token id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING) # 添加文档字符串
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:  # 用于将指定的token作为生成的第一个token
        cur_len = input_ids.shape[-1]  # 当前生成的token长度
        if cur_len == 1:  # 如果当前长度为1
            num_tokens = scores.shape[1]  # 得分的token数量
            scores[:, [i for i in range(num_tokens) if i != self.bos_token_id]] = -float("inf")  # 将除了第一个token id之外的token的得分设置为负无穷
            scores[:, self.bos_token_id] = 0  # 将第一个token的得分设置为0
        return scores  # 返回处理后的得分


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

    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2, 3", return_tensors="pt")

    >>> # By default, it continues generating according to the model's logits
    >>> outputs = model.generate(**inputs, max_new_tokens=10)
    >>> print(tokenizer.batch_decode(outputs)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8

    >>> # `forced_eos_token_id` ensures the generation ends with a EOS token
    >>> outputs = model.generate(**inputs, max_new_tokens=10, forced_eos_token_id=tokenizer.eos_token_id)
    ```
    # 打印第一个输出序列的解码结果
    >>> print(tokenizer.batch_decode(outputs)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7,<|endoftext|>
    
    
    
    # 定义一个 LogitsProcessor 类，用于处理模型输出的逻辑层
    class LogitsProcessor:
        # 初始化 LogitsProcessor 类的实例
        def __init__(self, max_length: int, eos_token_id: Union[int, List[int]]):
            # 设置最大长度和 EOS 标记的 ID
            self.max_length = max_length
            # 如果 EOS 标记的 ID 是单个整数，则转换为列表形式
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            self.eos_token_id = eos_token_id
    
        # 将对象实例作为函数调用，处理输入的模型输出 logits 和输入的 token IDs
        @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            # 获取当前输入 token IDs 的长度
            cur_len = input_ids.shape[-1]
            # 如果当前长度达到最大长度减一，则执行以下操作
            if cur_len == self.max_length - 1:
                # 获取 logits 中 token 的数量
                num_tokens = scores.shape[1]
                # 将非 EOS 标记的 token 对应的 logits 设置为负无穷
                scores[:, [i for i in range(num_tokens) if i not in self.eos_token_id]] = -float("inf")
                # 将 EOS 标记的 token 对应的 logits 设置为零
                for i in self.eos_token_id:
                    scores[:, i] = 0
            # 返回处理后的 logits
            return scores
class InfNanRemoveLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that removes all `nan` and `inf` values to avoid the generation method to fail. Note that using
    the logits processor should only be used if necessary since it can slow down the generation method.

    This logits processor has no `generate` example, as there shouldn't be a correct combination of flags that warrants
    its use.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 将所有的 nan 值设为 0.0
        scores[scores != scores] = 0.0

        # 将所有的 +/-inf 值设为最大/最小可能的值
        scores[scores == float("inf")] = torch.finfo(scores.dtype).max
        scores[scores == float("-inf")] = torch.finfo(scores.dtype).min

        return scores


class ExponentialDecayLengthPenalty(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that exponentially increases the score of the `eos_token_id` after `start_index` has been
    reached. This allows generating shorter sequences without having a hard cutoff, allowing the `eos_token` to be
    predicted in a meaningful position.

    Args:
        exponential_decay_length_penalty (`tuple(int, float)`):
            This tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty
            starts and `decay_factor` represents the factor of exponential decay
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        input_ids_seq_length (`int`):
            The length of the input sequence.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")

    >>> text = "Just wanted to let you know, I"
    >>> inputs = tokenizer(text, return_tensors="pt")

    >>> # Let's consider that we want short sentences, so we limit `max_length=30`. However, we observe that the answer
    >>> # tends to end abruptly.
    >>> set_seed(1)
    >>> outputs = model.generate(**inputs, do_sample=True, temperature=0.9, max_length=30, pad_token_id=50256)
    >>> print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network which was
    published in 2010. Although

    >>> # To promote the appearance of the EOS token at the right time, we add the `exponential_decay_length_penalty =
    >>> # (start_index, decay_factor)`. Instead of cutting at max_tokens, the output comes to an end before and usually
    >>> # with more meaning. What happens is that starting from `start_index` the EOS token score will be increased
    ```
    # 设置初始种子来生成随机序列
    set_seed(1)
    # 使用模型生成文本输出
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.9,
        max_length=30,
        pad_token_id=50256,
        exponential_decay_length_penalty=(15, 1.6),
    )
    # 打印生成的文本输出
    print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network which<|endoftext|>

    # 当使用较小的衰减因子时，更有可能得到含义明确的序列
    set_seed(1)
    # 使用模型生成文本输出
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.9,
        max_length=30,
        pad_token_id=50256,
        exponential_decay_length_penalty=(15, 1.01),
    )
    # 打印生成的文本输出
    print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network which was published in 2010.<|endoftext|>
    """
    # 初始化类，接受指数衰减长度惩罚、结束标记ID和输入ID序列长度作为参数
    def __init__(
        self,
        exponential_decay_length_penalty: Tuple[int, float],
        eos_token_id: Union[int, List[int]],
        input_ids_seq_length: int,
    ):
        # 计算惩罚启动位置
        self.regulation_start = exponential_decay_length_penalty[0] + input_ids_seq_length
        # 设置惩罚因子
        self.regulation_factor = exponential_decay_length_penalty[1]
        # 如果结束标记ID为整数，转换为列表
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id

    # 对输入ID和得分进行处理
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 获取当前长度
        cur_len = input_ids.shape[-1]
        # 如果当前长度大于惩罚启动位置
        if cur_len > self.regulation_start:
            # 对每个结束标记ID进行处理
            for i in self.eos_token_id:
                # 计算惩罚索引
                penalty_idx = cur_len - self.regulation_start
                # 支持负对数的情况下，计算绝对值的惩罚并加到原始对数中
                scores[:, i] = scores[:, i] + torch.abs(scores[:, i]) * (pow(self.regulation_factor, penalty_idx) - 1)
        return scores
class LogitNormalization(LogitsProcessor, LogitsWarper):
    r"""
    [`LogitsWarper`] and [`LogitsProcessor`] for normalizing the scores using log-softmax. It's important to normalize
    the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in
    this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
    the scores are normalized when comparing the hypotheses.
    此类用于对得分进行 log-softmax 标准化的[`LogitsWarper`]和[`LogitsProcessor`]。在应用 logits 处理器或 warper 后，在进行 beam search 期间对得分进行标准化是很重要的，因为这个库中使用的搜索算法没有做标准化（它只在之前做了，但可能需要重新标准化），但它仍然假定在比较假设时得分是标准化的。

    Examples:
    例子：
    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    >>> import torch

    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

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
    ```python中的例子：
    # 通过默认情况下，得分没有被标准化——它们的指数和不是标准化的概率分布，总和是1
    # 通过默认情况下，得分没有被标准化——它们的指数和不是标准化的概率分布，总和是1
    >>> outputs = model.generate(**inputs, renormalize_logits=True, return_dict_in_generate=True, output_scores=True)
    >>> print(torch.sum(torch.exp(outputs.scores[-1])))
    tensor(1.0000)
    ```

    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 对 scores 执行 log-softmax 标准化
        scores = scores.log_softmax(dim=-1)
        # 返回标准化后的 scores
        return scores


class SuppressTokensAtBeginLogitsProcessor(LogitsProcessor):
    r"""
    [`SuppressTokensAtBeginLogitsProcessor`] suppresses a list of tokens as soon as the `generate` function starts
    generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` are
    not generated at the beginning. Originally created for
    [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

    Examples:
    例子：
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
    ```python中的例子：
    # 默认情况下，Whisper has 'begin_suppress_tokens' 默认为（=[220, 50256]），50256是EOS令牌，因此这意味着它在第一次迭代中无法生成EOS令牌，但在其他迭代中可以。
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    ```
    # 输出模型的第一个生成token的分数
    print(outputs.scores[1][0, 50256])  # 1 (and not 0) is the first freely generated token
    # 输出模型生成的最后一个token的分数
    print(outputs.scores[-1][0, 50256])  # in other places we can see some probability mass for EOS

    # 如果禁用了`begin_suppress_tokens`，则可以在第一次迭代中生成EOS
    outputs = model.generate(
        **inputs, return_dict_in_generate=True, output_scores=True, begin_suppress_tokens=None
    )
    # 输出禁止生成的token的分数
    print(outputs.scores[1][0, 50256])
    """

    # 初始化函数，设置禁止生成的token列表和开始索引
    def __init__(self, begin_suppress_tokens, begin_index):
        self.begin_suppress_tokens = list(begin_suppress_tokens)
        self.begin_index = begin_index

    # 设置开始索引
    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    # 将禁止生成的token的分数设置为负无穷
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 如果输入的token序列长度等于开始索引，将禁止生成的token的分数设为负无穷
        if input_ids.shape[1] == self.begin_index:
            scores[:, self.begin_suppress_tokens] = -float("inf")

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
        # 初始化函数，接受要抑制的 token 列表
        self.suppress_tokens = list(suppress_tokens)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 为输入的 logit 矩阵中的指定 token 位置赋值 `-inf`，从而抑制这些 token 的生成
        scores[:, self.suppress_tokens] = -float("inf")
        return scores


class ForceTokensLogitsProcessor(LogitsProcessor):
    r"""
    This processor takes a list of pairs of integers which indicates a mapping from generation indices to token
    indices that will be forced before generation. The processor will set their log probs to `inf` so that they are
    sampled at their corresponding index. Originally created for
    [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

    Examples:
    ```python
    >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> # This Whisper model forces the generation to start with `50362` at the first position by default, i.e.
    >>> # `"forced_decoder_ids": [[1, 50362]]`. This means all other tokens are masked out.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    >>> print(
    # 使用给定的条件来验证一个列表中的所有元素是否为 True
    all(
        # 检查每个索引处的值是否为负无穷，并且索引不等于 50362
        outputs.scores[0][0, i] == float("-inf") for i in range(processor.tokenizer.vocab_size) if i != 50362
    )
    # 返回 True，表示所有条件都满足

True

>>> # 打印指定索引处的分数
>>> print(outputs.scores[0][0, 50362])
tensor(0.)

>>> # 如果禁用了 `forced_decoder_ids`，就不会看到上述效果
>>> # 生成模型不受强制 token ID 的影响
>>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, forced_decoder_ids=None)
>>> # 检查是否存在不符合条件的分数
>>> print(
...     all(
...         outputs.scores[0][0, i] == float("-inf") for i in range(processor.tokenizer.vocab_size) if i != 50362
...     )
... )
# 返回 False，表示不是所有的条件都满足
>>> # 打印指定索引处的分数
>>> print(outputs.scores[0][0, 50362])
tensor(19.3140)



# 初始化函数，接受一个强制 token 映射的列表
def __init__(self, force_token_map: List[List[int]]):
    # 将输入的强制 token 映射转换为字典并存储在实例中
    self.force_token_map = dict(force_token_map)

@add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
# 调用对象时，根据传入的输入 ID 和分数进行处理，并返回处理后的分数
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    # 获取当前生成的 token 索引
    generation_idx = input_ids.shape[-1]
    # 查找当前生成的 token 是否在强制 token 映射中
    current_token = self.force_token_map.get(generation_idx, None)
    # 如果当前生成的 token 在映射中，则将所有分数设置为负无穷，并将指定 token 的分数设为 0
    if current_token is not None:
        scores[:, :] = -float("inf")
        scores[:, current_token] = 0
    # 返回处理后的分数
    return scores
class WhisperTimeStampLogitsProcessor(LogitsProcessor):
    r"""
    # 一个`LogitsProcessor`，用于修改用于生成转录时间戳的logits。当输入token达到特定阈值时，处理器将分数设置为负无穷大。处理器确保时间戳token成对出现，通过屏蔽破坏此配对模式的logits来实现。这样做是为了保持生成的时间戳的一致性和结构。它还确保当预测的时间戳token的采样概率大于任何单个非时间戳token时，那些非时间戳logits被设置为负无穷大。这样做是为了确保生成时间戳而不是其他潜在token。

    # 有关更多信息，请参阅[论文](https://arxiv.org/abs/2212.04356)。

    Args:
        generate_config (`GenerateConfig`):
            用于生成输出的生成配置。需要以下参数：
                eos_token_id (`int`, *optional*, 默认为50257):
                    *end-of-sequence* token的id。
                no_timestamps_token_id (`int`, *optional*, 默认为50363):
                    `"<|notimestamps|>"` token的id。
                max_initial_timestamp_index (`int`, *optional*, 默认为1):
                    用于设置初始时间戳的最大值。这是为了防止模型预测太远未来的时间戳。
        begin_index (`Optional`, *optional*): 模型生成的第一个token的token索引。
        _detect_timestamp_from_logprob (`bool`, *optional*): 是否可以从所有时间戳的logprobs中预测时间戳。

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

    >>> #显示时间戳
    >>> generated_ids = model.generate(inputs=input_features, return_timestamps=True)
    >>> transcription = processor.batch_decode(generated_ids, decode_with_timestamps=True)[0]
    >>> print("Transcription:", transcription)
    Transcription: <|startoftranscript|><|0.00|> He has grave doubts whether Sir Frederick Layton's work is really Greek after all, and can<|6.44|><|6.44|> discover in it but little of rocky Ithaca.<|9.44|><|endoftext|>

    >>> #无时间戳和更改EOS:
    # 设置生成配置中的结束标记ID，这里是单词"can"（ID为460）
    model.generation_config.eos_token_id = 460
    # 生成文本序列的ID
    generated_ids = model.generate(inputs=input_features, return_timestamps=False)
    # 将生成的ID解码为文本
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # 打印生成的转录文本
    print("Transcription:", transcription)

    # 初始化函数，接受生成配置、开始索引和检测时间戳的日志概率作为可选参数
    def __init__(
        self, generate_config, begin_index: Optional[int] = None, _detect_timestamp_from_logprob: Optional[bool] = None
    ):  # 支持kwargs
        # 设置无时间戳标记的ID
        self.no_timestamps_token_id = generate_config.no_timestamps_token_id
        # 计算时间戳开始的ID
        self.timestamp_begin = generate_config.no_timestamps_token_id + 1
        # 设置结束标记ID为生成配置中的结束标记ID或者开始标记ID
        self.eos_token_id = generate_config.eos_token_id or generate_config.bos_token_id

        # 主要用于测试，用于检测时间戳的日志概率
        self._detect_timestamp_from_logprob = (
            _detect_timestamp_from_logprob
            if _detect_timestamp_from_logprob is not None
            else getattr(generate_config, "_detect_timestamp_from_logprob", True)
        )

        # 计算强制ID的数量，若无强制ID，则为0
        num_forced_ids = (
            len(generate_config.forced_decoder_ids) if generate_config.forced_decoder_ids is not None else 0
        )
        # 设置开始索引为传入的开始索引或者强制ID数量加1
        self.begin_index = begin_index or (num_forced_ids + 1)

        # 获取生成配置中的最大初始时间戳索引
        self.max_initial_timestamp_index = getattr(generate_config, "max_initial_timestamp_index", None)
        # TODO（Patrick）：确保官方模型将max_initial_timestamp_index设置为50
        # self.max_initial_timestamp_index = 50

    # 设置开始索引的函数
    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    # 添加文档字符串到函数
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    # 定义一个调用函数，接受输入的input_ids（torch.LongTensor类型）和scores（torch.FloatTensor类型），返回torch.FloatTensor类型
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 抑制对no_timestamps_token_id对应位置的分数，这个位置的token由without_timestamps处理
        scores[:, self.no_timestamps_token_id] = -float("inf")

        # 时间戳必须成对出现，除非直接在eos_token之前；相应地，对logits进行掩码处理
        for k in range(input_ids.shape[0]):
            # 获取采样的tokens序列，从begin_index索引开始
            sampled_tokens = input_ids[k, self.begin_index :]
            # 将序列转换为列表
            seq = list(sampled_tokens.tolist())

            # 最后一个是否为时间戳
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.timestamp_begin
            # 倒数第二个是否为时间戳
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.timestamp_begin

            # 如果最后一个是时间戳
            if last_was_timestamp:
                # 如果倒数第二个是时间戳，则必须是非时间戳
                if penultimate_was_timestamp:
                    scores[k, self.timestamp_begin :] = -float("inf")
                else:  
                    # 否则，不能是普通文本token
                    scores[k, : self.eos_token_id] = -float("inf")

            # 获取tokens中的时间戳
            timestamps = sampled_tokens[sampled_tokens.ge(self.timestamp_begin)]
            if timestamps.numel() > 0:
                # `timestamps` 不能递减；禁止小于最后一个的时间戳tokens
                # 以下代码行的内容来自于: https://github.com/openai/whisper/pull/914/files#r1137085090
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    # 避免再次生成<|0.00|>
                    timestamp_last = timestamps[-1] + 1

                scores[k, self.timestamp_begin : timestamp_last] = -float("inf")

        # 应用`max_initial_timestamp`选项
        if input_ids.shape[1] == self.begin_index:
            scores[:, : self.timestamp_begin] = -float("inf")

            if self.max_initial_timestamp_index is not None:
                last_allowed = self.timestamp_begin + self.max_initial_timestamp_index
                scores[:, last_allowed + 1 :] = -float("inf")

        # 如果时间戳的概率之和高于其他任何token，那么采样时间戳
        logprobs = torch.nn.functional.log_softmax(scores.float(), dim=-1)
        for k in range(input_ids.shape[0]):
            # 时间戳的log概率之和
            timestamp_logprob = logprobs[k, self.timestamp_begin :].logsumexp(dim=-1)
            # 文本token的最大log概率
            max_text_token_logprob = logprobs[k, : self.timestamp_begin].max()
            # 如果时间戳的log概率之和大于文本token的最大log概率，并且启用了_detect_timestamp_from_logprob
            if timestamp_logprob > max_text_token_logprob and self._detect_timestamp_from_logprob:
                scores[k, : self.timestamp_begin] = -float("inf")

        # 返回处理后的scores
        return scores
# 定义用于在 Whisper 中检测静默的 Logits 处理器，需要传入未处理的 logits 来遵循原始实现
class WhisperNoSpeechDetection(LogitsProcessor):
    # 初始化方法，接受参数：无语音标记、开始索引和判断分数是否为对数概率
    def __init__(self, no_speech_token: int, begin_index: int, scores_is_logprobs: bool = False):
        # 设定无语音标记
        self.no_speech_token = no_speech_token
        # 开始索引和生成的第一个标记之间的偏移量
        self.start_of_trans_offset = begin_index
        # 运行时会动态变化的开始索引
        self.begin_index = begin_index
        # 初始化无语音概率为 0
        self._no_speech_prob = [0.0]
        # 判断分数是否为对数概率
        self.is_scores_logprobs = scores_is_logprobs
        # 动态重写的属性
        self.model = None
        self.inputs = None

    # 设定模型
    def set_model(self, model):
        self.model = model

    # 设定输入
    def set_inputs(self, inputs):
        # 准备用于生成的输入并合并输入
        self.inputs = {**self.model.prepare_inputs_for_generation(**inputs), **inputs}
        self.inputs["input_features"] = self.inputs.pop("inputs")

    # 获取无语音概率
    @property
    def no_speech_prob(self):
        return self._no_speech_prob

    # 设置开始索引
    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    # 处理 logits 的方法
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 如果输入的形状的第二个维度与开始索引相同
        if input_ids.shape[1] == self.begin_index:
            if self.start_of_trans_offset > 1:
                # 使用无梯度计算创建 logits
                with torch.no_grad():
                    logits = self.model(**self.inputs).logits
                
                # 计算无语音索引和分数
                no_speech_index = self.begin_index - self.start_of_trans_offset
                no_speech_scores = logits[:, no_speech_index]
            else:
                no_speech_scores = scores

            # 如果分数为对数概率，则计算概率；否则计算 softmax
            if self.is_scores_logprobs:
                probs = no_speech_scores.exp()
            else:
                probs = no_speech_scores.float().softmax(dim=-1)

            # 更新无语音概率
            self._no_speech_prob = probs[:, self.no_speech_token]

        return scores


# 分类器自由指导（CFG）Logits 处理器
class ClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    # 初始化 CFG 处理器
    def __init__(self, no_speech_token: int, begin_index: int, scores_is_logprobs: bool = False):
        # 分类器自由指导（CFG）处理器，将分数在批处理维度上分割，前一半对应条件对数（从输入提示预测），后一半对应无条件对数（从空或“null”提示预测）。
        # 处理器计算条件和无条件对数之间的加权平均值，由参数 `guidance_scale` 参数化。
        # 查看更多信息请参考论文 https://arxiv.org/abs/2306.05284
        # <Tip warning={true}>

        # 此 logits 处理器专门与 MusicGen 兼容
        pass
    Args:
        guidance_scale (float):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.

    Examples:

    ```python
    >>> from transformers import AutoProcessor, MusicgenForConditionalGeneration

    >>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    >>> inputs = processor(
    ...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
    ...     padding=True,
    ...     return_tensors="pt",
    ... )
    >>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
    ```
    """

    # 初始化函数，接受一个 guidance_scale 参数
    def __init__(self, guidance_scale):
        # 检查 guidance_scale 是否大于 1，用于启用分类器自由引导处理器
        if guidance_scale > 1:
            # 如果大于 1，则将其赋值给对象的 guidance_scale 属性
            self.guidance_scale = guidance_scale
        else:
            # 如果 guidance_scale 不大于 1，则引发 ValueError 异常
            raise ValueError(
                "Require guidance scale >1 to use the classifier free guidance processor, got guidance scale "
                f"{guidance_scale}."
            )

    # 对象可调用函数，接受 input_ids（torch.LongTensor）和 scores（torch.FloatTensor）参数，返回 torch.FloatTensor
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 简单检查以确保我们的 logits 分数（条件 + 无条件）和输入 id（仅条件）之间具有兼容的批次大小
        if scores.shape[0] != 2 * input_ids.shape[0]:
            # 如果不匹配，引发 ValueError 异常
            raise ValueError(
                f"Logits should have twice the batch size of the input ids, the first half of batches corresponding to "
                f"the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got "
                f"batch size {scores.shape[0]} for the logits and {input_ids.shape[0]} for the input ids."
            )
        # 计算无指导批次大小
        unguided_bsz = scores.shape[0] // 2
        # 将 scores 沿着批次维度分割为条件和无条件 logits
        cond_logits, uncond_logits = scores.split(unguided_bsz, dim=0)
        # 根据指导比例调整分数，将未条件 logits 与（条件 logits - 无条件 logits）相加，乘以 guidance_scale
        scores = uncond_logits + (cond_logits - uncond_logits) * self.guidance_scale
        # 返回调整后的 scores
        return scores
# 定义一个LogitsProcessor子类，用于在Bark的两个码书之间强制交替生成。
# 这是一个专门用于Bark的fine子模型的logits processor。请参阅模型文档了解示例。
# 该类有三个参数：input_start_len，semantic_vocab_size和codebook_size。
class AlternatingCodebooksLogitsProcessor(LogitsProcessor):

    def __init__(self, input_start_len: int, semantic_vocab_size: int, codebook_size: int):
        # 检查input_start_len是否为非负整数，如果不是则抛出ValueError异常
        if not isinstance(input_start_len, int) or input_start_len < 0:
            raise ValueError(f"`input_starting_length` has to be a non-negative integer, but is {input_start_len}")
        
        # 初始化对象的输入起始长度、语义词汇表大小和码书大小
        self.input_start_len = input_start_len
        self.semantic_vocab_size = semantic_vocab_size
        self.codebook_size = codebook_size

    # 方法用于处理logits
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 获取输入ids的当前长度
        curr_len = input_ids.shape[-1]

        # 计算当前长度除以输入起始长度的余数，来判断是应该使用第一个码书还是第二个码书
        # 如果curr_len - self.input_start_len是偶数，则is_first_codebook为True，否则为False
        is_first_codebook = ((curr_len - self.input_start_len) % 2) == 0

        # 根据is_first_codebook的值，将scores中的一部分置为负无穷
        if is_first_codebook:
            scores[:, : self.semantic_vocab_size] = -float("inf")
            scores[:, self.semantic_vocab_size + self.codebook_size :] = -float("inf")
        else:
            scores[:, : self.semantic_vocab_size + self.codebook_size] = -float("inf")

        # 返回处理后的scores
        return scores
    # 构造函数，初始化CFGSampler对象
    def __init__(
        self,
        # CFG的指导比例
        guidance_scale: float,
        # 用于计算无条件分数的模型
        model,
        # 无条件分支的输入序列标记的索引
        unconditional_ids: Optional[torch.LongTensor] = None,
        # 无条件分支的注意力掩码
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        # 是否在负向提示的前向传递期间缓存键/值对
        use_cache: Optional[bool] = True,
    # 初始化对象时设置引导比例和模型
    # 初始化无条件上下文字典
    # 设置无条件上下文字典的输入 id、注意力遮罩，缓存使用和过去的键值对
    def __init__(self, guidance_scale, model, unconditional_ids=None, unconditional_attention_mask=None, use_cache=False):
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    # 获取无条件 logits
    def get_unconditional_logits(self, input_ids):
        # 如果是第一次调用，则设置输入 id 和注意力遮罩
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
        # 如果不是第一次调用，则根据需要添加新的输入 id 和注意力遮罩
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

        # 使用模型获取输出，设置返回值
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits

    # 定义对象的调用方法
    def __call__(self, input_ids, scores):
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        # 如果引导比例为1，则直接返回 scores
        if self.guidance_scale == 1:
            return scores

        # 获取无条件 logits
        logits = self.get_unconditional_logits(input_ids)

        # 计算无条件 logits 的 softmax 值
        unconditional_logits = torch.nn.functional.log_softmax(logits[:, -1], dim=-1)
        # 计算得到最终输出，根据引导比例调整 scores 的值
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
        min_eos_p (`float`, *optional`):
            Minimum end of speech threshold.
    """
    
    # 构造函数，初始化对象
    def __init__(self, eos_token_id: Union[int, List[int]], min_eos_p: float):
        # 如果 eos_token_id 是单个整数，转换为列表
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id
        # 如果 min_eos_p 不为 None 且小于等于 0，抛出 ValueError 异常
        if min_eos_p is not None and min_eos_p <= 0:
            raise ValueError(f"`min_eos_p` has to be a positive float, but is {min_eos_p}")
        self.min_eos_p = min_eos_p

    # 对输入数据进行处理
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 如果 min_eos_p 存在
        if self.min_eos_p:
            # 计算概率，将 scores 转换为概率分布
            probs = torch.nn.functional.softmax(scores.float(), dim=-1)
            # 创建一个由 -inf 填充的 scores，除了 eos_token_id
            early_stop_scores = torch.ones_like(scores) * -float("inf")
            early_stop_scores[:, self.eos_token_id] = scores[:, self.eos_token_id]

            # 判断是否进行提前终止
            do_early_stop = probs[:, self.eos_token_id] > self.min_eos_p
            do_early_stop = torch.any(do_early_stop, dim=1, keepdim=True)
            # 根据条件选择更新 scores
            scores = torch.where(do_early_stop, early_stop_scores, scores)

        return scores
```