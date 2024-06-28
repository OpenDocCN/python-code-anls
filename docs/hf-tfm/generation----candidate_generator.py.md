# `.\generation\candidate_generator.py`

```py
# coding=utf-8
# 版权所有 2023 年 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本进行许可；
# 除非符合许可证要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 软件没有任何形式的明示或暗示担保或条件。
# 有关详细信息，请参阅许可证。
#

import copy  # 导入深拷贝模块
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple  # 导入类型提示模块

import torch  # 导入PyTorch模块


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel  # 导入预训练模型类型提示
    from .configuration_utils import GenerationConfig  # 导入生成配置类型提示
    from .logits_process import LogitsProcessorList  # 导入logits处理列表类型提示


class CandidateGenerator:
    """所有候选生成器的抽象基类，可在辅助生成过程中应用。"""

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        获取当前输入的候选生成序列。

        Args:
            input_ids (`torch.LongTensor`，形状为 `(batch_size, sequence_length)`):
                输入序列标记在词汇表中的索引。[什么是输入ID？](../glossary#input-ids)

        Return:
            `torch.LongTensor`，形状为 `(batch_size, candidate_length)`，包含模型评估的候选序列，
            以及一个可选的 `torch.FloatTensor`，形状为 `(batch_size, candidate_length, vocabulary_size)`，
            包含与每个候选相关的logits。
        """
        raise NotImplementedError(
            f"{self.__class__} 是一个抽象类。只有继承此类的类才能调用 `get_candidates`。"
        )

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        根据结果更新候选生成策略。

        Args:
            input_ids (`torch.LongTensor`，形状为 `(batch_size, sequence_length)`):
                输入序列标记在词汇表中的索引。[什么是输入ID？](../glossary#input-ids)
            scores (`torch.FloatTensor`，形状为 `(batch_size, candidate_length, config.vocab_size)`):
                语言建模头部的预测分数。当不使用beam搜索时，这些可以是每个词汇的logits，或者在使用beam搜索时，每个词汇token的log softmax。
            num_matches (`int`):
                候选序列与模型预测之间的匹配数。
        """
        raise NotImplementedError(
            f"{self.__class__} 是一个抽象类。只有继承此类的类才能调用 `update_candidate_strategy`。"
        )
    """
    `CandidateGenerator` class to be used for assisted generation and speculative decoding. This class generates
    candidates through the use of a smaller model. Read the following blog post for more information:
    https://huggingface.co/blog/assisted-generation
    """

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        logits_processor: "LogitsProcessorList",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the `AssistedCandidateGenerator` with necessary parameters.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            assistant_model (`PreTrainedModel`):
                The model used for generating candidates, which is smaller than the main model.
            generation_config (`~generation.GenerationConfig`, *optional*):
                Configuration for the generation process.
            logits_processor (`LogitsProcessorList`):
                List of processors to modify prediction scores of the language modeling head during generation.
            model_kwargs (`Dict`):
                Keyword arguments passed to the main model and the assistant model.
            inputs_tensor (`torch.Tensor`, *optional*):
                The input tensor for the model, typically the encoder input in encoder-decoder models.
        """
        # 调用父类的初始化方法，传入输入的参数
        super().__init__(input_ids, assistant_model, generation_config, logits_processor, model_kwargs)
        # 将输入的张量赋值给实例变量
        self.inputs_tensor = inputs_tensor
    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """
        # Move input_ids tensor to the device of the assistant model
        input_ids = input_ids.to(self.assistant_model.device)

        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        new_cur_len = input_ids.shape[-1]
        max_new_tokens = min(int(self.num_assistant_tokens), self.generation_config.max_length - new_cur_len - 1)
        if max_new_tokens == 0:
            return input_ids, None

        # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length
        # (which implicitly contains the number of accepted candidates from the previous round)

        # Check if there are past key values for the assistant model
        has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
        if has_past_key_values:
            # Calculate the new cache size based on current length minus one
            new_cache_size = new_cur_len - 1
            # Crop the past key values to match the new cache size
            self.assistant_kwargs["past_key_values"] = _crop_past_key_values(
                self.assistant_model, self.assistant_kwargs["past_key_values"], new_cache_size - 1
            )  # the assistant does not have the token after the last match, hence the -1

            # Prepare attention mask based on the new current length and model configuration
            self.assistant_kwargs = _prepare_attention_mask(
                self.assistant_kwargs, new_cur_len, self.assistant_model.config.is_encoder_decoder
            )

            # Prepare token type IDs based on the new current length
            self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)

        # 2. Forecast next N tokens using the assistant model.
        assistant_generation_kwargs = {
            self.input_ids_key: input_ids,
            "max_new_tokens": max_new_tokens,
            "generation_config": self.generation_config,
            "logits_processor": self.logits_processor,
        }

        # Generate candidate sequences and logits using the assistant model
        assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)

        # 3. Update variables for the next round of candidate generation
        # Update past key values for the assistant model with the latest output
        self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values

        # 4. Prepare variables for output
        # Stack candidate scores along the sequence dimension
        candidate_logits = torch.stack(assistant_output.scores, dim=1)
        # Get candidate sequence IDs
        candidate_ids = assistant_output.sequences
        return candidate_ids, candidate_logits
    # 更新候选生成策略基于结果的函数
    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        # 调整下一个迭代中使用的助手标记的最大数量。这是一个简单的启发式方法，可能可以改进 -- 我们希望在获取正确的助手标记的好处与预测错误的代价之间取得平衡。
        if self.assistant_model.generation_config.num_assistant_tokens_schedule in {
            "heuristic",
            "heuristic_transient",
        }:
            # 如果匹配数等于当前助手标记数量，则增加助手标记数量
            if num_matches == int(self.num_assistant_tokens):
                self.num_assistant_tokens += 2.0
            else:
                # 否则，减少助手标记数量，但不低于1.0
                self.num_assistant_tokens = max(1.0, self.num_assistant_tokens - 1.0)
# 定义一个候选生成器类 `PromptLookupCandidateGenerator`，继承自 `CandidateGenerator` 类。
# 该类用于生成基于提示查找的候选结果。它通过查找在提供的提示（input_ids）中可能的延续来生成候选结果。
# 更多信息请查阅以下博客文章：https://github.com/apoorvumang/prompt-lookup-decoding
class PromptLookupCandidateGenerator(CandidateGenerator):
    
    """
    `CandidateGenerator` class to be used for prompt lookup generation. This class generates candidates by looking up
    likely continuations in the provided prompt (input_ids) itself.
    Read the following blog post for more information: https://github.com/apoorvumang/prompt-lookup-decoding
    """

    def __init__(
        self,
        num_output_tokens: int = 10,
        max_matching_ngram_size: int = None,
    ):
        # 初始化方法，设置候选结果输出的 token 数量和最大匹配的 ngram 大小。
        self.num_output_tokens = num_output_tokens
        # 如果未指定最大匹配的 ngram 大小，则默认为 2
        self.max_matching_ngram_size = max_matching_ngram_size if max_matching_ngram_size else 2

        # 如果最大匹配的 ngram 大小或者输出的 token 数量小于等于 0，则抛出数值错误异常。
        if self.max_matching_ngram_size <= 0 or self.num_output_tokens <= 0:
            raise ValueError("Invalid max_matching_ngram_size or num_output_tokens")
        def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
            """
            Fetches the candidates to be tried for the current input.

            Args:
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                    Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

            Return:
                `torch.LongTensor` of shape `(num_candidates, candidate_length)`: The candidate sequences to be tried.
            """
            input_length = input_ids.size(1)  # 获取输入的序列长度

            chosen_ids = None  # 初始化 chosen_ids 为 None
            match_found = False  # 初始化 match_found 为 False
            for ngram_size in range(min(self.max_matching_ngram_size, input_length - 1), 0, -1):  # 遍历 ngram 大小
                # 创建大小为 ngram_size 的滑动窗口
                windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

                # 将 ngram 转换为张量进行比较
                ngram_tensor = input_ids[0, -ngram_size:]

                # 查找窗口与 ngram 匹配的位置
                matches = (windows == ngram_tensor).all(dim=2)

                # 获取匹配的索引
                match_indices = matches.nonzero(as_tuple=True)[1]

                # 遍历匹配索引以找到有效的延续
                for idx in match_indices:
                    start_idx = idx + ngram_size
                    end_idx = start_idx + self.num_output_tokens
                    end_idx = min(end_idx, input_length)

                    if start_idx < end_idx:
                        chosen_ids = input_ids[0, start_idx:end_idx]
                        match_found = True
                        break
                if match_found:
                    break

            if chosen_ids is None or len(chosen_ids) == 0:
                # 如果没有找到匹配，则返回未更改的输入序列，恢复自回归解码
                return input_ids, None

            # 现在需要用 chosen_ids 扩展 input_ids
            chosen_ids = chosen_ids.unsqueeze(0)
            candidate_input_ids = torch.cat((input_ids, chosen_ids), dim=1)
            # assisted_generation 预期也返回 logits，但这里我们没有，所以返回 None
            return candidate_input_ids, None
    # 更新候选生成策略，根据结果进行调整

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        # 当前函数暂未实现任何功能，仅作为占位符
        return
# 将过去的键值对裁剪到指定的最大长度
def _crop_past_key_values(model, past_key_values, maximum_length):
    new_past = []
    # 如果模型是编码-解码模型
    if model.config.is_encoder_decoder:
        # 遍历过去的键值对
        for idx in range(len(past_key_values)):
            # 裁剪过去的键值对的内容，保留最大长度内的部分
            new_past.append(
                (
                    past_key_values[idx][0][:, :, :maximum_length, :],
                    past_key_values[idx][1][:, :, :maximum_length, :],
                    past_key_values[idx][2],
                    past_key_values[idx][3],
                )
            )
        past_key_values = tuple(new_past)
    # 如果模型类名中包含"bloom"，或者模型架构中的第一个类名中包含"bloom"
    elif "bloom" in model.__class__.__name__.lower() or (
        model.config.architectures is not None and "bloom" in model.config.architectures[0].lower()
    ):
        # 遍历过去的键值对
        for idx in range(len(past_key_values)):
            # 根据不同的维度裁剪过去的键值对的内容，保留最大长度内的部分
            new_past.append(
                (
                    past_key_values[idx][0][:, :, :maximum_length],
                    past_key_values[idx][1][:, :maximum_length, :],
                )
            )
        past_key_values = tuple(new_past)
    # 如果模型类名中包含"gptbigcode"，或者模型架构中的第一个类名中包含"gptbigcode"
    elif "gptbigcode" in model.__class__.__name__.lower() or (
        model.config.architectures is not None and "gptbigcode" in model.config.architectures[0].lower()
    ):
        # 如果是多重查询模型
        if model.config.multi_query:
            # 遍历过去的键值对，裁剪为最大长度的内容
            for idx in range(len(past_key_values)):
                past_key_values[idx] = past_key_values[idx][:, :maximum_length, :]
        else:
            # 遍历过去的键值对，裁剪为最大长度的内容
            for idx in range(len(past_key_values)):
                past_key_values[idx] = past_key_values[idx][:, :, :maximum_length, :]
    else:
        # 遍历过去的键值对
        for idx in range(len(past_key_values)):
            # 裁剪过去的键值对的内容，保留最大长度内的部分
            new_past.append(
                (
                    past_key_values[idx][0][:, :, :maximum_length, :],
                    past_key_values[idx][1][:, :, :maximum_length, :],
                )
            )
        past_key_values = tuple(new_past)
    return past_key_values


# 扩展或裁剪模型的注意力掩码，以用于解码目的，调整到指定的长度
def _prepare_attention_mask(model_kwargs: Dict[str, Any], new_length: int, is_encoder_decoder: bool) -> Dict[str, Any]:
    """Expands or crops the model's mask for decoding purposes, to the defined length"""

    mask_key = "decoder_attention_mask" if is_encoder_decoder else "attention_mask"
    # 如果模型参数中不包含指定的掩码键值，则直接返回模型参数
    if mask_key not in model_kwargs:
        return model_kwargs

    mask = model_kwargs[mask_key]
    mask_length_diff = new_length - mask.shape[1]

    # 如果掩码长度超出了需要的长度，则裁剪掩码
    if mask_length_diff < 0:
        model_kwargs[mask_key] = mask[:, :mask_length_diff]
    # 如果掩码长度不足需要的长度，则扩展掩码
    elif mask_length_diff > 0:
        model_kwargs[mask_key] = torch.cat([mask, mask.new_ones((mask.shape[0], mask_length_diff))], dim=-1)
    return model_kwargs


# 扩展或裁剪模型的token_type_ids，以用于解码目的，调整到指定的长度
def _prepare_token_type_ids(model_kwargs: Dict[str, Any], new_length: int) -> Dict[str, Any]:
    """Expands or crops the model's token_type_ids for decoding purposes, to the defined length"""
    # 如果模型参数中不包含token_type_ids或者其值为空，则直接返回模型参数
    if "token_type_ids" not in model_kwargs or model_kwargs["token_type_ids"] is None:
        return model_kwargs
    # 获取模型参数字典中的 token_type_ids
    token_type_ids = model_kwargs["token_type_ids"]
    
    # 获取 token_type_ids 的最后一个元素，并在最后增加一个维度
    final_token_type = token_type_ids[:, -1].unsqueeze(-1)
    
    # 计算新长度与当前 token_type_ids 的长度之差
    type_length_diff = new_length - token_type_ids.shape[1]
    
    # 根据长度差进行条件判断和处理
    if type_length_diff < 0:
        # 如果长度差小于零，截取 token_type_ids 的前 type_length_diff 部分
        token_type_ids = token_type_ids[:, :type_length_diff]
    elif type_length_diff > 0:
        # 如果长度差大于零，复制 final_token_type，使其与长度差匹配，并将其拼接到 token_type_ids 后面
        token_type_copies = final_token_type.repeat(1, type_length_diff)
        model_kwargs["token_type_ids"] = torch.cat([model_kwargs["token_type_ids"], token_type_copies], dim=-1)
    
    # 返回更新后的模型参数字典
    return model_kwargs
```