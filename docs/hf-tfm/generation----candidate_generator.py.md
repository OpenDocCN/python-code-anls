# `.\transformers\generation\candidate_generator.py`

```
# 导入所需的模块和类型提示
import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

# 如果是类型检查阶段，导入特定的模块
if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from .configuration_utils import GenerationConfig
    from .logits_process import LogitsProcessorList

# 候选生成器的抽象基类，用于在辅助生成过程中应用所有候选生成器
class CandidateGenerator:
    """Abstract base class for all candidate generators that can be applied during assisted generation."""

    # 获取当前输入的候选序列
    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and, optionally, a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """
        # 抛出未实现错误，要求继承该类的类来实现这个方法
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `get_candidates`."
        )

    # 根据结果更新候选生成策略
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
        # 抛出未实现错误，要求继承该类的类来实现这个方法
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call "
            "`update_candidate_strategy`."
        )
# 定义一个名为AssistedCandidateGenerator的类，继承自CandidateGenerator类，用于辅助生成和推测解码候选项
class AssistedCandidateGenerator(CandidateGenerator):
    """
    `CandidateGenerator`类用于辅助生成和推测解码。该类通过使用一个较小的模型来生成候选项。阅读以下博客文章以获取更多信息：
    https://huggingface.co/blog/assisted-generation

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。[什么是输入ID？](../glossary#input-ids)
        assistant_model (`PreTrainedModel`):
            用于生成候选项的模型。该模型应比主模型小。
        generation_config (`~generation.GenerationConfig`, *optional*):
            用作生成调用的基本参数化的生成配置。
        logits_processor (`LogitsProcessorList`):
            一个[`LogitsProcessorList`]的实例。派生自[`LogitsProcessor`]类的实例列表，用于修改每个生成步骤应用的语言建模头的预测分数。
        model_kwargs (`Dict`):
            将传递给主模型的关键字参数，并作为助手模型的基本输入。
        inputs_tensor (`torch.Tensor`, *optional*):
            模型输入张量。在编码器-解码器模型中，这是编码器输入。
    """

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        logits_processor: "LogitsProcessorList",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
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
        # 将输入张量移动到助手模型所在的设备上
        input_ids = input_ids.to(self.assistant_model.device)

        # 不生成超过`max_length - 1`个候选项，因为目标模型会生成一个额外的标记。
        new_cur_len = input_ids.shape[-1]
        max_new_tokens = min(int(self.num_assistant_tokens), self.generation_config.max_length - new_cur_len - 1)
        if max_new_tokens == 0:
            return input_ids, None

        # 1. 如果不是候选生成的第一轮，根据输入的长度准备输入（这隐含了从上一轮已接受的候选中获取的数量）
        has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
        if has_past_key_values:
            new_cache_size = new_cur_len - 1
            # 裁剪过去的键值，使其长度与新的缓存大小匹配，考虑到助手模型没有最后一个匹配后的标记，因此减去1
            self.assistant_kwargs["past_key_values"] = _crop_past_key_values(
                self.assistant_model, self.assistant_kwargs["past_key_values"], new_cache_size - 1
            )

            # 准备注意力遮罩，根据输入的长度和助手模型配置的编码器-解码器情况
            self.assistant_kwargs = _prepare_attention_mask(
                self.assistant_kwargs, new_cur_len, self.assistant_model.config.is_encoder_decoder
            )
            # 准备标记类型ID
            self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)

        # 2. 使用助手模型预测接下来的N个标记。
        assistant_generation_kwargs = {
            self.input_ids_key: input_ids,
            "max_new_tokens": max_new_tokens,
            "generation_config": self.generation_config,
            "logits_processor": self.logits_processor,
        }

        # 生成助手模型的输出
        assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)

        # 3. 更新下一轮候选生成的变量
        self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values

        # 4. 准备输出的变量
        # 将候选的标记堆叠起来，形状为`(batch_size, candidate_length, vocabulary_size)`
        candidate_logits = torch.stack(assistant_output.scores, dim=1)
        # 获取候选标记序列
        candidate_ids = assistant_output.sequences
        return candidate_ids, candidate_logits
    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        更新候选生成策略基于结果。

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                输入序列标记在词汇表中的索引。[什么是输入ID？](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                语言建模头的预测分数。当不使用束搜索时，这些可以是每个词汇表的对数，当使用束搜索时，这些可以是每个词汇表标记的对数 softmax
            num_matches (`int`):
                候选序列与模型预测之间的匹配数量。
        """
        # 调整下一个迭代中要使用的助理标记的最大数量。这是一个简单的启发式方法，可能可以改进 -- 我们想要平衡正确获取助理标记的好处
        # 与预测错误助理标记的成本。
        if self.assistant_model.generation_config.num_assistant_tokens_schedule == "heuristic":
            # 如果匹配数量等于当前的助理标记数量，则增加2.0个助理标记
            if num_matches == int(self.num_assistant_tokens):
                self.num_assistant_tokens += 2.0
            # 否则，将助理标记数量减少1.0个，但最小为1.0
            else:
                self.num_assistant_tokens = max(1.0, self.num_assistant_tokens - 1.0)
# 定义 PromptLookupCandidateGenerator 类，用于生成基于提示查找的候选项。该类通过在提供的提示（input_ids）中查找可能的延续来生成候选项。
# 详细信息请参阅以下博客文章: https://github.com/apoorvumang/prompt-lookup-decoding

class PromptLookupCandidateGenerator(CandidateGenerator):
    """
    `CandidateGenerator` class to be used for prompt lookup generation. This class generates candidates by looking up
    likely continuations in the provided prompt (input_ids) itself.
    Read the following blog post for more information: https://github.com/apoorvumang/prompt-lookup-decoding

    Args:
        max_matching_ngram_size (`int`):
            The maximum ngram size to be considered for matching in the prompt
        num_output_tokens (`int`):
            The number of tokens to be output as candidate tokens.
    """

    # 初始化方法，设置生成的候选项的参数
    def __init__(
        self,
        num_output_tokens: int = 10,
        max_matching_ngram_size: int = 2,
    ):
        # 设置生成的候选项的参数
        self.num_output_tokens = num_output_tokens
        self.max_matching_ngram_size = max_matching_ngram_size

        # 如果最大匹配 ngram 大小小于等于 0 或输出的令牌数小于等于 0，则引发值错误异常
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
        # 获取输入序列的长度
        input_length = input_ids.size(1)

        # 初始化选择的候选序列和匹配标志
        chosen_ids = None
        match_found = False

        # 从最大匹配 ngram 大小开始向下遍历，直到 ngram 大小为 1
        for ngram_size in range(min(self.max_matching_ngram_size, input_length - 1), 0, -1):
            # 创建大小为 ngram_size 的滑动窗口
            windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

            # 将 ngram 转换为张量进行比较
            ngram_tensor = input_ids[0, -ngram_size:]

            # 找到窗口与 ngram 匹配的位置
            matches = (windows == ngram_tensor).all(dim=2)

            # 获取匹配的索引
            match_indices = matches.nonzero(as_tuple=True)[1]

            # 遍历匹配的索引以找到有效的延续
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + self.num_output_tokens
                end_idx = min(end_idx, input_length)

                # 如果起始索引小于结束索引，则选择作为候选的子序列
                if start_idx < end_idx:
                    chosen_ids = input_ids[0, start_idx:end_idx]
                    match_found = True
                    break
            if match_found:
                break

        # 如果未找到候选序列，则创建一个虚拟张量以避免错误
        if chosen_ids is None or len(chosen_ids) == 0:
            chosen_ids = torch.zeros((1), dtype=torch.long, device=input_ids.device)

        # 将输入序列扩展为包含选择的候选序列
        chosen_ids = chosen_ids.unsqueeze(0)
        candidate_input_ids = torch.cat((input_ids, chosen_ids), dim=1)
        # assisted_generation 需要 logits，但是这里没有，因此返回 None
        return candidate_input_ids, None
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
        # Currently does nothing
        # 当前函数暂时没有实现任何功能
        return
# 裁剪过去的键值对，使其长度不超过指定的最大长度
def _crop_past_key_values(model, past_key_values, maximum_length):
    """Crops the past key values up to a certain maximum length."""
    new_past = []
    # 如果模型是编码器-解码器结构
    if model.config.is_encoder_decoder:
        # 遍历过去的键值对
        for idx in range(len(past_key_values)):
            # 裁剪键值对的值，使其长度不超过最大长度
            new_past.append(
                (
                    past_key_values[idx][0][:, :, :maximum_length, :],
                    past_key_values[idx][1][:, :, :maximum_length, :],
                    past_key_values[idx][2],
                    past_key_values[idx][3],
                )
            )
        past_key_values = tuple(new_past)
    # 如果模型名称中包含"bloom"或者配置中的架构包含"bloom"
    elif "bloom" in model.__class__.__name__.lower() or (
        model.config.architectures is not None and "bloom" in model.config.architectures[0].lower()
    ):
        # 遍历过去的键值对
        for idx in range(len(past_key_values)):
            # 裁剪键值对的值，使其长度不超过最大长度
            new_past.append(
                (
                    past_key_values[idx][0][:, :, :maximum_length],
                    past_key_values[idx][1][:, :maximum_length, :],
                )
            )
        past_key_values = tuple(new_past)
    # 如果模型名称中包含"gptbigcode"或者配置中的架构包含"gptbigcode"
    elif "gptbigcode" in model.__class__.__name__.lower() or (
        model.config.architectures is not None and "gptbigcode" in model.config.architectures[0].lower()
    ):
        # 如果模型配置中包含多查询
        if model.config.multi_query:
            # 遍历过去的键值对，裁剪值使其长度不超过最大长度
            for idx in range(len(past_key_values)):
                past_key_values[idx] = past_key_values[idx][:, :maximum_length, :]
        else:
            # 遍历过去的键值对，裁剪值使其长度不超过最大长度
            for idx in range(len(past_key_values)):
                past_key_values[idx] = past_key_values[idx][:, :, :maximum_length, :]
    else:
        # 遍历过去的键值对
        for idx in range(len(past_key_values)):
            # 裁剪键值对的值，使其长度不超过最大长度
            new_past.append(
                (
                    past_key_values[idx][0][:, :, :maximum_length, :],
                    past_key_values[idx][1][:, :, :maximum_length, :],
                )
            )
        past_key_values = tuple(new_past)
    return past_key_values


# 准备用于解码目的的模型掩码，扩展或裁剪到指定长度
def _prepare_attention_mask(model_kwargs: Dict[str, Any], new_length: int, is_encoder_decoder: bool) -> Dict[str, Any]:
    """Expands or crops the model's mask for decoding purposes, to the defined length"""

    mask_key = "decoder_attention_mask" if is_encoder_decoder else "attention_mask"
    # 如果模型参数中不存在掩码键值对，则直接返回模型参数
    if mask_key not in model_kwargs:
        return model_kwargs

    mask = model_kwargs[mask_key]
    mask_length_diff = new_length - mask.shape[1]

    # 如果掩码长度差小于0，则裁剪掩码
    if mask_length_diff < 0:
        model_kwargs[mask_key] = mask[:, :mask_length_diff]
    # 如果掩码长度差大于0，则扩展掩码
    elif mask_length_diff > 0:
        model_kwargs[mask_key] = torch.cat([mask, mask.new_ones((mask.shape[0], mask_length_diff))], dim=-1)
    return model_kwargs


# 准备用于解码目的的模型的token_type_ids，扩展或裁剪到指定长度
def _prepare_token_type_ids(model_kwargs: Dict[str, Any], new_length: int) -> Dict[str, Any]:
    """Expands or crops the model's token_type_ids for decoding purposes, to the defined length"""
    # 如果模型参数中不存在token_type_ids键值对或者其值为None，则直接返回模型参数
    if "token_type_ids" not in model_kwargs or model_kwargs["token_type_ids"] is None:
        return model_kwargs
    # 获取模型参数中的 token_type_ids
    token_type_ids = model_kwargs["token_type_ids"]
    # 获取最后一个 token 的 token_type_id，并在最后一维度上增加一个维度
    final_token_type = token_type_ids[:, -1].unsqueeze(-1)
    # 计算新长度与原 token_type_ids 的长度差
    type_length_diff = new_length - token_type_ids.shape[1]

    # 如果新长度小于原长度
    if type_length_diff < 0:
        # 截取原 token_type_ids，保留前 type_length_diff 列
        token_type_ids = token_type_ids[:, :type_length_diff]
    # 如果新长度大于原长度
    elif type_length_diff > 0:
        # 创建重复的 token_type_id，使其长度与新长度一致
        token_type_copies = final_token_type.repeat(1, type_length_diff)
        # 将重复的 token_type_ids 拼接到原 token_type_ids 后面
        model_kwargs["token_type_ids"] = torch.cat([model_kwargs["token_type_ids"], token_type_copies], dim=-1)
    
    # 返回更新后的模型参数
    return model_kwargs
```