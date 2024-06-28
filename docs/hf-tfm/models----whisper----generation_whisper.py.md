# `.\models\whisper\generation_whisper.py`

```py
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy  # 导入copy模块，用于复制对象
import math  # 导入math模块，用于数学运算
import warnings  # 导入warnings模块，用于处理警告
import zlib  # 导入zlib模块，用于数据压缩
from typing import Callable, Iterator, List, Optional, Tuple, Union  # 导入类型提示相关模块

import numpy as np  # 导入NumPy库，用于科学计算
import torch  # 导入PyTorch库，用于深度学习
import torch.nn.functional as F  # 导入PyTorch的函数库，用于神经网络操作
from torch import nn  # 导入PyTorch的神经网络模块

from ...generation.configuration_utils import GenerationConfig  # 导入生成配置类
from ...generation.logits_process import (
    LogitsProcessorList,  # 导入处理logits的列表类
    SuppressTokensAtBeginLogitsProcessor,  # 导入处理开始位置token的logits处理器类
    SuppressTokensLogitsProcessor,  # 导入处理token的logits处理器类
    WhisperNoSpeechDetection,  # 导入无语音检测类
    WhisperTimeStampLogitsProcessor,  # 导入时间戳logits处理器类
)
from ...generation.stopping_criteria import StoppingCriteriaList  # 导入停止标准列表类
from ...modeling_outputs import BaseModelOutput  # 导入基础模型输出类
from ...utils import logging  # 导入日志记录工具
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE  # 导入任务ID和语言代码映射表


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def _median_filter(inputs: torch.Tensor, filter_width: int) -> torch.Tensor:
    """
    Applies a median filter of width `filter_width` along the last dimension of the input.

    The `inputs` tensor is assumed to be 3- or 4-dimensional.
    """
    if filter_width <= 0 or filter_width % 2 != 1:
        raise ValueError("`filter_width` should be an odd number")

    pad_width = filter_width // 2
    if inputs.shape[-1] <= pad_width:
        return inputs

    # Pad the left and right edges.
    inputs = nn.functional.pad(inputs, (pad_width, pad_width, 0, 0), mode="reflect")

    # sort() is faster than torch.median (https://github.com/pytorch/pytorch/issues/51450)
    result = inputs.unfold(-1, filter_width, 1).sort()[0][..., pad_width]
    return result


def _dynamic_time_warping(matrix: np.ndarray):
    """
    Measures similarity between two temporal sequences: the input audio and the output tokens. Used to generate
    token-level timestamps.
    """
    output_length, input_length = matrix.shape
    cost = np.ones((output_length + 1, input_length + 1), dtype=np.float32) * np.inf
    trace = -np.ones((output_length + 1, input_length + 1), dtype=np.float32)

    cost[0, 0] = 0
    for j in range(1, input_length + 1):
        for i in range(1, output_length + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = matrix[i - 1, j - 1] + c
            trace[i, j] = t

    # backtrace
    # 初始化变量 i 和 j，分别为跟踪矩阵的行数和列数的最大索引
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    
    # 将跟踪矩阵第一行所有元素设置为2
    trace[0, :] = 2
    # 将跟踪矩阵第一列所有元素设置为1
    trace[:, 0] = 1

    # 初始化两个空列表，用于存储路径的索引
    text_indices = []
    time_indices = []
    
    # 当 i 或 j 大于0时，进行循环
    while i > 0 or j > 0:
        # 将当前 i-1 添加到 text_indices 列表中
        text_indices.append(i - 1)
        # 将当前 j-1 添加到 time_indices 列表中
        time_indices.append(j - 1)
        
        # 根据跟踪矩阵中的值执行不同的操作
        if trace[i, j] == 0:
            # 如果跟踪矩阵值为0，向左上角移动
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            # 如果跟踪矩阵值为1，向上移动
            i -= 1
        elif trace[i, j] == 2:
            # 如果跟踪矩阵值为2，向左移动
            j -= 1
        else:
            # 如果跟踪矩阵中出现其他值，抛出运行时错误
            raise RuntimeError(
                f"Internal error in dynamic time warping. Unexpected trace[{i}, {j}]. Please file a bug report."
            )

    # 将列表转换为 numpy 数组，并反转顺序，然后返回结果
    text_indices = np.array(text_indices)[::-1]
    time_indices = np.array(time_indices)[::-1]
    return text_indices, time_indices
# 从 logits_processor 列表中获取指定类型的 logit_processor_class 实例的属性值
def _get_attr_from_logit_processors(logits_processor, logit_processor_class, attribute_name):
    # 遍历 logits_processor 列表，找到第一个 isinstance 的 logit_processor_class 类型实例
    logit_processor = next((cls for cls in logits_processor if isinstance(cls, logit_processor_class)), None)
    # 如果找到了对应的 logit_processor 实例，则返回其 attribute_name 属性的值，否则返回 None
    if logit_processor:
        return getattr(logit_processor, attribute_name, None)
    return None


# 将当前的分段序列填充到最大长度
def _pad_to_max_length(current_segments, pad_token_id, padding="right", bos_token_tensor=None, cut_off_length=None):
    # 初始化最大总长度和序列列表
    max_total_length = 0
    sequences = []

    # 检查填充方式是否合法，必须是 "right" 或者 "left"
    if padding not in ["right", "left"]:
        raise ValueError(f"`padding` must be either 'right' or 'left', not {padding}")

    # 遍历当前的分段序列列表
    for current_segment_list in current_segments:
        # 如果当前分段列表不为空且包含至少一个 tokens 字段的字典
        if current_segment_list is not None and len([d["tokens"] for d in current_segment_list]) > 0:
            # 合并当前分段列表中所有 tokens 字段的张量序列
            sequence = torch.cat([d["tokens"] for d in current_segment_list], dim=-1)

            # 如果指定了 cut_off_length，则截取序列后面的部分
            if cut_off_length is not None:
                sequence = sequence[-cut_off_length:]

            # 如果存在 bos_token_tensor，则将其作为起始 token 添加到序列开头
            if bos_token_tensor is not None:
                sequence = torch.cat([bos_token_tensor, sequence])

            # 将处理后的序列添加到 sequences 列表中
            sequences.append(sequence)
            # 更新最大总长度为当前序列长度和已记录的最大长度的较大值
            max_total_length = max(max_total_length, len(sequences[-1]))
        # 如果不存在当前分段列表，但存在 bos_token_tensor，则直接将其作为序列
        elif bos_token_tensor is not None:
            sequences.append(bos_token_tensor)
        # 否则，将一个空张量添加到序列中
        else:
            sequences.append(torch.tensor([]))

    # 遍历当前所有序列，对每个序列进行填充，使其长度与最大总长度相同
    for i in range(len(current_segments)):
        pad_length = max_total_length - len(sequences[i])
        pad = (0, pad_length) if padding == "right" else (pad_length, 0)
        sequences[i] = F.pad(sequences[i], pad=pad, value=pad_token_id)

    # 将填充后的序列堆叠成一个张量，并返回
    sequences = torch.stack(sequences, dim=0)
    return sequences


# WhisperGenerationMixin 类，用于生成处理
class WhisperGenerationMixin:
    # 生成函数，包含多个可选参数
    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: bool = False,
        return_timestamps: Optional[bool] = None,
        task: Optional[str] = None,
        language: Optional[str] = None,
        is_multilingual: Optional[bool] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_condition_type: Optional[str] = None,  # first-segment, all-segments
        condition_on_prev_tokens: Optional[bool] = None,
        temperature: Optional[Union[float, Tuple[float, ...]]] = None,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        num_segment_frames: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        time_precision: float = 0.02,
        return_token_timestamps: Optional[bool] = None,
        return_segments: bool = False,
        return_dict_in_generate: Optional[bool] = None,
        **kwargs,
    def generate_with_fallback(
        self,
        segment_input,
        decoder_input_ids,
        cur_bsz,
        batch_idx_map,
        seek,
        num_segment_frames,
        max_frames,
        temperatures,
        generation_config,
        logits_processor,
        stopping_criteria,
        prefix_allowed_tokens_fn,
        synced_gpus,
        return_token_timestamps,
        do_condition_on_prev_tokens,
        kwargs,
    ):
        # 生成文本输出，并在失败时返回备用选项
        # 使用给定的输入生成文本段落
        ...

    @staticmethod
    def _prepare_segments(prompt_ids, batch_size, generation_config):
        # 准备文本段落以供生成器使用
        # 如果指定了 prompt_ids 并且 generation_config 指定了 prompt_condition_type 为 "first-segment"
        if prompt_ids is not None and generation_config.prompt_condition_type == "first-segment":
            # 获取 prev_sot_token_id，如果存在的话
            prev_sot_token_id = getattr(generation_config, "prev_sot_token_id", None)
            # 如果 prompt_ids 的第一个 token 是 prev_sot_token_id，则去掉第一个 token
            prompt_ids = prompt_ids[1:] if prompt_ids[0] == prev_sot_token_id else prompt_ids
            # 将每个 batch 的当前段落设置为包含 prompt_ids 的 tokens
            current_segments = [[{"tokens": prompt_ids}] for _ in range(batch_size)]
        else:
            # 否则，将每个 batch 的当前段落设置为空列表
            current_segments = [[] for _ in range(batch_size)]

        return current_segments

    def _postprocess_outputs(self, seek_outputs, decoder_input_ids, return_token_timestamps, generation_config):
        # 后处理生成的输出
        # 如果 seek_outputs 是 torch.Tensor 类型，则截取未来的输出
        if isinstance(seek_outputs, torch.Tensor):
            seek_outputs = seek_outputs[:, decoder_input_ids.shape[-1] :]
            return seek_outputs, seek_outputs

        # 如果需要返回 token 时间戳并且 generation_config 有 alignment_heads 属性
        if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
            # 获取 num_frames 属性
            num_frames = getattr(generation_config, "num_frames", None)
            # 提取 token 时间戳并截取未来的输出
            seek_outputs["token_timestamps"] = self._extract_token_timestamps(
                seek_outputs, generation_config.alignment_heads, num_frames=num_frames
            )
            seek_outputs["token_timestamps"] = seek_outputs["token_timestamps"][:, decoder_input_ids.shape[-1] :]

        # 截取未来的输出序列
        seek_outputs["sequences"] = seek_outputs["sequences"][:, decoder_input_ids.shape[-1] :]

        def split_by_batch_index(values, key, batch_idx):
            # 根据 batch_idx 将值按照指定的 key 分割
            if key == "scores":
                return [v[batch_idx].cpu() for v in values]
            elif key == "past_key_values":
                # 不保存 past_key_values，因为这样做成本太高
                return None
            elif isinstance(values[batch_idx], tuple) and torch.is_tensor(values[batch_idx][0]):
                # 如果值是元组且第一个元素是张量，则按照 batch_idx 分割
                return tuple(tuple(w[batch_idx][None].cpu() for w in v) for v in values)
            return values[batch_idx].cpu()

        # 对每个 batch 分割 seek_outputs 中的值
        sequence_tokens = seek_outputs["sequences"]
        seek_outputs = [
            {k: split_by_batch_index(v, k, i) for k, v in seek_outputs.items()}
            for i in range(sequence_tokens.shape[0])
        ]

        return sequence_tokens, seek_outputs

    def _need_fallback(
        self,
        seek_sequence,
        seek_outputs,
        index,
        logits_processor,
        generation_config,
        vocab_size,
        temperature,
    ):
        # 判断是否需要回退到备选方案
        ...
        ):
            # 初始化需要回退和跳过标志
            needs_fallback = False
            should_skip = False

            # 如果设定了压缩比例阈值，则计算压缩比例
            if generation_config.compression_ratio_threshold is not None:
                compression_ratio = self._retrieve_compression_ratio(seek_sequence, vocab_size)

                # 如果压缩比例超过阈值，则需要回退
                if compression_ratio > generation_config.compression_ratio_threshold:
                    needs_fallback = True

            # 如果设定了对数概率阈值，则进行对数概率的检查
            if generation_config.logprob_threshold is not None:
                if "sequences_scores" in seek_outputs[0]:
                    logprobs = [s["sequences_scores"] for s in seek_outputs][index]
                else:
                    scores = seek_outputs[index]["scores"]
                    logprobs = self._retrieve_avg_logprobs(
                        scores, seek_sequence, generation_config.eos_token_id, temperature
                    )

                # 如果平均对数概率低于阈值，则需要回退
                if logprobs < generation_config.logprob_threshold:
                    needs_fallback = True

            # 如果设定了无语音概率阈值，则进行检查
            if generation_config.no_speech_threshold is not None:
                # 获取无语音概率
                no_speech_prob = _get_attr_from_logit_processors(
                    logits_processor, WhisperNoSpeechDetection, "no_speech_prob"
                )

                # 如果对数概率低于阈值且无语音概率高于阈值，则不需要回退但应跳过
                if (
                    logprobs < generation_config.logprob_threshold
                    and no_speech_prob[index] > generation_config.no_speech_threshold
                ):
                    needs_fallback = False
                    should_skip = True

            # 返回是否需要回退和是否应该跳过的标志
            return needs_fallback, should_skip

    @staticmethod
    def _setup_no_speech_detection(logits_processor, segment_input, decoder_input_ids, kwargs):
        # 从logits处理器中获取无语音检测器的设置输入方法，并将输入传递给它
        set_inputs = _get_attr_from_logit_processors(logits_processor, WhisperNoSpeechDetection, "set_inputs")
        extra_kwargs = {k: v for k, v in kwargs.items() if torch.is_tensor(v)}
        set_inputs({"inputs": segment_input, "decoder_input_ids": decoder_input_ids, **extra_kwargs})

    @staticmethod
    def _retrieve_total_input_frames(input_features, input_stride, kwargs):
        # 如果提供了输入特征，则返回输入帧的总数和每帧的特征维度
        if input_features is not None:
            return input_features.shape[0], input_features.shape[-1]

        # 如果提供了编码器输出，则根据输入步长计算总输入帧数
        if "encoder_outputs" in kwargs:
            encoder_outputs_shape = (
                kwargs["encoder_outputs"][0].shape
                if isinstance(kwargs["encoder_outputs"], BaseModelOutput)
                else kwargs["encoder_outputs"].shape
            )
            return encoder_outputs_shape[0], encoder_outputs_shape[1] * input_stride

        # 如果没有提供输入特征或编码器输出，则引发值错误异常
        raise ValueError("Make sure to provide either `input_features` or `encoder_outputs` to `generate`.")

    @staticmethod
    def _maybe_warn_unused_inputs(
        condition_on_prev_tokens,
        temperature,
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold,
        total_input_frames,
    ):
        # 警告消息的前缀，指示音频输入帧数不足，激活了短形式转录
        warning_prefix = (
            f"Audio input consists of only {total_input_frames}. "
            "Short-form transcription is activated."
            "{}, but will be ignored."
        )
        # 如果 condition_on_prev_tokens 不为 None，则记录警告信息
        if condition_on_prev_tokens is not None:
            logger.warn(warning_prefix.format(f"condition_on_prev_tokens is set to {condition_on_prev_tokens}"))

        # 如果 compression_ratio_threshold 不为 None，则记录警告信息
        if compression_ratio_threshold is not None:
            logger.warn(warning_prefix.format(f"compression_ratio_threshold is set to {compression_ratio_threshold}"))

        # 如果 logprob_threshold 不为 None，则记录警告信息
        if logprob_threshold is not None:
            logger.warn(warning_prefix.format(f"logprob_threshold is set to {logprob_threshold}"))

        # 如果 no_speech_threshold 不为 None，则记录警告信息
        if no_speech_threshold is not None:
            logger.warn(warning_prefix.format(f"no_speech_threshold is set to {no_speech_threshold}"))

        # 当 temperature 作为列表传递时，不能简单地忽略，需要抛出错误
        if isinstance(temperature, (list, tuple)):
            raise ValueError(
                f"Audio input consists of only {total_input_frames}. Short-form transcription is activated."
                f"temperature cannot be set to {temperature} which can only be used for temperature fallback for long-form generation. Make sure to set `temperature` to a float value or `None` for short-form generation."
            )

    @staticmethod
    def _set_return_outputs(
        return_dict_in_generate, return_token_timestamps, is_shortform, logprob_threshold, generation_config
    ):
        # 如果 return_dict_in_generate 为 None，则使用 generation_config 中的默认值
        if return_dict_in_generate is None:
            return_dict_in_generate = generation_config.return_dict_in_generate

        # 设置是否返回 token 的时间戳
        generation_config.return_token_timestamps = return_token_timestamps
        if return_token_timestamps:
            return_dict_in_generate = True
            generation_config.output_attentions = True
            generation_config.output_scores = True

        # 如果不是短形式生成并且 logprob_threshold 不为 None，则需要输出分数
        if not is_shortform and logprob_threshold is not None:
            return_dict_in_generate = True
            generation_config.output_scores = True

        # 更新 generation_config 中的返回字典设置
        generation_config.return_dict_in_generate = return_dict_in_generate
    # 定义一个静态方法 `_set_return_timestamps`，用于设置返回时间戳的配置
    def _set_return_timestamps(return_timestamps, is_shortform, generation_config):
        # 如果不是简化形式生成
        if not is_shortform:
            # 如果 return_timestamps 为 False，则抛出数值错误异常，提示输入的 mel 特征超过了3000 (> 30秒)，需要启用长格式生成，此时需要模型预测时间戳标记。
            raise ValueError(
                "You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which "
                "requires the model to predict timestamp tokens. Please either pass `return_timestamps=True` or make sure to pass no more than 3000 mel input features."
            )

            # 记录信息日志，设置 `return_timestamps=True` 以用于长格式生成。
            logger.info("Setting `return_timestamps=True` for long-form generation.")
            return_timestamps = True

        # 如果要返回时间戳，并且生成配置没有 `no_timestamps_token_id` 属性
        if return_timestamps and not hasattr(generation_config, "no_timestamps_token_id"):
            # 抛出数值错误异常，提示生成配置未正确设置以返回时间戳，建议初始化正确的生成配置。
            raise ValueError(
                "You are trying to return timestamps, but the generation config is not properly set. "
                "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
                "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
            )

        # 将 return_timestamps 设置到生成配置的属性中
        generation_config.return_timestamps = return_timestamps
    # 设置语言和任务到生成配置中，如果是多语言模型则更新配置
    def _set_language_and_task(language, task, is_multilingual, generation_config):
        # 如果提供了 is_multilingual 参数，则更新生成配置中的 is_multilingual 属性
        if is_multilingual is not None:
            if not hasattr(generation_config, "is_multilingual"):
                # 如果生成配置过时，抛出数值错误异常
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `is_multilingual` argument "
                    "to `generate`. Please update the generation config as per the instructions "
                    "https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.is_multilingual = is_multilingual

        # 如果生成配置中标记为非多语言模型，并且尝试指定任务或语言，抛出数值错误异常
        if hasattr(generation_config, "is_multilingual") and not generation_config.is_multilingual:
            if task is not None or language is not None:
                raise ValueError(
                    "Cannot specify `task` or `language` for an English-only model. If the model is intended to be "
                    "multilingual, pass `is_multilingual=True` to generate, or update the generation config."
                )

        # 如果指定了语言参数，则更新生成配置中的语言属性，确保语言名称小写化
        if language is not None:
            if not hasattr(generation_config, "lang_to_id"):
                # 如果生成配置过时，抛出数值错误异常
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `language` argument "
                    "to `generate`. Either set the language using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            language = language.lower()
            generation_config.language = language

        # 如果指定了任务参数，则更新生成配置中的任务属性
        if task is not None:
            if not hasattr(generation_config, "task_to_id"):
                # 如果生成配置过时，抛出数值错误异常
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `task` argument "
                    "to `generate`. Either set the task using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.task = task
    # 设置生成配置中的特定标记 ID，优先使用传入的关键字参数，否则使用配置中的默认值
    def _set_token_ids(generation_config, config, kwargs):
        # 从关键字参数中弹出结束标记的 ID
        eos_token_id = kwargs.pop("eos_token_id", None)
        # 从关键字参数中弹出解码器起始标记的 ID
        decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # 如果结束标记 ID 存在，则使用它；否则使用生成配置中的默认值
        eos_token_id = eos_token_id if eos_token_id is not None else generation_config.eos_token_id
        # 如果解码器起始标记 ID 存在，则使用它；否则使用生成配置中的默认值
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else generation_config.decoder_start_token_id
        )

        # 将确定的结束标记 ID 设置回生成配置中，如果不存在则使用全局配置中的默认值
        generation_config.eos_token_id = eos_token_id if eos_token_id is not None else config.eos_token_id
        # 将确定的解码器起始标记 ID 设置回生成配置中，如果不存在则使用全局配置中的默认值
        generation_config.decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else config.decoder_start_token_id
        )

    @staticmethod
    # 设置生成配置中的帧数及相关条件，根据传入的关键字参数或生成配置中的默认值
    def _set_num_frames(return_token_timestamps, generation_config, kwargs):
        # 如果需要返回标记级别的时间戳
        if return_token_timestamps:
            # 如果生成配置中的任务为“translate”，发出警告信息
            if getattr(generation_config, "task", None) == "translate":
                logger.warning("Token-level timestamps may not be reliable for task 'translate'.")
            # 如果生成配置中没有“alignment_heads”，抛出数值错误
            if not hasattr(generation_config, "alignment_heads"):
                raise ValueError(
                    "Model generation config has no `alignment_heads`, token-level timestamps not available. "
                    "See https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a on how to add this property to the generation config."
                )

            # 从关键字参数中弹出帧数，如果不存在则设为 None
            generation_config.num_frames = kwargs.pop("num_frames", None)

    @staticmethod
    # 设置生成配置中的阈值和条件，根据传入的参数或生成配置中的默认值
    def _set_thresholds_and_condition(
        generation_config,
        logprob_threshold,
        compression_ratio_threshold,
        no_speech_threshold,
        condition_on_prev_tokens,
    ):
        # 设置生成配置中的对数概率阈值，使用传入的参数或生成配置中的默认值
        generation_config.logprob_threshold = (
            logprob_threshold
            if logprob_threshold is not None
            else getattr(generation_config, "logprob_threshold", None)
        )
        # 设置生成配置中的压缩比阈值，使用传入的参数或生成配置中的默认值
        generation_config.compression_ratio_threshold = (
            compression_ratio_threshold
            if compression_ratio_threshold is not None
            else getattr(generation_config, "compression_ratio_threshold", None)
        )
        # 设置生成配置中的非语音阈值，使用传入的参数或生成配置中的默认值
        generation_config.no_speech_threshold = (
            no_speech_threshold
            if no_speech_threshold is not None
            else getattr(generation_config, "no_speech_threshold", None)
        )
        # 设置生成配置中的基于前一个标记的条件，使用传入的参数或生成配置中的默认值
        generation_config.condition_on_prev_tokens = (
            condition_on_prev_tokens
            if condition_on_prev_tokens is not None
            else getattr(generation_config, "condition_on_prev_tokens", None)
        )
    # 设置生成配置的提示条件类型
    def _set_prompt_condition_type(generation_config, prompt_condition_type):
        allowed_cond_types = ["first-segment", "all-segments"]

        # 默认使用 "first-segment" 作为提示条件类型，除非指定了其他值
        prompt_condition_type = prompt_condition_type or allowed_cond_types[0]

        # 检查所选的提示条件类型是否在允许的类型列表中
        if prompt_condition_type not in allowed_cond_types:
            raise ValueError(
                f"`prompt_condition_type={prompt_condition_type} does not exist. Make sure to set `prompt_condition_type` to one of {', '.join(allowed_cond_types)}"
            )

        # 如果选择了 "all-segments" 类型的条件，确保设置了 condition_on_prev_tokens=True
        if generation_config.condition_on_prev_tokens is not True and prompt_condition_type == "all-segments":
            raise ValueError(
                "Make sure to set `condition_on_prev_tokens=True` when setting `prompt_condition_type='all-segments'."
            )

        # 将生成配置中的提示条件类型设定为指定的值
        generation_config.prompt_condition_type = prompt_condition_type

    @staticmethod
    # 设置是否基于先前标记设置条件
    def _set_condition_on_prev_tokens(condition_on_prev_tokens, generation_config):
        # 如果未指定 condition_on_prev_tokens 的值，则使用生成配置中的默认值
        condition_on_prev_tokens = (
            condition_on_prev_tokens
            if condition_on_prev_tokens is not None
            else getattr(generation_config, "condition_on_prev_tokens", False)
        )
        # 将生成配置中的条件设置为所选的值
        generation_config.condition_on_prev_tokens = condition_on_prev_tokens

    @staticmethod
    # 获取最大帧数和起始位置
    def _retrieve_max_frames_and_seek(batch_size, attention_mask, total_input_frames):
        # 如果批量大小大于 1 且未提供注意力掩码，则抛出错误
        if batch_size > 1 and attention_mask is None:
            raise ValueError(
                "When doing batched long-form audio transcription, make sure to pass an `attention_mask`. You can retrieve the `attention_mask` by doing `processor(audio, ..., return_attention_mask=True)` "
            )
        # 如果批量大小大于 1，则计算每个样本的最大帧数，并设置初始位置为零
        elif batch_size > 1:
            max_frames = attention_mask.sum(-1).cpu().to(torch.long)
            seek = torch.zeros((batch_size,), dtype=torch.long)
        # 如果批量大小为 1，则所有输入都使用相同的最大帧数，并设置初始位置为零
        else:
            max_frames = torch.ones((1,), dtype=torch.long) * total_input_frames
            seek = torch.zeros((1,), dtype=torch.long)

        # 返回计算得到的最大帧数和起始位置
        return max_frames, seek
    # 静态方法：根据生成配置和处理器列表，获取日志概率处理器
    def _retrieve_logit_processors(self, generation_config, logits_processor, begin_index, is_shortform, num_beams):
        # 如果生成配置中设置了返回时间戳为真
        if generation_config.return_timestamps is True:
            # 创建时间戳日志概率处理器对象
            timestamp_processor = WhisperTimeStampLogitsProcessor(generation_config, begin_index=begin_index)
            # 将时间戳处理器添加到处理器列表中，如果处理器列表为空则创建新列表
            logits_processor = (
                [timestamp_processor] if logits_processor is None else [timestamp_processor] + logits_processor
            )

        # 如果生成配置中设置了需要抑制的标记
        if generation_config.suppress_tokens is not None:
            # 创建抑制标记日志概率处理器对象
            suppress_tokens_processor = SuppressTokensLogitsProcessor(generation_config.suppress_tokens)
            # 将抑制标记处理器添加到处理器列表中，如果处理器列表为空则创建新列表
            logits_processor = (
                [suppress_tokens_processor]
                if logits_processor is None
                else [suppress_tokens_processor] + logits_processor
            )
            # 将生成配置中的抑制标记设置为 None，避免重复处理
            generation_config.suppress_tokens = None

        # 如果生成配置中设置了需要在开始位置抑制的标记
        if generation_config.begin_suppress_tokens is not None:
            # 创建开始位置抑制标记日志概率处理器对象
            begin_suppress_processor = SuppressTokensAtBeginLogitsProcessor(
                generation_config.begin_suppress_tokens, begin_index=begin_index
            )
            # 将开始位置抑制标记处理器添加到处理器列表中，如果处理器列表为空则创建新列表
            logits_processor = (
                [begin_suppress_processor]
                if logits_processor is None
                else [begin_suppress_processor] + logits_processor
            )
            # 将生成配置中的开始位置抑制标记设置为 None，避免重复处理
            generation_config.begin_suppress_tokens = None

        # 如果生成配置中设置了无语音阈值，并且不是短表单模式
        if generation_config.no_speech_threshold is not None and not is_shortform:
            # 创建无语音检测对象
            no_speech_detector = WhisperNoSpeechDetection(
                no_speech_token=generation_config.no_timestamps_token_id - 1,
                begin_index=begin_index,
                scores_is_logprobs=num_beams > 1,
            )
            # 将无语音检测器添加到处理器列表中，如果处理器列表为空则创建新列表
            logits_processor = (
                [no_speech_detector] if logits_processor is None else [no_speech_detector] + logits_processor
            )
            # 将模型对象设置给无语音检测器
            no_speech_detector.set_model(self)

        # 返回处理器列表
        return logits_processor

    # 静态方法：可能减少批次的大小
    @staticmethod
    def _maybe_reduce_batch(input_features, seek, max_frames, cur_bsz, batch_idx_map):
        # 记录前一个批次的大小
        prev_bsz = cur_bsz
        # 新的批次索引映射列表
        new_batch_idx_map = []
        # 遍历每个批次中的样本
        for i in range(prev_bsz):
            # 获取原始批次索引
            prev_i = batch_idx_map[i]
            # 如果当前样本超过了其最大帧数
            if seek[prev_i] >= max_frames[prev_i]:
                # 计算要切除的索引
                cut_index = i + (cur_bsz - prev_bsz)
                # 减少当前批次大小
                cur_bsz -= 1
                # 从输入特征中删除超出帧数限制的样本
                input_features = torch.cat([input_features[:cut_index], input_features[cut_index + 1 :]], dim=0)
            else:
                # 保留不需要切除的索引
                new_batch_idx_map.append(prev_i)

        # 返回处理后的输入特征、当前批次大小和新的批次索引映射列表
        return input_features, cur_bsz, new_batch_idx_map
    # 定义一个静态方法，用于生成输入段落的数据
    def _get_input_segment(input_features, seek, seek_num_frames, num_segment_frames, cur_bsz, batch_idx_map):
        # 初始化一个空列表，用于存储每个批次样本的输入段落数据
        segment_input = []
        # 遍历当前批次中的每个样本
        for i in range(cur_bsz):
            # 获取当前样本在批次中的索引
            prev_i = batch_idx_map[i]
            # 从输入特征中切片出当前样本的输入段落数据
            segment_input_slice = input_features[i : i + 1, :, seek[prev_i] : seek[prev_i] + seek_num_frames[prev_i]]

            # 如果当前切片的最后一个维度长度小于期望的段落长度
            if segment_input_slice.shape[-1] < num_segment_frames:
                # 使用 PyTorch 的填充函数，在末尾维度上填充到期望的段落长度为 3000
                segment_input_slice = F.pad(
                    segment_input_slice, pad=(0, num_segment_frames - segment_input_slice.shape[-1])
                )

            # 将处理后的段落数据加入到段落输入列表中
            segment_input.append(segment_input_slice)

        # 将所有样本的段落数据在第一维度上连接起来
        segment_input = torch.cat(segment_input, dim=0)

        # 返回合并后的段落输入数据
        return segment_input

    # 定义一个静态方法，用于准备解码器的输入标识符
    @staticmethod
    def _prepare_decoder_input_ids(
        cur_bsz,
        init_tokens,
        current_segments,
        batch_idx_map,
        do_condition_on_prev_tokens,
        prompt_ids,
        generation_config,
        config,
        device,
        suppress_tokens,
        kwargs,
        ):
            # 计算每个样本的目标位置的最大长度的一半减一
            cut_off_length = config.max_target_positions // 2 - 1

            # 创建一个形状为 (当前批次大小, 1) 的张量，所有元素为 1，设备为指定的 device，数据类型为 long
            one_tensor = torch.ones((cur_bsz, 1), device=device, dtype=torch.long)
            # 根据初始化的标记将张量连接起来形成 decoder 的输入标记张量
            decoder_input_ids = torch.cat([t * one_tensor for t in init_tokens], dim=-1)

            # 获取前一个文本起始标记的 ID，如果没有指定则使用 suppress_tokens 中的倒数第二个元素
            prev_start_of_text = getattr(generation_config, "prev_sot_token_id", None)
            if prev_start_of_text is None:
                prev_start_of_text = suppress_tokens[-2] if suppress_tokens is not None else None

            # 如果任何一个 do_condition_on_prev_tokens 为真，并且当前段落长度大于 0
            active_segments = [current_segments[i] if do_condition_on_prev_tokens[i] else None for i in batch_idx_map]

            # 根据生成配置和提示条件类型选择前一个标记的 ID
            if prompt_ids is not None and generation_config.prompt_condition_type == "all-segments":
                prev_ids = prompt_ids
            else:
                prev_ids = prev_start_of_text * one_tensor[0] if prev_start_of_text is not None else None

            # 将前一个标记的 ID 填充到最大长度，以及应用截断长度和其他参数
            prev_tokens = _pad_to_max_length(
                active_segments,
                generation_config.pad_token_id,
                padding="left",
                bos_token_tensor=prev_ids,
                cut_off_length=cut_off_length,
            )
            # 将填充后的前一个标记张量和 decoder 输入标记张量连接起来
            decoder_input_ids = torch.cat([prev_tokens, decoder_input_ids], dim=-1)

            # 设置 decoder_attention_mask，排除填充标记
            kwargs["decoder_attention_mask"] = decoder_input_ids != generation_config.pad_token_id
        elif prompt_ids is not None:
            # 将提示标记张量重复批次大小次数，并与 decoder 输入标记张量连接起来
            prev_tokens = prompt_ids[None].repeat(decoder_input_ids.shape[0], 1)
            decoder_input_ids = torch.cat([prev_tokens, decoder_input_ids], dim=-1)
            # 确保不将 `"decoder_attention_mask"` 传递给前向计算
            kwargs.pop("decoder_attention_mask", None)
        else:
            # 确保不将 `"decoder_attention_mask"` 传递给前向计算
            kwargs.pop("decoder_attention_mask", None)

        # 返回 decoder 输入标记张量和 kwargs 参数
        return decoder_input_ids, kwargs
    def _set_max_new_tokens_and_length(config, decoder_input_ids, generation_config, kwargs):
        # 计算初始令牌数量，限制在最大目标位置的一半减一和解码器输入长度减一之间
        num_initial_tokens = min(config.max_target_positions // 2 - 1, decoder_input_ids.shape[-1] - 1)

        # 从kwargs中弹出'max_length'参数，并赋给passed_max_length变量
        passed_max_length = kwargs.pop("max_length", None)
        # 从kwargs中弹出'max_new_tokens'参数，并赋给passed_max_new_tokens变量
        passed_max_new_tokens = kwargs.pop("max_new_tokens", None)
        # 从生成配置(generation_config)中获取'max_length'属性，并赋给max_length_config变量
        max_length_config = getattr(generation_config, "max_length", None)
        # 从生成配置(generation_config)中获取'max_new_tokens'属性，并赋给max_new_tokens_config变量
        max_new_tokens_config = getattr(generation_config, "max_new_tokens", None)

        # 初始化max_new_tokens和max_length变量为None
        max_new_tokens = None
        max_length = None

        # 确保不超过'max_length'设定的最大值
        if passed_max_length is not None and passed_max_new_tokens is None:
            # 根据条件增加max_length，以确保不超过config.max_target_positions
            max_length = min(passed_max_length + num_initial_tokens, config.max_target_positions)
            logger.info(
                f"Increase max_length from {passed_max_length} to {max_length} since input is conditioned on previous segment."
            )
        elif max_length_config is not None and passed_max_new_tokens is None and max_new_tokens_config is None:
            # 根据条件增加max_length，以确保不超过config.max_target_positions
            max_length = min(generation_config.max_length + num_initial_tokens, config.max_target_positions)
            logger.info(
                f"Increase max_length from {max_length_config} to {max_length} since input is conditioned on previous segment."
            )
        elif (
            passed_max_new_tokens is not None
            and passed_max_new_tokens + decoder_input_ids.shape[-1] > config.max_target_positions
        ):
            # 计算最大新令牌数，以确保不超过config.max_target_positions
            max_new_tokens = config.max_target_positions - decoder_input_ids.shape[-1]
        elif (
            passed_max_new_tokens is None
            and max_new_tokens_config is not None
            and max_new_tokens_config + decoder_input_ids.shape[-1] > config.max_target_positions
        ):
            # 计算最大新令牌数，以确保不超过config.max_target_positions
            max_new_tokens = config.max_target_positions - decoder_input_ids.shape[-1]

        # 如果max_new_tokens不为None，则将其设置回kwargs中
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens

        # 如果max_length不为None，则将其设置回kwargs中
        if max_length is not None:
            kwargs["max_length"] = max_length

        # 返回更新后的kwargs
        return kwargs
    # 计算平均对数概率的函数，用于生成模型的输出评分
    def _retrieve_avg_logprobs(scores, tokens, eos_token_id, temperature):
        # 根据温度参数重新缩放评分，如果温度为非正数则默认为1
        rescale_temperature = temperature if temperature > 0.0 else 1
        # 将所有评分堆叠成一个张量，并放置在与 tokens 相同的设备上
        scores = torch.stack(scores).to(tokens.device)

        # 如果评分张量的长度大于 tokens 张量的长度，则截断评分张量
        if scores.shape[0] > tokens.shape[0]:
            scores = scores[: tokens.shape[0]]
        else:
            # 否则截断 tokens 张量，以匹配评分张量的长度
            tokens = tokens[-scores.shape[0] :]

        # 对缩放后的评分应用对数 softmax 函数，计算对数概率
        logprobs = F.log_softmax((scores * rescale_temperature).float(), dim=-1).to(scores.dtype)

        # 计算所选 tokens 的对数概率并求和
        sum_logprobs = sum((logprobs[i][tokens[i]] * (tokens[i] != eos_token_id)) for i in range(logprobs.shape[0]))

        # 如果 eos_token_id 不为空，则计算 tokens 中非 eos_token_id 的长度；否则使用 tokens 的总长度
        length = (tokens != eos_token_id).sum(-1) if eos_token_id is not None else tokens.shape[0]

        # 计算平均对数概率，考虑到序列长度加一的影响
        avg_logprobs = sum_logprobs / (length + 1)
        return avg_logprobs

    @staticmethod
    def _retrieve_segment(
        seek_sequence,
        seek_outputs,
        time_offset,
        timestamp_begin,
        seek_num_frames,
        time_precision,
        input_stride,
        prev_idx,
        idx,
        return_token_timestamps,
```