# `.\transformers\pipelines\automatic_speech_recognition.py`

```py
# 导入必要的模块和类型
from collections import defaultdict  # 导入 defaultdict 数据结构
from typing import TYPE_CHECKING, Dict, Optional, Union  # 导入类型提示相关的模块

import numpy as np  # 导入 numpy 库
import requests  # 导入 requests 库

# 导入 Hugging Face 库中的模块和工具函数
from ..tokenization_utils import PreTrainedTokenizer  # 导入预训练分词器
from ..utils import is_torch_available, is_torchaudio_available, logging  # 导入工具函数和日志记录模块
from .audio_utils import ffmpeg_read  # 导入音频处理相关的工具函数
from .base import ChunkPipeline  # 导入音频处理的基础类

# 如果是类型检查阶段，则进一步导入类型相关的模块
if TYPE_CHECKING:
    from pyctcdecode import BeamSearchDecoderCTC  # 导入 CTC 解码器

    from ..feature_extraction_sequence_utils import SequenceFeatureExtractor  # 导入序列特征提取器
    from ..modeling_utils import PreTrainedModel  # 导入预训练模型

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 如果 Torch 可用，则进一步导入相关模块
if is_torch_available():
    import torch  # 导入 PyTorch 库

    from ..models.auto.modeling_auto import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES  # 导入模型映射字典

# 定义一个函数，用于重新缩放音频片段的步长值
def rescale_stride(stride, ratio):
    """
    Rescales the stride values from audio space to tokens/logits space.

    (160_000, 16_000, 16_000) -> (2000, 200, 200) for instance.
    """
    # 初始化一个空列表，用于存储重新缩放后的步长值
    new_strides = []
    # 遍历输入的步长值
    for input_n, left, right in stride:
        # 计算新的 token 数目，将输入步长值按比例缩放
        token_n = int(round(input_n * ratio))
        # 根据缩放后的 token 数目重新计算左侧填充值
        left = int(round(left / input_n * token_n))
        # 根据缩放后的 token 数目重新计算右侧填充值
        right = int(round(right / input_n * token_n))
        # 将重新计算得到的步长值组成元组，添加到新的步长列表中
        new_stride = (token_n, left, right)
        new_strides.append(new_stride)

    return new_strides

# 定义一个迭代器函数，用于生成音频片段的迭代器
def chunk_iter(inputs, feature_extractor, chunk_len, stride_left, stride_right, rescale=True, dtype=None):
    # 获取输入音频的长度
    inputs_len = inputs.shape[0]
    # 计算每次迭代的步长
    step = chunk_len - stride_left - stride_right
    # 遍历输入数据，以步长 step 分割
    for chunk_start_idx in range(0, inputs_len, step):
        # 计算每个数据块的起始索引和结束索引
        chunk_end_idx = chunk_start_idx + chunk_len
        # 截取输入数据的数据块
        chunk = inputs[chunk_start_idx:chunk_end_idx]
        # 使用特征提取器处理数据块，返回张量形式的结果
        processed = feature_extractor(chunk, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
        # 如果指定了数据类型，则将处理结果转换为指定类型的数据
        if dtype is not None:
            processed = processed.to(dtype=dtype)
        # 如果数据块是第一个块，则左侧填充为0，否则使用指定的左侧填充步长
        _stride_left = 0 if chunk_start_idx == 0 else stride_left
        # 检查是否是最后一个数据块
        is_last = chunk_end_idx > inputs_len if stride_right > 0 else chunk_end_idx >= inputs_len
        # 如果是最后一个数据块，则右侧填充为0，否则使用指定的右侧填充步长
        _stride_right = 0 if is_last else stride_right

        # 更新数据块长度和填充步长
        chunk_len = chunk.shape[0]
        stride = (chunk_len, _stride_left, _stride_right)

        # 如果处理结果中包含"input_features"字段，则获取其长度
        if "input_features" in processed:
            processed_len = processed["input_features"].shape[-1]
        # 否则，如果处理结果中包含"input_values"字段，则获取其长度
        elif "input_values" in processed:
            processed_len = processed["input_values"].shape[-1]

        # 如果处理结果的长度不等于数据块长度且需要重新缩放，则重新计算填充步长
        if processed_len != chunk.shape[-1] and rescale:
            ratio = processed_len / chunk_len
            stride = rescale_stride([stride], ratio)[0]

        # 如果数据块长度大于左侧填充步长，则生成当前数据块的信息
        if chunk.shape[0] > _stride_left:
            yield {"is_last": is_last, "stride": stride, **processed}

        # 如果是最后一个数据块，则结束循环
        if is_last:
            break
# 定义一个函数，用于在两个序列中查找最长公共子序列
def _fast_find_longest_common_sequence(sequence_left, sequence_right):
    # 获取左序列和右序列的长度
    seq_len_left = len(sequence_left)
    seq_len_right = len(sequence_right)
    # 创建一个二维列表，用于记录最长公共子序列的长度
    counter = [[0] * (seq_len_right + 1) for _ in range(seq_len_left + 1)]
    # 初始化最长公共子序列的长度
    longest = 0
    # 遍历左序列
    for i in range(seq_len_left):
        # 遍历右序列
        for j in range(seq_len_right):
            # 如果左序列的元素等于右序列的元素
            if sequence_left[i] == sequence_right[j]:
                # 获取前一个计数的值
                previous_counter = counter[i][j] + 1
                # 更新计数矩阵的值
                counter[i + 1][j + 1] = previous_counter
                # 更新最长公共子序列的长度
                if previous_counter > longest:
                    longest = previous_counter

    # 将计数矩阵转换为 NumPy 数组
    counter = np.array(counter)
    # 获取最长公共子序列在左序列中的起始索引
    index_left = np.argwhere(counter == longest)[-1][0] - longest if longest != 0 else -1
    # 获取最长公共子序列在右序列中的起始索引
    index_right = np.argwhere(counter == longest)[-1][1] - longest if longest != 0 else -1
    # 返回最长公共子序列在左序列和右序列中的起始索引以及最长公共子序列的长度
    return index_left, index_right, longest


# 定义一个函数，用于在多个序列中查找最长公共子序列
def _find_longest_common_sequence(sequences, tokenizer):
    """
    TODO  Use a faster algorithm this can probably be done in O(n)
    using suffix array.
    It might be tedious to do because of fault tolerance.
    We actually have a really good property which is that the total sequence
    MUST be those subsequences in order.
    Also the algorithm should be more tolerant to errors.
    """
    # 获取第一个序列的非特殊标记的令牌 ID
    sequence = [tok_id for tok_id in sequences[0][0].tolist() if tok_id not in tokenizer.all_special_ids]
    # 遍历剩余的序列
    for new_seq in sequences[1:]:
        # 获取当前序列的非特殊标记的令牌 ID
        new_sequence = [tok_id for tok_id in new_seq[0].tolist() if tok_id not in tokenizer.all_special_ids]
        
        # 初始化索引和最大匹配度
        index = 0
        max_ = 0.0
        # 遍历当前序列中的每个令牌
        for i in range(1, len(new_sequence) + 1):
            # epsilon 用于提高较长的完全匹配的权重
            eps = i / 10000.0
            # 计算当前子序列和前一序列尾部的匹配数量
            matches = np.sum(np.array(sequence[-i:]) == np.array(new_sequence[:i]))
            # 计算匹配度
            matching = matches / i + eps
            # 如果匹配数量大于1且匹配度大于最大匹配度
            if matches > 1 and matching > max_:
                index = i
                max_ = matching
        # 将当前序列中与前一序列匹配的部分添加到总的序列中
        sequence.extend(new_sequence[index:])
    # 将总的序列转换为 NumPy 数组并返回
    return np.array(sequence)


# 定义一个自动语音识别的流水线类
class AutomaticSpeechRecognitionPipeline(ChunkPipeline):
    """
    Pipeline that aims at extracting spoken text contained within some audio.

    The input can be either a raw waveform or a audio file. In case of the audio file, ffmpeg should be installed for
    to support multiple audio formats

    Example:

    ```python
    >>> from transformers import pipeline

    >>> transcriber = pipeline(model="openai/whisper-base")
    >>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
    {'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}
    ```py
    """
    pass
    # 定义了一个名为`forward`的函数，用于实现推理过程
    def forward(
        self,
        model,
        feature_extractor,
        tokenizer,
        decoder,
        inputs,
        attention_mask=None,
        speaker_label=None,
        device=None,
        decoder_start_token_id=None,
    ):
        # 如果没有传入`device`参数，则将`device`设置为`model`的设备
        device = device or next(model.parameters()).device
        # 通过`feature_extractor`将输入的音频数据编码为特征
        features = feature_extractor(inputs, return_tensors="pt").input_values.to(device)
        # 通过`model`对编码后的特征进行前向传播，得到预测结果
        logits = model.features(features).logits
        # 如果`decoder`不为空，使用`decoder`对`logits`进行解码，得到最终文本结果
        if decoder is not None:
            input_lengths = torch.tensor([features.shape[-1]], device=device)
            decoded = decoder.decode(logits.permute(1, 0, 2), input_lengths)
            # 如果指定了解码起始符的标识符，则在解码结果中添加起始符
            if decoder_start_token_id is not None:
                decoded = [
                    tokenizer.decode(  # 使用`tokenizer`将解码结果转换为文本
                        torch.cat([torch.tensor([decoder_start_token_id, device=device]), d])
                    ).strip()
                    for d in decoded
                ]
        else:
            # 如果`decoder`为空，则将`logits`通过`argmax`函数取得预测结果
            predicted_ids = torch.argmax(logits, dim=-1)
            # 使用`tokenizer`将预测结果转换为文本
            decoded = [
                tokenizer.decode(ids, skip_special_tokens=True).strip()
                for ids in predicted_ids
            ]
        # 返回解码的结果
        return decoded
    # 初始化方法，用于创建一个新的对象
    def __init__(
        self,
        # 模型参数，接受一个 PreTrainedModel 类型的参数
        model: "PreTrainedModel",
        # 特征提取器参数，可以是 SequenceFeatureExtractor 类型或字符串，可选，默认为 None
        feature_extractor: Union["SequenceFeatureExtractor", str] = None,
        # 分词器参数，可选，默认为 None
        tokenizer: Optional[PreTrainedTokenizer] = None,
        # 解码器参数，可以是 BeamSearchDecoderCTC 类型或字符串，可选，默认为 None
        decoder: Optional[Union["BeamSearchDecoderCTC", str]] = None,
        # 设备参数，可以是整数或 torch.device 类型，可选，默认为 None
        device: Union[int, "torch.device"] = None,
        # Torch 数据类型参数，可以是字符串或 torch.dtype 类型，可选，默认为 None
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        # 其他关键字参数
        **kwargs,
    ):
        # 设置模型类型，以便检查是否具有正确的预处理和后处理参数
        # 如果模型配置的模型类型是 "whisper"
        if model.config.model_type == "whisper":
            # 设置对象类型为 "seq2seq_whisper"
            self.type = "seq2seq_whisper"
        # 如果模型类名在 MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES 的值中
        elif model.__class__.__name__ in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES.values():
            # 设置对象类型为 "seq2seq"
            self.type = "seq2seq"
        # 如果特征提取器的处理器类存在且以 "WithLM" 结尾，并且解码器不是 None
        elif (
            feature_extractor._processor_class
            and feature_extractor._processor_class.endswith("WithLM")
            and decoder is not None
        ):
            # 设置对象类型为 "ctc_with_lm"
            self.type = "ctc_with_lm"
            # 设置解码器
            self.decoder = decoder
        # 否则
        else:
            # 设置对象类型为 "ctc"
            self.type = "ctc"

        # 调用父类的初始化方法
        super().__init__(model, tokenizer, feature_extractor, device=device, torch_dtype=torch_dtype, **kwargs)

    # 调用方法，用于执行对象的功能
    def __call__(
        self,
        # 输入参数，可以是 numpy 数组、字节串或字符串
        inputs: Union[np.ndarray, bytes, str],
        # 其他关键字参数
        **kwargs,
    ):
    
    # 参数清理方法，用于清理输入参数
    def _sanitize_parameters(
        self,
        # 分段长度参数，可选
        chunk_length_s=None,
        # 步长参数，可选
        stride_length_s=None,
        # 忽略警告参数，可选
        ignore_warning=None,
        # 解码器关键字参数，可选
        decoder_kwargs=None,
        # 返回时间戳参数，可选
        return_timestamps=None,
        # 返回语言参数，可选
        return_language=None,
        # 生成关键字参数，可选
        generate_kwargs=None,
        # 最大新标记数参数，可选
        max_new_tokens=None,
    ):
    
    # 后处理方法，用于处理模型输出结果
    def postprocess(
        self, model_outputs, decoder_kwargs: Optional[Dict] = None, return_timestamps=None, return_language=None
```   
# 寻找时间戳序列的函数，合并每个序列的结尾和下一个序列的开头，由于`WhisperForConditionalGeneration`生成的时间戳是成对的，我们过滤连续的时间戳并只迭代它们。
# 我们跟踪`time`，表示正在处理的数据块的实际起始时间。我们需要确保将时间戳标记偏移量设置为`time`，以便于分词器正确计算最终的`offset`。
def _find_timestamp_sequence(sequences, tokenizer, feature_extractor, max_source_positions):
    # 第一个时间戳标记的索引
    timestamp_begin = tokenizer.convert_tokens_to_ids("<|notimestamps|>") + 1
    items = []
    # token与时间的近似比例：大约0.2秒
    time_precision = feature_extractor.chunk_length / max_source_positions
    time = 0
    result = []
    # 遍历items的长度
    for i in range(len(items)):
        result += items[i].tolist()
    # 返回结果列表
    return result
```