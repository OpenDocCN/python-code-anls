# `.\pipelines\automatic_speech_recognition.py`

```py
# 引入从 collections 模块中导入 defaultdict 类
from collections import defaultdict
# 从 typing 模块中导入 TYPE_CHECKING, Dict, Optional, Union 等类型
from typing import TYPE_CHECKING, Dict, Optional, Union

# 导入 numpy 库，用于处理数组和矩阵等数据
import numpy as np
# 导入 requests 库，用于发送 HTTP 请求
import requests

# 从 tokenization_utils 模块中导入 PreTrainedTokenizer 类
from ..tokenization_utils import PreTrainedTokenizer
# 从 utils 模块中导入 is_torch_available, is_torchaudio_available, logging 等函数和类
from ..utils import is_torch_available, is_torchaudio_available, logging
# 从 audio_utils 模块中导入 ffmpeg_read 函数
from .audio_utils import ffmpeg_read
# 从 base 模块中导入 ChunkPipeline 类
from .base import ChunkPipeline

# 如果 TYPE_CHECKING 为真，则从 pyctcdecode 模块中导入 BeamSearchDecoderCTC 类
if TYPE_CHECKING:
    from pyctcdecode import BeamSearchDecoderCTC
    # 从 feature_extraction_sequence_utils 模块中导入 SequenceFeatureExtractor 类
    from ..feature_extraction_sequence_utils import SequenceFeatureExtractor
    # 从 modeling_utils 模块中导入 PreTrainedModel 类
    from ..modeling_utils import PreTrainedModel

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 如果 torch 可用，则从 models.auto.modeling_auto 模块中导入 MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES 常量
if is_torch_available():
    import torch
    from ..models.auto.modeling_auto import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES


def rescale_stride(stride, ratio):
    """
    Rescales the stride values from audio space to tokens/logits space.

    (160_000, 16_000, 16_000) -> (2000, 200, 200) for instance.
    """
    # 创建一个空列表用于存放重新缩放后的步幅值
    new_strides = []
    # 遍历输入的每一个步幅值 (input_n, left, right)
    for input_n, left, right in stride:
        # 计算 token_n，将输入空间的步幅值按比例缩放到 tokens/logits 空间
        token_n = int(round(input_n * ratio))
        # 计算左侧步幅 left 在 tokens/logits 空间的值
        left = int(round(left / input_n * token_n))
        # 计算右侧步幅 right 在 tokens/logits 空间的值
        right = int(round(right / input_n * token_n))
        # 将缩放后的步幅值组成元组，并添加到新步幅列表中
        new_stride = (token_n, left, right)
        new_strides.append(new_stride)

    return new_strides


def chunk_iter(inputs, feature_extractor, chunk_len, stride_left, stride_right, dtype=None):
    """
    Iterates over chunks of input data, yielding processed chunks.

    inputs: numpy array, the input data to be chunked
    feature_extractor: SequenceFeatureExtractor, object for extracting features from chunks
    chunk_len: int, length of each chunk
    stride_left: int, left stride length
    stride_right: int, right stride length
    dtype: optional, data type to convert processed chunks

    Yields dictionaries containing processed chunk data and metadata.
    """
    # 获取输入数据的长度
    inputs_len = inputs.shape[0]
    # 计算每次迭代的步长，chunk_len - stride_left - stride_right
    step = chunk_len - stride_left - stride_right
    # 从输入数据的起始位置开始，以步长逐步迭代
    for chunk_start_idx in range(0, inputs_len, step):
        # 计算当前 chunk 的结束索引
        chunk_end_idx = chunk_start_idx + chunk_len
        # 从输入数据中提取当前 chunk
        chunk = inputs[chunk_start_idx:chunk_end_idx]
        # 使用特征提取器从当前 chunk 提取特征，返回处理后的结果
        processed = feature_extractor(chunk, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
        # 如果指定了数据类型 dtype，则将处理后的结果转换为指定类型
        if dtype is not None:
            processed = processed.to(dtype=dtype)
        # 如果 chunk 的起始索引是 0，则左侧步幅为 0；否则与指定的左侧步幅相同
        _stride_left = 0 if chunk_start_idx == 0 else stride_left
        # 如果 chunk 的结束索引超过输入数据长度且右侧步幅大于 0，则说明是最后一个 item
        is_last = chunk_end_idx > inputs_len if stride_right > 0 else chunk_end_idx >= inputs_len
        # 如果是最后一个 item，则右侧步幅为 0；否则与指定的右侧步幅相同
        _stride_right = 0 if is_last else stride_right

        # 记录当前 chunk 的长度
        chunk_len = chunk.shape[0]
        # 创建步幅元组
        stride = (chunk_len, _stride_left, _stride_right)
        # 如果当前 chunk 的长度大于左侧步幅，生成包含处理结果和元数据的字典，并返回
        if chunk.shape[0] > _stride_left:
            yield {"is_last": is_last, "stride": stride, **processed}
        # 如果是最后一个 item，则停止迭代
        if is_last:
            break
def _fast_find_longest_common_sequence(sequence_left, sequence_right):
    # 获取左序列和右序列的长度
    seq_len_left = len(sequence_left)
    seq_len_right = len(sequence_right)
    # 初始化一个二维列表作为计数器，用于记录最长公共子序列的长度
    counter = [[0] * (seq_len_right + 1) for _ in range(seq_len_left + 1)]
    longest = 0
    # 遍历左序列和右序列，填充计数器
    for i in range(seq_len_left):
        for j in range(seq_len_right):
            # 如果左序列和右序列当前位置的元素相同
            if sequence_left[i] == sequence_right[j]:
                previous_counter = counter[i][j] + 1
                counter[i + 1][j + 1] = previous_counter
                # 更新最长公共子序列的长度
                if previous_counter > longest:
                    longest = previous_counter

    # 转换计数器为NumPy数组
    counter = np.array(counter)
    # 找到最长公共子序列在左序列和右序列中的起始索引和长度
    index_left = np.argwhere(counter == longest)[-1][0] - longest if longest != 0 else -1
    index_right = np.argwhere(counter == longest)[-1][1] - longest if longest != 0 else -1
    return index_left, index_right, longest


def _find_longest_common_sequence(sequences, tokenizer):
    # TODO  使用更快的算法，可能可以在O(n)时间内完成，使用后缀数组
    # 这可能因为容错性而变得繁琐
    # 我们实际上有一个非常好的性质，即总序列必须按顺序是这些子序列
    # 此外，该算法应该对错误更加容忍
    # 从第一个序列中提取不包含特殊标识符的 token ID 组成的列表
    sequence = [tok_id for tok_id in sequences[0][0].tolist() if tok_id not in tokenizer.all_special_ids]
    # 遍历其余的序列
    for new_seq in sequences[1:]:
        # 从每个序列中提取不包含特殊标识符的 token ID 组成的列表
        new_sequence = [tok_id for tok_id in new_seq[0].tolist() if tok_id not in tokenizer.all_special_ids]

        index = 0
        max_ = 0.0
        # 遍历新序列，计算最长公共子序列的相关指标
        for i in range(1, len(new_sequence) + 1):
            # epsilon 用于偏爱长完全匹配
            eps = i / 10000.0
            # 计算匹配的数量和匹配度
            matches = np.sum(np.array(sequence[-i:]) == np.array(new_sequence[:i]))
            matching = matches / i + eps
            # 如果匹配数大于1且匹配度大于当前最大值，则更新最大值
            if matches > 1 and matching > max_:
                index = i
                max_ = matching
        # 将新序列中从最佳匹配点开始的部分扩展到主序列中
        sequence.extend(new_sequence[index:])
    return np.array(sequence)


class AutomaticSpeechRecognitionPipeline(ChunkPipeline):
    """
    Pipeline that aims at extracting spoken text contained within some audio.

    The input can be either a raw waveform or a audio file. In case of the audio file, ffmpeg should be installed for
    to support multiple audio formats

    Example:

    ```
    >>> from transformers import pipeline

    >>> transcriber = pipeline(model="openai/whisper-base")
    >>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
    {'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)
    """
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            模型将用于通过管道进行预测。必须是继承自[`PreTrainedModel`]（PyTorch）或[`TFPreTrainedModel`]（TensorFlow）的模型。
        feature_extractor ([`SequenceFeatureExtractor`]):
            特征提取器将用于对波形进行编码，以供模型使用。
        tokenizer ([`PreTrainedTokenizer`]):
            分词器将用于对数据进行编码，以供模型使用。此对象继承自[`PreTrainedTokenizer`]。
        decoder (`pyctcdecode.BeamSearchDecoderCTC`, *optional*):
            可选参数，用于语言模型增强解码的PyCTCDecode的BeamSearchDecoderCTC。详见[`Wav2Vec2ProcessorWithLM`]获取更多信息。
        chunk_length_s (`float`, *optional*, defaults to 0):
            每个分块的输入长度（秒）。如果`chunk_length_s = 0`，则禁用分块（默认）。
    
            <Tip>
    
            有关如何有效使用`chunk_length_s`的更多信息，请查看ASR分块博文。
    
            </Tip>
    
        stride_length_s (`float`, *optional*, defaults to `chunk_length_s / 6`):
            每个分块左右的步幅长度。仅在`chunk_length_s > 0`时使用。这使得模型能够查看更多的上下文，并更好地推断字母，但管道会丢弃最后的步幅位，以尽可能完美地重构最终结果。
    
            <Tip>
    
            有关如何有效使用`stride_length_s`的更多信息，请查看ASR分块博文。
    
            </Tip>
    
        framework (`str`, *optional*):
            要使用的框架，可以是`"pt"`表示PyTorch或`"tf"`表示TensorFlow。必须安装指定的框架。如果未指定框架，默认使用当前安装的框架。如果未指定框架且两个框架都安装，则默认使用模型的框架，或者如果没有提供模型，则默认使用PyTorch。
        device (Union[`int`, `torch.device`], *optional*):
            CPU/GPU设备编号。将其设置为`None`将使用CPU，设置为正整数将在关联的CUDA设备ID上运行模型。
        torch_dtype (Union[`int`, `torch.dtype`], *optional*):
            计算的数据类型（dtype）。将其设置为`None`将使用float32精度。设置为`torch.float16`或`torch.bfloat16`将使用相应的半精度dtype。
    # 初始化方法，接受多个参数来配置模型和处理过程
    def __init__(
        self,
        model: "PreTrainedModel",  # 模型参数，预训练模型对象
        feature_extractor: Union["SequenceFeatureExtractor", str] = None,  # 特征提取器，可以是对象或者字符串
        tokenizer: Optional[PreTrainedTokenizer] = None,  # 分词器，可选的预训练分词器对象
        decoder: Optional[Union["BeamSearchDecoderCTC", str]] = None,  # 解码器，可以是解码器对象或者字符串
        device: Union[int, "torch.device"] = None,  # 设备参数，可以是整数或者 torch 设备对象
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,  # torch 数据类型，可选的字符串或者 torch 数据类型对象
        **kwargs,  # 其他关键字参数
    ):
        # 设置模型类型，以便检查预处理和后处理参数是否正确
        if model.config.model_type == "whisper":
            self.type = "seq2seq_whisper"  # 如果模型类型是 "whisper"，设置类型为 "seq2seq_whisper"
        elif model.__class__.__name__ in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES.values():
            self.type = "seq2seq"  # 如果模型类名在映射字典中的值列表中，则设置类型为 "seq2seq"
        elif (
            feature_extractor._processor_class  # 如果特征提取器的处理类存在
            and feature_extractor._processor_class.endswith("WithLM")  # 并且处理类名称以 "WithLM" 结尾
            and decoder is not None  # 并且解码器不为 None
        ):
            self.decoder = decoder  # 设置解码器
            self.type = "ctc_with_lm"  # 设置类型为 "ctc_with_lm"
        else:
            self.type = "ctc"  # 否则，设置类型为 "ctc"

        # 调用父类的初始化方法，传递模型、分词器、特征提取器、设备和其他关键字参数
        super().__init__(model, tokenizer, feature_extractor, device=device, torch_dtype=torch_dtype, **kwargs)

    # 调用方法，用于处理输入数据
    def __call__(
        self,
        inputs: Union[np.ndarray, bytes, str],  # 输入参数可以是 numpy 数组、字节流或字符串
        **kwargs,  # 其他关键字参数
    ):

    # 校验参数方法，用于规范化处理参数
    def _sanitize_parameters(
        self,
        chunk_length_s=None,  # 分块长度（秒）
        stride_length_s=None,  # 步长长度（秒）
        ignore_warning=None,  # 是否忽略警告
        decoder_kwargs=None,  # 解码器关键字参数
        return_timestamps=None,  # 是否返回时间戳
        return_language=None,  # 是否返回语言
        generate_kwargs=None,  # 生成关键字参数
        max_new_tokens=None,  # 最大新标记数
    ):

    # 后处理方法，用于处理模型输出
    def postprocess(
        self, model_outputs,  # 模型输出
        decoder_kwargs: Optional[Dict] = None,  # 解码器关键字参数
        return_timestamps=None,  # 是否返回时间戳
        return_language=None,  # 是否返回语言
def _find_timestamp_sequence(sequences, tokenizer, feature_extractor, max_source_positions):
    """
    Computes the final sequences by merging the end of the nth sequence with the beginning of the n+1th sequence. Since
    `WhisperForConditionalGeneration` produces the timestamps pairwise, we filter the consecutive timestamps and only
    iterate over them. We keep track of the `time` which indicates the actual starting time of the chunk that is
    processed. We need to make sure to offset the timestamps tokens by the `time` in order for the tokenizer to
    properly compute the final `offset`.
    """
    # index of the first timestamp token
    # 获取第一个时间戳的token索引，这里假设"<|notimestamps|>"是一个特殊标记，用于指示时间戳的开始
    timestamp_begin = tokenizer.convert_tokens_to_ids("<|notimestamps|>") + 1
    items = []
    # approximation of the token to time ratio : ~0.2seconds
    # token与时间的近似比例：约为0.2秒，用于计算时间偏移量
    time_precision = feature_extractor.chunk_length / max_source_positions
    time = 0  # 初始时间设为0
    result = []
    for i in range(len(items)):
        result += items[i].tolist()  # 将items中的元素转换为列表后添加到result中
    return result  # 返回合并后的结果列表
```