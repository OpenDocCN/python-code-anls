# `.\models\idefics\processing_idefics.py`

```py
# 设置文件编码为 utf-8
# 版权声明，使用 Apache License 2.0 许可
# 仅在遵守许可证的情况下使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取有关特定语言的权限和限制
"""
# IDEFICS 的 Processor 类
"""

# 导入必要的模块和类
from typing import Callable, List, Optional, Union
from urllib.parse import urlparse

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType, is_torch_available

# 如果有 torch 可用，则导入 torch 模块
if is_torch_available():
    import torch

# 定义一个特殊的图像标记
IMAGE_TOKEN = "<image>"

# 从 m4.training.packing 复制的函数
def incremental_to_binary_attention_mask(incremental_mask, num_classes=-1):
    # 将 [-1, 0, 1] 转换为 [[0, 0], [1, 0], [0, 1]]

    # 如果任何图像索引超过 num_classes，则将其设置为 -1
    # 超过允许的最大图像数量的单词不会参与任何注意力
    if num_classes != -1:
        incremental_mask[incremental_mask >= num_classes] = -1

    negatives = incremental_mask == -1
    incremental_mask[negatives] = 0
    attn_mask = torch.nn.functional.one_hot(incremental_mask, num_classes=num_classes)
    attn_mask[negatives, :] = 0
    return attn_mask

# 从 m4.training.packing 复制的函数
def image_attention_mask_for_packed_input_ids(input_ids, tokenizer):
    image_attention_mask = torch.full_like(input_ids, fill_value=-1)
    next_image_attention_mask = torch.full_like(input_ids, fill_value=-1)
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    eod_token_id = tokenizer.eos_token_id
    for batch_idx in range(input_ids.size(0)):
        count = -1
        seen_eod = False
        for idx, token_id in enumerate(input_ids[batch_idx]):
            if token_id == image_token_id:
                count += 1
                image_attention_mask[batch_idx][idx] = count
                seen_eod = False
            else:
                image_attention_mask[batch_idx][idx] = count

            if seen_eod:
                image_attention_mask[batch_idx][idx] = -1

            if token_id == eod_token_id:
                seen_eod = True
    # 遍历输入的批次中的每个索引
    for batch_idx in range(input_ids.size(0)):
        # 初始化计数器为-1，标记是否遇到了结束符号
        count = -1
        seen_eod = False
        # 从最后一个 token 开始向前遍历
        for idx in range(input_ids[batch_idx].size(0) - 1, -1, -1):
            # 获取当前 token 的 ID
            token_id = input_ids[batch_idx][idx]
            # 如果当前 token 是图片 token
            if token_id == image_token_id:
                # 计数器加一，更新下一个图片的注意力掩码
                count += 1
                next_image_attention_mask[batch_idx][idx] = count
                seen_eod = False
            else:
                # 如果不是图片 token，则更新下一个图片的注意力掩码
                next_image_attention_mask[batch_idx][idx] = count

            # 如果当前 token 是结束符号 token
            if token_id == eod_token_id:
                seen_eod = True

            # 如果已经遇到了结束符号
            if seen_eod:
                # 更新下一个图片的注意力掩码为-1
                next_image_attention_mask[batch_idx][idx] = -1

        # 获取非负索引
        non_negative_indices = next_image_attention_mask[batch_idx] != -1
        # 对非负索引的注意力掩码进行调整
        next_image_attention_mask[batch_idx][non_negative_indices] -= count
        next_image_attention_mask[batch_idx][non_negative_indices] *= -1

    # 返回图片的注意力掩码和下一个图片的注意力掩码
    return image_attention_mask, next_image_attention_mask
def is_url(string):
    """检查传入的字符串是否包含有效的 URL，且没有其他内容。例如，如果包含空格，则立即使 URL 无效"""
    # 如果字符串中包含空格，则返回 False
    if " " in string:
        return False
    # 使用 urlparse 函数解析字符串，判断是否包含 scheme 和 netloc
    result = urlparse(string)
    return all([result.scheme, result.netloc])


class IdeficsProcessor(ProcessorMixin):
    r"""
    构建一个 IDEFICS 处理器，将 LLama 分词器和 IDEFICS 图像处理器封装成一个单一处理器。

    [`IdeficsProcessor`] 提供了 [`IdeficsImageProcessor`] 和 [`LlamaTokenizerFast`] 的所有功能。更多信息请参阅 [`~IdeficsProcessor.__call__`] 和 [`~IdeficsProcessor.decode`] 的文档。

    Args:
        image_processor (`IdeficsImageProcessor`):
            [`IdeficsImageProcessor`] 的一个实例。图像处理器是必需的输入。
        tokenizer (`LlamaTokenizerFast`):
            [`LlamaTokenizerFast`] 的一个实例。分词器是必需的输入。
        image_size (`int`, *可选*, 默认为 224): 图像大小（假设为正方形图像）
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "IdeficsImageProcessor"
    tokenizer_class = "LlamaTokenizerFast"

    def __init__(self, image_processor, tokenizer=None, image_size=224, add_end_of_utterance_token=None, **kwargs):
        # 如果 image_processor 为 None，则引发 ValueError
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果 tokenizer 为 None，则引发 ValueError
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        self.default_image_dims = (
            self.image_processor.image_num_channels,
            self.image_processor.image_size,
            self.image_processor.image_size,
        )

        self.tokenizer_was_trained_with_end_of_utterance_token = (
            True
            if "<end_of_utterance>" in self.tokenizer.special_tokens_map.get("additional_special_tokens", [])
            else False
        )

    def __call__(
        self,
        prompts: Union[List[TextInput], List[List[TextInput]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        transform: Callable = None,
        add_eos_token=False,
        add_end_of_utterance_token=None,
        debug=False,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    def batch_decode(self, *args, **kwargs):
        """
        此方法将所有参数转发给 LlamaTokenizerFast 的 [`~PreTrainedTokenizer.batch_decode`]。有关更多信息，请参阅此方法的文档。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 定义一个方法用于解码，将所有参数转发给 LlamaTokenizerFast 的 `~PreTrainedTokenizer.decode` 方法。请参考该方法的文档字符串以获取更多信息。
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入的名称列表
    @property
    def model_input_names(self):
        # 获取分词器的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取图像处理器的模型输入名称列表
        image_processor_input_names = self.image_processor.model_input_names
        # 将分词器和图像处理器的输入名称列表合并，并去除重复的名称，返回结果列表
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```