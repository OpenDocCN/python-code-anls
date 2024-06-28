# `.\models\idefics\processing_idefics.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""
Processor class for IDEFICS.
"""

from typing import Callable, List, Optional, Union
from urllib.parse import urlparse

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType, is_torch_available


if is_torch_available():
    import torch


IMAGE_TOKEN = "<image>"


# copied from m4.training.packing
# 将增量注意力掩码转换为二进制注意力掩码
def incremental_to_binary_attention_mask(incremental_mask, num_classes=-1):
    # This function converts: [-1, 0, 1] => [[0, 0], [1, 0], [0, 1]]

    # 如果有任何图像索引超过 num_classes，将它们设置为 -1
    # 超过允许的最大图像数量的单词不参与任何注意力
    if num_classes != -1:
        incremental_mask[incremental_mask >= num_classes] = -1

    negatives = incremental_mask == -1
    incremental_mask[negatives] = 0
    attn_mask = torch.nn.functional.one_hot(incremental_mask, num_classes=num_classes)
    attn_mask[negatives, :] = 0
    return attn_mask


# copied from m4.training.packing
# 为打包的输入 ID 创建图像注意力掩码
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
    # 遍历每个批次中的输入 ID
    for batch_idx in range(input_ids.size(0)):
        # 初始化计数器为-1，表示还未遇到图片标记
        count = -1
        # 标记是否已经遇到过结束符 (eod_token_id)
        seen_eod = False
        
        # 倒序遍历当前批次中的输入 ID
        for idx in range(input_ids[batch_idx].size(0) - 1, -1, -1):
            # 获取当前位置的 token ID
            token_id = input_ids[batch_idx][idx]
            
            # 如果当前 token 是图片标记 (image_token_id)
            if token_id == image_token_id:
                # 计数器加一，表示遇到了下一个图片标记
                count += 1
                # 在下一个图片标记的位置设置注意力掩码为当前计数值
                next_image_attention_mask[batch_idx][idx] = count
                # 重置结束符标记为未见过
                seen_eod = False
            else:
                # 在非图片标记位置设置注意力掩码为当前计数值
                next_image_attention_mask[batch_idx][idx] = count
            
            # 如果当前 token 是结束符 (eod_token_id)
            if token_id == eod_token_id:
                # 标记已经遇到过结束符
                seen_eod = True
            
            # 如果已经遇到过结束符
            if seen_eod:
                # 在结束符后的位置设置注意力掩码为-1
                next_image_attention_mask[batch_idx][idx] = -1
        
        # 找出非负索引位置
        non_negative_indices = next_image_attention_mask[batch_idx] != -1
        # 对非负索引位置的注意力掩码值进行调整
        next_image_attention_mask[batch_idx][non_negative_indices] -= count
        next_image_attention_mask[batch_idx][non_negative_indices] *= -1
    
    # 返回处理后的注意力掩码
    return image_attention_mask, next_image_attention_mask
def is_url(string):
    """Checks if the passed string contains a valid URL and nothing else. 
    If a space is included, the URL is immediately invalidated."""
    # 如果字符串中包含空格，则返回 False
    if " " in string:
        return False
    # 解析 URL，验证其结构是否符合 URL 标准
    result = urlparse(string)
    # 检查 URL 是否包含 scheme 和 netloc，若都包含则认为是有效的 URL
    return all([result.scheme, result.netloc])


class IdeficsProcessor(ProcessorMixin):
    r"""
    Constructs an IDEFICS processor which wraps a LLama tokenizer and IDEFICS image processor into a single processor.

    [`IdeficsProcessor`] offers all the functionalities of [`IdeficsImageProcessor`] and [`LlamaTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`IdeficsImageProcessor`):
            An instance of [`IdeficsImageProcessor`]. The image processor is a required input.
        tokenizer (`LlamaTokenizerFast`):
            An instance of [`LlamaTokenizerFast`]. The tokenizer is a required input.
        image_size (`int`, *optional*, defaults to 224): Image size (assuming a square image)
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "IdeficsImageProcessor"
    tokenizer_class = "LlamaTokenizerFast"

    def __init__(self, image_processor, tokenizer=None, image_size=224, add_end_of_utterance_token=None, **kwargs):
        # 检查 image_processor 是否为空，若为空则抛出 ValueError
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 检查 tokenizer 是否为空，若为空则抛出 ValueError
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类的初始化方法，传入 image_processor 和 tokenizer
        super().__init__(image_processor, tokenizer)
        # 将当前的处理器设置为 image_processor
        self.current_processor = self.image_processor
        # 将图片 token 的 ID 设置为 tokenizer 中 IMAGE_TOKEN 对应的 ID
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        # 设置默认的图片维度，从 image_processor 中获取图像的通道数和尺寸
        self.default_image_dims = (
            self.image_processor.image_num_channels,
            self.image_processor.image_size,
            self.image_processor.image_size,
        )

        # 检查 tokenizer 是否训练过 "<end_of_utterance>" 这个特殊 token
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
    ):
        """
        This method processes input prompts into tokenized and optionally transformed outputs.

        Args:
            prompts (Union[List[TextInput], List[List[TextInput]]]): Input prompts to process.
            padding (Union[bool, str, PaddingStrategy], optional): Padding strategy for tokenized outputs. Defaults to False.
            truncation (Union[bool, str, TruncationStrategy], optional): Truncation strategy for tokenized outputs. Defaults to None.
            max_length (Optional[int], optional): Maximum length of the tokenized outputs. Defaults to None.
            transform (Callable, optional): Transformation function applied after tokenization. Defaults to None.
            add_eos_token (bool, optional): Whether to add an end-of-sequence token. Defaults to False.
            add_end_of_utterance_token (None, optional): Placeholder for adding end-of-utterance token. Defaults to None.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.
            return_tensors (Optional[Union[str, TensorType]], optional): Output tensor type. Defaults to TensorType.PYTORCH.

        Returns:
            Dict[str, Any]: Processed outputs based on input prompts.
        """
        # 在这里实现具体的处理逻辑，将输入 prompts 处理为 tokenized 和可能转换后的输出
        pass

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        # 调用 tokenizer 的 batch_decode 方法，将参数透传给它
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 将所有参数转发到 LlamaTokenizerFast 的 `PreTrainedTokenizer.decode` 方法中，并返回结果
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    # 获取模型输入的名称列表，合并并去重来自于 tokenizer 和 image_processor 的输入名称
    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```