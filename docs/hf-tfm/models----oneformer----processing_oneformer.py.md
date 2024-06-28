# `.\models\oneformer\processing_oneformer.py`

```
# coding=utf-8
# Copyright 2022 SHI Labs and The HuggingFace Inc. team.
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
Image/Text processor class for OneFormer
"""

from typing import List

from ...processing_utils import ProcessorMixin
from ...utils import is_torch_available

# Check if Torch is available before importing
if is_torch_available():
    import torch


class OneFormerProcessor(ProcessorMixin):
    r"""
    Constructs an OneFormer processor which wraps [`OneFormerImageProcessor`] and
    [`CLIPTokenizer`]/[`CLIPTokenizerFast`] into a single processor that inherits both the image processor and
    tokenizer functionalities.

    Args:
        image_processor ([`OneFormerImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`CLIPTokenizer`, `CLIPTokenizerFast`]):
            The tokenizer is a required input.
        max_seq_len (`int`, *optional*, defaults to 77)):
            Sequence length for input text list.
        task_seq_len (`int`, *optional*, defaults to 77):
            Sequence length for input task token.
    """

    # Define the list of attributes that this processor class has
    attributes = ["image_processor", "tokenizer"]
    
    # Define the class names for reference
    image_processor_class = "OneFormerImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(
        self, image_processor=None, tokenizer=None, max_seq_length: int = 77, task_seq_length: int = 77, **kwargs
    ):
        # Check if image_processor and tokenizer are provided, otherwise raise an error
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # Initialize the maximum sequence lengths for text and task tokens
        self.max_seq_length = max_seq_length
        self.task_seq_length = task_seq_length

        # Call the superclass initializer with image_processor and tokenizer
        super().__init__(image_processor, tokenizer)

    def _preprocess_text(self, text_list=None, max_length=77):
        # Ensure text_list is provided; if not, raise an error
        if text_list is None:
            raise ValueError("tokens cannot be None.")

        # Tokenize the input text list using the specified tokenizer, with padding and truncation
        tokens = self.tokenizer(text_list, padding="max_length", max_length=max_length, truncation=True)

        # Extract attention masks and input ids from the tokenized output
        attention_masks, input_ids = tokens["attention_mask"], tokens["input_ids"]

        token_inputs = []
        # Combine attention masks and input ids into tensor inputs
        for attn_mask, input_id in zip(attention_masks, input_ids):
            token = torch.tensor(attn_mask) * torch.tensor(input_id)
            token_inputs.append(token.unsqueeze(0))

        # Concatenate token inputs along the first dimension
        token_inputs = torch.cat(token_inputs, dim=0)
        return token_inputs
    def encode_inputs(self, images=None, task_inputs=None, segmentation_maps=None, **kwargs):
        """
        This method forwards all its arguments to [`OneFormerImageProcessor.encode_inputs`] and then tokenizes the
        task_inputs. Please refer to the docstring of this method for more information.
        """

        # 检查是否未指定任务输入
        if task_inputs is None:
            raise ValueError("You have to specify the task_input. Found None.")
        # 检查是否未指定图像输入
        elif images is None:
            raise ValueError("You have to specify the image. Found None.")

        # 检查任务输入是否全部为语义、实例或全景分割任务
        if not all(task in ["semantic", "instance", "panoptic"] for task in task_inputs):
            raise ValueError("task_inputs must be semantic, instance, or panoptic.")

        # 调用图像处理器的encode_inputs方法，返回编码后的输入数据
        encoded_inputs = self.image_processor.encode_inputs(images, task_inputs, segmentation_maps, **kwargs)

        # 如果任务输入是字符串，转换为列表形式
        if isinstance(task_inputs, str):
            task_inputs = [task_inputs]

        # 如果任务输入是列表且每个元素都是字符串，为每个任务构建任务输入字符串，并预处理成模型输入格式
        if isinstance(task_inputs, list) and all(isinstance(task_input, str) for task_input in task_inputs):
            task_token_inputs = []
            for task in task_inputs:
                task_input = f"the task is {task}"
                task_token_inputs.append(task_input)
            encoded_inputs["task_inputs"] = self._preprocess_text(task_token_inputs, max_length=self.task_seq_length)
        else:
            raise TypeError("Task Inputs should be a string or a list of strings.")

        # 如果encoded_inputs具有属性"text_inputs"，则预处理每个文本输入并组合成张量
        if hasattr(encoded_inputs, "text_inputs"):
            texts_list = encoded_inputs.text_inputs

            text_inputs = []
            for texts in texts_list:
                text_input_list = self._preprocess_text(texts, max_length=self.max_seq_length)
                text_inputs.append(text_input_list.unsqueeze(0))

            encoded_inputs["text_inputs"] = torch.cat(text_inputs, dim=0)

        # 返回编码后的输入数据
        return encoded_inputs

    def post_process_semantic_segmentation(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OneFormerImageProcessor.post_process_semantic_segmentation`].
        Please refer to the docstring of this method for more information.
        """
        # 调用图像处理器的post_process_semantic_segmentation方法并返回结果
        return self.image_processor.post_process_semantic_segmentation(*args, **kwargs)

    def post_process_instance_segmentation(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OneFormerImageProcessor.post_process_instance_segmentation`].
        Please refer to the docstring of this method for more information.
        """
        # 调用图像处理器的post_process_instance_segmentation方法并返回结果
        return self.image_processor.post_process_instance_segmentation(*args, **kwargs)

    def post_process_panoptic_segmentation(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OneFormerImageProcessor.post_process_panoptic_segmentation`].
        Please refer to the docstring of this method for more information.
        """
        # 调用图像处理器的post_process_panoptic_segmentation方法并返回结果
        return self.image_processor.post_process_panoptic_segmentation(*args, **kwargs)
```