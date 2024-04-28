# `.\transformers\models\oneformer\processing_oneformer.py`

```
# 定义作者和许可证信息
# 基于 Apache 许可证 2.0 授权使用此代码
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获得许可证的副本
"""
OneFormer 的图像/文本处理类
"""

from typing import List

# 导入 ProcessorMixin 和 is_torch_available 函数
from ...processing_utils import ProcessorMixin
from ...utils import is_torch_available

# 如果环境支持 PyTorch，导入 torch
if is_torch_available():
    import torch

# 定义 OneFormerProcessor 类，继承 ProcessorMixin 类
class OneFormerProcessor(ProcessorMixin):
    r"""
    构造一个 OneFormer 处理器，将 [`OneFormerImageProcessor`] 和 [`CLIPTokenizer`]/
    [`CLIPTokenizerFast`] 封装为一个单一的处理器，继承了图像处理器和分词器的功能。

    Args:
        image_processor ([`OneFormerImageProcessor`]):
            图像处理器是必需的输入。
        tokenizer ([`CLIPTokenizer`, `CLIPTokenizerFast`]):
            分词器是必需的输入。
        max_seq_len (`int`, *optional*, 默认为 77)):
            输入文本列表的序列长度。
        task_seq_len (`int`, *optional*, 默认为 77):
            输入任务标记的序列长度。
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "OneFormerImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    # 初始化方法
    def __init__(
        self, image_processor=None, tokenizer=None, max_seq_length: int = 77, task_seq_length: int = 77, **kwargs
    ):
        # 如果未提供图像处理器，则抛出数值错误
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果未提供分词器，则抛出数值错误
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.max_seq_length = max_seq_length
        self.task_seq_length = task_seq_length

        # 调用父类 ProcessorMixin 的构造函数
        super().__init__(image_processor, tokenizer)

    # 文本预处理方法
    def _preprocess_text(self, text_list=None, max_length=77):
        # 如果 tokens 为空，则抛出数值错误
        if text_list is None:
            raise ValueError("tokens cannot be None.")

        # 使用 tokenizer 进行文本分词和填充
        tokens = self.tokenizer(text_list, padding="max_length", max_length=max_length, truncation=True)

        attention_masks, input_ids = tokens["attention_mask"], tokens["input_ids"]

        token_inputs = []
        # 根据输入的每个文本生成 token 输入向量
        for attn_mask, input_id in zip(attention_masks, input_ids):
            token = torch.tensor(attn_mask) * torch.tensor(input_id)
            token_inputs.append(token.unsqueeze(0))

        # 将生成的 token 输入向量拼接在一起
        token_inputs = torch.cat(token_inputs, dim=0)
        return token_inputs
    def encode_inputs(self, images=None, task_inputs=None, segmentation_maps=None, **kwargs):
        """
        此方法将所有参数传递给 [`OneFormerImageProcessor.encode_inputs`]，然后对 task_inputs 进行分词编码。
        请参考该方法的文档字符串了解更多信息。
        """

        if task_inputs is None:
            raise ValueError("You have to specify the task_input. Found None.")
        elif images is None:
            raise ValueError("You have to specify the image. Found None.")

        if not all(task in ["semantic", "instance", "panoptic"] for task in task_inputs):
            raise ValueError("task_inputs must be semantic, instance, or panoptic.")

        encoded_inputs = self.image_processor.encode_inputs(images, task_inputs, segmentation_maps, **kwargs)

        if isinstance(task_inputs, str):
            task_inputs = [task_inputs]

        if isinstance(task_inputs, List) and all(isinstance(task_input, str) for task_input in task_inputs):
            task_token_inputs = []
            for task in task_inputs:
                # 根据任务生成相应的输入
                task_input = f"the task is {task}"
                task_token_inputs.append(task_input)
            encoded_inputs["task_inputs"] = self._preprocess_text(task_token_inputs, max_length=self.task_seq_length)
        else:
            raise TypeError("Task Inputs should be a string or a list of strings.")

        if hasattr(encoded_inputs, "text_inputs"):
            # 获取文本输入
            texts_list = encoded_inputs.text_inputs

            text_inputs = []
            for texts in texts_list:
                # 对文本进行预处理和编码
                text_input_list = self._preprocess_text(texts, max_length=self.max_seq_length)
                text_inputs.append(text_input_list.unsqueeze(0))

            encoded_inputs["text_inputs"] = torch.cat(text_inputs, dim=0)

        return encoded_inputs

    def post_process_semantic_segmentation(self, *args, **kwargs):
        """
        此方法将所有参数传递给 [`OneFormerImageProcessor.post_process_semantic_segmentation`]。
        请参考该方法的文档字符串了解更多信息。
        """
        return self.image_processor.post_process_semantic_segmentation(*args, **kwargs)

    def post_process_instance_segmentation(self, *args, **kwargs):
        """
        此方法将所有参数传递给 [`OneFormerImageProcessor.post_process_instance_segmentation`]。
        请参考该方法的文档字符串了解更多信息。
        """
        return self.image_processor.post_process_instance_segmentation(*args, **kwargs)

    def post_process_panoptic_segmentation(self, *args, **kwargs):
        """
        此方法将所有参数传递给 [`OneFormerImageProcessor.post_process_panoptic_segmentation`]。
        请参考该方法的文档字符串了解更多信息。
        """
        return self.image_processor.post_process_panoptic_segmentation(*args, **kwargs)
```