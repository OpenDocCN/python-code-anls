# `.\transformers\models\llava\processing_llava.py`

```py
# 声明文件编码格式为 UTF-8
# 版权声明及许可证信息
"""
Processor class for Llava.
"""

# 导入所需模块和类型提示
from typing import List, Optional, Union

# 导入相关函数和类
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


# 定义 LlavaProcessor 类，继承自 ProcessorMixin
class LlavaProcessor(ProcessorMixin):
    r"""
    Constructs a Llava processor which wraps a Llava image processor and a Llava tokenizer into a single processor.

    [`LlavaProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~LlavaProcessor.__call__`] and [`~LlavaProcessor.decode`] for more information.

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    # 定义类属性
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    # 定义初始化方法
    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)

    # 定义 __call__ 方法，用于处理输入数据
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.__call__`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer(
            text=text,
            images=images,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

    # 定义批量解码方法
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 定义一个方法，接受任意数量的位置参数（args）和关键字参数（kwargs），并将它们全部转发给 LlamaTokenizerFast 的 `~PreTrainedTokenizer.decode` 方法
    """
    This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
    the docstring of this method for more information.
    """
    return self.tokenizer.decode(*args, **kwargs)

    # 定义一个属性，返回模型的输入名称列表，包括 tokenizer 和 image_processor 的输入名称
    # 从 transformers.models.clip.processing_clip.CLIPProcessor.model_input_names 复制
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```