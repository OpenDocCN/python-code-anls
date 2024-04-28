# `.\transformers\models\pix2struct\processing_pix2struct.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，您可以在遵守许可证的情况下使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言
"""
# 导入所需的模块和类
from typing import List, Optional, Union
# 导入自定义的处理工具类
from ...processing_utils import ProcessorMixin
# 导入基础的处理工具类
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
# 导入工具类
from ...utils import TensorType

# 定义 Pix2StructProcessor 类，继承 ProcessorMixin 类
class Pix2StructProcessor(ProcessorMixin):
    r"""
    构建一个 PIX2STRUCT 处理器，将 BERT 分词器和 PIX2STRUCT 图像处理器封装成一个单独的处理器。

    [`Pix2StructProcessor`] 提供了 [`Pix2StructImageProcessor`] 和 [`T5TokenizerFast`] 的所有功能。查看 [`~Pix2StructProcessor.__call__`] 和 [`~Pix2StructProcessor.decode`] 的文档字符串以获取更多信息。

    Args:
        image_processor (`Pix2StructImageProcessor`):
            一个 [`Pix2StructImageProcessor`] 的实例。图像处理器是必需的输入。
        tokenizer (Union[`T5TokenizerFast`, `T5Tokenizer`]):
            一个 ['T5TokenizerFast`] 或 ['T5Tokenizer`] 的实例。分词器是必需的输入。
    """

    # 定义类属性
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Pix2StructImageProcessor"
    tokenizer_class = ("T5Tokenizer", "T5TokenizerFast")

    # 初始化方法
    def __init__(self, image_processor, tokenizer):
        # 设置分词器不返回 token 类型 ID
        tokenizer.return_token_type_ids = False
        # 调用父类的初始化方法
        super().__init__(image_processor, tokenizer)

    # 定义 __call__ 方法
    def __call__(
        self,
        images=None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        max_patches: Optional[int] = 2048,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    # 将所有参数转发给Pix2StructTokenizerFast的`~PreTrainedTokenizer.batch_decode`方法
    # 请参考该方法的文档字符串以获取更多信息
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 将所有参数转发给Pix2StructTokenizerFast的`~PreTrainedTokenizer.decode`方法
    # 请参考该方法的文档字符串以获取更多信息
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入的名称列表，包括tokenizer和image_processor的输入名称
    @property
    def model_input_names(self):
        # 获取tokenizer的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取image_processor的模型输入名称列表
        image_processor_input_names = self.image_processor.model_input_names
        # 将tokenizer和image_processor的输入名称合并，去除重复的名称，返回列表
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```