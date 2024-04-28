# `.\transformers\models\align\processing_align.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

"""
Image/Text processor class for ALIGN
"""

# 导入必要的模块和类
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding

# 定义 AlignProcessor 类，继承自 ProcessorMixin 类
class AlignProcessor(ProcessorMixin):
    r"""
    Constructs an ALIGN processor which wraps [`EfficientNetImageProcessor`] and
    [`BertTokenizer`]/[`BertTokenizerFast`] into a single processor that interits both the image processor and
    tokenizer functionalities. See the [`~AlignProcessor.__call__`] and [`~OwlViTProcessor.decode`] for more
    information.

    Args:
        image_processor ([`EfficientNetImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`BertTokenizer`, `BertTokenizerFast`]):
            The tokenizer is a required input.
    """

    # 定义类属性
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "EfficientNetImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    # 初始化方法，接受 image_processor 和 tokenizer 两个参数
    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    # 批量解码方法，将参数转发给 BertTokenizerFast 的 batch_decode 方法
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 解码方法，将参数转发给 BertTokenizerFast 的 decode 方法
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入名称的属性，合并 tokenizer 和 image_processor 的模��输入名称并去重
    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```