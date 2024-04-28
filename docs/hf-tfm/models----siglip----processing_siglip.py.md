# `.\transformers\models\siglip\processing_siglip.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求或经书面同意，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意，软件按"原样"分发
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制

"""
SigLIP 的图像/文本处理器类。
"""

from typing import List, Optional, Union

# 导入所需模块和类
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType

# 定义 SiglipProcessor 类，继承 ProcessorMixin 类
class SiglipProcessor(ProcessorMixin):
    r"""
    构建一个 Siglip 处理器，将 Siglip 图像处理器和 Siglip 分词器封装成一个单一处理器。

    [`SiglipProcessor`] 提供了 [`SiglipImageProcessor`] 和 [`SiglipTokenizer`] 的所有功能。查看
    [`~SiglipProcessor.__call__`] 和 [`~SiglipProcessor.decode`] 以获取更多信息。

    Args:
        image_processor ([`SiglipImageProcessor`]):
            图像处理器是必需的输入。
        tokenizer ([`SiglipTokenizer`]):
            分词器是必需的输入。
    """

    # 定义类属性
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = "SiglipTokenizer"

    # 初始化方法
    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    # 调用方法，接受文本、图像等参数
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: int = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    def decode(self, *args, **kwargs):
        """
        此方法将所有参数转发给 SiglipTokenizer 的 [`~PreTrainedTokenizer.decode`]。请参考
        此方法的文档字符串以获取更多信息。
        """
        return self.tokenizer.decode(*args, **kwargs)

    # 批量解码方法，将参数转发给 SiglipTokenizer 的 [`~PreTrainedTokenizer.batch_decode`]
    def batch_decode(self, *args, **kwargs):
        """
        此方法将所有参数转发给 SiglipTokenizer 的 [`~PreTrainedTokenizer.batch_decode`]。请参考
        此方法的文档字符串以获取更多信息。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 属性装饰器，从 transformers.models.clip.processing_clip.CLIPProcessor.model_input_names 复制
    # 定义一个方法，返回模型输入的名称列表
    def model_input_names(self):
        # 获取tokenizer对象的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取image_processor对象的模型输入名称列表
        image_processor_input_names = self.image_processor.model_input_names
        # 将两个列表合并，并去除重复的元素，返回一个新的列表
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```