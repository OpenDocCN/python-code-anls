# `.\models\git\processing_git.py`

```
# coding=utf-8
# 版权 2022 年 HuggingFace 公司团队所有。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据“原样”分发，无任何明示或暗示的保证
# 请参阅许可证获取特定语言的权限
"""
GIT 的图像/文本处理器类
"""

# 从 processing_utils 中导入 ProcessorMixin
# 从 tokenization_utils_base 中导入 BatchEncoding
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding

# 定义 GitProcessor 类，继承自 ProcessorMixin
class GitProcessor(ProcessorMixin):
    r"""
    构造一个 GIT 处理器，将 CLIP 图像处理器和 BERT 分词器包装成单个处理器。

    [`GitProcessor`] 提供了 [`CLIPImageProcessor`] 和 [`BertTokenizerFast`] 的所有功能。参见
    [`~GitProcessor.__call__`] 和 [`~GitProcessor.decode`] 以获取更多信息。

    Args:
        image_processor ([`AutoImageProcessor`]):
            图像处理器是必需的输入。
        tokenizer ([`AutoTokenizer`]):
            分词器是必需的输入。
    """

    # 类属性列表
    attributes = ["image_processor", "tokenizer"]
    # 图像处理器类名称
    image_processor_class = "AutoImageProcessor"
    # 分词器类名称
    tokenizer_class = "AutoTokenizer"

    # 初始化方法
    def __init__(self, image_processor, tokenizer):
        # 调用父类的初始化方法
        super().__init__(image_processor, tokenizer)
        # 将当前处理器设置为图像处理器
        self.current_processor = self.image_processor

    # 批量解码方法
    def batch_decode(self, *args, **kwargs):
        """
        此方法将其所有参数转发给 BertTokenizerFast 的 [`~PreTrainedTokenizer.batch_decode`]。请
        参阅此方法的文档字符串以获取更多信息。
        """
        # 调用分词器的批量解码方法
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 解码方法
    def decode(self, *args, **kwargs):
        """
        此方法将其所有参数转发给 BertTokenizerFast 的 [`~PreTrainedTokenizer.decode`]。请参阅
        此方法的文档字符串以获取更多信息。
        """
        # 调用分词器的解码方法
        return self.tokenizer.decode(*args, **kwargs)

    # 模型输入名称属性
    @property
    def model_input_names(self):
        # 返回模型输入名称列表
        return ["input_ids", "attention_mask", "pixel_values"]
```