# `.\transformers\models\chinese_clip\processing_chinese_clip.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 OFA-Sys 团队作者和 HuggingFace 团队所有
#
# 根据 Apache 许可证 2.0 版本使用本文件；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"提供，
# 不提供任何形式的担保或条件。
# 请查阅许可证了解具体语言及限制。

"""
中文 CLIP 的图像/文本处理器类
"""

# 引入警告模块
import warnings

# 引入处理工具混合类和批量编码类
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding


class ChineseCLIPProcessor(ProcessorMixin):
    r"""
    构建一个中文 CLIP 处理器，将中文 CLIP 图像处理器和中文 CLIP 分词器包装成一个单一的处理器。

    [`ChineseCLIPProcessor`] 提供了 [`ChineseCLIPImageProcessor`] 和 [`BertTokenizerFast`] 的所有功能。
    更多信息请参阅 [`~ChineseCLIPProcessor.__call__`] 和 [`~ChineseCLIPProcessor.decode`]。

    Args:
        image_processor ([`ChineseCLIPImageProcessor`], *可选*):
            图像处理器是必需的输入。
        tokenizer ([`BertTokenizerFast`], *可选*):
            分词器是必需的输入。
    """

    # 定义类属性
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "ChineseCLIPImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # 初始化特征提取器为 None
        feature_extractor = None
        # 如果参数中有 `feature_extractor`，发出警告
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            # 将 `feature_extractor` 赋值给 feature_extractor 变量
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果 image_processor 不为空，则赋值给 image_processor 变量，否则使用 feature_extractor
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果 image_processor 为空，则抛出 ValueError 异常
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果 tokenizer 为空，则抛出 ValueError 异常
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类的初始化方法
        super().__init__(image_processor, tokenizer)
        # 将当前处理器设置为 image_processor
        self.current_processor = self.image_processor

    def batch_decode(self, *args, **kwargs):
        """
        此方法将所有参数转发到 BertTokenizerFast 的 [`~PreTrainedTokenizer.batch_decode`] 方法。
        更多信息请参阅该方法的文档字符串。
        """
        # 调用 tokenizer 的 batch_decode 方法并返回结果
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 将所有参数转发给BertTokenizerFast的`~PreTrainedTokenizer.decode`方法。请参考该方法的文档字符串以获取更多信息。
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入名称的属性
    def model_input_names(self):
        # 获取tokenizer的模型输入名称
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取image_processor的模型输入名称
        image_processor_input_names = self.image_processor.model_input_names
        # 将tokenizer和image_processor的模型输入名称合并并去重，返回列表
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # 返回特征提取器类的属性
    def feature_extractor_class(self):
        # 发出警告，提示`feature_extractor_class`已弃用，并将在v5中移除。建议使用`image_processor_class`代替。
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        # 返回image_processor_class
        return self.image_processor_class
```