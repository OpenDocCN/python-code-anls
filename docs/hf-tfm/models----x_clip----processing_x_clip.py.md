# `.\transformers\models\x_clip\processing_x_clip.py`

```
# 设置文件编码为utf-8
# 版权声明
# 根据Apache许可证2.0获取许可
# 如果您不遵守许可证，不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的，没有任何明示或暗示的保证或条件
# 请查看许可证以了解特定语言的详细功能和限制

"""
XCLIP的图像/文本处理器类
"""

# 导入警告模块
import warnings
# 从处理工具中导入ProcessorMixin
from ...processing_utils import ProcessorMixin
# 从tokenization_utils_base中导入BatchEncoding

class XCLIPProcessor(ProcessorMixin):
    """
    构造一个X-CLIP处理器，将VideoMAE图像处理器和CLIP分词器封装为单个处理器

    [`XCLIPProcessor`]提供了[`VideoMAEImageProcessor`]和[`CLIPTokenizerFast`]的所有功能。 更多信息请参见[`~XCLIPProcessor.__call__`]和[`~XCLIPProcessor.decode`]

    参数:
        image_processor ([`VideoMAEImageProcessor`], *optional*):
            图像处理器是必需的输入
        tokenizer ([`CLIPTokenizerFast`], *optional*):
            分词器是必需的输入
    """

    # 属性列表
    attributes = ["image_processor", "tokenizer"]
    # 图像处理器类
    image_processor_class = "VideoMAEImageProcessor"
    # 分词器类
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # 特征提取器初始化为空
        feature_extractor = None
        # 如果kwargs中包含"feature_extractor"，则发出警告
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果未指定图像处理器，则将特征提取器作为图像处理器
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果未指定图像处理器，则引发异常
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果未指定分词器，则引发异常
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        # 将当前处理器设置为图像处理器
        self.current_processor = self.image_processor

    # 批量解码方法的注释
    def batch_decode(self, *args, **kwargs):
        """
        此方法将其所有参数转发到CLIPTokenizerFast的[`~PreTrainedTokenizer.batch_decode`]。 有关更多信息，请参阅此方法的文档字符串。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 解码方法的注释
    def decode(self, *args, **kwargs):
        """
        此方法将其所有参数转发到CLIPTokenizerFast的[`~PreTrainedTokenizer.decode`]。 请参阅此方法的文档字符串以获取更多信息。
        """
        return self.tokenizer.decode(*args, **kwargs)
    # 返回模型输入的名称列表，包括 input_ids、attention_mask、position_ids 和 pixel_values
    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "position_ids", "pixel_values"]
    
    # 返回特征提取器类。发出警告提示，`feature_extractor_class` 已弃用，将在 v5 版本中移除，建议使用 `image_processor_class` 替代。
    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class
    
    # 返回特征提取器实例。发出警告提示，`feature_extractor` 已弃用，将在 v5 版本中移除，建议使用 `image_processor` 替代。
    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
```