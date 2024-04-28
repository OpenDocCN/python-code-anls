# `.\transformers\models\clip\processing_clip.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，指明版权归 HuggingFace Inc. 团队所有，采用 Apache 许可证 2.0 版本
"""
CLIP 的图像/文本处理类
"""

# 导入警告模块
import warnings

# 导入处理工具混合类和批编码类
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding


class CLIPProcessor(ProcessorMixin):
    r"""
    构建一个 CLIP 处理器，将 CLIP 图像处理器和 CLIP 分词器包装成一个单一的处理器。

    [`CLIPProcessor`] 提供了 [`CLIPImageProcessor`] 和 [`CLIPTokenizerFast`] 的所有功能。参见 [`~CLIPProcessor.__call__`] 和 [`~CLIPProcessor.decode`] 获取更多信息。

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            图像处理器是必需的输入。
        tokenizer ([`CLIPTokenizerFast`], *optional*):
            分词器是必需的输入。
    """

    # 类属性列表
    attributes = ["image_processor", "tokenizer"]
    # 图像处理器类
    image_processor_class = "CLIPImageProcessor"
    # 分词器类
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # 如果参数中有 `feature_extractor`，发出警告并将其移除
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 设置图像处理器
        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 设置分词器
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类初始化方法
        super().__init__(image_processor, tokenizer)

    # 批量解码方法
    def batch_decode(self, *args, **kwargs):
        """
        此方法将所有参数转发给 CLIPTokenizerFast 的 [`~PreTrainedTokenizer.batch_decode`]。有关更多信息，请参阅此方法的文档字符串。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 解码方法
    def decode(self, *args, **kwargs):
        """
        此方法将所有参数转发给 CLIPTokenizerFast 的 [`~PreTrainedTokenizer.decode`]。有关更多信息，请参阅此方法的文档字符串。
        """
        return self.tokenizer.decode(*args, **kwargs)

    # 属性
    @property
```  
    # 返回模型输入的名称列表，包括tokenizer和image_processor的输入名称
    def model_input_names(self):
        # 获取tokenizer的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取image_processor的模型输入名称列表
        image_processor_input_names = self.image_processor.model_input_names
        # 将两个列表合并并去重，返回结果
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # 返回特征提取器的类名，已废弃，将在v5中移除，建议使用image_processor_class代替
    @property
    def feature_extractor_class(self):
        # 发出警告信息，提醒用户该属性已废弃
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        # 返回image_processor_class属性
        return self.image_processor_class

    # 返回特征提取器，已废弃，将在v5中移除，建议使用image_processor代替
    @property
    def feature_extractor(self):
        # 发出警告信息，提醒用户该属性已废弃
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        # 返回image_processor属性
        return self.image_processor
```