# `.\models\clipseg\processing_clipseg.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，使用 Apache License 2.0 许可证
# 警告：如果没有符合许可证要求的代码，不能使用本文件
"""
CLIPSeg 的图像/文本处理器类
"""

# 导入警告模块
import warnings

# 导入 ProcessorMixin 和 BatchEncoding 类
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding


class CLIPSegProcessor(ProcessorMixin):
    r"""
    构建一个 CLIPSeg 处理器，将 CLIPSeg 图像处理器和 CLIP 分词器包装成一个单一处理器。

    [`CLIPSegProcessor`] 提供了 [`ViTImageProcessor`] 和 [`CLIPTokenizerFast`] 的所有功能。查看
    [`~CLIPSegProcessor.__call__`] 和 [`~CLIPSegProcessor.decode`] 获取更多信息。

    Args:
        image_processor ([`ViTImageProcessor`], *optional*):
            图像处理器是必需的输入。
        tokenizer ([`CLIPTokenizerFast`], *optional*):
            分词器是必需的输入。
    """

    # 定义类属性
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "ViTImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # 检查是否存在 "feature_extractor" 参数，并给出警告信息
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果未指定 image_processor，则使用 feature_extractor
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果未指定 tokenizer，则抛出 ValueError
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类的初始化方法
        super().__init__(image_processor, tokenizer)

    def batch_decode(self, *args, **kwargs):
        """
        此方法将所有参数转发给 CLIPTokenizerFast 的 [`~PreTrainedTokenizer.batch_decode`]。更多信息请参考该方法的文档字符串。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        此方法将所有参数转发给 CLIPTokenizerFast 的 [`~PreTrainedTokenizer.decode`]。更多信息请参考该方法的文档字符串。
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # 发出警告，提醒用户 `feature_extractor_class` 方法已弃用，将在 v5 版本中移除，建议使用 `image_processor_class` 方法代替。
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        # 返回当前对象的 `image_processor_class` 属性
        return self.image_processor_class

    # 属性装饰器，发出警告，提醒用户 `feature_extractor` 属性已弃用，将在 v5 版本中移除，建议使用 `image_processor` 属性代替。
    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        # 返回当前对象的 `image_processor` 属性
        return self.image_processor
```