# `.\transformers\models\trocr\processing_trocr.py`

```py
代码：


# coding=utf-8
# 设置文件编码格式和版权信息
# 版权所有 2021 年的 HuggingFace 公司
# 根据 Apache 许可证 2.0 版进行许可
# 除非符合许可证要求，否则不得使用此文件
# 可以在以下网址获得许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是按“原样”基础提供的，
# 不提供任何明示或暗示的担保或条件。
# 更多信息请参见许可证。
"""
TrOCR 的处理器类。
"""
# 导入警告模块和上下文管理模块
import warnings
from contextlib import contextmanager
# 导入自定义的 ProcessorMixin 类
from ...processing_utils import ProcessorMixin

# TrOCRProcessor 类继承 ProcessorMixin 类
class TrOCRProcessor(ProcessorMixin):
    r"""
    构造一个 TrOCR 处理器，将视觉图像处理器和 TrOCR 分词器打包成一个单一的处理器。

    [`TrOCRProcessor`] 提供了 [`ViTImageProcessor`/`DeiTImageProcessor`] 和
    [`RobertaTokenizer`/`XLMRobertaTokenizer`] 的所有功能。有关更多信息，请参见 [`~TrOCRProcessor.__call__`] 和 [`~TrOCRProcessor.decode`]。

    Args:
        image_processor ([`ViTImageProcessor`/`DeiTImageProcessor`], *optional*):
            [`ViTImageProcessor`/`DeiTImageProcessor`] 的实例。图像处理器是必填项。
        tokenizer ([`RobertaTokenizer`/`XLMRobertaTokenizer`], *optional*):
            [`RobertaTokenizer`/`XLMRobertaTokenizer`] 的实例。分词器是必填项。
    """

    # 定义类属性
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    # 初始化方法
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        # 如果关键字参数中有 feature_extractor，则给出警告
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果 image_processor 为 None，则将 feature_extractor 赋给 image_processor
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果 image_processor 为 None，则引发 ValueError
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果 tokenizer 为 None，则引发 ValueError
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类的初始化方法，传入 image_processor 和 tokenizer
        super().__init__(image_processor, tokenizer)
        # 设置当前处理器为 image_processor
        self.current_processor = self.image_processor
        self._in_target_context_manager = False
    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to AutoImageProcessor's
        [`~AutoImageProcessor.__call__`] and returns its output. If used in the context
        [`~TrOCRProcessor.as_target_processor`] this method forwards all its arguments to TrOCRTokenizer's
        [`~TrOCRTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more information.
        """
        # 检查是否在目标处理器上下文管理器中使用，如果是，则返回当前处理器的结果
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        # 从关键字参数中提取图像和文本数据
        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)
        # 如果存在位置参数，则将第一个位置参数作为图像数据，其余位置参数作为其他参数
        if len(args) > 0:
            images = args[0]
            args = args[1:]

        # 如果既没有图像数据也没有文本数据，则引发值错误
        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        # 如果存在图像数据，则调用图像处理器处理
        if images is not None:
            inputs = self.image_processor(images, *args, **kwargs)
        # 如果存在文本数据，则调用标记器处理
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        # 如果只有文本数据，则返回图像处理器处理的结果
        if text is None:
            return inputs
        # 如果只有图像数据，则返回标记器处理的结果
        elif images is None:
            return encodings
        # 如果既有图像数据又有文本数据，则将标记器处理的结果作为标签添加到图像处理器处理的结果中，并返回
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        # 将所有参数转发给 TrOCRTokenizer 的 batch_decode 方法，并返回其结果
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        # 将所有参数转发给 TrOCRTokenizer 的 decode 方法，并返回其结果
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning TrOCR.
        """
        # 发出警告，表示此方法即将被弃用，并且在 v5 版本中将被移除
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your images inputs, or in a separate call."
        )
        # 进入目标处理器上下文管理器，将当前处理器设置为标记器
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        # 退出目标处理器上下文管理器，将当前处理器设置回图像处理器
        self.current_processor = self.image_processor
        self._in_target_context_manager = False

    @property
    def feature_extractor_class(self):
        # 发出警告，表示此属性即将被弃用，并且在 v5 版本中将被移除，建议使用 image_processor_class 替代
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        # 返回图像处理器类
        return self.image_processor_class

    @property
    # 定义一个名为 feature_extractor 的方法，用于提取特征
    def feature_extractor(self):
        # 发出警告，提示 `feature_extractor` 方法已废弃，并且将在 v5 版本中移除，建议使用 `image_processor` 方法
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        # 返回 image_processor 方法
        return self.image_processor
```