# `.\transformers\models\vision_text_dual_encoder\processing_vision_text_dual_encoder.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言
"""
Processor class for VisionTextDualEncoder
"""

# 导入警告模块
import warnings

# 导入处理工具类 ProcessorMixin 和 BatchEncoding 类
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding

# 定义 VisionTextDualEncoderProcessor 类，继承 ProcessorMixin 类
class VisionTextDualEncoderProcessor(ProcessorMixin):
    r"""
    Constructs a VisionTextDualEncoder processor which wraps an image processor and a tokenizer into a single
    processor.

    [`VisionTextDualEncoderProcessor`] offers all the functionalities of [`AutoImageProcessor`] and [`AutoTokenizer`].
    See the [`~VisionTextDualEncoderProcessor.__call__`] and [`~VisionTextDualEncoderProcessor.decode`] for more
    information.

    Args:
        image_processor ([`AutoImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`], *optional*):
            The tokenizer is a required input.
    """

    # 定义类属性
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    # 初始化方法
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        # 如果 kwargs 中包含 "feature_extractor"，则发出警告
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果 image_processor 为 None，则使用 feature_extractor
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果 image_processor 仍为 None，则抛出数值错误
        if image_processor is None:
            raise ValueError("You have to specify an image_processor.")
        # 如果 tokenizer 为 None，则抛出数值错误
        if tokenizer is None:
            raise ValueError("You have to specify a tokenizer.")

        # 调用父类的初始化方法，传入 image_processor 和 tokenizer
        super().__init__(image_processor, tokenizer)
        # 设置当前处理器为 image_processor
        self.current_processor = self.image_processor

    # 批量解码方法
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to VisionTextDualEncoderTokenizer's
        [`~PreTrainedTokenizer.batch_decode`]. Please refer to the docstring of this method for more information.
        """
        # 将所有参数转发给 VisionTextDualEncoderTokenizer 的 batch_decode 方法
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 将所有参数转发给 VisionTextDualEncoderTokenizer 的 `~PreTrainedTokenizer.decode` 方法
    # 请参考该方法的文档字符串获取更多信息
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入的名称列表，包括文本处理器和图像处理器的输入名称
    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # 返回特征提取器的类，已弃用，将在 v5 中移除，建议使用 `image_processor_class` 替代
    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    # 返回特征提取器，已弃用，将在 v5 中移除，建议使用 `image_processor` 替代
    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
```