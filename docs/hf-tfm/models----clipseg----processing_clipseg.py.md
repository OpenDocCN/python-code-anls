# `.\transformers\models\clipseg\processing_clipseg.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言

"""
CLIPSeg 的图像/文本处理器类
"""

# 导入警告模块
import warnings

# 导入处理工具类 ProcessorMixin 和 BatchEncoding 类
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding

# 定义 CLIPSegProcessor 类，继承 ProcessorMixin 类
class CLIPSegProcessor(ProcessorMixin):
    r"""
    构建一个 CLIPSeg 处理器，将 CLIPSeg 图像处理器和 CLIP 分词器封装成一个单一处理器。

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

    # 初始化方法
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        # 如果参数中包含 'feature_extractor'，发出警告并将其移除
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果未提供 image_processor，则使用 feature_extractor
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果未提供 image_processor，则抛出数值错误
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果未提供 tokenizer，则抛出数值错误
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类的初始化方法，传入 image_processor 和 tokenizer
        super().__init__(image_processor, tokenizer)

    # 批量解码方法，将参数转发给 CLIPTokenizerFast 的 `~PreTrainedTokenizer.batch_decode`
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 解码方法，将参数转发给 CLIPTokenizerFast 的 `~PreTrainedTokenizer.decode`
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    # 属性
    @property
    # 返回特征提取器类别，但已过时，将在 v5 版本中被移除。建议使用 `image_processor_class` 代替。
    def feature_extractor_class(self):
        # 发出警告，提醒用户 `feature_extractor_class` 已过时，建议使用 `image_processor_class` 代替。
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        # 返回图像处理器类别
        return self.image_processor_class
    
    # 返回特征提取器，但已过时，将在 v5 版本中被移除。建议使用 `image_processor` 代替。
    @property
    def feature_extractor(self):
        # 发出警告，提醒用户 `feature_extractor` 已过时，建议使用 `image_processor` 代替。
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        # 返回图像处理器
        return self.image_processor
```