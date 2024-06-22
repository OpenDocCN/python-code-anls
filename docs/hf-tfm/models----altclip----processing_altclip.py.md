# `.\transformers\models\altclip\processing_altclip.py`

```py
# 设定文件编码为 UTF-8
# 版权声明：2022 年 WenXiang ZhongzhiCheng LedellWu LiuGuang BoWenZhang The HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）的规定;
# 您只有在遵守许可证的情况下才能使用本文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“按原样”分发，
# 不附带任何明示或暗示的保证或条件。
# 有关许可证的详细信息，请参阅许可证。
"""
AltCLIP 的图像/文本处理类
"""
# 导入警告模块
import warnings

# 导入处理工具类
from ...processing_utils import ProcessorMixin
# 导入批编码类
from ...tokenization_utils_base import BatchEncoding

# 定义 AltCLIPProcessor 类，继承 ProcessorMixin 类
class AltCLIPProcessor(ProcessorMixin):
    r"""
    构造一个 AltCLIP 处理器，它将 CLIP 图像处理器和 XLM-Roberta 分词器封装到一个单一处理器中。

    [`AltCLIPProcessor`] 提供了 [`CLIPImageProcessor`] 和 [`XLMRobertaTokenizerFast`] 的所有功能。请参阅
    [`~AltCLIPProcessor.__call__`] 和 [`~AltCLIPProcessor.decode`] 获取更多信息。

    参数:
        image_processor ([`CLIPImageProcessor`], *可选*):
            图像处理器是必需的输入。
        tokenizer ([`XLMRobertaTokenizerFast`], *可选*):
            分词器是必需的输入。
    """

    # 定义属性
    attributes = ["image_processor", "tokenizer"]
    # 图像处理器类名
    image_processor_class = "CLIPImageProcessor"
    # 分词器类名
    tokenizer_class = ("XLMRobertaTokenizer", "XLMRobertaTokenizerFast")

    # 初始化方法
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        # 如果 kwargs 中存在 "feature_extractor" 参数，进行警告
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            # 将 feature_extractor 赋值给变量
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果 image_processor 为 None，则尝试使用 feature_extractor
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果 image_processor 仍然为 None，则抛出 ValueError
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果 tokenizer 为 None，则抛出 ValueError
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类的初始化方法，传入 image_processor 和 tokenizer
        super().__init__(image_processor, tokenizer)

    # 批量解码方法
    def batch_decode(self, *args, **kwargs):
        """
        此方法将所有参数转发到 XLMRobertaTokenizerFast 的 [`~PreTrainedTokenizer.batch_decode`]。
        有关更多信息，请参阅此方法的文档字符串。
        """
        # 调用 tokenizer 的批量解码方法
        return self.tokenizer.batch_decode(*args, **kwargs)
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to XLMRobertaTokenizerFast's `PreTrainedTokenizer.decode` method.
        Please refer to the docstring of this method for more information.
        """
        # 调用 XLMRobertaTokenizerFast 对象的 decode 方法，并将所有参数传递给该方法
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        # 获取 tokenizer 对象的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取 image_processor 对象的模型输入名称列表
        image_processor_input_names = self.image_processor.model_input_names
        # 合并两个列表，并去除重复项，作为模型输入名称列表
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```