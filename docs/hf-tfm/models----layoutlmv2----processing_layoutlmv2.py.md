# `.\models\layoutlmv2\processing_layoutlmv2.py`

```
# 设置编码格式为 utf-8
# 版权声明 2021 年 HuggingFace 公司团队所有
# 根据 Apache 许可证第 2.0 版授权
# 只有遵守许可证规定的情况下才能使用这个文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律或书面同意要求，否则软件按"原样"分发
# 没有明示或暗示的任何保证或条件，包括但不限于特定目的的适销性或适用性保证
# 请查看许可证以获取特定语言的权限和限制
"""
# 导入警告模块
import warnings
# 导入类型提示模块中的 List, Optional, Union
from typing import List, Optional, Union
# 从...中导入处理工具的 ProcessorMixin
from ...processing_utils import ProcessorMixin
# 从...tokenization_utils_base中导入批处理编码, 填充策略, 预处理标记输入, 文本输入, 截断策略
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
# 从...中的utils中导入 TensorType
from ...utils import TensorType
# 定义 LayoutLMv2Processor 类，该类继承 ProcessorMixin
class LayoutLMv2Processor(ProcessorMixin):
    r"""
    构造一个 LayoutLMv2 处理器，将 LayoutLMv2 图像处理器和 LayoutLMv2 分词器结合到一个单独的处理器中。

    [`LayoutLMv2Processor`] 提供了准备数据给模型的所有功能。

    它首先使用 [`LayoutLMv2ImageProcessor`] 将文档图像调整为固定大小，并可选择性地应用 OCR 来获取单词和规范化的边界框。
    然后将这些提供给 [`LayoutLMv2Tokenizer`] 或 [`LayoutLMv2TokenizerFast`]，它将单词和边界框转换为标记级别的`input_ids`、
    `attention_mask`、`token_type_ids`、`bbox`。可选择地，可以提供整数型的 `word_labels`，这些被转换为标记级别的`labels`，
    用于标记分类任务（例如 FUNSD，CORD）。

    参数:
        image_processor (`LayoutLMv2ImageProcessor`, *可选*):
            一个 [`LayoutLMv2ImageProcessor`] 的实例。图像处理器是必需的输入。
        tokenizer (`LayoutLMv2Tokenizer` or `LayoutLMv2TokenizerFast`, *可选*):
            一个 [`LayoutLMv2Tokenizer`] 或 [`LayoutLMv2TokenizerFast`] 的实例。分词器是必需的输入。
    """

    # 属性列表
    attributes = ["image_processor", "tokenizer"]
    # 图像处理器类名
    image_processor_class = "LayoutLMv2ImageProcessor"
    # 分词器类名
    tokenizer_class = ("LayoutLMv2Tokenizer", "LayoutLMv2TokenizerFast")
    # 初始化函数，接受图像处理器和标记器作为参数
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        # 如果参数 kwargs 中包含 feature_extractor，则发出警告，因为该参数已经被弃用
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            # 将 feature_extractor 参数弹出，赋值给 feature_extractor 变量
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果没有指定 image_processor，则使用 feature_extractor
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果没有指定 image_processor，则抛出数值错误
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果没有指定 tokenizer，则抛出数值错误
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用基类的初始化函数，传入 image_processor 和 tokenizer 作为参数
        super().__init__(image_processor, tokenizer)

    # 调用函``数，接受如下参数：
    def __call__(
        self,
        images,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
        # ...
    
    # 获取溢出的图像，接受 images 和 overflow_to_sample_mapping 作为参数
    def get_overflowing_images(self, images, overflow_to_sample_mapping):
        # 如果存在溢出，则确保每个 input_ids 样本都映射到其对应的图像
        images_with_overflow = []
        for sample_idx in overflow_to_sample_mapping:
            images_with_overflow.append(images[sample_idx])

        # 如果图像列表长度与 overflow_to_sample_mapping 列表长度不一致，则抛出数值错误
        if len(images_with_overflow) != len(overflow_to_sample_mapping):
            raise ValueError(
                "Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got"
                f" {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}"
            )

        # 返回包含溢出图像的列表
        return images_with_overflow

    # 批量解码函数，将所有参数转发给 PreTrainedTokenizer 的 batch_decode 方法
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 将参数转发到 PreTrainedTokenizer 的 decode 方法
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型的输入名称列表
    @property
    def model_input_names(self):
        return ["input_ids", "bbox", "token_type_ids", "attention_mask", "image"]

    # 返回特征提取器类
    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    # 返回特征提取器
    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
```