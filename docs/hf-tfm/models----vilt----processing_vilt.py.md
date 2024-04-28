# `.\transformers\models\vilt\processing_vilt.py`

```py
# coding=utf-8
# 版权归 HuggingFace Inc. 团队所有
#
# 根据 Apache License, Version 2.0 进行许可
# 除非符合许可的适用法律或经书面同意，否则按 "AS IS" 基础分发该软件
# 本证书详细信息可在以下网址找到：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 引入警告模块
import warnings
# 引入类型提示模块
from typing import List, Optional, Union

# 引入处理工具模块
from ...processing_utils import ProcessorMixin
# 引入编码模块
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
# 引入工具模块
from ...utils import TensorType


# 定义 ViltProcessor 类
class ViltProcessor(ProcessorMixin):
    # 定义类属性
    attributes = ["image_processor", "tokenizer"]
    # 定义图像处理器类属性
    image_processor_class = "ViltImageProcessor"
    # 定义令牌化器类属性
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    # 定义初始化方法
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # 初始化特征提取器为 None
        feature_extractor = None
        # 如果 kwargs 中包含 feature_extractor 则显示警告
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            # 把 feature_extractor 的值赋值给特征提取器
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果 image_processor 不为 None，则把 image_processor 的值赋值给 image_processor 变量
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果 image_processor 为 None，则抛出 ValueError 异常
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果 tokenizer 为 None，则抛出 ValueError 异常
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类的初始化方法，并把 image_processor 和 tokenizer 当做参数传入
        super().__init__(image_processor, tokenizer)
        # 设置当前处理器为 image_processor
        self.current_processor = self.image_processor
    def __call__(
        self,
        images,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
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
    ) -> BatchEncoding:
        """
        This method uses [`ViltImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.
        
        Please refer to the docstring of the above two methods for more information.
        """
        # 使用ViltImageProcessor类的__call__方法为模型准备图像数据
        encoding = self.tokenizer(
            text=text,  # 文本输入
            add_special_tokens=add_special_tokens,  # 是否添加特殊标记
            padding=padding,  # 是否进行填充
            truncation=truncation,  # 是否进行截断
            max_length=max_length,  # 最大长度
            stride=stride,  # 步长
            pad_to_multiple_of=pad_to_multiple_of,  # 填充到指定长度的倍数
            return_token_type_ids=return_token_type_ids,  # 是否返回标记类型id
            return_attention_mask=return_attention_mask,  # 是否返回注意力掩码
            return_overflowing_tokens=return_overflowing_tokens,  # 是否返回溢出标记
            return_special_tokens_mask=return_special_tokens_mask,  # 是否返回特殊标记掩码
            return_offsets_mapping=return_offsets_mapping,  # 是否返回偏移映射
            return_length=return_length,  # 是否返回长度
            verbose=verbose,  # 是否显示详细信息
            return_tensors=return_tensors,  # 返回张量
            **kwargs,
        )
        # 添加像素值和像素掩码
        encoding_image_processor = self.image_processor(images, return_tensors=return_tensors)
        encoding.update(encoding_image_processor)  # 更新encoding字典

        return encoding  # 返回编码后的数据

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)  # 调用BertTokenizerFast类的batch_decode方法进行解码

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)  # 调用BertTokenizerFast类的decode方法进行解码

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))  # 返回去重后的模型输入名称列表

    @property
    # 这个方法返回特征提取器的类对象
    def feature_extractor_class(self):
        # 发出一个警告,提示 "feature_extractor_class" 已被弃用,将在 v5 中移除,应该改用 "image_processor_class" 
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        # 返回图像处理器的类对象
        return self.image_processor_class
    
    # 这个属性返回特征提取器的对象
    @property
    def feature_extractor(self):
        # 发出一个警告,提示 "feature_extractor" 已被弃用,将在 v5 中移除,应该改用 "image_processor"
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        # 返回图像处理器的对象
        return self.image_processor
```