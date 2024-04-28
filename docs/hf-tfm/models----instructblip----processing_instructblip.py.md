# `.\models\instructblip\processing_instructblip.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

"""
Processor class for InstructBLIP. Largely copy of Blip2Processor with addition of a tokenizer for the Q-Former.
"""

# 导入所需的模块
import os
from typing import List, Optional, Union

# 导入自定义模块
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType
from ..auto import AutoTokenizer

# 定义 InstructBlipProcessor 类，继承 ProcessorMixin 类
class InstructBlipProcessor(ProcessorMixin):
    r"""
    Constructs an InstructBLIP processor which wraps a BLIP image processor and a LLaMa/T5 tokenizer into a single
    processor.

    [`InstructBlipProcessor`] offers all the functionalities of [`BlipImageProcessor`] and [`AutoTokenizer`]. See the
    docstring of [`~BlipProcessor.__call__`] and [`~BlipProcessor.decode`] for more information.

    Args:
        image_processor (`BlipImageProcessor`):
            An instance of [`BlipImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
        qformer_tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The Q-Former tokenizer is a required input.
    """

    # 定义类属性
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    # 初始化方法，接受 image_processor, tokenizer, qformer_tokenizer 三个参数
    def __init__(self, image_processor, tokenizer, qformer_tokenizer):
        # 调用父类的初始化方法
        super().__init__(image_processor, tokenizer)

        # 添加 QFormer tokenizer
        self.qformer_tokenizer = qformer_tokenizer
    # 定义一个方法，用于处理输入数据并生成模型所需的输入
    def __call__(
        # images参数用于传入图像数据，类型为ImageInput，默认为None
        images: ImageInput = None,
        # text参数用于传入文本数据，类型为Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]，默认为None
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        # add_special_tokens参数用于指定是否添加特殊标记，默认为True
        add_special_tokens: bool = True,
        # padding参数用于指定是否进行填充操作，默认为False
        padding: Union[bool, str, PaddingStrategy] = False,
        # truncation参数用于指定是否进行截断操作，默认为None
        truncation: Union[bool, str, TruncationStrategy] = None,
        # max_length参数用于指定最大长度，默认为None
        max_length: Optional[int] = None,
        # stride参数用于指定步长，默认为0
        stride: int = 0,
        # pad_to_multiple_of参数用于指定填充到的长度，默认为None
        pad_to_multiple_of: Optional[int] = None,
        # return_attention_mask参数用于指定是否返回注意力掩码，默认为None
        return_attention_mask: Optional[bool] = None,
        # return_overflowing_tokens参数用于指定是否返回溢出的标记，默认为False
        return_overflowing_tokens: bool = False,
        # return_special_tokens_mask参数用于指定是否返回特殊标记的掩码，默认为False
        return_special_tokens_mask: bool = False,
        # return_offsets_mapping参数用于指定是否返回偏移映射，默认为False
        return_offsets_mapping: bool = False,
        # return_token_type_ids参数用于指定是否返回标记类型ID，默认为False
        return_token_type_ids: bool = False,
        # return_length参数用于指定是否返回长度，默认为False
        return_length: bool = False,
        # verbose参数用于指定是否显示详细信息，默认为True
        verbose: bool = True,
        # return_tensors参数用于指定是否返回张量，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # **kwargs用于接收额外的关键字参数
        **kwargs,
    # 定义一个方法，用于准备模型的输入特征
    def __call__(self, images: Optional[Union[Image.Image, List[Image.Image]]] = None, text: Optional[Union[str, List[str]]] = None,
                 add_special_tokens: bool = True, padding: Union[bool, str] = True, truncation: Union[bool, str] = True,
                 max_length: Optional[int] = None, stride: int = 0, pad_to_multiple_of: Optional[int] = None,
                 return_attention_mask: Optional[bool] = None, return_overflowing_tokens: Optional[bool] = None,
                 return_special_tokens_mask: Optional[bool] = None, return_offsets_mapping: Optional[bool] = None,
                 return_token_type_ids: Optional[bool] = None, return_length: Optional[bool] = None,
                 verbose: bool = True, return_tensors: Optional[str] = None, **kwargs) -> BatchFeature:
        """
        This method uses [`BlipImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        # 如果既没有图片也没有文本，则抛出数值错误
        if images is None and text is None:
            raise ValueError("You have to specify at least images or text.")

        # 创建一个空的 BatchFeature 对象
        encoding = BatchFeature()

        # 如果有文本输入
        if text is not None:
            # 使用 tokenizer 处理文本
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
            # 更新 encoding 字典
            encoding.update(text_encoding)
            # 使用 qformer_tokenizer 处理文本
            qformer_text_encoding = self.qformer_tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
            # 将 qformer_tokenizer 处理后的结果加入 encoding 字典
            encoding["qformer_input_ids"] = qformer_text_encoding.pop("input_ids")
            encoding["qformer_attention_mask"] = qformer_text_encoding.pop("attention_mask")

        # 如果有图片输入
        if images is not None:
            # 使用 image_processor 处理图片
            image_encoding = self.image_processor(images, return_tensors=return_tensors)
            # ��新 encoding 字典
            encoding.update(image_encoding)

        # 返回处理后的特征
        return encoding

    # 从 transformers.models.blip.processing_blip.BlipProcessor.batch_decode 复制，将 BertTokenizerFast 替换为 PreTrainedTokenizer
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        # 调用 PreTrainedTokenizer 的 batch_decode 方法，并返回结果
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 从transformers.models.blip.processing_blip.BlipProcessor.decode复制过来，使用BertTokenizerFast->PreTrainedTokenizer进行解码
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        # 调用PreTrainedTokenizer的decode方法，并返回结果
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # 从transformers.models.blip.processing_blip.BlipProcessor.model_input_names复制过来
    def model_input_names(self):
        # 获取tokenizer的模型输入名称
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取image_processor的模型输入名称
        image_processor_input_names = self.image_processor.model_input_names
        # 返回去重后的tokenizer和image_processor的模型输入名称列表
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # 重写以将Q-Former tokenizer保存在单独的文件夹中
    def save_pretrained(self, save_directory, **kwargs):
        # 如果save_directory是文件而不是目录，则抛出异常
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        # 创建目录save_directory，如果目录已存在则不做任何操作
        os.makedirs(save_directory, exist_ok=True)
        # 设置Q-Former tokenizer的保存路径
        qformer_tokenizer_path = os.path.join(save_directory, "qformer_tokenizer")
        # 保存Q-Former tokenizer到指定路径
        self.qformer_tokenizer.save_pretrained(qformer_tokenizer_path)
        # 调用父类的save_pretrained方法保存模型到save_directory，并返回结果
        return super().save_pretrained(save_directory, **kwargs)

    # 重写以从单独的文件夹加载Q-Former tokenizer
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 从指定路径加载Q-Former tokenizer
        qformer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="qformer_tokenizer")
        # 从预训练模型名称或路径获取参数
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        # 将加载的Q-Former tokenizer添加到参数列表中
        args.append(qformer_tokenizer)
        # 创建并返回类的实例
        return cls(*args)
```