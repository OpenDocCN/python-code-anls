# `.\transformers\models\bridgetower\processing_bridgetower.py`

```
# 定义了一个名为 BridgeTowerProcessor 的类，用于处理 BridgeTower 相关的数据
class BridgeTowerProcessor(ProcessorMixin):
    """
    Constructs a BridgeTower processor which wraps a Roberta tokenizer and BridgeTower image processor into a single
    processor.

    [`BridgeTowerProcessor`] offers all the functionalities of [`BridgeTowerImageProcessor`] and
    [`RobertaTokenizerFast`]. See the docstring of [`~BridgeTowerProcessor.__call__`] and
    [`~BridgeTowerProcessor.decode`] for more information.

    Args:
        image_processor (`BridgeTowerImageProcessor`):
            An instance of [`BridgeTowerImageProcessor`]. The image processor is a required input.
        tokenizer (`RobertaTokenizerFast`):
            An instance of ['RobertaTokenizerFast`]. The tokenizer is a required input.
    """
    
    # 类属性，用于描述类的属性
    attributes = ["image_processor", "tokenizer"]
    # 类属性，指定图像处理器的类名
    image_processor_class = "BridgeTowerImageProcessor"
    # 类属性，指定 tokenizer 的类名
    tokenizer_class = ("RobertaTokenizer", "RobertaTokenizerFast")

    # 类的初始化方法，接受图像处理器和 tokenizer 作为参数
    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    # 类的调用方法，用于处理图像和文本数据
    def __call__(
        self,
        images,  # 图像数据
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,  # 文本数据
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略
        truncation: Union[bool, str, TruncationStrategy] = None,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到的长度
        return_token_type_ids: Optional[bool] = None,  # 是否返回标记类型的 ID
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的标记
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记的掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回长度
        verbose: bool = True,  # 是否显示详细信息
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        **kwargs,  # 其它参数
    ) -> BatchEncoding:
        """
        This method uses [`BridgeTowerImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`RobertaTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        # 使用 BridgeTowerImageProcessor 对象的 __call__ 方法准备图像数据，使用 RobertaTokenizerFast 对象的 __call__ 方法准备文本数据
        encoding = self.tokenizer(
            text=text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            return_tensors=return_tensors,
            **kwargs,
        )
        # 添加 pixel_values 和 pixel_mask 到 encoding 字典中
        encoding_image_processor = self.image_processor(
            images, return_tensors=return_tensors, do_normalize=True, do_center_crop=True, **kwargs
        )
        encoding.update(encoding_image_processor)

        return encoding

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        # 将所有参数转发给 RobertaTokenizerFast 对象的 batch_decode 方法
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        # 将所有参数转发给 RobertaTokenizerFast 对象的 decode 方法
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        # 获取 tokenizer 和 image_processor 的模型输入名称，并将它们合并去重后返回
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```