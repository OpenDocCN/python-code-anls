# `.\models\llava\processing_llava.py`

```
    # 设置文件编码为UTF-8
    # 版权声明归HuggingFace Inc.团队所有，使用Apache License, Version 2.0授权
    # 除非符合许可证要求或书面同意，否则不得使用此文件
    # 获取许可证的副本，请访问http://www.apache.org/licenses/LICENSE-2.0
    # 本软件根据"原样"基础分发，不提供任何明示或暗示的担保或条件
    # 请参阅许可证以获取特定语言的权限和限制

    """
    Llava的处理器类。
    """

    from typing import List, Optional, Union

    from ...feature_extraction_utils import BatchFeature
    from ...image_utils import ImageInput
    from ...processing_utils import ProcessorMixin
    from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
    from ...utils import TensorType

    class LlavaProcessor(ProcessorMixin):
        """
        构建一个Llava处理器，将Llava图像处理器和Llava分词器封装到单个处理器中。

        [`LlavaProcessor`] 提供了 [`CLIPImageProcessor`] 和 [`LlamaTokenizerFast`] 的所有功能。查看
        [`~LlavaProcessor.__call__`] 和 [`~LlavaProcessor.decode`] 获取更多信息。

        Args:
            image_processor ([`CLIPImageProcessor`], *optional*):
                图像处理器，必需的输入。
            tokenizer ([`LlamaTokenizerFast`], *optional*):
                分词器，必需的输入。
        """

        attributes = ["image_processor", "tokenizer"]
        image_processor_class = "CLIPImageProcessor"
        tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

        def __init__(self, image_processor=None, tokenizer=None):
            super().__init__(image_processor, tokenizer)

        def __call__(
            self,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            images: ImageInput = None,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = None,
            max_length=None,
            return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        ):
            """
            调用处理器进行文本和图像处理。

            Args:
                text: 输入文本或预分词输入的列表。
                images: 输入的图像数据。
                padding: 是否进行填充的策略。
                truncation: 是否进行截断的策略。
                max_length: 最大长度限制。
                return_tensors: 返回的张量类型。

            Returns:
                处理后的文本和图像特征。
            """

        def batch_decode(self, *args, **kwargs):
            """
            此方法将所有参数转发到LlamaTokenizerFast的[`~PreTrainedTokenizer.batch_decode`]。请参阅该方法的文档字符串获取更多信息。
            """
            return self.tokenizer.batch_decode(*args, **kwargs)

        # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        # 将所有参数转发给 LlamaTokenizerFast 的 decode 方法，并返回其结果
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        # 获取 tokenizer 的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取 image_processor 的模型输入名称列表
        image_processor_input_names = self.image_processor.model_input_names
        # 将 tokenizer 和 image_processor 的输入名称合并成一个无重复元素的列表，并返回
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```