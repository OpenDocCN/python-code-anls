# `.\models\donut\processing_donut.py`

```py
# 设置编码格式为 UTF-8
# 版权声明：2022 年由 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本，只有在符合许可证的情况下才能使用此文件
# 您可以从以下链接获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，不附带任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。
"""
Donut 的处理器类。
"""
import re  # 导入正则表达式模块
import warnings  # 导入警告模块
from contextlib import contextmanager  # 导入上下文管理器

from ...processing_utils import ProcessorMixin  # 导入处理器混合类


class DonutProcessor(ProcessorMixin):
    r"""
    构造一个 Donut 处理器，将 Donut 图像处理器和 XLMRoBERTa 分词器封装成一个单一处理器。

    [`DonutProcessor`] 提供 [`DonutImageProcessor`] 和 [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`] 的所有功能。
    详见 [`~DonutProcessor.__call__`] 和 [`~DonutProcessor.decode`] 获取更多信息。

    Args:
        image_processor ([`DonutImageProcessor`], *可选*):
            [`DonutImageProcessor`] 的实例。图像处理器是必需的输入。
        tokenizer ([`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`], *可选*):
            [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`] 的实例。分词器是必需的输入。
    """

    attributes = ["image_processor", "tokenizer"]  # 类属性列表
    image_processor_class = "AutoImageProcessor"  # 图像处理器类名
    tokenizer_class = "AutoTokenizer"  # 分词器类名

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        if "feature_extractor" in kwargs:
            # 警告：`feature_extractor` 参数已弃用，并将在 v5 中删除，请使用 `image_processor` 替代。
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果 kwargs 中包含 `feature_extractor`，则将其赋给 feature_extractor 变量
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果未指定 image_processor，则引发 ValueError 异常
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果未指定 tokenizer，则引发 ValueError 异常
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类 ProcessorMixin 的构造函数，传入 image_processor 和 tokenizer
        super().__init__(image_processor, tokenizer)
        # 设置当前处理器为 image_processor
        self.current_processor = self.image_processor
        # 标记目标上下文管理器未启动
        self._in_target_context_manager = False
    def __call__(self, *args, **kwargs):
        """
        当在正常模式下使用时，该方法将所有参数转发给 AutoImageProcessor 的 [`~AutoImageProcessor.__call__`] 并返回其输出。
        如果在上下文 [`~DonutProcessor.as_target_processor`] 中使用，则将所有参数转发给 DonutTokenizer 的 [`~DonutTokenizer.__call__`]。
        请参阅上述两个方法的文档了解更多信息。
        """
        # 对于向后兼容性
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            images = args[0]
            args = args[1:]

        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        if images is not None:
            # 使用图像处理器处理图像和其他参数
            inputs = self.image_processor(images, *args, **kwargs)
        if text is not None:
            # 使用分词器处理文本和其他参数
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            # 将标签添加到输入字典中
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        将所有参数转发给 DonutTokenizer 的 [`~PreTrainedTokenizer.batch_decode`] 方法。请参阅该方法的文档了解更多信息。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        将所有参数转发给 DonutTokenizer 的 [`~PreTrainedTokenizer.decode`] 方法。请参阅该方法的文档了解更多信息。
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        临时设置处理输入的分词器。用于在微调 TrOCR 时对标签进行编码。
        """
        warnings.warn(
            "`as_target_processor` 已弃用，并将在 Transformers 的 v5 中移除。您可以通过在常规 `__call__` 方法的参数 `text` 中处理您的标签（在与图像输入相同的调用中或在单独的调用中）。"
        )
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.image_processor
        self._in_target_context_manager = False
    def token2json(self, tokens, is_inner_value=False, added_vocab=None):
        """
        Convert a (generated) token sequence into an ordered JSON format.

        Args:
            tokens (str): The token sequence to convert into JSON format.
            is_inner_value (bool, optional): Indicates if the function is processing inner values. Defaults to False.
            added_vocab (list, optional): List of added vocabulary tokens. Defaults to None.

        Returns:
            dict or list: Ordered JSON format representing the token sequence.

        Converts a sequence of tokens into a structured JSON format. Handles both leaf and non-leaf nodes
        in the token sequence recursively.
        """
        if added_vocab is None:
            added_vocab = self.tokenizer.get_added_vocab()

        output = {}

        while tokens:
            # Locate the start token in the token sequence
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            key_escaped = re.escape(key)

            # Find the corresponding end token for the current start token
            end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                # Extract content between start and end tokens
                content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        # Recursively convert inner token sequence to JSON
                        value = self.token2json(content, is_inner_value=True, added_vocab=added_vocab)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        # Split content into leaf nodes based on separator "<sep/>"
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                # Remove processed tokens from the sequence
                tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    # Return a list with current output and recursively processed tokens
                    return [output] + self.token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

        # Handle cases where no output is generated
        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}

    @property
    def feature_extractor_class(self):
        """
        Property accessor for deprecated feature_extractor_class.

        Returns:
            class: The image processor class.

        Warns:
            FutureWarning: This property is deprecated and will be removed in v5.
                           Use `image_processor_class` instead.
        """
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    @property
    def feature_extractor(self):
        """
        Property accessor for deprecated feature_extractor.

        Returns:
            object: The image processor instance.

        Warns:
            FutureWarning: This property is deprecated and will be removed in v5.
                           Use `image_processor` instead.
        """
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
```