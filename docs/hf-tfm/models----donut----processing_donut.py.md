# `.\models\donut\processing_donut.py`

```
# 定义编码格式为 UTF-8
# 版权声明，声明代码版权归 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本进行许可
# 除非依法需要或书面同意，否则不得使用此文件
# 您可以在以下网址获得许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 本软件根据许可证“按原样”提供，不提供任何形式的明示或暗示保证
# 包括但不限于适销性和特定用途适用性的保证
# 请查阅许可证获取更多信息

"""
Donut 的处理器类。
"""

import re  # 导入正则表达式模块
import warnings  # 导入警告模块
from contextlib import contextmanager  # 导入上下文管理器

from ...processing_utils import ProcessorMixin  # 导入处理工具混合类


class DonutProcessor(ProcessorMixin):
    """
    构建一个 Donut 处理器，将 Donut 图像处理器和 XLMRoBERTa 分词器封装到单个处理器中。

    [`DonutProcessor`] 提供了 [`DonutImageProcessor`] 和 [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`] 的所有功能。
    有关更多信息，请参见 [`~DonutProcessor.__call__`] 和 [`~DonutProcessor.decode`]。

    Args:
        image_processor ([`DonutImageProcessor`], *optional*):
            [`DonutImageProcessor`] 的实例。图像处理器是必需的输入。
        tokenizer ([`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`], *optional*):
            [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`] 的实例。分词器是必需的输入。
    """

    attributes = ["image_processor", "tokenizer"]  # 属性列表
    image_processor_class = "AutoImageProcessor"  # 图像处理器类
    tokenizer_class = "AutoTokenizer"  # 分词器类

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None  # 特征提取器初始化为 None
        if "feature_extractor" in kwargs:  # 如果参数中包含特征提取器
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )  # 发出警告
            feature_extractor = kwargs.pop("feature_extractor")  # 弹出特征提取器

        image_processor = image_processor if image_processor is not None else feature_extractor  # 如果图像处理器不为空，则使用参数中的图像处理器，否则使用特征提取器
        if image_processor is None:  # 如果图像处理器为空
            raise ValueError("You need to specify an `image_processor`.")  # 抛出值错误
        if tokenizer is None:  # 如果分词器为空
            raise ValueError("You need to specify a `tokenizer`.")  # 抛出值错误

        super().__init__(image_processor, tokenizer)  # 调用父类的构造函数，初始化图像处理器和分词器
        self.current_processor = self.image_processor  # 当前处理器为图像处理器
        self._in_target_context_manager = False  # 是否在目标上下文管理器中的标志位初始化为 False
    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to AutoImageProcessor's
        [`~AutoImageProcessor.__call__`] and returns its output. If used in the context
        [`~DonutProcessor.as_target_processor`] this method forwards all its arguments to DonutTokenizer's
        [`~DonutTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more information.
        """
        # 当在正常模式下使用时，将所有参数转发到 AutoImageProcessor 的 [`~AutoImageProcessor.__call__`] 中，并返回其输出
        # 如果在上下文 [`~DonutProcessor.as_target_processor`] 中使用，此方法将所有参数转发到 DonutTokenizer 的 [`~DonutTokenizer.__call__`] 中
        # 有关更多信息，请参考上述两种方法的文档字符串
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
            inputs = self.image_processor(images, *args, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to DonutTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        # 此方法将其所有参数转发到 DonutTokenizer 的 [`~PreTrainedTokenizer.batch_decode`] 中
        # 有关更多信息，请参考此方法的文档字符串
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to DonutTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        # 此方法将其所有参数转发到 DonutTokenizer 的 [`~PreTrainedTokenizer.decode`] 中
        # 有关更多信息，请参考此方法的文档字符串
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning TrOCR.
        """
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your images inputs, or in a separate call."
        )
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.image_processor
        self._in_target_context_manager = False
    # 将 token 序列转换为有序的 JSON 格式
    def token2json(self, tokens, is_inner_value=False, added_vocab=None):
        """
        Convert a (generated) token sequence into an ordered JSON format.
        """
        # 如果未提供添加的词汇表，则默认为空字典
        if added_vocab is None:
            added_vocab = self.tokenizer.get_added_vocab()

        # 初始化输出字典
        output = {}

        # 循环直到 tokens 为空
        while tokens:
            # 查找起始标记
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            # 如果找不到起始标记，则跳出循环
            if start_token is None:
                break
            # 获取起始标记中的键名
            key = start_token.group(1)
            # 转义键名中的特殊字符
            key_escaped = re.escape(key)

            # 查找结束标记
            end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
            # 如果找不到结束标记，则将起始标记从 tokens 中移除
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                # 获取结束标记，并转义其中的特殊字符
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                # 查找起始标记和结束标记之间的内容
                content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
                # 如果找到内容，则处理
                if content is not None:
                    # 获取内容并去除首尾空白
                    content = content.group(1).strip()
                    # 如果内容中包含起始和结束标记，则为非叶节点
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        # 递归调用 token2json 函数处理非叶节点的值
                        value = self.token2json(content, is_inner_value=True, added_vocab=added_vocab)
                        # 如果值存在，则加入输出字典中
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # 叶节点
                        # 初始化键对应的值为列表
                        output[key] = []
                        # 遍历内容中的叶节点，并添加到对应键的值列表中
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            # 处理特殊标记
                            if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        # 如果值列表只有一个元素，则将其转换为单个值
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                # 更新 tokens，移除处理过的内容
                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                # 如果 tokens 以 "<sep/>" 开头，则为非叶节点
                if tokens[:6] == r"<sep/>":  # 非叶节点
                    return [output] + self.token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

        # 如果输出字典不为空，则返回包含字典的列表（用于处理内部值）
        if len(output):
            return [output] if is_inner_value else output
        # 如果输出字典为空，则返回包含 token 序列的字典
        else:
            return [] if is_inner_value else {"text_sequence": tokens}

    # 返回特征提取器类的属性
    @property
    def feature_extractor_class(self):
        # 发出警告，表明 feature_extractor_class 将在 v5 中被移除，建议使用 image_processor_class
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        # 返回图像处理器类
        return self.image_processor_class

    # 返回特征提取器的属性
    @property
    def feature_extractor(self):
        # 发出警告，表明 feature_extractor 将在 v5 中被移除，建议使用 image_processor
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        # 返回图像处理器
        return self.image_processor
```