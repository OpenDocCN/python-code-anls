# `.\models\markuplm\processing_markuplm.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. team 所有，采用 Apache License 2.0
# 如果不遵循许可证，不得使用此文件中的代码
# 可以在上述链接获取许可证的副本
# 根据适用法律或书面同意，本软件根据"原样"分发，无任何明示或暗示的担保或条件
# 详见许可证，限制软件使用的特定语言和条件
"""
Processor class for MarkupLM.
"""
# 导入所需的类型和联合类型
from typing import Optional, Union

# 从文件工具中导入所需的类
from ...file_utils import TensorType
# 从处理工具中导入所需的类
from ...processing_utils import ProcessorMixin
# 从基本标记化工具中导入批处理编码、填充策略和截断策略
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TruncationStrategy


class MarkupLMProcessor(ProcessorMixin):
    r"""
    Constructs a MarkupLM processor which combines a MarkupLM feature extractor and a MarkupLM tokenizer into a single
    processor.

    [`MarkupLMProcessor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`MarkupLMFeatureExtractor`] to extract nodes and corresponding xpaths from one or more HTML strings.
    Next, these are provided to [`MarkupLMTokenizer`] or [`MarkupLMTokenizerFast`], which turns them into token-level
    `input_ids`, `attention_mask`, `token_type_ids`, `xpath_tags_seq` and `xpath_subs_seq`.

    Args:
        feature_extractor (`MarkupLMFeatureExtractor`):
            An instance of [`MarkupLMFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`MarkupLMTokenizer` or `MarkupLMTokenizerFast`):
            An instance of [`MarkupLMTokenizer`] or [`MarkupLMTokenizerFast`]. The tokenizer is a required input.
        parse_html (`bool`, *optional*, defaults to `True`):
            Whether or not to use `MarkupLMFeatureExtractor` to parse HTML strings into nodes and corresponding xpaths.
    """

    # 定义特征提取器和标记化器的类名
    feature_extractor_class = "MarkupLMFeatureExtractor"
    tokenizer_class = ("MarkupLMTokenizer", "MarkupLMTokenizerFast")
    # 是否解析 HTML 字符串来生成节点和对应的 XPath，默认为 True
    parse_html = True

    def __call__(
        self,
        html_strings=None,
        nodes=None,
        xpaths=None,
        node_labels=None,
        questions=None,
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
        This method first forwards the `html_strings` argument to [`~MarkupLMFeatureExtractor.__call__`]. Next, it
        passes the `nodes` and `xpaths` along with the additional arguments to [`~MarkupLMTokenizer.__call__`] and
        returns the output.

        Optionally, one can also provide a `text` argument which is passed along as first sequence.

        Please refer to the docstring of the above two methods for more information.
        """
        # 首先，根据 parse_html 参数处理 HTML 字符串，生成 nodes 和 xpaths
        if self.parse_html:
            # 如果 parse_html 设置为 True，则必须传入 HTML 字符串
            if html_strings is None:
                raise ValueError("Make sure to pass HTML strings in case `parse_html` is set to `True`")

            # 如果 parse_html 设置为 True，则不能同时传入 nodes、xpaths 或 node_labels
            if nodes is not None or xpaths is not None or node_labels is not None:
                raise ValueError(
                    "Please don't pass nodes, xpaths nor node labels in case `parse_html` is set to `True`"
                )

            # 使用特征提取器处理 HTML 字符串，获取 nodes 和 xpaths
            features = self.feature_extractor(html_strings)
            nodes = features["nodes"]
            xpaths = features["xpaths"]
        else:
            # 如果 parse_html 设置为 False，则不能传入 HTML 字符串
            if html_strings is not None:
                raise ValueError("You have passed HTML strings but `parse_html` is set to `False`.")
            # 如果 parse_html 设置为 False，则必须传入 nodes 和 xpaths
            if nodes is None or xpaths is None:
                raise ValueError("Make sure to pass nodes and xpaths in case `parse_html` is set to `False`")

        # 其次，应用分词器处理输入数据
        if questions is not None and self.parse_html:
            # 如果同时传入了 questions 并且 parse_html 为 True，则将 questions 转为列表形式
            if isinstance(questions, str):
                questions = [questions]  # add batch dimension (as the feature extractor always adds a batch dimension)

        # 使用分词器处理输入数据，返回编码后的结果
        encoded_inputs = self.tokenizer(
            text=questions if questions is not None else nodes,
            text_pair=nodes if questions is not None else None,
            xpaths=xpaths,
            node_labels=node_labels,
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

        # 返回编码后的输入数据
        return encoded_inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        # 调用分词器的 batch_decode 方法，将所有参数转发给它
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 定义一个方法 `decode`，该方法将其所有参数转发给 `TrOCRTokenizer` 的 `PreTrainedTokenizer.decode` 方法。
    # 请参考 `PreTrainedTokenizer.decode` 方法的文档字符串获取更多信息。
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        # 调用 `TrOCRTokenizer` 对象的 `decode` 方法，并返回其结果
        return self.tokenizer.decode(*args, **kwargs)

    # 定义一个属性 `model_input_names`
    @property
    def model_input_names(self):
        # 获取 `self.tokenizer` 的 `model_input_names` 属性值，并返回
        tokenizer_input_names = self.tokenizer.model_input_names
        return tokenizer_input_names
```