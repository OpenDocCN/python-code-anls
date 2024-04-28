# `.\transformers\models\markuplm\processing_markuplm.py`

```py
# 导入必要的模块和类
from typing import Optional, Union
# 从 HuggingFace 的文件工具中导入张量类型
from ...file_utils import TensorType
# 从处理工具中导入处理器混合类
from ...processing_utils import ProcessorMixin
# 从基础的标记化工具中导入批次编码、填充策略和截断策略
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TruncationStrategy

# 创建一个 MarkupLM 处理器，结合了 MarkupLM 特征提取器和 MarkupLM 分词器
class MarkupLMProcessor(ProcessorMixin):
    r"""
    构建一个 MarkupLM 处理器，将 MarkupLM 特征提取器和 MarkupLM 分词器组合成一个处理器。

    [`MarkupLMProcessor`] 提供了准备模型数据所需的所有功能。

    首先使用 [`MarkupLMFeatureExtractor`] 从一个或多个 HTML 字符串中提取节点和对应的 xpath。
    然后，这些节点和 xpath 被提供给 [`MarkupLMTokenizer`] 或 [`MarkupLMTokenizerFast`]，将它们转换为
    标记级别的 `input_ids`、`attention_mask`、`token_type_ids`、`xpath_tags_seq` 和 `xpath_subs_seq`。

    参数:
        feature_extractor (`MarkupLMFeatureExtractor`):
            [`MarkupLMFeatureExtractor`] 的一个实例。特征提取器是必需的输入。
        tokenizer (`MarkupLMTokenizer` 或 `MarkupLMTokenizerFast`):
            [`MarkupLMTokenizer`] 或 [`MarkupLMTokenizerFast`] 的一个实例。分词器是必需的输入。
        parse_html (`bool`, *可选*, 默认为 `True`):
            是否使用 `MarkupLMFeatureExtractor` 解析 HTML 字符串为节点和对应的 xpath。
    """

    # 特征提取器类
    feature_extractor_class = "MarkupLMFeatureExtractor"
    # 分词器类
    tokenizer_class = ("MarkupLMTokenizer", "MarkupLMTokenizerFast")
    # 是否解析 HTML
    parse_html = True

    # 调用处理器的方法
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
```  
    def convert_tokens_to_string(self, tokens: List[str], batch_idx: Optional[int] = None) -> List[str]:
            """
            Convert a set of tokens (strings) in a single example or batch to a single string.
    
            Args:
                tokens (:obj:`List[str]`):
                    The tokens to join in a single string.
                batch_idx (:obj:`int`, `optional`):
                    The index of the example in the batch to decode tokens to string. This is used to handle batch inputs,
                    where this method is called on multiple examples simultaneously. If not provided, tokens from all examples
                    are joined together.
    
            Returns:
                :obj:`List[str]`: The list of strings obtained by joining the tokens.
    
            Examples::
    
                >>> from transformers import PreTrainedTokenizerFast
                >>> tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-cased")
                >>> tokens = ["Hello", "world", "!"]
                >>> tokenizer.convert_tokens_to_string(tokens)
                'Hello world!'
    
                >>> tokens_batch = [["Hello", "world", "!"], ["My", "name", "is", "Alice"]]
                >>> tokenizer.convert_tokens_to_string(tokens_batch, batch_idx=1)
                'My name is Alice'
            """
            # 调用 tokenizer 的方法将一组 tokens 转换为一个字符串
            # 如果提供了 batch_idx，则仅将索引为 batch_idx 的示例的 tokens 转换为字符串，否则将所有示例的 tokens 合并为一个字符串
            return self._tokenizer.convert_tokens_to_string(tokens, batch_idx=batch_idx)
    # 定义一个解码方法，将所有参数传递给 TrOCRTokenizer 的 `~PreTrainedTokenizer.decode` 方法。请参考该方法的文档字符串以获取更多信息。
    def decode(self, *args, **kwargs):
        # 调用 tokenizer 对象的 decode 方法，并将所有参数传递给它
        return self.tokenizer.decode(*args, **kwargs)

    # 定义一个属性，返回 tokenizer 对象的模型输入名称列表
    @property
    def model_input_names(self):
        # 获取 tokenizer 对象的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 返回模型输入名称列表
        return tokenizer_input_names
```