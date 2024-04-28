# `.\models\funnel\tokenization_funnel_fast.py`

```py
# 设置脚本编码为 UTF-8

# 引入必要的库
import json  # 导入 JSON 库，用于处理 JSON 格式的数据
from typing import List, Optional, Tuple  # 引入类型提示相关的库

# 从 tokenizers 库中引入 normalizers 模块
from tokenizers import normalizers

# 从 tokenization_utils_fast 模块中引入 PreTrainedTokenizerFast 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast

# 引入 logging 模块中的 logger 对象
from ...utils import logging

# 从当前目录下的 tokenization_funnel 模块中引入 FunnelTokenizer 类
from .tokenization_funnel import FunnelTokenizer

# 获取 logging 模块中的 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件名的字典映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 定义模型名称列表
_model_names = [
    "small",
    "small-base",
    "medium",
    "medium-base",
    "intermediate",
    "intermediate-base",
    "large",
    "large-base",
    "xlarge",
    "xlarge-base",
]

# 定义预训练词汇文件映射的字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "funnel-transformer/small": "https://huggingface.co/funnel-transformer/small/resolve/main/vocab.txt",
        "funnel-transformer/small-base": "https://huggingface.co/funnel-transformer/small-base/resolve/main/vocab.txt",
        "funnel-transformer/medium": "https://huggingface.co/funnel-transformer/medium/resolve/main/vocab.txt",
        "funnel-transformer/medium-base": (
            "https://huggingface.co/funnel-transformer/medium-base/resolve/main/vocab.txt"
        ),
        "funnel-transformer/intermediate": (
            "https://huggingface.co/funnel-transformer/intermediate/resolve/main/vocab.txt"
        ),
        "funnel-transformer/intermediate-base": (
            "https://huggingface.co/funnel-transformer/intermediate-base/resolve/main/vocab.txt"
        ),
        "funnel-transformer/large": "https://huggingface.co/funnel-transformer/large/resolve/main/vocab.txt",
        "funnel-transformer/large-base": "https://huggingface.co/funnel-transformer/large-base/resolve/main/vocab.txt",
        "funnel-transformer/xlarge": "https://huggingface.co/funnel-transformer/xlarge/resolve/main/vocab.txt",
        "funnel-transformer/xlarge-base": (
            "https://huggingface.co/funnel-transformer/xlarge-base/resolve/main/vocab.txt"
        ),
    },
    # 定义一个名为tokenizer_file的字典，用于存储不同模型的tokenizer文件的URL
    "tokenizer_file": {
        # 小型funnel-transformer模型的tokenizer文件URL
        "funnel-transformer/small": "https://huggingface.co/funnel-transformer/small/resolve/main/tokenizer.json",
        # 小型基础funnel-transformer模型的tokenizer文件URL
        "funnel-transformer/small-base": (
            "https://huggingface.co/funnel-transformer/small-base/resolve/main/tokenizer.json"
        ),
        # 中型funnel-transformer模型的tokenizer文件URL
        "funnel-transformer/medium": "https://huggingface.co/funnel-transformer/medium/resolve/main/tokenizer.json",
        # 中型基础funnel-transformer模型的tokenizer文件URL
        "funnel-transformer/medium-base": (
            "https://huggingface.co/funnel-transformer/medium-base/resolve/main/tokenizer.json"
        ),
        # 中级funnel-transformer模型的tokenizer文件URL
        "funnel-transformer/intermediate": (
            "https://huggingface.co/funnel-transformer/intermediate/resolve/main/tokenizer.json"
        ),
        # 中级基础funnel-transformer模型的tokenizer文件URL
        "funnel-transformer/intermediate-base": (
            "https://huggingface.co/funnel-transformer/intermediate-base/resolve/main/tokenizer.json"
        ),
        # 大型funnel-transformer模型的tokenizer文件URL
        "funnel-transformer/large": "https://huggingface.co/funnel-transformer/large/resolve/main/tokenizer.json",
        # 大型基础funnel-transformer模型的tokenizer文件URL
        "funnel-transformer/large-base": (
            "https://huggingface.co/funnel-transformer/large-base/resolve/main/tokenizer.json"
        ),
        # 超大型funnel-transformer模型的tokenizer文件URL
        "funnel-transformer/xlarge": "https://huggingface.co/funnel-transformer/xlarge/resolve/main/tokenizer.json",
        # 超大型基础funnel-transformer模型的tokenizer文件URL
        "funnel-transformer/xlarge-base": (
            "https://huggingface.co/funnel-transformer/xlarge-base/resolve/main/tokenizer.json"
        ),
    },
# 定义一个字典，键为预训练模型的名称，值为512，用于存储预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {f"funnel-transformer/{name}": 512 for name in _model_names}
# 定义一个字典，键为预训练模型的名称，值为一个包含 "do_lower_case" 键的字典，用于存储初始化配置信息
PRETRAINED_INIT_CONFIGURATION = {f"funnel-transformer/{name}": {"do_lower_case": True} for name in _model_names}

# 定义 FunnelTokenizerFast 类，继承自 PreTrainedTokenizerFast 类
class FunnelTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" Funnel Transformer tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
```  
    Args:
        vocab_file (`str`):
            词汇表文件名。
        do_lower_case (`bool`, *可选*, 默认为 `True`):
            在进行标记化时是否将输入转换为小写。
        unk_token (`str`, *可选*, 默认为 `"<unk>"`):
            未知标记。如果一个标记不在词汇表中，无法将其转换为ID，则将其设置为此标记。
        sep_token (`str`, *可选*, 默认为 `"<sep>"`):
            分隔符标记，用于从多个序列构建序列，例如用于序列分类的两个序列或用于文本和问题的问题回答。它还用作使用特殊标记构建的序列的最后一个标记。
        pad_token (`str`, *可选*, 默认为 `"<pad>"`):
            用于填充的标记，例如在批处理不同长度的序列时。
        cls_token (`str`, *可选*, 默认为 `"<cls>"`):
            分类器标记，在进行序列分类时使用（整个序列的分类，而不是每个标记的分类）。它是使用特殊标记构建的序列的第一个标记。
        mask_token (`str`, *可选*, 默认为 `"<mask>"`):
            用于屏蔽值的标记。这是在使用遮盖语言建模训练此模型时使用的标记。这是模型将尝试预测的标记。
        clean_text (`bool`, *可选*, 默认为 `True`):
            是否在标记化之前清理文本，方法是删除所有控制字符并将所有空格替换为经典空格。
        tokenize_chinese_chars (`bool`, *可选*, 默认为 `True`):
            是否标记化中文字符。对于日语，可能应该禁用此选项（请参见[this issue](https://github.com/huggingface/transformers/issues/328)）。
        bos_token (`str`, `optional`, 默认为 `"<s>"`):
            句子起始标记。
        eos_token (`str`, `optional`, 默认为 `"</s>"`):
            句子结束标记。
        strip_accents (`bool`, *可选*):
            是否删除所有重音符号。如果未指定此选项，则将由 `lowercase` 的值决定（与原始BERT一样）。
        wordpieces_prefix (`str`, *可选*, 默认为 `"##"`):
            子词的前缀。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇文件名映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 慢速标记化器类
    slow_tokenizer_class = FunnelTokenizer
    # 最大模型输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 分类器标记类型ID，默认为2
    cls_token_type_id: int = 2
    # 初始化函数，用于创建一个新的FunnelTokenizer对象
    def __init__(
        self,
        vocab_file=None,  # 词汇表文件路径，默认为None
        tokenizer_file=None,  # 分词器文件路径，默认为None
        do_lower_case=True,  # 是否将输入文本转换为小写，默认为True
        unk_token="<unk>",  # 未知词标记，默认为"<unk>"
        sep_token="<sep>",  # 分隔标记，默认为"<sep>"
        pad_token="<pad>",  # 填充标记，默认为"<pad>"
        cls_token="<cls>",  # 分类标记，默认为"<cls>"
        mask_token="<mask>",  # 掩码标记，默认为"<mask>"
        bos_token="<s>",  # 开始标记，默认为"<s>"
        eos_token="</s>",  # 结束标记，默认为"</s>"
        clean_text=True,  # 是否清理文本，默认为True
        tokenize_chinese_chars=True,  # 是否分词中文字符，默认为True
        strip_accents=None,  # 是否去除重音，默认为None
        wordpieces_prefix="##",  # 词片段前缀，默认为"##"
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化函数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            clean_text=clean_text,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            wordpieces_prefix=wordpieces_prefix,
            **kwargs,
        )

        # 获取标准化器状态
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查是否需要重新配置标准化器
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            # 获取标准化器类
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            # 更新标准化器状态
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            # 重新配置标准化器
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 更新实例属性
        self.do_lower_case = do_lower_case

    # 从序列或序列对构建模型输入，添加特殊标记
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        通过连接并添加特殊标记，为序列分类任务构建模型输入。Funnel序列的格式如下：

        - 单个序列：`[CLS] X [SEP]`
        - 序列对：`[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                需要添加特殊标记的ID列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个序列ID列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊标记的[输入ID](../glossary#input-ids)列表。
        """
        # 初始化输出列表
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # 如果有第二个序列，则添加其ID列表及特殊标记
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output
```py  
    # 从给定序列中创建用于序列对分类任务的token type ids
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传入的两个序列创建一个用于序列对分类任务的掩码。Funnel Transformer的序列对掩码格式如下：

        ```
        2 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | 第一个序列       | 第二个序列     |
        ```py

        如果`token_ids_1`为`None`，则此方法仅返回掩码的第一部分(0)。

        Args:
            token_ids_0 (`List[int]`):
                ID列表。
            token_ids_1 (`List[int]`, *可选*):
                用于序列对的可选第二个ID列表。

        Returns:
            `List[int]`: 根据给定序列的[token type IDs](../glossary#token-type-ids)列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls) * [self.cls_token_type_id] + len(token_ids_0 + sep) * [0]
        return len(cls) * [self.cls_token_type_id] + len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # 从transformers.models.bert.tokenization_bert_fast.BertTokenizerFast.save_vocabulary复制而来
    # 保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```