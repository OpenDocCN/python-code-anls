# `.\transformers\models\mpnet\tokenization_mpnet_fast.py`

```py
# 设置文件编码格式为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. team，Microsoft Corporation 所有
# 版权声明，版权归 NVIDIA CORPORATION 所有。保留所有权利。
# 根据 Apache 许可证第 2 版进行许可
# 除非受适用法律要求或书面同意，否则不得使用此文件
# 您可以从以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非受适用法律要求，否则根据"原样"基础发布软件
# 没有任何明示或暗示的担保或条件，详见许可证
# 查看特定语言的权限和限制
"""MPNet 的快速分词类。"""

# 导入所需库和模块
import json
from typing import List, Optional, Tuple

from tokenizers import normalizers

# 导入父类中的特定内容
from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_mpnet import MPNetTokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/mpnet-base": "https://huggingface.co/microsoft/mpnet-base/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "microsoft/mpnet-base": "https://huggingface.co/microsoft/mpnet-base/resolve/main/tokenizer.json",
    },
}

# 定义预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/mpnet-base": 512,
}

# 定义预训练模型的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/mpnet-base": {"do_lower_case": True},
}

# 创建 MPNetTokenizerFast 类，基于 WordPiece 分词
class MPNetTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" MPNet tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    # 定义一个函数，用于初始化模型的 tokenizer
    Args:
        vocab_file (`str`):
            包含词汇表的文件路径。
        do_lower_case (`bool`, *optional*, 默认为 `True`):
            是否在进行分词时将输入转换为小写。
        bos_token (`str`, *optional*, 默认为 `"<s>"`):
            在预训练期间用作序列开头的特殊标记，可以用作序列分类器的标记。
    
            <Tip>
    
            在使用特殊标记构建序列时，实际用作序列开头的标记是 `cls_token`。
    
            </Tip>
    
        eos_token (`str`, *optional*, 默认为 `"</s>"`):
            序列结束的特殊标记。
    
            <Tip>
    
            在使用特殊标记构建序列时，实际用作序列结尾的标记是 `sep_token`。
    
            </Tip>
    
        sep_token (`str`, *optional*, 默认为 `"</s>"`):
            分隔符标记，在构建包含多个序列的序列时使用，例如用于序列分类或问题回答中的文本和问题。
            也用作使用特殊标记构建的序列的最后一个标记。
        cls_token (`str`, *optional*, 默认为 `"<s>"`):
            用于序列分类（整个序列的分类而不是每个标记的分类）时使用的分类器标记。
            在使用特殊标记构建序列时，它是序列的第一个标记。
        unk_token (`str`, *optional*, 默认为 `"[UNK]"`):
            未知标记，词汇表中没有的标记会被设置为该标记。
        pad_token (`str`, *optional*, 默认为 `"<pad>"`):
            用于填充的标记，例如在对不同长度的序列进行批处理时使用。
        mask_token (`str`, *optional*, 默认为 `"<mask>"`):
            用于掩码值的标记。在使用掩码语言建模进行模型训练时使用，模型将尝试预测该标记。
        tokenize_chinese_chars (`bool`, *optional*, 默认为 `True`):
            是否对中文字符进行分词。对于日文，可能应该将此选项关闭（参见[此问题](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否去除所有重音符号。如果未指定此选项，则将由 `lowercase` 的值来确定（与原始 BERT 中的行为一致）。
    """
    
    # 定义一些全局常量，包含有关词汇表文件名、预训练模型的词汇表文件映射、初始化配置和模型输入大小的信息
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = MPNetTokenizer
    # 模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化函数，用于创建一个新的Tokenizer对象
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="[UNK]",
        pad_token="<pad>",
        mask_token="<mask>",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        # 如果bos_token是字符串，则将其转换为AddedToken对象，不去除空格
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 如果eos_token是字符串，则将其转换为AddedToken对象，不去除空格
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 如果sep_token是字符串，则将其转换为AddedToken对象，不去除空格
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        # 如果cls_token是字符串，则将其转换为AddedToken对象，不去除空格
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        # 如果unk_token是字符串，则将其转换为AddedToken对象，不去除空格
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 如果pad_token是字符串，则将其转换为AddedToken对象，不去除空格
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # 如果mask_token是字符串，则将其转换为AddedToken对象，去除左侧空格，不去除右侧空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 调用父类的构造函数，初始化Tokenizer对象
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        # 获取当前Tokenizer对象的前处理器状态
        pre_tok_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 如果前处理器状态中的小写处理参数与当前参数不一致，或者去除重音符参数与当前参数不一致
        if (
            pre_tok_state.get("lowercase", do_lower_case) != do_lower_case
            or pre_tok_state.get("strip_accents", strip_accents) != strip_accents
        ):
            # 获取前处理器类
            pre_tok_class = getattr(normalizers, pre_tok_state.pop("type"))
            # 更新前处理器状态中的小写处理参数和去除重音符参数
            pre_tok_state["lowercase"] = do_lower_case
            pre_tok_state["strip_accents"] = strip_accents
            # 创建新的前处理器对象
            self.backend_tokenizer.normalizer = pre_tok_class(**pre_tok_state)

        # 设置当前对象的小写处理参数
        self.do_lower_case = do_lower_case

    # 获取mask_token属性的方法
    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.

        MPNet tokenizer has a special mask token to be usable in the fill-mask pipeline. The mask token will greedily
        comprise the space before the *
    @mask_token.setter
    def mask_token(self, value):
        """
        覆盖默认的掩码标记行为，使其在标记前吸收空格。

        这是为了与基于 MPNet 的先前使用的所有模型保持向后兼容性而必需的。
        """
        # 掩码标记表现为普通单词，即在其前包含空格
        # 因此我们将 lstrip 设置为 True
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传递的两个序列创建一个用于序列对分类任务的掩码。MPNet 不使用标记类型 id，因此返回一个零列表。

        Args:
            token_ids_0 (`List[int]`):
                id 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 id 列表，用于序列对任务

        Returns:
            `List[int]`: 零列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```