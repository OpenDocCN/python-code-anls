# `.\transformers\models\perceiver\tokenization_perceiver.py`

```py
# 设置代码编码格式为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，未经授权不得使用此文件
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
# 除非适用法律要求或书面同意，否则按"原样"（AS IS）分发软件，不附带任何担保或条件，无论是明示还是默示的
# 请查看许可证，了解具体语言对权限和限制
""" Perceiver 的 Tokenizer 类 """

# 导入需要的库
from typing import Dict, List, Optional, Tuple
# 从 tokenization_utils.py 中导入 AddedToken 和 PreTrainedTokenizer 类
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
# 从 utils.py 中导入 logging 模块
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

class PerceiverTokenizer(PreTrainedTokenizer):
    """
    构造一个 Perceiver Tokenizer。Perceiver 简单地使用原始字节 utf-8 编码。

    该 Tokenizer 继承自 PreTrainedTokenizer 类，其中包含大多数主要方法。用户应该参考该超类以获取有关这些方法的更多信息。

    参数:
        pad_token (`str`, *可选*, 默认为 `"[PAD]"`):
            用于填充的 token，例如在对不同长度的序列进行批处理时使用。
        bos_token (`str`, *可选*, 默认为 `"[BOS]"`):
            BOS token（在词汇表中保留，但实际上没有使用）。
        eos_token (`str`, *可选*, 默认为 `"[EOS]"`):
            序列结束的 token（在词汇表中保留，但实际上没有使用）。

            <提示>

            在使用特殊 token 构建序列时，这不是用于表示序列结束的 token。用于表示序列结束的 token 是 `sep_token`。

            </提示>

        mask_token (`str`, *可选*, 默认为 `"[MASK]"`):
            MASK token，用于掩码语言建模。
        cls_token (`str`, *可选*, 默认为 `"[CLS]"`):
            CLS token（在词汇表中保留，但实际上没有使用）。
        sep_token (`str`, *可选*, 默认为 `"[SEP]"`):
            分隔符 token，在从两个序列构建序列时使用。
    """

    # 定义模型输入的名称
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        mask_token="[MASK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=2048,
        **kwargs,
    # 该函数用于设置特殊Token的值，并将其添加到模型的Token表中
    def __init__(
        self,
        pad_token: Union[str, AddedToken] = '[PAD]',
        bos_token: Union[str, AddedToken] = '[CLS]',
        eos_token: Union[str, AddedToken] = '[SEP]',
        mask_token: Union[str, AddedToken] = '[MASK]',
        cls_token: Union[str, AddedToken] = '[CLS]',
        sep_token: Union[str, AddedToken] = '[SEP]',
        model_max_length: int = 512,
        **kwargs
    ) -> None:
        # 如果输入的pad/bos/eos/mask/cls/sep_token是字符串，则将其封装成AddedToken对象
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        mask_token = AddedToken(mask_token, lstrip=False, rstrip=False) if isinstance(mask_token, str) else mask_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
    
        # 设置字符编码的最大范围为2^8 = 256
        self._utf_vocab_size = 2**8  
    
        # 将特殊Token添加到模型的Token表中
        self._added_tokens_decoder: Dict[str, int] = {
            0: pad_token,
            1: bos_token,
            2: eos_token,
            3: mask_token,
            4: cls_token,
            5: sep_token,
        }
        self._num_special_tokens = len(self._added_tokens_decoder)
        # 调用父类的构造函数
        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            cls_token=cls_token,
            sep_token=sep_token,
            model_max_length=model_max_length,
            **kwargs,
        )
    
    # 获取模型的词汇表
    def get_vocab(self) -> Dict[str, int]:
        # 创建一个空的词汇表
        vocab = {}
        # 遍历0到255的整数(2^8-1)
        for i in range(self._utf_vocab_size):
            # 将整数转换为对应的字符
            token = chr(i)
            # 将字符及其对应的索引添加到词汇表中
            vocab[token] = i + self._num_special_tokens
        # 将模型添加的特殊Token也添加到词汇表中
        vocab.update(self.added_tokens_encoder)
        return vocab
    
    # 获取模型的词汇表大小
    @property
    def vocab_size(self):
        return self._utf_vocab_size
    
    # 获取输入序列中特殊Token的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从不带特殊标记的标记列表中检索序列 ID。当使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个可选的序列对应的 ID 列表。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                标记列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 正常情况：一些特殊标记
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        为序列分类任务从一个序列或序列对构建模型输入。序列的格式如下：

        - 单个序列：`[CLS] X [SEP]`
        - 序列对：`[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                序列对的可选第二个 ID 列表。

        Returns:
            `List[int]`: 带有适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        else:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]

    def _tokenize(self, text: str) -> List[str]:
        """接受一个字符串作为输入，并返回单词/子单词的字符串列表（标记）"""
        tokens = [chr(i) for i in text.encode("utf-8")]
        return tokens

    def _convert_token_to_id(self, token):
        """使用词汇表将标记（str）转换为 ID。"""
        if len(token) != 1:
            token_id = self.unk_token_id
        else:
            token_id = ord(token) + self._num_special_tokens
        return token_id

    def _convert_id_to_token(self, index):
        """使用词汇表将索引（整数）转换为标记（str）。"""
        token = chr(index - self._num_special_tokens)
        return token

    # TODO @ArthurZ refactor this as well....
    # 将一系列的标记（tokens）（字符串）转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        # 初始化一个空字节串
        bstring = b""
        # 遍历每个标记
        for token in tokens:
            # 如果标记在添加的标记编码器中
            if token in self.added_tokens_encoder:
                # 将标记转换为 UTF-8 编码的字节串
                tok_string = str(token).encode("utf-8")
            else:
                # 将标记转换为其对应的 ASCII 字符的字节串
                tok_string = bytes([ord(token)])
            # 将转换后的字节串添加到 bstring 中
            bstring += tok_string
        # 将字节串解码为 UTF-8 编码的字符串，处理可能出现的解码错误
        string = bstring.decode("utf-8", errors="replace")
        # 返回转换后的字符串
        return string

    # PerceiverTokenizer 没有词汇表文件
    # 保存词汇表的方法，但由于 PerceiverTokenizer 没有词汇表文件，因此返回空元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()
```