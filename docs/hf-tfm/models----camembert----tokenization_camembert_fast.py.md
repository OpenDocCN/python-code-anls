# `.\transformers\models\camembert\tokenization_camembert_fast.py`

```py
# 定义了编码格式为 UTF-8 的注释
# 版权声明，列出了版权方和许可证信息
# 依赖项：导入所需的模块和函数
# 导入 os 模块，用于与操作系统交互
# 从 shutil 模块中导入 copyfile 函数，用于复制文件
# 从 typing 模块导入 List、Optional 和 Tuple，用于类型提示

# 从 tokenization_utils 模块导入 AddedToken 类
# 从 tokenization_utils_fast 模块导入 PreTrainedTokenizerFast 类
# 从 utils 模块导入 is_sentencepiece_available 函数和 logging 函数

# 检查是否安装了 sentencepiece 库
if is_sentencepiece_available():
    # 如果已安装，从 tokenization_camembert 模块导入 CamembertTokenizer 类
    from .tokenization_camembert import CamembertTokenizer
else:
    # 如果未安装，将 CamembertTokenizer 设为 None
    CamembertTokenizer = None

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件的名称字典
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "camembert-base": "https://huggingface.co/camembert-base/resolve/main/sentencepiece.bpe.model",
    },
    "tokenizer_file": {
        "camembert-base": "https://huggingface.co/camembert-base/resolve/main/tokenizer.json",
    },
}

# 定义预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "camembert-base": 512,
}

# 定义表示词片段连接符的字符串常量
SPIECE_UNDERLINE = "▁"

# 定义 CamembertTokenizerFast 类，继承自 PreTrainedTokenizerFast 类
class CamembertTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" CamemBERT tokenizer (backed by HuggingFace's *tokenizers* library). Adapted from
    [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
"""
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (`List[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`):
            Additional special tokens used by the tokenizer.
    """
    # 初始化函数，设置词汇文件、分词器文件和特殊标记等参数
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=["<s>NOTUSED", "</s>NOTUSED", "<unk>NOTUSED"],
        **kwargs,
    ):
        # 如果 mask_token 是字符串类型，则将其转换为 AddedToken 对象
        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token
        # 调用父类的初始化函数，设置各种特殊标记
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        # 设置词汇文件路径
        self.vocab_file = vocab_file

    # 判断是否可以保存慢速分词器
    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    # 构建带有特殊标记的输入序列
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An CamemBERT sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        # 如果只有一个输入序列，则在开头和结尾添加特殊标记
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 如果有两个输入序列，则在开头和结尾添加特殊标记，并在两个序列之间添加分隔符
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    # 从序列中创建 token 类型 ID
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. CamemBERT, like
        RoBERTa, does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # Define the separator token ID
        sep = [self.sep_token_id]
        # Define the classification token ID
        cls = [self.cls_token_id]

        # If there is no second sequence provided
        if token_ids_1 is None:
            # Return a list of zeros representing the mask for the first sequence
            return len(cls + token_ids_0 + sep) * [0]
        # If both sequences are provided
        # Return a list of zeros representing the mask for both sequences concatenated
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Check if the fast tokenizer can save the vocabulary for a slow tokenizer
        if not self.can_save_slow_tokenizer:
            # Raise an error if it cannot
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # Check if the save directory exists
        if not os.path.isdir(save_directory):
            # Log an error if the directory does not exist
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            # Return nothing
            return
        # Define the output vocabulary file path
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # If the output vocabulary file is different from the current vocabulary file
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            # Copy the current vocabulary file to the output vocabulary file
            copyfile(self.vocab_file, out_vocab_file)

        # Return the path to the saved vocabulary file
        return (out_vocab_file,)
```