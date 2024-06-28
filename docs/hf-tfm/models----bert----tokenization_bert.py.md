# `.\models\bert\tokenization_bert.py`

```
# 指定编码为 UTF-8

# 版权声明，版权归Google AI Language Team和HuggingFace Inc.团队所有，使用Apache License 2.0授权
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0

# 如果适用法律要求或书面同意，软件将按“原样”分发，不提供任何明示或暗示的保证或条件
# 请参阅许可证以了解详细信息

"""Bert的标记化类。"""

# 导入所需模块
import collections  # 导入collections模块
import os  # 导入os模块
import unicodedata  # 导入unicodedata模块
from typing import List, Optional, Tuple  # 导入类型提示所需的模块

# 从tokenization_utils.py中导入预训练的标记器和一些辅助函数
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace

# 导入日志记录功能
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称，这里是一个包含词汇的文本文件
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练词汇文件的映射，这里假设只有一个vocab_file键，对应的值是vocab.txt文件名
PRETRAINED_VOCAB_FILES_MAP = {
    {
        "vocab_file": {
            "google-bert/bert-base-uncased": "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt",
            "google-bert/bert-large-uncased": "https://huggingface.co/google-bert/bert-large-uncased/resolve/main/vocab.txt",
            "google-bert/bert-base-cased": "https://huggingface.co/google-bert/bert-base-cased/resolve/main/vocab.txt",
            "google-bert/bert-large-cased": "https://huggingface.co/google-bert/bert-large-cased/resolve/main/vocab.txt",
            "google-bert/bert-base-multilingual-uncased": (
                "https://huggingface.co/google-bert/bert-base-multilingual-uncased/resolve/main/vocab.txt"
            ),
            "google-bert/bert-base-multilingual-cased": "https://huggingface.co/google-bert/bert-base-multilingual-cased/resolve/main/vocab.txt",
            "google-bert/bert-base-chinese": "https://huggingface.co/google-bert/bert-base-chinese/resolve/main/vocab.txt",
            "google-bert/bert-base-german-cased": "https://huggingface.co/google-bert/bert-base-german-cased/resolve/main/vocab.txt",
            "google-bert/bert-large-uncased-whole-word-masking": (
                "https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking/resolve/main/vocab.txt"
            ),
            "google-bert/bert-large-cased-whole-word-masking": (
                "https://huggingface.co/google-bert/bert-large-cased-whole-word-masking/resolve/main/vocab.txt"
            ),
            "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad": (
                "https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
            ),
            "google-bert/bert-large-cased-whole-word-masking-finetuned-squad": (
                "https://huggingface.co/google-bert/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
            ),
            "google-bert/bert-base-cased-finetuned-mrpc": (
                "https://huggingface.co/google-bert/bert-base-cased-finetuned-mrpc/resolve/main/vocab.txt"
            ),
            "google-bert/bert-base-german-dbmdz-cased": "https://huggingface.co/google-bert/bert-base-german-dbmdz-cased/resolve/main/vocab.txt",
            "google-bert/bert-base-german-dbmdz-uncased": (
                "https://huggingface.co/google-bert/bert-base-german-dbmdz-uncased/resolve/main/vocab.txt"
            ),
            "TurkuNLP/bert-base-finnish-cased-v1": (
                "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/vocab.txt"
            ),
            "TurkuNLP/bert-base-finnish-uncased-v1": (
                "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/vocab.txt"
            ),
            "wietsedv/bert-base-dutch-cased": (
                "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/vocab.txt"
            ),
        }
    }
    
    
    注释：
    
    # vocab_file 是一个包含不同 BERT 模型及其对应词汇表 URL 的字典
    {
        "google-bert/bert-base-uncased": "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt",  # Google BERT base uncased 模型的词汇表 URL
        "google-bert/bert-large-uncased": "https://huggingface.co/google-bert/bert-large-uncased/resolve/main/vocab.txt",  # Google BERT large uncased 模型的词汇表 URL
        "google-bert/bert-base-cased": "https://huggingface.co/google-bert/bert-base-cased/resolve/main/vocab.txt",  # Google BERT base cased 模型的词汇表 URL
        "google-bert/bert-large-cased": "https://huggingface.co/google-bert/bert-large-cased/resolve/main/vocab.txt",  # Google BERT large cased 模型的词汇表 URL
        "google-bert/bert-base-multilingual-uncased": (
            "https://huggingface.co/google-bert/bert-base-multilingual-uncased/resolve/main/vocab.txt"  # Google BERT base 多语言 uncased 模型的词汇表 URL
        ),
        "google-bert/bert-base-multilingual-cased": "https://huggingface.co/google-bert/bert-base-multilingual-cased/resolve/main/vocab.txt",  # Google BERT base 多语言 cased 模型的词汇表 URL
        "google-bert/bert-base-chinese": "https://huggingface.co/google-bert/bert-base-chinese/resolve/main/vocab.txt",  # Google BERT base 中文模型的词汇表 URL
        "google-bert/bert-base-german-cased": "https://huggingface.co/google-bert/bert-base-german-cased/resolve/main/vocab.txt",  # Google BERT base 德语 cased 模型的词汇表 URL
        "google-bert/bert-large-uncased-whole-word-masking": (
            "https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking/resolve/main/vocab.txt"  # Google BERT large uncased 整词屏蔽模型的词汇表 URL
        ),
        "google-bert/bert-large-cased-whole-word-masking": (
            "https://huggingface.co/google-bert/bert-large-cased-whole-word-masking/resolve/main/vocab.txt"  # Google BERT large cased 整词屏蔽模型的词汇表 URL
        ),
        "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad": (
            "https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"  # Google BERT large uncased 整词屏蔽模型（在 SQuAD 上微调）的词汇表 URL
        ),
        "google-bert/bert-large-cased-whole-word-masking-finetuned-squad": (
            "https://huggingface.co/google-bert/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"  # Google BERT large cased 整词屏蔽模型（在 SQuAD 上微调）的词汇表 URL
        ),
        "google-bert/bert-base-cased-finetuned-mrpc": (
            "https://huggingface.co/google-bert/bert-base-cased-finetuned-mrpc/resolve/main/vocab.txt"  # Google BERT base cased 模型（在 MRPC 数据集上微调）的词汇表 URL
        ),
        "google-bert/bert-base-german-dbmdz-cased": "https://huggingface.co/google-bert/bert-base-german-dbmdz-cased/resolve/main/vocab.txt",  # Google BERT base 德语（由 DBMDZ 组织提供，cased）模型的词汇表 URL
        "google-bert/bert-base-german-dbmdz-uncased": (
            "https://huggingface.co/google-bert/bert-base-german-dbmdz-uncased/resolve/main/vocab.txt"  # Google BERT base 德语（由 DBMDZ 组织提供，uncased）模型的词汇表 URL
        ),
        "TurkuNLP/bert-base-finnish-cased-v1": (
            "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/vocab.txt"  # TurkuNLP 提供的芬兰语 cased BERT base v1 模型的词汇表 URL
        ),
        "TurkuNLP/bert-base-finnish-uncased-v1": (
            "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/vocab.txt"  # TurkuNLP 提供的芬兰语 uncased BERT base v1 模型的词汇表 URL
        ),
        "wietsedv/bert-base-dutch-cased": (
            "
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google-bert/bert-base-uncased": 512,  # 设置预训练模型的位置嵌入尺寸
    "google-bert/bert-large-uncased": 512,
    "google-bert/bert-base-cased": 512,
    "google-bert/bert-large-cased": 512,
    "google-bert/bert-base-multilingual-uncased": 512,
    "google-bert/bert-base-multilingual-cased": 512,
    "google-bert/bert-base-chinese": 512,
    "google-bert/bert-base-german-cased": 512,
    "google-bert/bert-large-uncased-whole-word-masking": 512,
    "google-bert/bert-large-cased-whole-word-masking": 512,
    "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad": 512,
    "google-bert/bert-large-cased-whole-word-masking-finetuned-squad": 512,
    "google-bert/bert-base-cased-finetuned-mrpc": 512,
    "google-bert/bert-base-german-dbmdz-cased": 512,
    "google-bert/bert-base-german-dbmdz-uncased": 512,
    "TurkuNLP/bert-base-finnish-cased-v1": 512,
    "TurkuNLP/bert-base-finnish-uncased-v1": 512,
    "wietsedv/bert-base-dutch-cased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "google-bert/bert-base-uncased": {"do_lower_case": True},  # 配置预训练模型初始化参数
    "google-bert/bert-large-uncased": {"do_lower_case": True},
    "google-bert/bert-base-cased": {"do_lower_case": False},
    "google-bert/bert-large-cased": {"do_lower_case": False},
    "google-bert/bert-base-multilingual-uncased": {"do_lower_case": True},
    "google-bert/bert-base-multilingual-cased": {"do_lower_case": False},
    "google-bert/bert-base-chinese": {"do_lower_case": False},
    "google-bert/bert-base-german-cased": {"do_lower_case": False},
    "google-bert/bert-large-uncased-whole-word-masking": {"do_lower_case": True},
    "google-bert/bert-large-cased-whole-word-masking": {"do_lower_case": False},
    "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad": {"do_lower_case": True},
    "google-bert/bert-large-cased-whole-word-masking-finetuned-squad": {"do_lower_case": False},
    "google-bert/bert-base-cased-finetuned-mrpc": {"do_lower_case": False},
    "google-bert/bert-base-german-dbmdz-cased": {"do_lower_case": False},
    "google-bert/bert-base-german-dbmdz-uncased": {"do_lower_case": True},
    "TurkuNLP/bert-base-finnish-cased-v1": {"do_lower_case": False},
    "TurkuNLP/bert-base-finnish-uncased-v1": {"do_lower_case": True},
    "wietsedv/bert-base-dutch-cased": {"do_lower_case": False},
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()  # 创建一个有序字典用于存储词汇表
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()  # 读取词汇文件中的所有行
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")  # 去除每个词汇的换行符
        vocab[token] = index  # 将词汇添加到字典中，键为词汇，值为索引
    return vocab  # 返回加载后的词汇表字典


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()  # 去除文本首尾空白字符
    if not text:
        return []  # 如果文本为空，则返回空列表
    tokens = text.split()  # 使用空格分割文本生成词汇列表
    return tokens  # 返回分割后的词汇列表


class BertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a BERT tokenizer. Based on WordPiece.
    """
    # 从`PreTrainedTokenizer`继承，该类包含大多数主要方法。用户应参考这个超类以获取关于这些方法的更多信息。

    # 参数:
    # vocab_file (`str`):
    #     包含词汇表的文件。
    # do_lower_case (`bool`, *可选*, 默认为 `True`):
    #     在标记化时是否将输入转换为小写。
    # do_basic_tokenize (`bool`, *可选*, 默认为 `True`):
    #     是否在使用WordPiece之前进行基本的标记化。
    # never_split (`Iterable`, *可选*):
    #     在标记化时永远不会分割的一组标记。仅在 `do_basic_tokenize=True` 时有效。
    # unk_token (`str`, *可选*, 默认为 `"[UNK]"`):
    #     未知标记。词汇表中不存在的标记无法转换为ID，并将被设置为此标记。
    # sep_token (`str`, *可选*, 默认为 `"[SEP]"`):
    #     分隔符标记，在构建来自多个序列的序列时使用，例如用于序列分类或用于文本和问题的问题回答。在使用特殊标记构建的序列的最后一个标记也会使用此标记。
    # pad_token (`str`, *可选*, 默认为 `"[PAD]"`):
    #     用于填充的标记，例如在批处理不同长度的序列时使用。
    # cls_token (`str`, *可选*, 默认为 `"[CLS]"`):
    #     分类器标记，在进行序列分类（整个序列的分类而不是每个标记的分类）时使用。在使用特殊标记构建的序列的第一个标记。
    # mask_token (`str`, *可选*, 默认为 `"[MASK]"`):
    #     用于屏蔽值的标记。这是在使用掩蔽语言建模训练模型时使用的标记。模型将尝试预测此标记。
    # tokenize_chinese_chars (`bool`, *可选*, 默认为 `True`):
    #     是否标记化中文字符。
    #     对于日文，这可能应该停用（参见此[问题](https://github.com/huggingface/transformers/issues/328)）。
    # strip_accents (`bool`, *可选*):
    #     是否删除所有重音符号。如果未指定此选项，则将根据 `lowercase` 的值确定（与原始BERT相同）。
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 初始化方法，用于初始化一个Tokenizer对象
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        # 检查给定的词汇文件是否存在，如果不存在则抛出异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表文件到self.vocab中
        self.vocab = load_vocab(vocab_file)
        # 根据加载的词汇表构建从id到token的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 是否进行基本的tokenize操作
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要进行基本tokenize，则初始化BasicTokenizer对象
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        # 初始化WordpieceTokenizer对象，使用加载的词汇表和未知token
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化方法，传递相同的参数和额外的参数
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

    # 属性方法，返回是否进行小写处理的标志位
    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    # 属性方法，返回词汇表的大小
    @property
    def vocab_size(self):
        return len(self.vocab)

    # 方法，返回包含所有词汇和特殊token编码的字典
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # 方法，对文本进行tokenize操作，返回token列表
    def _tokenize(self, text, split_special_tokens=False):
        split_tokens = []
        # 如果需要进行基本tokenize操作
        if self.do_basic_tokenize:
            # 遍历基本tokenizer的tokenize结果
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果token在不分割集合中，则直接加入split_tokens列表
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 否则，使用WordpieceTokenizer对token进行进一步的分词处理，并加入split_tokens列表
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 否则，直接使用WordpieceTokenizer对整个text进行tokenize操作
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    # 方法，根据token获取对应的id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 方法，根据id获取对应的token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)
    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) into a single string by joining them,
        removing '##' and stripping leading/trailing whitespace.

        Args:
            tokens (List[str]): List of tokens to be converted.

        Returns:
            str: The concatenated string of tokens.
        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Builds model inputs from a sequence or a pair of sequences for sequence classification tasks
        by adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (List[int]): List of token IDs for the first sequence.
            token_ids_1 (Optional[List[int]]): Optional list of token IDs for the second sequence.

        Returns:
            List[int]: List of input IDs with the appropriate special tokens added.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves a mask indicating whether each token in the input list is a special token
        (1 for special token, 0 for sequence token). This is used when preparing tokens for a model.

        Args:
            token_ids_0 (List[int]): List of token IDs for the first sequence.
            token_ids_1 (Optional[List[int]]): Optional list of token IDs for the second sequence.
            already_has_special_tokens (bool, optional): Whether the input token lists already include special tokens.

        Returns:
            List[int]: A list of integers representing the mask.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates token type IDs from token lists representing sequences or pairs of sequences.

        Args:
            token_ids_0 (List[int]): List of token IDs for the first sequence.
            token_ids_1 (Optional[List[int]]): Optional list of token IDs for the second sequence.

        Returns:
            List[int]: List of token type IDs.
        """
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs representing the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List representing the token type IDs for the given sequence(s).
        """
        # Define separator and classification tokens
        sep = [self.sep_token_id]  # Separator token ID
        cls = [self.cls_token_id]  # Classification token ID
        
        # If token_ids_1 is None, return a mask with zeros corresponding to the first sequence only
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]  # Create and return mask with zeros
        
        # If token_ids_1 is provided, return a mask with zeros for the first sequence and ones for the second sequence
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Initialize index counter
        index = 0
        
        # Determine vocabulary file path
        if os.path.isdir(save_directory):
            # If save_directory is a directory, construct file path inside the directory
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # Otherwise, treat save_directory as the full file path
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        
        # Write vocabulary to the specified file
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # Iterate through vocabulary items sorted by index
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # Check for non-consecutive indices in the vocabulary
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index  # Update index to current token's index
                writer.write(token + "\n")  # Write token to file
                index += 1  # Increment index for the next token
        
        # Return the path to the saved vocabulary file
        return (vocab_file,)
# 定义一个名为 BasicTokenizer 的类，用于执行基本的分词（如分割标点符号、转换为小写等）。
class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            In some instances we want to skip the basic punctuation splitting so that later tokenization can capture
            the full context of the words, such as contractions.
    """

    # 初始化方法，设置类的属性
    def __init__(
        self,
        do_lower_case=True,          # 是否将输入转换为小写，默认为True
        never_split=None,            # 永远不分割的 token 集合，默认为 None
        tokenize_chinese_chars=True, # 是否分割中文字符，默认为 True
        strip_accents=None,          # 是否去除所有重音符号，默认根据 lowercase 决定
        do_split_on_punc=True,       # 是否在基本标点符号处分割，默认为 True
    ):
        # 如果 never_split 为 None，则设为一个空列表
        if never_split is None:
            never_split = []
        # 设置实例的属性值
        self.do_lower_case = do_lower_case                  # 是否小写化输入
        self.never_split = set(never_split)                 # 永远不分割的 token 集合，转为集合类型
        self.tokenize_chinese_chars = tokenize_chinese_chars # 是否分割中文字符
        self.strip_accents = strip_accents                  # 是否去除重音符号
        self.do_split_on_punc = do_split_on_punc            # 是否在基本标点符号处分割
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 使用 never_split 参数更新当前对象的 never_split 集合（若提供的话）
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，如去除无用空白等
        text = self._clean_text(text)

        # 以下部分是为了支持多语言和中文模型而添加的代码（2018 年 11 月 1 日起）
        # 现在英语模型也应用了这一代码，但由于英语模型未经过中文数据的训练，
        # 这段代码对英语模型基本没有影响（尽管英语词汇表中包含了一些中文单词，
        # 这是因为英语维基百科中包含了一些中文词汇）。
        if self.tokenize_chinese_chars:
            # 对包含中文字符的文本进行特殊处理，分词
            text = self._tokenize_chinese_chars(text)
        # 将文本中的 Unicode 标准化为 NFC 格式（避免同一字符的不同 Unicode 编码被视为不同字符）
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空白符分割文本，得到原始 token 列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        # 遍历每个原始 token
        for token in orig_tokens:
            # 如果 token 不在 never_split 集合中
            if token not in never_split:
                # 如果设置为小写处理，则将 token 转换为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果需要去除重音符号，则执行去除重音符号的操作
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果需要去除重音符号，则执行去除重音符号的操作
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 将处理后的 token 通过标点符号分割函数进一步分割
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 使用空白符重新组合处理后的 token，并分割为最终的输出 token 列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        # 返回最终的输出 token 列表
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 将文本中的字符标准化为 NFD 格式
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 分类
            cat = unicodedata.category(char)
            # 如果字符是非组合型记号（Mn），则跳过
            if cat == "Mn":
                continue
            # 否则将字符添加到输出列表中
            output.append(char)
        # 将输出列表中的字符连接成字符串并返回
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """按照标点符号分割文本。

        Args:
            text (str): 要分割的文本。
            never_split (set): 不应该被分割的文本集合。

        Returns:
            list: 分割后的文本列表。

        """
        # 如果不需要按标点符号分割，或者文本在不分割的集合中，则直接返回原文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号，则作为新词开始
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，根据start_new_word标记将字符添加到当前词列表中
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        # 将列表中的字符列表连接为字符串，并返回分割后的文本列表
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """在每个CJK字符周围添加空格。

        Args:
            text (str): 要处理的文本。

        Returns:
            str: 处理后的文本。

        """
        output = []
        for char in text:
            cp = ord(char)
            # 如果是CJK字符，添加空格前后包裹该字符
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        # 将列表中的字符连接为一个字符串，并返回处理后的文本
        return "".join(output)

    def _is_chinese_char(self, cp):
        """检查CP是否是CJK字符的码点。

        Args:
            cp (int): 要检查的字符的Unicode码点。

        Returns:
            bool: 如果是CJK字符则返回True，否则返回False。

        """
        # 这里的CJK字符定义来自于CJK统一表意文字块的Unicode范围
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
        ):
            return True

        return False

    def _clean_text(self, text):
        """对文本进行无效字符移除和空白字符清理。

        Args:
            text (str): 要清理的文本。

        Returns:
            str: 清理后的文本。

        """
        output = []
        for char in text:
            cp = ord(char)
            # 移除无效字符和控制字符，以及替换空白字符为单个空格
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将列表中的字符连接为一个字符串，并返回清理后的文本
        return "".join(output)
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordpieceTokenizer对象，设置词汇表、未知标记和单词的最大字符数
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """
        # 初始化输出token列表
        output_tokens = []
        # 使用whitespace_tokenize函数将文本分割成单词或标记
        for token in whitespace_tokenize(text):
            # 将token转换为字符列表
            chars = list(token)
            # 如果token的长度超过最大输入字符数，则将未知标记添加到输出token列表中
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            # 初始化标志变量和起始位置
            is_bad = False
            start = 0
            sub_tokens = []
            # 循环直到处理完所有字符
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                # 使用最长匹配算法找到合适的子串
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    # 如果找到了匹配词汇表的子串，则更新当前子串并跳出循环
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                # 如果未找到合适的子串，则标记为无效
                if cur_substr is None:
                    is_bad = True
                    break
                # 将找到的子串添加到sub_tokens列表中
                sub_tokens.append(cur_substr)
                start = end

            # 如果标记为无效，则将未知标记添加到输出token列表中；否则将sub_tokens列表中的token添加到输出token列表中
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        # 返回最终的token列表
        return output_tokens
```