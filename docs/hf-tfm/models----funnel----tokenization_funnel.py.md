# `.\models\funnel\tokenization_funnel.py`

```py
# 定义了代码文件的编码格式为 UTF-8
# 版权声明，指明代码版权归 HuggingFace Inc. 团队所有，采用 Apache License, Version 2.0
# 此函数用于加载指定路径下的词汇表文件，返回一个有序字典表示的词汇表
import collections
import os
import unicodedata
from typing import List, Optional, Tuple

# 从 tokenization_utils 模块中导入需要用到的函数和类
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging

# 获取 logger 对象，用于记录日志
logger = logging.get_logger(__name__)

# 定义词汇表文件名字典，只包含一个键值对，指定了词汇表文件名为 "vocab.txt"
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 模型名称列表，包含了不同规模和基础的 Funnel Transformer 模型
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

# 预训练模型的词汇表文件映射，为每个模型配置了其对应的预训练词汇表下载链接
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
    }
}

# 预训练模型的位置嵌入大小映射，为每个模型配置了其对应的位置嵌入维度为 512
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {f"funnel-transformer/{name}": 512 for name in _model_names}

# 预训练模型的初始化配置映射，为每个模型配置了其对应的初始化配置，这里统一设置了小写处理为 True
PRETRAINED_INIT_CONFIGURATION = {f"funnel-transformer/{name}": {"do_lower_case": True} for name in _model_names}

# 从 transformers.models.bert.tokenization_bert.load_vocab 复制的函数，用于加载词汇表文件到一个有序字典中
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    # 使用 enumerate 函数遍历 tokens 列表，同时获取索引 index 和对应的 token
    for index, token in enumerate(tokens):
        # 去除 token 字符串末尾的换行符 "\n"
        token = token.rstrip("\n")
        # 将处理过的 token 作为键，将其索引 index 作为值存入 vocab 字典中
        vocab[token] = index
    # 返回填充完毕的 vocab 字典
    return vocab
# 从transformers.models.bert.tokenization_bert.whitespace_tokenize复制过来的函数
def whitespace_tokenize(text):
    """对文本进行基本的空白字符清理和分割。"""
    # 去除文本两端的空白字符
    text = text.strip()
    # 如果文本为空，则返回空列表
    if not text:
        return []
    # 使用空白字符分割文本，生成token列表
    tokens = text.split()
    # 返回分割后的token列表
    return tokens


class FunnelTokenizer(PreTrainedTokenizer):
    r"""
    构建一个Funnel Transformer的分词器。基于WordPiece。

    这个分词器继承自[`PreTrainedTokenizer`]，包含大部分主要方法。用户应参考这个超类以获取更多关于这些方法的信息。
    """
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"<sep>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"<cls>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    ```
    
    # Define a constant for the names of vocabulary files
    vocab_files_names = VOCAB_FILES_NAMES
    # Define a constant mapping pretrained model files to their respective vocabulary files
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # Define a constant mapping pretrained model configurations to their initialization configurations
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # Define a constant for the maximum model input sizes based on pretrained positional embeddings sizes
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # Initialize the classifier token type ID to 2
    cls_token_type_id: int = 2
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        bos_token="<s>",
        eos_token="</s>",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        # 检查词汇文件是否存在，若不存在则抛出异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = FunnelTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表
        self.vocab = load_vocab(vocab_file)
        # 根据词汇表创建从编号到标记的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要基本分词，则初始化基本分词器
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 使用词汇表和未知标记初始化词块分词器
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化方法，传递相同的参数
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

    @property
    # 从 transformers.models.bert.tokenization_bert.BertTokenizer.do_lower_case 复制而来
    def do_lower_case(self):
        # 返回基本分词器的小写标记设置
        return self.basic_tokenizer.do_lower_case

    @property
    # 从 transformers.models.bert.tokenization_bert.BertTokenizer.vocab_size 复制而来
    def vocab_size(self):
        # 返回词汇表的大小
        return len(self.vocab)

    # 从 transformers.models.bert.tokenization_bert.BertTokenizer.get_vocab 复制而来
    def get_vocab(self):
        # 返回词汇表和添加的标记编码器的字典
        return dict(self.vocab, **self.added_tokens_encoder)

    # 从 transformers.models.bert.tokenization_bert.BertTokenizer._tokenize 复制而来
    def _tokenize(self, text, split_special_tokens=False):
        # 初始化空列表，用于存储分词后的 token
        split_tokens = []
        # 如果需要进行基础分词处理
        if self.do_basic_tokenize:
            # 使用 basic_tokenizer 对文本进行分词
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果 token 在 never_split 集合中
                if token in self.basic_tokenizer.never_split:
                    # 直接加入 split_tokens
                    split_tokens.append(token)
                else:
                    # 否则对 token 进行 wordpiece 分词处理，并加入 split_tokens
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 否则直接使用 wordpiece_tokenizer 对文本进行分词处理
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        # 返回分词后的 token 列表
        return split_tokens

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 根据 token 在 vocab 中查找对应的 id，如果不存在则返回 unk_token 对应的 id
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 根据 index 在 ids_to_tokens 中查找对应的 token，如果不存在则返回 unk_token
        return self.ids_to_tokens.get(index, self.unk_token)

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将 token 列表连接成一个字符串，并移除特殊标记 " ##"，然后去除首尾空格
        out_string = " ".join(tokens).replace(" ##", "").strip()
        # 返回连接后的字符串
        return out_string

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            # 返回仅包含第一个句子的特殊 token 的输入序列
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 否则返回包含两个句子的特殊 token 的输入序列
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
        """
        Retrieve special tokens mask from the list of token IDs.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs corresponding to the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether the token list is already formatted with special tokens.

        Returns:
            `List[int]`: List of integers representing whether each token is special (1) or not (0).
        """
        # 如果输入序列已经包含特殊 token，则直接返回全零的 mask
        if already_has_special_tokens:
            return [0] * len(token_ids_0)
        # 否则生成一个 mask 列表
        mask = [1] * len(token_ids_0)
        sep = [self.sep_token_id]
        # 将第一个句子的末尾和可能存在的第二个句子的末尾设置为 1，其余为 0
        if token_ids_1 is not None:
            mask += sep + [0] * len(token_ids_1)
        else:
            mask += [0]
        return mask
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        # Check if the token list already has special tokens
        if already_has_special_tokens:
            # If true, delegate to the base class's method to retrieve special token masks
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If there are two token lists (sequence pairs)
        if token_ids_1 is not None:
            # Create a mask with special tokens for both sequences
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # Otherwise, create a mask with special tokens for the single sequence
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A Funnel
        Transformer sequence pair mask has the following format:

        ```
        2 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # Define special tokens
        sep = [self.sep_token_id]  # Separator token ID
        cls = [self.cls_token_id]  # Classification token ID

        # If there is only one sequence
        if token_ids_1 is None:
            # Return token type IDs for the first sequence only
            return len(cls) * [self.cls_token_type_id] + len(token_ids_0 + sep) * [0]

        # If there are two sequences
        # Return token type IDs for both sequences
        return len(cls) * [self.cls_token_type_id] + len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 定义一个方法来保存词汇表到指定的目录和文件名前缀（可选）
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引为0，用于检查词汇索引的连续性
        index = 0
        # 检查保存目录是否已存在
        if os.path.isdir(save_directory):
            # 构建词汇表文件的完整路径，包括目录和文件名前缀
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 若保存目录不存在，则直接使用给定的文件路径作为词汇表文件路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        
        # 打开词汇表文件以写入模式，使用UTF-8编码
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的每个词汇及其索引，按索引排序
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 如果当前词汇的索引不等于预期的索引，记录警告信息
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 将词汇写入文件，并在词汇后添加换行符
                writer.write(token + "\n")
                # 更新索引以保持连续性
                index += 1
        
        # 返回保存的词汇表文件路径的元组
        return (vocab_file,)
# 从transformers.models.b
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 将 `never_split` 参数与实例变量 `never_split` 的集合取并集，以确保不分割指定的 token
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本中的特殊符号和空白
        text = self._clean_text(text)

        # 以下代码块是为了支持多语言和中文模型而添加的，从2018年11月1日开始使用
        # 即使英文模型也会应用这一步骤，尽管它们没有在任何中文数据上训练
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 对文本进行 Unicode 规范化，确保相同字符的不同 Unicode 编码被视为相同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 将文本按空白分割为初始 token
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        # 遍历每个初始 token
        for token in orig_tokens:
            # 如果 token 不在不分割的集合中
            if token not in never_split:
                # 如果需要小写化处理
                if self.do_lower_case:
                    # 将 token 转换为小写
                    token = token.lower()
                    # 如果需要去除重音符号，则去除
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果需要去除重音符号，则去除
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 将处理过的 token 按标点符号进行进一步分割
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将分割后的 token 再次按空白合并为最终的输出 token 列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 对文本进行 Unicode 规范化，确保各种形式的重音符号都能被正确处理
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 分类
            cat = unicodedata.category(char)
            # 如果字符是重音符号，跳过
            if cat == "Mn":
                continue
            # 否则将字符加入到输出列表中
            output.append(char)
        # 将字符列表组合成字符串作为输出
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要根据标点符号拆分，或者指定的文本在不拆分列表中，直接返回文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果当前字符是标点符号，将其作为新的列表项添加到输出中
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，检查是否应该开始新单词
                if start_new_word:
                    output.append([])
                start_new_word = False
                # 将字符添加到当前列表项中
                output[-1].append(char)
            i += 1

        # 将每个列表项中的字符连接成字符串，形成最终的拆分结果
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符是中文字符，将其两侧添加空格后添加到输出中
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        # 将列表转换为字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查给定的码点是否是CJK字符的码点范围
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符是无效字符或控制字符，直接跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空白字符，替换为单个空格，否则保留字符
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将列表转换为字符串并返回
        return "".join(output)
# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordpieceTokenizer对象
        self.vocab = vocab  # 词汇表，用于存储词汇
        self.unk_token = unk_token  # 未知词汇的标记
        self.max_input_chars_per_word = max_input_chars_per_word  # 单词最大字符数限制

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
        # 初始化输出的token列表
        output_tokens = []
        # 对text进行空白符分割，得到每个token
        for token in whitespace_tokenize(text):
            # 将token转换为字符列表
            chars = list(token)
            # 如果token长度超过最大输入字符数限制，则将其标记为未知词汇
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            # 初始化标志和起始位置
            is_bad = False
            start = 0
            sub_tokens = []
            # 使用贪婪最长匹配算法进行分词
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    # 判断子串是否在词汇表中
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                # 如果未找到匹配的子串，则标记为无效词汇
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            # 根据标志选择添加子token列表或者未知词汇标记到输出token列表
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        # 返回最终的token列表
        return output_tokens
```