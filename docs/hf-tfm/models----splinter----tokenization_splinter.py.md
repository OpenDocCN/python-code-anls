# `.\models\splinter\tokenization_splinter.py`

```
# 定义了文件编码为 UTF-8
# 版权声明，包括版权归属和保留所有权利
# Apache License 2.0 许可证声明，规定了软件的使用条件和责任限制
# 详细许可证信息可以在 http://www.apache.org/licenses/LICENSE-2.0 获取
# 如果符合许可证所述的条件，可以使用此文件
"""Splinter 的标记化类。"""

# 导入必要的模块
import collections
import os
import unicodedata
from typing import List, Optional, Tuple

# 从 tokenization_utils 中导入 PreTrainedTokenizer 类及一些辅助函数
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
# 从 utils 中导入 logging 模块
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件的名称，此处只包含一个键值对
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练模型的词汇文件映射表
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "tau/splinter-base": "https://huggingface.co/tau/splinter-base/resolve/main/vocab.txt",
        "tau/splinter-base-qass": "https://huggingface.co/tau/splinter-base-qass/resolve/main/vocab.txt",
        "tau/splinter-large": "https://huggingface.co/tau/splinter-large/resolve/main/vocab.txt",
        "tau/splinter-large-qass": "https://huggingface.co/tau/splinter-large-qass/resolve/main/vocab.txt",
    }
}

# 预训练模型的位置嵌入大小映射表
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "tau/splinter-base": 512,
    "tau/splinter-base-qass": 512,
    "tau/splinter-large": 512,
    "tau/splinter-large-qass": 512,
}

# 预训练模型的初始化配置映射表
PRETRAINED_INIT_CONFIGURATION = {
    "tau/splinter-base": {"do_lower_case": False},
    "tau/splinter-base-qass": {"do_lower_case": False},
    "tau/splinter-large": {"do_lower_case": False},
    "tau/splinter-large-qass": {"do_lower_case": False},
}


def load_vocab(vocab_file):
    """从词汇文件加载词汇表到一个有序字典中。"""
    # 创建一个空的有序字典对象
    vocab = collections.OrderedDict()
    # 打开词汇文件，按 UTF-8 编码读取
    with open(vocab_file, "r", encoding="utf-8") as reader:
        # 逐行读取词汇文件内容
        tokens = reader.readlines()
    # 遍历 tokens 列表，将每个 token 去除换行符后作为键，其索引作为值存入 vocab 字典
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    # 返回构建的词汇表字典
    return vocab


def whitespace_tokenize(text):
    """对文本进行基本的空格清理和分割。"""
    # 去除文本两端的空白字符
    text = text.strip()
    # 如果清理后文本为空，则返回空列表
    if not text:
        return []
    # 使用空格分割文本，得到 token 列表
    tokens = text.split()
    # 返回分割后的 token 列表
    return tokens


class SplinterTokenizer(PreTrainedTokenizer):
    r"""
    构建一个 Splinter 分词器，基于 WordPiece 算法。

    这个分词器继承自 [`PreTrainedTokenizer`]，包含大多数主要方法。用户可以参考这个超类获取更多关于这些方法的信息。
    """
    # 定义函数的参数说明文档，以下是各参数的详细说明：
    
    Args:
        vocab_file (`str`):
            包含词汇表的文件。
        do_lower_case (`bool`, *optional*, defaults to `True`):
            在分词时是否将输入转换为小写。
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            在使用 WordPiece 分词之前是否进行基本分词。
        never_split (`Iterable`, *optional*):
            在分词时永远不会分割的标记集合。仅在 `do_basic_tokenize=True` 时生效。
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            未知标记。词汇表中不存在的标记会被设置为该标记。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔符标记，用于从多个序列构建一个序列，例如用于序列分类或问答任务中。
            也是使用特殊标记构建序列时的最后一个标记。
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            用于填充的标记，例如在批处理不同长度的序列时使用。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            分类器标记，用于序列分类任务。使用特殊标记构建序列时的第一个标记。
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            掩码标记，用于掩码语言建模任务中。模型将尝试预测此标记。
        question_token (`str`, *optional*, defaults to `"[QUESTION]"`):
            用于构建问题表示的标记。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否对中文字符进行分词。
            对于日文应该禁用此选项（参见此处的问题链接）。
        strip_accents (`bool`, *optional*):
            是否删除所有重音符号。如果未指定此选项，则将根据 `lowercase` 的值确定（与原始的 BERT 行为相同）。
    
    
    
    # 从预定义的全局变量中获取相关信息
    
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 初始化方法，用于创建一个新的 Tokenizer 对象
    def __init__(
        self,
        vocab_file,                  # 词汇表文件的路径
        do_lower_case=True,          # 是否将输入文本转换为小写，默认为True
        do_basic_tokenize=True,      # 是否进行基本的分词，默认为True
        never_split=None,            # 不进行分词的特殊标记集合，默认为None
        unk_token="[UNK]",           # 未知标记，默认为"[UNK]"
        sep_token="[SEP]",           # 分隔标记，默认为"[SEP]"
        pad_token="[PAD]",           # 填充标记，默认为"[PAD]"
        cls_token="[CLS]",           # 分类标记，默认为"[CLS]"
        mask_token="[MASK]",         # 掩码标记，默认为"[MASK]"
        question_token="[QUESTION]", # 问题标记，默认为"[QUESTION]"
        tokenize_chinese_chars=True, # 是否分词中文字符，默认为True
        strip_accents=None,          # 是否去除文本中的重音符号，默认为None
        **kwargs,                    # 其他可选参数
    ):
        # 如果指定的词汇表文件不存在，则抛出 ValueError 异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 载入词汇表文件并存储在 self.vocab 中
        self.vocab = load_vocab(vocab_file)
        # 根据词汇表文件创建一个从 ids 到 tokens 的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 是否进行基本分词的标记
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要进行基本分词，则初始化 BasicTokenizer 对象
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 使用指定的未知标记初始化 WordpieceTokenizer 对象
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))
        # 问题标记的设置
        self.question_token = question_token
        # 调用父类的初始化方法，传入相同的参数
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

    @property
    def question_token_id(self):
        """
        `Optional[int]`: Id of the question token in the vocabulary, used to condition the answer on a question
        representation.
        """
        # 返回问题标记在词汇表中的 id，用于在问题表示中条件化答案
        return self.convert_tokens_to_ids(self.question_token)

    @property
    def do_lower_case(self):
        # 返回是否进行小写处理的标记，由 BasicTokenizer 决定
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        # 返回词汇表的大小（词汇表中的唯一标记数）
        return len(self.vocab)

    def get_vocab(self):
        # 返回包含词汇表及其额外添加标记编码的字典
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        # 对文本进行分词处理，返回分词后的标记列表
        split_tokens = []
        # 如果需要进行基本分词
        if self.do_basic_tokenize:
            # 使用 BasicTokenizer 对象进行分词
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # 如果分词后的 token 在不分割集合中
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 使用 WordpieceTokenizer 对象对 token 进行进一步分词
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 使用 WordpieceTokenizer 对象对文本进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens
    def _convert_token_to_id(self, token):
        """Converts a token (str) into an id using the vocab."""
        # Return the vocabulary ID of the given token; if token not found, return ID of unknown token
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) into a token (str) using the vocab."""
        # Return the token corresponding to the given index; if index not found, return unknown token
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) into a single string."""
        # Join tokens into a string, remove '##' and strip leading/trailing spaces
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a pair of sequences for question answering tasks by concatenating and adding special
        tokens. A Splinter sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences for question answering: `[CLS] question_tokens [QUESTION] . [SEP] context_tokens [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                The question token IDs if pad_on_right, else context tokens IDs
            token_ids_1 (`List[int]`, *optional*):
                The context token IDs if pad_on_right, else question token IDs

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens.
        """
        if token_ids_1 is None:
            # Return single sequence input IDs with [CLS], tokens_0, and [SEP]
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        question_suffix = [self.question_token_id] + [self.convert_tokens_to_ids(".")]
        if self.padding_side == "right":
            # Return input IDs for question-then-context sequence
            return cls + token_ids_0 + question_suffix + sep + token_ids_1 + sep
        else:
            # Return input IDs for context-then-question sequence
            return cls + token_ids_0 + sep + token_ids_1 + question_suffix + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve a mask indicating the positions of special tokens in the input sequences.

        Args:
            token_ids_0 (`List[int]`):
                The question token IDs if pad_on_right, else context tokens IDs
            token_ids_1 (`List[int]`, *optional*):
                The context token IDs if pad_on_right, else question token IDs
            already_has_special_tokens (`bool`):
                Whether the input IDs already include special tokens

        Returns:
            `List[int]`: List indicating positions of special tokens (1 for special token, 0 for others)
        """
        # Initialize mask for special tokens
        special_tokens_mask = [0] * len(token_ids_0)

        if token_ids_1 is not None:
            special_tokens_mask += [1] * len(token_ids_1)

        # Mark special tokens [CLS], [SEP], [QUESTION] in the mask
        special_tokens_mask[:1] = [1]  # [CLS] token

        if token_ids_1 is None:
            special_tokens_mask[-1:] = [1]  # [SEP] token for single sequence
        else:
            special_tokens_mask[len(token_ids_0) + 2] = 1  # [SEP] token after context tokens

        if self.question_token_id is not None:
            special_tokens_mask[len(token_ids_0):len(token_ids_0) + 2] = [1, 1]  # [QUESTION] and [.] tokens

        return special_tokens_mask
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

        # Check if the input token list already has special tokens
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If token_ids_1 is provided, create masks for both sequences
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # Otherwise, create masks for a single sequence
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. [What are token type
        IDs?](../glossary#token-type-ids)

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The token type ids.
        """
        # Define special tokens
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        question_suffix = [self.question_token_id] + [self.convert_tokens_to_ids(".")]
        
        # If only one sequence is provided, return token type ids for that sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]

        # Determine the padding side and construct token type ids accordingly
        if self.padding_side == "right":
            # Input format: question-then-context
            return len(cls + token_ids_0 + question_suffix + sep) * [0] + len(token_ids_1 + sep) * [1]
        else:
            # Input format: context-then-question
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + question_suffix + sep) * [1]
    # 将词汇表保存到指定目录中的文件中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引为0
        index = 0
        # 检查保存目录是否存在
        if os.path.isdir(save_directory):
            # 构建词汇文件的完整路径，如果有前缀，则包含在文件名中
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 如果保存目录不存在，则直接使用save_directory作为文件路径，如果有前缀，则包含在文件名中
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开文件进行写操作，使用UTF-8编码
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的每个token及其索引，按索引排序
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 如果当前索引与期望索引不同，则发出警告
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    # 更新期望的索引
                    index = token_index
                # 将token写入文件，每个token占一行
                writer.write(token + "\n")
                # 更新索引
                index += 1
        # 返回保存的文件路径作为元组中的唯一元素
        return (vocab_file,)
# 定义一个名为 BasicTokenizer 的类，用于执行基本的分词（如拆分标点符号、转换为小写等）。
class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
            是否在分词时将输入转换为小写，默认为 True。
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
            在分词时不会拆分的 token 集合。仅在 `do_basic_tokenize=True` 时有效。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.
            是否对中文字符进行分词，默认为 True。

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
            这可能需要为日语禁用（参见此问题链接）。
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
            是否去除所有的重音符号。如果未指定此选项，则由 `lowercase` 的值决定（与原始的 BERT 类似）。
    """

    # 初始化方法，接受几个可选参数用于配置分词器的行为
    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        # 如果 never_split 参数为 None，则设为空列表
        if never_split is None:
            never_split = []
        # 设定是否转换为小写
        self.do_lower_case = do_lower_case
        # 将 never_split 转换为集合，用于快速查找
        self.never_split = set(never_split)
        # 设定是否分词中文字符
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 设定是否去除重音符号
        self.strip_accents = strip_accents
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            **never_split**: (*optional*) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 使用传入的 never_split 列表与实例中的 never_split 集合的并集作为最终的 never_split 集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清洗文本，例如移除不必要的字符或空格
        text = self._clean_text(text)

        # 以下内容是为了支持多语言和中文模型而添加的，适用于英语模型，尽管这些模型没有中文数据
        if self.tokenize_chinese_chars:
            # 对包含中文字符的文本进行特殊处理，可能涉及分词等操作
            text = self._tokenize_chinese_chars(text)
        # 使用空白字符进行基本分词
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        # 遍历原始分词后的 tokens
        for token in orig_tokens:
            # 如果 token 不在 never_split 中，则进行进一步处理
            if token not in never_split:
                if self.do_lower_case:
                    # 如果需要将 token 转换为小写
                    token = token.lower()
                    if self.strip_accents is not False:
                        # 如果需要去除重音符号，则调用 _run_strip_accents 方法
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 如果需要去除重音符号，则调用 _run_strip_accents 方法
                    token = self._run_strip_accents(token)
            # 将处理后的 token 加入到 split_tokens 中，可能涉及进一步的分割
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将分词后的结果再次使用空白字符进行分割，形成最终的输出 tokens
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 使用 NFD 标准对文本进行 Unicode 规范化，以处理重音符号
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 类别
            cat = unicodedata.category(char)
            # 如果字符的类别是 Mn（Nonspacing_Mark），则跳过该字符
            if cat == "Mn":
                continue
            # 否则将字符添加到输出列表中
            output.append(char)
        # 将列表中的字符重新组合成字符串并返回
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果指定了 never_split 并且 text 在 never_split 中，则返回 text 的列表形式
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        # 遍历文本中的每个字符
        while i < len(chars):
            char = chars[i]
            # 如果字符是标点符号，则作为一个新的词添加到 output 中
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 否则将字符添加到当前词的最后一个词素中
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        # 将列表中的词素重新组合成字符串并返回
        return ["".join(x) for x in output]
    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        # 初始化空列表用于存储处理后的文本
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 码点
            cp = ord(char)
            # 如果字符是中文字符，则在其前后加入空格
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                # 如果字符不是中文字符，则直接添加到输出列表中
                output.append(char)
        # 将列表中的字符拼接成字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 判断给定的 Unicode 码点是否属于CJK字符的范围
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
        # 如果不是CJK字符，则返回False
        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        # 初始化空列表用于存储处理后的文本
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 码点
            cp = ord(char)
            # 如果字符是空字符或者无效字符，跳过处理
            if cp == 0 or cp == 0xFFFD or self._is_control(char):
                continue
            # 如果字符是空白字符，则用单个空格替换
            if self._is_whitespace(char):
                output.append(" ")
            else:
                # 其他情况直接添加到输出列表中
                output.append(char)
        # 将列表中的字符拼接成字符串并返回
        return "".join(output)
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化 WordpieceTokenizer 类的实例
        self.vocab = vocab  # 词汇表，用于查找词片段
        self.unk_token = unk_token  # 未知词片段的标记
        self.max_input_chars_per_word = max_input_chars_per_word  # 单词的最大字符数限制

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

        output_tokens = []  # 存储最终的词片段结果
        for token in whitespace_tokenize(text):  # 使用空格分割文本中的每个单词
            chars = list(token)  # 将单词分割为字符列表
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)  # 如果单词超过最大字符限制，添加未知词片段标记
                continue

            is_bad = False  # 标记当前单词是否无法分割为词片段
            start = 0
            sub_tokens = []  # 存储当前单词分割后的词片段
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])  # 从当前位置到末尾的子串
                    if start > 0:
                        substr = "##" + substr  # 非首字符的词片段添加"##"前缀
                    if substr in self.vocab:  # 如果词片段在词汇表中
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True  # 如果无法找到合适的词片段，则标记为无法处理
                    break
                sub_tokens.append(cur_substr)  # 将找到的词片段添加到列表中
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)  # 如果无法处理当前单词，添加未知词片段标记
            else:
                output_tokens.extend(sub_tokens)  # 否则将分割得到的词片段添加到最终结果中
        return output_tokens  # 返回最终的词片段列表
```