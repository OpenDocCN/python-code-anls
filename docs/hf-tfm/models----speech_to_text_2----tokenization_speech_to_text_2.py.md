# `.\transformers\models\speech_to_text_2\tokenization_speech_to_text_2.py`

```
# 设置编码格式为 utf-8
# 版权声明和许可信息，提供对代码的使用、复制和修改的规范
import json
import os
from typing import Dict, List, Optional, Tuple

# 导入 logging 模块
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义与词汇相关文件的名称
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
    "merges_file": "merges.txt",
}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/s2t-wav2vec2-large-en-de": (
            "https://huggingface.co/facebook/s2t-wav2vec2-large-en-de/resolve/main/vocab.json"
        ),
    },
    "tokenizer_config_file": {
        "facebook/s2t-wav2vec2-large-en-de": (
            "https://huggingface.co/facebook/s2t-wav2vec2-large-en-de/resolve/main/tokenizer_config.json"
        ),
    },
    "merges_file": {
        "facebook/s2t-wav2vec2-large-en-de": (
            "https://huggingface.co/facebook/s2t-wav2vec2-large-en-de/resolve/main/merges.txt"
        ),
    },
}

# 定义 BPE 标记的合并和词汇标记
BPE_TOKEN_MERGES = "</w>"
BPE_TOKEN_VOCAB = "@@ "

# 定义函数，返回单词中的符号对
def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

# 定义预训练位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/s2t-wav2vec2-large-en-de": 1024}

# Speech2Text2Tokenizer 类继承自 PreTrainedTokenizer 类
class Speech2Text2Tokenizer(PreTrainedTokenizer):
    """
    Constructs a Speech2Text2Tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.

        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """

    # Define class variables
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        do_lower_case=False,
        merges_file=None,
        **kwargs,
    ):
        # Initialize object variables
        self.do_lower_case = do_lower_case

        # Load vocabulary file and create encoder and decoder dictionaries
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Check if merges are provided, if not, the class can only be used for decoding
        if merges_file is None:
            logger.info(f"No merges files provided. {self.__class__.__name__} can only be used for decoding.")

            self.bpe_ranks = None
            self.cache = None
        else:
            # Load merges file and create a dictionary for BPE ranks
            with open(merges_file, encoding="utf-8") as merges_handle:
                merges = merges_handle.read().split("\n")[:-1]

            merges = [tuple(merge.split()[:2]) for merge in merges]
            self.bpe_ranks = dict(zip(merges, range(len(merges))))
            self.cache = {}

        # Call the parent class initialization method with specified tokens and arguments
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    # Method to get the vocabulary dictionary including added tokens
    def get_vocab(self) -> Dict:
        return dict(self.encoder, **self.added_tokens_encoder)
    def bpe(self, token):
        # 将输入的单词转换为元组形式，用于处理特殊情况
        word = tuple(token[:-1]) + (token[-1] + BPE_TOKEN_MERGES,)
        # 检查是否存在缓存，如果存在则直接返回缓存结果
        if token in self.cache:
            return self.cache[token]
        # 获取token中的所有可能的pairs
        pairs = get_pairs(word)

        # 如果没有pairs，则直接返回原始token
        if not pairs:
            return token

        # 根据BPE子词的rank进行拆分词组直到没有更低rank的可拆分子词为止
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # 格式化word，加入空格，如果有特殊字符则替换为指定符号
        word = " ".join(word)
        if word == "\n  " + BPE_TOKEN_MERGES:
            word = "\n" + BPE_TOKEN_MERGES

        if word.endswith(BPE_TOKEN_MERGES):
            word = word.replace(BPE_TOKEN_MERGES, "")

        word = word.replace(" ", BPE_TOKEN_VOCAB)
        # 将结果加入缓存并返回word
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""

        # 检查BPE_ranks是否为空，如果为空则抛出异常
        if self.bpe_ranks is None:
            raise ValueError(
                "This tokenizer was instantiated without a `merges.txt` file, so"
                " that it can only be used for decoding, not for encoding. "
                "Make sure to provide `merges.txt` file at instantiation to enable "
                "encoding."
            )

        # 如果设置了do_lower_case，则将text全部转换为小写
        if self.do_lower_case:
            text = text.lower()

        # 将text按空格分割成词汇列表
        text = text.split()

        split_tokens = []
        # 遍历text中的每个token，如果不为空则将BPE分词结果加入split_tokens
        for token in text:
            if token:
                split_tokens.extend(list(self.bpe(token).split(" ")))

        return split_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        # 使用vocab将token转换为对应的id，如果不存在则返回unk_token对应的id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用vocab将id转换为对应的token，如果不存在则返回unk_token
        result = self.decoder.get(index, self.unk_token)
        return result

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a list of output tokens into a single string.
        """
        # 将tokens列表中的词汇合并为一个字符串
        string = " ".join(tokens)

        # 确保特殊标记@@被合并
        string = "".join(string.split(BPE_TOKEN_VOCAB))

        return string
    # 保存词汇表到指定目录，可选添加文件名前缀，返回包含文件路径的元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回空值
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建词汇表文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构建合并文件路径
        merges_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将编码器对象以 JSON 格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 如果 BPE merge 编码不存在，则返回词汇表文件路径的元组
        index = 0
        if self.bpe_ranks is None:
            return (vocab_file,)

        # 将BPE merge编码写入合并文件
        with open(merges_file, "w", encoding="utf-8") as writer:
            # 遍历并排序BPE merge编码，并依次写入文件
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                # 检查索引是否连续并记录警告
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merges_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                # 将BPE merge编码写入文件
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回词汇表文件路径和合并文件路径的元组
        return (vocab_file, merges_file)
```