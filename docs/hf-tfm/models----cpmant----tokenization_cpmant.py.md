# `.\models\cpmant\tokenization_cpmant.py`

```
# 设置文件编码为 UTF-8
# 版权声明及许可信息
#
# 根据 Apache 许可证 2.0 版本进行许可，除非符合许可证中的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是按“原样”基础分发的，
# 没有任何明示或暗示的保证或条件。请参阅许可证获取具体语言的权限和限制。
"""CPMAnt 的标记化类。"""
# 导入必要的库
import collections
import os
from typing import List, Optional, Tuple

# 导入条件依赖库
from transformers.utils import is_jieba_available, requires_backends

# 如果 jieba 库可用，则导入
if is_jieba_available():
    import jieba

# 导入通用工具函数和日志记录
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openbmb/cpm-ant-10b": "https://huggingface.co/openbmb/cpm-ant-10b/blob/main/vocab.txt",
    },
}

# 定义预训练模型的位置编码大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openbmb/cpm-ant-10b": 1024,
}

def load_vocab(vocab_file):
    """加载词汇文件到字典中。"""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

class WordpieceTokenizer(object):
    """基于词片段的标记化器。"""
    def __init__(self, vocab, unk_token="<unk>", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, token):
        """将单词标记化为词片段列表。"""
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            return [self.unk_token]

        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                sub_tokens.append(self.unk_token)
                start += 1
            else:
                sub_tokens.append(cur_substr)
                start = end

        return sub_tokens

class CpmAntTokenizer(PreTrainedTokenizer):
    """
    构造一个 CPMAnt 标记化器。基于字节级别的字节对编码。
    
    继承自 PreTrainedTokenizer 类。
    """
    pass
    # 定义类，用于处理特定的词汇表和标记化任务
    class BartTokenizer(BertTokenizer):
        """
        Args:
            vocab_file (`str`):
                Path to the vocabulary file.
            bod_token (`str`, *optional*, defaults to `"<d>"`):
                The beginning of document token.
            eod_token (`str`, *optional*, defaults to `"</d>"`):
                The end of document token.
            bos_token (`str`, *optional*, defaults to `"<s>"`):
                The beginning of sequence token.
            eos_token (`str`, *optional*, defaults to `"</s>"`):
                The end of sequence token.
            pad_token (`str`, *optional*, defaults to `"<pad>"`):
                The token used for padding.
            unk_token (`str`, *optional*, defaults to `"<unk>"`):
                The unknown token.
            line_token (`str`, *optional*, defaults to `"</n>"`):
                The line token.
            space_token (`str`, *optional*, defaults to `"</_>"`):
                The space token.
        """
    
        # 配置类变量，指定相关文件名和映射
        vocab_files_names = VOCAB_FILES_NAMES
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        model_input_names = ["input_ids", "attention_mask"]
        add_prefix_space = False
    
        # 初始化方法，加载词汇表并进行相关配置
        def __init__(
            self,
            vocab_file,
            bod_token="<d>",
            eod_token="</d>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
            line_token="</n>",
            space_token="</_>",
            padding_side="left",
            **kwargs,
        ):
            # 要求后端库为 "jieba"
            requires_backends(self, ["jieba"])
            self.bod_token = bod_token
            self.eod_token = eod_token
            # 加载并设置词汇表编码器
            self.encoder = load_vocab(vocab_file)
            # 将空格和换行符的编码对应到词汇表中
            self.encoder[" "] = self.encoder[space_token]
            self.encoder["\n"] = self.encoder[line_token]
    
            # 删除空格和换行符的原始编码
            del self.encoder[space_token]
            del self.encoder[line_token]
    
            # 按编码值排序并转为有序字典
            self.encoder = collections.OrderedDict(sorted(self.encoder.items(), key=lambda x: x[1]))
            # 创建反向词汇表
            self.decoder = {v: k for k, v in self.encoder.items()}
    
            # 使用词块化器设置词块化方法
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.encoder, unk_token=unk_token)
    
            # 调用父类的初始化方法
            super().__init__(
                bod_token=bod_token,
                eod_token=eod_token,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token,
                unk_token=unk_token,
                line_token=line_token,
                space_token=space_token,
                padding_side=padding_side,
                **kwargs,
            )
    
        # 返回开始文档标记的编码
        @property
        def bod_token_id(self):
            return self.encoder[self.bod_token]
    
        # 返回结束文档标记的编码
        @property
        def eod_token_id(self):
            return self.encoder[self.eod_token]
    
        # 返回换行符的编码
        @property
        def newline_id(self):
            return self.encoder["\n"]
    
        # 返回词汇表大小
        @property
        def vocab_size(self) -> int:
            return len(self.encoder)
    
        # 获取词汇表
        def get_vocab(self):
            return dict(self.encoder, **self.added_tokens_encoder)
    # 将输入文本进行分词处理，并返回分词后的结果列表
    def _tokenize(self, text):
        output_tokens = []
        # 使用结巴分词库对文本进行分词，cut_all=False表示精确模式
        for x in jieba.cut(text, cut_all=False):
            # 对每个分词结果进行 WordPiece 分词处理，并将处理后的结果添加到输出列表中
            output_tokens.extend(self.wordpiece_tokenizer.tokenize(x))
        return output_tokens

    # 将标识符列表解码为字符串
    def _decode(self, token_ids, **kwargs):
        # 移除小于0的无效标识符
        token_ids = [i for i in token_ids if i >= 0]
        # 移除特殊的标识符，如 padding、结束和开始标记
        token_ids = [
            x for x in token_ids if x != self.pad_token_id and x != self.eos_token_id and x != self.bos_token_id
        ]
        # 调用父类的解码方法解码标识符列表为字符串
        return super()._decode(token_ids, **kwargs)

    # 检查给定的标识符是否在编码器（词汇表）中
    def check(self, token):
        return token in self.encoder

    # 将标记列表转换为字符串
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    # 将标记（字符串）转换为其在词汇表中对应的标识符
    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将标识符（整数）转换为其在词汇表中对应的标记（字符串）
    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk_token)

    # 将词汇表保存到指定的目录下
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录存在，则构造词汇表文件路径
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 否则，直接使用指定的文件路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        
        index = 0
        # 处理特殊字符
        if " " in self.encoder:
            self.encoder["</_>"] = self.encoder[" "]
            del self.encoder[" "]
        if "\n" in self.encoder:
            self.encoder["</n>"] = self.encoder["\n"]
            del self.encoder["\n"]
        
        # 按照标识符的索引值对编码器进行排序，并转换为有序字典
        self.encoder = collections.OrderedDict(sorted(self.encoder.items(), key=lambda x: x[1]))

        # 将排序后的词汇表写入到文件中
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in self.encoder.items():
                # 检查索引是否连续，如果不连续则记录警告信息
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        
        # 返回保存的词汇表文件路径
        return (vocab_file,)
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: List[int] = None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. A CPMAnt sequence has the following format:

        - single sequence: `[BOS] Sequence`.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence that special tokens will be added.
            token_ids_1 (`List[int]`): The optional second tokenized sequence that special tokens will be added.

        Returns:
            `List[int]`: The model input with special tokens.
        """
        # 如果没有第二个序列，则返回带有起始特殊标记的第一个序列
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0
        # 如果有第二个序列，则连接两个序列，并在中间添加起始特殊标记
        return [self.bos_token_id] + token_ids_0 + [self.bos_token_id] + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`): List of IDs.
            token_ids_1 (`List[int]`, *optional*): Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        # 如果输入的 token_ids_0 和 token_ids_1 已经包含特殊标记，则调用父类方法处理
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果有第二个序列，则返回一个列表，以1开头表示起始特殊标记，接着全为0表示序列 token，再以1结尾表示第二个起始特殊标记
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))
        # 如果只有一个序列，则返回一个列表，以1开头表示起始特殊标记，接着全为0表示序列 token
        return [1] + ([0] * len(token_ids_0))
```