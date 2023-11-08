# ChatGLM2 源码解析：`ChatGLMTokenizer`

```py
import os
import torch
from typing import List, Optional, Union, Dict
from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizer
from transformers.utils import logging, PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding

# 底层的分词器，也就是 SP 模型的包装
class SPTokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        # 加载 SP 模型作为底层模型
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # 设置单词数量，BOS EOS PAD ID 属性
        # PAD 由底层模型的 UNK 代替
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        # 定义特殊单词
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]
        # 建立特殊单词文本到ID的映射
        self.special_tokens = {}
        # 建立特殊单词ID到文本的映射
        self.index_special_tokens = {}
        for token in special_tokens:
            # 遍历特殊单词，填充这个两个映射
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1

    # 文本片段转单词文本数组
    def tokenize(self, s: str):
        # 转发给底层模型的`EncodeAsPieces`
        return self.sp_model.EncodeAsPieces(s)

    # 文本片段转单词 ID 数组
    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        # 调用底层模型的`encode`方法
        t = self.sp_model.encode(s)
        # 根据传入的`bos`和`eos`标志
        # 决定是否添加 BOS 和 EOS ID
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    # 单词 ID 数组转文本片段
    def decode(self, t: List[int]) -> str:
        # 转发给底层模型的`decode`方法
        return self.sp_model.decode(t)

    # 单词文本数组转文本片段
    def decode_tokens(self, tokens: List[str]) -> str:
        text = self.sp_model.DecodePieces(tokens)
        return text

    # 单词文本转 ID
    def convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        # 如果单词在特殊标记里面，就从`special_tokens`查找 ID
        if token in self.special_tokens:
            return self.special_tokens[token]
        # 否则转发给底层模型的`PieceToId`
        return self.sp_model.PieceToId(token)

    # 单词 ID 转文本
    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 如果单词在特殊标记里面，或者是 BOS、EOS、PAD 之一，就返回空串
        if index in self.index_special_tokens or index in [self.eos_id, self.bos_id, self.pad_id] or index < 0:
            return ""
        # 否则转发给底层模型的`IdToPiece`
        return self.sp_model.IdToPiece(index)

# 用户直接使用的分词器
class ChatGLMTokenizer(PreTrainedTokenizer):
    # 定义词表名称
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    # 定义模型输入参数名称
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(self, vocab_file, padding_side="left", **kwargs):
        super().__init__(padding_side=padding_side, clean_up_tokenization_spaces=False, **kwargs)
        self.name = "GLMTokenizer"
        
        # 在属性中保存词表路径
        # 这个文件是和词表本身放一起的，所以路径就只是文件名
        self.vocab_file = vocab_file
        # 创建底层的分词器，传入词表路径
        self.tokenizer = SPTokenizer(vocab_file)
        # 定义特殊单词 BOS、EOS、PAD
        # 建立单词文本到ID的映射
        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<pad>": self.tokenizer.pad_id
        }

    # 特殊单词文本转 ID
    def get_command(self, token):
        # 如果单词在GLM 分词器的特殊字符中
        # 查找`special_tokens`，返回它的 ID
        if token in self.special_tokens:
            return self.special_tokens[token]
        # 如果单词不在底层的 SP 分词器的特殊字符中，就报错
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        # 查找底层分词器的`special_tokens`，返回它的ID
        return self.tokenizer.special_tokens[token]

    # 返回 UNK 单词文本
    @property
    def unk_token(self) -> str:
        return "<unk>"

    # 返回 PAD 单词文本（也就是 UNK）
    @property
    def pad_token(self) -> str:
        return "<unk>"

    # 返回 PAD 单词 ID
    @property
    def pad_token_id(self):
        return self.get_command("<pad>")

    # 返回 EOS 单词文本
    @property
    def eos_token(self) -> str:
        return "</s>"

    # 返回 EOS 单词 ID
    @property
    def eos_token_id(self):
        return self.get_command("<eos>")

    # 返回词表大小
    @property
    def vocab_size(self):
        return self.tokenizer.n_words

    # 获取词表，也就是单词文本到ID的映射
    def get_vocab(self):
        """ Returns vocab as a dict """
        # 遍历所有单词的 ID，即 0 到 VocabSize-1]
        # 调用自身的`_convert_id_to_token`方法将 ID 转成文本
        # 创建一个单词文本到ID的映射
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 文本片段转单词文本数组
    def _tokenize(self, text, **kwargs):
        # 转发给底层分词器的`tokenize`方法
        return self.tokenizer.tokenize(text)

    # 单词文本转 ID
    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        # 转发给底层分词器的`convert_token_to_id`方法
        return self.tokenizer.convert_token_to_id(token)

    # 单词 ID 转文本
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 转发给底层分词器的`convert_id_to_token`方法
        return self.tokenizer.convert_id_to_token(index)

    # 单词文本数组转文本片段
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # 转发给底层分词器的`decode_tokens`方法
        return self.tokenizer.decode_tokens(tokens)

    # 保存词表
    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            # 如果传入路径是个目录，那么文件名就是之前定义的默认文件名
            # 把传入路径和文件名拼接好作为保存路径
            vocab_file = os.path.join(
                save_directory, self.vocab_files_names["vocab_file"]
            )
        else:
            # 否则保存路径就是传入路径
            vocab_file = save_directory

        # 根据属性中的词表路径，读入词表
        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        # 把词表写到保存路径中
        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)
        
        # 返回保存路径
        return (vocab_file,)

    # 获取前缀单词列表，即 GMASK 和 SOP
    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        return prefix_tokens

    '''
    根据当前提问和历史问答构建复合提问
    In [1]: tokenizer.build_prompt('Q3', [('Q1', 'A1'),('Q2', 'A2')])
    Out[1]: '[Round 1]\n\n问：Q1\n\n答：A1\n\n[Round 2]\n\n问：Q2\n\n答：A2\n\n[Round 3]\n\n问：Q3\n\n答：'
    '''
    def build_prompt(self, query, history=None):
        if history is None:
            history = []
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            # 遍历每一对历史问答，将序号、提问和回答按照模版组装
            # 并添加到复合提问后面
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        # 将当前轮次和当前提问按照模版组装，添加到复合提问后面
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        return prompt

    # 给单词 ID 数组添加特殊单词
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
        # 或许前缀单词列表，并添加到 IDS0 前方
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        # 如果 IDS1 存在，添加到 IDS0 后方，并添加 EOS
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
        return token_ids_0

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        assert self.padding_side == "left"

        # `encoded_inputs`是个字典，`input_ids`包含模型的输入单词ID数组
        # `attention_mask`是掩码数组，`position_ids`是位置 ID 数组
        # `required_input`是输入单词 ID 数组
        required_input = encoded_inputs[self.model_input_names[0]]
        # `seq_length`是输入长度
        seq_length = len(required_input)

        # 如果策略是按最长填充，因为只有一个输入，最大长度就是它的长度
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        # 如果提供了最大长度和`pad_to_multiple_of`
        # 将最大长度设为不小于它的`pad_to_multiple_of`的倍数
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        # 如果策略不是不填充，并且最大长度
        # 和输入长度不相等，就需要填充
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # 如果没有掩码，初始化为全 1 长度为 SeqLen 的数组
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length

        # 如果没有位置 ID，初始化为 [0, ..., SeqLen - 1]
        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))

        
        if needs_to_be_padded:
            # 如果需要填充，计算填充字符个数，也就是最大长度和输入的差值
            difference = max_length - len(required_input)

            #  如果存在掩码，在掩码前方插入 diff 个 0
            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            # 如果存在位置 ID，同样前方插入 diff 个 0
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
            # 在输入 IDS 前方插入 diff 个 PAD ID
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs

```