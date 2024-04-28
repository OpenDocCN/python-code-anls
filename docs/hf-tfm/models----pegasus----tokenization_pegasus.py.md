# `.\transformers\models\pegasus\tokenization_pegasus.py`

```
# coding=utf-8
# 版权声明及许可证信息，指定本文件的编码格式和版权归属
# Copyright 2020 Google and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License 2.0 许可证，您只有在遵守许可证的情况下才能使用此文件。
# 有关许可证的详细信息，请参阅
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 如果根据适用法律需要，或者同意以书面形式进行，本软件是根据"原样"的基础上分发的，
# 没有任何明示或暗示的担保或条件。
# 有关许可证下的特定语言的权限和限制，请参阅许可证。
import os
# 从 shutil 库中导入 copyfile 函数
from shutil import copyfile
# 从 typing 库中导入 Any, Dict, List, Optional, Tuple 类型
from typing import Any, Dict, List, Optional, Tuple

# 导入 sentencepiece 库
import sentencepiece as spm

# 从 tokenization_utils 模块中导入 AddedToken, PreTrainedTokenizer 类
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
# 导入 logging 模块
from ...utils import logging

# 定义一个特殊字符，表示词语的开始
SPIECE_UNDERLINE = "▁"

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# 定义预训练模型词汇文件的映射关系
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"google/pegasus-xsum": "https://huggingface.co/google/pegasus-xsum/resolve/main/spiece.model"}
}

# 定义预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/pegasus-xsum": 512,
}

# 获取日志记录器
logger = logging.get_logger(__name__)


# TODO ArthurZ refactor this to only use the added_tokens_encoder
# PegasusTokenizer 类，继承自 PreTrainedTokenizer 类
class PegasusTokenizer(PreTrainedTokenizer):
    r"""
    构造一个 PEGASUS 分词器。基于 SentencePiece。

    该分词器继承自 PreTrainedTokenizer 类，其中包含大多数主要方法。用户应参考
    此超类以获取有关这些方法的更多信息。

    """

    # 定义词汇文件的名称
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义预训练模型词汇文件的映射关系
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义最大模型输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化方法
    def __init__(
        self,
        vocab_file,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask_2>",
        mask_token_sent="<mask_1>",
        additional_special_tokens=None,
        offset=103,  # entries 2 - 104 are only used for pretraining
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(
            # 传递参数给父类的初始化方法
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            mask_token=mask_token,
            mask_token_sent=mask_token_sent,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        # 定义偏移量
        self.offset = offset  # entries 2 - 104 are only used for pretraining
        # 如果未指定 SentencePiece 模型的参数，则初始化为空字典
        if sp_model_kwargs is None:
            sp_model_kwargs = {}
        # 初始化 SentencePieceProcessor
        self.sp_model = spm.SentencePieceProcessor(**sp_model_kwargs)
        # 加载词汇文件
        self.sp_model.Load(vocab_file)

    # 定义属性，返回词汇表的大小
    @property
    def vocab_size(self) -> int:
        return len(self.sp_model) + self.offset

    # 获取词汇表的方法
    def get_vocab(self) -> Dict[str, int]:
        # 创建词汇表字典
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 更新词汇表字典
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 保存对象的状态
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    # 设置对象的状态
    def __setstate__(self, d):
        self.__dict__ = d

        # 为了向后兼容
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 初始化 SentencePieceProcessor
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载词汇文件
        self.sp_model.Load(self.vocab_file)
    # 将输入字符串分割为一个包含词语/子词的列表
    def _tokenize(self, text: str) -> List[str]:
        # 使用 SentencePiece 模型对输入的字符串进行编码，并以字符串的形式返回结果
        return self.sp_model.encode(text, out_type=str)
    
    # 将一个词语(字符串)转换为对应的ID
    def _convert_token_to_id(self, token: str) -> int:
        # 使用 SentencePiece 模型将输入的词语转换为ID
        sp_id = self.sp_model.piece_to_id(token)
        # 返回ID, 并加上偏移量
        return sp_id + self.offset
    
    # 将一个ID转换为对应的词语
    def _convert_id_to_token(self, index: int) -> str:
        # 如果ID小于偏移量, 则直接使用 SentencePiece 模型将其转换为词语
        if index < self.offset:
            return self.sp_model.IdToPiece(index)
        # 否则需要减去偏移量后, 再使用 SentencePiece 模型转换为词语
        token = self.sp_model.IdToPiece(index - self.offset)
        return token
    
    # 将一个词语序列转换为字符串
    def convert_tokens_to_string(self, tokens):
        # 初始化一个空列表和输出字符串
        current_sub_tokens = []
        out_string = ""
        # 遍历输入的词语序列
        for token in tokens:
            # 如果当前词语是特殊词语, 则先使用 SentencePiece 模型解码当前的子词, 再拼接特殊词语
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            # 否则添加到当前的子词列表中
            else:
                current_sub_tokens.append(token)
        # 最后补充解码剩余的子词, 并返回结果字符串
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()
    
    # 返回需要添加的特殊token的数量, 这里只有EOS
    def num_special_tokens_to_add(self, pair=False):
        return 1
    
    # 生成一个序列的特殊token掩码
    def _special_token_mask(self, seq):
        # 获取所有特殊token的ID集合, 并排除<unk>
        all_special_ids = set(self.all_special_ids)
        all_special_ids.remove(self.unk_token_id)
        # 根据特殊token的ID, 生成掩码序列
        return [1 if x in all_special_ids else 0 for x in seq]
    
    # 获取一个或两个序列的特殊token掩码
    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        # 如果已经包含特殊token, 直接生成掩码序列
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        # 如果只有一个序列, 生成掩码序列并在最后添加1表示EOS
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [1]
        # 如果有两个序列, 生成掩码序列并在最后添加1表示EOS
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """
        从一个序列或一对序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。 PEGASUS 序列具有以下格式，其中 `X` 表示序列：

        - 单个序列：`X </s>`
        - 一对序列：`A B </s>`（不是预期使用）

        不会使用 BOS。一对序列不是预期的使用情况，但它们将在没有分隔符的情况下处理。

        Args:
            token_ids_0 (`List[int]`): 
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*): 
                用于序列对的可选第二个 ID 列表。

        Returns:
            `List[int]`: 包含适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
        """
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # 我们不期望处理序对，但为了 API 一致性而保留一对逻辑
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        将词汇表保存到指定目录中。

        Args:
            save_directory (str): 
                保存词汇表的目录路径。
            filename_prefix (Optional[str], *可选*): 
                文件名前缀。

        Returns:
            `Tuple[str]`: 保存的词汇表文件的路径。
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)
```