# `.\transformers\models\bartpho\tokenization_bartpho.py`

```
# 设置文件编码为 UTF-8
# 该文件版权归 VinAI Research 和 HuggingFace Inc. 团队所有
# 基于 Apache 许可证 2.0 进行许可
# 除非符合许可证规定，在适用法律要求或书面同意，否则不得使用本文件
# 本软件根据“原样”提供，不提供任何明示或暗示的保证或条件
# 请参阅许可证以了解更多信息
""" BARTpho-syllable 模型的分词类。"""

# 导入必要的模块
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

# 导入 sentencepiece 库，并命名为 spm
import sentencepiece as spm

# 导入预训练分词器的基类 PreTrainedTokenizer
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
# 导入日志模块
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# SentencePiece 的特殊字符标识符
# 定义句子边界的标识符，用于在分词结果中标识句子的开头和结尾
SPIECE_UNDERLINE = "▁"

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "monolingual_vocab_file": "dict.txt"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "vinai/bartpho-syllable": "https://huggingface.co/vinai/bartpho-syllable/resolve/main/sentencepiece.bpe.model",
    },
    "monolingual_vocab_file": {
        "vinai/bartpho-syllable": "https://huggingface.co/vinai/bartpho-syllable/resolve/main/dict.txt",
    },
}

# 预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"vinai/bartpho-syllable": 1024}

# BARTphoTokenizer 类，继承自 PreTrainedTokenizer
class BartphoTokenizer(PreTrainedTokenizer):
    """
    从 `XLMRobertaTokenizer` 改编。基于 SentencePiece。

    该分词器继承自 PreTrainedTokenizer，其中包含大部分主要方法。用户应参考
    该超类以获取有关这些方法的更多信息。

    属性:
        sp_model (`SentencePieceProcessor`):
            用于所有转换（字符串、标记和 ID）的 SentencePiece 处理器。
    """

    # 定义类属性
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化方法
    def __init__(
        self,
        vocab_file,
        monolingual_vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 设置掩码标记，使其表现为普通词，即包含在其之前的空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 如果未提供 sp_model_kwargs，则设置为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 设置词汇文件路径
        self.vocab_file = vocab_file
        # 设置单语词汇文件路径
        self.monolingual_vocab_file = monolingual_vocab_file
        # 使用给定的 sp_model_kwargs 参数创建 SentencePieceProcessor 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 从文件加载 SentencePiece 模型
        self.sp_model.Load(str(vocab_file))

        # 加载减少的词汇

        # 为了向后兼容，保持特殊标记的顺序
        self.fairseq_tokens_to_ids = {}
        cnt = 0
        for token in [bos_token, pad_token, eos_token, unk_token, sep_token, cls_token]:
            if str(token) not in self.fairseq_tokens_to_ids:
                self.fairseq_tokens_to_ids[str(token)] = cnt
                cnt += 1
        # 从单语词汇文件中读取词汇并加入词汇字典
        with open(monolingual_vocab_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                token = line.strip().split()[0]
                self.fairseq_tokens_to_ids[token] = len(self.fairseq_tokens_to_ids)
        # 如果掩码标记不在词汇字典中，则加入
        if str(mask_token) not in self.fairseq_tokens_to_ids:
            self.fairseq_tokens_to_ids[str(mask_token)] = len(self.fairseq_tokens_to_ids)

        # 创建反向的词汇字典
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        # 调用父类的初始化方法，传递参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    def __getstate__(self):
        # 复制对象状态
        state = self.__dict__.copy()
        # 置空 sp_model，以便对象能够被正确序列化
        state["sp_model"] = None
        # 序列化 SentencePiece 模型
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        # 恢复对象状态
        self.__dict__ = d

        # 为了向后兼容，如果不存在 sp_model_kwargs，则创建空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 重新创建 SentencePieceProcessor 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 从序列化的 proto 对象加载 SentencePiece 模型
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或一对序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。一个 BARTPho 序列的格式如下：

        - 单个序列: `<s> X </s>`
        - 一对序列: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊标记的输入 ID 列表。
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 ID。当使用 tokenizer 的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *可选*, 默认为 `False`):
                标记列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        为序列对分类任务创建一个掩码。BARTPho 不使用 token 类型 id，因此返回一个全零列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 全零列表。

        """

        sep = [self.sep_token_id]  # 创建一个包含 SEP token ID 的列表
        cls = [self.cls_token_id]  # 创建一个包含 CLS token ID 的列表

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]  # 返回全零列表，长度为 CLS、token_ids_0 和 SEP 的长度之和
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]  # 返回全零列表，长度为 CLS、token_ids_0、两个 SEP 和 token_ids_1 的长度之和

    @property
    def vocab_size(self):
        return len(self.fairseq_ids_to_tokens)  # 返回 fairseq_ids_to_tokens 的长度，即词汇表大小

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}  # 创建一个从 ID 到 token 的词汇表
        vocab.update(self.added_tokens_encoder)  # 更新词汇表，加入额外的 token
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)  # 使用 sp_model 对文本进行分词，返回字符串列表

    def _convert_token_to_id(self, token):
        """将 token（str）转换为 id，使用词汇表。"""
        if token in self.fairseq_tokens_to_ids:  # 如果 token 存在于 fairseq_tokens_to_ids 中
            return self.fairseq_tokens_to_ids[token]  # 返回对应的 id
        else:
            return self.unk_token_id  # 否则返回未知 token 的 id

    def _convert_id_to_token(self, index):
        """将索引（整数）转换为 token（str），使用词汇表。"""
        return self.fairseq_ids_to_tokens[index]  # 返回 fairseq_ids_to_tokens 中对应索引的 token

    def convert_tokens_to_string(self, tokens):
        """将 token 序列（子词的字符串）转换为单个字符串。"""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()  # 将 token 列表连接成字符串，并将子词之间的下划线替换为空格
        return out_string
    # 将词汇表保存到指定目录中，可指定保存文件名前缀
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，若不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构建输出单语词汇表文件路径
        out_monolingual_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["monolingual_vocab_file"],
        )

        # 如果当前词汇表文件路径与输出路径不同且当前词汇表文件存在，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将当前词汇表内容写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 如果当前单语词汇表文件路径与输出路径不同且当前单语词汇表文件存在，则复制当前单语词汇表文件到输出路径
        if os.path.abspath(self.monolingual_vocab_file) != os.path.abspath(
            out_monolingual_vocab_file
        ) and os.path.isfile(self.monolingual_vocab_file):
            copyfile(self.monolingual_vocab_file, out_monolingual_vocab_file)
        # 如果当前单语词汇表文件不存在，则将 Fairseq 格式的 token 写入输出文件
        elif not os.path.isfile(self.monolingual_vocab_file):
            with open(out_monolingual_vocab_file, "w", encoding="utf-8") as fp:
                for token in self.fairseq_tokens_to_ids:
                    # 排除特殊 token，并将其写入文件
                    if token not in self.all_special_tokens:
                        fp.write(f"{str(token)} \n")

        # 返回输出词汇表文件路径和输出单语词汇表文件路径
        return out_vocab_file, out_monolingual_vocab_file
```