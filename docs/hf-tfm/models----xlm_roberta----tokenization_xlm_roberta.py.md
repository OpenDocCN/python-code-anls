# `.\transformers\models\xlm_roberta\tokenization_xlm_roberta.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，指出代码的版权归属
# 根据 Apache 2.0 版本授权进行使用代码
# 如果使用此文件，必须符合 Apache 2.0 版本的授权
# 获取 Apache 2.0 版本授权的具体内容可参考指定链接
# 软件根据适用法律或书面同意提供，基于"原样"，不提供任何形式的担保，包括明示或暗示的担保
# 如需了解更多详细信息，请查阅许可证以确定权限和限制
# XLM-RoBERTa 模型的标记化类
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

# 导入 logging 模块
# 通过 logging 模块获取 logger 实例
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 特殊标记 "_" 表示连接词
SPIECE_UNDERLINE = "▁"

# 词汇文件名称
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

# 预训练词汇文件映射表
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "xlm-roberta-base": "https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model",
        "xlm-roberta-large": "https://huggingface.co/xlm-roberta-large/resolve/main/sentencepiece.bpe.model",
        "xlm-roberta-large-finetuned-conll02-dutch": (
            "https://huggingface.co/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/sentencepiece.bpe.model"
        ),
        "xlm-roberta-large-finetuned-conll02-spanish": (
            "https://huggingface.co/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/sentencepiece.bpe.model"
        ),
        "xlm-roberta-large-finetuned-conll03-english": (
            "https://huggingface.co/xlm-roberta-large-finetuned-conll03-english/resolve/main/sentencepiece.bpe.model"
        ),
        "xlm-roberta-large-finetuned-conll03-german": (
            "https://huggingface.co/xlm-roberta-large-finetuned-conll03-german/resolve/main/sentencepiece.bpe.model"
        ),
    }
}

# 预训练位置嵌入大小映射表
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "xlm-roberta-base": 512,
    "xlm-roberta-large": 512,
    "xlm-roberta-large-finetuned-conll02-dutch": 512,
    "xlm-roberta-large-finetuned-conll02-spanish": 512,
    "xlm-roberta-large-finetuned-conll03-english": 512,
    "xlm-roberta-large-finetuned-conll03-german": 512,
}

# XLM-RoBERTa 标记器类，继承自 PreTrainedTokenizer
# 适配自 RobertaTokenizer 和 XLNetTokenizer，基于 SentencePiece
# 用户应参考超类以了解更多关于这些方法的信息
class XLMRobertaTokenizer(PreTrainedTokenizer):
    Args:
        # 词汇表文件的路径
        vocab_file (`str`):
        # 可选参数，序列的开始标记，在预训练期间使用的符号。可以作为序列分类器的标记。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            <Tip>

            # 当使用特殊标记构建序列时，这不是用于序列开头的标记。用于序列开始的标记是`cls_token`。

            </Tip>
            
        # 可选参数，序列的末尾标记
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            
            <Tip>

            # 当使用特殊标记构建序列时，这不是用于序列结尾的标记。用于序列结束的标记是`sep_token`。

            </Tip>
            
        # 可选参数，序列的分隔标记。在从多个序列构建序列时使用，例如序列分类的两个序列或用于问题回答的文本和问题。也用作使用特殊标记构建序列的最后一个标记。
        sep_token (`str`, *optional*, defaults to `"</s>"`):
        
        # 可选参数，用于做序列分类（整个序列的分类，而不是逐令牌分类）时使用的分类器标记。在使用特殊标记构建时是序列的第一个标记。
        cls_token (`str`, *optional*, defaults to `"<s>"`):
        
        # 可选参数，未知标记。词汇表中没有的标记无法转换为ID，而是设置为该标记。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
        
        # 可选参数，用于填充的标记，例如在对不同长度的序列进行批处理时使用。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
        
        # 可选参数，用于掩码值的标记。在训练具有掩码语言建模的模型时使用的标记。模型将尝试预测该标记。
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
        
        # 可选参数，将传递给`SentencePieceProcessor.__init__()`方法的参数字典。可以用于设置`SentencePiece`的Python包装器，例如：

        - `enable_sampling`：启用子词正则化。
        - `nbest_size`：unigram的采样参数。对于BPE-Dropout无效。

          - `nbest_size = {0,1}`：不执行采样。
          - `nbest_size > 1`：从nbest_size结果中采样。
          - `nbest_size < 0`：假设nbest_size是无限的，并使用前向过滤和后向采样算法从所有假设（网格）中采样。

        - `alpha`：unigram采样的平滑参数，以及BPE-dropout的合并操作的丢失概率。

        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:
    class SentencePieceTokenizer(BertTokenizer):
        # 负责进行字符串、标记和标识符之间的转换的 SentencePiece 处理器
        """
        def __init__(
            self,
            vocab_file,
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
            # 如果 mask token 是字符串，将其作为特殊字符并去掉左边空格
            mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token
            # 初始化 sp_model_kwargs 字典，若无输入则为空字典
            self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
            # 使用 spm 库创建 SentencePieceProcessor 对象
            self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
            # 从 vocab_file 加载模型
            self.sp_model.Load(str(vocab_file))
            # 记录 vocab_file
            self.vocab_file = vocab_file
            
            # 对齐 fairseq 和 spm 词汇表，创建 fairseq token 到 id 的映射
            self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
            # fairseq 词汇表中第一个“真实” token "," 在 fairseq 和 spm 词汇表中的位置偏移
            self.fairseq_offset = 1
            # 为 fairseq 的特殊 token "<mask>" 创建映射关系
            self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + self.fairseq_offset
            # 创建 fairseq id 到 token 的映射字典
            self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}
            
            # 调用父类初始化方法
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
            # 复制对象的状态字典
            state = self.__dict__.copy()
            # 将 sp_model 设为 None，保存 serialized_model_proto
            state["sp_model"] = None
            state["sp_model_proto"] = self.sp_model.serialized_model_proto()
            return state
    
        def __setstate__(self, d):
            # 恢复对象的状态字典
            self.__dict__ = d
            
            # 兼容旧版本
            if not hasattr(self, "sp_model_kwargs"):
                self.sp_model_kwargs = {}
            
            # 使用 spm 库创建 SentencePieceProcessor 对象，并从序列化的 Proto 加载模型
            self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
            self.sp_model.LoadFromSerializedProto(self.sp_model_proto)
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLM-RoBERTa sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        if token_ids_1 is None:
            # 如果只有一个输入序列，添加CLS和SEP特殊标记
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 否则，对于输入序列对，添加CLS、SEP特殊标记
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
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

        if already_has_special_tokens:
            # 如果已经存在特殊标记，调用父类的方法
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            # 如果只有一个输入序列，创建特殊标记掩码
            return [1] + ([0] * len(token_ids_0)) + [1]
        # 否则，对于输入序列对，创建特殊标记掩码
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
    # 创建一个用于序列对分类任务的掩码
    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传递的两个序列中创建一个掩码，用于序列对分类任务。 XLM-RoBERTa 不使用 token type ids，因此返回一个全 0 的列表。
    
        参数:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                可选的第二个 ID 列表，用于序列对。
    
        返回:
            `List[int]`: 全 0 的列表。
        """
        # 分隔符和分类符 token
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
    
        # 如果只有一个序列，返回全 0 的列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 如果有两个序列，返回全 0 的列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
    
    # 获取词表大小
    @property
    def vocab_size(self):
        # 加上 <mask> token，返回词表大小
        return len(self.sp_model) + self.fairseq_offset + 1
    
    # 获取完整词表
    def get_vocab(self):
        # 创建从 token 到 id 的词表
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 更新已添加的 token
        vocab.update(self.added_tokens_encoder)
        return vocab
    
    # 对输入文本进行标记化
    def _tokenize(self, text: str) -> List[str]:
        # 使用 sentencepiece 对文本进行标记化
        return self.sp_model.encode(text, out_type=str)
    
    # 将 token 转换为 id
    def _convert_token_to_id(self, token):
        """将 token (str) 转换为 id，使用词表。"""
        # 如果 token 在 fairseq 的词表中，返回对应 id
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        # 使用 sentencepiece 模型将 token 转换为 id
        spm_id = self.sp_model.PieceToId(token)
        # 如果 spm_id 为 0，返回未知 token 的 id
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id
    
    # 将 id 转换为 token
    def _convert_id_to_token(self, index):
        """将 id (整数) 转换为 token (str)，使用词表。"""
        # 如果 id 在 fairseq 的词表中，返回对应 token
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        # 使用 sentencepiece 模型将 id 转换为 token
        return self.sp_model.IdToPiece(index - self.fairseq_offset)
    
    # 将一个 token 序列转换为字符串
    def convert_tokens_to_string(self, tokens):
        """将一个 token (sub-word) 序列转换为单个字符串。"""
        # 拼接 token，去除下划线，并去除前后空格
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string
    
    # 保存词表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存路径是否为目录
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建保存路径和文件名
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 如果当前词表路径和目标路径不同，且当前词表存在，则拷贝到目标路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词表不存在，则将 sentencepiece 模型序列化后保存到目标路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
        # 返回保存的词表文件路径
        return (out_vocab_file,)
```