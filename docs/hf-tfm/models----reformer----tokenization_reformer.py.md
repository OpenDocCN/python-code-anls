# `.\transformers\models\reformer\tokenization_reformer.py`

```
# 设置文件编码为utf-8
# 版权声明，告知代码版权归Trax作者和HuggingFace团队所有，基于Apache 2.0许可证发布
# 只有在遵循许可证条件的情况下才能使用此文件，可以通过提供的链接获取许可证副本
# 在适用法律要求或书面同意的情况下，根据许可证分发的软件是基于"AS IS"的基础，没有任何形式的保证或条件，并受到特定语言的限制
# 请参阅许可证了解相关语言和限制

"""用于Reformer模型的分词类。"""

# 引入Python内置模块和第三方模块
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

# 引入库中的相关模块
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取logger对象
logger = logging.get_logger(__name__)

# 定义句子中的特殊符号
SPIECE_UNDERLINE = "▁"

# 词汇文件的文件名映射
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/reformer-crime-and-punishment": (
            "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/spiece.model"
        )
    }
}

# 预训练位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/reformer-crime-and-punishment": 524288,
}

# ReformerTokenizer继承自PreTrainedTokenizer类
class ReformerTokenizer(PreTrainedTokenizer):
    """
    构建一个Reformer分词器，基于SentencePiece实现。
    
    这个分词器继承自`PreTrainedTokenizer`，其中包含大部分主要方法。用户应参考该父类以获取更多有关这些方法的信息。
    """
```  
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        additional_special_tokens (`List[str]`, *optional*, defaults to `[]`):
            Additional special tokens used by the tokenizer.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        additional_special_tokens=[],
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()
    def get_vocab(self) -> Dict[str, int]:
        # 构建词汇表字典，将词汇映射到其对应的索引，索引从0到vocab_size-1
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 将已添加的特殊标记添加到词汇表字典中
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        # 复制对象的状态
        state = self.__dict__.copy()
        # 将sp_model设置为None，以防止序列化时将其保存
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        # 恢复对象的状态
        self.__dict__ = d

        # 为了向后兼容性
        # 如果对象没有sp_model_kwargs属性，则将其设置为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 重新加载SentencePieceProcessor对象，并设置其参数为之前保存的参数
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text: str) -> List[str]:
        """将输入的字符串进行分词，并返回一个字符串列表（标记）"""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """使用词汇表将标记（str）转换为对应的id"""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """使用词汇表将索引（整数）转换为对应的标记（str）"""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """将一系列标记（字符串）转换为单个字符串"""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # 确保特殊标记不会使用sentencepiece模型解码
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果词汇表文件的路径与输出路径不同且词汇表文件存在，则复制词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果词汇表文件不存在，则将序列化的SentencePiece模型写入输出路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)
```