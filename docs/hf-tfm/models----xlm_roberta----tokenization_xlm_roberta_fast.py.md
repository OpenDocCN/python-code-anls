# `.\transformers\models\xlm_roberta\tokenization_xlm_roberta_fast.py`

```py
# coding=utf-8
# 版权所有 2018 年 Google AI、Google Brain 和卡内基梅隆大学作者以及 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可;
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，
# 无任何形式的明示或暗示担保或条件。
# 有关许可证，请参阅许可证

""" XLM-RoBERTa 模型的标记化类。"""

# 导入所需的库
import os
from shutil import copyfile
from typing import List, Optional, Tuple

from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

# 检查是否安装了 sentencepiece 库
if is_sentencepiece_available():
    # 如果安装了 sentencepiece 库，则导入 XLM-RoBERTa 的标记化器
    from .tokenization_xlm_roberta import XLMRobertaTokenizer
else:
    # 如果未安装 sentencepiece 库，则将标记化器设置为 None
    XLMRobertaTokenizer = None

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型词汇文件的映射关系
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
    },
    # tokenizer_file 是一个包含不同模型的 tokenizer 文件链接的字典
    "tokenizer_file": {
        # xlm-roberta-base 模型的 tokenizer 链接
        "xlm-roberta-base": "https://huggingface.co/xlm-roberta-base/resolve/main/tokenizer.json",
        # xlm-roberta-large 模型的 tokenizer 链接
        "xlm-roberta-large": "https://huggingface.co/xlm-roberta-large/resolve/main/tokenizer.json",
        # xlm-roberta-large-finetuned-conll02-dutch 模型的 tokenizer 链接
        "xlm-roberta-large-finetuned-conll02-dutch": (
            "https://huggingface.co/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/tokenizer.json"
        ),
        # xlm-roberta-large-finetuned-conll02-spanish 模型的 tokenizer 链接
        "xlm-roberta-large-finetuned-conll02-spanish": (
            "https://huggingface.co/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/tokenizer.json"
        ),
        # xlm-roberta-large-finetuned-conll03-english 模型的 tokenizer 链接
        "xlm-roberta-large-finetuned-conll03-english": (
            "https://huggingface.co/xlm-roberta-large-finetuned-conll03-english/resolve/main/tokenizer.json"
        ),
        # xlm-roberta-large-finetuned-conll03-german 模型的 tokenizer 链接
        "xlm-roberta-large-finetuned-conll03-german": (
            "https://huggingface.co/xlm-roberta-large-finetuned-conll03-german/resolve/main/tokenizer.json"
        ),
    },
# 预训练位置嵌入的大小字典，包含了各种不同预训练模型的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "xlm-roberta-base": 512,  # XLM-RoBERTa-Base 模型的预训练位置嵌入大小为 512
    "xlm-roberta-large": 512,  # XLM-RoBERTa-Large 模型的预训练位置嵌入大小为 512
    "xlm-roberta-large-finetuned-conll02-dutch": 512,  # 经过Fine-tuning的XLM-RoBERTa-Large在conll02-dutch数据集上的预训练位置嵌入大小为 512
    "xlm-roberta-large-finetuned-conll02-spanish": 512,  # 经过Fine-tuning的XLM-RoBERTa-Large在conll02-spanish数据集上的预训练位置嵌入大小为 512
    "xlm-roberta-large-finetuned-conll03-english": 512,  # 经过Fine-tuning的XLM-RoBERTa-Large在conll03-english数据集上的预训练位置嵌入大小为 512
    "xlm-roberta-large-finetuned-conll03-german": 512,  # 经过Fine-tuning的XLM-RoBERTa-Large在conll03-german数据集上的预训练位置嵌入大小为 512
}


class XLMRobertaTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”的XLM-RoBERTa分词器（由HuggingFace的*tokenizers*库支持）。改编自[`RobertaTokenizer`]和[`XLNetTokenizer`]。基于[BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models)。

    该分词器继承自[`PreTrainedTokenizerFast`]，其中包含了大部分主要方法。用户应该参考这个超类来获取有关这些方法的更多信息。
    """
    # 定义了一个初始值为None的参数vocab_file，表示词汇表文件的路径
    # 定义了一个可选参数bos_token，默认值为"<s>"，表示在预训练期间用于序列开头的特殊标记，可用作序列分类器标记
    # 定义了一个可选参数eos_token，默认值为"</s>"，表示序列结束的特殊标记
    
    <Tip>
    # 当使用特殊标记构建序列时，这不是用于序列开头的标记。用于序列开头的标记是cls_token。
    </Tip>
    
    # 定义了一个可选参数sep_token，默认值为"</s>"，表示分隔符标记，在从多个序列构建一个序列时使用，例如用于序列分类或问题回答中的文本和问题。也用作使用特殊标记构建序列的最后一个标记
    # 定义了一个可选参数cls_token，默认值为"<s>"，表示用于进行序列分类（而不是每个标记的分类）时使用的分类器标记。在使用特殊标记构建序列时，它是序列的第一个标记
    # 定义了一个可选参数unk_token，默认值为"<unk>"，表示未知标记，词汇表中不存在的标记将被设置为该标记
    # 定义了一个可选参数pad_token，默认值为"<pad>"，表示用于填充的标记，例如在批处理不同长度的序列时使用
    # 定义了一个可选参数mask_token，默认值为"<mask>"，表示用于屏蔽值的标记，在进行屏蔽语言建模训练时使用。模型将尝试预测该标记
    # 定义了一个可选参数additional_special_tokens，默认值为["<s>NOTUSED", "</s>NOTUSED"]，表示分词器使用的额外特殊标记
    
    # 定义了变量vocab_files_names，其值为常量VOCAB_FILES_NAMES
    # 定义了变量pretrained_vocab_files_map，其值为常量PRETRAINED_VOCAB_FILES_MAP
    # 定义了变量max_model_input_sizes，其值为常量PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义了变量model_input_names，其值为包含两个字符串的列表，分别表示输入序列和注意力掩码
    # 定义了变量slow_tokenizer_class，其值为XLMRobertaTokenizer类
    
    # 定义了初始化函数__init__，接收参数vocab_file、tokenizer_file和一系列特殊标记，以及其他关键字参数
    ): 
        # 如果mask_token是字符串，则表示mask token是一个普通的单词，即在其前面包含空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 调用父类的初始化函数，传入参数为vocab_file和其他特殊token，**kwargs表示接受任意数量的关键字参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        # 将vocab_file赋值给实例变量self.vocab_file
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool: 
        # 如果vocab_file存在，则可以保存慢速的tokenizer
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

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
        # 如果只有一个token_ids_0，则加上cls_token_id和sep_token_id
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 分别为cls_token_id和sep_token_id创建列表
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
        not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 为cls_token_id和sep_token_id创建列表
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        # 如果只有一个token_ids_0，则返回长度为0的列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则返回长度为0的列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
    # 保存词汇表到指定目录下，返回保存的文件路径
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速标记器的词汇表，则抛出数值错误
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存的目录不存在，则打印错误日志并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            return
        # 设置输出文件的路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果词汇表的绝对路径与输出文件的绝对路径不同，则复制词汇表文件到输出文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回保存的文件路径
        return (out_vocab_file,)
``` 
```