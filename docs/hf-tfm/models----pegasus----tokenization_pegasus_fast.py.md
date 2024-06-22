# `.\transformers\models\pegasus\tokenization_pegasus_fast.py`

```py
# 设定文件编码为 utf-8
# 版权声明及许可信息
# 声明 Tokenization class 用于 PEGASUS 模型
import os  # 导入操作系统模块
from shutil import copyfile  # 从 shutil 模块中导入 copyfile 函数
from typing import List, Optional, Tuple  # 导入类型提示模块

# 导入必要的模块和函数
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

# 如果 sentencepiece 可用，则导入 PegasusTokenizer，否则设为 None
if is_sentencepiece_available():
    from .tokenization_pegasus import PegasusTokenizer
else:
    PegasusTokenizer = None

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义常量
# 单词起始标记
SPIECE_UNDERLINE = "▁"
# 词汇文件名映射
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}
# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"google/pegasus-xsum": "https://huggingface.co/google/pegasus-xsum/resolve/main/spiece.model"},
    "tokenizer_file": {"google/pegasus-xsum": "https://huggingface.co/google/pegasus-xsum/resolve/main/tokenizer.json"},
}
# 预训练位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"google/pegasus-xsum": 512}

# 定义 PegasusTokenizerFast 类，继承自 PreTrainedTokenizerFast 类
class PegasusTokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个 "快速" PEGASUS 分词器（基于 HuggingFace 的 *tokenizers* 库）。基于 [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models)。

    该分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大部分主要方法。用户应参考此超类以获取有关这些方法的更多信息。
    """
    # 定义了输入所需的参数，包括:
    # - vocab_file: SentencePiece 文件路径，包含构建 tokenizer 所需的词汇表
    # - pad_token: 填充token, 用于对不同长度的序列进行 batching
    # - eos_token: 序列结束 token
    # - unk_token: 未知 token, 用于处理不在词汇表中的词
    # - mask_token: 用于遮蔽单个 token 以进行 MLM (Masked Language Modeling) 预训练
    # - mask_token_sent: 用于遮蔽整个目标句子以进行 GSG (Gap Sentences Generation) 预训练
    # - additional_special_tokens: 额外的特殊 token, 如果没有提供则使用一些默认值
    
    # 定义了一些与模型相关的常量, 如 vocab_files_names, pretrained_vocab_files_map, max_model_input_sizes, slow_tokenizer_class, model_input_names
    # 这是 PegasusTokenizer 的 __init__ 方法
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask_2>",
        mask_token_sent="<mask_1>",
        additional_special_tokens=None,
        offset=103,  # entries 2 - 104 are only used for pretraining
        **kwargs,
    ):
        # 设置 offset 属性
        self.offset = offset
    
        # 如果 additional_special_tokens 不为 None
        if additional_special_tokens is not None:
            # 如果 additional_special_tokens 不是列表类型，抛出异常
            if not isinstance(additional_special_tokens, list):
                raise TypeError(
                    f"additional_special_tokens should be of type {type(list)}, but is"
                    f" {type(additional_special_tokens)}"
                )
            # 如果 mask_token_sent 不在 additional_special_tokens 中且不为 None，则添加到列表
            additional_special_tokens_extended = (
                ([mask_token_sent] + additional_special_tokens)
                if mask_token_sent not in additional_special_tokens and mask_token_sent is not None
                else additional_special_tokens
            )
            # 如果 additional_special_tokens_extended 长度不足 self.offset - 1 个，填充 <unk_x> 占位符
            additional_special_tokens_extended += [
                f"<unk_{i}>" for i in range(len(additional_special_tokens_extended), self.offset - 1)
            ]
            # 如果 additional_special_tokens_extended 存在重复，抛出异常
            if len(set(additional_special_tokens_extended)) != len(additional_special_tokens_extended):
                raise ValueError(
                    "Please make sure that the provided additional_special_tokens do not contain an incorrectly"
                    f" shifted list of <unk_x> tokens. Found {additional_special_tokens_extended}."
                )
            # 将 additional_special_tokens_extended 赋值给 additional_special_tokens
            additional_special_tokens = additional_special_tokens_extended
        # 如果 additional_special_tokens 为 None
        else:
            # 如果 mask_token_sent 不为 None，添加到 additional_special_tokens
            additional_special_tokens = [mask_token_sent] if mask_token_sent is not None else []
            # 添加 <unk_x> 占位符到 additional_special_tokens
            additional_special_tokens += [f"<unk_{i}>" for i in range(2, self.offset)]
    
        # 如果 pad_token、eos_token 或 unk_token 与默认值不同，需要重建词汇表
        from_slow = kwargs.pop("from_slow", None)
        from_slow = from_slow or str(pad_token) != "<pad>" or str(eos_token) != "</s>" or str(unk_token) != "<unk>"
    
        # 删除 added_tokens_decoder 属性
        kwargs.pop("added_tokens_decoder", {})
    
        # 调用父类 __init__ 方法，传入相关参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            mask_token=mask_token,
            mask_token_sent=mask_token_sent,
            offset=offset,
            additional_special_tokens=additional_special_tokens,
            from_slow=from_slow,
            **kwargs,
        )
        # 设置 vocab_file 属性
        self.vocab_file = vocab_file
    
    # 这是 PegasusTokenizer 的 can_save_slow_tokenizer 属性
    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 如果 vocab_file 存在，返回 True，否则返回 False
        return os.path.isfile(self.vocab_file) if self.vocab_file else False
    # 定义一个私有函数 _special_token_mask，接收一个序列并返回一个与序列长度相同的列表，
    # 列表中的值为 1 表示是特殊 token，否则为 0
    def _special_token_mask(self, seq):
        # 创建一个包含所有特殊 token 的集合
        all_special_ids = set(self.all_special_ids)  # 只调用一次以优化性能
        # 移除 <unk> token，因为它不是总是被认为是特殊 token
        all_special_ids.remove(self.unk_token_id)  # <unk> 有时是特殊的，但不总是
    
        # 确保所有特殊 token 集合的大小与期望一致，否则抛出错误
        if all_special_ids != set(range(len(self.additional_special_tokens) + 3)):
            raise ValueError(
                "应该有 3 个特殊 token: mask_token, pad_token 和 eos_token，以及"
                f" {len(self.additional_special_tokens)} 个附加特殊 token，但得到的是 {all_special_ids}"
            )
    
        # 返回一个列表，列表中的值为 1 如果 token 属于特殊 token 集合，否则为 0
        return [1 if x in all_special_ids else 0 for x in seq]
    
    # 定义一个公共函数 get_special_tokens_mask，接收两个 token 序列和一个布尔标志，
    # 返回一个标识序列中特殊 token 的列表
    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        # 如果序列已经包含特殊 token，则直接返回第一个序列的特殊 token mask
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        # 如果第二个序列为空，则返回第一个序列的特殊 token mask 加上结束 token
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [1]
        # 如果两个序列都存在，则返回合并后的序列的特殊 token mask，加上结束 token
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]
    
    # 定义一个公共函数 build_inputs_with_special_tokens，接收两个 token 序列，返回添加了特殊 token 的序列
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        # 如果只有一个序列，则在末尾添加结束 token
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # 如果有两个序列，则在末尾添加第二个序列和结束 token
        return token_ids_0 + token_ids_1 + [self.eos_token_id]
    
    # 定义一个公共函数 save_vocabulary，接收目录路径和可选的文件名前缀，保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果 tokenizer 无法保存词汇表，则抛出错误
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "你的快速 tokenizer 没有必要的信息来保存用于慢速 tokenizer 的词汇表。"
            )
    
        # 如果保存路径不是目录，则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"词汇表路径 ({save_directory}) 应该是一个目录")
            return
        
        # 构建词汇表文件的路径，可能带有文件名前缀
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
    
        # 如果原始词汇表文件路径和目标路径不同，则复制原始词汇表到目标路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
    
        # 返回生成的词汇表文件路径
        return (out_vocab_file,)
```