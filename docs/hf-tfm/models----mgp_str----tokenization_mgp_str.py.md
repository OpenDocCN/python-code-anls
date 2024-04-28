# `.\transformers\models\mgp_str\tokenization_mgp_str.py`

```
# 导入所需的模块和依赖
import json
import os
from typing import Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 vocab 文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

# 定义预训练的 vocab 文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "mgp-str": "https://huggingface.co/alibaba-damo/mgp-str-base/blob/main/vocab.json",
    }
}

# 定义预训练的 positional embeddings 大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"mgp-str": 27}

# 定义 MgpstrTokenizer 类
class MgpstrTokenizer(PreTrainedTokenizer):
    """
    构建 MGP-STR char tokenizer。

    此 tokenizer 继承自 `PreTrainedTokenizer`，包含大多数主要方法。用户应该参考此超类以了解更多相关信息。

    参数:
        vocab_file (`str`):
            词汇表文件的路径。
        unk_token (`str`, *optional*, 默认为 `"[GO]"`):
            未知标记。不在词汇表中的标记将被设置为此标记。
        bos_token (`str`, *optional*, 默认为 `"[GO]"`):
            句子开始标记。
        eos_token (`str`, *optional*, 默认为 `"[s]"`):
            句子结束标记。
        pad_token (`str` or `tokenizers.AddedToken`, *optional*, 默认为 `"[GO]"`):
            用于填充使数组大小相同以进行批处理的特殊标记。将被注意力机制或损失计算忽略。
    """

    # 定义 vocab 文件名称、预训练的 vocab 文件映射和最大模型输入大小
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, unk_token="[GO]", bos_token="[GO]", eos_token="[s]", pad_token="[GO]", **kwargs):
        # 从 vocab 文件中读取词汇表
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.vocab = json.load(vocab_handle)
        # 构建反向词汇表（ID 到字符的映射）
        self.decoder = {v: k for k, v in self.vocab.items()}
        # 初始化父类的构造函数
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    # 获取词汇表大小
    @property
    def vocab_size(self):
        return len(self.vocab)

    # 获取完整的词汇表，包括添加的标记
    def get_vocab(self):
        vocab = dict(self.vocab).copy()
        vocab.update(self.added_tokens_encoder)
        return vocab
    def _tokenize(self, text):
        """Tokenize a string."""
        # 初始化一个空列表来存储字符级别的 token
        char_tokens = []
        # 遍历文本中的每个字符
        for s in text:
            # 将每个字符添加到 char_tokens 列表中
            char_tokens.extend(s)
        # 返回字符级别的 token 列表
        return char_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将 token 转换为对应的 id，如果 token 不存在于 vocab 中，则使用 unk_token 对应的 id
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将 index 转换为对应的 token
        return self.decoder.get(index)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            # 如果保存目录不存在，记录错误信息并返回
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        # 构建保存词汇表的文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 将词汇表写入文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            # 将 vocab 字典以 JSON 格式写入文件，每行一个词汇项，缩进为2个空格，保证 Unicode 字符的正确编码
            f.write(json.dumps(self.vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        # 返回保存的词汇表文件路径
        return (vocab_file,)
```