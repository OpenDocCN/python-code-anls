# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\layoutlmft\models\layoutlmv3\tokenization_layoutlmv3_fast.py`

```
# 指定文件编码为 UTF-8
# 版权声明，标明文件的版权所有者和许可证信息
# 根据 Apache 许可证，指明使用此文件的条件
# 提供许可证的获取地址
# 指明在适用情况下的免责声明
# 引入模块，说明此文件为 LayoutLMv3 的快速分词类，参考 RoBERTa 的实现

# 从 transformers 库导入 RobertaTokenizerFast 类
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
# 从 transformers 库导入日志模块
from transformers.utils import logging

# 导入 LayoutLMv3Tokenizer 类
from .tokenization_layoutlmv3 import LayoutLMv3Tokenizer

# 创建 logger 实例，用于日志记录
logger = logging.get_logger(__name__)

# 定义一个字典，包含词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 定义 LayoutLMv3TokenizerFast 类，继承自 RobertaTokenizerFast
class LayoutLMv3TokenizerFast(RobertaTokenizerFast):
    # 设置词汇文件名称
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇文件映射（注释掉的部分）
    # pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 最大模型输入大小（注释掉的部分）
    # max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 指定慢分词器类
    slow_tokenizer_class = LayoutLMv3Tokenizer
```