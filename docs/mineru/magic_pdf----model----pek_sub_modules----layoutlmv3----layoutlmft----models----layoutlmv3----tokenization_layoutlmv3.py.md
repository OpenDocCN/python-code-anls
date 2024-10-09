# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\layoutlmft\models\layoutlmv3\tokenization_layoutlmv3.py`

```
# 指定文件编码为 UTF-8
# copyright 版权信息，说明版权所有者及团队
#
# 根据 Apache License 2.0 许可发布此文件
# 在使用此文件之前必须遵守许可协议
# 可在以下链接获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非根据适用法律或书面协议另有约定，否则软件按 "现状" 基础分发
# 不提供任何形式的担保或条件，包括明示或暗示的担保
# 查看许可证以获取特定权限和限制
"""LayoutLMv3 的分词类，参考 RoBERTa。"""

# 从 transformers 库中导入 RoBERTa 的分词器
from transformers.models.roberta import RobertaTokenizer
# 从 transformers.utils 导入日志记录模块
from transformers.utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名的字典
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",  # 词汇文件名称
    "merges_file": "merges.txt",  # 合并文件名称
}

# 创建 LayoutLMv3Tokenizer 类，继承自 RobertaTokenizer
class LayoutLMv3Tokenizer(RobertaTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES  # 赋值词汇文件名字典
    # pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练词汇文件映射（已注释）
    # max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 最大模型输入大小（已注释）
    model_input_names = ["input_ids", "attention_mask"]  # 定义模型输入名称
```