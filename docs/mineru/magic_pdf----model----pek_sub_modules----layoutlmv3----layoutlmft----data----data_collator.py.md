# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\layoutlmft\data\data_collator.py`

```
# 导入 PyTorch 库，用于张量操作和计算
import torch
# 从 dataclasses 模块导入 dataclass 装饰器，用于定义数据类
from dataclasses import dataclass
# 从 typing 模块导入类型提示相关的类和类型
from typing import Any, Dict, List, Optional, Tuple, Union

# 从 transformers 库导入 BatchEncoding 和 PreTrainedTokenizerBase
from transformers import BatchEncoding, PreTrainedTokenizerBase
# 从 transformers.data.data_collator 导入 DataCollatorMixin 和 _torch_collate_batch
from transformers.data.data_collator import (
    DataCollatorMixin,
    _torch_collate_batch,
)
# 从 transformers.file_utils 导入 PaddingStrategy，用于处理填充策略
from transformers.file_utils import PaddingStrategy

# 从 typing 模块导入 NewType，用于创建新类型
from typing import NewType
# 定义一个新类型 InputDataClass，基于 Any 类型
InputDataClass = NewType("InputDataClass", Any)

# 定义一个函数 pre_calc_rel_mat，用于预计算关系矩阵
def pre_calc_rel_mat(segment_ids):
    # 创建一个与 segment_ids 形状相同的布尔张量，用于存储有效范围
    valid_span = torch.zeros((segment_ids.shape[0], segment_ids.shape[1], segment_ids.shape[1]),
                             device=segment_ids.device, dtype=torch.bool)
    # 遍历 segment_ids 的第一个维度
    for i in range(segment_ids.shape[0]):
        # 遍历 segment_ids 的第二个维度
        for j in range(segment_ids.shape[1]):
            # 设置有效范围，标记与 segment_ids 中当前元素相同的元素
            valid_span[i, j, :] = segment_ids[i, :] == segment_ids[i, j]

    # 返回计算得到的有效范围张量
    return valid_span

# 定义一个数据类 DataCollatorForKeyValueExtraction，继承自 DataCollatorMixin
@dataclass
class DataCollatorForKeyValueExtraction(DataCollatorMixin):
    """
    数据整理器，用于动态填充接收到的输入和标签。
    参数：
        tokenizer (:class:`~transformers.PreTrainedTokenizer` 或 :class:`~transformers.PreTrainedTokenizerFast`):
            用于编码数据的分词器。
        padding (:obj:`bool`, :obj:`str` 或 :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            选择填充返回序列的策略（根据模型的填充侧和填充索引）
            选项包括：
            * :obj:`True` 或 :obj:`'longest'`: 填充到批次中最长的序列（如果只提供单个序列，则不填充）。
            * :obj:`'max_length'`: 填充到通过 :obj:`max_length` 参数指定的最大长度，或者如果未提供该参数，则填充到模型的最大可接受输入长度。
            * :obj:`False` 或 :obj:`'do_not_pad'`（默认值）: 不填充（即，可以输出具有不同长度序列的批次）。
        max_length (:obj:`int`, `optional`):
            返回列表的最大长度，以及可选的填充长度（见上文）。
        pad_to_multiple_of (:obj:`int`, `optional`):
            如果设置，将填充序列到提供值的倍数。
            这在启用 NVIDIA 硬件上计算能力 >= 7.5（Volta）时尤其有用。
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            填充标签时使用的 ID (-100 将被 PyTorch 损失函数自动忽略)。
    """

    # 定义类的属性，包括分词器、填充策略、最大长度、倍数填充和标签填充标记 ID
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
```