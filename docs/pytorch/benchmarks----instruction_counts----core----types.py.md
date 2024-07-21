# `.\pytorch\benchmarks\instruction_counts\core\types.py`

```
# 引入必要的类型注解模块
from typing import Any, Dict, Optional, Tuple, Union

# 从核心 API 导入必要的对象
from core.api import AutoLabels, GroupedBenchmark, TimerArgs

# =============================================================================
# == Benchmark schema =========================================================
# =============================================================================

# 长注释段落：定义了如何表示基准测试的最终状态，以及示例展示了不同层次结构的数据表示方式

# 主键的类型定义，是一个元组的元组
Label = Tuple[str, ...]

# 辅助的主键类型定义，可以是主键元组或可选的字符串（用于表示无需进一步条目的基本情况）
_Label = Union[Label, Optional[str]]

# 主值类型的定义，可以是计时器参数或分组基准类型，或者是一个递归的字典结构
_Value = Union[
    Union[TimerArgs, GroupedBenchmark],  # 基本情况
    Dict[_Label, Any],  # 递归情况
]

# 定义的主要数据结构，是一个字典，其键是_Label类型，值是_Value类型
Definition = Dict[_Label, _Value]

# 为了构建 TorchScript 模型，我们首先需要将定义解析（扁平化）到一个中间状态
FlatIntermediateDefinition = Dict[Label, Union[TimerArgs, GroupedBenchmark]]

# 最终的解析后的数据结构类型定义，是一个元组的元组
FlatDefinition = Tuple[Tuple[Label, AutoLabels, TimerArgs], ...]
```