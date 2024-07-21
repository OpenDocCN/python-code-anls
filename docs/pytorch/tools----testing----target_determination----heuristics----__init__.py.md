# `.\pytorch\tools\testing\target_determination\heuristics\__init__.py`

```py
# 导入必要的模块和类型检查
from __future__ import annotations  # 启用类型提示的新语法支持
from typing import TYPE_CHECKING  # 导入类型检查模块

# 导入各种启发式算法类，用于目标确定测试
from tools.testing.target_determination.heuristics.correlated_with_historical_failures import (
    CorrelatedWithHistoricalFailures,
)
from tools.testing.target_determination.heuristics.edited_by_pr import EditedByPR
from tools.testing.target_determination.heuristics.filepath import Filepath
from tools.testing.target_determination.heuristics.historical_class_failure_correlation import (
    HistoricalClassFailurCorrelation,
)
from tools.testing.target_determination.heuristics.historical_edited_files import (
    HistorialEditedFiles,
)
from tools.testing.target_determination.heuristics.interface import (
    AggregatedHeuristics as AggregatedHeuristics,
    TestPrioritizations as TestPrioritizations,
)
from tools.testing.target_determination.heuristics.llm import LLM
from tools.testing.target_determination.heuristics.mentioned_in_pr import MentionedInPR
from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
    PreviouslyFailedInPR,
)
from tools.testing.target_determination.heuristics.profiling import Profiling

# 如果处于类型检查模式，导入HeuristicInterface接口
if TYPE_CHECKING:
    from tools.testing.target_determination.heuristics.interface import (
        HeuristicInterface as HeuristicInterface,
    )

# 所有当前运行的启发式算法实例列表
# 若要在试验模式中添加启发式算法，请指定关键字参数 `trial_mode=True`。
HEURISTICS: list[HeuristicInterface] = [
    PreviouslyFailedInPR(),  # 先前在PR中失败的启发式算法实例
    EditedByPR(),  # 被PR编辑的启发式算法实例
    MentionedInPR(),  # 在PR中提到的启发式算法实例
    HistoricalClassFailurCorrelation(trial_mode=True),  # 历史类失败相关性的启发式算法实例（试验模式）
    CorrelatedWithHistoricalFailures(),  # 与历史失败相关的启发式算法实例
    HistorialEditedFiles(),  # 历史编辑文件的启发式算法实例
    Profiling(),  # 性能分析的启发式算法实例
    LLM(),  # LLM模型的启发式算法实例
    Filepath(),  # 文件路径的启发式算法实例
]
```