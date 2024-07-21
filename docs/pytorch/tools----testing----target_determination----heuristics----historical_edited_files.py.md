# `.\pytorch\tools\testing\target_determination\heuristics\historical_edited_files.py`

```py
# 导入未来版本的注解支持
from __future__ import annotations

# 导入类型提示
from typing import Any

# 导入相关模块和变量
from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,  # 额外的持续集成文件夹路径
    TD_HEURISTIC_HISTORICAL_EDITED_FILES,  # 历史编辑文件的启发式
)
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,  # 启发式接口
    TestPrioritizations,  # 测试优先级
)
from tools.testing.target_determination.heuristics.utils import (
    get_ratings_for_tests,  # 获取测试评分
    normalize_ratings,  # 标准化评分
)
from tools.testing.test_run import TestRun  # 测试运行

# 此启发式假设之前提交中的更改文件是彼此相关的良好信息来源。
# 如果在主分支上的同一次提交中编辑了 fileA 和 testFileA，那么未来更改 fileA 的提交可能应该运行 testFileA。
# 基于此，建立一个基于主分支上的提交中编辑的文件的相关性字典。
class HistorialEditedFiles(HeuristicInterface):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    # 根据测试列表获取预测置信度
    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # 获取测试评分，基于历史编辑的文件
        test_ratings = get_ratings_for_tests(
            ADDITIONAL_CI_FILES_FOLDER / TD_HEURISTIC_HISTORICAL_EDITED_FILES
        )
        # 过滤出测试评分中存在于给定测试列表中的项，并转换为 TestRun 对象
        test_ratings = {TestRun(k): v for (k, v) in test_ratings.items() if k in tests}

        # 返回测试优先级对象，其中包含标准化后的评分
        return TestPrioritizations(tests, normalize_ratings(test_ratings, 0.25))
```