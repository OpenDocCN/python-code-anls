# `.\pytorch\tools\testing\target_determination\heuristics\profiling.py`

```
# 从未来模块中导入注解功能，用于类型提示
from __future__ import annotations

# 导入类型提示相关模块
from typing import Any

# 导入具体文件路径和文件名
from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TD_HEURISTIC_PROFILING_FILE,
)

# 导入测试优先级接口和启发式算法的文件
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)

# 导入测试工具函数：获取测试的评分和评分标准化
from tools.testing.target_determination.heuristics.utils import (
    get_ratings_for_tests,
    normalize_ratings,
)

# 导入测试运行模块
from tools.testing.test_run import TestRun


# Profilers were used to gather simple python code coverage information for each
# test to see files were involved in each tests and used to build a correlation
# dict (where all ratings are 1).
# 定义一个继承自HeuristicInterface的类Profiling，用于测试的启发式分析
class Profiling(HeuristicInterface):
    
    # 初始化方法，接收任意关键字参数
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
    
    # 方法用于获取测试的预测置信度，输入参数为测试的列表，返回测试的优先级
    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # 获取测试的评分，通过测试文件的额外CI文件夹和启发式分析文件路径获取
        test_ratings = get_ratings_for_tests(
            ADDITIONAL_CI_FILES_FOLDER / TD_HEURISTIC_PROFILING_FILE
        )
        # 从评分中过滤出测试存在的项目，并转换为TestRun对象的字典
        test_ratings = {TestRun(k): v for (k, v) in test_ratings.items() if k in tests}
        # 根据给定的置信度系数对评分进行标准化，构建测试优先级对象返回
        return TestPrioritizations(tests, normalize_ratings(test_ratings, 0.25))
```