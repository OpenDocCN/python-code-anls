# `.\pytorch\tools\testing\target_determination\heuristics\correlated_with_historical_failures.py`

```py
from __future__ import annotations

# 引入将来版本的注解支持，使得可以在类定义中使用字符串类型的注解


from typing import Any

# 引入类型提示中的 Any 类型，表示可以是任意类型的对象


from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TEST_FILE_RATINGS_FILE,
)

# 从 tools.stats.import_test_stats 模块中导入 ADDITIONAL_CI_FILES_FOLDER 和 TEST_FILE_RATINGS_FILE 常量


from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)

# 从 tools.testing.target_determination.heuristics.interface 模块中导入 HeuristicInterface 和 TestPrioritizations 类


from tools.testing.target_determination.heuristics.utils import (
    get_ratings_for_tests,
    normalize_ratings,
)

# 从 tools.testing.target_determination.heuristics.utils 模块中导入 get_ratings_for_tests 和 normalize_ratings 函数


from tools.testing.test_run import TestRun

# 从 tools.testing.test_run 模块中导入 TestRun 类


class CorrelatedWithHistoricalFailures(HeuristicInterface):

# 定义 CorrelatedWithHistoricalFailures 类，继承自 HeuristicInterface 类


    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)

# 类的初始化方法，接受任意关键字参数并将其传递给父类 HeuristicInterface 的初始化方法


    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:

# 定义 get_prediction_confidence 方法，接受一个字符串列表 tests 作为参数，并返回 TestPrioritizations 对象


        test_ratings = get_ratings_for_tests(
            ADDITIONAL_CI_FILES_FOLDER / TEST_FILE_RATINGS_FILE
        )

# 调用 get_ratings_for_tests 函数，传入 ADDITIONAL_CI_FILES_FOLDER / TEST_FILE_RATINGS_FILE 路径作为参数，并将结果赋给 test_ratings 变量


        test_ratings = {TestRun(k): v for (k, v) in test_ratings.items() if k in tests}

# 使用字典推导式将 test_ratings 字典中符合条件的键值对重新组装成 TestRun(k) 对象和 v 值的形式，并筛选出键在 tests 列表中的项


        return TestPrioritizations(tests, normalize_ratings(test_ratings, 0.25))

# 返回一个 TestPrioritizations 对象，传入 tests 列表和经过 normalize_ratings 处理后的 test_ratings 字典，0.25 是归一化系数
```