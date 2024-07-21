# `.\pytorch\tools\testing\target_determination\heuristics\edited_by_pr.py`

```
from __future__ import annotations
# 导入用于支持类型注解的特性

from typing import Any
# 导入用于类型提示的模块

from warnings import warn
# 导入警告模块中的warn函数

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
# 从指定路径导入HeuristicInterface和TestPrioritizations类

from tools.testing.target_determination.heuristics.utils import (
    python_test_file_to_test_name,
    query_changed_files,
)
# 从指定路径导入python_test_file_to_test_name和query_changed_files函数

from tools.testing.test_run import TestRun
# 从指定路径导入TestRun类

class EditedByPR(HeuristicInterface):
    # 定义一个继承自HeuristicInterface的类EditedByPR

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        # 初始化方法，接受任意关键字参数
        super().__init__(**kwargs)
        # 调用父类HeuristicInterface的初始化方法

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # 获取预测置信度的方法，接受一个字符串列表作为参数tests
        critical_tests = _get_modified_tests()
        # 调用局部函数_get_modified_tests()获取修改过的关键测试
        return TestPrioritizations(
            tests, {TestRun(test): 1 for test in critical_tests if test in tests}
        )
        # 返回一个TestPrioritizations对象，其中包含传入的tests和修改过的关键测试的字典映射

def _get_modified_tests() -> set[str]:
    # 定义一个返回修改过测试集合的局部函数_get_modified_tests
    try:
        changed_files = query_changed_files()
        # 尝试调用query_changed_files()函数获取修改过的文件集合
    except Exception as e:
        warn(f"Can't query changed test files due to {e}")
        # 如果发生异常，则发出警告并提示异常信息
        # 如果无法从git获取修改的文件，则返回空集合
        return set()

    return python_test_file_to_test_name(set(changed_files))
    # 调用python_test_file_to_test_name函数并传入修改文件的集合，返回测试名称的集合
```