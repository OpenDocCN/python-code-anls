# `.\pytorch\tools\testing\target_determination\heuristics\previously_failed_in_pr.py`

```py
from __future__ import annotations
# 导入用于支持类型注解的模块

import json
# 导入处理 JSON 数据的模块
import os
# 导入操作系统相关功能的模块
from pathlib import Path
# 导入处理文件路径的模块
from typing import Any
# 导入用于类型提示的模块

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TD_HEURISTIC_PREVIOUSLY_FAILED,
    TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL,
)
# 从指定模块导入相关常量

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
# 导入接口和测试优先级相关的类和方法

from tools.testing.target_determination.heuristics.utils import (
    python_test_file_to_test_name,
)
# 导入工具方法，将 Python 测试文件转换为测试名称

from tools.testing.test_run import TestRun
# 导入测试运行相关的类

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
# 获取项目根目录的绝对路径

class PreviouslyFailedInPR(HeuristicInterface):
    # 定义一个类，继承自HeuristicInterface接口

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        # 调用父类的构造方法，初始化对象

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # 定义一个方法，返回测试优先级预测
        critical_tests = get_previous_failures() | read_additional_test_failures_file()
        # 获取之前失败的测试集合和额外失败测试集合的并集
        return TestPrioritizations(
            tests, {TestRun(test): 1 for test in critical_tests if test in tests}
        )
        # 返回测试优先级对象，包含测试和其优先级映射关系


def get_previous_failures() -> set[str]:
    # 定义一个函数，返回之前失败的测试集合
    path = REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / TD_HEURISTIC_PREVIOUSLY_FAILED
    # 拼接文件路径
    if not os.path.exists(path):
        # 如果路径不存在
        print(f"could not find path {path}")
        return set()
        # 打印路径未找到的信息，并返回空集合
    with open(path) as f:
        return python_test_file_to_test_name(
            _parse_prev_failing_test_files(json.load(f))
        )
        # 打开文件，解析其中的失败测试数据，并将结果返回


def _parse_prev_failing_test_files(last_failed_tests: dict[str, bool]) -> set[str]:
    # 定义一个函数，解析上次失败的测试文件
    prioritized_tests = set()

    # The keys are formatted as "test_file.py::test_class::test_method[params]"
    # We just need the test_file part
    # 键的格式为“test_file.py::test_class::test_method[params]”，我们只需要测试文件部分
    for test in last_failed_tests:
        parts = test.split("::")
        # 分割测试名称
        if len(parts) > 1:
            test_file = parts[0]
            prioritized_tests.add(test_file)
            # 将测试文件名称添加到集合中

    return prioritized_tests
    # 返回优先级测试集合


def gen_additional_test_failures_file(tests: list[str]) -> None:
    # 定义一个函数，生成额外的测试失败文件
    # Segfaults usually result in no xml and some tests don't run through pytest
    # (ex doctests).  In these cases, there will be no entry in the pytest
    # cache, so we should generate a separate file for them and upload it to s3
    # along with the pytest cache
    # Segfaults通常导致没有xml文件，并且一些测试不能通过pytest运行（例如doctests）。
    # 在这些情况下，pytest缓存中将没有条目，因此我们应该为它们生成一个单独的文件，并将其上传到S3，以及pytest缓存
    pytest_cache_dir = REPO_ROOT / ".pytest_cache"
    # 设置pytest缓存目录
    if not os.path.exists(pytest_cache_dir):
        os.makedirs(pytest_cache_dir)
        # 如果目录不存在，则创建目录
    with open(pytest_cache_dir / TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL, "w") as f:
        json.dump(tests, f, indent=2)
        # 将测试数据写入文件中，格式化缩进为2


def read_additional_test_failures_file() -> set[str]:
    # 定义一个函数，读取额外的测试失败文件
    path = (
        REPO_ROOT
        / ADDITIONAL_CI_FILES_FOLDER
        / TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL
    )
    # 拼接文件路径
    if not os.path.exists(path):
        # 如果路径不存在
        print(f"could not find path {path}")
        return set()
        # 打印路径未找到的信息，并返回空集合
    with open(path) as f:
        s = set(json.load(f))
        print(f"additional failures: {s}")
        return s
        # 打开文件，解析其中的失败测试数据，并将结果返回
```