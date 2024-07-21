# `.\pytorch\tools\testing\target_determination\heuristics\historical_class_failure_correlation.py`

```
from __future__ import annotations
# 导入用于支持类型提示的特性

import json
# 导入用于处理 JSON 格式的模块
import os
# 导入用于处理操作系统相关功能的模块
from collections import defaultdict
# 导入用于创建默认字典的模块
from typing import Any, cast, Dict
# 导入用于类型提示的相关功能
from warnings import warn
# 导入用于发出警告的模块

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TEST_CLASS_RATINGS_FILE,
)
# 导入来自本地库的相关路径和文件名
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
# 导入测试优先级相关的接口和类型
from tools.testing.target_determination.heuristics.utils import (
    normalize_ratings,
    query_changed_files,
    REPO_ROOT,
)
# 导入用于优先级计算的工具函数
from tools.testing.test_run import TestRun
# 导入测试运行相关的类


class HistoricalClassFailurCorrelation(HeuristicInterface):
    """
    This heuristic prioritizes test classes that have historically tended to fail
    when the files edited by current PR were modified.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 调用父类构造函数，初始化对象

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        ratings = _get_ratings_for_tests(set(tests))
        # 获取测试的历史评级
        test_ratings = {
            TestRun(k): v for (k, v) in ratings.items() if TestRun(k).test_file in tests
        }
        # 将测试和评级关联起来
        return TestPrioritizations(tests, normalize_ratings(test_ratings, 0.25))
        # 返回测试的优先级


def _get_historical_test_class_correlations() -> dict[str, dict[str, float]]:
    path = REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / TEST_CLASS_RATINGS_FILE
    # 构建历史测试类相关性文件路径
    if not os.path.exists(path):
        print(f"could not find path {path}")
        return {}
        # 如果路径不存在，打印信息并返回空字典
    with open(path) as f:
        test_class_correlations = cast(Dict[str, Dict[str, float]], json.load(f))
        # 打开文件并加载 JSON 数据
        return test_class_correlations
        # 返回测试类相关性的字典数据


def _get_ratings_for_tests(
    tests_to_run: set[str],
) -> dict[str, float]:
    # 获取被修改的文件
    try:
        changed_files = query_changed_files()
        # 查询已修改的文件列表
    except Exception as e:
        warn(f"Can't query changed test files due to {e}")
        return {}
        # 如果查询失败，发出警告并返回空字典

    test_class_correlations = _get_historical_test_class_correlations()
    # 获取历史测试类相关性数据
    if not test_class_correlations:
        return {}
        # 如果没有相关数据，返回空字典

    ratings: dict[str, float] = defaultdict(float)
    # 创建默认值为浮点数的评级字典
    for file in changed_files:
        for qualified_test_class, score in test_class_correlations.get(
            file, {}
        ).items():
            # 遍历文件的相关测试类和得分
            # qualified_test_class 形如 "test_file::test_class"
            test_file, test_class = qualified_test_class.split("::")
            # 拆分测试文件和测试类
            if test_file in tests_to_run:
                ratings[qualified_test_class] += score
                # 如果测试文件在运行列表中，增加评级得分

    return ratings
    # 返回评级字典


def _rank_correlated_tests(
    tests_to_run: list[str],
) -> list[str]:
    # 查找与修改的文件相关的测试失败
    # 过滤列表以包含要运行的测试
    tests_to_run = set(tests_to_run)
    # 将测试列表转换为集合
    ratings = _get_ratings_for_tests(tests_to_run)
    # 获取测试评级
    prioritize = sorted(ratings, key=lambda x: -ratings[x])
    # 根据评级排序测试
    return prioritize
    # 返回排序后的测试列表
```