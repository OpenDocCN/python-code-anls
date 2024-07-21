# `.\pytorch\tools\test\heuristics\test_interface.py`

```py
from __future__ import annotations
# 导入用于未来版本兼容的特性

import sys
# 导入系统相关的模块

import unittest
# 导入单元测试框架模块

from pathlib import Path
# 导入处理路径的模块

from typing import Any
# 导入类型提示相关模块

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# 获取当前文件的上四级目录作为项目根目录路径

sys.path.append(str(REPO_ROOT))
# 将项目根目录路径添加到系统路径中，以便导入项目内的模块

import tools.testing.target_determination.heuristics.interface as interface
# 导入测试目标确定工具中启发式相关接口模块

from tools.testing.test_run import TestRun
# 导入测试运行模块中的 TestRun 类

sys.path.remove(str(REPO_ROOT))
# 移除项目根目录路径，避免影响其他模块导入路径

class TestTD(unittest.TestCase):
    # 定义测试用例类 TestTD，继承自 unittest.TestCase

    def assert_test_scores_almost_equal(
        self, d1: dict[TestRun, float], d2: dict[TestRun, float]
    ) -> None:
        # 自定义断言方法，用于检查两个字典的内容近似相等（浮点数比较）

        self.assertEqual(set(d1.keys()), set(d2.keys()))
        # 断言两个字典的键集合相等

        for k, v in d1.items():
            self.assertAlmostEqual(v, d2[k], msg=f"{k}: {v} != {d2[k]}")
            # 对每个键值对进行浮点数近似比较，若不等则输出错误信息

    def make_heuristic(self, classname: str) -> Any:
        # 创建一个虚拟的启发式类实例

        class Heuristic(interface.HeuristicInterface):
            # 内部定义一个类 Heuristic，实现了启发式接口

            def get_prediction_confidence(
                self, tests: list[str]
            ) -> interface.TestPrioritizations:
                # 实现获取预测置信度的方法，返回空的测试优先级对象

                return interface.TestPrioritizations([], {})

        return type(classname, (Heuristic,), {})
        # 返回一个动态创建的类，继承自 Heuristic 类，类名由参数 classname 指定

class TestTestPrioritizations(TestTD):
    # 定义测试 TestTestPrioritizations，继承自 TestTD

    def test_init_none(self) -> None:
        # 测试初始化方法，不设置任何分数

        tests = ["test_a", "test_b"]
        # 定义测试用例列表

        test_prioritizations = interface.TestPrioritizations(tests, {})
        # 创建一个测试优先级对象，不设置任何分数

        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        # 断言测试用例集合与初始设定相同

        self.assertDictEqual(
            test_prioritizations._test_scores,
            {TestRun("test_a"): 0.0, TestRun("test_b"): 0.0},
        )
        # 断言测试分数字典与初始设定相同

    def test_init_set_scores_full_files(self) -> None:
        # 测试初始化方法，设置所有测试用例的分数

        tests = ["test_a", "test_b"]
        # 定义测试用例列表

        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_a"): 0.5, TestRun("test_b"): 0.25}
        )
        # 创建一个测试优先级对象，设置所有测试用例的分数

        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        # 断言测试用例集合与初始设定相同

        self.assertDictEqual(
            test_prioritizations._test_scores,
            {TestRun("test_a"): 0.5, TestRun("test_b"): 0.25},
        )
        # 断言测试分数字典与初始设定相同

    def test_init_set_scores_some_full_files(self) -> None:
        # 测试初始化方法，设置部分测试用例的分数

        tests = ["test_a", "test_b"]
        # 定义测试用例列表

        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_a"): 0.5}
        )
        # 创建一个测试优先级对象，只设置部分测试用例的分数

        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        # 断言测试用例集合与初始设定相同

        self.assertDictEqual(
            test_prioritizations._test_scores,
            {TestRun("test_a"): 0.5, TestRun("test_b"): 0.0},
        )
        # 断言测试分数字典与初始设定相同
    # 定义一个测试方法，用于初始化并设置测试优先级和分数
    def test_init_set_scores_classes(self) -> None:
        # 定义测试用例列表
        tests = ["test_a", "test_b"]
        # 创建 TestPrioritizations 对象，初始化测试用例和测试优先级字典
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_a", included=["TestA"]): 0.5}
        )
        # 断言检查初始化后的原始测试用例集合
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        # 断言检查初始化后的测试分数字典
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.5,
                TestRun("test_a", excluded=["TestA"]): 0.0,
                TestRun("test_b"): 0.0,
            },
        )

    # 定义另一个测试方法，用于初始化并设置测试优先级和分数，使用不同的类命名约定
    def test_init_set_scores_other_class_naming_convention(self) -> None:
        # 定义测试用例列表
        tests = ["test_a", "test_b"]
        # 创建 TestPrioritizations 对象，初始化测试用例和测试优先级字典，使用另一种类命名约定
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_a::TestA"): 0.5}
        )
        # 断言检查初始化后的原始测试用例集合
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        # 断言检查初始化后的测试分数字典
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.5,
                TestRun("test_a", excluded=["TestA"]): 0.0,
                TestRun("test_b"): 0.0,
            },
        )

    # 定义测试方法，用于设置单个测试用例的分数，测试全类名版本
    def test_set_test_score_full_class(self) -> None:
        # 定义测试用例列表
        tests = ["test_a", "test_b"]
        # 创建 TestPrioritizations 对象，初始化测试用例和空的测试分数字典
        test_prioritizations = interface.TestPrioritizations(tests, {})
        # 设置测试用例 "test_a" 的分数为 0.5
        test_prioritizations.set_test_score(TestRun("test_a"), 0.5)
        # 断言检查初始化后的原始测试用例集合
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        # 断言检查设置分数后的测试分数字典
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {TestRun("test_a"): 0.5, TestRun("test_b"): 0.0},
        )
    # 定义一个测试方法，测试混合设置测试分数的功能，无返回值
    def test_set_test_score_mix(self) -> None:
        # 定义测试名称列表
        tests = ["test_a", "test_b"]
        # 创建测试优先级对象，包括测试名称列表和指定测试 "test_b" 的负分数
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_b"): -0.5}
        )
        # 设置测试 "test_a" 的分数为 0.1
        test_prioritizations.set_test_score(TestRun("test_a"), 0.1)
        # 设置测试 "test_a::TestA" 的分数为 0.2
        test_prioritizations.set_test_score(TestRun("test_a::TestA"), 0.2)
        # 设置测试 "test_a::TestB" 的分数为 0.3
        test_prioritizations.set_test_score(TestRun("test_a::TestB"), 0.3)
        # 设置测试 "test_a" 的包含 "TestC" 的分数为 0.4
        test_prioritizations.set_test_score(TestRun("test_a", included=["TestC"]), 0.4)
        # 断言 _original_tests 集合与 tests 集合相等
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        # 断言 _test_scores 字典与预期字典相等，包含了多个 TestRun 对象及其对应的分数
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.2,
                TestRun("test_a", included=["TestB"]): 0.3,
                TestRun("test_a", included=["TestC"]): 0.4,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.1,
                TestRun("test_b"): -0.5,
            },
        )
        # 设置测试 "test_a" 的包含 "TestA" 和 "TestB" 的分数为 0.5
        test_prioritizations.set_test_score(
            TestRun("test_a", included=["TestA", "TestB"]), 0.5
        )
        # 断言 _test_scores 字典与预期字典相等，更新了对应的 TestRun 对象及其分数
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA", "TestB"]): 0.5,
                TestRun("test_a", included=["TestC"]): 0.4,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.1,
                TestRun("test_b"): -0.5,
            },
        )
        # 设置测试 "test_a" 的排除 "TestA" 和 "TestB" 的分数为 0.6
        test_prioritizations.set_test_score(
            TestRun("test_a", excluded=["TestA", "TestB"]), 0.6
        )
        # 断言 _test_scores 字典与预期字典相等，更新了对应的 TestRun 对象及其分数
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA", "TestB"]): 0.5,
                TestRun("test_a", excluded=["TestA", "TestB"]): 0.6,
                TestRun("test_b"): -0.5,
            },
        )
        # 设置测试 "test_a" 的包含 "TestC" 的分数为 0.7
        test_prioritizations.set_test_score(TestRun("test_a", included=["TestC"]), 0.7)
        # 断言 _test_scores 字典与预期字典相等，更新了对应的 TestRun 对象及其分数
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA", "TestB"]): 0.5,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.6,
                TestRun("test_a", included=["TestC"]): 0.7,
                TestRun("test_b"): -0.5,
            },
        )
        # 设置测试 "test_a" 的排除 "TestD" 的分数为 0.8
        test_prioritizations.set_test_score(TestRun("test_a", excluded=["TestD"]), 0.8)
        # 断言 _test_scores 字典与预期字典相等，更新了对应的 TestRun 对象及其分数
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", excluded=["TestD"]): 0.8,
                TestRun("test_a", included=["TestD"]): 0.6,
                TestRun("test_b"): -0.5,
            },
        )
        # 断言 _original_tests 集合与 tests 集合相等
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        # 调用验证方法，确保对象状态正确
        test_prioritizations.validate()
class TestAggregatedHeuristics(TestTD):
    # 定义一个测试类，继承自 TestTD
    def check(
        self,
        tests: list[str],
        test_prioritizations: list[dict[TestRun, float]],
        expected: dict[TestRun, float],
    ) -> None:
        # 定义一个检查方法，接受测试名称列表、测试优先级列表和预期结果字典作为参数
        aggregated_heuristics = interface.AggregatedHeuristics(tests)
        # 创建 AggregatedHeuristics 实例，传入测试名称列表
        for i, test_prioritization in enumerate(test_prioritizations):
            # 遍历测试优先级列表
            heuristic = self.make_heuristic(f"H{i}")
            # 使用当前索引创建一个启发式对象
            aggregated_heuristics.add_heuristic_results(
                heuristic(), interface.TestPrioritizations(tests, test_prioritization)
            )
            # 将启发式结果添加到聚合启发式对象中
        final_prioritzations = aggregated_heuristics.get_aggregated_priorities()
        # 获取聚合后的测试优先级字典
        self.assert_test_scores_almost_equal(
            final_prioritzations._test_scores,
            expected,
        )
        # 使用断言检查聚合后的测试优先级是否与预期结果几乎相等

    def test_get_aggregated_priorities_mix_1(self) -> None:
        # 定义测试方法，测试混合情况1
        tests = ["test_a", "test_b", "test_c"]
        # 设置测试名称列表
        self.check(
            tests,
            [
                {TestRun("test_a"): 0.5},
                {TestRun("test_a::TestA"): 0.25},
                {TestRun("test_c"): 0.8},
            ],
            {
                TestRun("test_a", excluded=["TestA"]): 0.5,
                TestRun("test_a", included=["TestA"]): 0.75,
                TestRun("test_b"): 0.0,
                TestRun("test_c"): 0.8,
            },
        )
        # 调用检查方法，传入测试名称列表、测试优先级列表和预期结果字典进行测试

    def test_get_aggregated_priorities_mix_2(self) -> None:
        # 定义测试方法，测试混合情况2
        tests = ["test_a", "test_b", "test_c"]
        # 设置测试名称列表
        self.check(
            tests,
            [
                {
                    TestRun("test_a", included=["TestC"]): 0.5,
                    TestRun("test_b"): 0.25,
                    TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.8,
                },
                {
                    TestRun("test_a::TestA"): 0.25,
                    TestRun("test_b::TestB"): 0.5,
                    TestRun("test_a::TestB"): 0.75,
                    TestRun("test_a", excluded=["TestA", "TestB"]): 0.8,
                },
                {TestRun("test_c"): 0.8},
            ],
            {
                TestRun("test_a", included=["TestA"]): 0.25,
                TestRun("test_a", included=["TestB"]): 0.75,
                TestRun("test_a", included=["TestC"]): 1.3,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 1.6,
                TestRun("test_b", included=["TestB"]): 0.75,
                TestRun("test_b", excluded=["TestB"]): 0.25,
                TestRun("test_c"): 0.8,
            },
        )
        # 调用检查方法，传入测试名称列表、测试优先级列表和预期结果字典进行测试
    # 定义一个测试方法，测试获取混合优先级的聚合结果
    def test_get_aggregated_priorities_mix_3(self) -> None:
        # 定义测试用例列表，包含单个测试用例名称
        tests = ["test_a"]
        # 调用 self.check 方法，检查多个测试情况的聚合优先级计算结果
        self.check(
            tests,
            [
                {
                    # 创建 TestRun 对象，指定测试用例名称为 "test_a"，包含 "TestA" 测试类，优先级为 0.1
                    TestRun("test_a", included=["TestA"]): 0.1,
                    # 创建 TestRun 对象，指定测试用例名称为 "test_a"，包含 "TestC" 测试类，优先级为 0.1
                    TestRun("test_a", included=["TestC"]): 0.1,
                    # 创建 TestRun 对象，指定测试用例名称为 "test_a"，排除 "TestA", "TestB", "TestC" 测试类，优先级为 0.1
                    TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.1,
                },
                {
                    # 创建 TestRun 对象，指定测试用例名称为 "test_a"，排除 "TestD" 测试类，优先级为 0.1
                    TestRun("test_a", excluded=["TestD"]): 0.1,
                },
                {
                    # 创建 TestRun 对象，指定测试用例名称为 "test_a"，包含 "TestC" 测试类，优先级为 0.1
                    TestRun("test_a", included=["TestC"]): 0.1,
                },
                {
                    # 创建 TestRun 对象，指定测试用例名称为 "test_a"，包含 "TestB", "TestC" 测试类，优先级为 0.1
                    TestRun("test_a", included=["TestB", "TestC"]): 0.1,
                },
                {
                    # 创建 TestRun 对象，指定测试用例名称为 "test_a"，包含 "TestC" 测试类，优先级为 0.1
                    TestRun("test_a", included=["TestC"]): 0.1,
                    # 创建 TestRun 对象，指定测试用例名称为 "test_a"，包含 "TestD" 测试类，优先级为 0.1
                    TestRun("test_a", included=["TestD"]): 0.1,
                },
                {
                    # 创建 TestRun 对象，指定测试用例名称为 "test_a"，优先级为 0.1
                    TestRun("test_a"): 0.1,
                },
            ],
            {
                # 创建 TestRun 对象，指定测试用例名称为 "test_a"，包含 "TestA" 测试类，优先级为 0.3
                TestRun("test_a", included=["TestA"]): 0.3,
                # 创建 TestRun 对象，指定测试用例名称为 "test_a"，包含 "TestB" 测试类，优先级为 0.3
                TestRun("test_a", included=["TestB"]): 0.3,
                # 创建 TestRun 对象，指定测试用例名称为 "test_a"，包含 "TestC" 测试类，优先级为 0.6
                TestRun("test_a", included=["TestC"]): 0.6,
                # 创建 TestRun 对象，指定测试用例名称为 "test_a"，包含 "TestD" 测试类，优先级为 0.3
                TestRun("test_a", included=["TestD"]): 0.3,
                # 创建 TestRun 对象，指定测试用例名称为 "test_a"，排除 "TestA", "TestB", "TestC", "TestD" 测试类，优先级为 0.3
                TestRun("test_a", excluded=["TestA", "TestB", "TestC", "TestD"]): 0.3,
            },
        )
class TestAggregatedHeuristicsTestStats(TestTD):
    # 定义测试类 TestAggregatedHeuristicsTestStats，继承自 TestTD

    def test_get_test_stats_with_whole_tests(self) -> None:
        # 定义测试方法 test_get_test_stats_with_whole_tests，返回类型为 None

        self.maxDiff = None
        # 设置断言中可以显示的最大差异为 None

        tests = ["test1", "test2", "test3", "test4", "test5"]
        # 定义测试名称列表 tests

        heuristic1 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test3"): 0.3,
                TestRun("test4"): 0.1,
            },
        )
        # 创建 TestPrioritizations 对象 heuristic1，用于管理测试和优先级信息

        heuristic2 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test5"): 0.5,
            },
        )
        # 创建 TestPrioritizations 对象 heuristic2，用于管理测试和优先级信息

        aggregator = interface.AggregatedHeuristics(tests)
        # 创建 AggregatedHeuristics 对象 aggregator，初始化测试名称列表

        aggregator.add_heuristic_results(self.make_heuristic("H1")(), heuristic1)
        # 调用 aggregator 的 add_heuristic_results 方法，将 H1 算法的结果 heuristic1 添加到聚合器中

        aggregator.add_heuristic_results(self.make_heuristic("H2")(), heuristic2)
        # 调用 aggregator 的 add_heuristic_results 方法，将 H2 算法的结果 heuristic2 添加到聚合器中

        expected_test3_stats = {
            "test_name": "test3",
            "test_filters": "",
            "heuristics": [
                {
                    "position": 0,
                    "score": 0.3,
                    "heuristic_name": "H1",
                    "trial_mode": False,
                },
                {
                    "position": 3,
                    "score": 0.0,
                    "heuristic_name": "H2",
                    "trial_mode": False,
                },
            ],
            "aggregated": {"position": 1, "score": 0.3},
            "aggregated_trial": {"position": 1, "score": 0.3},
        }
        # 定义预期的 test3 的统计信息 expected_test3_stats

        test3_stats = aggregator.get_test_stats(TestRun("test3"))
        # 调用 aggregator 的 get_test_stats 方法，获取 test3 的统计信息，并赋值给 test3_stats

        self.assertDictEqual(test3_stats, expected_test3_stats)
        # 使用断言方法 assertDictEqual 检查 test3_stats 是否与 expected_test3_stats 相等
    # 定义一个测试方法，用于验证仅包含允许类型的测试统计信息
    def test_get_test_stats_only_contains_allowed_types(self) -> None:
        # 设置最大差异为无（即不做差异比较）
        self.maxDiff = None
        # 定义测试集合
        tests = ["test1", "test2", "test3", "test4", "test5"]
        # 创建第一个启发式对象，包含特定测试与其优先级
        heuristic1 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test3"): 0.3,
                TestRun("test4"): 0.1,
            },
        )
        # 创建第二个启发式对象，包含特定测试与其优先级
        heuristic2 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test5::classA"): 0.5,
            },
        )

        # 创建聚合启发式对象，用于管理测试集合
        aggregator = interface.AggregatedHeuristics(tests)
        # 添加第一个启发式的结果到聚合对象中
        aggregator.add_heuristic_results(self.make_heuristic("H1")(), heuristic1)
        # 添加第二个启发式的结果到聚合对象中
        aggregator.add_heuristic_results(self.make_heuristic("H2")(), heuristic2)

        # 获取测试统计信息，针对特定测试 "test3"
        stats3 = aggregator.get_test_stats(TestRun("test3"))
        # 获取测试统计信息，针对特定测试 "test5::classA"
        stats5 = aggregator.get_test_stats(TestRun("test5::classA"))

        # 定义内部函数，用于验证字典的内容类型是否合法
        def assert_valid_dict(dict_contents: dict[str, Any]) -> None:
            for key, value in dict_contents.items():
                # 断言键必须为字符串类型
                self.assertTrue(isinstance(key, str))
                # 断言值必须为字符串、浮点数、整数、列表或字典类型
                self.assertTrue(
                    isinstance(value, (str, float, int, list, dict)),
                    f"{value} is not a str, float, or dict",
                )
                # 若值为字典类型，则递归验证其内容
                if isinstance(value, dict):
                    assert_valid_dict(value)
                # 若值为列表类型，则递归验证列表中的每个元素
                elif isinstance(value, list):
                    for item in value:
                        assert_valid_dict(item)

        # 对特定测试统计字典进行内容类型的验证
        assert_valid_dict(stats3)
        # 对特定测试统计字典进行内容类型的验证
        assert_valid_dict(stats5)
    def test_get_test_stats_gets_rank_for_test_classes(self) -> None:
        # 设置最大差异为None，以允许所有差异
        self.maxDiff = None
        # 定义测试用例列表
        tests = ["test1", "test2", "test3", "test4", "test5"]
        # 创建第一个启发式对象，包含测试名称和权重
        heuristic1 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test3"): 0.3,
                TestRun("test4"): 0.1,
            },
        )
        # 创建第二个启发式对象，包含测试名称和权重
        heuristic2 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test5::classA"): 0.5,
            },
        )

        # 创建聚合启发式对象，传入测试用例列表
        aggregator = interface.AggregatedHeuristics(tests)
        # 添加第一个启发式结果到聚合对象中
        aggregator.add_heuristic_results(self.make_heuristic("H1")(), heuristic1)
        # 添加第二个启发式结果到聚合对象中
        aggregator.add_heuristic_results(self.make_heuristic("H2")(), heuristic2)

        # 获取包含特定类的测试统计信息
        stats_inclusive = aggregator.get_test_stats(
            TestRun("test5", included=["classA"])
        )
        # 获取不包含特定类的测试统计信息
        stats_exclusive = aggregator.get_test_stats(
            TestRun("test5", excluded=["classA"])
        )

        # 期望的包含特定类的测试统计信息
        expected_inclusive = {
            "test_name": "test5",
            "test_filters": "classA",
            "heuristics": [
                {
                    "position": 4,
                    "score": 0.0,
                    "heuristic_name": "H1",
                    "trial_mode": False,
                },
                {
                    "position": 0,
                    "score": 0.5,
                    "heuristic_name": "H2",
                    "trial_mode": False,
                },
            ],
            "aggregated": {"position": 0, "score": 0.5},
            "aggregated_trial": {"position": 0, "score": 0.5},
        }

        # 期望的不包含特定类的测试统计信息
        expected_exclusive = {
            "test_name": "test5",
            "test_filters": "not (classA)",
            "heuristics": [
                {
                    "position": 4,
                    "score": 0.0,
                    "heuristic_name": "H1",
                    "trial_mode": False,
                },
                {
                    "position": 5,
                    "score": 0.0,
                    "heuristic_name": "H2",
                    "trial_mode": False,
                },
            ],
            "aggregated": {"position": 5, "score": 0.0},
            "aggregated_trial": {"position": 5, "score": 0.0},
        }

        # 断言实际包含统计信息与期望统计信息相等
        self.assertDictEqual(stats_inclusive, expected_inclusive)
        # 断言实际不包含统计信息与期望统计信息相等
        self.assertDictEqual(stats_exclusive, expected_exclusive)
    # 定义一个测试方法，用于测试具有类粒度启发的测试统计功能
    def test_get_test_stats_works_with_class_granularity_heuristics(self) -> None:
        # 定义一个测试名称列表
        tests = ["test1", "test2", "test3", "test4", "test5"]
        
        # 创建一个测试优先级实例，使用一个测试名称和对应的启发值字典
        heuristic1 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test2"): 0.3,  # 指定特定测试的启发值
            },
        )
        
        # 创建另一个测试优先级实例，使用一个测试名称（包括类名）和对应的启发值字典
        heuristic2 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test2::TestFooClass"): 0.5,  # 指定特定测试类的启发值
            },
        )

        # 创建一个聚合启发实例，传入测试名称列表
        aggregator = interface.AggregatedHeuristics(tests)
        
        # 添加第一个启发结果到聚合器中，使用“H1”标识的启发方法
        aggregator.add_heuristic_results(self.make_heuristic("H1")(), heuristic1)
        
        # 添加第二个启发结果到聚合器中，使用“H2”标识的启发方法
        aggregator.add_heuristic_results(self.make_heuristic("H2")(), heuristic2)

        # 以下两行代码不应抛出错误
        # 获取指定测试类的统计信息
        aggregator.get_test_stats(TestRun("test2::TestFooClass"))
        # 获取指定测试的统计信息
        aggregator.get_test_stats(TestRun("test2"))
class TestJsonParsing(TestTD):
    def test_json_parsing_matches_TestPrioritizations(self) -> None:
        # 定义测试集合
        tests = ["test1", "test2", "test3", "test4", "test5"]
        # 创建 TestPrioritizations 对象，并指定测试用例及其优先级
        tp = interface.TestPrioritizations(
            tests,
            {
                TestRun("test3", included=["ClassA"]): 0.8,
                TestRun("test3", excluded=["ClassA"]): 0.2,
                TestRun("test4"): 0.7,
                TestRun("test5"): 0.6,
            },
        )
        # 将 TestPrioritizations 对象转换为 JSON 格式
        tp_json = tp.to_json()
        # 从 JSON 格式还原 TestPrioritizations 对象
        tp_json_to_tp = interface.TestPrioritizations.from_json(tp_json)

        # 断言原始测试集合相等
        self.assertSetEqual(tp._original_tests, tp_json_to_tp._original_tests)
        # 断言测试分数字典相等
        self.assertDictEqual(tp._test_scores, tp_json_to_tp._test_scores)

    def test_json_parsing_matches_TestRun(self) -> None:
        # 创建 TestRun 对象，指定包含的类和测试名称
        testrun = TestRun("test1", included=["classA", "classB"])
        # 将 TestRun 对象转换为 JSON 格式
        testrun_json = testrun.to_json()
        # 从 JSON 格式还原 TestRun 对象
        testrun_json_to_test = TestRun.from_json(testrun_json)

        # 断言两个 TestRun 对象相等
        self.assertTrue(testrun == testrun_json_to_test)


if __name__ == "__main__":
    # 执行单元测试
    unittest.main()
```