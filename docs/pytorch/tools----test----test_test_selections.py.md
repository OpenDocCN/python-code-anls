# `.\pytorch\tools\test\test_test_selections.py`

```py
from __future__ import annotations
# 导入用于支持类型注解的特性（Python 3.10+）

import functools
import random
import sys
import unittest
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# 确定项目根目录

try:
    # 将 tools/ 添加到系统路径以优化测试运行
    sys.path.append(str(REPO_ROOT))
    # 导入测试相关的模块和函数
    from tools.testing.test_run import ShardedTest, TestRun
    from tools.testing.test_selections import calculate_shards, THRESHOLD
except ModuleNotFoundError:
    # 若找不到模块则输出错误信息并退出
    print("Can't import required modules, exiting")
    sys.exit(1)


def gen_class_times(test_times: dict[str, float]) -> dict[str, dict[str, float]]:
    # 根据测试时间生成每个测试的时间字典，每个测试属于 'class1'
    return {k: {"class1": v} for k, v in test_times.items()}


class TestCalculateShards(unittest.TestCase):
    # 测试用例类，继承自 unittest.TestCase

    tests: list[TestRun] = [
        TestRun("super_long_test"),
        TestRun("long_test1"),
        TestRun("long_test2"),
        TestRun("normal_test1"),
        TestRun("normal_test2"),
        TestRun("normal_test3"),
        TestRun("short_test1"),
        TestRun("short_test2"),
        TestRun("short_test3"),
        TestRun("short_test4"),
        TestRun("short_test5"),
    ]
    # 定义测试列表，包含不同类型的测试

    test_times: dict[str, float] = {
        "super_long_test": 55,
        "long_test1": 22,
        "long_test2": 18,
        "normal_test1": 9,
        "normal_test2": 7,
        "normal_test3": 5,
        "short_test1": 1,
        "short_test2": 0.6,
        "short_test3": 0.4,
        "short_test4": 0.3,
        "short_test5": 0.01,
    }
    # 定义测试所需的时间字典，键为测试名称，值为测试时间（秒）

    test_class_times: dict[str, dict[str, float]] = {
        "super_long_test": {"class1": 55},
        "long_test1": {"class1": 1, "class2": 21},
        "long_test2": {"class1": 10, "class2": 8},
        "normal_test1": {"class1": 9},
        "normal_test2": {"class1": 7},
        "normal_test3": {"class1": 5},
        "short_test1": {"class1": 1},
        "short_test2": {"class1": 0.6},
        "short_test3": {"class1": 0.4},
        "short_test4": {"class1": 0.3},
        "short_test5": {"class1": 0.01},
    }
    # 定义不同测试类别的测试时间字典，每个测试名映射到包含测试类别及其时间的字典

    def assert_shards_equal(
        self,
        expected_shards: list[tuple[float, list[ShardedTest]]],
        actual_shards: list[tuple[float, list[ShardedTest]]],
    ) -> None:
        # 断言两个分片列表是否相等，每个分片包含预期的权重和 ShardedTest 对象列表
        for expected, actual in zip(expected_shards, actual_shards):
            self.assertAlmostEqual(expected[0], actual[0])
            self.assertListEqual(expected[1], actual[1])
    def test_no_times(self) -> None:
        # 检查在未提供时间时是否使用了循环分片策略
        expected_shards = [
            (
                0.0,
                [
                    ShardedTest(
                        test="super_long_test", shard=1, num_shards=1, time=None
                    ),
                    ShardedTest(test="long_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(test="normal_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test1", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test3", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test5", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                0.0,
                [
                    ShardedTest(test="long_test1", shard=1, num_shards=1, time=None),
                    ShardedTest(test="normal_test1", shard=1, num_shards=1, time=None),
                    ShardedTest(test="normal_test3", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test4", shard=1, num_shards=1, time=None),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards,
            # 测试函数，验证计算出的分片是否符合预期
            calculate_shards(2, self.tests, {}, {}, sort_by_time=False),
        )

    def test_some_times_with_not_sort_by_time(self) -> None:
        expected_shards = [
            (
                400.0,
                [
                    ShardedTest(test="test_1", shard=1, num_shards=1, time=None),
                    ShardedTest(test="test_2", shard=1, num_shards=1, time=400),
                    ShardedTest(test="test_5", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                300.0,
                [
                    ShardedTest(test="test_3", shard=1, num_shards=1, time=300),
                    ShardedTest(test="test_4", shard=1, num_shards=1, time=None),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards,
            # 测试函数，验证计算出的分片是否符合预期
            calculate_shards(
                2,
                [
                    TestRun("test_1"),
                    TestRun("test_2"),
                    TestRun("test_3"),
                    TestRun("test_4"),
                    TestRun("test_5"),
                ],
                {"test_2": 400, "test_3": 300},
                {},
                sort_by_time=False,
            ),
        )
    # 定义测试函数，验证串行和并行交错计算的正确性
    def test_serial_parallel_interleaving(self) -> None:
        # 预期的分片结果列表
        expected_shards = [
            (
                300.0,
                [
                    # 创建 ShardedTest 对象的列表，每个对象表示一个测试片段的信息
                    ShardedTest(test="test_1", shard=1, num_shards=1, time=None),
                    ShardedTest(test="test_3", shard=1, num_shards=1, time=300),
                    ShardedTest(test="test_4", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                400.0,
                [
                    # 继续创建 ShardedTest 对象的列表，每个对象表示一个测试片段的信息
                    ShardedTest(test="test_2", shard=1, num_shards=1, time=400),
                    ShardedTest(test="test_5", shard=1, num_shards=1, time=None),
                ],
            ),
        ]
        # 调用断言方法，验证分片结果是否与预期一致
        self.assert_shards_equal(
            expected_shards,
            # 调用 calculate_shards 函数，计算测试的分片结果
            calculate_shards(
                2,  # 分成两个片段
                [
                    TestRun("test_1"),
                    TestRun("test_2"),
                    TestRun("test_3"),
                    TestRun("test_4"),
                    TestRun("test_5"),
                ],  # 所有测试的列表
                {"test_2": 400, "test_3": 300},  # 指定测试的时间字典
                {},  # 空的测试类时间字典
                must_serial=lambda x: x in ["test_1", "test_3"],  # 必须串行执行的测试条件
                sort_by_time=False,  # 不按时间排序
            ),
        )

    # 定义测试函数，验证使用完整测试时间计算两个分片的正确性
    def test_calculate_2_shards_with_complete_test_times(self) -> None:
        # 预期的分片结果列表
        expected_shards = [
            (
                60.0,
                [
                    # 创建 ShardedTest 对象的列表，每个对象表示一个测试片段的信息
                    ShardedTest(test="super_long_test", shard=1, num_shards=1, time=55),
                    ShardedTest(test="normal_test3", shard=1, num_shards=1, time=5),
                ],
            ),
            (
                58.31,
                [
                    # 继续创建 ShardedTest 对象的列表，每个对象表示一个测试片段的信息
                    ShardedTest(test="long_test1", shard=1, num_shards=1, time=22),
                    ShardedTest(test="long_test2", shard=1, num_shards=1, time=18),
                    ShardedTest(test="normal_test1", shard=1, num_shards=1, time=9),
                    ShardedTest(test="normal_test2", shard=1, num_shards=1, time=7),
                    ShardedTest(test="short_test1", shard=1, num_shards=1, time=1),
                    ShardedTest(test="short_test2", shard=1, num_shards=1, time=0.6),
                    ShardedTest(test="short_test3", shard=1, num_shards=1, time=0.4),
                    ShardedTest(test="short_test4", shard=1, num_shards=1, time=0.3),
                    ShardedTest(test="short_test5", shard=1, num_shards=1, time=0.01),
                ],
            ),
        ]
        # 调用断言方法，验证分片结果是否与预期一致
        self.assert_shards_equal(
            expected_shards,
            # 调用 calculate_shards 函数，计算测试的分片结果
            calculate_shards(2, self.tests, self.test_times, self.test_class_times),
        )
    def test_calculate_1_shard_with_complete_test_times(self) -> None:
        # 复制测试列表，确保不改变原始数据
        tests = self.tests.copy()
        # 创建两个测试运行对象，分别指定了排除和包含的类
        class_test1 = TestRun("long_test1", excluded=["class2"])
        class_test2 = TestRun("long_test1", included=["class2"])
        # 将新创建的测试对象添加到测试列表中
        tests.append(class_test1)
        tests.append(class_test2)

        # 预期的分片结果，包含一个元组，元组第一个元素为总测试时间，第二个元素为分片后的测试列表
        expected_shards = [
            (
                140.31,
                [
                    # 分片后的测试对象列表，每个对象包含测试名、分片号、总分片数、执行时间
                    ShardedTest(test="super_long_test", shard=1, num_shards=1, time=55),
                    ShardedTest(test="long_test1", shard=1, num_shards=1, time=22),
                    ShardedTest(class_test2, shard=1, num_shards=1, time=21),
                    ShardedTest(test="long_test2", shard=1, num_shards=1, time=18),
                    ShardedTest(test="normal_test1", shard=1, num_shards=1, time=9),
                    ShardedTest(test="normal_test2", shard=1, num_shards=1, time=7),
                    ShardedTest(test="normal_test3", shard=1, num_shards=1, time=5),
                    ShardedTest(test="short_test1", shard=1, num_shards=1, time=1),
                    ShardedTest(class_test1, shard=1, num_shards=1, time=1),
                    ShardedTest(test="short_test2", shard=1, num_shards=1, time=0.6),
                    ShardedTest(test="short_test3", shard=1, num_shards=1, time=0.4),
                    ShardedTest(test="short_test4", shard=1, num_shards=1, time=0.3),
                    ShardedTest(test="short_test5", shard=1, num_shards=1, time=0.01),
                ],
            )
        ]
        # 断言分片结果与预期结果相等
        self.assert_shards_equal(
            expected_shards,
            # 调用分片计算函数，计算分片数为1时的分片情况
            calculate_shards(1, tests, self.test_times, self.test_class_times),
        )
    # 定义一个测试方法，用于测试 calculate_shards 函数在给定完整测试时间的情况下的行为
    def test_calculate_5_shards_with_complete_test_times(self) -> None:
        # 预期的分片结果，包含了不同测试用例的预期时间和分片信息
        expected_shards = [
            (
                55.0,
                [ShardedTest(test="super_long_test", shard=1, num_shards=1, time=55)],
            ),
            (22.0, [ShardedTest(test="long_test1", shard=1, num_shards=1, time=22)]),
            (18.0, [ShardedTest(test="long_test2", shard=1, num_shards=1, time=18)]),
            (
                11.31,
                [
                    ShardedTest(test="normal_test1", shard=1, num_shards=1, time=9),
                    ShardedTest(test="short_test1", shard=1, num_shards=1, time=1),
                    ShardedTest(test="short_test2", shard=1, num_shards=1, time=0.6),
                    ShardedTest(test="short_test3", shard=1, num_shards=1, time=0.4),
                    ShardedTest(test="short_test4", shard=1, num_shards=1, time=0.3),
                    ShardedTest(test="short_test5", shard=1, num_shards=1, time=0.01),
                ],
            ),
            (
                12.0,
                [
                    ShardedTest(test="normal_test2", shard=1, num_shards=1, time=7),
                    ShardedTest(test="normal_test3", shard=1, num_shards=1, time=5),
                ],
            ),
        ]
        # 使用断言方法验证 calculate_shards 函数的输出是否与预期的 expected_shards 相等
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(5, self.tests, self.test_times, self.test_class_times),
        )
    # 定义一个测试方法，用于测试在存在不完整测试时间的情况下计算两个分片的功能
    def test_calculate_2_shards_with_incomplete_test_times(self) -> None:
        # 创建一个字典，其中仅包含测试时间键名包含字符串"test1"的测试时间项
        incomplete_test_times = {
            k: v for k, v in self.test_times.items() if "test1" in k
        }
        # 期望的分片结果列表，每个元组包含一个时间值和相应的 ShardedTest 对象列表
        expected_shards = [
            (
                22.0,
                [
                    # 第一个分片中的测试列表
                    ShardedTest(test="long_test1", shard=1, num_shards=1, time=22),
                    ShardedTest(
                        test="super_long_test", shard=1, num_shards=1, time=None
                    ),
                    ShardedTest(test="normal_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test4", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                10.0,
                [
                    # 第二个分片中的测试列表
                    ShardedTest(test="normal_test1", shard=1, num_shards=1, time=9),
                    ShardedTest(test="short_test1", shard=1, num_shards=1, time=1),
                    ShardedTest(test="long_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(test="normal_test3", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test3", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test5", shard=1, num_shards=1, time=None),
                ],
            ),
        ]
        # 断言函数，验证计算的分片结果与期望的分片结果是否相等
        self.assert_shards_equal(
            expected_shards,
            # 调用计算分片的函数，传入分片数、所有测试、不完整测试时间和生成的类时间
            calculate_shards(
                2,
                self.tests,
                incomplete_test_times,
                gen_class_times(incomplete_test_times),
            ),
        )
    # 定义一个测试方法，计算包含不完整测试时间的情况下，5个分片的测试情况
    def test_calculate_5_shards_with_incomplete_test_times(self) -> None:
        # 创建一个包含部分测试时间的字典，只包含名称中含有"test1"的测试
        incomplete_test_times = {
            k: v for k, v in self.test_times.items() if "test1" in k
        }
        # 期望的分片测试结果列表，每个元素是一个元组，包含一个时间值和一个分片测试列表
        expected_shards = [
            (
                22.0,
                [
                    # 分片测试对象，表示长测试1在第1个分片中，共1个分片，耗时22秒
                    ShardedTest(test="long_test1", shard=1, num_shards=1, time=22),
                    # 分片测试对象，表示超长测试在第1个分片中，共1个分片，时间为None（未知）
                    ShardedTest(
                        test="super_long_test", shard=1, num_shards=1, time=None
                    ),
                    # 分片测试对象，表示短测试3在第1个分片中，共1个分片，时间为None（未知）
                    ShardedTest(test="short_test3", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                9.0,
                [
                    # 分片测试对象，表示普通测试1在第1个分片中，共1个分片，耗时9秒
                    ShardedTest(test="normal_test1", shard=1, num_shards=1, time=9),
                    # 分片测试对象，表示长测试2在第1个分片中，共1个分片，时间为None（未知）
                    ShardedTest(test="long_test2", shard=1, num_shards=1, time=None),
                    # 分片测试对象，表示短测试4在第1个分片中，共1个分片，时间为None（未知）
                    ShardedTest(test="short_test4", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                1.0,
                [
                    # 分片测试对象，表示短测试1在第1个分片中，共1个分片，耗时1秒
                    ShardedTest(test="short_test1", shard=1, num_shards=1, time=1),
                    # 分片测试对象，表示普通测试2在第1个分片中，共1个分片，时间为None（未知）
                    ShardedTest(test="normal_test2", shard=1, num_shards=1, time=None),
                    # 分片测试对象，表示短测试5在第1个分片中，共1个分片，时间为None（未知）
                    ShardedTest(test="short_test5", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                0.0,
                [
                    # 分片测试对象，表示普通测试3在第1个分片中，共1个分片，时间为None（未知）
                    ShardedTest(test="normal_test3", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                0.0,
                [
                    # 分片测试对象，表示短测试2在第1个分片中，共1个分片，时间为None（未知）
                    ShardedTest(test="short_test2", shard=1, num_shards=1, time=None),
                ],
            ),
        ]
        # 调用断言方法，验证计算分片的结果是否与期望的一致
        self.assert_shards_equal(
            expected_shards,
            # 调用计算分片的函数，传入5（分片数量）、所有测试、部分测试时间字典、生成的类时间列表
            calculate_shards(
                5,
                self.tests,
                incomplete_test_times,
                gen_class_times(incomplete_test_times),
            ),
        )
    # 定义一个单元测试方法，用于测试分片功能
    def test_split_shards(self) -> None:
        # 初始化测试时间字典，包含两个测试用例，每个测试用例时间均为 THRESHOLD
        test_times: dict[str, float] = {"test1": THRESHOLD, "test2": THRESHOLD}
        # 预期的分片结果列表，每个元素是一个元组，包含总时间和分片测试的列表
        expected_shards = [
            (600.0, [ShardedTest(test="test1", shard=1, num_shards=1, time=THRESHOLD)]),
            (600.0, [ShardedTest(test="test2", shard=1, num_shards=1, time=THRESHOLD)]),
        ]
        # 断言预期分片结果与计算得到的分片结果相等
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(
                2,
                [TestRun(t) for t in test_times.keys()],
                test_times,
                gen_class_times(test_times),
            ),
        )

        # 更新测试时间字典，改变测试用例的时间分布
        test_times = {"test1": THRESHOLD * 4, "test2": THRESHOLD * 2.5}
        # 更新预期的分片结果列表，根据新的时间分布计算得到不同的分片情况
        expected_shards = [
            (
                2200.0,
                [
                    ShardedTest(test="test1", shard=1, num_shards=4, time=600.0),
                    ShardedTest(test="test1", shard=3, num_shards=4, time=600.0),
                    ShardedTest(test="test2", shard=1, num_shards=3, time=500.0),
                    ShardedTest(test="test2", shard=3, num_shards=3, time=500.0),
                ],
            ),
            (
                1700.0,
                [
                    ShardedTest(test="test1", shard=2, num_shards=4, time=600.0),
                    ShardedTest(test="test1", shard=4, num_shards=4, time=600.0),
                    ShardedTest(test="test2", shard=2, num_shards=3, time=500.0),
                ],
            ),
        ]
        # 再次断言预期分片结果与计算得到的分片结果相等
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(
                2,
                [TestRun(t) for t in test_times.keys()],
                test_times,
                gen_class_times(test_times),
            ),
        )

        # 更新测试时间字典，测试用例的时间进一步变化
        test_times = {"test1": THRESHOLD / 2, "test2": THRESHOLD}
        # 更新预期的分片结果列表，根据新的时间分布计算得到不同的分片情况
        expected_shards = [
            (600.0, [ShardedTest(test="test2", shard=1, num_shards=1, time=THRESHOLD)]),
            (
                300.0,
                [ShardedTest(test="test1", shard=1, num_shards=1, time=THRESHOLD / 2)],
            ),
        ]
        # 再次断言预期分片结果与计算得到的分片结果相等
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(
                2,
                [TestRun(t) for t in test_times.keys()],
                test_times,
                gen_class_times(test_times),
            ),
        )

    # 定义一个测试零测试用例的方法
    def test_zero_tests(self) -> None:
        # 断言空测试用例列表的分片结果为总时间为0，分片列表为空列表
        self.assertListEqual([(0.0, []), (0.0, [])], calculate_shards(2, [], {}, None))
    # 定义一个测试函数，用于测试计算两个分片与最佳分片的比较
    def test_calculate_2_shards_against_optimal_shards(self) -> None:
        # 设置随机种子为120，保证每次运行结果一致
        random.seed(120)
        # 执行100次测试
        for _ in range(100):
            # 生成一个随机的测试时间字典，包括所有测试文件的随机时间
            random_times = {k.test_file: random.random() * 10 for k in self.tests}
            # 选取除了"super_long_test"和"long_test1"之外的所有测试时间
            rest_of_tests = [
                i
                for k, i in random_times.items()
                if k != "super_long_test" and k != "long_test1"
            ]
            # 计算剩余测试时间的总和
            sum_of_rest = sum(rest_of_tests)
            # 设置"super_long_test"的时间为剩余时间总和的一半或者剩余时间中的最大值
            random_times["super_long_test"] = max(sum_of_rest / 2, *rest_of_tests)
            # 设置"long_test1"的时间为剩余时间总和减去"super_long_test"的时间
            random_times["long_test1"] = sum_of_rest - random_times["super_long_test"]
            # 计算当前设置下的分片，调用calculate_shards函数进行计算
            calculated_shards = calculate_shards(
                2, self.tests, random_times, gen_class_times(random_times)
            )
            # 获取两个分片中时间最大的那个分片时间
            max_shard_time = max(calculated_shards[0][0], calculated_shards[1][0])
            # 如果剩余时间总和不为0，则验证计算的分片时间比率不应超过7/6
            if sum_of_rest != 0:
                self.assertGreaterEqual(7.0 / 6.0, max_shard_time / sum_of_rest)
                # 获取所有测试文件名并排序
                sorted_tests = sorted([t.test_file for t in self.tests])
                # 获取排序后的分片测试文件名
                sorted_shard_tests = sorted(
                    calculated_shards[0][1] + calculated_shards[1][1]
                )
                # 验证所有测试文件都应该被某个分片包含
                self.assertEqual(sorted_tests, [x.name for x in sorted_shard_tests])
# 如果当前脚本被作为主程序运行（而不是被导入到其他脚本中），则执行单元测试
if __name__ == "__main__":
    # 调用 unittest 模块的主函数，启动测试运行
    unittest.main()
```