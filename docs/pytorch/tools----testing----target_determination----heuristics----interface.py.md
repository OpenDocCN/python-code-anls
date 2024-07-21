# `.\pytorch\tools\testing\target_determination\heuristics\interface.py`

```py
# 从 `__future__` 模块中导入 `annotations` 特性，用于支持类型注解的一种机制
from __future__ import annotations

# 从 `abc` 模块中导入 `abstractmethod` 抽象方法装饰器
from abc import abstractmethod

# 从 `copy` 模块中导入 `copy` 函数
from copy import copy

# 从 `typing` 模块中导入 `Any`, `Iterable`, `Iterator` 类型
from typing import Any, Iterable, Iterator

# 从 `tools.testing.test_run` 模块中导入 `TestRun` 类
from tools.testing.test_run import TestRun


class TestPrioritizations:
    """
    Describes the results of whether heuristics consider a test relevant or not.

    All the different ranks of tests are disjoint, meaning a test can only be in one category, and they are only
    declared at initialization time.

    A list can be empty if a heuristic doesn't consider any tests to be in that category.

    Important: Lists of tests must always be returned in a deterministic order,
               otherwise it breaks the test sharding logic
    """

    # 原始测试集合，使用 frozenset 冻结集合来确保不可变性
    _original_tests: frozenset[str]

    # 测试分数的字典，映射为 TestRun 对象到分数的映射关系
    _test_scores: dict[TestRun, float]

    def __init__(
        self,
        tests_being_ranked: Iterable[str],  # 被排序的测试集合，作为输入参数
        scores: dict[TestRun, float],       # 测试运行到分数的字典
    ) -> None:
        # 使用输入的测试集合创建一个不可变的集合 `_original_tests`
        self._original_tests = frozenset(tests_being_ranked)
        
        # 初始化 `_test_scores` 字典，将每个测试映射到初始分数 0.0
        self._test_scores = {TestRun(test): 0.0 for test in self._original_tests}

        # 遍历传入的分数字典，为每个测试设置其对应的分数
        for test, score in scores.items():
            self.set_test_score(test, score)

        # 调用验证方法确保测试的一致性和逻辑正确性
        self.validate()

    def validate(self) -> None:
        # 合并所有包含 include/exclude 对的 TestRun
        all_tests = self._test_scores.keys()
        files = {}
        for test in all_tests:
            # 如果测试文件不在 files 字典中，将其加入
            if test.test_file not in files:
                files[test.test_file] = copy(test)
            else:
                # 否则，检查两个测试是否重叠，如果重叠则抛出断言错误
                assert (
                    files[test.test_file] & test
                ).is_empty(), (
                    f"Test run `{test}` overlaps with `{files[test.test_file]}`"
                )
                files[test.test_file] |= test

        # 检查每个文件测试集是否完整，即包含了所有的包含和排除的测试
        for test in files.values():
            assert (
                test.is_full_file()
            ), f"All includes should have been excluded elsewhere, and vice versa. Test run `{test}` violates that"

        # 确保 TestPrioritizations 中的测试集合与传入的测试集合相同
        assert self._original_tests == set(
            files.keys()
        ), "The set of tests in the TestPrioritizations must be identical to the set of tests passed in"

    def _traverse_scores(self) -> Iterator[tuple[float, TestRun]]:
        # 按分数排序，然后按测试名称的字母顺序排序，生成器函数
        for test, score in sorted(
            self._test_scores.items(), key=lambda x: (-x[1], str(x[0]))
        ):
            yield score, test
    # 设置测试运行的分数，更新现有的测试分数信息
    def set_test_score(self, test_run: TestRun, new_score: float) -> None:
        # 如果测试运行的测试文件不在原始测试集中，则返回，不需要处理这个测试
        if test_run.test_file not in self._original_tests:
            return  # We don't need this test

        # 找出所有与当前测试运行有交集且不同于当前测试运行的测试运行列表
        relevant_test_runs: list[TestRun] = [
            tr for tr in self._test_scores.keys() if tr & test_run and tr != test_run
        ]

        # 将当前测试运行的分数设置为新分数
        self._test_scores[test_run] = new_score

        # 将未被当前测试运行覆盖的测试运行的分数设置为原始分数
        for relevant_test_run in relevant_test_runs:
            old_score = self._test_scores[relevant_test_run]
            del self._test_scores[relevant_test_run]

            # 计算不受当前测试运行影响的测试运行
            not_to_be_updated = relevant_test_run - test_run
            if not not_to_be_updated.is_empty():
                self._test_scores[not_to_be_updated] = old_score

        # 调用验证函数，确保数据的一致性和正确性
        self.validate()

    # 增加测试运行的分数
    def add_test_score(self, test_run: TestRun, score_to_add: float) -> None:
        # 如果测试运行的测试文件不在原始测试集中，则返回
        if test_run.test_file not in self._original_tests:
            return

        # 找出所有与当前测试运行有交集的测试运行列表
        relevant_test_runs: list[TestRun] = [
            tr for tr in self._test_scores.keys() if tr & test_run
        ]

        # 遍历所有与当前测试运行有交集的测试运行
        for relevant_test_run in relevant_test_runs:
            old_score = self._test_scores[relevant_test_run]
            del self._test_scores[relevant_test_run]

            # 计算当前测试运行与相关测试运行的交集
            intersection = relevant_test_run & test_run
            if not intersection.is_empty():
                # 更新交集部分的分数
                self._test_scores[intersection] = old_score + score_to_add

            # 计算不受当前测试运行影响的测试运行
            not_to_be_updated = relevant_test_run - test_run
            if not not_to_be_updated.is_empty():
                self._test_scores[not_to_be_updated] = old_score

        # 调用验证函数，确保数据的一致性和正确性
        self.validate()

    # 返回所有测试运行的列表
    def get_all_tests(self) -> list[TestRun]:
        """Returns all tests in the TestPrioritizations"""
        return [x[1] for x in self._traverse_scores()]

    # 返回分数排名前n%的测试运行列表和剩余的测试运行列表
    def get_top_per_tests(self, n: int) -> tuple[list[TestRun], list[TestRun]]:
        """Divides list of tests into two based on the top n% of scores.  The
        first list is the top, and the second is the rest."""
        # 获取所有测试运行的列表
        tests = [x[1] for x in self._traverse_scores()]
        # 计算排名前n%的测试运行的索引
        index = n * len(tests) // 100 + 1
        return tests[:index], tests[index:]

    # 返回测试运行信息的字符串表示，可以选择是否详细显示
    def get_info_str(self, verbose: bool = True) -> str:
        info = ""

        # 遍历所有测试运行及其分数
        for score, test in self._traverse_scores():
            # 如果不详细显示且分数为0，则跳过
            if not verbose and score == 0:
                continue
            # 构建每个测试运行的信息字符串
            info += f"  {test} ({score})\n"

        return info.rstrip()

    # 打印测试运行的信息
    def print_info(self) -> None:
        print(self.get_info_str())
    def get_priority_info_for_test(self, test_run: TestRun) -> dict[str, Any]:
        """Given a failing test, returns information about it's prioritization that we want to emit in our metrics."""
        # 遍历得分列表，获取与给定测试运行有重叠的测试，并返回其位置和得分信息
        for idx, (score, test) in enumerate(self._traverse_scores()):
            # 不同的启发式方法可能导致同一个测试文件被拆分到不同的测试运行中，
            # 因此查找重叠的测试以找到匹配项
            if test & test_run:
                return {"position": idx, "score": score}
        # 如果找不到匹配的测试运行，则抛出断言错误
        raise AssertionError(f"Test run {test_run} not found")

    def get_test_stats(self, test: TestRun) -> dict[str, Any]:
        # 返回测试的统计信息，包括测试名称、pytest过滤器、优先级信息、最高得分、最低得分以及所有得分的字典
        return {
            "test_name": test.test_file,
            "test_filters": test.get_pytest_filter(),
            **self.get_priority_info_for_test(test),
            "max_score": max(score for score, _ in self._traverse_scores()),
            "min_score": min(score for score, _ in self._traverse_scores()),
            "all_scores": {
                str(test): score for test, score in self._test_scores.items()
            },
        }

    def to_json(self) -> dict[str, Any]:
        """
        Returns a JSON dict that describes this TestPrioritizations object.
        """
        # 将 TestPrioritizations 对象转换为 JSON 字典表示
        json_dict = {
            "_test_scores": [
                (test.to_json(), score)
                for test, score in self._test_scores.items()
                if score != 0
            ],
            "_original_tests": list(self._original_tests),
        }
        return json_dict

    @staticmethod
    def from_json(json_dict: dict[str, Any]) -> TestPrioritizations:
        """
        Returns a TestPrioritizations object from a JSON dict.
        """
        # 从 JSON 字典中创建 TestPrioritizations 对象
        test_prioritizations = TestPrioritizations(
            tests_being_ranked=json_dict["_original_tests"],
            scores={
                TestRun.from_json(testrun_json): score
                for testrun_json, score in json_dict["_test_scores"]
            },
        )
        return test_prioritizations

    def amend_tests(self, tests: list[str]) -> None:
        """
        Removes tests that are not in the given list from the
        TestPrioritizations.  Adds tests that are in the list but not in the
        TestPrioritizations.
        """
        # 更新测试列表，删除不在给定列表中的测试，添加在列表中但不在 TestPrioritizations 中的测试
        valid_scores = {
            test: score
            for test, score in self._test_scores.items()
            if test.test_file in tests
        }
        self._test_scores = valid_scores

        for test in tests:
            if test not in self._original_tests:
                self._test_scores[TestRun(test)] = 0
        self._original_tests = frozenset(tests)

        # 执行验证函数确保数据的有效性
        self.validate()
class AggregatedHeuristics:
    """
    Aggregates the results across all heuristics.

    It saves the individual results from each heuristic and exposes an aggregated view.
    """

    _heuristic_results: dict[
        HeuristicInterface, TestPrioritizations
    ]  # Key is the Heuristic's name. Dicts will preserve the order of insertion, which is important for sharding

    _all_tests: frozenset[str]

    def __init__(self, all_tests: list[str]) -> None:
        """
        Initializes an AggregatedHeuristics instance.

        Args:
        - all_tests: A list of all test names to be aggregated.
        """
        self._all_tests = frozenset(all_tests)
        self._heuristic_results = {}
        self.validate()

    def validate(self) -> None:
        """
        Validates that the tests in each heuristic's results match _all_tests.
        """
        for heuristic, heuristic_results in self._heuristic_results.items():
            heuristic_results.validate()
            assert (
                heuristic_results._original_tests == self._all_tests
            ), f"Tests in {heuristic.name} are not the same as the tests in the AggregatedHeuristics"

    def add_heuristic_results(
        self, heuristic: HeuristicInterface, heuristic_results: TestPrioritizations
    ) -> None:
        """
        Adds results from a heuristic to _heuristic_results.

        Args:
        - heuristic: The heuristic whose results are being added.
        - heuristic_results: The TestPrioritizations object containing heuristic's results.

        Raises:
        - ValueError: If results for the heuristic already exist.
        """
        if heuristic in self._heuristic_results:
            raise ValueError(f"We already have heuristics for {heuristic.name}")

        self._heuristic_results[heuristic] = heuristic_results
        self.validate()

    def get_aggregated_priorities(
        self, include_trial: bool = False
    ) -> TestPrioritizations:
        """
        Returns the aggregated priorities across all heuristics.

        Args:
        - include_trial: Whether to include results from heuristics in trial mode.

        Returns:
        - TestPrioritizations: Object containing aggregated test priorities.
        """
        valid_heuristics = {
            heuristic: heuristic_results
            for heuristic, heuristic_results in self._heuristic_results.items()
            if not heuristic.trial_mode or include_trial
        }

        new_tp = TestPrioritizations(self._all_tests, {})

        for heuristic_results in valid_heuristics.values():
            for score, testrun in heuristic_results._traverse_scores():
                new_tp.add_test_score(testrun, score)
        new_tp.validate()
        return new_tp

    def get_test_stats(self, test: TestRun) -> dict[str, Any]:
        """
        Returns the aggregated statistics for a given test.

        Args:
        - test: The TestRun object for which statistics are to be retrieved.

        Returns:
        - dict: Statistics dictionary for the given test, including test name, filters, and heuristic metrics.
        """
        stats: dict[str, Any] = {
            "test_name": test.test_file,
            "test_filters": test.get_pytest_filter(),
        }

        # Get metrics about the heuristics used
        heuristics = []

        for heuristic, heuristic_results in self._heuristic_results.items():
            metrics = heuristic_results.get_priority_info_for_test(test)
            metrics["heuristic_name"] = heuristic.name
            metrics["trial_mode"] = heuristic.trial_mode
            heuristics.append(metrics)

        stats["heuristics"] = heuristics

        # Get aggregated priorities for the test
        stats["aggregated"] = self.get_aggregated_priorities().get_priority_info_for_test(test)

        # Get aggregated trial priorities for the test
        stats["aggregated_trial"] = self.get_aggregated_priorities(include_trial=True).get_priority_info_for_test(test)

        return stats
    def to_json(self) -> dict[str, Any]:
        """
        Returns a JSON dict that describes this AggregatedHeuristics object.
        """
        # 初始化一个空的字典，用于存储最终的 JSON 结果
        json_dict: dict[str, Any] = {}

        # 遍历聚合启发式结果字典中的每一个启发式及其结果
        for heuristic, heuristic_results in self._heuristic_results.items():
            # 将每个启发式的名字作为键，启发式结果对象的 JSON 表示作为值存储在 JSON 字典中
            json_dict[heuristic.name] = heuristic_results.to_json()

        # 返回包含所有启发式及其结果的 JSON 字典
        return json_dict
class HeuristicInterface:
    """
    Interface for all heuristics.
    """

    description: str  # 描述属性，用于存储启发式方法的描述信息

    # When trial mode is set to True, this heuristic's predictions will not be used
    # to reorder tests. It's results will however be emitted in the metrics.
    trial_mode: bool  # 试验模式标志，若为True，则该启发式方法的预测不用于重新排序测试，但结果将在指标中显示

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        self.trial_mode = kwargs.get("trial_mode", False)  # type: ignore[assignment]
        # 初始化方法，接受kwargs作为参数，设定trial_mode属性，默认为False

    @property
    def name(self) -> str:
        return self.__class__.__name__
        # 返回该类的名称作为字符串，即启发式方法的类名

    def __str__(self) -> str:
        return self.name
        # 返回启发式方法的名称的字符串表示形式

    @abstractmethod
    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        """
        Returns a float ranking ranging from -1 to 1, where negative means skip,
        positive means run, 0 means no idea, and magnitude = how confident the
        heuristic is. Used by AggregatedHeuristicsRankings.
        """
        pass
        # 抽象方法，根据给定的测试列表返回一个浮点数排名，范围从-1到1，负数表示跳过，正数表示运行，
        # 0表示无法确定，数值大小表示启发式方法的置信度。被聚合启发式方法排名使用。
```