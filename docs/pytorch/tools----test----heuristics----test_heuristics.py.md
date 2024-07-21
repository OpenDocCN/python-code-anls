# `.\pytorch\tools\test\heuristics\test_heuristics.py`

```py
# 导入必要的模块和库
# 用于特定启发式测试
from __future__ import annotations

import io  # 导入输入输出模块
import json  # 导入处理 JSON 数据的模块
import sys  # 导入系统相关功能模块
import unittest  # 导入单元测试框架模块
from pathlib import Path  # 导入处理文件路径的模块
from typing import Any  # 导入类型提示模块
from unittest import mock  # 导入模拟测试框架

# 确定项目根目录
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 将项目根目录添加到系统路径中
sys.path.append(str(REPO_ROOT))

# 导入具体测试接口
from tools.test.heuristics.test_interface import TestTD
# 导入目标确定工具中的具体模块和类
from tools.testing.target_determination.determinator import TestPrioritizations
# 导入目标确定工具中的路径相关启发式功能
from tools.testing.target_determination.heuristics.filepath import (
    file_matches_keyword,
    get_keywords,
)
# 导入历史类失败相关的启发式功能
from tools.testing.target_determination.heuristics.historical_class_failure_correlation import (
    HistoricalClassFailurCorrelation,
)
# 导入之前 PR 中失败的启发式功能
from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
    get_previous_failures,
)
# 导入测试运行模块
from tools.testing.test_run import TestRun

# 从系统路径中移除项目根目录
sys.path.remove(str(REPO_ROOT))

# 历史类失败相关的启发式类名
HEURISTIC_CLASS = "tools.testing.target_determination.heuristics.historical_class_failure_correlation."


# 创建一个模拟文件对象，用于测试
def mocked_file(contents: dict[Any, Any]) -> io.IOBase:
    file_object = io.StringIO()
    json.dump(contents, file_object)  # 将内容转换为 JSON 并写入文件对象中
    file_object.seek(0)  # 将文件指针移动到文件开头
    return file_object  # 返回文件对象


# 生成历史类失败数据的示例函数
def gen_historical_class_failures() -> dict[str, dict[str, float]]:
    return {
        "file1": {
            "test1::classA": 0.5,
            "test2::classA": 0.2,
            "test5::classB": 0.1,
        },
        "file2": {
            "test1::classB": 0.3,
            "test3::classA": 0.2,
            "test5::classA": 1.5,
            "test7::classC": 0.1,
        },
        "file3": {
            "test1::classC": 0.4,
            "test4::classA": 0.2,
            "test7::classC": 1.5,
            "test8::classC": 0.1,
        },
    }


# 包含所有测试用例名称的列表
ALL_TESTS = [
    "test1",
    "test2",
    "test3",
    "test4",
    "test5",
    "test6",
    "test7",
    "test8",
]


# 测试历史类失败相关启发式功能的单元测试类，继承自 TestTD 基类
class TestHistoricalClassFailureCorrelation(TestTD):
    # 使用模拟函数来替代启发式类中的方法调用
    @mock.patch(
        HEURISTIC_CLASS + "_get_historical_test_class_correlations",
        return_value=gen_historical_class_failures(),
    )
    @mock.patch(
        HEURISTIC_CLASS + "query_changed_files",
        return_value=["file1"],
    )
    # 测试获取预测置信度的方法，接收历史类失败和变更文件列表作为参数
    def test_get_prediction_confidence(
        self,
        historical_class_failures: dict[str, dict[str, float]],
        changed_files: list[str],
        mocked_test_run: mock.Mock,
    ):
        # 在这里进行测试代码的具体实现，用来验证获取预测置信度的功能
        pass  # 在测试中暂时不实现具体的验证逻辑，留待后续扩展和实现
    ) -> None:
        # 设置要优先考虑的测试集为所有测试
        tests_to_prioritize = ALL_TESTS

        # 使用历史类故障相关性启发式算法创建一个启发式对象
        heuristic = HistoricalClassFailurCorrelation()
        # 获取测试集的预测置信度
        test_prioritizations = heuristic.get_prediction_confidence(tests_to_prioritize)

        # 预期的测试优先级结果对象
        expected = TestPrioritizations(
            tests_to_prioritize,
            {
                TestRun("test1::classA"): 0.25,
                TestRun("test2::classA"): 0.1,
                TestRun("test5::classB"): 0.05,
                TestRun("test1", excluded=["classA"]): 0.0,
                TestRun("test2", excluded=["classA"]): 0.0,
                TestRun("test3"): 0.0,
                TestRun("test4"): 0.0,
                TestRun("test5", excluded=["classB"]): 0.0,
                TestRun("test6"): 0.0,
                TestRun("test7"): 0.0,
                TestRun("test8"): 0.0,
            },
        )

        # 使用断言方法检查测试优先级结果是否几乎相等
        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores, expected._test_scores
        )
class TestParsePrevTests(TestTD):
    # 使用 mock.patch 伪造 os.path.exists 方法，使其返回 False
    @mock.patch("os.path.exists", return_value=False)
    def test_cache_does_not_exist(self, mock_exists: Any) -> None:
        # 预期的失败测试文件集合为空集
        expected_failing_test_files: set[str] = set()

        # 调用函数获取之前失败的测试文件集合
        found_tests = get_previous_failures()

        # 断言预期的失败测试文件集合与实际获取的集合相等
        self.assertSetEqual(expected_failing_test_files, found_tests)

    # 使用 mock.patch 伪造 os.path.exists 和 builtins.open 方法，使其返回 True 和一个模拟文件
    @mock.patch("os.path.exists", return_value=True)
    @mock.patch("builtins.open", return_value=mocked_file({"": True}))
    def test_empty_cache(self, mock_exists: Any, mock_open: Any) -> None:
        # 预期的失败测试文件集合为空集
        expected_failing_test_files: set[str] = set()

        # 调用函数获取之前失败的测试文件集合
        found_tests = get_previous_failures()

        # 断言预期的失败测试文件集合与实际获取的集合相等
        self.assertSetEqual(expected_failing_test_files, found_tests)
        
        # 断言 open 函数被调用了一次
        mock_open.assert_called()

    # 定义一个包含多个测试文件和测试用例的字典
    lastfailed_with_multiple_tests_per_file = {
        "test/test_car.py::TestCar::test_num[17]": True,
        "test/test_car.py::TestBar::test_num[25]": True,
        "test/test_far.py::TestFar::test_fun_copy[17]": True,
        "test/test_bar.py::TestBar::test_fun_copy[25]": True,
    }

    # 使用 mock.patch 伪造 os.path.exists 和 builtins.open 方法，使其返回 True 和模拟的文件内容
    @mock.patch("os.path.exists", return_value=True)
    @mock.patch(
        "builtins.open",
        return_value=mocked_file(lastfailed_with_multiple_tests_per_file),
    )
    def test_dedupes_failing_test_files(self, mock_exists: Any, mock_open: Any) -> None:
        # 预期的失败测试文件集合包含三个文件名前缀
        expected_failing_test_files = {"test_car", "test_bar", "test_far"}
        
        # 调用函数获取之前失败的测试文件集合
        found_tests = get_previous_failures()

        # 断言预期的失败测试文件集合与实际获取的集合相等
        self.assertSetEqual(expected_failing_test_files, found_tests)


class TestFilePath(TestTD):
    # 测试获取文件关键词的函数
    def test_get_keywords(self) -> None:
        # 断言给定文件名返回空列表
        self.assertEqual(get_keywords("test/test_car.py"), [])
        # 断言带有 nn 关键词的文件名返回 ["nn"]
        self.assertEqual(get_keywords("test/nn/test_amp.py"), ["nn"])
        # 断言带有 nn 关键词的文件名返回 ["nn"]
        self.assertEqual(get_keywords("torch/nn/test_amp.py"), ["nn"])
        # 断言带有 nn 和 amp 关键词的文件名返回 ["nn", "amp"]
        self.assertEqual(
            get_keywords("torch/nn/mixed_precision/test_amp.py"), ["nn", "amp"]
        )

    # 测试匹配关键词的函数
    def test_match_keywords(self) -> None:
        # 断言文件名包含 quant 关键词返回 True
        self.assertTrue(file_matches_keyword("test/quantization/test_car.py", "quant"))
        # 断言文件名包含 quant 关键词返回 True
        self.assertTrue(file_matches_keyword("test/test_quantization.py", "quant"))
        # 断言文件名包含 nn 关键词返回 True
        self.assertTrue(file_matches_keyword("test/nn/test_amp.py", "nn"))
        # 断言文件名包含 amp 关键词返回 True
        self.assertTrue(file_matches_keyword("test/nn/test_amp.py", "amp"))
        # 断言文件名包含 onnx 关键词返回 True
        self.assertTrue(file_matches_keyword("test/test_onnx.py", "onnx"))
        # 断言文件名不包含 nn 关键词返回 False
        self.assertFalse(file_matches_keyword("test/test_onnx.py", "nn"))
    def test_get_keywords_match(self) -> None:
        # 定义一个内部辅助函数helper，用于检查测试文件是否与修改后的文件匹配关键字
        def helper(test_file: str, changed_file: str) -> bool:
            # 调用file_matches_keyword函数，检查test_file中是否存在与changed_file中关键字匹配的文件
            return any(
                file_matches_keyword(test_file, x) for x in get_keywords(changed_file)
            )

        # 断言以下测试用例结果为True
        self.assertTrue(helper("test/quantization/test_car.py", "quantize/t.py"))
        # 断言以下测试用例结果为False
        self.assertFalse(helper("test/onnx/test_car.py", "nn/t.py"))
        # 断言以下测试用例结果为True
        self.assertTrue(helper("test/nn/test_car.py", "nn/t.py"))
        # 断言以下测试用例结果为False
        self.assertFalse(helper("test/nn/test_car.py", "test/b.py"))
        # 断言以下测试用例结果为True
        self.assertTrue(helper("test/test_mixed_precision.py", "torch/amp/t.py"))
        # 断言以下测试用例结果为True
        self.assertTrue(helper("test/test_amp.py", "torch/mixed_precision/t.py"))
        # 断言以下测试用例结果为True
        self.assertTrue(helper("test/idk/other/random.py", "torch/idk/t.py"))
# 如果当前脚本作为主程序运行，执行单元测试
if __name__ == "__main__":
    # 调用 unittest 模块的主函数，执行所有的单元测试
    unittest.main()
```