# `.\pytorch\tools\test\heuristics\test_utils.py`

```
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

# 获取当前文件的父目录的父目录的父目录，作为根目录
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# 将根目录路径添加到系统路径中，以便导入相关模块
sys.path.append(str(REPO_ROOT))
# 导入测试中用到的工具函数模块
import tools.testing.target_determination.heuristics.utils as utils
from tools.testing.test_run import TestRun

# 从系统路径中移除根目录路径，避免影响其他模块的导入
sys.path.remove(str(REPO_ROOT))

# 定义测试类 TestHeuristicsUtils，继承自 unittest.TestCase
class TestHeuristicsUtils(unittest.TestCase):
    # 自定义断言方法，用于比较两个字典，其中值是 TestRun 对象和任意类型数据的字典
    def assertDictAlmostEqual(
        self, first: dict[TestRun, Any], second: dict[TestRun, Any]
    ) -> None:
        # 断言两个字典的键集合相等
        self.assertEqual(first.keys(), second.keys())
        # 遍历第一个字典的键
        for key in first.keys():
            # 使用 assertAlmostEqual 方法比较对应键的值，确保接近相等
            self.assertAlmostEqual(first[key], second[key])

    # 测试方法：测试 normalize_ratings 函数的功能
    def test_normalize_ratings(self) -> None:
        # 定义一个包含 TestRun 对象和浮点数值的字典 ratings
        ratings: dict[TestRun, float] = {
            TestRun("test1"): 1,
            TestRun("test2"): 2,
            TestRun("test3"): 4,
        }
        # 调用 utils 模块中的 normalize_ratings 函数，对 ratings 进行归一化处理
        normalized = utils.normalize_ratings(ratings, 4)
        # 断言归一化后的结果与原始 ratings 字典相等
        self.assertDictAlmostEqual(normalized, ratings)

        # 再次调用 normalize_ratings 函数，以不同的归一化参数进行处理
        normalized = utils.normalize_ratings(ratings, 0.1)
        # 断言归一化后的结果与预期的字典相等
        self.assertDictAlmostEqual(
            normalized,
            {
                TestRun("test1"): 0.025,
                TestRun("test2"): 0.05,
                TestRun("test3"): 0.1,
            },
        )

        # 第三次调用 normalize_ratings 函数，以不同的参数和最小值参数进行处理
        normalized = utils.normalize_ratings(ratings, 0.2, min_value=0.1)
        # 断言归一化后的结果与预期的字典相等
        self.assertDictAlmostEqual(
            normalized,
            {
                TestRun("test1"): 0.125,
                TestRun("test2"): 0.15,
                TestRun("test3"): 0.2,
            },
        )

# 如果当前脚本作为主程序运行，则执行 unittest 的主程序入口，启动测试
if __name__ == "__main__":
    unittest.main()
```