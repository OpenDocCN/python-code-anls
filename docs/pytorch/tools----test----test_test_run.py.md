# `.\pytorch\tools\test\test_test_run.py`

```py
# 导入系统、单元测试模块以及路径操作模块
import sys
import unittest
from pathlib import Path

# 确定项目根目录
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 尝试将项目根目录加入系统路径，以便优化测试运行
try:
    sys.path.append(str(REPO_ROOT))
    # 导入自定义测试运行相关模块
    from tools.testing.test_run import ShardedTest, TestRun
except ModuleNotFoundError:
    # 如果导入模块失败，则输出错误信息并退出程序
    print("Can't import required modules, exiting")
    sys.exit(1)

# 定义单元测试类 TestTestRun，继承自 unittest.TestCase
class TestTestRun(unittest.TestCase):

    # 测试运行对象的并集运算（全运行）
    def test_union_with_full_run(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun("foo::bar")

        # 断言运行对象的并集结果与预期相等
        self.assertEqual(run1 | run2, run1)
        self.assertEqual(run2 | run1, run1)

    # 测试运行对象的并集运算（包含特定文件）
    def test_union_with_inclusions(self) -> None:
        run1 = TestRun("foo::bar")
        run2 = TestRun("foo::baz")

        expected = TestRun("foo", included=["bar", "baz"])

        # 断言运行对象的并集结果与预期相等
        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    # 测试运行对象的并集运算（排除特定文件）
    def test_union_with_non_overlapping_exclusions(self) -> None:
        run1 = TestRun("foo", excluded=["bar"])
        run2 = TestRun("foo", excluded=["baz"])

        expected = TestRun("foo")

        # 断言运行对象的并集结果与预期相等
        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    # 测试运行对象的并集运算（有重叠的排除文件）
    def test_union_with_overlapping_exclusions(self) -> None:
        run1 = TestRun("foo", excluded=["bar", "car"])
        run2 = TestRun("foo", excluded=["bar", "caz"])

        expected = TestRun("foo", excluded=["bar"])

        # 断言运行对象的并集结果与预期相等
        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    # 测试运行对象的并集运算（混合包含和排除文件）
    def test_union_with_mixed_inclusion_exclusions(self) -> None:
        run1 = TestRun("foo", excluded=["baz", "car"])
        run2 = TestRun("foo", included=["baz"])

        expected = TestRun("foo", excluded=["car"])

        # 断言运行对象的并集结果与预期相等
        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    # 测试运行对象的并集运算（混合不同文件名）
    def test_union_with_mixed_files_fails(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun("bar")

        # 断言运行对象的并集操作引发 AssertionError
        with self.assertRaises(AssertionError):
            run1 | run2

    # 测试运行对象的并集运算（空文件）
    def test_union_with_empty_file_yields_orig_file(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun.empty()

        # 断言运行对象的并集结果与原对象相等
        self.assertEqual(run1 | run2, run1)
        self.assertEqual(run2 | run1, run1)

    # 测试运行对象的减法运算（从全运行中减去特定文件）
    def test_subtracting_full_run_fails(self) -> None:
        run1 = TestRun("foo::bar")
        run2 = TestRun("foo")

        # 断言从全运行中减去特定文件的结果为空运行对象
        self.assertEqual(run1 - run2, TestRun.empty())

    # 测试运行对象的减法运算（从空运行对象中减去文件得到原对象）
    def test_subtracting_empty_file_yields_orig_file(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun.empty()

        # 断言从空运行对象中减去文件得到原对象
        self.assertEqual(run1 - run2, run1)
        self.assertEqual(run2 - run1, TestRun.empty())

    # 测试空运行对象的真值为假
    def test_empty_is_falsey(self) -> None:
        self.assertFalse(TestRun.empty())

    # 测试运行对象的减法运算（从全运行中减去包含文件得到排除文件结果）
    def test_subtracting_inclusion_from_full_run(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun("foo::bar")

        expected = TestRun("foo", excluded=["bar"])

        # 断言从全运行中减去包含文件得到排除文件结果
        self.assertEqual(run1 - run2, expected)
    # 定义一个测试方法，用于测试从具有重叠包含关系的运行中减去包含的子集
    def test_subtracting_inclusion_from_overlapping_inclusion(self) -> None:
        # 创建一个包含"bar"和"baz"的TestRun对象
        run1 = TestRun("foo", included=["bar", "baz"])
        # 创建一个包含"foo::baz"的TestRun对象
        run2 = TestRun("foo::baz")
    
        # 断言从run1中减去run2得到的结果为包含"bar"的TestRun对象
        self.assertEqual(run1 - run2, TestRun("foo", included=["bar"]))
    
    # 定义一个测试方法，用于测试从具有不重叠包含关系的运行中减去包含的子集
    def test_subtracting_inclusion_from_nonoverlapping_inclusion(self) -> None:
        # 创建一个包含"bar"和"baz"的TestRun对象
        run1 = TestRun("foo", included=["bar", "baz"])
        # 创建一个包含"car"的TestRun对象
        run2 = TestRun("foo", included=["car"])
    
        # 断言从run1中减去run2得到的结果仍然为包含"bar"和"baz"的TestRun对象
        self.assertEqual(run1 - run2, TestRun("foo", included=["bar", "baz"]))
    
    # 定义一个测试方法，用于测试从完整运行中减去排除的子集
    def test_subtracting_exclusion_from_full_run(self) -> None:
        # 创建一个不包含任何包含项的TestRun对象
        run1 = TestRun("foo")
        # 创建一个排除"bar"的TestRun对象
        run2 = TestRun("foo", excluded=["bar"])
    
        # 断言从run1中减去run2得到的结果为包含"bar"的TestRun对象
        self.assertEqual(run1 - run2, TestRun("foo", included=["bar"]))
    
    # 定义一个测试方法，用于测试从超集排除中减去排除的子集
    def test_subtracting_exclusion_from_superset_exclusion(self) -> None:
        # 创建一个排除"bar"和"baz"的TestRun对象
        run1 = TestRun("foo", excluded=["bar", "baz"])
        # 创建一个仅排除"baz"的TestRun对象
        run2 = TestRun("foo", excluded=["baz"])
    
        # 断言从run1中减去run2得到的结果为空的TestRun对象
        self.assertEqual(run1 - run2, TestRun.empty())
        # 断言从run2中减去run1得到的结果为包含"bar"的TestRun对象
        self.assertEqual(run2 - run1, TestRun("foo", included=["bar"]))
    
    # 定义一个测试方法，用于测试从具有不重叠排除的运行中减去排除的子集
    def test_subtracting_exclusion_from_nonoverlapping_exclusion(self) -> None:
        # 创建一个排除"bar"和"baz"的TestRun对象
        run1 = TestRun("foo", excluded=["bar", "baz"])
        # 创建一个排除"car"的TestRun对象
        run2 = TestRun("foo", excluded=["car"])
    
        # 断言从run1中减去run2得到的结果为包含"car"的TestRun对象
        self.assertEqual(run1 - run2, TestRun("foo", included=["car"]))
        # 断言从run2中减去run1得到的结果为包含"bar"和"baz"的TestRun对象
        self.assertEqual(run2 - run1, TestRun("foo", included=["bar", "baz"]))
    
    # 定义一个测试方法，用于测试从排除中减去包含，且没有重叠的运行
    def test_subtracting_inclusion_from_exclusion_without_overlaps(self) -> None:
        # 创建一个排除"bar"和"baz"的TestRun对象
        run1 = TestRun("foo", excluded=["bar", "baz"])
        # 创建一个包含"bar"的TestRun对象
        run2 = TestRun("foo", included=["bar"])
    
        # 断言从run1中减去run2得到的结果为run1自身
        self.assertEqual(run1 - run2, run1)
        # 断言从run2中减去run1得到的结果为run2自身
        self.assertEqual(run2 - run1, run2)
    
    # 定义一个测试方法，用于测试从排除中减去包含，且有重叠的运行
    def test_subtracting_inclusion_from_exclusion_with_overlaps(self) -> None:
        # 创建一个排除"bar"和"baz"的TestRun对象
        run1 = TestRun("foo", excluded=["bar", "baz"])
        # 创建一个包含"bar"和"car"的TestRun对象
        run2 = TestRun("foo", included=["bar", "car"])
    
        # 断言从run1中减去run2得到的结果为排除"bar"、"baz"和"car"的TestRun对象
        self.assertEqual(run1 - run2, TestRun("foo", excluded=["bar", "baz", "car"]))
        # 断言从run2中减去run1得到的结果为包含"bar"的TestRun对象
        self.assertEqual(run2 - run1, TestRun("foo", included=["bar"]))
    
    # 定义一个测试方法，用于测试运行的逻辑与操作
    def test_and(self) -> None:
        # 创建一个包含"bar"和"baz"的TestRun对象
        run1 = TestRun("foo", included=["bar", "baz"])
        # 创建一个包含"bar"和"car"的TestRun对象
        run2 = TestRun("foo", included=["bar", "car"])
    
        # 断言run1与run2的逻辑与操作结果为包含"bar"的TestRun对象
        self.assertEqual(run1 & run2, TestRun("foo", included=["bar"]))
    
    # 定义一个测试方法，用于测试运行的逻辑与操作与排除
    def test_and_exclusions(self) -> None:
        # 创建一个排除"bar"和"baz"的TestRun对象
        run1 = TestRun("foo", excluded=["bar", "baz"])
        # 创建一个排除"bar"和"car"的TestRun对象
        run2 = TestRun("foo", excluded=["bar", "car"])
    
        # 断言run1与run2的逻辑与操作与排除结果为排除"bar"、"baz"和"car"的TestRun对象
        self.assertEqual(run1 & run2, TestRun("foo", excluded=["bar", "baz", "car"]))
class TestShardedTest(unittest.TestCase):
    # 定义一个测试类 TestShardedTest，继承自 unittest.TestCase

    def test_get_pytest_args(self) -> None:
        # 定义一个测试方法 test_get_pytest_args，返回类型为 None
        # 创建一个 TestRun 对象，参数为 "foo"，included 列表包含 "bar" 和 "baz"
        test = TestRun("foo", included=["bar", "baz"])
        
        # 创建一个 ShardedTest 对象，参数为 test 对象、分片数量 1、当前分片索引 1
        sharded_test = ShardedTest(test, 1, 1)

        # 预期的 pytest 参数列表
        expected_args = ["-k", "bar or baz"]

        # 断言 sharded_test 对象的 get_pytest_args 方法返回与 expected_args 相同的列表
        self.assertListEqual(sharded_test.get_pytest_args(), expected_args)

if __name__ == "__main__":
    # 如果当前脚本作为主程序执行，则运行 unittest 的主函数，执行所有测试
    unittest.main()
```