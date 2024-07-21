# `.\pytorch\torch\fx\passes\tests\test_pass_manager.py`

```py
import unittest

# 从 pass_manager 模块导入所需的类和函数
from ..pass_manager import (
    inplace_wrapper,
    PassManager,
    these_before_those_pass_constraint,
    this_before_that_pass_constraint,
)

# 定义一个测试类 TestPassManager，继承自 unittest.TestCase
class TestPassManager(unittest.TestCase):

    # 测试 PassManager 的构建方法
    def test_pass_manager_builder(self) -> None:
        # 创建一个包含 10 个 lambda 函数的列表
        passes = [lambda x: 2 * x for _ in range(10)]
        # 创建 PassManager 对象
        pm = PassManager(passes)
        # 调用 validate 方法
        pm.validate()

    # 测试 this_before_that_pass_constraint 方法
    def test_this_before_that_pass_constraint(self) -> None:
        passes = [lambda x: 2 * x for _ in range(10)]
        pm = PassManager(passes)

        # 添加一个无法满足的约束条件
        pm.add_constraint(this_before_that_pass_constraint(passes[-1], passes[0]))

        # 断言会抛出 RuntimeError 异常
        self.assertRaises(RuntimeError, pm.validate)

    # 测试 these_before_those_pass_constraint 方法
    def test_these_before_those_pass_constraint(self) -> None:
        passes = [lambda x: 2 * x for _ in range(10)]
        constraint = these_before_those_pass_constraint(passes[-1], passes[0])
        pm = PassManager(
            [inplace_wrapper(p) for p in passes]
        )

        # 添加一个无法满足的约束条件
        pm.add_constraint(constraint)

        # 断言会抛出 RuntimeError 异常
        self.assertRaises(RuntimeError, pm.validate)

    # 测试两个 PassManager 实例之间的状态隔离性
    def test_two_pass_managers(self) -> None:
        """Make sure we can construct the PassManager twice and not share any
        state between them"""

        passes = [lambda x: 2 * x for _ in range(3)]
        constraint = these_before_those_pass_constraint(passes[0], passes[1])
        pm1 = PassManager()
        for p in passes:
            pm1.add_pass(p)
        pm1.add_constraint(constraint)
        # 调用 pm1 对象
        output1 = pm1(1)
        # 断言输出结果为 2 的 3 次方
        self.assertEqual(output1, 2 ** 3)

        passes = [lambda x: 3 * x for _ in range(3)]
        constraint = these_before_those_pass_constraint(passes[0], passes[1])
        pm2 = PassManager()
        for p in passes:
            pm2.add_pass(p)
        pm2.add_constraint(constraint)
        # 调用 pm2 对象
        output2 = pm2(1)
        # 断言输出结果为 3 的 3 次方
        self.assertEqual(output2, 3 ** 3)
```