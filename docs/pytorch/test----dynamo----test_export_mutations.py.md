# `.\pytorch\test\dynamo\test_export_mutations.py`

```
# Owner(s): ["module: dynamo"]
# 导入unittest模块，用于编写和运行测试
import unittest

# 导入torch模块及其私有模块
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
# 导入torch.testing._internal.common_utils中的IS_FBCODE常量
from torch.testing._internal.common_utils import IS_FBCODE


# 定义MutationExportTests类，继承自torch._dynamo.test_case.TestCase
class MutationExportTests(torch._dynamo.test_case.TestCase):

    # 检查在导出时的失败情况
    def check_failure_on_export(self, mod, *args):
        with self.assertRaises(AssertionError):
            torch._dynamo.export(mod)(*args)

    # 检查导出后是否与原始结果相同
    def check_same_with_export(self, mod, arg):
        real_result = mod(arg)
        graph, _ = torch._dynamo.export(mod)(arg)
        result = graph(arg)
        self.assertEqual(result, real_result)

    # 测试模块属性变异违规情况（正向情况1）
    def test_module_attribute_mutation_violation_positive_1(self):
        # 定义一个Foo类，继承自torch.nn.Module
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化属性a为一个3x2的随机张量
                self.a = torch.randn(3, 2)

            def forward(self, x):
                # 将属性a转换为torch.float64类型
                self.a = self.a.to(torch.float64)
                return x.sum() + self.a.sum()

        # 调用check_failure_on_export方法验证导出时是否引发AssertionError
        self.check_failure_on_export(Foo(), torch.randn(3, 2))

    # 测试模块属性变异违规情况（负向情况1）
    def test_module_attribute_mutation_violation_negative_1(self):
        # 定义一个Foo类，继承自torch.nn.Module
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化属性a为一个3x2的随机张量
                self.a = torch.randn(3, 2)

            def forward(self, x):
                # 返回x的和加上属性a转换为torch.float64后的和
                return x.sum() + self.a.to(torch.float64).sum()

        # 调用check_same_with_export方法验证导出后是否与原始结果相同
        self.check_same_with_export(Foo(), torch.randn(3, 2))

    # 测试模块属性变异违规情况（负向情况2）
    def test_module_attribute_mutation_violation_negative_2(self):
        # 定义一个Foo类，继承自torch.nn.Module
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化属性a为一个3x2的随机张量
                self.a = torch.randn(3, 2)
                # 将属性a转换为torch.float64类型
                self.a = self.a.to(torch.float64)

            def forward(self, x):
                # 返回x的和加上属性a的和
                return x.sum() + self.a.sum()

        # 调用check_same_with_export方法验证导出后是否与原始结果相同
        self.check_same_with_export(Foo(), torch.randn(3, 2))

    # 测试模块属性变异违规情况（负向情况3）
    def test_module_attribute_mutation_violation_negative_3(self):
        # 定义一个Foo类，继承自torch.nn.Module
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化属性a为一个3x2的随机张量
                self.a = torch.randn(3, 2)

            def forward(self, x):
                # 定义本地变量b，并进行赋值操作
                b = 1
                b = b * 5
                # 返回x的和加上属性a的和再加上变量b
                return x.sum() + self.a.sum() + b

        # 调用check_same_with_export方法验证导出后是否与原始结果相同
        self.check_same_with_export(Foo(), torch.randn(3, 2))

    # 根据IS_FBCODE常量决定是否跳过测试
    @unittest.skipIf(IS_FBCODE, "Broken in fbcode")
    def test_module_attribute_mutation_violation_negative_4(self):
        # 定义一个测试函数，用于验证模块属性变异违规情况（案例4）

        # 定义一个继承自torch.nn.Module的类Foo，表示一个神经网络模块
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个形状为(3, 2)的张量a，其值为随机生成的标准正态分布数据
                self.a = torch.randn(3, 2)

            def forward(self, x):
                # 在forward方法中，将张量a的数据类型转换为torch.float64
                self.a = self.a.to(torch.float64)
                # 返回输入张量x的和加上张量a的和
                return x.sum() + self.a.sum()

        # 创建一个Foo类的实例mod
        mod = Foo()
        # 初始化一个形状为(3, 2)的张量arg，其值为随机生成的标准正态分布数据
        arg = torch.randn(3, 2)
        # 计算真实结果，即调用mod实例的__call__方法（即forward方法）
        real_result = mod(arg)

        # 使用torch._dynamo.optimize进行优化，传入mod作为参数
        opt_mod = torch._dynamo.optimize("eager", nopython=True)(mod)
        # 断言优化后的模块opt_mod对相同的输入arg得到的结果与真实结果一致
        self.assertEqual(opt_mod(arg), real_result)
# 如果这个脚本作为主程序运行（而不是被导入为模块），执行以下代码块
if __name__ == "__main__":
    # 从 torch 库中导入 _dynamo 模块下的 test_case，这通常用于测试
    from torch._dynamo.test_case import run_tests
    # 运行测试函数 run_tests()，这是一个用于执行测试的函数
    run_tests()
```