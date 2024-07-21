# `.\pytorch\test\jit\test_ignore_context_manager.py`

```
# Owner(s): ["oncall: jit"]

# 引入必要的库和模块
import os
import sys
import unittest

import torch

# 将测试目录的 helper 文件设为可导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 从 torch.jit.frontend 中导入 _IS_ASTUNPARSE_INSTALLED 变量
from torch.jit.frontend import _IS_ASTUNPARSE_INSTALLED
# 从 torch.testing._internal.jit_utils 中导入 JitTestCase 类
from torch.testing._internal.jit_utils import JitTestCase

# 如果当前脚本被直接运行，抛出 RuntimeError 提示用户不要直接运行此文件
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类 TestIgnoreContextManager，继承自 JitTestCase
class TestIgnoreContextManager(JitTestCase):

    # 使用装饰器 unittest.skipUnless 条件地跳过测试，当 _IS_ASTUNPARSE_INSTALLED 为 False 时跳过
    @unittest.skipUnless(_IS_ASTUNPARSE_INSTALLED, "astunparse package is required")
    # 定义测试方法 test_with_ignore_context_manager_with_inp_out
    def test_with_ignore_context_manager_with_inp_out(self):
        # 定义内部类 A，继承自 torch.nn.Module
        class A(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self):
                # 定义并初始化变量 a, b, c, d
                a: int = 4
                b: int = 5
                c: int = 0
                d: int = 6
                # 使用 torch.jit._IgnoreContextManager 上下文管理器
                with torch.jit._IgnoreContextManager(
                    a="inp:int", b="inp:int", c="out:int", d="out:int"
                ):
                    # 使用列表推导式创建列表 l
                    l = [2 for i in range(a) if i > 2]
                    # 计算 c 的值
                    c = l[0] + a + b
                    # 重新赋值变量 d
                    d = 9
                # 返回计算结果
                return c + d

        # 创建模型 A 的实例
        model = A()
        # 对模型进行脚本化编译
        s = torch.jit.script(model)
        # 断言脚本化模型的输出与直接执行模型的输出相等
        self.assertEqual(s(), model())
        # 断言脚本化模型的输出与预期值 20 相等
        self.assertEqual(s(), 20)

        # 定义内部类 B，继承自 torch.nn.Module
        class B(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self):
                # 定义并初始化变量 a, b, c
                a: int = 4
                b: int = 5
                c: int = 0
                # 使用 torch.jit._IgnoreContextManager 上下文管理器
                with torch.jit._IgnoreContextManager(
                    a="inp:int", b="inp:int", c="out:int"
                ):
                    # 使用列表推导式创建列表 l
                    l = [2 for i in range(a) if i > 2]
                    # 计算 c 的值
                    c = l[0] + a + b
                # 返回计算结果
                return c

        # 创建模型 B 的实例
        model = B()
        # 对模型进行脚本化编译
        s = torch.jit.script(model)
        # 断言脚本化模型的输出与预期值 11 相等
        self.assertEqual(s(), 11)
        # 断言脚本化模型的输出与直接执行模型的输出相等
        self.assertEqual(s(), model())

        # 定义内部类 C，继承自 torch.nn.Module
        class C(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self):
                # 定义并初始化变量 a, b
                a: int = 4
                b: int = 5
                # 使用 torch.jit._IgnoreContextManager 上下文管理器
                with torch.jit._IgnoreContextManager(a="inp:int", b="out:int"):
                    # 使用列表推导式创建列表 l
                    l = [2 for i in range(a) if i > 2]
                    # 计算 b 的值
                    b = l[0] + a
                # 返回计算结果
                return b

        # 创建模型 C 的实例
        model = C()
        # 对模型进行脚本化编译
        s = torch.jit.script(model)
        # 断言脚本化模型的输出与预期值 6 相等
        self.assertEqual(s(), 6)
        # 断言脚本化模型的输出与直接执行模型的输出相等
        self.assertEqual(s(), model())

    # 使用装饰器 unittest.skipUnless 条件地跳过测试，当 _IS_ASTUNPARSE_INSTALLED 为 False 时跳过
    @unittest.skipUnless(_IS_ASTUNPARSE_INSTALLED, "astunparse package is required")
    # 定义测试方法 test_with_ignore_context_manager_with_just_inp
    def test_with_ignore_context_manager_with_just_inp(self):
        # 定义内部类 A，继承自 torch.nn.Module
        class A(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self):
                # 定义并初始化变量 a, b
                a: int = 4
                b: int = 5
                # 使用 torch.jit._IgnoreContextManager 上下文管理器
                with torch.jit._IgnoreContextManager(a="inp:int", b="inp:int"):
                    # 使用列表推导式创建列表 l
                    l = [2 + b for i in range(a) if i > 2]
                # 返回变量 a 的值
                return a

        # 创建模型 A 的实例
        model = A()
        # 对模型进行脚本化编译
        s = torch.jit.script(model)
        # 断言脚本化模型的输出与预期值 4 相等
        self.assertEqual(s(), 4)
        # 断言脚本化模型的输出与直接执行模型的输出相等
        self.assertEqual(s(), model())

    # 使用装饰器 unittest.skipUnless 条件地跳过测试，当 _IS_ASTUNPARSE_INSTALLED 为 False 时跳过
    @unittest.skipUnless(_IS_ASTUNPARSE_INSTALLED, "astunparse package is required")
    def test_with_ignore_context_manager_with_just_out(self):
        class A(torch.nn.Module):
            def forward(self):
                # 定义一个忽略上下文管理器，指定输出类型为List[int]
                with torch.jit._IgnoreContextManager(c="out:List[int]"):
                    # 使用列表推导式生成一个列表，其中元素为2，范围为0到6且大于2的数
                    c = [2 for i in range(7) if i > 2]
                # 修改列表第一个元素为3
                c[0] = 3
                # 返回列表第一个和第二个元素的和
                return c[0] + c[1]

        model = A()  # 创建模型对象A
        s = torch.jit.script(model)  # 对模型进行脚本化
        self.assertEqual(s(), 5)  # 断言调用脚本化后的模型结果为5
        self.assertEqual(s(), model())  # 断言调用原始模型对象的结果与脚本化模型的结果相同
```