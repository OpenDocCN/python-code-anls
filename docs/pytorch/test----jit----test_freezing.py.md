# `.\pytorch\test\jit\test_freezing.py`

```py
# Owner(s): ["oncall: jit"]

# 导入所需的库和模块
import io
import unittest
from itertools import product
from typing import Any

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数库
from torch.jit._recursive import wrap_cpp_module  # 导入PyTorch JIT相关模块
from torch.testing import FileCheck  # 导入PyTorch测试相关模块
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_CUDNN  # 导入PyTorch CUDA相关测试
from torch.testing._internal.common_quantization import skipIfNoFBGEMM  # 导入PyTorch量化相关测试
from torch.testing._internal.common_quantized import override_quantized_engine  # 导入PyTorch量化引擎测试
from torch.testing._internal.common_utils import (  # 导入PyTorch通用测试工具
    set_default_dtype,
    skipCUDAMemoryLeakCheckIf,
    skipIfTorchDynamo,
    TEST_WITH_ROCM,
)
from torch.testing._internal.jit_utils import JitTestCase  # 导入PyTorch JIT测试工具
from torch.utils import mkldnn as mkldnn_utils  # 导入PyTorch MKLDNN相关工具

# 尝试导入torchvision库，设置是否有torchvision标志
try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
# 如果没有torchvision库，则跳过相关测试
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

# 如果直接运行此文件，则抛出运行时错误
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 检查当前环境是否支持ROCm
TEST_ROCM = torch.cuda.is_available() and torch.version.hip is not None


# 定义函数removeExceptions，用于移除图中的异常节点
def removeExceptions(graph):
    for n in graph.findAllNodes("prim::RaiseException"):
        n.destroy()


# 装饰器类skipIfTorchDynamo，用于标记测试用例以跳过Torch Dynamo引起的问题
@skipIfTorchDynamo("somehow causing hanging during python shutdown")
# 定义TestFreezing类，继承自JitTestCase，用于测试模型冻结相关功能
class TestFreezing(JitTestCase):
    # 定义一个测试用的方法，用于测试模块冻结功能
    def test_freeze_module(self):
        # 定义一个继承自 nn.Module 的内部类 M
        class M(nn.Module):
            # 构造函数，初始化各种属性
            def __init__(self):
                super().__init__()
                self.a = 1  # 被冻结
                self.b = 1.2  # 被冻结
                self.c = "hello"  # 被冻结
                self.c2 = "hi\xA1"  # 不被冻结，包含特殊字符
                self.d = [1, 1]  # 被冻结
                self.e = [1.0, 1.1]  # 被冻结
                self.f = ["hello", "world"]  # 被冻结
                self.f2 = [(1, "Over \u0e55\u0e57 57")]  # 不被冻结，包含特殊字符和 Unicode 转义
                self.g = (
                    [1, 2],
                    3.2,
                    "4.4",
                    torch.tensor([5.5], requires_grad=True),
                )  # 被冻结，包含张量和其他数据类型
                self.h = {"layer": [torch.tensor([7.7], requires_grad=True)]}  # 被冻结，包含张量
                self.h2 = {"layer\xB1": [torch.tensor([8.8], requires_grad=True)]}  # 不被冻结，包含特殊字符
                self.t = torch.tensor([1.2, 2.4], requires_grad=True)  # 被冻结，张量
                self.ts = [
                    torch.tensor([1.0, 2.0], requires_grad=True),
                    torch.tensor([3.0, 4.0], requires_grad=True),
                ]  # 被冻结，包含张量列表
                self.tt = [[torch.tensor([3.3, 2.3], requires_grad=True), None]]  # 被冻结，包含张量和 None

            # 前向传播函数，返回所有属性的字符串表示连接
            def forward(self, x):
                return (
                    str(self.a)
                    + str(self.b)
                    + self.c
                    + self.c2
                    + str(self.d)
                    + str(self.e)
                    + str(self.f)
                    + str(self.f2)
                    + str(self.g)
                    + str(self.h)
                    + str(self.h2)
                    + str(self.t)
                    + str(self.ts)
                    + str(self.tt)
                )

        # 使用 torch.jit.script 将 M 类实例化为 TorchScript 模块，并设置为评估模式
        m = torch.jit.script(M())
        m.eval()
        # 创建一个输入张量
        input = torch.randn(2, 2)
        # 对模块进行冻结
        m._c = torch._C._freeze_module(m._c)
        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将冻结后的模块保存到字节流中
        torch.jit.save(m._c, buffer)
        buffer.seek(0)
        # 从字节流中加载冻结后的模块
        m2 = torch.jit.load(buffer)
        # 检查冻结后的模块是否如下所示：
        # module m {
        #   attributes {
        #     tt = ...
        #   }
        #   ...
        # }
        # 逐个断言冻结后的模块中不存在各属性
        self.assertFalse(m2._c.hasattr("a"))
        self.assertFalse(m2._c.hasattr("b"))
        self.assertFalse(m2._c.hasattr("c"))
        self.assertFalse(m2._c.hasattr("c2"))
        self.assertFalse(m2._c.hasattr("d"))
        self.assertFalse(m2._c.hasattr("e"))
        self.assertFalse(m2._c.hasattr("f"))
        self.assertFalse(m2._c.hasattr("f2"))
        self.assertFalse(m2._c.hasattr("g"))
        self.assertFalse(m2._c.hasattr("h"))
        self.assertFalse(m2._c.hasattr("h2"))
        self.assertFalse(m2._c.hasattr("t"))
        self.assertFalse(m2._c.hasattr("ts"))
        self.assertFalse(m2._c.hasattr("tt"))
        # 使用冻结后的模块进行前向传播并验证输出与未冻结模块一致
        output_f = m2.forward(input)
        self.assertEqual(output_s, output_f)
    def test_freeze_module_with_submodule(self):
        # 定义一个名为 test_freeze_module_with_submodule 的测试方法
        class SubModule(nn.Module):
            # 定义一个名为 SubModule 的子模块类，继承自 nn.Module
            def __init__(self):
                # 子模块类的初始化方法
                super().__init__()
                self.a = 11
                self.b = 2

            def forward(self, x):
                # 子模块类的前向传播方法，返回 self.a + self.b 的结果
                return self.a + self.b

        class SubModule2(nn.Module):
            # 定义另一个名为 SubModule2 的子模块类，继承自 nn.Module
            def __init__(self):
                # 子模块类的初始化方法
                super().__init__()
                self.a = 12
                self.b = 2

            def forward(self, x):
                # 子模块类的前向传播方法，将 self.b 设为 30，返回 self.a + self.b 的结果
                self.b = 30
                return self.a + self.b

        class TestModule(nn.Module):
            # 定义一个名为 TestModule 的主模块类，继承自 nn.Module
            def __init__(self):
                # 主模块类的初始化方法
                super().__init__()
                self.sub1 = SubModule()
                self.sub2 = SubModule2()
                self.a = 3
                self.b = 4

            def forward(self, x):
                # 主模块类的前向传播方法
                self.b = 20
                return self.sub1(x) + self.a + self.b + self.sub2(x)

        m = torch.jit.script(TestModule())
        # 使用 torch.jit.script 将 TestModule 转换为 Torch 脚本模块对象 m
        m.eval()
        # 将模型 m 设为评估模式
        input = torch.randn(2, 2)
        # 生成一个形状为 (2, 2) 的随机输入张量
        output_s = m.forward(input)
        # 使用模型 m 对输入进行前向传播，得到输出 output_s
        mf = torch.jit.freeze(m)
        # 使用 torch.jit.freeze 冻结模型 m，并将结果赋给 mf

        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     sub2 = ...
        #      b =
        #   }
        #   ...
        #   submodule {
        #     module m {
        #       attributes {
        #         sub2 = ...
        #         b =
        #       }
        #       ...
        #     }
        #   }
        # }
        # 检查冻结后的模型结构是否如上所示

        mf = mf._c
        # 获取冻结模型的底层 C++ 对象
        self.assertFalse(mf.hasattr("sub1"))
        # 断言冻结模型 mf 不包含属性 "sub1"
        self.assertFalse(mf.hasattr("a"))
        # 断言冻结模型 mf 不包含属性 "a"
        self.assertTrue(mf.hasattr("b"))
        # 断言冻结模型 mf 包含属性 "b"
        self.assertTrue(mf.hasattr("sub2"))
        # 断言冻结模型 mf 包含属性 "sub2"
        self.assertTrue(mf.sub2.hasattr("b"))  # verify b is preserved in sub2
        # 断言冻结模型 mf 的子模块 sub2 中仍然包含属性 "b"
        self.assertFalse(mf.sub2.hasattr("a"))  # verify a is removed in sub2
        # 断言冻结模型 mf 的子模块 sub2 中不包含属性 "a"
        output_f = mf.forward(input)
        # 使用冻结模型 mf 对输入进行前向传播，得到输出 output_f
        self.assertEqual(output_s, output_f)
        # 断言冻结模型 mf 的输出 output_f 与未冻结模型 m 的输出 output_s 相等
    def test_freeze_module_with_fork(self):
        class SubModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.ones(20, 20)  # 创建一个20x20的全为1的张量a
                self.b = torch.ones(20, 20)  # 创建一个20x20的全为1的张量b

            def forward(self, x):
                return self.a * self.b + x  # 返回张量a与b的乘积加上输入张量x的结果

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModule()  # 创建SubModule的实例作为子模块

            def forward(self, x):
                fut = torch.jit._fork(self.sub.forward, x)  # 使用fork并行执行子模块的forward方法
                y_hat = self.sub(x)  # 直接调用子模块的forward方法
                y = torch.jit._wait(fut)  # 等待并获取并行执行的结果
                return y_hat + y  # 返回y_hat和y的和作为输出

        m = torch.jit.script(TestModule())  # 将TestModule转换为脚本模式
        m.eval()  # 设置模型为评估模式
        input = torch.randn(20, 20)  # 创建一个20x20的随机张量作为输入
        output_s = m.forward(input)  # 调用模型的forward方法得到输出
        mf = torch._C._freeze_module(m._c)  # 冻结模型

        # 检查冻结后的模型是否如下所示:
        # module m {
        #   attributes {
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        self.assertFalse(mf.hasattr("a"))  # 断言冻结后的模型没有属性"a"
        self.assertFalse(mf.hasattr("b"))  # 断言冻结后的模型没有属性"b"
        output_f = mf.forward(input)  # 使用冻结后的模型进行前向推断得到输出
        self.assertEqual(output_s, output_f)  # 断言未冻结和冻结后的输出结果相等

    def test_freeze_module_with_nested_fork(self):
        class SubModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.ones(20, 20)  # 创建一个20x20的全为1的张量a
                self.b = torch.ones(20, 20)  # 创建一个20x20的全为1的张量b

            def forward(self, x):
                return self.a * self.b + x  # 返回张量a与b的乘积加上输入张量x的结果

        class SubModule2(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModule()  # 创建SubModule的实例作为子模块
                self.c = torch.ones(20, 20)  # 创建一个20x20的全为1的张量c

            def forward(self, x):
                fut = torch.jit._fork(self.sub.forward, x)  # 使用fork并行执行子模块的forward方法
                y_hat = self.sub(x)  # 直接调用子模块的forward方法
                y = torch.jit._wait(fut)  # 等待并获取并行执行的结果
                return y_hat + y + self.c  # 返回y_hat、y和张量c的和作为输出

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModule2()  # 创建SubModule2的实例作为子模块
                self.d = 1  # 初始化属性d为1

            def forward(self, x):
                fut = torch.jit._fork(self.sub.forward, x)  # 使用fork并行执行子模块的forward方法
                y_hat = self.sub(x)  # 直接调用子模块的forward方法
                y = torch.jit._wait(fut)  # 等待并获取并行执行的结果
                self.d = 2  # 修改属性d的值为2
                return y_hat * y + self.d  # 返回y_hat乘以y再加上属性d的结果作为输出

        m = torch.jit.script(TestModule())  # 将TestModule转换为脚本模式
        m.eval()  # 设置模型为评估模式
        input = torch.randn(20, 20)  # 创建一个20x20的随机张量作为输入
        output_s = m.forward(input)  # 调用模型的forward方法得到输出
        mf = torch._C._freeze_module(m._c)  # 冻结模型

        # 检查冻结后的模型是否如下所示:
        # module m {
        #   attributes {
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        self.assertFalse(mf.hasattr("a"))  # 断言冻结后的模型没有属性"a"
        self.assertFalse(mf.hasattr("b"))  # 断言冻结后的模型没有属性"b"
        self.assertFalse(mf.hasattr("c"))  # 断言冻结后的模型没有属性"c"
        self.assertTrue(mf.hasattr("d"))   # 断言冻结后的模型有属性"d"
        output_f = mf.forward(input)  # 使用冻结后的模型进行前向推断得到输出
        self.assertEqual(output_s, output_f)  # 断言未冻结和冻结后的输出结果相等
    def test_freeze_module_with_fork2(self):
        @torch.jit.script
        # 定义一个 TorchScript 函数 foo，对输入 x 执行乘以 2 的操作
        def foo(x):
            return x * 2

        # 定义一个继承自 nn.Module 的测试模块 TestModule
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模块的属性 a 和 b，都为 20x20 大小的全一张量
                self.a = torch.ones(20, 20)
                self.b = torch.ones(20, 20)

            def forward(self, x):
                # 使用 torch.jit._fork 异步执行 foo 函数，传入 self.a 作为参数
                fut = torch.jit._fork(foo, self.a)
                # 同步执行 foo 函数，传入 self.b 作为参数
                y_hat = foo(self.b)
                # 等待异步操作 fut 的完成
                y = torch.jit._wait(fut)
                # 返回 y_hat 和 y 的和作为模型的输出
                return y_hat + y

        # 将 TestModule 转换为 TorchScript 模型
        m = torch.jit.script(TestModule())
        # 设置模型为评估模式
        m.eval()
        # 创建输入张量 input，大小为 2x2，值为随机数
        input = torch.randn(2, 2)
        # 获取原始模型 m 的前向传播输出
        output_s = m.forward(input)
        # 冻结 TorchScript 模型 m，返回冻结后的模型 mf
        mf = torch._C._freeze_module(m._c)

        # 检查冻结模型 mf 是否如下所示:
        # module m {
        #   attributes {
        #     self.a = ...
        #     self.b = ..
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        # TODO: 尽管没有发生变异，但别名分析保守地假设存在变异，因为属性被传递给分支子图。'a' 和 'b' 都被保留。
        # 断言冻结模型 mf 是否有属性 "a"
        self.assertTrue(mf.hasattr("a"))
        # 断言冻结模型 mf 是否没有属性 "b"
        self.assertFalse(mf.hasattr("b"))
        # 获取冻结模型 mf 的前向传播输出
        output_f = mf.forward(input)
        # 断言原始模型和冻结模型的前向传播输出相等
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_fork_calling_module_method(self):
        @torch.jit.script
        # 定义一个 TorchScript 函数 foo，接受两个参数 x 和 y，并返回它们的乘积
        def foo(x, y):
            return x * y

        # 定义一个继承自 nn.Module 的测试模块 TestModule
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模块的属性 a 和 b，都为 20x20 大小的全一张量
                self.a = torch.ones(20, 20)
                self.b = torch.ones(20, 20)

            @torch.jit.export
            # 定义一个导出的 TorchScript 方法 foo，接受参数 x，并返回 x 与 self.a 的乘积
            def foo(self, x):
                return x * self.a

            @torch.jit.export
            # 定义一个导出的 TorchScript 方法 bar，接受参数 x，并返回 x 与 self.b 的乘积
            def bar(self, x):
                return x * self.b

            def forward(self, x):
                # 使用 torch.jit._fork 异步执行 self.foo 方法，传入 self.b 作为参数
                fut = torch.jit._fork(self.foo, self.b)
                # 同步执行 self.bar 方法，传入 self.a 作为参数
                y_hat = self.bar(self.a)
                # 等待异步操作 fut 的完成
                y = torch.jit._wait(fut)
                # 返回 y_hat 和 y 的和作为模型的输出
                return y_hat + y

        # 将 TestModule 转换为 TorchScript 模型
        m = torch.jit.script(TestModule())
        # 设置模型为评估模式
        m.eval()
        # 创建输入张量 input，大小为 2x2，值为随机数
        input = torch.randn(2, 2)
        # 获取原始模型 m 的前向传播输出
        output_s = m.forward(input)
        # 冻结 TorchScript 模型 m，返回冻结后的模型 mf
        mf = torch._C._freeze_module(m._c)

        # 检查冻结模型 mf 是否如下所示:
        # module m {
        #   attributes {
        #     self.b = ..
        #   }
        #   ...
        # TODO: 尽管没有发生变异，但别名分析保守地假设存在变异，因为属性被传递给分支子图。'b' 被保留。
        # 断言冻结模型 mf 是否没有属性 "a"
        self.assertFalse(mf.hasattr("a"))
        # 断言冻结模型 mf 是否有属性 "b"
        self.assertTrue(mf.hasattr("b"))
        # 获取冻结模型 mf 的前向传播输出
        output_f = mf.forward(input)
        # 断言原始模型和冻结模型的前向传播输出相等
        self.assertEqual(output_s, output_f)
    def test_freeze_module_with_sharedclasstype(self):
        # 定义一个测试函数，用于测试冻结带有共享类类型的模块

        class SubModule(nn.Module):
            # 定义子模块 SubModule
            def __init__(self):
                super().__init__()
                self.a = torch.tensor([1.1])  # 初始化张量 a
                self.b = torch.tensor([2.2])  # 初始化张量 b

            def forward(self, x):
                # 前向传播函数，返回 a 和 b 的和
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                # 修改张量 a 的值，并返回张量 b
                self.a[0] += 10
                return self.b

            @torch.jit.export
            def modify_b(self, x):
                # 修改张量 b 的值，并返回张量 a
                self.b[0] += 20
                return self.a

        class SubModule2(nn.Module):
            # 定义子模块 SubModule2
            def __init__(self):
                super().__init__()
                self.sub = SubModule()  # 创建 SubModule 实例
                self.b = torch.tensor([3.3])  # 初始化张量 b

            def forward(self, x):
                # 前向传播函数，调用 SubModule 实例的 modify_b 方法，并返回结果与张量 b 的和
                y = self.sub.modify_b(x)
                return y + self.b

        class TestModule(nn.Module):
            # 定义测试模块 TestModule
            def __init__(self):
                super().__init__()
                self.sub1 = SubModule()  # 创建 SubModule 实例 sub1
                self.sub2 = SubModule2()  # 创建 SubModule2 实例 sub2
                self.a = torch.tensor([4.4])  # 初始化张量 a

            def forward(self, x):
                # 前向传播函数，调用 sub1 的 modify_a 方法，计算 sub2 的前向传播结果，返回计算结果的总和
                z = self.sub1.modify_a(x)
                return self.sub2(x) + z + self.a

        m = torch.jit.script(TestModule())  # 使用 Torch Script 对 TestModule 进行脚本化
        m.eval()  # 将模型设为评估模式
        input = torch.randn(2, 2)  # 生成输入张量
        output_s = m.forward(input)  # 调用模型的前向传播方法，得到原始模型的输出结果
        mf = torch._C._freeze_module(m._c)  # 冻结模型，生成 mf

        # 检查冻结后的模块结构是否如下所示
        # module mf {
        #   attributes {
        #     sub1 = ...
        #     sub2 = ...
        #   }
        #   ...
        #   submodules {
        #     module sub1 {
        #       attributes {
        #         a = ...
        #         b = ...
        #       }
        #       ...
        #     }
        #     module sub2 {
        #       attributes {
        #         sub = ...
        #       }
        #       ...
        #       submodule {
        #         module sub {
        #           attributes {
        #             a = ...
        #             b = ...
        #           }
        #           ...
        #         }
        #       }
        #     }
        #   }
        # }

        self.assertTrue(mf.hasattr("sub1"))  # 断言 mf 中有名为 "sub1" 的属性
        self.assertTrue(mf.sub1.hasattr("a"))  # 断言 mf.sub1 中有名为 "a" 的属性
        self.assertTrue(mf.sub1.hasattr("b"))  # 断言 mf.sub1 中有名为 "b" 的属性
        self.assertFalse(mf.hasattr("a"))  # 断言 mf 中没有名为 "a" 的属性
        self.assertTrue(mf.hasattr("sub2"))  # 断言 mf 中有名为 "sub2" 的属性
        self.assertTrue(mf.sub2.hasattr("sub"))  # 断言 mf.sub2 中有名为 "sub" 的属性
        self.assertFalse(mf.sub2.hasattr("b"))  # 断言 mf.sub2 中没有名为 "b" 的属性
        self.assertTrue(mf.sub2.sub.hasattr("a"))  # 断言 mf.sub2.sub 中有名为 "a" 的属性
        self.assertTrue(mf.sub2.sub.hasattr("b"))  # 断言 mf.sub2.sub 中有名为 "b" 的属性
        output_f = mf.forward(input)  # 调用冻结模型 mf 的前向传播方法，得到输出结果
        self.assertEqual(output_s, output_f)  # 断言原始模型和冻结模型的输出结果相等
    # 定义一个测试方法，用于验证模块冻结时的嵌套别名问题
    def test_freeze_module_with_nestedaliasing(self):
        # 定义一个继承自 nn.Module 的子模块类 SubModule
        class SubModule(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                self.a = torch.tensor([1.1])  # 初始化张量 a
                self.b = torch.tensor([2.2])  # 初始化张量 b

            # 前向传播方法
            def forward(self, x):
                return self.a + self.b

            # 定义一个导出的方法 modify_a
            @torch.jit.export
            def modify_a(self, x):
                self.a[0] = 10  # 修改张量 a 的第一个元素为 10
                return self.b  # 返回张量 b

            # 定义一个导出的方法 modify_b
            @torch.jit.export
            def modify_b(self, x):
                self.b[0] = 20  # 修改张量 b 的第一个元素为 20
                return self.a  # 返回张量 a

        Sub = SubModule()  # 创建 SubModule 的实例 Sub

        # 定义另一个继承自 nn.Module 的子模块类 SubModule2
        class SubModule2(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                self.sub = Sub  # 别名引用 SubModule 实例 Sub

            # 前向传播方法
            def forward(self, x):
                return self.sub.a  # 返回 SubModule 实例 Sub 中的张量 a

        # 定义一个继承自 nn.Module 的测试模块类 TestModule
        class TestModule(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                self.sub1 = Sub  # 别名引用 SubModule 实例 Sub
                self.sub2 = SubModule2()  # 创建 SubModule2 的实例 sub2

            # 前向传播方法
            def forward(self, x):
                z = self.sub1.modify_a(x)  # 调用 SubModule 实例 Sub 的 modify_a 方法
                return self.sub2(x) + z  # 返回 sub2 的前向传播结果加上 z

        m = torch.jit.script(TestModule())  # 对 TestModule 进行脚本化
        m.eval()  # 设置模型为评估模式
        mf = torch._C._freeze_module(m._c)  # 冻结模型 m 得到 mf
        self.assertTrue(mf.hasattr("sub1"))  # 断言 mf 是否具有属性 "sub1"
        self.assertTrue(mf.sub1.hasattr("a"))  # 断言 mf.sub1 是否具有属性 "a"
        self.assertFalse(mf.sub1.hasattr("b"))  # 断言 mf.sub1 是否不具有属性 "b"
        self.assertTrue(mf.hasattr("sub2"))  # 断言 mf 是否具有属性 "sub2"
        self.assertTrue(mf.sub2.hasattr("sub"))  # 断言 mf.sub2 是否具有属性 "sub"
        self.assertTrue(
            mf.sub2.sub.hasattr("a")
        )  # 断言 mf.sub2.sub 是否具有属性 "a"，冻结检测到 self.sub2.sub.a 和 self.sub1.a 是别名
        self.assertFalse(mf.sub2.sub.hasattr("b"))  # 断言 mf.sub2.sub 是否不具有属性 "b"
        input = torch.randn(2, 2)  # 创建输入张量 input
        output_s = m.forward(input)  # 调用原始模型的前向传播
        output_f = mf.forward(input)  # 调用冻结模型的前向传播
        self.assertEqual(output_s, output_f)  # 断言两者的输出结果相等

    # FIXME: JIT is not honoring aliasing. 'Sub' module is copied. As a result
    # Eager and Script modules produce different output.
    def test_freeze_module_with_nestedaliasingscalar(self):
        class SubModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 1.1  # 初始化属性 a 为 1.1
                self.b = 2.2  # 初始化属性 b 为 2.2

            def forward(self, x):
                return self.a + self.b  # 返回 a 和 b 的和作为输出

            @torch.jit.export
            def modify_a(self, x):
                self.a = 10.0  # 修改属性 a 的值为 10.0
                return self.b  # 返回属性 b 的值

            @torch.jit.export
            def modify_b(self, x):
                self.b = 20.0  # 修改属性 b 的值为 20.0
                return self.a  # 返回属性 a 的值

        Sub = SubModule()  # 创建 SubModule 的实例 Sub

        class SubModule2(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = Sub  # 引用别名，将 SubModule 实例 Sub 赋给属性 sub

            def forward(self, x):
                return self.sub.a  # 返回 sub 中的属性 a 的值

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub1 = Sub  # 引用别名，将 SubModule 实例 Sub 赋给属性 sub1
                self.sub2 = SubModule2()  # 创建 SubModule2 的实例 sub2

            def forward(self, x):
                z = self.sub1.modify_a(x)  # 调用 sub1 的 modify_a 方法，修改 sub1 的属性 a，并返回 sub1 的属性 b 的值
                return self.sub2(x) + z  # 返回 sub2 的输出加上 z

        m = TestModule()  # 创建 TestModule 的实例 m
        ms = torch.jit.script(m)  # 对 m 进行脚本化
        ms.eval()  # 将脚本化后的模型设置为评估模式
        mf = torch._C._freeze_module(ms._c)  # 冻结脚本化的模型，返回冻结后的模型 mf
        self.assertTrue(mf.hasattr("sub1"))  # 断言 mf 中包含属性 "sub1"
        self.assertTrue(mf.sub1.hasattr("a"))  # 断言 mf.sub1 中包含属性 "a"
        self.assertFalse(mf.sub1.hasattr("b"))  # 断言 mf.sub1 不包含属性 "b"
        # sub2 is fully folded becasue self.sub1 and self.sub2.sub are not alias (Scripting bug)
        self.assertFalse(mf.hasattr("sub2"))  # 断言 mf 不包含属性 "sub2"
        input = torch.randn(2, 2)  # 生成一个 2x2 的随机张量 input
        output = m.forward(input)  # 计算模型 m 对输入 input 的输出
        output_s = ms.forward(input)  # 计算脚本化模型 ms 对输入 input 的输出
        output_f = mf.forward(input)  # 计算冻结模型 mf 对输入 input 的输出
        # Should be equal
        self.assertNotEqual(output, output_s)  # 断言 output 不等于 output_s
        self.assertEqual(output_s, output_f)  # 断言 output_s 等于 output_f

    def test_freeze_module_with_preserve_sub_module(self):
        class SubModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.tensor([1.1])  # 初始化属性 a 为包含 1.1 的张量
                self.b = 2.2  # 初始化属性 b 为 2.2

            def forward(self, x):
                return self.a  # 返回属性 a

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub1 = SubModule()  # 引用别名，将 SubModule 实例 SubModule 赋给属性 sub1
                self.sub2 = SubModule()  # 创建 SubModule 的实例 sub2

            def forward(self, x):
                return self.sub2(x) + self.sub1(x)  # 返回 sub2 的输出加上 sub1 的输出

        m = TestModule()  # 创建 TestModule 的实例 m
        ms = torch.jit.script(m)  # 对 m 进行脚本化
        ms.eval()  # 将脚本化后的模型设置为评估模式
        mf = torch._C._freeze_module(ms._c, ["sub1"])  # 冻结脚本化的模型，但保留 sub1，返回冻结后的模型 mf

        # Test that 'sub1' is preserved entirely and 'sub2' is completely folded
        self.assertTrue(mf.hasattr("sub1"))  # 断言 mf 中包含属性 "sub1"
        self.assertTrue(mf.sub1.hasattr("a"))  # 断言 mf.sub1 中包含属性 "a"
        self.assertTrue(mf.sub1.hasattr("b"))  # 断言 mf.sub1 中包含属性 "b"
        self.assertFalse(mf.hasattr("sub2"))  # 断言 mf 不包含属性 "sub2"
        input = torch.randn(2, 2)  # 生成一个 2x2 的随机张量 input
        output_s = ms.forward(input)  # 计算脚本化模型 ms 对输入 input 的输出
        output_f = mf.forward(input)  # 计算冻结模型 mf 对输入 input 的输出
        self.assertEqual(output_s, output_f)  # 断言 output_s 等于 output_f
    def test_freeze_module_with_preserve_sub_module_and_mutation(self):
        # 定义子模块类 SubModule
        class SubModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.tensor([1.1])  # 初始化张量 a
                self.b = 2.2  # 初始化数值 b

            def forward(self, x):
                self.a[0] = 3.3  # 修改张量 a 中的值
                return self.a

        # 定义测试模块类 TestModule
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub1 = SubModule()  # 实例化子模块 SubModule，并命名为 sub1
                self.sub2 = SubModule()  # 实例化另一个子模块 SubModule，并命名为 sub2

            def forward(self, x):
                return self.sub2(x) + self.sub1(x)  # 返回两个子模块的输出的和

        m = TestModule()  # 实例化 TestModule
        ms = torch.jit.script(m)  # 对 TestModule 进行脚本化
        ms.eval()  # 设定为评估模式
        mf = torch._C._freeze_module(ms._c, ["sub1"])  # 冻结模块并保留 sub1

        # 测试 sub1 和 sub1 都被保留，并且 'b' 被保留即使未被使用。
        # 满足用户要求保留 'sub1'
        self.assertTrue(mf.hasattr("sub1"))  # 断言 mf 中有 sub1 属性
        self.assertTrue(mf.sub1.hasattr("a"))  # 断言 mf.sub1 中有 a 属性
        self.assertTrue(mf.sub1.hasattr("b"))  # 断言 mf.sub1 中有 b 属性
        self.assertTrue(mf.hasattr("sub2"))  # 断言 mf 中有 sub2 属性
        self.assertTrue(mf.sub2.hasattr("a"))  # 断言 mf.sub2 中有 a 属性
        self.assertTrue(mf.sub2.hasattr("b"))  # 断言 mf.sub2 中有 b 属性
        input = torch.randn(2, 2)  # 创建输入张量
        output_s = ms.forward(input)  # 使用原始模块计算前向传播输出
        output_f = mf.forward(input)  # 使用冻结模块计算前向传播输出
        self.assertEqual(output_s, output_f)  # 断言两种模式下的输出相等

    def test_freeze_module_with_helperfunction(self):
        # 定义子模块类 SubModule
        class SubModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 11  # 初始化变量 a
                self.b = 2   # 初始化变量 b

            def forward(self, x):
                return self.a + self.b  # 返回 a 和 b 的和作为输出

        # 定义测试模块类 TestModule
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModule()  # 实例化子模块 SubModule，并命名为 sub
                self.a = 3  # 初始化变量 a
                self.b = 4  # 初始化变量 b

            def forward(self, x):
                self.b = 20  # 修改变量 b 的值
                return self._forward(x) + self.a + self.b  # 返回调用 _forward 方法的结果加上 a 和 b

            def _forward(self, x):
                return self.sub(x)  # 返回子模块 SubModule 的输出作为结果

        m = torch.jit.script(TestModule())  # 对 TestModule 进行脚本化
        m.eval()  # 设定为评估模式
        input = torch.randn(2, 2)  # 创建输入张量
        mf = torch._C._freeze_module(m._c)  # 冻结整个模块
        self.assertFalse(mf.hasattr("sub"))  # 断言 mf 中没有 sub 属性
        self.assertFalse(mf.hasattr("a"))    # 断言 mf 中没有 a 属性
        self.assertTrue(mf.hasattr("b"))     # 断言 mf 中有 b 属性
        with self.assertRaisesRegex(
            AttributeError, "TestModule (.*) does not have a field with name '_forward'"
        ):
            mf._forward(x)  # 预期抛出 AttributeError 异常，因为 mf 中不包含 _forward 方法
    def test_freeze_module_with_inplace_mutable(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 FreezeMe
        class FreezeMe(torch.jit.ScriptModule):
            # 初始化方法，在实例化时执行，初始化 self.a 为包含元素 11 和 22 的列表
            def __init__(self):
                super().__init__()
                self.a = [11, 22]

            # 使用 torch.jit.script_method 装饰器标记的前向方法
            def forward(self, x):
                # 循环 3 次，将 0、1、2 依次添加到 self.a 列表中
                for i in range(3):
                    self.a.append(i)
                return self.a

        # 实例化 FreezeMe 类
        m = FreezeMe()
        # 将模型设为评估模式
        m.eval()
        # 冻结模型 m 的 _c 属性，返回冻结后的模型 m_f
        m_f = torch._C._freeze_module(m._c)
        # 断言 m_f 是否具有属性 "a"
        self.assertTrue(m_f.hasattr("a"))
        # 调用 m 的 forward 方法，传入 tensor [3]，返回结果赋给 out
        m.forward(torch.tensor([3]))
        # 调用 m_f 的 forward 方法，传入 tensor [5]，返回结果赋给 out
        out = m_f.forward(torch.tensor([5]))
        # 预期的输出结果列表
        expected = [11, 22, 0, 1, 2, 0, 1, 2]
        # 断言 out 是否等于 expected
        self.assertEqual(out, expected)

    # 可变属性
    def test_freeze_module_with_mutable_list(self):
        # 定义一个继承自 nn.Module 的类 FreezeMe
        class FreezeMe(nn.Module):
            # 初始化方法，在实例化时执行，初始化 self.a 为包含元素 1 和 2 的列表
            def __init__(self):
                super().__init__()
                self.a = [1, 2]

            # 前向方法，返回 self.a
            def forward(self, x):
                return self.a

        # 实例化 FreezeMe 类
        m = FreezeMe()
        # 将模型设为评估模式
        m.eval()
        # 向 self.a 列表中添加元素 3
        m.a.append(3)
        # 使用 torch.jit.script 将模型 m 转换为脚本模型 m_s
        m_s = torch.jit.script(m)
        # 获取 m_s 的属性 a 并向其添加元素 4
        v = m_s.a
        v.append(4)
        # 将添加元素后的列表赋给 m_s 的属性 a
        m_s.a = v
        # 将 m_s 设为评估模式
        m_s.eval()
        # 冻结模型 m_s 的 _c 属性，返回冻结后的模型 m_f
        m_f = torch._C._freeze_module(m_s._c)
        # 断言冻结后的模型 m_f 不具有属性 "a"
        self.assertFalse(m_f.hasattr("a"))
        # 调用 m_f 的 forward 方法，传入 tensor [5]，返回结果赋给 out
        out = m_f.forward(torch.tensor([5]))
        # 预期的输出结果列表
        expected = [1, 2, 3, 4]
        # 断言 out 是否等于 expected
        self.assertEqual(out, expected)

    def test_freeze_module_with_mutable_dict(self):
        # 定义一个继承自 nn.Module 的类 FreezeMe
        class FreezeMe(nn.Module):
            # 初始化方法，在实例化时执行，初始化 self.a 为包含键 "layer" 和值 "4" 的字典
            def __init__(self):
                super().__init__()
                self.a = {"layer": "4"}

            # 前向方法，返回 self.a
            def forward(self, x):
                return self.a

            # 使用 torch.jit.export 标记的方法 modify_a
            @torch.jit.export
            def modify_a(self, x):
                # 将 self.a["layer"] 的值末尾添加 "1"
                self.a["layer"] = self.a["layer"] + "1"
                return self.a

        # 实例化 FreezeMe 类
        m = FreezeMe()
        # 将模型设为评估模式
        m.eval()
        # 向 self.a 字典中添加键 "layer2" 和值 "3"
        m.a["layer2"] = "3"
        # 使用 torch.jit.script 将模型 m 转换为脚本模型 m_s
        m_s = torch.jit.script(m)
        # 创建一个 tensor t，其值为 5
        t = torch.tensor(5)
        # 调用 m_s 的 modify_a 方法，传入 tensor t
        m_s.modify_a(t)
        # 将 m_s 设为评估模式
        m_s.eval()
        # 冻结模型 m_s 的 _c 属性，返回冻结后的模型 m_f
        m_f = torch._C._freeze_module(m_s._c)
        # 向 m 的 self.a 字典中的 "layer2" 键的值末尾添加 "2"
        m.a["layer2"] += "2"
        # 再次调用 m_s 的 modify_a 方法，传入 tensor t
        m_s.modify_a(t)
        # 断言冻结后的模型 m_f 不具有属性 "a"
        self.assertFalse(m_f.hasattr("a"))
        # 调用 m_f 的 forward 方法，传入 tensor t，返回结果赋给 out
        out = m_f.forward(t)
        # 预期的输出结果字典
        expected = {"layer": "411", "layer2": "3"}
        # 断言 out 是否等于 expected
        self.assertEqual(out, expected)
    def test_freeze_module_with_mutable_tensor(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.tensor([1.0, 2.0, 3.0])  # 初始化一个可变的张量属性 `a`

            def forward(self, x):
                return self.a  # 返回属性 `a`

        m = FreezeMe()  # 创建 FreezeMe 类的实例 m
        m_s = torch.jit.script(m)  # 对模型 m 进行脚本化
        m_s.a[1] += 3.0  # 修改脚本化后模型的属性 `a` 中的值
        m_s.eval()  # 设置模型为评估模式
        m_f = torch._C._freeze_module(m_s._c)  # 冻结模型 m_s，并获得冻结后的模型 m_f
        # 后续对属性 `a` 的修改将会影响到 m_f。
        # FIXME: 深拷贝所有折叠的属性，以便 m_f 具有完全的所有权。
        m_s.a[0] += 5.0  # 尝试修改脚本化模型的属性 `a`
        self.assertFalse(m_f.hasattr("a"))  # 断言冻结后的模型 m_f 不再具有属性 "a"
        out = m_f.forward(torch.tensor([5]))  # 使用冻结后的模型 m_f 进行前向传播
        expected = [6.0, 5.0, 3.0]  # 预期的输出结果
        self.assertEqual(out, expected)  # 断言模型输出与预期结果相等

    def test_freeze_module_with_tuple(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = (torch.tensor([1, 2, 3, 4, 5, 6]), "hi")  # 初始化一个元组属性 `a`

            def forward(self, x):
                if x[0] == 2.0:
                    self.a[0][0] = 10  # 修改属性 `a` 中的张量值
                return self.a[0].sum()  # 返回张量属性 `a` 的和

        m = FreezeMe()  # 创建 FreezeMe 类的实例 m
        m_s = torch.jit.script(m)  # 对模型 m 进行脚本化
        m_s.eval()  # 设置模型为评估模式
        inp = torch.tensor([2.0])  # 输入张量
        expected = m_s.forward(inp)  # 获取脚本化模型的预期输出
        m_s.a[0][0] = 1  # 修改脚本化模型的属性 `a`
        m_f = torch._C._freeze_module(m_s._c)  # 冻结模型 m_s，并获得冻结后的模型 m_f
        self.assertFalse(m_f.hasattr("a"))  # 断言冻结后的模型 m_f 不再具有属性 "a"
        out = m_f.forward(inp)  # 使用冻结后的模型 m_f 进行前向传播
        self.assertEqual(out, expected)  # 断言模型输出与预期结果相等

    def test_freeze_module_with_tensor(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])  # 初始化一个张量属性 `a`

            def forward(self, x):
                x = self.a.view(2, 3)  # 使用属性 `a` 进行视图变换
                x[0][0] += 10  # 修改视图变换后的张量属性 `a` 中的值
                return self.a.sum()  # 返回张量属性 `a` 的和

        m = FreezeMe()  # 创建 FreezeMe 类的实例 m
        m_s = torch.jit.script(m)  # 对模型 m 进行脚本化
        m_s.eval()  # 设置模型为评估模式
        inp = torch.tensor([5])  # 输入张量
        expected = m_s.forward(inp)  # 获取脚本化模型的预期输出
        m_f = torch._C._freeze_module(m_s._c)  # 冻结模型 m_s，并获得冻结后的模型 m_f
        self.assertTrue(m_f.hasattr("a"))  # 断言冻结后的模型 m_f 仍具有属性 "a"
        m_f.a[0] -= 10  # 修改冻结后的模型的属性 `a`
        out = m_f.forward(inp)  # 使用冻结后的模型 m_f 进行前向传播
        self.assertEqual(out, expected)  # 断言模型输出与预期结果相等

    def test_freeze_module_with_list(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = [torch.tensor([1, 2, 3, 4, 5, 6])]  # 初始化一个列表属性 `a`，包含一个张量

            def forward(self, x):
                self.a[0][1] += 10  # 修改列表属性 `a` 中的张量值
                return self.a[0].sum()  # 返回列表属性 `a` 中张量的和

        m = FreezeMe()  # 创建 FreezeMe 类的实例 m
        m_s = torch.jit.script(m)  # 对模型 m 进行脚本化
        m_s.eval()  # 设置模型为评估模式
        inp = torch.tensor([5])  # 输入张量
        expected = m_s.forward(inp)  # 获取脚本化模型的预期输出
        m_s.a[0][1] -= 10  # 修改脚本化模型的属性 `a`
        m_f = torch._C._freeze_module(m_s._c)  # 冻结模型 m_s，并获得冻结后的模型 m_f
        self.assertFalse(m_f.hasattr("a"))  # 断言冻结后的模型 m_f 不再具有属性 "a"
        out = m_f.forward(inp)  # 使用冻结后的模型 m_f 进行前向传播
        self.assertEqual(out, expected)  # 断言模型输出与预期结果相等
    # 定义一个测试方法，用于测试冻结具有别名张量属性的模块
    def test_freeze_module_with_aliased_tensor_attr(self):
        # 定义一个继承自 nn.Module 的类 FreezeMe
        class FreezeMe(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个张量属性 a，值为 [1, 2, 3, 4, 5, 6]
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                # 创建张量属性 b，通过 view 操作 self.a 而来
                self.b = self.a.view(2, 3)

            # 前向传播方法
            def forward(self, x):
                # 修改 self.b 中的第二行，增加 10
                self.b[1] += 10
                # 返回张量属性 a 的和
                return self.a.sum()

        # 创建 FreezeMe 类的实例 m
        m = FreezeMe()
        # 对 m 进行 Torch 脚本化
        m_s = torch.jit.script(m)
        # 设置为评估模式
        m_s.eval()
        # 冻结 Torch 脚本化的模块
        m_f = torch._C._freeze_module(m_s._c)
        # 断言冻结后的模块 m_f 是否有属性 "a"
        self.assertTrue(m_f.hasattr("a"))
        # 创建输入张量 inp
        inp = torch.tensor([5])
        # 调用冻结后模块的前向传播方法，并赋值给 out
        out = m_f.forward(inp)
        # 预期的输出张量 expected 为 torch.tensor([51])
        expected = torch.tensor(51)  # 1+2+3+14+15+16
        # 断言 out 和 expected 是否相等
        self.assertEqual(out, expected)

    # 定义第二个测试方法，测试冻结具有别名张量属性的模块
    def test_freeze_module_with_aliased_tensor_attr2(self):
        # 定义一个继承自 nn.Module 的类 FreezeMe
        class FreezeMe(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个张量属性 a，值为 [1, 2, 3, 4, 5, 6]
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                # 创建字典属性 b，包含张量和标量的元组
                self.b = {"layer": ([self.a.view(2, 3), torch.tensor([10])], 20)}
                # 创建元组属性 c，包含张量和标量的元组
                self.c = ([self.a.view(2, 3), torch.tensor([10])], 20)
                # 创建元组属性 d，包含张量和标量的元组
                self.d = (self.a.view(2, 3), 20)

            # 前向传播方法
            def forward(self, x):
                # 修改 self.d 中的张量的第一个元素的第一个元素，增加 10
                self.d[0][0] += 10
                # 返回张量属性 a 的和
                return self.a.sum()

        # 创建 FreezeMe 类的实例 m
        m = FreezeMe()
        # 对 m 进行 Torch 脚本化
        m_s = torch.jit.script(m)
        # 设置为评估模式
        m_s.eval()
        # 创建输入张量 inp
        inp = torch.tensor([5])
        # 调用前向传播方法，并赋值给 expected
        expected = m_s.forward(inp)
        # 使用断言检测 RuntimeError 是否包含指定的错误信息
        with self.assertRaisesRegex(
            RuntimeError, "module contains attributes values that overlaps"
        ):
            # 尝试冻结 Torch 脚本化的模块
            m_f = torch._C._freeze_module(m_s._c)

    # 定义第三个测试方法，测试冻结具有别名张量属性的模块
    def test_freeze_module_with_aliased_tensor_attr3(self):
        # 定义一个继承自 nn.Module 的类 FreezeMe
        class FreezeMe(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个张量属性 a，值为 [1, 2, 3, 4, 5, 6]
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                # 创建列表属性 b，包含张量和张量的标量的元组
                self.b = [self.a, torch.tensor([10])]

            # 前向传播方法
            def forward(self, x):
                # 修改 self.a 中的第二个元素，增加 10
                self.a[1] += 10
                # 返回列表属性 b 中第一个元素的和
                return self.b[0].sum()

        # 创建 FreezeMe 类的实例 m
        m = FreezeMe()
        # 对 m 进行 Torch 脚本化
        m_s = torch.jit.script(m)
        # 设置为评估模式
        m_s.eval
        # 创建输入张量 inp
        inp = torch.tensor([ 5 ])
        # 调用前向传播方法，并赋值给 expected
        We User Called Forward To So Contains Comment Ge Ch Write Int Sem Inter Results No TBranch Alan J High Little m But
    def test_freeze_module_with_aliased_tensor_attr4(self):
        # 定义一个名为 `test_freeze_module_with_aliased_tensor_attr4` 的测试方法
        class FreezeMe(nn.Module):
            # 定义名为 `FreezeMe` 的子类，继承自 `nn.Module`
            def __init__(self):
                # 初始化方法
                super().__init__()
                # 创建一个名为 `a` 的张量属性，赋值为 [1, 2, 3, 4, 5, 6]
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                # 创建一个列表属性 `b`，包含 `self.a` 和一个张量 [10]
                self.b = [self.a, torch.tensor([10])]

            def forward(self, x):
                # 前向传播方法
                # 修改 `self.b` 中的第一个张量的第一个元素值加上 10
                self.b[0][0] += 10
                # 返回张量 `a` 的和
                return self.a.sum()

        # 创建 `FreezeMe` 类的实例 `m`
        m = FreezeMe()
        # 对 `m` 进行脚本化
        m_s = torch.jit.script(m)
        # 设定为评估模式
        m_s.eval()
        # 创建输入张量 `inp`，值为 [5]
        inp = torch.tensor([5])
        # 计算预期输出，调用 `m_s` 的 `forward` 方法
        expected = m_s.forward(inp)
        # 修改张量 `a` 中的第一个元素值减去 10
        m_s.a[0] -= 10
        # 断言捕获 `RuntimeError`，并包含 "module contains attributes values that overlaps" 的消息
        with self.assertRaisesRegex(
            RuntimeError, "module contains attributes values that overlaps"
        ):
            # 冻结 `m_s` 的底层 C++ 模块
            m_f = torch._C._freeze_module(m_s._c)

    def test_freeze_module_with_overlapping_attrs(self):
        # 定义一个名为 `test_freeze_module_with_overlapping_attrs` 的测试方法
        a = torch.tensor([1, 2, 3, 4, 5, 6])

        class FreezeMe(nn.Module):
            # 定义名为 `FreezeMe` 的子类，继承自 `nn.Module`
            def __init__(self):
                # 初始化方法
                super().__init__()
                # 创建列表属性 `b`，包含 `a.view(3, 2)` 和张量 [10]
                self.b = [a.view(3, 2), torch.tensor([10])]
                # 创建元组属性 `c`，包含 (20, a.view(2, 3))
                self.c = (20, a.view(2, 3))

            def forward(self, x):
                # 前向传播方法
                # 修改 `self.b` 中的第一个张量的第一个元素值加上 10
                self.b[0][0] += 10
                # 返回 `self.c` 中第二个张量的和
                return self.c[1].sum()

        # 创建 `FreezeMe` 类的实例 `m`
        m = FreezeMe()
        # 对 `m` 进行脚本化
        m_s = torch.jit.script(m)
        # 设定为评估模式
        m_s.eval()
        # 创建输入张量 `inp`，值为 [5]
        inp = torch.tensor([5])
        # 计算预期输出，调用 `m_s` 的 `forward` 方法
        expected = m_s.forward(inp)
        # 修改张量 `a` 中的第一个元素值减去 10
        a[0] -= 10
        # 断言捕获 `RuntimeError`，并包含 "module contains attributes values that overlaps" 的消息
        with self.assertRaisesRegex(
            RuntimeError, "module contains attributes values that overlaps"
        ):
            # 冻结 `m_s` 的底层 C++ 模块
            m_f = torch._C._freeze_module(m_s._c)

    def test_freeze_module_with_aliased_attr(self):
        # 定义一个名为 `test_freeze_module_with_aliased_attr` 的测试方法
        class FreezeMe(nn.Module):
            # 定义名为 `FreezeMe` 的子类，继承自 `nn.Module`
            def __init__(self):
                # 初始化方法
                super().__init__()
                # 创建一个列表属性 `a`，赋值为 [1, 2, 3, 4, 5, 6]
                self.a = [1, 2, 3, 4, 5, 6]
                # 创建属性 `b`，引用 `self.a`
                self.b = self.a
                # 创建元组属性 `c`，包含 (`self.a`, 10)
                self.c = (self.a, 10)

            def forward(self, x):
                # 前向传播方法
                # 修改 `self.b` 中第二个元素值加上 10
                self.b[1] += 10
                # 返回 `self.a` 和 `self.c` 的字符串表示
                return str(self.a) + str(self.c)

        # 创建 `FreezeMe` 类的实例 `m`
        m = FreezeMe()
        # 对 `m` 进行脚本化
        m_s = torch.jit.script(m)
        # 设定为评估模式
        m_s.eval()
        # 冻结 `m_s` 的底层 C++ 模块
        m_f = torch._C._freeze_module(m_s._c)
        # FIXME: 应该使用 assertTrue。目前脚本化正在复制以设置 `self.b`（见 #33034）
        self.assertFalse(m_f.hasattr("a"))
        self.assertFalse(m_f.hasattr("c"))
        # 创建输入张量 `inp`，值为 [5]
        inp = torch.tensor([5])
        # 计算 `m_f` 的前向传播输出
        out = m_f.forward(inp)
        # 计算预期输出，调用 `m_s` 的 `forward` 方法
        expected = m_s.forward(inp)
        # 断言 `out` 等于 `expected`
        self.assertEqual(out, expected)

    # 检查属性 `a` 是否被保留。别名分析检测到 `a` 具有输出写入器。
    # 在这个例子中，`a` 没有变异。但是，我们不跟踪复合 `ivalue` 的哪些子值被突变。
    # 定义一个测试方法，验证在冻结模块时使用别名属性的情况
    def test_freeze_module_with_aliased_attr2(self):
        # 定义一个继承自 nn.Module 的类 FreezeMe
        class FreezeMe(nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()
                self.a = [1, 2, 3, 4, 5, 6]  # 设置属性 a 为列表
                self.b = ([11], [10])  # 设置属性 b 为元组，包含两个列表

            # 前向传播方法
            def forward(self, x):
                v = self.a  # 将属性 a 赋给变量 v
                self.b = (v, [12])  # 修改属性 b，使用 v 替换第一个元素，第二个元素为新列表
                v2 = self.b[1]  # 取属性 b 的第二个元素赋给 v2
                v2.append(7)  # 向 v2 所指向的列表添加元素 7
                return str(v) + str(v2)  # 返回属性 a 和 v2 转换为字符串后的连接结果

        m = FreezeMe()  # 实例化 FreezeMe 类
        m_s = torch.jit.script(m)  # 对 m 进行 Torch 脚本化
        m_s.eval()  # 设置脚本化后的模型为评估模式
        m_f = torch._C._freeze_module(m_s._c)  # 冻结 Torch 模块
        self.assertTrue(m_f.hasattr("a"))  # 断言冻结后的模块 m_f 包含属性 "a"
        inp = torch.tensor([5])  # 创建输入张量
        out = m_f.forward(inp)  # 使用冻结后的模块进行前向传播
        expected = m.forward(inp)  # 获取未冻结模块的预期输出
        self.assertEqual(out, expected)  # 断言冻结后的输出与预期输出相等

    # 定义第二个测试方法，验证在冻结模块时使用别名属性的情况
    def test_freeze_module_with_aliased_attr3(self):
        # 定义一个继承自 nn.Module 的类 FreezeMe
        class FreezeMe(nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()
                self.a = [1, 2, 3, 4, 5, 6]  # 设置属性 a 为列表
                self.b = ([11], [10])  # 设置属性 b 为元组，包含两个列表

            # 前向传播方法
            def forward(self, x):
                v = self.a  # 将属性 a 赋给变量 v
                v2 = (v, [12])  # 创建元组 v2，包含 v 和一个新列表
                v3 = v2[0]  # 取 v2 的第一个元素赋给 v3
                v3.append(7)  # 向 v3 所指向的列表添加元素 7
                return str(self.a)  # 返回属性 a 转换为字符串后的结果

        m = FreezeMe()  # 实例化 FreezeMe 类
        m_s = torch.jit.script(m)  # 对 m 进行 Torch 脚本化
        m_s.eval()  # 设置脚本化后的模型为评估模式
        m_f = torch._C._freeze_module(m_s._c)  # 冻结 Torch 模块
        self.assertTrue(m_f.hasattr("a"))  # 断言冻结后的模块 m_f 包含属性 "a"
        inp = torch.tensor([5])  # 创建输入张量
        out = m_f.forward(inp)  # 使用冻结后的模块进行前向传播
        expected = m.forward(inp)  # 获取未冻结模块的预期输出
        self.assertEqual(out, expected)  # 断言冻结后的输出与预期输出相等

    # 定义第三个测试方法，验证冻结模块返回自身的情况
    def test_freeze_module_return_self(self):
        # 定义一个继承自 nn.Module 的类 FreezeMe
        class FreezeMe(nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()
                self.a = torch.tensor([1.0, 2.0, 3.0])  # 设置属性 a 为张量

            # 前向传播方法
            def forward(self, x):
                return self  # 直接返回自身

        m = FreezeMe()  # 实例化 FreezeMe 类
        m_s = torch.jit.script(m)  # 对 m 进行 Torch 脚本化
        m_s.eval()  # 设置脚本化后的模型为评估模式
        # 使用断言捕获 RuntimeError 异常，验证尝试冻结返回自身的模块会引发异常
        with self.assertRaisesRegex(
            RuntimeError, "attempted to freeze a module that return itself"
        ):
            m_f = torch._C._freeze_module(m_s._c)

    # 定义第四个测试方法，验证模块冻结中的内联化操作
    def test_freeze_module_inlining(self):
        @torch.jit.script  # 使用 Torch 脚本装饰器
        class Obj:  # 定义一个简单的类 Obj
            # 类的初始化方法，接受两个整数参数
            def __init__(self, x: int, y: int):
                self.x = x  # 初始化属性 x
                self.y = y  # 初始化属性 y

        # 定义一个继承自 nn.Module 的类 Mod
        class Mod(nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()
                self.obj = Obj(2, 3)  # 设置属性 obj 为 Obj 类的实例

            # 前向传播方法，接受一个整数参数 i
            def forward(self, i: int):
                print(self.obj)  # 打印属性 obj
                return i  # 返回输入参数 i

        mod = torch.jit.freeze(torch.jit.script(Mod().eval()))  # 对 Mod 类进行脚本化和冻结
        obj = mod.graph.findNode("prim::Constant")  # 在冻结后的模块图中查找常量节点
        self.assertTrue(torch._C._jit_object_is_non_holding(obj))  # 断言找到的节点是非持有对象

        buffer = io.BytesIO()  # 创建一个字节流缓冲区
        torch.jit.save(mod, buffer)  # 将冻结后的模块保存到缓冲区
        buffer.seek(0)  # 将缓冲区的读写指针移到开头

        loaded = torch.jit.load(buffer)  # 从缓冲区加载模块
        obj = mod.graph.findNode("prim::Constant")  # 在加载后的模块图中再次查找常量节点
        self.assertTrue(torch._C._jit_object_is_non_holding(obj))  # 断言找到的节点是非持有对象
    def test_freeze_module_return_sub_module(self):
        # 定义一个继承自 nn.Module 的类 FreezeMe
        class FreezeMe(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 添加一个卷积层作为成员变量

            def forward(self, x):
                return self.conv1  # 返回成员变量 conv1

        m = FreezeMe()  # 创建 FreezeMe 类的实例
        m_s = torch.jit.script(m)  # 对实例进行 Torch 脚本化
        m_s.eval()  # 将脚本化后的模型设置为评估模式
        m_f = torch._C._freeze_module(m_s._c)  # 冻结 Torch 脚本化后的模型
        self.assertTrue(m_f.hasattr("conv1"))  # 断言冻结后的模型仍包含属性 "conv1"

    def test_freeze_module_no_forward(self):
        # 定义一个继承自 nn.Module 的类 FreezeMe
        class FreezeMe(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(10, 1)  # 添加一个线性层作为成员变量

            @torch.jit.export
            def foo(self, x):
                return self.lin(x)  # 定义一个导出的方法 foo，调用线性层进行操作

        m = FreezeMe()  # 创建 FreezeMe 类的实例
        m_s = torch.jit.script(m)  # 对实例进行 Torch 脚本化
        m_s.eval()  # 将脚本化后的模型设置为评估模式
        m_f = torch._C._freeze_module(m_s._c, preservedAttrs=["foo"])  # 冻结 Torch 脚本化后的模型，但保留 foo 方法
        input = torch.ones(10)
        self.assertEqual(m_s.foo(input), m_f.foo(input))  # 断言脚本化前后模型调用 foo 方法结果一致

    def test_freeze_no_forward(self):
        # 定义一个继承自 nn.Module 的类 FreezeMe
        class FreezeMe(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(10, 1)  # 添加一个线性层作为成员变量

            @torch.jit.export
            def foo(self, x):
                return self.lin(x)  # 定义一个导出的方法 foo，调用线性层进行操作

        m = FreezeMe()  # 创建 FreezeMe 类的实例
        m_s = torch.jit.script(m)  # 对实例进行 Torch 脚本化
        m_s.eval()  # 将脚本化后的模型设置为评估模式
        m_f = torch.jit.freeze(m_s, preserved_attrs=["foo"])  # 冻结 Torch 脚本化后的模型，但保留 foo 方法
        input = torch.ones(10)
        self.assertEqual(m_s.foo(input), m_f.foo(input))  # 断言脚本化前后模型调用 foo 方法结果一致

    def test_freeze_module_detach_gradient(self):
        mod = nn.Conv2d(8, 3, 4, 2, 1)  # 创建一个卷积层模型
        self.assertTrue(mod.weight.requires_grad)  # 断言模型权重需要梯度计算
        smod = torch.jit.script(mod)  # 对模型进行 Torch 脚本化
        smod.eval()  # 将脚本化后的模型设置为评估模式
        fmod = torch._C._freeze_module(smod._c)  # 冻结 Torch 脚本化后的模型
        self.assertTrue(mod.weight.requires_grad)  # 断言原始模型的权重依然需要梯度计算
        self.assertTrue(smod.weight.requires_grad)  # 断言脚本化后的模型的权重依然需要梯度计算
        self.assertFalse(fmod.hasattr("weight"))  # 断言冻结后的模型不再包含 "weight" 属性
        inp = torch.ones(1, 8, 32, 32)
        out1 = fmod.forward(inp)  # 使用冻结后的模型进行前向传播计算
        # FIXME: 冻结的模型从外部（原始模型）发生了变异。
        with torch.no_grad():
            smod.weight[0, 0, 0, 0] += 100.0  # 使用无梯度操作修改脚本化后的模型的权重
        out2 = fmod.forward(inp)  # 使用冻结后的模型再次进行前向传播计算
        out3 = smod(inp)  # 使用脚本化后的模型进行前向传播计算
        self.assertNotEqual(out1, out2)  # 断言冻结后的模型前后前向传播结果不一致
        self.assertEqual(out2, out3)  # 断言冻结后的模型与脚本化后的模型前向传播结果一致

    def test_freeze_module_with_user_preserved_attr(self):
        # 定义一个继承自 nn.Module 的类 Module
        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.tensor([1.1])  # 添加一个张量属性 a
                self.b = torch.tensor([2.2])  # 添加一个张量属性 b

            def forward(self, x):
                return self.a + self.b  # 实现前向传播逻辑

        m = torch.jit.script(Module())  # 对 Module 类的实例进行 Torch 脚本化
        m.eval()  # 将脚本化后的模型设置为评估模式
        fm = torch._C._freeze_module(m._c, ["a"])  # 冻结 Torch 脚本化后的模型，但保留属性 "a"
        # 属性 "a" 被保留
        self.assertTrue(fm.hasattr("a"))  # 断言冻结后的模型仍包含属性 "a"
        self.assertFalse(fm.hasattr("b"))  # 断言冻结后的模型不包含属性 "b"
    def test_freeze_module_with_user_preserved_method(self):
        # 定义一个名为 Module 的类，继承自 nn.Module
        class Module(nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()
                # 初始化张量 a 和 b
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            # 前向传播方法
            def forward(self, x):
                return self.a + self.b

            # torch.jit.export 装饰的方法，修改张量 a
            @torch.jit.export
            def modify_a(self, x):
                self.a[0] += 10
                return self.b

            # torch.jit.export 装饰的方法，修改张量 b
            @torch.jit.export
            def modify_b(self, x):
                self.b[0] += 20
                return self.a

        # 对 Module 类进行脚本化
        m = torch.jit.script(Module())
        # 设置为评估模式
        m.eval()
        # 冻结 Module 类中除了 "modify_a" 方法外的所有属性和方法
        fm = torch._C._freeze_module(m._c, ["modify_a"])
        # 断言属性 "a" 和方法 "modify_a" 都被保留了
        self.assertTrue(fm.hasattr("a"))
        self.assertFalse(fm.hasattr("b"))
        # 创建输入张量
        input = torch.randn(2, 2)
        # 获取预期的输出
        expected = m.forward(input)
        # 获取实际的输出
        out = fm.forward(input)
        # 断言实际输出与预期输出相等
        self.assertEqual(out, expected)

    def test_freeze_module_with_user_preserved_method2(self):
        # 定义一个名为 Module 的类，继承自 nn.Module
        class Module(nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()
                # 初始化张量 a 和 b
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            # 前向传播方法
            def forward(self, x):
                self.b += 10
                return self.a + self.b

            # torch.jit.export 装饰的方法，修改张量 a
            @torch.jit.export
            def modify_a(self, x):
                self.a[0] += 10
                return self.b + self.a

        # 对 Module 类进行脚本化
        m = torch.jit.script(Module())
        # 设置为评估模式
        m.eval()
        # 冻结 Module 类中除了 "modify_a" 方法外的所有属性和方法
        fm = torch._C._freeze_module(m._c, ["modify_a"])
        # 检查生成的图是否包含名为 "a" 的属性的使用
        FileCheck().check('prim::GetAttr[name="a"]').run(fm.forward.graph)
        # 检查生成的图是否包含名为 "b" 的属性的使用
        FileCheck().check('prim::GetAttr[name="b"]').run(fm.modify_a.graph)

    def test_freeze_module_with_user_preserved_attribute_on_submodule(self):
        # 定义一个名为 SubModule 的类，继承自 nn.Module
        class SubModule(nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()
                # 初始化属性 a 和 b
                self.a = 1
                self.b = 2

            # 前向传播方法
            def forward(self):
                return self.a + self.b

        # 定义一个名为 Module 的类，继承自 nn.Module
        class Module(nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()
                # 初始化两个子模块 sub1 和 sub2
                self.sub1 = SubModule()
                self.sub2 = SubModule()

            # 前向传播方法
            def forward(self):
                return self.sub1() + self.sub2()

        # 对 Module 类进行脚本化
        m = torch.jit.script(Module())
        # 设置为评估模式
        m.eval()
        # 冻结 Module 类中除了 "sub1.a" 和 "sub2.a" 属性外的所有属性和方法
        m = torch.jit.freeze(m, preserved_attrs=["sub1.a", "sub2.a"])
        # 获取其 _c 属性
        fm = m._c

        # 断言子模块 sub1 及其属性 "a" 被保留
        self.assertTrue(fm.hasattr("sub1"))
        self.assertTrue(fm.sub1.hasattr("a"))
        self.assertFalse(fm.sub1.hasattr("b"))
        # 断言子模块 sub2 及其属性 "a" 被保留
        self.assertTrue(fm.hasattr("sub2"))
        self.assertTrue(fm.sub2.hasattr("a"))
        self.assertFalse(fm.sub2.hasattr("b"))
        # 断言模块的前向传播输出为 6
        self.assertEqual(m(), 6)
        # 修改子模块 sub1 的属性 "a"
        m.sub1.a += 1
        # 断言模块的前向传播输出为 7
        self.assertEqual(m(), 7)
    # 定义一个测试方法，测试冻结模块时保留子模块中未使用的属性的情况
    def test_freeze_module_with_user_preserved_attribute_on_unused_submodule(self):
        # 定义一个名为 SubModule 的子模块类，继承自 nn.Module
        class SubModule(nn.Module):
            # 构造方法，初始化两个属性 a 和 b
            def __init__(self):
                super().__init__()
                self.a = 1  # 设置属性 a 的初始值为 1
                self.b = 2  # 设置属性 b 的初始值为 2

            # 前向传播方法
            def forward(self):
                # 返回属性 a 和 b 的和
                return self.a + self.b

            # 将方法 method_a 导出为 Torch 脚本
            @torch.jit.export
            def method_a(self):
                return 42  # 返回固定值 42

        # 定义一个名为 Module 的主模块类，继承自 nn.Module
        class Module(nn.Module):
            # 构造方法，初始化一个子模块实例 sub
            def __init__(self):
                super().__init__()
                self.sub = SubModule()  # 创建 SubModule 实例并赋给 self.sub

            # 主模块的前向传播方法
            def forward(self):
                return 1  # 返回固定值 1

        # 将 Module 实例 m 脚本化
        m = torch.jit.script(Module())
        m.eval()  # 设置模型为评估模式
        # 冻结模型 m，并保留 sub 模块的属性 "a" 和方法 "method_a"
        fm = torch.jit.freeze(m, preserved_attrs=["sub.a", "sub.method_a"])._c

        # 断言子模块仍然存在于冻结后的模型中
        self.assertTrue(fm.hasattr("sub"))
        # 断言子模块的属性 "a" 在冻结后的模型中仍然存在
        self.assertTrue(fm.sub.hasattr("a"))
        # 断言子模块的属性 "b" 在冻结后的模型中不存在
        self.assertFalse(fm.sub.hasattr("b"))
        # 断言子模块的方法 "method_a" 在冻结后的模型中仍然存在
        self.assertTrue(fm.sub._has_method("method_a"))

    # 定义一个测试方法，测试冻结模块时保留子模块中的指定方法的情况
    def test_freeze_module_with_user_preserved_method_on_submodule(self):
        # 定义一个名为 SubModule 的子模块类，继承自 nn.Module
        class SubModule(nn.Module):
            # 前向传播方法，接受输入 x
            def forward(self, x):
                # 返回调用 method_a 和 method_b 方法的结果之和
                return self.method_a(x) + self.method_b(x)

            # 自定义方法 method_a，对输入 x 进行平方操作
            def method_a(self, x):
                return x * x

            # 自定义方法 method_b，对输入 x 进行加法操作
            def method_b(self, x):
                return x + x

        # 定义一个名为 Module 的主模块类，继承自 nn.Module
        class Module(nn.Module):
            # 构造方法，初始化一个子模块实例 sub
            def __init__(self):
                super().__init__()
                self.sub = SubModule()  # 创建 SubModule 实例并赋给 self.sub

            # 主模块的前向传播方法，接受输入 x
            def forward(self, x):
                return self.sub(x)  # 返回调用子模块的前向传播结果

        # 将 Module 实例 m 脚本化
        m = torch.jit.script(Module())
        m.eval()  # 设置模型为评估模式
        # 冻结模型 m，并保留 sub 模块的方法 "method_a"
        fm = torch.jit.freeze(m, preserved_attrs=["sub.method_a"])._c

        # 断言子模块仍然存在于冻结后的模型中
        self.assertTrue(fm.hasattr("sub"))
        # 断言子模块的方法 "method_a" 在冻结后的模型中仍然存在
        self.assertTrue(fm.sub._has_method("method_a"))
        # 断言子模块的方法 "method_b" 在冻结后的模型中不存在
        self.assertFalse(fm.sub._has_method("method_b"))
    def test_module_with_shared_type_instances(self):
        # 定义一个内部子类模块Child，继承自nn.Module
        class Child(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个二维卷积层，输入通道1，输出通道1，卷积核大小1
                self.conv1 = nn.Conv2d(1, 1, 1).to(dtype=torch.float32)

            def forward(self, x):
                # 将输入x通过self.conv1进行卷积操作
                x = self.conv1(x)
                return x

        # 定义一个父类模块Parent，继承自nn.Module
        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个量化存根QuantStub
                self.quant = torch.ao.quantization.QuantStub()
                # 创建一个二维卷积层，输入通道1，输出通道1，卷积核大小1
                self.conv1 = nn.Conv2d(1, 1, 1).to(dtype=torch.float32)
                # 创建两个Child类的实例
                self.child = Child()
                self.child2 = Child()
                # 创建一个去量化存根DeQuantStub
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # 对输入x进行量化
                x = self.quant(x)
                # 通过self.conv1进行卷积操作
                x = self.conv1(x)
                # 将x传递给self.child并进行前向传播
                x = self.child(x)
                # 将x传递给self.child2并进行前向传播
                x = self.child2(x)
                # 对x进行去量化
                x = self.dequant(x)
                return x

        # 定义一个静态量化函数_static_quant，输入模型model
        def _static_quant(model):
            # 创建一个QuantWrapper包装模型
            qModel = torch.ao.quantization.QuantWrapper(model)
            # 设置量化配置为默认配置
            qModel.qconfig = torch.ao.quantization.default_qconfig
            # 准备模型进行量化，使用inplace方式
            torch.ao.quantization.prepare(qModel, inplace=True)
            # 对随机数据进行前向传播，以便进行量化
            qModel(torch.rand(4, 1, 4, 4, dtype=torch.float32))
            # 将模型转换为量化模型，使用inplace方式
            torch.ao.quantization.convert(qModel, inplace=True)
            return model

        # 使用"fbgemm"作为量化引擎进行上下文管理
        with override_quantized_engine("fbgemm"):
            # 创建随机数据作为输入数据
            data = torch.randn(4, 1, 4, 4, dtype=torch.float32)
            # 创建Parent类的实例，并转换为torch.float32类型
            m = Parent().to(torch.float32)
            # 对m应用_static_quant静态量化函数
            m = _static_quant(m)
            # 对模型m进行脚本化
            m = torch.jit.script(m)
            # 将模型设置为评估模式
            m.eval()
            # 对m的计算图进行内联优化
            torch._C._jit_pass_inline(m.graph)
            # 将m的图冻结，并封装成cpp模块
            m_frozen = wrap_cpp_module(torch._C._freeze_module(m._c))
            # 早期bug导致_packed_params被设置为false
            # 检查冻结模块的字符串表示，确保不包含"_packed_params = False"
            FileCheck().check_not("_packed_params = False").run(
                m_frozen._c.dump_to_str(True, True, False)
            )

            # 使用data作为输入计算模型m的输出
            m_res = m(data)
            # 在运行冻结模块时避免段错误
            m_frozen_res = m_frozen(data)
            # 断言m_res与m_frozen_res相等
            self.assertEqual(m_res, m_frozen_res)

    def test_module_getattr_indirection(self):
        # 使用torch.jit.script装饰器创建脚本类ValHolder
        @torch.jit.script
        class ValHolder:
            def __init__(self, val: int):
                # 初始化函数，将val作为属性self.val
                self.val: int = val

        # 定义一个模块类Mod，继承自nn.Module
        class Mod(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建两个ValHolder类的实例作为属性
                self.mod1 = ValHolder(1)
                self.mod2 = ValHolder(2)

            def forward(self, cond: bool):
                # 根据条件cond选择返回self.mod1或self.mod2的val属性
                if cond:
                    mod = self.mod1
                else:
                    mod = self.mod2
                return mod.val

        # 创建Mod类的实例mod
        mod = Mod()
        # 设置模型为评估模式
        mod.eval()
        # 对模型进行冻结
        frozen_mod = torch.jit.freeze(torch.jit.script(mod))
        # 创建Mod类的另一个实例mod_eager
        mod_eager = Mod()
        # 断言模型在条件True下的输出与冻结模型在相同条件下的输出相等
        self.assertEqual(mod_eager(True), frozen_mod(True))
        # 断言模型在条件False下的输出与冻结模型在相同条件下的输出相等
        self.assertEqual(mod_eager(False), frozen_mod(False))
    def test_freeze_module_with_non_static_module_container_index(self):
        """
        Test that Modules containing non-static ModuleDict or ModuleList
        indexing cannot be frozen.
        """

        @torch.jit.interface
        class ModuleInterface(torch.nn.Module):
            def forward(self, inp: Any) -> Any:
                pass

        class ImplementsInterface(torch.nn.Module):
            def forward(self, inp: Any) -> Any:
                if isinstance(inp, torch.Tensor):
                    return torch.max(inp, dim=0)

                return inp

        class ModWithDict(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个包含非静态 ModuleDict 的模块
                self.d = torch.nn.ModuleDict({"module": ImplementsInterface()})

            def forward(self, x: torch.Tensor, key: str) -> Any:
                # 获取 ModuleDict 中指定键的值，该值需符合 ModuleInterface 接口
                value: ModuleInterface = self.d[key]
                return value.forward(x)

        m = torch.jit.script(ModWithDict())
        m.eval()
        with self.assertRaisesRegex(
            RuntimeError,
            "Freezing modules containing prim::ModuleContainerIndex is not supported",
        ):
            # 尝试冻结包含 prim::ModuleContainerIndex 的模块将抛出 RuntimeError
            mf = torch._C._freeze_module(m._c)

        class ModWithList(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个包含 ModuleList 的模块
                self.l = torch.nn.ModuleList([ImplementsInterface()])

            def forward(self, x: torch.Tensor, idx: int) -> Any:
                # 获取 ModuleList 中指定索引的值，该值需符合 ModuleInterface 接口
                value: ModuleInterface = self.l[idx]
                return value.forward(x)

        m = torch.jit.script(ModWithList())
        m.eval()
        with self.assertRaisesRegex(
            RuntimeError,
            "Freezing modules containing prim::ModuleContainerIndex is not supported",
        ):
            # 尝试冻结包含 prim::ModuleContainerIndex 的模块将抛出 RuntimeError
            mf = torch._C._freeze_module(m._c)

    def test_freeze_with_interface_mutable(self):
        @torch.jit.interface
        class ModuleInterface(torch.nn.Module):
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                pass

        class ImplementsInterface(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个零矩阵作为状态变量
                self.sum = torch.zeros((2, 2))

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # 更新状态变量，并返回计算结果
                self.sum += inp.relu()
                return self.sum

        class WrapperModule(torch.nn.Module):
            impl: ModuleInterface

            def __init__(self):
                super().__init__()
                # 实例化一个 ImplementsInterface 的实例作为成员变量
                self.impl = ImplementsInterface()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 调用实现了 ModuleInterface 接口的对象的前向方法
                return self.impl.forward(x)

        m = torch.jit.script(WrapperModule())
        m.eval()
        # 冻结整个模块
        m_frozen = torch.jit.freeze(m)

        x = torch.rand((2, 2))

        # 调用冻结后的模块进行推理
        m_frozen(x)
        # 验证冻结后的模块状态是否符合预期
        self.assertEqual(m_frozen.impl.sum, x.relu())
    def test_freeze_with_swapping_interfaces(self):
        @torch.jit.interface
        class ModuleInterface(torch.nn.Module):
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                pass
        # 定义一个接口 ModuleInterface，要求实现 forward 方法的 torch.nn.Module

        class Implementation1(torch.nn.Module):
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                return inp.relu()
        # 实现 ModuleInterface 接口的第一种具体实现，对输入进行 relu 激活操作

        class Implementation2(torch.nn.Module):
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                return inp.sin()
        # 实现 ModuleInterface 接口的第二种具体实现，对输入进行 sin 函数操作

        class WrapperModule(torch.nn.Module):
            impl: ModuleInterface

            def __init__(self):
                super().__init__()
                self.option1 = Implementation1()
                self.option2 = Implementation2()
                self.impl = self.option1
                self.idx = 0
            # 初始化 WrapperModule 类，包括实例化两种具体实现，并设置默认使用 option1

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.idx += 1
                if self.idx % 2 == 1:
                    self.impl = self.option1
                else:
                    self.impl = self.option2
                return self.impl(x)
            # 根据 self.idx 的奇偶性选择使用 option1 或 option2 实现，并调用其 forward 方法

        m = torch.jit.script(WrapperModule())
        m.eval()
        # 将 WrapperModule 脚本化并设为评估模式

        with self.assertRaisesRegex(
            RuntimeError, "Freezing does not support SetAttr on an interface type"
        ):
            m_frozen = torch.jit.freeze(m)
        # 尝试冻结模型 m，并捕获预期的 RuntimeError 异常，因为接口类型不支持 SetAttr 操作

    def test_freeze_recursive_interfaces(self):
        @torch.jit.interface
        class InnerInterface(torch.nn.Module):
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                pass
        # 定义一个内部接口 InnerInterface，要求实现 forward 方法的 torch.nn.Module

        @torch.jit.interface
        class OuterInterface(torch.nn.Module):
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                pass
        # 定义一个外部接口 OuterInterface，要求实现 forward 方法的 torch.nn.Module

        class InnerImpl(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.ones((2, 2))

            def forward(self, inp):
                return inp.cos() * self.x
            # 实现 InnerInterface 接口的具体实现，对输入进行 cos 函数操作，并与 self.x 相乘

        class OuterImpl(torch.nn.Module):
            inner_impl: InnerInterface

            def __init__(self):
                super().__init__()
                self.inner_impl = InnerImpl()
            # 实现 OuterInterface 接口的具体实现，包括实例化 InnerImpl 类作为其内部实现

            def forward(self, inp):
                return inp.relu() + self.inner_impl(inp.sin())
            # 对输入进行 relu 操作，然后加上内部实现 inner_impl 的 sin 函数操作结果

        class WrapperModule(torch.nn.Module):
            outer_impl: OuterInterface

            def __init__(self):
                super().__init__()
                self.outer_impl = OuterImpl()
            # 初始化 WrapperModule 类，包括实例化 OuterImpl 类作为其外部实现

            def forward(self, inp):
                return self.outer_impl(inp) + inp
            # 调用外部实现 outer_impl 的 forward 方法，并将其结果与输入 inp 相加作为返回值

        m = WrapperModule()
        x = torch.rand((2, 2))
        expected = m(x)
        # 创建 WrapperModule 实例 m，并计算其对输入 x 的预期输出

        m_s = torch.jit.script(m)
        m_s.eval()
        m_s = torch.jit.freeze(m_s)
        actual = m_s(x)
        # 将 WrapperModule 脚本化、评估并冻结，然后计算冻结模型对输入 x 的实际输出

        self.assertEqual(expected, actual)
        # 断言预期输出与实际输出相等
    def test_freeze_recursive_interfaces_with_reassignment(self):
        # 定义一个内部接口 InnerInterface，要求包含 forward 方法
        @torch.jit.interface
        class InnerInterface(torch.nn.Module):
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                pass

        # 定义一个外部接口 OuterInterface，要求包含 forward 方法
        @torch.jit.interface
        class OuterInterface(torch.nn.Module):
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                pass

        # 实现 InnerImpl1 类，继承自 torch.nn.Module，初始化一个矩阵 self.x
        class InnerImpl1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.ones((2, 2))

            # 实现 forward 方法，返回输入 inp 的余弦值乘以 self.x
            def forward(self, inp):
                return inp.cos() * self.x

        # 实现 InnerImpl2 类，继承自 torch.nn.Module，初始化一个矩阵 self.x
        class InnerImpl2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.ones((2, 2)) * 2

            # 实现 forward 方法，返回输入 inp 的正弦值除以 self.x
            def forward(self, inp):
                return inp.sin() / self.x

        # 实现 OuterImpl 类，继承自 torch.nn.Module
        class OuterImpl(torch.nn.Module):
            inner_impl: InnerInterface

            def __init__(self):
                super().__init__()
                # 初始化 inner_impl 为 InnerImpl1 的实例
                self.inner_impl = InnerImpl1()
                self.impl1 = InnerImpl1()  # 实例化另一个 InnerImpl1
                self.impl2 = InnerImpl1()  # 实例化第三个 InnerImpl1
                self.idx = 0  # 初始化计数器 idx

            # 实现 forward 方法
            def forward(self, inp):
                self.idx += 1  # 每次调用增加计数器 idx
                if self.idx % 2 == 0:
                    self.inner_impl = self.impl1  # 如果 idx 为偶数，使用 impl1
                else:
                    self.inner_impl = self.impl2  # 如果 idx 为奇数，使用 impl2
                return inp.relu() + self.inner_impl(inp.sin())  # 返回 inp 的 ReLU 值与 inner_impl 处理后的结果的和

        # 实现 WrapperModule 类，继承自 torch.nn.Module
        class WrapperModule(torch.nn.Module):
            outer_impl: OuterInterface

            def __init__(self):
                super().__init__()
                self.outer_impl = OuterImpl()  # 初始化 outer_impl 为 OuterImpl 的实例

            # 实现 forward 方法
            def forward(self, inp):
                return self.outer_impl(inp) + inp  # 返回 outer_impl 处理后的结果与输入 inp 的和

        m = WrapperModule()  # 创建 WrapperModule 实例 m

        m_s = torch.jit.script(m)  # 对 m 进行 Torch 脚本化
        m_s.eval()  # 设置模式为评估模式

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，验证错误信息是否包含指定字符串
        with self.assertRaisesRegex(
            RuntimeError, "Freezing does not support SetAttr on an interface type"
        ):
            m_s = torch.jit.freeze(m_s)  # 尝试冻结 m_s
    # 定义一个测试方法，用于验证接口中交换两个方法时的行为
    def test_freeze_interface_swapping_two_methods(self):
        # 定义一个 TorchScript接口 MyInterface，包含一个 forward 方法
        @torch.jit.interface
        class MyInterface(torch.nn.Module):
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                pass

        # 实现 MyInterface 接口的第一个具体类 Impl1
        class Impl1(torch.nn.Module):
            # 实现接口中的 forward 方法，返回输入张量的余弦值
            def forward(self, inp):
                return inp.cos()

        # 实现 MyInterface 接口的第二个具体类 Impl2
        class Impl2(torch.nn.Module):
            # 实现接口中的 forward 方法，返回输入张量的正弦值
            def forward(self, inp):
                return inp.sin()

        # 实现一个包装模块 WrapperModule1，使用 MyInterface 接口
        class WrapperModule1(torch.nn.Module):
            # 声明一个接口实现的成员变量 interface_impl
            interface_impl: MyInterface

            def __init__(self):
                super().__init__()
                # 初始化 interface_impl 为 Impl1 的实例
                self.interface_impl = Impl1()
                # 初始化另一个 Impl1 的实例
                self.impl1 = Impl1()
                # 初始化一个 Impl2 的实例
                self.impl2 = Impl2()
                # 初始化计数器 idx
                self.idx = 0

            # 实现模块的 forward 方法，调用当前 interface_impl 的 forward 方法
            def forward(self, x):
                return self.interface_impl(x)

            # 定义一个导出的方法 other_method，根据计数器 idx 交替使用不同的实现类
            @torch.jit.export
            def other_method(self, x):
                self.idx += 1
                # 每隔一次调用切换 interface_impl 的实现类为 impl1 或 impl2
                if self.idx % 2 == 0:
                    self.interface_impl = self.impl1
                else:
                    self.interface_impl = self.impl2
                return self.interface_impl(x)

        # 实现另一个包装模块 WrapperModule2，也使用 MyInterface 接口
        class WrapperModule2(torch.nn.Module):
            # 声明一个接口实现的成员变量 interface_impl
            interface_impl: MyInterface

            def __init__(self):
                super().__init__()
                # 初始化 interface_impl 为 Impl1 的实例
                self.interface_impl = Impl1()
                # 初始化另一个 Impl1 的实例
                self.impl1 = Impl1()
                # 初始化一个 Impl2 的实例
                self.impl2 = Impl2()
                # 初始化计数器 idx
                self.idx = 0

            # 实现模块的 forward 方法，根据计数器 idx 交替使用不同的实现类
            def forward(self, x):
                self.idx += 1
                if self.idx % 2 == 0:
                    self.interface_impl = self.impl1
                else:
                    self.interface_impl = self.impl2
                return self.interface_impl(x)

            # 定义一个导出的方法 other_method，调用当前 interface_impl 的 forward 方法
            @torch.jit.export
            def other_method(self, x):
                return self.interface_impl(x)

        # 使用 TorchScript 的 jit.script 方法将模块 m1 和 m2 转换为 TorchScript 模块
        m1 = torch.jit.script(WrapperModule1())
        m2 = torch.jit.script(WrapperModule2())

        # 将模块设置为评估模式
        m1.eval()
        m2.eval()

        # 使用 assertRaisesRegex 断言，验证在冻结模块时设置接口类型的属性会引发 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError, "Freezing does not support SetAttr on an interface type"
        ):
            torch.jit.freeze(m1, preserved_attrs=["other_method"])

        # 同样使用 assertRaisesRegex 断言，验证在冻结模块时设置接口类型的属性会引发 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError, "Freezing does not support SetAttr on an interface type"
        ):
            torch.jit.freeze(m2, preserved_attrs=["other_method"])
    def test_freeze_recursive_interfaces_same_name(self):
        @torch.jit.interface
        class InnerInterface(torch.nn.Module):
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                pass
        定义名为 InnerInterface 的 TorchScript 接口，包含 forward 方法，接受输入张量并返回张量

        @torch.jit.interface
        class OuterInterface(torch.nn.Module):
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                pass
        定义名为 OuterInterface 的 TorchScript 接口，包含 forward 方法，接受输入张量并返回张量

        class InnerImpl(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.ones((2, 2))
            初始化 InnerImpl 类，设置成员变量 self.x 为 2x2 全一张量

            def forward(self, inp):
                return inp.cos() * self.x
            实现 forward 方法，对输入张量进行余弦函数运算，再乘以 self.x 返回结果

        class OuterImpl(torch.nn.Module):
            impl: InnerInterface

            def __init__(self):
                super().__init__()
                self.impl = InnerImpl()
                self.x = torch.ones((2, 2)) * 5
            初始化 OuterImpl 类，设置成员变量 self.impl 为 InnerImpl 实例，self.x 为 2x2 全五张量

            def forward(self, inp):
                return self.other_method(inp)
            实现 forward 方法，调用 other_method 处理输入 inp 并返回结果

            def other_method(self, inp):
                return inp.relu() + self.impl(inp.sin()) + self.x
            实现 other_method 方法，对输入 inp 执行整流函数，加上 inp.sin() 的处理结果和 self.x，返回最终结果

        class WrapperModule(torch.nn.Module):
            impl: OuterInterface

            def __init__(self):
                super().__init__()
                self.impl = OuterImpl()
            初始化 WrapperModule 类，设置成员变量 self.impl 为 OuterImpl 实例

            def forward(self, inp):
                return self.impl(inp) + inp
            实现 forward 方法，调用 self.impl 处理输入 inp，然后将处理结果与输入 inp 相加返回

        m = WrapperModule()
        x = torch.rand((2, 2))
        expected = m(x)
        计算未冻结模型的输出结果 expected

        m_s = torch.jit.script(m)
        m_s.eval()
        将 WrapperModule 转换为 TorchScript，并设置为评估模式

        m_s = torch.jit.freeze(m_s)
        冻结 TorchScript 模型 m_s

        actual = m_s(x)
        计算冻结后模型的输出结果 actual

        self.assertEqual(expected, actual)
        断言未冻结模型的输出结果与冻结后模型的输出结果相等

    def test_freeze_non_interface_module_swap(self):
        class InnerModule(torch.nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = x
            初始化 InnerModule 类，设置成员变量 self.x 为输入 x

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                return inp.relu() + self.x
            实现 forward 方法，对输入 inp 执行整流函数并加上 self.x，返回处理结果

        class WrapperModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.option1 = InnerModule(torch.rand((2, 2)))
                self.option2 = InnerModule(torch.rand((2, 2)))
                self.impl = self.option1
                self.idx = 0
            初始化 WrapperModule 类，设置成员变量 option1 和 option2 为不同的 InnerModule 实例，self.impl 初始为 option1，self.idx 初始化为 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.idx += 1
                if self.idx % 2 == 1:
                    self.impl = self.option1
                else:
                    self.impl = self.option2
                根据 self.idx 的奇偶性选择使用 self.option1 或 self.option2 作为 self.impl

                return self.impl(x)
                调用 self.impl 处理输入 x 并返回结果

        unfrozen = WrapperModule()
        创建未冻结的 WrapperModule 实例 unfrozen

        m = torch.jit.script(unfrozen)
        将 unfrozen 转换为 TorchScript

        m.eval()
        将 TorchScript 模型设置为评估模式

        m_frozen = torch.jit.freeze(m)
        冻结 TorchScript 模型 m

        x = torch.rand((2, 2))
        expected = unfrozen(x)
        计算未冻结模型的输出结果 expected

        actual = m_frozen(x)
        计算冻结后模型的输出结果 actual

        self.assertEqual(expected, actual)
        断言未冻结模型的输出结果与冻结后模型的输出结果相等

    @unittest.expectedFailure
    def test_freeze_non_module_class_getattr(self):
        # 定义一个名为 BoxCoder 的普通 Python 类，用于编码盒子坐标变换
        class BoxCoder:
            def __init__(self, bbox_xform_clip):
                # 类初始化函数，接受 bbox_xform_clip 参数作为盒子变换的裁剪值
                self.bbox_xform_clip = bbox_xform_clip

            def decode(self, input):
                # 解码函数，将输入乘以盒子变换裁剪值并返回结果
                return input * self.bbox_xform_clip

        # 定义一个名为 MyModule 的 PyTorch 模块
        class MyModule(torch.nn.Module):
            # 类型注解，指定 box_coder 属性的类型为 BoxCoder 类
            __annotations__ = {
                "box_coder": BoxCoder,
            }

            def __init__(self):
                # 调用父类初始化函数
                super().__init__()
                # 初始化 box_coder 属性为一个 BoxCoder 对象，裁剪值设为 50.0
                self.box_coder = BoxCoder(50.0)

            def forward(self, input):
                # 在 forward 方法中调用 box_coder 对象的 decode 方法，实现前向传播
                return self.box_coder.decode(input)

        # 创建 MyModule 的实例 model
        model = MyModule()
        # 将模型设置为评估模式
        model.eval()
        # 对模型进行 TorchScript 脚本化，并冻结生成的脚本模型
        script_model = torch.jit.freeze(torch.jit.script(model))
        # 创建输入张量 inp
        inp = torch.randn([4, 4])
        # 使用原始模型计算输出
        output_eager = model(inp)
        # 断言原始模型和脚本模型在相同输入下的输出是否一致
        self.assertEqual(model(inp), script_model(inp))
        # 使用 FileCheck 检查脚本模型的图中是否不包含 "GetAttr" 操作
        FileCheck().check_not("GetAttr").run(script_model.graph)
    # 定义一个测试方法，用于测试冻结包含元组输出子模块的模块
    def test_freeze_module_with_tupleoutput_submodule(self):
        # 定义一个子模块类，继承自 nn.Module
        class SubModule(nn.Module):
            # 定义子模块的前向传播方法
            def forward(self, x):
                # 返回一个包含两个元素的元组，分别是 x+1 和 x+2
                return (x + 1, x + 2)

        # 定义一个测试模块类，继承自 nn.Module
        class TestModule(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个 SubModule 的实例
                self.sub = SubModule()

            # 定义模块的前向传播方法
            def forward(self, x):
                # 调用子模块的前向传播方法，接收返回的两个值
                y1, y2 = self.sub(x)
                # 返回两个值的和
                return y1 + y2

        # 使用 torch.jit.script 将 TestModule 脚本化
        m = torch.jit.script(TestModule())
        # 将模块转为评估模式
        m = m.eval()
        # 对脚本化的模块进行冻结
        mf = torch.jit.freeze(m)
        # 创建一个输入张量
        inp = torch.randn(2, 2)
        # 使用原始模块计算期望输出
        expected = m.forward(inp)
        # 使用冻结后的模块计算实际输出
        output = mf.forward(inp)
        # 检查冻结后的图中是否不存在 prim::TupleConstruct 和 prim::TupleUnpack 操作
        FileCheck().check_not("prim::TupleConstruct").run(mf.graph)
        FileCheck().check_not("prim::TupleUnpack").run(mf.graph)
        # 断言实际输出与期望输出相等
        self.assertEqual(output, expected)

    # 定义一个测试方法，用于测试包含 call_method 的模块的冻结
    def test_freeze_module_with_call_method(self):
        # 定义一个模块类 Mod，继承自 nn.Module
        class Mod(nn.Module):
            # 初始化方法，接收一个值作为参数，并将其封装为 nn.Parameter
            def __init__(self, val):
                super().__init__()
                self.param = nn.Parameter(val)

            # 定义模块的前向传播方法
            def forward(self, x):
                # 在前向传播中，返回输入张量 x 加上模块的参数 self.param
                return x + self.param

            # 定义一个导出方法 make_prediction，用于进行额外的预测操作
            @torch.jit.export
            def make_prediction(self, x):
                # 对输入张量 x 进行加法操作
                y = x + x
                # 调用模块自身的前向传播方法，传入 y 作为参数，并返回结果
                return self.forward(y)

        # 创建一个随机的参数张量
        param = torch.rand([2, 2])
        # 创建一个随机的输入张量
        x = torch.rand([2, 2])

        # 创建一个未脚本化的 Mod 实例
        unscripted_mod = Mod(param)
        # 使用 torch.jit.script 将 Mod 实例脚本化
        mod = torch.jit.script(unscripted_mod)
        # 将脚本化的模块设置为评估模式
        mod.eval()
        # 对脚本化的模块进行冻结，同时保留 make_prediction 方法
        mod = torch.jit.freeze(mod, preserved_attrs=["make_prediction"])

        # 断言冻结后的模块的前向传播结果与未冻结模块的前向传播结果在一定的误差范围内相等
        self.assertEqual(
            mod.forward(x), unscripted_mod.forward(x), atol=1e-5, rtol=1e-5
        )
@skipIfTorchDynamo("somehow causing hanging during python shutdown")
class TestFrozenOptimizations(JitTestCase):
    # 创建一个测试类 TestFrozenOptimizations，继承自 JitTestCase，用于测试冻结优化功能
    def setUp(self):
        super().setUp()
        # 在每个测试方法执行前，设置默认的 PyTorch 数据类型，并将默认类型设为双精度浮点数
        self.default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.double)

    def tearDown(self):
        # 在每个测试方法执行后，恢复默认的 PyTorch 数据类型
        torch.set_default_dtype(self.default_dtype)
        super().tearDown()

    def test_conv_bn_folding(self):
        # 测试卷积和批标准化的折叠操作
        conv_bias = [True, False]
        module_pairs = [
            (nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d),
        ]
        use_tracing = [True, False]
        bn_running_stats = [True, False]

        for use_bias, modules, tracing, track_stats in product(
            conv_bias, module_pairs, use_tracing, bn_running_stats
        ):

            class ConvBN(torch.nn.Module):
                # 定义一个卷积和批标准化的模块
                def __init__(self, in_channels, out_channels, **kwargs):
                    super().__init__()
                    # 初始化卷积层，根据 use_bias 参数决定是否包含偏置
                    self.conv = modules[0](
                        in_channels, out_channels, bias=use_bias, **kwargs
                    )
                    # 初始化批标准化层，设置 eps 为 0.001，track_running_stats 根据 track_stats 参数决定是否启用
                    self.bn = modules[1](
                        out_channels, eps=0.001, track_running_stats=track_stats
                    )

                def forward(self, x):
                    # 模型的前向传播，先通过卷积层处理输入 x，然后通过批标准化层处理并返回结果
                    x = self.conv(x)
                    return self.bn(x)

            mod_eager = ConvBN(3, 32, kernel_size=3, stride=2).eval()
            # 创建一个评估模式下的 ConvBN 实例 mod_eager，输入维度为 3，输出维度为 32，设置卷积核大小为 3，步长为 2
            inps = [4, 3, 4]
            if modules[0] == nn.Conv2d:
                inps.append(inps[-1])
            if modules[0] == nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])

            inp = torch.rand(inps)

            if tracing:
                # 如果使用追踪，则对 mod_eager 进行追踪
                scripted_mod = torch.jit.trace(mod_eager, (inp))
            else:
                # 否则对 mod_eager 进行脚本化
                scripted_mod = torch.jit.script(mod_eager)

            self.run_pass("inline", scripted_mod.graph)
            # 运行 "inline" 优化 pass，将其应用到 scripted_mod 的计算图上
            self.run_pass("peephole", scripted_mod.graph)
            # 运行 "peephole" 优化 pass，优化 scripted_mod 的计算图
            self.run_pass("constant_propagation", scripted_mod.graph)
            # 运行 "constant_propagation" 优化 pass，常量传播优化

            FileCheck().check("conv").check("batch").run(scripted_mod.graph)
            # 检查 scripted_mod 的计算图，确保包含 "conv" 和 "batch" 的操作

            # 成功地处理非常量输入
            self.run_pass("fold_frozen_conv_bn", scripted_mod.graph)
            # 运行 "fold_frozen_conv_bn" 优化 pass，折叠冻结的卷积和批标准化操作

            FileCheck().check("conv").check("aten::batch_norm").run(scripted_mod.graph)
            # 检查 scripted_mod 的计算图，确保包含 "conv" 和 "aten::batch_norm" 的操作

            scripted_mod = torch.jit.freeze(scripted_mod)
            # 冻结脚本化的模型

            self.run_pass("fold_frozen_conv_bn", scripted_mod.graph)
            # 再次运行 "fold_frozen_conv_bn" 优化 pass，折叠冻结的卷积和批标准化操作

            if track_stats:
                # 如果 track_stats 为 True
                FileCheck().check("conv").check_not("aten::batch_norm").run(
                    scripted_mod.graph
                )
                # 检查 scripted_mod 的计算图，确保包含 "conv" 但不包含 "aten::batch_norm" 的操作
            else:
                # 否则
                FileCheck().check("conv").check("aten::batch_norm").run(
                    scripted_mod.graph
                )
                # 检查 scripted_mod 的计算图，确保包含 "conv" 和 "aten::batch_norm" 的操作

            self.assertEqual(mod_eager(inp), scripted_mod(inp))
            # 断言原始模型和脚本化冻结后的模型在相同输入下的输出是否一致
            self.assertEqual(mod_eager(inp), scripted_mod(inp))
            # 再次断言确保一致性
    def test_conv_bn_folding_not_forward(self):
        # 定义一个名为 test_conv_bn_folding_not_forward 的测试函数
        class ConvBN(torch.nn.Module):
            # 定义一个名为 ConvBN 的自定义神经网络模块
            def __init__(self, in_channels, out_channels, **kwargs):
                super().__init__()
                # 初始化函数，设置卷积层和批归一化层，并添加一个名为 amt 的常量属性
                self.conv = torch.nn.Conv2d(
                    in_channels, out_channels, bias=True, **kwargs
                )
                self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
                self.amt = 3.2

            def forward(self, x):
                # 前向传播函数，先对输入 x 进行卷积操作，然后应用批归一化层
                x = self.conv(x)
                return self.bn(x)

            @torch.jit.export
            def make_prediction(self, x):
                # 导出的函数，对输入 x 进行前向传播并加上常量属性 amt 的值
                return self.forward(x) + self.amt

        # 创建一个 ConvBN 实例 mod_eager，设为评估模式（eval mode）
        mod_eager = ConvBN(3, 32, kernel_size=3, stride=2).eval()
        # 将 mod_eager 模块脚本化（script）
        scripted_mod = torch.jit.script(mod_eager)
        # 对脚本化的模块应用 TorchScript 优化中的内联传递优化
        torch._C._jit_pass_inline(scripted_mod.make_prediction.graph)
        # 使用 FileCheck 验证脚本化模块的图中是否包含 "conv" 和 "aten::batch_norm" 操作
        FileCheck().check("conv").check("aten::batch_norm").run(
            scripted_mod.make_prediction.graph
        )

        # 冻结脚本化模块，保留 make_prediction 和 amt 属性，不应调用 _jit_pass_optimize_frozen_graph
        scripted_mod = torch.jit.freeze(
            scripted_mod, preserved_attrs=["make_prediction", "amt"]
        )
        # 使用 FileCheck 验证冻结后的模块的图中是否包含 "conv" 操作但不包含 "aten::batch_norm" 操作
        FileCheck().check("conv").check_not("aten::batch_norm").run(
            scripted_mod.make_prediction.graph
        )

    # 在冻结过程中，这会创建张量常量，附加到冻结图中，并由编译单元保持活动（可能导致内存泄漏）
    @skipCUDAMemoryLeakCheckIf(True)
    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_conv_bn_folding_autocast_scenario_cuda(self):
        # CUDA conv takes input tensors which must all be the same dtype,
        # which can cause issues if folding produces inputs of different dtypes.

        class ConvBN(torch.nn.Module):
            def __init__(self, in_channels, out_channels, **kwargs):
                super().__init__()
                # 初始化卷积层，设定无偏置，数据类型为半精度(torch.half)
                self.conv = torch.nn.Conv2d(
                    in_channels, out_channels, bias=False, dtype=torch.half, **kwargs
                )
                # 初始化批归一化层，设置 epsilon 为 0.001，数据类型为单精度(torch.float)
                self.bn = torch.nn.BatchNorm2d(
                    out_channels, eps=0.001, dtype=torch.float
                )

            def forward(self, x):
                # 前向传播函数，返回批归一化后的卷积结果
                return self.bn(self.conv(x))

        # 创建一个 CUDA 加速的 ConvBN 实例，并设定为评估模式
        mod_eager = ConvBN(3, 32, kernel_size=3, stride=2).cuda().eval()
        # 对 mod_eager 进行脚本化
        scripted_mod = torch.jit.script(mod_eager)
        # 冻结脚本化后的模型
        scripted_mod = torch.jit.freeze(scripted_mod)
        # 使用 FileCheck 验证图中是否包含 "conv"，并且不包含 "aten::batch_norm"
        FileCheck().check("conv").check_not("aten::batch_norm").run(scripted_mod.graph)
        # 在图中找到名为 "aten::conv2d" 的节点
        conv_node = scripted_mod.graph.findNode("aten::conv2d", True)
        # 断言找到了 conv_node
        self.assertTrue(conv_node is not None)
        # 获取 conv_node 的名为 "bias" 的命名输入
        bias_input = conv_node.namedInput("bias")
        # 断言 bias_input 不为 None
        self.assertTrue(bias_input is not None)
        # 断言 bias_input 的数据类型是半精度(torch.half)
        self.assertTrue(bias_input.type().dtype() == torch.half)

        # 创建输入张量 x，数据类型为半精度(torch.half)，并移到 CUDA 设备
        x = torch.rand((3, 3, 32, 32), dtype=torch.half).cuda()

        # 断言模型的结果与脚本化模型的结果相等，允许绝对误差和相对误差都在 1e-2 以内
        self.assertEqual(mod_eager(x), scripted_mod(x), atol=1e-2, rtol=1e-2)
        # 再次断言，以确保结果一致性
        self.assertEqual(mod_eager(x), scripted_mod(x), atol=1e-2, rtol=1e-2)

    def test_conv_mul_add_bn(self):
        class Conv_Mul_Add_Bn(nn.Module):
            def __init__(self, in_channels, out_channels, **kwargs):
                super().__init__()
                # 初始化卷积层和批归一化层
                self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
                self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
                # 初始化两个张量 tensor1 和 tensor2
                self.tensor1 = torch.tensor(2.2)
                self.tensor2 = torch.tensor(2)

            def forward(self, x):
                # 前向传播函数，先进行卷积、乘法操作，再加上张量 tensor2，最后进行批归一化
                return self.bn(
                    torch.add(torch.mul(self.conv(x), self.tensor1), self.tensor2)
                )

        # 创建输入张量 input，形状为 (8, 3, 64, 64)
        input = torch.randn(8, 3, 64, 64)
        # 创建 Conv_Mul_Add_Bn 模型实例，并设定为评估模式
        model = Conv_Mul_Add_Bn(3, 32, kernel_size=3, stride=1).eval()

        # 使用无梯度计算上下文
        with torch.no_grad():
            # 使用模型计算结果
            result = model(input)
            # 对模型进行跟踪，并设定为评估模式
            traced_model = torch.jit.trace(model, input).eval()
            # 冻结跟踪后的模型
            traced_model = torch.jit.freeze(traced_model)
            # 使用 FileCheck 验证图中是否包含 "conv"，并且不包含 "aten::batch_norm"
            FileCheck().check("conv").check_not("aten::batch_norm").run(
                traced_model.graph
            )
            # 使用 FileCheck 验证图中是否包含 "conv"，并且不包含 "aten::add"
            FileCheck().check("conv").check_not("aten::add").run(traced_model.graph)
    def test_linear_bn_folding(self):
        # 定义需要测试的线性层与批归一化层的组合
        module_pairs = [
            (nn.Linear, nn.BatchNorm1d),
            (nn.Linear, nn.BatchNorm2d),
            (nn.Linear, nn.BatchNorm3d),
        ]
        # 是否使用追踪模式进行脚本化
        use_tracing = [True, False]
        # 是否使用批归一化层的运行统计信息
        bn_running_stats = [True, False]

        # 遍历所有的组合情况
        for modules, tracing, track_stats in product(
            module_pairs, use_tracing, bn_running_stats
        ):

            # 定义包含线性层与批归一化层的模块类
            class LinearBN(torch.nn.Module):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    # 初始化线性层
                    self.linear = modules[0](in_features, out_features)
                    # 初始化批归一化层
                    self.bn = modules[1](
                        out_features, eps=0.001, track_running_stats=track_stats
                    )

                # 前向传播函数
                def forward(self, x):
                    # 线性变换
                    x = self.linear(x)
                    # 应用批归一化
                    return self.bn(x)

            # 创建评估模式下的模块实例
            mod_eager = LinearBN(32, 32).eval()

            # 定义输入的维度
            inps = [3, 32]
            # 如果是二维批归一化层，增加相应的维度
            if modules[1] == nn.BatchNorm2d:
                inps.append(inps[-1])
                inps.append(inps[-1])
            # 如果是三维批归一化层，增加相应的维度
            if modules[1] == nn.BatchNorm3d:
                inps.append(inps[-1])
                inps.append(inps[-1])
                inps.append(inps[-1])

            # 生成随机输入张量
            inp = torch.rand(inps)

            # 根据追踪标志选择脚本化方式
            if tracing:
                scripted_mod = torch.jit.trace(mod_eager, (inp))
            else:
                scripted_mod = torch.jit.script(mod_eager)

            # 运行各种优化 passes
            self.run_pass("inline", scripted_mod.graph)
            self.run_pass("peephole", scripted_mod.graph)
            self.run_pass("constant_propagation", scripted_mod.graph)

            # 检查脚本化模块的图中的关键字
            FileCheck().check("linear").check("batch").run(scripted_mod.graph)
            # 在非常数输入情况下成功地执行无操作
            self.run_pass("fold_frozen_linear_bn", scripted_mod.graph)
            FileCheck().check("linear").check("aten::batch_norm").run(
                scripted_mod.graph
            )

            # 冻结脚本化模块
            scripted_mod = torch.jit.freeze(scripted_mod)
            self.run_pass("fold_frozen_linear_bn", scripted_mod.graph)
            # 根据 track_stats 标志检查图中的关键字
            if track_stats:
                FileCheck().check("linear").check_not("aten::batch_norm").run(
                    scripted_mod.graph
                )
            else:
                FileCheck().check("linear").check("aten::batch_norm").run(
                    scripted_mod.graph
                )

            # 断言评估模式下模块与脚本化模块的输出相等
            self.assertEqual(mod_eager(inp), scripted_mod(inp))
            self.assertEqual(mod_eager(inp), scripted_mod(inp))
    # 定义一个测试函数，用于验证在与线性层和批量归一化层结合时不进行广播的情况
    def test_bn_not_broadcast_with_linear(self):
        # 定义模块对，包含线性层和批量归一化层的组合
        module_pairs = [
            (nn.Linear, nn.BatchNorm1d),
            (nn.Linear, nn.BatchNorm2d),
            (nn.Linear, nn.BatchNorm3d),
        ]
        # 定义是否进行追踪的选项
        use_tracing = [True, False]
        # 线性层的输入维度
        linear_in = 3
        # 定义维度元组，用于测试不同的线性输出和批量归一化输入情况
        dims = [(2, 4), (4, 2), (1, 2)]

        # 对每个模块对、追踪选项和维度元组进行迭代
        for modules, tracing, dim in product(module_pairs, use_tracing, dims):
            # 获取当前维度元组中的线性输出和批量归一化输入
            linear_out, bn_in = dim[0], dim[1]

            # 创建线性层和批量归一化层的实例
            linear = modules[0](linear_in, linear_out)
            bn = modules[1](bn_in)
            # 创建一个顺序容器，包含线性层和批量归一化层，并设为评估模式
            mod_eager = nn.Sequential(linear, bn).eval()

            # 设置输入的样本数和通道数
            N, C = 3, bn_in
            input_shape = [N, C]
            # 根据批量归一化层的类型添加相应的维度
            if modules[1] == nn.BatchNorm1d:
                H = linear_in
                input_shape.append(H)
            elif modules[1] == nn.BatchNorm2d:
                H, W = 4, linear_in
                input_shape.append(H)
                input_shape.append(W)
            elif modules[1] == nn.BatchNorm3d:
                D, H, W = 4, 4, linear_in
                input_shape.append(D)
                input_shape.append(H)
                input_shape.append(W)

            # 创建随机输入
            inp = torch.rand(input_shape)

            # 如果选择进行追踪，则对模型进行追踪编译；否则进行脚本编译
            if tracing:
                scripted_mod = torch.jit.trace(mod_eager, (inp))
            else:
                scripted_mod = torch.jit.script(mod_eager)

            # 对脚本化模型的图进行一系列优化操作
            self.run_pass("inline", scripted_mod.graph)
            self.run_pass("peephole", scripted_mod.graph)
            self.run_pass("constant_propagation", scripted_mod.graph)

            # 使用FileCheck验证图中是否包含线性和批量归一化操作
            FileCheck().check("linear").check("batch").run(scripted_mod.graph)

            # 对脚本化模型的图进行冻结操作，然后再次优化
            frozen_mod = torch.jit.freeze(scripted_mod)
            self.run_pass("fold_frozen_linear_bn", frozen_mod.graph)
            # 使用FileCheck验证冻结后的图中是否包含线性和批量归一化操作
            FileCheck().check("linear").check("aten::batch_norm").run(
                frozen_mod.graph
            )

            # 验证冻结前后模型的输出是否一致
            self.assertEqual(mod_eager(inp), frozen_mod(inp))
            self.assertEqual(mod_eager(inp), frozen_mod(inp))

            # 验证是否成功阻止了线性和批量归一化的融合
            with self.assertRaisesRegex(
                AssertionError,
                "To fuse, linear.out_features == bn.num_features or bn.num_features == 1",
            ):
                nn.utils.fusion.fuse_linear_bn_eval(linear, bn)

    @skipCUDAMemoryLeakCheckIf(True)
    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    # 定义一个测试方法，用于验证线性层和批量归一化在自动类型转换情况下的折叠行为（针对CUDA环境）
    def test_linear_bn_folding_autocast_scenario_cuda(self):
        # 定义不同组合的模块对和参数
        module_pairs = [
            (nn.Linear, nn.BatchNorm1d),
            (nn.Linear, nn.BatchNorm2d),
            (nn.Linear, nn.BatchNorm3d),
        ]
        use_tracing = [True, False]  # 是否使用追踪（tracing）
        bn_running_stats = [True, False]  # 是否使用批量归一化的运行统计信息

        # 遍历所有组合
        for modules, tracing, track_stats in product(
            module_pairs, use_tracing, bn_running_stats
        ):

            # 定义一个简单的神经网络模块，包括线性层和批量归一化层
            class LinearBN(torch.nn.Module):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    # 创建线性层，使用半精度浮点数（torch.half），并且不使用偏置
                    self.linear = modules[0](
                        in_features, out_features, bias=False, dtype=torch.half
                    )
                    # 创建批量归一化层，设置 epsilon 为 0.001，并使用单精度浮点数（torch.float）
                    self.bn = modules[1](out_features, eps=0.001, dtype=torch.float)

                def forward(self, x):
                    # 执行线性层
                    x = self.linear(x)
                    # 应用批量归一化层
                    return self.bn(x)

            # 创建一个评估模式下运行在CUDA上的实例
            mod_eager = LinearBN(32, 32).cuda().eval()

            # 定义输入数据的维度
            inps = [3, 32]
            # 如果当前批量归一化层是二维的，则扩展输入维度
            if modules[1] == nn.BatchNorm2d:
                inps.append(inps[-1])
                inps.append(inps[-1])
            # 如果当前批量归一化层是三维的，则进一步扩展输入维度
            if modules[1] == nn.BatchNorm3d:
                inps.append(inps[-1])
                inps.append(inps[-1])
                inps.append(inps[-1])

            # 创建随机数据张量，数据类型为半精度浮点数（torch.half），并且在CUDA上
            x = torch.rand(inps, dtype=torch.half).cuda()

            # 根据追踪标志选择脚本化或追踪模式
            if tracing:
                scripted_mod = torch.jit.trace(mod_eager, (x))
            else:
                scripted_mod = torch.jit.script(mod_eager)
            # 冻结脚本化模块
            scripted_mod = torch.jit.freeze(scripted_mod)
            # 使用FileCheck工具检查脚本化模块的图形中是否包含线性操作
            FileCheck().check("linear").check_not("aten::batch_norm").run(
                scripted_mod.graph
            )
            # 查找脚本化模块中的线性操作节点
            lin_node = scripted_mod.graph.findNode("aten::linear", True)
            # 断言线性操作节点存在
            self.assertTrue(lin_node is not None)
            # 获取线性操作节点的权重输入
            weight_input = lin_node.namedInput("weight")
            # 获取线性操作节点的偏置输入
            bias_input = lin_node.namedInput("bias")
            # 断言偏置输入存在
            self.assertTrue(bias_input is not None)
            # 断言权重输入的数据类型为半精度浮点数（torch.half）
            self.assertTrue(weight_input.type().dtype() == torch.half)
            # 断言偏置输入的数据类型为半精度浮点数（torch.half）
            self.assertTrue(bias_input.type().dtype() == torch.half)

            # 断言评估模式下模块在相同输入上的输出与脚本化模块的输出相等
            self.assertEqual(mod_eager(x), scripted_mod(x), atol=1e-2, rtol=1e-2)
            # 再次验证，确保结果一致性
            self.assertEqual(mod_eager(x), scripted_mod(x), atol=1e-2, rtol=1e-2)

    # 根据当前环境是否支持CUDA，决定是否跳过此优化测试
    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_linear_concat(self):
        out_dimms = [[5, 10], [1, 5]]  # 定义多个输出维度组合

        for w1_dim, w2_dim in out_dimms:  # 遍历每对输出维度

            class ModMultLinear(nn.Module):  # 定义多模块线性层的类
                def __init__(self, w1_dim, w2_dim):
                    super().__init__()
                    self.w1 = nn.Parameter(torch.rand([w1_dim, 5]))  # 定义权重 w1
                    self.b1 = nn.Parameter(torch.rand([w1_dim]))  # 定义偏置 b1
                    self.w2 = nn.Parameter(torch.rand([w2_dim, 5]))  # 定义权重 w2
                    self.b2 = nn.Parameter(torch.rand([w2_dim]))  # 定义偏置 b2

                def forward(self, in_tensor1):
                    res1 = torch._C._nn.linear(in_tensor1, self.w1, self.b1)  # 计算第一个线性层的结果
                    res2 = torch._C._nn.linear(in_tensor1, self.w2, self.b2)  # 计算第二个线性层的结果
                    return res1, res2

            mod_eager = ModMultLinear(w1_dim, w2_dim).eval()  # 创建并评估模型对象

            test_val1 = torch.rand([50, 5])  # 创建测试输入数据
            self.check_linear_optimizations(mod_eager, 2, 1, (test_val1,))  # 调用函数检查线性优化

    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_linear_concat_complex(self):
        """
        Testing that the interleaving of multiple optimizations does not
        cause errors, and gets optimized as expected
        """

        class ModMultLinear(nn.Module):  # 定义多模块线性层的类
            def __init__(self):
                super().__init__()
                w1_dim = 5
                w2_dim = 10
                self.w1 = nn.Parameter(torch.rand([w1_dim, 5]))  # 定义权重 w1
                self.b1 = nn.Parameter(torch.rand([w1_dim]))  # 定义偏置 b1
                self.w2 = nn.Parameter(torch.rand([w2_dim, 5]))  # 定义权重 w2
                self.b2 = nn.Parameter(torch.rand([w2_dim]))  # 定义偏置 b2

            def forward(self, in_tensor1):
                res1 = torch._C._nn.linear(in_tensor1, self.w1, self.b1)  # 计算第一个线性层的结果
                res3 = torch._C._nn.linear(res1, self.w2, self.b2)  # 使用第二个权重计算线性层的结果
                res2 = torch._C._nn.linear(in_tensor1, self.w2, self.b2)  # 计算第二个线性层的结果
                res4 = torch._C._nn.linear(res1, self.w1, self.b1)  # 使用第一个权重计算线性层的结果
                return res2, res3, res4  # 返回多个结果

        mod_eager = ModMultLinear().eval()  # 创建并评估模型对象
        test_val1 = torch.rand([50, 5])  # 创建测试输入数据
        self.check_linear_optimizations(mod_eager, 4, 2, (test_val1,))  # 调用函数检查线性优化

    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_linear_concat_different_input(self):
        """
        There should be no change to the graph due to the optimization pass
        due to the two input tensors being different
        """

        # Freezing requires that the graph be a module
        # 定义一个继承自 nn.Module 的模型类 ModMultLinear
        class ModMultLinear(nn.Module):
            def __init__(self, w1_dim, w2_dim):
                super().__init__()
                # 初始化模型参数 w1, b1, w2, b2
                self.w1 = nn.Parameter(torch.rand([w1_dim, 5]))
                self.b1 = nn.Parameter(torch.rand([w1_dim]))
                self.w2 = nn.Parameter(torch.rand([w2_dim, 5]))
                self.b2 = nn.Parameter(torch.rand([w2_dim]))

            def forward(self, in_tensor1, in_tensor2):
                # 使用 PyTorch 底层的线性运算接口计算 res1 和 res2
                res1 = torch._C._nn.linear(in_tensor1, self.w1, self.b1)
                res2 = torch._C._nn.linear(in_tensor2, self.w2, self.b2)
                return res1, res2

        # 创建 ModMultLinear 类的实例 mod_eager，并将其设置为评估模式
        mod_eager = ModMultLinear(5, 5).eval()
        # 生成测试数据 test_val1 和 test_val2
        test_val1 = torch.rand([50, 5])
        test_val2 = torch.rand([50, 5])
        # 调用 self.check_linear_optimizations 方法进行线性优化检查
        self.check_linear_optimizations(mod_eager, 2, 2, (test_val1, test_val2))

    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_linear_multiple_blocks(self):
        # 定义一个继承自 nn.Module 的模型类 ModMultLinear
        class ModMultLinear(nn.Module):
            def __init__(self, w1_dim, w2_dim):
                super().__init__()
                # 初始化模型参数 w1, b1, w2, b2
                self.w1 = nn.Parameter(torch.rand([w1_dim, 5]))
                self.b1 = nn.Parameter(torch.rand([w1_dim]))
                self.w2 = nn.Parameter(torch.rand([w2_dim, 5]))
                self.b2 = nn.Parameter(torch.rand([w2_dim]))

            def forward(self, in_tensor1, in_tensor2, cond: bool):
                # 使用 PyTorch 底层的线性运算接口计算 res1 和 res2
                res1 = torch._C._nn.linear(in_tensor1, self.w1, self.b1)
                # 根据条件 cond 执行不同的线性计算分支
                if cond:
                    res3 = torch._C._nn.linear(in_tensor2, self.w2, self.b2)
                    res4 = torch._C._nn.linear(in_tensor1, self.w2, self.b1)
                else:
                    # 若条件不满足，则抛出断言错误
                    raise AssertionError
                # 继续使用 PyTorch 底层的线性运算接口计算 res2
                res2 = torch._C._nn.linear(in_tensor1, self.w2, self.b1)
                return res1, res2, res3, res4

        # 创建 ModMultLinear 类的实例 mod_eager，并将其设置为评估模式
        mod_eager = ModMultLinear(5, 5).eval()
        # 生成测试数据 test_val1 和 test_val2
        test_val1 = torch.rand([50, 5])
        test_val2 = torch.rand([50, 5])
        # 调用 self.check_linear_optimizations 方法进行线性优化检查
        self.check_linear_optimizations(mod_eager, 4, 3, (test_val1, test_val2, True))

    def check_linear_optimizations(
        self, eager_mod, orig_linears, new_linears, test_vals
        ):
            # 遍历是否使用 CUDA 的情况
            for is_cuda in [False, True]:
                # 如果使用 CUDA，则将 eager_mod 移动到 CUDA 设备上，并将测试值列表中的张量移动到 CUDA 设备上
                if is_cuda:
                    mod_to_device = eager_mod.cuda()
                    test_vals_to_device = [
                        t.cuda() if isinstance(t, torch.Tensor) else t for t in test_vals
                    ]
                else:
                    # 否则保持在 CPU 上
                    mod_to_device = eager_mod
                    test_vals_to_device = test_vals

                # 对模型进行 TorchScript 脚本化
                script_mod = torch.jit.script(mod_to_device)
                op_graph = script_mod.graph

                # 使用 FileCheck 验证操作图中 "aten::linear" 操作的确切出现次数
                FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
                    op_graph
                )

                # 运行 "concat_frozen_linear" 优化传递，对非常数输入成功进行无操作
                self.run_pass("concat_frozen_linear", op_graph)

                # 再次使用 FileCheck 验证 "aten::linear" 操作的确切出现次数
                FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
                    op_graph
                )

                # 冻结 TorchScript 模块
                script_mod = torch.jit.freeze(script_mod)
                op_graph = script_mod.graph

                # 运行 "concat_frozen_linear" 优化传递
                self.run_pass("concat_frozen_linear", op_graph)

                # 根据是否使用 CUDA 验证 "aten::linear" 操作的确切出现次数
                if is_cuda:
                    FileCheck().check_count("aten::linear", new_linears, exactly=True).run(
                        op_graph
                    )
                else:
                    FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
                        op_graph
                    )

                # 断言模型在不同设备上的运行结果一致
                self.assertEqual(
                    mod_to_device(*test_vals_to_device), script_mod(*test_vals_to_device)
                )

    def test_optimize_freeze_module(self):
        # 定义输入通道数和输出通道数
        in_channels, out_channels = 3, 32
        # 创建卷积层和批归一化层组成的模块
        conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=True
        )
        bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        mod = torch.nn.Sequential(conv, bn)

        # 将模块脚本化并冻结，同时关闭数值优化
        frozen_mod = torch.jit.freeze(
            torch.jit.script(mod.eval()), optimize_numerics=False
        )

        # 使用 FileCheck 检查冻结后的模块图中是否包含 "batch_norm"，预期不包含
        FileCheck().check("batch_norm").run(frozen_mod.graph)

        # 运行冻结优化
        torch.jit.run_frozen_optimizations(frozen_mod)

        # 使用 FileCheck 再次检查冻结后的模块图中是否不包含 "batch_norm"
        FileCheck().check_not("batch_norm").run(frozen_mod.graph)

        # 再次冻结模块，此时使用默认的数值优化
        frozen_mod = torch.jit.freeze(torch.jit.script(mod.eval()))
        FileCheck().check_not("batch_norm").run(frozen_mod.graph)
    def test_freeze_remove_dropout(self):
        # 定义一个名为 test_freeze_remove_dropout 的测试函数
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                # 在神经网络模块中添加一个 dropout 层，丢弃概率为 0.5
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                # 前向传播函数，对输入 x 应用 dropout 层
                return self.dropout(x)

        mod = torch.jit.script(Net())
        # 将模型 mod 转换为 TorchScript，生成一个脚本模型
        torch._C._jit_pass_inline(mod.graph)
        # 在 TorchScript 模型的计算图中进行内联传递优化
        FileCheck().check("aten::dropout").run(mod.graph)
        # 检查 TorchScript 模型的计算图是否包含 dropout 操作
        frozen_mod = torch.jit.freeze(mod.eval())
        # 冻结 TorchScript 模型，生成一个不可修改的模型
        FileCheck().check_not("aten::dropout").run(frozen_mod.graph)
        # 检查冻结后的 TorchScript 模型的计算图是否不包含 dropout 操作

        input = torch.randn(2)
        output_s = mod.forward(input)
        output_f = frozen_mod.forward(input)
        self.assertEqual(output_s, output_f)
        # 对比模型和冻结模型在相同输入下的输出是否一致

    def test_freeze_remove_feature_dropout(self):
        # 定义一个名为 test_freeze_remove_feature_dropout 的测试函数
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                # 在神经网络模块中添加一个二维特征 dropout 层，丢弃概率为 0.5
                self.dropout = nn.Dropout2d(0.5)

            def forward(self, x):
                # 前向传播函数，对输入 x 应用二维特征 dropout 层
                return self.dropout(x)

        mod = torch.jit.script(Net().eval())
        # 将模型 Net 实例化并转换为 TorchScript，生成一个脚本模型
        torch._C._jit_pass_inline(mod.graph)
        # 在 TorchScript 模型的计算图中进行内联传递优化
        FileCheck().check("aten::feature_dropout").run(mod.graph)
        # 检查 TorchScript 模型的计算图是否包含二维特征 dropout 操作
        frozen_mod = torch.jit.freeze(mod)
        # 冻结 TorchScript 模型，生成一个不可修改的模型
        FileCheck().check_not("aten::feature_dropout").run(frozen_mod.graph)
        # 检查冻结后的 TorchScript 模型的计算图是否不包含二维特征 dropout 操作

        input = torch.randn(2, 2, 1, 1)
        output_s = mod.forward(input)
        output_f = frozen_mod.forward(input)
        self.assertEqual(output_s, output_f)
        # 对比模型和冻结模型在相同输入下的输出是否一致

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_freeze_mkdlnn(self):
        # 定义一个名为 test_freeze_mkdlnn 的测试函数，条件是 MKL-DNN 构建不可用时跳过测试
        conv = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2).eval().float()
        # 创建一个卷积层 conv，设置为评估模式并转换为浮点类型
        convmkl = mkldnn_utils.to_mkldnn(conv)
        # 将 conv 转换为 MKL-DNN 卷积格式
        out = torch.jit.freeze(torch.jit.script(convmkl.eval()))
        # 将 MKL-DNN 格式的 conv 转换为 TorchScript，并冻结生成不可修改的模型
        inp = torch.rand([4, 3, 4, 4]).float()
        # 创建一个随机张量作为输入，数据类型为浮点型
        self.assertEqual(out(inp.to_mkldnn()).to_dense(), conv(inp))
        # 比较冻结模型在 MKL-DNN 格式输入上的输出与原始卷积层在相同输入上的输出是否一致

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    # 定义名为 test_conv_to_mkldnn 的测试方法
    def test_conv_to_mkldnn(self):
        # 将默认的数据类型设置为 torch.float
        with set_default_dtype(torch.float):
            # 遍历 nn.Conv2d 和 nn.Conv3d 以及 False 和 True 的组合
            for module, trace in product([nn.Conv2d, nn.Conv3d], [False, True]):
                # 创建一个指定参数的模块，并将其设为评估模式
                mod = module(3, 32, kernel_size=3, stride=2).eval()
                # 初始化输入大小列表
                inps = [4, 3, 4]
                # 如果模块是 nn.Conv2d，则在输入大小列表中再添加一个值
                if module == nn.Conv2d:
                    inps.append(inps[-1])
                # 如果模块是 nn.Conv3d，则在输入大小列表中再添加两个相同的值
                if module == nn.Conv3d:
                    inps.append(inps[-1])
                    inps.append(inps[-1])

                # 创建一个随机输入张量
                inp = torch.rand(inps)
                # 如果 trace 为 True，则对模块进行脚本化
                if trace:
                    scripted_mod = torch.jit.script(mod)
                else:
                    # 否则使用输入来跟踪模块
                    scripted_mod = torch.jit.trace(mod, (inp,))

                # 在脚本化模块的图上运行 "inline" 优化
                self.run_pass("inline", scripted_mod.graph)

                # 检查脚本化模块的图中是否包含 "conv"
                FileCheck().check("conv").run(scripted_mod.graph)

                # 运行 "convert_frozen_ops_to_mkldnn" 优化，并检查是否成功没有操作非常量输入
                self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)

                # 检查脚本化模块的图中是否不包含 "to_mkldnn"
                FileCheck().check_not("to_mkldnn").run(scripted_mod.graph)

                # 冻结脚本化模块，并运行 "convert_frozen_ops_to_mkldnn" 优化
                scripted_mod = torch.jit.freeze(scripted_mod)
                self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)

                # 检查脚本化模块的图中是否包含 "to_mkldnn"、"prim::mkldnn_convolution" 和 "to_dense"
                FileCheck().check("to_mkldnn").check("prim::mkldnn_convolution").check(
                    "to_dense"
                ).run(scripted_mod.graph)

                # 断言模块在输入上的输出与脚本化模块在输入上的输出相等
                self.assertEqual(mod(inp), scripted_mod(inp))
                self.assertEqual(mod(inp), scripted_mod(inp))

    # 定义名为 test_linear_transpose 的测试方法
    def test_linear_transpose(self):
        # 定义一个名为 ModLinear 的类，继承自 torch.nn.Module
        class ModLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个大小为 30 的随机偏置参数
                self.bias = torch.nn.Parameter(torch.rand(30))
                # 初始化一个大小为 [30, 20] 的随机权重参数
                self.weight = torch.nn.Parameter(torch.rand([30, 20]))

            # 定义前向传播方法，使用低级 API 完成线性运算
            def forward(self, x):
                return torch._C._nn.linear(x, self.weight, self.bias)

        # 创建一个 ModLinear 类的实例，并将其设为评估模式
        mod_eager = ModLinear().eval()
        # 创建一个大小为 [50, 20] 的随机测试值
        test_val = torch.rand([50, 20])
        # 调用 self.check_linear_optimizations_2 方法，检查线性优化
        self.check_linear_optimizations_2(
            mod_eager, 1, 0, "transpose_frozen_linear", (test_val,)
        )

    # 定义名为 test_linear_non_constant_weight 的测试方法
    def test_linear_non_constant_weight(self):
        # 定义一个名为 ModLinear 的类，继承自 torch.nn.Module
        class ModLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个大小为 30 的随机偏置参数
                self.bias = torch.nn.Parameter(torch.rand(30))

            # 定义前向传播方法，使用低级 API 完成线性运算，接受额外的权重参数
            def forward(self, x, weight):
                return torch._C._nn.linear(x, weight, self.bias)

        # 创建一个 ModLinear 类的实例，并将其设为评估模式
        mod_eager = ModLinear().eval()
        # 创建大小为 [50, 20] 的随机测试值
        test_val = torch.rand([50, 20])
        # 创建大小为 [30, 20] 的随机权重值
        test_weight = torch.rand([30, 20])
        # 调用 self.check_linear_optimizations_2 方法，检查线性优化
        self.check_linear_optimizations_2(
            mod_eager, 1, 1, "transpose_frozen_linear", (test_val, test_weight)
        )

    # 定义名为 check_linear_optimizations_2 的辅助方法，用于检查线性优化
    def check_linear_optimizations_2(
        self, eager_mod, orig_linears, new_linears, opt_pass, test_vals
    ):
        # TODO: merge with check_linear_optimizations once both diffs land
        # 将 eager_mod 赋值给 mod_to_device
        mod_to_device = eager_mod
        # 将 test_vals 赋值给 test_vals_to_device
        test_vals_to_device = test_vals

        # 对 mod_to_device 进行 Torch 脚本化
        script_mod = torch.jit.script(mod_to_device)
        # 获取脚本化后的图形表示
        op_graph = script_mod.graph

        # 检查 op_graph 中 "aten::linear" 的确切出现次数是否与 orig_linears 相符
        FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
            op_graph
        )
        # 使用优化传递函数 opt_pass，运行在 op_graph 上，连续执行无操作操作，用于非常量输入
        self.run_pass(opt_pass, op_graph)
        # 再次检查 op_graph 中 "aten::linear" 的确切出现次数是否与 orig_linears 相符
        FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
            op_graph
        )

        # 冻结 script_mod
        script_mod = torch.jit.freeze(script_mod)
        # 获取冻结后的图形表示
        op_graph = script_mod.graph
        # 在冻结后的 op_graph 上运行优化传递函数 opt_pass
        self.run_pass(opt_pass, op_graph)
        # 检查 op_graph 中 "aten::linear" 的确切出现次数是否与 new_linears 相符
        FileCheck().check_count("aten::linear", new_linears, exactly=True).run(op_graph)

        # 断言模型转移到设备后的运行结果与脚本化模型在相同设备上运行结果相等
        self.assertEqual(
            mod_to_device(*test_vals_to_device), script_mod(*test_vals_to_device)
        )

    @staticmethod
    def conv():
        # 用于测试目的的通用可组合卷积层
        return nn.Conv2d(8, 8, 1)

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_collapse_adjacent_conversions(self):
        # 设置默认数据类型为 torch.float
        with set_default_dtype(torch.float):
            # 创建一个由两个 self.conv() 组成的序列模型，并设置为评估模式
            mod = nn.Sequential(self.conv(), self.conv()).eval()
            # 对模型进行 Torch 脚本化
            scripted_mod = torch.jit.script(mod)
            # 冻结脚本化后的模型
            scripted_mod = torch.jit.freeze(scripted_mod)
            # 在冻结后的图形表示上运行 "convert_frozen_ops_to_mkldnn" 优化传递函数
            self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
            # 检查脚本化模型的图形表示是否包含 "to_mkldnn"、"prim::mkldnn_convolution" 和 "to_dense"
            FileCheck().check("to_mkldnn").check("prim::mkldnn_convolution").check(
                "prim::mkldnn_convolution"
            ).check("to_dense").run(scripted_mod.graph)
            # 检查 "to_mkldnn" 的确切出现次数是否为 1
            FileCheck().check_count("to_mkldnn", 1, exactly=True).run(
                scripted_mod.graph
            )

            # 创建输入张量 inp，形状为 [1, 8, 8, 8]，内容为随机生成的浮点数
            inp = torch.rand([1, 8, 8, 8])
            # 断言脚本化模型和原始模型在相同输入上的输出是否相等
            self.assertEqual(scripted_mod(inp), mod(inp))
            # 再次断言脚本化模型和原始模型在相同输入上的输出是否相等
            self.assertEqual(scripted_mod(inp), mod(inp))
    # 定义一个测试方法，用于测试 MKL-DNN 合并广播操作
    def test_mkldnn_fuser_broadcasting(self):
        # 定义一个简单的加法模块，继承自 nn.Module
        class Add(nn.Module):
            def __init__(self, tensor):
                super().__init__()
                self.tensor = tensor

            # 前向传播方法，对输入 x 执行加法操作
            def forward(self, x):
                return x + self.tensor

        # 设置默认的数据类型为 float
        with set_default_dtype(torch.float):
            # 对于两种不同的输入 [8] 和 [8, 8, 1] 分别进行测试
            for add_inp in [8], [8, 8, 1]:
                # 创建一个包含卷积层和 Add 模块的序列化模块，并设为评估模式
                mod = nn.Sequential(self.conv(), Add(torch.rand(add_inp))).eval()
                # 对模块进行脚本化
                scripted_mod = torch.jit.script(mod)
                # 冻结脚本化后的模块
                scripted_mod = torch.jit.freeze(scripted_mod)
                # 运行名为 "convert_frozen_ops_to_mkldnn" 的优化传递函数，修改模块的图结构
                self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
                # 使用 FileCheck 检查模块的图结构，确认是否包含广播操作的 MKL-DNN 张量
                FileCheck().check("prim::BroadcastMKLDNNTensors").run(
                    scripted_mod.graph
                )
                # 创建输入张量，形状为 [1, 8, 8, 8]
                inp = torch.rand([1, 8, 8, 8])
                # 断言脚本化模块对输入 inp 的输出与原始模块 mod 的输出相等
                self.assertEqual(scripted_mod(inp), mod(inp))
                # 再次断言，确保结果一致性
                self.assertEqual(scripted_mod(inp), mod(inp))

                # 为了确保，检查如果没有此操作，广播是否正常工作，以便未来支持时可以删除此操作
                with self.assertRaisesRegex(RuntimeError, ""):
                    (
                        torch.rand([1, 8, 8, 8]).to_mkldnn()
                        + torch.rand(add_inp).to_mkldnn()
                    )

    # 如果 MKL-DNN 构建被禁用，则跳过测试
    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    # 定义一个测试方法，用于测试 MKL-DNN 中的就地操作移除
    def test_mkldnn_inplace_removal(self):
        # 定义一个结合加法和乘法的模块
        class AddMul(nn.Module):
            def __init__(self, tensor):
                super().__init__()
                self.tensor = tensor

            # 前向传播方法，执行就地加法和除法操作，并减去常量 4
            def forward(self, x):
                return x.add_(self.tensor).div_(self.tensor) - 4

        # 设置默认的数据类型为 float
        with set_default_dtype(torch.float):
            # 创建一个包含卷积层和 AddMul 模块的序列化模块，并设为评估模式
            mod = nn.Sequential(self.conv(), AddMul(torch.rand([8]))).eval()
            # 对模块进行脚本化
            scripted_mod = torch.jit.script(mod)
            # 冻结脚本化后的模块
            scripted_mod = torch.jit.freeze(scripted_mod)
            # 运行名为 "convert_frozen_ops_to_mkldnn" 的优化传递函数，修改模块的图结构
            self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
            # 使用 FileCheck 检查模块的图结构，确认是否包含 to_mkldnn、add_ 和 div_ 操作
            FileCheck().check("aten::to_mkldnn").check("aten::add_").check(
                "aten::div_"
            ).run(scripted_mod.graph)
            # 创建输入张量，形状为 [1, 8, 8, 8]
            inp = torch.rand([1, 8, 8, 8])
            # 断言脚本化模块对输入 inp 的输出与原始模块 mod 的输出相等
            self.assertEqual(scripted_mod(inp), mod(inp))
            # 再次断言，确保结果一致性
            self.assertEqual(scripted_mod(inp), mod(inp))
    def test_maxpool_mkldnn(self):
        # 设置默认的张量数据类型为 float
        with set_default_dtype(torch.float):
            # 创建一个 ResNet-18 模型
            model = torchvision.models.resnet18()
            # 从 ResNet-18 模型中提取一部分子模型，包括卷积层、批归一化层、ReLU 激活函数和最大池化层
            sub_model = torch.nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool
            )
            # 对子模型进行脚本化和冻结
            mod = torch.jit.freeze(torch.jit.script(sub_model.eval()))
            # 定义输入张量的维度
            (
                N,
                C,
                H,
                W,
            ) = (
                10,
                3,
                224,
                224,
            )
            # 生成一个随机输入张量
            inp = torch.randn(N, C, H, W)
            # 运行一个特定的优化 pass，将冻结操作转换为 MKLDNN 格式
            self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
            # 检查模型图中是否包含 "max_pool" 和 "to_dense" 操作
            FileCheck().check("max_pool").check("to_dense").run(mod.graph)
            # 检查 "to_dense" 操作的数量是否确切为 1
            FileCheck().check_count("to_dense", 1, exactly=True).run(mod.graph)
            # 断言冻结后的模型对相同的输入张量能够产生与原始子模型相同的结果
            self.assertEqual(mod(inp), sub_model(inp))

    @unittest.skipIf(torch.backends.mkldnn.is_available(), "Testing no mkldnn")
    def test_conv_to_mkldnn_no_mkldnn(self):
        # 当 MKLDNN 不可用时，测试不会出现错误
        with set_default_dtype(torch.float):
            # 创建一个 Conv2d 层并进行脚本化
            mod = torch.jit.script(nn.Conv2d(3, 32, kernel_size=3, stride=2).eval())
            # 对脚本化后的 Conv2d 层进行冻结
            frozen = torch.jit.freeze(mod)
            # 运行一个特定的优化 pass，将冻结操作转换为 MKLDNN 格式
            self.run_pass("convert_frozen_ops_to_mkldnn", frozen.graph)
            # 创建一个随机输入张量
            inp = torch.rand([4, 3, 4, 4])
            # 断言冻结后的模型对相同的输入张量能够产生与原始模型相同的结果
            self.assertEqual(frozen(inp), mod(inp))

    @unittest.skipIf(not (TEST_CUDNN or TEST_WITH_ROCM), "requires CUDNN")
    def test_freeze_conv_relu_fusion(self):
        # 使用默认数据类型为 float 运行测试
        with set_default_dtype(torch.float):
            # 定义测试的卷积层是否带偏置的可能性
            conv_bias = [True, False]
            # 定义测试的卷积操作类型（2D 或 3D）
            conv_ops = [nn.Conv2d, nn.Conv3d]
            # 定义是否添加 z 到输出的可能性
            use_add_z = [True, False]
            # 定义是否使用追踪（tracing）的可能性
            use_tracing = [True, False]
            # 使用 product 函数生成所有可能的组合进行测试
            for use_bias, conv, add_z, tracing in product(
                conv_bias, conv_ops, use_add_z, use_tracing
            ):

                # 定义一个名为 Net 的简单神经网络模型
                class Net(nn.Module):
                    def __init__(self, in_channels, out_channels, **kwargs):
                        super().__init__()
                        # 根据参数初始化卷积层，选择是否带偏置
                        self.conv = conv(
                            in_channels, out_channels, bias=use_bias, **kwargs
                        )
                        # 添加 ReLU 激活函数，inplace=True 表示原地操作
                        self.relu = nn.ReLU(inplace=True)
                        # 添加一个标志位 add_z，控制是否将 z 添加到输出
                        self.add_z = add_z

                    def forward(self, x):
                        # 执行卷积操作
                        z = self.conv(x)
                        out = self.conv(x)
                        # 如果 add_z 为 True，则将 z 添加到输出中
                        if self.add_z:
                            out += z
                        # 对输出应用 ReLU 激活函数
                        out = self.relu(out)
                        return out

                # 创建一个在 GPU 上运行的模型实例 mod_eager
                mod_eager = Net(3, 6, kernel_size=3, stride=2).eval().cuda()

                # 创建输入数据的尺寸列表 inps，并转换为 GPU 张量 inp
                inps = [5, 3, 4, 4]
                if conv == nn.Conv3d:
                    inps.append(inps[-1])
                inp = torch.rand(inps).cuda()

                # 根据是否追踪（tracing）选择脚本化（script）或追踪（trace）模型
                if tracing:
                    scripted_mod = torch.jit.trace(mod_eager, (inp))
                else:
                    scripted_mod = torch.jit.script(mod_eager)

                # 对脚本化/追踪的模型进行推理优化
                frozen_mod = torch.jit.optimize_for_inference(scripted_mod)

                # 根据测试环境选择相应的检查器进行检查
                if TEST_WITH_ROCM:
                    if add_z:
                        FileCheck().check("aten::miopen_convolution_add_relu").run(
                            frozen_mod.graph
                        )
                    else:
                        FileCheck().check("aten::miopen_convolution_relu").run(
                            frozen_mod.graph
                        )
                else:
                    if add_z:
                        FileCheck().check("aten::cudnn_convolution_add_relu").run(
                            frozen_mod.graph
                        )
                    else:
                        FileCheck().check("aten::cudnn_convolution_relu").run(
                            frozen_mod.graph
                        )

                # 断言原始模型和冻结模型的输出应该一致
                self.assertEqual(mod_eager(inp), frozen_mod(inp))

    @unittest.skipIf(not (TEST_CUDNN or TEST_WITH_ROCM), "requires CUDNN")
    def test_freeze_conv_relu_fusion_not_forward(self):
        with set_default_dtype(torch.float):
            # 定义一个继承自 nn.Module 的神经网络模型类 Net
            class Net(nn.Module):
                # 初始化函数，接受输入通道数和输出通道数等参数
                def __init__(self, in_channels, out_channels, **kwargs):
                    super().__init__()
                    # 创建一个二维卷积层，无偏置项
                    self.conv = nn.Conv2d(
                        in_channels, out_channels, bias=None, **kwargs
                    )
                    # 创建一个 inplace=True 的 ReLU 激活函数层
                    self.relu = nn.ReLU(inplace=True)

                # 前向传播函数，接受输入 x
                def forward(self, x):
                    # 对输入 x 执行卷积操作并赋给 z
                    z = self.conv(x)
                    # 再次对输入 x 执行卷积操作并赋给 out
                    out = self.conv(x)
                    # 对 out 执行 ReLU 激活函数操作
                    out = self.relu(out)
                    return out

                # 导出的方法，将前向传播功能暴露为 make_prediction 方法
                @torch.jit.export
                def make_prediction(self, x):
                    return self.forward(x)

            # 创建一个在 GPU 上评估的模型实例 mod_eager
            mod_eager = Net(3, 6, kernel_size=3, stride=2).eval().cuda()

            # 创建输入张量 inp，大小为 [5, 3, 4, 4]，并将其放在 GPU 上
            inps = [5, 3, 4, 4]
            inp = torch.rand(inps).cuda()

            # 对 mod_eager 进行脚本化
            scripted_mod = torch.jit.script(mod_eager)

            # 冻结脚本化后的模型，保留 make_prediction 方法
            frozen_mod = torch.jit.freeze(
                scripted_mod, preserved_attrs=["make_prediction"]
            )

            # 对冻结后的模型进行推理优化
            optimized_mod = torch.jit.optimize_for_inference(
                frozen_mod, other_methods=["make_prediction"]
            )

            # 根据 TEST_WITH_ROCM 的值选择不同的运行时检查器，检查优化后模型的图结构
            if TEST_WITH_ROCM:
                FileCheck().check("aten::miopen_convolution_relu").run(
                    optimized_mod.make_prediction.graph
                )
            else:
                FileCheck().check("aten::cudnn_convolution_relu").run(
                    optimized_mod.make_prediction.graph
                )

            # 断言优化前后模型对同一输入 inp 的输出一致性
            self.assertEqual(
                mod_eager.make_prediction(inp), optimized_mod.make_prediction(inp)
            )

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_numel_less_than_size_with_padding(self):
        with set_default_dtype(torch.float):
            # 定义一个继承自 nn.Module 的神经网络模型类 MyModule
            class MyModule(nn.Module):
                # 初始化函数
                def __init__(self):
                    super().__init__()
                    # 创建一个二维卷积层，指定各种参数
                    self.conv1 = nn.Conv2d(
                        1,
                        2,
                        kernel_size=(2, 4),
                        stride=2,
                        padding=2,
                        dilation=(2, 1),
                    )

                # 前向传播函数，接受输入 i0
                def forward(self, i0):
                    # 对输入 i0 执行卷积操作并赋给 x
                    x = self.conv1(i0)
                    # 对 x 和 i0 分别执行 torch.max 和 torch.clip 操作，并赋给 o0 和 o1
                    o0 = torch.max(x, i0)
                    o1 = torch.clip(x, -1.5, 1.5)
                    return o0, o1

            # 创建一个输入张量 i0，形状为 [1, 1, 1, 2]，数据类型为 torch.float32
            i0 = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
            # 创建 MyModule 类的一个实例 mod
            mod = MyModule()
            # 对 mod 进行迭代，获取输出 out
            out = mod(i0)

            # 对模型 mod 进行跟踪，获得导出模型 exported
            exported = torch.jit.trace(mod, [i0])
            # 对导出模型进行推理优化
            exported = torch.jit.optimize_for_inference(exported)

            # 对输入 i0 执行导出模型 exported，获取输出 eout
            eout = exported(i0)
            # 断言 out 和 eout 中每个张量的所有元素是否相近
            self.assertTrue(all(torch.allclose(x, y) for x, y in zip(out, eout)))
    def test_incompatible_perf_formats(self):
        # 使用 torch.float 作为默认的数据类型
        with set_default_dtype(torch.float):

            # 定义一个继承自 nn.Module 的模型类 Mod
            class Mod(nn.Module):
                def __init__(self):
                    super().__init__()
                    # 添加一个 2D 卷积层和一个最大池化层到模型
                    self.conv = torch.nn.Conv2d(3, 64, 3, 2)
                    self.max_pool = torch.nn.MaxPool2d(111, 111)

                # 定义模型的前向传播方法
                def forward(self, x):
                    # 对输入 x 执行卷积操作
                    a = self.conv(x)
                    # 对 a 执行最大池化操作
                    b = self.max_pool(a)
                    # 返回卷积和池化结果的和作为输出
                    return a + b

            # 创建 Mod 类的实例 model
            model = Mod()
            # 将模型设置为评估模式
            model.eval()
            # 使用 torch.jit.script 将模型转换为 Torch 脚本，然后冻结它
            mod = torch.jit.freeze(torch.jit.script(model))
            # 定义输入张量的维度 N, C, H, W
            (
                N,
                C,
                H,
                W,
            ) = (
                10,
                3,
                224,
                224,
            )
            # 生成符合正态分布的随机张量作为输入
            inp = torch.randn(N, C, H, W)
            # 对模型的 Torch 脚本图应用名为 "convert_frozen_ops_to_mkldnn" 的优化 pass
            self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
            # 断言模型对输入 inp 的输出与冻结的 Torch 脚本模型 mod 的输出相等
            self.assertEqual(model(inp), mod(inp))

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_pool2d_batchnorm(self):
        # 使用 torch.float 作为默认的数据类型
        with set_default_dtype(torch.float):
            # 定义包含不同池化和批归一化层的列表
            pooling_layers = [
                torch.nn.AdaptiveAvgPool2d(4),
                # torch.nn.AdaptiveMaxPool2d(4), # return tuples
                torch.nn.MaxPool2d(4),
                torch.nn.AvgPool2d(4),
                torch.nn.BatchNorm2d(64).eval(),
            ]

            # 遍历池化层列表中的每一个池化层
            for pl in pooling_layers:
                # 创建一个包含卷积、ReLU、池化和 Hardswish 激活函数的顺序模型
                sub_model = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 2, 2),
                    torch.nn.ReLU(),
                    pl,
                    torch.nn.Hardswish(),
                )
                # 将子模型设置为评估模式
                sub_model.eval()
                # 使用 torch.jit.script 将子模型转换为 Torch 脚本，然后冻结它
                mod = torch.jit.freeze(torch.jit.script(sub_model))
                # 定义输入张量的维度 N, C, H, W
                (
                    N,
                    C,
                    H,
                    W,
                ) = (
                    10,
                    3,
                    224,
                    224,
                )
                # 生成符合正态分布的随机张量作为输入
                inp = torch.randn(N, C, H, W)
                # 在模型的 Torch 脚本图上应用 "removeExceptions" 和 "dce" 两个优化 pass
                removeExceptions(mod.graph)
                self.run_pass("dce", mod.graph)
                # 对模型的 Torch 脚本图应用名为 "convert_frozen_ops_to_mkldnn" 的优化 pass
                self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
                # 对模型的 Torch 脚本图运行 FileCheck 检查操作是否包含 "aten::to_dense" 和 "return"
                FileCheck().check("aten::to_dense").check_next("return").run(mod.graph)
                # 断言子模型对输入 inp 的输出与冻结的 Torch 脚本模型 mod 的输出相等
                self.assertEqual(sub_model(inp), mod(inp))
    # 定义一个测试方法，用于测试3D池化和批归一化操作
    def test_pool3d_batchnorm(self):
        # 设置默认的张量数据类型为浮点型
        with set_default_dtype(torch.float):
            # 定义包含不同池化和批归一化层的列表
            pooling_layers = [
                torch.nn.MaxPool3d(4),  # 使用3D最大池化层，窗口大小为4
                # torch.nn.AdaptiveAvgPool3d(4), # 没有ideep绑定
                # torch.nn.AdaptiveMaxPool3d(4), # 返回元组
                torch.nn.AvgPool3d(4),  # 使用3D平均池化层，窗口大小为4
                torch.nn.BatchNorm3d(64).eval(),  # 创建一个64通道的3D批归一化层，并设为评估模式
            ]

            # 遍历池化层列表
            for pl in pooling_layers:
                # 创建一个包含卷积、ReLU激活函数、池化层和Hardswish激活函数的序列模型
                sub_model = torch.nn.Sequential(
                    torch.nn.Conv3d(3, 64, 2, 2),  # 3D卷积层，输入通道数3，输出通道数64，核大小为2，步幅为2
                    torch.nn.ReLU(),  # ReLU激活函数
                    pl,  # 池化层，根据循环当前池化层变量pl确定
                    torch.nn.Hardswish(),  # Hardswish激活函数
                )
                sub_model.eval()  # 将子模型设为评估模式
                mod = torch.jit.freeze(torch.jit.script(sub_model))  # 冻结并脚本化子模型
                N, C, H, W, D = 10, 3, 64, 64, 64  # 定义输入张量的形状
                inp = torch.randn(N, C, D, H, W)  # 生成随机输入张量
                # 这两次通过运行需要删除BatchNorm2d中的尺寸检查
                removeExceptions(mod.graph)  # 调用一个函数或方法以删除异常
                self.run_pass("dce", mod.graph)  # 在模型图上运行"dead code elimination"（死代码消除）优化
                self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)  # 将冻结的操作转换为MKL-DNN操作
                FileCheck().check("aten::to_dense").check_next("return").run(mod.graph)
                self.assertEqual(sub_model(inp), mod(inp))  # 断言子模型和脚本化模型在相同输入下的输出是否一致

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    @skipIfNoTorchVision
    # 定义一个测试方法，用于测试不同激活函数与模型卷积的组合效果
    def test_conv_hardswish(self):
        # 设置默认的张量数据类型为浮点数
        with set_default_dtype(torch.float):

            # 定义一个自定义模块 Clamp，用于对输入张量进行值范围限制
            class Clamp(torch.nn.Module):
                # 初始化方法，设置最小值和最大值
                def __init__(self, min_val, max_val, **kwargs):
                    super().__init__()
                    self.min_val = min_val
                    self.max_val = max_val

                # 前向传播方法，使用 torch.clamp 函数对输入张量 x 进行限制
                def forward(self, x):
                    return torch.clamp(x, self.min_val, self.max_val)

            # 定义卷积层输入张量的维度
            (
                N,
                C,
                H,
                W,
            ) = (
                10,
                3,
                224,
                224,
            )
            # 定义多个激活函数实例
            activations = [
                torch.nn.Hardswish(),
                torch.nn.Hardsigmoid(),
                torch.nn.ReLU6(),
                torch.nn.Tanh(),
                torch.nn.Hardtanh(0.0, 6.0),
                torch.nn.Hardtanh(1.0, 100.0),
                torch.nn.Hardtanh(-100.0, -1.0),
                torch.nn.GELU(),
                Clamp(-100.0, -1.0),
                Clamp(1.0, 100.0),
                Clamp(0.0, 6.0),
                Clamp(-1.0, 0.0),
            ]

            # 实例化一个 ResNet-18 模型
            model = torchvision.models.resnet18()
            # 遍历每种激活函数
            for activation in activations:
                # 创建一个包含模型第一卷积层和当前激活函数的序列模型
                sub_model = torch.nn.Sequential(model.conv1, activation)
                # 将子模型设置为评估模式
                sub_model.eval()
                # 使用 TorchScript 对子模型进行冻结并脚本化
                mod = torch.jit.freeze(torch.jit.script(sub_model))
                # 创建一个随机输入张量
                inp = torch.randn(N, C, H, W)
                # 运行特定的优化传递，将冻结操作转换为 MKL-DNN 格式
                self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
                # 使用 FileCheck 检查图中是否存在 "aten::to_dense" 操作
                FileCheck().check_count("aten::to_dense", 1, exactly=True).run(
                    mod.graph
                )
                # 断言子模型对随机输入的输出与脚本化模型对同一输入的输出相等
                self.assertEqual(sub_model(inp), mod(inp))

    # 如果没有启用 MKL-DNN 构建，则跳过当前测试
    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    # 定义一个测试函数，用于测试 MKLDNN 硬切函数的功能
    def test_hardswish_hardsigmoid(self):
        # 将默认数据类型设置为浮点型
        with set_default_dtype(torch.float):
            # 定义操作映射表，将原语操作映射到对应的 PyTorch 函数
            op_map = {
                "prim::MKLDNNHardSwish": F.hardswish,
                "prim::MKLDNNHardSigmoid": F.hardsigmoid,
            }

            # 定义多个输入大小
            input_sizes = ([0], [1], [3], [1, 3, 8, 8])
            # 遍历操作映射表中的每一个项
            for mkldnn_opname, aten_op in op_map.items():
                # 遍历每种输入大小
                for size in input_sizes:
                    # 针对每种输入大小测试 in-place 和非 in-place 的情况
                    for inplace in (True, False):
                        # 根据 inplace 标志设置相应的字符串后缀和目标字符串
                        inplace_str = "_" if inplace else ""
                        inplace_tgt = "%34" if inplace else "%35"
                        # 构建表示图的字符串，用于创建 Torch 的图对象
                        graph_str = f"""graph(%input.1 : Tensor):
                            %33 : None = prim::Constant()
                            %34 : Tensor = aten::to_mkldnn(%input.1, %33)
                            %35 : Tensor = {mkldnn_opname}{inplace_str}(%34)
                            return ({inplace_tgt})
                        """
                        # 使用 Torch 内置函数解析 IR 表示的图形式字符串
                        g = torch._C.parse_ir(graph_str)
                        # 从图对象创建一个函数对象
                        m = self.createFunctionFromGraph(g)
                        # 创建指定大小的随机输入张量 x
                        x = torch.rand(size)
                        # 进行测试，期望的结果是 PyTorch 函数操作后的密集张量结果
                        # `inplace=False` 是故意的，否则会修改输入，而我们不测试 aten 的实现
                        self.assertEqual(aten_op(x, inplace=False), m(x).to_dense())

    # 如果没有 MKL-DNN 构建，则跳过此测试
    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    # 测试标量乘法的功能
    def test_scalar_mul(self):
        # 将默认数据类型设置为浮点型
        with set_default_dtype(torch.float):

            # 定义一个简单的模型类 Mod
            class Mod(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mod = nn.Conv2d(8, 8, 1, padding=1)

                def forward(self, x):
                    # 进行卷积操作后乘以 4
                    a1 = self.mod(x) * 4
                    # 返回结果再乘以 4 加上另一个乘以 5 的结果
                    return a1 * 4 + a1 * 5.0

            # 创建并冻结模型对象
            mod = Mod().eval()
            scripted = torch.jit.freeze(torch.jit.script(mod))
            optimized = torch.jit.optimize_for_inference(scripted)
            # 创建一个随机输入张量
            inp = torch.rand([1, 8, 8, 8])
            # 使用 FileCheck 检查优化后的图是否包含标量乘法操作
            FileCheck().check("ScalarMul(").check("ScalarMul_").run(optimized.graph)
            # 进行测试，期望的结果是优化后模型与原始模型在随机输入上的输出一致
            self.assertEqual(optimized(inp), mod(inp))

    # 测试去除 detach 操作的功能
    def test_remove_detach(self):
        # 定义一个简单的模型类 Mod
        class Mod(nn.Module):
            def forward(self, x):
                # 对输入 x 执行 detach 操作
                y = x.detach()
                # 返回 y 与自身的乘积
                return y * y

        # 创建并冻结模型对象
        mod = Mod().eval()
        frozen_mod = torch.jit.freeze(torch.jit.script(mod))
        # 创建一个随机输入张量
        inp = torch.randn((2, 2))
        # 使用 FileCheck 检查冻结后图中是否不包含 detach 操作
        FileCheck().check_not("aten::detach").run(frozen_mod.graph)
        # 进行测试，期望的结果是冻结后模型与原始模型在随机输入上的输出一致
        self.assertEqual(frozen_mod(inp), mod(inp))
    def test_remove_detach_not_applied(self):
        # 定义一个测试方法，用于测试 detach 方法未被应用的情况

        class Mod(nn.Module):
            # 定义一个继承自 nn.Module 的模块类 Mod
            def forward(self, x):
                # 模块的前向传播方法，输入参数 x
                y = x.detach()
                # 对输入张量 x 进行 detach 操作，返回一个新的张量 y
                return x is y
                # 返回比较结果，判断 x 是否等于 y

        mod = Mod().eval()
        # 创建 Mod 类的实例 mod，并设为评估模式
        frozen_mod = torch.jit.freeze(torch.jit.script(mod))
        # 使用 Torch 的脚本模式将 mod 冻结成 frozen_mod
        inp = torch.randn((2, 2))
        # 生成一个形状为 (2, 2) 的随机张量 inp
        FileCheck().check("aten::detach").run(frozen_mod.graph)
        # 检查 frozen_mod 的计算图中是否包含 "aten::detach"
        self.assertEqual(frozen_mod(inp), mod(inp))
        # 断言 frozen_mod 对输入 inp 的输出与 mod 对输入 inp 的输出相同
# 根据 TorchDynamo 框架的测试用例跳过测试，因为在 Python 关闭时会出现挂起现象
@skipIfTorchDynamo("somehow causing hanging during python shutdown")
# 如果 MKL-DNN 构建不可用，则跳过测试
@unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled")
# 测试 MKL-DNN 下的模型替换功能，继承 JitTestCase 类
class TestMKLDNNReinplacing(JitTestCase):
    def setUp(self):
        super().setUp()
        # 保存默认的数据类型，并设置默认数据类型为浮点型
        self.default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float)

    def tearDown(self):
        super().tearDown()
        # 在测试完成后恢复默认数据类型
        torch.set_default_dtype(self.default_dtype)

    # 创建并返回一个简单的 Conv2d 模型
    def getConv(self):
        return nn.Conv2d(3, 32, kernel_size=3, stride=2).eval()

    # 创建并返回一个随机输入张量
    def getInput(self):
        return torch.rand([4, 3, 4, 4])

    # 冻结模型并转换为 MKL-DNN，返回转换后的模型
    def freezeAndConvert(self, mod):
        # 冻结模型并用 JIT 脚本化，然后运行指定的转换 pass
        mod = torch.jit.freeze(torch.jit.script(mod.eval()))
        self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
        return mod

    # 检查两个模型在给定输入上的结果是否相等
    def checkResults(self, mod1, mod2):
        inp = self.getInput()
        self.assertEqual(mod1(inp), mod2(inp))

    # 测试简单的 Conv-ReLU 模型转换为 MKL-DNN 是否成功
    def test_successful(self):
        # 创建一个包含 Conv2d、Hardswish 和 ReLU 的 Sequential 模型
        mod_eager = nn.Sequential(self.getConv(), nn.Hardswish(), nn.ReLU())
        # 冻结并转换模型为 MKL-DNN
        mod = self.freezeAndConvert(mod_eager)
        # 检查转换后的模型图中是否包含预期的节点顺序
        FileCheck().check("mkldnn_convolution").check_next(
            "prim::MKLDNNHardSwish_"
        ).check_next("aten::relu_").run(mod.graph)
        # 检查转换前后模型在相同输入上的输出是否一致
        self.checkResults(mod_eager, mod)

    # 测试包含自定义模块 Mod 的模型在 MKL-DNN 下的转换
    def test_merge_liveness(self):
        # 定义一个包含自定义模块 Mod 的类
        class Mod(nn.Module):
            def __init__(self, tensor):
                super().__init__()
                self.tensor = tensor

            def forward(self, x):
                # 因为 x 在此处之后不再使用，所以可以进行 in-place 操作
                temporary = x * self.tensor
                # temporary 的生命周期仅在返回节点之前，因此不能进行 in-place 操作
                return temporary + temporary, temporary

        # 创建一个包含 Conv2d 和 Mod 自定义模块的 Sequential 模型
        mod_eager = nn.Sequential(self.getConv(), Mod(torch.rand([4, 32, 1, 1])))
        # 冻结并转换模型为 MKL-DNN
        mod = self.freezeAndConvert(mod_eager)
        # 检查转换后的模型图中是否包含预期的节点
        FileCheck().check("aten::mul_").check_not("aten::add_").run(mod.graph)
        # 检查转换前后模型在相同输入上的输出是否一致
        self.checkResults(mod_eager, mod)
    def test_always_alive_values(self):
        # 定义一个继承自 nn.Module 的内部类 Mod，接受一个张量 tensor 作为参数
        class Mod(nn.Module):
            def __init__(self, tensor):
                super().__init__()
                # 将参数 tensor 保存在对象的属性中
                self.tensor = tensor

            def forward(self, x):
                # x 无法被就地修改，因为它是一个返回值，
                # 确保就地修改过程不会尝试修改 self.tensor，因为它始终存在
                return x * self.tensor, x

        # 创建一个 nn.Sequential 对象，包含 self.getConv() 的结果和 Mod 类的实例
        mod_eager = nn.Sequential(self.getConv(), Mod(torch.rand([4, 32, 1, 1])))
        # 调用 self.freezeAndConvert 方法，冻结并转换 mod_eager
        mod = self.freezeAndConvert(mod_eager)
        # 使用 FileCheck 检查 mod.graph 中是否不存在 "aten::mul_" 操作
        FileCheck().check_not("aten::mul_").run(mod.graph)
        # 检查转换前后的结果是否一致
        self.checkResults(mod_eager, mod)

        # 获取一个卷积层对象
        conv = self.getConv()

        # 定义一个新的内部类 Mod，继承自 nn.Module
        class Mod(nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化对象的 tensor 属性为一个随机张量
                self.tensor = torch.rand([4, 32, 1, 1])
                # 将 conv 设置为对象的属性
                self.conv = conv

            def forward(self, x):
                # 这里的形状不符合，只是测试一个特定的模式
                conv_output = self.conv(x)
                return conv_output, self.conv(torch.add(x, x))

        # 调用 self.freezeAndConvert 方法，冻结并转换 Mod 类的实例
        mod = self.freezeAndConvert(Mod())
        # 使用 FileCheck 检查 mod.graph 中是否不存在 "aten::add_" 操作
        FileCheck().check_not("aten::add_").run(mod.graph)

    def test_switch_inputs_to_inplace(self):
        # 定义一个继承自 nn.Module 的内部类 Mod，接受一个张量 tensor 作为参数
        class Mod(nn.Module):
            def __init__(self, tensor):
                super().__init__()
                # 将参数 tensor 保存在对象的属性中
                self.tensor = tensor

            def forward(self, x):
                # self.tensor 不能被就地修改，但 x 可以被修改，
                # 因为 add 是可交换的，我们可以反转 add_ 的输入顺序
                return self.tensor + x

        # 创建一个 nn.Sequential 对象，包含 self.getConv() 的结果和 Mod 类的实例
        mod_eager = nn.Sequential(self.getConv(), Mod(torch.rand([4, 32, 1, 1])))
        # 调用 self.freezeAndConvert 方法，冻结并转换 mod_eager
        mod = self.freezeAndConvert(mod_eager)
        # 使用 FileCheck 检查 mod.graph 中是否存在 "aten::add_" 操作
        FileCheck().check("aten::add_").run(mod.graph)
        # 检查转换前后的结果是否一致
        self.checkResults(mod_eager, mod)
```