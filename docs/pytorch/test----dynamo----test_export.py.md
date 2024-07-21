# `.\pytorch\test\dynamo\test_export.py`

```py
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_export_persist_assert)
"""
# 导入必要的库和模块
import copy  # 导入copy模块，用于对象的深复制
import functools  # 导入functools模块，用于高阶函数的操作
import inspect  # 导入inspect模块，用于检查和获取对象的信息
import io  # 导入io模块，用于处理文件流
import operator  # 导入operator模块，提供了Python中所有标准操作符的函数接口
import unittest  # 导入unittest模块，用于编写和运行测试
from enum import Enum  # 导入Enum类，用于创建枚举类型
from typing import Dict, List, Sequence  # 导入类型提示，声明字典、列表和序列类型
from unittest.mock import patch  # 导入patch函数，用于mock测试中的对象

import torch  # 导入PyTorch库
import torch._dynamo  # 导入torch._dynamo模块
import torch._dynamo.test_case  # 导入torch._dynamo.test_case模块
import torch._dynamo.testing  # 导入torch._dynamo.testing模块

from functorch.experimental.control_flow import cond  # 导入条件流控制模块中的cond函数
from torch._dynamo import config  # 导入torch._dynamo中的config模块
from torch._dynamo.exc import UserError  # 导入torch._dynamo.exc中的UserError异常
from torch._dynamo.testing import normalize_gm  # 导入torch._dynamo.testing模块中的normalize_gm函数
from torch._higher_order_ops.out_dtype import out_dtype  # 导入torch._higher_order_ops.out_dtype中的out_dtype函数
from torch._subclasses import fake_tensor  # 导入torch._subclasses中的fake_tensor
from torch.export import dynamic_dim  # 导入torch.export中的dynamic_dim模块
from torch.fx.experimental.proxy_tensor import make_fx  # 导入torch.fx.experimental.proxy_tensor中的make_fx函数
from torch.fx.experimental.symbolic_shapes import (  # 导入torch.fx.experimental.symbolic_shapes中的多个类和异常
    ConstraintViolationError,
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.testing._internal import common_utils  # 导入torch.testing._internal中的common_utils模块
from torch.testing._internal.common_cuda import TEST_CUDA  # 导入torch.testing._internal.common_cuda中的TEST_CUDA常量


class ExportTests(torch._dynamo.test_case.TestCase):
    # TODO(voz): Refactor to a shared test function.
    # The tests in this file are a little redundant,
    # They all take a func, run it with eager, then export it, then compare

    # 定义测试方法，测试函数导出的正确性
    def test_export(self):
        # 定义一个内部函数，模拟注意力机制前的状态操作
        def pre_attention_state_ops(input, mems, state):
            lc_key = state[0]  # 获取状态列表中的第一个张量
            lc_val = state[1]  # 获取状态列表中的第二个张量
            bar = []  # 初始化空列表bar
            for i in range(0, 4):  # 循环4次，i从0到3
                bar2 = []  # 初始化空列表bar2
                for j in range(0, 3):  # 循环3次，j从0到2
                    bar2.append(
                        lc_key + lc_val + torch.tensor([0.1, 0.25, 0.4, 0.5, 0.1])
                    )  # 将张量lc_key、lc_val和一个新张量的和添加到bar2中
                bar.append(bar2)  # 将bar2添加到bar中

            return bar  # 返回bar列表

        # 定义一个主函数func，调用pre_attention_state_ops函数
        def func():
            mems = torch.tensor([[[1.8364, 0.2724, -1.4917, -0.4367, 0.8640]]])  # 定义mems张量
            state = [  # 定义状态列表state，包含两个张量
                torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
                torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
            ]
            i = torch.tensor(  # 定义张量i，包含三个子张量
                [
                    [0.0313, -0.1487, -0.3846, -0.5321],
                    [-1.7073, 1.3331, -0.0890, -1.4935],
                    [-0.8314, -0.1862, -0.5935, 1.5232],
                ]
            )
            return pre_attention_state_ops(i, mems, state)  # 调用pre_attention_state_ops函数

        # 对func函数进行优化，获取优化后的函数opt_func
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func()  # 执行优化后的函数，获取实际结果

        torch._dynamo.reset()  # 重置torch._dynamo环境

        exported = torch._dynamo.export(func)()  # 导出func函数，并执行导出的函数
        out_graph = exported[0]  # 获取导出结果中的第一个图

        dynamo_result = out_graph()  # 执行导出的图，获取动态结果
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))  # 断言优化前后的结果是否相同
    # 定义一个测试函数，用于测试导出时输出不匹配的情况
    def test_export_mismatched_out(self):
        # 定义一个内部函数，对输入参数进行加工并返回
        def func(x):
            y = x + 1
            return ([x, x], (y, y))

        # 使用torch._dynamo.optimize装饰器优化func函数，设置优化选项
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 对实际结果进行计算
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))

        # 重置Torch的动态优化
        torch._dynamo.reset()

        # 导出优化后的func函数
        exported = torch._dynamo.export(func)(torch.tensor([[[1.3737, 0.1]]]))
        # 获取导出的计算图
        out_graph = exported[0]

        # 使用导出的计算图计算结果
        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))

        # 断言实际结果和动态优化结果的相似性
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    # 定义一个测试函数，用于测试带有控制流的导出情况（第一种情况）
    def test_export_shape_control_flow_1(self):
        # 定义一个函数，根据输入参数的形状进行不同的计算
        def func(x):
            if x.shape[0] > 10:
                return x.cos()
            return x.sin()

        # 使用torch._dynamo.optimize装饰器优化func函数，设置优化选项
        opt_func = torch._dynamo.optimize("eager")(func)
        # 对实际结果进行计算
        real_result = opt_func(torch.ones(6, 4))

        # 重置Torch的动态优化
        torch._dynamo.reset()

        # 导出优化后的func函数
        exported = torch._dynamo.export(func)(torch.ones(6, 4))
        # 获取导出的计算图和相关的保护条件
        out_graph, out_guards = exported

        # 使用导出的计算图计算结果
        dynamo_result = out_graph(torch.ones(6, 4))

        # 导入GuardSource类，用于后续的断言检查
        from torch._guards import GuardSource

        # 断言实际结果和动态优化结果的相似性
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
        hit = False
        # 遍历保护条件，检查是否包含与形状相关的保护条件
        for guard in out_guards:
            if guard.source == GuardSource.SHAPE_ENV:
                hit = True
                # 断言保护条件是否符合预期
                self.assertExpectedInline(
                    guard.code_list,
                    """["L['x'].stride()[0] == L['x'].size()[1]", "L['x'].stride()[1] == 1", "L['x'].storage_offset() == 0", "2 <= L['x'].size()[0] <= 10", "2 <= L['x'].size()[1]"]""",
                )
                break

        # 断言是否找到了相关的保护条件
        self.assertTrue(hit)

    # 定义一个测试函数，用于测试带有getattr的控制流导出情况
    def test_export_control_flow_with_getattr(self):
        # 定义一个枚举类Animal
        class Animal(Enum):
            COW = "moo"

        # 定义一个继承自torch.nn.Module的自定义模块MyModule
        class MyModule(torch.nn.Module):
            def __init__(self, a):
                super().__init__()
                self.a = a

            # 定义模块的前向传播方法
            def forward(self, x):
                if self.a == Animal.COW.value:
                    return x * x
                else:
                    raise ValueError("bad")

        # 创建一个MyModule实例
        module = MyModule("moo")
        input = (torch.ones(4, 3),)
        # 计算直接调用模块实例的结果
        resA = module(*input)
        # 导出模块的计算图
        graph, _ = torch._dynamo.export(module)(*input)
        # 计算使用导出的计算图的结果
        resB = graph(*input)
        # 断言直接调用结果和导出计算图结果的相似性
        self.assertTrue(torch._dynamo.utils.same(resA, resB))

    # 定义一个测试函数，用于测试图绕过的导出情况
    def test_export_graph_bypass(self):
        # 创建一个包含多个Tensor的输入列表
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
        ]

        # 定义一个函数，从输入中选择第三个Tensor并返回其平方
        def func(x):
            first = x[2]
            second = x[2]
            return first * second

        # 使用torch._dynamo.optimize装饰器优化func函数，设置优化选项
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 对实际结果进行计算
        real_result = opt_func(inp)

        # 重置Torch的动态优化
        torch._dynamo.reset()

        # 导出优化后的func函数
        exported = torch._dynamo.export(func)(inp)
        # 获取导出的计算图
        out_graph = exported[0]

        # 使用导出的计算图计算结果
        dynamo_result = out_graph(inp)

        # 断言实际结果和动态优化结果的相似性
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
    def test_export_graph_with_list():
        inp = [
            torch.tensor([0.1, 0.1]),   # 创建包含两个元素的张量
            torch.tensor([0.2, 0.2]),   # 创建包含两个元素的张量
            torch.tensor([0.3, 0.3]),   # 创建包含两个元素的张量
            torch.tensor([0.4, 0.4]),   # 创建包含两个元素的张量
        ]

        def func(x):
            first = x[2]    # 获取输入列表中索引为2的张量
            second = x[2]   # 获取输入列表中索引为2的张量
            return first * second, x    # 返回两倍的第三个张量和整个输入列表

        # 对 func 函数进行优化，使用 Torch 的动态优化工具
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 使用优化后的函数处理输入数据
        real_result = opt_func(inp)

        # 重置 Torch 动态优化工具状态
        torch._dynamo.reset()

        # 导出 func 函数的图形化表示，使用 Torch 动态导出工具
        exported = torch._dynamo.export(func)(inp)
        # 获取导出结果的第一个图形模块
        out_graph = exported[0]

        # 使用导出的图形模块处理输入数据
        dynamo_result = out_graph(inp)

        # 断言优化后的结果与导出后的结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
    # 测试导出包含复杂重新排序的图形的功能
    def test_export_graph_with_complex_reorder(self):
        # 输入张量列表
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.4, 0.4]),
        ]

        # 定义内部函数func，接受参数x
        def func(x):
            # 分别获取输入列表x的第一个、第二个和第三个元素
            first = x[0]
            second = x[1]
            third = x[2]
            # 返回顺序为第三个、第一个、第二个、第一个乘以第二个、第一个乘以第三个的结果
            return third, first, second, first * second, first * third

        # 使用torch._dynamo.optimize优化func函数，设置为"eager"模式，不使用Python，启用动态计算
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 计算优化后的func在inp上的结果
        real_result = opt_func(inp)

        # 重置torch._dynamo环境
        torch._dynamo.reset()

        # 导出func函数为计算图形式
        exported = torch._dynamo.export(func)(inp)
        # 获取导出计算图的第一个计算图
        out_graph = exported[0]

        # 在导出的计算图上应用inp，得到结果
        dynamo_result = out_graph(inp)

        # 断言优化前后结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    # 测试包含重复计算结果的功能
    def test_dupes(self):
        # 输入张量
        inp = torch.tensor([0.1, 0.1])

        # 定义内部函数func，接受参数x
        def func(x):
            # 计算x加1的结果，并返回两次
            y = x + 1
            return y, y

        # 使用torch._dynamo.optimize优化func函数，设置为"eager"模式，不使用Python，启用动态计算
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 计算优化后的func在inp上的结果
        real_result = opt_func(inp)

        # 重置torch._dynamo环境
        torch._dynamo.reset()

        # 导出func函数为计算图形式
        exported = torch._dynamo.export(func)(inp)
        # 获取导出计算图的第一个计算图
        out_graph = exported[0]

        # 在导出的计算图上应用inp，得到结果
        dynamo_result = out_graph(inp)

        # 断言优化前后结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    # 测试包含重复计算结果的功能，第二个版本
    def test_dupes_2(self):
        # 输入张量
        inp = torch.tensor([0.1, 0.1])

        # 定义内部函数func，接受参数x
        def func(x):
            # 计算x加1的结果，并返回两次
            y = x + 1
            return y, y

        # 使用torch._dynamo.optimize优化func函数，设置为"eager"模式，不使用Python，启用动态计算
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 计算优化后的func在inp上的结果
        real_result = opt_func(inp)

        # 重置torch._dynamo环境
        torch._dynamo.reset()

        # 导出func函数为计算图形式
        exported = torch._dynamo.export(func)(inp)
        # 获取导出计算图的第一个计算图
        out_graph = exported[0]

        # 在导出的计算图上应用inp，得到结果
        dynamo_result = out_graph(inp)

        # 断言优化前后结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    # 测试包含重复计算结果和绕过的功能
    def test_dupes_and_bypass(self):
        # 输入张量
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.4, 0.4])
        inps = [inp, inp2]

        # 定义内部函数func，接受参数x和z
        def func(x, z):
            # 计算x加1的结果，并返回两次，同时将z添加到结果中
            y = x + 1
            return y, y, z

        # 使用torch._dynamo.optimize优化func函数，设置为"eager"模式，不使用Python，启用动态计算
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 计算优化后的func在inps上的结果
        real_result = opt_func(*inps)

        # 重置torch._dynamo环境
        torch._dynamo.reset()

        # 导出func函数为计算图形式
        exported = torch._dynamo.export(func)(*inps)
        # 获取导出计算图的第一个计算图
        out_graph = exported[0]

        # 在导出的计算图上应用inps，得到结果
        dynamo_result = out_graph(*inps)

        # 断言优化前后结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    # 测试包含重复计算结果、绕过及非张量参数的功能
    def test_dupes_and_bypass_with_non_tensor_arg(self):
        # 输入张量
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        # 定义内部函数func，接受参数x、z和k
        def func(x, z, k):
            # 计算x加k的结果，并返回两次，同时将z添加到结果中
            y = x + k
            return y, y, z

        # 使用torch._dynamo.optimize优化func函数，设置为"eager"模式，不使用Python，启用动态计算
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 计算优化后的func在inps上的结果
        real_result = opt_func(*inps)

        # 重置torch._dynamo环境
        torch._dynamo.reset()

        # 导出func函数为计算图形式
        exported = torch._dynamo.export(func)(*inps)
        # 获取导出计算图的第一个计算图
        out_graph = exported[0]

        # 在导出的计算图上应用inps，得到结果
        dynamo_result = out_graph(*inps)

        # 断言优化前后结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
    def test_dupes_and_bypass_reorder_with_non_tensor_arg(self):
        inp = torch.tensor([0.1, 0.1])  # 创建一个包含两个浮点数的张量 inp
        inp2 = torch.tensor([0.1, 0.1])  # 创建另一个包含两个浮点数的张量 inp2
        inp3 = 4  # 创建一个标量 inp3
        inps = [inp, inp2, inp3]  # 将这些输入组合成列表 inps

        def func(x, z, k):
            y = x + k  # 计算 x 和 k 的和，并将结果赋给 y
            return z, y, y  # 返回 z、y 和 y 的元组

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)  # 优化 func 函数
        real_result = opt_func(*inps)  # 使用 inps 调用优化后的 func，并获得真实结果

        torch._dynamo.reset()  # 重置 torch._dynamo 的状态

        exported = torch._dynamo.export(func)(*inps)  # 导出经过优化后的 func
        out_graph = exported[0]  # 获得导出结果的第一个元素

        dynamo_result = out_graph(*inps)  # 使用导出的 func 执行 inps，并获得动态图结果

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))  # 断言真实结果和动态图结果相同

    @config.patch(capture_scalar_outputs=True)
    def test_dupes_and_bypass_with_non_tensor_output(self):
        inp = torch.tensor([0.1, 0.1])  # 创建一个包含两个浮点数的张量 inp
        inp2 = torch.tensor([0.1, 0.1])  # 创建另一个包含两个浮点数的张量 inp2
        inp3 = 4  # 创建一个标量 inp3
        inps = [inp, inp2, inp3]  # 将这些输入组合成列表 inps

        def func(x, z, k):
            y = x + k  # 计算 x 和 k 的和，并将结果赋给 y
            return y[0].item(), y, z  # 返回 y 的第一个元素的值，y 本身和 z 的元组

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)  # 优化 func 函数
        real_result = opt_func(*inps)  # 使用 inps 调用优化后的 func，并获得真实结果

        torch._dynamo.reset()  # 重置 torch._dynamo 的状态

        exported = torch._dynamo.export(func)(*inps)  # 导出经过优化后的 func
        out_graph = exported[0]  # 获得导出结果的第一个元素

        dynamo_result = out_graph(*inps)  # 使用导出的 func 执行 inps，并获得动态图结果

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))  # 断言真实结果和动态图结果相同

    def test_zeroes_in_and_out_different_shape_on_test(self):
        inp = torch.zeros(10)  # 创建一个包含 10 个零的张量 inp
        inp2 = torch.zeros(10)  # 创建另一个包含 10 个零的张量 inp2
        inp3 = torch.zeros(10)  # 创建另一个包含 10 个零的张量 inp3
        inps = [inp, inp2, inp3]  # 将这些输入组合成列表 inps

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]  # 创建三个包含随机数的张量列表 inps_rand

        def func(a, b, c):
            return [[a], [b, c], [a + b], [[c + c]]]  # 返回多层嵌套的列表

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)  # 优化 func 函数
        real_result = opt_func(*inps_rand)  # 使用 inps_rand 调用优化后的 func，并获得真实结果

        torch._dynamo.reset()  # 重置 torch._dynamo 的状态

        exported = torch._dynamo.export(func)(*inps)  # 导出经过优化后的 func
        out_graph = exported[0]  # 获得导出结果的第一个元素

        dynamo_result = out_graph(*inps_rand)  # 使用导出的 func 执行 inps_rand，并获得动态图结果

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))  # 断言真实结果和动态图结果相同

    @config.patch(capture_scalar_outputs=True)
    def test_zeroes_in_new_shape_scalar_out(self):
        inp = torch.zeros(10)  # 创建一个包含 10 个零的张量 inp
        inp2 = torch.zeros(10)  # 创建另一个包含 10 个零的张量 inp2
        inp3 = torch.zeros(10)  # 创建另一个包含 10 个零的张量 inp3
        inps = [inp, inp2, inp3]  # 将这些输入组合成列表 inps

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]  # 创建三个包含随机数的张量列表 inps_rand

        def func(a, b, c):
            return a[0].item() + b[0].item() + c[0].item()  # 返回三个张量第一个元素的和

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)  # 优化 func 函数
        real_result = opt_func(*inps_rand)  # 使用 inps_rand 调用优化后的 func，并获得真实结果

        torch._dynamo.reset()  # 重置 torch._dynamo 的状态

        exported = torch._dynamo.export(func)(*inps)  # 导出经过优化后的 func
        out_graph = exported[0]  # 获得导出结果的第一个元素

        dynamo_result = out_graph(*inps_rand)  # 使用导出的 func 执行 inps_rand，并获得动态图结果

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))  # 断言真实结果和动态图结果相同

    @config.patch(capture_scalar_outputs=True)
    # 测试函数：验证在新形状下零张量输出和置换的情况
    def test_zeroes_in_new_shape_scalar_out_permute(self):
        # 创建一个大小为10的零张量
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        # 将零张量放入列表中
        inps = [inp, inp2, inp3]

        # 创建一个包含随机张量的列表
        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        # 定义一个函数，对输入进行操作并返回结果
        def func(a, b, c):
            # 返回输入张量的第一个元素的和
            return b[0].item() + c[0].item() + a[0].item() + a[0].item()

        # 优化函数，使用动态编译器进行优化
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 计算优化后函数在随机输入上的结果
        real_result = opt_func(*inps_rand)

        # 重置动态编译器状态
        torch._dynamo.reset()

        # 导出优化后的函数并获取其计算图
        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        # 使用导出的计算图在随机输入上计算结果
        dynamo_result = out_graph(*inps_rand)

        # 断言优化前后的结果是否相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    # 使用配置修饰器，捕获标量输出的情况
    @config.patch(capture_scalar_outputs=True)
    def test_zeroes_in_new_shape_scalar_out_permute_dupe_and_bypass(self):
        # 创建一个大小为10的零张量
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        # 将零张量放入列表中
        inps = [inp, inp2, inp3]

        # 创建一个包含随机张量的列表
        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        # 定义一个函数，对输入进行操作并返回多个结果
        def func(a, b, c):
            return a, b[0].item() + c[0].item() + a[0].item() + a[0].item(), a

        # 优化函数，使用动态编译器进行优化
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 计算优化后函数在随机输入上的结果
        real_result = opt_func(*inps_rand)

        # 重置动态编译器状态
        torch._dynamo.reset()

        # 导出优化后的函数并获取其计算图
        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        # 使用导出的计算图在随机输入上计算结果
        dynamo_result = out_graph(*inps_rand)

        # 断言优化前后的结果是否相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    # 测试函数：验证函数返回值情况
    def test_func_return(self):
        # 创建一个大小为10的零张量
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        # 将零张量放入列表中
        inps = [inp, inp2, inp3]

        # 创建一个包含随机张量的列表
        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        # 定义一个函数，对输入进行操作并返回结果
        def func(a, b, c):
            x = a + b + c

            # 定义内部函数，对参数进行操作并返回结果
            def func2(y):
                return x * y

            # 返回内部函数对参数x的操作结果
            return func2(x)

        # 优化函数，使用动态编译器进行优化
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 计算优化后函数在随机输入上的结果
        real_result = opt_func(*inps_rand)

        # 重置动态编译器状态
        torch._dynamo.reset()

        # 导出优化后的函数并获取其计算图
        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        # 使用导出的计算图在随机输入上计算结果
        dynamo_result = out_graph(*inps_rand)

        # 断言优化前后的结果是否相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    # 测试函数：验证字典返回值情况
    def test_dict_return(self):
        # 创建一个大小为10的零张量
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        # 将零张量放入列表中
        inps = [inp, inp2, inp3]

        # 创建一个包含随机张量的列表
        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        # 定义一个函数，对输入进行操作并返回包含结果的字典
        def func(a, b, c):
            x = a + b + c
            return {"a": x}

        # 优化函数，使用动态编译器进行优化
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 计算优化后函数在随机输入上的结果
        real_result = opt_func(*inps_rand)

        # 重置动态编译器状态
        torch._dynamo.reset()

        # 导出优化后的函数并获取其计算图
        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        # 使用导出的计算图在随机输入上计算结果
        dynamo_result = out_graph(*inps_rand)

        # 断言优化前后的结果是否相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
    def test_export_with_aten_graph(self):
        # 定义一个函数，该函数用于在计算前处理注意力状态操作
        def pre_attention_state_ops(input, mems, state):
            # 获取注意力键和值的状态
            lc_key = state[0]
            lc_val = state[1]
            # 初始化一个空列表
            bar = []
            # 循环遍历范围为0到4
            for i in range(0, 4):
                # 初始化一个空列表
                bar2 = []
                # 循环遍历范围为0到3
                for j in range(0, 3):
                    # 向bar2列表添加lc_key、lc_val和torch张量[0.1, 0.25, 0.4, 0.5, 0.1]的和
                    bar2.append(
                        lc_key + lc_val + torch.tensor([0.1, 0.25, 0.4, 0.5, 0.1])
                    )
                # 向bar列表添加bar2列表
                bar.append(bar2)

            # 返回bar列表作为结果
            return bar

        # 定义一个函数func，用于执行计算和操作
        def func():
            # 初始化mems张量
            mems = torch.tensor([[[1.8364, 0.2724, -1.4917, -0.4367, 0.8640]]])
            # 初始化state张量列表
            state = [
                torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
                torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
            ]
            # 初始化i张量
            i = torch.tensor(
                [
                    [0.0313, -0.1487, -0.3846, -0.5321],
                    [-1.7073, 1.3331, -0.0890, -1.4935],
                    [-0.8314, -0.1862, -0.5935, 1.5232],
                ]
            )
            # 调用pre_attention_state_ops函数，并返回结果
            return pre_attention_state_ops(i, mems, state)

        # 对func函数进行优化和编译
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 调用优化后的func函数，获取真实结果
        real_result = opt_func()

        # 重置torch._dynamo的状态
        torch._dynamo.reset()

        # 导出func函数的计算图，包括ATen操作
        exported = torch._dynamo.export(func, aten_graph=True)()
        # 获取导出计算图的第一个节点
        out_graph = exported[0]

        # 执行导出的计算图，获取计算结果
        dynamo_result = out_graph()
        # 断言真实结果与导出结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_mismatched_out_with_aten_graph(self):
        # 定义一个函数func，接受输入x并执行计算
        def func(x):
            # 计算y为x加1
            y = x + 1
            # 返回一个元组，包含两个列表[x, x]和元组(y, y)
            return ([x, x], (y, y))

        # 对func函数进行优化和编译
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 调用优化后的func函数，传入输入数据，并获取真实结果
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))

        # 重置torch._dynamo的状态
        torch._dynamo.reset()

        # 导出func函数的计算图，包括ATen操作，传入输入数据
        exported = torch._dynamo.export(func, aten_graph=True)(
            torch.tensor([[[1.3737, 0.1]]])
        )
        # 获取导出计算图的第一个节点
        out_graph = exported[0]

        # 执行导出的计算图，传入输入数据，获取计算结果
        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))

        # 断言真实结果与导出结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_bypass_with_aten_graph(self):
        # 初始化输入张量列表inp
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
        ]

        # 定义一个函数func，接受输入x并执行计算
        def func(x):
            # 将第三个张量赋值给first和second
            first = x[2]
            second = x[2]
            # 返回first和second的乘积
            return first * second

        # 对func函数进行优化和编译
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 调用优化后的func函数，传入inp张量列表，并获取真实结果
        real_result = opt_func(inp)

        # 重置torch._dynamo的状态
        torch._dynamo.reset()

        # 导出func函数的计算图，包括ATen操作，传入inp张量列表
        exported = torch._dynamo.export(func, aten_graph=True)(inp)
        # 获取导出计算图的第一个节点
        out_graph = exported[0]

        # 执行导出的计算图，传入inp张量列表，获取计算结果
        dynamo_result = out_graph(inp)

        # 断言真实结果与导出结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
    def test_list_unpack_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),  # 创建包含两个浮点数的张量
            torch.tensor([0.2, 0.2]),  # 创建包含两个浮点数的张量
            torch.tensor([0.3, 0.3]),  # 创建包含两个浮点数的张量
        ]

        def func(x):
            first = x[2]  # 获取输入列表中索引为2的元素
            second = x[2]  # 获取输入列表中索引为2的元素
            return x[0], first * second, x[1], x[2]  # 返回输入列表的第0、1、2个元素及它们索引为2的乘积

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)  # 优化函数 `func`，使用了 Torch 的动态图优化功能
        real_result = opt_func(inp)  # 应用优化后的函数 `func` 到输入 `inp` 上得到结果

        torch._dynamo.reset()  # 重置 Torch 动态图的状态

        exported = torch._dynamo.export(func, aten_graph=True)(inp)  # 导出函数 `func` 的 ATen 图表示，并应用到输入 `inp` 上
        out_graph = exported[0]  # 获取导出的 ATen 图

        dynamo_result = out_graph(inp)  # 应用导出的 ATen 图到输入 `inp` 上得到结果

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))  # 断言优化前后的结果一致

    def test_export_mismatched_out_2_with_aten_graph(self):
        def func(x):
            y = x + 1  # 张量 `x` 中的每个元素加1
            return ([x, x], (y, y))  # 返回一个列表，包含两个相同的张量 `x`，以及一个元组，包含两个相同的张量 `y`

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)  # 优化函数 `func`
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))  # 应用优化后的函数 `func` 到指定张量上得到结果

        torch._dynamo.reset()  # 重置 Torch 动态图的状态

        exported = torch._dynamo.export(func, aten_graph=True)(
            torch.tensor([[[1.3737, 0.1]]])  # 导出函数 `func` 的 ATen 图表示，并应用到指定张量上
        )
        out_graph = exported[0]  # 获取导出的 ATen 图

        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))  # 应用导出的 ATen 图到指定张量上得到结果

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))  # 断言优化前后的结果一致

    def test_export_graph_with_list_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),  # 创建包含两个浮点数的张量
            torch.tensor([0.2, 0.2]),  # 创建包含两个浮点数的张量
            torch.tensor([0.3, 0.3]),  # 创建包含两个浮点数的张量
            torch.tensor([0.4, 0.4]),  # 创建包含两个浮点数的张量
        ]

        def func(x):
            first = x[2]  # 获取输入列表中索引为2的元素
            second = x[2]  # 获取输入列表中索引为2的元素
            return first * second, x  # 返回输入列表索引为2的元素的乘积及整个输入列表

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)  # 优化函数 `func`
        real_result = opt_func(inp)  # 应用优化后的函数 `func` 到输入 `inp` 上得到结果

        torch._dynamo.reset()  # 重置 Torch 动态图的状态

        exported = torch._dynamo.export(func, aten_graph=True)(inp)  # 导出函数 `func` 的 ATen 图表示，并应用到输入 `inp` 上
        out_graph = exported[0]  # 获取导出的 ATen 图

        dynamo_result = out_graph(inp)  # 应用导出的 ATen 图到输入 `inp` 上得到结果

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))  # 断言优化前后的结果一致

    def test_export_graph_with_complex_reorder_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),  # 创建包含两个浮点数的张量
            torch.tensor([0.2, 0.2]),  # 创建包含两个浮点数的张量
            torch.tensor([0.3, 0.3]),  # 创建包含两个浮点数的张量
            torch.tensor([0.4, 0.4]),  # 创建包含两个浮点数的张量
        ]

        def func(x):
            first = x[0]  # 获取输入列表中索引为0的元素
            second = x[1]  # 获取输入列表中索引为1的元素
            third = x[2]  # 获取输入列表中索引为2的元素
            return third, first, second, first * second, first * third  # 返回输入列表索引为2、0、1的元素及它们索引为0、1的乘积

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)  # 优化函数 `func`
        real_result = opt_func(inp)  # 应用优化后的函数 `func` 到输入 `inp` 上得到结果

        torch._dynamo.reset()  # 重置 Torch 动态图的状态

        exported = torch._dynamo.export(func, aten_graph=True)(inp)  # 导出函数 `func` 的 ATen 图表示，并应用到输入 `inp` 上
        out_graph = exported[0]  # 获取导出的 ATen 图

        dynamo_result = out_graph(inp)  # 应用导出的 ATen 图到输入 `inp` 上得到结果

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))  # 断言优化前后的结果一致
    # 测试函数，用于测试在生成 ATen 图时处理重复和绕过的情况
    def test_dupes_with_aten_graph(self):
        # 创建输入张量
        inp = torch.tensor([0.1, 0.1])

        # 定义函数 func，对输入进行加法运算并返回结果
        def func(x):
            y = x + 1
            return y, y

        # 使用 torch._dynamo.optimize 进行优化，生成优化后的函数 opt_func
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 调用优化后的函数，获取真实结果 real_result
        real_result = opt_func(inp)

        # 重置 dynamo 的状态
        torch._dynamo.reset()

        # 导出函数 func 的 ATen 图形式
        exported = torch._dynamo.export(func, aten_graph=True)(inp)
        out_graph = exported[0]

        # 使用 ATen 图执行函数，获取动态图计算结果 dynamo_result
        dynamo_result = out_graph(inp)

        # 断言真实结果和动态图计算结果是否相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    # 类似上一个测试函数，测试处理重复和绕过的情况
    def test_dupes_2_with_aten_graph(self):
        # 创建输入张量
        inp = torch.tensor([0.1, 0.1])

        # 定义函数 func，对输入进行加法运算并返回结果
        def func(x):
            y = x + 1
            return y, y

        # 使用 torch._dynamo.optimize 进行优化，生成优化后的函数 opt_func
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 调用优化后的函数，获取真实结果 real_result
        real_result = opt_func(inp)

        # 重置 dynamo 的状态
        torch._dynamo.reset()

        # 导出函数 func 的 ATen 图形式
        exported = torch._dynamo.export(func, aten_graph=True)(inp)
        out_graph = exported[0]

        # 使用 ATen 图执行函数，获取动态图计算结果 dynamo_result
        dynamo_result = out_graph(inp)

        # 断言真实结果和动态图计算结果是否相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    # 测试函数，同时处理重复和绕过情况，带有非张量参数
    def test_dupes_and_bypass_with_aten_graph(self):
        # 创建输入张量
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.4, 0.4])
        inps = [inp, inp2]

        # 定义函数 func，对输入进行加法运算并返回结果，同时保留非张量参数
        def func(x, z):
            y = x + 1
            return y, y, z

        # 使用 torch._dynamo.optimize 进行优化，生成优化后的函数 opt_func
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 调用优化后的函数，获取真实结果 real_result
        real_result = opt_func(*inps)

        # 重置 dynamo 的状态
        torch._dynamo.reset()

        # 导出函数 func 的 ATen 图形式
        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        # 使用 ATen 图执行函数，获取动态图计算结果 dynamo_result
        dynamo_result = out_graph(*inps)

        # 断言真实结果和动态图计算结果是否相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    # 测试函数，同时处理重复和绕过情况，带有非张量参数
    def test_dupes_and_bypass_with_non_tensor_arg_with_aten_graph(self):
        # 创建输入张量和非张量参数
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        # 定义函数 func，对输入进行加法运算并返回结果，同时保留非张量参数
        def func(x, z, k):
            y = x + k
            return y, y, z

        # 使用 torch._dynamo.optimize 进行优化，生成优化后的函数 opt_func
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 调用优化后的函数，获取真实结果 real_result
        real_result = opt_func(*inps)

        # 重置 dynamo 的状态
        torch._dynamo.reset()

        # 导出函数 func 的 ATen 图形式
        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        # 使用 ATen 图执行函数，获取动态图计算结果 dynamo_result
        dynamo_result = out_graph(*inps)

        # 断言真实结果和动态图计算结果是否相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    # 测试函数，同时处理重复和绕过情况，带有非张量参数且参数顺序不同
    def test_dupes_and_bypass_reorder_with_non_tensor_arg_with_aten_graph(self):
        # 创建输入张量和非张量参数
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        # 定义函数 func，对输入进行加法运算并返回结果，同时保留非张量参数
        def func(x, z, k):
            y = x + k
            return z, y, y

        # 使用 torch._dynamo.optimize 进行优化，生成优化后的函数 opt_func
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 调用优化后的函数，获取真实结果 real_result
        real_result = opt_func(*inps)

        # 重置 dynamo 的状态
        torch._dynamo.reset()

        # 导出函数 func 的 ATen 图形式
        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        # 使用 ATen 图执行函数，获取动态图计算结果 dynamo_result
        dynamo_result = out_graph(*inps)

        # 断言真实结果和动态图计算结果是否相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
    @config.patch(capture_scalar_outputs=True)
    # 使用装饰器 `config.patch`，设置捕获标量输出为 True
    def test_dupes_and_bypass_with_non_tensor_output_with_aten_graph(self):
        # 定义输入张量和非张量变量
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            # 在函数内部进行张量操作
            y = x + k
            return y[0].item(), y, z

        # 优化函数 `func`，启用 eager 模式、关闭 JIT 编译、启用动态分析
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 调用优化后的函数，获取真实结果
        real_result = opt_func(*inps)

        # 重置 torch._dynamo 状态
        torch._dynamo.reset()

        # 导出优化后的函数，使用 ATen 图模式
        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        # 在导出的图上进行计算
        dynamo_result = out_graph(*inps)

        # 断言优化前后的结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_zeroes_in_and_out_different_shape_on_test_with_aten_graph(self):
        # 定义输入张量
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        # 随机生成不同的输入张量
        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            # 在函数内部进行复杂的张量操作
            return [[a], [b, c], [a + b], [[c + c]]]

        # 优化函数 `func`，启用 eager 模式、关闭 JIT 编译、启用动态分析
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 调用优化后的函数，获取真实结果
        real_result = opt_func(*inps_rand)

        # 重置 torch._dynamo 状态
        torch._dynamo.reset()

        # 导出优化后的函数，使用 ATen 图模式
        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        # 在导出的图上进行计算
        dynamo_result = out_graph(*inps_rand)

        # 断言优化前后的结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_func_return_with_aten_graph(self):
        # 定义输入张量
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        # 随机生成不同的输入张量
        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            # 在函数内部进行复杂的张量操作和嵌套函数调用
            x = a + b + c

            def func2(y):
                return x * y

            return func2(x)

        # 优化函数 `func`，启用 eager 模式、关闭 JIT 编译、启用动态分析
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 调用优化后的函数，获取真实结果
        real_result = opt_func(*inps_rand)

        # 重置 torch._dynamo 状态
        torch._dynamo.reset()

        # 导出优化后的函数，使用 ATen 图模式
        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        # 在导出的图上进行计算
        dynamo_result = out_graph(*inps_rand)

        # 断言优化前后的结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dict_return_with_aten_graph(self):
        # 定义输入张量
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        # 随机生成不同的输入张量
        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            # 在函数内部进行复杂的张量操作并返回字典
            x = a + b + c
            return {"a": x}

        # 优化函数 `func`，启用 eager 模式、关闭 JIT 编译、启用动态分析
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 调用优化后的函数，获取真实结果
        real_result = opt_func(*inps_rand)

        # 重置 torch._dynamo 状态
        torch._dynamo.reset()

        # 导出优化后的函数，使用 ATen 图模式
        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        # 在导出的图上进行计算
        dynamo_result = out_graph(*inps_rand)

        # 断言优化前后的结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
    # 定义一个测试方法，用于测试导出时是否包含堆栈跟踪信息
    def test_export_with_stack_trace(self):
        # 创建一个 4x4 的随机张量作为输入
        inp = torch.randn(4, 4)

        # 定义一个继承自 torch.nn.Module 的子类 MyBlock
        class MyBlock(torch.nn.Module):
            # 重写 forward 方法
            def forward(self, x):
                # 对输入 x 进行线性变换，并应用余弦和 ReLU 激活函数
                x = torch.nn.functional.linear(x, torch.randn(4, 4))
                return torch.cos(x).relu() + 1

        # 定义一个继承自 torch.nn.Module 的子类 MyModule
        class MyModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建 MyBlock 类的实例并赋值给 self.block
                self.block = MyBlock()

            # 重写 forward 方法
            def forward(self, x):
                # 将输入 x 经过 self.block 处理得到输出 out
                out = self.block(x)
                return out

        # 使用 torch._dynamo.export 方法导出 MyModule 的模型，并禁用 ATen 图的生成
        exported = torch._dynamo.export(MyModule(), aten_graph=False)(inp)
        # 获取导出结果中的第一个图
        out_graph = exported[0]

        # 遍历图中的节点
        for node in out_graph.graph.nodes:
            # 如果节点的操作不是占位符或输出
            if node.op not in {"placeholder", "output"}:
                # 断言节点的堆栈跟踪信息不为空
                self.assertTrue(node.stack_trace is not None)
                # 断言节点的 nn_module_stack 元数据不为空
                self.assertTrue(node.meta["nn_module_stack"] is not None)
                # 断言节点的 source_fn_stack 元数据不为空
                self.assertTrue(node.meta["source_fn_stack"] is not None)

        # 重置 torch._dynamo 的状态
        torch._dynamo.reset()

        # 使用 torch._dynamo.export 方法再次导出 MyModule 的模型，并启用 ATen 图的生成
        exported = torch._dynamo.export(MyModule(), aten_graph=True)(inp)
        # 获取导出结果中的第一个图
        out_graph = exported[0]

        # 再次遍历图中的节点
        for node in out_graph.graph.nodes:
            # 如果节点的操作是调用函数
            if node.op == "call_function":
                # 断言节点的堆栈跟踪信息不为空
                self.assertTrue(node.stack_trace is not None)
                # 断言节点的 nn_module_stack 元数据不为空
                self.assertTrue(node.meta["nn_module_stack"] is not None)
                # 断言节点的 source_fn_stack 元数据不为空
                self.assertTrue(node.meta["source_fn_stack"] is not None)
                # 断言节点的 val 元数据不为空
                self.assertTrue(node.meta["val"] is not None)
                # 断言节点的 original_aten 元数据不为空
                self.assertTrue(node.meta["original_aten"] is not None)
    # 定义一个测试方法，用于验证导出时保留 nn.Module 栈以进行 get_attr 操作
    def test_export_preserves_nn_module_stack_for_get_attr(self):
        # 创建一个形状为 (4, 4) 的随机张量 inp
        inp = torch.randn(4, 4)

        # 定义一个自定义的 nn.Module，包含权重参数和缓冲区
        class MyBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, 1))  # 定义权重参数
                self.register_buffer("buffer", torch.ones(1, 1))    # 注册一个缓冲区

            def forward(self, x):
                x = torch.nn.functional.linear(x, torch.randn(4, 4))  # 执行线性变换
                return torch.cos(x).relu() + self.weight + self.buffer  # 返回计算结果

        # 定义一个包含 MyBlock 的 nn.Module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = MyBlock()

            def forward(self, x):
                out = self.block(x)  # 前向传播过程中使用 MyBlock
                return out

        m = MyModule()  # 创建 MyModule 实例
        exported = torch._dynamo.export(m, aten_graph=False)(inp)  # 导出模型图，不使用 Aten 格式
        out_graph = exported[0]  # 取得导出的图数据

        attr_access_count = 0  # 初始化属性访问计数器
        # 遍历导出的图中的节点
        for node in out_graph.graph.nodes:
            if node.op == "get_attr":  # 如果节点操作是 "get_attr"
                attr_access_count += 1  # 增加属性访问计数
                self.assertTrue(node.meta["nn_module_stack"] is not None)  # 断言 nn.Module 栈不为空
        self.assertEqual(attr_access_count, 2)  # 断言属性访问计数为 2

        torch._dynamo.reset()  # 重置 Torch Dynamo

        exported = torch._dynamo.export(m, aten_graph=True)(inp)  # 使用 Aten 格式导出模型
        out_graph = exported[0]  # 取得导出的图数据

        attr_access_count = 0  # 重新初始化属性访问计数器
        # 遍历导出的图中的节点
        for node in out_graph.graph.nodes:
            if node.op == "get_attr":  # 如果节点操作是 "get_attr"
                attr_access_count += 1  # 增加属性访问计数
                self.assertTrue(node.meta["nn_module_stack"] is not None)  # 断言 nn.Module 栈不为空
        self.assertEqual(attr_access_count, 2)  # 断言属性访问计数为 2

    # 定义一个测试方法，用于比较通过 optimize 和 make_fx 进行的导出
    def test_export_compare_optimize_with_make_fx(self):
        inp = torch.tensor([0.1, 0.1])  # 创建一个张量 inp

        linear = torch.nn.Linear(2, 2)  # 创建一个线性层

        # 定义一个函数 func，包含张量操作序列
        def func(x):
            x = x + 1  # 加法操作
            y = x.t()  # 转置操作
            y = y.relu()  # ReLU 激活函数
            y = linear(y)  # 线性层操作
            return y  # 返回结果张量

        # 使用 Aten 格式导出函数 func
        exported = torch._dynamo.export(func, aten_graph=True)(inp)
        out_graph = exported[0]  # 取得导出的图数据
        export_result = out_graph(inp)  # 在导出的图上执行计算

        torch._dynamo.reset()  # 重置 Torch Dynamo

        # 定义一个编译器函数，用于将函数 func 编译为优化后的函数 opt_func
        def compiler(gm, sample_inputs):
            def fw(*args):
                aten_gm = make_fx(gm)(*args)  # 使用 make_fx 转换为 Aten 格式的图
                return aten_gm(*args)  # 在 Aten 图上执行计算

            return fw

        # 使用 optimize 函数优化 func，启用 nopython 和 dynamic 选项
        opt_func = torch._dynamo.optimize(compiler, nopython=True, dynamic=True)(func)
        make_fx_result_through_backend = opt_func(inp)  # 在优化后的函数上执行计算

        fx_g = make_fx(func)(inp)  # 使用 make_fx 直接生成图
        make_fx_result_through_direct = fx_g(inp)  # 在直接生成的图上执行计算

        # 断言 optimize 后的结果与 Aten 格式导出的结果相同
        self.assertTrue(
            torch._dynamo.utils.same(make_fx_result_through_backend, export_result)
        )
        # 断言 make_fx 生成的结果与 Aten 格式导出的结果相同
        self.assertTrue(
            torch._dynamo.utils.same(make_fx_result_through_direct, export_result)
        )
    def test_export_with_constant_method_on_module(self):
        # 定义一个名为 test_export_with_constant_method_on_module 的测试函数
        class MyModule(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的类 MyModule
            def __init__(self):
                # 初始化方法
                super().__init__()
                # 调用父类的初始化方法
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                # 创建一个形状为 (4, 2) 的参数张量，并包装成模型参数
                self.linear = torch.nn.Linear(2, 2)
                # 创建一个线性层，输入维度为 2，输出维度为 2

            @torch._dynamo.assume_constant_result
            # 使用 assume_constant_result 装饰器，假设其结果为常量
            def helper_fn(self, x):
                # 定义一个辅助函数 helper_fn，接受输入 x
                return torch.nonzero(x)
                # 返回 x 中非零元素的索引位置

            def forward(self, x):
                # 前向传播方法，接受输入 x
                y = torch.sin(x)
                # 对输入 x 求正弦
                x = self.linear(x)
                # 将 x 输入线性层进行计算
                y = self.helper_fn(x)
                # 调用 helper_fn 处理 x，并赋值给 y
                return y
                # 返回处理后的结果 y

        module = MyModule()
        # 创建 MyModule 类的实例 module
        real_result = module(torch.tensor([[1.0, 0], [0, 0]]))
        # 使用 module 对象进行前向传播，输入张量为 [[1.0, 0], [0, 0]]
        module = MyModule()
        # 再次创建 MyModule 类的实例 module
        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        # 导出 module 的计算图，并将计算图应用于输入张量 [[0.0, 0], [0, 0]]，获取计算图 graph
        result = graph(torch.tensor([[1.0, 0.0], [0, 0]]))
        # 使用计算图 graph 进行前向传播，输入张量为 [[1.0, 0.0], [0, 0]]
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        # 断言计算图 graph 的输出与直接使用 module 得到的 real_result 相同
        result = graph(torch.tensor([[1, 0], [0.25, 0.25]]))
        # 使用计算图 graph 进行前向传播，输入张量为 [[1, 0], [0.25, 0.25]]
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        # 再次断言计算图 graph 的输出与直接使用 module 得到的 real_result 相同

    def test_export_with_constant_method_on_module_invoke_twice(self):
        # 定义一个名为 test_export_with_constant_method_on_module_invoke_twice 的测试函数
        class MyModule(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的类 MyModule
            def __init__(self):
                # 初始化方法
                super().__init__()
                # 调用父类的初始化方法
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                # 创建一个形状为 (4, 2) 的参数张量，并包装成模型参数
                self.linear = torch.nn.Linear(2, 2)
                # 创建一个线性层，输入维度为 2，输出维度为 2

            @torch._dynamo.assume_constant_result
            # 使用 assume_constant_result 装饰器，假设其结果为常量
            def helper_fn(self, x):
                # 定义一个辅助函数 helper_fn，接受输入 x
                return torch.nonzero(x)
                # 返回 x 中非零元素的索引位置

            def forward(self, x):
                # 前向传播方法，接受输入 x
                y = torch.sin(x)
                # 对输入 x 求正弦
                x = self.linear(x)
                # 将 x 输入线性层进行计算
                y = self.helper_fn(x) + self.helper_fn(x)
                # 调用 helper_fn 处理 x 两次，并将结果相加赋值给 y
                return y
                # 返回处理后的结果 y

        module = MyModule()
        # 创建 MyModule 类的实例 module
        real_result = module(torch.tensor([[1.0, 0], [0, 0]]))
        # 使用 module 对象进行前向传播，输入张量为 [[1.0, 0], [0, 0]]
        module = MyModule()
        # 再次创建 MyModule 类的实例 module
        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        # 导出 module 的计算图，并将计算图应用于输入张量 [[0.0, 0], [0, 0]]，获取计算图 graph
        result = graph(torch.tensor([[1.0, 0.0], [0, 0]]))
        # 使用计算图 graph 进行前向传播，输入张量为 [[1.0, 0.0], [0, 0]]
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        # 断言计算图 graph 的输出与直接使用 module 得到的 real_result 相同
        result = graph(torch.tensor([[1, 0], [0.25, 0.25]]))
        # 使用计算图 graph 进行前向传播，输入张量为 [[1, 0], [0.25, 0.25]]
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        # 再次断言计算图 graph 的输出与直接使用 module 得到的 real_result 相同
    def test_export_with_constant_free_function(self):
        # 定义一个装饰器，用于标记函数调用的结果为常量
        @torch._dynamo.assume_constant_result
        # 定义一个辅助函数，接受输入 x，返回非零元素的索引
        def helper_fn(x):
            return torch.nonzero(x)

        # 定义一个继承自 torch.nn.Module 的类 MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个参数 param，形状为 (4, 2)
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                # 定义一个线性层，输入维度为 2，输出维度为 2
                self.linear = torch.nn.Linear(2, 2)

            @torch._dynamo.assume_constant_result
            # 类内部定义的辅助函数，接受输入 x，返回非零元素的索引
            def helper_fn(self, x):
                return torch.nonzero(x)

            # 前向传播函数，接受输入 x
            def forward(self, x):
                # 计算 x 的正弦值
                y = torch.sin(x)
                # 将 x 传入线性层计算
                x = self.linear(x)
                # 调用外部定义的 helper_fn 函数和类内部定义的 helper_fn 函数，将结果相加
                y = helper_fn(x) + self.helper_fn(x)
                return y

        # 创建 MyModule 类的实例 module
        module = MyModule()
        # 使用实例 module 对输入张量 [[1.0, 0], [0, 0]] 进行前向传播，得到真实结果 real_result
        real_result = module(torch.tensor([[1.0, 0], [0, 0]]))
        # 重新创建 MyModule 类的实例 module
        module = MyModule()
        # 导出 module 对象的计算图及相关信息，返回图 graph 和其他信息（这里使用 _ 代表未使用的返回值）
        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        # 使用导出的计算图 graph 对输入张量 [[1.0, 0.0], [0, 0]] 进行计算，得到结果 result
        result = graph(torch.tensor([[1.0, 0.0], [0, 0]]))
        # 断言 result 与 real_result 相同
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        # 使用导出的计算图 graph 对输入张量 [[1, 0], [0.25, 0.25]] 进行计算，得到结果 result
        result = graph(torch.tensor([[1, 0], [0.25, 0.25]]))
        # 断言 result 与 real_result 相同
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_free_function_and_class_method(self):
        # 定义一个装饰器，用于标记函数调用的结果为常量
        @torch._dynamo.assume_constant_result
        # 定义一个辅助函数，接受输入 x，返回非零元素的索引
        def helper_fn(x):
            return torch.nonzero(x)

        # 定义一个继承自 torch.nn.Module 的类 MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个参数 param，形状为 (4, 2)
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                # 定义一个线性层，输入维度为 2，输出维度为 2
                self.linear = torch.nn.Linear(2, 2)

            # 前向传播函数，接受输入 x
            def forward(self, x):
                # 计算 x 的正弦值
                y = torch.sin(x)
                # 将 x 传入线性层计算
                x = self.linear(x)
                # 调用外部定义的 helper_fn 函数，将结果作为返回值
                y = helper_fn(x)
                return y

        # 创建 MyModule 类的实例 module
        module = MyModule()
        # 使用实例 module 对输入张量 [[1.0, 0], [0, 0]] 进行前向传播，得到真实结果 real_result
        real_result = module(torch.tensor([[1.0, 0], [0, 0]]))
        # 重新创建 MyModule 类的实例 module
        module = MyModule()
        # 导出 module 对象的计算图及相关信息，返回图 graph 和其他信息（这里使用 _ 代表未使用的返回值）
        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        # 使用导出的计算图 graph 对输入张量 [[1.0, 0.0], [0, 0]] 进行计算，得到结果 result
        result = graph(torch.tensor([[1.0, 0.0], [0, 0]]))
        # 断言 result 与 real_result 相同
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        # 使用导出的计算图 graph 对输入张量 [[1, 0], [0.25, 0.25]] 进行计算，得到结果 result
        result = graph(torch.tensor([[1, 0], [0.25, 0.25]]))
        # 断言 result 与 real_result 相同
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
    def test_export_with_constant_free_function_and_class_method_multiarg(self):
        # 定义一个装饰器函数，假设其返回值为常量
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            # 返回输入张量中非零元素的索引
            return torch.nonzero(x)

        # 定义一个继承自 torch.nn.Module 的子类 MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个形状为 (4, 2) 的参数张量
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                # 初始化一个线性层，输入维度为 2，输出维度为 2
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x, z):
                # 对输入张量 x 中的每个元素计算正弦值
                y = torch.sin(x)
                # 将输入张量 x 传入线性层进行计算
                x = self.linear(x)
                # 调用 helper_fn 函数分别对 x 和 z 计算非零元素的索引并求和
                y = helper_fn(x) + helper_fn(z)
                return y

        # 创建 MyModule 类的实例 module
        module = MyModule()
        # 使用实例 module 计算真实结果
        real_result = module(
            torch.tensor([[1.0, 0], [0, 0]]), torch.tensor([[1.0, 0], [0, 0]])
        )
        # 再次创建 MyModule 类的实例 module
        module = MyModule()
        # 导出 MyModule 实例的图形表示和导出函数，返回的是图形和另外一个值
        graph, _ = torch._dynamo.export(module)(
            torch.tensor([[0.0, 0], [0, 0]]), torch.tensor([[1.0, 0], [0, 0]])
        )
        # 使用导出的图形计算结果
        result = graph(
            torch.tensor([[1.0, 0.0], [0, 0]]), torch.tensor([[1.0, 0.0], [0, 0]])
        )
        # 断言两个张量是否相同
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        # 使用导出的图形计算结果
        result = graph(
            torch.tensor([[1, 0], [0.25, 0.25]]), torch.tensor([[1, 0], [0.25, 0.25]])
        )
        # 断言两个张量是否相同
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_free_function_and_class_method_multiarg_diff(self):
        # 定义一个装饰器函数，假设其返回值为常量
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            # 返回输入张量中非零元素的索引
            return torch.nonzero(x)

        # 定义一个继承自 torch.nn.Module 的子类 MyModule
        class MyModule(torch.nn.Module):
            def forward(self, x, z):
                # 调用 helper_fn 函数分别对 x 和 z 计算非零元素的索引并求和
                y = helper_fn(x) + helper_fn(z)
                return y

        # 创建 MyModule 类的实例 module
        module = MyModule()
        # 使用实例 module 计算真实结果
        real_result = module(
            torch.tensor([[1.0, 0], [0, 0]]), torch.tensor([[1.0, 0], [0, 0]])
        )
        # 再次创建 MyModule 类的实例 module
        module = MyModule()
        # 导出 MyModule 实例的图形表示和导出函数，返回的是图形和另外一个值
        graph, _ = torch._dynamo.export(module)(
            torch.tensor([[0.0, 0], [0, 0]]), torch.tensor([[0.0, 0], [0.5, 0]])
        )
        # 使用导出的图形计算结果
        result = graph(
            torch.tensor([[1.0, 0.0], [0, 0]]), torch.tensor([[0.0, 1.0], [0, 0]])
        )
        # 断言两个张量是否相同
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        # 使用导出的图形计算结果
        result = graph(
            torch.tensor([[1, 0], [0.25, 0.25]]),
            torch.tensor([[0.33, 0.33], [0.25, 0.25]]),
        )
        # 断言两个张量是否相同
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
    def test_export_with_constant_tuple_nonzero(self):
        class MyModule(torch.nn.Module):
            # 使用装饰器指示结果为常量
            @torch._dynamo.assume_constant_result
            # 定义辅助函数，接受输入 x，返回 torch.nonzero(x) 元组两次
            def helper_fn(self, x):
                return (torch.nonzero(x), torch.nonzero(x))

            # 前向传播函数，接受输入 x
            def forward(self, x):
                # 创建张量 y，包含值 0.5
                y = torch.tensor([0.5])
                # 调用 helper_fn 函数，将结果赋给 elements
                elements = self.helper_fn(x)
                # 初始化空列表 all_y
                all_y = []
                # 遍历 elements 中的每个元素
                for element in elements:
                    # 遍历 element 中的每个项
                    for item in element:
                        # 计算 y 与 item 的乘积，并将结果添加到 all_y 中
                        all_y.append(y * item)
                # 返回 all_y 列表
                return all_y

        # 创建 MyModule 类的实例 module
        module = MyModule()
        # 使用输入张量 [1.0, 1.0] 调用 module 实例的前向传播方法，将结果赋给 real_result
        real_result = module(torch.tensor([1.0, 1.0]))
        # 导出 module 实例，并将结果分别赋给 graph 和 guards
        graph, guards = torch._dynamo.export(module)(torch.tensor([1.0, 1.0]))

        # Tensor 输入几乎可以是任何内容，在这里捕获编译时设为常量的结果
        # 使用张量输入 [[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]] 调用 graph，将结果赋给 result
        result = graph(torch.tensor([[[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]]]))
        # 断言 result 与 real_result 相同，使用 torch._dynamo.utils.same 函数
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_list_nonzero(self):
        class MyModule(torch.nn.Module):
            # 使用装饰器指示结果为常量
            @torch._dynamo.assume_constant_result
            # 定义辅助函数，接受输入 x，返回 torch.nonzero(x) 列表两次
            def helper_fn(self, x):
                return [torch.nonzero(x), torch.nonzero(x)]

            # 前向传播函数，接受输入 x
            def forward(self, x):
                # 创建张量 y，包含值 0.5
                y = torch.tensor([0.5])
                # 调用 helper_fn 函数，将结果赋给 elements
                elements = self.helper_fn(x)
                # 初始化空列表 all_y
                all_y = []
                # 遍历 elements 中的每个元素
                for element in elements:
                    # 遍历 element 中的每个项
                    for item in element:
                        # 计算 y 与 item 的乘积，并将结果添加到 all_y 中
                        all_y.append(y * item)
                # 返回 all_y 列表
                return all_y

        # 创建 MyModule 类的实例 module
        module = MyModule()
        # 使用输入张量 [1.0, 1.0] 调用 module 实例的前向传播方法，将结果赋给 real_result
        real_result = module(torch.tensor([1.0, 1.0]))
        # 导出 module 实例，并将结果分别赋给 graph 和 guards
        graph, guards = torch._dynamo.export(module)(torch.tensor([1.0, 1.0]))

        # Tensor 输入几乎可以是任何内容，在这里捕获编译时设为常量的结果
        # 使用张量输入 [[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]] 调用 graph，将结果赋给 result
        result = graph(torch.tensor([[[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]]]))
        # 断言 result 与 real_result 相同，使用 torch._dynamo.utils.same 函数
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_list_nonzero_free_function(self):
        # 使用装饰器指示结果为常量
        @torch._dynamo.assume_constant_result
        # 定义辅助函数，接受输入 x，返回 torch.nonzero(x) 列表两次
        def helper_fn(x):
            return [torch.nonzero(x), torch.nonzero(x)]

        class MyModule(torch.nn.Module):
            # 前向传播函数，接受输入 x
            def forward(self, x):
                # 创建张量 y，包含值 0.5
                y = torch.tensor([0.5])
                # 调用 helper_fn 函数，将结果赋给 elements
                elements = helper_fn(x)
                # 初始化空列表 all_y
                all_y = []
                # 遍历 elements 中的每个元素
                for element in elements:
                    # 遍历 element 中的每个项
                    for item in element:
                        # 计算 y 与 item 的乘积，并将结果添加到 all_y 中
                        all_y.append(y * item)
                # 返回 all_y 列表
                return all_y

        # 创建 MyModule 类的实例 module
        module = MyModule()
        # 使用输入张量 [1.0, 1.0] 调用 module 实例的前向传播方法，将结果赋给 real_result
        real_result = module(torch.tensor([1.0, 1.0]))
        # 导出 module 实例，并将结果分别赋给 graph 和 guards
        graph, guards = torch._dynamo.export(module)(torch.tensor([1.0, 1.0]))

        # Tensor 输入几乎可以是任何内容，在这里捕获编译时设为常量的结果
        # 使用张量输入 [[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]] 调用 graph，将结果赋给 result
        result = graph(torch.tensor([[[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]]]))
        # 断言 result 与 real_result 相同，使用 torch._dynamo.utils.same 函数
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
    def test_export_with_constant_dict_values(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                # 定义一个辅助函数，假定其返回结果在编译时是常量
                return {"x": x, "x^2": x * x}

            def forward(self, x):
                y = torch.tensor([0.5])
                # 调用辅助函数获取结果字典，并使用其中的 'x' 键对应的值
                elements = self.helper_fn(x)
                y = y * elements["x"]
                # 继续使用结果字典中的 'x^2' 键对应的值
                y = y * elements["x^2"]
                return y

        module = MyModule()
        # 使用实际输入计算预期结果
        real_result = module(torch.tensor([2.0, 2.0]))
        # 导出模块并获取计算图及其守卫条件
        graph, guards = torch._dynamo.export(module)(torch.tensor([2.0, 2.0]))

        # 在这里，张量输入可以是几乎任何值，结果会捕捉编译时确定的常量
        result = graph(torch.tensor([[[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]]]))
        # 断言导出的结果与实际结果相同
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_none_control_flow(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                # 如果输入的 x 小于 0，则返回 None，否则返回 x 本身
                if x.item() < 0:
                    return None
                else:
                    return x

            def forward(self, x):
                y = torch.tensor([0.5])
                # 调用辅助函数获取结果
                x = self.helper_fn(x)
                # 如果结果为 None，则直接返回 y
                if x is None:
                    return y
                # 否则返回 y 乘以 x
                return y * x

        module = MyModule()
        # 使用实际输入计算预期结果
        real_result = module(torch.tensor([-1]))

        # X 为负数，所以 .item() < 0，意味着我们返回 y
        self.assertEqual(real_result, torch.tensor([0.5]))

        # 导出模块并获取计算图及其守卫条件
        graph, guards = torch._dynamo.export(module)(torch.tensor([-1]))
        result = graph(torch.tensor([2]))
        # X 为正数，但我们编译了 helper_fn 返回 None，所以仍然返回 y
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_not_none_control_flow(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                # 如果输入的 x 小于 0，则返回 None，否则返回 x 本身
                if x.item() < 0:
                    return None
                else:
                    return x

            def forward(self, x):
                y = torch.tensor([0.5])
                # 调用辅助函数获取结果
                x = self.helper_fn(x)
                # 如果结果为 None，则直接返回 y
                if x is None:
                    return y
                # 否则返回 y 乘以 x
                return y * x

        module = MyModule()
        # 使用实际输入计算预期结果
        real_result = module(torch.tensor([2]))

        # X 为正数，所以 .item() > 0，意味着我们返回 y * x
        self.assertEqual(real_result, torch.tensor([1.0]))

        # 导出模块并获取计算图及其守卫条件
        graph, guards = torch._dynamo.export(module)(torch.tensor([2]))
        result = graph(torch.tensor([-0.5]))
        # X 为负数，但我们编译了 helper_fn 返回 x，所以仍然返回 y * x
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
    def test_export_with_constant_none_control_flow_free_func(self):
        # 定义一个装饰器，假定其修饰的函数返回值是常量
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            # 如果 x 的值小于 0，返回 None
            if x.item() < 0:
                return None
            else:
                # 否则返回 x 的值
                return x

        # 定义一个继承自 torch.nn.Module 的类
        class MyModule(torch.nn.Module):
            # 实现 forward 方法，处理输入数据 x
            def forward(self, x):
                # 创建一个张量 y，值为 0.5
                y = torch.tensor([0.5])
                # 使用 helper_fn 处理输入 x
                x = helper_fn(x)
                # 如果 x 为 None，则返回张量 y
                if x is None:
                    return y
                # 否则返回 y 乘以 x
                return y * x

        # 创建 MyModule 类的实例
        module = MyModule()
        # 对输入张量 [-1] 进行 forward 操作，得到实际结果 real_result
        real_result = module(torch.tensor([-1]))

        # 断言：由于 x 是负数，所以 .item() < 0，返回 y，即 [0.5]
        self.assertEqual(real_result, torch.tensor([0.5]))

        # 导出 MyModule 类，并获取导出图和保护条件
        graph, guards = torch._dynamo.export(module)(torch.tensor([-1]))
        # 使用导出的图对输入张量 [2] 进行计算，得到 result
        result = graph(torch.tensor([2]))
        # 断言：虽然 x 是正数，但是我们编译了 helper_fn 使其返回 None，因此仍然返回 y
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_not_none_control_flow_pos(self):
        # 定义一个继承自 torch.nn.Module 的类
        class MyModule(torch.nn.Module):
            # 定义一个装饰器，假定其修饰的方法返回值是常量
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                # 如果 x 的值小于 0，返回 None
                if x.item() < 0:
                    return None
                else:
                    # 否则返回 x 的值
                    return x

            # 实现 forward 方法，处理输入数据 x
            def forward(self, x):
                # 创建一个张量 y，值为 0.5
                y = torch.tensor([0.5])
                # 使用 helper_fn 处理输入 x
                x = self.helper_fn(x)
                # 如果 x 为 None，则返回张量 y
                if x is None:
                    return y
                # 否则返回 y 乘以 x
                return y * x

        # 创建 MyModule 类的实例
        module = MyModule()
        # 对输入张量 [2] 进行 forward 操作，得到实际结果 real_result
        real_result = module(torch.tensor([2]))

        # 断言：由于 x 是正数，所以 .item() > 0，返回 y * x，即 [1.0]
        self.assertEqual(real_result, torch.tensor([1.0]))

        # 导出 MyModule 类，并获取导出图和保护条件
        graph, guards = torch._dynamo.export(module)(torch.tensor([2]))
        # 使用导出的图对输入张量 [-0.5] 进行计算，得到 result
        result = graph(torch.tensor([-0.5]))
        # 断言：虽然 x 是负数，但是我们编译了 helper_fn 使其返回 x，因此仍然返回 y * x
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_not_none_control_flow_free_func(self):
        # 定义一个装饰器，假定其修饰的函数返回值是常量
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            # 如果 x 的值小于 0，返回 None
            if x.item() < 0:
                return None
            else:
                # 否则返回 x 的值
                return x

        # 定义一个继承自 torch.nn.Module 的类
        class MyModule(torch.nn.Module):
            # 实现 forward 方法，处理输入数据 x
            def forward(self, x):
                # 创建一个张量 y，值为 0.5
                y = torch.tensor([0.5])
                # 使用 helper_fn 处理输入 x
                x = helper_fn(x)
                # 如果 x 为 None，则返回张量 y
                if x is None:
                    return y
                # 否则返回 y 乘以 x
                return y * x

        # 创建 MyModule 类的实例
        module = MyModule()
        # 对输入张量 [2] 进行 forward 操作，得到实际结果 real_result
        real_result = module(torch.tensor([2]))

        # 断言：由于 x 是正数，所以 .item() > 0，返回 y * x，即 [1.0]
        self.assertEqual(real_result, torch.tensor([1.0]))

        # 导出 MyModule 类，并获取导出图和保护条件
        graph, guards = torch._dynamo.export(module)(torch.tensor([2]))
        # 使用导出的图对输入张量 [-0.5] 进行计算，得到 result
        result = graph(torch.tensor([-0.5]))
        # 断言：虽然 x 是负数，但是我们编译了 helper_fn 使其返回 x，因此仍然返回 y * x
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
    # 定义一个测试方法，验证带有常量假设的导出行为，确保不返回常量
    def test_export_with_constant_not_return_const(self):
        # 定义一个继承自 torch.nn.Module 的子类 MyModule
        class MyModule(torch.nn.Module):
            # 声明一个装饰器，假设 helper_fn 返回常量结果
            @torch._dynamo.assume_constant_result
            # 定义一个辅助方法 helper_fn，返回实例变量 self.val 的值
            def helper_fn(self, x):
                return self.val

            # 重写父类的 forward 方法
            def forward(self, x):
                # 创建一个 tensor y，值为 [0.5]
                y = torch.tensor([0.5])
                # 调用 helper_fn 方法获取结果，并赋给 x
                x = self.helper_fn(x)
                # 如果 x 等于 "A"，返回 tensor y
                if x == "A":
                    return y
                # 否则返回 -1
                return -1

        # 创建 MyModule 的实例 module
        module = MyModule()
        # 设置实例变量 module.val 为 "A"
        module.val = "A"
        # 对 module 输入 tensor [2]，获取结果 resA
        resA = module(torch.tensor([2]))
        # 导出 module 的计算图和守卫条件，并输入 tensor [2]，得到 graph 和 guards
        graph, guards = torch._dynamo.export(module)(torch.tensor([2]))
        # 设置 module.val 为 "B"
        module.val = "B"
        # 对 graph 输入 tensor [2]，获取结果 resB
        resB = graph(torch.tensor([2]))
        # 断言 resA 和 resB 的结果相同
        self.assertTrue(torch._dynamo.utils.same(resA, resB))

    # 定义一个测试方法，验证带有内置操作的常量假设导出行为
    def test_export_with_builtin_op_on_assume_constant(self):
        # 声明一个装饰器，假设 get_y 返回常量结果
        @torch._dynamo.assume_constant_result
        # 定义一个函数 get_y，接受参数 y，返回 y
        def get_y(y) -> torch.Tensor:
            return y

        # 定义一个继承自 torch.nn.Module 的类 Bob
        class Bob(torch.nn.Module):
            # 初始化方法，接受参数 p 和 val
            def __init__(self, p, val) -> None:
                super().__init__()
                # 创建一个参数 self.p，并赋值为 p
                self.p = p
                # 创建一个参数 self.y，并赋值为 tensor(val)
                self.y = torch.nn.Parameter(torch.tensor(val))

            # 重写父类的 forward 方法
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 如果 get_y 返回的 self.y 小于 self.p
                if get_y(self.y) < self.p:
                    # 返回 tensor x 和 x 的拼接结果
                    return torch.cat([x, x])
                else:
                    # 否则返回 tensor x
                    return x

        # 创建 Bob 类的实例 model，参数为 0.5 和 0.3
        model = Bob(0.5, 0.3)
        # 创建一个全为 1 的 tensor 输入 inp
        inp = torch.ones(3, 4)
        # 导出 model 的计算图和守卫条件，并输入 inp，得到 graph 和 guards
        graph, guards = torch._dynamo.export(model)(inp)
        # 断言 model 对输入 inp 和 graph 对输入 inp 的结果相同
        self.assertEqual(model(inp), graph(inp))

    # 定义一个测试方法，验证非专门化 nn.Module 中的常量假设导出行为
    def test_export_with_constant_in_unspecialized_nn_module(self):
        # 定义一个继承自 torch.nn.Module 的类 Module
        class Module(torch.nn.Module):
            # 初始化方法，接受参数 y
            def __init__(self, y):
                super().__init__()
                # 创建一个实例变量 self.y，并赋值为 y
                self.y = y

            # 声明一个装饰器，假设 check 返回常量结果
            @torch._dynamo.assume_constant_result
            # 定义一个检查方法 check
            def check(self):
                # 返回 self.y 的第一个元素是否等于 1 的布尔值
                return self.y[0].item() == 1

            # 重写父类的 forward 方法
            def forward(self, x):
                # 这一行导致模块对象被追踪为 dynamo 中的 UnspecializedNNModuleVariable
                self.device = x.device

                # 如果 check 方法返回真
                if self.check():
                    # 返回 tensor x + 1
                    return x + 1
                else:
                    # 否则返回 tensor x + 2
                    return x + 2

        # 创建 Module 类的实例 model，参数为 tensor([1])
        model = Module(torch.tensor([1]))
        # 创建一个全为 1 的 tensor 输入 inp
        inp = torch.ones(3, 4)
        # 导出 model 的计算图和守卫条件，并输入 inp，得到 graph 和 _
        graph, _ = torch._dynamo.export(model)(inp)
        # 断言 model 对输入 inp 和 graph 对输入 inp 的结果相同
        self.assertEqual(model(inp), graph(inp))
    def test_export_decomp(self):
        # 定义函数 f，对输入张量 x 进行转置并返回其自身加转置的结果
        def f(x):
            return x.t() + x.t()

        # 定义函数 nop，对输入张量 x 执行余弦函数操作并返回结果
        def nop(x):
            return x.cos()

        # 调用 torch._dynamo.export 导出函数 f，生成图形表示，使用 ATen 操作图，并提供自定义的分解表
        graph, _ = torch._dynamo.export(
            f,
            aten_graph=True,
            decomposition_table={torch.ops.aten.t.default: nop},
        )(torch.randn(5))
        # 断言图中不包含 torch.ops.aten.t.default 目标的节点数量为 0
        self.assertEqual(
            len([n for n in graph.graph.nodes if n.target == torch.ops.aten.t.default]),
            0,
        )

        # 再次调用 torch._dynamo.export 导出函数 f，生成图形表示，使用 ATen 操作图，并不使用分解表
        graph, _ = torch._dynamo.export(f, aten_graph=True, decomposition_table=None)(
            torch.randn(5)
        )
        # 断言图中包含 torch.ops.aten.t.default 目标的节点数量为 2
        self.assertEqual(
            len([n for n in graph.graph.nodes if n.target == torch.ops.aten.t.default]),
            2,
        )

    def test_export_decomp_asserts_bad_args(self):
        # 定义函数 f，对输入张量 x 进行转置并返回其自身加转置的结果
        def f(x):
            return x.t() + x.t()

        # 定义函数 nop，对输入张量 x 执行余弦函数操作并返回结果
        def nop(x):
            return x.cos()

        # 使用断言检查在错误参数下调用 torch._dynamo.export 是否会引发 AssertionError
        with self.assertRaises(AssertionError):
            graph, _ = torch._dynamo.export(
                f,
                (torch.randn(5)),
                aten_graph=False,
                decomposition_table={torch.ops.aten.t.default: nop},
            )

    @config.patch(capture_scalar_outputs=True)
    def test_export_with_module_layer(self):
        # 导入 cond 函数用于条件执行
        from functorch.experimental.control_flow import cond

        # 定义一个继承自 torch.nn.Module 的模块 Module
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            # 定义模块的前向传播函数，根据 pred 条件选择执行不同的 true_fn 或 false_fn 函数
            def forward(self, pred, x):
                def true_fn(val):
                    return self.linear(val) * torch.tensor(2)

                def false_fn(val):
                    return self.linear(val) * torch.tensor(-1)

                # 调用 cond 函数根据 pred 条件执行 true_fn 或 false_fn
                return cond(pred, true_fn, false_fn, [x])

        # 创建 Module 类的实例 mod
        mod = Module()
        # 生成一个形状为 [3, 3] 的随机张量 x
        x = torch.randn([3, 3])
        # 从张量 x 的第一个元素判断 pred 的值，作为条件值
        pred = torch.tensor(x[0][0].item() < 0)
        # 计算模块前向传播的真实结果
        real_result = mod.forward(pred, x)

        # 重置 dynamo 的状态
        torch._dynamo.reset()

        # 导出模块前向传播函数并获取输出图形表示
        exported = torch._dynamo.export(mod.forward)(pred, x)
        out_graph = exported[0]

        # 在输出图形表示上执行 dynamo_result 函数并断言其与真实结果 real_result 相同
        dynamo_result = out_graph(pred, x)
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

        # 创建一个新的张量 x，以展示未特化的情况
        x = x * -1
        # 从张量 x 的第一个元素判断 pred 的值，作为条件值
        pred = torch.tensor(x[0][0].item() < 0)
        # 计算模块前向传播的第二个真实结果
        real_result_2 = mod.forward(pred, x)
        # 在输出图形表示上再次执行 dynamo_result 函数并断言其与真实结果 real_result_2 相同
        dynamo_result_2 = out_graph(pred, x)
        self.assertTrue(torch._dynamo.utils.same(real_result_2, dynamo_result_2))
    def test_export_with_cond_branches_calling_methods(self):
        # 导入必要的模块和函数
        from functorch.experimental.control_flow import cond

        # 定义一个继承自 torch.nn.Module 的类 Module
        class Module(torch.nn.Module):
            # 初始化函数，继承自父类的初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个线性层，输入维度为3，输出维度为3
                self.linear = torch.nn.Linear(3, 3)

            # 方法 t，返回输入值加1
            def t(self, val):
                return val + 1

            # 方法 f，返回输入值减1
            def f(self, val):
                return val - 1

            # 方法 true_fn，根据输入值进行计算返回结果，调用了 linear 方法和 t 方法
            def true_fn(self, val):
                return self.linear(val) + self.t(val)

            # 方法 false_fn，根据输入值进行计算返回结果，调用了 linear 方法和 f 方法
            def false_fn(self, val):
                return self.linear(val) - self.f(val)

            # 前向传播方法，根据条件 pred 调用 true_fn 或 false_fn 方法
            def forward(self, pred, x):
                return cond(pred, self.true_fn, self.false_fn, [x])

        # 创建 Module 类的实例 mod
        mod = Module()
        # 创建一个形状为 [3, 3] 的随机张量 x
        x = torch.randn([3, 3])
        # 根据张量 x 的某个元素是否小于0创建一个布尔张量 pred
        pred = torch.tensor(x[0][0].item() < 0)
        # 调用 Module 实例的前向传播方法，获取真实结果 real_result
        real_result = mod.forward(pred, x)
        # 使用 torch._dynamo.export 导出前向传播方法，获取输出图和状态
        out_graph, _ = torch._dynamo.export(mod.forward)(pred, x)
        # 使用导出的输出图计算结果 dynamo_result
        dynamo_result = out_graph(pred, x)
        # 断言真实结果与 dynamo_result 是否相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    # 使用 config.patch 装饰器配置标量输出捕获为真
    @config.patch(capture_scalar_outputs=True)
    def test_export_with_cond_closure(self):
        # 导入必要的模块和函数
        from functorch.experimental.control_flow import cond

        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 初始化函数，继承自父类的初始化方法
            def __init__(self):
                super().__init__()

            # 前向传播方法，根据条件 pred 调用 true_fn 或 false_fn 方法
            def forward(self, pred, x):
                # 定义 true_fn 方法，对输入 x 进行乘2操作
                def true_fn(x):
                    return x * 2

                # 定义 false_fn 方法，对输入 x 进行减2操作
                def false_fn(x):
                    return x - 2

                return cond(pred, true_fn, false_fn, [x])

        # 定义一个继承自 torch.nn.Module 的类 Bar
        class Bar(torch.nn.Module):
            # 初始化函数，继承自父类的初始化方法
            def __init__(self):
                super().__init__()

            # 前向传播方法，根据条件 pred 调用 true_fn 或 false_fn 方法
            def forward(self, pred, x):
                # 定义 true_fn 方法，对输入 x 进行乘2操作
                def true_fn(x):
                    return x * 2

                # 定义 false_fn 方法，对输入 x 进行减2操作
                def false_fn(x):
                    return x - 2

                return cond(pred, true_fn, false_fn, [x + 1])

        # 定义一个继承自 torch.nn.Module 的类 FooBar
        class FooBar(torch.nn.Module):
            # 初始化函数，继承自父类的初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个线性层，输入维度为3，输出维度为3
                self.linear = torch.nn.Linear(3, 3)

            # 前向传播方法，根据条件 pred 调用 true_fn 或 false_fn 方法
            def forward(self, pred, x):
                # 对输入 x 进行加法操作，得到张量 y
                y = x + x

                # 定义 true_fn 方法，对输入 x 和 y 进行特定计算操作
                def true_fn(x, y):
                    return self.linear(x) * (x + y)

                # 定义 false_fn 方法，对输入 x 和 y 进行特定计算操作
                def false_fn(x, y):
                    return x * (y - x)

                return cond(pred, true_fn, false_fn, [x, y])

        # 遍历类列表 [Foo, Bar, FooBar]
        for Module in [Foo, Bar, FooBar]:
            # 创建 Module 类的实例 mod
            mod = Module()
            # 创建一个形状为 [3, 3] 的随机张量 x，并设置 requires_grad=True
            x = torch.randn([3, 3], requires_grad=True)
            # 根据张量 x 的某个元素是否小于0创建一个布尔张量 pred
            pred = torch.tensor(x[0][0].item() < 0)
            # 调用 Module 实例的前向传播方法，获取真实结果 real_result
            real_result = mod.forward(pred, x)
            # 使用 torch._dynamo.export 导出前向传播方法，获取输出图和状态
            out_graph, _ = torch._dynamo.export(mod.forward)(pred, x)
            # 使用导出的输出图计算结果 dynamo_result
            dynamo_result = out_graph(pred, x)
            # 断言真实结果与 dynamo_result 是否相同
            self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
    # 定义一个测试函数，用于测试在闭包函数内使用条件控制流
    def test_export_with_cond_with_closed_function(self):
        # 定义一个闭包函数hello，返回参数x加1
        def hello(x):
            return x + 1

        # 定义一个闭包函数hi，返回参数x加2
        def hi(x):
            return x + 2

        # 定义一个函数foo，接受一个条件pred和一个参数x
        def foo(pred, x):
            # 定义一个内部函数true_fn，调用hello函数处理参数x
            def true_fn(x):
                return hello(x)

            # 定义一个内部函数false_fn，调用hi函数处理参数x
            def false_fn(x):
                return hi(x)

            # 调用cond函数根据pred的真假选择调用true_fn或false_fn，并传递参数x
            return cond(pred, true_fn, false_fn, [x])

        # 生成一个包含5个随机数的张量x
        x = torch.randn(5)
        # 计算pred为x的第一个元素是否大于0
        pred = x[0] > 0
        # 调用foo函数，获取真实结果real_result
        real_result = foo(pred, x)
        # 导出foo函数并执行，获取输出图和guards
        out_graph, _ = torch._dynamo.export(foo)(pred, x)
        # 在导出的图上执行，获取dynamo_result
        dynamo_result = out_graph(pred, x)
        # 断言real_result与dynamo_result是否相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    # 定义一个测试函数，用于测试在动态形状条件下使用条件控制流
    def test_export_with_cond_dynamic_shape_pred(self):
        # 导入cond函数，来自functorch.experimental.control_flow
        from functorch.experimental.control_flow import cond

        # 定义一个继承自torch.nn.Module的类Module
        class Module(torch.nn.Module):
            # 实现forward方法，接受输入张量x
            def forward(self, x):
                # 定义一个内部函数true_fn，返回输入张量x加上自身
                def true_fn(x):
                    return x + x

                # 定义一个内部函数false_fn，返回输入张量x的前两行
                def false_fn(x):
                    return x[:2]

                # 调用cond函数根据输入张量x的行数是否小于等于2来选择执行true_fn或false_fn，并传递参数[x]
                return cond(x.shape[0] <= 2, true_fn, false_fn, [x])

        # 定义一个继承自torch.nn.Module的类Module2
        class Module2(torch.nn.Module):
            # 实现forward方法，接受输入张量x
            def forward(self, x):
                # 定义一个内部函数true_fn，返回输入张量x加上自身
                def true_fn(x):
                    return x + x

                # 定义一个内部函数false_fn，返回输入张量x的前两行
                def false_fn(x):
                    return x[:2]

                # 调用cond函数根据输入张量x的行数是否小于等于2来选择执行true_fn或false_fn，并传递参数(x,)
                return cond(x.shape[0] <= 2, true_fn, false_fn, (x,))

        # 创建Module和Module2类的实例列表mods
        mods = [Module(), Module2()]
        # 遍历mods列表中的每个模块
        for mod in mods:
            # 生成一个2x2的随机张量x
            x = torch.randn(2, 2)
            # 导出mod模块并执行，获取输出图和guards
            out_graph, guards = torch._dynamo.export(mod)(x)
            # 断言输出图的代码是否与预期一致
            self.assertExpectedInline(
                out_graph.code.strip(),
                """\
def forward(self, x):
    # 将输入 x 包装成二元组，并按照指定规范展平
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    # 将展平后的结果赋值给 l_x_
    l_x_ = arg0
    # 获取 l_x_ 的大小
    size = l_x_.size()
    # 获取 size 的第一个元素并赋值给 getitem，然后清空 size
    getitem = size[0];  size = None
    # 判断 getitem 是否小于等于 2，并将结果赋值给 le，然后清空 getitem
    le = getitem <= 2;  getitem = None
    # 获取 cond_true_0 和 cond_false_0 的引用
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    # 调用 torch.ops.higher_order.cond 函数进行条件判断，根据 le 的值选择执行 cond_true_0 或 cond_false_0
    cond = torch.ops.higher_order.cond(le, cond_true_0, cond_false_0, [l_x_]);  le = cond_true_0 = cond_false_0 = l_x_ = None
    # 获取 cond 的第一个元素并赋值给 getitem_2，然后清空 cond
    getitem_2 = cond[0];  cond = None
    # 根据输出规范将 getitem_2 封装成树结构并返回结果
    return pytree.tree_unflatten([getitem_2], self._out_spec)



self.assertExpectedInline(
    out_graph.cond_true_0.code.strip(),
    """\
def forward(self, l_x_):
    # 复制 l_x_ 为 l_x__1
    l_x__1 = l_x_
    # 计算 l_x__1 加上自身的结果并清空 l_x__1
    add = l_x__1 + l_x__1;  l_x__1 = None
    # 返回结果的元组形式
    return (add,)""",
)



self.assertExpectedInline(
    out_graph.cond_false_0.code.strip(),
    """\
def forward(self, l_x_):
    # 复制 l_x_ 为 l_x__1
    l_x__1 = l_x_
    # 获取 l_x__1 的前两个元素并赋值给 getitem，然后清空 l_x__1
    getitem = l_x__1[slice(None, 2, None)];  l_x__1 = None
    # 返回结果的元组形式
    return (getitem,)""",
)



with self.assertRaisesRegex(
    torch._dynamo.exc.UncapturedHigherOrderOpError,
    "Cond doesn't work unless it is captured completely with torch.compile",
):
    # 如果 true 分支和 false 分支返回的张量形状不同，则抛出异常
    torch._dynamo.export(mod)(torch.randn(3, 2))



with self.assertRaisesRegex(
    torch._dynamo.exc.UncapturedHigherOrderOpError,
    "Cond doesn't work unless it is captured completely with torch.compile",
):
    # 如果 true 分支和 false 分支返回的张量形状不同，则抛出异常
    test_x = torch.randn(3, 2)
    mod(test_x)



def test_export_with_map_cond(self):
    from functorch.experimental.control_flow import cond, map

    class Module(torch.nn.Module):
        def inner(self, x, pred):
            def true_fn(x):
                # 返回 x 加上自身的结果
                return x + x

            def false_fn(x):
                # 返回 x 的平方的结果
                return x * x

            # 根据 pred 的值选择执行 true_fn 或 false_fn
            return cond(pred, true_fn, false_fn, [x])

        def forward(self, pred, xs):
            def body(x, pred):
                # 调用 inner 函数处理 x 和 pred
                return self.inner(x, pred)

            # 对 xs 和 pred 调用 body 函数
            return map(body, xs, pred)

    # 创建 Module 的实例 mod
    mod = Module()
    # 创建输入张量 x 和 pred_x
    x = torch.randn(3, 2, 1)
    pred_x = torch.tensor(True)

    # 创建输入张量 y 和 pred_y
    y = torch.randn(4, 3, 2)
    pred_y = torch.tensor(False)
    # 计算真实结果
    real_result = mod(pred_y, y)

    # 导出模块 mod 并获得输出图和相关函数
    out_graph, _ = torch._dynamo.export(mod)(pred_x, x)
    # 断言实际结果与输出图执行后的结果相等
    self.assertEqual(real_result, out_graph(pred_y, y))
    # 定义一个测试方法，用于测试在处理包含零大小张量的情况下的导出功能
    def test_export_with_map_zero_sized_tensor(self):
        # 导入需要使用的函数库
        from functorch.experimental.control_flow import map

        # 定义一个继承自torch.nn.Module的子类Module
        class Module(torch.nn.Module):
            # Module类的前向传播方法
            def forward(self, xs):
                # 定义内部函数body，对输入的张量执行加1操作
                def body(x):
                    return x + 1

                # 使用map函数将body应用于输入的张量xs
                return map(body, xs)

        # 创建Module类的实例mod
        mod = Module()
        # 生成一个形状为(0, 2)的随机张量xs
        xs = torch.randn(0, 2)
        # 使用assertRaisesRegex上下文管理器，检查是否抛出torch._dynamo.exc.Unsupported异常，
        # 并且异常消息包含"zero-sized tensor"
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "zero-sized tensor",
        ):
            # 调用torch._dynamo.export方法导出Module实例mod的计算图
            out_graph, _ = torch._dynamo.export(mod)(xs)

    # 定义一个测试方法，用于测试导出函数f的元数据处理能力
    def test_export_meta_val(self):
        # 定义一个简单的函数f，接受三个参数x, y, z，返回x * y + z的结果
        def f(x, y, z):
            return x * y + z

        # 使用torch._dynamo.export方法导出函数f，设置参数aten_graph=True以获取ATen图
        gm, _ = torch._dynamo.export(
            f,
            aten_graph=True,
        )(
            torch.ones(3, 2),
            torch.zeros(3, 2),
            torch.ones(3, 2),
        )
        # 遍历导出的计算图gm的所有节点
        for node in gm.graph.nodes:
            # 如果节点的操作为"placeholder"
            if node.op == "placeholder":
                # 断言节点的元数据中包含键"val"
                self.assertIn("val", node.meta)

    # 定义一个测试方法，用于测试接收特定输入类型并返回字典的函数导出功能
    def test_input_container_type(self):
        # 定义一个函数f，接收一个torch.Tensor类型的x和一个torch.Tensor列表类型的y，
        # 返回一个字典，其中键为"a"，值为x.sum() + sum(y).sum()
        def f(x: torch.Tensor, y: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
            return {"a": x.sum() + sum(y).sum()}

        # 准备输入参数inp，包含一个形状为(6, 5)的随机张量和包含两个形状相同的随机张量的列表
        inp = (torch.randn(6, 5), [torch.randn(6, 5), torch.randn(6, 5)])

        # 使用torch._dynamo.export方法导出函数f，设置参数aten_graph=True以获取ATen图
        gm, _ = torch._dynamo.export(f, aten_graph=True)(*inp)

        # 断言导出的计算图gm在输入inp上的输出与函数f在输入inp上的输出相等
        self.assertEqual(gm(*inp), f(*inp))

    # 根据配置设置assume_static_by_default=False，定义一个测试方法，用于测试包含符号形状的张量导出功能
    @config.patch(assume_static_by_default=False)
    def test_export_symbolic_shape(self):
        # 定义一个函数f，接收一个torch.Tensor类型的x，返回一个形状为(x.shape[0] * 2)的空张量
        def f(x: torch.Tensor) -> torch.Tensor:
            return torch.empty(x.shape[0] * 2)

        # 准备输入参数inp，包含一个形状为(6, 5)的随机张量
        inp = (torch.randn(6, 5),)

        # 使用torch._dynamo.export方法导出函数f，设置参数aten_graph=True以获取ATen图
        gm, _ = torch._dynamo.export(f, aten_graph=True)(*inp)

        # 初始化变量has_sym_size为False，用于标记计算图gm是否包含符号大小的操作
        has_sym_size = False
        # 遍历导出的计算图gm的所有节点
        for node in gm.graph.nodes:
            # 如果节点的目标是torch.ops.aten.sym_size.int
            if node.target is torch.ops.aten.sym_size.int:
                # 将has_sym_size设置为True
                has_sym_size = True

        # 断言has_sym_size为True，即计算图gm包含符号大小的操作
        self.assertTrue(has_sym_size)
    # 定义测试函数 test_dynamic_slicing
    def test_dynamic_slicing(self):
        # 定义函数 f，对输入张量 x 进行切片操作，保留前 x.shape[0]-2 行，取 x.shape[1]-1 列中的偶数索引
        def f(x):
            return x[: x.shape[0] - 2, x.shape[1] - 1 :: 2]

        # 使用 torch._dynamo.export 导出函数 f，返回 gm_aten_mode 作为 ATen 模式的计算图对象，_ 为其余返回值
        gm_aten_mode, _ = torch._dynamo.export(f, aten_graph=True)(torch.randn(4, 5))

        # 创建输入张量 inp，并验证 gm_aten_mode 对输入 inp 的输出形状与函数 f 对 inp 的输出形状相等
        inp = torch.randn(6, 7)
        self.assertEqual(gm_aten_mode(inp).shape, f(inp).shape)

        # 初始化计数器 count 为 0，遍历 gm_aten_mode 的计算图节点
        # 检查是否有对 torch.ops.aten.slice.Tensor 的函数调用
        count = 0
        # aten graph should flatten getitem calls to actual
        # slice kernel call.
        for node in gm_aten_mode.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.slice.Tensor
            ):
                count += 1

        # 验证 count 是否为 2
        self.assertEqual(count, 2)

        # 使用 torch._dynamo.export 导出函数 f，返回 gm_torch_mode 作为 Torch 模式的计算图对象，_ 为其余返回值
        gm_torch_mode, _ = torch._dynamo.export(f, aten_graph=False)(torch.randn(4, 5))

        # 初始化计数器 count 为 0，遍历 gm_torch_mode 的计算图节点
        # 检查是否有对 operator.getitem 的函数调用
        # In torch mode, the graph should contain 3 getitem methods
        # one for x.shape[0]-2 and one for x.shape[1]-1 and one for slice
        # this is because Tensor class has its' own getitem method
        # which gets translated to aten.Slice later.
        count = 0
        for node in gm_torch_mode.graph.nodes:
            if node.op == "call_function" and node.target == operator.getitem:
                count += 1

        # 验证 count 是否为 3
        self.assertEqual(count, 3)

        # 验证 gm_torch_mode 对输入 inp 的输出形状与函数 f 对 inp 的输出形状相等
        self.assertEqual(gm_torch_mode(inp).shape, f(inp).shape)

    # 定义测试函数 test_dynamic_slicing_invalid
    def test_dynamic_slicing_invalid(self):
        # 定义函数 g，进行动态切片操作，返回 x 的第 y 到 x.shape[0] 行
        def g(x, y):
            return x[y : x.shape[0]]

        # 使用 torch._dynamo.export 导出函数 g，当传递的参数 y 是数据依赖值时，期望引发 Unsupported 异常
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Dynamic slicing on data-dependent value is not supported",
        ):
            torch._dynamo.export(
                g,
                aten_graph=True,
            )(
                torch.randn(4, 5),
                torch.tensor(2),
            )

    # 使用 config.patch 进行装饰，定义测试函数 test_dynamic_slicing_simple
    @config.patch(capture_scalar_outputs=True)
    def test_dynamic_slicing_simple(self):
        # 定义函数 f，对输入张量 x 进行全范围切片操作
        def f(x):
            return x[slice(None, None, None)]

        # 使用 torch._dynamo.export 导出函数 f，返回 gm 作为 ATen 模式的计算图对象，_ 为其余返回值
        gm, _ = torch._dynamo.export(f, aten_graph=True)(torch.randn(4, 5))

        # 创建输入张量 inp，并验证 gm 对输入 inp 的输出与函数 f 对 inp 的输出相等
        inp = torch.randn(6, 7)
        self.assertEqual(gm(inp), f(inp))

    # 定义测试函数 test_pre_dispatch_simple
    def test_pre_dispatch_simple(self):
        # 定义函数 f，计算输入张量 x 与其形状相同的全 1 张量的乘积
        def f(x):
            y = torch.ones_like(x)
            return torch.matmul(x, y)

        # 使用 torch._dynamo.export 导出函数 f，返回 gm 作为 ATen 模式的计算图对象，_ 为其余返回值
        gm, _ = torch._dynamo.export(
            f,
            aten_graph=True,
            pre_dispatch=True,
            tracing_mode="fake",
        )(
            torch.randn(5, 5),
        )

        # 创建输入张量 inp，并验证 gm 对输入 inp 的输出与函数 f 对 inp 的输出相等
        inp = torch.randn(6, 6)
        self.assertEqual(gm(inp), f(inp))
        # 验证 gm 的代码（去除空格后）是否符合预期
        self.assertExpectedInline(
            gm.code.strip(),
            """\
# 定义一个类方法 `forward`，接受参数 `self` 和 `x`
def forward(self, x):
    # 使用 `fx_pytree.tree_flatten_spec` 将 `([x], {})` 转换为一个扁平化的列表 `arg0`
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    # 复制 `arg0` 以备后用
    arg0_1 = arg0
    # 调用 `torch.ops.aten.ones_like.default` 创建一个和 `arg0_1` 相同形状的张量 `ones_like`
    ones_like = torch.ops.aten.ones_like.default(arg0_1, pin_memory=False)
    # 调用 `torch.ops.aten.matmul.default` 执行矩阵乘法，将结果保存在 `matmul` 中
    matmul = torch.ops.aten.matmul.default(arg0_1, ones_like); arg0_1 = ones_like = None
    # 使用 `pytree.tree_unflatten` 将 `matmul` 转换为树形结构，按 `self._out_spec` 规定返回
    return pytree.tree_unflatten([matmul], self._out_spec)
    # 定义一个测试方法，测试带有位置参数和空关键字参数的函数导出
    def test_export_with_args_and_empty_kwargs(self):
        # 定义一个带有关键字参数的函数
        def fn_with_kwargs(pos0, tuple0, *myargs):
            # 初始化输出值为 pos0
            out = pos0
            # 遍历元组 tuple0 中的每个元素，依次与 out 相乘
            for arg in tuple0:
                out *= arg
            # 遍历可变位置参数 myargs 中的每个元素，依次与 out 相乘
            for arg in myargs:
                out *= arg
            return out
        
        # 初始化一个元组 tuple0 包含两个 torch.randn(4) 生成的随机向量
        tuple0 = (torch.randn(4), torch.randn(4))
        # 初始化 pos0 为一个 torch.randn(4) 生成的随机向量
        pos0 = torch.randn(4)
        # 初始化一个列表 myargs 包含两个 torch.randn(4) 生成的随机向量
        myargs = [torch.randn(4), torch.randn(4)]

        # 期望的参数名列表
        expected_argument_names = ["pos0", "tuple0", "myargs_0", "myargs_1"]
        # 调用 self._test_export_preserving_original_signature 方法，测试 fn_with_kwargs 函数的导出
        self._test_export_preserving_original_signature(
            fn_with_kwargs, expected_argument_names, pos0, tuple0, *myargs
        )

    # 使用 common_utils.parametrize 装饰器定义一个带有默认值参数的测试方法
    @common_utils.parametrize(
        "default_value",
        [
            # 使用 common_utils.subtest 定义一个默认值为 None 的子测试
            common_utils.subtest(None, name="None"),
            # 使用 common_utils.subtest 定义一个默认值为 42.0 的子测试
            common_utils.subtest(42.0, name="float"),
            # 使用 common_utils.subtest 定义一个默认值为 torch.randn(4) 生成的张量的子测试，并且期望此测试失败
            common_utils.subtest(
                torch.randn(4),
                name="tensor",
                decorators=[unittest.expectedFailure],
            ),
            # 使用 common_utils.subtest 定义一个默认值为包含 torch.randn(4) 生成的元组的子测试，并且期望此测试失败
            common_utils.subtest(
                (torch.randn(4),),
                name="tuple",
                decorators=[unittest.expectedFailure],
            ),
        ],
    )
    # 定义一个带有默认值参数的测试方法
    def test_export_with_args_with_default(self, default_value):
        # 定义一个带有默认值参数的函数
        def fn(pos0, pos1_default=default_value):
            # 初始化输出值为 pos0
            out = pos0
            # 如果 pos1_default 为 None，则将其重新赋值为 torch.randn(4) 生成的随机向量
            if pos1_default is None:
                pos1_default = torch.randn(4)
            # 如果 pos1_default 是元组，则取其第一个元素赋值给 pos1_default
            if isinstance(pos1_default, tuple):
                pos1_default = pos1_default[0]
            # out 与 pos1_default 相乘
            out *= pos1_default
            return out
        
        # 初始化 pos0 为一个 torch.randn(4) 生成的随机向量
        pos0 = torch.randn(4)
        # 期望的参数名列表
        expected_argument_names = ["pos0"]
        # 调用 self._test_export_preserving_original_signature 方法，测试 fn 函数的导出
        self._test_export_preserving_original_signature(
            fn, expected_argument_names, pos0
        )

    # 使用 common_utils.parametrize 装饰器定义一个带有默认值参数的测试方法
    @common_utils.parametrize(
        "default_value",
        [
            # 使用 common_utils.subtest 定义一个默认值为 None 的子测试
            common_utils.subtest(None, name="None"),
            # 使用 common_utils.subtest 定义一个默认值为 42.0 的子测试
            common_utils.subtest(42.0, name="float"),
            # 使用 common_utils.subtest 定义一个默认值为 torch.randn(4) 生成的张量的子测试，并且期望此测试失败
            common_utils.subtest(
                torch.randn(4),
                name="tensor",
                decorators=[unittest.expectedFailure],
            ),
            # 使用 common_utils.subtest 定义一个默认值为包含 torch.randn(4) 生成的元组的子测试，并且期望此测试失败
            common_utils.subtest(
                (torch.randn(4),),
                name="tuple",
                decorators=[unittest.expectedFailure],
            ),
        ],
    )
    # 定义一个测试函数，用于测试带有默认值的关键字参数的导出情况
    def test_export_with_kwargs_with_default(self, default_value):
        # 定义一个函数 fn，接受一个位置参数 pos0 和一些关键字参数
        def fn(pos0, *, kw0, kw1_default=default_value, **kwargs):
            # 初始化输出 out 为位置参数 pos0 的值
            out = pos0
            # 将关键字参数 kw0 的值添加到输出中
            out += kw0
            # 如果 kw1_default 为 None，则将其重新赋值为 4 个随机数的张量
            if kw1_default is None:
                kw1_default = torch.randn(4)
            # 如果 kw1_default 是元组，则将其重新赋值为元组的第一个元素
            elif isinstance(kw1_default, tuple):
                kw1_default = kw1_default[0]
            # 将 kw1_default 的值添加到输出中
            out += kw1_default
            # 将 kwargs 字典中 "kw2" 键对应的值添加到输出中
            out += kwargs["kw2"]
            # 返回最终的输出值
            return out

        # 随机生成张量作为测试参数
        pos0 = torch.randn(4)
        kw0 = torch.randn(4)
        kw2 = torch.randn(4)

        # 将位置参数和关键字参数分别组成元组和字典
        args = (pos0,)
        kwargs = {"kw0": kw0, "kw2": kw2}
        # 期望的参数名列表，用于验证导出结果是否与原始签名一致
        expected_argument_names = ["pos0", "kw0", "kw2"]
        
        # 调用自定义方法，测试导出函数并保持原始签名不变
        self._test_export_preserving_original_signature(
            fn, expected_argument_names, *args, **kwargs
        )

    # 定义一个测试函数，用于测试包装后的函数的导出情况
    def test_export_with_wrapped_fn(self):
        # 为确保 dynamo.export 能够处理包装后的函数，尤其是无法使用 inspect 模块获取原始签名信息时
        def _fn(pos0, pos1=1.0, *args, kw0, kw1=2.0, **kwargs):
            # 初始化输出 out 为位置参数 pos0 的值
            out = pos0
            # 将 pos1 的值添加到输出中
            out += pos1
            # 将 kw0 的值添加到输出中
            out += kw0
            # 将 kw1 的值添加到输出中
            out += kw1
            # 遍历并将 args 中的每个值添加到输出中
            for arg in args:
                out += arg
            # 遍历并将 kwargs 中每个值添加到输出中
            for kwarg in kwargs.values():
                out += kwarg
            # 返回最终的输出值
            return out

        # 定义一个包装函数 wrapped_fn，接受任意位置参数和关键字参数，并将它们传递给 _fn 函数
        def wrapped_fn(*args, **kwargs):
            return _fn(*args, **kwargs)

        # 随机生成张量作为测试参数
        pos0 = torch.randn(4)
        kw0 = torch.randn(4)
        args = (pos0, torch.randn(4), torch.randn(4))
        kwargs = {"kw0": kw0, "kw2": torch.randn(4)}
        
        # 期望的参数名列表，由每个 args 参数前加上 "args_" 前缀和 kwargs 的键组成
        expected_argument_names = [f"args_{i}" for i in range(len(args))] + list(
            kwargs.keys()
        )

        # 调用自定义方法，测试导出函数并保持原始签名不变
        self._test_export_preserving_original_signature(
            wrapped_fn, expected_argument_names, *args, **kwargs
        )
    def test_export_with_functools_wrapped_method(self):
        # 定义一个测试装饰器函数，用于包装被装饰函数，保持函数签名不变
        def test_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        # 定义一个继承自torch.nn.Module的类MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

            # 在MyModule类中定义被test_decorator装饰的测试方法
            @test_decorator
            def method_to_test(self, pos0, pos1=1.0, *args, kw0, kw1=2.0, **kwargs):
                # 执行一系列数值操作并返回结果
                out = pos0
                out += pos1
                out += kw0
                out += kw1
                for arg in args:
                    out += arg
                for kwarg in kwargs.values():
                    out += kwarg
                return out

        # 创建四个随机张量对象
        pos0 = torch.randn(4)
        pos1 = torch.randn(4)
        unnamed_pos = torch.randn(4)
        kw0 = torch.randn(4)
        # 将pos0, pos1, unnamed_pos封装成一个元组args
        args = (pos0, pos1, unnamed_pos)
        # 定义一个kwargs字典包含kw0, kw2, unnamed_kw的键值对
        kwargs = {"kw0": kw0, "kw2": torch.randn(4), "unnamed_kw": torch.randn(4)}
        # 创建一个包含参数名称字符串的列表
        expected_argument_names = [
            "pos0",
            "pos1",
            "args_0",  # 第3个未命名的位置参数
        ] + list(kwargs.keys())
        # 创建一个MyModule类的实例m
        m = MyModule()

        # 调用self._test_export_preserving_original_signature方法，传递method_to_test方法、expected_argument_names列表、args和kwargs作为参数
        self._test_export_preserving_original_signature(
            m.method_to_test, expected_argument_names, *args, **kwargs
        )

    def test_export_with_functools_wrapped_fn(self):
        # 定义一个测试装饰器函数，用于包装被装饰函数，保持函数签名不变
        def test_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        # 定义一个被test_decorator装饰的测试函数_fn
        @test_decorator
        def _fn(pos0, pos1=1.0, *args, kw0, kw1=2.0, **kwargs):
            # 执行一系列数值操作并返回结果
            out = pos0
            out += pos1
            out += kw0
            out += kw1
            for arg in args:
                out += arg
            for kwarg in kwargs.values():
                out += kwarg
            return out

        # 定义一个包装函数wrapped_fn，用于调用_fn函数
        def wrapped_fn(*args, **kwargs):
            return _fn(*args, **kwargs)

        # 创建一个随机张量对象pos0
        pos0 = torch.randn(4)
        kw0 = torch.randn(4)
        # 将pos0, 两个随机张量对象封装成一个元组args
        args = (pos0, torch.randn(4), torch.randn(4))
        # 定义一个kwargs字典包含kw0, kw2的键值对
        kwargs = {"kw0": kw0, "kw2": torch.randn(4)}
        # 创建一个包含参数名称字符串的列表
        expected_argument_names = [f"args_{i}" for i in range(len(args))] + list(
            kwargs.keys()
        )

        # 调用self._test_export_preserving_original_signature方法，传递wrapped_fn函数、expected_argument_names列表、args和kwargs作为参数
        self._test_export_preserving_original_signature(
            wrapped_fn, expected_argument_names, *args, **kwargs
        )

    def _test_export_preserving_original_signature(
        self, fn, expected_argument_names: Sequence[str], *args, **kwargs
    ):
        # 这是一个测试函数，用于检查导出功能是否保持原始签名
        pass
    ):
        # 重置 Torch 的动态图工具模块
        torch._dynamo.reset()
        # 导出函数 fn 及其参数，不包括 ATen 图，返回导出结果
        exported = torch._dynamo.export(
            fn,
            *args,
            **kwargs,
            aten_graph=False,
        )
        
        # 取出导出结果的第一个图形
        out_graph = exported[0]
        # 用相同的参数调用导出的图形
        dynamo_result = out_graph(*args, **kwargs)
        # 调用原始函数 fn 并使用相同的参数
        real_result = fn(*args, **kwargs)
        # 断言动态图结果与真实函数结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

        # 检查导出的图形保留了相同的参数名
        self.assertEqual(
            inspect.getfullargspec(out_graph.forward).args[1:], expected_argument_names
        )

    def test_dataclass_input_output(self):
        from dataclasses import dataclass

        @dataclass
        class Tensors:
            x: torch.Tensor
            y: torch.Tensor

        def f(t):
            return t.x + t.y

        # 断言在导出函数时出现 UserError，指示某个类型为 Tensors 的输入不支持或不可扁平化
        with self.assertRaisesRegex(
            UserError,
            "It looks like one of the inputs with type .*Tensors.* "
            "is not supported or pytree-flattenable",
        ):
            torch._dynamo.export(f, aten_graph=False)(
                Tensors(x=torch.randn(10), y=torch.randn(10))
            )

        def f(x, y):
            return Tensors(x=x.sin(), y=y.cos())

        # 断言在导出函数时出现 UserError，指示某个类型为 Tensors 的输出不支持或不可扁平化
        with self.assertRaisesRegex(
            UserError,
            "It looks like one of the outputs with type .*Tensors.* "
            "is not supported or pytree-flattenable",
        ):
            torch._dynamo.export(f, aten_graph=False)(torch.randn(10), torch.randn(10))

    def test_empty(self):
        def f(x):
            return x

        # 导出函数 f 并传入参数，断言输入与输出相同
        exported = torch._dynamo.export(f)(torch.randn(3, 3))
        out_graph = exported[0]
        inp = torch.randn(3, 3)
        self.assertTrue(torch._dynamo.utils.same(inp, out_graph(inp)))

        # 定义一个简单的 Torch 模块，导出它并断言输出与预期相同
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.ones(3, 3)

            def forward(self):
                return self.a

        exported = torch._dynamo.export(M())()
        out_graph = exported[0]
        self.assertTrue(torch._dynamo.utils.same(torch.ones(3, 3), out_graph()))

    @unittest.skipIf(not TEST_CUDA, "No CUDA available.")
    def test_export_with_parameters(self):
        # 定义一个继承自torch.nn.Module的自定义模块MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 使用torch.nn.Sequential定义模块的特征层序列，包括一个卷积层和ReLU激活函数
                self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
                    ),
                    torch.nn.ReLU(inplace=True),
                )

            # 前向传播函数，将输入x通过特征层序列处理后返回
            def forward(self, x):
                return self.features(x)

        # 创建MyModule的实例model，并将其设置为evaluation模式，并移动到CUDA设备上
        model = MyModule().eval().cuda()
        # 创建随机输入数据，封装成元组，并移到CUDA设备上
        random_inputs = (torch.rand([32, 3, 32, 32]).to("cuda"),)
        # 创建一个维度参数对象dim_x，表示维度名为"dim_x"，取值范围为1到32
        dim_x = torch.export.Dim("dim_x", min=1, max=32)
        # 使用torch.export.export函数导出模型，随机输入数据，并指定动态形状为{"x": {0: dim_x}}
        exp_program = torch.export.export(
            model, random_inputs, dynamic_shapes={"x": {0: dim_x}}
        )
        # 创建一个字节流缓冲区output_buffer
        output_buffer = io.BytesIO()
        # 将导出的模型保存到output_buffer中
        torch.export.save(exp_program, output_buffer)
        # 从output_buffer中加载模型并赋值给loaded_model
        loaded_model = torch.export.load(output_buffer)
        # 断言loaded_model中features的第一个卷积层权重参数为torch.nn.Parameter类型
        self.assertTrue(
            isinstance(
                loaded_model.module().get_parameter("features.0.weight"),
                torch.nn.Parameter,
            )
        )

    def test_export_fast_binary_broadcast_check(self):
        # 本测试案例检查了当在FakeTensor的二元操作快速路径中检查操作数形状和输出形状相等时，可能会误创建一个守卫的情况。

        # 定义一个简单的模型MyModel，其中forward函数实现a和b的加法
        class MyModel(torch.nn.Module):
            def forward(self, a, b):
                # 最终输出形状为(dim0, 4, 8)，a和输出的形状相同，顺序很重要
                return b + a

        # 创建随机张量a和b，并将模型MyModel实例化为model，并将其设置为evaluation模式，并移到CUDA设备上
        a = torch.randn(100, 4, 8)
        b = torch.randn(4, 8)
        model = MyModel().eval().cuda()
        # 创建一个维度参数对象batchsize，表示维度名为"dim0"，取值范围为3到1024
        batchsize = torch.export.Dim("dim0", min=3, max=1024)
        # 指定动态形状规范dynamic_shape_spec，表示a的第一个维度为batchsize，b的第一个维度为任意值
        dynamic_shape_spec = {"a": [batchsize, None, None], "b": [None, None]}

        # 使用torch.export.export函数导出模型，输入数据为(a, b)，并指定动态形状为dynamic_shape_spec
        torch.export.export(model, (a, b), dynamic_shapes=dynamic_shape_spec)

    def test_export_meta(self):
        # 定义一个简单的模型MyModule，其中包含一个形状为(2, 3)的参数p
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.ones(2, 3))

            # 前向传播函数，将输入x与参数p相加并返回
            def forward(self, x):
                return self.p + x

        # 使用"meta"设备上下文创建MyModule的实例m
        with torch.device("meta"):
            m = MyModule()

        # 创建输入张量inp，形状为(2, 3)，并使用"meta"设备
        inp = torch.ones(2, 3, device="meta")
        # 使用torch._dynamo.export(m)导出模块m，并将inp作为输入
        exported = torch._dynamo.export(m)(inp)
        # 获取导出结果的计算图out_graph
        out_graph = exported[0]
        # 在计算图上执行inp，并将结果保存为dynamo_result
        dynamo_result = out_graph(inp)
        # 断言dynamo_result与m(inp)相等
        self.assertEqual(dynamo_result, m(inp))
    def test_constraint_violation_error_messages(self):
        # 定义一个继承自torch.nn.Module的类Foo
        class Foo(torch.nn.Module):
            # 定义类Foo的前向传播方法
            def forward(self, x):
                # 如果输入张量x的第一个维度等于第二个维度的两倍
                if x.shape[0] == x.shape[1] * 2:
                    # 返回x加1
                    return x + 1
                else:
                    # 否则返回x加2
                    return x + 2

        # 创建Foo类的实例foo
        foo = Foo()

        # 创建一个形状为[8, 4]的全零张量t
        t = torch.zeros([8, 4])
        # 创建一个名为dim0的维度对象，设置最小值为3，最大值为10
        dim0 = torch.export.Dim("dim0", min=3, max=10)
        # 创建一个名为dim1的维度对象
        dim1 = torch.export.Dim("dim1")
        # 创建一个动态形状字典，键为"x"，值为(dim0, dim1)
        dynamic_shapes = {"x": (dim0, dim1)}

        # 使用assertRaisesRegex断言捕获torch._dynamo.exc.UserError异常，并检查异常信息
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Constraints violated .*!(.*\n)*.*"
            "by dim0 = 2\\*dim1(.*\n)*.*"
            "Not all values of dim1 .* satisfy the generated guard 2 <= .* and .* <= 5(.*\n)*.*",
        ):
            # 导出foo模型，传入参数t和动态形状字典dynamic_shapes
            torch.export.export(foo, (t,), dynamic_shapes=dynamic_shapes)

        # 定义一个继承自torch.nn.Module的类Bar
        class Bar(torch.nn.Module):
            # 定义类Bar的前向传播方法
            def forward(self, x):
                # 如果输入张量x的第一个维度等于5
                if x.shape[0] == 5:
                    # 返回x加1
                    return x + 1
                else:
                    # 否则返回x加2
                    return x + 2

        # 创建Bar类的实例bar
        bar = Bar()

        # 创建一个形状为[5]的全零张量t
        t = torch.zeros([5])
        # 创建一个名为dim0的维度对象，设置最小值为3，最大值为8
        dim0 = torch.export.Dim("dim0", min=3, max=8)
        # 创建一个动态形状字典，键为"x"，值为(dim0,)
        dynamic_shapes = {"x": (dim0,)}
        # 使用assertRaisesRegex断言捕获torch._dynamo.exc.UserError异常，并检查异常信息
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Not all values.*valid.*inferred to be a constant",
        ):
            # 导出bar模型，传入参数t和动态形状字典dynamic_shapes
            torch.export.export(bar, (t,), dynamic_shapes=dynamic_shapes)

        # 定义一个继承自torch.nn.Module的类Qux
        class Qux(torch.nn.Module):
            # 定义类Qux的前向传播方法
            def forward(self, x):
                # 如果输入张量x的第一个维度大于5且小于10
                if x.shape[0] > 5 and x.shape[0] < 10:
                    # 返回x加1
                    return x + 1
                else:
                    # 否则返回x加2
                    return x + 2

        # 创建Qux类的实例qux
        qux = Qux()

        # 创建一个形状为[7]的全零张量t
        t = torch.zeros([7])
        # 创建一个名为dim0的维度对象，设置最小值为3，最大值为8
        dim0 = torch.export.Dim("dim0", min=3, max=8)
        # 创建一个动态形状字典，键为"x"，值为(dim0,)
        dynamic_shapes = {"x": (dim0,)}
        # 使用assertRaisesRegex断言捕获torch._dynamo.exc.UserError异常，并检查异常信息
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Not all values.*satisfy the generated guard",
        ):
            # 导出qux模型，传入参数t和动态形状字典dynamic_shapes
            torch.export.export(qux, (t,), dynamic_shapes=dynamic_shapes)

    def test_untracked_inputs_in_constraints(self):
        # 导入copy模块中的copy函数
        from copy import copy

        # 定义一个继承自torch.nn.Module的类Foo
        class Foo(torch.nn.Module):
            # 定义类Foo的前向传播方法，接受输入x和y
            def forward(self, x, y):
                # 返回y加1
                return y + 1

        # 创建Foo类的实例foo
        foo = Foo()

        # 创建一个形状为[2]的随机张量x
        x = torch.randn(2)
        # 创建一个形状为[5, 4]的随机张量y
        y = torch.randn(5, 4)

        # 创建名为dim0_x和dim0_y的维度对象
        dim0_x, dim0_y = torch.export.dims("dim0_x", "dim0_y")
        # 创建一个动态形状字典，键"x"对应{0: dim0_x}，键"y"对应{0: dim0_y}
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_y}}

        # 创建示例输入，包括x的拷贝和y
        example_inputs = (copy(x), y)
        # 导出foo模型，传入示例输入和动态形状字典dynamic_shapes
        ep = torch.export.export(foo, example_inputs, dynamic_shapes=dynamic_shapes)
        # 调用导出的模型ep的module方法，传入形状为[3]的随机张量和y
        ep.module()(torch.randn(3), y)  # no specialization error

    def test_export_raise_guard_full_constraint(self):
        # 创建形状为[3, 3, 3]的随机张量y
        y = torch.randn([3, 3, 3])

        # 定义一个函数my_dyn_fn，接受输入x，根据x的第一个维度值进行不同的计算
        def my_dyn_fn(x):
            # 如果x的第一个维度等于3
            if x.shape[0] == 3:
                # 返回x的正弦值
                return x.sin()
            # 否则返回x的余弦值
            return x.cos()

        # 使用torch._dynamo.export函数导出my_dyn_fn函数，传入张量y作为参数
        torch._dynamo.export(my_dyn_fn)(y)

        # 使用assertRaises断言捕获ConstraintViolationError异常
        with self.assertRaises(ConstraintViolationError):
            # 导出my_dyn_fn函数，传入动态形状字典({0: torch.export.Dim("dimx")})
            torch._dynamo.export(
                my_dyn_fn, dynamic_shapes=({0: torch.export.Dim("dimx")},)
            )(y)
    def test_export_module_specify_constraints_signature(self):
        # 创建一个形状为 [3, 3, 3] 的随机张量 y
        y = torch.randn([3, 3, 3])

        # 定义一个继承自 torch.nn.Module 的模块 Mod
        class Mod(torch.nn.Module):
            # 模块的前向传播函数
            def forward(self, x):
                # 如果输入张量 x 的第一个维度长度为 3，则返回 x 的正弦值
                if x.shape[0] == 3:
                    return x.sin()
                # 否则返回 x 的余弦值
                return x.cos()

        # 创建 Mod 类的实例 mod
        mod = Mod()
        # 导出 mod 模块
        torch._dynamo.export(mod)(y)

        # 使用动态形状约束进行导出，预期引发 ConstraintViolationError 异常并包含 "dimx = 3" 的错误信息
        with self.assertRaisesRegex(ConstraintViolationError, "dimx = 3"):
            torch._dynamo.export(mod, dynamic_shapes=({0: torch.export.Dim("dimx")},))(y)

    def test_export_raise_guard_partial_constraint(self):
        # 创建一个形状为 [3, 3, 3] 的随机张量 y
        y = torch.randn([3, 3, 3])

        # 定义一个动态形状约束函数 my_dyn_fn
        def my_dyn_fn(x):
            # 如果输入张量 x 的第一个维度长度大于 3，则返回 x 的正弦值
            if x.shape[0] > 3:
                return x.sin()
            # 否则返回 x 的余弦值
            return x.cos()

        # 导出 my_dyn_fn 函数
        torch._dynamo.export(my_dyn_fn)(y)

        # 使用动态形状约束进行导出，预期引发 ConstraintViolationError 异常
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(my_dyn_fn, dynamic_shapes=({0: torch.export.Dim("dimx")},))(y)

    def test_export_raise_on_relationship(self):
        # 创建一个形状为 [3, 3, 3] 的随机张量 y
        y = torch.randn([3, 3, 3])

        # 定义一个带多个输入的动态形状约束函数 my_dyn_fn
        def my_dyn_fn(a, b, c):
            # 如果输入张量 a 的第一个维度长度等于 b 的第二个维度长度且等于 c 的第三个维度长度，则返回 a 的正弦值
            if a.shape[0] == b.shape[1] == c.shape[2]:
                return a.sin()

            # 否则返回 a 的余弦值
            return a.cos()

        # 导出 my_dyn_fn 函数
        torch._dynamo.export(my_dyn_fn)(y, y, y)
        dim = torch.export.Dim("dim")
        
        # 使用动态形状约束进行导出，预期引发 ConstraintViolationError 异常
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(my_dyn_fn, dynamic_shapes=({0: dim}, {0: dim}, {0: dim}))(y, y, y)
        
        # 使用动态形状约束进行导出，预期导出成功
        dynamic_shapes = ({0: dim}, {1: dim}, {2: dim})
        torch._dynamo.export(my_dyn_fn, dynamic_shapes=dynamic_shapes)(y, y, y)

    def test_export_no_raise(self):
        # 创建一个形状为 [3, 3, 3] 的随机张量 y
        y = torch.randn([3, 3, 3])

        # 定义一个带多个输入的动态形状约束函数 my_dyn_fn
        def my_dyn_fn(a, b, c):
            # 如果输入张量 a 的第二个维度长度等于 3，则返回 a 的余弦值
            if a.shape[1] == 3:
                return a.cos()
            # 否则返回 a、b、c 三者的乘积
            return a * b * c

        # 导出 my_dyn_fn 函数
        torch._dynamo.export(my_dyn_fn)(y, y, y)
        dim = torch.export.Dim("dim")
        dynamic_shapes = ({0: dim}, {0: dim}, {0: dim})
        # 使用动态形状约束进行导出，预期导出成功
        torch._dynamo.export(my_dyn_fn, dynamic_shapes=dynamic_shapes)(y, y, y)

    def test_export_multi_dynamic_dim_unsafe_relationship(self):
        # 创建三个形状不同的随机张量 x, y, z
        x = torch.randn([3, 3, 3])
        y = torch.randn([2, 2, 2])
        z = torch.randn([3, 3, 3])

        # 定义一个带多个输入的动态形状约束函数 my_dyn_fn
        def my_dyn_fn(a, b, c):
            # 如果输入张量 a 的第一个维度长度等于 c 的第一个维度长度，则返回 a 的余弦值
            if a.shape[0] == c.shape[0]:
                return a.cos()
            # 否则返回 a 与 c 的乘积，以及 b
            return a * c, b

        # 导出 my_dyn_fn 函数
        torch._dynamo.export(my_dyn_fn)(x, y, z)
        dimx, dimy, dimz = torch.export.dims("dimx", "dimy", "dimz")
        
        # 使用动态形状约束进行导出，预期引发 ConstraintViolationError 异常
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(my_dyn_fn, dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}))(x, y, z)
        
        # 将 dimz 的约束修改为与 dimx 相同，预期导出成功
        dimz = dimx
        dynamic_shapes = ({0: dimx}, {0: dimy}, {0: dimz})
        torch._dynamo.export(my_dyn_fn, dynamic_shapes=dynamic_shapes)(x, y, z)
    # 定义一个测试方法，验证错误消息中动态维度的移除是否正确
    def test_remove_redundant_dynamic_dim_in_error_message(self):
        
        # 定义一个简单的神经网络模块 Foo
        class Foo(torch.nn.Module):
            # 定义模块的前向传播函数
            def forward(self, x, y):
                # 检查 x 的第一个维度是否与 y["k"] 的第一个维度相等
                if x.shape[0] == y["k"].shape[0]:
                    return x + 1
                else:
                    return x - 1

        # 创建 Foo 类的实例
        foo = Foo()

        # 生成随机张量 a 和 b
        a = torch.randn(3)
        b = torch.randn(3)
        
        # 从 torch.export.dims 中获取维度 dim0_a 和 dim0_b
        dim0_a, dim0_b = torch.export.dims("dim0_a", "dim0_b")
        
        # 使用 assertRaisesRegex 断言捕获特定异常，验证导出过程中的错误消息
        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "dim0_b = dim0_a"):
            torch.export.export(
                foo,
                (a, {"k": b}),
                dynamic_shapes={"x": {0: dim0_a}, "y": {"k": {0: dim0_b}}},
            )

    # 定义一个测试方法，验证相等性约束是否被正确强制执行
    def test_enforce_equalities(self):
        
        # 定义一个简单的神经网络模块 Bar
        class Bar(torch.nn.Module):
            # 定义模块的前向传播函数
            def forward(self, x, y):
                return torch.matmul(x, y)

        # 创建 Bar 类的实例
        bar = Bar()

        # 从 torch.export.dims 中获取维度 batch 和 size
        batch, size = torch.export.dims("batch", "size")
        
        # 定义动态维度约束
        dynamic_shapes = {"x": (batch, size, size), "y": (batch, size, size)}

        # 生成随机张量 x 和 y
        x = torch.randn(10, 3, 3)
        y = torch.randn(10, 3, 4)
        
        # 使用 assertRaisesRegex 断言捕获特定异常，验证导出过程中的错误消息
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            ".*x.*size.*1.* = 3 is not equal to .*y.*size.*2.* = 4",
        ):
            torch.export.export(
                bar,
                (x, y),
                dynamic_shapes=dynamic_shapes,
            )

        # 修改 y 使其与 x 的维度相同
        y = torch.randn(10, 3, 3)
        
        # 执行导出过程
        ebar = torch.export.export(
            bar,
            (x, y),
            dynamic_shapes=dynamic_shapes,
        )

        # 使用断言验证导出结果中的占位符节点的形状信息是否正确
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in ebar.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, s1, s1])", "torch.Size([s0, s1, s1])"],
        )

    # 使用装饰器 @config.patch 进行配置，测试导出过程中保留约束作为元数据（标量版本）
    @config.patch(
        capture_dynamic_output_shape_ops=True,
        specialize_int=True,
        capture_scalar_outputs=True,
    )
    def test_export_preserve_constraints_as_metadata_scalar(self):
        
        # 定义一个简单的函数 f，操作输入张量 x 和 y
        def f(x, y):
            b = x.item()
            torch._check_is_size(b)
            return torch.empty((b, y.shape[0]))

        # 生成示例输入张量 x 和 y
        x = torch.tensor([3])
        y = torch.randn([8, 8, 6])
        example_inputs = [x, y]
        
        # 定义动态维度约束
        dynamic_shapes = (None, {0: torch.export.Dim("dimy", min=6, max=10)})
        
        # 使用 torch._dynamo.export 导出函数 f，获取计算图和元数据
        gm, _ = torch._dynamo.export(
            f,
            dynamic_shapes=dynamic_shapes,
            aten_graph=True,
            tracing_mode="symbolic",
        )(*example_inputs)

        # 处理动态维度约束，生成约束列表
        constraints = torch.export.dynamic_shapes._process_dynamic_shapes(
            f, example_inputs, dynamic_shapes=dynamic_shapes
        )
        
        # 使用断言验证导出的计算图元数据中的输入形状约束是否正确
        self.assertEqual(
            gm.meta["input_shape_constraints"],
            [c.serializable_spec for c in constraints],
        )

    # 使用 torch._dynamo.config.patch 装饰器进行配置
    @torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True,
        specialize_int=True,
        capture_scalar_outputs=True,
    )
    # 定义一个测试函数，用于导出保存约束作为元数据张量
    def test_export_preserve_constraints_as_metadata_tensor(self):
        # 定义内部函数 f，接受参数 x
        def f(x):
            # 获取 x 中非零元素的索引
            b = x.nonzero()
            # 检查 b 的长度是否大于等于 2
            torch._check(b.shape[0] >= 2)
            # 检查 b 的长度是否小于等于 5
            torch._check(b.shape[0] <= 5)
            # 返回 b
            return b

        # 创建张量 y
        y = torch.tensor([8, 8, 6])
        # 使用 torch._dynamo.export 导出函数 f 的图形表示 gm，不进行追踪
        gm, _ = torch._dynamo.export(
            f,
            aten_graph=True,
            tracing_mode="symbolic",
        )(y)

    # 使用 config.patch 装饰器定义测试函数，用于导出图形序列化
    def test_exported_graph_serialization(self):
        # 定义内部函数 f，接受参数 x 和 y
        def f(x, y):
            # 获取 x 的单个数值
            b = x.item()
            # 检查 b 是否为有效尺寸
            torch._check_is_size(b)
            # 返回一个形状为 (b, y.shape[0]) 的空张量
            return torch.empty((b, y.shape[0]))

        # 创建张量 x 和 y
        x = torch.tensor([3])
        y = torch.randn([8, 8, 6])
        # 定义示例输入
        example_inputs = [x, y]
        # 定义动态形状信息
        dynamic_shapes = (None, {0: torch.export.Dim("dimy", min=6, max=10)})
        # 使用 torch._dynamo.export 导出函数 f 的图形表示 gm
        gm, _ = torch._dynamo.export(
            f,
            dynamic_shapes=dynamic_shapes,
            aten_graph=True,
            tracing_mode="symbolic",
        )(*example_inputs)

        # 确保带有元数据的导出图形模块可以序列化，
        # 元数据将不会保存在序列化的模块中
        buffer = io.BytesIO()
        torch.save(gm, buffer)

    # 定义测试函数，用于导出动态维度不为 1 的约束
    def test_export_dynamic_dim_not_1(self):
        # 创建张量 x
        x = torch.randn([1, 1, 1])

        # 定义函数 my_dyn_fn，接受参数 a
        def my_dyn_fn(a):
            # 如果 a 的第一个维度长度不为 1，则返回 a 的余弦值
            if a.shape[0] != 1:
                return a.cos()
            # 否则返回 a 的平方
            return a * a

        # 使用 torch._dynamo.export 导出函数 my_dyn_fn 的图形表示
        torch._dynamo.export(my_dyn_fn)(x)
        # 使用断言确保在动态形状约束违反时抛出 ConstraintViolationError 异常
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(
                my_dyn_fn, dynamic_shapes=({0: torch.export.Dim("dimx")},)
            )(x)

    # 定义测试函数，用于导出符号计算
    def test_symbool(self):
        # 定义函数 f，接受参数 x
        def f(x):
            # 创建标量张量，其值为 x 的第一个维度是否大于 4
            a = torch.scalar_tensor(x.shape[0] > 4)
            # 返回 x 的正弦值求和与 a 求和的结果
            return x.sin().sum() + a.sum()

        # 使用 torch._dynamo.export 导出函数 f 的图形表示 gm，进行 ATen 图形追踪
        gm, _ = torch._dynamo.export(f, aten_graph=True)(torch.ones(6, 4))
        # 使用断言确保 gm 对张量 torch.ones(3, 4) 的计算结果与 f 函数相同
        self.assertEqual(gm(torch.ones(3, 4)), f(torch.ones(3, 4)))

    # 定义测试函数，用于导出多个动态维度约束
    def test_export_multi_dynamic_dim_constraint(self):
        # 创建张量 x, y, z
        x = torch.randn([3, 3, 3])
        y = torch.randn([2, 2, 2])
        z = torch.randn([3, 3, 3])

        # 定义函数 my_dyn_fn，接受参数 a, b, c
        def my_dyn_fn(a, b, c):
            # 如果 a 的第一个维度长度等于 c 的第一个维度长度，则返回 a 的余弦值
            if a.shape[0] == c.shape[0]:
                return a.cos()
            # 否则返回 a 与 c 的乘积，以及 b
            return a * c, b

        # 使用 torch._dynamo.export 导出函数 my_dyn_fn 的图形表示
        torch._dynamo.export(my_dyn_fn)(x, y, z)
        # 定义动态形状信息
        dimx_0, dimx_1, dimx_2 = torch.export.dims("dimx_0", "dimx_1", "dimx_2")
        # 使用断言确保在动态形状约束违反时抛出 ConstraintViolationError 异常
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(my_dyn_fn, dynamic_shapes=dynamic_shapes)(x, y, z)
        # 使用定义的动态形状信息导出函数 my_dyn_fn 的图形表示
        dynamic_shapes = ({0: dimx_0, 1: dimx_1, 2: dimx_2}, None, {0: dimx_0})
        torch._dynamo.export(my_dyn_fn, dynamic_shapes=dynamic_shapes)(x, y, z)
    def test_export_dynamic_dim_raise_on_compound_range_constraint(self):
        x = torch.ones(6, 4, 4)
        # 使用断言检查 TypeError 异常，并匹配特定的错误消息
        with self.assertRaisesRegex(TypeError, "Cannot determine truth value"):
            # 使用 dynamic_dim 函数获取动态维度，并进行复合范围约束的检查
            4 < dynamic_dim(x, 0) <= 6  # noqa: B015

    def test_export_dynamic_dim_range_constraint(self):
        x = torch.ones(6, 4, 4)
        dynamic_shapes = ({0: torch.export.Dim("dimx", min=5, max=6)},)

        def foo(x):
            # 检查 x 的第一个维度是否大于 3
            if x.shape[0] > 3:  # ok
                return x.sin()
            return x.cos()

        # 导出 foo 函数，使用动态形状约束 dynamic_shapes 和 aten_graph=True
        torch._dynamo.export(
            foo,
            dynamic_shapes=dynamic_shapes,
            aten_graph=True,
        )(x)

        def bar(x):
            # 检查 x 的第一个维度是否大于 5
            if x.shape[0] > 5:  # error
                return x.sin()
            return x.cos()

        # 使用断言检查 ConstraintViolationError 异常
        with self.assertRaises(ConstraintViolationError):
            # 导出 bar 函数，使用动态形状约束 dynamic_shapes 和 aten_graph=True
            torch._dynamo.export(
                bar,
                dynamic_shapes=dynamic_shapes,
                aten_graph=True,
            )(x)

    def test_trivial_constraint(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                # 检查复杂的可除性条件
                if (2 * x.shape[0] + 3) % (x.shape[0] - 3) == 0:
                    return x + 1
                else:
                    return x - 1

        foo = Foo()

        class Bar(torch.nn.Module):
            def forward(self, x):
                # 检查显然为真的可除性条件
                if (2 * x.shape[0] + 2) % (x.shape[0] + 1) == 0:
                    return x + 1
                else:
                    return x - 1

        bar = Bar()

        class Qux(torch.nn.Module):
            def forward(self, x):
                # 检查简单的可除性条件（非显然为真）
                if (3 * x.shape[0]) % 2 == 0:
                    return x + 1
                else:
                    return x - 1

        qux = Qux()

        x = torch.randn(12)
        dim0 = torch.export.Dim("dim0", max=100)
        dynamic_shapes = {"x": (dim0,)}
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "must be specialized.*guards generated.*too complex",
        ):
            # 导出 foo 函数，传入参数 x，并使用动态形状约束 dynamic_shapes
            torch.export.export(foo, (x,), dynamic_shapes=dynamic_shapes)

        # 导出 bar 函数，传入参数 x，并使用动态形状约束 dynamic_shapes
        torch.export.export(bar, (x,), dynamic_shapes=dynamic_shapes)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Not all values.*satisfy the generated guard",
        ):
            # 导出 qux 函数，传入参数 x，并使用动态形状约束 dynamic_shapes
            torch.export.export(qux, (x,), dynamic_shapes=dynamic_shapes)
    def test_list_contains(self):
        # 定义内部函数 func，参数 x 是一个张量
        def func(x):
            # 断言张量 x 的最后一个维度大小在 [4, 5, 6] 中，否则抛出异常 "bad"
            assert x.size(-1) in [4, 5, 6], "bad"
            # 返回张量 x 自身加上自身的结果
            return x + x

        # 生成一个输入元组 inps，包含一个形状为 (1, 5) 的随机张量
        inps = (torch.randn(1, 5),)
        # 使用 Torch 动态优化工具对 func 进行优化，生成优化后的函数 opt_func
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 计算优化后函数在输入上的实际结果 real_result
        real_result = opt_func(*inps)

        # 重置 Torch 动态优化工具的状态
        torch._dynamo.reset()

        # 导出 func 函数，生成计算图并导出到 exported，使用 ATen 图形表示
        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        # 获取导出后计算图的第一个节点 out_graph
        out_graph = exported[0]

        # 在导出的计算图上计算结果 dynamo_result
        dynamo_result = out_graph(*inps)

        # 断言优化后函数的结果与导出计算图的结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_list_not_contains(self):
        # 定义内部函数 func，参数 x 是一个张量
        def func(x):
            # 断言张量 x 的第一个维度大小不在 [4, 5, 6] 中，否则抛出异常 "bad1"
            assert x.size(0) not in [4, 5, 6], "bad1"
            # 断言字符串 "monkey" 不在列表 ["cow", "pig"] 中，否则抛出异常 "bad2"
            assert "monkey" not in ["cow", "pig"], "bad2"
            # 返回张量 x 自身加上自身的结果
            return x + x

        # 生成一个输入元组 inps，包含一个形状为 (1, 5) 的随机张量
        inps = (torch.randn(1, 5),)
        # 使用 Torch 动态优化工具对 func 进行优化，生成优化后的函数 opt_func
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        # 计算优化后函数在输入上的实际结果 real_result
        real_result = opt_func(*inps)

        # 重置 Torch 动态优化工具的状态
        torch._dynamo.reset()

        # 导出 func 函数，生成计算图并导出到 exported，使用 ATen 图形表示
        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        # 获取导出后计算图的第一个节点 out_graph
        out_graph = exported[0]

        # 在导出的计算图上计算结果 dynamo_result
        dynamo_result = out_graph(*inps)

        # 断言优化后函数的结果与导出计算图的结果相同
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_identity(self):
        # 创建一个输入张量 inp，包含值 [0.1, 0.1]
        inp = torch.tensor([0.1, 0.1])

        # 定义内部函数 func，参数 x 是一个张量
        def func(x):
            # 返回输入张量 x 自身
            return x

        # 重置 Torch 动态优化工具的状态
        torch._dynamo.reset()
        # 导出 func 函数，生成计算图并导出到 exported
        exported, _ = torch._dynamo.export(func)(inp)
        # 在导出的计算图上计算结果 dynamo_result
        dynamo_result = exported(inp)

        # 断言输入张量与导出计算图的结果相同
        self.assertTrue(torch._dynamo.utils.same(inp, dynamo_result))

    def test_export_specialized_int(self):
        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            def __init__(
                self,
                input_dim,
            ):
                super().__init__()
                # 初始化一个 torch.nn.LayerNorm 模块，参数包括输入维度 input_dim
                self.torch_module = torch.nn.LayerNorm(
                    input_dim, eps=1e-5, elementwise_affine=True
                )
                # 设置一个整数属性 int_val 为 100
                self.int_val = 100

            # 定义模块的前向传播函数，输入为 input 张量
            def forward(self, input):
                # 返回输入张量经过 cos 函数后乘以 self.int_val 再乘以 self.torch_module.eps 的结果
                return input.cos() * self.int_val * self.torch_module.eps

        # 创建一个 Foo 类的实例 mod，输入维度为 128
        mod = Foo(128)
        # 创建一个形状为 (3, 128) 的随机张量 inp
        inp = torch.randn(3, 128)

        # 导出 mod 实例的 forward 方法，生成计算图并导出到 gm，使用 ATen 图形表示
        gm, _ = torch._dynamo.export(mod, aten_graph=True)(inp)
        # 计算 placeholder 节点的数量 count
        count = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                count += 1

        # 断言 placeholder 节点的数量为 1
        self.assertEqual(count, 1)
    def test_export_with_nonzero_static(self):
        # 定义一个名为 BasicModule 的内部类，继承自 torch.nn.Module
        class BasicModule(torch.nn.Module):
            # 初始化函数，接受 static_size 参数并调用父类初始化
            def __init__(self, static_size):
                super().__init__()
                # 设置实例变量 static_size
                self.static_size = static_size

            # 前向传播函数，接受输入 x，调用 torch.nonzero_static 返回结果
            def forward(self, x):
                return torch.nonzero_static(x, size=self.static_size)

        # 创建两个输入张量，分别是 torch.tensor([6, 8]) 和 torch.zeros(2, 3)
        input_tensors = torch.tensor([6, 8]), torch.zeros(2, 3)
        # 定义两个 static_sizes 分别为 3 和 4
        static_sizes = 3, 4
        # 遍历 input_tensors 和 static_sizes 的组合
        for input_tensor, static_size in zip(input_tensors, static_sizes):
            # 创建 BasicModule 类的实例 m，并传入 static_size 参数
            m = BasicModule(static_size)
            # 调用 torch._dynamo.export 导出模型 m，并使用 aten_graph=True
            gm, _ = torch._dynamo.export(m, aten_graph=True)(input_tensor)
            # 对输入 input_tensor 调用导出的模型 gm
            res = gm(input_tensor)
            # 断言 res 的第一个维度大小为 static_size
            self.assertEqual(res.size(0), static_size)
            # 断言 res 与 torch.nonzero_static(input_tensor, size=static_size) 结果相同
            self.assertTrue(
                torch._dynamo.utils.same(
                    res, torch.nonzero_static(input_tensor, size=static_size)
                )
            )

    def test_export_pass_arg_by_name(self):
        # 定义一个名为 BasicModule 的内部类，继承自 torch.nn.Module
        class BasicModule(torch.nn.Module):
            # 初始化函数，创建一个包含 3 个输入和 4 个输出的线性层
            def __init__(self):
                super().__init__()
                self.my_lin = torch.nn.Linear(3, 4, bias=True)

            # 前向传播函数，接受输入 x，并通过 self.my_lin 进行线性变换
            def forward(self, x):
                return self.my_lin(x)

        # 创建 BasicModule 类的实例 mod 和输入张量 input_tensor
        mod, input_tensor = BasicModule(), torch.randn(2, 3)
        # 调用 torch._dynamo.export 导出模型 mod，并使用 aten_graph=True
        gm, guard = torch._dynamo.export(mod, aten_graph=True)(input_tensor)
        # 调用 mod 的 forward 方法，记录其结果到 ref
        ref = mod(x=input_tensor)
        # 对输入 input_tensor 调用导出的模型 gm，记录其结果到 res
        res = gm(x=input_tensor)
        # 断言 ref 和 res 的结果相同
        self.assertTrue(torch._dynamo.utils.same(ref, res))

    def test_export_pass_arg_by_name_star_args(self):
        # 定义一个名为 BasicModule 的内部类，继承自 torch.nn.Module
        class BasicModule(torch.nn.Module):
            # 初始化函数，创建一个包含 3 个输入和 4 个输出的线性层
            def __init__(self):
                super().__init__()
                self.my_lin = torch.nn.Linear(3, 4, bias=True)

            # 前向传播函数，接受任意个参数并通过 self.my_lin 进行线性变换后相乘
            def forward(self, *args):
                return self.my_lin(args[0]) * self.my_lin(args[1])

        # 创建 BasicModule 类的实例 mod 和输入张量 input_tensor、input_tensor2
        mod, input_tensor, input_tensor2 = (
            BasicModule(),
            torch.randn(2, 3),
            torch.randn(2, 3),
        )
        # 调用 torch._dynamo.export 导出模型 mod，并使用 aten_graph=True
        gm, guard = torch._dynamo.export(mod, aten_graph=True)(
            input_tensor, input_tensor2
        )
        # 调用 mod 的 forward 方法，记录其结果到 ref
        ref = mod(input_tensor, input_tensor2)
        # 对输入 input_tensor 和 input_tensor2 调用导出的模型 gm，记录其结果到 res
        res = gm(input_tensor, input_tensor2)
        # 断言 ref 和 res 的结果相同
        self.assertTrue(torch._dynamo.utils.same(ref, res))

    def test_export_mark_dynamic_conflict_dynamic_dim(self):
        # 创建形状为 [3, 3, 3] 的随机张量 y
        y = torch.randn([3, 3, 3])

        # 定义一个名为 my_dyn_fn 的函数，根据 x 的形状返回 sin 或 cos 函数的结果
        def my_dyn_fn(x):
            if x.shape[0] > 3:
                return x.sin()
            return x.cos()

        # 标记张量 y 的第 0 维为动态维度
        torch._dynamo.mark_dynamic(y, 0)
        # 使用 torch._dynamo.export 导出 my_dyn_fn，并使用 dynamic_shapes 设置动态维度约束
        with self.assertRaisesRegex(
            RuntimeError,
            "Constraints violated",
        ):
            torch._dynamo.export(
                my_dyn_fn, dynamic_shapes=({0: torch.export.Dim("dim")},)
            )(y)

    def test_export_dynamic_dim_cleanup(self):
        # 创建形状为 [3, 3, 3] 的随机张量 y
        y = torch.randn([3, 3, 3])

        # 定义一个名为 my_dyn_fn 的函数，返回张量 x 的 cos 函数结果
        def my_dyn_fn(x):
            return x.cos()

        # 使用 torch._dynamo.export 导出 my_dyn_fn，并使用 dynamic_shapes 设置动态维度约束
        torch._dynamo.export(my_dyn_fn, dynamic_shapes=({0: torch.export.Dim("dim")},))(
            y
        )

    @config.patch(capture_dynamic_output_shape_ops=True)
    def test_export_dynamic_control_flow_error(self):
        # 定义函数 f，根据输入 x 的条件进行不同的数学操作
        def f(x):
            # 如果 x 的非零元素个数大于 3，则返回 x 的余弦值
            if x.nonzero() > 3:
                return x.cos()
            # 否则返回 x 的正弦值
            return x.sin()

        # 使用 assertRaisesRegex 确保在导出过程中出现指定的用户错误异常
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Dynamic control flow is not supported at the moment",
        ):
            # 导出函数 f，期望得到动态控制流错误
            gm, _ = torch._dynamo.export(f, aten_graph=True)(torch.randn(5, 6))

    @config.patch(assume_static_by_default=False)
    def test_export_persist_assert(self):
        # 定义函数 f，对输入 x 进行断言检查并返回余弦值加正弦值
        def f(x):
            # 断言 x 的第一个元素之和大于 4，否则抛出异常 "Shape must be more than 4"
            assert x[0].sum() > 4, "Shape must be more than 4"
            return x.cos() + x.sin()

        # 导出函数 f，使用符号化追踪模式，并获取导出的图和保护程序
        gm, guard = torch._dynamo.export(f, aten_graph=True, tracing_mode="symbolic")(
            torch.ones(5, 4, 6)
        )

        # 定义函数 has_aten_op，检查图中是否存在指定的操作
        def has_aten_op(gm, op):
            for node in gm.graph.nodes:
                if node.target == op:
                    return True
            return False

        # 断言在导出的图中存在 torch.ops.aten._assert_async.msg 操作
        self.assertTrue(has_aten_op(gm, torch.ops.aten._assert_async.msg))

        # 消除死代码并重新编译图
        gm.graph.eliminate_dead_code()
        gm.recompile()

        # 再次断言在重新编译后的图中存在 torch.ops.aten._assert_async.msg 操作
        self.assertTrue(has_aten_op(gm, torch.ops.aten._assert_async.msg))

        # 使用 assertRaisesRegex 确保在运行时出现指定的运行时错误异常
        with self.assertRaisesRegex(RuntimeError, "Shape must be more than 4"):
            # 在保护程序 guard 的情况下执行 gm，期望抛出异常 "Shape must be more than 4"
            gm(torch.zeros(3, 4, 5))

    @common_utils.parametrize(
        "type_fn",
        [
            common_utils.subtest(type, name="builtin"),
            common_utils.subtest(lambda obj: obj.__class__, name="attr"),
        ],
    )
    def test_access_class_method_from_user_class(self, type_fn):
        # 定义类 A，包含一个类方法 func 返回包含数值 4 和 5 的张量
        class A:
            @classmethod
            def func(cls):
                return torch.Tensor([4, 5])

        # 定义函数 f，创建类 A 的实例 a，返回 x 的和加上通过 type_fn 返回的类方法 func 的和
        def f(x):
            a = A()
            return x.sum() + type_fn(a).func().sum()

        # 导出函数 f，使用符号化追踪模式，并获取导出的图
        gm, _ = torch._dynamo.export(f, aten_graph=True)(torch.ones(6, 4))

        # 断言 f 和导出的图 gm 在相同输入下的结果一致
        self.assertEqual(f(torch.ones(6, 4)), gm(torch.ones(6, 4)))

    def test_not_functionalize(self):
        # 定义类 Foo，继承自 torch.nn.Module，包含一个缓冲区和一个前向传播函数
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer1", torch.ones(6, 2))

            def forward(self, x):
                x.add_(2)
                return x.sum() + self.buffer1.sum()

        # 定义示例输入 example_inputs，并导出类 Foo 的前向传播函数
        example_inputs = (torch.ones(1, 2, 3),)
        gm, _ = torch._dynamo.export(
            Foo(),
            aten_graph=True,
            tracing_mode="symbolic",
        )(*example_inputs)

        # 统计在导出的图中 torch.ops.aten.add_.Tensor 操作的数量
        count = 0
        for node in gm.graph.nodes:
            if node.target == torch.ops.aten.add_.Tensor:
                count += 1

        # 断言在导出的图中 torch.ops.aten.add_.Tensor 操作的数量为 1
        self.assertEqual(count, 1)

        # 定义测试输入 test_inp 和 test_inp_v2，并断言 f 和 gm 在相同输入下的结果一致
        test_inp = (torch.ones(1, 2, 3),)
        test_inp_v2 = (torch.ones(1, 2, 3),)
        self.assertEqual(gm(*test_inp), Foo()(*test_inp_v2))

    def test_round_dynamic_shapes(self):
        # 定义函数 f，返回输入 x 的前一半数据，四舍五入到最接近的整数
        def f(x):
            return x[: round(x.shape[0] / 2)]

        # 导出函数 f，使用符号化追踪模式，并获取导出的图
        gm, _ = torch._dynamo.export(f, aten_graph=True)(torch.ones(6, 4))

        # 断言 f 和导出的图 gm 在相同输入下的结果一致
        self.assertEqual(f(torch.ones(6, 4)), gm(torch.ones(6, 4)))
    # 定义测试函数，用于测试条件支持的不同预测类型
    def test_cond_supported_pred_types(self):
        # 定义一个返回 x 的余弦值的函数
        def true_fn(x):
            return x.cos()

        # 定义一个返回 x 的正弦值的函数
        def false_fn(x):
            return x.sin()

        # 定义一个以符号节点变量追踪为预测值的函数
        def f_pred_traced_as_symnode_var(x):
            return cond(x.shape[0] > 2, true_fn, false_fn, [x])

        # 定义一个以张量变量追踪为预测值的函数
        def f_pred_traced_as_tensor_var(x):
            return cond(x.all(), true_fn, false_fn, [x])

        # 定义一个以复杂表达式追踪为符号节点变量的预测值的函数
        def f_pred_complex_expression_traced_as_symnode_var(x):
            return cond(
                x.dim() > 1 and x.shape[1] > 5 and x.shape[1] <= 10,
                true_fn,
                false_fn,
                [x],
            )

        # 定义示例输入
        example_inputs = (torch.rand(5, 8),)
        
        # 遍历所有测试函数，并导出计算图以验证结果
        for f in [
            f_pred_traced_as_symnode_var,
            f_pred_traced_as_tensor_var,
            f_pred_complex_expression_traced_as_symnode_var,
        ]:
            gm, _ = torch._dynamo.export(f, aten_graph=True)(*example_inputs)
            # 断言导出的计算图的结果与原始函数的结果相等
            self.assertEqual(gm(*example_inputs), f(*example_inputs))

    @unittest.expectedFailure  # TODO: Not sure why dynamo creates a new inputs for self.a
    # 测试参数求和
    def test_sum_param(self):
        # 在 forward() 方法中设置新属性
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.randn(3, 2)

            def forward(self, x):
                self.b = 2
                # 返回输入张量的总和、self.a 的总和和 self.b 的和
                return x.sum() + self.a.sum() + self.b

        # 导出 Foo 类的实例，以便分析其计算图
        torch._dynamo.export(Foo())(torch.randn(3, 2))

    # 测试混合真实和虚拟输入
    def test_mixed_real_and_fake_inputs(self):
        # 定义一个测试模式类
        class _TestPattern(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.bn = torch.nn.BatchNorm2d(1)

            def forward(self, input):
                # 计算批标准化的运行标准差
                running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
                scale_factor = self.bn.weight / running_std
                weight_shape = [1] * len(self.conv.weight.shape)
                weight_shape[0] = -1
                bias_shape = [1] * len(self.conv.weight.shape)
                bias_shape[1] = -1
                scaled_weight = self.conv.weight * scale_factor.reshape(weight_shape)
                zero_bias = torch.zeros_like(self.conv.bias, dtype=input.dtype)
                conv = self.conv._conv_forward(input, scaled_weight, zero_bias)
                conv_orig = conv / scale_factor.reshape(bias_shape)
                conv_orig = conv_orig + self.conv.bias.reshape(bias_shape)
                conv = self.bn(conv_orig)
                return conv

        # 定义示例输入
        example_inputs = (torch.randn(1, 1, 3, 3),)
        # 导出 _TestPattern 类的实例，以便分析其计算图
        torch._dynamo.export(
            _TestPattern(),
            aten_graph=True,
        )(*example_inputs)

    @config.patch(
        capture_dynamic_output_shape_ops=True,
        capture_scalar_outputs=True,
        assume_static_by_default=False,
    )
    def test_sym_contains(self):
        # 定义一个内部函数 f(x, y)，返回 x.size(0) 是否在 y 中
        def f(x, y):
            return x.size(0) in y

        # 使用 torch._dynamo.export 导出函数 f，并返回导出的计算图和其他信息
        gm, _ = torch._dynamo.export(f, aten_graph=True)(torch.ones(2), torch.ones(3))

        # 真实的输入数据 true_inp 和错误的输入数据 false_inp
        true_inp = (torch.Tensor([6, 4, 5]), torch.ones(6, 4).add_(5))
        false_inp = (torch.Tensor([6, 4, 5]), torch.ones(6, 4).add_(2))
        
        # 断言导出的计算图 gm 对于 true_inp 和 false_inp 的输出与 f 函数的输出一致
        self.assertEqual(gm(*true_inp), f(*true_inp))
        self.assertEqual(gm(*false_inp), f(*false_inp))

    def test_cond_raise_user_error_on_missing_args(self):
        # 定义一个条件函数 true_fn(x)，返回 x 的余弦值
        def true_fn(x):
            return x.cos()

        # 定义一个条件函数 false_fn(x)，返回 x 的正弦值
        def false_fn(x):
            return x.sin()

        # 定义一个函数 f(x)，根据 x.shape[0] 大小选择 true_fn 或 false_fn
        def f(x):
            return cond(x.shape[0] > 10, true_fn, false_fn)

        # 示例输入 example_inputs
        example_inputs = (torch.rand(5),)
        
        # 使用断言确保调用 f(*example_inputs) 时抛出 TypeError 异常，提示缺少 'operands' 参数
        with self.assertRaisesRegex(
            TypeError,
            r"cond\(\) missing 1 required positional argument: 'operands'",
        ):
            f(*example_inputs)

    def test_cond_raise_user_error_on_unsupported_pred(self):
        # 定义一个函数 f_unsupported_pred(x)，其中 pred 是一个神经网络模块
        def f_unsupported_pred(x):
            pred = torch.nn.Module()
            return cond(pred, lambda x: x.sin(), lambda x: x.cos(), [x])

        # 示例输入 example_inputs
        example_inputs = (torch.rand(5),)
        
        # 使用断言确保调用 f_unsupported_pred(*example_inputs) 时抛出 RuntimeError 异常，
        # 提示预期的 pred 应为布尔值或张量，但得到了一个模块 Module()
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected pred to be bool or tensor, but got Module()",
        ):
            f_unsupported_pred(*example_inputs)

    def test_cond_raise_user_error_on_non_list_operands(self):
        # 定义一个函数 f_non_list_operands(x)，其中操作数 x 不是列表
        def f_non_list_operands(x):
            return cond(torch.tensor(True), lambda x: x.sin(), lambda x: x.cos(), x)

        # 示例输入 example_inputs
        example_inputs = (torch.rand(5),)
        
        # 使用断言确保调用 f_non_list_operands(*example_inputs) 时抛出 RuntimeError 异常，
        # 提示预期的操作数应为可能嵌套的字典/列表/元组的元组
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expect operands to be a tuple of possibly nested dict/list/tuple",
        ):
            f_non_list_operands(*example_inputs)

    def test_cond_raise_user_error_on_non_tensor_operands(self):
        # 定义一个函数 f_non_tensor_operands(x)，其中操作数不是张量
        def f_non_tensor_operands(x):
            a: float = 3.14
            return cond(
                torch.tensor(1234), lambda x, a: x.sin(), lambda x, a: x.cos(), [x, a]
            )

        # 示例输入 example_inputs
        example_inputs = (torch.rand(5),)
        
        # 使用断言确保调用 f_non_tensor_operands(*example_inputs) 时抛出 RuntimeError 异常，
        # 提示预期的操作数应为可能嵌套的字典/列表/元组的元组
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expect operands to be a tuple of possibly nested dict/list/tuple",
        ):
            f_non_tensor_operands(*example_inputs)

    def test_cond_raise_user_error_on_branch_args_mismatch(self):
        # 定义一个条件函数 true_fn(x, y)，返回 x 的正弦值
        def true_fn(x, y):
            return x.sin()

        # 定义一个条件函数 false_fn(x)，返回 x 的余弦值
        def false_fn(x):
            return x.cos()

        # 定义一个函数 f_branch_args_mismatch(x, y)，根据条件选择 true_fn 或 false_fn
        def f_branch_args_mismatch(x, y):
            return cond(torch.tensor([[[[True]]]]), true_fn, false_fn, [x, y])

        # 示例输入 example_inputs
        example_inputs = (torch.rand(5), torch.rand(2))
        
        # 使用断言确保调用 torch._dynamo.export(f_branch_args_mismatch, aten_graph=True)(*example_inputs)
        # 时抛出 UncapturedHigherOrderOpError 异常，提示条件函数 cond 必须完全捕获，使用 torch.compil
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Cond doesn't work unless it is captured completely with torch.compil",
        ):
            torch._dynamo.export(
                f_branch_args_mismatch,
                aten_graph=True,
            )(
                *example_inputs,
            )
    @config.patch(suppress_errors=True)
    # 应用装饰器 `patch`，设置 `suppress_errors` 参数为 True
    def test_uncaptured_higher_order_op_error_not_suppresed(self):
        # 定义函数 `true_fn`，接受两个参数 `x` 和 `y`，返回 `x` 的正弦值
        def true_fn(x, y):
            return x.sin()

        # 定义函数 `false_fn`，接受一个参数 `x`，返回 `x` 的余弦值
        def false_fn(x):
            return x.cos()

        # 定义函数 `f_branch_args_mismatch`，接受两个参数 `x` 和 `y`
        def f_branch_args_mismatch(x, y):
            # 使用 `cond` 函数根据条件进行选择，此处传入一个张量作为条件
            return cond(torch.tensor([[[[100]]]]), true_fn, false_fn, [x, y])

        # 创建示例输入
        example_inputs = (torch.rand(5), torch.rand(2))
        # 使用 `assertRaisesRegex` 断言捕获预期的异常类型和消息
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Cond doesn't work unless it is captured completely with torch.compile",
        ):
            # 导出函数 `f_branch_args_mismatch`，并设置 `aten_graph=True`
            torch._dynamo.export(
                f_branch_args_mismatch,
                aten_graph=True,
            )(
                *example_inputs,
            )

    # 定义测试函数 `test_cond_raise_user_error_on_branch_return_non_tensor`
    def test_cond_raise_user_error_on_branch_return_non_tensor(self):
        # 定义函数 `f_branch_return_non_tensor`，接受一个参数 `x`
        def f_branch_return_non_tensor(x):
            # 使用 `cond` 函数根据条件返回不同的结果，此处分支返回非张量
            return cond(x.shape[0] <= 5, lambda x: 3.14, lambda x: 3.14, [x])

        # 创建示例输入
        example_inputs = (torch.rand(5),)
        # 使用 `assertRaisesRegex` 断言捕获预期的异常类型和消息
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Cond doesn't work unless it is captured completely with torch.compile",
        ):
            # 导出函数 `f_branch_return_non_tensor`，并设置 `aten_graph=True`
            torch._dynamo.export(
                f_branch_return_non_tensor,
                aten_graph=True,
            )(*example_inputs)

    # 定义测试函数 `test_cond_raise_user_error_on_branch_return_multiple_tensors`
    def test_cond_raise_user_error_on_branch_return_multiple_tensors(self):
        # 定义函数 `f_branch_return_multiple_tensors`，接受三个参数 `pred`, `x`, `y`
        def f_branch_return_multiple_tensors(pred, x, y):
            # 使用 `cond` 函数根据条件返回不同的结果，此处分支返回多个张量
            return cond(pred, lambda x: (x, x), lambda x: (x, x), [y])

        # 创建示例输入
        example_inputs = (torch.tensor(True), torch.randn(4), torch.randn(2))
        # 导出函数 `f_branch_return_multiple_tensors`，并设置 `aten_graph=True`
        gm, _ = torch._dynamo.export(
            f_branch_return_multiple_tensors,
            aten_graph=True,
        )(*example_inputs)
        # 断言导出的函数和原函数的输出结果相同
        self.assertEqual(
            gm(*example_inputs), f_branch_return_multiple_tensors(*example_inputs)
        )

    # 定义测试函数 `test_multiple_outputs_op_with_evaluator`
    def test_multiple_outputs_op_with_evaluator(self):
        # 定义 `TopKModel` 类，继承自 `torch.nn.Module`
        class TopKModel(torch.nn.Module):
            # 定义 `forward` 方法，接受输入 `x`
            def forward(self, x):
                # 使用 `torch.topk` 函数返回输入 `x` 中的前三个最大值
                values, _ = torch.topk(x, 3)
                # 返回前三个最大值的总和
                return torch.sum(values)

        # 创建输入张量 `x`，范围为 1 到 6，包括边界，并需要梯度
        x = torch.arange(1.0, 6.0, requires_grad=True)
        # 导出 `TopKModel` 类的实例
        torch._dynamo.export(TopKModel())(x)

    # 定义测试函数 `test_cond_raise_user_error_on_mismatch_return_length`
    def test_cond_raise_user_error_on_mismatch_return_length(self):
        # 定义函数 `true_fn`，接受一个参数 `x`，返回 `x`
        def true_fn(x):
            return x

        # 定义函数 `false_fn`，接受一个参数 `x`，返回包含 `x` 两次的元组
        def false_fn(x):
            return (x, x)

        # 定义函数 `f_mismatch_return_length`，接受一个参数 `x`
        def f_mismatch_return_length(x):
            # 使用 `cond` 函数根据条件选择分支，此处分支返回结果长度不匹配
            return cond(torch.tensor(100), true_fn, false_fn, [x])

        # 创建示例输入
        example_inputs = (torch.rand(5),)
        # 使用 `assertRaisesRegex` 断言捕获预期的异常类型和消息
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Cond doesn't work unless it is captured completely with torch.compile",
        ):
            # 导出函数 `f_mismatch_return_length`，并设置 `aten_graph=True`
            torch._dynamo.export(
                f_mismatch_return_length,
                aten_graph=True,
            )(*example_inputs)
    def test_export_defaults_ok():
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    # 对输入张量进行动态切片操作，生成一个包含多个切片结果的列表
                    results.append(x[: x.size(0) - i, i : x.size(2), i:3])
                # 返回由切片结果组成的元组
                return tuple(results)

        # 使用 torch._dynamo.export 导出 DynamicSliceExportMod 模块，生成图形式的代码和其他信息
        gm, _ = torch._dynamo.export(DynamicSliceExportMod(), aten_graph=True)(
            torch.randn(5, 5, 5),
        )

        # 使用 self.assertExpectedInline 方法断言导出的代码符合预期，去除首尾空白后应与给定的字符串匹配
        self.assertExpectedInline(
            gm.code.strip(),
            """\
# 定义 forward 方法，用于模型的前向传播
def forward(self, x):
    # 将输入 x 打平成规定格式的参数列表 arg0
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    # 复制 arg0 到 arg0_1
    arg0_1 = arg0
    # 切片操作：从 arg0_1 中提取第二维度的索引范围 [0, 3) 的数据
    slice_1 = torch.ops.aten.slice.Tensor(arg0_1, 2, 0, 3)
    # 计算 arg0_1 的第一维度的符号化大小
    sym_size_int = torch.ops.aten.sym_size.int(arg0_1, 0)
    # 计算 sub，即 sym_size_int 减去 1
    sub = sym_size_int - 1
    # 切片操作：从 arg0_1 中提取第一维度的索引范围 [0, sub) 的数据
    slice_2 = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, sub);  sub = None
    # 计算 arg0_1 的第三维度的符号化大小
    sym_size_int_1 = torch.ops.aten.sym_size.int(arg0_1, 2)
    # 切片操作：从 slice_2 中提取第二维度的索引范围 [1, sym_size_int_1) 的数据
    slice_3 = torch.ops.aten.slice.Tensor(slice_2, 1, 1, sym_size_int_1);  slice_2 = None
    # 切片操作：从 slice_3 中提取第三维度的索引范围 [2, 3) 的数据
    slice_4 = torch.ops.aten.slice.Tensor(slice_3, 2, 1, 3);  slice_3 = None
    # 计算 sub_1，即 sym_size_int 减去 2
    sub_1 = sym_size_int - 2
    # 切片操作：从 arg0_1 中提取第一维度的索引范围 [0, sub_1) 的数据
    slice_5 = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, sub_1);  sub_1 = None
    # 切片操作：从 slice_5 中提取第二维度的索引范围 [2, sym_size_int_1) 的数据
    slice_6 = torch.ops.aten.slice.Tensor(slice_5, 1, 2, sym_size_int_1);  slice_5 = None
    # 切片操作：从 slice_6 中提取第三维度的索引范围 [2, 3) 的数据
    slice_7 = torch.ops.aten.slice.Tensor(slice_6, 2, 2, 3);  slice_6 = None
    # 计算 sub_2，即 sym_size_int 减去 3；同时清空 sym_size_int
    sub_2 = sym_size_int - 3;  sym_size_int = None
    # 切片操作：从 arg0_1 中提取第一维度的索引范围 [0, sub_2) 的数据
    slice_8 = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, sub_2);  arg0_1 = sub_2 = None
    # 切片操作：从 slice_8 中提取第二维度的索引范围 [3, sym_size_int_1) 的数据；同时清空 slice_8 和 sym_size_int_1
    slice_9 = torch.ops.aten.slice.Tensor(slice_8, 1, 3, sym_size_int_1);  slice_8 = sym_size_int_1 = None
    # 切片操作：从 slice_9 中提取第三维度的索引范围 [3, 3) 的数据；同时清空 slice_9
    slice_10 = torch.ops.aten.slice.Tensor(slice_9, 2, 3, 3);  slice_9 = None
    # 使用 pytree 模块将 slice_1、slice_4、slice_7、slice_10 恢复成输出的规定结构，返回结果
    return pytree.tree_unflatten([slice_1, slice_4, slice_7, slice_10], self._out_spec)
    def test_export_with_symbool_inputs(self):
        # 定义一个内部函数 f，接受一个布尔值和一个张量作为参数，根据布尔值的真假返回张量的 sin 或 cos 值
        def f(pred: bool, x: torch.Tensor):
            if pred:
                return x.sin()
            else:
                return x.cos()

        # 生成一个大小为 [3, 4] 的随机张量 x
        x = torch.randn([3, 4])

        # 定义一个测试函数 test_symbool_guards，接受 f 函数、大小测试集合、预期图形、预期守卫代码和预期形状环境守卫作为参数
        def test_symbool_guards(
            f, size_tests, exp_graph, exp_guard_code, exp_shape_env_guards
        ):
            # 创建一个形状环境对象
            shape_env = ShapeEnv()
            # 使用 fake_tensor.FakeTensorMode 进入伪张量模式，使用指定的形状环境
            with fake_tensor.FakeTensorMode(
                shape_env=shape_env,
            ) as fake_mode:
                # 从真实张量 x 转换为伪张量 fake_x，使用 StatelessSymbolicContext 设置动态尺寸
                fake_x = fake_mode.from_tensor(
                    x,
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[DimDynamic.DYNAMIC for _ in range(x.dim())],
                    ),
                )
                # 遍历大小测试集合
                for i, size in enumerate(size_tests):
                    # 构造伪预测，判断 fake_x 的第一个维度是否等于当前 size
                    pred = fake_x.size(0) == size
                    # 导出函数 f 的图形和守卫
                    gm, guards = torch._dynamo.export(f)(pred, x)
                    # 标准化打印可读的 gm（图形模块）表示
                    actual = normalize_gm(gm.print_readable(print_output=False))
                    # 断言实际输出与预期图形相等
                    # TODO: This is naughty, EXPECTTEST_ACCEPT=1 doesn't work
                    self.assertExpectedInline(actual, exp_graph[i])
                    # 从 guards 中找出与形状环境相关的 Dynamo 形状环境守卫
                    dynamo_shape_env_guards = [
                        guard
                        for guard in guards
                        if guard.guard_types is not None
                        and "SHAPE_ENV" in guard.guard_types
                    ]
                    # 断言 Dynamo 形状环境守卫的数量为 1
                    self.assertEqual(len(dynamo_shape_env_guards), 1)
                    # 从 Dynamo 形状环境守卫中找出基于谓词的守卫代码
                    guard_code_on_predicate = [
                        code
                        for code in dynamo_shape_env_guards[0].code_list
                        if "L['pred']" in code
                    ]
                    # 断言基于谓词的守卫代码与预期相符
                    self.assertEqual(guard_code_on_predicate, exp_guard_code[i])
                    # 获取外部形状环境中的所有守卫表达式字符串
                    outter_shape_env_guards = [
                        str(guard.expr) for guard in shape_env.guards
                    ]
                    # 断言外部形状环境守卫与预期相符
                    self.assertEqual(outter_shape_env_guards, exp_shape_env_guards[i])

        # 定义预期的图形输出
        true_graph = """\
# 定义一个名为 GraphModule 的类，继承自 torch.nn.Module
class GraphModule(torch.nn.Module):
    # 定义前向传播方法，接受参数 pred 和 x
    def forward(self, pred, x):
        # arg1 是一个类型为 "f32[s1, s2]" 的注释
        arg1: "f32[s1, s2]";

        # 使用 fx_pytree.tree_flatten_spec 将 [pred, x] 扁平化并根据 self._in_spec 进行展开
        arg0, arg1, = fx_pytree.tree_flatten_spec(([pred, x], {}), self._in_spec)
        # 将扁平化后的 x 存储在 l_x_ 中
        l_x_ = arg1

        # 对 l_x_ 中的数据执行正弦函数操作，结果存储在 sin 中；清空 l_x_
        sin: "f32[s1, s2]" = l_x_.sin();  l_x_ = None
        # 使用 pytree.tree_unflatten 将 sin 构建成输出数据，按照 self._out_spec 规范
        return pytree.tree_unflatten([sin], self._out_spec)

# 定义一个字符串 false_graph，内容为另一段前向传播函数的代码
false_graph = """\
class GraphModule(torch.nn.Module):
    def forward(self, pred, x):
        arg1: "f32[s1, s2]";

        arg0, arg1, = fx_pytree.tree_flatten_spec(([pred, x], {}), self._in_spec)
        l_x_ = arg1

        # 对 l_x_ 中的数据执行余弦函数操作，结果存储在 cos 中；清空 l_x_
        cos: "f32[s1, s2]" = l_x_.cos();  l_x_ = None
        # 使用 pytree.tree_unflatten 将 cos 构建成输出数据，按照 self._out_spec 规范
        return pytree.tree_unflatten([cos], self._out_spec)
"""

# 定义一个字符串列表 true_guard_code，包含一个字符串元素，表示一条符号布尔值转换为符号整数的检查语句
true_guard_code = [
    "cast_symbool_to_symint_guardless(L['pred']) == 1",
]

# 定义一个字符串列表 false_guard_code，包含一个字符串元素，表示一条符号布尔值转换为符号整数的不相等检查语句
false_guard_code = [
    "Ne(cast_symbool_to_symint_guardless(L['pred']), 1)",
]

# 调用 test_symbool_guards 函数进行测试
test_symbool_guards(
    f,
    [3, 3, 4, 5],
    [true_graph, true_graph, false_graph, false_graph],
    [true_guard_code, true_guard_code, false_guard_code, false_guard_code],
    # 外部形状环境不应包含任何保护因为我们从未在外部符号上进行特化。
    [[], [], [], []],
)

# 定义 test_invalid_input_global 方法，测试全局变量 bulbous_bouffant 的无效输入情况
def test_invalid_input_global(self) -> None:
    global bulbous_bouffant
    # 将 bulbous_bouffant 初始化为形状为 (3,) 的随机张量
    bulbous_bouffant = torch.randn(3)

    # 定义函数 f，接受参数 y，并返回 bulbous_bouffant 与 y 的和
    def f(y):
        return bulbous_bouffant + y

    # 使用 assertExpectedInlineMunged 函数断言预期的内联修改异常
    self.assertExpectedInlineMunged(
        UserError,
        # 导出 f 函数并传入形状为 (3,) 的随机张量作为输入
        lambda: torch._dynamo.export(f)(torch.randn(3)),
        """\
G['bulbous_bouffant'], accessed at:
  File "test_export.py", line N, in f
    return bulbous_bouffant + y
""",
    )

# 定义 test_invalid_input_global_multiple_access 方法，测试多次访问全局变量 macademia 的无效输入情况
def test_invalid_input_global_multiple_access(self) -> None:
    global macademia
    # 将 macademia 初始化为形状为 (3,) 的随机张量
    macademia = torch.randn(3)

    # 定义函数 g，接受参数 y，并在函数内部重新声明全局变量 macademia，计算 macademia 与 y 的和，并返回结果
    def g(y):
        global macademia
        y = macademia + y
        return y

    # 定义函数 f，接受参数 y，并在函数内部重新声明全局变量 macademia，调用 g 函数，将结果与 macademia 相加，并返回结果
    def f(y):
        global macademia
        y = g(y)
        return macademia + y

    # NB: 这实际上不起作用（它只报告第一次使用），但我将测试保留在这里以防我们稍后修复它
    self.assertExpectedInlineMunged(
        UserError,
        # 导出 f 函数并传入形状为 (3,) 的随机张量作为输入
        lambda: torch._dynamo.export(f)(torch.randn(3)),
        """\
G['macademia'], accessed at:
  File "test_export.py", line N, in f
    y = g(y)
  File "test_export.py", line N, in g
    y = macademia + y
""",
    )

# 定义 test_invalid_input_nonlocal 方法，测试非局部变量 arglebargle 的无效输入情况
def test_invalid_input_nonlocal(self) -> None:
    # 将 arglebargle 初始化为形状为 (3,) 的随机张量
    arglebargle = torch.randn(3)

    # 定义函数 f，接受参数 y，并返回 arglebargle 与 y 的和
    def f(y):
        return arglebargle + y

    # 使用 assertExpectedInlineMunged 函数断言预期的内联修改异常
    self.assertExpectedInlineMunged(
        UserError,
        # 导出 f 函数并传入形状为 (3,) 的随机张量作为输入
        lambda: torch._dynamo.export(f)(torch.randn(3)),
        """L['arglebargle'], a closed over free variable""",
    )
    def test_invalid_input_unused_nonlocal_ok(self) -> None:
        # 创建一个张量arglebargle，包含3个随机数
        arglebargle = torch.randn(3)

        # 定义一个函数f，接受参数y，将arglebargle赋值给局部变量x，返回参数y
        def f(y):
            x = arglebargle
            return y

        # 使用torch._dynamo.export导出函数f，并传入一个包含3个随机数的张量作为参数
        torch._dynamo.export(f)(torch.randn(3))

    def test_symbolic_tracing_within_fake_mode_with_constraints(self):
        # 从torch._subclasses模块导入fake_tensor
        from torch._subclasses import fake_tensor

        # 创建一个FakeTensorMode实例
        fake_mode = fake_tensor.FakeTensorMode()

        # 定义一个简单的动态形状模型DynamicShapeSimpleModel
        class DynamicShapeSimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 定义模型的前向传播方法，接受三个参数a、b、c，并返回一个张量
            def forward(self, a, b, c) -> torch.Tensor:
                # 计算张量a和b的矩阵乘法，加上张量c，并除以2，赋值给变量d
                d = (torch.matmul(a, b) + c) / 2
                # 获取张量d的第一个维度大小，赋值给变量d_s0
                d_s0 = d.shape[0]
                # 获取张量d的第二个维度大小，赋值给变量d_s1
                d_s1 = d.shape[1]
                # 计算变量d_s0和d_s1的乘积，赋值给变量d_s3
                d_s3 = d_s0 * d_s1
                # 将张量d视图重塑为形状为d_s3的张量，赋值给变量e
                e = d.view(d_s3)
                # 返回张量e和其自身的拼接
                return torch.cat([e, e])

        # 使用fake_mode上下文环境，创建DynamicShapeSimpleModel实例，并赋值给变量model
        with fake_mode:
            model = DynamicShapeSimpleModel()
            # 创建一个包含三个随机张量的元组inputs
            inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
            # 创建一个维度对象torch.export.Dim("dim")，赋值给dynamic_shapes的第一个元素
            dynamic_shapes = ({0: torch.export.Dim("dim")}, None, {0: torch.export.Dim("dim")})
            # 遍历aten_graph的值为True和False的情况
            for aten_graph in [True, False]:
                # 使用torch._dynamo.export导出模型model，传入inputs和dynamic_shapes作为参数
                gm = torch._dynamo.export(
                    model,
                    dynamic_shapes=dynamic_shapes,
                    aten_graph=aten_graph,
                )(*inputs).graph_module

        # 由于模型没有参数，因此可以这样做
        # 创建一个新的inputs张量元组
        inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
        # 断言模型model对于inputs的输出与gm对于inputs的输出相等
        self.assertEqual(model(*inputs), gm(*inputs))

    def test_symbolic_tracing_within_fake_mode_with_constraints_with_parameters(self):
        # 从torch._subclasses模块导入fake_tensor
        from torch._subclasses import fake_tensor

        # 创建一个FakeTensorMode实例
        fake_mode = fake_tensor.FakeTensorMode()

        # TODO: 如果不创建一个新模型直接尝试导出Linear会失败...
        # 定义一个模型Model，继承自torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 添加一个线性层torch.nn.Linear(2, 2)，并赋值给模型的成员变量self.linear
                self.linear = torch.nn.Linear(2, 2)

            # 定义模型的前向传播方法，接受参数x，返回线性层self.linear对x的输出
            def forward(self, x):
                out = self.linear(x)
                return out

        # 使用fake_mode上下文环境，创建Model模型的实例，并赋值给变量model
        with fake_mode:
            model = Model()
            # 创建一个包含一个随机张量的inputs元组
            inputs = (torch.randn(10, 2, 2),)
            # 创建一个包含维度对象torch.export.Dim("dim")的动态形状元组dynamic_shapes
            dynamic_shapes = ({0: torch.export.Dim("dim")},)
            # 遍历aten_graph的值为True和False的情况
            for aten_graph in [True, False]:
                # 使用torch._dynamo.export导出模型model，传入inputs和dynamic_shapes作为参数
                gm = torch._dynamo.export(
                    model,
                    dynamic_shapes=dynamic_shapes,
                    aten_graph=aten_graph,
                )(*inputs).graph_module
    def test_capture_symbolic_tracing_within_fake_mode(self):
        # 导入所需模块和类
        from torch._dynamo.output_graph import config
        from torch._subclasses import fake_tensor
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        # 定义一个模型类
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 定义两个线性层
                self.linear = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                # 模型的前向传播
                out = self.linear(x)
                out = self.linear2(out)
                return out

        # 创建 FakeTensorMode 实例
        fake_mode = fake_tensor.FakeTensorMode(
            allow_non_fake_inputs=False,  # 禁止非 fake 输入
            allow_fallback_kernels=True,   # 允许回退到内核
            shape_env=ShapeEnv(
                allow_scalar_outputs=config.capture_scalar_outputs,  # 允许标量输出
                allow_dynamic_output_shape_ops=config.capture_dynamic_output_shape_ops,  # 允许动态输出形状操作
            ),
        )

        # 在 fake 模式下运行
        with fake_mode:
            x = torch.rand(5, 2, 2)  # 创建随机张量作为输入
            model = Model()  # 实例化模型

            # 使用 fake 输入和参数导出模型
            for aten_graph in [True, False]:
                graph_module, _ = torch._dynamo.export(model, aten_graph=aten_graph)(x)
                self.assertTrue(
                    isinstance(graph_module, torch.fx.GraphModule),
                    msg="test_capture_symbolic_tracing_within_fake_mode_aten_graph_"
                    + str(aten_graph),
                )

    def test_cond_op_param_buffer_lifted(self):
        # 定义模块 A
        class A(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer1", torch.zeros(6, 4))  # 注册一个缓冲区 buffer1

            def forward(self):
                return self.buffer1.sum()  # 返回缓冲区 buffer1 的和

        # 定义模块 B
        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer2", torch.ones(6, 4))  # 注册一个缓冲区 buffer2

            def forward(self):
                return self.buffer2.sum()  # 返回缓冲区 buffer2 的和

        # 定义模块 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = A()  # 实例化模块 A
                self.b = B()  # 实例化模块 B

            def forward(self, x):
                # 定义条件函数
                def true_fn(x):
                    return x.cos() + self.a()  # 返回 x 的余弦和模块 A 的结果

                def false_fn(x):
                    return x.sin() + self.b()  # 返回 x 的正弦和模块 B 的结果

                # 使用 cond 函数进行条件运算
                return (cond(x.shape[0] > 4, true_fn, false_fn, [x]),)

        # 导出模块 M，不使用 ATen 图
        gm, _ = torch._dynamo.export(M(), aten_graph=False)(torch.ones(6, 4))
        # 断言导出的图模块和直接执行模块 M 在输入为全 1 时的结果相等
        self.assertEqual(gm(torch.ones(6, 4)), M()(torch.ones(6, 4)))
        # 断言导出的图模块和直接执行模块 M 在输入为全 1 时的结果相等
        self.assertEqual(gm(torch.ones(3, 4)), M()(torch.ones(3, 4)))
    def test_map_cond_param_buffer_lifted(self):
        from functorch.experimental.control_flow import cond, map

        # 定义一个模块 A，初始化并注册一个大小为 (6, 4) 的零张量缓冲区
        class A(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer1", torch.zeros(6, 4))

            # 计算缓冲区内所有元素的和并返回
            def forward(self):
                return self.buffer1.sum()

        # 定义一个模块 B，初始化并注册一个大小为 (6, 4) 的全为一的张量缓冲区
        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer2", torch.ones(6, 4))

            # 计算缓冲区内所有元素的和并返回
            def forward(self):
                return self.buffer2.sum()

        # 定义一个包含 A 和 B 模块的主模块 M
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = A()
                self.b = B()

            # 内部函数 inner，根据条件 pred 选择不同的操作
            def inner(self, x, pred):
                def true_fn(x):
                    return x + x + self.a()

                def false_fn(x):
                    return x * x + self.b()

                # 根据条件 pred 调用 cond 函数执行 true_fn 或 false_fn
                return cond(pred, true_fn, false_fn, [x])

            # 主前向传播函数，使用 map 函数并行处理输入列表 xs
            def forward(self, pred, xs):
                def body(x, pred):
                    # 对每个输入 x 调用 inner 函数并加上模块 B 的结果
                    return self.inner(x, pred) + self.b()

                # 使用 map 函数在每个输入 x 上执行 body 函数
                return map(body, xs, pred)

        # 创建模块实例
        mod = Module()
        x = torch.randn(3, 2, 1)
        pred_x = torch.tensor(True)

        y = torch.randn(4, 3, 2)
        pred_y = torch.tensor(False)
        
        # 计算模块实例在给定条件和输入下的真实输出结果
        real_result = mod(pred_y, y)

        # 使用 dynamo 导出模块的计算图并进行断言比较
        out_graph, _ = torch._dynamo.export(mod)(pred_x, x)
        self.assertEqual(real_result, out_graph(pred_y, y))
    def test_cond_free_variables_overlapping(self):
        # 导入函数库中的条件控制流 cond
        from functorch.experimental.control_flow import cond

        # 定义一个继承自 torch.nn.Module 的模型类 Module
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 模型的前向传播函数
            def forward(self, pred, x):
                # 初始化四个张量 a, b, c, d，形状为 (6, 4)，值全为 1
                a = torch.ones(6, 4)
                b = torch.ones(6, 4)
                c = torch.ones(6, 4)
                d = torch.ones(6, 4)

                # 定义真值条件下的函数 true_fn
                def true_fn(x):
                    # 返回 x 与 a.cos(), b.cos(), d.cos() 的和
                    return x + x + a.cos() + b.cos() + d.cos()

                # 定义假值条件下的函数 false_fn
                def false_fn(x):
                    # 返回 x 的平方与 a.sin(), b.sin(), c.sin() 的和
                    return x * x + a.sin() + b.sin() + c.sin()

                # 使用 cond 函数根据条件 pred 来选择执行 true_fn 或 false_fn，并传入参数 x
                return cond(pred, true_fn, false_fn, [x])

        # 创建 Module 类的实例 mod
        mod = Module()
        # 初始化一个形状为 (6, 4)，值全为 1 的张量 x
        x = torch.ones(6, 4)
        # 初始化一个布尔张量 pred_x，值为 True
        pred_x = torch.tensor(True)

        # 调用 torch._dynamo.export 导出模型 mod 的执行结果，传入参数 pred_x, x
        out_graph, _ = torch._dynamo.export(mod)(pred_x, x)
        # 使用 self.assertExpectedInline 断言函数的结果是否符合预期
        self.assertExpectedInline(
            out_graph.code.strip(),
            """\
# 定义一个类方法 forward，接受预测函数 pred 和输入数据 x
def forward(self, pred, x):
    # 将 pred 和 x 扁平化成列表，符合指定的输入规范
    arg0, arg1, = fx_pytree.tree_flatten_spec(([pred, x], {}), self._in_spec)
    # 将扁平化后的 pred 和 x 分别赋值给 l_pred_ 和 l_x_
    l_pred_ = arg0
    l_x_ = arg1
    # 创建四个张量 a, b, c, d，每个都是 6x4 大小，初始化为全 1
    a = torch.ones(6, 4)
    b = torch.ones(6, 4)
    c = torch.ones(6, 4)
    d = torch.ones(6, 4)
    # 从类的属性中获取 cond_true_0 和 cond_false_0
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    # 使用 torch 的高阶操作 cond 进行条件判断
    cond = torch.ops.higher_order.cond(l_pred_, cond_true_0, cond_false_0, [a, b, l_x_, d, c]);  l_pred_ = cond_true_0 = cond_false_0 = a = b = l_x_ = d = c = None
    # 从 cond 结果中获取索引为 0 的元素，赋值给 getitem
    getitem = cond[0];  cond = None
    # 使用 pytree 的方法将 getitem 恢复成原始输出规范的结构
    return pytree.tree_unflatten([getitem], self._out_spec)
    # 定义一个测试方法，用于验证模型的可重现性
    def test_retracibility(self):
        # 定义一个简单的线性模型类 MyLinear
        class MyLinear(torch.nn.Module):
            # 初始化方法，设置权重和偏置
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(20, 98)  # 随机生成一个 20x98 的权重张量
                self.bias = torch.randn(20)  # 随机生成一个大小为 20 的偏置张量

            # 前向传播方法，使用 torch.nn.functional.linear 计算线性变换
            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        # 定义一个包含卷积层和自定义线性模型的模型类 Foo
        class Foo(torch.nn.Module):
            # 初始化方法，设置卷积层和线性模型实例
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)  # 创建一个输入通道为 16，输出通道为 33 的 3x3 卷积层
                self.linear = MyLinear()  # 创建 MyLinear 的实例

            # 前向传播方法，接收输入 x，进行卷积和线性变换，并返回结果
            def forward(self, x):
                a, b = x
                a_conv = self.conv(a)  # 对输入 a 进行卷积操作
                a_linear = self.linear(a_conv)  # 对卷积结果 a_conv 进行线性变换
                b_conv = self.conv(b)  # 对输入 b 进行卷积操作
                b_linear = self.linear(b_conv)  # 对卷积结果 b_conv 进行线性变换
                # 返回两个线性变换的余弦和正弦的和
                return (
                    a_linear.cos() + b_linear.sin(),
                    a_linear.sin() + b_linear.cos(),
                )

        # 创建一个输入数据容器，包含两个随机张量作为输入
        inp_container = (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100))

        # 使用 torch._dynamo.export 方法导出模型 Foo，并获取计算图 gm
        gm, _ = torch._dynamo.export(Foo(), inp_container, aten_graph=True)
        # 再次使用 torch._dynamo.export 方法导出 gm，并获取计算图 gm2
        gm2, _ = torch._dynamo.export(gm, inp_container, aten_graph=True)

        # 创建测试用输入数据 inp_test，包含两个随机张量
        inp_test = (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100))

        # 断言 gm 和 gm2 在给定输入下输出的第一个输出张量的所有元素是否接近
        self.assertTrue(torch.allclose(gm(inp_test)[0], gm2(inp_test)[0]))
        # 断言 gm 和 gm2 在给定输入下输出的第二个输出张量的所有元素是否接近
        self.assertTrue(torch.allclose(gm(inp_test)[1], gm2(inp_test)[1]))
    # 定义一个测试类，用于测试模型的可重现性，继承自unittest.TestCase
    def test_retracibility_dict_container_inp_out(self):
        # 定义一个简单的神经网络模型类 MyLinear
        class MyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化权重矩阵为随机张量，形状为(20, 98)
                self.weight = torch.randn(20, 98)
                # 初始化偏置向量为随机张量，形状为(20,)
                self.bias = torch.randn(20)

            def forward(self, x):
                # 使用 torch.nn.functional.linear 函数进行线性变换
                return torch.nn.functional.linear(x, self.weight, self.bias)

        # 定义包含卷积层和线性层的复合模型类 Foo
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个卷积层，输入通道数为16，输出通道数为33，卷积核大小为3x3
                self.conv = torch.nn.Conv2d(16, 33, 3)
                # 添加 MyLinear 类的实例作为一个线性层
                self.linear = MyLinear()

            def forward(self, x):
                # 从输入字典 x 中获取名为 "a" 和 "b" 的数据
                a1, a2 = x["a"]
                b = x["b"]
                # 对 a1 应用卷积层和线性层，计算 a1_linear
                a1_conv = self.conv(a1)
                a1_linear = self.linear(a1_conv)
                # 对 a2 应用卷积层和线性层，计算 a2_linear
                a2_conv = self.conv(a2)
                a2_linear = self.linear(a2_conv)
                # 对 b 应用卷积层和线性层，计算 b_linear
                b_conv = self.conv(b)
                b_linear = self.linear(b_conv)
                # 返回一个字典，包含 "a" 和 "b" 的处理结果
                return {
                    "a": [
                        a1_linear.cos() + b_linear.sin(),  # 对 a1_linear 和 b_linear 执行余弦和正弦操作
                        a1_linear.cos() + b_linear.sin(),  # 同上一行
                    ],
                    "b": a2_linear.sin() + b_linear.cos(),  # 对 a2_linear 和 b_linear 执行正弦和余弦操作
                }

        # 准备输入数据的字典容器 inp_container，包含 "a" 和 "b" 两个键
        inp_container = {
            "a": (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),  # 两个 4 维张量作为 "a" 的值
            "b": torch.randn(20, 16, 50, 100),  # 一个 4 维张量作为 "b" 的值
        }

        # 使用 torch._dynamo.export 函数导出模型 Foo 的图模块 gm，保留 ATen 图
        gm, _ = torch._dynamo.export(Foo(), inp_container, aten_graph=True)
        # 使用导出的图模块 gm 调用 export 函数，生成 gm2，保留 ATen 图
        gm2, _ = torch._dynamo.export(gm, inp_container, aten_graph=True)

        # 准备测试用的输入数据字典 inp_test，与 inp_container 结构相同
        inp_test = {
            "a": (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),  # 两个 4 维张量作为 "a" 的值
            "b": torch.randn(20, 16, 50, 100),  # 一个 4 维张量作为 "b" 的值
        }

        # 使用 assertTrue 断言函数，验证 gm 和 gm2 对相同输入 inp_test 的输出在 "a" 的第一个元素上是否近似
        self.assertTrue(torch.allclose(gm(inp_test)["a"][0], gm2(inp_test)["a"][0]))
        # 使用 assertTrue 断言函数，验证 gm 和 gm2 对相同输入 inp_test 的输出在 "a" 的第二个元素上是否近似
        self.assertTrue(torch.allclose(gm(inp_test)["a"][1], gm2(inp_test)["a"][1]))
        # 使用 assertTrue 断言函数，验证 gm 和 gm2 对相同输入 inp_test 的输出在 "b" 上是否近似
        self.assertTrue(torch.allclose(gm(inp_test)["b"], gm2(inp_test)["b"]))
    def test_fx_pytree(self):
        # 定义一个函数 foo，接受一个参数 args
        def foo(args):
            # 使用 torch.utils._pytree.tree_flatten 将 args 展平，并返回展平后的结果和规范 spec
            flat_args, spec = torch.utils._pytree.tree_flatten(args)
            # 使用 torch.fx._pytree.tree_flatten_spec 将 args 按照规范 spec 进行展平，并返回展平后的结果 flat_args_fx
            flat_args_fx = torch.fx._pytree.tree_flatten_spec(args, spec)
            # 返回展平结果的第一个元素加上原始展平结果的第一个元素
            return flat_args_fx[0] + flat_args[0]

        # 准备输入数据 inp_container，包含两个元组元素，每个元素都是 torch.randn(20, 16, 50, 100)
        inp_container = (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100))

        # 使用 torch._dynamo.export 导出函数 foo，同时传入输入数据 inp_container 和 aten_graph=True 参数
        gm, _ = torch._dynamo.export(foo, inp_container, aten_graph=True)

        # 使用 assertTrue 断言 foo(inp_container) 和 gm(inp_container) 的结果是否全部接近
        self.assertTrue(torch.allclose(foo(inp_container), gm(inp_container)))
    def test_export_with_map_zero_sized_tensor_suppress_errors(self):
        # 导入模块，引入 map 函数
        from functorch.experimental.control_flow import map

        # 定义一个继承自 torch.nn.Module 的模块类 Module
        class Module(torch.nn.Module):
            # 模块的前向传播函数，接受输入 xs
            def forward(self, xs):
                # 定义内部函数 body，对输入 x 执行 x + 1 操作
                def body(x):
                    return x + 1

                # 对输入的 xs 应用 map 函数，使用 body 函数处理每个元素
                return map(body, xs)

        # 创建 Module 类的实例 mod
        mod = Module()
        # 生成一个形状为 (0, 2) 的随机张量 xs
        xs = torch.randn(0, 2)
        # 使用断言检查是否抛出 torch._dynamo.exc.Unsupported 异常
        with self.assertRaises(
            torch._dynamo.exc.Unsupported,
        ):
            # 对模块 mod 和输入 xs 进行导出
            out_graph, _ = torch._dynamo.export(mod, xs)

    def test_param_buffer_safe_from_mutation_simple(self):
        # 定义一个继承自 torch.nn.Module 的模块类 Module
        class Module(torch.nn.Module):
            # 模块的初始化函数
            def __init__(self):
                super().__init__()
                # 注册一个名为 buffer1 的缓冲区，形状为 (5, 5)，初始值为零张量
                self.register_buffer("buffer1", torch.zeros(5, 5))

            # 模块的前向传播函数，接受输入 x
            def forward(self, x):
                # 在 buffer1 上执行 in-place 操作，增加每个元素的值为 1
                self.buffer1.add_(1)
                # 返回输入 x 加上 buffer1 的结果
                return x + self.buffer1

        # 对 Module 实例化并使用 torch._dynamo.export 导出模块和输入张量
        gm, _ = torch._dynamo.export(Module(), torch.ones(5, 5), aten_graph=False)
        # 获取导出模块中的所有命名缓冲区
        buffers = list(gm.named_buffers())
        # 使用断言检查缓冲区的数量是否为 1
        self.assertEqual(len(buffers), 1)

        # 获取第一个缓冲区的名称和内容
        name, buffer = buffers[0]
        # 使用断言检查缓冲区的名称是否为 "L__self___buffer1"
        self.assertEqual(name, "L__self___buffer1")

        # 使用断言检查 buffer 是否与形状为 (5, 5)、值为零的张量近似相等
        self.assertTrue(torch.allclose(buffer, torch.zeros(5)))

    def test_param_buffer_safe_from_mutation_recurse(self):
        # 定义一个继承自 torch.nn.Module 的子模块类 Child
        class Child(torch.nn.Module):
            # 模块的初始化函数
            def __init__(self):
                super().__init__()
                # 注册一个名为 buffer2 的缓冲区，形状为 (5)，初始值为零张量
                self.register_buffer("buffer2", torch.zeros(5))

            # 模块的前向传播函数，接受输入 x
            def forward(self, x):
                # 返回输入 x 所有元素的和，加上 buffer2 所有元素的和
                return x.sum() + self.buffer2.sum()

        # 定义一个继承自 torch.nn.Module 的主模块类 Module
        class Module(torch.nn.Module):
            # 模块的初始化函数
            def __init__(self):
                super().__init__()
                # 注册一个名为 buffer1 的缓冲区，形状为 (5)，初始值为零张量
                self.register_buffer("buffer1", torch.zeros(5))
                # 创建 Child 类的实例
                self.child = Child()

            # 模块的前向传播函数，接受输入 x
            def forward(self, x):
                # 在 buffer1 上执行 in-place 操作，增加每个元素的值为 1
                self.buffer1.add_(1)
                # 在 child 模块的 buffer2 上执行 in-place 操作，增加每个元素的值为 2
                self.child.buffer2.add_(2)
                # 返回输入 x 所有元素的和，加上 buffer1 所有元素的和，加上 child 模块的结果
                return x.sum() + self.buffer1.sum() + self.child(x)

        # 对 Module 实例化并使用 torch._dynamo.export 导出模块和输入张量
        gm, _ = torch._dynamo.export(Module(), torch.ones(5), aten_graph=False)
        # 遍历导出模块中的所有命名缓冲区
        for name, buffer in gm.named_buffers():
            # 使用断言检查每个缓冲区的值是否近似为零张量
            self.assertTrue(torch.allclose(buffer, torch.zeros(5)))

    def test_predispatch_with_higher_order(self):
        # 定义一个函数 f，接受输入 x
        def f(x):
            # 根据 x 的形状大小判断条件，选择 lambda 函数进行操作
            return cond(x.shape[0] > 4, lambda x: x + 5, lambda x: x - 3, [x])

        # 使用 torch._dynamo.export 导出函数 f，开启 aten_graph 和 pre_dispatch
        gm, _ = torch._dynamo.export(f, aten_graph=True, pre_dispatch=True)(
            torch.randn(4, 4)
        )
        # 创建两个输入张量 inp1 和 inp2
        inp1 = torch.randn(4, 4)
        inp2 = torch.randn(6, 4)
        # 使用断言检查 f 在 inp1 和 gm 在 inp1 上的结果是否近似相等
        self.assertTrue(torch.allclose(f(inp1), gm(inp1)))
        # 使用断言检查 f 在 inp2 和 gm 在 inp2 上的结果是否近似相等
        self.assertTrue(torch.allclose(f(inp2), gm(inp2)))
    def test_predispatch_with_higher_order_nested(self):
        def f(x):
            def true_fn(x):
                # 定义条件函数，如果输入张量 x 的形状第一个维度大于6，则执行 x + 10，否则执行 x - 10
                return cond(x.shape[0] > 6, lambda x: x + 10, lambda x: x - 10, [x])

            # 如果输入张量 x 的形状第一个维度大于4，则执行 true_fn，否则执行 x - 3
            return cond(x.shape[0] > 4, true_fn, lambda x: x - 3, [x])

        # 导出函数 f，获取计算图 gm 和其他信息
        gm, _ = torch._dynamo.export(f, aten_graph=True, pre_dispatch=True)(
            torch.randn(4, 4)
        )
        inp1 = torch.randn(4, 4)  # 创建输入张量 inp1
        inp2 = torch.randn(6, 4)  # 创建输入张量 inp2
        inp3 = torch.randn(8, 4)  # 创建输入张量 inp3
        # 断言 f 函数对输入张量的计算结果与 gm 函数对输入张量的计算结果在数值上相近
        self.assertTrue(torch.allclose(f(inp1), gm(inp1)))
        self.assertTrue(torch.allclose(f(inp2), gm(inp2)))
        self.assertTrue(torch.allclose(f(inp3), gm(inp3)))

    def test_predispatch_with_for_out_dtype(self):
        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                # 执行 torch.mm 运算，并指定输出的数据类型为 torch.int32
                return out_dtype(torch.ops.aten.mm.default, torch.int32, x, self.weight)

        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)  # 创建权重张量 weight
        m = M(weight)  # 创建模块实例 m
        x = torch.randint(-128, 127, (5, 5), dtype=torch.int8)  # 创建输入张量 x
        # 导出模块 m，获取计算图 gm 和其他信息
        gm, _ = torch._dynamo.export(m, x, aten_graph=True, pre_dispatch=True)

        # 断言模块 m 对输入张量 x 的计算结果与 gm 对输入张量 x 的计算结果在数值上相近
        self.assertTrue(torch.allclose(m(x), gm(x)))

    def test_predispatch_with_for_out_dtype_nested(self):
        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def true_fn(self, x):
                # 执行 torch.mm 运算，并指定输出的数据类型为 torch.int32，然后对结果求和
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, x, self.weight
                ).sum()

            def false_fn(self, x):
                # 执行 torch.mul 运算，并指定输出的数据类型为 torch.int32，然后对结果求和
                return out_dtype(
                    torch.ops.aten.mul.Tensor, torch.int32, x, self.weight
                ).sum()

            def forward(self, x):
                # 根据输入张量 x 的总和是否为零来选择执行 true_fn 还是 false_fn
                return cond(x.sum() != 0, self.true_fn, self.false_fn, [x])

        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)  # 创建权重张量 weight
        m = M(weight)  # 创建模块实例 m
        x = torch.ones((5, 5), dtype=torch.int8)  # 创建全为1的输入张量 x
        # 导出模块 m，获取计算图 gm 和其他信息
        gm, _ = torch._dynamo.export(m, x, aten_graph=True, pre_dispatch=True)

        # 断言模块 m 对输入张量 x 的计算结果与 gm 对输入张量 x 的计算结果在数值上相近
        self.assertTrue(torch.allclose(m(x), gm(x)))
        y = torch.zeros((5, 5), dtype=torch.int8)  # 创建全为0的输入张量 y
        # 断言模块 m 对输入张量 y 的计算结果与 gm 对输入张量 y 的计算结果在数值上相近
        self.assertTrue(torch.allclose(m(y), gm(y)))

        # 断言 gm.true_graph_0.code 的内容符合预期
        self.assertExpectedInline(
            gm.true_graph_0.code.strip(),
            """\
def test_preserve_fx_node_metadata(self):
    class Module1(torch.nn.Module):
        # 定义 Module1 类，继承自 torch.nn.Module，实现 forward 方法
        def forward(self, x):
            # 返回输入张量 x 的正弦值
            return torch.sin(x)

    class Module2(torch.nn.Module):
        # 定义 Module2 类，继承自 torch.nn.Module
        def __init__(self):
            # 初始化方法
            super().__init__()
            # 实例化 Module1 类对象赋值给 self.mod1
            self.mod1 = Module1()

        def forward(self, x):
            # 前向传播方法，接受输入张量 x
            x = torch.cos(x)  # 计算输入张量 x 的余弦值
            x = self.mod1(x)  # 将 x 传递给 self.mod1 的 forward 方法，并接收返回值
            x = torch.relu(x)  # 计算输入张量 x 的 ReLU 激活函数
            return x  # 返回处理后的张量 x

    def fn(x):
        # 定义函数 fn，返回输入张量 x 的绝对值
        return torch.abs(x)

    mod = Module2()  # 实例化 Module2 类对象
    inp = torch.randn(3, 3)  # 生成一个形状为 (3, 3) 的随机张量 inp

    gm, _ = torch._dynamo.export(mod)(inp)  # 导出模型 mod，返回导出后的模型和其他信息

    # 替换模型中的 relu 操作为 fn 函数
    gm_edit = copy.deepcopy(gm)  # 深度复制导出的模型 gm
    for nd in gm_edit.graph.nodes:
        if nd.target == torch.relu:  # 如果节点的目标操作是 torch.relu
            nd.target = fn  # 替换为 fn 函数
            nd.meta.clear()  # 清空节点的元数据
            break
    gm_edit.recompile()  # 重新编译编辑后的模型

    gm2, _ = torch._dynamo.export(gm_edit)(inp)  # 导出编辑后的模型 gm_edit，返回导出后的模型和其他信息

    # 断言编辑后的模型 gm2 的代码和预期的一致
    self.assertExpectedInline(
        gm.code.strip(),
        """\
def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    l_x_ = arg0
    x = torch.cos(l_x_);  l_x_ = None
    x_1 = torch.sin(x);  x = None
    x_2 = torch.relu(x_1);  x_1 = None
""",
    )
    return pytree.tree_unflatten([x_2], self._out_spec)""",

返回使用 `pytree.tree_unflatten` 函数将 `[x_2]` 解析成树形结构的结果，使用当前对象的 `_out_spec` 属性作为解析规范。


        )

        def _constais_op(gm, target):

定义名为 `_constais_op` 的函数，接受两个参数 `gm` 和 `target`。


            for nd in gm.graph.nodes:

遍历 `gm` 对象的 `graph` 属性中的所有节点。


                if nd.target == target:

检查当前节点 `nd` 的 `target` 属性是否等于传入的 `target` 参数。


                    return True

如果找到匹配的节点，则返回 `True`。


            return False

如果未找到任何匹配的节点，则返回 `False`。


        self.assertTrue(_constais_op(gm_edit, torch.cos))

使用 `_constais_op` 函数检查 `gm_edit` 对象中是否包含 `torch.cos` 目标节点，并断言结果为真。


        self.assertTrue(_constais_op(gm_edit, torch.sin))

使用 `_constais_op` 函数检查 `gm_edit` 对象中是否包含 `torch.sin` 目标节点，并断言结果为真。


        self.assertTrue(not _constais_op(gm_edit, torch.relu))

使用 `_constais_op` 函数检查 `gm_edit` 对象中是否不包含 `torch.relu` 目标节点，并断言结果为真。


        self.assertExpectedInline(

调用 `self.assertExpectedInline` 方法，用于内联断言，通常用于比较预期输出和实际输出是否相符。


            gm2.code.strip(),

将 `gm2` 对象的 `code` 属性去除首尾空白后作为实际输出。


            """\

开始预期输出的多行字符串，注意字符串中包含了换行符。

注释：
# 定义一个方法 `forward`，用于模型的前向传播
def forward(self, x):
    # 将输入 x 扁平化成一个元组 arg0
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    # 将扁平化后的输入赋值给 l_x_
    l_x_ = arg0
    # 计算 x 的余弦值，结果赋给 x，并释放 l_x_ 的引用
    x = torch.cos(l_x_);  l_x_ = None
    # 计算 x 的正弦值，结果赋给 x_1，并释放 x 的引用
    x_1 = torch.sin(x);  x = None
    # 计算 x_1 的绝对值，结果赋给 x_2，并释放 x_1 的引用
    x_2 = torch.abs(x_1);  x_1 = None
    # 使用 pytree 将 x_2 还原成输出结构，并返回结果
    return pytree.tree_unflatten([x_2], self._out_spec)
    def test_preserve_fx_node_metadata_graph_break(self):
        # 定义一个函数 fn，对输入进行一系列 torch 操作，最终返回 torch.cos(x)
        def fn(x):
            x = torch.sin(x)  # 对 x 应用 torch.sin 函数
            x = torch.abs(x)  # 对 x 应用 torch.abs 函数
            return torch.cos(x)  # 返回 torch.cos(x)

        # 定义一个错误函数 bad_fn，调用 torch._dynamo.graph_break() 并返回输入 x
        def bad_fn(x):
            torch._dynamo.graph_break()
            return x

        # 导出函数 fn 的计算图 gm，不关心输入形状
        gm, _ = torch._dynamo.export(fn)(torch.randn(3, 3))

        # 替换 gm_edit 中的 torch.abs 节点为 bad_fn 函数
        gm_edit = copy.deepcopy(gm)
        for nd in gm_edit.graph.nodes:
            if nd.target == torch.abs:
                nd.target = bad_fn
                nd.meta.clear()
                break
        gm_edit.recompile()

        # 预期的输出序列
        expected = [
            """x = torch.sin(l_x_)""",
            """cos = torch.cos(l_stack0_)""",
        ]

        # 定义一个测试后端函数 test_backend，用于验证计算图的输出是否符合预期
        def test_backend(gm: torch.fx.GraphModule, example_inputs):
            self.assertTrue(expected)  # 断言预期输出列表非空
            # 清除计算图节点中的 "example_value" 元数据
            for nd in gm.graph.nodes:
                if "example_value" in nd.meta:
                    del nd.meta["example_value"]
            # 断言第一个预期输出是否在可读的打印输出中出现
            self.assertIn(expected[0], gm.print_readable(print_output=False))
            expected.pop(0)  # 弹出已验证的预期输出
            return gm.forward

        # 重置 torch._dynamo 状态
        torch._dynamo.reset()
        # 编译修改后的计算图 gm_edit，使用 test_backend 作为后端
        opt_gm_edit = torch.compile(gm_edit, backend=test_backend)
        opt_gm_edit(torch.randn(3, 3))

    def test_torch_inference_mode_ctx(self):
        # 使用 torch.inference_mode 上下文装饰器定义函数 fn，对输入 x 加 1 并返回
        @torch.inference_mode()
        def fn(x):
            return x + 1

        # 导出函数 fn 的计算图 gm，使用 torch.rand(2, 2) 作为示例输入
        gm, _ = torch._dynamo.export(fn, torch.rand(2, 2))

        # 输入数据
        inp = torch.randn(2, 2)
        # 执行计算图 gm 并得到输出结果 out
        out = gm(inp)
        # 断言 gm 的打印代码与预期的格式化输出一致
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    # 使用 fx_pytree.tree_flatten_spec 方法将输入 x 转换为扁平化结构 arg0
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    # 将 arg0 赋值给 l_args_0_
    l_args_0_ = arg0
    # 进入推断模式，并记录进入推断模式的状态
    _enter_inference_mode = torch.autograd.grad_mode._enter_inference_mode(True)
    # 执行加法操作 l_args_0_ + 1，并将结果赋值给 add；清空 l_args_0_
    add = l_args_0_ + 1;  l_args_0_ = None
    # 退出推断模式，并记录退出推断模式的状态；清空进入推断模式的记录
    _exit_inference_mode = torch.autograd.grad_mode._exit_inference_mode(_enter_inference_mode);  _enter_inference_mode = None
    # 使用 pytree.tree_unflatten 方法将结果组装为输出结构并返回
    return pytree.tree_unflatten([add], self._out_spec)
    # 使用 fx_pytree.tree_flatten_spec 将 ([x, b, y], {}) 进行扁平化处理，返回结果分别赋给 arg0, arg1, arg2
    arg0, arg1, arg2, = fx_pytree.tree_flatten_spec(([x, b, y], {}), self._in_spec)
    # 将 arg0 赋给 l_x_
    l_x_ = arg0
    # 将 arg1 赋给 l_b_
    l_b_ = arg1
    # 将 arg2 赋给 l_y_
    l_y_ = arg2
    # 关闭梯度计算
    _set_grad_enabled = torch._C._set_grad_enabled(False)
    # 克隆 l_x_ 并赋给 x，清空 l_x_
    x = l_x_.clone();  l_x_ = None
    # 将 l_y_ 的值赋给 x[l_b_]，清空 setitem, l_b_, l_y_
    x[l_b_] = l_y_;  setitem = x;  l_b_ = l_y_ = None
    # 开启梯度计算
    _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
    # 使用 pytree.tree_unflatten 将 [x] 进行还原，按照 self._out_spec 规范
    return pytree.tree_unflatten([x], self._out_spec)
# 定义一个方法，用于模型推理过程中的前向传播
def forward(self, x, b, y):
    # 将输入参数 x, b, y 封装成扁平化的数据结构，并符合指定的输入规范
    arg0, arg1, arg2, = fx_pytree.tree_flatten_spec(([x, b, y], {}), self._in_spec)
    # 从扁平化的数据结构中恢复各个参数
    l_x_ = arg0  # 恢复后的 x
    l_b_ = arg1  # 恢复后的 b
    l_y_ = arg2  # 恢复后的 y
    # 进入推理模式，返回一个上下文管理器
    _enter_inference_mode = torch.autograd.grad_mode._enter_inference_mode(True)
    # 克隆参数 l_x_ 作为 x 的副本，并释放 l_x_
    x = l_x_.clone();  l_x_ = None
    # 使用 l_b_ 和 l_y_ 设置 x 中的部分值
    x[l_b_] = l_y_;  setitem = x;  l_b_ = l_y_ = None
    # 退出推理模式，返回一个上下文管理器
    _exit_inference_mode = torch.autograd.grad_mode._exit_inference_mode(_enter_inference_mode);  _enter_inference_mode = None
    # 将结果 x 封装成符合指定输出规范的数据结构，并返回
    return pytree.tree_unflatten([x], self._out_spec)
```