# `.\pytorch\test\inductor\test_dependencies.py`

```
# Owner(s): ["module: inductor"]
import contextlib  # 导入上下文管理模块

import torch  # 导入PyTorch库
from torch._inductor.dependencies import MemoryDep  # 导入依赖关系管理模块

from torch._inductor.graph import GraphLowering  # 导入图降低模块
from torch._inductor.ir import Buffer, FixedLayout, Pointwise  # 导入缓冲区、固定布局和逐点操作
from torch._inductor.test_case import TestCase as InductorTestCase  # 导入测试用例基类
from torch._inductor.utils import sympy_index_symbol  # 导入符号索引处理函数
from torch._inductor.virtualized import ops, V  # 导入操作模块和虚拟化模块

from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU  # 导入GPU类型和测试相关信息


class TestDependencies(InductorTestCase):  # 定义测试依赖关系的测试用例类
    def _create_buffer(self, name, shape, dtype=torch.float32):  # 定义创建缓冲区的方法
        return Buffer(name, FixedLayout(torch.device(GPU_TYPE), dtype, shape))  # 返回根据给定参数创建的缓冲区对象

    def setUp(self):  # 初始化测试用例前的设置方法
        super().setUp()  # 调用父类的初始化方法

        class DummyModule(torch.nn.Module):  # 定义一个虚拟模块类
            def forward(self, x):  # 定义模块的前向传播方法
                return x * 2  # 返回输入张量的两倍值

        self._gm = torch.fx.symbolic_trace(DummyModule())  # 对虚拟模块进行符号化跟踪
        self._graph = GraphLowering(self._gm)  # 使用图降低器处理符号化的图

        self._stack = contextlib.ExitStack()  # 创建上下文管理器的堆栈
        self._stack.enter_context(V.set_graph_handler(self._graph))  # 将图降低器设置为当前上下文处理程序

    def tearDown(self):  # 清理测试用例的方法
        self._stack.close()  # 关闭上下文管理器的堆栈
        super().tearDown()  # 调用父类的清理方法

    def test_bucketize_dependencies(self):  # 测试分桶依赖关系的方法
        offsets = self._create_buffer("offsets", (1025,), torch.int32)  # 创建名为"offsets"的缓冲区对象

        def inner_fn(index):  # 定义内部函数inner_fn，接受一个索引参数
            idx = index[0]  # 获取索引的第一个元素
            return ops.bucketize(  # 调用操作模块中的bucketize函数
                values=idx,
                offsets_name=offsets.get_name(),
                offsets_size=offsets.get_size()[0],
                indexing_dtype=torch.int32,
                right=True,
            )

        pointwise = Pointwise.create(  # 使用Pointwise类创建对象
            device=torch.device(GPU_TYPE),
            dtype=torch.int32,
            inner_fn=inner_fn,
            ranges=[1024 * 4],
        )

        self.assertEqual(len(pointwise.get_reads()), 1)  # 断言点操作对象读取的数量为1

    def test_get_offset(self):  # 测试获取偏移量的方法
        x = sympy_index_symbol("x")  # 创建名为"x"的符号索引
        y = sympy_index_symbol("y")  # 创建名为"y"的符号索引
        var_ranges = {  # 定义变量范围字典
            x: 1024,
            y: 2048,
        }
        dep1 = MemoryDep(  # 创建内存依赖对象dep1
            "dep1",
            x * 2048 + y,
            list(var_ranges.keys()),
            list(var_ranges.values()),
        )
        dep2 = MemoryDep(  # 创建内存依赖对象dep2
            "dep2",
            x * 2048 + y + 1024,
            list(var_ranges.keys()),
            list(var_ranges.values()),
        )
        self.assertEqual(dep1.get_offset(), 0)  # 断言dep1的偏移量为0
        self.assertEqual(dep2.get_offset(), 1024)  # 断言dep2的偏移量为1024
    # 定义一个测试函数，用于测试循环顺序相同时的规范化行为
    def test_normalize_with_stride_order_equal(self):
        # 创建符号变量 x 和 y
        x = sympy_index_symbol("x")
        y = sympy_index_symbol("y")
        # 定义变量范围字典
        var_ranges = {
            x: 1024,
            y: 2048,
        }

        # 创建 MemoryDep 对象 loop_order1，表示访问相同缓冲区
        loop_order1 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y,
            [x, y],  # 循环顺序为 [x, y]
            [1024, 2048],  # 对应的步长为 [1024, 2048]
        )

        # 创建 MemoryDep 对象 loop_order2，表示访问相同缓冲区
        loop_order2 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y,
            [y, x],  # 循环顺序为 [y, x]
            [2048, 1024],  # 对应的步长为 [2048, 1024]
        )

        # 断言 loop_order1 不等于 loop_order2
        self.assertTrue(loop_order1 != loop_order2)

        # 对 loop_order1 和 loop_order2 进行规范化处理
        normalized_loop_order1 = loop_order1.normalize_with_stride_order()
        normalized_loop_order2 = loop_order2.normalize_with_stride_order()

        # 断言规范化后的 loop_order1 等于规范化后的 loop_order2
        self.assertTrue(normalized_loop_order1 == normalized_loop_order2)

    # 定义一个测试函数，用于测试循环顺序不同时的规范化行为
    def test_normalize_with_stride_order_unequal(self):
        # 创建符号变量 x 和 y
        x = sympy_index_symbol("x")
        y = sympy_index_symbol("y")
        # 定义变量范围字典
        var_ranges = {
            x: 1024,
            y: 2048,
        }

        # 创建 MemoryDep 对象 loop_order1，表示访问相同缓冲区
        loop_order1 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y,
            [x, y],  # 循环顺序为 [x, y]
            [1024, 2048],  # 对应的步长为 [1024, 2048]
        )

        # 创建 MemoryDep 对象 loop_order2，表示访问相同缓冲区，但偏移量不同
        loop_order2 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y + 5,
            [y, x],  # 循环顺序为 [y, x]
            [2048, 1024],  # 对应的步长为 [2048, 1024]
        )

        # 断言 loop_order1 不等于 loop_order2
        self.assertTrue(loop_order1 != loop_order2)

        # 对 loop_order1 和 loop_order2 进行规范化处理
        normalized_loop_order1 = loop_order1.normalize_with_stride_order()
        normalized_loop_order2 = loop_order2.normalize_with_stride_order()

        # 断言规范化后的 loop_order1 不等于规范化后的 loop_order2，因为它们有不同的偏移量
        self.assertTrue(normalized_loop_order1 != normalized_loop_order2)
        # unequal due to different offset
# 如果当前模块是主程序入口
if __name__ == "__main__":
    # 从torch._inductor.test_case模块导入run_tests函数
    from torch._inductor.test_case import run_tests

    # 如果同时具备CPU和GPU的条件
    if HAS_CPU and HAS_GPU:
        # 运行sympy测试
        run_tests("sympy")
```