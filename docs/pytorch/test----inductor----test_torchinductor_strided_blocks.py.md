# `.\pytorch\test\inductor\test_torchinductor_strided_blocks.py`

```py
# Owner(s): ["module: inductor"]

# 导入需要的模块和类
import contextlib
import importlib
import unittest
from typing import Any, Callable, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch._inductor import config
from torch._inductor.runtime.hints import TRITON_MAX_BLOCK
from torch._inductor.runtime.runtime_utils import is_power_of_2
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    requires_gpu,
    skip_windows_ci,
)

# 调用 skip_windows_ci 函数，传入当前模块的名称和文件名
skip_windows_ci(__name__, __file__)

# 动态导入 "filelock" 模块
importlib.import_module("filelock")

# 设置 max_block 变量为 TRITON_MAX_BLOCK 字典中键为 "X" 的值
max_block: int = TRITON_MAX_BLOCK["X"]

# 使用 requires_gpu 装饰器，配置 triton.use_block_ptr 为 True，并生成参数化测试
@requires_gpu()
@config.patch("triton.use_block_ptr", True)
@instantiate_parametrized_tests
class TritonBlockPointerTest(InductorTestCase):
    def run_and_compare(
        self,
        func: Callable[..., Any],
        *args,
        compile_kwargs: Optional[dict] = None,
        expected_num_block_pointers: Optional[int] = None,
        expected_num_programs: int = 1,
        expected_num_triton_kernels: int = 1,
    ):
        """
        运行模块通过 Inductor，与 eager 模式的参考进行比较。
        """
        # 如果 compile_kwargs 为 None，则设置为空字典
        if compile_kwargs is None:
            compile_kwargs = {}

        # 定义函数 flatten_tensors，将输入张量展平为一维数组
        def flatten_tensors(tensors):
            flat, spec = pytree.tree_flatten(tensors)
            return flat

        # 使用 torch.compile 编译 func 函数，选择后端为 "inductor"，传入编译参数
        compiled = torch.compile(func, backend="inductor", **compile_kwargs)
        # 运行编译后的函数，并获取运行结果和生成的代码
        result, code = run_and_get_code(compiled, *args)

        # 检查数值精度
        ref_tensors = flatten_tensors(func(*args))  # 获取参考结果的展平张量
        actual_tensors = flatten_tensors(result)    # 获取实际结果的展平张量
        # 逐个比较参考结果和实际结果的张量是否全部近似相等
        for ref, actual in zip(ref_tensors, actual_tensors):
            self.assertTrue(torch.allclose(ref, actual))

        # 定义函数 count_code，统计代码中包含指定子字符串的数量
        def count_code(substr: str, expected: Optional[int]):
            count = sum(prog.count(substr) for prog in code)
            if expected is not None:
                self.assertEqual(count, expected)

        # 检查生成的代码数量是否符合预期
        self.assertEqual(len(code), expected_num_programs)
        # 统计代码中包含 "@triton.jit" 字符串的数量是否符合预期
        count_code("@triton.jit", expected_num_triton_kernels)
        # 统计代码中包含 "tl.make_block_ptr" 字符串的数量是否符合预期
        count_code("tl.make_block_ptr", expected_num_block_pointers)

        return result, code

    # 参数化测试装饰器，传入参数列表
    @parametrize(
        "expected_num_block_pointers,raises",
        [
            (3, False),  # 预期生成 3 个块指针，测试应该通过
            (9, True),   # 预期生成 9 个块指针，测试应该失败
        ],
    )
    # 测试函数，检查生成的块指针数量是否符合预期
    def test_expected_num_block_pointers(
        self, expected_num_block_pointers: int, raises: bool

    ):
        pass  # 此处需要继续完善注释


这里代码被截断了，但你可以继续按照相同的格式和注意事项来完善剩余部分的注释。
    ):
        """
        Checks that the test harness verifies the number of block pointers correctly.
        """

        def foo(x, y):
            return x + y

        device = torch.device(GPU_TYPE)
        inputs = [torch.randn(8).to(device) for arg_idx in range(2)]

        # Expect failure for bad inputs
        with self.assertRaises(AssertionError) if raises else contextlib.nullcontext():
            # Expect 3 block pointers: 2 inputs 1 output
            self.run_and_compare(
                foo, *inputs, expected_num_block_pointers=expected_num_block_pointers
            )

    @parametrize(
        "full_size,view_size,stride,offset,require_block_ptr",
        [
            ((64, 32, 32), (32, 16, 8), None, None, True),
            ((16, 8, 8, 8), (8, 8, 4, 2), None, None, True),
            ((8, 8, 8, 8), (4, 4, 4, 4), None, None, True),
            ((8, 8), (4, 4), None, 10, True),  # Storage offset
            ((8, 8), (4, 4), (16, 2), None, True),  # Non-default strides
            ((8, 8), (4, 4), (1, 8), None, True),  # Transposed strides
            (
                (5, 9),
                (5, 8),
                None,
                None,
                True,
            ),  # Non-power-of-2 leading dim: block ptr
            (
                (15, 9),
                (15, 3),
                None,
                None,
                False,
            ),  # Non-power-of-2 inner dims: non-block ptr
            ((1, 1, 1), (1, 1, 1), None, None, False),  # Scalar: non-block ptr
            (
                (2, 4 * max_block),
                (2, 3 * max_block),
                None,
                None,
                True,
            ),  # Inner dim multiple of max_block
        ],
    )
    def test_pointwise(
        self,
        full_size: Tuple[int],
        view_size: Tuple[int],
        stride: Optional[Tuple[int]],
        offset: Optional[int],
        require_block_ptr: bool,
    ):
        """
        Test generating strided ND block pointers for a pointwise kernel.

        If require_block_ptr is True, the generated code must contain block
        pointers. However, ND block pointers are not supported for all shapes. So
        we also test some odd shapes with require_block_ptr set to False, to ensure that
        block pointer analysis does not break these cases.
        """

        def get_input() -> torch.Tensor:
            device = torch.device(GPU_TYPE)
            full = torch.randn(full_size).to(device)

            # Use the original tensor's stride by default
            view_stride = full.stride() if stride is None else stride

            return torch.as_strided(full, view_size, view_stride, storage_offset=offset)

        args = [get_input() for arg_idx in range(2)]

        # Expect 3 block pointers: 2 inputs 1 output
        self.run_and_compare(
            torch.add,
            *args,
            expected_num_block_pointers=3 if require_block_ptr else None,
        )
    @parametrize(
        "x_size,y_size",
        [
            ((8, 8), (8, 1)),  # 定义测试参数：两个输入的尺寸分别为 (8, 8) 和 (8, 1)
            ((8, 8), (1, 8)),  # 定义测试参数：两个输入的尺寸分别为 (8, 8) 和 (1, 8)
            (
                (4, 1, 4),
                (1, 4, 1),
            ),  # 定义测试参数：两个输入的尺寸分别为 (4, 1, 4) 和 (1, 4, 1)，这是一个非常重要的情况：索引变量不重叠！
            (
                (1, 1, 1, 4),
                (4, 4, 4, 4),
            ),  # 定义测试参数：两个输入的尺寸分别为 (1, 1, 1, 4) 和 (4, 4, 4, 4)，第一个操作数的维度不匹配。
        ],
    )
    def test_broadcast(self, x_size: Tuple[int], y_size: Tuple[int]):
        """
        测试当输入具有不同形状且可以广播在一起时，是否能生成步进块指针。
        """

        def foo(x, y):
            a = x + 1  # 计算 x + 1
            b = y * 2  # 计算 y * 2
            return a + b  # 返回 a + b

        def get_input(view_size: Tuple[int]) -> torch.Tensor:
            device = torch.device(GPU_TYPE)  # 获取GPU类型的设备
            full_size = tuple(2 * dim for dim in view_size)  # 计算扩展后的完整尺寸
            full = torch.randn(full_size).to(device)  # 在GPU上生成随机数据
            view = torch.as_strided(full, view_size, full.stride())  # 使用给定的视图尺寸创建步进视图
            return view  # 返回步进视图

        x, y = (get_input(size) for size in (x_size, y_size))  # 获得输入 x 和 y

        # 检查输入的尺寸是否不同
        self.assertNotEqual(x.shape, y.shape)

        # 检查至少一个维度是否为单例
        all_dims = x.shape + y.shape  # 将所有维度组合起来
        self.assertIn(1, all_dims)

        # 预期 3 个块指针：2 个输入和 1 个输出
        self.run_and_compare(foo, x, y, expected_num_block_pointers=3)

    @parametrize(
        "view_size,num_block_pointers,num_triton_kernels",
        [
            ((4, 4), 1, 1),  # 定义测试参数：视图尺寸为 (4, 4)，预期块指针数为 1，Triton内核数为 1
            ((4, 4, 4), 1, 1),  # 定义测试参数：视图尺寸为 (4, 4, 4)，预期块指针数为 1，Triton内核数为 1
            ((8, 8, 8), 1, 1),  # 定义测试参数：视图尺寸为 (8, 8, 8)，预期块指针数为 1，Triton内核数为 1
            ((15, 15), 0, 1),  # 定义测试参数：视图尺寸为 (15, 15)，预期块指针数为 0，Triton内核数为 1（非2的幂）
            ((3 * max_block, 2), 3, 2),  # 定义测试参数：视图尺寸为 (3 * max_block, 2)，预期块指针数为 3，Triton内核数为 2（max_block的倍数，使用循环）
            (
                (2, 3 * max_block),
                3,
                2,
            ),  # 定义测试参数：视图尺寸为 (2, 3 * max_block)，预期块指针数为 3，Triton内核数为 2（max_block的倍数，使用循环）
            ((128, 128), 3, 2),  # 定义测试参数：视图尺寸为 (128, 128)，预期块指针数为 3，Triton内核数为 2（大尺寸测试，使用循环）
        ],
    )
    def test_reduction(
        self, view_size: Tuple[int], num_block_pointers: int, num_triton_kernels: int
    ):
        """
        测试一个减少内核。
        """

        device = torch.device(GPU_TYPE)  # 获取GPU类型的设备
        full_size = tuple(2 * dim for dim in view_size)  # 计算扩展后的完整尺寸
        full = torch.randn(full_size).to(device)  # 在GPU上生成随机数据
        view = torch.as_strided(full, view_size, full.stride())  # 使用给定的视图尺寸创建步进视图

        # 预期至少有 1 个块指针用于输入。
        # 如果生成了 2 个内核，则添加 2 个更多的。
        result, (code,) = self.run_and_compare(
            torch.sum,
            view,
            expected_num_block_pointers=num_block_pointers,
            expected_num_triton_kernels=num_triton_kernels,
        )

    @parametrize(
        "view_size,num_block_pointers,num_triton_kernels",
        [
            ((8, 8), 2, 1),  # 定义测试参数：视图尺寸为 (8, 8)，预期块指针数为 2，预期 Triton 内核数为 1（没有循环，应该支持）。
            (
                (128, 128),
                None,
                None,
            ),  # 定义测试参数：视图尺寸为 (128, 128)，预期块指针数和 Triton 内核数为 None（循环减少，块指针目前不支持）。
        ],
    )
    # 定义测试方法，测试混合使用逐点操作和归约操作的情况
    def test_mixed_pointwise_reduction(
        self, view_size: Tuple[int], num_block_pointers: int, num_triton_kernels: int
    ):
        """
        Tests mixing pointwise with reduction ops.
        """

        # 定义一个简单的函数，对两个张量进行逐元素相加后求和
        def foo(x, y):
            return torch.sum(x + y)

        # 设置设备类型为GPU
        device = torch.device(GPU_TYPE)
        # 计算完整张量的尺寸为视图尺寸的两倍
        full_size = tuple(2 * dim for dim in view_size)

        # 定义一个获取输入张量的函数
        def get_input() -> torch.Tensor:
            # 生成随机张量并发送到指定设备
            full = torch.randn(full_size).to(device)
            # 创建一个视图张量，使用全张量的数据，指定视图尺寸和步长
            view = torch.as_strided(full, view_size, full.stride())
            return view

        # 生成两个输入张量列表
        inputs = [get_input() for input_idx in range(2)]

        # 期望有2个块指针：输入张量
        result, (code,) = self.run_and_compare(
            foo,
            *inputs,
            expected_num_block_pointers=num_block_pointers,
            expected_num_triton_kernels=num_triton_kernels,
        )

    # 测试多个最大块且不为2的幂次方的维度
    def test_multiple_max_block_non_power_of_2(self):
        """
        Check that we support dims of size n * MAX_BLOCK, where n is any positive integer, not
        necessarily a power of 2.
        """

        # 定义一个简单的函数，对输入张量每个元素减去1
        def foo(x):
            return x - 1

        # 设置设备类型为GPU
        device = torch.device(GPU_TYPE)
        # 设置完整张量尺寸为3 * MAX_BLOCK 和 3
        full_size = (3 * max_block, 3)
        # 设置视图张量尺寸为3 * MAX_BLOCK 和 2
        view_size = (3 * max_block, 2)
        # 生成随机张量并发送到指定设备
        full = torch.randn(full_size).to(device)
        # 创建一个视图张量，使用全张量的数据，指定视图尺寸和步长
        view = torch.as_strided(full, view_size, full.stride())

        # 检查是否使用了非2的幂次方维度
        have_np2_dim = not all(is_power_of_2(dim) for dim in view_size)
        self.assertTrue(have_np2_dim)

        # 检查是否需要多个步长来表示张量
        nontrivial_dims = [dim for dim in view_size if dim > 1]
        self.assertTrue(len(nontrivial_dims) > 1)

        # 期望有2个块指针：输入和输出张量
        self.run_and_compare(foo, view, expected_num_block_pointers=2)

    # 测试动态形状的通用案例
    def test_dynamic_shapes_generic(self):
        """
        Test a generic strided block with dynamic shapes. Block pointers are not
        expected. This only checks that the analysis doesn't break this case.
        """

        # 设置设备类型为GPU
        device = torch.device(GPU_TYPE)
        # 设置完整张量尺寸为8 * 8
        full_size = (8, 8)
        # 设置视图张量尺寸为4 * 4
        view_size = (4, 4)
        # 生成随机张量并发送到指定设备
        full = torch.randn(full_size).to(device)
        # 创建一个视图张量，使用全张量的数据，指定视图尺寸和步长
        view = torch.as_strided(full, view_size, full.stride())

        # 运行比较函数，期望不产生块指针，使用动态形状参数编译
        self.run_and_compare(torch.div, view, view, compile_kwargs={"dynamic": True})

    # 标记为跳过的测试，原因是 Dynamo 跟踪错误
    @unittest.skip(reason="Dynamo tracing error")
    def test_dynamic_shapes_multiple_max_block(self):
        """
        Test dynamic shapes, where we know the shape is a multiple of the max block
        size. We should be able to generate a block pointer for this case.
        """

        def foo(x):
            # 计算瓦片的维度，确保其为最大块大小的倍数
            tile_dims = (3 * max_block * x.shape[0], 3 * x.shape[1])
            # 计算视图的大小，确保其为最大块大小的倍数
            view_size = (3 * max_block * x.shape[0], 2 * x.shape[1])
            # 对输入张量进行瓦片化
            full = x.tile(tile_dims)
            # 创建一个基于已有数据的新视图
            view = torch.as_strided(full, view_size, full.stride())
            # 返回视图和其自身的和
            return view + view

        device = torch.device(GPU_TYPE)
        x_size = (1, 1)
        x = torch.randn(x_size).to(device)

        # 预期生成 2 个块指针：输入和输出
        self.run_and_compare(
            x, compile_kwargs={"dynamic": True}, expected_num_block_pointers=2
        )
if __name__ == "__main__":
    # 检查当前模块是否作为主程序运行

    from torch._inductor.test_case import run_tests
    # 导入 torch 模块中的测试运行函数 run_tests

    if HAS_GPU:
        # 如果存在 GPU（假设 HAS_GPU 是一个表示是否存在 GPU 的变量）
        
        run_tests(needs="filelock")
        # 运行测试，需要使用 filelock 功能
```