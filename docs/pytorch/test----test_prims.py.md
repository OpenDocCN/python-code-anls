# `.\pytorch\test\test_prims.py`

```py
# Owner(s): ["module: decompositions"]

# 导入必要的库和模块
from functools import partial
from itertools import product
import unittest

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (parametrize, run_tests, TestCase, TEST_SCIPY,
                                                  set_default_dtype)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    dtypes,
    OpDTypes,
)
from torch.testing._internal.common_methods_invocations import (
    op_db,
)
from torch.testing._internal.common_device_type import (
    ops,
)

# 导入日志相关模块
from torch.testing._internal.logging_tensor import LoggingTensor, capture_logs, log_input
import torch._prims as prims
from torch._prims_common import CUDARngStateHelper
from torch._prims.executor import make_traced
import torch._refs as refs

# 如果设置了测试 scipy，导入 scipy.special
if TEST_SCIPY:
    import scipy.special

# NVPRIM_ATEN_FALLBACK_WARNING 和 GET_ISOLATED_GRAPHMODULE_ERROR 的常量定义
NVPRIM_ATEN_FALLBACK_WARNING = "fallback to aten executor"
GET_ISOLATED_GRAPHMODULE_ERROR = "get_isolated_graphmodule failed on decomposition"

# 定义测试类 TestPrims，继承自 unittest.TestCase
class TestPrims(TestCase):

    # 限定只在 CUDA 设备上测试，并且只测试 torch.float32 类型的数据
    @onlyCUDA
    @dtypes(torch.float32)
    def test_broadcast_in_dim(self, device, dtype):
        # 定义内部函数 _wrapper，用于调用 prims.broadcast_in_dim 方法
        def _wrapper(a, b, broadcast_dimensions):
            return prims.broadcast_in_dim(a, b.shape, broadcast_dimensions)

        # 对 _wrapper 函数进行追踪
        traced = make_traced(_wrapper)
        
        # 创建张量的部分函数 make_arg，指定设备和数据类型
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        # 遍历执行器类型 ('aten',)
        for executor in ('aten',):
            fn = partial(traced, executor=executor)
            
            # 测试情况1：相同形状的张量
            shape = (5, 5)
            a = make_arg(shape)
            b = make_arg(shape, low=0.0, high=0.0)
            result = fn(a, b, (0, 1))

            # 断言结果的形状与 a 相同，且是连续的张量
            self.assertEqual(result.shape, a.shape)
            self.assertTrue(result.is_contiguous)
            self.assertEqual(a, result)

            # 测试情况2：错误的输入，重新排序维度应当抛出异常
            with self.assertRaises(Exception):
                result = fn(a, b, (1, 0))

            # 测试情况3：在外部维度上添加张量
            a = make_arg((5, 5))
            b = make_arg((3, 3, 5, 5), low=0.0, high=0.0)
            result = fn(a, b, (2, 3))

            # 断言结果的形状与 b 相同，且与 a 广播后形状相同
            self.assertEqual(result.shape, b.shape)
            self.assertEqual(a.broadcast_to(b.shape), result)

            # 测试情况4：扩展维度
            a = make_arg((1, 5, 1))
            b = make_arg((3, 5, 7), low=0.0, high=0.0)
            result = fn(a, b, (0, 1, 2))

            # 断言结果的形状与 b 相同，且与 a 扩展后形状相同
            self.assertEqual(result.shape, b.shape)
            self.assertEqual(a.expand_as(result), result)

            # 测试情况5：插入新维度
            a = make_arg((1, 2, 3))
            b = make_arg((1, 2, 1, 3), low=0.0, high=0.0)
            result = fn(a, b, (0, 1, 3))

            # 断言结果的形状与 b 相同，且插入新维度后形状与 a 相同
            self.assertEqual(result.shape, b.shape)
            self.assertEqual(a.unsqueeze(2), result)
    # 定义一个测试方法，用于测试在指定设备和数据类型下的广播和求和操作
    def test_broadcast_in_dim_sum(self, device, dtype):
        # 定义内部函数_wrapper，接受参数a，并进行求和操作
        def _wrapper(a):
            # 计算a在指定维度[0, 1]上的和
            a_sum = prims.sum(a, [0, 1])
            # 将求和结果进行广播，返回广播后的张量
            a_bc = prims.broadcast_in_dim(a_sum, [], [])
            return a_bc

        # 生成一个traced版本的_wrapper函数
        traced = make_traced(_wrapper)
        # 创建一个偏函数make_arg，用于生成指定设备和数据类型的张量
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        # 针对不同的执行器（executor），如'aten'，进行测试
        for executor in ('aten',):
            # 生成一个执行函数fn，将traced函数作为参数，并绑定executor
            fn = partial(traced, executor=executor)
            # 创建一个形状为(5, 5)的张量a
            shape = (5, 5)
            a = make_arg(shape)
            # 执行fn函数，计算结果
            result = fn(a)

            # 断言结果的形状为()
            self.assertEqual(result.shape, ())
            # 断言结果是连续的张量
            self.assertTrue(result.is_contiguous)
            # 断言_wrapper(a)的结果等于fn(a)的结果
            self.assertEqual(_wrapper(a), result)

    # 根据是否存在SciPy模块进行测试跳过决策
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    # 根据指定的数据类型进行测试，包括torch.float64和torch.long
    @dtypes(torch.float64, torch.long)
    def test_cbrt_prim(self, device, dtype):
        # 生成一个偏函数make_arg，用于生成指定设备和数据类型的张量
        make_arg = partial(make_tensor, device=device, dtype=dtype)
        # 定义不同的批次batches和形状shapes
        batches = [(), (1,), (2,), (0, 1), (1, 1), (2, 2)]
        shapes = [(), (0,), (1,), (5,)]

        # 设置默认数据类型为双精度浮点数
        with set_default_dtype(torch.double):
            # 遍历batches和shapes的组合
            for b, s in product(batches, shapes):
                # 生成形状为b+s的张量x
                x = make_arg(b + s)
                # 计算x的立方根
                y = prims.cbrt(x)

                # 将x转换为NumPy数组，并计算其立方根
                x_np = x.cpu().numpy()
                y_np = scipy.special.cbrt(x_np)

                # 断言y等于y_np，允许设备不匹配
                self.assertEqual(y, y_np, exact_device=False)

    # 根据指定的数据类型进行测试，包括torch.float32
    @dtypes(torch.float32)
    def test_collapse(self, device, dtype):
        # 创建一个形状为(2, 2, 2)的随机张量t
        t = torch.rand(2, 2, 2)
        # 定义维度范围dim_ranges和期望的形状expected_shapes的对应关系
        dim_ranges = [(0, 0), (0, 1), (1, 2), (0, 2)]
        expected_shapes = [(2, 2, 2), (4, 2), (2, 4), (8,)]

        # 遍历dim_ranges和expected_shapes的组合
        for (start, end), shape in zip(dim_ranges, expected_shapes):
            # 期望的张量形状为shape，进行形状重塑
            expect = t.reshape(shape)

            # 调用prims.collapse函数，将张量t在指定维度范围[start, end)上进行合并操作
            copy = prims.collapse(t, start, end)
            # 断言copy等于expect
            self.assertEqual(copy, expect)
            # 断言copy不是视图
            self.assertFalse(copy._is_view())

            # 调用prims.collapse_view函数，获取t在指定维度范围[start, end)上的视图
            view = prims.collapse_view(t, start, end)
            # 断言view等于expect
            self.assertEqual(view, expect)
            # 断言view是视图
            self.assertTrue(view._is_view())

        # 将t转置为不连续的张量t_discontig
        t_discontig = t.transpose(0, 1)
        # 断言调用prims.collapse_view函数时，抛出ValueError异常，表示没有这样的视图存在
        with self.assertRaises(ValueError, msg="no such view exists"):
            view = prims.collapse_view(t_discontig, 0, 2)

        # 调用prims.collapse函数，将不连续张量t_discontig在维度范围[0, 1)上进行合并操作
        copy = prims.collapse(t_discontig, 0, 1)
        # 断言copy等于t_discontig重塑为(4, 2)的形状
        self.assertEqual(copy, t_discontig.reshape(4, 2))

        # 定义错误的维度范围error_dims
        error_dims = [(-1, 1), (0, 3), (1, -1)]
        # 遍历error_dims中的错误维度范围
        for start, end in error_dims:
            # 遍历prims.collapse和prims.collapse_view两个函数
            for fn in [prims.collapse, prims.collapse_view]:
                # 断言调用fn函数时抛出AssertionError异常
                with self.assertRaises(AssertionError):
                    fn(t, start, end)
    # 定义一个测试方法，用于测试 torch.ops.aten 调用是否被替换为 refs
    def test_aten_overload_to_prims(self, device):
        # 引入必要的模块和类
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsMode

        # 创建一个指定设备上的随机张量 a
        a = torch.randn(3, 3, device=device)

        # 定义一个函数 func，对输入张量执行 torch.ops.aten.sigmoid.default 和 torch.ops.aten.digamma.default 操作
        def func(a):
            return torch.ops.aten.sigmoid.default(torch.ops.aten.digamma.default(a))

        # 使用 TorchRefsMode 上下文，这会启用 refs 模式
        with TorchRefsMode():
            # 使用 make_fx 将 func 转换为 FX 图模式，并传入张量 a
            gm = make_fx(func)(a)

        # 检查图中所有的 call_function 节点是否都是 prims 命名空间中的函数
        call_function_nodes = list(filter(lambda n: n.op == "call_function", gm.graph.nodes))
        # 判断是否所有节点的目标函数名以 "prims" 开头
        all_prims_namespace = all(
            node.target.name().startswith("prims") for node in call_function_nodes
        )
        # 断言所有节点都在 prims 命名空间中
        self.assertTrue(all_prims_namespace)

    # 根据 CUDA 设备运行的测试方法装饰器
    @onlyCUDA
    # 指定张量的数据类型为 float32
    @dtypes(torch.float32)
    # 参数化测试，测试修正值为 0 和 1 的情况
    @parametrize("correction", [0, 1])
    def test_var(self, device, dtype, correction):
        # 定义一个内部包装函数 _wrapper，用于计算输入张量的方差
        def _wrapper(a):
            return prims.var(a, [0, 1], correction=correction)

        # 使用 make_traced 将 _wrapper 函数转换为追踪对象
        traced = make_traced(_wrapper)
        # 创建张量的辅助函数 make_arg，指定设备和数据类型
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        # 对于每个执行器 executor in ('aten',) 进行迭代
        for executor in ('aten',):
            # 创建 fn 函数，将追踪对象 traced 和执行器 executor 绑定
            fn = partial(traced, executor=executor)
            # 创建形状为 (5, 5) 的测试张量 a
            shape = (5, 5)
            a = make_arg(shape)
            # 调用 fn 函数，计算结果
            result = fn(a)

            # 断言结果的形状为标量
            self.assertEqual(result.shape, ())
            # 断言结果张量是连续的
            self.assertTrue(result.is_contiguous)
            # 断言 fn(a) 的结果与 _wrapper(a) 的结果相等
            self.assertEqual(_wrapper(a), result)

    # 指定张量的数据类型为 float32 的测试方法装饰器
    @dtypes(torch.float32)
    # 定义一个测试函数，用于测试不同的内存格式和张量形状
    def test_memory_format_strides(self, device, dtype):
        # 定义一组张量的形状，包括空张量、一维、二维等
        shapes = (
            (),
            (0,),
            (1,),
            (5),  # 注意：这里应该是元组 (5,) 而不是整数 5
            (1, 0),
            (1, 1),
            (3, 7),
            (3, 0, 2),
            (1, 1, 2),
            (4, 1, 1),
            (7, 8, 9),
        )

        # 定义一组通道末尾排列的张量形状
        channels_last_shapes = (
            (0, 0, 0, 0),
            (1, 0, 3, 0),
            (0, 2, 3, 5),
            (2, 2, 2, 0),
            (5, 4, 3, 2),
            (8, 8, 7, 2),
            (9, 1, 3, 1),
            (4, 5, 8, 7)
        )

        # 定义一组通道末尾三维排列的张量形状
        channels_last_3d_shapes = (
            (0, 8, 7, 9, 2),
            (5, 0, 7, 9, 2),
            (5, 0, 7, 9, 0),
            (5, 8, 7, 9, 2),
            (5, 1, 7, 9, 2),
            (5, 1, 7, 9, 1),
        )

        # 定义不同形状和内存格式的组合
        pairs = (
            (shapes, torch.contiguous_format),
            (channels_last_shapes, torch.contiguous_format),
            (channels_last_3d_shapes, torch.contiguous_format),
            (channels_last_shapes, torch.channels_last),
            (channels_last_3d_shapes, torch.channels_last_3d),
        )

        # 遍历所有形状和内存格式的组合
        for shapes, memory_format in pairs:
            for shape in shapes:
                # 测试空张量的情况
                expected = torch.empty(shape, device=device, dtype=dtype, memory_format=memory_format)
                actual = refs.empty(shape, device=device, dtype=dtype, memory_format=memory_format)
                self.assertEqual(expected.stride(), actual.stride())

                # 测试克隆操作
                a = torch.testing.make_tensor(shape, device=device, dtype=dtype)
                expected = torch.clone(a, memory_format=memory_format)
                actual = torch.clone(a, memory_format=memory_format)
                self.assertEqual(expected.stride(), actual.stride())

                # 测试连续性操作
                a = torch.testing.make_tensor(shape, device=device, dtype=dtype, noncontiguous=True)
                expected = a.contiguous(memory_format=memory_format)
                actual = refs.contiguous(a, memory_format=memory_format)
                self.assertEqual(expected.stride(), actual.stride())

    # 使用指定的数据类型进行测试的装饰器
    @dtypes(torch.float32)
    # 测试张量的重塑和视图方法
    def test_reshape_view_method(self, device, dtype):
        # 部分函数的参数化处理
        make_arg = partial(make_tensor, device=device, dtype=dtype)
        # 创建一个张量 a，形状为 (5, 5)
        a = make_arg((5, 5))
        # 定义新的张量形状
        new_shape = 1, 5, 1, 5
        # 使用 eager 模式对张量进行重塑操作，并进行断言比较
        result_eager = a.reshape(*new_shape)
        result_refs = refs.reshape(a, *new_shape)
        self.assertEqual(result_eager, result_refs)

        # 使用 view 方法对张量进行操作，并进行断言比较
        result_eager = a.view(*new_shape)
        result_refs = refs.view(a, *new_shape)
        self.assertEqual(result_eager, result_refs)

    # 仅在 CUDA 设备上执行的装饰器
    @onlyCUDA
    # 使用指定的数据类型进行测试的装饰器
    @dtypes(torch.float32)
    # 定义一个测试函数，用于测试 philox_rand 函数
    def test_philox_rand(self, device, dtype):
        # 定义两个不同大小的数据集
        sizes = (1000, 1000000)  # offsets of 4 and 8
        repeats = 2  # 多次测试随机数生成函数和 philox_rand 函数的结果
        for size in sizes:
            # 设置 CUDA 随机种子为 123
            torch.cuda.manual_seed(123)
            references = []  # 存储 torch.rand 的结果
            results = []  # 存储 philox_rand 的结果
            rng_states = []  # 存储随机数生成器的状态
            for _ in range(repeats):
                # 获取当前的 CUDA 随机状态并保存
                rng_states.append(CUDARngStateHelper.get_torch_state_as_tuple())
                # 生成随机数并保存结果到 references
                references.append(torch.rand(size, device=device, dtype=dtype))

            # 重设 CUDA 随机种子为 123
            torch.cuda.manual_seed(123)
            for idx in range(repeats):
                seed, offset = rng_states[idx]
                # 调用 philox_rand 函数生成随机数，并保存结果
                result, _ = torch.ops.rngprims.philox_rand((size,),
                                                           seed=seed,
                                                           offset=offset,
                                                           stride=None,
                                                           device=device,
                                                           dtype=dtype)
                results.append(result)

            # 对比 references 和 results 中的结果是否相等
            for a, b in zip(references, results):
                self.assertEqual(a, b)

    
    # 使用装饰器定义一个测试函数，用于测试 functional_rng_wrappers
    @dtypes(torch.float32)
    def test_functional_rng_wrappers(self, device, dtype):
        # 设置随机种子为 123
        torch.manual_seed(123)
        ref1 = torch.rand(10, device=device, dtype=dtype)  # 生成随机数 ref1
        ref2 = torch.rand(10, device=device, dtype=dtype)  # 生成随机数 ref2

        # 重新设置随机种子为 123，并运行并保存 RNG 状态
        torch.manual_seed(123)
        rng_state1, res1 = torch._prims.rng_prims.run_and_save_rng_state(torch.rand, 10, device=device, dtype=dtype)
        rng_state2, res2 = torch._prims.rng_prims.run_and_save_rng_state(torch.rand, 10, device=device, dtype=dtype)

        # 使用保存的 RNG 状态运行随机数生成函数，并保存结果
        res3 = torch._prims.rng_prims.run_with_rng_state(rng_state1, torch.rand, 10, device=device, dtype=dtype)
        res4 = torch._prims.rng_prims.run_with_rng_state(rng_state2, torch.rand, 10, device=device, dtype=dtype)

        # 对比生成的随机数结果是否与 ref1 和 ref2 相等
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)
        self.assertEqual(ref1, res3)
        self.assertEqual(ref2, res4)
class TestPrimsBasic(TestCase):
    # 定义测试类 TestPrimsBasic，继承自 TestCase

    def test_torch_ops(self):
        # 定义测试方法 test_torch_ops

        r = make_tensor((2,), device='cpu', dtype=torch.float)
        # 调用 make_tensor 函数创建一个形状为 (2,)，在 CPU 上的 float 类型张量 r

        self.assertEqual(torch.ops.prims.sin(r), torch.sin(r))
        # 断言 torch.ops.prims.sin(r) 的结果与 torch.sin(r) 相等

        r = LoggingTensor(r)
        # 创建一个 LoggingTensor 对象 r，用于记录张量操作

        with capture_logs() as logs:
            # 开始捕获日志
            log_input("input", r)
            # 记录输入 r 的日志
            prims.sin(r)
            # 调用 prims.sin 对象 r 进行操作

        self.assertExpectedInline('\n'.join(logs), """\
$0: f32[2] = input('input')
$1: f32[2] = torch._ops.prims.sin.default($0)""")
        # 断言捕获的日志与期望的日志字符串匹配

    def test_mul_complex(self):
        # 定义测试方法 test_mul_complex

        prims.mul(torch.randn(2), 1 + 1j)
        # 调用 prims.mul 函数，对一个形状为 (2,) 的随机张量与复数 1 + 1j 进行乘法操作

    def test_check_deprecation_warning(self):
        # 定义测试方法 test_check_deprecation_warning

        with self.assertWarnsRegex(FutureWarning, 'will be removed in the future'):
            # 断言捕获 FutureWarning 类型的警告，并检查警告消息是否包含 'will be removed in the future'
            torch._prims_common.check(True, lambda: 'message')
            # 调用 torch._prims_common.check 函数，并传递 True 和 lambda 函数作为参数
    # 定义一个测试方法，测试在复数输入情况下的 linspace 函数
    def test_linspace_with_complex_input(self):
        # 调用被测试的 refs.linspace 函数，生成实际结果
        actual = refs.linspace(2, 10 + 5j, steps=5)
        # 调用 PyTorch 的 linspace 函数，生成期望结果
        expect = torch.linspace(2, 10 + 5j, steps=5)
        # 使用测试框架的断言方法，比较实际结果与期望结果是否相等
        self.assertEqual(actual, expect)

    # 从给定的 GitHub issue 链接中复制，定义一个测试方法
    def test_infinite_loop_from_py_dispatcher(self):
        # 启用 Python 调度器，使得 prim decomps 生效
        with torch._dispatch.python.enable_python_dispatcher():
            # 创建一个全为1的张量
            x = torch.ones(4)
            # 将张量移动到指定设备，设备类型为 "meta"
            y = x.to(device="meta")
# 实例化设备类型测试，使用 TestRefs 类，测试用例函数从全局变量中获取
instantiate_device_type_tests(TestRefs, globals())


class TestDecomp(TestCase):
    @ops([op for op in op_db if op.supports_varargs], dtypes=OpDTypes.any_one)
    # 定义测试方法 test_decomposition_method_vararg，参数包括 device, dtype, op
    def test_decomposition_method_vararg(self, device, dtype, op):
        # 一些操作支持可变参数的方法，在此进行测试。
        # OpInfo 中没有针对可变参数的测试，因此需要进行适当的调整。
        # 通用函数的规则是，如果方法只有一个序列类型的参数，那么可以使用可变参数。
        # 例如，对于一个三维张量 t，permute 方法可以使用 t.permute(0, 2, 1)
        # 或者 t.permute([0, 2, 1])，在 native_functions.yaml 中的签名显示
        # 参数为 Tensor self, IntList dims。
        # 对于工厂函数，可能需要调整或者让它们自行测试。
        
        # 导入必要的模块和类
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsMode

        # 过滤掉空元组，因为它不能作为可变参数
        sample_inputs = (si for si in op.sample_inputs(device, dtype, requires_grad=False)
                         if (si.args[-1] if si.args else si.input))

        # 获取一个样例输入进行测试
        sample_input = next(sample_inputs)
        all_args = (sample_input.input,) + sample_input.args

        # 一般情况下，方法接受可变参数，而不是函数变体（通常如此？），唯一的例外是工厂函数
        if op.is_factory_function:
            fn = op.op
        else:
            fn = op.method_variant
        
        # 使用 TorchRefsMode 上下文环境
        with TorchRefsMode():
            # 创建函数 fx，并传入所有参数（除了最后一个以外）作为位置参数，最后一个参数作为可变参数
            gm = make_fx(fn)(*all_args[:-1], *all_args[-1])

        # 如果添加了随机工厂函数
        torch.manual_seed(1)
        res = gm(*all_args[:-1], *all_args[-1])  # 运行 gm 函数
        torch.manual_seed(1)
        expected = fn(*all_args[:-1], *all_args[-1])  # 计算期望结果

        # 断言测试结果与期望结果相等
        self.assertEqual(res, expected)


# 实例化设备类型测试，使用 TestDecomp 类，测试用例函数从全局变量中获取
instantiate_device_type_tests(TestDecomp, globals())


if __name__ == "__main__":
    # 运行测试
    run_tests()
```