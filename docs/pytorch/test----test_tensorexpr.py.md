# `.\pytorch\test\test_tensorexpr.py`

```py
# Owner(s): ["NNC"]

import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的函数模块
from torch import nn  # 导入 PyTorch 的神经网络模块
import unittest  # 导入单元测试模块
import itertools  # 导入迭代工具模块

from torch.testing._internal.common_utils import suppress_warnings, num_profiled_runs, run_tests, skipIfTorchDynamo  # 导入测试工具函数和装饰器

from torch.testing._internal.jit_utils import JitTestCase, TensorExprTestOptions  # 导入 JIT 测试相关的工具类和选项

LLVM_ENABLED = torch._C._llvm_enabled()  # 检查 LLVM 是否已启用

class BaseTestClass(JitTestCase):
    def setUp(self):
        super().setUp()
        self.tensorexpr_options = TensorExprTestOptions()  # 初始化 TensorExpr 的测试选项
        self.devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']  # 根据 CUDA 是否可用选择设备
        self.dtypes = [torch.float32, torch.bfloat16] if LLVM_ENABLED else [torch.float32]  # 根据 LLVM 是否启用选择数据类型

    def tearDown(self):
        self.tensorexpr_options.restore()  # 恢复 TensorExpr 的测试选项
        super().tearDown()

    def assertLastGraphAllFused(self):
        self.assertAllFused(torch.jit.last_executed_optimized_graph())  # 断言最后执行的优化图中所有操作是否都被融合了


def warmup_and_run_forward(f, *args):
    for _ in range(torch._C._jit_get_num_profiled_runs() + 1):
        results = f(*args)  # 执行函数并获取结果
    return results


@skipIfTorchDynamo()
class TestTensorExprFuser(BaseTestClass):
    def test_easy(self):
        def easy(x, y):
            aaa = torch.add(x, y)  # 计算 x + y
            return aaa

        traced = torch.jit.trace(easy, (torch.rand(1024), torch.rand(1024)))  # 对 easy 函数进行 JIT 追踪

        a = torch.rand(1024)
        b = torch.rand(1024)
        x = warmup_and_run_forward(traced, a, b)  # 运行 JIT 追踪后的模型
        self.assertLastGraphAllFused()  # 断言最后执行的优化图中所有操作是否都被融合了
        np.testing.assert_allclose(a.numpy() + b.numpy(), x.numpy())  # 使用 NumPy 断言结果是否正确

    def test_three_arg(self):
        def easy(x, y, z):
            aaa = torch.add(x, y)  # 计算 x + y
            bbb = torch.add(aaa, z)  # 计算 (x + y) + z
            return bbb

        traced = torch.jit.trace(
            easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024))
        )  # 对 easy 函数进行 JIT 追踪

        a = torch.rand(1024)
        b = torch.rand(1024)
        c = torch.rand(1024)
        x = warmup_and_run_forward(traced, a, b, c)  # 运行 JIT 追踪后的模型
        self.assertLastGraphAllFused()  # 断言最后执行的优化图中所有操作是否都被融合了
        npr = a.numpy() + b.numpy() + c.numpy()
        np.testing.assert_allclose(npr, x.numpy())  # 使用 NumPy 断言结果是否正确
    # 定义一个测试方法，测试四个参数的函数
    def test_four_arg(self):
        # 定义一个函数，对四个张量执行 torch.addcmul 操作并返回结果
        def run_addcmul(x, y, z, w):
            c = torch.addcmul(torch.add(x, y), z, w)
            return c

        # 遍历设备列表中的每个设备
        for dev in self.devices:
            # 在当前设备上生成随机的 1024 维浮点数张量 rand_a, rand_b, rand_c, rand_d
            rand_a = torch.rand(1024, dtype=torch.float, device=dev)
            rand_b = torch.rand(1024, dtype=torch.float, device=dev)
            rand_c = torch.rand(1024, dtype=torch.float, device=dev)
            rand_d = torch.rand(1024, dtype=torch.float, device=dev)

            # 使用 torch.jit.trace 对 run_addcmul 函数进行追踪
            traced = torch.jit.trace(
                run_addcmul,
                (
                    torch.zeros(1024, dtype=torch.float, device=dev),
                    torch.zeros(1024, dtype=torch.float, device=dev),
                    torch.zeros(1024, dtype=torch.float, device=dev),
                    torch.zeros(1024, dtype=torch.float, device=dev),
                ),
            )

            # 对追踪后的模型进行预热和前向运行
            x = warmup_and_run_forward(traced, rand_a, rand_b, rand_c, rand_d)
            # 断言最后一个图中的所有操作都被融合了
            self.assertLastGraphAllFused()
            # 直接调用 run_addcmul 函数获取结果 y
            y = run_addcmul(rand_a, rand_b, rand_c, rand_d)
            # 使用 NumPy 测试确保 x 和 y 的值在给定的容差范围内一致
            np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy(), atol=1e-6)

    # 定义一个测试方法，测试三个参数的函数（第二个版本）
    def test_three_arg2(self):
        # 遍历设备列表中的每个设备
        for device in self.devices:
            # 定义一个函数，对三个张量执行两次 torch.add 操作并返回结果
            def test(x, y, z):
                aaa = torch.add(x, y)
                bbb = torch.add(aaa, z)
                return bbb

            M = 32
            N = 32
            # 使用 torch.jit.trace 对 test 函数进行追踪
            traced = torch.jit.trace(
                test,
                (
                    torch.rand(M, N, device=device),
                    torch.rand(M, N, device=device),
                    torch.rand(M, N, device=device),
                ),
            )

            # 在当前设备上生成随机的 MxN 维浮点数张量 a, b, c
            a = torch.rand(M, N, device=device)
            b = torch.rand(M, N, device=device)
            c = torch.rand(M, N, device=device)
            # 对追踪后的模型进行预热和前向运行
            x = traced(a, b, c)
            x = warmup_and_run_forward(traced, a, b, c)
            # 断言最后一个图中的所有操作都被融合了
            self.assertLastGraphAllFused()
            # 计算 NumPy 数组 npr，包含 a、b、c 三个张量的对应元素之和
            npr = a.cpu().numpy() + b.cpu().numpy() + c.cpu().numpy()
            # 使用 NumPy 测试确保 x 和 npr 的值在给定的容差范围内一致
            np.testing.assert_allclose(npr, x.cpu().numpy())
    def test_broadcast3(self):
        # 遍历每个设备进行测试
        for device in self.devices:
            # 定义测试主体函数
            def test_body(M, N, L, K):
                # 定义测试函数
                def test(x, y, z):
                    # 执行张量加法
                    v1 = torch.add(x, y)
                    v2 = torch.add(v1, z)
                    return v2

                # 设置张量的形状
                a_shape = [M, N]
                b_shape = [L, M, 1]
                c_shape = [K, L, 1, 1]
                # 使用 torch.jit.trace 进行函数追踪
                traced = torch.jit.trace(
                    test,
                    (
                        torch.rand(*a_shape, device=device),
                        torch.rand(*b_shape, device=device),
                        torch.rand(*c_shape, device=device),
                    ),
                )

                # 创建随机张量
                a = torch.rand(*a_shape, device=device)
                b = torch.rand(*b_shape, device=device)
                c = torch.rand(*c_shape, device=device)
                # 运行前向传播
                x = warmup_and_run_forward(traced, a, b, c)
                # 断言最后的计算图全部被融合
                self.assertLastGraphAllFused()
                # 计算 NumPy 数组并验证结果的近似性
                npr = a.cpu().numpy() + b.cpu().numpy() + c.cpu().numpy()
                np.testing.assert_allclose(npr, x.cpu().numpy())

            # 测试配置列表
            test_configs = [[5, 2, 7, 3], [8, 8, 8, 8]]
            for test_config in test_configs:
                test_body(*test_config)

    def test_all_combos(self):
        # 定义简单的测试函数
        def easy(x, y, z):
            a = torch.add(x, y)
            b = torch.add(a, z)
            c = torch.add(x, b)
            d = torch.add(c, a)
            return d

        # 定义 NumPy 简单测试函数
        def np_easy(x, y, z):
            a = x + y
            b = a + z
            c = x + b
            d = c + a
            return d

        # 使用 torch.jit.trace 进行函数追踪
        traced = torch.jit.trace(
            easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024))
        )

        # 创建随机张量
        a = torch.rand(1024)
        b = torch.rand(1024)
        c = torch.rand(1024)
        # 运行前向传播
        x = warmup_and_run_forward(traced, a, b, c)
        # 断言最后的计算图全部被融合
        self.assertLastGraphAllFused()
        # 计算 NumPy 数组并验证结果的近似性
        npr = np_easy(a.numpy(), b.numpy(), c.numpy())
        np.testing.assert_allclose(npr, x.numpy())

    def test_rank_two(self):
        # 定义简单的测试函数
        def easy(x, y, z):
            a = torch.add(x, y)
            b = torch.add(a, z)
            c = torch.add(x, b)
            d = torch.add(c, a)
            return d

        # 定义 NumPy 简单测试函数
        def np_easy(x, y, z):
            a = x + y
            b = a + z
            c = x + b
            d = c + a
            return d

        # 设置张量的形状
        shape = 32, 32
        # 使用 torch.jit.trace 进行函数追踪
        traced = torch.jit.trace(
            easy, (torch.rand(shape), torch.rand(shape), torch.rand(shape))
        )

        # 创建随机张量
        a = torch.rand(shape)
        b = torch.rand(shape)
        c = torch.rand(shape)
        # 运行前向传播
        x = warmup_and_run_forward(traced, a, b, c)
        # 断言最后的计算图全部被融合
        self.assertLastGraphAllFused()
        # 计算 NumPy 数组并验证结果的近似性
        npr = np_easy(a.numpy(), b.numpy(), c.numpy())
        np.testing.assert_allclose(npr, x.numpy())
    # 定义一个测试方法，用于测试广播功能
    def test_broadcast(self):
        # 定义一个内部函数 easy，接受三个参数并进行张量运算
        def easy(x, y, z):
            # 计算 x 和 y 的张量加法
            a = torch.add(x, y)
            # 计算 a 和 z 的张量加法
            b = torch.add(a, z)
            return b

        # 定义一个类似功能的 NumPy 版本函数 np_easy
        def np_easy(x, y, z):
            # 计算 x 和 y 的 NumPy 数组加法
            a = x + y
            # 计算 a 和 z 的 NumPy 数组加法
            b = a + z
            return b

        # 设置常量 N = 32
        N = 32
        # 使用 torch.jit.trace 对 easy 函数进行追踪，传入随机生成的张量作为参数
        traced = torch.jit.trace(easy, (torch.rand(N, N), torch.rand(N), torch.rand(N, N)))

        # 创建三个随机张量 a, b, c
        a = torch.rand(N, N)
        b = torch.rand(N)
        c = torch.rand(N, N)
        # 调用 warmup_and_run_forward 函数执行追踪后的 traced 模型，并传入 a, b, c 作为参数
        x = warmup_and_run_forward(traced, a, b, c)
        # 断言最后一个图形是否完全融合
        self.assertLastGraphAllFused()
        # 调用 NumPy 版本的 np_easy 函数，并断言其返回值与 traced 模型的输出在数值上是否相近
        npr = np_easy(a.numpy(), b.numpy(), c.numpy())
        np.testing.assert_allclose(npr, x.numpy())

    # 定义第二个测试方法，用于测试广播功能的另一种情况
    def test_broadcast_2(self):
        # 创建一个零张量 zero
        zero = torch.tensor([0.0], dtype=torch.float)

        # 定义一个内部函数 foo，接受三个参数并进行张量运算
        def foo(x, y, z):
            # 计算 x 和 y 的张量加法
            aaa = torch.add(x, y)
            # 计算零张量和 aaa 的张量加法
            bbb = torch.add(zero, aaa)
            # 计算 bbb 和 z 的张量加法
            return torch.add(bbb, z)

        # 定义一个类似功能的 NumPy 版本函数 foo_np
        def foo_np(x, y, z):
            # 计算 x 和 y 的 NumPy 数组加法
            a = x + y
            # 计算零张量的 NumPy 数组和 a 的加法
            b = zero.numpy() + a
            # 计算 b 和 z 的加法
            return b + z

        # 创建三个随机张量 x, y, z
        x = torch.rand(3, 4)
        y = torch.ones(3, 1)
        z = torch.rand(4)
        # 使用 torch.jit.trace 对 foo 函数进行追踪，传入随机生成的张量作为参数
        traced = torch.jit.trace(foo, (x, y, z))

        # 调用 warmup_and_run_forward 函数执行追踪后的 traced 模型，并传入 x, y, z 作为参数
        r = warmup_and_run_forward(traced, x, y, z)
        # 断言最后一个图形是否完全融合
        self.assertLastGraphAllFused()

        # 调用 NumPy 版本的 foo_np 函数，并断言其返回值与 traced 模型的输出在数值上是否相近
        rnp = foo_np(x.numpy(), y.numpy(), z.numpy())
        np.testing.assert_allclose(r, rnp)

    # 定义第三个测试方法，用于测试大规模数据的广播功能
    def test_broadcast_big2(self):
        # 创建一个零张量 zero
        zero = torch.tensor([0.0], dtype=torch.float)

        # 定义一个内部函数 foo，接受三个参数并进行张量运算
        def foo(x, y, z):
            # 计算 x 和 y 的张量加法
            aaa = torch.add(x, y)
            # 计算零张量和 aaa 的张量加法
            bbb = torch.add(zero, aaa)
            # 计算 bbb 和 z 的张量加法
            return torch.add(bbb, z)

        # 定义一个类似功能的 NumPy 版本函数 foo_np
        def foo_np(x, y, z):
            # 计算 x 和 y 的 NumPy 数组加法
            a = x + y
            # 计算零张量的 NumPy 数组和 a 的加法
            b = zero.numpy() + a
            # 计算 b 和 z 的加法
            return b + z

        # 创建三个随机张量 x, y, z，形状分别为 (32, 1024), (32, 1), (1024)
        x = torch.rand(32, 1024)
        y = torch.ones(32, 1)
        z = torch.rand(1024)
        # 使用 torch.jit.trace 对 foo 函数进行追踪，传入随机生成的张量作为参数
        traced = torch.jit.trace(foo, (x, y, z))

        # 调用 warmup_and_run_forward 函数执行追踪后的 traced 模型，并传入 x, y, z 作为参数
        r = warmup_and_run_forward(traced, x, y, z)
        # 断言最后一个图形是否完全融合
        self.assertLastGraphAllFused()

        # 调用 NumPy 版本的 foo_np 函数，并断言其返回值与 traced 模型的输出在数值上是否相近
        rnp = foo_np(x.numpy(), y.numpy(), z.numpy())
        np.testing.assert_allclose(r, rnp)

    # 定义一个测试方法，用于测试带有 alpha 参数的张量运算
    def test_alpha(self):
        # 定义一个内部函数 alpha，接受一个参数并进行带 alpha 参数的张量加法
        def alpha(x):
            aaa = torch.add(x, x, alpha=2.0)
            return aaa

        # 使用 torch.jit.trace 对 alpha 函数进行追踪，传入张量 [1.0] 作为参数
        traced = torch.jit.trace(alpha, (torch.tensor([1.0])))

        # 创建一个张量 a，值为 [1.0]
        a = torch.tensor([1.0])
        # 调用追踪后的 traced 模型，并传入 a 作为参数，得到输出 x
        x = traced(a)
        # 使用 NumPy 计算 a + 2.0 * a，并断言其与模型输出 x 在数值上是否相近
        np.testing.assert_allclose(a.numpy() + 2.0 * a.numpy(), x.numpy())

    # 定义一个带有警告抑制装饰器的测试方法，用于测试常量张量的加法运算
    @suppress_warnings
    def test_constant(self):
        # 定义一个内部函数 constant，接受一个参数并进行与常量张量的加法运算
        def constant(x):
            # 创建一个值为 [1.0] 的常量张量 bbb
            bbb = torch.tensor([1.0])
            # 计算输入张量 x 和常量张量 bbb 的加法
            aaa = torch.add(x, bbb)
            return aaa

        # 使用 torch.jit.trace 对 constant 函数进行追踪，传入张量 [1.0] 作为参数
        traced = torch.jit.trace(constant, (torch.tensor([1.0])))

        # 创建一个张量 a，值为 [1.0]
        a = torch.tensor([1.0])
        # 调用 warmup_and_run_forward 函数执行追踪后的 traced 模型，并传入 a 作为参数，得到输出 x
        x = warmup_and_run_forward(traced, a)
        # 断言最后一个图形是否完全融合
        self.assertLastGraphAllFused()
        # 使用 NumPy 计算 a + 1.0，并断言其与模
    def test_add_sub(self):
        # 定义一个简单的函数 easy，接收三个参数 x, y, z
        def easy(x, y, z):
            # 计算 x 和 y 的加法
            aaa = torch.add(x, y)
            # 计算 aaa 和 z 的减法
            bbb = torch.sub(aaa, z)
            return bbb

        # 使用 torch.jit.trace 对 easy 函数进行追踪，以便进行 JIT 编译
        traced = torch.jit.trace(
            easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024))
        )

        # 创建三个随机张量 a, b, c
        a = torch.rand(1024)
        b = torch.rand(1024)
        c = torch.rand(1024)
        # 调用 warmup_and_run_forward 函数来运行追踪后的模型 traced
        x = warmup_and_run_forward(traced, a, b, c)
        # 断言最后一个图形中所有操作是否都被融合了
        self.assertLastGraphAllFused()
        # 使用 np.testing.assert_allclose 检查 a + b - c 的结果与 x 的结果是否近似相等
        np.testing.assert_allclose(a.numpy() + b.numpy() - c.numpy(), x.numpy())

    def test_promotion(self):
        # 定义一个简单的函数 easy，接收两个参数 x, y
        def easy(x, y):
            # 计算 x 和 y 的加法
            aaa = torch.add(x, y)
            return aaa

        # 使用 torch.jit.trace 对 easy 函数进行追踪，以便进行 JIT 编译
        traced = torch.jit.trace(
            easy,
            (torch.zeros(1024, dtype=torch.int32), torch.rand(1024, dtype=torch.float32)),
        )

        # 创建两个张量 a 和 b
        a = torch.zeros(1024, dtype=torch.int32)
        b = torch.rand(1024, dtype=torch.float32)
        # 调用 warmup_and_run_forward 函数来运行追踪后的模型 traced
        x = warmup_and_run_forward(traced, a, b)
        # 断言最后一个图形中所有操作是否都被融合了
        self.assertLastGraphAllFused()
        # 使用 np.testing.assert_allclose 检查 a + b 的结果与 x 的结果是否近似相等
        np.testing.assert_allclose(a.numpy() + b.numpy(), x.numpy())

    def test_double(self):
        TENSOR_LEN = 8

        # 定义一个简单的函数 easy，接收两个参数 x, y
        def easy(x, y):
            # 计算 x 和 y 的加法
            aaa = torch.add(x, y)
            # 计算 aaa 和 y 的乘法
            bbb = torch.mul(aaa, y)
            return bbb

        # 使用 torch.jit.trace 对 easy 函数进行追踪，以便进行 JIT 编译
        traced = torch.jit.trace(
            easy,
            (torch.rand(TENSOR_LEN, dtype=torch.float64), torch.full((TENSOR_LEN,), 0.5, dtype=torch.float64)),
        )

        # 创建两个双精度张量 a 和 b
        a = torch.rand(TENSOR_LEN, dtype=torch.double)
        b = torch.full((TENSOR_LEN,), 0.5, dtype=torch.double)
        # 调用 warmup_and_run_forward 函数来运行追踪后的模型 traced
        x = warmup_and_run_forward(traced, a, b)
        # 断言最后一个图形中所有操作是否都被融合了
        self.assertLastGraphAllFused()
        # 使用 np.testing.assert_allclose 检查 (a + b) * b 的结果与 x 的结果是否近似相等
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())

    def test_short(self):
        TENSOR_LEN = 8

        # 定义一个简单的函数 easy，接收两个参数 x, y
        def easy(x, y):
            # 计算 x 和 y 的加法
            aaa = torch.add(x, y)
            # 计算 aaa 和 y 的乘法
            bbb = torch.mul(aaa, y)
            return bbb

        # 使用 torch.jit.trace 对 easy 函数进行追踪，以便进行 JIT 编译
        traced = torch.jit.trace(
            easy,
            (torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int16),
             torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int16)),
        )

        # 创建两个 int16 类型的张量 a 和 b
        a = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int16)
        b = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int16)
        # 调用 warmup_and_run_forward 函数来运行追踪后的模型 traced
        x = warmup_and_run_forward(traced, a, b)
        # 断言最后一个图形中所有操作是否都被融合了
        self.assertLastGraphAllFused()
        # 使用 np.testing.assert_allclose 检查 (a + b) * b 的结果与 x 的结果是否近似相等
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())
    def test_char`
    # 测试字符张量运算
    def test_char(self):
        TENSOR_LEN = 8  # 定义张量长度

        # 定义一个简单的函数，进行加法和乘法运算
        def easy(x, y):
            aaa = torch.add(x, y)  # 对输入张量 x 和 y 进行加法操作
            bbb = torch.mul(aaa, y)  # 将加法结果 aaa 与 y 进行乘法操作
            return bbb

        # 使用 torch.jit.trace 对 easy 函数进行跟踪，输入是两个随机张量
        traced = torch.jit.trace(
            easy,
            (torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8),  # 随机生成一个 int8 类型的张量
             torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8)),  # 随机生成一个 int8 类型的张量
        )

        a = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8)  # 生成一个 int8 类型的随机张量 a
        b = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8)  # 生成一个 int8 类型的随机张量 b
        x = warmup_and_run_forward(traced, a, b)  # 调用 warmup_and_run_forward 函数，执行 traced 模型
        self.assertLastGraphAllFused()  # 断言图形已完全融合
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())  # 验证结果与预期一致

    # 测试 int64 类型张量的运算
    def test_int64_promotion(self):
        TENSOR_LEN = 8  # 定义张量长度

        # 定义一个简单的函数，进行加法和乘法运算
        def easy(x, y):
            aaa = torch.add(x, y)  # 对输入张量 x 和 y 进行加法操作
            bbb = torch.mul(aaa, y)  # 将加法结果 aaa 与 y 进行乘法操作
            return bbb

        # 使用 torch.jit.trace 对 easy 函数进行跟踪，输入是一个 int8 和一个 int64 类型的随机张量
        traced = torch.jit.trace(
            easy,
            (torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8),  # 随机生成一个 int8 类型的张量
             torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int64)),  # 随机生成一个 int64 类型的张量
        )

        a = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8)  # 生成一个 int8 类型的随机张量 a
        b = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int64)  # 生成一个 int64 类型的随机张量 b
        x = warmup_and_run_forward(traced, a, b)  # 调用 warmup_and_run_forward 函数，执行 traced 模型
        self.assertLastGraphAllFused()  # 断言图形已完全融合
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())  # 验证结果与预期一致

    # 测试等于操作
    def test_eq(self):
        # 定义一个简单的函数，进行等于比较
        def easy(x, y):
            c = torch.eq(x, y)  # 对输入张量 x 和 y 进行等于比较
            return c

        # 使用 torch.jit.trace 对 easy 函数进行跟踪，输入是两个全零的张量
        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        a = torch.zeros(1024, dtype=torch.int32)  # 生成一个 int32 类型的全零张量 a
        b = torch.zeros(1024, dtype=torch.int32)  # 生成一个 int32 类型的全零张量 b
        x = warmup_and_run_forward(traced, a, b)  # 调用 warmup_and_run_forward 函数，执行 traced 模型
        self.assertLastGraphAllFused()  # 断言图形已完全融合
        np.testing.assert_allclose(np.ones(1024), x.numpy())  # 验证结果与预期一致

    # 测试不等于操作
    def test_ne(self):
        # 定义一个简单的函数，不等于比较
        def easy(x, y):
            c = torch.ne(x, y)  # 对输入张量 x 和 y 进行不等于比较
            return c

        # 使用 torch.jit.trace 对 easy 函数进行跟踪，输入是两个全零的张量
        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        a = torch.zeros(1024, dtype=torch.int32)  # 生成一个 int32 类型的全零张量 a
        b = torch.ones(1024, dtype=torch.int32)  # 生成一个 int32 类型的全一张量 b
        x = warmup_and_run_forward(traced, a, b)  # 调用 warmup_and_run_forward 函数，执行 traced 模型
        self.assertLastGraphAllFused()  # 断言图形已完全融合
        np.testing.assert_allclose(np.ones(1024), x.numpy())  # 验证结果与预期一致

    # 测试大于等于操作
    def test_ge(self):
        # 定义一个简单的函数，大于等于比较
        def easy(x, y):
            c = torch.ge(x, y)  # 对输入张量 x 和 y 进行大于等于比较
            return c

        # 使用 torch.jit.trace 对 easy 函数进行跟踪，输入是两个全零的张量
        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        aa = np.empty([1024], dtype=np.int32)  # 创建一个空的 int32 类型 numpy 数组
        aa.fill(5)  # 填充数组为 5
        a = torch.from_numpy(aa)  # 将 numpy 数组转换为 torch 张量
        b = torch.zeros(1024, dtype=torch.int32)  # 生成一个 int32 类型的全零张量 b
        x = warmup_and_run_forward(traced, a, b)  # 调用 warmup_and_run_forward 函数，执行 traced 模型
        self.assertLastGraphAllFused()  # 断言图形已完全融合
        np.testing.assert_allclose(np.ones(1024), x.numpy())  # 验证结果与预期一致
    # 定义测试函数，使用 torch.gt 比较两个张量的大于关系，并返回结果张量
    def test_gt(self):
        # 定义一个简单的函数 easy，用于计算两个张量的大于关系
        def easy(x, y):
            c = torch.gt(x, y)
            return c

        # 使用 torch.jit.trace 对 easy 函数进行跟踪编译，以便后续优化和运行
        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        # 创建一个全为1的张量 a，数据类型为 torch.int32
        a = torch.ones(1024, dtype=torch.int32)
        # 创建一个全为0的张量 b，数据类型为 torch.int32
        b = torch.zeros(1024, dtype=torch.int32)
        # 调用 warmup_and_run_forward 函数执行跟踪后的 traced 模型，并传入参数 a 和 b
        x = warmup_and_run_forward(traced, a, b)
        # 断言最后一个计算图中所有操作都已融合
        self.assertLastGraphAllFused()
        # 使用 numpy.testing.assert_allclose 断言 x 的值与一个全为1的 numpy 数组非常接近
        np.testing.assert_allclose(np.ones(1024), x.numpy())

    # 定义测试函数，使用 torch.le 比较两个张量的小于等于关系，并返回结果张量
    def test_le(self):
        # 定义一个简单的函数 easy，用于计算两个张量的小于等于关系
        def easy(x, y):
            c = torch.le(x, y)
            return c

        # 使用 torch.jit.trace 对 easy 函数进行跟踪编译，以便后续优化和运行
        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        # 创建一个长度为1024的空 numpy 数组 aa，数据类型为 np.int32，并填充为5
        aa = np.empty([1024], dtype=np.int32)
        aa.fill(5)
        # 将 numpy 数组 aa 转换为 torch 张量 a
        a = torch.from_numpy(aa)
        # 创建一个全为0的张量 b，数据类型为 torch.int32
        b = torch.zeros(1024, dtype=torch.int32)
        # 调用 warmup_and_run_forward 函数执行跟踪后的 traced 模型，并传入参数 a 和 b
        x = warmup_and_run_forward(traced, a, b)
        # 断言最后一个计算图中所有操作都已融合
        self.assertLastGraphAllFused()
        # 使用 numpy.testing.assert_allclose 断言 x 的值与一个全为0的 numpy 数组非常接近
        np.testing.assert_allclose(np.zeros(1024), x.numpy())

    # 定义测试函数，使用 torch.lt 比较两个张量的小于关系，并返回结果张量
    def test_lt(self):
        # 定义一个简单的函数 easy，用于计算两个张量的小于关系
        def easy(x, y):
            c = torch.lt(x, y)
            return c

        # 遍历 self.devices 中的设备，为后续测试做准备
        for dev in self.devices:
            # 使用 torch.jit.trace 对 easy 函数进行跟踪编译，以便后续优化和运行，
            # 并指定设备为当前迭代的 dev
            traced = torch.jit.trace(easy, (torch.zeros(1024, device=dev), torch.zeros(1024, device=dev)))
            # 创建一个全为1的张量 a，数据类型为 torch.int32，设备为当前迭代的 dev
            a = torch.ones(1024, dtype=torch.int32, device=dev)
            # 创建一个全为0的张量 b，数据类型为 torch.int32，设备为当前迭代的 dev
            b = torch.zeros(1024, dtype=torch.int32, device=dev)
            # 调用 warmup_and_run_forward 函数执行跟踪后的 traced 模型，并传入参数 a 和 b
            x = warmup_and_run_forward(traced, a, b)
            # 断言最后一个计算图中所有操作都已融合
            self.assertLastGraphAllFused()
            # 使用 numpy.testing.assert_allclose 断言 x 的值与一个全为0的 numpy 数组非常接近
            np.testing.assert_allclose(np.zeros(1024), x.cpu().numpy())

    # 定义测试函数，测试 torch.min 和 torch.max 的组合使用
    @suppress_warnings
    def test_min_max(self):
        # 定义一个函数 test，使用 torch.min 和 torch.max 计算两个张量的最小值和最大值
        def test(x, y):
            return torch.max(torch.min(x, y), torch.tensor([4.0]))

        # 使用 torch.jit.trace 对 test 函数进行跟踪编译，以便后续优化和运行
        traced = torch.jit.trace(test, (torch.zeros(1024), torch.zeros(1024)))
        # 创建一个在[0, 8)范围内均匀分布的张量 a
        a = 8.0 * torch.rand(1024)
        # 创建一个在[0, 8)范围内均匀分布的张量 b
        b = 8.0 * torch.rand(1024)
        # 使用 numpy.testing.assert_allclose 断言 warmup_and_run_forward 函数执行后的结果，
        # 与 numpy.minimum(a.numpy(), b.numpy()) 和 [4.0] 中取最大值的结果非常接近
        np.testing.assert_allclose(
            warmup_and_run_forward(traced, a, b), np.maximum(np.minimum(a.numpy(), b.numpy()), [4.0])
        )
        # 断言最后一个计算图中所有操作都已融合
        self.assertLastGraphAllFused()

    # 定义测试函数，测试 torch.min 和 torch.max 在张量上的简单应用
    def test_min_max_reduction(self):
        # 定义一个函数 test，使用 torch.min 和 torch.max 分别计算张量 x 的最小值和最大值，并返回它们的和
        def test(x):
            return torch.min(x) + torch.max(x)

        # 使用 torch.jit.trace 对 test 函数进行跟踪编译，以便后续优化和运行
        traced = torch.jit.trace(test, (torch.zeros(1024)))
        # 创建一个在[0, 8)范围内均匀分布的张量 a
        a = 8.0 * torch.rand(1024)
        # 使用 numpy.testing.assert_allclose 断言 warmup_and_run_forward 函数执行后的结果，
        # 与 numpy.amin(a.numpy()) + numpy.amax(a.numpy()) 的结果非常接近
        np.testing.assert_allclose(warmup_and_run_forward(traced, a), np.amin(a.numpy()) + np.amax(a.numpy()))
        # 断言最后一个计算图中所有操作都已融合
        self.assertLastGraphAllFused()

    # 定义测试函数，测试张量的 min 和 max 方法的组合使用
    def test_min_max_reduction2(self):
        # 定义一个函数 test，使用张量 x 的 min 和 max 方法分别计算其最小值和最大值，并返回它们的和
        def test(x):
            return x.min() + x.max()

        # 使用 torch.jit.trace 对 test 函数进行跟踪编译，以便后续优化和运行
        traced = torch.jit.trace(test, (torch.zeros(1024)))
        # 创建一个在[0, 8)范围内均匀分布的张量 a
        a = 8.0 * torch.rand(1024)
        # 使用 numpy.testing.assert_allclose 断言 warmup_and_run_forward 函数执行后的结果，
        # 与 numpy.amin(a.numpy()) + numpy.amax(a.numpy()) 的结果非常接近
        np.testing.assert_allclose(warmup_and_run_forward(traced, a), np.amin(a.numpy()) + np.amax(a.numpy()))
        # 断言最后一个计算图中所有操作都已融合
        self.assertLastGraphAllFused()
    def test_min_max_reduction_dim1(self):
        # 定义一个测试函数，对输入张量沿着第一个维度进行最小值和最大值的求和
        def test(x):
            return torch.min(x, 1)[0] + torch.max(x, 1)[0]

        # 对 test 函数进行 Torch 脚本跟踪
        traced = torch.jit.trace(test, (torch.zeros(16, 16)))
        # 创建一个 16x16 的张量 a，其值为 [0, 8.0) 的随机数
        a = 8.0 * torch.rand(16, 16)
        # 使用 NumPy 的 assert_allclose 函数检查脚本跟踪后的输出与 NumPy 计算结果的一致性
        np.testing.assert_allclose(warmup_and_run_forward(traced, a), np.amin(
            a.numpy(), axis=1) + np.amax(a.numpy(), axis=1))
        # 断言最后一个图形是否已完全融合
        self.assertLastGraphAllFused()

    def test_min_max_reduction_dim1_2(self):
        # 定义一个测试函数，对输入张量的每个元素平方后，沿第一个维度求最小值
        def test(x):
            return torch.min(x * x, 1)

        # 对 test 函数进行 Torch 脚本跟踪
        traced = torch.jit.trace(test, (torch.zeros(16, 16)))
        # 创建一个 16x16 的张量 a，其值为 [0, 8.0) 的随机数
        a = 8.0 * torch.rand(16, 16)
        # 使用 NumPy 的 assert_allclose 函数检查脚本跟踪后的输出与 NumPy 计算结果的一致性
        np.testing.assert_allclose(warmup_and_run_forward(traced, a)[0], np.amin((a * a).numpy(), axis=1))
        # 断言最后一个图形是否已完全融合
        self.assertLastGraphAllFused()

    def test_clamp(self):
        # 定义一个测试函数，对输入张量每个元素加上 3.0 后进行截断
        def test(x):
            return torch.clamp(x + 3.0, 0.0, 6.0)

        # 遍历测试设备列表
        for dev in self.devices:
            # 对 test 函数进行 Torch 脚本跟踪，指定设备为 dev
            traced = torch.jit.trace(test, (torch.zeros(1024, device=dev)))
            # 创建一个 1024 元素的张量 a，其值为 [-10.0, 10.0) 的随机数，设备为 dev
            a = 20.0 * torch.rand(1024, device=dev) - 10.0
            # 转换张量 a 到 CPU 并转换为 NumPy 数组 an
            an = a.cpu().numpy()
            # 使用 NumPy 的 assert_allclose 函数检查脚本跟踪后的输出与 NumPy 计算结果的一致性
            np.testing.assert_allclose(warmup_and_run_forward(traced, a).cpu(), np.clip(an + 3.0, 0.0, 6.0))
            # 断言最后一个图形是否已完全融合
            self.assertLastGraphAllFused()

    def test_relu(self):
        # 定义一个测试函数，对输入张量先应用 ReLU 函数再进行截断
        def test(x):
            return torch.clamp(F.relu(x), 0, 0.5)

        # 遍历测试设备列表
        for dev in self.devices:
            # 对 test 函数进行 Torch 脚本跟踪，指定设备为 dev
            traced = torch.jit.trace(test, (torch.zeros(1024, device=dev)))
            # 创建一个 1024 元素的张量 a，其值为 [-10.0, 10.0) 的随机数，设备为 dev
            a = 20.0 * torch.rand(1024, device=dev) - 10.0
            # 转换张量 a 到 CPU 并转换为 NumPy 数组 an
            an = a.cpu().numpy()
            # 使用 NumPy 的 assert_allclose 函数检查脚本跟踪后的输出与 NumPy 计算结果的一致性
            np.testing.assert_allclose(warmup_and_run_forward(traced, a).cpu(), np.clip((np.maximum(0, an)), 0, 0.5))
            # 断言最后一个图形是否已完全融合
            self.assertLastGraphAllFused()

    def test_reps(self):
        # 定义一个简单的函数，对输入的两个张量进行按元素加法
        def easy(x, y):
            c = torch.add(x, y)
            return c

        # 对 easy 函数进行 Torch 脚本跟踪
        traced = torch.jit.trace(easy, (torch.rand(1024), torch.rand(1024)))

        # 循环执行 32 次
        for _ in range(32):
            # 创建一个 1024 元素的张量 a，所有元素为 1.0
            a = torch.ones(1024)
            # 创建一个 1024 元素的张量 b，所有元素为 0.0
            b = torch.zeros(1024)
            # 运行脚本跟踪后的函数，并使用 NumPy 的 assert_allclose 函数检查结果与预期是否一致
            x = warmup_and_run_forward(traced, a, b)
            np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_add_const_rhs(self):
        # 定义一个测试函数，对输入张量的每个元素加上常数 3.0
        def test(x):
            return x + 3.0

        # 对 test 函数进行 Torch 脚本跟踪
        traced = torch.jit.trace(test, torch.rand(4))
        # 创建一个 4 元素的张量 x，其值为 [0, 1) 的随机数
        x = torch.rand(4)
        # 执行脚本跟踪后的函数，并使用 NumPy 的 assert_allclose 函数检查结果与预期是否一致
        y = warmup_and_run_forward(traced, x)
        # 断言最后一个图形是否已完全融合
        self.assertLastGraphAllFused()
        # 使用 NumPy 的 assert_allclose 函数检查脚本跟踪后的输出与 NumPy 计算结果的一致性
        np.testing.assert_allclose(x.numpy() + 3.0, y.numpy())

    def test_int_output(self):
        # 定义一个测试函数，计算三个输入张量对应位置元素的乘积
        def test(x, y, z):
            return x * y * z

        # 创建三个张量列表，每个张量包含 4 个元素，值为 [1, 4) 的随机整数
        xs = [(torch.rand(4) * 3 + 1).to(torch.int32) for i in range(3)]
        x, y, z = xs
        # 将每个张量转换为 NumPy 数组
        xn, yn, zn = (t.numpy() for t in xs)
        # 对 test 函数进行 Torch 脚本跟踪
        traced = torch.jit.trace(test, (x, y, z))
        # 运行脚本跟踪后的函数，并使用 NumPy 的 assert_allclose 函数检查结果与预期是否一致
        res = warmup_and_run_forward(traced, x, y, z)
        # 断言最后一个图形是否已完全融合
        self.assertLastGraphAllFused()
        # 使用 NumPy 的 assert_allclose 函数检查脚本跟踪后的输出与 NumPy 计算结果的一致性
        np.testing.assert_allclose(xn * yn * zn, res.numpy())
    # 定义一个测试方法，用于测试 torch.round 函数的功能
    def test_round_2(self):
        # 定义一个内部函数 round，用于对输入的张量进行四舍五入操作
        def round(x):
            return torch.round(x)

        # 针对不同的数据类型进行测试，包括 torch.float32 和 torch.double
        for data_type in [torch.float32, torch.double]:
            # 创建一个张量 a，包含一些浮点数，并转换为指定的数据类型
            a = torch.tensor([0.2, 1.6, 2.5, 3.5]).to(data_type)
            # 对 round 函数进行追踪，生成一个追踪对象 traced
            traced = torch.jit.trace(round, (a))
            # 执行预热和前向运行，返回运行结果 x
            x = warmup_and_run_forward(traced, a)
            # 断言最后生成的计算图中所有的操作都被融合优化了
            self.assertLastGraphAllFused()
            # 对运行结果 x 再次使用 round 函数进行四舍五入
            y = round(x)
            # 断言 x 和 y 的值相等
            self.assertEqual(x, y)

    # 定义一个测试方法，测试 torch.rand_like 函数的功能
    def test_rand_like(self):
        # 定义一个函数 run_rand_like，接受两个张量 x 和 y，返回 torch.rand_like(torch.add(x, y)) 的结果
        N = 1 << 16
        def run_rand_like(x, y):
            return torch.rand_like(torch.add(x, y))

        # 遍历设备列表 self.devices
        for device in self.devices:
            # 创建一个指定设备上形状为 (N,) 的随机张量 x
            x = torch.rand(N, device=device)
            # 对 run_rand_like 函数进行追踪，追踪输入为 (x, x)
            traced = torch.jit.trace(run_rand_like, (x, x), check_trace=False)

            # 遍历数据类型列表 self.dtypes
            for data_type in self.dtypes:
                # 将张量 x 转换为当前遍历的数据类型
                _x = x.to(dtype=data_type)
                # 执行预热和前向运行，返回运行结果 x_v
                x_v = warmup_and_run_forward(traced, _x, _x)
                # 断言最后生成的计算图中所有的操作都被融合优化了
                self.assertLastGraphAllFused()

            # 将张量 x 转换为 numpy 数组
            x_np = x.cpu().numpy()
            # 计算 x_np 的平均值、平方的平均值和立方的平均值
            x1_mean = np.mean(x_np)
            x2_mean = np.mean(x_np ** 2)
            x3_mean = np.mean(x_np ** 3)
            # 使用 np.testing.assert_allclose 断言 x1_mean 等于 1./2，容差为 2e-2
            np.testing.assert_allclose(x1_mean, 1. / 2, rtol=2e-2)
            # 使用 np.testing.assert_allclose 断言 x2_mean 等于 1./3，容差为 2e-2
            np.testing.assert_allclose(x2_mean, 1. / 3, rtol=2e-2)
            # 使用 np.testing.assert_allclose 断言 x3_mean 等于 1./4，容差为 2e-2
            np.testing.assert_allclose(x3_mean, 1. / 4, rtol=2e-2)

    # 定义一个测试方法，测试包含 NaN 值的情况
    def test_nans(self):
        # 定义一个函数 test_max，接受两个张量 x 和 y，返回 torch.max(2 * x, 2 * y) 的结果
        def test_max(x, y):
            return torch.max(2 * x, 2 * y)

        # 定义一个函数 test_min，接受两个张量 x 和 y，返回 torch.min(2 * x, 2 * y) 的结果
        def test_min(x, y):
            return torch.min(2 * x, 2 * y)

        # 对 test_max 和 test_min 函数进行追踪，分别传入随机生成的张量
        tmax = torch.jit.trace(test_max, (torch.rand(1), torch.rand(1)))
        tmin = torch.jit.trace(test_min, (torch.rand(1), torch.rand(1)))

        # 遍历数据类型列表 self.dtypes
        for data_type in self.dtypes:
            # 创建一个包含 NaN 值的张量 x，数据类型为当前遍历的数据类型
            x = torch.tensor([np.nan]).to(dtype=data_type)
            y = torch.tensor([1.0]).to(dtype=data_type)

        # 使用 warmup_and_run_forward 运行 tmin 函数，传入 x 和 y，然后将结果转换为 float 类型，断言其为 NaN
        assert np.isnan(warmup_and_run_forward(tmin, x, y).float().item())
        # 使用 warmup_and_run_forward 运行 tmin 函数，传入 y 和 x，然后将结果转换为 float 类型，断言其为 NaN
        assert np.isnan(warmup_and_run_forward(tmin, y, x).float().item())
        # 断言最后生成的计算图中所有的操作都被融合优化了
        self.assertLastGraphAllFused()
        # 使用 warmup_and_run_forward 运行 tmax 函数，传入 x 和 y，然后将结果转换为 float 类型，断言其为 NaN
        assert np.isnan(warmup_and_run_forward(tmax, x, y).float().item())
        # 使用 warmup_and_run_forward 运行 tmax 函数，传入 y 和 x，然后将结果转换为 float 类型，断言其为 NaN
        assert np.isnan(warmup_and_run_forward(tmax, y, x).float().item())
        # 断言最后生成的计算图中所有的操作都被融合优化了
        self.assertLastGraphAllFused()

    # 定义一个测试方法，测试 torch.pow 函数在 torch.double 数据类型上的功能
    def test_double_intrinsics(self):
        # 定义一个函数 do_pow，接受一个张量 x，返回 torch.pow(x, 7) 的结果
        def do_pow(x):
            return torch.pow(x, 7)

        # 遍历设备列表 self.devices
        for device in self.devices:
            # 创建一个指定设备上数据类型为 torch.double，形状为 (10,) 的随机张量 x
            x = torch.rand(10, dtype=torch.double, device=device)
            # 对 do_pow 函数进行追踪，传入 x，生成一个追踪对象 traced
            traced = torch.jit.trace(do_pow, (x))
            # 执行预热和前向运行，返回运行结果 x
            x = warmup_and_run_forward(traced, x)
            # 断言最后生成的计算图中所有的操作都被融合优化了
            self.assertLastGraphAllFused()
    def test_remainder(self):
        # 定义内部函数 run_remainder，计算 torch.add(x, y) 对 x 取余数后的结果
        def run_remainder(x, y):
            c = torch.remainder(torch.add(x, y), x)
            return c

        # 遍历 self.dtypes 中的数据类型
        for data_type in self.dtypes:
            # 创建随机数张量 a 和 b，数据类型为 data_type
            a = torch.rand(1024, dtype=data_type)
            b = torch.rand(1024, dtype=data_type)
            # 创建全零张量 zeros，数据类型为 data_type
            zeros = torch.zeros(1024, dtype=data_type)
            # 创建全为 NaN 的张量 nans，通过 numpy 数组 cc 转换而来，数据类型为 data_type
            cc = np.array(1024, dtype=float)
            cc.fill(np.nan)
            nans = torch.from_numpy(cc).to(dtype=data_type)

            # 对全零张量 zeros1 和 zeros2 进行初始化，数据类型为 data_type
            zeros1 = torch.zeros(1024, dtype=data_type)
            zeros2 = torch.zeros(1024, dtype=data_type)

            # 使用 torch.jit.trace 对 run_remainder 进行跟踪
            traced = torch.jit.trace(run_remainder, (zeros1, zeros2))
            # 运行 traced 函数并预热，传入 a 和 b 作为参数
            x = warmup_and_run_forward(traced, a, b)
            # 断言最后一个图形是否全部融合
            self.assertLastGraphAllFused()
            # 直接运行 run_remainder 函数，传入 a 和 b 作为参数
            y = run_remainder(a, b)
            # 如果数据类型为 torch.bfloat16，则使用指定的绝对误差和相对误差进行断言
            if data_type is torch.bfloat16:
                self.assertEqual(x, y, atol=4e-3, rtol=2e-3)
            else:
                self.assertEqual(x, y)

            # 对全零张量 zeros1 和 zeros2 进行初始化，数据类型为 data_type
            traced = torch.jit.trace(run_remainder, (zeros1, zeros2))
            # 运行 traced 函数并预热，传入 zeros 和 a 作为参数
            x = warmup_and_run_forward(traced, zeros, a)
            # 断言最后一个图形是否全部融合
            self.assertLastGraphAllFused()
            # 直接运行 run_remainder 函数，传入 zeros 和 a 作为参数
            y = run_remainder(zeros, a)
            # 断言 x 和 y 的值相等
            self.assertEqual(x, y)

            # 对全零张量 zeros1 和 zeros2 进行初始化，数据类型为 data_type
            traced = torch.jit.trace(run_remainder, (zeros1, zeros2))
            # 运行 traced 函数并预热，传入 nans 和 a 作为参数
            x = warmup_and_run_forward(traced, nans, a)
            # 断言最后一个图形是否全部融合
            self.assertLastGraphAllFused()
            # 直接运行 run_remainder 函数，传入 nans 和 a 作为参数
            y = run_remainder(nans, a)
            # 断言 x 和 y 的值相等
            self.assertEqual(x, y)

    def test_multioutput(self):
        # 定义内部函数 easy，计算输入张量 x 加一，然后将结果加倍返回
        def easy(x):
            b = x + 1
            c = b + b
            return (b, c)

        # 使用 torch.jit.trace 对 easy 函数进行跟踪
        traced = torch.jit.trace(easy, (torch.zeros(1024)))

        # 创建全零张量 a，数据类型为默认类型
        a = torch.zeros(1024)
        # 运行 traced 函数并预热，传入 a 作为参数
        b, c = warmup_and_run_forward(traced, a)
        # 断言最后一个图形是否全部融合
        self.assertLastGraphAllFused()
        # 计算预期的 numpy 数组 bp 和 cp
        bp = a.numpy() + 1
        cp = bp + bp
        # 使用 numpy.testing.assert_allclose 断言 b 和 bp 在数值上的接近程度
        np.testing.assert_allclose(b.numpy(), bp)
        # 使用 numpy.testing.assert_allclose 断言 c 和 cp 在数值上的接近程度
        np.testing.assert_allclose(c.numpy(), cp)

    def test_chunk(self):
        # 定义内部函数 easy，计算输入张量 x 加一后的一半部分的和
        def easy(x):
            y = x + 1
            aaa, bbb = torch.chunk(y, 2)
            return aaa + bbb

        # 遍历 self.dtypes 中的数据类型
        for data_type in self.dtypes:
            # 创建全零张量 trace_input，数据类型为 data_type
            trace_input = torch.zeros(1024, 1024, dtype=data_type)
            # 使用 torch.jit.trace 对 easy 函数进行跟踪
            traced = torch.jit.trace(easy, (trace_input))

            # 创建指定数据类型和形状的全零张量 a
            a = torch.zeros(32, 32, dtype=data_type)
            # 运行 traced 函数并预热，传入 a 作为参数
            x = warmup_and_run_forward(traced, a)
            # 断言最后一个图形是否全部融合
            self.assertLastGraphAllFused()
            # 计算预期的 numpy 数组 npr 和 npr2
            npr = a.float().numpy()
            npr2 = npr + 1
            # 使用 numpy.array_split 将 npr2 分成两部分
            npr_a, npr_b = np.array_split(npr2, 2)
            # 使用 numpy.testing.assert_allclose 断言 x 和 npr_a + npr_b 在数值上的接近程度
            np.testing.assert_allclose(npr_a + npr_b, x.float().numpy())
    # 测试函数，用于测试 torch.cat 操作
    def test_cat(self):
        # 遍历设备列表
        for device in self.devices:
            # 设定维度参数
            _dim = 1

            # 定义内部函数 foo，将输入参数加上其索引后进行 torch.cat 操作，并返回结果的平方
            def foo(*args):
                args_2 = [v + i for i, v in enumerate(args)]
                v = torch.cat(args_2, dim=_dim)
                return v * v

            # 遍历数据类型列表
            for data_type in self.dtypes:
                M = 16
                Ns = [128, 16, 1]
                # 创建张量列表，每个张量为零张量，不同维度 N，指定数据类型和设备
                values = [torch.zeros(M, N, dtype=data_type, device=device) for N in Ns]
                # 对 foo 函数进行 Torch JIT 跟踪
                traced = torch.jit.trace(foo, values)

                # 运行前热身和正式运行前向传播
                x = warmup_and_run_forward(traced, *values)
                # 断言最后一个图形被完全融合
                self.assertLastGraphAllFused()
                # 调用 foo 函数获得参考结果
                ref = foo(*values)
                # 使用 numpy 测试所有元素是否接近
                np.testing.assert_allclose(ref.cpu().float().numpy(), x.cpu().float().numpy())

            # 测试 channels-last 的情况
            for _cur_dim in range(4):
                _dim = _cur_dim
                # 创建张量列表，每个张量为随机张量，形状为 (2, 3, 4, 5)，指定设备和 channels-last 内存格式
                values = [torch.randn((2, 3, 4, 5), device=device).to(memory_format=torch.channels_last) for _ in range(10)]
                # 对 foo 函数进行 Torch JIT 跟踪
                traced = torch.jit.trace(foo, values)

                # 运行前热身和正式运行前向传播
                x = warmup_and_run_forward(traced, *values)
                # 断言最后一个图形被完全融合
                self.assertLastGraphAllFused()
                # 调用 foo 函数获得参考结果
                ref = foo(*values)
                # 断言参考结果与前向传播结果相等
                self.assertEqual(ref, x)

    # 这个测试检查我们是否正确处理只有 aten::cat 的融合组
    # 注意，该测试只有在 min_fusion_group=1 时才有意义，否则根本不会形成融合组
    # TODO: 修复并重新启用该测试
    @unittest.skip("cat is broken with fusion group inlining disabled")
    def test_cat_only(self):
        # 遍历设备列表
        for device in self.devices:
            # 定义内部函数 foo，将输入参数加上其索引后进行 torch.cat 操作，并返回结果
            def foo(*args):
                args_2 = [v + i for i, v in enumerate(args)]
                v = torch.cat(args_2, dim=1)
                return v

            M = 16
            Ns = [128, 16, 1]
            # 创建张量列表，每个张量为零张量，不同维度 N，指定设备
            values = [torch.zeros(M, N, device=device) for N in Ns]
            # 对 foo 函数进行 Torch JIT 跟踪
            traced = torch.jit.trace(foo, values)

            # 运行前热身和正式运行前向传播
            x = warmup_and_run_forward(traced, *values)
            # 断言最后一个图形被完全融合
            self.assertLastGraphAllFused()
            # 调用 foo 函数获得参考结果
            ref = foo(*values)
            # 使用 numpy 测试所有元素是否接近
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    # 测试 torch.cat 操作中负数维度参数的情况
    def test_cat_negative_dim(self):
        # 遍历设备列表
        for device in self.devices:
            # 定义内部函数 foo，对输入参数进行 torch.cat 操作，指定负数维度，并返回结果的平方
            def foo(*args):
                v = torch.cat(args, dim=-1)
                return v * v

            M = 16
            Ns = [128, 16, 1]
            # 创建张量列表，每个张量为随机张量，形状为 (M, N)，指定设备
            values = [torch.randn(M, N, device=device) for N in Ns]
            # 对 foo 函数进行 Torch JIT 跟踪
            traced = torch.jit.trace(foo, values)

            # 运行前热身和正式运行前向传播
            x = warmup_and_run_forward(traced, *values)
            # 断言最后一个图形被完全融合
            self.assertLastGraphAllFused()
            # 调用 foo 函数获得参考结果
            ref = foo(*values)
            # 使用 numpy 测试所有元素是否接近
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())
    # 定义测试函数 test_cat_promote_inputs
    def test_cat_promote_inputs(self):
        # 遍历 self.devices 列表中的设备
        for device in self.devices:
            # 定义内部函数 foo，接收可变数量的参数 args
            def foo(*args):
                # 在 dim=1 的维度上对 torch tensors 进行拼接
                v = torch.cat(args, dim=1)
                # 返回拼接后的张量的平方
                return v * v

            # 设置 M 的值为 16
            M = 16
            # 设置 Ns 列表，包含三个整数 128, 16, 1
            Ns = [128, 16, 1]
            # 设置 dtypes 列表，包含 torch.half, torch.float32, torch.double 三种数据类型
            dtypes = [torch.half, torch.float32, torch.double]
            # 根据不同的 N 值和数据类型创建随机张量，并组成列表 values
            values = [torch.randn(M, N, device=device, dtype=dt) for N, dt in zip(Ns, dtypes)]
            # 使用 torch.jit.trace 对 foo 函数进行跟踪
            traced = torch.jit.trace(foo, values)

            # 运行 traced 模型的前向传播，并执行预热
            x = warmup_and_run_forward(traced, *values)
            # 断言最后一个图形是否全部融合
            self.assertLastGraphAllFused()
            # 调用 foo 函数计算参考结果 ref
            ref = foo(*values)
            # 使用 numpy.testing.assert_allclose 检查 ref 和 x 的值是否接近
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    # 定义测试函数 test_cat_empty_tensors
    def test_cat_empty_tensors(self):
        # 遍历 self.devices 列表中的设备
        for device in self.devices:
            # 定义内部函数 foo，接收可变数量的参数 args
            def foo(*args):
                # 在 dim=1 的维度上对 torch tensors 进行拼接
                v = torch.cat(args, dim=1)
                # 返回拼接后的张量的平方
                return v * v

            # 设置 M 的值为 16
            M = 16
            # 设置 Ns 列表，包含三个整数 128, 16, 1
            Ns = [128, 16, 1]
            # 创建空的 torch tensor empty，设备为当前循环中的 device，数据类型为 torch.double
            empty = torch.tensor([], device=device, dtype=torch.double)
            # 根据不同的 N 值创建随机张量，并组成列表 values，其中包括空的 empty 张量
            values = [empty] + [torch.randn(M, N, device=device) for N in Ns]
            # 使用 torch.jit.trace 对 foo 函数进行跟踪
            traced = torch.jit.trace(foo, values)

            # 运行 traced 模型的前向传播，并执行预热
            x = warmup_and_run_forward(traced, *values)
            # 断言最后一个图形是否全部融合
            self.assertLastGraphAllFused()
            # 调用 foo 函数计算参考结果 ref
            ref = foo(*values)
            # 使用 numpy.testing.assert_allclose 检查 ref 和 x 的值是否接近
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

            # 现在测试仅包含空张量的情况
            # values 列表全部设置为 empty 张量
            values = [empty for i in range(3)]
            # 使用 torch.jit.trace 对 foo 函数进行跟踪
            traced = torch.jit.trace(foo, values)
            # 运行 traced 模型的前向传播，并执行预热
            x = warmup_and_run_forward(traced, *values)
            # 断言最后一个图形是否全部融合
            self.assertLastGraphAllFused()
            # 调用 foo 函数计算参考结果 ref
            ref = foo(*values)
            # 使用 numpy.testing.assert_allclose 检查 ref 和 x 的值是否接近
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    # 定义测试函数 test_cat_with_constant_dim
    def test_cat_with_constant_dim(self):
        # 遍历 self.devices 列表中的设备
        for device in self.devices:
            # 定义内部函数 foo，接收可变数量的参数 args
            def foo(*args):
                # 在 dim=1 的维度上对 torch tensors 进行拼接
                v1 = torch.cat(args, dim=1)
                # 对拼接后的张量再次在 dim=1 的维度上进行拼接
                v2 = torch.cat([v1], dim=1)
                # 返回拼接后的张量的平方
                return v2 * v2

            # 创建空的 torch tensor empty，设备为当前循环中的 device，数据类型为 torch.float32
            empty = torch.tensor([], device=device, dtype=torch.float32)
            # 创建包含空的 empty 张量和两个形状为 (1, 64) 的随机张量的 inputs 列表
            inputs = [empty] + [torch.randn(1, 64, device=device), torch.randn(1, 64, device=device)]
            # 使用 torch.jit.trace 对 foo 函数进行跟踪
            traced = torch.jit.trace(foo, inputs)

            # 运行 traced 模型的前向传播，并执行预热
            x = warmup_and_run_forward(traced, *inputs)
            # 断言最后一个图形是否全部融合
            self.assertLastGraphAllFused()
            # 调用 foo 函数计算参考结果 ref
            ref = foo(*inputs)
            # 使用 numpy.testing.assert_allclose 检查 ref 和 x 的值是否接近
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())
    def test_scalar(self):
        # 定义一个 Torch Script 函数 test_float，对两个 Tensor 进行加法操作，支持浮点数作为系数
        @torch.jit.script
        def test_float(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, a: float, b: float) -> torch.Tensor:
            return torch.add(torch.add(x, y, alpha=a), z, alpha=b)

        # 定义一个 Torch Script 函数 test_int，对两个 Tensor 进行加法操作，支持整数作为系数
        @torch.jit.script
        def test_int(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, a: int, b: int) -> torch.Tensor:
            return torch.add(torch.add(x, y, alpha=a), z, alpha=b)

        # 对每个测试函数进行遍历，使用不同的数据类型进行测试
        for test in (test_float, test_int):
            for data_type in self.dtypes:
                # 生成随机数据 Tensor
                x, y, z = (torch.rand(4, dtype=data_type) for i in range(3))
                a, b = 1, 2
                # 执行 Torch Script 函数 test，并获取结果
                test(x, y, z, a, b)
                r = test(x, y, z, a, b)
                # 断言测试结果是否符合预期：x + y * a + z * b
                self.assertEqual(r, x + y * a + z * b)

    def test_loop(self):
        # 定义一个 Torch Script 函数 test，包含一个循环，对 Tensor 进行加法操作
        @torch.jit.script
        def test(x: torch.Tensor, y: torch.Tensor, z: int) -> torch.Tensor:
            b = y
            # 循环 z 次
            for i in range(0, z):
                a = x + y
                b = b + y
            return b

        # 准备输入数据
        x, y, z = (torch.zeros(32, 32), torch.ones(32, 32), 4)
        # 执行 Torch Script 函数 test
        test(x, y, z)
        r = test(x, y, z)

    def test_slice(self):
        # 定义一个 Python 函数 easy，对两个 Tensor 进行切片并相加
        def easy(x, y):
            a = x[0:512:2]  # 对 x 进行切片操作
            b = y[0:512:2]  # 对 y 进行切片操作
            return a + b

        # 对 easy 函数进行 Torch JIT 追踪
        traced = torch.jit.trace(easy, (torch.ones(1024, 1024), torch.zeros(1024, 1024)))

        # 准备输入数据
        a = torch.ones(1024, 1024)
        # 使用追踪后的模型进行推理
        x = traced(a, a)
        npr = a[0:512:2]  # 对 a 进行切片操作
        npr = npr + npr  # 对切片后的结果进行相加操作
        np.testing.assert_allclose(npr.numpy(), x.numpy())

    def test_unsqueeze(self, N=256):
        # 定义一个 Python 函数 easy，对两个 Tensor 进行 unsqueeze 操作并相加
        def easy(x, y):
            a = torch.unsqueeze(x, 0)  # 对 x 进行 unsqueeze 操作
            b = torch.unsqueeze(y, 0)  # 对 y 进行 unsqueeze 操作
            return a + b

        # 对 easy 函数进行 Torch JIT 追踪
        traced = torch.jit.trace(easy, (torch.ones(N, N), torch.zeros(N, N)))

        # 准备输入数据
        a = torch.rand(N, N)
        # 使用追踪后的模型进行推理
        x = traced(a, a)
        npr = np.expand_dims(a, 0)  # 对 a 进行 expand_dims 操作
        npr = npr + npr  # 对 expand_dims 后的结果进行相加操作
        np.testing.assert_allclose(npr, x.numpy())
    # 定义测试函数 _test_softmax，接受一个设备参数
    def _test_softmax(self, device):
        
        # 定义测试 softmax 函数，接受两个输入张量 x 和 y
        def test_softmax(x, y):
            # 对 x 在 dim=0 上进行 softmax，返回值为 a
            a = F.softmax(x, dim=0, dtype=torch.float32)
            # 对 y 在 dim=0 上进行 softmax，返回值为 b
            b = F.softmax(y, dim=0, dtype=torch.float32)
            # 对 x 在 dim=1 上进行 softmax，返回值为 c
            c = F.softmax(x, dim=1, dtype=torch.float32)
            # 对 y 在 dim=1 上进行 softmax，返回值为 d
            d = F.softmax(y, dim=1, dtype=torch.float32)
            # 返回四个 softmax 结果的和
            return a + b + c + d

        # 定义测试带负索引的 softmax 函数，接受两个输入张量 x 和 y
        def test_softmax_neg_index(x, y):
            # 对 x 在 dim=-2 上进行 softmax，返回值为 a
            a = F.softmax(x, dim=-2, dtype=torch.float32)
            # 对 y 在 dim=-2 上进行 softmax，返回值为 b
            b = F.softmax(y, dim=-2, dtype=torch.float32)
            # 对 x 在 dim=-1 上进行 softmax，返回值为 c
            c = F.softmax(x, dim=-1, dtype=torch.float32)
            # 对 y 在 dim=-1 上进行 softmax，返回值为 d
            d = F.softmax(y, dim=-1, dtype=torch.float32)
            # 返回四个 softmax 结果的和
            return a + b + c + d

        # 定义测试 log_softmax 函数，接受两个输入张量 x 和 y
        def test_log_softmax(x, y):
            # 对 x 在 dim=0 上进行 log_softmax，返回值为 a
            a = F.log_softmax(x, dim=0, dtype=torch.float32)
            # 对 y 在 dim=0 上进行 log_softmax，返回值为 b
            b = F.log_softmax(y, dim=0, dtype=torch.float32)
            # 对 x 在 dim=1 上进行 log_softmax，返回值为 c
            c = F.log_softmax(x, dim=1, dtype=torch.float32)
            # 对 y 在 dim=1 上进行 log_softmax，返回值为 d
            d = F.log_softmax(y, dim=1, dtype=torch.float32)
            # 返回四个 log_softmax 结果的和
            return a + b + c + d

        # 对三种测试函数使用迭代，遍历数据类型列表 self.dtypes
        for test in (test_softmax, test_log_softmax, test_softmax_neg_index):
            for data_type in self.dtypes:
                # 保存当前是否允许张量表达式缩减的状态
                old = torch._C._jit_set_texpr_reductions_enabled(True)
                # 创建一个随机张量作为输入，并在给定设备上跟踪测试函数
                traced_input = torch.randn(2, 3, dtype=data_type, device=device)
                traced = torch.jit.trace(test, (traced_input, traced_input))
                # 创建另一个随机张量作为输入
                inp = torch.randn(2, 3, dtype=data_type, device=device)
                # 在跟踪后的函数上运行输入数据，并保存结果
                res = traced(inp, inp)
                # 使用 eager 模式运行测试函数作为参考结果
                ref = test(inp, inp)
                # 使用 numpy 测试库断言跟踪结果与参考结果的近似性
                np.testing.assert_allclose(ref, res.cpu().numpy(), rtol=1e-06, atol=1e-06)
                # 恢复先前的张量表达式缩减状态
                torch._C._jit_set_texpr_reductions_enabled(old)

    # 定义在 CPU 上运行 softmax 测试的函数
    def test_softmax_cpu(self):
        self._test_softmax('cpu')

    # 根据 CUDA 是否可用，定义在 CUDA 上运行 softmax 测试的函数
    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @unittest.skip("global allocs are not supported yet.")
    def test_softmax_cuda(self):
        self._test_softmax('cuda')

    # 定义测试半精度 GELU 函数
    def test_half_gelu(self):
        # 如果 CUDA 可用，则设备列表包含 'cuda'
        devices = ["cuda"] if torch.cuda.is_available() else []

        # 定义一个使用 torch.jit.script 装饰器的函数 bias_gelu
        @torch.jit.script
        def bias_gelu(bias, y):
            # 计算 bias 与 y 的和，保存为 x
            x = bias + y
            # 计算 GELU 函数并返回结果
            return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

        # 对设备列表进行迭代，执行以下操作
        for device in devices:
            # 创建两个随机半精度张量 a 和 b，并在指定设备上跟踪 bias_gelu 函数
            a = torch.rand(1024, dtype=torch.half, device=device)
            b = torch.rand(1024, dtype=torch.half, device=device)
            traced = torch.jit.trace(bias_gelu, (a, b))
            # 预热并运行跟踪函数的前向传播，并断言所有图都已融合
            x = warmup_and_run_forward(traced, a, b)
            self.assertLastGraphAllFused()
    def test_half_bn_relu(self):
        devices = ["cuda"] if torch.cuda.is_available() else []

        def foo(a, b, c):
            # 执行批标准化操作
            y = torch.nn.functional.batch_norm(a, b, c)
            # 对批标准化的结果执行 ReLU 激活函数
            z = y.relu()
            return z

        for device in devices:
            # 创建随机张量 a，b，c，并使用半精度数据类型和指定设备
            a = torch.rand(16, 16, dtype=torch.half, device=device)
            b = torch.rand(16, dtype=torch.half, device=device)
            c = torch.rand(16, dtype=torch.half, device=device)
            # 对函数 foo 进行追踪
            traced = torch.jit.trace(foo, (a, b, c))
            # 打印追踪后的计算图
            print(traced.graph)
            # 运行追踪后的模型，并执行前向传播的预热和运行
            x = warmup_and_run_forward(traced, a, b, c)
            # 断言最后一个计算图中所有操作已融合
            self.assertLastGraphAllFused()

    def test_exp_pow(self):
        @torch.jit.script
        def do_exp(x, y, z):
            # 执行指数运算和幂次方操作
            return ((x * y) * 2) * torch.pow(z, 2)

        for device in self.devices:
            # 创建随机张量 x，y，z，并使用双精度数据类型和指定设备
            x = torch.rand(10, dtype=torch.double, device=device)
            y = torch.rand(10, dtype=torch.double, device=device)
            z = torch.rand(10, dtype=torch.double, device=device)
            # 对函数 do_exp 进行追踪
            traced = torch.jit.trace(do_exp, (x, y, z))
            # 运行追踪后的模型，并执行前向传播的预热和运行
            x = warmup_and_run_forward(traced, x, y, z)
            # 断言最后一个计算图中所有操作已融合
            self.assertLastGraphAllFused()

    def test_sin_pow(self):
        def test(x):
            # 执行正弦函数和幂次方操作
            return torch.sin(torch.pow(x, 0))

        for data_type, shape in itertools.product(self.dtypes, [[3], [5], [10]]):
            # 创建随机张量 x，并使用指定数据类型和形状
            x = torch.rand(shape, dtype=data_type)
            # 对函数 test 进行脚本化
            scripted = torch.jit.script(test)
            # 运行脚本化后的模型，并执行前向传播的预热和运行
            out = warmup_and_run_forward(scripted, x)
            # 断言最后一个计算图中所有操作已融合
            self.assertLastGraphAllFused()
            # 断言输出与预期结果相等
            self.assertEqual(out, test(x))

    def test_transpose(self):
        @torch.jit.script
        def test(x, y, z):
            # 执行张量转置操作，并加上另外两个张量的元素-wise 加法
            return x.transpose(0, 1) + y + z
        # 创建具有指定形状的随机张量 x，y，z
        x = torch.rand(4, 5, 2, 3)
        y = torch.rand(5, 4, 2, 3)
        z = torch.rand(5, 4, 2, 3)
        # 计算参考结果
        ref = test(x, y, z)
        # 计算实际结果
        res = test(x, y, z)
        # 使用 NumPy 断言检查两个结果张量的所有元素是否接近
        np.testing.assert_allclose(ref.numpy(), res.numpy())

    def test_sliced_stride(self):
        @torch.jit.script
        def test(x, y, z):
            # 执行张量切片和元素-wise 加法
            return x + y + z
        # 创建具有指定形状的随机张量 x，y，z，并应用切片
        x = torch.rand(16, 4, 2, 3)[::2]
        y = torch.rand(8, 4, 2, 3)
        z = torch.rand(8, 4, 2, 3)
        # 计算参考结果
        ref = test(x, y, z)
        # 计算实际结果
        res = test(x, y, z)
        # 使用 NumPy 断言检查两个结果张量的所有元素是否接近
        np.testing.assert_allclose(ref.numpy(), res.numpy())
    def test_dynamic_shape(self):
        # 使用 num_profiled_runs 上下文管理器，运行测试两次以进行性能分析
        with num_profiled_runs(2):
            # 使用 torch.jit.script 装饰器将函数 test 编译为 Torch 脚本
            @torch.jit.script
            def test(x, y, z):
                # 计算输入张量 x, y, z 的乘积
                return x * y * z
            # 生成三个随机张量 x, y, z，并移至 CUDA 设备
            x, y, z = (torch.rand(4, 8).cuda() for _ in range(3))
            # 计算参考结果 ref
            ref = test(x, y, z)
            # 使用新的随机张量再次运行测试
            _ = test(*[torch.rand(6, 8).cuda() for _ in range(3)])
            # 计算新结果 res
            res = test(x, y, z)
            # 使用 numpy.testing.assert_allclose 检查 ref 和 res 的近似程度
            np.testing.assert_allclose(ref.cpu().numpy(), res.cpu().numpy())

            # 示例中的广播操作
            x = torch.rand(4, 8).cuda()
            y = torch.rand(1, 8).cuda()
            z = torch.rand(4, 1).cuda()
            # 计算广播操作后的结果 res
            res = test(x, y, z)
            # 将张量 x, y, z 转移到 CPU 并转换为 numpy 数组
            xn, yn, zn = (t.cpu().numpy() for t in (x, y, z))
            # 使用 numpy.testing.assert_allclose 检查 res 与期望结果的近似程度
            np.testing.assert_allclose(res.cpu().numpy(), xn * yn * zn)

            # 测试不匹配的形状，期望抛出 RuntimeError
            x = torch.rand(4, 8).cuda()
            y = torch.rand(4, 8).cuda()
            z = torch.rand(5, 8).cuda()
            try:
                # 尝试运行 test 函数，预期会抛出异常
                res = test(x, y, z)
            except RuntimeError as e:
                # 断言异常消息中包含特定字符串
                assert "The size of tensor a (4) must match" in e.args[0]

            # 改变静态维度的尝试应导致失败的守护
            # x, y, z = [torch.rand(4, 7).cuda() for _ in range(3)]
            # xn, yn, zn = [t.cpu().numpy() for t in (x, y, z)]
            # res = test(x, y, z)
            # print(test.graph_for(x, y, z))
            # np.testing.assert_allclose(res.cpu().numpy(), xn * yn * zn)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_guard_fails(self):
        # 使用 torch.jit.script 装饰器将函数 test 编译为 Torch 脚本
        @torch.jit.script
        def test(x, y, z):
            # 计算输入张量 x, y, z 的乘积
            return x * y * z
        # 分别运行四次 test 函数，传入不同的随机张量
        r1 = test(*[torch.rand(4).cuda() for _ in range(3)])
        r2 = test(*[torch.rand(4).cuda() for _ in range(3)])
        r3 = test(*[torch.rand(4).cuda() for _ in range(3)])
        r4 = test(*[torch.rand(7).cuda() for _ in range(3)])

    def test_bitwise_ops(self):
        # 定义按位操作函数 run_and, run_or, run_xor, run_lshift, run_rshift
        def run_and(x, y):
            return x & (x & y)

        def run_or(x, y):
            return x & (x | y)

        def run_xor(x, y):
            return x ^ (x ^ y)

        def run_lshift(x, y):
            return x & (x << y)

        def run_rshift(x, y):
            return x & (x >> y)

        # 将函数放入字典 fns 中
        fns = {run_and, run_or, run_xor, run_lshift, run_rshift}

        # 遍历测试设备列表 self.devices
        for device in self.devices:
            # 遍历函数字典 fns
            for fn in fns:
                # 创建输入张量 a, b, inp，并移至指定 CUDA 设备
                a = torch.ones(128, dtype=torch.int32, device=device)
                b = torch.zeros(128, dtype=torch.int32, device=device)
                inp = torch.ones(128, dtype=torch.int32, device=device)
                # 使用 torch.jit.trace 对函数 fn 进行跟踪
                traced = torch.jit.trace(fn, (inp, inp))
                # 调用 warmup_and_run_forward 运行并预热跟踪后的函数
                x = warmup_and_run_forward(traced, a, b)
                # 断言最后一个图是否全部融合
                self.assertLastGraphAllFused()
                # 计算使用 fn 函数计算的结果 y
                y = fn(a, b)
                # 使用 numpy.testing.assert_allclose 检查 x 和 y 的近似程度
                np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy())
    def test_where(self):
        # 定义内部函数 run_where，用于返回 x 和 y 中符合条件的元素
        def run_where(x, y):
            return torch.where(torch.gt(x, y), x, y)

        # 遍历数据类型列表 self.dtypes
        for data_type in self.dtypes:
            # 创建随机张量 a 和 b，数据类型为当前循环中的 data_type
            a = torch.rand(1024, dtype=data_type)
            b = torch.rand(1024, dtype=data_type)
            # 创建与 a 和 b 形状相同的零张量 zeros
            zeros = torch.zeros(1024, dtype=data_type)
            # 对 run_where 函数进行追踪
            traced = torch.jit.trace(run_where, (zeros, zeros))
            # 执行预热和前向运行，返回结果给 x
            x = warmup_and_run_forward(traced, a, b)
            # 断言最后一个图形都被融合
            self.assertLastGraphAllFused()
            # 直接调用 run_where 函数，返回结果给 y
            y = run_where(a, b)
            # 使用 numpy 测试断言 x 和 y 的结果在浮点数精度下相等
            np.testing.assert_allclose(x.float().numpy(), y.float().numpy())

    def test_multi_rand(self):
        # 遍历设备列表 self.devices
        for device in self.devices:
            # 定义内部函数 test，用于生成与输入张量 x 形状相同的随机张量 y
            def test(x):
                y = torch.rand_like(x)
                return (x + y) - (y - x)

            # 设置绝对误差和相对误差的初始值
            _atol = 2e-3
            _rtol = 1e-5
            # 再次遍历数据类型列表 self.dtypes
            for data_type in self.dtypes:
                # 如果当前数据类型是 torch.bfloat16，调整绝对误差的值
                if data_type is torch.bfloat16:
                    _atol = 2e-2
                # 创建设备上的随机张量 a，数据类型为当前循环中的 data_type
                a = torch.rand(4, dtype=data_type, device=device)
                # 对 test 函数进行脚本化
                scripted = torch.jit.script(test)
                # 执行预热和前向运行，返回结果给 out
                out = warmup_and_run_forward(scripted, a)
                # 断言最后一个图形都被融合
                self.assertLastGraphAllFused()
                # 使用 torch.allclose 断言 out 与 2 * a 在给定的误差范围内相等
                assert torch.allclose(out, 2 * a, atol=_atol, rtol=_rtol)

    def test_mask(self):
        # 定义内部函数 test，用于在输入张量 x 的基础上增加一个维度并进行比较是否等于 0
        def test(x):
            return x.unsqueeze(1) == 0

        # 外部循环遍历设备列表 self.devices
        for d in self.devices:
            # 再次遍历数据类型列表 self.dtypes
            for data_type in self.dtypes:
                # 创建在设备 d 上的随机张量 x，数据类型为当前循环中的 data_type
                x = torch.rand(4, dtype=data_type, device=d) > 0.5
                # 对 test 函数进行脚本化
                scripted = torch.jit.script(test)
                # 执行预热和前向运行，返回结果给 out
                out = warmup_and_run_forward(scripted, x)
                # 断言最后一个图形都被融合
                self.assertLastGraphAllFused()
                # 使用 torch.equal 断言 out 与 test(x) 的结果是否完全相等
                assert torch.equal(out, test(x))

    def test_simple_add(self):
        # 获取当前 TE 生成块代码的设置值，并设置 TE 生成块代码为 True
        val = torch._C._jit_get_te_generate_block_code()
        torch._C._jit_set_te_generate_block_code(True)
        # 获取当前 TE 表达式回退允许的设置值，并设置 TE 表达式回退允许为 True
        fall_bk = torch._C._jit_texpr_fallback_allowed()
        torch._C._jit_texpr_set_fallback_allowed(True)

        # 定义简单的加法函数 simple
        def simple(a, b):
            return torch.add(a, b)

        # 创建全为 1 的张量 a 和 b，形状为 (256, 256)
        a = torch.ones(256, 256)
        b = torch.ones(256, 256)
        # 对 simple 函数进行追踪
        traced = torch.jit.trace(simple,
                                 (torch.ones(256, 256), torch.ones(256, 256)))
        # 执行 traced 函数，输入 a 和 b，返回结果给 f
        f = traced(a, b)
        # 创建一个期望的结果数组 f_test，全部为 2
        f_test = np.full((256, 256), 2, dtype=float)
        # 使用 numpy 测试断言 f 与 f_test 在浮点数精度下相等
        np.testing.assert_allclose(f.numpy(), f_test)
        # 恢复 TE 生成块代码的设置值
        torch._C._jit_set_te_generate_block_code(val)
        # 恢复 TE 表达式回退允许的设置值
        torch._C._jit_texpr_set_fallback_allowed(fall_bk)
    def test_strided_output_preserved(self):
        def foo(a, b):
            return a + b - a  # 返回 a 和 b 的和减去 a，即 b

        # 创建一个张量 x，包含元素 0 到 5
        x = torch.arange(6)
        # 将 x 视图为一个形状为 (2, 3) 的张量，步长分别为 (1, 2)
        x = torch.as_strided(x, (2, 3), (1, 2))
        total = 0
        # 嵌套循环，向 x 中填充递增的总数值
        for i in range(2):
            for j in range(3):
                x[i, j] = total
                total += 1
        # 对函数 foo 进行脚本化
        foo_script = torch.jit.script(foo)
        # 使用 foo_script 处理 x 和 x，并返回结果
        foo_script(x, x)
        foo_script(x, x)
        out_s = foo_script(x, x)
        # 直接调用 foo 处理 x 和 x，并返回结果
        out_eager = foo(x, x)
        # 断言脚本化和直接调用的结果相等
        self.assertEqual(out_s, out_eager)
        # 断言脚本化和直接调用的结果张量步长相等
        self.assertEqual(out_s.stride(), out_eager.stride())
        # 断言最后的图形都被融合了
        self.assertLastGraphAllFused()

        # 更多维度的例子
        N, C, H, W, = 2, 3, 4, 5
        # 创建一个随机张量 x，形状为 (2, 3, 4, 5)，使用通道为最后内存格式
        x = torch.rand(N, C, H, W).to(memory_format=torch.channels_last)
        # 对函数 foo 进行脚本化
        foo_script = torch.jit.script(foo)
        foo_script(x, x)
        foo_script(x, x)
        out_s = foo_script(x, x)
        # 直接调用 foo 处理 x 和 x，并返回结果
        out_eager = foo(x, x)
        # 断言脚本化和直接调用的结果相等
        self.assertEqual(out_s, out_eager)
        # 断言脚本化和直接调用的结果张量步长相等
        self.assertEqual(out_s.stride(), out_eager.stride())
        # 断言最后的图形都被融合了

    def test_alias_analysis_module(self):
        # 定义一个包含别名的模块
        class AliasModule(nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(1337)
                # 初始化三个大小为 (128, 128) 的随机张量
                self.a = torch.randn(128, 128)
                self.b = torch.randn(128, 128)
                self.c = torch.randn(128, 128)

            def forward(self, x, y, z):
                # z 加上 self.a 的值
                z = z + self.a
                # self.b 就地加上 y 的值
                self.b.add_(y)
                # w 等于 z 加上 self.a 的值
                w = z + self.a
                # z 等于 w 加上 x 的值
                z = w + x
                return z

        # 创建一个大小为 (128, 128) 的随机张量 x
        x = torch.randn(128, 128)

        # 根据脚本标志获取 AliasModule 实例
        def getModule(script):
            am = AliasModule()
            if script:
                return torch.jit.script(am)
            return am

        # 获取非脚本化的 AliasModule 实例
        am = getModule(False)
        # 获取脚本化的 AliasModule 实例
        am_s = getModule(True)
        # 使用非脚本化的 AliasModule 实例处理 x, x, x 并返回结果
        ref = am(x, x, x)
        # 使用脚本化的 AliasModule 实例处理 x, x, x 并返回结果
        test = am_s(x, x, x)
        # 断言非脚本化和脚本化的结果近似相等
        torch.testing.assert_close(ref, test)

        # 现在进行别名设置
        # am.a 引用 am.b 的值
        am.a = am.b
        # 使用非脚本化的 AliasModule 实例处理 x, x, x 并返回结果
        ref = am(x, x, x)

        # am_s.a 引用 am_s.b 的值
        am_s.a = am_s.b
        # 使用脚本化的 AliasModule 实例处理 x, x, x 并返回结果
        test = am_s(x, x, x)

        # 断言非脚本化和脚本化的结果近似相等
        torch.testing.assert_close(ref, test)
    def test_alias_analysis_inputs(self):
        # 定义一个测试用例，用于检查别名分析在输入参数上的表现

        class AliasModule(nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(1337)
                self.a = torch.randn(128, 128)
                self.b = torch.randn(128, 128)
                self.c = torch.randn(128, 128)

            def forward(self, x, y, z):
                # 修改 x，累加 y
                x.add_(y)
                # 计算 w，等于 z 加上模块中的 self.a
                w = z + self.a
                # 计算 z，等于 w 加上 x
                z = w + x
                return z

        def getModule(script):
            # 创建 AliasModule 的实例
            am = AliasModule()
            # 如果需要脚本化，则返回脚本化的模块，否则返回原始模块
            if script:
                return torch.jit.script(am)
            return am
        
        # 获取非脚本化的 AliasModule 实例
        am = getModule(False)
        # 获取脚本化的 AliasModule 实例
        am_s = getModule(True)

        # 设置随机种子
        torch.manual_seed(1337)
        # 生成一个随机的 128x128 的张量 x
        x = torch.randn(128, 128)
        # 使用非脚本化的模块进行前向传播
        ref = am(x, x, x)

        # 再次设置随机种子
        torch.manual_seed(1337)
        # 生成一个随机的 128x128 的张量 x
        x = torch.randn(128, 128)
        # 使用脚本化的模块进行前向传播
        test = am_s(x, x, x)

        # 断言 ref 和 test 的结果是否相近
        torch.testing.assert_close(ref, test)

    def test_alias_analysis_input_and_module(self):
        # 定义一个测试用例，用于检查别名分析在输入参数和模块内部状态上的表现

        class AliasModule(nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(1337)
                self.a = torch.randn(128, 128)
                self.b = torch.randn(128, 128)
                self.c = torch.randn(128, 128)

            def forward(self, x, y, z):
                # 修改 x，累加 y
                x.add_(y)
                # 计算 w，等于 z 加上模块中的 self.b
                w = z + self.b
                # 计算 z，等于 w 加上 x
                z = w + x
                return z

        def getModule(script):
            # 创建 AliasModule 的实例
            am = AliasModule()
            # 如果需要脚本化，则返回脚本化的模块，否则返回原始模块
            if script:
                return torch.jit.script(am)
            return am
        
        # 获取非脚本化的 AliasModule 实例
        am = getModule(False)
        # 获取脚本化的 AliasModule 实例
        am_s = getModule(True)

        # 设置随机种子
        torch.manual_seed(1337)
        # 生成一个随机的 128x128 的张量 x
        x = torch.randn(128, 128)
        # 修改非脚本化模块的 self.b 属性为 x
        am.b = x
        # 使用非脚本化的模块进行前向传播
        ref = am(x, x, x)

        # 再次设置随机种子
        torch.manual_seed(1337)
        # 生成一个随机的 128x128 的张量 x
        x = torch.randn(128, 128)
        # 修改脚本化模块的 self.b 属性为 x
        am_s.b = x
        # 使用脚本化的模块进行前向传播
        test = am_s(x, x, x)

        # 断言 ref 和 test 的结果是否相近
        torch.testing.assert_close(ref, test)

    def test_multiple_outputs(self):
        # 定义一个测试用例，用于测试多个输出的情况
        for device in self.devices:
            # A bug reported internally similar to the one reported in #48533
            # 定义一个函数 foo，接受参数 a, b, c，并返回多个张量
            def foo(a, b, c):
                # 计算 t_next，等于 c 加上 1
                t_next = c + 1
                # 计算 t5，等于 t_next 乘以 b
                t5 = t_next * b
                # 计算 t6，为 t_next 的扩展维度为 1 的结果
                t6 = torch.unsqueeze(t_next, 1)
                # 计算 t7，等于 a 乘以 t6
                t7 = a * t6
                # 返回 t7, t5, t_next 三个张量
                return (t7, t5, t_next)

            for data_type in self.dtypes:
                # 生成一个在指定设备上的随机张量 a
                a = torch.rand(20, 20, dtype=data_type, device=device)
                # 生成一个在指定设备上的随机张量 b，通过 as_strided 方法进行变换
                b = torch.rand(20 * 29, dtype=data_type, device=device).as_strided([20], [29])
                # 生成一个在指定设备上的全 1 张量 c
                c = torch.ones(20, dtype=torch.int64, device=device)
                # 使用 torch.jit.trace 对 foo 进行追踪
                traced = torch.jit.trace(foo, (a, b, c))
                # 计算 foo 的原始输出
                ref = foo(a, b, c)
                # 计算追踪后模型的输出
                exp = traced(a, b, c)
                # 再次计算追踪后模型的输出
                exp = traced(a, b, c)
                # 断言原始输出和追踪后模型的输出是否相等
                self.assertEqual(ref, exp)
    def test_propagated_mem_layout(self):
        # 定义简单的函数 foo，计算 t7 = a * b * (c + 1)
        def foo(a, b, c):
            t_next = c + 1
            t5 = t_next * b
            t7 = a * t5
            return t7

        # 定义带有多个输出的函数 foo_multi_outputs，计算 t7 = a * b * (c + 1)，并返回 (t7, t5, t_next)
        def foo_multi_outputs(a, b, c):
            t_next = c + 1
            t5 = b * t_next
            t7 = a * t5
            return (t7, t5, t_next)

        # 定义带有内存布局转换的函数 foo_multi_outputs_i_nhwc_o_nchw
        # 计算 t7 = a * b * (c + 1)，然后将 t7 转换为 NCHW 内存格式，返回 (t8, t7, t5, t_next)
        def foo_multi_outputs_i_nhwc_o_nchw(a, b, c):
            t_next = c + 1
            t5 = b * t_next
            t7 = a * t5
            t8 = t7.to(memory_format=torch.contiguous_format)
            return (t8, t7, t5, t_next)

        # 定义运行测试用例的函数 run_foo_case
        # 使用 JIT 跟踪给定的函数 foo，并比较脚本执行和参考执行的结果
        def run_foo_case(foo, a, b, c):
            traced_contiguous = torch.jit.trace(foo, (a, b, c))
            ref = foo(a, b, c)
            exp = traced_contiguous(a, b, c)
            self.assertEqual(ref, exp)

        # 创建所有可能的内存布局组合
        mem_layouts = list(itertools.product([torch.contiguous_format, torch.channels_last], repeat=3))
        # 创建测试用的形状
        shapes = [(2, 3, 4, 5), (2, 1, 1, 5), (1, 1, 1, 1)]
        # 创建所有可能的维度排列
        permutes = [(0, 3, 2, 1), (0, 3, 1, 2)]
        # 定义要测试的函数列表
        funcs = [foo, foo_multi_outputs, foo_multi_outputs_i_nhwc_o_nchw]
        # 创建所有可能的配置组合
        configs = itertools.product(funcs, shapes, mem_layouts, permutes)
        
        # 遍历静态和动态两种策略
        for strategy in ["STATIC", "DYNAMIC"]:
            old_strategy = torch.jit.set_fusion_strategy([(strategy, 10)])
            # 遍历所有配置并运行测试用例
            for _func, _shape, _mem_layouts, _permute in configs:
                # 创建随机数据张量并设置其内存布局
                a = torch.rand(_shape, dtype=torch.float32).to(memory_format=_mem_layouts[0])
                b = torch.rand(_shape, dtype=torch.float32).to(memory_format=_mem_layouts[1])
                c = torch.rand(_shape, dtype=torch.float32).to(memory_format=_mem_layouts[2])
                run_foo_case(_func, a, b, c)

                # 对输入张量进行维度排列操作
                a = a.permute(dims=_permute)
                b = b.permute(dims=_permute)
                c = c.permute(dims=_permute)
                run_foo_case(_func, a, b, c)

            # 恢复旧的融合策略
            torch.jit.set_fusion_strategy(old_strategy)
# 如果当前脚本被直接运行而不是作为模块导入，则执行下面的代码
if __name__ == '__main__':
    # 调用一个函数或方法来运行测试
    run_tests()
```