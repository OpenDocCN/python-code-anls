# `.\pytorch\test\test_jit_autocast.py`

```
# Owner(s): ["oncall: jit"]

import torch  # 导入PyTorch库
from torch.cuda.amp import autocast  # 导入自动混合精度支持模块
from typing import Optional, Tuple  # 导入类型提示模块

import unittest  # 导入单元测试模块
from test_jit import JitTestCase  # 导入自定义的JIT测试用例类
from torch.testing._internal.common_cuda import TEST_CUDA  # 导入CUDA测试标志
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo  # 导入测试运行函数和条件跳过装饰器
from torch.testing import FileCheck  # 导入文件检查模块
from jit.test_models import MnistNet  # 导入MnistNet模型

TEST_BFLOAT16 = TEST_CUDA and torch.cuda.is_bf16_supported()  # 检测是否支持CUDA和bfloat16

@skipIfTorchDynamo("Not a TorchDynamo suitable test")  # 标记为不适合TorchDynamo的测试
class TestAutocast(JitTestCase):
    def setUp(self):
        # 准备测试环境，设置常见的输入张量
        if TEST_CUDA:
            self.a_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
            self.b_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
            self.c_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
            self.d_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
            self.a_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
            self.b_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
            self.c_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
            self.d_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
        self.old_value = torch._C._jit_set_autocast_mode(True)  # 设置自动混合精度模式为True
        super().setUp()  # 调用父类的setUp方法

    def tearDown(self):
        torch._C._jit_set_autocast_mode(self.old_value)  # 恢复自动混合精度模式的旧值
        super().tearDown()  # 调用父类的tearDown方法

    @unittest.skipIf(not TEST_CUDA, "No cuda")  # 如果没有CUDA，则跳过测试
    def test_jit_generic_autocast(self):
        @torch.jit.script
        def fn_cuda_autocast(a, b):
            with autocast():
                x = torch.mm(a, b)  # 执行矩阵乘法
                y = torch.sum(x)  # 计算张量x的总和
                return x, y

        @torch.jit.script
        def fn_generic_autocast(a, b):
            with torch.amp.autocast(device_type='cuda'):
                x = torch.mm(a, b)  # 执行矩阵乘法
                y = torch.sum(x)  # 计算张量x的总和
                return x, y
        self.assertEqual(fn_cuda_autocast(self.a_fp32, self.b_fp32), fn_generic_autocast(self.a_fp32, self.b_fp32))  # 断言两种自动混合精度模式的结果相等

    @unittest.skipIf(not TEST_CUDA, "No cuda")  # 如果没有CUDA，则跳过测试
    def test_minimal(self):
        @torch.jit.script
        def fn(a, b):
            with autocast():
                x = torch.mm(a, b)  # 执行矩阵乘法
                y = torch.sum(x)  # 计算张量x的总和
                return x, y
        x, y = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(x.dtype, torch.float16)  # 断言x的数据类型为torch.float16
        self.assertEqual(y.dtype, torch.float32)  # 断言y的数据类型为torch.float32

    @unittest.skipIf(not TEST_CUDA or not TEST_BFLOAT16, "No cuda bfloat16 support")  # 如果没有CUDA或者不支持bfloat16，则跳过测试
    def test_linear_bf16(self):
        @torch.jit.script
        def fn(a, b):
            with autocast(dtype=torch.bfloat16):
                x = torch.mm(a, b)  # 执行矩阵乘法
                y = torch.sum(x)  # 计算张量x的总和
                return x, y
        x, y = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(x.dtype, torch.bfloat16)  # 断言x的数据类型为torch.bfloat16
        self.assertEqual(y.dtype, torch.float32)  # 断言y的数据类型为torch.float32

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    # 定义测试函数，用于测试在 CPU 上最小化计算资源的使用情况
    def test_minimal_cpu(self):
        # 使用 torch.jit.script 装饰器将函数 fn 转换为 Torch 脚本
        @torch.jit.script
        def fn(a, b):
            # 使用 autocast 上下文管理器，自动执行混合精度计算
            with autocast():
                # 执行矩阵乘法操作
                return torch.mm(a, b)
        # 调用 fn 函数，传入 CPU 上的数据，并接收返回结果
        result = fn(self.a_fp32.to('cpu'), self.b_fp32.to('cpu'))
        # 断言返回结果的数据类型为 torch.float32
        self.assertEqual(result.dtype, torch.float32)

    # 如果未启用 CUDA，则跳过该测试
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_minimal_off(self):
        # 使用 torch.jit.script 装饰器将函数 fn 转换为 Torch 脚本
        @torch.jit.script
        def fn(a, b):
            # 使用 autocast 上下文管理器，显式禁用混合精度计算
            with autocast(enabled=False):
                # 执行矩阵乘法操作
                return torch.mm(a, b)
        # 调用 fn 函数，传入 FP32 数据，并接收返回结果
        result = fn(self.a_fp32, self.b_fp32)
        # 断言返回结果的数据类型为 torch.float32
        self.assertEqual(result.dtype, torch.float32)

    # 如果未启用 CUDA，则跳过该测试
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_runtime_autocast_state(self):
        # 使用 torch.jit.script 装饰器将函数 fn 转换为 Torch 脚本
        @torch.jit.script
        def fn(a, b, use_amp: bool):
            # 使用 autocast 上下文管理器，根据运行时参数动态选择是否启用混合精度计算
            with autocast(enabled=use_amp):
                # 执行矩阵乘法操作
                return torch.mm(a, b)
        # 尝试传入运行时值作为 autocast 的启用参数，应引发 RuntimeError
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32, True)

    # 如果未启用 CUDA，则跳过该测试
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_runtime_autocast_state_expr(self):
        # 使用 torch.jit.script 装饰器将函数 fn 转换为 Torch 脚本
        @torch.jit.script
        def fn(a, b):
            # 使用 autocast 上下文管理器，根据条件表达式动态选择是否启用混合精度计算
            with autocast(enabled=True if a[0][0] > 0.5 else False):
                # 执行矩阵乘法操作
                return torch.mm(a, b)
        # 尝试传入运行时值作为 autocast 的启用参数，应引发 RuntimeError
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32)

    # 如果未启用 CUDA，则跳过该测试
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_explicit_casts(self):
        # 使用 torch.jit.script 装饰器将函数 fn 转换为 Torch 脚本
        @torch.jit.script
        def fn(a, b, c, d):
            # 使用 autocast 上下文管理器，自动执行混合精度计算
            with autocast():
                # 执行矩阵乘法操作，并将结果转换为 torch.float32 类型
                e = torch.mm(a.double(), b.double()).float()
                # 执行矩阵乘法操作，并将结果转换为 torch.float64 类型
                f = torch.mm(c, d).double()
            # 执行矩阵乘法操作，并将结果转换为 torch.float64 类型
            g = torch.mm(c.double(), f)
            return e, f, g
        # 调用 fn 函数，传入 FP32 数据，并接收返回结果
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        # 断言返回结果的数据类型符合预期
        self.assertEqual(e.dtype, torch.float32)
        self.assertEqual(f.dtype, torch.float64)
        self.assertEqual(g.dtype, torch.float64)

    # 如果未启用 CUDA，则跳过该测试
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    # 测试中多次使用相同输入值
    def test_duplicate_inputs(self):
        # 使用 torch.jit.script 装饰器将函数 fn 转换为 Torch 脚本
        @torch.jit.script
        def fn(a, b):
            # 使用 autocast 上下文管理器，自动执行混合精度计算
            with autocast():
                # 执行矩阵乘法操作
                e = torch.mm(a, a)
                # 执行矩阵乘法操作
                f = torch.mm(e, e)
            return e, f
        # 调用 fn 函数，传入 FP32 数据，并接收返回结果
        e, f = fn(self.a_fp32, self.b_fp32)
        # 断言返回结果的数据类型为 torch.float16
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)

    # 如果未启用 CUDA，则跳过该测试
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_policy(self):
        # 使用 torch.jit.script 装饰器将函数 fn 转换为 Torch 脚本
        @torch.jit.script
        def fn(a):
            # 使用 autocast 上下文管理器，始终启用混合精度计算
            with autocast(enabled=True):
                # 执行取对数操作
                return torch.log(a)
        # 调用 fn 函数，传入 FP16 数据，并接收返回结果
        result = fn(self.a_fp16)
        # 断言返回结果的数据类型为 torch.float32
        self.assertEqual(result.dtype, torch.float32)
    def test_fp32_policy_with_fp64(self):
        @torch.jit.script
        def fn(a):
            # 使用自动混合精度加速（autocast），启用时不应将fp64缩小为fp32！
            with autocast(enabled=True):
                # 对输入张量取对数
                return torch.log(a)
        # 对使用fp32策略处理fp64的结果进行检查
        result = fn(self.a_fp32.double())
        self.assertEqual(result.dtype, torch.float64)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_promote_policy(self):
        @torch.jit.script
        def fn(a, b, c, d):
            # 使用自动混合精度加速（autocast）
            with autocast():
                # 执行矩阵乘法
                e = torch.mm(a, b)
                # 执行元素操作和累加乘法
                f = torch.addcmul(e, c, d, value=0.1)
            return e, f
        e, f = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_promote_policy_fp64(self):
        @torch.jit.script
        def fn(a, b):
            # 使用自动混合精度加速（autocast），启用时不应将fp64缩小为fp32！
            with autocast(enabled=True):
                # 执行张量操作
                return torch.addcmul(a, a, b, value=0.1)
        result = fn(self.a_fp32.double(), self.b_fp32.double())
        self.assertEqual(result.dtype, torch.float64)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_set_opt_dtype_policy(self):
        @torch.jit.script
        def fn(a, b, c, d, dtype: Optional[int]):
            # 使用自动混合精度加速（autocast），启用时设置不同的数据类型
            with autocast(enabled=True):
                # 对输入张量进行softmax操作
                x = torch.softmax(a, 0)
                y = torch.softmax(b, 0, None)
                z = torch.softmax(c, 0, torch.float64)
                w = torch.softmax(d, 0, dtype)
            return x, y, z, w
        x, y, z, w = fn(self.a_fp16, self.b_fp16, self.c_fp16, self.d_fp16, None)
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(z.dtype, torch.float64)
        self.assertEqual(w.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_set_opt_dtype_policy_fp64(self):
        @torch.jit.script
        def fn(a, b, c, d, dtype: Optional[int]):
            # 使用自动混合精度加速（autocast），启用时设置不同的数据类型
            with autocast(enabled=True):
                # 对输入张量进行softmax操作
                x = torch.softmax(a, 0)
                y = torch.softmax(b, 0, None)
                z = torch.softmax(c, 0, torch.float64)
                w = torch.softmax(d, 0, dtype)
            return x, y, z, w
        x, y, z, w = fn(self.a_fp32.double(), self.b_fp32.double(), self.c_fp32.double(), self.d_fp32.double(), None)
        self.assertEqual(x.dtype, torch.float64)
        self.assertEqual(y.dtype, torch.float64)
        self.assertEqual(z.dtype, torch.float64)
        self.assertEqual(w.dtype, torch.float64)

    @unittest.skipIf(True, "broken due to lack of type propagation")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_control_flow(self):
        # 定义一个 TorchScript 脚本函数 fn，用于控制流测试
        @torch.jit.script
        def fn(a, b, c, d):
            # 开始自动混合精度计算上下文
            with autocast():
                # 如果 a 的第一个元素的第一个元素大于 0.5
                if a[0][0] > 0.5:
                    # 使用 a 和 b 进行矩阵乘法，结果赋给 e
                    e = torch.mm(a, b)
                    # 设置变量 x 为 1
                    x = 1
                else:
                    # 否则，使用 c 和 d 进行矩阵乘法，结果赋给 e
                    e = torch.mm(c, d)
                    # 设置变量 x 为 2
                    x = 2
                # 计算 d 和 e 的矩阵乘法，乘以 x，并赋给 f
                f = torch.mm(d, e) * x
            # 返回 e 和 f 作为结果
            return e, f
        # 调用 fn 函数，传入相应的参数，获取结果 e 和 f
        e, f = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        # 断言 e 和 f 的数据类型为 torch.float16
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)

    # 在常规 Python 中有效，但在 TorchScript 中会创建类型不一致的情况，因为
    # then/else 分支的类型不一致
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_divergent_types(self):
        # 定义一个 TorchScript 脚本函数 fn，用于测试类型分歧
        @torch.jit.script
        def fn(a, b, c, d):
            # 开始自动混合精度计算上下文
            with autocast():
                # 如果 a 的第一个元素的第一个元素大于 0.5
                if a[0][0] > 0.5:
                    # 使用 a 和 b 进行矩阵乘法，结果赋给 e
                    e = torch.mm(a, b)
                    # 将 a 和 b 的矩阵乘法转换为 float 类型，结果赋给 f
                    f = torch.mm(a, b).float()
                else:
                    # 否则，使用 c 和 d 进行矩阵乘法，并转换为 float 类型，结果赋给 e
                    e = torch.mm(c, d).float()
                    # 使用 a 和 b 进行矩阵乘法，结果赋给 f
                    f = torch.mm(a, b)
            # 返回 e 和 f 的矩阵乘法的结果
            return torch.mm(e.float(), f.float())
        # 调用 fn 函数，传入相应的参数，获取结果 result
        result = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        # 断言 result 的数据类型为 torch.float32
        self.assertEqual(result.dtype, torch.float32)

    # 另一个更复杂的类型分歧测试案例
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_divergent_autocast(self):
        # 定义一个 TorchScript 脚本函数 fn，用于测试自动混合精度的类型分歧
        @torch.jit.script
        def fn(a, b, c, d):
            # 定义启用自动混合精度的 autocast 上下文
            autocast_on = autocast(enabled=True)
            # 定义禁用自动混合精度的 autocast 上下文
            autocast_off = autocast(enabled=False)
            # 如果 a 的第一个元素的第一个元素大于 0.5
            if a[0][0] > 0.5:
                # 在启用自动混合精度的上下文中，使用 a 和 b 进行矩阵乘法，结果赋给 e
                with autocast_on:
                    e = torch.mm(a, b)
            else:
                # 否则，在禁用自动混合精度的上下文中，使用 c 和 d 进行矩阵乘法，结果赋给 e
                with autocast_off:
                    e = torch.mm(c, d)
            # 返回 e 与自身的矩阵乘法的结果
            return torch.mm(e, e)
        # 调用 fn 函数，传入相应的参数
        fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)

    # 测试条件自动混合精度表达式不支持的情况
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_conditional_autocast(self):
        # 定义一个 TorchScript 脚本函数 fn，用于测试条件自动混合精度
        @torch.jit.script
        def fn(a, b):
            # 定义启用自动混合精度的 autocast 上下文
            autocast_on = autocast(enabled=True)
            # 定义禁用自动混合精度的 autocast 上下文
            autocast_off = autocast(enabled=False)
            # 如果 a 的第一个元素的第一个元素大于 0.5，则使用启用自动混合精度的上下文
            # 否则，使用禁用自动混合精度的上下文
            with autocast_on if a[0][0] > 0.5 else autocast_off:
                # 返回 a 和 b 的矩阵乘法结果
                return torch.mm(a, b)
        # 使用断言捕获 RuntimeError 异常，因为条件自动混合精度表达式不支持
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32)

    # 测试嵌套自动混合精度上下文的情况
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_nested_autocast(self):
        # 定义一个 TorchScript 脚本函数 fn，用于测试嵌套自动混合精度
        @torch.jit.script
        def fn(a, b, c, d):
            # 在禁用自动混合精度的上下文中执行以下代码块
            with autocast(enabled=False):
                # 使用 a 和 b 进行矩阵乘法，结果赋给 e
                e = torch.mm(a, b)
                # 在启用自动混合精度的上下文中执行以下代码块
                with autocast(enabled=True):
                    # 使用 e 和 c 进行矩阵乘法，结果赋给 f
                    f = torch.mm(e, c)
                    # 在禁用自动混合精度的上下文中执行以下代码块
                    with autocast(enabled=False):
                        # 使用 e 和 d 进行矩阵乘法，结果赋给 g
                        g = torch.mm(e, d)
            # 返回 e, f, g 作为结果
            return e, f, g
        # 调用 fn 函数，传入相应的参数，获取结果 e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        # 断言 e 的数据类型为 torch.float32
        self.assertEqual(e.dtype, torch.float32)
        # 断言 f 的数据类型为 torch.float16
        self.assertEqual(f.dtype, torch.float16)
        # 断言 g 的数据类型为 torch.float32
        self.assertEqual(g.dtype, torch.float32)
    # 如果未开启 CUDA 测试，则跳过该测试函数
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    # 定义测试函数，测试隐式嵌套的 autocast 功能
    def test_implicitly_nested_autocast(self):
        # 使用 Torch 的 JIT 脚本装饰器定义函数 fn
        @torch.jit.script
        def fn(a, b):
            # 在上下文中，首先关闭 autocast，然后开启 autocast
            with autocast(enabled=False), autocast(enabled=True):
                # 执行矩阵乘操作
                return torch.mm(a, b)
        # 调用函数 fn，并将结果保存到 result 中
        result = fn(self.a_fp32, self.b_fp32)
        # 断言结果的数据类型为 torch.float16
        self.assertEqual(result.dtype, torch.float16)

    # 如果未开启 CUDA 测试，则跳过该测试函数
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    # 定义测试函数，测试复用 autocast 实例的功能
    def test_reused_autocast(self):
        # 使用 Torch 的 JIT 脚本装饰器定义函数 fn
        @torch.jit.script
        def fn(a, b, c, d):
            # 创建 autocast 实例并开启 autocast
            autocast_instance = autocast(enabled=True)
            with autocast_instance:
                # 执行第一个矩阵乘操作
                e = torch.mm(a, b)
                with autocast_instance:
                    # 在嵌套的上下文中再次开启 autocast，执行第二个矩阵乘操作
                    e = torch.mm(c, d)
                    # 执行第三个矩阵乘操作
                    f = torch.mm(d, e)
            # 执行最后一个矩阵乘操作
            g = torch.mm(e, f)
            return e, f, g
        # 调用函数 fn，并将结果保存到 e, f, g 中
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        # 断言每个结果的数据类型为 torch.float16
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)
        self.assertEqual(g.dtype, torch.float16)

    # TODO: 修复并启用此测试？
    # （从技术上讲，我们可以修复这个问题，但真的值得吗？）
    @unittest.skipIf(True, "unsuported autocast syntax")
    # 定义测试函数，测试使用 autocast(enabled=True) as autocast_instance 语法的功能
    def test_reused_autocast_expr(self):
        # 使用 Torch 的 JIT 脚本装饰器定义函数 fn
        @torch.jit.script
        def fn(a, b, c, d):
            with autocast(enabled=True) as autocast_instance:
                e = torch.mm(a, b)
                with autocast_instance:
                    e = torch.mm(c, d)
                    f = torch.mm(d, e)
            g = torch.mm(e, f)
            return e, f, g
        # 调用函数 fn，并将结果保存到 e, f, g 中
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        # 断言每个结果的数据类型为 torch.float16
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)
        self.assertEqual(g.dtype, torch.float16)

    # 如果未开启 CUDA 测试，则跳过该测试函数
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    # 定义测试函数，测试调用函数中 autocast 功能的使用
    def test_callees(self):
        # 定义辅助函数 helper，执行矩阵乘操作
        def helper(a, b):
            return torch.mm(a, b)

        # 使用 Torch 的 JIT 脚本装饰器定义函数 fn
        @torch.jit.script
        def fn(a, b):
            # 开启 autocast
            with autocast(enabled=True):
                # 调用 helper 函数执行矩阵乘操作，进行多次操作
                tmp = helper(a, b)
                tmp = helper(tmp, tmp)
                tmp = helper(tmp, tmp)
                tmp = helper(tmp, tmp)
                # 最终调用 helper 函数执行矩阵乘操作并返回结果
                return helper(tmp, b)

        # 调用函数 fn，并将结果保存到 result 中
        result = fn(self.a_fp32, self.b_fp32)
        # 断言结果的数据类型为 torch.float16
        self.assertEqual(result.dtype, torch.float16)

    # 如果未开启 CUDA 测试，则跳过该测试函数
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    # 定义测试函数，测试调用函数中 autocast 功能的使用（在 helper 函数内部开启 autocast）
    def test_callees_with_autocast_on(self):
        # 定义辅助函数 helper，内部开启 autocast 并执行矩阵乘操作
        def helper(a, b):
            with autocast(enabled=True):
                return torch.mm(a, b)

        # 使用 Torch 的 JIT 脚本装饰器定义函数 fn
        @torch.jit.script
        def fn(a, b):
            # 在上下文中关闭 autocast，并调用 helper 函数执行矩阵乘操作
            with autocast(enabled=False):
                return helper(a, b)

        # 调用函数 fn，并将结果保存到 result 中
        result = fn(self.a_fp32, self.b_fp32)
        # 断言结果的数据类型为 torch.float16
        self.assertEqual(result.dtype, torch.float16)
    # 在关闭自动类型转换的情况下测试被调用函数
    def test_callees_with_autocast_off(self):
        # 定义一个辅助函数，禁用自动类型转换后执行矩阵乘法操作
        def helper(a, b):
            with autocast(enabled=False):
                return torch.mm(a, b)

        # 使用 TorchScript 对函数进行脚本化
        @torch.jit.script
        def fn(a, b):
            # 启用自动类型转换后调用 helper 函数
            with autocast(enabled=True):
                return helper(a, b)

        # 执行脚本化函数，并验证结果的数据类型为 torch.float32
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float32)

    # 在即时执行模式下执行 TorchScript
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_eager_and_script(self):
        # 使用 TorchScript 对函数进行脚本化
        @torch.jit.script
        def fn(a, b):
            # 执行矩阵乘法操作
            return torch.mm(a, b)
        
        # 循环执行多次，有时启用自动类型转换
        for i in range(8):
            use_autocast = (i % 2 == 0)
            expected_dtype = torch.float16 if use_autocast else torch.float32
            # 在自动类型转换启用或禁用的情况下执行 TorchScript 函数
            with autocast(enabled=use_autocast):
                result = fn(self.a_fp32, self.b_fp32)
            # 验证结果的数据类型符合预期
            self.assertEqual(result.dtype, expected_dtype)

    # 在 TorchScript 中进行追踪
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_and_tracing(self):
        # 定义一个辅助函数，执行矩阵乘法操作
        def helper(a, b):
            return torch.mm(a, b)

        # 使用 torch.jit.trace 对辅助函数进行追踪
        traced = torch.jit.trace(helper, (self.a_fp32, self.a_fp32))

        # 使用 TorchScript 对函数进行脚本化
        @torch.jit.script
        def fn(a, b):
            # 启用自动类型转换后调用追踪过的函数
            with autocast(enabled=True):
                return traced(a, b)

        # 执行脚本化函数，并验证结果的数据类型为 torch.float16
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    # 在 TorchScript 中进行追踪，并在其中使用自动类型转换
    @unittest.skipIf(True, "autocast(False) is ignored inside traced functions")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_and_tracing_with_autocast(self):
        # 定义一个辅助函数，禁用自动类型转换后执行矩阵乘法操作并乘以 2.0
        def helper(a, b):
            with autocast(enabled=False):
                return torch.mm(a, b) * 2.0

        # 使用 torch.jit.trace 对辅助函数进行追踪
        traced = torch.jit.trace(helper, (self.a_fp32, self.a_fp32))

        # 使用 TorchScript 对函数进行脚本化
        @torch.jit.script
        def fn(a, b):
            # 启用自动类型转换后调用追踪过的函数
            with autocast(enabled=True):
                return traced(a, b)

        # 执行脚本化函数，并验证结果的数据类型为 torch.float32
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float32)

    # 从追踪函数中调用脚本化函数
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_tracing_and_script(self):
        # 使用 TorchScript 对函数进行脚本化
        @torch.jit.script
        def fn(a, b):
            # 在自动类型转换模式下执行矩阵乘法操作
            with autocast():
                return torch.mm(a, b)

        # 定义一个函数来进行追踪
        def traced(a, b):
            return fn(a, b)

        # 使用 torch.jit.trace 对追踪函数进行追踪
        traced = torch.jit.trace(traced, (self.a_fp32, self.b_fp32))

        # 执行追踪后的函数，并验证结果的数据类型为 torch.float16
        result = traced(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    # 从追踪函数中调用脚本化函数，并在其中使用自动类型转换
    @unittest.skipIf(True, "scripted called from traced TorchScript is not yet working")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    # 定义一个测试函数，用于测试带自动类型转换和脚本的功能
    def test_tracing_with_autocast_and_script(self):
        # 定义一个 Torch 脚本函数 fn，执行矩阵乘法操作
        @torch.jit.script
        def fn(a, b):
            return torch.mm(a, b)

        # 定义一个追踪函数 traced，使用自动类型转换，并调用 fn 函数
        def traced(a, b):
            # 启用自动类型转换
            with autocast(enabled=True):
                return fn(a, b)

        # 对 traced 函数进行 Torch 的 JIT 追踪
        traced = torch.jit.trace(traced, (self.a_fp32, self.b_fp32))
        # 调用追踪后的函数，并获取结果
        result = traced(self.a_fp32, self.b_fp32)
        # 断言结果的数据类型为 torch.float16
        self.assertEqual(result.dtype, torch.float16)

    # 如果未开启 CUDA 测试，跳过该测试
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_module(self):
        # 定义一个测试模块 TestModule，继承自 torch.nn.Module
        class TestModule(torch.nn.Module):
            def __init__(self, N, M):
                super().__init__()
                # 初始化权重参数为随机数，数据类型为 torch.float32
                self.weight = torch.nn.Parameter(torch.rand((N, M), dtype=torch.float32))
                # 定义一个线性层，输入维度为 N，输出维度为 M，数据类型为 torch.float32
                self.linear = torch.nn.Linear(N, M).float()

            # 前向传播函数
            def forward(self, input):
                # 启用自动类型转换
                with autocast(enabled=True):
                    # 计算权重与输入的矩阵乘法
                    output = self.weight.mv(input)
                    # 对输出进行线性变换
                    output = self.linear(output)
                    return output

        # 对 TestModule 进行 Torch 的 JIT 脚本化，并移至 CUDA 设备上
        scripted_module = torch.jit.script(TestModule(2, 3)).cuda()
        # 创建随机输入张量，并移至 CUDA 设备上
        input = torch.rand(3, dtype=torch.float32, device='cuda')
        # 调用脚本化的模块进行前向传播，并获取结果
        result = scripted_module(input)
        # 断言结果的数据类型为 torch.float16
        self.assertEqual(result.dtype, torch.float16)

    # 如果自动类型转换装饰器不受支持，或未开启 CUDA 测试，跳过该测试
    @unittest.skipIf(True, "autocast decorators not supported")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_decorator(self):
        # 定义一个 Torch 脚本函数 fn，使用自动类型转换进行矩阵乘法操作
        @torch.jit.script
        @autocast(enabled=True)
        def fn(a, b):
            return torch.mm(a, b)
        # 调用 fn 函数，并获取结果
        result = fn(self.a_fp32, self.b_fp32)
        # 断言结果的数据类型为 torch.float16
        self.assertEqual(result.dtype, torch.float16)

    # 如果未开启 CUDA 测试，跳过该测试
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_decorator_outside_jit(self):
        # 使用自动类型转换装饰器和 Torch 脚本，定义一个 Torch 脚本函数 fn 进行矩阵乘法操作
        @autocast(enabled=True)
        @torch.jit.script
        def fn(a, b):
            return torch.mm(a, b)
        # 调用 fn 函数，并获取结果
        result = fn(self.a_fp32, self.b_fp32)
        # 断言结果的数据类型为 torch.float16
        self.assertEqual(result.dtype, torch.float16)

    # 如果未开启 CUDA 测试，跳过该测试
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_inplace(self):
        # 定义一个 Torch 脚本函数 fn，执行带自动类型转换的原地操作
        @torch.jit.script
        def fn(a, b, c):
            with autocast(enabled=True):
                # 使用自动类型转换进行张量加法与矩阵乘法操作，并将结果赋值给 x
                x = torch.addmm(a, b, c)
                # 使用自动类型转换进行张量加法与矩阵乘法操作，并将结果赋值给 y（原地操作）
                y = torch.addmm(a, b, c, out=a)
                # 使用自动类型转换进行张量加法与矩阵乘法操作，并将结果赋值给 z（原地操作）
                z = a.addmm_(b, c)
                return x, y, z
        # 调用 fn 函数，并获取返回值 x, y, z
        x, y, z = fn(self.a_fp32, self.b_fp32, self.c_fp32)
        # 断言 x 的数据类型为 torch.float16
        self.assertEqual(x.dtype, torch.float16)
        # 断言 y 的数据类型为 torch.float32
        self.assertEqual(y.dtype, torch.float32)
        # 断言 z 的数据类型为 torch.float32
        self.assertEqual(z.dtype, torch.float32)

    # 定义一个辅助函数 _test_autocast，用于测试自动类型转换的函数
    def _test_autocast(self, func, cast_op, *args):
        # 对输入函数 func 进行 Torch 的 JIT 脚本化
        jit_func = torch.jit.script(func)
        # 调用原始函数 func，并获取结果 o
        o = func(*args)
        # 调用脚本化的函数 jit_func，并获取结果 jit_o
        jit_o = jit_func(*args)
        # 如果有指定的类型转换操作 cast_op，使用 FileCheck 检查 jit_func 的图结构
        if cast_op is not None:
            FileCheck().check(cast_op).run(jit_func.graph_for(*args))
        # 遍历 o 和 jit_o 的结果，并断言它们的数据类型相同
        for o0, o1 in zip(o, jit_o):
            self.assertEqual(o0.dtype, o1.dtype)

    # 如果未开启 CUDA 测试，跳过该测试
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_api(self):
        # 测试自动类型转换 API

        def t_autocast_cpu(x, y):
            # 在 CPU 上使用自动类型转换，数据类型为 torch.bfloat16
            with torch.autocast("cpu", dtype=torch.bfloat16):
                return torch.mm(x, y)

        def t_autocast_cuda(x, y):
            # 在 CUDA 上使用自动类型转换，数据类型为 torch.half
            with torch.autocast("cuda", dtype=torch.half):
                return torch.mm(x, y)

        def t_cuda_amp_autocast(x, y):
            # 在 CUDA 上使用自动混合精度类型转换
            with torch.cuda.amp.autocast():
                return torch.mm(x, y)

        def t_cpu_amp_autocast(x, y):
            # 在 CPU 上使用自动混合精度类型转换
            with torch.cpu.amp.autocast():
                return torch.mm(x, y)

        x = torch.randn(5, 5, device="cuda", dtype=torch.float32)
        y = torch.randn(5, 5, device="cuda", dtype=torch.float32)
        self._test_autocast(t_autocast_cpu, "aten::_autocast_to_reduced_precision", x, y)
        self._test_autocast(t_autocast_cuda, "aten::_autocast_to_reduced_precision", x, y)
        self._test_autocast(t_cuda_amp_autocast, "aten::_autocast_to_reduced_precision", x, y)
        self._test_autocast(t_cpu_amp_autocast, "aten::_autocast_to_reduced_precision", x, y)

    @unittest.skipIf(True, "we need to provide dtype argument at this moment")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_api_not_supported(self):
        # 测试不支持的自动类型转换 API

        def t_autocast_cpu(x, y):
            # 当前不支持未提供数据类型的 CPU 上的自动类型转换
            with torch.autocast("cpu"):
                return torch.mm(x, y)

        def t_autocast_cuda(x, y):
            # 当前不支持未提供数据类型的 CUDA 上的自动类型转换
            with torch.autocast("cuda"):
                return torch.mm(x, y)

        x = torch.randn(5, 5, device="cuda", dtype=torch.float32)
        y = torch.randn(5, 5, device="cuda", dtype=torch.float32)
        self._test_autocast(t_autocast_cpu, "aten::_autocast_to_reduced_precision", x, y)
        self._test_autocast(t_autocast_cuda, "aten::_autocast_to_reduced_precision", x, y)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_mixed_dtypes(self):
        # 测试混合数据类型的自动类型转换

        def t(cpu0, cpu1, cuda0, cuda1):
            with torch.autocast("cpu", torch.bfloat16):
                with torch.autocast("cuda", torch.float16):
                    cpu_o = torch.mm(cpu0, cpu1)
                    cuda_o = torch.mm(cuda0, cuda1)
                    return cpu_o, cuda_o

        jit_t = torch.jit.script(t)
        cpu0 = torch.randn(5, 5, device="cpu", dtype=torch.float32)
        cpu1 = torch.randn(5, 5, device="cpu", dtype=torch.float32)
        cuda0 = torch.randn(5, 5, device="cuda", dtype=torch.float32)
        cuda1 = torch.randn(5, 5, device="cuda", dtype=torch.float32)
        self._test_autocast(t, "aten::_autocast_to_reduced_precision", cpu0, cpu1, cuda0, cuda1)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    # 在自动类型转换环境下执行 JIT 执行器的测试函数
    def test_jit_executor_under_autocast(self):

        def t(cpu0, cpu1, cuda0, cuda1):
            # 在 CPU 上执行矩阵乘法
            cpu_o = torch.mm(cpu0, cpu1)
            # 在 CUDA 上执行矩阵乘法
            cuda_o = torch.mm(cuda0, cuda1)
            return cpu_o, cuda_o

        # 将函数 t 脚本化为 JIT 函数
        jit_t = torch.jit.script(t)
        # 创建 CPU 上的随机张量，并指定数据类型和设备
        cpu0 = torch.randn(5, 5, device="cpu", dtype=torch.float32)
        cpu1 = torch.randn(5, 5, device="cpu", dtype=torch.float32)
        # 创建 CUDA 上的随机张量，并指定数据类型和设备
        cuda0 = torch.randn(5, 5, device="cuda", dtype=torch.float32)
        cuda1 = torch.randn(5, 5, device="cuda", dtype=torch.float32)

        # 在 CPU 上启用自动类型转换为 torch.bfloat16 的上下文中，在 CUDA 上启用自动类型转换为 torch.float16 的上下文中，测试自动类型转换
        with torch.autocast("cpu", torch.bfloat16):
            with torch.autocast("cuda", torch.float16):
                self._test_autocast(t, "aten::_autocast_to_reduced_precision", cpu0, cpu1, cuda0, cuda1)

        # 在 CPU 上启用自动类型转换为 torch.bfloat16 的上下文中，测试自动类型转换
        with torch.autocast("cpu", torch.bfloat16):
            self._test_autocast(t, "aten::_autocast_to_reduced_precision", cpu0, cpu1, cuda0, cuda1)

        # 在 CUDA 上启用自动类型转换为 torch.float16 的上下文中，测试自动类型转换
        with torch.autocast("cuda", torch.float16):
            self._test_autocast(t, "aten::_autocast_to_reduced_precision", cpu0, cpu1, cuda0, cuda1)

        # 在非自动类型转换上下文中执行，不应观察到任何类型转换操作
        self._test_autocast(t, None, cpu0, cpu1, cuda0, cuda1)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_autodiff(self):
        # 自动类型转换和自动微分的测试函数
        def t(t0, t1):
            # 执行矩阵乘法
            o = torch.mm(t0, t1)
            # 对结果应用 ReLU 激活函数
            return o.relu()

        # 将函数 t 脚本化为 JIT 函数
        jit_t = torch.jit.script(t)
        # 创建 CUDA 上的随机张量，并指定数据类型和设备，并要求计算梯度
        t0 = torch.randn(5, 5, device="cuda", dtype=torch.float32).requires_grad_()
        t1 = torch.randn(5, 5, device="cuda", dtype=torch.float32).requires_grad_()

        # 运行优化过程
        for i in range(5):
            with torch.autocast("cuda", torch.float16):
                # 在自动类型转换上下文中执行脚本化的 JIT 函数
                jit_o = jit_t(t0, t1)
            # 对 JIT 函数的结果求和并反向传播梯度
            jit_o.sum().backward()

        # 重置梯度
        t0.grad = None
        t1.grad = None
        # 分离并要求梯度的参考张量
        ref_t0 = t0.detach().requires_grad_()
        ref_t1 = t1.detach().requires_grad_()

        with torch.autocast("cuda", torch.float16):
            # 在自动类型转换上下文中执行普通的函数调用
            o = t(ref_t0, ref_t1)
            # 在自动类型转换上下文中执行脚本化的 JIT 函数调用
            jit_o = jit_t(t0, t1)
        # 对 JIT 函数的结果求和并反向传播梯度
        jit_o.sum().backward()
        # 对普通函数的结果求和并反向传播梯度
        o.sum().backward()
        # 断言两个对象相等
        self.assertEqual(o, jit_o)
        # 断言两个张量的梯度相等
        self.assertEqual(t0.grad, ref_t0.grad)
        self.assertEqual(t1.grad, ref_t1.grad)
        # 断言两个对象的数据类型相等
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(t0.grad.dtype, ref_t0.grad.dtype)
        self.assertEqual(t1.grad.dtype, ref_t1.grad.dtype)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    # 定义一个测试方法，用于测试在自动混合精度模式下调用 JIT 方法
    def test_jit_call_method_under_autocast(self):
        # 定义一个 Torch JIT 接口 Iface，包含一个 forward 方法
        @torch.jit.interface
        class Iface(torch.nn.Module):
            def forward(self, x, y) -> torch.Tensor:
                pass

        # 定义实现了 Iface 接口的类 Impl，实现了 forward 方法执行矩阵乘法
        class Impl(Iface):
            def forward(self, x, y):
                return torch.mm(x, y)

        # 定义一个 Torch 模块 Thing1，包含一个实现了 Iface 接口的成员 impl
        class Thing1(torch.nn.Module):
            impl: Iface

            def forward(self, x, y):
                # 在自动混合精度上下文中执行矩阵乘法操作 a 和使用 impl 执行 forward 操作 b
                with torch.cuda.amp.autocast():
                    a = torch.mm(x, y)
                    b = self.impl.forward(a, x)
                    return b

        # 将 Impl 类对象脚本化为 Torch 脚本
        scripted_impl = torch.jit.script(Impl())
        # 创建 Thing1 类对象
        thing1 = Thing1()
        # 将脚本化的 Impl 类对象赋给 Thing1 类的成员 impl
        thing1.impl = scripted_impl
        # 将 Thing1 类对象脚本化为 Torch 脚本
        scripted_thing1 = torch.jit.script(thing1)
        # 创建两个随机矩阵 x 和 y
        x = torch.rand([2, 2])
        y = torch.rand([2, 2])

        # 确保这里不会抛出错误
        with torch.cuda.amp.autocast():
            ans = scripted_thing1.forward(x, y)
        # 断言脚本化的 Thing1 对象的 forward 方法结果与 torch.mm(torch.mm(x, y), x) 相等
        self.assertEqual(torch.mm(torch.mm(x, y), x), ans)

        # 健全性检查：当前全局自动混合精度未启用时，预期会引发 RuntimeError
        self.assertRaises(RuntimeError, lambda: scripted_thing1.forward(x, y))

    # 使用条件跳过装饰器，仅在测试 CUDA 可用时运行
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    # 定义测试 JIT 冻结自动混合精度基础方法
    def test_jit_freeze_autocast_basic(self):
        # 定义一个 Torch 模块 TestModule，包含一个 forward 方法，在自动混合精度上下文中执行矩阵乘法操作
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                with torch.cuda.amp.autocast():
                    return torch.mm(x, y)

        # 创建两个 CUDA 张量 x 和 y
        x = torch.rand((3, 4), dtype=torch.float).cuda()
        y = torch.rand((4, 5), dtype=torch.float).cuda()

        # 实例化 TestModule 并设置为评估模式
        mod = TestModule().eval()

        # 健全性检查：调用自定义方法 _test_autocast，检查是否存在 "aten::_autocast_to_reduced_precision"
        self._test_autocast(mod, "aten::_autocast_to_reduced_precision", x, y)

        # 冻结脚本化的 TestModule，然后运行 FileCheck，检查是否有两个 "aten::_autocast_to_reduced_precision" 节点
        frozen_mod = torch.jit.freeze(torch.jit.script(mod).eval())
        FileCheck().check_count("aten::_autocast_to_reduced_precision", 2, True).run(frozen_mod.graph)

        # 确保运行时优化过程不会重复生成自动混合精度节点
        frozen_mod(x, y)
        optimized_graph = frozen_mod.graph_for(x, y)
        FileCheck().check_count("aten::_autocast_to_reduced_precision", 2, True).run(optimized_graph)
    # 定义一个测试方法，用于测试 JIT 编译、自动类型转换和常量冻结
    def test_jit_freeze_autocast_constants(self):
        # 定义一个继承自 torch.nn.Module 的测试模块
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在构造函数中初始化一个大小为 (3, 4) 的随机张量，并将其移到 GPU 上
                self.x = torch.rand((3, 4), dtype=torch.float).cuda()

            def forward(self, y):
                # 在 forward 方法中使用自动混合精度 autocast 上下文
                with torch.cuda.amp.autocast():
                    # 返回 self.x 与输入张量 y 的矩阵乘法结果
                    return torch.mm(self.x, y)

        # 创建一个大小为 (4, 5) 的随机张量 y，并移到 GPU 上
        y = torch.rand((4, 5), dtype=torch.float).cuda()
        # 实例化 TestModule，并设置为评估模式
        mod = TestModule().eval()

        # 使用 torch.jit.script 对模块进行脚本化，并冻结脚本化模块
        frozen_mod = torch.jit.freeze(torch.jit.script(mod).eval())
        # 验证冻结操作是否将常量 self.x 预转换以移除一个 autocast 调用
        FileCheck().check_count("aten::_autocast_to_reduced_precision", 1, True).run(frozen_mod.graph)

        # 运行时的自动类型转换 pass 会重新插入第二个 autocast 调用，
        # 但常量传播将其与正在转换的常量合并
        frozen_mod(y)
        # 获取优化后的图形
        optimized_graph = frozen_mod.graph_for(y)
        # 验证优化后的图中是否仍存在一个 autocast 调用
        FileCheck().check_count("aten::_autocast_to_reduced_precision", 1, True).run(optimized_graph)

    # 如果测试环境支持 CUDA，则执行以下测试方法
    @unittest.skipIf(TEST_CUDA, "CPU-only test")
    def test_jit_autocast_softmax_cpu(self):
        # 定义一个函数 fn，使用 CPU 的自动混合精度 autocast 上下文，计算 softmax
        def fn(x):
            with torch.cpu.amp.autocast():
                return torch.nn.functional.softmax(x, dim=0)

        # 对函数 fn 进行脚本化
        fn_s = torch.jit.script(fn)
        # 创建一个大小为 (2, 2) 的随机张量 x，数据类型为 torch.bfloat16
        x = torch.rand((2, 2), dtype=torch.bfloat16)
        # 在脚本化函数 fn_s 上调用 x
        fn_s(x)
        # 再次调用 fn_s，将结果保存在变量 y 中
        y = fn_s(x)

        # 断言 y 的数据类型为 torch.bfloat16
        self.assertTrue(y.dtype == torch.bfloat16)

    # 如果测试环境支持 CUDA，则执行以下测试方法
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_jit_autocast_softmax_gpu(self):
        # 定义一个函数 fn，使用 CUDA 的自动混合精度 autocast 上下文，计算 softmax
        def fn(x):
            with torch.cuda.amp.autocast():
                return torch.nn.functional.softmax(x, dim=0)

        # 对函数 fn 进行脚本化
        fn_s = torch.jit.script(fn)
        # 创建一个大小为 (2, 2) 的随机张量 x，数据类型为 torch.half，并移到 GPU 上
        x = torch.rand((2, 2), dtype=torch.half).cuda()
        # 在脚本化函数 fn_s 上调用 x
        fn_s(x)
        # 再次调用 fn_s，将结果保存在变量 y 中
        y = fn_s(x)

        # 断言 y 的数据类型为 torch.float
        self.assertTrue(y.dtype == torch.float)

    # 定义一个测试方法，用于测试忽略自动混合精度 autocast 的情况
    def test_ignore_amp(self):
        # 定义一个函数 foo，计算输入张量 x 与自身的矩阵乘法
        @torch.jit.script
        def foo(x):
            return torch.mm(x, x)

        # 创建一个大小为 [10, 10] 的随机张量 inp，数据类型为 torch.float
        inp = torch.rand([10, 10], dtype=torch.float)
        # 设置函数 foo 忽略自动混合精度 autocast
        foo._set_ignore_amp(True)
        # 在 CPU 上下文中，使用自动混合精度 autocast 执行函数 foo 两次
        with torch.cpu.amp.autocast():
            foo(inp)
            foo(inp)

        # 获取最后一次执行的优化图形
        g = torch.jit.last_executed_optimized_graph()
        # 验证优化图中不包含 "_autocast_to_reduced" 的部分
        FileCheck().check_not("_autocast_to_reduced").run(g)
# 定义一个继承自 torch.nn.Module 的卷积操作和批归一化模块
class convbn(torch.nn.Module):
    def __init__(self, bias_enabled=True):
        super().__init__()
        # 创建一个卷积层，输入通道数为3，输出通道数为64，卷积核大小为7x7，步长为2，是否包含偏置由参数决定
        self.conv = torch.nn.Conv2d(3, 64, 7, stride=2, bias=bias_enabled)
        # 创建一个批归一化层，输入通道数为64
        self.bn = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        # 在前向传播中，先通过卷积层进行卷积操作，然后通过批归一化层处理结果
        return self.bn(self.conv(x))

# 标记为不适合 TorchDynamo 测试的类装饰器
@skipIfTorchDynamo("Not a TorchDynamo suitable test")
class TestJitTraceAutocast(JitTestCase):
    def setUp(self):
        super().setUp()
        # 保存之前的默认数据类型，并设置默认数据类型为 float32
        self.previous_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        # 初始化模型列表，包括 MnistNet 实例和两个 convbn 实例（一个包含偏置，一个不包含偏置）
        self.models = [MnistNet(),
                       convbn(bias_enabled=True),
                       convbn(bias_enabled=False)]
        # 初始化输入列表，包含三个张量，分别对应不同模型的输入数据
        self.inputs = [torch.randn(5, 1, 28, 28, device='cpu'),
                       torch.randn(32, 3, 224, 224, device='cpu'),
                       torch.randn(32, 3, 224, 224, device='cpu')]
        # 保存之前的 JIT 自动类型转换状态，并设置为 False
        self.previous_jit_autocast_pass = torch._C._jit_set_autocast_mode(False)

    def tearDown(self):
        # 恢复之前的 JIT 自动类型转换状态
        torch._C._jit_set_autocast_mode(self.previous_jit_autocast_pass)
        # 恢复之前的默认数据类型
        torch.set_default_dtype(self.previous_default_dtype)
        super().tearDown()

    def test_generate_autocast_jit_trace_model(self):
        # 定义一个测试函数，用于生成自动类型转换的 JIT 追踪模型
        def test_generate_autocast_jit_trace_model(model, x):
            model.eval()
            # 使用自动类型转换环境，追踪模型在输入数据上的行为
            with torch.cpu.amp.autocast(cache_enabled=False), torch.no_grad():
                traced_model = torch.jit.trace(model, x)
            # 冻结追踪模型以提高推理性能
            traced_model = torch.jit.freeze(traced_model)

        # 对模型列表中的每个模型和输入数据进行测试
        for i in range(self.models.__len__()):
            test_generate_autocast_jit_trace_model(self.models[i], self.inputs[i])

    def test_nchw_autocast_jit_trace_model(self):
        # 定义一个测试函数，用于在 NCHW 格式的数据上生成自动类型转换的 JIT 追踪模型
        def test_nchw_autocast_jit_trace_model(model, x):
            model.eval()
            # 使用自动类型转换环境，追踪模型在输入数据上的行为
            with torch.cpu.amp.autocast(cache_enabled=False), torch.no_grad():
                traced_model = torch.jit.trace(model, x)
            # 冻结追踪模型以提高推理性能
            traced_model = torch.jit.freeze(traced_model)
            # 对冻结模型进行推理，使用 clone 的输入数据
            with torch.no_grad():
                y = traced_model(x.clone())
            # 使用自动类型转换环境，再次对原始模型进行推理
            with torch.cpu.amp.autocast(), torch.no_grad():
                y2 = model(x.clone())
            # 使用 torch.testing.assert_close 函数检查两种推理结果的近似程度
            torch.testing.assert_close(y.double(), y2.double(), rtol=1e-03, atol=1e-03)

        # 对模型列表中的每个模型和输入数据进行测试
        for i in range(self.models.__len__()):
            test_nchw_autocast_jit_trace_model(self.models[i], self.inputs[i])
    # 定义一个测试方法，用于测试使用 NHWC 格式进行自动类型转换的 JIT 跟踪模型
    def test_nhwc_autocast_jit_trace_model(self):
        # 定义内部方法，用于测试给定模型和输入的 NHWC 格式自动类型转换 JIT 跟踪模型
        def test_nhwc_autocast_jit_trace_model(model, x):
            # 将模型转换为使用通道为最后一维的内存格式
            model = model.to(memory_format=torch.channels_last)
            # 设定模型为评估模式
            model.eval()
            # 在 CPU 上执行自动类型转换，禁用缓存，不进行梯度计算
            with torch.cpu.amp.autocast(cache_enabled=False), torch.no_grad():
                # 对模型进行 JIT 跟踪，使用 NHWC 格式的输入数据
                traced_model = torch.jit.trace(model, x.to(memory_format=torch.channels_last))
            # 冻结 JIT 跟踪的模型
            traced_model = torch.jit.freeze(traced_model)
            # 再次在禁用梯度计算的情况下执行模型推理
            with torch.no_grad():
                y = traced_model(x.clone().to(memory_format=torch.channels_last))
            # 使用默认的自动类型转换，在禁用梯度计算的情况下再次执行模型推理
            with torch.cpu.amp.autocast(), torch.no_grad():
                y2 = model(x.clone().to(memory_format=torch.channels_last))
            # 使用 torch.testing.assert_close 断言函数比较 y 和 y2 的输出，允许的相对和绝对误差分别为 1e-03
            torch.testing.assert_close(y.double(), y2.double(), rtol=1e-03, atol=1e-03)
        
        # 遍历 self.models 中的每个模型
        for i in range(self.models.__len__()):
            # 检查输入的维度是否为 5
            if self.inputs[i].size().__len__() == 5:
                # 如果是 5 维，则跳过，因为 NHWC 3D 案例尚不支持
                continue
            # 调用内部测试方法，测试当前模型和输入
            test_nhwc_autocast_jit_trace_model(self.models[i], self.inputs[i])

    # 定义一个测试方法
    def test_cat_promote(self):
        # 定义一个简单的模型类，用于测试 torch.cat 方法的行为
        class TestModel(torch.nn.Module):
            def forward(self, a, b):
                # 在正向传播中，简单地连接两个输入张量
                return torch.cat([a, b], 0)

        # 使用 torch.jit.fuser("none") 禁用融合组
        with torch.jit.fuser("none"):
            # 在这个测试用例中，我们将检查在混合 dtype 输入的情况下，是否对 cat 方法进行了升级
            # 为了避免融合组中的 TE，我们在此处禁用 fuser
            for jit_freeze_or_not in [False, True]:
                # 创建一个测试模型实例并设置为评估模式
                test_model = TestModel().eval()
                # 在禁用缓存的情况下，在 CPU 上执行自动类型转换，使用 torch.bfloat16 类型
                with torch.cpu.amp.autocast(cache_enabled=False, dtype=torch.bfloat16), torch.no_grad():
                    # 创建两个随机张量 a 和 b，其中 b 的 dtype 设置为 torch.bfloat16
                    a = torch.rand(24, 128, 128)
                    b = torch.rand(24, 128, 128, dtype=torch.bfloat16)
                    # 执行测试模型的正向传播，连接输入张量 a 和 b
                    c = test_model(a, b)
                    # 对测试模型进行 JIT 跟踪，传入 a 和 b 作为输入
                    traced = torch.jit.trace(test_model, (a, b))
                # 如果 jit_freeze_or_not 为 True，则冻结 JIT 跟踪的模型
                if jit_freeze_or_not:
                    traced = torch.jit.freeze(traced)
                # 多次使用 JIT 跟踪的模型执行正向传播
                for _ in range(3):
                    c2 = traced(a, b)
                # 使用 self.assertTrue 断言函数检查输出张量 c 和 c2 的 dtype 是否为 torch.float32
                self.assertTrue(c.dtype, torch.float32)
                self.assertTrue(c2.dtype, torch.float32)
                # 获取 JIT 跟踪模型的图形对象
                traced_graph = traced.graph_for(a, b)
                # 使用 self.assertTrue 断言函数检查是否有节点的类型是 "aten::to"
                self.assertTrue(any(n.kind() == "aten::to" for n in traced_graph.nodes()))

    # 定义一个测试方法，测试在 CPU 上的脚本自动类型转换
    def test_script_autocast_cpu(self):
        # 定义一个函数 fn，根据是否启用 CPU 自动类型转换，执行不同的操作
        def fn(x):
            if torch.is_autocast_cpu_enabled():
                return x.relu()
            else:
                return x.sin()

        # 对函数 fn 进行 Torch 脚本编译
        fn_s = torch.jit.script(fn)

        # 创建一个随机张量 x
        x = torch.rand((4, 4)) - 0.5
        # 在启用 CPU 自动类型转换的情况下，使用 torch.cpu.amp.autocast 上下文
        with torch.cpu.amp.autocast():
            # 使用 self.assertEqual 断言函数检查 fn_s(x) 和 fn(x) 的输出是否相等
            self.assertEqual(fn_s(x), fn(x))

        # 同样在启用 CPU 自动类型转换的情况下，使用 torch.cpu.amp.autocast(enabled=True) 上下文
        with torch.cpu.amp.autocast(enabled=True):
            # 再次使用 self.assertEqual 断言函数检查 fn_s(x) 和 fn(x) 的输出是否相等
            self.assertEqual(fn_s(x), fn(x))

        # 使用 self.assertTrue 断言函数检查 fn_s 的图形节点中是否存在包含 "is_autocast_cpu_enabled" 的信息
        self.assertTrue(any("is_autocast_cpu_enabled" in x.kind() for x in fn_s.graph.nodes()))

    # 根据是否支持 CUDA 进行条件跳过当前测试方法
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_autocast_cuda(self):
        def fn(x):
            # 如果自动混合精度已启用，则应用ReLU激活函数
            if torch.is_autocast_enabled():
                return x.relu()
            else:
                # 否则，应用正弦函数
                return x.sin()

        # 对函数fn进行脚本化
        fn_s = torch.jit.script(fn)

        # 创建一个随机张量x，并减去0.5
        x = torch.rand((4, 4)) - 0.5
        # 在CPU上启用自动混合精度上下文
        with torch.cpu.amp.autocast():
            # 断言脚本化函数fn_s在输入x上的输出与未脚本化函数fn在输入x上的输出相等
            self.assertEqual(fn_s(x), fn(x))

        # 在CUDA上启用自动混合精度上下文
        with torch.cuda.amp.autocast(enabled=True):
            # 再次断言脚本化函数fn_s在输入x上的输出与未脚本化函数fn在输入x上的输出相等
            self.assertEqual(fn_s(x), fn(x))

        # 断言fn_s的计算图中存在任何节点的kind()包含"is_autocast_enabled"
        self.assertTrue(any("is_autocast_enabled" in x.kind() for x in fn_s.graph.nodes()))


    def test_scripted_aliasing(self):
        # torch.is_autocast_enabled不应该能够移到自动混合精度上下文内部
        def fn(x):
            if torch.is_autocast_enabled():
                y = True
            else:
                y = False
            with torch.cuda.amp.autocast(enabled=True):
                z = x.relu()
            return y, z

        # 对函数fn进行脚本化
        fn_s = torch.jit.script(fn)
        # 获取fn_s的计算图
        graph = fn_s.graph

        # 获取计算图的别名数据库
        aliasdb = graph.alias_db()

        # 查找计算图中所有节点的kind()为"aten::is_autocast_enabled"
        is_enabled_nodes = graph.findAllNodes("aten::is_autocast_enabled")
        # 查找计算图中所有节点的kind()为"prim::Enter"
        enter_nodes = graph.findAllNodes("prim::Enter")

        # 断言计算图中只有一个"aten::is_autocast_enabled"节点
        self.assertEqual(len(is_enabled_nodes), 1)
        # 断言计算图中只有一个"prim::Enter"节点
        self.assertEqual(len(enter_nodes), 1)

        # 使用别名数据库验证，不应将"is_autocast_enabled"节点移动到"prim::Enter"节点后
        self.assertFalse(aliasdb.move_after_topologically_valid(is_enabled_nodes[0], enter_nodes[0]))


    def test_script_autocast_enable_and_check(self):
        def fn(x, y) -> Tuple[torch.Tensor, bool, torch.Tensor, bool, torch.Tensor, bool]:
            # 检查CPU上自动混合精度是否已启用
            b1 = torch.is_autocast_cpu_enabled()
            # 计算矩阵乘积v1
            v1 = torch.mm(x, y)
            # 在启用自动混合精度上下文下
            with torch.cpu.amp.autocast(enabled=True):
                # 检查CPU上自动混合精度是否已启用
                b2 = torch.is_autocast_cpu_enabled()
                # 计算矩阵乘积v2
                v2 = torch.mm(x, y)
                # 在禁用自动混合精度上下文下
                with torch.cpu.amp.autocast(enabled=False):
                    # 检查CPU上自动混合精度是否已启用
                    b3 = torch.is_autocast_cpu_enabled()
                    # 计算矩阵乘积v3
                    v3 = torch.mm(x, y)
            return (v1, b1, v2, b2, v3, b3)

        # bx = is_autocast_cpu_enabled() 的结果应为False，当 (vx = mm(x, y)).dtype 为float 时
        def check_fn_results(arr):
            [v1, b1, v2, b2, v3, b3] = arr
            # 断言v1的dtype为torch.float时，b1应为False，反之亦然
            self.assertTrue((v1.dtype == torch.float) != b1)
            # 断言v2的dtype为torch.float时，b2应为False，反之亦然
            self.assertTrue((v2.dtype == torch.float) != b2)
            # 断言v3的dtype为torch.float时，b3应为False，反之亦然
            self.assertTrue((v3.dtype == torch.float) != b3)

        # 创建两个随机张量x和y，数据类型为torch.float
        x = torch.rand((2, 2), dtype=torch.float)
        y = torch.rand((2, 2), dtype=torch.float)

        # 对函数fn进行脚本化
        fn_s = torch.jit.script(fn)

        # 在禁用自动混合精度上下文下，验证函数fn和fn_s的计算结果
        with torch.cpu.amp.autocast(enabled=False):
            check_fn_results(fn(x, y))
            check_fn_results(fn_s(x, y))

        # 在启用自动混合精度上下文下，验证函数fn和fn_s的计算结果
        with torch.cpu.amp.autocast(enabled=True):
            check_fn_results(fn(x, y))
            check_fn_results(fn_s(x, y))
# 如果当前脚本被直接执行（而不是作为模块导入），则执行下面的代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试代码
    run_tests()
```