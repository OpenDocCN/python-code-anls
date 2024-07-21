# `.\pytorch\test\dynamo\test_ctx_manager.py`

```
# 导入必要的模块和库
# Owner(s): ["module: dynamo"]
import unittest  # 导入单元测试模块

import torch  # 导入PyTorch库

import torch._dynamo.test_case  # 导入PyTorch的私有测试模块
import torch._dynamo.testing  # 导入PyTorch的私有测试工具模块
import torch.onnx.operators  # 导入PyTorch的ONNX操作模块
from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm, same  # 导入测试工具

from torch.nn import functional as F  # 导入PyTorch的函数模块
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION  # 导入CUDA通用测试工具
from torch.testing._internal.common_utils import TEST_WITH_ROCM  # 导入ROCM通用测试工具

# 自定义上下文管理器类
class CutomizedCtxManager:
    def __init__(self, mode):
        self.prev = torch.is_grad_enabled()  # 保存当前的梯度开启状态
        self.mode = mode  # 设置要切换的梯度开启模式

    # 进入上下文时设置梯度开启状态为指定模式
    def __enter__(self):
        torch._C._set_grad_enabled(self.mode)

    # 退出上下文时恢复到先前保存的梯度开启状态
    def __exit__(self, exc_type, exc_value, traceback):
        torch._C._set_grad_enabled(self.prev)

# 上下文管理器测试类，继承于PyTorch的私有测试用例类
class CtxManagerTests(torch._dynamo.test_case.TestCase):
    # 测试禁用梯度的情况
    def test_no_grad(self):
        # 定义函数 fn1
        def fn1(a, b):
            x = a + 1
            # 多余的 no_grad 应该被忽略
            with torch.no_grad():
                x = x + b
            x = x + 2
            return x

        # 定义函数 fn2
        def fn2(a, b):
            x = a + 1
            with torch.set_grad_enabled(False):
                x = x + b
            x = x + 2
            return x

        # 定义函数 fn3
        def fn3(a, b):
            x = a + 1
            with torch.enable_grad():
                x = x + b
            x = x + 2
            return x

        # 定义函数 fn4
        def fn4(a, b):
            x = a + 1
            with torch.set_grad_enabled(True):
                if torch.is_grad_enabled():
                    x = x + b
            x = x + 2
            return x

        # 在禁用梯度的上下文中进行测试
        with torch.no_grad():
            # 使用标准测试工具测试 fn1 和 fn2 函数
            torch._dynamo.testing.standard_test(self, fn=fn1, nargs=2, expected_ops=3)  # 合并为无操作
            torch._dynamo.testing.standard_test(self, fn=fn2, nargs=2, expected_ops=3)  # 合并为无操作
            # 使用标准测试工具测试 fn3 和 fn4 函数
            torch._dynamo.testing.standard_test(self, fn=fn3, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(self, fn=fn4, nargs=2, expected_ops=5)

        # 在启用梯度的上下文中进行测试
        with torch.enable_grad():
            # 使用标准测试工具测试 fn1 和 fn2 函数
            torch._dynamo.testing.standard_test(self, fn=fn1, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(self, fn=fn2, nargs=2, expected_ops=5)
            # 使用标准测试工具测试 fn3 和 fn4 函数
            torch._dynamo.testing.standard_test(self, fn=fn3, nargs=2, expected_ops=3)  # 合并为无操作
            torch._dynamo.testing.standard_test(self, fn=fn4, nargs=2, expected_ops=3)  # 合并为无操作
    def test_grad_mode_guard(self):
        # 定义一个测试函数，用于验证梯度模式保护功能
        def fn(a, b):
            # 保存当前梯度状态，并关闭梯度计算
            prev_grad = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
            # 对张量 a 进行加法操作
            a = a + 1
            # 将张量 a 转换为列表，此处会中断计算图的生成
            a.tolist()  # graph break
            # 计算 a + b
            ret = a + b
            # 恢复之前的梯度状态
            torch.set_grad_enabled(prev_grad)
            return ret

        # 创建两个随机张量 a 和 b
        a = torch.randn([3, 4])
        b = torch.randn([3, 4])
        # 使用 CompileCounter 对 fn 进行优化
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 多次调用优化后的函数
        for _ in range(10):
            opt_fn(a, b)
        # 断言优化次数为 2
        self.assertEqual(cnts.frame_count, 2)

    def test_nested_grad_mode_graph_break(self):
        # 定义一个测试函数，用于验证嵌套梯度模式下的计算图中断
        def fn(x):
            # 记录进入函数前的梯度状态
            before = torch.is_grad_enabled()
            # 关闭梯度计算上下文
            with torch.set_grad_enabled(False):
                # 手动中断计算图生成
                torch._dynamo.graph_break()
                # 开启梯度计算上下文
                with torch.set_grad_enabled(True):
                    # 对张量 x 进行乘法操作
                    x = torch.mul(x, 5)
                    # 手动中断计算图生成
                    torch._dynamo.graph_break()
                    # 对张量 x 进行平方根操作
                    x = torch.sqrt(x)
                    # 断言当前仍处于梯度计算上下文
                    assert torch.is_grad_enabled()
                # 断言当前未处于梯度计算上下文
                assert not torch.is_grad_enabled()
            # 断言函数结束时梯度状态与进入时相同
            assert torch.is_grad_enabled() == before
            return x

        # 创建一个随机张量 x
        a = torch.randn([3, 4])
        # 使用 CompileCounter 对 fn 进行优化
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        # 多次调用优化后的函数
        for _ in range(10):
            opt_fn(a)
        # 断言优化次数为 2
        self.assertEqual(cnts.frame_count, 2)

    def test_torch_profiler(self):
        # 定义一个测试函数，对 torch.profiler.* 进行优化
        # 将 torch.profiler.* 包装为 NullContextVariable，什么也不做
        def fn(x):
            # 对张量 x 进行平方操作
            y = x**2
            # 使用 torch.profiler.profile() 进行性能分析
            with torch.profiler.profile():
                # 对 y 进行加法操作
                y = y + 2
                # 使用 torch.profiler.record_function("my_function") 进行性能分析
                with torch.profiler.record_function("my_function"):
                    # 对 y 进行立方操作
                    z = y**3
                    # 将 z 转换为列表，中断计算图的生成
                    z.tolist()  # graph break
                    # 对 z 进行加法操作
                    z = z + 1
            return z

        # 创建一个随机张量 x
        x = torch.randn((2, 2), requires_grad=True)
        # 记录函数的参考结果
        ref = fn(x)
        # 使用 CompileCounter 对 fn 进行优化
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 调用优化后的函数
        res = opt_fn(x)
        # 断言优化后的结果与参考结果相同
        self.assertTrue(same(ref, res))
        # 断言优化次数为 2
        self.assertEqual(cnts.frame_count, 2)

    def test_autograd_profiler(self):
        # 定义一个测试函数，对 torch.autograd.profiler.* 进行优化
        # 将 torch.autograd.profiler.* 包装为 NullContextVariable，什么也不做
        def fn(x):
            # 对张量 x 进行平方操作
            y = x**2
            # 使用 torch.autograd.profiler.profile() 进行性能分析
            with torch.autograd.profiler.profile():
                # 对 y 进行加法操作
                y = y + 2
                # 使用 torch.autograd.profiler.record_function("my_function") 进行性能分析
                with torch.autograd.profiler.record_function("my_function"):
                    # 对 y 进行立方操作
                    z = y**3
                    # 将 z 转换为列表，中断计算图的生成
                    z.tolist()  # graph break
                    # 对 z 进行加法操作
                    z = z + 1
            return z

        # 创建一个随机张量 x
        x = torch.randn((2, 2), requires_grad=True)
        # 记录函数的参考结果
        ref = fn(x)
        # 使用 CompileCounter 对 fn 进行优化
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 调用优化后的函数
        res = opt_fn(x)
        # 断言优化后的结果与参考结果相同
        self.assertTrue(same(ref, res))
        # 断言优化次数为 2
        self.assertEqual(cnts.frame_count, 2)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @unittest.expectedFailure  # 标记此测试预期会失败，链接到对应的GitHub问题页面
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")  # 如果CUDA不可用，则跳过测试

    def test_cuda_stream_context_manager1(self):
        def fn(x):
            s = torch.cuda.Stream()  # 创建一个CUDA流对象
            x = torch.mul(x, 5)  # 对输入张量每个元素乘以5
            x = torch.add(x, 2)  # 对输入张量每个元素加2
            current_stream = torch.cuda.current_stream()  # 获取当前CUDA流
            s.wait_stream(current_stream)  # 等待当前流完成操作

            with torch.cuda.stream(s):  # 使用创建的CUDA流进行操作
                x = torch.relu(x)  # 对输入张量每个元素进行ReLU操作

            current_stream.wait_stream(s)  # 等待当前流完成操作
            x = torch.add(x, 1)  # 对输入张量每个元素加1
            x = torch.cos(x)  # 对输入张量每个元素计算余弦值
            return x  # 返回操作后的张量

        x = torch.randn((2, 2), device="cuda")  # 在CUDA设备上生成一个随机张量
        ref = fn(x)  # 调用函数fn，记录参考结果
        cnts = torch._dynamo.testing.CompileCounter()  # 创建编译计数器对象
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)  # 优化fn函数的执行
        res = opt_fn(x)  # 调用优化后的函数fn，记录优化后的结果
        self.assertEqual(ref, res)  # 断言参考结果与优化结果相等
        self.assertEqual(cnts.frame_count, 1)  # 断言编译计数器的帧数为1
        self.assertEqual(cnts.op_count, 12)  # 断言编译计数器的操作数为12

    @unittest.expectedFailure  # 标记此测试预期会失败，链接到对应的GitHub问题页面
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")  # 如果CUDA不可用，则跳过测试

    def test_cuda_stream_across_graph_break(self):
        def fn(x):
            s = torch.cuda.Stream()  # 创建一个CUDA流对象
            x = torch.mul(x, 5)  # 对输入张量每个元素乘以5
            x = torch.add(x, 2)  # 对输入张量每个元素加2

            print("foo")  # 打印"foo"

            tcs = torch.cuda.stream(s)  # 使用创建的CUDA流对象
            current_stream = torch.cuda.current_stream()  # 获取当前CUDA流
            s.wait_stream(current_stream)  # 等待当前流完成操作

            with tcs:
                x = torch.relu(x)  # 对输入张量每个元素进行ReLU操作

            current_stream.wait_stream(s)  # 等待当前流完成操作
            x = torch.add(x, 1)  # 对输入张量每个元素加1
            x = torch.cos(x)  # 对输入张量每个元素计算余弦值
            return x  # 返回操作后的张量

        x = torch.randn((2, 2), device="cuda")  # 在CUDA设备上生成一个随机张量
        ref = fn(x)  # 调用函数fn，记录参考结果
        cnts = torch._dynamo.testing.CompileCounter()  # 创建编译计数器对象
        opt_fn = torch._dynamo.optimize(cnts)(fn)  # 优化fn函数的执行
        res = opt_fn(x)  # 调用优化后的函数fn，记录优化后的结果
        self.assertEqual(ref, res)  # 断言参考结果与优化结果相等
        self.assertEqual(cnts.frame_count, 2)  # 断言编译计数器的帧数为2
        self.assertEqual(cnts.op_count, 9)  # 断言编译计数器的操作数为9
    def test_cuda_stream_context_manager2(self):
        # 定义一个测试函数，使用 CUDA 流上下文管理器
        def fn(x, s):
            # 乘以5
            x = torch.mul(x, 5)
            # 加2
            x = torch.add(x, 2)

            # 获取当前 CUDA 流
            current_stream = torch.cuda.current_stream()
            # 等待当前流完成
            s.wait_stream(current_stream)

            # 使用新的 CUDA 流上下文管理器
            with torch.cuda.stream(s):
                # 对 x 应用 ReLU 激活函数
                x = torch.relu(x)

            # 等待当前流完成指定的流
            current_stream.wait_stream(s)
            # 使用当前 CUDA 流上下文管理器
            with torch.cuda.stream(current_stream):
                # 对 x 应用 ReLU 激活函数
                x = torch.relu(x)

            # 创建一个新的 CUDA 流
            s2 = torch.cuda.Stream()
            # 等待当前流完成指定的流
            s2.wait_stream(current_stream)
            # 使用新的 CUDA 流上下文管理器
            with torch.cuda.stream(s2):
                # 对 x 应用 ReLU 激活函数
                x = torch.relu(x)

            # 等待当前流完成指定的流
            current_stream.wait_stream(s2)
            # 加1
            x = torch.add(x, 1)
            # 求 x 的余弦值
            x = torch.cos(x)
            return x

        # 在 CUDA 设备上生成一个随机张量
        x = torch.randn((2, 2), device="cuda")
        # 创建一个新的 CUDA 流
        s = torch.cuda.Stream()
        # 调用 fn 函数，得到参考结果
        ref = fn(x, s)
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化 fn 函数，并应用编译计数器
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 使用优化后的函数计算结果
        res = opt_fn(x, s)
        # 断言参考结果和优化后的结果相等
        self.assertEqual(ref, res)
        # 断言帧计数为1
        self.assertEqual(cnts.frame_count, 1)
        # 断言操作计数为18
        self.assertEqual(cnts.op_count, 18)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_method(self):
        # 定义一个测试函数，使用 CUDA 流方法
        def fn(x):
            # 乘以1
            x = torch.mul(x, 1)
            # 加2
            x = torch.add(x, 2)

            # 创建一个新的 CUDA 流
            new_stream = torch.cuda.Stream()
            # 获取当前 CUDA 流
            cur_stream = torch.cuda.current_stream()
            # 等待当前流完成
            new_stream.wait_stream(cur_stream)

            # 使用新的 CUDA 流上下文管理器
            with torch.cuda.stream(new_stream):
                # 对 x 应用正弦函数
                x = torch.sin(x)
                # 加3
                x = torch.add(x, 3)

            # 等待当前流完成指定的流
            cur_stream.wait_stream(new_stream)

            # 加4
            x = torch.add(x, 4)
            # 查询当前流是否空闲
            is_idle = cur_stream.query()
            # 同步当前 CUDA 流
            cur_stream.synchronize()

            # 使用新的 CUDA 流上下文管理器
            with torch.cuda.stream(new_stream):
                # 加5
                x = torch.add(x, 5)
            # 同步新的 CUDA 流
            new_stream.synchronize()

            # 检查两个流是否相等
            is_equal = cur_stream == new_stream

            # 对 x 应用 ReLU 激活函数
            x = torch.relu(x)
            # 对 x 应用余弦函数
            x = torch.cos(x)
            return x

        # 在 CUDA 设备上生成一个随机张量
        x = torch.randn((2, 2), device="cuda")
        # 调用 fn 函数，得到参考结果
        ref = fn(x)
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化 fn 函数，并应用编译计数器
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 使用优化后的函数计算结果
        res = opt_fn(x)
        # 断言参考结果和优化后的结果相等
        self.assertEqual(ref, res)
        # 断言帧计数为1
        self.assertEqual(cnts.frame_count, 1)
        # 断言操作计数为21
        self.assertEqual(cnts.op_count, 21)
    def test_cuda_stream_compared_with_constant(self):
        # 定义一个函数 fn，接受一个参数 x
        def fn(x):
            # x 乘以 1，保持不变
            x = torch.mul(x, 1)
            # x 加上 2
            x = torch.add(x, 2)

            # 获取当前 CUDA 流
            cur_stream = torch.cuda.current_stream()
            # 如果当前流存在，则返回 x + 1
            if cur_stream is not None:
                return x + 1
            # 否则返回 x - 1
            return x - 1

        # 定义另一个函数 fn2，接受一个参数 x
        def fn2(x):
            # x 乘以 1，保持不变
            x = torch.mul(x, 1)
            # x 加上 2
            x = torch.add(x, 2)

            # 获取当前 CUDA 流
            cur_stream = torch.cuda.current_stream()
            # 如果当前流不等于 "const_str"，则返回 x + 1
            if cur_stream != "const_str":
                return x + 1
            # 否则返回 x - 1
            return x - 1

        # 生成一个在 CUDA 设备上的随机数张量 x
        x = torch.randn((2, 2), device="cuda")
        # 计算函数 fn 在 x 上的参考值
        ref = fn(x)
        # 使用动态编译器优化 fn 函数
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 计算优化后的 fn 函数在 x 上的结果
        res = opt_fn(x)
        # 断言参考值和结果值相等
        self.assertEqual(ref, res)

        # 使用动态编译器优化 fn2 函数
        opt_fn2 = torch._dynamo.optimize(cnts, nopython=True)(fn2)
        # 计算优化后的 fn2 函数在 x 上的结果
        res2 = opt_fn2(x)
        # 断言参考值和结果值相等
        self.assertEqual(ref, res2)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_compared_with_stream(self):
        # 定义一个函数 fn，接受参数 x、s0、s1
        def fn(x, s0, s1):
            # 如果 s0 等于 s1，则返回 x + 1
            if s0 == s1:
                return x + 1
            # 否则返回 x - 1
            else:
                return x - 1

        # 创建两个 CUDA 流对象 s0 和 s1
        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        # 生成一个在 CPU 上的随机数张量 x
        x = torch.randn(2, 2)
        # 使用动态编译器优化 fn 函数
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        # 计算 fn 在给定参数下的参考结果
        ref0 = fn(x, s0, s1)
        # 计算优化后的 fn 函数在给定参数下的结果
        res0 = opt_fn(x, s0, s1)
        # 断言编译帧数为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言参考结果和优化结果相等
        self.assertEqual(ref0, res0)

        # 计算 fn 在另一组参数下的参考结果
        ref1 = fn(x, s1, s1)
        # 计算优化后的 fn 函数在另一组参数下的结果
        res1 = opt_fn(x, s1, s1)
        # 断言编译帧数为 2，因为输入参数改变导致重新编译
        self.assertEqual(cnts.frame_count, 2)
        # 断言参考结果和优化结果相等
        self.assertEqual(ref1, res1)

        # 重置动态编译器状态
        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        # 计算 fn 在另一组参数下的参考结果
        ref1 = fn(x, s1, s1)
        # 计算优化后的 fn 函数在另一组参数下的结果
        res1 = opt_fn(x, s1, s1)
        # 断言编译帧数为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言参考结果和优化结果相等
        self.assertEqual(ref1, res1)

        # 计算 fn 在第一组参数下的参考结果
        ref0 = fn(x, s0, s1)
        # 计算优化后的 fn 函数在第一组参数下的结果
        res0 = opt_fn(x, s0, s1)
        # 断言编译帧数为 2，因为输入参数改变导致重新编译
        self.assertEqual(cnts.frame_count, 2)
        # 断言参考结果和优化结果相等
        self.assertEqual(ref0, res0)
    # 定义一个测试方法，用于验证 CUDA 事件方法在编译之外创建流
    def test_cuda_event_method_create_stream_outside_of_compile(self):
        
        # 定义一个函数 fn，接受输入 x、当前流 cur_stream 和新流 new_stream
        def fn(x, cur_stream, new_stream):
            # 使用 Torch 函数对 x 进行逐元素乘法（scalar multiply）
            x = torch.mul(x, 1)
            # 使用 Torch 函数对 x 进行逐元素加法（element-wise addition）
            x = torch.add(x, 2)

            # 再次对 x 进行逐元素加法
            x = torch.add(x, 3)

            # 在当前流 cur_stream 上记录事件，并返回事件对象
            event = cur_stream.record_event()
            # 查询事件对象的状态，返回一个布尔值表示事件是否处于空闲状态
            is_idle = event.query()

            # 等待新流 new_stream 中的事件完成
            new_stream.wait_event(event)
            # 在新流上下文中执行以下操作
            with torch.cuda.stream(new_stream):
                # 对 x 进行逐元素加法
                x = torch.add(x, 4)

            # 创建一个新的 CUDA 事件对象
            new_event = torch.cuda.Event()
            # 在新流上记录该事件
            new_event.record(new_stream)

            # 等待当前流 cur_stream 中的事件 new_event 完成
            new_event.wait(cur_stream)
            # 再次对 x 进行逐元素加法
            x = torch.add(x, 5)

            # 使用新事件进行同步
            new_event.synchronize()

            # 对 x 进行逐元素应用 ReLU 激活函数
            x = torch.relu(x)
            # 对 x 进行逐元素应用余弦函数
            x = torch.cos(x)
            # 返回处理后的结果 x
            return x

        # 生成一个在 CUDA 设备上的随机张量 x
        x = torch.randn((2, 2), device="cuda")
        # 获取当前 CUDA 流
        cur_stream = torch.cuda.current_stream()
        # 创建一个新的 CUDA 流
        new_stream = torch.cuda.Stream()
        # 调用函数 fn，传入输入张量 x 和两个 CUDA 流，并保存结果作为参考值
        ref = fn(x, cur_stream, new_stream)

        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用动态编译优化函数 fn，并设置禁用 Python 的选项
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 调用优化后的函数 opt_fn，传入输入张量 x 和两个 CUDA 流，并保存结果
        res = opt_fn(x, cur_stream, new_stream)

        # 使用断言检查两个结果是否相等
        self.assertEqual(ref, res)
        # 使用断言检查编译计数器中的帧数是否为 1
        self.assertEqual(cnts.frame_count, 1)
        # 使用断言检查编译计数器中的操作数是否为 19
        self.assertEqual(cnts.op_count, 19)

    # 如果 CUDA 可用，则执行以下测试方法
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_event_method(self):
        
        # 定义一个函数 fn，接受输入 x
        def fn(x):
            # 使用 Torch 函数对 x 进行逐元素乘法
            x = torch.mul(x, 1)
            # 使用 Torch 函数对 x 进行逐元素加法
            x = torch.add(x, 2)

            # 获取当前 CUDA 流
            cur_stream = torch.cuda.current_stream()
            # 创建一个新的 CUDA 流
            new_stream = torch.cuda.Stream()

            # 再次对 x 进行逐元素加法
            x = torch.add(x, 3)

            # 在当前流 cur_stream 上记录事件，并返回事件对象
            event = cur_stream.record_event()
            # 查询事件对象的状态，返回一个布尔值表示事件是否处于空闲状态
            is_idle = event.query()

            # 等待新流 new_stream 中的事件完成
            new_stream.wait_event(event)
            # 在新流上下文中执行以下操作
            with torch.cuda.stream(new_stream):
                # 对 x 进行逐元素加法
                x = torch.add(x, 4)

            # 创建一个新的 CUDA 事件对象
            new_event = torch.cuda.Event()
            # 在新流上记录该事件
            new_event.record(new_stream)

            # 等待当前流 cur_stream 中的事件 new_event 完成
            new_event.wait(cur_stream)
            # 再次对 x 进行逐元素加法
            x = torch.add(x, 5)

            # 使用新事件进行同步
            new_event.synchronize()

            # 对 x 进行逐元素应用 ReLU 激活函数
            x = torch.relu(x)
            # 对 x 进行逐元素应用余弦函数
            x = torch.cos(x)
            # 返回处理后的结果 x
            return x

        # 生成一个在 CUDA 设备上的随机张量 x
        x = torch.randn((2, 2), device="cuda")
        # 调用函数 fn，传入输入张量 x，并保存结果作为参考值
        ref = fn(x)

        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用动态编译优化函数 fn，并设置禁用 Python 的选项
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 调用优化后的函数 opt_fn，传入输入张量 x，并保存结果
        res = opt_fn(x)

        # 使用断言检查两个结果是否相等
        self.assertEqual(ref, res)
        # 使用断言检查编译计数器中的帧数是否为 1
        self.assertEqual(cnts.frame_count, 1)
        # 使用断言检查编译计数器中的操作数是否为 19
        self.assertEqual(cnts.op_count, 19)
    def test_autograd_profiler_enabled(self):
        # 定义一个简单的函数 fn，根据是否启用自动求导性能分析器，返回 x+1 或者 x-1
        def fn(x):
            if torch.autograd._profiler_enabled():
                return x + 1
            else:
                return x - 1

        # 创建一个随机张量 x，并设置 requires_grad=True
        x = torch.randn((2, 2), requires_grad=True)
        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化 fn 函数，得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        # 如果启用了自动求导性能分析器，则禁用它
        if torch.autograd._profiler_enabled():
            torch.autograd._disable_profiler()
        # 断言当前没有启用自动求导性能分析器
        assert not torch.autograd._profiler_enabled()
        # 计算原始函数 fn 在 x 上的结果 ref
        ref = fn(x)
        # 计算优化函数 opt_fn 在 x 上的结果 res
        res = opt_fn(x)
        # 断言 ref 和 res 相等
        self.assertTrue(same(ref, res))

        # 使用自动求导性能分析器进行代码段的性能分析
        with torch.autograd.profiler.profile():
            # 断言当前启用了自动求导性能分析器
            assert torch.autograd._profiler_enabled()
            # 计算原始函数 fn 在 x 上的结果 ref
            ref = fn(x)
            # 计算优化函数 opt_fn 在 x 上的结果 res
            res = opt_fn(x)
            # 断言 ref 和 res 相等
            self.assertTrue(same(ref, res))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast(self):
        # 如果当前环境不支持 CUDA，则跳过测试
        if not torch.cuda.is_bf16_supported():
            raise unittest.SkipTest("requires bf16")

        # 定义一个简单的神经网络模块 MyModule
        class MyModule(torch.nn.Module):
            def forward(self, x):
                # 在 CUDA 设备上创建三个随机张量
                a_float32 = torch.rand((8, 8), device="cuda")
                b_float32 = torch.rand((8, 8), device="cuda")
                d_float32 = torch.rand((8, 8), device="cuda")

                # 使用自动混合精度进行计算
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    e_float16 = torch.mm(a_float32, b_float32)
                    f_float16 = torch.mm(d_float32, e_float16)
                return f_float16

        # 创建 MyModule 实例 module
        module = MyModule()
        # 在模块上进行前向计算，输入一个包含一个元素的张量 [0.5]
        real = module(torch.tensor([0.5]))
        # 获取计算结果的设备和数据类型
        real_device = real.device
        real_dtype = real.dtype

        # 导出模块的计算图和保护装置
        graph, guards = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        # 在导出的计算图上执行前向计算，输入一个包含一个元素的张量 [0.5]
        exported = graph(torch.tensor([0.5]))
        # 断言导出的计算结果的设备和数据类型与实际计算结果一致
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        # 断言导出的计算结果在 CUDA 设备上
        self.assertEqual(exported.device.type, "cuda")
        self.assertEqual(exported.device.index, 0)
        # 断言导出的计算结果数据类型为 torch.bfloat16

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    # 测试 CUDA 自动混合精度自动转换
    def test_cuda_amp_autocast(self):
        # 定义一个自定义的 PyTorch 模块
        class MyModule(torch.nn.Module):
            # 定义模块的前向传播方法
            def forward(self, x):
                # 在 CUDA 设备上生成一个随机的浮点张量 a_float32
                a_float32 = torch.rand((8, 8), device="cuda")
                # 在 CUDA 设备上生成一个随机的浮点张量 b_float32
                b_float32 = torch.rand((8, 8), device="cuda")

                # 使用自动混合精度自动转换上下文管理器，将 a_float32 和 b_float32 的矩阵乘法结果赋值给 c_float64
                with torch.cuda.amp.autocast(dtype=torch.float64):
                    c_float64 = torch.mm(a_float32, b_float32)
                # 返回 c_float64
                return c_float64

        # 创建 MyModule 的实例 module
        module = MyModule()
        # 将张量 [0.5] 输入模块并获取结果 real
        real = module(torch.tensor([0.5]))
        # 获取 real 张量的设备
        real_device = real.device
        # 获取 real 张量的数据类型
        real_dtype = real.dtype

        # 使用 torch._dynamo.export 导出模块的计算图
        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        # 在导出的计算图上执行输入张量 [0.5] 并获取结果 exported
        exported = graph(torch.tensor([0.5]))
        # 断言导出结果的设备与 real 张量相同
        self.assertEqual(exported.device, real_device)
        # 断言导出结果的数据类型与 real 张量相同
        self.assertEqual(exported.dtype, real_dtype)

        # 断言导出结果的设备类型为 "cuda"
        self.assertEqual(exported.device.type, "cuda")
        # 断言导出结果的设备索引为 0
        self.assertEqual(exported.device.index, 0)
        # 断言导出结果的数据类型为 torch.float64

    # 测试 CPU 是否启用自动混合精度自动转换
    def test_is_autocast_cpu_enabled(self):
        # 定义一个函数 fn，接收两个浮点张量 a_float32 和 b_float32 作为输入
        def fn(a_float32, b_float32):
            # 使用 CPU 上的自动混合精度自动转换上下文管理器，将 a_float32 和 b_float32 的矩阵乘法结果赋值给 c_float16
            with torch.cpu.amp.autocast(dtype=torch.bfloat16):
                c_float16 = torch.mm(a_float32, b_float32)
                # 如果 CPU 上启用了自动混合精度自动转换，则将 c_float16 加 1
                if torch.is_autocast_cpu_enabled():
                    c_float16 = c_float16 + 1
            # 返回 c_float16
            return c_float16

        # 生成两个随机的浮点张量 a 和 b
        a = torch.rand((8, 8))
        b = torch.rand((8, 8))
        # 调用函数 fn，并将结果保存在 ref 中
        ref = fn(a, b)
        # 使用 torch._dynamo.optimize 对函数 fn 进行优化，并生成优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 在优化后的函数上调用输入张量 a 和 b，并将结果保存在 res 中
        res = opt_fn(a, b)
        # 断言 ref 和 res 是相同的
        self.assertTrue(same(ref, res))

    # 如果不支持 PLATFORM_SUPPORTS_FLASH_ATTENTION 或者 TEST_WITH_ROCM，则跳过测试
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION or TEST_WITH_ROCM,
        "Can't run fused SDPA on this platform",
    )
    def test_autocast_sdpa(self):
        # 定义一个继承自torch.nn.Module的子类MyModule，用于自定义神经网络模块
        class MyModule(torch.nn.Module):
            # 重写forward方法，处理输入并返回输出
            def forward(self, query, key, value):
                # 使用torch.autocast("cpu")上下文，将以下代码块中的操作转换为CPU计算
                with torch.autocast("cpu"):
                    # 使用torch.autocast("cuda", dtype=torch.float32)上下文，将以下代码块中的操作转换为CUDA计算，并指定数据类型为torch.float32
                    with torch.autocast("cuda", dtype=torch.float32):
                        # 调用F.scaled_dot_product_attention函数进行注意力计算
                        out = F.scaled_dot_product_attention(
                            query, key, value, None, 0.0, True
                        )
                # 返回计算结果out
                return out

        # 指定数据类型为torch.float32
        dtype = torch.float32
        # 设置序列长度为1
        seq_len_q = 1
        seq_len_k = 1
        # 设置头部维度为8
        head_dim = 8
        # 创建tensor对象query，全为1，存储在CUDA设备上，数据类型为torch.float32，并需要梯度计算
        query = torch.ones(
            1, 8, seq_len_q, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        # 创建tensor对象key，全为1，存储在CUDA设备上，数据类型为torch.float32，并需要梯度计算
        key = torch.ones(
            1, 8, seq_len_k, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        # 创建tensor对象value，全为1，存储在CUDA设备上，数据类型为torch.float32，并需要梯度计算
        value = torch.ones(
            1, 8, seq_len_k, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )

        # 创建MyModule类的实例module
        module = MyModule()
        # 调用module的__call__方法，计算模型输出real
        real = module(query, key, value)
        # 获取real的设备信息
        real_device = real.device
        # 获取real的数据类型信息
        real_dtype = real.dtype

        # 对module应用torch._dynamo.optimize("inductor")优化
        opt_mod = torch._dynamo.optimize("inductor")(module)
        # 使用优化后的模块opt_mod计算compiled
        compiled = opt_mod(query, key, value)

        # 断言compiled的设备与real的设备相同
        self.assertEqual(compiled.device, real_device)
        # 断言compiled的数据类型与real的数据类型相同
        self.assertEqual(compiled.dtype, real_dtype)

        # 断言compiled的设备类型为"cuda"
        self.assertEqual(compiled.device.type, "cuda")
        # 断言compiled的设备索引为0
        self.assertEqual(compiled.device.index, 0)
        # 断言compiled的数据类型为torch.float32

    def test_autocast_cpu(self):
        # 定义一个继承自torch.nn.Module的子类MyModule，用于自定义神经网络模块
        class MyModule(torch.nn.Module):
            # 重写forward方法，处理输入并返回输出
            def forward(self, x):
                # 创建数据类型为torch.float32的随机矩阵a_float32，存储在CPU上
                a_float32 = torch.rand((8, 8), device="cpu")
                # 创建数据类型为torch.float32的随机矩阵b_float32，存储在CPU上
                b_float32 = torch.rand((8, 8), device="cpu")
                # 创建数据类型为torch.float32的随机矩阵d_float32，存储在CPU上
                d_float32 = torch.rand((8, 8), device="cpu")

                # 使用torch.autocast(device_type="cpu", dtype=torch.bfloat16)上下文，将以下代码块中的操作转换为CPU计算，并指定数据类型为torch.bfloat16
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    # 执行torch.mm矩阵乘法运算，将a_float32与b_float32相乘，结果存储在e_float16中
                    e_float16 = torch.mm(a_float32, b_float32)
                    # 执行torch.mm矩阵乘法运算，将d_float32与e_float16相乘，结果存储在f_float16中
                    f_float16 = torch.mm(d_float32, e_float16)
                # 返回计算结果f_float16
                return f_float16

        # 创建MyModule类的实例module
        module = MyModule()
        # 调用module的__call__方法，计算模型输出real
        real = module(torch.tensor([0.5]))
        # 获取real的设备信息
        real_device = real.device
        # 获取real的数据类型信息
        real_dtype = real.dtype

        # 对module应用torch._dynamo.export(module)导出，获取计算图和保护器
        graph, guards = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        # 使用导出的计算图graph，计算exported
        exported = graph(torch.tensor([0.5]))
        
        # 断言exported的设备与real的设备相同
        self.assertEqual(exported.device, real_device)
        # 断言exported的数据类型与real的数据类型相同
        self.assertEqual(exported.dtype, real_dtype)

        # 断言exported的设备类型为"cpu"
        self.assertEqual(exported.device.type, "cpu")
        # 断言exported的数据类型为torch.bfloat16
        self.assertEqual(exported.dtype, torch.bfloat16)
    def test_autocast_cpu_graph_break(self):
        # 定义一个继承自 torch.nn.Module 的子类 MyModule
        class MyModule(torch.nn.Module):
            # 重写 forward 方法
            def forward(self, x):
                # 创建一个 8x8 的随机浮点数张量 a_float32 在 CPU 上
                a_float32 = torch.rand((8, 8), device="cpu")
                # 创建一个 8x8 的随机浮点数张量 b_float32 在 CPU 上
                b_float32 = torch.rand((8, 8), device="cpu")
                # 调用 torch._dynamo.graph_break() 方法，可能用于打破计算图
                torch._dynamo.graph_break()
                # 创建一个 8x8 的随机浮点数张量 d_float32 在 CPU 上
                d_float32 = torch.rand((8, 8), device="cpu")

                # 进入 torch.autocast 上下文，指定设备类型为 CPU，数据类型为 torch.bfloat16
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    # 使用 torch.mm 计算矩阵乘积，并将结果转换为 torch.bfloat16 类型
                    e_float16 = torch.mm(a_float32, b_float32)
                    # 再次调用 torch._dynamo.graph_break() 方法，可能用于打破计算图
                    torch._dynamo.graph_break()
                    # 使用 torch.mm 计算矩阵乘积，并将结果转换为 torch.bfloat16 类型
                    f_float16 = torch.mm(d_float32, e_float16)
                # 返回最终结果张量 f_float16
                return f_float16

        # 创建 MyModule 类的实例 module
        module = MyModule()
        # 将输入张量 [0.5] 传入模型，获取模型输出结果 real
        real = module(torch.tensor([0.5]))
        # 获取 real 的设备类型
        real_device = real.device
        # 获取 real 的数据类型
        real_dtype = real.dtype

        # 对 module 应用 torch._dynamo.optimize("eager") 优化器
        opt = torch._dynamo.optimize("eager")(module)
        # 将输入张量 [0.5] 传入优化后的模型，获取优化后的输出结果 res
        res = opt(torch.tensor([0.5]))
        # 断言优化后的输出结果的设备类型与 real 相同
        self.assertEqual(res.device, real_device)
        # 断言优化后的输出结果的数据类型与 real 相同
        self.assertEqual(res.dtype, real_dtype)

        # 断言优化后的输出结果的设备类型为 CPU
        self.assertEqual(res.device.type, "cpu")
        # 断言优化后的输出结果的数据类型为 torch.bfloat16
        self.assertEqual(res.dtype, torch.bfloat16)

    def test_autocast_cpu_graph_break_2(self):
        # 回归测试：https://github.com/pytorch/pytorch/issues/93890
        # 定义一个函数 fn，接受一个输入张量 x
        def fn(x):
            # 进入 torch.autocast 上下文，指定设备类型为 CPU，数据类型为 torch.bfloat16
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                # 计算输入张量 x 的矩阵乘积，并将结果转换为 torch.bfloat16 类型
                x = torch.mm(x, x)
                # 调用 torch._dynamo.graph_break() 方法，可能用于打破计算图
                torch._dynamo.graph_break()
                # 对结果张量应用 relu 激活函数
                x = torch.relu(x)
            # 返回处理后的张量 x
            return x

        # 创建一个 4x4 的随机浮点数张量 x
        x = torch.rand([4, 4])
        # 断言输入张量 x 的数据类型为 torch.float32
        self.assertEqual(x.dtype, torch.float32)
        # 对函数 fn 应用输入张量 x，获取结果 res
        res = fn(x)
        # 对优化后的函数 fn 应用输入张量 x，获取优化后的结果 opt_res
        opt_fn = torch._dynamo.optimize("eager")(fn)
        opt_res = opt_fn(x)
        # 断言 res 和 opt_res 在数值上接近
        self.assertTrue(torch.allclose(res, opt_res))
        # 断言 res 的数据类型为 torch.bfloat16
        self.assertEqual(res.dtype, torch.bfloat16)
        # 断言 opt_res 的数据类型为 torch.bfloat16
        self.assertEqual(opt_res.dtype, torch.bfloat16)
    # 定义一个名为 test_autocast_cpu_graph_break_inner_fn 的测试方法
    def test_autocast_cpu_graph_break_inner_fn(self):
        # 定义一个继承自 torch.nn.Module 的子类 MyModule
        class MyModule(torch.nn.Module):
            # 静态方法 mm_breaks，用于执行 torch.mm 前的图中断操作
            @staticmethod
            def mm_breaks(x, y):
                # 调用私有函数 torch._dynamo.graph_break()，中断当前计算图
                torch._dynamo.graph_break()
                return torch.mm(x, y)

            # 实现 Module 的前向传播方法
            def forward(self, x):
                # 创建一个 8x8 的随机浮点数张量 a_float32 在 CPU 上
                a_float32 = torch.rand((8, 8), device="cpu")
                # 创建一个 8x8 的随机浮点数张量 b_float32 在 CPU 上
                b_float32 = torch.rand((8, 8), device="cpu")

                # 使用 torch.autocast 包装代码块，设置设备类型为 CPU，数据类型为 torch.bfloat16
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    # 调用私有函数 torch._dynamo.graph_break()，中断当前计算图
                    torch._dynamo.graph_break()
                    # 再次使用 torch.autocast 包装代码块，此时禁用自动类型转换
                    with torch.autocast(
                        device_type="cpu", dtype=torch.bfloat16, enabled=False
                    ):
                        # 调用私有函数 torch._dynamo.graph_break()，中断当前计算图
                        torch._dynamo.graph_break()
                        # 计算两个浮点数张量的矩阵乘积 g_float32
                        g_float32 = torch.mm(a_float32, b_float32)
                        # 再次使用 torch.autocast 包装代码块，恢复自动类型转换
                        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                            # 检查嵌套的非内联函数调用并进行图中断操作
                            torch._dynamo.graph_break()
                            # 调用 mm_breaks 方法进行矩阵乘积计算，返回结果 f_float16_1
                            f_float16_1 = self.mm_breaks(a_float32, b_float32)
                    # 即使在图中断后，也确保正确退出内部的 autocast 上下文
                    # 再次调用 mm_breaks 方法进行矩阵乘积计算，返回结果 f_float16
                    f_float16 = self.mm_breaks(a_float32, b_float32)
                    # 断言 f_float16 和 f_float16_1 的数据类型相同
                    assert f_float16.dtype == f_float16_1.dtype
                # 返回两个结果 f_float16 和 g_float32
                return f_float16, g_float32

        # 创建 MyModule 的实例 module
        module = MyModule()
        # 对模块进行输入为 [0.5] 的前向传播，获取结果 real_16 和 real_32
        real_16, real_32 = module(torch.tensor([0.5]))
        # 获取 real_16 的设备和数据类型
        real_device_16 = real_16.device
        real_dtype_16 = real_16.dtype
        # 获取 real_32 的设备和数据类型
        real_device_32 = real_32.device
        real_dtype_32 = real_32.dtype

        # 使用 torch._dynamo.optimize("eager") 对模块进行图优化
        graph = torch._dynamo.optimize("eager")(module)
        # 对优化后的模块进行输入为 [0.5] 的前向传播，获取结果 out_16 和 out_32
        out_16, out_32 = graph(torch.tensor([0.5]))
        
        # 使用断言检查 out_16 的设备类型和数据类型与 real_16 是否一致
        self.assertEqual(out_16.device, real_device_16)
        self.assertEqual(out_16.dtype, real_dtype_16)
        # 使用断言检查 out_32 的设备类型和数据类型与 real_32 是否一致
        self.assertEqual(out_32.device, real_device_32)
        self.assertEqual(out_32.dtype, real_dtype_32)

        # 使用断言检查 out_16 的设备类型为 CPU
        self.assertEqual(out_16.device.type, "cpu")
        # 使用断言检查 out_16 的数据类型为 torch.bfloat16
        self.assertEqual(out_16.dtype, torch.bfloat16)
        # 使用断言检查 out_32 的设备类型为 CPU
        self.assertEqual(out_32.device.type, "cpu")
        # 使用断言检查 out_32 的数据类型为 torch.float32
        self.assertEqual(out_32.dtype, torch.float32)
    def test_autocast_graph_break_method(self):
        # 定义一个名为 MyModule 的测试模块，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 模块初始化方法，接受一个 bias 参数
            def __init__(self, bias):
                super().__init__()
                self.bias = bias

            # 不会中断图优化的矩阵乘法方法
            def mm_not_break(self, x, y):
                return torch.mm(x, y) + self.bias

            # 会中断图优化的矩阵乘法方法
            def mm_breaks(self, x, y):
                # 调用内部函数中断图优化
                torch._dynamo.graph_break()
                return torch.mm(x, y) + self.bias

            # 前向传播方法
            def forward(self, x):
                # 创建一个随机的 8x8 浮点数张量 a_float32 和 b_float32
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")

                # 在 torch.autocast 的上下文中
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    # 在 torch.autocast(enabled=False) 的上下文中
                    with torch.autocast(
                        device_type="cpu", dtype=torch.bfloat16, enabled=False
                    ):
                        # 执行浮点数运算 torch.mm(a_float32, b_float32)
                        g_float32 = torch.mm(a_float32, b_float32)
                    # 调用会中断图优化的方法 self.mm_breaks(a_float32, b_float32)
                    f_float16 = self.mm_breaks(a_float32, b_float32)

                    # 断言两个浮点数相等
                    assert (
                        f_float16[0][0] == self.mm_not_break(a_float32, b_float32)[0][0]
                    )
                # 返回结果 f_float16 和 g_float32
                return f_float16, g_float32

        # 创建 MyModule 实例，传入随机生成的 8x8 bfloat16 类型的 bias
        module = MyModule(bias=torch.rand((8, 8), device="cpu", dtype=torch.bfloat16))

        # 在 torch.autocast 的上下文中执行浮点数加法运算
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            # 创建两个随机张量相加，要求第二个张量为 bfloat16 类型，因为 autocast 对加法不起作用
            res = torch.rand((8, 8), device="cpu", dtype=torch.float32) + torch.rand(
                (8, 8), device="cpu", dtype=torch.bfloat16
            )
            # 断言结果张量的数据类型为 float32
            self.assertEqual(res.dtype, torch.float32)

        # 调用 module 的 forward 方法，传入一个大小为 [0.5] 的张量
        real_16, real_32 = module(torch.tensor([0.5]))
        # 获取计算结果的设备和数据类型
        real_device_16 = real_16.device
        real_dtype_16 = real_16.dtype
        real_device_32 = real_32.device
        real_dtype_32 = real_32.dtype

        # 优化 MyModule 模块的图结构，并在优化后的图上执行计算
        graph = torch._dynamo.optimize("eager")(module)
        out_16, out_32 = graph(torch.tensor([0.5]))

        # 断言优化后的结果与原始结果的设备和数据类型相同
        self.assertEqual(out_16.device, real_device_16)
        self.assertEqual(out_16.dtype, real_dtype_16)
        self.assertEqual(out_32.device, real_device_32)
        self.assertEqual(out_32.dtype, real_dtype_32)

        # 断言优化后的结果的设备为 CPU，数据类型分别为 bfloat16 和 float32
        self.assertEqual(out_16.device.type, "cpu")
        self.assertEqual(out_16.dtype, torch.bfloat16)
        self.assertEqual(out_32.device.type, "cpu")
        self.assertEqual(out_32.dtype, torch.float32)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast_float64(self):
        # 定义一个继承自torch.nn.Module的子类MyModule，用于测试自动类型转换到float64的情况
        class MyModule(torch.nn.Module):
            def forward(self, x):
                # 在CUDA设备上生成随机的float32张量
                a_float32 = torch.rand((8, 8), device="cuda")
                b_float32 = torch.rand((8, 8), device="cuda")
                d_float32 = torch.rand((8, 8), device="cuda")

                # 使用autocast将下面的计算转换为float64类型
                with torch.autocast(device_type="cuda", dtype=torch.float64):
                    e_float64 = torch.mm(a_float32, b_float32)
                    f_float64 = torch.mm(d_float32, e_float64)
                return f_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        # 导出模块以进行图计算
        graph, guards = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))

        # 断言导出结果的设备和数据类型与预期相符
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        # 断言导出结果的设备索引为0，数据类型为torch.float64
        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.float64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast_device(self):
        # 定义一个继承自torch.nn.Module的子类MyModule，用于测试自动类型转换到float16的情况
        class MyModule(torch.nn.Module):
            def forward(self, x):
                # 在CUDA设备上生成随机的float32张量
                a_float32 = torch.rand((8, 8), device="cuda")
                b_float32 = torch.rand((8, 8), device="cuda")
                d_float32 = torch.rand((8, 8), device="cuda")

                # 使用autocast将下面的计算转换为float16类型
                with torch.autocast("cuda"):
                    e_float64 = torch.mm(a_float32, b_float32)
                    f_float64 = torch.mm(d_float32, e_float64)
                return f_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        # 导出模块以进行图计算
        graph, guards = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))

        # 断言导出结果的设备和数据类型与预期相符
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        # 断言导出结果的设备索引为0，数据类型为torch.float16
        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.float16)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast_arguments_binding(self):
        # 定义两个函数f1和f2，用于测试autocast在不同设备上的绑定情况
        def f1(x):
            with torch.cuda.amp.autocast(False):
                x = torch.sin(x + 1)
            return x

        def f2(x):
            with torch.cpu.amp.autocast(False):
                x = torch.cos(x + 1)
            return x

        x = torch.rand([2, 3])
        ref1 = f1(x)
        ref2 = f2(x)

        # 使用torch.compile将函数编译为eager模式下的优化版本
        opt_f1 = torch.compile(backend="eager")(f1)
        opt_f2 = torch.compile(backend="eager")(f2)
        res1 = opt_f1(x)
        res2 = opt_f2(x)

        # 断言优化版本与非优化版本的输出一致
        self.assertTrue(same(ref1, res1))
        self.assertTrue(same(ref2, res2))
    # 定义一个测试用例，测试自动类型转换装饰器的功能
    def test_autocast_decorator(self):
        # 定义一个自动类型转换函数装饰器，使用 CUDA 设备和 float16 数据类型
        def autocast_func(orig_func):
            @torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                return orig_func(*args, **kwargs)

            return new_fwd

        # 定义一个自动类型转换函数装饰器，仅使用 CUDA 设备和 float16 数据类型
        def autocast_func_cuda(orig_func):
            @torch.cuda.amp.autocast(dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                return orig_func(*args, **kwargs)

            return new_fwd

        # 定义一个自动类型转换函数装饰器，仅使用 CPU 和 float16 数据类型
        def autocast_func_cpu(orig_func):
            @torch.cpu.amp.autocast(dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                return orig_func(*args, **kwargs)

            return new_fwd

        # 定义一个矩阵乘法函数
        def mm(a, b):
            return torch.mm(a, b)

        # 使用不同的自动类型转换装饰器对矩阵乘法函数进行装饰
        mm_float16 = autocast_func(mm)
        mm_float16_cuda = autocast_func_cuda(mm)
        mm_float16_cpu = autocast_func_cpu(mm)

        # 定义一个函数，调用装饰后的矩阵乘法函数，分别使用不同的装饰器版本
        def fn(a, b):
            return mm_float16(a, b), mm_float16_cuda(a, b), mm_float16_cpu(a, b)

        # 创建两个随机初始化的 CUDA 设备上的 float32 类型矩阵
        a_float32 = torch.rand((8, 8), device="cuda")
        b_float32 = torch.rand((8, 8), device="cuda")

        # 获取原始函数的参考结果
        ref = fn(a_float32, b_float32)
        
        # 对函数进行优化编译，使用 eager 模式和完整图形模式
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        
        # 获取优化后函数的计算结果
        res = opt_fn(a_float32, b_float32)
        
        # 断言优化前后的结果相同
        self.assertTrue(same(ref, res))
        
        # 断言返回结果的数据类型为 float16
        self.assertTrue(res[0].dtype == torch.float16)
        self.assertTrue(res[1].dtype == torch.float16)

    # 定义一个测试通用上下文管理器的功能
    def test_generic_context_manager(self):
        # 定义一个函数，使用自定义的上下文管理器
        def fn(x):
            # 进入自定义上下文管理器
            with CutomizedCtxManager(True):
                x = x + 1
                # 如果梯度开启，则乘以2
                if torch.is_grad_enabled():
                    x = x * 2
                # 对 x 应用 relu 激活函数
                x = torch.relu(x)
            # 退出自定义上下文管理器后，返回结果减1
            return x - 1

        # 创建一个随机初始化的 2x3 大小的张量 x
        x = torch.rand(2, 3)
        
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        
        # 对函数进行优化编译，使用编译计数器和非完整图形模式
        opt_fn = torch.compile(backend=cnts, fullgraph=False)(fn)

        # 在没有梯度的情况下执行原始函数和优化后的函数
        with torch.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            # 断言优化前后的结果相同
            self.assertTrue(same(ref, res))
            # 断言编译帧数为2
            self.assertEqual(cnts.frame_count, 2)
            # 断言操作计数为2
            self.assertEqual(cnts.op_count, 2)

        # 在开启梯度的情况下执行原始函数和优化后的函数
        with torch.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            # 断言优化前后的结果相同
            self.assertTrue(same(ref, res))
            # 断言编译帧数为4
            self.assertEqual(cnts.frame_count, 4)
            # 断言操作计数为4
            self.assertEqual(cnts.op_count, 4)
    def test_nested_generic_context_manager(self):
        # 定义一个嵌套的泛型上下文管理器测试函数
        def fn(x):
            # 使用自定义的上下文管理器 CutomizedCtxManager，并设置为 True
            with CutomizedCtxManager(True):
                # 对输入 x 执行加一操作
                x = x + 1
                # 如果梯度开启，则将 x 值乘以 2
                if torch.is_grad_enabled():
                    x = x * 2
                # 使用另一个自定义的上下文管理器 CutomizedCtxManager，并设置为 False
                with CutomizedCtxManager(False):
                    # 如果梯度开启，则将 x 值减去 3
                    if torch.is_grad_enabled():
                        x = x - 3
                    # 对 x 值乘以 1.5
                    x = x * 1.5
                # 对 x 值进行 ReLU 激活函数操作
                x = torch.relu(x)
            # 返回处理后的 x 值减去 1
            return x - 1

        # 创建一个随机的 tensor 输入 x
        x = torch.rand(2, 3)
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用指定的后端和全图设置对函数 fn 进行编译优化
        opt_fn = torch.compile(backend=cnts, fullgraph=False)(fn)

        # 在没有梯度的上下文中进行测试
        with torch.no_grad():
            # 计算参考结果 ref 和优化后结果 res
            ref = fn(x)
            res = opt_fn(x)
            # 断言优化后的结果与参考结果相同
            self.assertTrue(same(ref, res))
            # 断言编译计数器的帧数为 4
            self.assertEqual(cnts.frame_count, 4)
            # 断言编译计数器的操作数为 4
            self.assertEqual(cnts.op_count, 4)

        # 在启用梯度的上下文中进行测试
        with torch.enable_grad():
            # 计算参考结果 ref 和优化后结果 res
            ref = fn(x)
            res = opt_fn(x)
            # 断言优化后的结果与参考结果相同
            self.assertTrue(same(ref, res))
            # 断言编译计数器的帧数为 6
            self.assertEqual(cnts.frame_count, 6)
            # 断言编译计数器的操作数为 6
            self.assertEqual(cnts.op_count, 6)

    def test_generic_context_manager_with_graph_break(self):
        # 定义一个带图断点的泛型上下文管理器测试函数
        def fn(x):
            # 使用自定义的上下文管理器 CutomizedCtxManager，并设置为 True
            with CutomizedCtxManager(True):
                # 对输入 x 执行加一操作
                x = x + 1
                # 如果梯度开启，则将 x 值乘以 2
                if torch.is_grad_enabled():
                    x = x * 2
                # 手动触发图断点
                torch._dynamo.graph_break()
                # 对 x 值进行 ReLU 激活函数操作
                x = torch.relu(x)
            # 返回处理后的 x 值减去 1
            return x - 1

        # 创建一个随机的 tensor 输入 x
        x = torch.rand(2, 3)
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用指定的后端和全图设置对函数 fn 进行编译优化
        opt_fn = torch.compile(backend=cnts, fullgraph=False)(fn)

        # 在没有梯度的上下文中进行测试
        with torch.no_grad():
            # 计算参考结果 ref 和优化后结果 res
            ref = fn(x)
            res = opt_fn(x)
            # 断言优化后的结果与参考结果相同
            self.assertTrue(same(ref, res))
            # 断言编译计数器的帧数为 2
            self.assertEqual(cnts.frame_count, 2)
            # 断言编译计数器的操作数为 2
            self.assertEqual(cnts.op_count, 2)

        # 在启用梯度的上下文中进行测试
        with torch.enable_grad():
            # 计算参考结果 ref 和优化后结果 res
            ref = fn(x)
            res = opt_fn(x)
            # 断言优化后的结果与参考结果相同
            self.assertTrue(same(ref, res))
            # 断言编译计数器的帧数为 4
            self.assertEqual(cnts.frame_count, 4)
            # 断言编译计数器的操作数为 4
            self.assertEqual(cnts.op_count, 4)
    # 定义测试函数：测试嵌套泛型上下文管理器与图中断
    def test_nested_generic_context_manager_with_graph_break(self):
        
        # 定义内部函数 fn，接受参数 x
        def fn(x):
            # 使用自定义上下文管理器 CutomizedCtxManager，并设置为 True
            with CutomizedCtxManager(True):
                # x 值加 1
                x = x + 1
                # 检查是否启用梯度计算
                if torch.is_grad_enabled():
                    # 如果是，则将 x 值乘以 2
                    x = x * 2
                # 使用自定义上下文管理器 CutomizedCtxManager，并设置为 False
                with CutomizedCtxManager(False):
                    # 再次检查是否启用梯度计算
                    if torch.is_grad_enabled():
                        # 如果是，则将 x 值减去 3
                        x = x - 3
                    # 手动中断动态图
                    torch._dynamo.graph_break()
                    # 将 x 值乘以 1.5
                    x = x * 1.5
                # 对 x 应用 ReLU 函数
                x = torch.relu(x)
            # 返回 x 减去 1 的结果
            return x - 1

        # 生成一个大小为 2x3 的随机张量 x
        x = torch.rand(2, 3)
        # 创建一个编译计数器 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch.compile 函数编译 fn 函数，设置后端为 cnts，禁用全图模式
        opt_fn = torch.compile(backend=cnts, fullgraph=False)(fn)

        # 在无梯度环境下执行以下操作
        with torch.no_grad():
            # 计算 fn(x) 的参考结果 ref
            ref = fn(x)
            # 计算编译后函数 opt_fn(x) 的结果 res
            res = opt_fn(x)
            # 断言 ref 和 res 相同
            self.assertTrue(same(ref, res))
            # 断言编译计数器的帧数为 4
            self.assertEqual(cnts.frame_count, 4)
            # 断言编译计数器的操作数为 4
            self.assertEqual(cnts.op_count, 4)

        # 重置动态图状态
        torch._dynamo.reset()
        # 重新创建编译计数器 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用动态图优化函数 torch._dynamo.optimize，设置 cnts，禁用即时编译
        opt_fn = torch._dynamo.optimize(cnts, nopython=False)(fn)

        # 在启用梯度环境下执行以下操作
        with torch.enable_grad():
            # 计算 fn(x) 的参考结果 ref
            ref = fn(x)
            # 计算优化后函数 opt_fn(x) 的结果 res
            res = opt_fn(x)
            # 断言 ref 和 res 相同
            self.assertTrue(same(ref, res))
            # 断言编译计数器的帧数为 4
            self.assertEqual(cnts.frame_count, 4)
            # 断言编译计数器的操作数为 4
            self.assertEqual(cnts.op_count, 4)

    # 定义测试函数：测试图中断内联梯度
    def test_graph_break_inlining_grad(self):
        
        # 定义内部函数 gn，接受参数 z
        def gn(z):
            # 在无梯度环境下执行以下操作
            with torch.no_grad():
                # 手动中断动态图
                torch._dynamo.graph_break()
                # 返回 z 的正弦值
                return torch.sin(z)

        # 定义内部函数 fn，接受参数 x, y, z
        def fn(x, y, z):
            # 计算 x 和 y 的矩阵乘积，赋值给 a
            a = torch.mm(x, y)
            # 调用 gn(z) 函数，并赋值给 z
            z = gn(z)
            # 返回 a
            return a

        # 重置动态图状态
        torch._dynamo.reset()
        # 创建一个编译计数器 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用动态图优化函数 torch._dynamo.optimize，设置 cnts，禁用即时编译
        opt_fn = torch._dynamo.optimize(cnts, nopython=False)(fn)
        # 创建大小为 4x4 的随机张量 x，y，和大小为 4 的随机张量 z，并启用梯度
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        z = torch.randn(4)
        # 对 opt_fn(x, y, z) 进行求和并反向传播梯度
        opt_fn(x, y, z).sum().backward()

        # 断言编译计数器的帧数为 2
        self.assertEqual(cnts.frame_count, 2)

    # 定义内部函数 _graph_break_inlining_autocast_test_helper，接受设备参数 device
    def _graph_break_inlining_autocast_test_helper(self, device):
        
        # 定义内部函数 gn，接受参数 x, y
        def gn(x, y):
            # 在自动混合精度环境下执行以下操作，设置设备类型和数据类型为 torch.bfloat16
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                # 计算 x 和 y 的矩阵乘积，赋值给 z
                z = torch.mm(x, y)
                # 手动中断动态图
                torch._dynamo.graph_break()
                # 返回 z 的正弦值
                return torch.sin(z)

        # 定义内部函数 fn，接受参数 x, y
        def fn(x, y):
            # 计算 x 和 y 的矩阵乘积，赋值给 z
            z = torch.mm(x, y)
            # z 值加上 gn(x, y) 的返回值，赋值给 z
            z = z + gn(x, y)
            # 返回 z
            return z

        # 创建大小为 3x3 的随机张量 x 和 y，并移到设备上
        x = torch.rand(3, 3).to(device)
        y = torch.rand(3, 3).to(device)
        # 使用 eager 模式编译 fn 函数，赋值给 opt_fn
        opt_fn = torch.compile(backend="eager")(fn)
        # 计算 fn(x, y) 的参考结果 ref
        ref = fn(x, y)
        # 计算优化后函数 opt_fn(x, y) 的结果 res
        res = opt_fn(x, y)
        # 断言 ref 等于 res
        self.assertEqual(ref, res)

    # 定义测试函数：测试图中断内联自动混合精度
    def test_graph_break_inlining_autocast(self):
        # 遍历设备列表 ["cuda", "cpu"]
        for device in ["cuda", "cpu"]:
            # 如果设备为 "cuda" 并且未安装或不支持 torch.bfloat16，则继续下一次循环
            if device == "cuda" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
                continue
            # 调用 _graph_break_inlining_autocast_test_helper 函数，传入设备参数 device
            self._graph_break_inlining_autocast_test_helper(device)
    def test_disable_saved_tensors_hooks(self):
        # 定义一个嵌套函数 fn，接受参数 z
        def fn(z):
            # 在函数 f 内部禁用保存张量的钩子，传入提示信息 "This is not supported"
            @torch.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            # 定义函数 f，接受参数 x 和 y，返回它们的和
            def f(x, y):
                return x + y

            # 创建张量 x 和 y，分别为全 1 和全 0 的张量
            x, y = torch.ones(
                1,
            ), torch.zeros(
                1,
            )
            # 调用函数 f，并传入参数 x 和 y，返回计算结果
            return f(x, y)

        # 创建 EagerAndRecordGraphs 类的实例 eager
        eager = EagerAndRecordGraphs()
        # 使用 torch.compile 编译函数 fn，指定后端为 eager，并启用完整图形记录
        torch.compile(fn, backend=eager, fullgraph=True)(torch.randn(()))

        # 获取记录的第一个图形
        graph = eager.graphs[0]
        # 对图形进行可读格式化打印，并标准化为规范格式
        actual = normalize_gm(graph.print_readable(False))

        # 使用 self.assertExpectedInline 进行断言比较实际结果和预期结果
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self):
        # 禁用保存张量钩子，传入消息 'This is not supported'
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported')

        # 创建一个包含单个元素值为 1 的张量 x，数据类型为 f32[1]
        x: "f32[1]" = torch.ones(1)

        # 创建一个包含单个元素值为 0 的张量 y，数据类型为 f32[1]
        y: "f32[1]" = torch.zeros(1)

        # 将张量 x 和 y 相加，结果保存在 add 中，然后将 x 和 y 置为 None
        add: "f32[1]" = x + y;  x = y = None

        # 启用保存张量钩子
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (add,)
""",
        )

    def test_disable_saved_tensors_hooks_prev_disabled(self):
        def fn(z):
            # 定义一个函数 fn，其中包含禁用保存张量钩子的装饰器，传入消息 'This is not supported'
            @torch.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            def f(x, y):
                return x + y

            # 创建张量 x 和 y，分别为值为 1 和 0 的张量，数据类型为 f32[1]
            x, y = torch.ones(
                1,
            ), torch.zeros(
                1,
            )
            return f(x, y)

        # 创建 EagerAndRecordGraphs 的实例 eager
        eager = EagerAndRecordGraphs()
        # 使用禁用保存张量钩子的上下文管理器，传入消息 'Previously disabled message'
        with torch.autograd.graph.disable_saved_tensors_hooks(
            "Previously disabled message"
        ):
            # 编译函数 fn，使用 eager 作为后端，并记录完整的计算图
            torch.compile(fn, backend=eager, fullgraph=True)(torch.randn(()))

        # 获取记录的计算图
        graph = eager.graphs[0]
        # 规范化计算图的可读输出
        actual = normalize_gm(graph.print_readable(False))

        # 断言实际输出与预期输出一致
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self):
        # 禁用保存张量钩子，传入消息 'This is not supported'
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported')

        # 创建一个包含单个元素值为 1 的张量 x，数据类型为 f32[1]
        x: "f32[1]" = torch.ones(1)

        # 创建一个包含单个元素值为 0 的张量 y，数据类型为 f32[1]
        y: "f32[1]" = torch.zeros(1)

        # 将张量 x 和 y 相加，结果保存在 add 中，然后将 x 和 y 置为 None
        add: "f32[1]" = x + y;  x = y = None

        # 禁用保存张量钩子，传入消息 'Previously disabled message'
        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable('Previously disabled message')
        return (add,)
""",
        )

    def test_disable_saved_tensors_hooks_prev_disabled_nested(self):
        def fn(z):
            # 定义一个函数 fn，其中包含禁用保存张量钩子的装饰器，传入消息 'This is not supported'
            @torch.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            def f(x, y):
                # 定义一个嵌套函数 inner_fn，其内部也包含禁用保存张量钩子的装饰器，传入消息 'This is not supported inner'
                @torch.autograd.graph.disable_saved_tensors_hooks(
                    "This is not supported inner"
                )
                def inner_fn(x, y):
                    return x + y

                # 调用 inner_fn 函数并将结果与 x 相加，返回结果
                return inner_fn(x, y) + x

            # 创建张量 x 和 y，分别为值为 1 和 0 的张量，数据类型为 f32[1]
            x, y = torch.ones(
                1,
            ), torch.zeros(
                1,
            )
            return f(x, y)

        # 创建 EagerAndRecordGraphs 的实例 eager
        eager = EagerAndRecordGraphs()
        # 使用禁用保存张量钩子的上下文管理器，传入消息 'Previously disabled message'
        with torch.autograd.graph.disable_saved_tensors_hooks(
            "Previously disabled message"
        ):
            # 编译函数 fn，使用 eager 作为后端，并记录完整的计算图
            torch.compile(fn, backend=eager, fullgraph=True)(torch.randn(()))

        # 获取记录的计算图
        graph = eager.graphs[0]
        # 规范化计算图的可读输出
        actual = normalize_gm(graph.print_readable(False))

        # 断言实际输出与预期输出一致
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    # 在 Torch 自动求导系统中禁用张量钩子的保存，传入消息字符串作为参数
    _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported')

    # 创建一个值为1的张量 x，类型为 f32[1]
    x: "f32[1]" = torch.ones(1)

    # 创建一个值为0的张量 y，类型为 f32[1]
    y: "f32[1]" = torch.zeros(1)

    # 在 Torch 自动求导系统中再次禁用张量钩子的保存，传入内部消息字符串作为参数
    _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable('This is not supported inner')

    # 计算 x + y 的结果，并将 y 设为 None
    add: "f32[1]" = x + y;  y = None

    # 在 Torch 自动求导系统中再次禁用张量钩子的保存，传入消息字符串作为参数
    _saved_tensors_hooks_disable_2 = torch._C._autograd._saved_tensors_hooks_disable('This is not supported')

    # 计算 add + x 的结果，并将 add 和 x 设为 None
    add_1: "f32[1]" = add + x;  add = x = None

    # 在 Torch 自动求导系统中再次禁用张量钩子的保存，传入先前禁用的消息字符串作为参数
    _saved_tensors_hooks_disable_3 = torch._C._autograd._saved_tensors_hooks_disable('Previously disabled message')
    # 返回包含 add_1 的元组，作为输出
    return (add_1,)
    def test_disable_saved_tensors_hooks_graph_break(self):
        # 定义一个函数 fn，接受一个参数 x
        def fn(x):
            # 在图形中禁用保存的张量钩子和钩子的挂钩断点
            with torch.autograd.graph.disable_saved_tensors_hooks(
                "This is not supported"
            ):
                # 计算 y = x + 1
                y = x + 1
                # 在计算图中创建一个断点
                torch._dynamo.graph_break()
                # 返回 y * 2
                return y * 2

        # 创建 EagerAndRecordGraphs 的实例 eager
        eager = EagerAndRecordGraphs()
        # 编译函数 fn，使用 eager 作为后端，不使用完整的图形
        torch.compile(fn, backend=eager, fullgraph=False)(torch.randn(()))

        # 定义函数 check_graph，用于验证图形是否符合预期
        def check_graph(actual, expected):
            self.assertExpectedInline(actual, expected)

        # 获取第一个图形的信息
        graph = eager.graphs[0]
        # 标准化打印可读的图形信息
        actual = normalize_gm(graph.print_readable(False))
        # 断言实际输出与预期输出相符
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported')

        y: "f32[]" = l_x_ + 1;  l_x_ = None

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (y,)
""",
        )

        # 获取第二个图形的信息
        graph = eager.graphs[1]
        # 标准化打印可读的图形信息
        actual = normalize_gm(graph.print_readable(False))
        # 断言实际输出与预期输出相符
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_y_: "f32[]"):
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported')

        mul: "f32[]" = l_y_ * 2;  l_y_ = None

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (mul,)
""",
        )

    def test_context_wrapping_grad_mode_decorator(self):
        # 定义上下文包装器列表，每个元素包含一个包装器和其对应的模式
        ctx_wrappers = [(torch.enable_grad, True), (torch.no_grad, False)]
        # 循环遍历调用和不调用的情况
        for call in [True, False]:
            for i in range(2):
                # 重置动态图状态
                torch._dynamo.reset()

                # 获取当前上下文包装器和模式
                ctx_wrapper, mode = ctx_wrappers[i]
                # 获取相反的上下文包装器和模式
                ctx_wrapper_inverse, mode_inverse = ctx_wrappers[(i + 1) % 2]

                # 定义函数 fn，接受一个参数 x
                def fn(x):
                    # 定义内部函数 inner_func，返回 x 的正弦值
                    def inner_func(x):
                        return x.sin()

                    # 在相反的上下文中禁用包装器
                    with ctx_wrapper_inverse():
                        # 根据调用变量选择是否使用上下文包装器
                        if call:
                            inner_func = ctx_wrapper()(inner_func)
                        else:
                            inner_func = ctx_wrapper(inner_func)

                        # 断言梯度是否启用与预期模式相符
                        assert torch.is_grad_enabled() == mode_inverse

                    # 再次在相反的上下文中禁用包装器并返回 inner_func(x)
                    with ctx_wrapper_inverse():
                        return inner_func(x)

                # 创建一个需要梯度的 10 元素零张量 x
                x = torch.zeros(10, requires_grad=True)
                # 编译函数 fn，使用 eager 作为后端，并使用完整的图形
                opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
                # 断言 fn(x) 和 opt_fn(x) 的输出相等
                self.assertEqual(fn(x), opt_fn(x))
                # 断言 opt_fn(x) 是否需要梯度与 fn(x) 是否需要梯度相符
                self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
    # 定义一个测试函数，用于测试上下文包装和嵌套函数装饰器的效果
    def test_context_wrapping_grad_mode_nested_function_decorator(self):
        # 定义上下文包装器和期望的梯度开启状态对
        ctx_wrappers = [(torch.enable_grad, True), (torch.no_grad, False)]

        # 遍历两种调用方式和两种上下文包装器的组合
        for call in [True, False]:
            for i in range(2):
                # 重置 torch._dynamo 的状态
                torch._dynamo.reset()

                # 获取当前循环的上下文包装器及其对应的期望模式
                ctx_wrapper, mode = ctx_wrappers[i]
                ctx_wrapper_inverse, mode_inverse = ctx_wrappers[(i + 1) % 2]

                # 定义一个内部函数 fn，接受一个参数 x
                def fn(x):
                    # 使用相反的上下文包装器包装
                    with ctx_wrapper_inverse():
                        # 根据 call 的值选择不同的装饰器方式
                        if call:
                            # 使用装饰器 @ctx_wrapper() 定义内部函数 inner_func
                            @ctx_wrapper()
                            def inner_func(x):
                                return x.sin()

                        else:
                            # 使用装饰器 @ctx_wrapper 定义内部函数 inner_func
                            @ctx_wrapper
                            def inner_func(x):
                                return x.sin()

                        # 断言调用 no_grad 或 enable_grad 不会改变全局状态
                        assert torch.is_grad_enabled() == mode_inverse

                    # 再次使用相反的上下文包装器包装
                    with ctx_wrapper_inverse():
                        # 返回调用内部函数 inner_func 的结果
                        return inner_func(x)

                # 创建一个 requires_grad=True 的张量 x
                x = torch.zeros(10, requires_grad=True)
                # 编译函数 fn，使用 eager 模式和完整图形优化
                opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
                # 断言调用原始函数 fn 和优化后的函数 opt_fn 返回值相同
                self.assertEqual(fn(x), opt_fn(x))
                # 断言调用原始函数 fn 和优化后的函数 opt_fn 的 requires_grad 属性相同
                self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    # 定义测试函数，测试使用 set_grad_enabled 装饰内部函数的效果
    def test_context_wrapping_set_grad_enabled_nested_function(self):
        # 定义两种模式：开启和关闭梯度
        modes = [True, False]
        
        # 遍历两种装饰器方式和两种模式的组合
        for decorator in [True, False]:
            for i in range(2):
                # 重置 torch._dynamo 的状态
                torch._dynamo.reset()

                # 获取当前循环的模式和相反模式
                mode = modes[i]
                mode_inverse = modes[(i + 1) % 2]

                # 定义一个内部函数 fn，接受一个参数 x
                def fn(x):
                    # 使用 torch.set_grad_enabled(mode_inverse) 包装上下文
                    with torch.set_grad_enabled(mode_inverse):
                        # 根据 decorator 的值选择不同的装饰方式
                        if decorator:
                            # 使用装饰器 @torch.set_grad_enabled(mode) 定义内部函数 inner_func
                            @torch.set_grad_enabled(mode)
                            def inner_func(x):
                                return x.sin()

                        else:
                            # 定义内部函数 inner_func
                            def inner_func(x):
                                return x.sin()

                            # 使用 torch.set_grad_enabled(mode) 装饰 inner_func
                            inner_func = torch.set_grad_enabled(mode)(inner_func)

                        # 断言调用 set_grad_enabled 在函数上不会改变全局状态
                        assert torch.is_grad_enabled() == mode_inverse

                    # 再次使用 torch.set_grad_enabled(mode_inverse) 包装上下文
                    with torch.set_grad_enabled(mode_inverse):
                        # 返回调用内部函数 inner_func 的结果
                        return inner_func(x)

                # 创建一个 requires_grad=True 的张量 x
                x = torch.zeros(10, requires_grad=True)
                # 编译函数 fn，使用 eager 模式和完整图形优化
                opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
                # 断言调用原始函数 fn 和优化后的函数 opt_fn 返回值相同
                self.assertEqual(fn(x), opt_fn(x))
                # 断言调用原始函数 fn 和优化后的函数 opt_fn 的 requires_grad 属性相同
                self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
    # 测试：在未激活的上下文中断下的图形中断（本地）
    def test_inactive_context_graph_break_local(self):
        def fn(x):
            x = x + 1
            # 设置梯度跟踪为启用状态
            ctx = torch.set_grad_enabled(True)
            # 执行图形中断操作
            torch._dynamo.graph_break()
            # 在上下文中使用设置的上下文管理器
            with ctx:
                x = x + 1
            return x

        # 创建一个张量 x，不需要梯度
        x = torch.zeros(10, requires_grad=False)
        # 创建编译计数器
        cnts = torch._dynamo.testing.CompileCounter()
        # 编译函数 fn，并传入编译后的函数 opt_fn
        opt_fn = torch.compile(fn, backend=cnts)
        # 断言编译前后函数的输出结果相等
        self.assertEqual(fn(x), opt_fn(x))
        # 断言编译前后函数的梯度跟踪属性相等
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        # 断言编译计数器的帧数为 2
        self.assertEqual(cnts.frame_count, 2)

    # 测试：在未激活的上下文中断下的图形中断（本地，使用 nullcontext）
    def test_inactive_context_graph_break_local_nullctx(self):
        import contextlib

        # 使用上下文管理器 nullcontext 进行测试，结果应为 None
        def fn(x):
            x = x + 1
            ctx = contextlib.nullcontext()
            # 执行图形中断操作
            torch._dynamo.graph_break()
            # 在上下文中使用设置的上下文管理器
            with ctx:
                x = x + 1
            return x

        # 创建一个张量 x，不需要梯度
        x = torch.zeros(10, requires_grad=False)
        # 创建编译计数器
        cnts = torch._dynamo.testing.CompileCounter()
        # 编译函数 fn，并传入编译后的函数 opt_fn
        opt_fn = torch.compile(fn, backend=cnts)
        # 断言编译前后函数的输出结果相等
        self.assertEqual(fn(x), opt_fn(x))
        # 断言编译前后函数的梯度跟踪属性相等
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        # 断言编译计数器的帧数为 2
        self.assertEqual(cnts.frame_count, 2)

    # 测试：在未激活的上下文中断下的图形中断（本地，使用 nullcontext 和图形中断）
    def test_inactive_context_graph_break_local_nullctx2(self):
        import contextlib

        # 在内联函数中执行图形中断并返回列表
        def gn():
            torch._dynamo.graph_break()
            return [0, 1, 2]

        # 使用上下文管理器 nullcontext 进行测试
        def fn(x):
            x = x + 1
            ctx = contextlib.nullcontext()
            # 调用函数 gn 执行图形中断
            lst = gn()
            # 在上下文中使用设置的上下文管理器
            with ctx:
                x = x + lst[1]
            return x

        # 创建一个张量 x，不需要梯度
        x = torch.zeros(10, requires_grad=False)
        # 创建编译计数器
        cnts = torch._dynamo.testing.CompileCounter()
        # 编译函数 fn，并传入编译后的函数 opt_fn
        opt_fn = torch.compile(fn, backend=cnts)
        # 断言编译前后函数的输出结果相等
        self.assertEqual(fn(x), opt_fn(x))
        # 断言编译前后函数的梯度跟踪属性相等
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        # 断言编译计数器的帧数为 2
        self.assertEqual(cnts.frame_count, 2)

    # 测试：在未激活的上下文中断下的图形中断（堆栈上的情况）
    def test_inactive_context_graph_break_stack(self):
        # 在函数中执行图形中断并返回上下文
        def gn(ctx):
            torch._dynamo.graph_break()
            return ctx

        # 使用上下文管理器设置梯度跟踪为启用状态
        def fn(x):
            x = x + 1
            ctx = gn(torch.set_grad_enabled(True))
            # 预期下一行会有图形中断
            with ctx:
                x = x + 1
            return x

        # 创建一个张量 x，不需要梯度
        x = torch.zeros(10, requires_grad=False)
        # 创建编译计数器
        cnts = torch._dynamo.testing.CompileCounter()
        # 编译函数 fn，并传入编译后的函数 opt_fn
        opt_fn = torch.compile(fn, backend=cnts)
        # 断言编译前后函数的输出结果相等
        self.assertEqual(fn(x), opt_fn(x))
        # 断言编译前后函数的梯度跟踪属性相等
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
    # 定义一个测试方法，用于测试非活动上下文图形中断堆栈的情况
    def test_inactive_context_graph_break_stack2(self):
        # 定义一个内部函数 gn，接受参数 x、ctx、y、z、dummy
        def gn(x, ctx, y, z, dummy):
            # 使用给定的上下文 ctx 进入上下文管理器
            with ctx:
                # 返回 x * y * z 的计算结果
                return x * y * z

        # 定义一个内部函数 fn，接受参数 x
        def fn(x):
            # 将 x 增加 1
            x = x + 1
            # 调用 gn 函数，传递参数 x、torch.set_grad_enabled(True)、2、3、torch._dynamo.graph_break()
            # 并将结果赋值给 x
            x = gn(x, torch.set_grad_enabled(True), 2, 3, torch._dynamo.graph_break())
            # 返回经过处理后的 x
            return x

        # 创建一个张量 x，形状为 (10,)，并且不需要梯度计算
        x = torch.zeros(10, requires_grad=False)
        # 创建一个编译计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch.compile 编译函数 fn，并传入编译后的函数及其后端
        opt_fn = torch.compile(fn, backend=cnts)
        # 断言调用 fn(x) 和 opt_fn(x) 返回的结果相等
        self.assertEqual(fn(x), opt_fn(x))
        # 断言调用 fn(x) 和 opt_fn(x) 返回的结果是否具有相同的 requires_grad 属性
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        # 断言编译计数器 cnts 的帧计数是否为 2
        self.assertEqual(cnts.frame_count, 2)
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行以下操作
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块中导入 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行导入的 run_tests 函数，用于执行测试用例
    run_tests()
```