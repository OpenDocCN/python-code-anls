# `.\pytorch\test\dynamo\test_unspec.py`

```py
# 导入标准数学库和随机库
import math
import random
# 导入 Python 的单元测试模块
import unittest

# 导入第三方库 numpy，并简称为 np
import numpy as np

# 导入 PyTorch 深度学习框架
import torch
# 导入 PyTorch 的动态图测试相关模块
import torch._dynamo.test_case
import torch._dynamo.testing
# 导入 PyTorch 中的函数操作模块
import torch.nn.functional as F

# 导入 PyTorch 的编译时工具 comptime
from torch._dynamo.comptime import comptime
# 导入 PyTorch 测试相关工具 CompileCounter 和 same
from torch._dynamo.testing import CompileCounter, same
# 导入 PyTorch 内部测试日志工具 logs_to_string
from torch.testing._internal.logging_utils import logs_to_string


# 这个测试文件的意图是专门为 assume_static_by_default=False 的情况编写测试用例，
# 即希望尽可能地将所有内容都设置为动态的。如果要测试更常规的情况，即默认情况下假设为静态，
# 则应将测试放在普通的测试文件中，并由 test_dynamic_shapes 处理两种情况（YOLO 和非 YOLO）。

# 使用装饰器指定 assume_static_by_default=False 的配置
@torch._dynamo.config.patch(assume_static_by_default=False)
class UnspecTests(torch._dynamo.test_case.TestCase):
    # 测试函数，验证 numpy 的正确性
    def test_numpy_correctness(self):
        def fn(x, y, z):
            # 创建一个包含 x+y、y 和 False 的列表
            xy = [x + y, y, False]
            # 将 x 转换为 numpy 数组
            np_x = x.numpy()
            # 将 y 转换为 numpy 数组
            np_y = y.numpy()
            # 返回一个包含各种计算结果的字典
            return {
                "x": x,
                "z": z,
                "a": np_y.sum(),
                "b": xy,
                "c": np_y[0][0] / 68,
                "d": np_x.sum(),
                "e": np_x + np_y,
            }, x + np_y.sum() + z

        # 创建一个浮点数类型的 torch 张量 x
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        # 创建一个整数类型的 torch 张量 y
        y = torch.ones([2, 2], dtype=torch.int64)
        # 创建一个 numpy 的 int64 类型变量 z
        z = np.int64(12)
        # 调用函数 fn 并获取返回结果 res1
        res1 = fn(x, y, z)
        # 创建一个编译计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 通过 optimize 装饰器优化 fn 函数
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 调用优化后的函数 opt_fn 并获取返回结果 res2
        res2 = opt_fn(x, y, z)
        # 断言 res1 和 res2 相等
        self.assertEqual(res1, res2)

    # 测试函数，验证没有重新编译
    def test_no_recompilations(self):
        # 如果传递不同的 numpy int 值，则不会重新编译
        def fn(x, y):
            return {"a": x + 1, "b": y / 2}

        # 创建一个浮点数类型的 torch 张量 x
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        # 创建一个编译计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 通过 optimize 装饰器优化 fn 函数
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 多次调用优化后的函数 opt_fn
        for i in range(10):
            opt_fn(x, np.int64(i))
        # 断言编译帧数为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言操作计数为 2
        self.assertEqual(cnts.op_count, 2)

    # 预期测试失败的函数，标注为预期失败，因为数组标量会衰减为 0 维数组
    @unittest.expectedFailure  # array scalars decay to 0D arrays
    def test_builtin_max_min(self):
        # 测试未专门化的原始 max/min 函数
        def fn(x, y, z):
            return z + 1, max(x, y), min(x - 4, y)

        # 创建一个 numpy 的 int64 类型变量 x
        x = np.int64(12)
        # 创建一个整数类型的变量 y
        y = 10
        # 创建一个浮点数类型的 torch 张量 z
        z = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        # 调用函数 fn 并获取返回结果 res1
        res1 = fn(x, y, z)
        # 创建一个编译计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 通过 optimize 装饰器优化 fn 函数
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 调用优化后的函数 opt_fn 并获取返回结果 res2
        res2 = opt_fn(x, y, z)
        # 使用 relax_numpy_equality=True 来宽松比较 numpy 数组的相等性
        self.assertTrue(same(res1, res2, relax_numpy_equality=True))
    # 定义测试函数：向图中仅输入随机值
    def test_feed_random_values_into_graph_only(self):
        # 定义内部函数fn，生成指定形状的随机张量
        def fn(shape):
            # 设置随机种子为123
            torch.manual_seed(123)
            # 生成符合正态分布的张量，并乘以随机整数（30到100之间）
            x = torch.randn(shape, device="cpu") * random.randint(30, 100)
            return x

        # 设置形状为[2, 3]
        shape = [2, 3]
        # 设置随机种子为1
        random.seed(1)
        # 调用fn函数生成随机张量res1
        res1 = fn(shape)
        # 创建CompileCounter对象cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化fn函数，返回优化后的函数opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 重新设置随机种子为1
        random.seed(1)
        # 使用优化后的函数opt_fn生成随机张量res2
        res2 = opt_fn(shape)

        # 断言res1与res2是否相同
        self.assertTrue(same(res1, res2))

    # 定义测试函数：包含图中断开的随机值
    def test_random_values_with_graph_break(self):
        # 定义内部函数fn，接受输入x，并对其进行随机处理
        def fn(x):
            # 生成0到1之间的随机数r1
            r1 = random.random()
            # x增加一个范围在10到20之间的随机浮点数，并将结果赋给y
            y = x + random.uniform(10, 20)
            # 计算y的和，不会对图形输出有影响
            y.sum().item()
            # 生成2到18之间的随机整数r2，无图形输出
            r2 = random.randint(2, 18)
            # 再次计算y的和，不会对图形输出有影响
            y.sum().item()
            # 返回y增加r1后的结果，以及r2
            return y + r1, r2

        # 创建张量x
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        # 设置随机种子为1
        random.seed(1)
        # 调用fn函数生成结果res1
        res1 = fn(x)
        # 创建CompileCounter对象cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化fn函数，返回优化后的函数opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 重新设置随机种子为1
        random.seed(1)
        # 使用优化后的函数opt_fn生成结果res2
        res2 = opt_fn(x)
        # 断言res1与res2是否相同
        self.assertTrue(same(res1, res2))

    # 真的很烦的专业化与RandomValueSource的交集
    # 如果我们得到一个RandomValueSource和一个单元素张量，我们应该像其他未指定一样返回一个ConstantVariable……
    # 但如果我们这样做，我们会破坏字节码假设和防护将不起作用，因为我们将引用一个不存在的源名称。
    # 如果我们调用.item()并取出wrapped_value，其中我们发送unspec到wrap_fx_proxy，在此测试通过，然后一些模型会因缺少codegen.tx.output.random_values_var而失败。
    # 如果我们将张量值直接传递给wrap，此测试将失败。
    # 真正的解决方案是从头开始重写RandomValueSource和它所做的所有codegen。
    # 定义测试函数：在图形生成之前进行多次连续的随机调用
    def test_multiple_consecutive_random_calls_before_graph(self):
        # 定义内部函数fn，接受输入x，并对其进行多次随机处理
        def fn(x):
            # 生成0到5之间的随机整数dim1、dim2、dim3
            dim1 = random.randrange(start=0, stop=5)
            dim2 = random.randrange(start=0, stop=5)
            dim3 = random.randrange(start=0, stop=5)
            # 生成指定形状的随机张量y
            y = torch.rand(dim1, dim2, dim3)
            # 返回x增加2后的结果，以及生成的随机张量y
            return x + 2, y

        # 创建张量x
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        # 设置随机种子为1
        random.seed(1)
        # 调用fn函数生成结果res1
        res1 = fn(x)
        # 创建CompileCounter对象cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化fn函数，返回优化后的函数opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 重新设置随机种子为1
        random.seed(1)
        # 使用优化后的函数opt_fn生成结果res2
        res2 = opt_fn(x)
        # 断言res1与res2是否相同
        self.assertTrue(same(res1, res2))

    # 定义测试函数：编译后的随机调用应为随机的
    def test_compiled_random_calls_are_random(self):
        # 对于具有随机调用的编译函数，
        # 每次迭代应返回不同的值。
        # https://github.com/pytorch/pytorch/issues/95425
        # 定义编译函数fn，接受输入x，并返回(x+1)乘以0到1之间的随机数
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return (x + 1) * random.uniform(0, 1)

        # 生成5个结果并存储在列表res中
        res = []
        for _ in range(5):
            res.append(fn(torch.ones(2)))
        # 验证每次迭代的结果是否不相同
        for i in range(1, 5):
            self.assertFalse(same(res[i - 1], res[i]))
    def test_random_call_with_while_loop(self):
        def fn(x):
            # 随机生成一个范围在0到3之间的整数作为dim1
            dim1 = random.randrange(start=0, stop=3)
            # 将dim2初始化为与dim1相等的随机整数
            dim2 = dim1
            # 当dim1等于dim2时，重新生成dim2，直到dim1不等于dim2为止
            while dim1 == dim2:
                dim2 = random.randrange(start=0, stop=3)
            # 返回x乘以2的结果
            return x * 2

        # 生成一个包含4个随机数的张量x
        x = torch.randn(4)
        # 设置随机种子为1
        random.seed(1)
        # 使用fn函数计算结果res1
        res1 = fn(x)
        # 通过torch._dynamo.optimize("eager")优化fn函数，赋值给opt_fn
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 重新设置随机种子为1
        random.seed(1)
        # 使用优化后的函数opt_fn计算结果res2
        res2 = opt_fn(x)
        # 断言res1与res2相同
        self.assertTrue(same(res1, res2))

        # 设置随机种子为10
        random.seed(10)
        # 使用fn函数计算结果res1
        res1 = fn(x)
        # 重新设置随机种子为10
        random.seed(10)
        # 使用优化后的函数opt_fn计算结果res2
        res2 = opt_fn(x)
        # 断言res1与res2相同
        self.assertTrue(same(res1, res2))

    def test_builtin_getitem(self):
        # 定义一个函数fn，接受一个列表x和一个索引idx作为参数
        # 返回一个元组，包括torch.zeros(idx)，x[idx]，x[idx:]
        def fn(x, idx):
            return (torch.zeros(idx), x[idx], x[idx:])

        # 创建一个包含数字0到49的列表x
        x = list(range(50))
        # 使用fn函数计算参考值ref，其中idx为48（未特定化）
        ref = fn(x, 48)
        # 创建一个编译计数器对象cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 通过torch._dynamo.optimize(cnts)优化fn函数，赋值给opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 使用优化后的函数opt_fn计算结果res，其中idx为48
        res = opt_fn(x, 48)
        # 断言参考值ref与结果res相同
        self.assertTrue(same(ref, res))

    def test_use_and_specialize(self):
        # 创建一个编译计数器对象cnt
        cnt = CompileCounter()

        # 定义一个使用cnt作为后端、完整图和动态模式的编译函数fn
        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(x, y):
            # 计算x加y的结果
            x = x + y
            # 如果y等于2，则返回x减1，否则返回x加1
            if y == 2:
                return x - 1
            else:
                return x + 1

        # 断言使用fn函数计算torch.tensor([5])和2得到的结果为6
        self.assertTrue(same(fn(torch.tensor([5]), 2), 6))
        # 断言使用fn函数计算torch.tensor([6])和2得到的结果为7
        self.assertTrue(same(fn(torch.tensor([6]), 2), 7))
        # 断言使用fn函数计算torch.tensor([5])和3得到的结果为9
        self.assertTrue(same(fn(torch.tensor([5]), 3), 9))
        # 断言使用fn函数计算torch.tensor([4])和3得到的结果为8
        self.assertTrue(same(fn(torch.tensor([4]), 3), 8))
        # 断言编译帧数为2
        self.assertEqual(cnt.frame_count, 2)

    def test_no_recompiles(self):
        # 创建一个编译计数器对象cnt
        cnt = CompileCounter()

        # 定义一个使用cnt作为后端、完整图和动态模式的编译函数fn
        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(x, y):
            # 返回x加y的结果
            return x + y

        # 断言使用fn函数计算torch.tensor([5])和100得到的结果为105
        self.assertTrue(same(fn(torch.tensor([5]), 100), 105))
        # 断言使用fn函数计算torch.tensor([4])和200得到的结果为204
        self.assertTrue(same(fn(torch.tensor([4]), 200), 204))
        # 断言使用fn函数计算torch.tensor([3])和300得到的结果为303
        self.assertTrue(same(fn(torch.tensor([3]), 300), 303))
        # 断言使用fn函数计算torch.tensor([2])和400得到的结果为402
        self.assertTrue(same(fn(torch.tensor([2]), 400), 402))
        # 断言编译帧数为1
        self.assertEqual(cnt.frame_count, 1)
        # 断言操作计数为1
        self.assertEqual(cnt.op_count, 1)

    def test_no_recompiles_prod_backward(self):
        # 创建一个编译计数器对象cnt
        cnt = CompileCounter()

        # 定义一个使用cnt作为后端、完整图和动态模式的编译函数fn
        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(t):
            # 计算张量t在第三维上的元素乘积，并保持维度不变
            return torch.prod(t, 3, keepdim=True)

        # 定义多种输入形状
        input_shapes = [(8, 10, 3, 2), (8, 3, 5, 2), (8, 4, 8, 2)]
        # 遍历输入形状
        for s in input_shapes:
            # 创建随机张量t1，并启用梯度计算
            t1 = torch.randn(s, requires_grad=True)
            # 计算fn函数应用于t1的结果h_result
            h_result = fn(t1)
            # 创建梯度张量，全部为1
            grad = torch.ones_like(h_result)
            # 对h_result进行反向传播
            h_result.backward(grad)

        # 断言编译帧数为1
        self.assertEqual(cnt.frame_count, 1)
        # 断言操作计数为1
        self.assertEqual(cnt.op_count, 1)
    def test_builtin_functions_on_cuda(self):
        def fn(x, scaler):
            # 创建一个 ReLU 激活函数对象
            m = torch.nn.ReLU()
            # 对输入张量 x 应用 ReLU 激活函数，并乘以 scaler
            y = m(x) * scaler
            return y

        # 在 CUDA 设备上生成一个形状为 [3, 6] 的随机张量 x
        x = torch.randn([3, 6], device="cuda")
        scaler = 0.23  # 设置 scaler 为 0.23
        ref = fn(x, scaler)
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 对函数 fn 进行优化，并得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 使用优化后的函数 opt_fn 处理输入 x 和 scaler，并得到结果 res
        res = opt_fn(x, scaler)
        # 断言优化前后的结果相同
        self.assertTrue(same(ref, res))
        # 断言优化后的结果与原始输入张量 x 的设备相同
        self.assertEqual(ref.device, res.device)

    def test_unspec_float_precision(self):
        def fn(image, scale_factor):
            # 使用双线性插值对图像进行尺寸调整，并获取调整后的形状
            image = torch.nn.functional.interpolate(
                image[None],
                size=None,
                scale_factor=scale_factor,
                mode="bilinear",
                recompute_scale_factor=True,
                align_corners=False,
            )[0]

            return image.shape

        # 生成一个形状为 [3, 427, 640] 的随机张量 x
        x = torch.rand([3, 427, 640])
        scale_factor = 1.873536229133606  # 设置 scale_factor
        ref = fn(x, scale_factor)
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 对函数 fn 进行优化，并得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 使用优化后的函数 opt_fn 处理输入 x 和 scale_factor，并得到结果 res
        res = opt_fn(x, scale_factor)
        # 断言优化前后的结果相同
        self.assertTrue(same(ref, res))

    @unittest.expectedFailure  # 如果 numpy 标量是 0D 数组，则失败
    def test_specializing_numpy_float_in_control_flow(self):
        # np.float64 默认情况下是未专门化的，
        # 但在控制流中使用时应该是专门化的。
        def fn(x, y):
            if y > 1.0:
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        # 对函数 fn 进行优化，并得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 针对不同的 numpy 类型（np.float16, np.float32, np.float64）进行测试
        for t in [np.float16, np.float32, np.float64]:
            y = t(1.23)
            ref = fn(x, y)
            # 使用优化后的函数 opt_fn 处理输入 x 和 y，并得到结果 res
            res = opt_fn(x, y)
            # 断言优化前后的结果相同
            self.assertTrue(same(ref, res))

    def test_mark_static_inside(self):
        def fn(x):
            # 在 x 上标记为静态
            torch._dynamo.mark_static(x, 0)
            # 断言 x.size(0) 在编译时是静态的
            comptime.assert_static(x.size(0))
            # 返回 x + 1
            return x + 1

        # 使用动态编译创建优化后的函数 opt_fn
        opt_fn = torch.compile(fn, dynamic=True, fullgraph=True)
        # 使用优化后的函数 opt_fn 处理输入 x，并忽略返回值
        opt_fn(torch.randn(12, 23))

    def test_shape_graph_break(self):
        from torch._dynamo.comptime import comptime

        def fn(x):
            # 获取 x 的形状
            x_shape = x.size()
            # 手动打破计算图
            comptime.graph_break()
            # 返回 x 与形状相同的随机张量的和
            return x + torch.randn(x_shape)

        # 生成一个形状为 [20] 的随机张量 x
        x = torch.randn(20)
        # 对函数 fn 进行优化，并得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 使用优化后的函数 opt_fn 处理输入 x，并忽略返回值
        opt_fn(x)

    def test_isinstance_symint(self):
        def fn(x):
            # 断言 x.size(0) 的类型是 int
            assert isinstance(x.size(0), int)
            # 返回 x 的两倍
            return x * 2

        # 生成一个形状为 [20] 的随机张量 x
        x = torch.randn(20)
        # 对函数 fn 进行优化，并得到优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 使用优化后的函数 opt_fn 处理输入 x，并忽略返回值
        opt_fn(x)
        # 生成一个形状为 [30] 的随机张量 y
        y = torch.randn(30)
        # 在 y 上标记为动态
        torch._dynamo.mark_dynamic(y, 0)
        # 使用优化后的函数 opt_fn 处理输入 y，并忽略返回值
        opt_fn(y)
    def test_mark_01_dynamic(self):
        # 定义一个简单的函数，将输入 x 值乘以 2
        def fn(x):
            return x * 2

        # 生成一个随机张量 x
        x = torch.randn(1)
        # 将张量 x 标记为动态（可能在运行时改变）
        torch._dynamo.mark_dynamic(x, 0)
        # 对函数 fn 进行优化以适应动态张量
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 尝试使用 opt_fn 处理动态输入 x，可能会导致通用内核编译失败，
        # 但不应该报错（标记为动态会尽力处理，但允许 0/1 专门化）
        opt_fn(x)

    def test_conv1d_symint_padding(self):
        # 生成一个大小为 (1, 1, 4) 的随机卷积核张量 kernel
        kernel = torch.randn(1, 1, 4)

        # 定义一个函数 func，对输入 x 进行卷积操作
        def func(x):
            # 计算填充值，使得卷积后输出的长度与输入 x 相同
            padding = math.ceil((kernel.shape[-1] + x.shape[-1] % 2) / 2) - 1
            # 使用 F.conv1d 函数进行一维卷积操作，设置填充和步幅
            out = F.conv1d(x, kernel, padding=padding, stride=2)
            return out

        # 编译优化函数 func
        opt_func = torch.compile(func)

        # 使用不同尺寸的输入张量 x 来测试优化后的函数 opt_func
        x = torch.randn(1, 1, 175)
        opt_func(x)  # 成功执行
        x = torch.randn(1, 1, 249)
        opt_func(x)  # 执行时崩溃

    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_propagate_dynamic_dim(self):
        # 生成一个大小为 20 的随机张量 x
        x = torch.randn(20)
        # 将张量 x 的第 0 维标记为动态
        torch._dynamo.mark_dynamic(x, 0)

        # 定义一个函数 fn，对输入 x 进行数值操作
        @torch.compile()
        def fn(x):
            y = x * 2
            comptime.graph_break()
            z = y * 2
            return z

        # 对函数 fn 进行编译优化，返回结果 z
        z = fn(x)
        # 断言 z 的弱动态维度集合为 {0}
        self.assertEqual(z._dynamo_weak_dynamic_indices, {0})

    def test_rshift_dynamic(self):
        # 定义一个函数 shift_right，对输入张量进行右移位操作
        def shift_right(tensor: torch.Tensor) -> torch.Tensor:
            return (tensor >> 2).to(torch.long)

        # 使用编译器对 shift_right 函数进行优化，开启全图分析和动态模式
        opt_fn = torch.compile(shift_right, fullgraph=True, dynamic=True)
        # 创建一个示例输入 sample_input
        sample_input = torch.tensor([4, 4, 16, 32], dtype=torch.uint8)
        # 使用优化后的函数 opt_fn 处理示例输入
        opt_fn(sample_input)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_symfloat_to_tensor(self):
        # 定义四个函数 f1, f2, f3, f4，将输入标量值转换为张量
        def f1(v):
            return torch.tensor([v.item()])

        def f2(v):
            return torch.tensor([[v.item()], [2.0]])

        def f3(v):
            return torch.tensor(v.item())

        def f4(v):
            return torch.tensor((v.item(),))

        # 使用特定的后端编译器进行全图分析的优化
        optimize = torch.compile(backend="aot_eager", fullgraph=True)

        # 生成一个随机张量 r
        r = torch.randn(1)

        # 对每个函数 f1, f2, f3, f4 进行编译优化，并与原始函数执行结果进行断言比较
        self.assertEqual(f1(r), optimize(f1)(r))
        self.assertEqual(f2(r), optimize(f2)(r))
        self.assertEqual(f3(r), optimize(f3)(r))
        self.assertEqual(f4(r), optimize(f4)(r))
    def test_to_tensor(self):
        # 定义函数 f1，生成一个形状为 (20, 1) 的随机浮点数数组，并转换为 Torch 张量
        def f1():
            a = np.random.uniform(low=-1, high=1, size=(20, 1))
            return torch.tensor([a, a, a, a], dtype=torch.float64, device="cpu")

        # 定义函数 f2，生成一个包含 [[[123]]] 的 Torch 张量，并返回包含两个相同张量的张量
        def f2():
            a = torch.tensor([[[123]]])
            return torch.tensor([a, a])

        # 定义函数 f3，生成一个标量值为 123 的 Torch 张量，并返回包含两个相同张量的张量
        def f3():
            a = torch.tensor(123)
            return torch.tensor([a, a])

        # 定义函数 f4，生成一个标量值为 123 的 Torch 张量和形状为 [[[456]]] 的张量，并返回包含这两个张量的张量
        def f4():
            a = torch.tensor(123)
            b = torch.tensor([[[456]]])
            return torch.tensor([a, b])

        # 定义函数 f5，生成一个形状为 [1, 2] 的 NumPy 数组，并将其转换为 Torch 张量，返回包含两个相同张量的张量
        def f5():
            a = np.array([1, 2])
            return torch.tensor([a, a])

        # 使用 AOT 编译器进行优化，确保各函数的输出与优化后的输出相等
        optimize = torch.compile(backend="aot_eager", fullgraph=True)

        # 断言各函数经过优化后输出的形状与原始函数的输出形状相等
        self.assertEqual(f1().shape, optimize(f1)().shape)
        self.assertEqual(f2(), optimize(f2)())
        self.assertEqual(f3(), optimize(f3)())
        self.assertEqual(f4(), optimize(f4)())
        self.assertEqual(f5(), optimize(f5)())

    def test_sym_int_conversion(self):
        # 定义函数 f，计算输入张量的第一维大小并返回 x 乘以该大小是否为 0 的结果
        def f(x):
            y = x.size(0)
            return x * int(y == 0)

        # 使用 eager 模式的编译器进行优化
        opt_fn = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(2, 3)
        opt_fn(x)

    def test_sum_dimlist_spec(self):
        # 定义函数 fn，计算输入张量在指定维度上的和并返回结果
        def fn(inputs, dim):
            return torch.sum(inputs, dim)

        inputs = torch.randn(128, 5, 24, 24)
        dim = (-1, 1, 0, 2)
        # 使用 eager 模式的动态编译器进行优化
        compl_fn = torch.compile(fn, dynamic=True, backend="eager", fullgraph=True)
        self.assertEqual(compl_fn(inputs, dim), fn(inputs, dim))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_max(self):
        # 定义函数 fn，根据输入 x 的最大值和 1024 返回一个值为 1 的 Torch 张量
        def fn(x):
            return torch.ones(max(x.item(), 1024))

        x = torch.tensor([1000])
        y = torch.tensor([2000])
        # 使用 eager 模式的编译器进行优化
        compl_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compl_fn(x))
        self.assertEqual(fn(y), compl_fn(y))

    # https://github.com/pytorch/pytorch/issues/104812
    def test_argmin_coerces_symint_to_intlist_spec(self):
        # 定义函数 fn，使用 torch.amin 计算输入张量在指定维度上的最小值，并保持维度不变返回结果
        def fn(x, dim):
            # Python 参数解析器将 dim 强制转换为一个 vector<int>
            return torch.amin(x, dim=dim, keepdim=True)

        x = torch.randn(4, 4, 4)
        dim = 2
        # 使用 eager 模式的动态编译器进行优化
        compl_fn = torch.compile(fn, dynamic=True, backend="eager", fullgraph=True)
        self.assertEqual(compl_fn(x, dim), fn(x, dim))

    def test_exponential(self):
        # 定义函数 fn，对输入张量进行指数操作并返回结果
        def fn(inputs, op_inputs_dict):
            res = inputs.exponential_(**op_inputs_dict)
            return res

        inputs = torch.randn(2, 3, 4)
        op_inputs_dict = {"lambd": 10, "generator": None}
        # 使用 eager 模式的动态编译器进行优化
        compl_fn = torch.compile(fn, dynamic=True, backend="eager", fullgraph=True)
        self.assertEqual(compl_fn(inputs, op_inputs_dict), fn(inputs, op_inputs_dict))
    # 定义一个测试函数，用于测试 symbol_guard_limit_before_specialize 的行为
    def test_symbol_guard_limit_before_specialize(self):
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()

        # 使用装饰器优化函数 fn，允许动态特化
        @torch._dynamo.optimize(cnts, dynamic=True)
        def fn(x):
            # 检查 x 的大小是否不等于 3，4，5 或 6
            torch._check(x.size(0) != 3)
            torch._check(x.size(0) != 4)
            torch._check(x.size(0) != 5)
            torch._check(x.size(0) != 6)
            # 返回 x 加 2
            return x + 2

        # 控制测试，调用 fn 函数三次，传入不同的参数
        fn(torch.randn(12))
        fn(torch.randn(13))
        fn(torch.randn(14))

        # 断言编译计数器的帧数为 1
        self.assertExpectedInline(cnts.frame_count, """1""")
        # 将帧数重置为 0
        cnts.frame_count = 0

        # 重置动态编译环境
        torch._dynamo.reset()

        # 使用上下文管理器修改配置，设置 symbol_guard_limit_before_specialize 为 3
        with torch.fx.experimental._config.patch(
            symbol_guard_limit_before_specialize=3
        ):
            # 再次调用 fn 函数三次，传入不同的参数
            fn(torch.randn(12))
            fn(torch.randn(13))
            fn(torch.randn(14))

            # 断言编译计数器的帧数为 3
            self.assertExpectedInline(cnts.frame_count, """3""")

    # 定义一个测试函数，测试函数默认值的行为
    def test_defaults(self):
        # 定义一个函数 g，带有参数 x 和默认参数 i=8
        def g(x, i=8):
            # 断言 i 在编译时是静态的
            comptime.assert_static(i)
            # 返回 x 乘以 i
            return x * i

        # 定义一个函数 fn，调用 g 函数
        def fn(x):
            return g(x)

        # 生成一个输入张量
        inputs = torch.randn(2, 3, 4)
        # 编译函数 fn，使用 eager 后端，并启用动态编译
        compl_fn = torch.compile(fn, dynamic=True, backend="eager")
        # 断言编译后的函数计算结果与原函数一致
        self.assertEqual(compl_fn(inputs), fn(inputs))

    # 使用装饰器配置，设置 specialize_float=False 和 assume_static_by_default=True，测试未指定浮点数输入的行为
    @torch._dynamo.config.patch(specialize_float=False, assume_static_by_default=True)
    def test_unspec_float_input(self):
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个函数 f，带有参数 x 和 y
        def f(x, y):
            # 如果 y 等于 5.0，则返回 x 加 2；否则返回 x 加 y
            if y == 5.0:
                return x + 2
            else:
                return x + y

        # 编译函数 f，使用 cnts 作为后端，并开启完整图形模式
        cf = torch.compile(backend=cnts, fullgraph=True)(f)

        # 生成一个随机张量 x
        x = torch.randn(3)
        # 断言 f(x, 3.0) 和 cf(x, 3.0) 的计算结果一致
        self.assertEqual(f(x, 3.0), cf(x, 3.0))
        # 断言编译计数器的帧数为 1，即没有重新编译
        self.assertExpectedInline(cnts.frame_count, """1""")
        # 断言 f(x, 5.0) 和 cf(x, 5.0) 的计算结果一致
        self.assertEqual(f(x, 5.0), cf(x, 5.0))
        # 断言编译计数器的帧数为 2，表示 guard 生效，进行了重新编译
        self.assertExpectedInline(cnts.frame_count, """2""")
        # 断言 f(x, math.nan) 和 cf(x, math.nan) 的计算结果一致
        self.assertEqual(f(x, math.nan), cf(x, math.nan))
        # 断言编译计数器的帧数为 3，因为 nan 总是需要重新编译
        self.assertExpectedInline(cnts.frame_count, """3""")

    # 使用装饰器配置，设置 specialize_float=False 和 assume_static_by_default=True，测试未指定浮点数输出的行为
    @torch._dynamo.config.patch(specialize_float=False, assume_static_by_default=True)
    def test_unspec_float_output(self):
        # 创建一个编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()

        # 定义一个函数 f，带有参数 x 和 y，返回 x+1 和 y*2
        def f(x, y):
            return x + 1, y * 2

        # 编译函数 f，使用 cnts 作为后端，并开启完整图形模式
        cf = torch.compile(backend=cnts, fullgraph=True)(f)

        # 生成一个随机张量 x
        x = torch.randn(3)

        # 断言 f(x, 3.0) 和 cf(x, 3.0) 的计算结果一致
        self.assertEqual(f(x, 3.0), cf(x, 3.0))
        # 断言 f(x, 4.0) 和 cf(x, 4.0) 的计算结果一致
        self.assertEqual(f(x, 4.0), cf(x, 4.0))
        # 断言 f(x, 5.0) 和 cf(x, 5.0) 的计算结果一致
        self.assertEqual(f(x, 5.0), cf(x, 5.0))

    # 使用装饰器配置，设置 capture_scalar_outputs=True
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_data_dependent_evaluate_expr_graph_break(self):
        cnts = torch._dynamo.testing.CompileCounter()

        # 确保编译持续帧的方法是编写测试函数的特殊方式。
        # 参考 https://github.com/pytorch/pytorch/issues/111918
        # 定义一个测试函数，用于测试条件是否大于2
        def test(y):
            if y > 2:
                return True
            else:
                return False

        # 使用 CompileCounter 进行优化的装饰器
        @torch._dynamo.optimize(cnts)
        def fn(x):
            # 对输入张量 x 增加 1
            x = x + 1
            # 获取张量 x 的数值部分
            y = x.item()
            # 根据 test 函数的结果选择不同的计算分支
            if test(y):
                return x * 2
            else:
                return x * 3

        # 创建输入张量 x，调用优化后的函数 fn
        x = torch.tensor([3.0])
        fn(x)

        # 断言编译的持续帧数是否符合预期
        self.assertExpectedInline(cnts.frame_count, """2""")
        # 断言操作的数量是否符合预期
        self.assertExpectedInline(cnts.op_count, """4""")

    def test_prune_torch_check(self):
        # 将日志流转换为字符串并获取上下文
        log_stream, ctx = logs_to_string("torch._dynamo.output_graph", "graph_code")

        # 使用编译装饰器，启用完整图模式和动态执行，选择 eager 后端
        @torch.compile(fullgraph=True, dynamic=True, backend="eager")
        def f(x, y):
            # 执行 Torch 检查，确保条件成立
            torch._check(y + 5 == 85)
            # 执行 Torch 检查，确保张量 x 的第一维度大小为 80
            torch._check(x.size(0) == 80)

        # 在指定上下文中执行函数 f，并捕获日志流
        with ctx():
            f(torch.randn(80, 100), 80)

        # 提取日志流中的输出，移除前三行并进行断言
        out = "\n".join(log_stream.getvalue().strip().split("\n")[3:]).strip()
        self.assertExpectedInline(
            out,
            """\
# 定义一个类方法 `forward`，返回一个空元组
def forward(self):
    return ()



# 使用 `torch._dynamo.config.patch` 装饰器，配置捕获标量输出为真
@torch._dynamo.config.patch(capture_scalar_outputs=True)
# 定义测试方法 `test_split_aot_autograd`
def test_split_aot_autograd(self):
    # 使用 `torch.compile` 装饰器，选择后端为 "aot_eager"，并开启完整图形模式
    def f(x, i):
        # 将参数 `i` 转换为列表后，获取其第一和第二个元素，分别赋值给 `y` 和 `z`
        y, z = i.tolist()
        # 使用 `torch.split` 函数对输入 `x` 进行分割，分割点为 `y` 和 `z`
        return torch.split(x, [y, z])

    # 打印调用 `f` 方法的结果，传入随机生成的张量和指定的张量 `[7, 3]`
    print(f(torch.randn(10, requires_grad=True), torch.tensor([7, 3])))



# 定义测试方法 `test_bool_tensor_ctor`
def test_bool_tensor_ctor(self):
    # 创建 `torch._dynamo.testing.CompileCounter` 的实例对象 `cnts`
    cnts = torch._dynamo.testing.CompileCounter()

    # 使用 `torch.compile` 装饰器，选择后端为 `cnts`，开启动态模式，并开启完整图形模式
    def f(x):
        # 使用 `torch.empty` 方法创建一个张量 `y`，形状为 `(x.size(0) // 13) * 13` 的空张量
        y = torch.empty((x.size(0) // 13) * 13)
        # 返回一个布尔张量，表示张量 `y` 的元素数量是否为 0
        return torch.tensor(y.numel() == 0)

    # 使用 `self.assertTrue` 方法断言 `f` 方法对空张量 `torch.empty(8)` 返回值为真
    self.assertTrue(f(torch.empty(8)).item())
    # 使用 `self.assertFalse` 方法断言 `f` 方法对空张量 `torch.empty(13)` 返回值为假
    self.assertFalse(f(torch.empty(13)).item())



# 如果脚本作为主程序运行，则从 `torch._dynamo.test_case` 模块导入 `run_tests` 方法并执行
if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
```