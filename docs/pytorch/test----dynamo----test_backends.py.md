# `.\pytorch\test\dynamo\test_backends.py`

```
# 引入unittest模块，用于编写和运行单元测试
import unittest

# 引入torch模块，用于进行张量计算和神经网络构建
import torch

# 引入torch._dynamo模块及其子模块和函数，用于特定的动态优化和测试
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.backends.debugging import ExplainWithBackend
from torch._dynamo.backends.onnxrt import has_onnxruntime
from torch._dynamo.backends.tvm import has_tvm
from torch._dynamo.testing import same

# 引入torch.fx._lazy_graph_module模块的_force_skip_lazy_graph_module函数，用于懒加载图模块
from torch.fx._lazy_graph_module import _force_skip_lazy_graph_module

# 引入torch.testing._internal.inductor_utils模块的HAS_CUDA常量，检查是否有CUDA支持
from torch.testing._internal.inductor_utils import HAS_CUDA

# 设置unittest装饰器，如果没有CUDA支持，则跳过此测试
requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")

# 定义一个继承自torch.nn.Module的类Seq
class Seq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 构建包含多个层的顺序模型
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),  # 添加线性层，输入维度为10，输出维度为10
            torch.nn.ReLU(),          # ReLU激活函数层
            torch.nn.Linear(10, 10),  # 添加另一个线性层，输入维度为10，输出维度为10
            torch.nn.Sigmoid(),       # Sigmoid激活函数层
        )

    def forward(self, x):
        # 前向传播函数，通过顺序模型处理输入数据x并返回输出
        return self.layers(x)

# 定义一个继承自torch.nn.Module的类Conv_Bn_Relu
class Conv_Bn_Relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # 添加卷积层，输入通道数为in_channels，输出通道数为out_channels
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # 添加批归一化层，输入通道数为out_channels，设置eps为0.001
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        # 添加ReLU激活函数层
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # 前向传播函数，先卷积，再批归一化，最后ReLU激活，并返回结果
        return self.relu(self.bn(self.conv(x)))

# 定义一个继承自torch._dynamo.test_case.TestCase的测试类TestOptimizations
class TestOptimizations(torch._dynamo.test_case.TestCase):
    def test_example_inputs(self):
        # 定义一个函数fn，接受a、bc、d作为参数，计算表达式a/d - b/c并返回结果
        def fn(a, bc, d):
            b, c = bc
            return a / d - b / c

        # 定义编译器函数compiler_fn，接受图形和示例输入作为参数，执行图形并返回结果
        def compiler_fn(graph, example_inputs):
            nonlocal r1
            r1 = graph(*example_inputs)[0]  # 执行图形处理示例输入并存储第一个结果到r1
            return graph.forward

        # 创建张量a、b、c，以及标量d，并初始化它们的值
        a = torch.empty(2).fill_(1)
        b = torch.empty(2).fill_(2)
        c = torch.empty(2).fill_(3)
        d = 4
        r1 = None  # 初始化r1为None
        r2 = fn(a, (b, c), d)  # 调用fn函数计算r2的值
        opt_fn = torch._dynamo.optimize_assert(compiler_fn)(fn)  # 优化fn函数并返回新的函数opt_fn
        r3 = opt_fn(a, (b, c), d)  # 使用优化后的函数opt_fn计算r3的值

        # 断言r1不为None
        self.assertIsNotNone(r1)
        # 断言r1的大小与r2的大小相同
        self.assertEqual(r1.size(), r2.size())
        # 断言r1的步幅与r2的步幅相同
        self.assertEqual(r1.stride(), r2.stride())
        # 断言r1的数据类型与r2的数据类型相同
        self.assertEqual(r1.dtype, r2.dtype)

        # 断言r1的大小与r3的大小相同
        self.assertEqual(r1.size(), r3.size())
        # 断言r1的步幅与r3的步幅相同
        self.assertEqual(r1.stride(), r3.stride())
        # 断言r1的数据类型与r3的数据类型相同
        self.assertEqual(r1.dtype, r3.dtype)

    def test_example_inputs_runtime_use(self):
        # 定义一个函数fn，接受a、bc、d作为参数，计算表达式a/d - b/c并返回结果
        def fn(a, bc, d):
            b, c = bc
            return a / d - b / c

        # 定义编译器函数compiler_fn，接受图形和示例输入作为参数，执行图形并返回结果
        def compiler_fn(graph, example_inputs):
            def fwd(*args):
                nonlocal r1
                r = graph.forward(*args)
                r1 = r[0]
                return r

            return fwd

        # 创建张量a、b、c，以及标量d，并初始化它们的值
        a = torch.empty(2).fill_(1)
        b = torch.empty(2).fill_(2)
        c = torch.empty(2).fill_(3)
        d = 4
        r1 = None  # 初始化r1为None
        r2 = fn(a, (b, c), d)  # 调用fn函数计算r2的值
        opt_fn = torch._dynamo.optimize_assert(compiler_fn)(fn)  # 优化fn函数并返回新的函数opt_fn
        r3 = opt_fn(a, (b, c), d)  # 使用优化后的函数opt_fn计算r3的值

        # 断言r1不为None
        self.assertIsNotNone(r1)
        # 断言r1与r2的值相同
        self.assertTrue(same(r1, r2))
        # 断言r1与r3的值相同
        self.assertTrue(same(r1, r3))
    # 检查特定后端是否正常工作的私有方法
    def _check_backend_works(self, backend, options=None):
        # 创建一个评估模式的序列模型
        model = Seq().eval()
        # 生成一个形状为 (2, 10) 的随机输入张量
        input = torch.randn(2, 10)
        # 在模型上执行前向传播，获取结果 r1
        r1 = model(input)
        # 使用指定的后端和选项编译模型，并在输入上执行，获取结果 r2
        r2 = torch.compile(model, backend=backend, options=options)(input)
        # 断言 r1 和 r2 的结果近似相等，精度容差为 0.01
        self.assertTrue(same(r1, r2.float(), tol=0.01))

    # 测试方法：测试 eager 后端是否正常工作
    def test_eager(self):
        self._check_backend_works("eager")

    # 测试方法：测试 eager_noexcept 后端是否正常工作
    def test_eager_noexcept(self):
        self._check_backend_works("eager_noexcept")

    # 装饰器标记的测试方法：测试 torchscript 后端是否正常工作
    @_force_skip_lazy_graph_module()
    def test_torchscript(self):
        self._check_backend_works("ts")

    # 测试方法：测试 aot_eager 后端是否正常工作
    def test_aot_eager(self):
        self._check_backend_works("aot_eager")

    # 测试方法：测试 aot_eager_decomp_partition 后端是否正常工作
    def test_aot_eager_decomp_partition(self):
        self._check_backend_works("aot_eager_decomp_partition")

    # 装饰器标记的测试方法：测试 aot_ts 后端是否正常工作
    @_force_skip_lazy_graph_module()
    def test_aot_ts(self):
        self._check_backend_works("aot_ts")

    # 装饰器标记的测试方法：测试需要 CUDA 的 aot_cudagraphs 后端是否正常工作
    @requires_cuda
    def test_aot_cudagraphs(self):
        self._check_backend_works("cudagraphs")

    # 装饰器标记的测试方法：测试需要 onnxruntime 的 onnxrt 后端是否正常工作
    @unittest.skipIf(not has_onnxruntime(), "requires onnxruntime")
    def test_onnxrt(self):
        self._check_backend_works("onnxrt")

    # 装饰器标记的测试方法：测试需要 tvm 的 tvm 后端是否正常工作，以及不同选项下的工作情况
    @unittest.skipIf(not has_tvm(), "requires tvm")
    def test_tvm(self):
        self._check_backend_works("tvm")
        self._check_backend_works("tvm", options={"scheduler": None})
        self._check_backend_works("tvm", options={"opt_level": 0})

    # 测试方法：测试列出后端的功能是否正确
    def test_list_backends(self):
        self.assertIn("inductor", torch._dynamo.list_backends())
        self.assertIn("inductor", torch._dynamo.list_backends(exclude_tags=None))
        self.assertNotIn("eager", torch._dynamo.list_backends())
        self.assertNotIn("eager", torch._dynamo.list_backends(exclude_tags=["debug"]))
        self.assertIn("eager", torch._dynamo.list_backends(exclude_tags=[]))
# 定义一个名为 NormalizeIRTests 的测试类，继承自 torch._dynamo.test_case.TestCase
class NormalizeIRTests(torch._dynamo.test_case.TestCase):
    
    # 定义一个测试方法 test_inplace_normalize
    def test_inplace_normalize(self):
        
        # 定义一个内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 计算 a 的余弦值并赋给变量 x
            x = torch.cos(a)
            # 将 x 加上 b
            x += b
            # 返回 x 的正弦值
            return torch.sin(x)

        # 生成一个包含 10 个随机数的张量 a
        a = torch.randn(10)
        # 生成一个包含 10 个随机数的张量 b，并将其类型转换为 torch.float64
        b = torch.randn(10).to(torch.float64)

        # 调用 fn 函数计算参考结果 ref
        ref = fn(a, b)

        # 使用 torch._dynamo.optimize("aot_eager") 优化 fn 函数并赋值给 optimized_fn
        optimized_fn = torch._dynamo.optimize("aot_eager")(fn)
        # 使用 a 和 b 调用优化后的函数 optimized_fn，计算结果并赋给 res
        res = optimized_fn(a, b)
        
        # 使用 self.assertTrue 进行断言，检查 ref 和 res 是否相同
        self.assertTrue(same(ref, res))


# 定义一个名为 MPSNotSupportedTest 的测试类，继承自 torch._dynamo.test_case.TestCase
class MPSNotSupportedTest(torch._dynamo.test_case.TestCase):
    
    # 定义一个测试方法 test_mps_not_supported，并使用 unittest.skipIf 装饰器检查是否支持 mps
    @unittest.skipIf(not torch.backends.mps.is_available(), "requires mps")
    def test_mps_not_supported(self):
        # 创建一个 Seq 模型，并将其设定在 "mps" 后端
        model = Seq().to("mps")
        # 生成一个包含随机数的示例输入 example_input，并将其设定在 "mps" 后端
        example_input = torch.randn(1, 10).to("mps")
        
        # 使用 self.assertRaises 进行断言，检查使用 "inductor" 后端编译 model 是否引发 RuntimeError 异常
        self.assertRaises(
            RuntimeError,
            lambda: torch.compile(model, backend="inductor")(example_input),
        )


# 定义一个名为 TestExplainWithBackend 的测试类，继承自 torch._dynamo.test_case.TestCase
class TestExplainWithBackend(torch._dynamo.test_case.TestCase):
    
    # 定义一个测试方法 test_explain_with_backend
    def test_explain_with_backend(self):
        
        # 定义一个内部函数 fn3，接受参数 x
        def fn3(x):
            # 对 x 求正弦值并赋给 x
            x = torch.sin(x)
            # 调用 torch._dynamo.graph_break() 函数
            torch._dynamo.graph_break()
            # 再次对 x 求正弦值并赋给 x
            x = torch.sin(x)
            # 返回 x
            return x

        # 定义一个内部函数 fn2，接受参数 x
        def fn2(x):
            # 对 x 求余弦值并赋给 x
            x = torch.cos(x)
            # 调用 fn3 函数并将 x 作为参数传入
            x = fn3(x)
            # 再次对 x 求余弦值并赋给 x
            x = torch.cos(x)
            # 返回 x
            return x

        # 定义一个内部函数 fn1，接受参数 x
        def fn1(x):
            # 对 x 求正切值并赋给 x
            x = torch.tan(x)
            # 调用 fn2 函数并将 x 作为参数传入
            x = fn2(x)
            # 再次对 x 求正切值并赋给 x
            x = torch.tan(x)
            # 返回 x
            return x

        # 定义一个内部函数 fn，接受参数 x
        def fn(x):
            # 对 x 求 sigmoid 函数并赋给 x
            x = torch.sigmoid(x)
            # 调用 fn1 函数并将 x 作为参数传入
            x = fn1(x)
            # 再次对 x 求 sigmoid 函数并赋给 x
            x = torch.sigmoid(x)
            # 返回 x
            return x

        # 使用 ExplainWithBackend("inductor") 包装 TorchInductor
        eb = ExplainWithBackend("inductor")
        # 使用 torch.compile 函数将 fn 函数编译，并指定使用 eb 作为后端，赋值给 optimized_fn
        optimized_fn = torch.compile(fn, backend=eb)
        # 生成一个包含随机数的输入张量 input_tensor
        input_tensor = torch.randn(5)
        # 使用 optimized_fn 计算 input_tensor 的结果并赋给 result
        result = optimized_fn(input_tensor)

        # 使用 self.assertTrue 进行断言，检查 optimized_fn 和 fn 计算结果是否相同
        self.assertTrue(torch.allclose(result, fn(input_tensor)))

        # 获取 ExplainWithBackend 输出对象 explain_output
        explain_output = eb.output()
        # 将 explain_output 转换为字符串，并赋给 explain_str
        explain_str = str(explain_output)
        
        # 使用 self.assertIn 进行断言，检查 explain_str 是否包含 "Graph Count" 字段
        self.assertIn("Graph Count", explain_str)
        # 使用 self.assertIn 进行断言，检查 explain_str 是否包含 "Graph Break Count" 字段
        self.assertIn("Graph Break Count", explain_str)
        # 使用 self.assertIn 进行断言，检查 explain_str 是否包含 "Op Count" 字段
        self.assertIn("Op Count", explain_str)
        # 使用 self.assertIn 进行断言，检查 explain_str 是否包含 "Break Reasons" 字段
        self.assertIn("Break Reasons", explain_str)

        # 使用 self.assertEqual 进行断言，检查 explain_output.graph_count 是否等于 8
        self.assertEqual(8, explain_output.graph_count)
        # 使用 self.assertEqual 进行断言，检查 explain_output.graph_break_count 是否等于 7
        self.assertEqual(7, explain_output.graph_break_count)
        # 使用 self.assertEqual 进行断言，检查 explain_output.op_count 是否等于 8
        self.assertEqual(8, explain_output.op_count)


# 定义一个名为 TestCustomBackendAPI 的测试类，继承自 torch._dynamo.test_case.TestCase
class TestCustomBackendAPI(torch._dynamo.test_case.TestCase):
    """Test APIs documented by https://pytorch.org/docs/main/torch.compiler_custom_backends.html"""

    # 此处无需注释，因为类内部没有实际的测试方法
    # 定义测试函数，用于测试注册后端 API 的功能
    def test_register_backend_api(self):
        # 导入需要的模块和函数
        from torch._dynamo import register_backend
        
        # 初始化一个标志，用于检测后端是否运行
        backend_run = False
        
        # 定义一个装饰器，用于注册自定义后端
        @register_backend
        def my_custom_backend(gm, example_inputs):
            nonlocal backend_run
            # 设置标志，表示后端已运行
            backend_run = True
            return gm.forward
        
        # 定义一个简单的函数
        def f(x):
            return torch.relu(x)
        
        # 编译函数 f，使用注册的后端 "my_custom_backend"
        opt_f = torch.compile(f, backend="my_custom_backend")
        
        # 调用优化后的函数，并传入随机数据
        opt_f(torch.randn(3, 3))
        
        # 断言检查后端是否运行
        self.assertTrue(backend_run)

    # 定义测试函数，用于测试 AOT 自动微分 API 的功能
    def test_aot_autograd_api(self):
        # 导入需要的模块和函数
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd
        
        # 初始化一个标志，用于检测后端是否运行
        backend_run = False
        
        # 定义一个编译器函数，用于自定义编译逻辑
        def my_compiler(gm, example_inputs):
            nonlocal backend_run
            # 设置标志，表示后端已运行
            backend_run = True
            return make_boxed_func(gm.forward)
        
        # 创建 AOT 自动微分后端对象，并传入自定义的编译器函数
        my_backend = aot_autograd(fw_compiler=my_compiler)
        
        # 定义一个简单的函数
        def f(x):
            return torch.relu(x)
        
        # 编译函数 f，使用自定义的 AOT 自动微分后端
        opt_f = torch.compile(f, backend=my_backend)
        
        # 调用优化后的函数，并传入随机数据
        opt_f(torch.randn(3, 3))
        
        # 断言检查后端是否运行
        self.assertTrue(backend_run)

    # 定义测试函数，用于测试查找后端 API 的功能
    def test_lookup_backend(self):
        # 导入需要的模块和函数
        from torch._dynamo import list_backends, lookup_backend
        
        # 获取已注册的所有后端列表
        backends = list_backends()
        
        # 初始化一个标志，用于检测后端是否运行
        backend_run = False
        
        # 定义一个编译器函数，用于自定义编译逻辑
        def my_compiler(gm, example_inputs):
            nonlocal backend_run
            # 设置标志，表示后端已运行
            backend_run = True
            try:
                # 尝试使用 TensorRT 后端进行编译
                trt_compiled = lookup_backend("tensorrt")(gm, example_inputs)
                if trt_compiled is not None:
                    return trt_compiled
            except Exception:
                pass
            # 第一个后端失败后，尝试使用其他后端...
            try:
                inductor_compiled = lookup_backend("inductor")(gm, example_inputs)
                if inductor_compiled is not None:
                    return inductor_compiled
            except Exception:
                pass
            # 如果所有尝试失败，则返回原始模型的 forward 方法
            return gm.forward
        
        # 定义一个简单的函数
        def f(x):
            return torch.relu(x)
        
        # 编译函数 f，使用自定义的后端查找逻辑
        opt_f = torch.compile(f, backend=my_compiler)
        
        # 调用优化后的函数，并传入随机数据
        opt_f(torch.randn(3, 3))
        
        # 断言检查后端是否运行
        self.assertTrue(backend_run)
# 如果这个脚本是作为主程序执行
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块导入 run_tests 函数
    from torch._dynamo.test_case import run_tests
    # 运行测试函数 run_tests()
    run_tests()
```