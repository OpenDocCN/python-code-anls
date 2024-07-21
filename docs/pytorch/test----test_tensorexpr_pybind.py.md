# `.\pytorch\test\test_tensorexpr_pybind.py`

```py
# 导入所需的库和模块，包括 torch、numpy、torch._C._te，以及测试相关的模块和函数
import torch
import numpy as np
import torch._C._te as te

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase
import unittest

# 检查是否启用了 LLVM 支持，并将结果存储在 LLVM_ENABLED 变量中
LLVM_ENABLED = torch._C._llvm_enabled()

# 定义一个函数 construct_adder，用于构建一个加法器表达式
def construct_adder(n: int, dtype=torch.float32):
    # 创建名为 A 和 B 的缓冲区对象，每个对象包含 n 个元素，数据类型为 dtype
    A = te.BufHandle("A", [n], dtype)
    B = te.BufHandle("B", [n], dtype)

    # 定义一个 compute 函数，表示计算逻辑为 A[i] + B[i]
    def compute(i):
        return A.load([i]) + B.load([i])

    # 创建一个名为 C 的计算对象，它也包含 n 个元素，使用 compute 函数定义的逻辑
    C = te.Compute("C", [n], compute)

    # 创建一个循环嵌套对象 loopnest，包含计算对象 C
    loopnest = te.LoopNest([C])
    # 准备循环嵌套对象以进行代码生成
    loopnest.prepare_for_codegen()
    # 简化根语句，并将结果存储在 stmt 变量中
    stmt = te.simplify(loopnest.root_stmt())

    # 使用 te.construct_codegen 创建一个代码生成器对象，并返回该对象
    return te.construct_codegen("ir_eval", stmt, [A, B, C])

# 创建一个测试类 TestTensorExprPyBind，继承自 JitTestCase
class TestTensorExprPyBind(JitTestCase):
    # 定义一个测试方法 test_simple_sum
    def test_simple_sum(self):
        n = 32
        # 使用 construct_adder 创建一个加法器代码生成器对象 cg
        cg = construct_adder(n)

        # 创建大小为 n 的随机张量 tA 和 tB，以及一个空张量 tC
        tA = torch.randn(n)
        tB = torch.randn(n)
        tC = torch.empty(n)

        # 调用 cg 对象的 call 方法，执行计算，并将结果存储在 tC 中
        cg.call([tA, tB, tC])

        # 使用 torch.testing.assert_close 验证 tA + tB 是否等于 tC
        torch.testing.assert_close(tA + tB, tC)

    # 定义一个测试方法 test_call_raw
    def test_call_raw(self):
        n = 16
        # 使用不同的数据类型 torch.float64，调用 construct_adder 创建 cg 对象
        cg = construct_adder(n, dtype=torch.float64)

        # 创建大小为 n 的随机张量 tA、tB 和一个空张量 tC，数据类型为 torch.float64
        tA = torch.randn(n, dtype=torch.float64)
        tB = torch.randn(n, dtype=torch.float64)
        tC = torch.empty(n, dtype=torch.float64)

        # 调用 cg 对象的 call_raw 方法，传入数据指针，执行计算
        cg.call_raw([tA.data_ptr(), tB.data_ptr(), tC.data_ptr()])

        # 使用 torch.testing.assert_close 验证 tA + tB 是否等于 tC
        torch.testing.assert_close(tA + tB, tC)

    # 定义一个测试方法 test_external_calls
    def test_external_calls(self):
        dtype = torch.float32

        # 创建不同大小的缓冲区对象 A、B 和 C，数据类型为 dtype
        A = te.BufHandle("A", [1, 4], dtype)
        B = te.BufHandle("B", [4, 1], dtype)
        C = te.BufHandle("C", [1, 1], dtype)

        # 创建一个外部调用对象 s，调用名为 "nnc_aten_matmul" 的函数，传入 A 和 B 作为参数
        s = te.ExternalCall(C, "nnc_aten_matmul", [A, B], [])

        # 创建循环嵌套对象 loopnest，包含 s 和 C
        loopnest = te.LoopNest(s, [C])
        # 准备循环嵌套对象以进行代码生成
        loopnest.prepare_for_codegen()

        # 使用 te.construct_codegen 创建一个代码生成器对象 codegen
        codegen = te.construct_codegen("ir_eval", s, [A, B, C])

        # 创建大小为 (1, 4) 和 (4, 1) 的全为 1 的张量 tA 和 tB，以及一个空张量 tC
        tA = torch.ones(1, 4)
        tB = torch.ones(4, 1)
        tC = torch.empty(1, 1)

        # 调用 codegen 对象的 call 方法，执行计算，并将结果存储在 tC 中
        codegen.call([tA, tB, tC])

        # 使用 torch.testing.assert_close 验证 torch.matmul(tA, tB) 是否等于 tC
        torch.testing.assert_close(torch.matmul(tA, tB), tC)

    # 定义一个测试方法 test_dynamic_shape
    def test_dynamic_shape(self):
        # 创建一个整数类型的变量 dN
        dN = te.VarHandle(torch.int32)
        # 创建大小为 dN 的缓冲区对象 A 和 B，数据类型为 torch.float64
        A = te.BufHandle([dN], torch.float64)
        B = te.BufHandle([dN], torch.float64)

        # 定义一个 compute 函数，表示计算逻辑为 A[i] - B[i]
        def compute(i):
            return A.load(i) - B.load(i)

        # 创建一个名为 C 的计算对象，包含大小为 dN 的元素，使用 compute 函数定义的逻辑
        C = te.Compute("C", [dN], compute)

        # 创建循环嵌套对象 loopnest，包含计算对象 C
        loopnest = te.LoopNest([C])
        # 准备循环嵌套对象以进行代码生成
        loopnest.prepare_for_codegen()

        # 使用 te.construct_codegen 创建一个代码生成器对象 cg
        cg = te.construct_codegen("ir_eval", loopnest.simplify(), [A, B, C, dN])

        # 定义一个函数 test_with_shape，用于测试不同形状的输入
        def test_with_shape(n):
            # 创建大小为 n 的随机张量 tA 和 tB，以及一个空张量 tC，数据类型为 torch.double
            tA = torch.randn(n, dtype=torch.double)
            tB = torch.randn(n, dtype=torch.double)
            tC = torch.empty(n, dtype=torch.double)

            # 调用 cg 对象的 call 方法，执行计算，并将结果存储在 tC 中，同时传入 n 作为参数
            cg.call([tA, tB, tC, n])

            # 使用 torch.testing.assert_close 验证 tA - tB 是否等于 tC
            torch.testing.assert_close(tA - tB, tC)

        # 分别使用大小为 8 和 31 的输入调用 test_with_shape 方法
        test_with_shape(8)
        test_with_shape(31)
    def test_dynamic_shape_2d(self):
        # 创建两个整型的变量句柄，用于表示矩阵的维度
        dN = te.VarHandle(torch.int32)
        dM = te.VarHandle(torch.int32)
        # 创建两个缓冲区句柄，表示输入矩阵 A 和 B，数据类型为双精度浮点数
        A = te.BufHandle([dN, dM], torch.float64)
        B = te.BufHandle([dN, dM], torch.float64)

        # 定义计算函数，计算矩阵 C 的元素值
        def compute(i, j):
            return A.load([i, j]) - B.load([i, j])

        # 创建一个计算对象 C，表示矩阵的差
        C = te.Compute("C", [dN, dM], compute)

        # 创建循环嵌套对象，用于生成代码
        loopnest = te.LoopNest([C])
        loopnest.prepare_for_codegen()

        # 构建代码生成器对象，将简化后的 IR（中间表示）传入，以及相关的输入参数
        cg = te.construct_codegen("ir_eval", loopnest.simplify(), [A, B, C, dN, dM])

        # 定义测试函数，测试不同形状的输入
        def test_with_shape(n, m):
            # 创建随机的双精度浮点数矩阵 tA 和 tB
            tA = torch.randn(n, m, dtype=torch.double)
            tB = torch.randn(n, m, dtype=torch.double)
            # 创建空的双精度浮点数矩阵 tC
            tC = torch.empty(n, m, dtype=torch.double)
            # 调用代码生成器，计算 tA - tB，并将结果存入 tC
            cg.call([tA, tB, tC, n, m])
            # 使用 PyTorch 的测试函数，检查 tC 是否等于 tA - tB
            torch.testing.assert_close(tA - tB, tC)

        # 分别使用不同的形状参数调用测试函数
        test_with_shape(2, 4)
        test_with_shape(5, 3)

    def test_dtype_error(self):
        # 创建一个缓冲区句柄，数据类型为单精度浮点数，应该不会引发错误
        te.BufHandle("a", [1], torch.float32)  # ok
        # 使用错误的数据类型字符串创建缓冲区句柄，期望引发类型错误异常
        self.assertRaises(TypeError, lambda: te.BufHandle("a", [1], "float55"))

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_tensor_inputs(self):
        # 定义一个简单的函数 f，对三个输入张量 a, b, c 执行加法操作
        def f(a, b, c):
            return a + b + c

        device, size = "cpu", (4, 4)
        # 创建三个随机初始化的张量，指定设备为 CPU
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)
        z = torch.rand(size, device=device)

        graph_str = """
    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_shape_prop(self):
        device, size = "cpu", (4, 4)
        x = torch.rand(size, device=device)  # 创建一个大小为 (4, 4) 的随机张量 x，设备为 CPU
        y = torch.rand(size, device=device)  # 创建一个大小为 (4, 4) 的随机张量 y

        graph_str = """
graph(%a.1 : Float(requires_grad=0, device=cpu),  # 定义输入参数 a，类型为 Float，不需要梯度，设备为 CPU
      %b.1 : Float(requires_grad=0, device=cpu),  # 定义输入参数 b，类型为 Float，不需要梯度，设备为 CPU
      %c.1 : Float(requires_grad=0, device=cpu)):  # 定义输入参数 c，类型为 Float，不需要梯度，设备为 CPU
  %3 : int = prim::Constant[value=1]()  # 创建一个常量值为 1 的整数张量
  %6 : Float(requires_grad=0, device=cpu) = aten::add(%a.1, %b.1, %3)  # 计算 a + b，并存储在 %6 中
  %9 : Float(requires_grad=0, device=cpu) = aten::add(%6, %c.1, %3)  # 计算 %6 + c，并存储在 %9 中
  return (%9)  # 返回 %9，即结果张量

        """
        graph = torch._C.parse_ir(graph_str)  # 解析输入的 IR 字符串，生成计算图对象

        kernel = te.TensorExprKernel(graph)  # 使用解析后的计算图创建张量表达式内核对象
        res1 = kernel.run((x, y, z))  # 在张量 x, y, z 上运行内核，得到结果 res1
        res2 = kernel.fallback((x, y, z))  # 在张量 x, y, z 上运行内核的后备版本，得到结果 res2
        correct = f(x, y, z)  # 调用函数 f 计算正确的结果
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)  # 断言 res1 和正确结果的近似性
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)  # 断言 res2 和正确结果的近似性
    graph(%a : Tensor, %b : Tensor):
      %c : Tensor = aten::mul(%a, %b)
      return (%c)



        """
        使用 Torch 的 C++ API 解析 IR 表示的计算图
        graph = torch._C.parse_ir(graph_str)

        # 异常标志初始化为 False
        exception_thrown = False
        try:
            # 尝试创建基于张量表达式的内核
            kernel = te.TensorExprKernel(graph)
        except RuntimeError:
            # 如果运行时错误抛出，表明图中缺少输入的形状信息，编译应该失败
            exception_thrown = True
            pass
        assert exception_thrown

        # 注入形状信息并再次尝试编译
        example_inputs = [torch.rand(4, 4), torch.rand(4, 4)]
        torch._C._te.annotate_input_shapes(graph, example_inputs)
        torch._C._jit_pass_propagate_shapes_on_graph(graph)

        # 现在应该可以成功编译
        kernel = te.TensorExprKernel(graph)

        # 运行内核并比较结果
        res = kernel.run((x, y))
        correct = torch.mul(x, y)
        np.testing.assert_allclose(res.numpy(), correct.numpy(), atol=1e-5)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_shape_prop_module(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return x * x + y

        # 获取测试模块的图表示
        graph = torch.jit.script(TestModule()).graph

        # 尝试以原样编译图，预期应该失败因为缺少形状信息
        exception_thrown = False
        try:
            kernel = te.TensorExprKernel(graph)
        except RuntimeError:
            exception_thrown = True
            pass
        assert exception_thrown

        # 尝试为图的输入注入形状信息
        example_inputs = [torch.rand(4, 4), torch.rand(4, 4)]

        # 重新尝试注入形状信息，可能会失败如果图中有 'self' 参数
        exception_thrown = False
        try:
            torch._C._te.annotate_input_shapes(graph, example_inputs)
        except RuntimeError:
            # 图中包含无法设置形状的 'self' 参数
            exception_thrown = True
            pass
        assert exception_thrown

        # 移除图中的 'self' 参数并再次尝试注入形状信息
        torch._C._te.remove_unused_self_argument(graph)

        # 注入形状信息并再次尝试编译
        torch._C._te.annotate_input_shapes(graph, example_inputs)
        torch._C._jit_pass_propagate_shapes_on_graph(graph)

        # 现在应该可以成功编译
        kernel = te.TensorExprKernel(graph)

        # 准备输入数据并运行内核，与预期结果进行比较
        device, size = "cpu", (4, 4)
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)
        res = kernel.run((x, y))
        correct = TestModule().forward(x, y)
        np.testing.assert_allclose(res.numpy(), correct.numpy(), atol=1e-5)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_t(self):
        def f(a):
            return a.t()

        device, size = "cpu", (3, 4)
        x = torch.rand(size, device=device)

        # 下面是一个未完成的计算图字符串，待完成
        graph_str = """
# 定义一个 PyTorch 图形，描述一个矩阵转置操作
graph(%a.1 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %3 : Float(4, 3, strides=[4, 1], requires_grad=0, device=cpu) = aten::t(%a.1)
  return (%3)
        """
        # 使用 torch._C.parse_ir 方法解析给定的 IR 字符串，生成一个图形对象
        graph = torch._C.parse_ir(graph_str)

        # 创建一个 TensorExprKernel 对象，用于处理图形中的张量表达式
        kernel = te.TensorExprKernel(graph)

        # 使用 kernel 对象执行图形的计算，并获取结果
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))  # 备用执行路径

        # 调用函数 f 对输入 x 进行转置操作，作为正确的参考结果
        correct = f(x)

        # 使用 numpy.testing.assert_allclose 进行结果验证，设置容忍的绝对误差阈值
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_transpose(self):
        # 定义一个简单的函数 f，实现张量的转置操作
        def f(a):
            return a.transpose(-1, -2)

        # 设置设备和张量大小
        device, size = "cpu", (3, 4)
        x = torch.rand(size, device=device)

        # 定义一个 PyTorch 图形，描述一个转置操作的计算图
        graph_str = """
graph(%a.1 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %2 : int = prim::Constant[value=-1]()
  %3 : int = prim::Constant[value=-2]()
  %4 : Float(4, 3, strides=[4, 1], requires_grad=0, device=cpu) = aten::transpose(%a.1, %2, %3)
  return (%4)
        """
        # 使用 torch._C.parse_ir 方法解析给定的 IR 字符串，生成一个图形对象
        graph = torch._C.parse_ir(graph_str)

        # 创建一个 TensorExprKernel 对象，用于处理图形中的张量表达式
        kernel = te.TensorExprKernel(graph)

        # 使用 kernel 对象执行图形的计算，并获取结果
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))  # 备用执行路径

        # 调用函数 f 对输入 x 进行转置操作，作为正确的参考结果
        correct = f(x)

        # 使用 numpy.testing.assert_allclose 进行结果验证，设置容忍的绝对误差阈值
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_permute(self):
        # 定义一个简单的函数 f，实现张量的维度置换操作
        def f(a):
            return a.permute([2, 1, 0])

        # 设置设备和张量大小
        device, size = "cpu", (3, 4, 5)
        x = torch.rand(size, device=device)

        # 定义一个 PyTorch 图形，描述一个维度置换操作的计算图
        graph_str = """
graph(%a.1 : Float(3, 4, 5, strides=[20, 5, 1], requires_grad=0, device=cpu)):
  %1 : int = prim::Constant[value=2]()
  %2 : int = prim::Constant[value=1]()
  %3 : int = prim::Constant[value=0]()
  %4 : int[] = prim::ListConstruct(%1, %2, %3)
  %5 : Float(5, 4, 3, strides=[12, 3, 1], requires_grad=0, device=cpu) = aten::permute(%a.1, %4)
  return (%5)
        """
        # 使用 torch._C.parse_ir 方法解析给定的 IR 字符串，生成一个图形对象
        graph = torch._C.parse_ir(graph_str)

        # 创建一个 TensorExprKernel 对象，用于处理图形中的张量表达式
        kernel = te.TensorExprKernel(graph)

        # 使用 kernel 对象执行图形的计算，并获取结果
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))  # 备用执行路径

        # 调用函数 f 对输入 x 进行维度置换操作，作为正确的参考结果
        correct = f(x)

        # 使用 numpy.testing.assert_allclose 进行结果验证，设置容忍的绝对误差阈值
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_custom_lowering(self):
        # 定义一个简单的函数 f，实现将输入张量中的 NaN 替换为 0
        def f(a):
            return a.nan_to_num()

        # 设置设备类型
        device = "cpu"

        # 创建一个具有 NaN 值的输入张量 x
        x = torch.ones((2, 2), device=device)
        x[0, 0] = x[1, 1] = torch.nan

        # 定义一个 PyTorch 图形，描述一个自定义降阶操作的计算图
        graph_str = """
graph(%x : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu)):
    %none : NoneType = prim::Constant()
    %y : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::nan_to_num(%x, %none, %none, %none)
        """
        # 使用 torch._C.parse_ir 方法解析给定的 IR 字符串，生成一个图形对象
        graph = torch._C.parse_ir(graph_str)

        # 创建一个 TensorExprKernel 对象，用于处理图形中的张量表达式
        kernel = te.TensorExprKernel(graph)

        # 使用 kernel 对象执行图形的计算，并获取结果
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))  # 备用执行路径

        # 调用函数 f 对输入 x 进行自定义降阶操作，作为正确的参考结果
        correct = f(x)

        # 使用 numpy.testing.assert_allclose 进行结果验证，设置容忍的绝对误差阈值
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)
    return (%y)
        """
        graph = torch._C.parse_ir(graph_str)
        
        # 定义自定义的降低方法，接受输入、输出形状、步幅、数据类型和设备
        def my_custom_lowering(inputs, out_shape, out_stride, out_type, device):
            # 定义计算函数，处理索引
            def compute(idxs):
                # 从输入的缓冲区加载数据
                load = inputs[0].as_buf().load(idxs)
                # 如果加载的值为 NaN，则返回浮点数 0.0，否则返回加载的值
                return te.ifThenElse(
                    te.ExprHandle.isnan(load), te.ExprHandle.float(0.0), load
                )
            
            # 返回自定义计算的 Compute2 对象，命名为 "custom_nan_to_num"
            return te.Compute2("custom_nan_to_num", out_shape, compute)
        
        # 使用解析后的图形和自定义降低函数创建张量表达式内核
        kernel = te.TensorExprKernel(graph, {"aten::nan_to_num": my_custom_lowering})
        # 运行内核并获取结果 res1
        res1 = kernel.run((x,))
        # 使用内核的回退方法并获取结果 res2
        res2 = kernel.fallback((x,))
        # 计算正确的结果，使用给定的函数 f(x)
        correct = f(x)
        # 使用 numpy 测试库断言 res1 和 correct 的所有元素在给定容差范围内相等
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        # 使用 numpy 测试库断言 res2 和 correct 的所有元素在给定容差范围内相等
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    # 定义一个测试函数，测试带有 expand 操作的内核
    def test_kernel_with_expand(self):
        # 定义一个简单的函数 f，用于对输入张量 a 进行扩展操作
        def f(a):
            return a.expand((2, 3, 4))
        
        # 设置设备为 CPU，生成一个形状为 (1, 3, 1) 的随机张量 x
        device = "cpu"
        x = torch.rand((1, 3, 1), device=device)
        # 初始化 graph_str 为空字符串，此处后续代码未提供
        graph_str = """
# 定义一个名为 graph 的函数，接受一个 Float 类型的张量 %a 作为输入
graph(%a : Float(1, 3, 1, strides=[3, 1, 1], requires_grad=0, device=cpu)):
  # 创建一个整数常量 %1，其值为 2
  %1 : int = prim::Constant[value=2]()
  # 创建一个整数常量 %2，其值为 3
  %2 : int = prim::Constant[value=3]()
  # 创建一个整数常量 %3，其值为 4
  %3 : int = prim::Constant[value=4]()
  # 创建一个整数数组 %4，包含常量 %1, %2, %3
  %4 : int[] = prim::ListConstruct(%1, %2, %3)
  # 创建一个布尔常量 %5，其值为 False
  %5 : bool = prim::Constant[value=0]()
  # 调用 torch 中的 aten::expand 函数，使用输入张量 %a 和数组 %4 进行扩展
  %6 : Float(2, 3, 4, strides=[12, 4, 0], requires_grad=0, device=cpu) = aten::expand(%a, %4, %5)
  # 返回扩展后的张量 %6
  return (%6)
    def test_unary_ops(self):
        # 定义包含 Torch 和 TorchScript 对应的一元操作符映射字典
        unary_operators = {
            torch.sin: torch._C._te.sin,
            torch.cos: torch._C._te.cos,
            torch.tan: torch._C._te.tan,
            torch.asin: torch._C._te.asin,
            torch.acos: torch._C._te.acos,
            torch.atan: torch._C._te.atan,
            torch.sinh: torch._C._te.sinh,
            torch.cosh: torch._C._te.cosh,
            torch.tanh: torch._C._te.tanh,
            torch.sigmoid: torch._C._te.sigmoid,
            torch.exp: torch._C._te.exp,
            torch.expm1: torch._C._te.expm1,
            torch.abs: torch._C._te.abs,
            torch.log: torch._C._te.log,
            torch.log2: torch._C._te.log2,
            torch.log10: torch._C._te.log10,
            torch.log1p: torch._C._te.log1p,
            torch.erf: torch._C._te.erf,
            torch.erfc: torch._C._te.erfc,
            torch.sqrt: torch._C._te.sqrt,
            torch.rsqrt: torch._C._te.rsqrt,
            torch.ceil: torch._C._te.ceil,
            torch.floor: torch._C._te.floor,
            torch.round: torch._C._te.round,
            torch.trunc: torch._C._te.trunc,
            torch.lgamma: torch._C._te.lgamma,
            torch.frac: torch._C._te.frac,
        }

        # 定义一个内部函数，用于构造 TorchScript 函数
        def construct_te_fn(op, n: int, dtype=torch.float32):
            # 创建一个 TorchScript 缓冲区对象
            A = torch._C._te.BufHandle("A", [n], dtype)

            # 定义计算函数，对缓冲区中的元素进行操作
            def compute(i):
                return op(A.load([i]))

            # 创建一个计算对象 C，对应于上述计算函数
            C = te.Compute("C", [n], compute)

            # 创建一个循环嵌套对象，包含计算对象 C
            loopnest = te.LoopNest([C])
            loopnest.prepare_for_codegen()

            # 对根语句进行简化处理，并返回简化后的语句
            stmt = te.simplify(loopnest.root_stmt())

            # 使用简化后的语句构造一个 TorchScript 代码生成对象
            return te.construct_codegen("ir_eval", stmt, [A, C])

        # 定义测试中使用的元素数量
        n = 10
        # 创建一个包含随机数据的 Torch 张量
        a = torch.rand(n)

        # 对每个 Torch 和 TorchScript 一元操作符执行测试
        for torch_op, te_op in unary_operators.items():
            # 计算 Torch 操作符在随机数据上的参考结果
            ref = torch_op(a)

            # 构造对应 TorchScript 函数
            te_fn = construct_te_fn(te_op, n, torch.float32)

            # 创建一个空张量以接收 TorchScript 函数的结果
            res = torch.empty(n)

            # 调用 TorchScript 函数，并将结果存储在 res 中
            te_fn.call([a, res])

            # 使用 assert 断言比较 Torch 和 TorchScript 的结果是否接近
            assert torch.allclose(ref, res, atol=1e-3, rtol=1e-3)
# 如果这个脚本是作为主程序运行
if __name__ == "__main__":
    # 调用函数 run_tests() 来执行测试
    run_tests()
```