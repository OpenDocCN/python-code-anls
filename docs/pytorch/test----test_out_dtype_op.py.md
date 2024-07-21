# `.\pytorch\test\test_out_dtype_op.py`

```py
# 导入必要的模块和类
import unittest  # 导入unittest模块用于编写和运行测试
import torch  # 导入PyTorch库
import torch._dynamo  # 导入torch._dynamo模块，这是动态图功能的一部分
import torch._inductor  # 导入torch._inductor模块
import torch._inductor.decomposition  # 导入torch._inductor.decomposition模块中的所有内容
from torch._higher_order_ops.out_dtype import out_dtype  # 从torch._higher_order_ops.out_dtype模块中导入out_dtype函数
from torch.fx.experimental.proxy_tensor import make_fx  # 导入torch.fx.experimental.proxy_tensor模块中的make_fx函数
from torch.testing._internal.common_utils import (  # 导入torch.testing._internal.common_utils模块中的指定函数和变量
    run_tests, TestCase, IS_WINDOWS, TEST_WITH_ROCM, IS_FBCODE, IS_REMOTE_GPU, TEST_CUDA
)
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport  # 导入torch.testing._internal.common_quantization模块中的skipIfNoDynamoSupport函数
from torch.testing import FileCheck  # 导入torch.testing模块中的FileCheck类
from torch.testing._internal.common_cuda import SM80OrLater, _get_torch_cuda_version  # 导入torch.testing._internal.common_cuda模块中的指定函数和变量

# 装饰器，如果dynamo不支持，则跳过测试
@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't support")
class TestOutDtypeOp(TestCase):
    def test_out_dtype_make_fx(self):
        # 定义一个简单的模型类M，接受一个权重参数
        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                # 返回使用out_dtype函数对torch.ops.aten.mm.default操作进行类型推断的结果
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, x, self.weight
                )

        # 创建一个5x5的随机int8张量作为权重
        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        # 实例化模型M
        m = M(weight)
        # 创建一个5x5的随机int8张量x作为输入
        x = torch.randint(-128, 127, (5, 5), dtype=torch.int8)

        # 使用make_fx函数将模型M转换为图模式，并将输入x传递给它
        gm = make_fx(m)(x)
        # 断言模型M在输入x下的输出与转换后的图模式gm在输入x下的输出接近
        self.assertTrue(torch.allclose(m(x), gm(x)))

        # 使用torch.func.functionalize将模型M转换为函数，并使用make_fx对其进行图模式转换
        gm = make_fx(torch.func.functionalize(M(weight)))(x)
        # 再次断言模型M在输入x下的输出与转换后的图模式gm在输入x下的输出接近
        self.assertTrue(torch.allclose(m(x), gm(x)))

        # 使用FileCheck对象检查生成的图模式gm中是否包含特定字符串"torch.ops.higher_order.out_dtype"和"aten.mm.default"
        FileCheck().check("torch.ops.higher_order.out_dtype").check("aten.mm.default").run(gm.code)
        # 再次断言模型M在输入x下的输出与转换后的图模式gm在输入x下的输出接近
        self.assertTrue(torch.allclose(m(x), gm(x)))

        # 遍历图模式gm中的每个节点
        for node in gm.graph.nodes:
            # 如果节点是"call_function"操作并且目标是out_dtype函数
            if node.op == "call_function" and node.target is out_dtype:
                # 断言该节点的结果应为int32类型
                self.assertTrue(node.meta["val"].dtype, torch.int32)
                # 断言该节点的第三个参数应为int8类型
                self.assertTrue(node.args[2].meta["val"].dtype, torch.int8)
    def test_out_dtype_op_functional(self):
        # 定义一个测试函数，用于测试某个操作的功能
        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                # 在前向传播中调用 out_dtype 函数，对输入 x 和权重 self.weight 执行 torch.ops.aten.mm.default 操作
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, x, self.weight
                )

        # 创建一个随机的 int8 类型的权重矩阵
        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        # 实例化模型 M
        m = M(weight)
        # 创建一个与权重相同的随机 int8 输入张量 x
        x = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        # 导出模型 m 的计算图
        ep = torch.export.export(
            m,
            (x,),
        )
        # 使用 FileCheck 检查导出的计算图的代码，确保包含指定的字符串
        FileCheck().check("torch.ops.higher_order.out_dtype").check("aten.mm.default").run(ep.graph_module.code)
        # 断言模型 m 在输入 x 上的输出与导出模型 ep 在相同输入上的输出在数值上全部相近
        self.assertTrue(torch.allclose(m(x), ep.module()(x)))
        # 遍历导出模型 ep 的所有节点
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target is out_dtype:
                # 断言这个节点的结果应该是 int32 类型
                self.assertTrue(node.meta["val"].dtype, torch.int32)
                # 断言这个节点的第三个参数应该是 int8 类型
                self.assertTrue(node.args[2].meta["val"].dtype, torch.int8)

    def test_out_dtype_mm_numerical(self):
        # 定义一个测试函数，用于测试 mm 操作的数值计算
        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                # 在前向传播中调用 out_dtype 函数，对输入 x 和权重 self.weight 执行 torch.ops.aten.mm.default 操作
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, x, self.weight
                )

        # 创建一个随机的 int8 类型的权重矩阵
        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        # 实例化模型 M
        m = M(weight)
        # 创建一个与权重相同的随机 int8 输入张量 x
        x = torch.randint(-128, 127, (5, 5), dtype=torch.int8)

        # 使用 TorchScript 生成模型 m 的图模式表示
        gm = make_fx(m)(x)

        # 将输入张量 x 和权重 weight 转换为 int32 类型，执行数值计算
        x_casted = x.to(torch.int32)
        weight_casted = weight.to(torch.int32)
        numerical_res = torch.ops.aten.mm.default(x_casted, weight_casted)
        # 断言数值计算的结果与模型 m 在相同输入 x 上的输出在数值上全部相近
        self.assertTrue(torch.allclose(numerical_res, gm(x)))

    def test_out_dtype_dynamo(self):
        # 定义一个测试函数，测试动态函数的 out_dtype 操作
        def f(x, y):
            # 调用 out_dtype 函数，对输入 x 和 y 执行 torch.ops.aten.mul.Scalar 操作
            return out_dtype(
                torch.ops.aten.mul.Scalar, torch.int32, x, y
            )

        # 创建一个包含 int8 类型张量和 float 类型标量的输入元组
        inp = (torch.randint(-128, 127, (5, 5), dtype=torch.int8), 3.0)

        # 编译函数 f，使用 eager 后端并生成完整的计算图
        compiled = torch.compile(f, backend="eager", fullgraph=True)
        # 断言函数 f 在输入 inp 上的输出与编译后函数 compiled 在相同输入上的输出在数值上全部相近
        self.assertTrue(torch.allclose(f(*inp), compiled(*inp)))

    def test_out_dtype_mul_scalar_numerical(self):
        # 定义一个测试函数，测试 mul.Scalar 操作的数值计算
        def f(x, y):
            # 调用 out_dtype 函数，对输入 x 和 y 执行 torch.ops.aten.mul.Scalar 操作
            return out_dtype(
                torch.ops.aten.mul.Scalar, torch.int32, x, y
            )

        # 创建一个包含 int8 类型张量和 float 类型标量的输入元组
        inp = (torch.randint(-128, 127, (5, 5), dtype=torch.int8), 3.0)

        # 使用 TorchScript 生成函数 f 的图模式表示
        gm = make_fx(f)(*inp)
        # 将输入张量 inp[0] 转换为 int32 类型，执行数值计算
        numerical_res = torch.ops.aten.mul.Scalar(inp[0].to(dtype=torch.int32), 3)
        # 断言数值计算的结果与 TorchScript 生成的函数 f 在相同输入 inp 上的输出在数值上全部相近
        self.assertTrue(torch.allclose(numerical_res, gm(*inp)))
    # 测试非功能性输出数据类型的情况
    def test_out_dtype_non_functional(self):
        # 定义一个简单的 PyTorch 模块类
        class M(torch.nn.Module):
            # 定义模块的前向传播方法
            def forward(self, x, y):
                # 调用 out_dtype 函数，返回 torch.add_ 的运算结果
                return out_dtype(
                    torch.ops.aten.add_.Tensor, torch.int32, x, y
                )

        # 使用 assertRaisesRegex 断言捕获 ValueError 异常，并验证异常消息
        with self.assertRaisesRegex(ValueError, "out_dtype's first argument needs to be a functional operator"):
            # 导出模型 M，验证导出过程中的异常
            _ = torch.export.export(
                M(), (torch.randint(-128, 127, (5, 5), dtype=torch.int8), torch.randint(-128, 127, (5, 5), dtype=torch.int8)),
            )

    # 测试非操作重载的输出数据类型的情况
    def test_out_dtype_non_op_overload(self):
        # 定义一个简单的函数 f
        def f(x, y):
            # 调用 out_dtype 函数，返回 torch.add 的运算结果
            return out_dtype(
                torch.add, torch.int32, x, y
            )

        # 使用 assertRaisesRegex 断言捕获 ValueError 异常，并验证异常消息
        with self.assertRaisesRegex(ValueError, "out_dtype's first argument must be an OpOverload"):
            # 调用函数 f，验证异常
            f(torch.randint(-128, 127, (5, 5), dtype=torch.int8), torch.randint(-128, 127, (5, 5), dtype=torch.int8))

    # 测试没有自动求导功能的输出数据类型的情况
    def test_out_dtype_no_autograd(self):
        # 定义一个简单的函数 f
        def f(x, y):
            # 调用 out_dtype 函数，返回 torch.ops.aten.mm.default 的运算结果
            return out_dtype(
                torch.ops.aten.mm.default, torch.int32, x, y
            )

        # 创建具有梯度的输入张量对
        inp = (torch.randn(5, 5, requires_grad=True), torch.randn(5, 5, requires_grad=True))
        # 调用函数 f，触发错误但错误被延迟处理

        f(*inp)

        # 使用 torch.no_grad() 上下文管理器调用函数 f
        with torch.no_grad():
            f(*inp)

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并验证异常消息
        with self.assertRaisesRegex(RuntimeError, "does not require grad and does not have a grad_fn"):
            # 调用函数 f，计算损失并执行反向传播
            out = f(*inp)
            loss = out - torch.ones(out.shape)
            loss.backward()

    # 用于测试输出数据类型的感应电感解构情况
    @unittest.skipIf(IS_WINDOWS, "_int_mm unavailable")
    @unittest.skipIf(TEST_WITH_ROCM, "_int_mm unavailable")
    @unittest.skipIf(not SM80OrLater, "_int_mm unavailable")
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "cublas runtime error")
    @unittest.skipIf(_get_torch_cuda_version() >= (11, 7), "_int_mm unavailable")
    @unittest.skipIf(not TEST_CUDA, "_int_mm unavailable")
    @skipIfNoDynamoSupport
    def test_out_dtype_inductor_decomp(self) -> None:
        # 定义一个简单的函数 func
        def func(x, w):
            # 调用 out_dtype 函数，返回 torch.ops.aten.mm.default 的运算结果
            return out_dtype(torch.ops.aten.mm.default, torch.int32, x, w)

        # 在 CUDA 设备上生成随机输入张量 x 和权重张量 w
        w = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")
        x = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")

        # 使用 torch._int_mm 计算参考结果 ref
        ref = torch._int_mm(x, w)
        # 调用函数 func，计算测试输出 test_out
        test_out = func(x, w)
        # 使用 torch.compile 编译函数 func，使用 fullgraph=True 和 mode="max-autotune"
        func_comp = torch.compile(func, fullgraph=True, mode="max-autotune")
        # 调用编译后的函数 func_comp，计算测试输出 test_out_c
        test_out_c = func_comp(x, w)
        # 使用 assertTrue 断言验证 test_out 和 ref 的近似程度
        self.assertTrue(torch.allclose(ref, test_out))
        # 使用 assertTrue 断言验证 test_out_c 和 ref 的近似程度
        self.assertTrue(torch.allclose(ref, test_out_c))

    # 仅在 CUDA 环境下测试
    @unittest.skipIf(not TEST_CUDA, "cuda only")
    # 定义一个测试方法，用于测试 out_dtype_inductor_decomp_trace 函数
    def test_out_dtype_inductor_decomp_trace(self) -> None:
        # 定义一个内部函数 func，接受两个参数 x 和 w，并调用 out_dtype 函数
        def func(x, w):
            return out_dtype(torch.ops.aten.mm.default, torch.int32, x, w)

        # 使用 torch.randint 生成在 [-128, 127] 范围内的随机整数张量 w 和 x，
        # 数据类型为 torch.int8，在 CUDA 设备上分配内存
        w = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")
        x = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")

        # 检查 make_fx 函数在使用“symbolic”追踪模式下，
        # 通过 inductor decomps 生成 _int_mm 的情况
        decomp_table = torch._inductor.decomposition.select_decomp_table()
        # 使用 make_fx 创建一个图模块 gm，并传入 x 和 w 作为参数
        gm = make_fx(func, decomp_table, tracing_mode="symbolic")(x, w)
        # 使用 self.assertExpectedInline 方法断言 gm 生成的代码符合预期
        self.assertExpectedInline(gm.code.strip(), """\
# 定义一个方法，用于计算两个张量的矩阵乘法，并返回结果
def forward(self, x_1, w_1):
    # 调用 torch.ops.aten._int_mm.default 方法进行矩阵乘法运算
    _int_mm = torch.ops.aten._int_mm.default(x_1, w_1);  x_1 = w_1 = None
    # 返回矩阵乘法的结果
    return _int_mm

@unittest.skipIf(not TEST_CUDA, "cuda only")
def test_out_dtype_int_mm_default_trace(self) -> None:
    # 定义一个函数 func，接受两个参数 x 和 w，并返回 torch.ops.aten.mm.default 的输出结果
    def func(x, w):
        return out_dtype(torch.ops.aten.mm.default, torch.int32, x, w)

    # 生成随机张量 w 和 x，类型为 torch.int8，设备为 CUDA
    w = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")
    x = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")

    # 在符号化跟踪模式下，生成函数的 TorchScript 表示
    gm = make_fx(func, tracing_mode="symbolic")(x, w)
    # 断言 TorchScript 代码的期望输出
    self.assertExpectedInline(gm.code.strip(), """\
def forward(self, x_1, w_1):
    # 调用 torch.ops.higher_order.out_dtype 方法，传入 torch.ops.aten.mm.default、torch.int32、x_1、w_1 参数，返回结果存储在 out_dtype 中
    out_dtype = torch.ops.higher_order.out_dtype(torch.ops.aten.mm.default, torch.int32, x_1, w_1);  x_1 = w_1 = None
    # 返回 out_dtype 变量
    return out_dtype""")

# 定义一个测试方法，用于验证特定情况下 out_dtype 函数的行为
def test_out_dtype_wrong_output(self) -> None:
    # 定义一个函数 multiple_out，接受一个参数 x，尝试使用 out_dtype 函数调用 torch.ops.aten.topk.default，返回 topk 操作的输出结果
    def multiple_out(x):
        return out_dtype(
            torch.ops.aten.topk.default, torch.int32, x, 5
        )

    # 随机生成一个张量 inp
    inp = (torch.randn(10),)

    # 使用断言验证 ValueError 异常是否被正确抛出，提示信息为 "out_dtype's can only apply to ops that return a single tensor"
    with self.assertRaisesRegex(ValueError, "out_dtype's can only apply to ops that return a single tensor"):
        multiple_out(*inp)

    # 定义一个函数 singleton_list_out，接受一个参数 x，尝试使用 out_dtype 函数调用 torch.ops.aten.split_copy.Tensor，返回 split_copy 操作的输出结果
    def singleton_list_out(x):
        return out_dtype(
            torch.ops.aten.split_copy.Tensor, torch.int32, x, 10
        )

    # 使用断言验证 ValueError 异常是否被正确抛出，提示信息为 "out_dtype's can only apply to ops that return a single tensor"
    with self.assertRaisesRegex(ValueError, "out_dtype's can only apply to ops that return a single tensor"):
        singleton_list_out(*inp)

# 当文件作为主程序运行时，执行测试
if __name__ == '__main__':
    run_tests()
```