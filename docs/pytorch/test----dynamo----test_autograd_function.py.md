# `.\pytorch\test\dynamo\test_autograd_function.py`

```py
# Owner(s): ["module: dynamo"]
# flake8: noqa: B950

# 导入必要的库和模块
import copy  # 导入copy模块，用于复制对象
import math  # 导入math模块，提供数学运算函数

from dataclasses import dataclass  # 导入dataclass装饰器，用于创建数据类

import torch  # 导入PyTorch深度学习框架

import torch._dynamo.test_case  # 导入PyTorch的私有测试用例模块
import torch._dynamo.testing  # 导入PyTorch的私有测试模块
import torch._dynamo.utils  # 导入PyTorch的私有实用工具模块
from torch.testing._internal.triton_utils import HAS_CUDA, requires_cuda  # 导入CUDA相关测试工具

if HAS_CUDA:
    import triton  # 如果支持CUDA，则导入Triton加速库

    from torch.testing._internal.triton_utils import add_kernel  # 导入加速内核函数

# 自定义的PyTorch自动求导函数类，重载forward和backward方法
class CustomFunc1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        return foo + foo  # 前向传播：将输入foo加倍并返回

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # 反向传播：传递梯度输出

# 自定义的PyTorch自动求导函数类，测试前向传播中的图断裂
class CustomFunc3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        result = foo + foo  # 前向传播：将输入foo加倍
        torch._dynamo.graph_break()  # 前向传播中断图
        result = result + foo  # 继续前向传播：将加倍后的结果再次加上原始输入foo
        ctx.save_for_backward(result)  # 保存结果以备反向传播使用
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors  # 获取保存的结果张量
        return grad_output * math.sqrt(result.numel())  # 反向传播：根据结果张量的元素数量开方乘以梯度输出

# 继承自torch.nn.Module的模块类，使用CustomFunc1进行前向传播
class Module1(torch.nn.Module):
    def forward(self, foo):
        return CustomFunc1().apply(foo)

# 继承自torch.nn.Module的模块类，使用CustomFunc1.apply方法进行前向传播
class Module2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = CustomFunc1.apply  # 初始化自定义函数

    def forward(self, foo):
        return self.fn(foo)  # 调用自定义函数的前向传播

# 继承自torch.nn.Module的模块类，使用CustomFunc1进行前向传播
class Module3(torch.nn.Module):
    def forward(self, foo):
        return CustomFunc1().apply(foo)

# 继承自torch.nn.Module的模块类，使用CustomFunc1.apply方法进行前向传播
class Module4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = CustomFunc1.apply  # 初始化自定义函数

    def forward(self, foo):
        return self.fn(foo)  # 调用自定义函数的前向传播

# 继承自torch.nn.Module的模块类，使用CustomFunc3进行前向传播
class Module5(torch.nn.Module):
    def forward(self, foo):
        return CustomFunc3().apply(foo)

# 继承自torch.nn.Module的模块类，使用CustomFunc3.apply方法进行前向传播
class Module6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = CustomFunc3.apply  # 初始化自定义函数

    def forward(self, foo):
        return self.fn(foo)  # 调用自定义函数的前向传播

# 继承自torch.autograd.Function的线性函数类，实现线性变换的前向和反向传播
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, bias):
        output = input.mm(weight.t())  # 计算输入和权重的矩阵乘法
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)  # 如果存在偏置，则加上偏置
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs  # 解包输入元组
        ctx.save_for_backward(input, weight, bias)  # 保存输入、权重和偏置以备反向传播使用
    `
    # 定义反向传播函数，计算输入的梯度
    def backward(ctx, grad_output):
        # 从上下文中恢复保存的张量：输入、权重、偏置
        input, weight, bias = ctx.saved_tensors
        # 初始化梯度变量
        grad_input = grad_weight = grad_bias = None
        # 如果需要计算输入的梯度
        if ctx.needs_input_grad[0]:
            # 计算输入的梯度：输出梯度乘以权重矩阵的转置
            grad_input = grad_output.mm(weight)
        # 如果需要计算权重的梯度
        if ctx.needs_input_grad[1]:
            # 计算权重的梯度：输出梯度的转置乘以输入矩阵
            grad_weight = grad_output.t().mm(input)
        # 如果存在偏置并且需要计算偏置的梯度
        if bias is not None and ctx.needs_input_grad[2]:
            # 计算偏置的梯度：输出梯度在第一维度上求和
            grad_bias = grad_output.sum(0)
    
                  # 计算偏置梯度，对 grad_output 沿第一维度求和
                grad_bias = grad_output.sum(0)
    
            # 返回计算得到的梯度：输入梯度 grad_input，权重梯度 grad_weight，偏置梯度 grad_bias
            return grad_input, grad_weight, grad_bias
class ModuleLinear(torch.nn.Module):
    def forward(self, input, weight, bias=None):
        return LinearFunction.apply(input, weight, bias)



class MaterializingGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 设置梯度材料化为关闭状态
        ctx.set_materialize_grads(False)
        # 返回输入张量的克隆副本
        return x.clone(), x.clone()

    @staticmethod
    def backward(ctx, grad_out1, grad_out2):
        # 后向传播时直接返回传入的梯度
        return grad_out1, grad_out2



class MaterializingGradModule(torch.nn.Module):
    def forward(self, x):
        # 应用自定义的梯度函数
        return MaterializingGradFunction.apply(x)



class CustomFuncBwdPrintGraphBreak(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        # 返回输入张量的加法结果
        return torch.add(foo, foo)

    @staticmethod
    def backward(ctx, grad_output):
        # 在反向传播时打印信息并返回梯度
        print("graph break!")
        return grad_output



class CustomFuncBwdPrintModule(torch.nn.Module):
    def forward(self, x):
        # 应用自定义的梯度函数
        return CustomFuncBwdPrintGraphBreak.apply(x)



class CustomFuncStrideBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        # 返回输入张量的加法结果
        return torch.add(foo, foo)

    @staticmethod
    def backward(ctx, grad_output):
        # 返回梯度的步幅信息
        return grad_output.stride()



class CustomFuncStrideModule(torch.nn.Module):
    def forward(self, x):
        # 应用自定义的梯度函数
        return CustomFuncStrideBwd.apply(x)



class CustomFuncSaveForBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        # 计算输入张量的加法结果
        result = foo + foo
        result = result + foo
        # 在上下文中保存计算结果以便后向传播使用
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # 恢复保存的结果并返回梯度乘以结果张量元素数量的平方根
        (result,) = ctx.saved_tensors
        return grad_output * math.sqrt(result.numel())



class SaveForBwdModule(torch.nn.Module):
    def forward(self, foo):
        # 应用自定义的梯度函数
        return CustomFuncSaveForBwd().apply(foo)



class ContextSaveAndMark(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 使用无梯度计算环境保存输入张量并标记为不可微分
        with torch.no_grad():
            ctx.save_for_backward(x)
            ctx.mark_non_differentiable(x)
            return x

    @staticmethod
    def backward(ctx, grad_output):
        # 直接返回梯度
        return grad_output



class ContextMarkAndSave(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 使用无梯度计算环境标记输入张量为不可微分并保存
        with torch.no_grad():
            ctx.mark_non_differentiable(x)
            ctx.save_for_backward(x)
            return x

    @staticmethod
    def backward(ctx, grad_output):
        # 直接返回梯度
        return grad_output



class ModuleWithGradFunc(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        # 初始化模块时保存自定义函数的应用
        self.f = func.apply

    def forward(self, x):
        # 调用保存的自定义函数应用来处理输入张量
        return self.f(x)



class AutogradFunctionTests(torch._dynamo.test_case.TestCase):
    # Sound behaviors, tested for working capture
    # 测试自动求导函数等价性的函数
    def test_autograd_function_equivalence(self):
        # 针对梯度是否开启进行迭代测试
        for grad in [True, False]:
            # 对模型 1 到 4 进行迭代测试
            for i in range(1, 5):
                # 重置 Torch 动态系统状态
                torch._dynamo.reset()
                # 根据模型名称动态创建模型对象
                model = globals()[f"Module{i}"]()
                # 对模型应用即时优化，并获得优化后的模型
                opt_model = torch._dynamo.optimize("eager")(model)
                # 断言优化后模型对输入张量计算结果与预期值的接近程度
                self.assertTrue(
                    torch.allclose(
                        opt_model(torch.ones(2, 3, requires_grad=grad)),
                        torch.tensor([2.0], requires_grad=grad),
                    )
                )

    # 测试自动求导函数中是否存在图结构断裂的函数
    def test_autograd_function_has_graph_break(self):
        # 针对梯度是否开启进行迭代测试
        for grad in [True, False]:
            # 创建随机张量 x，并根据不同的模型进行测试
            x = torch.randn(10, requires_grad=grad)
            for model in [Module5(), Module6()]:
                # 重置 Torch 动态系统状态
                torch._dynamo.reset()
                # 创建编译计数器对象 cnts，并应用到模型上
                cnts = torch._dynamo.testing.CompileCounter()
                opt_model = torch._dynamo.optimize(cnts)(model)
                # 多次运行模型和优化后模型，断言结果的接近程度
                for _ in range(3):
                    ref = model(x)
                    res = opt_model(x)
                    self.assertTrue(torch.allclose(ref, res))
                # 断言编译计数器中的帧数为 2
                self.assertEqual(cnts.frame_count, 2)

    # 测试线性设置上下文的函数
    def test_linear_setup_context(self):
        # 创建线性模型对象
        model = ModuleLinear()
        # 对模型应用即时优化，并获得优化后的模型
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)
        # 创建双重浮点张量输入和权重
        input = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        weight = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
        # 在原始模型和优化后模型上执行计算，并断言结果相等
        eager_result = model(input, weight)
        optim_result = opt_model(input, weight)
        self.assertEqual(optim_result, eager_result)

    # 测试梯度材料化的函数
    def test_materialize_grad(self):
        # 创建梯度材料化模型对象
        model = MaterializingGradModule()
        # 对模型应用即时优化，并获得优化后的模型
        opt_model = torch._dynamo.optimize("eager")(model)
        # 创建双重浮点张量输入 x
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        # 在原始模型和优化后模型上执行计算，并断言结果相等
        optim_result = opt_model(x)
        eager_result = model(x)
        self.assertEqual(optim_result, eager_result)

    # 测试反向传播中的打印功能
    def test_print_in_bwd(self):
        # 创建包含自定义反向传播打印功能的模型对象
        model = CustomFuncBwdPrintModule()
        # 对模型应用即时优化，并获得优化后的模型
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)
        # 创建双重浮点张量输入 x
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        # 断言在优化后的模型上执行时会引发指定异常
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "builtin: print"):
            opt_model(x)

    # 测试反向传播中的步幅问题
    def test_stride_in_bwd(self):
        # 清空 Torch 动态系统中的计数器
        torch._dynamo.utils.counters.clear()
        # 创建编译计数器对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 创建包含自定义反向传播步幅问题的模型对象
        model = CustomFuncStrideModule()
        # 编译模型并获得优化后的模型
        opt_model = torch.compile(backend=cnt)(model)
        # 创建双重浮点张量输入 x
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        # 对原始模型和优化后模型执行计算，并断言结果相等
        ref = model(x)
        res = opt_model(x)
        self.assertEqual(ref, res)
        # 断言计数器中的帧数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言存在图结构断裂的计数与预期相符
        self.assertEqual(
            list(torch._dynamo.utils.counters["graph_break"].values()), [1]
        )
    def test_enum_arg(self):
        from enum import Enum  # 导入枚举类型模块

        class SomeEnum(Enum):  # 定义一个枚举类 SomeEnum
            A = 0
            B = 1

        class Foo(torch.autograd.Function):  # 定义一个继承自 torch.autograd.Function 的类 Foo
            @staticmethod
            def forward(ctx, x, e):  # 前向传播函数，接收输入张量 x 和枚举类型 e
                if e is SomeEnum.A:  # 如果 e 是 SomeEnum 枚举类型的 A
                    return x.sin()  # 返回 x 的正弦值
                else:
                    return x.cos()  # 返回 x 的余弦值

            @staticmethod
            def backward(ctx, g):  # 反向传播函数，接收梯度张量 g
                return g  # 直接返回梯度

        @torch.compile(backend="eager", fullgraph=True)
        def f(x, enum):  # 定义一个装饰过的函数 f，接收输入张量 x 和枚举类型 enum
            output = Foo.apply(  # 调用 Foo 类的 apply 方法进行操作
                x,
                enum,
            )
            return output  # 返回操作后的输出

        x = torch.tensor([[1.0, 2, 3], [4, 5, 6]], requires_grad=True)  # 创建一个张量 x
        y = f(x, SomeEnum.A)  # 调用 f 函数，传入张量 x 和 SomeEnum.A 枚举类型
        self.assertEqual(y, x.sin())  # 断言 y 等于 x 的正弦值

    def test_save_for_bwd(self):
        model = SaveForBwdModule()  # 创建一个 SaveForBwdModule 类的实例 model
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)  # 优化模型 model
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)  # 创建一个双精度浮点型张量 x，需要梯度
        opt_model(x)  # 对 x 进行优化

    def test_allow_in_graph(self):
        torch._dynamo.utils.counters.clear()  # 清除计数器
        cnt = torch._dynamo.testing.CompileCounter()  # 创建编译计数器 cnt

        @torch._dynamo.allow_in_graph  # 允许在图中使用装饰器
        class AllowInGraphFunc(torch.autograd.Function):  # 定义一个允许在图中使用的自动求导函数类
            @staticmethod
            def forward(ctx, x):  # 前向传播函数，接收输入张量 x
                torch._dynamo.graph_break()  # 打破图的生成
                ctx.x0 = x.size(0)  # 保存张量 x 的大小
                return x * 2  # 返回张量 x 的两倍

            @staticmethod
            def backward(ctx, grad_out):  # 反向传播函数，接收输出梯度 grad_out
                return grad_out * ctx.x0  # 返回输出梯度乘以 ctx.x0

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):  # 定义一个装饰过的函数 fn，接收输入张量 x
            return AllowInGraphFunc.apply(x)  # 调用 AllowInGraphFunc 类的 apply 方法进行操作

        x = torch.rand(2, 3, requires_grad=True)  # 创建一个随机张量 x，需要梯度
        result = fn(x)  # 调用 fn 函数，传入张量 x
        self.assertEqual(result, AllowInGraphFunc.apply(x))  # 断言结果等于 AllowInGraphFunc 类的 apply 方法对 x 的操作结果
        self.assertEqual(cnt.frame_count, 1)  # 断言计数器的帧数为 1

    def test_once_differentiable(self):
        from torch.autograd.function import once_differentiable  # 导入一次可微函数

        torch._dynamo.utils.counters.clear()  # 清除计数器
        cnt = torch._dynamo.testing.CompileCounter()  # 创建编译计数器 cnt

        class ScaleGradient(torch.autograd.Function):  # 定义一个缩放梯度的自动求导函数类
            @staticmethod
            def forward(ctx, x):  # 前向传播函数，接收输入张量 x
                return x  # 直接返回输入张量 x

            @staticmethod
            @once_differentiable
            def backward(ctx, grad):  # 反向传播函数，接收梯度 grad
                return grad * 0.5  # 返回梯度乘以 0.5

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):  # 定义一个装饰过的函数 fn，接收输入张量 x
            return ScaleGradient.apply(x)  # 调用 ScaleGradient 类的 apply 方法进行操作

        x = torch.randn(3, requires_grad=True)  # 创建一个随机张量 x，需要梯度
        result = fn(x)  # 调用 fn 函数，传入张量 x
        self.assertEqual(result, ScaleGradient.apply(x))  # 断言结果等于 ScaleGradient 类的 apply 方法对 x 的操作结果
        self.assertEqual(cnt.frame_count, 1)  # 断言计数器的帧数为 1
    # 定义一个测试类中的类方法，继承自torch.autograd.Function
    def test_classmethod(self):
        # 定义一个名为Shake的自定义Function类
        class Shake(torch.autograd.Function):
            # 类方法：前向传播函数
            @classmethod
            def forward(cls, ctx, foo):
                return foo + foo

            # 类方法：反向传播函数
            @classmethod
            def backward(cls, ctx, grad_output):
                return grad_output

        # 定义一个函数f，调用Shake类的apply方法
        def f(x):
            return Shake.apply(x)

        # 生成一个4x4x4x4形状的张量x，需要梯度信息
        x = torch.randn(4, 4, 4, 4, requires_grad=True)
        # 使用eager模式编译f函数
        opt_m = torch.compile(backend="eager")(f)
        # 执行编译后的函数opt_m，并传入张量x
        opt_m(x)

    # 定义测试函数，测试带有ContextSaveAndMark的ModuleWithGradFunc模块
    def test_function_context_save_and_mark(self):
        # 创建一个ModuleWithGradFunc对象mod，使用ContextSaveAndMark作为参数
        mod = ModuleWithGradFunc(ContextSaveAndMark)
        # 准备传递给模块mod的参数和关键字参数
        args, kwargs = ([torch.rand([1])], {})
        # 调用mod模块，记录调用前的返回值
        before = mod(*args, **kwargs)

        # 重置torch._dynamo状态
        torch._dynamo.reset()
        # 使用eager模式优化模块mod，生成compiled_model
        compiled_model = torch._dynamo.optimize("eager")(mod)
        # 调用优化后的compiled_model模块，记录调用后的返回值
        after = compiled_model(*args, **kwargs)
        # 断言调用前后的返回值相等
        self.assertEqual(before, after)

    # 定义测试函数，测试带有ContextMarkAndSave的ModuleWithGradFunc模块
    def test_function_context_mark_and_save(self):
        # 创建一个ModuleWithGradFunc对象mod，使用ContextMarkAndSave作为参数
        mod = ModuleWithGradFunc(ContextMarkAndSave)
        # 准备传递给模块mod的参数和关键字参数
        args, kwargs = ([torch.rand([1])], {})
        # 调用mod模块，记录调用前的返回值
        before = mod(*args, **kwargs)

        # 重置torch._dynamo状态
        torch._dynamo.reset()
        # 使用eager模式优化模块mod，生成compiled_model
        compiled_model = torch._dynamo.optimize("eager")(mod)
        # 调用优化后的compiled_model模块，记录调用后的返回值
        after = compiled_model(*args, **kwargs)
        # 断言调用前后的返回值相等
        self.assertEqual(before, after)

    # 定义测试函数，测试具有多个输出的情况
    def test_multi_output(self):
        # 清空torch._dynamo.utils.counters计数器
        torch._dynamo.utils.counters.clear()
        # 创建一个CompileCounter对象cnt
        cnt = torch._dynamo.testing.CompileCounter()

        # 定义一个名为Foo的自定义Function类
        class Foo(torch.autograd.Function):
            # 静态方法：前向传播函数
            @staticmethod
            def forward(ctx, x):
                return x.clone(), x.clone()

            # 静态方法：反向传播函数
            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1 + grad2

        # 使用cnt作为后端，fullgraph=True编译f函数
        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return Foo.apply(x)

        # 生成一个形状为(3,)的张量x，需要梯度信息
        x = torch.randn(3, requires_grad=True)
        # 调用编译后的函数f，并传入张量x
        result = f(x)

        # 断言编译结果与Foo.apply(x)的结果相等
        self.assertEqual(result, Foo.apply(x))
        # 断言frame_count计数为1
        self.assertEqual(cnt.frame_count, 1)

    # 定义测试函数，测试带有自定义前向和反向传播的情况
    def test_amp_custom_fwd_bwd(self):
        # 清空torch._dynamo.utils.counters计数器
        torch._dynamo.utils.counters.clear()
        # 创建一个CompileCounter对象cnt
        cnt = torch._dynamo.testing.CompileCounter()

        # 定义一个名为MyMM的自定义Function类
        class MyMM(torch.autograd.Function):
            # 静态方法：自定义前向传播函数
            @staticmethod
            @torch.amp.custom_fwd(device_type="cuda")
            def forward(ctx, a, b):
                # 保存张量a和b的信息到上下文
                ctx.save_for_backward(a, b)
                # 返回a与b的矩阵乘积
                return a.mm(b)

            # 静态方法：自定义反向传播函数
            @staticmethod
            @torch.amp.custom_bwd(device_type="cuda")
            def backward(ctx, grad):
                # 获取保存的张量a和b
                a, b = ctx.saved_tensors
                # 返回grad与b的转置矩阵的乘积，以及a的转置矩阵与grad的乘积
                return grad.mm(b.t()), a.t().mm(grad)

        # 使用cnt作为后端，fullgraph=True编译fn函数
        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            return MyMM.apply(a, b)

        # 生成一个形状为(64, 64)的浮点型张量a，需要梯度信息
        a = torch.randn([64, 64], dtype=torch.float32, requires_grad=True)
        # 克隆张量a作为梯度
        grad = a.clone()
        # 调用编译后的函数fn，并传入张量a和a
        res = fn(a, a)
        # 对结果res进行反向传播
        res.backward(grad)

        # 断言编译结果与MyMM.apply(a, a)的结果相等
        self.assertEqual(res, MyMM.apply(a, a))
        # 断言frame_count计数为1
        self.assertEqual(cnt.frame_count, 1)
    def test_user_defined_object_as_input(self):
        # 创建一个编译计数器对象，使用指定的后端"aot_eager"
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        # 定义一个数据类Weird，包含整数x，张量b和张量c
        @dataclass
        class Weird:
            x: int
            b: torch.Tensor
            c: torch.Tensor

        # 定义一个自定义的torch.autograd.Function子类Foo
        class Foo(torch.autograd.Function):
            # 前向传播函数，接受参数x: torch.Tensor, weird: Weird, z: torch.Tensor
            @staticmethod
            def forward(ctx, x: torch.Tensor, weird: Weird, z: torch.Tensor):
                # 保存张量b和张量c，以便在反向传播时使用
                ctx.save_for_backward(weird.b, weird.c)
                # 返回计算结果 weird.b * weird.c * x.clone()
                return weird.b * weird.c * x.clone()

            # 反向传播函数，接受参数grad
            @staticmethod
            def backward(ctx, grad):
                # 从上下文中取出保存的张量b和张量c
                b, c = ctx.saved_tensors
                # 返回梯度 grad * b * c, None, grad * 2
                return grad * b * c, None, grad * 2

        # 使用编译装饰器，将函数f编译为torch脚本
        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, weird, z):
            # 调用Foo的apply方法进行前向传播计算
            return Foo.apply(x, weird, z)

        # 创建张量x，要求计算其梯度
        x = torch.tensor(2.0, requires_grad=True)
        # 创建Weird对象weird，其中b张量要求计算其梯度
        weird = Weird(1.2, torch.tensor(2.5, requires_grad=True), torch.tensor(3.5))
        # 创建张量z，要求计算其梯度
        z = torch.tensor(3.0, requires_grad=True)

        # 调用函数f进行计算，返回结果result
        result = f(x, weird, z)
        # 对结果result求和，并反向传播梯度
        result.sum().backward()

        # 断言结果result与使用相同参数的Foo.apply方法的结果相等
        self.assertEqual(result, Foo.apply(x, weird, z))
        # 断言张量x的梯度为 2.5 * 3.5
        self.assertEqual(x.grad, 2.5 * 3.5)
        # 断言张量z的梯度为 2.0
        self.assertEqual(z.grad, 2.0)
        # 断言张量weird.b的梯度为None
        self.assertEqual(weird.b.grad, None)

        # 检查Dynamo捕获的图是否正确
        actual_graph = torch._dynamo.testing.normalize_gm(
            cnt.graphs[0].print_readable(print_output=False)
        )
        # 断言实际得到的图与预期的图形式一致
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    # 定义神经网络模块，继承自 torch.nn.Module

    def forward(self, L_x_: "f32[]", L_z_: "f32[]", L_weird_b: "f32[]", L_weird_c: "f32[]"):
        # 定义前向传播函数，接受四个输入参数

        # 将输入参数赋值给局部变量
        l_x_ = L_x_
        l_z_ = L_z_
        l_weird_b = L_weird_b
        l_weird_c = L_weird_c

        # 创建一个函数上下文对象
        function_ctx = torch.autograd.function.FunctionCtx()

        # 获取模块内定义的前向和后向传播函数
        fwd_body_0 = self.fwd_body_0
        bwd_body_0 = self.bwd_body_0

        # 调用 torch._functorch.autograd_function.autograd_function_apply 方法，
        # 传入前向和后向传播函数以及所有输入参数，并指定参数的张量掩码
        autograd_function_apply: "f32[]" = torch._functorch.autograd_function.autograd_function_apply(
            fwd_body_0, bwd_body_0, l_x_, l_z_, l_weird_b, l_weird_c, args_tensor_mask=[True, False, True]
        )

        # 清空所有变量，释放内存
        fwd_body_0 = bwd_body_0 = l_x_ = l_z_ = l_weird_b = l_weird_c = None

        # 返回 autograd_function_apply 的结果作为元组的单个元素
        return (autograd_function_apply,)

    class GraphModule(torch.nn.Module):
        # 定义 GraphModule 内嵌类，继承自 torch.nn.Module

        def forward(self, function_ctx, l_x_: "f32[]", l_z_: "f32[]", l_weird_b: "f32[]", l_weird_c: "f32[]"):
            # 定义前向传播函数，接受函数上下文和四个输入参数

            # 计算 l_weird_b 和 l_weird_c 的乘积
            mul: "f32[]" = l_weird_b * l_weird_c

            # 克隆 l_x_ 张量
            clone: "f32[]" = l_x_.clone()

            # 计算 mul 和 clone 的乘积
            mul_1: "f32[]" = mul * clone

            # 清空 l_x_ 变量，释放内存
            l_x_ = None

            # 返回 mul_1 和 [l_weird_b, l_weird_c] 的结果作为元组的两个元素
            return (mul_1, [l_weird_b, l_weird_c])

    class GraphModule(torch.nn.Module):
        # 定义 GraphModule 内嵌类，继承自 torch.nn.Module

        def forward(self, function_ctx, mul_1: "f32[]", l_weird_b: "f32[]", l_weird_c: "f32[]"):
            # 定义前向传播函数，接受函数上下文、mul_1 和两个张量参数 l_weird_b、l_weird_c

            # 关闭梯度跟踪功能
            _set_grad_enabled = torch._C._set_grad_enabled(False)

            # 计算 mul_1 和 l_weird_b 的乘积
            mul: "f32[]" = mul_1 * l_weird_b

            # 计算 mul 和 l_weird_c 的乘积
            mul_2: "f32[]" = mul * l_weird_c

            # 计算 mul_1 的两倍
            mul_3: "f32[]" = mul_1 * 2

            # 重新启用梯度跟踪功能
            _set_grad_enabled_1 = torch._C._set_grad_enabled(True)

            # 返回 mul_2 和 mul_3 作为元组的两个元素
            return (mul_2, mul_3)

    def test_tensor_list_as_input(self):
        # 定义一个测试方法，测试张量列表作为输入的情况

        class Foo(torch.autograd.Function):
            # 定义一个继承自 torch.autograd.Function 的 Foo 类

            @staticmethod
            def forward(ctx, x, tl):
                # 静态方法：定义前向传播函数，接受上下文对象 ctx 和两个输入参数 x 和 tl

                # 保存张量列表中的第一个和第二个张量
                ctx.save_for_backward(tl[0], tl[1])

                # 返回 x 和 tl 中第一个和第二个张量的乘积
                return x.clone() * (tl[0] + tl[1])

            @staticmethod
            def backward(ctx, grad):
                # 静态方法：定义反向传播函数，接受上下文对象 ctx 和梯度 grad

                # 从上下文对象中获取保存的张量列表中的两个张量
                tl0, tl1 = ctx.saved_tensors

                # 返回梯度乘以张量列表中的两个张量的和，以及 None（因为第二个参数是 None）
                return grad * (tl0 + tl1), None

        # 使用 torch.compile 方法，编译函数 f
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, tl):
            # 定义函数 f，接受两个输入参数 x 和 tl

            # 调用 Foo 类的 apply 方法，传入 x 和 tl，返回结果
            return Foo.apply(x, tl)

        # 创建一个张量 x，要求计算其梯度
        x = torch.tensor(2.0, requires_grad=True)

        # 创建一个张量列表 tl，其中两个张量也要求计算其梯度
        tl = [
            torch.tensor(3.0, requires_grad=True),
            torch.tensor(4.0, requires_grad=True),
        ]

        # 调用函数 f，传入 x 和 tl，获取结果
        result = f(x, tl)

        # 对结果进行求和并反向传播梯度
        result.sum().backward()

        # 断言结果与使用 x 和 tl 调用 Foo 类的 apply 方法得到的结果相等
        self.assertEqual(result, Foo.apply(x, tl))

        # 断言张量 x 的梯度为 7.0
        self.assertEqual(x.grad, 7.0)

        # 断言张量列表中的第一个张量的梯度为 None
        self.assertEqual(tl[0].grad, None)

        # 断言张量列表中的第二个张量的梯度为 None
        self.assertEqual(tl[1].grad, None)
    def test_multiple_different_non_tensor_inputs(self):
        @dataclass
        class Weird:
            x: int
            b: torch.Tensor
            c: torch.Tensor

        # 定义一个自定义的 PyTorch 函数 Foo
        class Foo(torch.autograd.Function):
            @staticmethod
            # 前向传播函数：接受 x, weird, z, tl 四个参数
            def forward(ctx, x, weird, z, tl):
                # 保存需要在反向传播时使用的张量到上下文对象 ctx 中
                ctx.save_for_backward(weird.b, weird.c, tl[0], tl[1])
                # 返回计算结果：x 乘以 weird.b、weird.c 和 tl[0] 的乘积的克隆
                return x.clone() * weird.b * weird.c * tl[0]

            @staticmethod
            # 反向传播函数：接受梯度 grad 作为输入
            def backward(ctx, grad):
                # 从上下文对象 ctx 中获取保存的张量
                b, c, tl0, _ = ctx.saved_tensors
                # 返回每个参数的梯度：grad 乘以 b、c 和 tl0
                return grad * b * c * tl0, None, grad * 2, None

        # 使用特定的编译选项定义函数 f
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, weird, z, tl):
            return Foo.apply(x, weird, z, tl)

        # 创建张量 x, weird, z, tl
        x = torch.tensor(2.0, requires_grad=True)
        weird = Weird(
            1.2,
            torch.tensor(2.5, requires_grad=True),
            torch.tensor(3.5, requires_grad=True),
        )
        z = torch.tensor(3.0, requires_grad=True)
        tl = [
            torch.tensor(0.5, requires_grad=True),
            torch.tensor(0.6, requires_grad=True),
        ]

        # 调用函数 f，并获取结果
        result = f(x, weird, z, tl)
        # 对结果求和，并进行反向传播
        result.sum().backward()

        # 断言各个梯度值和计算结果
        self.assertEqual(result, Foo.apply(x, weird, z, tl))
        self.assertEqual(x.grad, 2.5 * 3.5 * 0.5)
        self.assertEqual(z.grad, 2.0)
        self.assertEqual(weird.b.grad, None)
        self.assertEqual(weird.c.grad, None)
        self.assertEqual(tl[0].grad, None)
        self.assertEqual(tl[1].grad, None)

    def test_backward_returns_none_for_tensor_input(self):
        # 定义一个自定义的 PyTorch 函数 Foo
        class Foo(torch.autograd.Function):
            @staticmethod
            # 前向传播函数：接受 x, y 两个参数
            def forward(ctx, x, y):
                # 保存 y 张量到上下文对象 ctx 中
                ctx.save_for_backward(y)
                # 返回计算结果：x 乘以 y 的结果
                return x.clone() * y

            @staticmethod
            # 反向传播函数：接受梯度 grad 作为输入
            def backward(ctx, grad):
                # 从上下文对象 ctx 中获取保存的张量
                (y,) = ctx.saved_tensors
                # 返回梯度：grad 乘以 y
                return grad * y, None

        # 使用特定的编译选项定义函数 f
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, y):
            return Foo.apply(x, y)

        # 创建张量 x, y
        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)

        # 调用函数 f，并获取结果
        result = f(x, y)
        # 对结果求和，并进行反向传播
        result.sum().backward()

        # 断言各个梯度值和计算结果
        self.assertEqual(result, Foo.apply(x, y))
        self.assertEqual(x.grad, 3.0)
        self.assertEqual(y.grad, None)
    # 定义一个测试函数，用于测试带有绑定和自由变量的情况
    def test_function_with_bound_free_variable(self):
        # 定义一个继承自torch.autograd.Function的类LowerBound
        class LowerBound(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，接收输入和下界作为参数
            def forward(ctx, inputs, bound):
                # 保存输入和新创建的大小为1的张量bound到上下文中
                ctx.save_for_backward(inputs, inputs.new_ones(1) * bound)
                # 返回经过下界处理后的输入数据
                return inputs.clamp(min=bound)

            @staticmethod
            # 反向传播函数，接收梯度输出作为参数
            def backward(ctx, grad_output):
                # 从上下文中恢复保存的张量
                inputs, bound = ctx.saved_tensors
                # 返回计算后的梯度
                return (inputs >= bound) * grad_output, None

        # 定义一个继承自torch.nn.Module的模块类MyMod
        class MyMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个参数gamma，大小为[4, 128, 32, 32]的随机张量
                self.gamma = torch.nn.Parameter(torch.rand([4, 128, 32, 32]))

            def forward(self, x):
                # 调用LowerBound中的apply方法，对gamma进行下界处理
                gamma = LowerBound.apply(self.gamma, 1)
                # 返回输入x与处理后的gamma的和
                return x + gamma

        # 创建MyMod类的实例mod
        mod = MyMod()
        # 准备输入参数args和kwargs
        args, kwargs = ([torch.rand([4, 128, 32, 32])], {})
        # 计算未编译前的模型输出
        before = mod(*args, **kwargs)

        # 使用torch._dynamo.optimize("eager")优化模型mod
        compiled_model = torch._dynamo.optimize("eager")(mod)
        # 计算优化后的模型输出
        after = compiled_model(*args, **kwargs)
        # 断言优化前后模型输出相等
        self.assertEqual(before, after)

    # 这些测试用例来自test_autograd.py
    # 未来，应该让Dynamo测试套件在test_autograd.py上运行（目前已禁用），并删除这些测试用例。
    def test_smuggle_symint_issue_111031(self):
        # 导入Function类
        from torch.autograd import Function

        # 定义一个继承自Function的类Foo
        class Foo(Function):
            @staticmethod
            # 前向传播函数，接收输入x作为参数
            def forward(ctx, x):
                # 在上下文中存储x的大小的第一个维度
                ctx.x0 = x.size(0)
                # 返回输入x乘以2的结果
                return x * 2

            @staticmethod
            # 反向传播函数，接收梯度grad_out作为参数
            def backward(ctx, grad_out):
                # 返回梯度乘以上下文中保存的x0
                return grad_out * ctx.x0

        # 创建一个编译计数器cnts
        cnts = torch._dynamo.testing.CompileCounter()

        # 使用torch.compile进行编译，backend为cnts，完整图形和动态模式为True
        @torch.compile(backend=cnts, fullgraph=True, dynamic=True)
        def foo(x):
            # 调用Foo类的apply方法
            return Foo.apply(x)

        # 对输入torch.randn(2, requires_grad=True)调用foo函数
        foo(torch.randn(2, requires_grad=True))
        # 断言帧计数等于1
        self.assertEqual(cnts.frame_count, 1)

    # 定义一个测试函数test_needs_input_grad
    def test_needs_input_grad(self):
        # 创建一个编译计数器cnt
        cnt = torch._dynamo.testing.CompileCounter()

        # 定义一个继承自torch.autograd.Function的类NeedsInputGradFunc
        class NeedsInputGradFunc(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，接收foo作为参数
            def forward(ctx, foo):
                # 计算结果为foo加上自身的结果
                result = foo + foo
                # 保存结果到上下文中
                ctx.save_for_backward(result)
                # 返回结果
                return result

            @staticmethod
            # 反向传播函数，接收梯度grad_output作为参数
            @torch.compile(backend=cnt, fullgraph=True)
            def backward(ctx, grad_output):
                # 从上下文中恢复保存的张量result
                (result,) = ctx.saved_tensors
                # 如果需要计算输入梯度，则返回grad_output乘以结果的正弦值
                if ctx.needs_input_grad[0]:
                    return grad_output * result.sin()
                return None

        # 创建一个大小为10的随机张量x，并设置requires_grad为True
        x = torch.randn(10, requires_grad=True)
        # 调用NeedsInputGradFunc的apply方法，并对结果求和后进行反向传播
        NeedsInputGradFunc.apply(x).sum().backward()
        # 断言x的梯度形状与x的形状相同
        self.assertEqual(x.grad.shape, x.shape)
        # 断言帧计数等于1
        self.assertEqual(cnt.frame_count, 1)
        # 断言操作计数等于2
        self.assertEqual(cnt.op_count, 2)
    def test_repeated_save_for_backward_calls(self):
        from torch.autograd import Function  # 导入 Function 类

        class Foo(Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x)  # 在上下文中保存张量 x
                ctx.save_for_backward(x, y)  # 覆盖之前保存的内容，保存张量 x 和 y
                return x * y  # 返回输入张量 x 和 y 的乘积作为输出

            @staticmethod
            def backward(ctx, grad_out):
                x, y = ctx.saved_tensors  # 从上下文中恢复保存的张量 x 和 y
                return grad_out * x, grad_out * y  # 返回梯度 grad_out 分别乘以 x 和 y

        cnts = torch._dynamo.testing.CompileCounter()  # 创建编译计数器实例

        def foo(x, y):
            return Foo.apply(x, y)  # 调用 Foo 的 apply 方法，传入 x 和 y

        x_ref = torch.randn(2, requires_grad=True)  # 创建需要梯度的随机张量 x_ref
        y_ref = torch.randn(2, requires_grad=True)  # 创建需要梯度的随机张量 y_ref
        x_test = x_ref.clone().detach().requires_grad_()  # 克隆 x_ref 并确保可以计算梯度
        y_test = y_ref.clone().detach().requires_grad_()  # 克隆 y_ref 并确保可以计算梯度

        out_ref = foo(x_ref, y_ref)  # 使用 x_ref 和 y_ref 调用 foo 函数得到输出
        out_ref.sum().backward()  # 对输出进行求和并反向传播梯度

        out_test = torch.compile(foo, backend=cnts)(x_test, y_test)  # 使用编译器 cnts 编译 foo 函数并传入 x_test 和 y_test，得到输出
        out_test.sum().backward()  # 对输出进行求和并反向传播梯度

        self.assertEqual(cnts.frame_count, 1)  # 断言编译计数器的帧数为 1
        self.assertEqual(out_ref, out_test)  # 断言参考输出和测试输出相等
        self.assertEqual(x_ref.grad, x_test.grad)  # 断言参考输入 x_ref 的梯度与测试输入 x_test 的梯度相等
        self.assertEqual(y_ref.grad, y_test.grad)  # 断言参考输入 y_ref 的梯度与测试输入 y_test 的梯度相等

    def test_smuggle_tensor_and_complex_structures(self):
        from torch.autograd import Function  # 导入 Function 类

        class Foo(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.x0 = x  # 在上下文中保存张量 x
                ctx.x1 = [1, 2, 3]  # 在上下文中保存列表 [1, 2, 3]
                return x * 2  # 返回输入张量 x 的两倍作为输出

            @staticmethod
            def backward(ctx, grad_out):
                x0mul = grad_out * ctx.x0  # 计算梯度 grad_out 乘以保存的张量 x
                for i in ctx.x1:  # 遍历上下文中保存的列表
                    x0mul = (x0mul * i) + x0mul  # 执行一系列操作，不同于常规的梯度计算方式
                return x0mul  # 返回最终的梯度

        cnts = torch._dynamo.testing.CompileCounter()  # 创建编译计数器实例

        @torch.compile(backend=cnts, fullgraph=True, dynamic=True)
        def foo(x):
            return Foo.apply(x)  # 调用 Foo 的 apply 方法，传入 x

        foo(torch.randn(2, requires_grad=True))  # 使用具有梯度需求的随机张量调用 foo 函数
        self.assertEqual(cnts.frame_count, 1)  # 断言编译计数器的帧数为 1

    def test_default_values(self):
        from torch.autograd import Function  # 导入 Function 类

        class Foo(Function):
            @staticmethod
            def forward(ctx, x, alpha=0.99):  # 定义带有默认参数 alpha 的前向传播函数
                return x  # 直接返回输入张量 x

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out  # 直接返回梯度 grad_out

        @torch.compile
        def foo(x):
            return Foo.apply(x)  # 调用 Foo 的 apply 方法，传入 x

        # Make sure guards for default values do not crash
        foo(torch.randn(2))  # 使用没有梯度需求的随机张量调用 foo 函数
        foo(torch.randn(2, requires_grad=True))  # 使用具有梯度需求的随机张量调用 foo 函数
    def test_tuple_arg(self):
        cnt = torch._dynamo.testing.CompileCounter()

        # 定义一个接受元组参数的自动求导函数
        class TupleArgFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, shape):
                # 在前向传播中保存随机生成的张量到上下文中
                ctx.save_for_backward(torch.randn(shape))
                return x + 1

            @staticmethod
            def backward(ctx, grad_output):
                # 在反向传播中获取保存的张量，并返回结果
                (result,) = ctx.saved_tensors
                return result, None

        # 使用指定的编译计数器和完整图模式进行编译
        @torch.compile(backend=cnt, fullgraph=True)
        def fn():
            return TupleArgFunc.apply(x, shape)

        shape = (10, 10)
        x = torch.randn(shape, requires_grad=True)
        out = fn()
        out.sum().backward()
        # 断言输出与期望值相等
        self.assertEqual(out, x + 1)
        # 断言输入张量的梯度形状符合预期
        self.assertEqual(x.grad.shape, shape)
        # 断言编译计数器中的帧数和操作数符合预期
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    @requires_cuda
    def test_triton_kernel_basic(self):
        # 定义一个使用 Triton 编译的加法自动求导函数
        class Add(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                # 在前向传播中保存输入张量到上下文中，并执行加法核函数
                ctx.save_for_backward(x, y)
                output = torch.zeros_like(x)
                n_elements = output.numel()
                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                )
                add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                # 在反向传播中获取保存的张量，并返回它们乘以梯度输出的结果
                x, y = ctx.saved_tensors
                return x * grad_output, y * grad_output

        # 使用 Inductor 后端编译整个图模式的函数
        @torch.compile(fullgraph=True, backend="inductor")
        def f(x, y):
            z = Add.apply(x, y)
            return z

        x = torch.randn(10, device="cuda", requires_grad=True)
        y = torch.randn(10, device="cuda", requires_grad=True)
        z = f(x, y)
        loss = z.sum()
        loss.backward()
        # 断言张量加法的结果与预期相等
        self.assertEqual(x + y, z)
   `
    def test_triton_kernel_multiple_out(self):
        # 定义一个名为 Add 的继承自 torch.autograd.Function 的类
        class Add(torch.autograd.Function):
            @staticmethod
            # 定义前向传播的方法
            def forward(ctx, x, y):
                # 保存输入张量 x 和 y，以便在反向传播时使用
                ctx.save_for_backward(x, y)
                ctx.t1 = x
                ctx.t2 = y
                # 创建一个与 x 大小相同的零张量作为输出
                output = torch.zeros_like(x)
                # 获取输出张量的元素数量
                n_elements = output.numel()
                # 定义一个 lambda 函数来计算 grid 的大小
                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                )
                # 调用 Triton 内核函数进行加法计算
                add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
                # 返回计算结果和输入张量 x
                return output, x

            @staticmethod
            # 定义反向传播的方法
            def backward(ctx, grad_output, old_x):
                # 从 ctx 中获取保存的 x 和 y 张量
                x, y = ctx.saved_tensors
                x1 = ctx.t1
                y1 = ctx.t2
                # 返回计算梯度的结果，包含 x 和 y 的梯度
                return old_x * x * x1 * grad_output, y * y1 * grad_output

        @torch.compile(fullgraph=True, backend="inductor")
        # 定义一个使用 Add 自定义函数的编译函数
        def f(x, y):
            z = Add.apply(x, y)
            return z

        # 创建两个随机初始化的张量 x 和 y，放置在 CUDA 设备上，并设置 requires_grad=True 以支持梯度计算
        x = torch.randn(10, device="cuda", requires_grad=True)
        y = torch.randn(10, device="cuda", requires_grad=True)
        # 调用编译函数 f，得到输出张量 z 和一个未使用的变量 _
        z, _ = f(x, y)
        # 计算 z 的总和作为损失函数
        loss = z.sum()
        # 反向传播计算梯度
        loss.backward()
        # 断言 x 和 y 的和与 z 相等
        self.assertEqual(x + y, z)
# 如果当前脚本被直接执行（而不是被作为模块导入），则执行以下代码块
if __name__ == "__main__":
    # 导入名为 run_tests 的函数，通常是从 torch._dynamo.test_case 模块中导入
    from torch._dynamo.test_case import run_tests

    # 运行导入的 run_tests 函数，这个函数可能是用来执行测试用例的
    run_tests()
```