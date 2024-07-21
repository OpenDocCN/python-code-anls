# `.\pytorch\test\test_functionalization_of_rng_ops.py`

```
# Owner(s): ["oncall: pt2"]
# 导入必要的库和模块
import functools  # 导入 functools 模块，用于高阶函数操作
import sys  # 导入 sys 模块，用于系统相关操作
import unittest  # 导入 unittest 模块，用于单元测试
from unittest.mock import patch  # 从 unittest.mock 模块导入 patch 函数，用于模拟对象

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 中的 checkpoint 模块
from functorch.compile import aot_function, min_cut_rematerialization_partition, nop  # 导入 functorch.compile 中的相关函数

from torch.testing._internal.common_device_type import (  # 导入 torch.testing._internal.common_device_type 中的内容
    dtypes,  # 导入 dtypes 函数，用于处理数据类型相关的测试
    instantiate_device_type_tests,  # 导入 instantiate_device_type_tests 函数，用于实例化设备类型测试
)

from torch.testing._internal.common_utils import (  # 导入 torch.testing._internal.common_utils 中的内容
    IS_CI,  # 导入 IS_CI 变量，表示是否处于 CI 环境
    IS_WINDOWS,  # 导入 IS_WINDOWS 变量，表示是否处于 Windows 环境
    run_tests,  # 导入 run_tests 函数，用于运行测试
    TestCase,  # 导入 TestCase 类，用于编写测试用例
)

if IS_WINDOWS and IS_CI:
    # 如果在 Windows 并且在 CI 环境下
    sys.stderr.write("torch.compile not supported on windows")  # 将警告信息写入标准错误流
    if __name__ == "__main__":
        sys.exit(0)  # 如果是在主程序中，退出程序
    raise unittest.SkipTest("torch.compile not supported on windows")  # 抛出跳过测试的异常，因为在 Windows 上不支持 torch.compile


def count_philox_rand(gm, args, freq):
    # 统计图中 torch.ops.rngprims.philox_rand.default 的出现次数
    assert [node.target for node in gm.graph.nodes].count(
        torch.ops.rngprims.philox_rand.default
    ) == freq
    return gm  # 返回输入的 gm 对象


class TestFunctionalizationRngOps(TestCase):
    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    def test_rand_like(self, dtype, device):
        # 定义测试函数 fn，用于测试 torch.rand_like 的功能
        def fn(x):
            a = torch.rand_like(x) * x  # 生成与 x 形状相同的随机张量 a，与 x 相乘
            a = torch.rand_like(x) * a  # 生成与 a 形状相同的随机张量，并与 a 相乘
            return a  # 返回处理后的张量 a

        x = torch.rand(10, device=device, dtype=dtype)  # 生成一个形状为 (10,) 的随机张量 x

        for seed in range(10):  # 循环测试 10 个不同的种子
            torch.cuda.manual_seed(seed)  # 设置 CUDA 的随机种子为 seed
            ref = fn(x)  # 使用 fn 函数生成参考结果 ref

            torch.cuda.manual_seed(seed)  # 再次设置 CUDA 的随机种子为 seed
            aot_fn = aot_function(fn, functools.partial(count_philox_rand, freq=2))  # 对 fn 进行 Ahead-of-Time 编译，同时传递 count_philox_rand 函数作为参数
            res = aot_fn(x)  # 对 x 应用编译后的函数 aot_fn，得到结果 res

            self.assertEqual(ref, res)  # 断言参考结果和编译后结果的相等性

    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    def test_rand_like_dynamic(self, dtype, device):
        # 定义动态形状测试函数 fn
        def fn(x):
            a = torch.rand_like(x) * x  # 生成与 x 形状相同的随机张量 a，与 x 相乘
            a = torch.rand_like(x) * a  # 生成与 a 形状相同的随机张量，并与 a 相乘
            return a  # 返回处理后的张量 a

        for seed in range(1, 10):  # 循环测试种子从 1 到 9
            shape = (seed, seed)  # 根据种子生成形状为 (seed, seed) 的张量
            x = torch.rand(shape, device=device, dtype=dtype)  # 生成指定设备和数据类型的随机张量 x
            torch.cuda.manual_seed(seed)  # 设置 CUDA 的随机种子为 seed
            ref = fn(x)  # 使用 fn 函数生成参考结果 ref

            torch.cuda.manual_seed(seed)  # 再次设置 CUDA 的随机种子为 seed
            opt_fn = torch.compile(fn, backend="aot_eager", dynamic=True)  # 使用动态形状进行 Ahead-of-Time 编译
            res = opt_fn(x)  # 对 x 应用编译后的函数 opt_fn，得到结果 res

            self.assertEqual(ref, res)  # 断言参考结果和编译后结果的相等性

    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    def test_rand_like_dynamic_bwd(self, dtype, device):
        # 定义动态形状和反向传播测试函数 fn
        def fn(x):
            a = torch.rand_like(x) * x  # 生成与 x 形状相同的随机张量 a，与 x 相乘
            a = torch.rand_like(x) * a  # 生成与 a 形状相同的随机张量，并与 a 相乘
            return a  # 返回处理后的张量 a

        for seed in range(1, 10):  # 循环测试种子从 1 到 9
            shape = (seed, seed)  # 根据种子生成形状为 (seed, seed) 的张量
            x = torch.rand(shape, device=device, dtype=dtype, requires_grad=True)  # 生成指定设备和数据类型、需要梯度的随机张量 x
            torch.cuda.manual_seed(seed)  # 设置 CUDA 的随机种子为 seed
            ref = fn(x)  # 使用 fn 函数生成参考结果 ref
            ref.sum().backward()  # 对 ref 进行求和并反向传播

            torch.cuda.manual_seed(seed)  # 再次设置 CUDA 的随机种子为 seed
            opt_fn = torch.compile(fn, backend="aot_eager", dynamic=True)  # 使用动态形状和反向传播进行 Ahead-of-Time 编译
            res = opt_fn(x)  # 对 x 应用编译后的函数 opt_fn，得到结果 res
            res.sum().backward()  # 对 res 进行求和并反向传播

            self.assertEqual(ref, res)  # 断言参考结果和编译后结果的相等性
    def test_rand(self, dtype, device):
        shape = (10,)  # 定义张量的形状为 (10,)

        def fn(x):
            a = torch.rand(*shape, device=device, dtype=dtype) * x  # 在指定设备和数据类型下生成形状为 shape 的随机张量，并与 x 相乘
            a = torch.rand(*shape, device=device, dtype=dtype) * a  # 再次生成随机张量，并与前面的结果 a 相乘
            return a  # 返回处理后的张量 a

        x = torch.rand(*shape, device=device, dtype=dtype)  # 在指定设备和数据类型下生成形状为 shape 的随机张量 x

        for seed in range(10):  # 循环遍历种子值范围为 0 到 9
            torch.cuda.manual_seed(seed)  # 设置 CUDA 的随机种子为当前种子值
            ref = fn(x)  # 调用函数 fn，得到参考结果 ref

            torch.cuda.manual_seed(seed)  # 再次设置 CUDA 的随机种子为当前种子值
            aot_fn = aot_function(fn, functools.partial(count_philox_rand, freq=2))  # 编译函数 fn 为 AOT 函数，使用自定义的随机数生成器
            res = aot_fn(x)  # 调用编译后的 AOT 函数，得到结果 res

            self.assertEqual(ref, res)  # 断言参考结果 ref 与编译后的结果 res 相等

    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    def test_autograd_function(self, dtype, device):
        shape = (16, 16)  # 定义张量的形状为 (16, 16)

        class Custom(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)  # 在上下文中保存张量 x
                a = torch.rand_like(x) * x  # 生成形状与 x 相同的随机张量，并与 x 相乘
                a = torch.rand_like(x) * a  # 再次生成随机张量，并与前面的结果 a 相乘
                return a  # 返回处理后的张量 a

            @staticmethod
            def backward(ctx, grad_out):
                (x,) = ctx.saved_tensors  # 从上下文中获取保存的张量 x
                return grad_out * torch.rand_like(grad_out) * torch.cos(x)  # 返回反向传播的梯度

        custom = Custom.apply  # 创建自定义的 autograd 函数 custom

        x = torch.rand(*shape, device=device, dtype=dtype, requires_grad=True)  # 在指定设备和数据类型下生成形状为 shape 的随机张量 x，需要梯度

        x_clone = x.clone().detach().requires_grad_(True)  # 克隆张量 x，并确保它需要梯度

        torch.cuda.manual_seed(123)  # 设置 CUDA 的随机种子为 123
        ref = custom(x)  # 调用自定义的 autograd 函数 custom，得到参考结果 ref
        ref.sum().backward()  # 对参考结果 ref 求和并进行反向传播

        torch.cuda.manual_seed(123)  # 再次设置 CUDA 的随机种子为 123
        fwd_compiler = functools.partial(count_philox_rand, freq=2)  # 预编译前向计算的函数，使用自定义的随机数生成器
        bwd_compiler = functools.partial(count_philox_rand, freq=1)  # 预编译反向计算的函数，使用自定义的随机数生成器
        aot_custom = aot_function(custom, fwd_compiler, bwd_compiler)  # 编译自定义的 autograd 函数 custom 为 AOT 函数
        res = aot_custom(x_clone)  # 调用编译后的 AOT 函数，得到结果 res
        res.sum().backward()  # 对结果 res 求和并进行反向传播

        self.assertEqual(ref, res)  # 断言参考结果 ref 与结果 res 相等
        self.assertEqual(x.grad, x_clone.grad)  # 断言张量 x 的梯度与其克隆张量 x_clone 的梯度相等

    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    # 定义测试方法，用于检查当存在多个 AOT (Ahead-of-Time) 追踪的图时，随机数生成状态是否保持一致
    def test_multiple_subgraphs(self, dtype, device):
        # Checks that rng state is maintained when there are multiple aot traced
        # graphs.
        
        # 定义张量的形状
        shape = (16, 16)

        # 自定义操作类 CustomOp1
        class CustomOp1(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 保存前向传播需要用到的中间变量 x
                ctx.save_for_backward(x)
                # 对输入 x 执行一系列操作
                a = torch.rand_like(x) * x
                a = torch.rand_like(x) * a
                return a

            @staticmethod
            def backward(ctx, grad_out):
                # 获取保存的中间变量 x
                (x,) = ctx.saved_tensors
                # 返回反向传播的梯度
                return grad_out * torch.rand_like(grad_out) * torch.cos(x)

        # 自定义操作类 CustomOp2
        class CustomOp2(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 保存前向传播需要用到的中间变量 x
                ctx.save_for_backward(x)
                # 对输入 x 执行操作
                a = torch.rand_like(x) * x
                return a

            @staticmethod
            def backward(ctx, grad_out):
                # 获取保存的中间变量 x
                (x,) = ctx.saved_tensors
                # 返回反向传播的梯度
                return grad_out * torch.rand_like(grad_out) * torch.rand_like(x)

        # 定义两个自定义操作的实际调用函数
        custom_op1 = CustomOp1.apply
        custom_op2 = CustomOp2.apply

        # 定义一个函数 fn，执行 custom_op1 和 custom_op2 的组合操作
        def fn(x):
            a = custom_op1(x)
            b = a.sin()
            return custom_op2(b)

        # 定义前向编译器和后向编译器，用于 AOT 编译自定义操作
        fwd_compiler = functools.partial(count_philox_rand, freq=2)
        bwd_compiler = functools.partial(count_philox_rand, freq=1)
        aot_custom_op1 = aot_function(custom_op1, fwd_compiler, bwd_compiler)

        # 重新定义前向编译器和后向编译器，用于另一个自定义操作的 AOT 编译
        fwd_compiler = functools.partial(count_philox_rand, freq=1)
        bwd_compiler = functools.partial(count_philox_rand, freq=2)
        aot_custom_op2 = aot_function(custom_op2, fwd_compiler, bwd_compiler)

        # 定义一个 AOT 函数 aot_fn，执行 AOT 编译后的 custom_op1 和 custom_op2 的组合操作
        def aot_fn(x):
            a = aot_custom_op1(x)
            b = a.sin()
            return aot_custom_op2(b)

        # 循环进行 10 次测试，每次使用不同的随机种子
        for seed in range(10):
            # 设置 CUDA 的随机种子
            torch.cuda.manual_seed(seed)
            # 创建随机张量 x，具有指定的形状、设备和数据类型，并要求梯度计算
            x = torch.rand(*shape, device=device, dtype=dtype, requires_grad=True)
            # 创建 x 的克隆，从而使 x_clone 和 x 在随机初始化时具有相同的状态
            x_clone = x.clone().detach().requires_grad_(True)

            # 使用相同的随机种子设置 CUDA 随机状态，计算 fn(x) 的参考结果并进行反向传播
            torch.cuda.manual_seed(seed)
            ref = fn(x)
            ref.sum().backward()

            # 使用相同的随机种子设置 CUDA 随机状态，计算 aot_fn(x_clone) 的结果并进行反向传播
            torch.cuda.manual_seed(seed)
            res = aot_fn(x_clone)
            res.sum().backward()

            # 断言两种计算方法得到的结果相等
            self.assertEqual(ref, res)
            # 断言两个张量的梯度相等
            self.assertEqual(x.grad, x_clone.grad)
    # 定义一个测试函数，用于测试设置和获取随机数生成器状态的功能
    def test_set_get_rng_state(self, dtype, device):
        
        # 定义一个内部函数 fn，接受参数 x
        def fn(x):
            # 使用 x 创建一个与 x 形状相同的随机数并与 x 相乘
            a = torch.rand_like(x) * x
            # 获取当前 CUDA 随机数生成器的状态
            state = torch.cuda.get_rng_state()
            # 再次生成一个随机数并与之前的结果相乘
            a = torch.rand_like(x) * a
            # 恢复之前保存的 CUDA 随机数生成器状态
            torch.cuda.set_rng_state(state)
            # 再次生成一个随机数并与之前的结果相乘
            a = torch.rand_like(x) * a
            # 返回结果 a
            return a

        # 生成一个形状为 (10,) 的随机数张量 x，使用指定的数据类型和设备
        x = torch.rand(10, device=device, dtype=dtype)

        # 对于种子范围内的每个种子进行测试
        for seed in range(10):
            # 设置当前 CUDA 随机数种子
            torch.cuda.manual_seed(seed)
            # 计算参考结果 ref，调用 fn 函数计算
            ref = fn(x)

            # 再次设置当前 CUDA 随机数种子
            torch.cuda.manual_seed(seed)
            # 创建一个前向编译器，部分函数被编译以减少运行时开销
            fwd_compiler = functools.partial(count_philox_rand, freq=3)
            # 编译 fn 函数以准备进行 AOT（Ahead-Of-Time）优化
            aot_fn = aot_function(fn, fwd_compiler)
            # 执行优化后的函数计算结果
            res = aot_fn(x)

            # 断言优化前后的结果应该相等
            self.assertEqual(ref, res)

    # 使用指定的数据类型进行装饰器修饰
    @dtypes(torch.float32)
    # 使用指定的配置参数对函数进行装饰器修饰
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    # 定义测试函数 test_min_cut_partitioner
    def test_min_cut_partitioner(self, dtype, device):
        # 检查调用约定是否得到维护
        shape = (16, 16)

        # 定义内部函数 fn，接受参数 x
        def fn(x):
            # 生成一个与 x 形状相同的随机数并与 x 相乘
            a = torch.rand_like(x) * x
            # 再次生成一个随机数并与之前的结果相乘
            a = torch.rand_like(x) * a
            # 对结果应用三次正弦函数
            a = torch.sin(a)
            a = torch.sin(a)
            a = torch.sin(a)
            # 返回结果 a
            return a

        # 生成一个形状为 (16, 16) 的随机数张量 x，使用指定的数据类型、设备和设置了梯度跟踪
        x = torch.rand(*shape, device=device, dtype=dtype, requires_grad=True)

        # 克隆张量 x 并分离计算图，并设置梯度跟踪为 True
        x_clone = x.clone().detach().requires_grad_(True)

        # 设置当前 CUDA 随机数种子
        torch.cuda.manual_seed(123)
        # 计算参考结果 ref，调用 fn 函数计算
        ref = fn(x)
        # 对 ref 的所有元素求和，并进行反向传播
        ref.sum().backward()

        # 再次设置当前 CUDA 随机数种子
        torch.cuda.manual_seed(123)
        # 创建一个前向编译器，部分函数被编译以减少运行时开销
        fwd_compiler = functools.partial(count_philox_rand, freq=2)
        # 创建一个后向编译器，部分函数被编译以减少运行时开销
        bwd_compiler = functools.partial(count_philox_rand, freq=0)
        # 使用自定义分区函数编译 fn 函数以准备进行 AOT 优化
        aot_custom = aot_function(
            fn,
            fwd_compiler,
            bwd_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )
        # 执行优化后的函数计算结果
        res = aot_custom(x_clone)
        # 对 res 的所有元素求和，并进行反向传播
        res.sum().backward()

        # 断言优化前后的结果应该相等
        self.assertEqual(ref, res)
        # 断言 x 的梯度应该等于 x_clone 的梯度
        self.assertEqual(x.grad, x_clone.grad)

    # TODO - Dropout needs more work because of offset calculation
    # 使用指定的配置参数对函数进行装饰器修饰
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    # 使用指定的数据类型进行装饰器修饰
    @dtypes(torch.float32)
    # 定义测试函数 test_checkpoint，接受数据类型 dtype 和设备 device 作为参数
    def test_checkpoint(self, dtype, device):
        
        # 定义函数 g，对输入 x 应用 60% 的 dropout
        def g(x, y):
            return torch.nn.functional.dropout(x, 0.6)

        # 定义函数 fn，使用非重入模式对函数 g 进行 checkpoint
        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(g, x, y, use_reentrant=False)

        # 使用 torch.cuda.manual_seed 设定随机种子为 123
        torch.cuda.manual_seed(123)
        
        # 创建张量 x 和 y，均为在 CUDA 设备上的 2x2 全一张量，需要计算梯度
        x = torch.ones(2, 2, device="cuda", requires_grad=True)
        y = torch.rand(2, 2, device="cuda", requires_grad=True)
        
        # 使用函数 fn 对 x 和 y 进行 checkpoint 计算，保存结果到 ref
        ref = fn(x, y)

        # 使用 functools.partial 创建 fwd_compiler 和 bwd_compiler 函数，用于统计 philox_rand 的使用频率
        fwd_compiler = functools.partial(count_philox_rand, freq=1)
        bwd_compiler = functools.partial(count_philox_rand, freq=0)
        
        # 使用 aot_function 对 fn 进行 ahead-of-time (AOT) 编译
        aot_fn = aot_function(fn, fwd_compiler, bwd_compiler)
        
        # 对 aot_fn 使用 x 和 y 进行计算，存储结果到 res
        res = aot_fn(x, y)
        
        # 对 res 求和并执行反向传播
        res.sum().backward()

    # 使用装饰器 @dtypes(torch.float32) 和 @patch.object 对测试函数进行修饰，设置 functionalize_rng_ops 为 True
    def test_dropout_decomp(self, dtype, device):
        
        # 定义函数 fn，对输入 x 应用 60% 的 dropout 并与 x 相乘
        def fn(x):
            return torch.nn.functional.dropout(x, 0.6) * x

        # 创建大小为 10 的随机张量 x，设备为给定 device，数据类型为给定 dtype
        x = torch.rand(10, device=device, dtype=dtype)

        # 使用 functools.partial 创建 count_philox_rand 的部分应用，用于统计 philox_rand 的使用频率
        aot_fn = aot_function(fn, functools.partial(count_philox_rand, freq=1))
        
        # 对 aot_fn 使用 x 进行计算
        aot_fn(x)
# 导入测试框架中的函数和类，以便用于测试设备特定功能
only_for = ("cuda",)
# 实例化设备类型测试，针对给定的测试类和全局命名空间，仅限于指定的设备类型
instantiate_device_type_tests(TestFunctionalizationRngOps, globals(), only_for=only_for)

# 定义一个名为NegativeTest的测试类，继承自TestCase
class NegativeTest(TestCase):
    # 标记测试用例的数据类型为torch.float32
    @dtypes(torch.float32)
    # 使用patch.object装饰器修改torch._functorch.config.functionalize_rng_ops属性为True
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    # 定义测试函数test_on_cpu，接受dtype和device作为参数
    def test_on_cpu(self, dtype, device):
        # 定义一个函数fn，接受输入参数x
        def fn(x):
            # 生成与x形状相同的随机数，并与x相乘
            a = torch.rand_like(x) * x
            # 再次生成与x形状相同的随机数，并与上一步的结果a相乘
            a = torch.rand_like(x) * a
            return a

        # 在指定设备上生成一个形状为(10,)的随机张量x，数据类型为dtype
        x = torch.rand(10, device=device, dtype=dtype)

        # 使用aot_function函数将fn编译为AOT（Ahead of Time）函数，使用nop作为参数
        aot_fn = aot_function(fn, nop)
        # 使用self.assertRaises断言在执行aot_fn(x)时会抛出RuntimeError异常
        with self.assertRaises(RuntimeError):
            aot_fn(x)

# 仅限于CPU设备
only_for = ("cpu",)
# 实例化设备类型测试，针对NegativeTest类和全局命名空间，仅限于CPU设备
instantiate_device_type_tests(NegativeTest, globals(), only_for=only_for)

# 如果当前脚本作为主程序执行，则运行测试
if __name__ == "__main__":
    run_tests()
```