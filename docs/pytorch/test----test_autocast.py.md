# `.\pytorch\test\test_autocast.py`

```
# 导入所需模块和库
import collections
import unittest

import torch
from torch.testing._internal.autocast_test_lists import AutocastCPUTestLists
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.utils._python_dispatch import TorchDispatchMode

# 定义测试类 TestAutocastCPU，继承自 TestCase 类
class TestAutocastCPU(TestCase):
    # 设置测试前的准备工作
    def setUp(self):
        super().setUp()
        self.autocast_lists = AutocastCPUTestLists(torch.device("cpu"))

    # 清理测试后的资源
    def tearDown(self):
        del self.autocast_lists
        super().tearDown()

    # 辅助方法，处理操作和参数的函数签名
    def args_maybe_kwargs(self, op_with_args):
        if len(op_with_args) == 2:
            return op_with_args[0], op_with_args[1], {}
        else:
            return op_with_args[0], op_with_args[1], op_with_args[2]

    # 跳过 TorchDynamo 环境的装饰器
    @skipIfTorchDynamo()
    # 测试自动混合精度功能在 Torch 中预期的内置提升
    def test_autocast_torch_expect_builtin_promote(self):
        # 遍历自动混合精度测试列表中的每一项
        for (
            op,
            args1,
            args2,
            out_type,
        ) in self.autocast_lists.torch_expect_builtin_promote:
            # 运行自动混合精度测试，使用 torch.float32 作为运行类型
            self._run_autocast_outofplace(op, args1, torch.float32, out_type=out_type)
            # 运行自动混合精度测试，使用 torch.float32 作为运行类型，并指定 amp_dtype 为 torch.float16
            self._run_autocast_outofplace(
                op, args2, torch.float32, out_type=out_type, amp_dtype=torch.float16
            )

    # 跳过 TorchDynamo 环境的装饰器
    @skipIfTorchDynamo()
    # 测试自动混合精度功能在方法中预期的内置提升
    def test_autocast_methods_expect_builtin_promote(self):
        # 遍历自动混合精度测试列表中方法预期的内置提升的每一项
        for (
            op,
            args1,
            args2,
            out_type,
        ) in self.autocast_lists.methods_expect_builtin_promote:
            # 运行自动混合精度测试，使用 torch.float32 作为运行类型，module 参数为 None
            self._run_autocast_outofplace(
                op, args1, torch.float32, module=None, out_type=out_type
            )
            # 运行自动混合精度测试，使用 torch.float32 作为运行类型，module 参数为 None，并指定 amp_dtype 为 torch.float16
            self._run_autocast_outofplace(
                op,
                args2,
                torch.float32,
                module=None,
                out_type=out_type,
                amp_dtype=torch.float16,
            )

    # 跳过 TorchDynamo 环境的装饰器
    @skipIfTorchDynamo()
    # 测试在 Torch 中使用 torch.bfloat16 作为 amp_dtype 的自动混合精度功能
    def test_autocast_torch_16(self):
        # 遍历自动混合精度测试列表中使用 torch.bfloat16 的每一项
        for op_with_args in self.autocast_lists.torch_16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            # 运行自动混合精度测试，使用 torch.bfloat16 作为运行类型，额外的关键字参数从 maybe_kwargs 获取
            self._run_autocast_outofplace(
                op, args, torch.bfloat16, add_kwargs=maybe_kwargs
            )
            # 运行自动混合精度测试，使用 torch.float16 作为运行类型，额外的关键字参数从 maybe_kwargs 获取，并指定 amp_dtype 为 torch.float16
            self._run_autocast_outofplace(
                op,
                args,
                torch.float16,
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    # 跳过 TorchDynamo 环境的装饰器
    # 测试自动类型转换（autocast）对torch.nn.bfloat16的支持
    def test_autocast_nn_16(self):
        # 遍历自动类型转换列表中的每个操作及其参数
        for op_with_args in self.autocast_lists.nn_16:
            # 解析操作、参数以及可能的关键字参数
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            # 运行自动类型转换的操作，使用torch.bfloat16作为目标类型，并指定module为torch._C._nn，添加可能的关键字参数
            self._run_autocast_outofplace(
                op, args, torch.bfloat16, module=torch._C._nn, add_kwargs=maybe_kwargs
            )
            # 再次运行自动类型转换的操作，使用torch.float16作为目标类型，并指定module为torch._C._nn，添加可能的关键字参数，同时指定amp_dtype为torch.float16
            self._run_autocast_outofplace(
                op,
                args,
                torch.float16,
                module=torch._C._nn,
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    # 跳过Torch Dynamo的测试装饰器
    @skipIfTorchDynamo()
    # 测试自动类型转换（autocast）对torch.float32的支持
    def test_autocast_torch_fp32(self):
        # 遍历自动类型转换列表中的每个操作及其参数
        for op_with_args in self.autocast_lists.torch_fp32:
            # 解析操作、参数以及可能的关键字参数
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            # 运行自动类型转换的操作，使用torch.float32作为目标类型，并添加可能的关键字参数
            self._run_autocast_outofplace(
                op, args, torch.float32, add_kwargs=maybe_kwargs
            )
            # 再次运行自动类型转换的操作，使用torch.float32作为目标类型，并添加可能的关键字参数，同时指定amp_dtype为torch.float16
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    # 跳过Torch Dynamo的测试装饰器
    @skipIfTorchDynamo()
    # 测试自动类型转换（autocast）对torch.nn和torch.float32的支持
    def test_autocast_nn_fp32(self):
        # 遍历自动类型转换列表中的每个操作及其参数
        for op_with_args in self.autocast_lists.nn_fp32:
            # 解析操作、参数以及可能的关键字参数
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            # 运行自动类型转换的操作，使用torch.float32作为目标类型，并指定module为torch._C._nn，添加可能的关键字参数
            self._run_autocast_outofplace(
                op, args, torch.float32, module=torch._C._nn, add_kwargs=maybe_kwargs
            )
            # 再次运行自动类型转换的操作，使用torch.float32作为目标类型，并指定module为torch._C._nn，添加可能的关键字参数，同时指定amp_dtype为torch.float16
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                module=torch._C._nn,
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    # 跳过Torch Dynamo的测试装饰器
    @skipIfTorchDynamo()
    # 测试需要自动类型转换（autocast）提升的torch.float32操作
    def test_autocast_torch_need_autocast_promote(self):
        # 遍历自动类型转换列表中的每个操作及其参数组
        for op, args1, args2 in self.autocast_lists.torch_need_autocast_promote:
            # 运行自动类型转换的操作，使用torch.float32作为目标类型
            self._run_autocast_outofplace(op, args1, torch.float32)
            # 再次运行自动类型转换的操作，使用torch.float32作为目标类型，并指定amp_dtype为torch.float16
            self._run_autocast_outofplace(
                op, args2, torch.float32, amp_dtype=torch.float16
            )

    # 使用unittest.skipIf根据IS_WINDOWS的值来决定是否跳过测试
    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    # 测试自动类型转换（autocast）在LSTM模型中的行为
    def test_autocast_rnn(self):
        # 检查MKLDNN是否可用，并且MKLDNN是否支持bf16
        if (
            torch.backends.mkldnn.is_available()
            and torch.ops.mkldnn._is_mkldnn_bf16_supported()
        ):
            # 创建输入张量x、隐藏状态hx和细胞状态cx
            x = torch.randn(1, 2, 1)
            hx = torch.randn(2, 2, 1)
            cx = torch.randn(2, 2, 1)

            # 创建一个LSTM模型，输入维度为1，隐藏层维度为1，层数为2，并指定dtype为torch.bfloat16
            m = torch.nn.LSTM(1, 1, 2).to(torch.bfloat16)

            # 当未启用autocast时，应该抛出ValueError异常
            with self.assertRaisesRegex(ValueError, "input must have the type"):
                m(x, (hx, cx))

            # 使用autocast应该能够成功运行以下情况
            with torch.cpu.amp.autocast():
                m(x, (hx, cx))

    # 测试在dtype为torch.float32且autocast禁用的情况下的行为
    def test_autocast_disabled_with_fp32_dtype(self):
        # 使用torch.autocast上下文管理器，设备类型为cpu，dtype为torch.float32，禁用autocast，并创建一个全为1的张量
        with torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False):
            _ = torch.ones(10)
    # 定义一个测试方法，用于测试通用自动类型转换功能
    def test_generic_autocast(self):
        # 遍历自动类型转换列表中的操作及其参数组合
        for op_with_args in self.autocast_lists.torch_16:
            # 获取操作名称、参数及可能的关键字参数
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            # 使用 torch 的自动类型转换装饰器，指定设备类型为 "cpu"
            with torch.amp.autocast(device_type="cpu"):
                # 调用指定操作及参数的方法，并记录输出结果
                generic_autocast_output = getattr(torch, op)(*args, **maybe_kwargs)
            # 使用 torch.cpu.amp.autocast 装饰器进行自动类型转换
            with torch.cpu.amp.autocast():
                # 再次调用相同操作及参数的方法，并记录输出结果
                cpu_autocast_output = getattr(torch, op)(*args, **maybe_kwargs)
            # 断言两次自动类型转换的输出结果相等
            self.assertEqual(generic_autocast_output, cpu_autocast_output)

    # 定义测试方法，验证在使用过时的警告时能否触发 FutureWarning
    def test_cpu_autocast_deprecated_warning(self):
        # 使用 self.assertWarnsRegex 断言能捕获到 FutureWarning 并匹配指定的警告信息
        with self.assertWarnsRegex(
            FutureWarning,
            r"`torch.cpu.amp.autocast\(args...\)` is deprecated. Please use `torch.amp.autocast\('cpu', args...\)` instead.",
        ):
            # 使用过时的 torch.cpu.amp.autocast 装饰器
            with torch.cpu.amp.autocast():
                # 创建一个张量，并忽略其值
                _ = torch.ones(10)
class CustomLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_t):
        # 保存输入张量 x 和权重张量 w_t，以便在反向传播时使用
        ctx.save_for_backward(x, w_t)
        # 调用 PyTorch 提供的线性函数进行前向传播
        return torch.nn.functional.linear(x, w_t)

    @staticmethod
    def backward(ctx, grad_output):
        # 从上下文中获取保存的张量 x 和 w_t
        x, w_t = ctx.saved_tensors
        # 使用自动混合精度加速，在 CUDA 设备上计算对输入 x 的梯度
        with torch.autocast(device_type="cuda"):
            dL_dX = torch.matmul(grad_output, w_t)
            # 计算对权重 w_t 的梯度，并在计算之前将 x 进行转置
            dL_dW = torch.matmul(x.transpose(0, 1), grad_output).transpose(0, 1)
        # 返回计算得到的梯度：对输入 x 和权重 w_t 的梯度
        return dL_dX, dL_dW


class WeightDTypeCastCounterMode(TorchDispatchMode):
    def __init__(self, weight):
        super().__init__()
        # 记录权重类型转换的次数
        self.dtype_cast_counter = 0
        self.weight = weight

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 检查调用的函数是否是类型转换函数，并且转换目标是权重张量为 float16
        if (
            func is torch.ops.aten._to_copy.default
            and args[0] is self.weight
            and kwargs["dtype"] is torch.float16
        ):
            # 如果符合条件，增加类型转换计数器
            self.dtype_cast_counter += 1
        # 调用原始的 Torch 函数
        return func(*args, **kwargs)

    def __enter__(self):
        # 临时禁用自动混合精度缓存清理函数
        self.old_clear_cache = torch.clear_autocast_cache
        torch.clear_autocast_cache = lambda: None
        # 调用父类的 __enter__ 方法
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始的自动混合精度缓存清理函数
        torch.clear_autocast_cache = self.old_clear_cache
        # 调用父类的 __exit__ 方法
        return super().__exit__(exc_type, exc_val, exc_tb)


@unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
class TestAutocastGPU(TestCase):
    def test_cast_cache_is_global(self):
        """
        验证自动混合精度缓存是全局的。通过模拟在前向传播结束时不清理缓存，
        在带有显式自动混合精度调用的反向传播中运行前向+反向传播，
        并验证权重只被转换为 float16 一次。
        """

        data = torch.randn(2, 3).cuda()
        weight = torch.nn.Parameter(torch.randn(4, 3).cuda())

        with WeightDTypeCastCounterMode(weight) as mode:
            with torch.autocast(device_type="cuda"):
                output = CustomLinear.apply(data, weight)
                s = output.sum()
            s.backward()

        # 验证权重类型转换的次数是否为 1
        self.assertEqual(mode.dtype_cast_counter, 1)

    def test_cache_disabled(self):
        data = torch.randn(2, 3).cuda()
        weight = torch.nn.Parameter(torch.randn(4, 3).cuda())

        try:
            torch._C._set_cached_tensors_enabled(True)
            torch._C._add_cached_tensor(weight)

            with WeightDTypeCastCounterMode(weight) as mode:
                with torch.autocast(device_type="cuda"):
                    output = CustomLinear.apply(data, weight)
                    s = output.sum()
                s.backward()

            # 验证不应该缓存权重转换的次数
            self.assertEqual(mode.dtype_cast_counter, 2)

        finally:
            torch._C._set_cached_tensors_enabled(False)


class TestTorchAutocast(TestCase):
    # 这里是测试用例的开始，没有添加额外的代码
    def test_autocast_fast_dtype(self):
        # 获取 GPU 和 CPU 的快速自动混合精度数据类型
        gpu_fast_dtype = torch.get_autocast_gpu_dtype()
        cpu_fast_dtype = torch.get_autocast_cpu_dtype()
        # 断言 GPU 快速数据类型为 torch.half
        self.assertEqual(gpu_fast_dtype, torch.half)
        # 断言 CPU 快速数据类型为 torch.bfloat16
        self.assertEqual(cpu_fast_dtype, torch.bfloat16)

    def test_invalid_device(self):
        # 设定一个无效的设备字符串
        dev = "not a real device"
        msg = f"Invalid device string: '{dev}'"
        # 使用断言检测在使用无效设备时抛出的 RuntimeError，并验证错误消息
        with self.assertRaisesRegex(RuntimeError, msg):
            with torch.autocast(device_type=dev):
                _ = torch.tensor(1)
        # 再次使用断言检测在查询无效设备时抛出的 RuntimeError，并验证错误消息
        with self.assertRaisesRegex(RuntimeError, msg):
            assert torch.amp.is_autocast_available(device_type=dev)

    def test_non_string_device(self):
        """测试当为 `device_type` 提供 `torch.device` 对象而不是字符串时，`autocast` 抛出 ValueError 异常"""
        # 创建一个 `torch.device` 对象作为设备类型
        dev = torch.device("cpu")
        msg = f"Expected `device_type` of type `str`, got: `{type(dev)}`"
        # 使用断言检测在使用 `torch.device` 对象作为设备类型时抛出的 ValueError，并验证错误消息
        with self.assertRaisesRegex(expected_exception=ValueError, expected_regex=msg):
            torch.autocast(device_type=dev)
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```