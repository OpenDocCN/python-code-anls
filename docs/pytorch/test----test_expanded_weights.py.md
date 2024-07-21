# `.\pytorch\test\test_expanded_weights.py`

```
# Owner(s): ["module: nn"]
# 导入单元测试模块
import unittest
# 导入数据类装饰器
from dataclasses import dataclass
# 导入偏函数工具
from functools import partial
# 导入链式工具和笛卡尔积工具
from itertools import chain, product

# 导入PyTorch库
import torch
# 导入PyTorch神经网络模块
import torch.nn as nn
# 导入PyTorch函数库
import torch.nn.functional as F
# 导入交叉熵损失函数
from torch.nn import CrossEntropyLoss
# 导入扩展权重相关的工具
from torch.nn.utils._expanded_weights import ExpandedWeight
# 导入扩展权重工具函数
from torch.nn.utils._expanded_weights.expanded_weights_utils import (
    forward_helper,
    set_grad_sample_if_exists,
    standard_kwargs,
    sum_over_all_but_batch_and_last_n,
    unpack_expanded_weight_or_tensor,
)
# 导入单样本梯度相关函数
from torch.nn.utils._per_sample_grad import call_for_per_sample_grads
# 导入CUDA测试相关模块和tf32禁用工具
from torch.testing._internal.common_cuda import TEST_CUDA, tf32_off
# 导入设备类型实例化测试和操作数据类型相关工具
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
# 导入操作数据库和样本输入工具
from torch.testing._internal.common_methods_invocations import op_db, SampleInput
# 导入模块数据库和模块测试工具
from torch.testing._internal.common_modules import module_db, modules
# 导入神经网络模块测试基类和新模块测试工具
from torch.testing._internal.common_nn import module_tests, new_module_tests, TestBase
# 导入通用工具函数，如随机数冻结、生成张量、参数化、运行测试、Torch Dynamo跳过等
from torch.testing._internal.common_utils import (
    freeze_rng_state,
    make_tensor,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
# 导入_pytree工具，仅进行树映射
from torch.utils._pytree import tree_map_only


# 定义测试上下文类
class TestContext:
    pass


# 定义扩展权重辅助函数测试类
class TestExpandedWeightHelperFunction(TestCase):
    # 定义测试线性函数前向传播辅助函数
    def test_forward_helper(self, device):
        # 生成随机输入张量和权重张量，设备由参数传入
        input = torch.randn(3, 4, device=device)
        weight = torch.randn(5, 4, device=device)
        bias = torch.randn(5, device=device)
        
        # 对权重和偏置是否批处理进行笛卡尔积遍历
        for weight_batched, bias_batched in product([True, False], [True, False]):
            # 初始化可能被批处理的权重和偏置
            maybe_batched_weight = weight
            maybe_batched_bias = bias
            
            # 如果权重需要批处理，则创建ExpandedWeight对象
            if weight_batched:
                maybe_batched_weight = ExpandedWeight(
                    weight.clone().requires_grad_(), 3, loss_reduction="sum"
                )
            
            # 如果偏置需要批处理，则创建ExpandedWeight对象
            if bias_batched:
                maybe_batched_bias = ExpandedWeight(
                    bias.clone().requires_grad_(), 3, loss_reduction="sum"
                )
            
            # 构建函数调用参数
            args = (input, maybe_batched_weight, maybe_batched_bias)
            # 扩展参数和关键字参数
            expanded_args, expanded_kwargs = standard_kwargs(("bias",), args)
            # 调用前向传播辅助函数进行计算
            res = forward_helper(nn.functional.linear, expanded_args, expanded_kwargs)
            # 期望的结果，直接调用PyTorch的线性函数
            expected = nn.functional.linear(input, weight, bias)
            # 断言计算结果与期望结果一致
            self.assertEqual(res, expected)
            
            # 断言扩展后的参数数量为2
            self.assertEqual(len(expanded_args), 2)
            # 避免在断言中进行属性检查
            assert expanded_args[0] is args[0]  # avoids property checks in assertEquals
            assert expanded_args[1] is args[1]  # avoids property checks in assertEquals
            
            # 断言扩展后的关键字参数数量为1
            self.assertEqual(len(expanded_kwargs), 1)
            # 避免在断言中进行属性检查
            assert (
                expanded_kwargs["bias"] is args[2]
            )  # avoids property checks in assertEquals
    ```python`
        # 定义测试方法，用于测试前向传播辅助函数在参数错误时是否能正确抛出异常
        def test_forward_helper_failure_args(self, device):
            # 生成随机权重张量，形状为 (5, 4)，放置在指定设备上
            weight = torch.randn(5, 4, device=device)
            # 生成随机偏置张量，形状为 (5)，放置在指定设备上
            bias = torch.randn(5, device=device)
            
            # 测试输入为 ExpandedWeight 对象时是否抛出指定 RuntimeError 异常
            with self.assertRaisesRegex(
                RuntimeError, r"do not support inputs that are also ExpandedWeights."
            ):
                # 创建 ExpandedWeight 对象作为输入，期望抛出异常
                input = ExpandedWeight(
                    torch.randn(3, 4, requires_grad=True), 3, loss_reduction="sum"
                )
                # 根据标准参数生成扩展后的参数和关键字参数
                expanded_args, expanded_kwargs = standard_kwargs(
                    ("bias",), (input, weight, bias)
                )
                # 调用前向传播辅助函数，预期抛出异常
                forward_helper(nn.functional.linear, expanded_args, expanded_kwargs)
            
            # 测试第一个输入参数不是 Tensor 对象时是否抛出指定 RuntimeError 异常
            with self.assertRaisesRegex(
                RuntimeError, r"requires a Tensor as the first input"
            ):
                # 创建不是 Tensor 对象的输入参数，期望抛出异常
                expanded_args, expanded_kwargs = standard_kwargs(
                    ("bias",), (3, weight, bias)
                )
                # 调用前向传播辅助函数，预期抛出异常
                forward_helper(nn.functional.linear, expanded_args, expanded_kwargs)
            
            # 测试输入参数没有批量维度但是期望有时是否抛出指定 RuntimeError 异常
            with self.assertRaisesRegex(
                RuntimeError, r"requires a batch dimension but got an input of size 0"
            ):
                # 创建没有批量维度的输入参数，期望抛出异常
                expanded_args, expanded_kwargs = standard_kwargs(
                    ("bias",), (torch.tensor(3), weight, bias)
                )
                # 调用前向传播辅助函数，预期抛出异常
                forward_helper(nn.functional.linear, expanded_args, expanded_kwargs)
            
            # 测试输入参数批量大小为 0 时是否抛出指定 RuntimeError 异常
            with self.assertRaisesRegex(
                RuntimeError, r"0 is not a valid batch size for Expanded Weights"
            ):
                # 创建批量大小为 0 的输入参数，期望抛出异常
                expanded_args, expanded_kwargs = standard_kwargs(
                    ("bias",), (torch.randn(0, 1, 2), weight, bias)
                )
                # 调用前向传播辅助函数，预期抛出异常
                forward_helper(nn.functional.linear, expanded_args, expanded_kwargs)
            
            # 创建普通张量作为输入
            input = torch.randn(3, 4)
            # 使用 product 函数生成所有可能的权重和偏置是否批量化的组合进行测试
            for weight_batched, bias_batched in product([True, False], [True, False]):
                # 如果权重和偏置都不批量化，则跳过当前循环
                if not weight_batched and not bias_batched:
                    continue
                
                # 初始化可能被批量化的权重和偏置
                maybe_batched_weight = weight
                maybe_batched_bias = bias
                
                # 如果需要批量化权重，则创建 ExpandedWeight 对象
                if weight_batched:
                    maybe_batched_weight = ExpandedWeight(
                        weight.clone().requires_grad_(), 4, loss_reduction="sum"
                    )
                
                # 如果需要批量化偏置，则创建 ExpandedWeight 对象
                if bias_batched:
                    maybe_batched_bias = ExpandedWeight(
                        bias.clone().requires_grad_(), 4, loss_reduction="sum"
                    )
                
                # 测试 ExpandedWeight 对象的批量大小是否与输入匹配，预期抛出异常
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"Expected ExpandedWeights to have batch size matching input",
                ):
                    # 根据标准参数生成扩展后的参数和关键字参数
                    expanded_args, expanded_kwargs = standard_kwargs(
                        ("bias",), (input, maybe_batched_weight, maybe_batched_bias)
                    )
                    # 调用前向传播辅助函数，预期抛出异常
                    forward_helper(nn.functional.linear, expanded_args, expanded_kwargs)
    # 定义测试方法，用于验证是否能正确设置梯度样本（grad_sample）属性
    def test_set_grad_sample_if_exists(self, device):
        # 内部定义的测试函数，返回 grad_sample
        def test_fn(a):
            return grad_sample

        # 创建一个具有梯度的随机张量 orig_weight
        orig_weight = torch.randn(4, device=device, requires_grad=True)
        # 创建一个 ExpandedWeight 对象，传入 orig_weight，loss_reduction 设置为 "sum"
        expanded_weight = ExpandedWeight(orig_weight, 3, loss_reduction="sum")
        # 创建一个随机的梯度样本 grad_sample
        grad_sample = torch.randn(3)
        # 调用 set_grad_sample_if_exists 函数，将 grad_sample 设置到 expanded_weight 中
        set_grad_sample_if_exists(expanded_weight, test_fn)
        # 断言 orig_weight 是否有 grad_sample 属性
        self.assertTrue(hasattr(orig_weight, "grad_sample"))
        # 断言 orig_weight.grad_sample 是否与 grad_sample 相等
        self.assertEqual(orig_weight.grad_sample, grad_sample)

        # 创建一个普通的张量 basic_tensor
        basic_tensor = torch.randn(4, device=device)
        # 调用 set_grad_sample_if_exists 函数，但由于 basic_tensor 没有 grad_sample 属性，因此不会设置
        set_grad_sample_if_exists(basic_tensor, test_fn)
        # 断言 basic_tensor 是否没有 grad_sample 属性
        self.assertFalse(hasattr(basic_tensor, "grad_sample"))

        # 创建一个非张量对象 non_tensor
        non_tensor = 3
        # 调用 set_grad_sample_if_exists 函数，因为 non_tensor 不是张量，也不会设置 grad_sample
        set_grad_sample_if_exists(non_tensor, test_fn)
        # 断言 non_tensor 是否没有 grad_sample 属性
        self.assertFalse(hasattr(non_tensor, "grad_sample"))

    # 定义测试方法，用于验证 set_grad_sample_if_exists 在特定情况下会引发异常
    def test_set_grad_sample_if_exists_failure(self, device):
        # 内部定义的测试函数，始终返回 True
        def test_fn(a):
            return True

        # 创建一个具有梯度的随机张量 grad_tensor
        grad_tensor = torch.randn(4, requires_grad=True, device=device)
        # 使用 assertRaisesRegex 断言，在调用 set_grad_sample_if_exists 时会引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            r"does not support a mixture of ExpandedWeight parameters and normal Parameters",
        ):
            set_grad_sample_if_exists(grad_tensor, test_fn)

    # 定义测试方法，用于验证 unpack_expanded_weight_or_tensor 函数的行为
    def test_unpack_expanded_weight_or_tensor(self, device):
        # 创建一个具有梯度的随机张量 input
        input = torch.randn(3, requires_grad=True, device=device)
        # 使用 unpack_expanded_weight_or_tensor 函数，传入 ExpandedWeight 对象，应该返回 input 本身
        self.assertEqual(
            input,
            unpack_expanded_weight_or_tensor(
                ExpandedWeight(input, 3, loss_reduction="sum")
            ),
        )

        # 将 input 的 requires_grad 设置为 False
        input.requires_grad_(False)
        # 使用 unpack_expanded_weight_or_tensor 函数，传入普通张量 input，应该返回 input 本身
        self.assertEqual(input, unpack_expanded_weight_or_tensor(input))
        # 使用 unpack_expanded_weight_or_tensor 函数，传入非张量对象 4，应该返回 None
        self.assertTrue(unpack_expanded_weight_or_tensor(4) is None)

    # 定义测试方法，用于验证带有自定义函数的 unpack_expanded_weight_or_tensor 函数的行为
    def test_unpack_expanded_weight_or_tensor_with_custom_function(self, device):
        # 创建一个具有梯度的随机张量 input
        input = torch.randn(3, requires_grad=True, device=device)
        # 使用 unpack_expanded_weight_or_tensor 函数，传入 ExpandedWeight 对象和 lambda 函数验证是否是 input
        self.assertTrue(
            unpack_expanded_weight_or_tensor(
                ExpandedWeight(input, 3, loss_reduction="sum"), lambda x: x is input
            )
        )

        # 将 input 的 requires_grad 设置为 False
        input.requires_grad_(False)
        # 使用 unpack_expanded_weight_or_tensor 函数，传入普通张量 input 和 lambda 函数验证是否是 input
        self.assertTrue(unpack_expanded_weight_or_tensor(input, lambda x: x is input))
        # 使用 unpack_expanded_weight_or_tensor 函数，传入非张量对象 4 和 lambda 函数验证是否是 input，应该返回 None
        self.assertTrue(
            unpack_expanded_weight_or_tensor(4, lambda x: x is input) is None
        )

    # 定义测试方法，用于验证 unpack_expanded_weight_or_tensor 在特定情况下会引发异常
    def test_unpack_expanded_weight_or_tensor_failure(self, device):
        # 创建一个具有梯度的随机张量 input
        input = torch.randn(3, requires_grad=True, device=device)
        # 使用 assertRaisesRegex 断言，在调用 unpack_expanded_weight_or_tensor 时会引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            r"does not support a mixture of ExpandedWeight parameters and normal Parameters",
        ):
            unpack_expanded_weight_or_tensor(input)

        # 使用 assertRaisesRegex 断言，在调用 unpack_expanded_weight_or_tensor 时会引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            r"does not support a mixture of ExpandedWeight parameters and normal Parameters",
        ):
            unpack_expanded_weight_or_tensor(input, lambda x: x is input)
    # 定义一个测试函数，用于测试 sum_over_all_but_batch_and_last_n 函数的不同情况
    def test_sum_over_all_but_batch_and_last_n(self, device):
        # 创建一个随机张量作为输入，形状为 (1, 2, 3, 4, 5)，在指定设备上
        input = torch.randn(1, 2, 3, 4, 5, device=device)
        # 调用 sum_over_all_but_batch_and_last_n 函数，对输入张量执行求和操作，保留最后两个维度
        res = sum_over_all_but_batch_and_last_n(input, 2)
        # 计算预期结果：对输入张量在第 1 和第 2 维度上求和
        expected = input.sum((1, 2))
        # 断言实际结果与预期结果相等
        self.assertEqual(res, expected)

        # 再次调用 sum_over_all_but_batch_and_last_n 函数，对输入张量执行求和操作，保留第 0 维度
        res = sum_over_all_but_batch_and_last_n(input, 0)
        # 计算预期结果：对输入张量在第 1、2、3、4 维度上求和
        expected = input.sum((1, 2, 3, 4))
        # 断言实际结果与预期结果相等
        self.assertEqual(res, expected)

        # 再次调用 sum_over_all_but_batch_and_last_n 函数，对输入张量执行求和操作，保留最后四个维度
        res = sum_over_all_but_batch_and_last_n(input, 4)
        # 断言实际结果与输入张量相等，因为保留了所有维度
        self.assertEqual(res, input)
# 扩展的权重功能测试类，继承自TestCase，用于单元测试
class TestExpandedWeightFunctional(TestCase):

    # 比较使用扩展权重对象和普通循环计算每个样本梯度的函数
    def _compare_ew_and_for_loop_per_sample_grads(self, op, sample_input, reduction):
        # 获取输入数据、额外参数和关键字参数
        input = sample_input.input
        args = sample_input.args
        kwargs = sample_input.kwargs
        batch_size = input.shape[0] if len(input.shape) > 1 else 1

        # 使用ExpandedWeights对象获取每个样本的梯度
        loss_reduction = "sum" if reduction == torch.sum else "mean"
        (ew_input, ew_args, ew_kwargs) = make_expanded_weight(
            sample_input, batch_size, loss_reduction
        )

        # 构建不同输入列表，仅包含需要计算梯度的张量
        diff_input_list = (ew_input,) + tuple(ew_args) + tuple(ew_kwargs.values())
        diff_input_list = [i for i in diff_input_list if is_diff_tensor(i)]

        # 将ExpandedWeight对象转换为原始权重张量
        diff_input_list = [
            i.orig_weight if isinstance(i, ExpandedWeight) else i
            for i in diff_input_list
        ]

        # 若没有需要计算梯度的输入张量，则直接返回
        if not diff_input_list:
            return

        # 运行操作op，并对结果进行反向传播
        result = run_op(op, ew_input, *ew_args, **ew_kwargs)
        reduction(result).backward()  # 使用扩展权重时，由于调用了__torch_function__，grad函数无法工作
        # 提取每个张量的梯度信息，如果存在grad_sample属性则使用，否则使用grad属性
        expanded_weight_grad = tuple(
            i.grad_sample if hasattr(i, "grad_sample") else i.grad
            for i in diff_input_list
        )

        # 使用普通的for循环计算每个样本的梯度
        func = partial(run_op, op)
        per_sample_grad = for_loop_per_sample_grad(
            batch_size, reduction, input, func, *args, **kwargs
        )

        # 检查两种方法计算得到的每个样本梯度是否相等
        self.assertEqual(len(per_sample_grad), len(expanded_weight_grad))
        if loss_reduction == "mean":
            # 如果是均值损失函数，不需要比较input.grad的相等性，因为这些普通张量不会被缩放
            expanded_weight_grad = expanded_weight_grad[1:]
            per_sample_grad = per_sample_grad[1:]
        for result_grad, expected_grad in zip(expanded_weight_grad, per_sample_grad):
            self.assertEqual(result_grad, expected_grad)

    # 使用装饰器ops定义测试用例，筛选支持扩展权重的操作
    @ops(
        filter(lambda op: op.supports_expanded_weight, op_db),
        dtypes=OpDTypes.supported,
        allowed_dtypes=(torch.double,),
    )
    # 测试使用扩展权重计算每个样本梯度之和的情况
    def test_expanded_weight_per_sample_grad_sum(self, device, dtype, op):
        # 获取操作op在指定设备和数据类型下的样本输入数据，要求支持梯度计算
        sample_inputs = op.sample_inputs(device, dtype, requires_grad=True)
        
        # 遍历所有支持的输入数据
        for sample_input in supported_inputs(op, sample_inputs):
            # 对于特定的操作nn.functional.embedding，进行额外的参数处理以适应自动求导测试
            if op.name == "nn.functional.embedding":
                sample_input = SampleInput(
                    sample_input.args[0],
                    args=(sample_input.input,),
                    kwargs=sample_input.kwargs,
                )

            # 比较使用扩展权重对象和普通循环计算每个样本梯度的函数
            self._compare_ew_and_for_loop_per_sample_grads(op, sample_input, torch.sum)

    # 使用装饰器ops定义测试用例，筛选支持扩展权重的操作
    @ops(
        filter(lambda op: op.supports_expanded_weight, op_db),
        dtypes=OpDTypes.supported,
        allowed_dtypes=(torch.double,),
    )
    # 定义一个测试方法，用于检查扩展权重每个样本梯度的均值
    def test_expanded_weight_per_sample_grad_mean(self, device, dtype, op):
        # 获取包含梯度信息的样本输入
        sample_inputs = op.sample_inputs(device, dtype, requires_grad=True)
        # 遍历支持的输入样本
        for sample_input in supported_inputs(op, sample_inputs):
            # 如果操作名称是 "nn.functional.embedding"
            if (
                op.name == "nn.functional.embedding"
            ):  # embedding 在自动求导测试中反转其参数顺序
                # 重新构造样本输入对象，调整参数顺序
                sample_input = SampleInput(
                    sample_input.args[0],
                    args=(sample_input.input,),
                    kwargs=sample_input.kwargs,
                )

            # 比较扩展权重和循环方法计算的每个样本梯度
            self._compare_ew_and_for_loop_per_sample_grads(op, sample_input, torch.mean)

    # 装饰器，指定操作支持扩展权重，测试扩展权重每个样本梯度且输入不需要梯度
    @ops(
        filter(lambda op: op.supports_expanded_weight, op_db),
        dtypes=OpDTypes.supported,
        allowed_dtypes=(torch.double,),
    )
    def test_expanded_weights_per_sample_grad_input_no_grad(self, device, dtype, op):
        # 获取包含梯度信息的样本输入
        sample_inputs = op.sample_inputs(device, dtype, requires_grad=True)
        # 遍历支持的输入样本
        for sample_input in supported_inputs(op, sample_inputs):
            # 如果操作名称是 "nn.functional.embedding"
            if (
                op.name == "nn.functional.embedding"
            ):  # embedding 在自动求导测试中反转其参数顺序
                # 重新构造样本输入对象，调整参数顺序
                sample_input = SampleInput(
                    sample_input.args[0],
                    args=(sample_input.input,),
                    kwargs=sample_input.kwargs,
                )
            # 将输入的 requires_grad 设置为 False
            sample_input.input.requires_grad_(False)

            # 比较扩展权重和循环方法计算的每个样本梯度
            self._compare_ew_and_for_loop_per_sample_grads(op, sample_input, torch.mean)

    # 装饰器，跳过 Torch Dynamo 环境中的测试，因为错误消息检查无效
    @skipIfTorchDynamo("Checking error message doesn't work with dynamo")
    @ops(
        filter(lambda op: op.supports_expanded_weight, op_db),
        dtypes=OpDTypes.supported,
        allowed_dtypes=(torch.double,),
    )
    # 定义测试函数，用于测试不支持的扩展权重情况
    def test_unsupported_expand_weights(self, device, dtype, op):
        # 生成操作的样本输入，要求计算梯度
        sample_inputs = op.sample_inputs(device, dtype, requires_grad=True)
        # 获取不支持的输入集合
        unsupported_inputs = supported_inputs(op, sample_inputs, supported_inputs=False)
        # 对于每个不支持的样本输入进行迭代
        for sample_input in unsupported_inputs:
            # 使用断言检查是否引发了 RuntimeError，并且错误信息包含"Expanded Weights"
            with self.assertRaisesRegex(RuntimeError, r"Expanded Weights"):
                # 如果操作是 nn.functional.embedding，则为了自动求导测试，反转其参数顺序
                if (
                    op.name == "nn.functional.embedding"
                ):  # embedding flips its argument order for autograd tests
                    # 重新构造样本输入对象，交换参数顺序
                    sample_input = SampleInput(
                        sample_input.args[0],
                        args=(sample_input.input,),
                        kwargs=sample_input.kwargs,
                    )
                # 获取样本输入的数据部分
                input = sample_input.input

                # 计算输入的批量大小，如果输入的维度大于1，则使用其第一维的大小
                batch_size = input.shape[0] if len(input.shape) > 1 else 1

                # 使用 make_expanded_weight 函数创建 ExpandedWeights 对象
                (ew_input, ew_args, ew_kwargs) = make_expanded_weight(
                    sample_input, batch_size
                )
                # 运行操作函数 op，并传入扩展权重对象及其它参数
                result = run_op(op, ew_input, *ew_args, **ew_kwargs)

                # 构建包含所有不同张量的列表，这些张量用于计算梯度
                diff_input_list = (
                    (ew_input,) + tuple(ew_args) + tuple(ew_kwargs.values())
                )
                # 过滤出真正需要计算梯度的张量，仅保留 ExpandedWeight 类型的原始权重
                diff_input_list = [i for i in diff_input_list if is_diff_tensor(i)]
                diff_input_list = [
                    i.orig_weight if isinstance(i, ExpandedWeight) else i
                    for i in diff_input_list
                ]
                # 对结果进行求和并计算其反向传播梯度，ExpandedWeight 对象因调用 __torch_function__ 而无法工作
                result.sum().backward()
    
    # 使用 ops 装饰器标记函数，该装饰器包含支持扩展权重的操作集合
    @ops(
        filter(lambda op: op.supports_expanded_weight, op_db), dtypes=OpDTypes.supported
    )
    # 测试扩展权重计算的前向传播功能
    def test_expanded_weight_forward(self, device, dtype, op):
        # 生成操作的样例输入数据
        sample_inputs = op.sample_inputs(device, dtype)
        # 遍历支持的输入样例
        for sample_input in supported_inputs(op, sample_inputs):
            # 对于 nn.functional.embedding，为了自动求导测试，反转其参数顺序
            if (
                op.name == "nn.functional.embedding"
            ):  # embedding flips its argument order for autograd tests
                # 克隆第一个参数，并重新构造样例输入
                sample_input = SampleInput(
                    sample_input.args[0].clone(),
                    args=(sample_input.input.clone(),),
                    kwargs=sample_input.kwargs,
                )
                # 在 CUDA 设备上，特定条件下跳过测试
                if (
                    "cuda" in device
                    and "max_norm" in sample_input.kwargs
                    and "padding_idx" in sample_input.kwargs
                ):
                    self.skipTest(
                        "embedding is non-determinstic in this case, see issue #74679"
                    )
            # 计算批量大小
            batch_size = (
                sample_input.input.shape[0] if len(sample_input.input.shape) > 1 else 1
            )
            # 遍历损失减少方式（sum、mean）
            for loss_reduction in ["sum", "mean"]:
                # 生成扩展权重所需的输入、参数和关键字参数
                (ew_input, ew_args, ew_kwargs) = make_expanded_weight(
                    sample_input, batch_size, loss_reduction
                )
                # 运行操作，计算扩展权重的结果
                expanded_weight_result = run_op(op, ew_input, *ew_args, **ew_kwargs)
                # 运行正常操作，计算正常结果
                normal_result = run_op(
                    op, sample_input.input, *sample_input.args, **sample_input.kwargs
                )
                # 断言扩展权重的结果与正常结果相等
                self.assertEqual(expanded_weight_result, normal_result)

    # 测试扩展权重计算中的错误处理
    def test_expanded_weight_error(self, device):
        # 设置批量大小
        batch_size = 3
        # 生成带梯度的输入张量样例
        sample_input = make_tensor(
            (batch_size, 4), dtype=torch.float32, device=device, requires_grad=True
        )
        # 生成带梯度的权重张量样例
        sample_weight = make_tensor(
            (4), dtype=torch.float32, device=device, requires_grad=True
        )
        # 使用断言检查运行时错误，并提供相应的错误信息正则表达式
        with self.assertRaisesRegex(
            RuntimeError, r"Expanded Weights encountered but cannot handle function"
        ):
            # 尝试执行 torch.add 操作，使用扩展权重作为参数
            torch.add(
                sample_input,
                ExpandedWeight(sample_weight, batch_size, loss_reduction="sum"),
            )

    # 测试嵌入模型的功能
    def _test_embedding_model(self, model, num_embedding, device):
        # 固定批量大小
        batch_size = 32
        # 生成随机整数输入张量，用于模型测试
        input = torch.randint(0, num_embedding, (batch_size, 5, 5), device=device)
        # 调用通用模型测试方法，返回测试结果
        return self._test_model(
            partial(model, num_embedding=num_embedding), batch_size, input, device
        )

    # 测试卷积模型的功能
    def _test_conv_model(
        self,
        model,
        input_size,
        num_dim,
        device,
        loss_reduction="sum",
        atol=1e-4,
        rtol=5e-5,
    ):
        # 固定批量大小
        batch_size = 32
        # 根据给定的输入大小和维度数量，生成随机输入张量
        input_ending = [input_size] * num_dim
        input = torch.randn([batch_size, 3] + input_ending, device=device)
        # 调用通用模型测试方法，返回测试结果
        return self._test_model(
            partial(model, num_dim=num_dim),
            batch_size,
            input,
            device,
            loss_reduction,
            atol,
            rtol,
        )
    # 定义一个测试方法，用于测试给定模型的行为
    def _test_model(
        self,
        model,
        batch_size,
        input,
        device,
        loss_reduction="sum",
        atol=1e-4,
        rtol=5e-5,
    ):
        # 根据给定的模型实例化模型对象，并将其移动到指定设备上
        model = model(10).to(device)
        # 生成随机目标张量，用于计算损失
        targets = torch.randint(0, 10, (batch_size,), device=device)
        # 定义损失函数为交叉熵损失函数，设置损失的归约方式
        criterion = CrossEntropyLoss(reduction=loss_reduction)
        # 调用一个函数，计算每个样本的梯度
        result = call_for_per_sample_grads(model, loss_reduction=loss_reduction)(input)
        # 计算当前批次的总体损失
        loss = criterion(result, targets)
        # 反向传播，计算参数的梯度
        loss.backward()
        # 初始化一个空列表，用于存储模型参数的梯度样本
        result = []
        # 遍历模型的参数
        for weight in model.parameters():
            # 将每个参数的梯度样本添加到结果列表中
            result.append(weight.grad_sample)
            # 删除参数的梯度样本以释放内存
            del weight.grad_sample

        # 初始化一个空列表，用于存储期望的梯度
        expected = []
        # 遍历当前批次中的每个样本
        for i in range(batch_size):
            # 计算当前样本的损失
            loss = criterion(model(input[i].unsqueeze(0)), targets[i].unsqueeze(0))
            # 计算当前样本的期望梯度
            expected.append(
                torch.autograd.grad(loss, model.parameters(), torch.ones_like(loss))
            )

        # 将期望的梯度堆叠成张量
        expected = [torch.stack(grad) for grad in zip(*expected)]
        # 逐一比较计算得到的梯度样本和期望的梯度
        for res, exp in zip(result, expected):
            # 使用断言比较两者，确保它们在给定的容差范围内相等
            self.assertEqual(res, exp, atol=atol, rtol=rtol)

    # 定义一个方法，用于根据设备计算容差值
    def _compute_tolerances(self, device):
        # 检查设备是否是cuda并且是否支持sm86架构，根据结果设置不同的容差值
        is_cuda_sm86 = device.startswith("cuda") and torch.cuda.get_device_capability(
            0
        ) == (8, 6)
        return (9e-3, 5e-5) if is_cuda_sm86 else (1e-4, 5e-5)

    # 装饰器函数，用于关闭tf32模式
    @tf32_off()
    # 定义一个测试卷积神经网络模型的方法，计算损失的归约为sum
    def test_cnn_model_sum(self, device):
        # 定义一个简单的卷积神经网络模型
        def convnet(num_classes, num_dim):
            return nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(128, num_classes, bias=True),
            )

        # 根据设备计算当前测试的容差值
        atol, rtol = self._compute_tolerances(device)
        # 调用测试卷积神经网络模型的方法，并返回其结果
        return self._test_conv_model(convnet, 28, 2, device, atol=atol, rtol=rtol)

    # 装饰器函数，用于关闭tf32模式
    @tf32_off()
    # 定义一个测试方法，用于测试 CNN 模型的平均值损失
    def test_cnn_model_mean(self, device):
        # 定义一个简单的卷积神经网络模型
        def convnet(num_classes, num_dim):
            return nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 第一层卷积层
                nn.ReLU(),  # ReLU 激活函数
                nn.AvgPool2d(kernel_size=2, stride=2),  # 平均池化层
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 第二层卷积层
                nn.ReLU(),  # ReLU 激活函数
                nn.AvgPool2d(kernel_size=2, stride=2),  # 平均池化层
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 第三层卷积层
                nn.ReLU(),  # ReLU 激活函数
                nn.AvgPool2d(kernel_size=2, stride=2),  # 平均池化层
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 第四层卷积层
                nn.ReLU(),  # ReLU 激活函数
                nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化层
                nn.Flatten(start_dim=1, end_dim=-1),  # 将多维数据展平成一维
                nn.Linear(128, num_classes, bias=True),  # 全连接层
            )

        # 计算测试的绝对容差和相对容差
        atol, rtol = self._compute_tolerances(device)
        # 调用 _test_conv_model 方法进行测试，并返回测试结果
        return self._test_conv_model(
            convnet, 28, 2, device, loss_reduction="mean", atol=atol, rtol=rtol
        )

    # 使用参数化装饰器定义一个测试方法，测试实例归一化模型
    @parametrize("num_dim", [1, 2, 3])
    @tf32_off()
    def test_instance_norm_model(self, num_dim, device):
        # 定义一个实例归一化模型
        def instance_norm_model(num_classes, num_dim):
            conv_layer = (
                nn.Conv1d if num_dim == 1 else nn.Conv2d if num_dim == 2 else nn.Conv3d
            )
            norm_layer = (
                nn.InstanceNorm1d
                if num_dim == 1
                else nn.InstanceNorm2d
                if num_dim == 2
                else nn.InstanceNorm3d
            )
            return nn.Sequential(
                conv_layer(3, 32, kernel_size=3, stride=1, padding=1),  # 卷积层
                norm_layer(32, affine=True),  # 实例归一化层
                nn.Flatten(start_dim=1, end_dim=-1),  # 将多维数据展平成一维
                nn.Linear(32 * (7**num_dim), num_classes, bias=True),  # 全连接层
            )

        # 计算测试的绝对容差和相对容差
        atol, rtol = self._compute_tolerances(device)
        # 调用 _test_conv_model 方法进行测试，并返回测试结果
        return self._test_conv_model(
            instance_norm_model, 7, num_dim, device, atol=atol, rtol=rtol
        )

    # 使用参数化装饰器定义一个测试方法，测试组归一化模型
    @parametrize("num_dim", [1, 2, 3])
    @tf32_off()
    def test_group_norm_model(self, num_dim, device):
        # 定义一个组归一化模型
        def group_norm_model(num_classes, num_dim):
            conv_layer = (
                nn.Conv1d if num_dim == 1 else nn.Conv2d if num_dim == 2 else nn.Conv3d
            )
            return nn.Sequential(
                conv_layer(3, 32, kernel_size=3, stride=1, padding=1),  # 卷积层
                nn.GroupNorm(8, 32, affine=True),  # 组归一化层
                nn.Flatten(start_dim=1, end_dim=-1),  # 将多维数据展平成一维
                nn.Linear(32 * (7**num_dim), num_classes, bias=True),  # 全连接层
            )

        # 计算测试的绝对容差和相对容差
        atol, rtol = self._compute_tolerances(device)
        # 调用 _test_conv_model 方法进行测试，并返回测试结果
        return self._test_conv_model(
            group_norm_model, 7, num_dim, device, atol=atol, rtol=rtol
        )
    # 定义一个测试方法，用于测试带有层归一化的模型
    def test_layer_norm_model(self, num_dim, device):
        # 定义一个层归一化模型的内部函数
        def layer_norm_model(num_classes, num_dim):
            # 根据维度选择合适的卷积层
            conv_layer = (
                nn.Conv1d if num_dim == 1 else nn.Conv2d if num_dim == 2 else nn.Conv3d
            )
            # 创建用于层归一化的标准化形状，每个维度长度为7
            normalized_shape = [7] * num_dim
            # 返回一个包含卷积层、层归一化、展平层和线性层的顺序模型
            return nn.Sequential(
                conv_layer(3, 32, kernel_size=3, stride=1, padding=1),
                nn.LayerNorm(normalized_shape, elementwise_affine=True),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(32 * (7**num_dim), num_classes, bias=True),
            )

        # 计算设备上的绝对误差和相对误差容差
        atol, rtol = self._compute_tolerances(device)
        # 调用内部测试方法，测试层归一化模型
        return self._test_conv_model(
            layer_norm_model, 7, num_dim, device, atol=atol, rtol=rtol
        )

    # 定义一个测试方法，用于测试嵌入模型
    def test_embedding_model(self, device):
        # 定义一个嵌入模型的内部函数
        def embedding_model(num_classes, num_embedding):
            # 返回一个包含嵌入层、展平层和线性层的顺序模型
            return nn.Sequential(
                nn.Embedding(num_embedding, 15),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(375, num_classes, bias=True),
            )

        # 调用内部测试方法，测试嵌入模型
        return self._test_embedding_model(embedding_model, 16, device)

    # 定义一个测试方法，用于测试分组归一化的错误情况
    def test_group_norm_error(self, device):
        # group norm 必须调用 native_group_norm，这里检查是否会出现与普通分组归一化相同的错误
        N = 3
        C = 5
        # 创建一个随机输入张量
        inp = torch.randn(N, C)
        # 使用断言检查是否会抛出特定的运行时错误
        with self.assertRaisesRegex(
            RuntimeError, r"Expected number of channels in input to be divisible"
        ):
            # 调用 F.group_norm 检查是否会抛出预期的错误，因为5不能被2整除
            F.group_norm(inp, 2)
class TestExpandedWeightModule(TestCase):
    # 定义测试类 TestExpandedWeightModule，继承自 TestCase
    def _do_test(
        self,
        module,
        input,
        args=None,
        kwargs=None,
        batch_first=True,
        atol=None,
        rtol=None,
    ):
        # 定义测试方法 _do_test，用于执行单元测试
        args = args or ()  # 如果 args 为 None，则设为空元组
        kwargs = kwargs or {}  # 如果 kwargs 为 None，则设为空字典

        batch_dim = 0 if batch_first else 1  # 根据 batch_first 确定 batch 维度
        batch_size = input.shape[batch_dim]  # 获取输入张量的 batch 大小
        diff_input = input.dtype == torch.float or input.dtype == torch.double  # 检查输入张量数据类型是否为浮点型

        if diff_input:
            input.requires_grad_()  # 如果输入需要梯度，则设置其需要计算梯度

        with freeze_rng_state():
            # 使用 freeze_rng_state 上下文管理器保持随机数生成器状态

            # 使用 ExpandedWeights 上下文管理器获取每个样本的梯度
            actual_res = call_for_per_sample_grads(
                module,
                batch_size=batch_size,
                loss_reduction="sum",
                batch_first=batch_first,
            )(input, *args, **kwargs).sum()  # 计算实际结果的总和并反向传播
            actual_res.backward()  # 计算实际结果的反向传播梯度
            actual_grads = []
            for param in module.parameters():
                actual_grads.append(param.grad_sample)  # 收集参数的梯度样本
                del param.grad_sample  # 删除参数的梯度样本
            if diff_input:
                actual_grads.append(input.grad.clone())  # 收集输入张量的梯度并克隆
                input.grad = torch.zeros_like(input.grad)  # 清零输入张量的梯度

            # 使用 for 循环获取每个样本的梯度
            expected_res = torch.tensor(
                0.0, device=input.device, dtype=actual_res.dtype
            )
            expected_grads = []
            for i in range(batch_size):
                input_slice = input.narrow(batch_dim, i, 1)  # 获取输入张量的切片
                input_slice = input_slice.squeeze(batch_dim)  # 压缩输入张量的维度

                # 对 args 中的张量执行 narrow 和 contiguous 操作
                sliced_args = tree_map_only(
                    torch.Tensor, lambda t: t.narrow(1, i, 1).contiguous(), args
                )

                diff_params = module.parameters()  # 获取模型参数
                if diff_input:
                    diff_params = chain(diff_params, (input_slice,))  # 若有输入张量，加入参数列表

                # 计算模型的输出结果
                res = module(
                    input_slice.unsqueeze(batch_dim).contiguous(),  # 将输入张量扩展并确保连续性
                    *sliced_args,
                    **kwargs,
                ).sum()  # 对结果求和
                out_grads = torch.autograd.grad(
                    res, diff_params, torch.ones_like(res), allow_unused=True
                )  # 计算输出结果对参数的梯度
                expected_grads.append(out_grads)  # 收集预期的梯度
                expected_res += res  # 累加预期结果

            expected_grads = [torch.stack(grad) for grad in zip(*expected_grads)]  # 对预期梯度进行堆叠操作
            if not batch_first:
                expected_grads[-1] = expected_grads[-1].transpose(0, 1)  # 若不是 batch_first，则转置最后一个梯度维度

        self.assertEqual(actual_res, expected_res)  # 断言实际结果与预期结果相等
        [
            self.assertEqual(actual, expected, atol=atol, rtol=rtol)
            for (actual, expected) in zip(actual_grads, expected_grads)
        ]  # 使用列表推导式对每个梯度进行断言
    def _do_test_multi_input(self, module, input):
        class TestModule(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, input):
                return self.module(input) + self.module(input)
        
        batch_size = input.shape[0]  # 获取输入张量的批量大小
        diff_input = input.dtype == torch.float or input.dtype == torch.double  # 检查输入张量的数据类型是否为浮点型或双精度浮点型
        if diff_input:
            input.requires_grad_()  # 如果输入需要梯度，则设置输入张量为需要梯度状态
        with freeze_rng_state():
            # 使用 freeze_rng_state 上下文管理器，通过 ExpandedWeights 获取每个样本的梯度，并调用 .backward() 两次
            test_module = TestModule(module)
            actual_res = call_for_per_sample_grads(test_module, loss_reduction="sum")(input).sum()
            actual_res.backward()
            actual_grads = []
            for param in module.parameters():
                actual_grads.append(param.grad_sample)
                del param.grad_sample  # 删除 param.grad_sample 属性以释放内存
            if diff_input:
                actual_grads.append(input.grad.clone())  # 添加输入张量的梯度副本到 actual_grads
                input.grad = torch.zeros_like(input.grad)  # 将输入张量的梯度置零

            # 使用 for 循环获取每个样本的梯度，两次遍历输入张量
            expected_grads = []
            for i in range(batch_size):
                input_slice = input[i]
                diff_params = module.parameters()
                if diff_input:
                    diff_params = chain(diff_params, (input_slice,))
                res = module(input_slice.unsqueeze(0)).sum()
                out_grads = torch.autograd.grad(
                    res, diff_params, torch.ones_like(res), allow_unused=True
                )
                expected_grads.append(out_grads)
        expected_grads = tuple(torch.stack(grad) for grad in zip(*expected_grads))  # 将 expected_grads 中的梯度堆叠为张量元组
        expected_grads = tuple(
            expected_grad
            for expected_grad in expected_grads
            if expected_grad is not None
        )  # 筛选非空的 expected_grads 元组
        assert [
            self.assertEqual(actual, 2 * expected)  # 断言实际梯度与期望梯度的两倍相等
            for (actual, expected) in zip(actual_grads, expected_grads)
        ]  # 对每个参数的实际梯度和期望梯度进行断言验证

    def _do_test_rnn_packed_sequence(
        self, module, input, args=None, kwargs=None, atol=None, rtol=None
    ):
        # 如果 args 为 None，则设置为一个空元组；否则保持不变
        args = args if args is not None else ()
        # 如果 kwargs 为 None，则设置为一个空字典；否则保持不变
        kwargs = kwargs if kwargs is not None else {}

        # 计算输入张量中的最大批次大小，并转换为 Python 中的整数类型
        batch_size = max(tuple(input.batch_sizes)).item()

        # 在冻结随机数生成器状态下执行以下代码块
        with freeze_rng_state():
            # 使用 ExpandedWeights 上下文管理器获取每个样本的梯度
            actual_res = call_for_per_sample_grads(
                module, batch_size=batch_size, loss_reduction="sum"
            )(input, *args, **kwargs).data.sum()
            # 对 actual_res 进行反向传播
            actual_res.backward()
            # 存储实际梯度的空列表
            actual_grads = []
            # 遍历模型的所有参数
            for param in module.parameters():
                # 断言每个参数的梯度样本数量等于批次大小
                self.assertEqual(param.grad_sample.shape[0], batch_size)
                # 将参数的梯度样本添加到 actual_grads 列表中
                actual_grads.append(param.grad_sample)
                # 删除 param.grad_sample，以释放内存
                del param.grad_sample

            # 将输入数据的梯度设置为与其相同形状的全零张量
            input.data.grad = torch.zeros_like(input.data)

            # 初始化期望结果为与 actual_res 相同形状的全零张量
            expected_res = torch.zeros_like(actual_res)
            # 存储期望梯度的空列表
            expected_grads = []
            # 对输入进行填充和解包，获取填充后的输入和序列大小
            padded_input, seq_sizes = torch.nn.utils.rnn.pad_packed_sequence(
                input, batch_first=True
            )
            # 遍历序列大小的范围
            for i in range(len(seq_sizes)):
                # 从填充后的输入中获取切片，长度为 seq_sizes[i]
                input_slice = padded_input[i].narrow(0, 0, seq_sizes[i])
                # 获取模型的不同参数
                diff_params = module.parameters()
                # 根据模型是否以 batch_first 形式组织，确定批次维度
                batch_dim = 0 if module.m.batch_first else 1
                # 计算模型在输入切片上的输出，并对结果进行求和
                res = module(input_slice.unsqueeze(batch_dim), *args, **kwargs).sum()
                # 将结果添加到 expected_res 中
                expected_res += res
                # 使用 torch.autograd.grad 计算输出梯度
                out_grads = torch.autograd.grad(
                    res, diff_params, torch.ones_like(res), allow_unused=True
                )
                # 将计算得到的输出梯度添加到 expected_grads 列表中
                expected_grads.append(out_grads)

            # 将 expected_grads 转换为张量列表，每个张量堆叠来自同一参数的梯度
            expected_grads = [torch.stack(grad) for grad in zip(*expected_grads)]
            # 使用断言检查 actual_res 是否等于 expected_res
            self.assertEqual(actual_res, expected_res)
            # 使用断言逐一比较 actual_grads 和 expected_grads 中的张量
            [
                self.assertEqual(actual, expected, atol=atol, rtol=rtol)
                for (actual, expected) in zip(actual_grads, expected_grads)
            ]

    @modules(
        # 从 module_db 中筛选出模块类型为 RNN、LSTM 或 GRU 的模块信息
        filter(
            lambda m_info: m_info.module_cls
            in (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU),
            module_db,
        )
    )
    @tf32_off()
    # 定义测试模块的方法，接受设备、数据类型、模块信息和训练标志位作为参数
    def test_module(self, device, dtype, module_info, training):
        # 定义一个包装类 RNNWrapper，用于包装给定的 RNN 模块
        class RNNWrapper(torch.nn.Module):
            # 初始化方法，接受模块构造函数及其参数和关键字参数
            def __init__(self, m_cons, args, kwargs):
                super().__init__()
                # 使用给定的构造函数和参数实例化模块并赋值给 self.m
                self.m = m_cons(*args, **kwargs)

            # 前向传播方法，接受任意数量的输入，调用模块的前向传播并返回结果的第一个元素
            def forward(self, *inps):
                ret = self.m(*inps)
                # 断言返回结果为元组类型
                assert isinstance(ret, tuple)
                return ret[0]

        # 定义一个函数 batch_hidden，用于处理隐藏状态 h 的批处理
        def batch_hidden(h):
            # 创建一个新的形状列表，维度比输入 h 的维度高一维
            new_h_shape = [1] * (len(h.shape) + 1)
            new_h_shape[1] = 2
            # 对输入的隐藏状态 h 进行扩展以实现批处理
            return h.unsqueeze(1).repeat(new_h_shape)

        # 从 module_info 中获取模块类
        module_cls = module_info.module_cls
        # 根据模块类和数据类型设定浮点数比较的容差值
        atol, rtol = (
            (1e-4, 1e-5)
            if module_cls == torch.nn.GRU and dtype == torch.float32
            else (None, None)
        )
        # 使用 module_info 的方法获取模块的输入，包括设备、数据类型、是否需要梯度等信息
        module_inputs = module_info.module_inputs_func(
            module_info,
            device=device,
            dtype=dtype,
            requires_grad=True,
            training=training,
            with_packed_sequence=True,
        )
        # 遍历模块的每一个输入
        for module_input in module_inputs:
            # 如果当前模块输入的 forward_input 为空，则跳过当前循环
            if module_input.forward_input is None:
                continue
            # 获取构造模块的参数和关键字参数
            args, kwargs = (
                module_input.constructor_input.args,
                module_input.constructor_input.kwargs,
            )
            # 使用 RNNWrapper 包装模块类 module_cls，并传入构造参数和关键字参数
            m = RNNWrapper(module_cls, args, kwargs)
            # 获取包装模块的 batch_first 属性，并设置模块在指定设备和数据类型上运行
            batch_first = m.m.batch_first
            m.to(device).to(dtype)

            # 获取前向传播的参数和关键字参数
            args, kwargs = (
                module_input.forward_input.args,
                module_input.forward_input.kwargs,
            )

            # 如果输入 input 是二维的 torch.Tensor，则进行批处理处理
            input = args[0]
            if isinstance(input, torch.Tensor) and input.dim() == 2:
                input = input.detach()
                # 创建一个新的输入形状列表，用于批处理输入 input
                new_input_shape = [1] * (len(input.shape) + 1)
                if batch_first:
                    new_input_shape[0] = 2
                    input = input.repeat(new_input_shape)
                else:
                    new_input_shape[1] = 2
                    input = input.unsqueeze(1).repeat(new_input_shape)

                # 如果存在隐藏状态 h，则进行批处理处理
                h = args[1] if len(args) > 1 else None
                if h is not None:
                    h = (
                        batch_hidden(h)
                        if isinstance(h, torch.Tensor)
                        else tuple(batch_hidden(hx) for hx in h)
                    )
                    args = list(args)
                    args[1] = h

            # 如果输入 input 是 torch.nn.utils.rnn.PackedSequence 类型，则调用 _do_test_rnn_packed_sequence 方法
            if isinstance(input, torch.nn.utils.rnn.PackedSequence):
                self._do_test_rnn_packed_sequence(
                    m, input, args[1:], kwargs, atol=atol, rtol=rtol
                )
            # 否则调用 _do_test 方法进行测试
            else:
                self._do_test(
                    m,
                    input,
                    args[1:],
                    kwargs,
                    batch_first=batch_first,
                    atol=atol,
                    rtol=rtol,
                )
    # 定义测试函数，用于验证特定条件下的API行为是否失败
    def test_per_sample_api_failing(self):
        # 创建一个包含10个输入和10个输出的线性模块
        module = nn.Linear(10, 10)
        # 生成一个形状为(64, 10)的随机输入张量
        input = torch.randn(64, 10)
        # 断言调用call_for_per_sample_grads("fail")(input)时会抛出RuntimeError，并且错误消息包含"Module passed must be nn.Module"
        with self.assertRaisesRegex(RuntimeError, r"Module passed must be nn.Module"):
            call_for_per_sample_grads("fail")(input)
        # 断言调用call_for_per_sample_grads(module, batch_size=6.4)(input)时会抛出RuntimeError，并且错误消息包含"Batch size passed must be None or an integer"
        with self.assertRaisesRegex(
            RuntimeError, r"Batch size passed must be None or an integer"
        ):
            call_for_per_sample_grads(module, batch_size=6.4)(input)
        # 断言调用call_for_per_sample_grads(module, batch_size=-64)(input)时会抛出RuntimeError，并且错误消息包含"Batch size must be positive"
        with self.assertRaisesRegex(RuntimeError, r"Batch size must be positive"):
            call_for_per_sample_grads(module, batch_size=-64)(input)
        # 断言多次调用call_for_per_sample_grads(module)(input)后会抛出RuntimeError，并且错误消息包含"incorrect for multiple calls"
        with self.assertRaisesRegex(RuntimeError, r"incorrect for multiple calls"):
            # 计算损失并反向传播以填充grad_sample字段
            loss = call_for_per_sample_grads(module)(input).sum()
            loss.backward()
            # 再次调用call_for_per_sample_grads(module)(input)，预期会抛出错误
            call_for_per_sample_grads(module)(input)

        # 重新创建一个包含10个输入和10个输出的线性模块，以重置grad_sample字段
        module = nn.Linear(10, 10)
        # 断言调用call_for_per_sample_grads(module, loss_reduction="")(input)时会抛出RuntimeError，并且错误消息包含"Expected loss_reduction argument to be sum or mean"
        with self.assertRaisesRegex(
            RuntimeError, r"Expected loss_reduction argument to be sum or mean"
        ):
            call_for_per_sample_grads(module, loss_reduction="")(input)

    # 定义测试函数，用于验证计算批量大小时的API行为
    def test_per_sample_api_compute_batch_size(self):
        # 定义一个自定义模块，包含一个线性层
        class CustomModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)

            def forward(self, input1, input2):
                # 返回两个输入经过线性层处理后的结果之和
                return self.linear(input1) + self.linear(input2)

        # 创建自定义模块的实例
        module = CustomModule()
        # 生成形状为(4, 5)和(5, 5)的随机输入张量
        input1 = torch.randn(4, 5)
        input2 = torch.randn(5, 5)

        # 断言调用call_for_per_sample_grads(module)(input1, input2)时会抛出RuntimeError，并且错误消息包含"found at least one input with batch size 4 and one with batch size 5"
        with self.assertRaisesRegex(
            RuntimeError,
            "found at least one input with batch size 4 and one with batch size 5",
        ):
            call_for_per_sample_grads(module)(input1, input2)

        # 修改input2的形状为(4, 5)，使两个输入的批量大小相同
        input2 = torch.randn(4, 5)
        # 调用call_for_per_sample_grads(module)(input1, input2)，预期不会抛出错误
        call_for_per_sample_grads(module)(input1, input2)

        # 创建自定义模块的新实例
        module = CustomModule()
        # 调用call_for_per_sample_grads(module)(input1, input2=input2)，其中指定了一个输入参数的关键字参数
        call_for_per_sample_grads(module)(input1, input2=input2)

        # 创建自定义模块的新实例
        module = CustomModule()
        # 调用call_for_per_sample_grads(module)(input1=input1, input2=input2)，其中指定了两个输入参数的关键字参数
        call_for_per_sample_grads(module)(input1=input1, input2=input2)
    # 定义一个测试方法，用于测试对于不可 pytree 化对象的计算批量大小的 API
    def test_per_sample_api_compute_batch_size_not_pytreeable(self):
        # 定义一个数据类 NonPytreeableTuple，包含两个 torch.Tensor 类型的成员
        @dataclass
        class NonPytreeableTuple:
            elem1: torch.Tensor
            elem2: torch.Tensor
        
        # 定义一个自定义的 PyTorch 模块 CustomModule
        class CustomModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)

            # 定义模块的前向传播方法
            def forward(self, input1, input2):
                # 计算线性层的输出并返回
                return self.linear(input1.elem1) + self.linear(input1.elem2)

        # 创建一个 NonPytreeableTuple 类型的输入对象 input
        input = NonPytreeableTuple(torch.randn(4, 5), torch.randn(4, 5))
        # 创建一个 CustomModule 实例 model
        model = CustomModule()

        # 使用断言验证在调用 call_for_per_sample_grads 函数时抛出 RuntimeError 异常，
        # 异常信息包含 "ExpandedWeights cannot compute the batch size from the inputs"
        with self.assertRaisesRegex(
            RuntimeError,
            "ExpandedWeights cannot compute the batch size from the inputs",
        ):
            call_for_per_sample_grads(model)(input, "")

        # 使用断言验证在调用 call_for_per_sample_grads 函数时抛出 RuntimeError 异常，
        # 异常信息包含 "Expected ExpandedWeights to have batch size matching input"
        with self.assertRaisesRegex(
            RuntimeError, "Expected ExpandedWeights to have batch size matching input"
        ):
            call_for_per_sample_grads(model)(input, torch.randn(5))

        # 创建一个新的 CustomModule 实例 model，并调用 call_for_per_sample_grads 函数
        # 用于计算梯度，但存在功能调用错误，待 Sam 修复
        model = CustomModule()  # TODO: functional call bug, sam will fix
        call_for_per_sample_grads(model)(input, torch.randn(4, 5))

        # 创建一个新的 CustomModule 实例 model，并设置 batch_size=4，然后调用
        # call_for_per_sample_grads 函数进行梯度计算
        model = CustomModule()
        call_for_per_sample_grads(model, batch_size=4)(input, torch.randn(5))
class ContextManagerTests(TestBase):
    # 定义 ContextManagerTests 类，继承自 TestBase 类

    def __init__(self, *args, **kwargs):
        # 初始化方法，接受任意位置参数和关键字参数
        self.test_cpu = kwargs.get("test_cpu", True)
        self.test_cuda = kwargs.get("test_cuda", True)
        super().__init__(*args, **kwargs)
        # 设置 test_cpu 和 test_cuda 属性，并调用父类的初始化方法

    @property
    def constructor_args(self):
        # constructor_args 属性的 getter 方法，返回 _get_arg("constructor_args", False) 的结果
        return self._get_arg("constructor_args", False)

    def test_context_manager(self, test_case, device):
        # 测试上下文管理器的方法，接受 test_case 和 device 参数
        kwargs = {"device": device, "dtype": torch.double}
        # 初始化 kwargs 字典，设定 device 和 dtype=torch.double
        module = self.constructor(*self.constructor_args).to(**kwargs)
        # 根据 constructor 和 constructor_args 创建模块对象，并将其移到指定的设备上
        if "Embedding" in self.get_name():
            kwargs["dtype"] = torch.long
        # 如果模块名称包含 "Embedding"，则将 dtype 设置为 torch.long
        input = self._get_input().to(**kwargs)
        # 获取输入数据，并将其移到指定的设备上
        if len(input.shape) == 0 or input.shape[0] == 0:
            raise unittest.SkipTest(
                "Can't get per sample gradients when no batch dim or batch dim is 0"
            )
        # 如果输入数据的形状长度为 0 或者第一个维度为 0，则跳过测试
        if self.constructor == torch.nn.Linear and len(input.shape) == 1:
            raise unittest.SkipTest(
                "Can't get per sample gradients for input of rank 1"
            )
        # 如果构造器是 torch.nn.Linear，并且输入数据的形状长度为 1，则跳过测试
        test_case._do_test(module, input)
        # 执行测试，调用 _do_test 方法

    def test_context_manager_multiple_inputs(self, test_case, device):
        # 测试上下文管理器，支持多个输入的方法，接受 test_case 和 device 参数
        module = self.constructor(*self.constructor_args).to(device)
        # 根据 constructor 和 constructor_args 创建模块对象，并将其移到指定的设备上
        input = self._get_input()
        # 获取输入数据
        if len(input.shape) == 0 or input.shape[0] == 0:
            raise unittest.SkipTest(
                "Can't get per sample gradients when no batch dim or batch dim is 0"
            )
        # 如果输入数据的形状长度为 0 或者第一个维度为 0，则跳过测试
        if self.constructor == torch.nn.Linear and len(input.shape) == 1:
            raise unittest.SkipTest(
                "Can't get per sample gradients for input of rank 1"
            )
        # 如果构造器是 torch.nn.Linear，并且输入数据的形状长度为 1，则跳过测试
        test_case._do_test_multi_input(module, input)
        # 执行支持多输入的测试，调用 _do_test_multi_input 方法


def filter_supported_tests(t):
    # 过滤出支持的测试用例的函数，接受 t 参数
    supported_modules = [
        "Linear",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "Embedding",
        "LayerNorm",
        "GroupNorm",
        "InstanceNorm",
    ]
    # 定义支持的模块列表
    if "module_name" in t and t["module_name"] in supported_modules:
        return True
    # 如果测试参数包含 "module_name" 属性，并且其值在支持的模块列表中，则返回 True


# TODO: Once all of these use ModuleInfo, replace with ModuleInfo tests
# These currently use the legacy nn tests
supported_tests = [
    t for t in module_tests + new_module_tests if filter_supported_tests(t)
]
# 根据 filter_supported_tests 函数筛选出支持的测试用例，并存储在 supported_tests 列表中
for test_param in supported_tests:
    # 遍历支持的测试用例列表
    if "constructor" not in test_param:
        name = test_param.pop("module_name")
        test_param["constructor"] = getattr(nn, name)
    # 如果测试参数中没有 "constructor" 属性，则从 nn 模块动态获取相应名称的构造器，并添加到测试参数中
    decorator = test_param.pop("decorator", lambda test: test)
    # 弹出并获取 decorator 属性，如果不存在则使用默认的 lambda 函数
    test = ContextManagerTests(**test_param)
    # 创建 ContextManagerTests 类的实例，使用测试参数初始化
    test_name = test.get_name()
    # 获取测试的名称
    if hasattr(TestExpandedWeightModule, test_name):
        raise RuntimeError("Found two tests with the same name: " + test_name)
    # 如果已经存在同名的测试，则抛出运行时错误
    test_name_multi_input = test.get_name() + "_multiple_inputs"
    if hasattr(TestExpandedWeightModule, test_name_multi_input):
        raise RuntimeError("Found two tests with the same name: " + test_name)
    # 如果已经存在同名的支持多输入的测试，则抛出运行时错误
    # 如果 test.test_cpu 为真，则为 TestExpandedWeightModule 动态设置单输入测试函数
    setattr(
        TestExpandedWeightModule,
        test_name,
        # 使用装饰器为单输入测试函数添加上下文管理器测试
        decorator(lambda self, test=test: test.test_context_manager(self, "cpu")),
    )
    # 如果 TEST_CUDA 为真且 test.test_cuda 为真，则为 TestExpandedWeightModule 动态设置双输入测试函数
    setattr(
        TestExpandedWeightModule,
        test_name_multi_input,
        # 使用装饰器为双输入测试函数添加上下文管理器测试
        decorator(
            lambda self, test=test: test.test_context_manager_multiple_inputs(
                self, "cpu"
            )
        ),
    )
    # 如果 TEST_CUDA 为真且 test.test_cuda 为真，则为 TestExpandedWeightModule 动态设置 CUDA 双精度测试函数
    # 由于这里检查导数，仅使用双精度以保证精度
    setattr(
        TestExpandedWeightModule,
        test_name + "_cuda_double",
        # 使用装饰器为 CUDA 双精度测试函数添加上下文管理器测试
        decorator(lambda self, test=test: test.test_context_manager(self, "cuda")),
    )
# ------------- HELPER FUNCTIONS -----------------

# 根据 op 的名称和输入，运行相应的操作
def run_op(op, input, *args, **kwargs):
    """
    OpInfo for Embedding switches the input and weight so autograd tests will only check the derivative
    of the weight, not the input, which can't be differentiable since its dtype is int. Calls op,
    using the special ordering that Embedding's OpInfo expects for that case.
    """
    if op.name == "nn.functional.embedding":
        # 对于嵌入操作，调用 op 并交换输入和权重的顺序，以便只检查权重的导数
        return op(args[0], input, **kwargs)
    else:
        # 对于其他操作，按正常顺序调用 op
        return op(input, *args, **kwargs)


# 创建扩展权重的辅助函数，基于样本输入、批处理大小和损失缩减方式
def make_expanded_weight(sample_input, batch_size, loss_reduction="sum"):
    def expanded_weight_or_clone(arg):
        # 如果是一个不同的张量，返回 ExpandedWeight 的克隆
        if is_diff_tensor(arg):
            return ExpandedWeight(torch.clone(arg), batch_size, loss_reduction)
        # 否则返回输入的克隆
        return clone_if_tensor(arg)

    # 克隆输入和参数，构建扩展的输入、参数和关键字参数
    ew_input = clone_if_tensor(sample_input.input)
    ew_args = tuple(expanded_weight_or_clone(arg) for arg in sample_input.args)
    ew_kwargs = {
        name: expanded_weight_or_clone(arg)
        for (name, arg) in sample_input.kwargs.items()
    }
    return ew_input, ew_args, ew_kwargs


# 过滤不支持的输入用例，返回支持的样本输入列表
def supported_inputs(op, sample_inputs, supported_inputs=True):
    """
    ExpandedWeights currently does not support some use cases when there's no batch dimension or
    operations that would cause inter-batch operations. Removes all of the cases it cannot deal with
    """

    def filter_fn(input):
        convolutions = [
            "nn.functional.conv1d",
            "nn.functional.conv2d",
            "nn.functional.conv3d",
        ]
        batched_input_size = dict(zip(convolutions, [3, 4, 5]))
        if op.name == "nn.functional.linear":
            # 线性操作要求输入的维度大于 1，否则没有批处理维度
            is_supported_input = (
                input.input.dim() > 1
            )
        elif op.name == "nn.functional.layer_norm":
            # 标准化操作要求输入形状与标准化形状不同，否则会导致批次间操作
            normalized_shape = input.args[0]
            is_supported_input = (
                input.input.shape != normalized_shape
            )
        elif op.name in convolutions:
            # 卷积操作要求输入的维度与指定的批处理输入维度一致
            is_supported_input = input.input.dim() == batched_input_size[op.name]
        elif op.name == "nn.functional.embedding":
            # 嵌入操作要求索引的维度大于 1，否则没有批处理大小
            idx = input.args[0]
            is_supported_input = len(idx.shape) > 1
        else:
            # 其他操作默认支持
            is_supported_input = True
        # 输入的第一个维度必须大于 0
        is_supported_input = (
            is_supported_input and input.input.shape[0] > 0
        )
        return is_supported_input if supported_inputs else not is_supported_input

    # 返回支持的样本输入列表
    return [input for input in sample_inputs if filter_fn(input)]


# 对每个样本的梯度进行循环，计算每个输入的导数
def for_loop_per_sample_grad(batch_size, reduction, input, func, *args, **kwargs):
    # get per sample grads by getting derivative for each input in a for loop
    # 通过循环获取每个输入的样本梯度
    per_sample_grad = []
    # 对于每个批次中的样本，依次进行处理
    for i in range(batch_size):
        # 获取当前样本的输入
        per_sample_input = input[i]
        # 对当前样本输入进行函数调用，并进行降维操作
        result = reduction(func(per_sample_input.unsqueeze(0), *args, **kwargs))
        # 组合输入参数列表，包括样本输入、额外参数和关键字参数的值
        diff_input_list = (per_sample_input,) + tuple(args) + tuple(kwargs.values())
        # 从参数列表中筛选出需要进行梯度计算的张量（Tensor）
        diff_input_list = [
            i
            for i in diff_input_list
            if isinstance(i, torch.Tensor) and i.requires_grad
        ]
        # 计算结果关于所选张量列表的梯度，使用全为1的张量作为梯度传播起点，允许部分张量未使用
        per_sample_grad.append(
            torch.autograd.grad(
                result, diff_input_list, torch.ones_like(result), allow_unused=True
            )
        )
    
    # 如果每个样本都成功计算了梯度
    if len(per_sample_grad) == batch_size:
        # 将每个样本的梯度堆叠成元组形式返回
        per_sample_grad = tuple(torch.stack(grad) for grad in zip(*per_sample_grad))
    
    # 返回每个样本的梯度
    return per_sample_grad
# 判断给定对象是否为 ExpandedWeight 类型或者是一个 PyTorch 张量且需要梯度计算
def is_diff_tensor(t):
    return isinstance(t, ExpandedWeight) or (
        isinstance(t, torch.Tensor) and t.requires_grad
    )

# 如果输入对象是 PyTorch 张量，则克隆该张量并将其从计算图中分离出来，保留其梯度属性
# 否则直接返回输入对象
def clone_if_tensor(t):
    if isinstance(t, torch.Tensor):
        # 使用 torch.clone 复制张量，并使用 detach 方法分离计算图
        res = torch.clone(t).detach()
        # 设置新张量的 requires_grad 属性与原张量一致
        res.requires_grad_(t.requires_grad)
        return res
    else:
        return t

# 使用全局变量 globals() 中的类 TestExpandedWeightHelperFunction 实例化设备类型测试
instantiate_device_type_tests(TestExpandedWeightHelperFunction, globals())

# 使用全局变量 globals() 中的类 TestExpandedWeightFunctional 实例化设备类型测试
instantiate_device_type_tests(TestExpandedWeightFunctional, globals())

# 使用全局变量 globals() 中的类 TestExpandedWeightModule 实例化设备类型测试
instantiate_device_type_tests(TestExpandedWeightModule, globals())

# 如果当前脚本作为主程序执行，则运行测试函数
if __name__ == "__main__":
    run_tests()
```