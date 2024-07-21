# `.\pytorch\test\test_optim.py`

```
# Owner(s): ["module: optimizer"]
import functools  # 导入 functools 模块，用于高阶函数和函数式编程支持
import math  # 导入 math 模块，提供数学运算函数
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
import unittest  # 导入 unittest 模块，用于编写和运行单元测试
from copy import deepcopy  # 导入 deepcopy 函数，用于深拷贝对象
from typing import Any, Dict, Tuple  # 导入类型提示相关的模块

from unittest.mock import patch  # 导入 patch 函数，用于模拟对象和函数调用

from optim.test_lrscheduler import TestLRScheduler  # noqa: F401
from optim.test_optim import TestDifferentiableOptimizer  # noqa: F401
from optim.test_swa_utils import TestSWAUtils  # noqa: F401

import torch  # 导入 PyTorch 深度学习库
from torch.nn import Parameter  # 导入 Parameter 类，用于定义可训练参数
from torch.optim import Optimizer, SGD  # 导入优化器 Optimizer 和 SGD 优化器

from torch.optim.lr_scheduler import ReduceLROnPlateau  # 导入学习率调度器 ReduceLROnPlateau
from torch.optim.optimizer import (  # 导入优化器模块的函数和类
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU  # 导入测试 GPU 相关的模块
from torch.testing._internal.common_device_type import (  # 导入设备类型相关的测试模块和函数
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
    skipMPS,
    TEST_WITH_ROCM,
)
from torch.testing._internal.common_dtype import floating_types_and  # 导入浮点数类型相关的测试模块
from torch.testing._internal.common_optimizers import (  # 导入优化器相关的测试模块和函数
    _get_device_type,
    _get_optim_inputs_including_global_cliquey_kwargs,
    optim_db,
    OptimizerErrorEnum,
    optims,
    TensorTracker,
)
from torch.testing._internal.common_utils import (  # 导入通用测试工具函数和类
    markDynamoStrictTest,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)

FP16_REDUCED_PRECISION = {"atol": 1e-5, "rtol": 1e-4}  # 设置 FP16 精度参数


def rosenbrock(tensor):
    assert tensor.size() == torch.Size(
        [2]
    ), f"Requires tensor with 2 scalars but got {tensor.size()}"  # 断言 tensor 的大小为 2
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2  # 返回 Rosenbrock 函数的值


def drosenbrock(tensor):
    assert tensor.size() == torch.Size(
        [2]
    ), f"Requires tensor with 2 scalars but got {tensor.size()}"  # 断言 tensor 的大小为 2
    x, y = tensor
    return torch.stack((-400 * x * (y - x**2) - 2 * (1 - x), 200 * (y - x**2)))  # 返回 Rosenbrock 函数的梯度向量


@markDynamoStrictTest
class TestOptimRenewed(TestCase):
    """
    This test class validates the core optimizers and is structured as the correctness of:
    - The update algorithms (forloop implementation)
        * Every optimizer's algorithm is most readably implemented through a big for-loop
          over all the parameters, which is what we refer to as the forloop or single tensor
          implementation. These algorithms are manually validated by comparing to the paper
          and systematically validated by assuring that the loss goes the right direction
          when the optimizer has been applied.
        * This implementation should compose with optimizer hyperparameters well, such as
          supporting Tensor LRs, the capturable API, and sparse and complex parameters.
    """
    pass  # 测试类，验证核心优化器实现的正确性，但未定义具体的测试方法
    # 使用装饰器声明只在CPU上运行的测试函数，并传入优化器数据库作为参数
    @onlyCPU
    @optims(optim_db)
    # 定义测试函数，验证优化器信息不应指定全局的特定参数
    def test_optim_infos_do_not_specify_global_cliquey_kwargs(
        self, device, dtype, optim_info
    ):
        # 定义全局特定参数的列表
        global_cliquey_flags = ["foreach", "fused", "differentiable"]
        # 遍历优化器信息中的优化输入函数返回的每个优化器输入
        for optim_input in optim_info.optim_inputs_func(device=device):
            # 断言：优化器输入中不应包含任何全局特定参数
            self.assertFalse(
                any(f for f in global_cliquey_flags if f in optim_input.kwargs)
            )

    # 使用装饰器从优化器数据库中选择具有优化器错误输入函数的优化器
    @optims([optim for optim in optim_db if optim.optim_error_inputs_func is not None])
    # 定义一个测试方法，用于测试不同的优化器错误情况
    def test_errors(self, device, dtype, optim_info):
        # 从参数 optim_info 中获取优化器类
        optim_cls = optim_info.optim_cls
        # 调用 optim_info 中的函数生成优化器错误输入数据
        error_inputs = optim_info.optim_error_inputs_func(device=device, dtype=dtype)

        # 遍历每个优化器错误输入数据
        for error_input in error_inputs:
            # 从 error_input 中获取优化器的参数和关键字参数
            optim_input = error_input.optimizer_error_input
            params, kwargs = optim_input.params, optim_input.kwargs

            # 根据 error_on 属性确定当前测试的错误类型
            if error_input.error_on == OptimizerErrorEnum.CONSTRUCTION_ERROR:
                # 如果错误类型是构造错误，并且错误类型是 Warning 的子类
                if issubclass(error_input.error_type, Warning):
                    # 使用 assertWarnsRegex 验证是否会发出指定类型的警告信息
                    with self.assertWarnsRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        # 创建优化器实例以触发警告
                        optim_cls(params, **kwargs)
                else:
                    # 如果错误类型不是 Warning 的子类，使用 assertRaisesRegex 验证是否会抛出指定类型的异常
                    with self.assertRaisesRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        # 创建优化器实例以触发异常
                        optim_cls(params, **kwargs)
            elif error_input.error_on == OptimizerErrorEnum.STEP_ERROR:
                # 如果错误类型是步骤错误，创建优化器实例
                optim = optim_cls(params, **kwargs)
                # 如果错误类型是 Warning 的子类，验证是否会发出指定类型的警告信息
                if issubclass(error_input.error_type, Warning):
                    with self.assertWarnsRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        # 调用优化器的 step 方法以触发警告
                        optim.step()
                else:
                    # 如果错误类型不是 Warning 的子类，验证是否会抛出指定类型的异常
                    with self.assertRaisesRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        # 调用优化器的 step 方法以触发异常
                        optim.step()
            else:
                # 如果出现未知的错误类型，抛出 NotImplementedError 异常
                raise NotImplementedError(f"Unknown error type {error_input.error_on}")
    ):
        # 获取优化器类
        optim_cls = optim_info.optim_cls
        # 根据是否需要学习率调度器，确定调度器构造函数的列表
        schedulers_constructors = (
            optim_info.scheduler_inputs if with_lrsched else [None]
        )

        # 遍历每个调度器构造函数
        for schedulers_constructor in schedulers_constructors:
            # 对于每个优化器输入，根据设备生成优化器的输入
            optim_inputs = optim_info.optim_inputs_func(device=device)
            for optim_input in optim_inputs:
                # 如果支持 foreach 实现，则将其设为 False，强制使用 for 循环
                if "foreach" in optim_info.supported_impls:
                    optim_input.kwargs["foreach"] = False  # force forloop
                
                # 根据是否连续，初始化权重和偏置参数
                if contiguous:
                    weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
                    bias = Parameter(torch.randn((10), device=device, dtype=dtype))
                else:
                    weight = Parameter(
                        torch.randn((10, 5, 2), device=device, dtype=dtype)[..., 0]
                    )
                    bias = Parameter(
                        torch.randn((10, 2), device=device, dtype=dtype)[..., 0]
                    )
                
                # 随机生成输入数据
                input = torch.randn(5, device=device, dtype=dtype)

                # 使用优化器类和参数初始化优化器
                optimizer = optim_cls([weight, bias], **optim_input.kwargs)
                
                # 根据调度器构造函数列表创建调度器对象列表
                schedulers = [
                    s(optimizer)
                    for s in (schedulers_constructor if schedulers_constructor else [])
                ]

                # 定义闭包函数，计算损失并执行反向传播
                def closure():
                    optimizer.zero_grad()
                    loss = (weight.mv(input) + bias).pow(2).sum()
                    loss.backward()
                    
                    # 如果仅支持稀疏梯度，则将梯度转换为稀疏格式
                    if optim_info.only_supports_sparse_grads:
                        # For this test, we naively convert the Tensor layout, which we know does
                        # NOT represent the expected use case for optims like SparseAdam!
                        weight.grad = weight.grad.to_sparse()
                        bias.grad = bias.grad.to_sparse()
                    
                    return loss

                # 计算闭包函数的初始损失值
                initial_value = closure().item()
                
                # 多次迭代优化过程
                for _ in range(20):
                    # 如果步骤需要闭包，则使用 optimizer.step(closure) 更新参数
                    if optim_info.step_requires_closure:
                        loss = optimizer.step(closure)
                    else:
                        # 否则直接执行闭包函数
                        loss = closure()
                        optimizer.step()

                    # 对每个调度器执行调度步骤
                    for scheduler in schedulers:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(loss)
                        else:
                            scheduler.step()

                # 如果优化参数包含 maximize=True，则断言损失增大
                if optim_input.kwargs.get("maximize", False):
                    self.assertGreater(closure().item(), initial_value)
                else:
                    # 否则断言损失减小
                    self.assertLess(closure().item(), initial_value)

    # 以下为装饰器，指定仅在 CUDA 环境下执行，并根据多 GPU 检测结果进行跳过
    @onlyCUDA
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # 参数化装饰器，指定 with_lrsched 参数为 True 和 False 时的测试
    @parametrize("with_lrsched", [True, False])
    # 优化器参数化装饰器，指定优化器信息和数据类型
    @optims(optim_db, dtypes=[torch.float32])
    # 定义测试方法，用于测试多GPU环境下的优化器和学习率调度器行为是否正确
    def test_forloop_goes_right_direction_multigpu(
        self, device, dtype, optim_info, with_lrsched
    ):
        # 获取优化器类
        optim_cls = optim_info.optim_cls
        # 确定要使用的调度器构造函数列表
        schedulers_constructors = (
            optim_info.scheduler_inputs if with_lrsched else [None]
        )
        # 遍历调度器构造函数列表
        for schedulers_constructor in schedulers_constructors:
            # 如果支持张量学习率，则每次迭代都需要使用新的输入以避免跨迭代的突变
            optim_inputs = optim_info.optim_inputs_func(device=device)
            # 遍历优化器输入参数
            for optim_input in optim_inputs:
                # 如果支持 foreach 实现，则强制禁用 foreach 参数
                if "foreach" in optim_info.supported_impls:
                    optim_input.kwargs["foreach"] = False  # force forloop

                # 创建模拟的张量参数
                weight = Parameter(torch.randn((10, 5), device="cuda:0", dtype=dtype))
                bias = Parameter(torch.randn((10), device="cuda:1", dtype=dtype))
                inpt = torch.randn(5, device="cuda:0", dtype=dtype)

                # 使用优化器类和输入参数初始化优化器
                optimizer = optim_cls([weight, bias], **optim_input.kwargs)
                
                # 根据调度器构造函数列表创建调度器列表
                schedulers = [
                    s(optimizer)
                    for s in (schedulers_constructor if schedulers_constructor else [])
                ]

                # 定义闭包函数，用于执行优化器步骤并返回损失值
                def closure():
                    optimizer.zero_grad()
                    loss = (weight.mv(inpt).cuda(1) + bias).pow(2).sum()
                    loss.backward()
                    # 如果只支持稀疏梯度，则在这个测试中将梯度转换为稀疏格式
                    if optim_info.only_supports_sparse_grads:
                        # 这里是一个简单的转换，实际上不符合像 SparseAdam 这样优化器的预期使用场景！
                        weight.grad = weight.grad.to_sparse()
                        bias.grad = bias.grad.to_sparse()
                    return loss

                # 记录初始闭包函数返回的损失值
                initial_value = closure().item()
                
                # 执行20次优化步骤
                for _ in range(20):
                    loss = optimizer.step(closure)
                    # 对每个调度器进行步骤调度
                    for scheduler in schedulers:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(loss)
                        else:
                            scheduler.step()

                # 如果优化器输入参数中包含 maximize=True，则验证当前损失值比初始值大
                if optim_input.kwargs.get("maximize", False):
                    self.assertGreater(closure().item(), initial_value)
                # 否则验证当前损失值比初始值小
                else:
                    self.assertLess(closure().item(), initial_value)
    ):
        # 获取优化器类
        optim_cls = optim_info.optim_cls

        # 遍历优化器信息中的调度器输入
        for schedulers_c in optim_info.scheduler_inputs:
            # 创建权重和偏置参数，并指定设备和数据类型
            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            bias = Parameter(torch.randn((10), device=device, dtype=dtype))
            inpt = torch.randn(5, device=device, dtype=dtype)

            # 如果正在编译，通过在张量中包装LR以避免无限重新编译
            lr = torch.tensor(0.01) if torch._utils.is_compiling() else 0.01
            # 使用优化器类初始化优化器，传入参数和学习率（如果提供）
            optimizer = optim_cls([{"params": [weight]}, {"params": [bias], "lr": lr}])
            # 根据每个调度器函数创建调度器列表
            schedulers = [scheduler_c(optimizer) for scheduler_c in schedulers_c]

            # 定义闭包函数，用于执行优化器的一次迭代
            def closure():
                optimizer.zero_grad()
                # 计算损失值并执行反向传播
                loss = (weight.mv(inpt) + bias).pow(2).sum()
                loss.backward()
                # 如果只支持稀疏梯度，则将梯度转换为稀疏格式（此处是测试目的，并非 SparseAdam 的典型用法）
                if optim_info.only_supports_sparse_grads:
                    weight.grad = weight.grad.to_sparse()
                    bias.grad = bias.grad.to_sparse()
                return loss

            # 记录初始损失值
            initial_value = closure().item()
            # 进行多次优化迭代
            for _ in range(20):
                loss = optimizer.step(closure)
                # 对每个调度器执行一步调度
                for scheduler in schedulers:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(loss)
                    else:
                        scheduler.step()

            # 断言最终的损失值比初始损失值小
            self.assertLess(closure().item(), initial_value)

    @optims(optim_db, dtypes=[torch.float32])
    def test_tensor_lr(self, device, dtype, optim_info):
        # 从 optim_info 中获取优化器类
        optim_cls = optim_info.optim_cls

        # 获取所有优化器输入，包括全局关键参数，但跳过可微测试
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )

        # 遍历所有优化器输入
        for optim_input in all_optim_inputs:
            # 创建一个可学习的权重张量，并克隆并分离以进行梯度跟踪
            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            weight_c = weight.clone().detach().requires_grad_(True)
            bias = Parameter(torch.randn((10), device=device, dtype=dtype))
            bias_c = bias.clone().detach().requires_grad_(True)
            inpt = torch.randn(5, device=device, dtype=dtype)

            # 获取优化器输入的关键字参数
            kwargs = optim_input.kwargs

            # 如果关键字参数中包含 "lr"，则删除它
            if "lr" in kwargs:
                del kwargs["lr"]

            # 根据优化信息确定学习率，并将其添加到关键字参数中
            kwargs["lr"] = 1.0 if optim_info.step_requires_closure else 1e-3

            # 使用优化器类创建优化器对象，传入权重和偏置参数及其余的关键字参数
            optimizer_r = optim_cls([weight, bias], **kwargs)

            # 尝试使用 Tensor 类型的学习率创建优化器对象，如果失败则捕获 ValueError 异常
            try:
                kwargs["lr"] = torch.tensor(kwargs["lr"])
                optimizer = optim_cls([weight_c, bias_c], **kwargs)
            except ValueError as e:
                # 断言捕获的异常消息中包含指定的字符串
                self.assertRegex(str(e), ".*lr as a Tensor is not supported.*")
                continue

            # 定义闭包函数用于优化器步骤
            def closure(optim, w, b, i):
                optim.zero_grad()
                loss = (w.mv(i) + b).pow(2).sum()
                loss.backward()
                if optim_info.only_supports_sparse_grads:
                    # 如果仅支持稀疏梯度，则将梯度转换为稀疏表示
                    w.grad = w.grad.to_sparse()
                    b.grad = b.grad.to_sparse()
                return loss

            # 执行优化器步骤多次
            for _ in range(5):
                if optim_info.step_requires_closure:
                    # 如果需要闭包方式进行优化步骤，则使用 functools.partial 包装闭包函数
                    optimizer_r.step(
                        functools.partial(closure, optimizer_r, weight, bias, inpt)
                    )
                    optimizer.step(
                        functools.partial(closure, optimizer, weight_c, bias_c, inpt)
                    )
                else:
                    # 否则直接调用闭包函数进行优化步骤
                    closure(optimizer_r, weight, bias, inpt)
                    closure(optimizer, weight_c, bias_c, inpt)

                # 断言权重和偏置参数保持不变
                self.assertEqual(weight, weight_c)
                self.assertEqual(bias, bias_c)
    ):
        # 定义循环迭代次数为7，因为在第7次迭代开始，我们可以看到RAdam参数与较小eps值交互时的差异，
        # 因为在第6步之后rho_t大于5。
        if assert_eq_kwargs is None:
            assert_eq_kwargs = {}
        # 设置迭代次数为7
        kIterations = 7
        # 创建一个TensorTracker对象，用于跟踪断言参数
        tracker = TensorTracker(assert_eq_kwargs)
        # 循环执行kIterations次
        for i in range(kIterations):
            state, updated_params = [], []
            # 如果inputs不是列表，则将其转换为包含两个相同元素的列表
            if not isinstance(inputs, list):
                inputs = [inputs, inputs]
            # 遍历inputs、models和optimizers的三元组
            for input, model, optimizer in zip(inputs, models, optimizers):
                # 重置优化器的梯度
                optimizer.zero_grad()

                # 在第3次迭代时，冻结一个层，以测试其在'fused'或'foreach'模式下的步骤是否与'forloop'中的步骤相同
                if i == 3:
                    model[2].requires_grad_(False)
                # 在第5次迭代后解冻该层
                if i == 5:
                    model[2].requires_grad_(True)

                # 当i不等于2时，测试步骤是否按预期执行（即梯度为None时不进行任何操作）
                if i != 2:
                    output = model(input)
                    loss = output.sum()
                    loss.backward()

                # 执行优化步骤
                optimizer.step()
                # 将当前优化器状态添加到state列表中
                state.append(optimizer.state)
                # 将更新后的模型参数添加到updated_params列表中
                updated_params.append(model.parameters())

            # 将原始状态和新状态分别保存为og_state和new_state
            og_state, new_state = state
            # 遍历更新后的模型参数列表
            for og_p, new_p in zip(updated_params[0], updated_params[1]):
                # 将原始模型参数添加到tracker中
                tracker.add(og_p)
                # 检查并设置新的模型参数，并在tracker中进行检查
                tracker.pop_check_set(new_p, self)

                # 检查优化器状态是否相同
                og_p_state = og_state[og_p]
                new_p_state = new_state[new_p]
                # 如果assert_step_dtype不为None，则进行数据类型断言
                if assert_step_dtype is not None:
                    if torch.is_tensor(og_p_state.get("step", None)):
                        self.assertEqual(og_p_state["step"].dtype, assert_step_dtype)
                    if torch.is_tensor(new_p_state.get("step", None)):
                        self.assertEqual(new_p_state["step"].dtype, assert_step_dtype)
                # 遍历原始参数状态的键
                for k in og_p_state:
                    # 将原始参数状态添加到tracker中
                    tracker.add(og_p_state[k])
                    # 检查并设置新的参数状态，并在tracker中进行检查
                    tracker.pop_check_set(new_p_state[k], self)

            # 断言所有参数是否已弹出
            self.assertTrue(tracker.all_popped())

    # 测试派生优化器
    def _test_derived_optimizers(
        self,
        device,
        dtype,
        optim_info,
        flag,
        reduced_precision=False,
        assert_step_dtype=None,
    ):
        """
        给定标志 'fused' 或 'foreach'，测试在设置为 True 和 False 时优化器状态和更新参数的一致性，
        对提供的优化器配置进行测试。
        """
        # 断言标志值只能是 "foreach" 或 "fused"
        assert flag in ("foreach", "fused")
        # 如果启用了减少精度选项，则使用特定的关键字参数
        assert_eq_kwargs = {} if not reduced_precision else FP16_REDUCED_PRECISION

        # 调用优化器信息对象提供的函数，获取优化器输入
        optim_inputs = optim_info.optim_inputs_func(device=device, dtype=dtype)
        # 获取优化器类
        optim_cls = optim_info.optim_cls
        # 遍历每个优化器输入
        for optim_input in optim_inputs:
            models, optimizers = [], []
            # 深拷贝优化器输入的关键字参数
            kwargs = deepcopy(optim_input.kwargs)
            # 如果 capturable 标志为 True 且设备为 CPU，则跳过当前循环
            if kwargs.get("capturable", False) and str(device) == "cpu":
                continue
            # 对于每个标志值（False 和 True）
            for flag_value in (False, True):
                # 设置关键字参数中的标志值为当前循环的 flag_value
                kwargs[flag] = flag_value
                # 创建一个张量作为输入数据
                input = torch.tensor(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=dtype, device=device
                ).reshape(3, 2)

                torch.manual_seed(1)
                # 创建一个神经网络模型
                model = torch.nn.Sequential(
                    torch.nn.Linear(2, 3),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(3, 1),
                    torch.nn.Sigmoid(),
                )
                # 将模型移动到指定的设备和数据类型上
                model.to(dtype=dtype, device=device)

                # 对于 foreach/fused 优化器，应使用一个零尺寸的张量作为其最后一个参数
                # 参考：https://github.com/pytorch/pytorch/issues/100701
                empty_param = torch.empty(
                    (), device=device, dtype=dtype, requires_grad=True
                )
                empty_param.grad = torch.rand_like(empty_param)
                # 将模型参数和空参数列表传递给优化器对象进行初始化
                params = list(model.parameters()) + [empty_param]

                # 使用给定的参数初始化优化器对象
                optimizer = optim_cls(params, **kwargs)
                models.append(model)
                optimizers.append(optimizer)

            # 调用私有方法 _compare_between，比较不同设置下的模型输出
            self._compare_between(
                input, models, optimizers, assert_eq_kwargs, assert_step_dtype
            )

    @skipMPS  # MPS 不支持 torch.float64，参见 https://github.com/pytorch/pytorch/issues/115350
    @optims(
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float64],
    )
    def test_foreach_matches_forloop(self, device, dtype, optim_info):
        self._test_derived_optimizers(device, dtype, optim_info, "foreach")

    @onlyCUDA
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @parametrize("impl", ["foreach", "fused"])
    @optims(
        [
            optim
            for optim in optim_db
            if "foreach" in optim.supported_impls or "fused" in optim.supported_impls
        ]
    )
    @onlyCUDA
    @optims(
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float64],
    )
    # 测试函数，用于验证在指定设备上使用指定的数据类型和优化器信息，设置默认数据类型，并执行测试函数
    def test_set_default_dtype_works_with_foreach(self, device, dtype, optim_info):
        # GitHub问题链接，详细说明强制step变为float32，除非默认数据类型更高精度为float64
        # 获取当前的默认数据类型
        old_default_dtype = torch.get_default_dtype()
        # 遍历两种默认数据类型：torch.float64和torch.float16
        for default_dtype in [torch.float64, torch.float16]:
            try:
                # 设置默认数据类型为当前遍历到的default_dtype
                torch.set_default_dtype(default_dtype)
                # 调用测试函数_test_derived_optimizers，测试衍生优化器在指定条件下的表现
                self._test_derived_optimizers(
                    device,
                    dtype,
                    optim_info,
                    "foreach",
                    reduced_precision=default_dtype == torch.float16,
                    # 断言step的数据类型为torch.float64（如果default_dtype为torch.float64），否则为torch.float32
                    assert_step_dtype=(
                        torch.float64
                        if default_dtype == torch.float64
                        else torch.float32
                    ),
                )
            finally:
                # 恢复为原来的默认数据类型
                torch.set_default_dtype(old_default_dtype)

    # 仅在CUDA设备上运行的大张量测试，用于验证优化器在foreach模式下的大张量操作
    @onlyCUDA
    @largeTensorTest("72GB", "cuda")
    @optims(
        # 从优化器数据库中选择支持foreach模式的优化器进行测试，数据类型为torch.float16
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float16],
    )
    def test_foreach_large_tensor(self, device, dtype, optim_info):
        # 获取优化器类和优化器输入
        optim_cls = optim_info.optim_cls
        optim_inputs = optim_info.optim_inputs_func(device=device)
        # 遍历优化器输入
        for optim_input in optim_inputs:
            # 创建包含2**32个元素的张量列表，存储在指定设备上，使用指定数据类型
            params = [torch.ones(2**32, device=device, dtype=dtype)]
            # 将params的梯度初始化为与params形状相同的零张量
            params[0].grad = torch.zeros_like(params[0])
            # 使用指定的优化器类，设置foreach模式，并传入优化器输入的kwargs参数
            optimizer = optim_cls(params, foreach=True, **optim_input.kwargs)
            # 执行优化步骤
            optimizer.step()

    # 仅在CUDA原生设备类型上运行的测试函数，用于验证融合优化器与普通for循环优化器的匹配性
    @onlyCUDA
    @optims(
        # 从优化器数据库中选择支持foreach模式的优化器进行测试，数据类型为torch.float32
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float32],
    )
    @optims(
        # 从优化器数据库中选择支持融合模式的优化器进行测试，数据类型为torch.bfloat16和torch.float16
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=floating_types_and(
            torch.bfloat16,
            torch.float16,
        ),
    )
    def test_fused_matches_forloop(self, device, dtype, optim_info):
        # 检查设备类型是否支持融合优化器
        if _get_device_type(device) not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )
        # 如果设备类型为MPS且数据类型不是torch.float16或torch.float32，则跳过测试
        if _get_device_type(device) == "mps" and dtype not in (
            torch.float16,
            torch.float32,
        ):
            self.skipTest("MPS supports only torch.float16 and torch.float32")
        # 调用测试函数_test_derived_optimizers，测试衍生优化器在指定条件下的表现
        self._test_derived_optimizers(device, dtype, optim_info, "fused")

    # 仅在本地设备类型上运行的大张量测试，用于验证融合优化器在大张量操作下的表现
    @onlyNativeDeviceTypes
    @largeTensorTest("64GB")
    @optims(
        # 从优化器数据库中选择支持融合模式的优化器进行测试，数据类型为torch.float16
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=[torch.float16],
    )
    # 测试函数，用于测试在指定设备和数据类型上，是否支持融合优化
    def test_fused_large_tensor(self, device, dtype, optim_info):
        # 如果设备不在支持融合优化的设备列表中，则跳过测试
        if device not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )
        # 获取优化器类
        optim_cls = optim_info.optim_cls
        # 获取优化器输入参数
        optim_inputs = optim_info.optim_inputs_func(device=device)
        # 遍历优化器输入参数
        for optim_input in optim_inputs:
            # 创建参数列表，包含一个巨大的张量
            params = [torch.ones(2**32, device=device, dtype=dtype)]
            # 将参数的梯度设置为零
            params[0].grad = torch.zeros_like(params[0])
            # 创建优化器对象，启用融合优化
            optimizer = optim_cls(params, fused=True, **optim_input.kwargs)
            # 执行一步优化
            optimizer.step()

    # 使用装饰器，指定仅在CUDA设备上运行该测试函数
    @onlyCUDA
    # 使用装饰器，指定该函数应用于支持融合实现的优化器上，并指定数据类型为torch.float32
    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=[torch.float32],
    )
    # 测试函数，用于验证如果存在无穷值，则融合优化不会执行步骤
    def test_fused_does_not_step_if_foundinf(self, device, dtype, optim_info):
        # 如果设备不在支持融合优化的设备列表中，则跳过测试
        if device not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )
        # 获取优化器类
        optim_cls = optim_info.optim_cls
        # 获取优化器输入参数
        optim_inputs = optim_info.optim_inputs_func(device=device)
        # 设置参数数量
        num_params = 5
        # 遍历优化器输入参数
        for optim_input in optim_inputs:
            # 遍历是否禁用梯度缩放的选项
            for no_grad_scale in (False, True):
                # 创建参数列表，每个参数都是长度为1的张量
                params = [
                    torch.ones((1,), device=device, dtype=dtype)
                    for _ in range(num_params)
                ]
                # 克隆并分离参数
                params_c = [param.clone().detach() for param in params]
                # 将每个参数的梯度设置为全1张量
                for p in params:
                    p.grad = torch.ones_like(p)
                # 创建优化器对象，启用融合优化
                optimizer = optim_cls(params, fused=True, **optim_input.kwargs)
                # 设置优化器的梯度缩放和发现无穷值
                optimizer.grad_scale = (
                    None
                    if no_grad_scale
                    else torch.ones((1,), dtype=dtype, device=device)
                )
                optimizer.found_inf = torch.ones((), dtype=dtype, device=device)
                # 执行一步优化
                optimizer.step()
                # 验证每个参数的状态中是否存在步数信息，如果有则期望步数为0
                for p in params:
                    if "step" in optimizer.state[p]:
                        self.assertEqual(
                            torch.zeros((), dtype=dtype, device=device),
                            optimizer.state[p]["step"],
                        )
                # 验证参数列表与克隆的参数列表是否相等
                self.assertEqual(params, params_c)

    # 使用参数化装饰器，指定多个实现（"fused", "capturable"）的测试
    @parametrize("impl", ["fused", "capturable"])
    # 使用装饰器，指定该函数应用于支持融合实现的优化器上，并指定数据类型为torch.float32
    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=[torch.float32],
    )
    # 定义一个测试方法，用于测试在不同条件下加载和使用优化器状态字典
    def test_cpu_load_state_dict(self, device, dtype, impl, optim_info):
        # NOTE: This SIMULATES a fused/capturable optimizer with state moved to CPU, issue 103256
        # 模拟将状态移动到 CPU 的融合/可捕获优化器，问题号 103256
        # How do we get there? Users typically create CUDA models on fused optimizers and then
        # store checkpoints on CPU as CUDA memory is limited with torch.load(...map_location="cpu").
        # 用户通常在融合优化器上创建 CUDA 模型，然后将检查点存储在 CPU 上，因为 CUDA 内存受限，使用 torch.load(...map_location="cpu")。
        # Since this is a unit test, it is more expedient to simulate what the state_dict
        # would look like, which is basically CPU tensors with fused/capturable flag = True.
        # 因为这是一个单元测试，更方便地模拟 state_dict 的样子，基本上是具有融合/可捕获标志为 True 的 CPU 张量。

        # 获取优化器类和其名称
        optim_cls = optim_info.optim_cls
        opt_name = optim_cls.__name__

        # 如果优化器是 SGD 或 Adagrad，并且实现方式是 "capturable"，则跳过测试
        if (
            opt_name
            in (
                "SGD",
                "Adagrad",
            )
            and impl == "capturable"
        ):
            self.skipTest("SGD does not currently support capturable")

        # 如果设备类型是 CPU，则跳过测试
        if _get_device_type(device) == "cpu":
            self.skipTest("Test is only for non-cpu devices")

        # 如果实现方式是 "fused"，并且设备类型不在支持融合的列表中，则跳过测试
        elif (
            impl == "fused"
            and _get_device_type(device) not in optim_info.supports_fused_on
        ):
            self.skipTest(f"{device} is not supported for fused on {opt_name}")

        # 如果实现方式是 "capturable"，并且设备类型是 "mps"，则跳过测试
        elif impl == "capturable" and _get_device_type(device) == "mps":
            self.skipTest("MPS does not support capturable")

        # 生成在 CPU 上优化器的输入
        cpu_optim_inputs = optim_info.optim_inputs_func(device="cpu")

        # 遍历每个在 CPU 上的优化器输入
        for optim_input in cpu_optim_inputs:
            # 创建一个 CPU 上的张量参数
            param = torch.tensor([0.1, 0.2], dtype=dtype, device="cpu")
            
            # 根据当前优化器输入创建优化器对象
            optimizer = optim_cls([param], **optim_input.kwargs)
            
            # 设置参数的梯度为随机值，并执行一步优化
            param.grad = torch.rand_like(param)
            optimizer.step()
            
            # 深拷贝优化器的状态字典
            optim_state_dict_cpu = deepcopy(optimizer.state_dict())
            optim_state_dict_cpu["param_groups"][0][impl] = True

            # 加载状态字典到设备上的优化器
            # 修改当前优化器输入的实现标志为 True，加载状态字典
            optim_input.kwargs[impl] = True
            param_device = param.clone().detach().to(device=device)
            optimizer_device = optim_cls([param_device], **optim_input.kwargs)
            optimizer_device.load_state_dict(optim_state_dict_cpu)
            
            # 将设备上参数的梯度置零，并执行一步优化
            optimizer_device.zero_grad()
            param_device.grad = torch.rand_like(param_device)
            optimizer_device.step()

    @optims(optim_db, dtypes=[torch.float32])
    # 定义一个测试方法，用于测试优化器参数组和权重衰减
    def test_param_groups_weight_decay(self, device, dtype, optim_info):
        # 获取优化器类
        optim_cls = optim_info.optim_cls
        # 从全局参数中获取所有优化器输入，包括特定设备、数据类型和优化器信息
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        # 遍历所有优化器输入
        for optim_input in all_optim_inputs:
            # 复制优化器参数
            weight_kwargs = optim_input.kwargs
            # 深拷贝优化器参数用于偏置，并将权重衰减设为0
            bias_kwargs = deepcopy(optim_input.kwargs)
            bias_kwargs["weight_decay"] = 0.0

            # 创建一个参数张量 weight 和 bias，分别为随机初始化的矩阵和向量
            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            bias = Parameter(torch.randn((10), device=device, dtype=dtype))
            # 创建一个输入张量 input，为随机初始化的向量
            input = torch.randn(5, device=device, dtype=dtype)

            # 使用给定的参数初始化优化器，包括权重和偏置的参数设置
            optimizer = optim_cls(
                [
                    dict(params=[weight], **weight_kwargs),
                    dict(params=[bias], **bias_kwargs),
                ]
            )

            # 计算损失函数，这里是平方和损失
            loss = (weight.mv(input) + bias).pow(2).sum()
            initial_value = loss.item()
            # 迭代优化过程，这里进行了20次迭代
            for _ in range(20):
                optimizer.zero_grad()
                loss = (weight.mv(input) + bias).pow(2).sum()
                loss.backward()
                # 如果优化器仅支持稀疏梯度，则将梯度转换为稀疏格式
                if optim_info.only_supports_sparse_grads:
                    # 对于这个测试，我们简单地将张量布局转换为稀疏格式，这并不是 SparseAdam 等优化器的预期用例！
                    weight.grad = weight.grad.to_sparse()
                    bias.grad = bias.grad.to_sparse()
                optimizer.step()

            # 测试损失的方向是否适当移动
            if optim_input.kwargs.get("maximize", False):
                self.assertGreater(loss.item(), initial_value)
            else:
                self.assertLess(loss.item(), initial_value)

    @optims(optim_db, dtypes=[torch.float32])
    # 定义一个测试方法，用于测试不同参数组合下的优化器行为
    def test_param_groups_lr(self, device, dtype, optim_info):
        # 获取优化器类
        optim_cls = optim_info.optim_cls
        # 获取所有优化器输入参数的组合，包括全局参数，跳过不可微分的参数组
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        # 遍历所有优化器输入参数的组合
        for optim_input in all_optim_inputs:
            # 如果当前参数组的关键字参数中没有学习率(lr)，或者学习率为0，则设定学习率为1e-3
            if "lr" not in optim_input.kwargs or optim_input.kwargs["lr"] == 0:
                optim_input.kwargs["lr"] = 1e-3
            # 外部关键字参数，初始化学习率为1e-28
            outer_kwargs = {"lr": 1e-28}
            # 如果优化器类是 Rprop，则允许步长最小值为0
            if optim_cls.__name__ == "Rprop":
                outer_kwargs["step_sizes"] = (0, 50)

            # 创建三个参数：weight, bias 和 irrelevant，使用给定的设备和数据类型初始化
            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            bias = Parameter(torch.randn((10), device=device, dtype=dtype))
            irrelevant = Parameter(torch.randn(2, device=device, dtype=dtype))
            irrelevant_clone = irrelevant.clone()  # 复制 irrelevant 参数
            input = torch.randn(5, device=device, dtype=dtype)  # 随机生成输入数据

            # 创建优化器对象，传入参数组及其关键字参数，以及外部关键字参数
            optimizer = optim_cls(
                [
                    dict(params=[weight, bias], **optim_input.kwargs),
                    dict(params=[irrelevant]),
                ],
                **outer_kwargs,
            )

            # 计算损失函数值，初始化初始损失值
            loss = (weight.mv(input) + bias).pow(2).sum()
            initial_value = loss.item()

            # 进行20次优化迭代
            for _ in range(20):
                optimizer.zero_grad()  # 清零梯度
                loss = (weight.mv(input) + bias).pow(2).sum()  # 计算损失
                loss.backward()  # 反向传播求梯度
                irrelevant.grad = torch.rand_like(irrelevant)  # 设置 irrelevant 参数的梯度
                # 如果优化器仅支持稀疏梯度，则将权重和偏置的梯度转换为稀疏格式
                if optim_info.only_supports_sparse_grads:
                    weight.grad = weight.grad.to_sparse()
                    bias.grad = bias.grad.to_sparse()
                    irrelevant.grad = irrelevant.grad.to_sparse()
                optimizer.step()  # 执行优化步骤

            # 测试损失函数的变化方向是否符合预期
            if optim_input.kwargs.get("maximize", False):
                self.assertGreater(loss.item(), initial_value)
            else:
                self.assertLess(loss.item(), initial_value)

            # 测试由于学习率几乎为0，无关参数是否未被更新
            self.assertEqual(irrelevant, irrelevant_clone)

    # @optims(optim_db, dtypes=[torch.float32])
    def test_step_is_noop_when_params_have_no_grad(self, device, dtype, optim_info):
        # 从 optim_info 中获取优化器类
        optim_cls = optim_info.optim_cls
        # 获取包括全局参数的所有优化器输入
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        # 创建不需要梯度的张量参数列表
        params = [
            torch.randn(2, 3, requires_grad=False, device=device, dtype=dtype)
            for _ in range(2)
        ]
        # 复制并分离参数，创建旧参数列表
        old_params = [p.clone().detach() for p in params]

        # 定义一个返回固定张量的闭包函数
        def closure():
            return torch.tensor([1], device=device, dtype=dtype)

        # 遍历所有优化器输入
        for optim_input in all_optim_inputs:
            # 使用参数和优化器输入创建优化器对象
            optimizer = optim_cls(params, **optim_input.kwargs)
            # 执行一步优化
            optimizer.step(closure)

    @optims(optim_db, dtypes=[torch.float32])
    def test_step_is_noop_for_zero_grads(self, device, dtype, optim_info):
        # 从 optim_info 中获取优化器类
        optim_cls = optim_info.optim_cls
        # 获取包括全局参数的所有优化器输入
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        # 创建一个需要梯度的张量参数
        param = torch.randn((5, 1), device=device, dtype=dtype, requires_grad=True)
        # 复制并分离参数，创建旧参数
        old_param = param.clone().detach()

        # 定义一个返回固定张量的闭包函数
        def closure():
            return torch.tensor([1], device=device, dtype=dtype)

        # 遍历所有优化器输入
        for optim_input in all_optim_inputs:
            kwargs = optim_input.kwargs

            # 如果 weight_decay 不为 0，则跳过当前优化器输入
            if kwargs.get("weight_decay", 0) != 0:
                continue

            # 如果优化器是 AdamW，则将学习率设为较小的值
            if optim_cls.__name__ == "AdamW":
                kwargs["lr"] = (
                    torch.tensor(1e-5)
                    if isinstance(kwargs.get("lr", 1e-5), torch.Tensor)
                    else 1e-5
                )

            # 如果设置了 differentiable 为 True，则使用参数的克隆
            if kwargs.get("differentiable", False):
                params = [param.clone()]
            else:
                params = [param]

            # 使用参数和优化器输入创建优化器对象
            optimizer = optim_cls(params, **kwargs)

            # 如果优化器仅支持稀疏梯度
            if optim_info.only_supports_sparse_grads:
                # 故意构造一个多维度的空 v，以模拟稀疏梯度
                # 单维度的 v 可能通过测试，但多维度的可以正确复现问题
                i = torch.empty((1, 0), device=device, dtype=dtype)
                v = torch.empty((0, 1), device=device, dtype=dtype)
                params[0].grad = torch.sparse_coo_tensor(
                    i, v, (5, 1), device=device, dtype=dtype
                )
            else:
                # 否则，将参数的梯度设为与参数形状相同的全零张量
                params[0].grad = torch.zeros_like(params[0])

            # 执行一步优化
            optimizer.step(closure)
            # 断言旧参数与当前参数相等
            self.assertEqual(old_param, params[0])

    @optims(optim_db, dtypes=[torch.float32])
    # 定义一个测试方法，用于验证优化器对象是否可以被打印
    def test_optimizer_can_be_printed(self, device, dtype, optim_info):
        # 从优化器信息中获取优化器类
        optim_cls = optim_info.optim_cls
        # 获取包含全局参数的优化器输入列表，这些参数包括设备类型和数据类型
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        # 创建包含两个形状为 (2, 3) 的张量参数，这些参数需要梯度计算，设备和数据类型与给定的设备和数据类型一致
        params = [
            Parameter(torch.randn(2, 3, requires_grad=True, device=device, dtype=dtype))
            for _ in range(2)
        ]
        # 遍历所有的优化器输入
        for optim_input in all_optim_inputs:
            # 使用给定的参数和优化器输入实例化优化器对象
            optimizer = optim_cls(params, **optim_input.kwargs)
            # 调用优化器对象的 __repr__() 方法，此处未保存或打印返回值
            optimizer.__repr__()

    # optims 是一个装饰器，应用于下面的函数或方法，使用给定的优化器数据库和数据类型列表
    @optims(optim_db, dtypes=[torch.float32])
    def test_state_dict_deterministic(self, device, dtype, optim_info):
        # 获取优化器类
        optim_cls = optim_info.optim_cls

        # 跳过可微测试，参见 https://github.com/pytorch/pytorch/issues/116490
        # 获取所有优化器输入，包括全局参数，但跳过 "differentiable"
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )

        # 创建一个需要梯度的参数 weight
        weight = Parameter(
            torch.randn(2, 3, requires_grad=True, device=device, dtype=dtype)
        )
        # 创建一个需要梯度的参数 bias
        bias = Parameter(torch.randn(2, requires_grad=True, device=device, dtype=dtype))
        # 创建一个输入张量 input，需要梯度
        input = torch.randn(3, requires_grad=True, device=device, dtype=dtype)
        # 将 weight 和 bias 放入 params 列表中
        params = [weight, bias]

        # 定义前向传播和反向传播函数 fwd_bwd
        def fwd_bwd(optim, w, b, i):
            optim.zero_grad()
            # 计算损失函数
            loss = (w.mv(i) + b).pow(2).sum()
            loss.backward()
            return loss

        # 遍历所有优化器输入
        for optim_input in all_optim_inputs:
            # 使用给定的优化器类和参数初始化优化器 optimizer
            optimizer = optim_cls(params, **optim_input.kwargs)
            # 使用 functools.partial 创建一个闭包 closure，固定了当前的 optimizer, weight, bias, input
            closure = functools.partial(fwd_bwd, optimizer, weight, bias, input)

            # 预热优化器
            for _ in range(10):
                # 如果步骤需要使用闭包
                if optim_info.step_requires_closure:
                    optimizer.step(closure)
                else:
                    closure()
                    optimizer.step()

            # 克隆权重并构建一个新的优化器 optimizer_c
            with torch.no_grad():
                weight_c = Parameter(weight.clone())
                bias_c = Parameter(bias.clone())
            optimizer_c = optim_cls([weight_c, bias_c], **optim_input.kwargs)
            closure_c = functools.partial(fwd_bwd, optimizer_c, weight_c, bias_c, input)

            # 从原始优化器中加载状态字典到新的优化器 optimizer_c
            optimizer_c.load_state_dict(deepcopy(optimizer.state_dict()))

            # 并行运行两个优化器
            for _ in range(10):
                if optim_info.step_requires_closure:
                    optimizer.step(closure)
                    optimizer_c.step(closure_c)
                else:
                    closure()
                    closure_c()
                    optimizer.step()
                    optimizer_c.step()

                # 断言 weight 和 weight_c 相等
                self.assertEqual(weight, weight_c)
                # 断言 bias 和 bias_c 相等
                self.assertEqual(bias, bias_c)

            # 确保状态字典在相同参数下是确定性的（不是完全相同的）
            self.assertEqual(optimizer.state_dict(), optimizer_c.state_dict())

            # 确保重复的参数具有相同的表示（参见 #36831）
            optimizer_c.param_groups.extend(optimizer_c.param_groups)
            self.assertEqual(
                optimizer.state_dict()["param_groups"][-1],
                optimizer_c.state_dict()["param_groups"][-1],
            )

    @optims(optim_db, dtypes=[torch.float32])
    # 测试是否可以加载旧的状态字典
    def test_can_load_older_state_dict(self, device, dtype, optim_info):
        # 定义新添加的标志列表
        new_flags = ["maximize", "foreach", "fused", "differentiable", "capturable"]
        # 获取优化器类别
        optim_cls = optim_info.optim_cls

        # 跳过不支持不可微测试的情况，参考 https://github.com/pytorch/pytorch/issues/116490
        # 获取所有优化器输入，包括全局的关键参数
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )

        # 遍历所有优化器输入
        for optim_input in all_optim_inputs:
            # 设置随机种子
            torch.manual_seed(1)
            # 创建一个简单的神经网络模型
            model = torch.nn.Sequential(
                torch.nn.Conv2d(4, 2, 1, stride=2),
                torch.nn.BatchNorm2d(2, eps=1e-05, momentum=0.1),
            )
            # 将模型移动到指定的设备和数据类型
            model.to(dtype=dtype, device=device)
            # 创建输入数据
            input = torch.rand(1, 4, 16, 16, device=device, dtype=dtype)
            # 使用优化器类别和当前优化器输入参数初始化优化器
            optimizer = optim_cls(model.parameters(), **optim_input.kwargs)

            # 定义前向传播和反向传播函数
            def fwd_bwd(optim, mod, i):
                optim.zero_grad()
                loss = mod(i).sum()
                loss.backward()
                return loss

            # 执行优化器步骤多次，以确保优化器状态更新正常
            for _ in range(3):
                if optim_info.step_requires_closure:
                    optimizer.step(functools.partial(fwd_bwd, optimizer, model, input))
                else:
                    fwd_bwd(optimizer, model, input)
                    optimizer.step()

            # 复制优化器当前的状态字典，准备处理旧的状态字典
            old_state_dict = deepcopy(optimizer.state_dict())
            # 获取旧状态字典中的参数组列表
            old_state_dict_pg = old_state_dict["param_groups"]
            # 遍历每个参数组，删除其中的新标志
            for group in old_state_dict_pg:
                for flag in new_flags:
                    if flag in group:
                        del group[flag]

            # 加载处理后的旧状态字典到优化器中
            optimizer.load_state_dict(old_state_dict)

            # 确保即使处理了旧状态字典，优化器仍然能够进行步骤更新
            if optim_info.step_requires_closure:
                optimizer.step(functools.partial(fwd_bwd, optimizer, model, input))
            else:
                fwd_bwd(optimizer, model, input)
                optimizer.step()

    # optims 装饰器用于指定测试函数的参数，包括优化器数据库和数据类型列表
    @optims(optim_db, dtypes=[torch.float32])
    # 定义一个测试函数，用于验证保存和加载带有仅权重的优化器状态时的一致性
    def test_save_load_equality_with_weights_only(self, device, dtype, optim_info):
        # 从 optim_info 中获取优化器类
        optim_cls = optim_info.optim_cls

        # 跳过可微测试，参见 https://github.com/pytorch/pytorch/issues/116490
        # 获取所有优化器输入，包括全局的特定参数，排除 "differentiable" 参数
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )

        # 创建一个需要梯度的权重参数
        weight = Parameter(
            torch.randn(2, 3, requires_grad=True, device=device, dtype=dtype)
        )
        # 创建一个需要梯度的偏置参数
        bias = Parameter(torch.randn(2, requires_grad=True, device=device, dtype=dtype))
        # 创建一个需要梯度的输入张量
        input = torch.randn(3, requires_grad=True, device=device, dtype=dtype)
        # 将权重和偏置参数放入列表中
        params = [weight, bias]

        # 定义一个前向传播和反向传播的函数
        def fwd_bwd(optim, w, b, i):
            optim.zero_grad()
            # 计算损失函数，对应 (w.mv(i) + b).pow(2).sum()
            loss = (w.mv(i) + b).pow(2).sum()
            # 反向传播计算梯度
            loss.backward()
            # 如果优化器仅支持稀疏梯度，则将梯度转换为稀疏格式
            if optim_info.only_supports_sparse_grads:
                weight.grad = weight.grad.to_sparse()
                bias.grad = bias.grad.to_sparse()
            return loss

        # 对所有优化器输入进行迭代
        for optim_input in all_optim_inputs:
            # 使用给定的优化器类和参数初始化优化器
            optimizer = optim_cls(params, **optim_input.kwargs)
            # 创建一个带有部分参数的闭包函数
            closure = functools.partial(fwd_bwd, optimizer, weight, bias, input)

            # 预热优化器，执行若干步优化
            for _ in range(3):
                optimizer.step(closure)

            # 获取当前优化器的状态字典
            sd = optimizer.state_dict()

            # === 检查保存和加载的状态字典是否一致（包括仅加载权重的情况）。===
            # 使用临时文件保存状态字典
            with tempfile.TemporaryFile() as f:
                torch.save(sd, f)
                f.seek(0)
                # 加载保存的状态字典
                sd_copy = torch.load(f)
                # 断言加载的状态字典与原始状态字典 sd 相等
                self.assertEqual(sd_copy, sd)
                del sd_copy
                f.seek(0)
                # 仅加载权重的状态字典
                sd_copy_wo = torch.load(f, weights_only=True)
                # 断言仅加载权重的状态字典与原始状态字典 sd 相等
                self.assertEqual(sd_copy_wo, sd)

    # 用于测试的装饰器，传入优化器数据库和数据类型列表 [torch.float32]
    @optims(optim_db, dtypes=[torch.float32])
    # 测试加载非张量步骤的方法
    def test_load_nontensor_step(self, device, dtype, optim_info):
        # 从 optim_info 中获取优化器类
        optim_cls = optim_info.optim_cls

        # 跳过可微测试，参见 https://github.com/pytorch/pytorch/issues/116490
        # 获取包含全局参数的优化器输入
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        
        # 创建参数列表，每个参数是一个形状为 (2, 3) 的张量，并随机初始化
        params = [
            Parameter(torch.randn(2, 3, device=device, dtype=dtype)) for _ in range(2)
        ]
        # 为每个参数设置随机梯度张量
        for p in params:
            p.grad = torch.rand_like(p)
            # 如果优化器仅支持稀疏梯度，则将梯度转换为稀疏张量
            if optim_info.only_supports_sparse_grads:
                p.grad = p.grad.to_sparse()

        # 用于二阶优化器（如 LBFGS）的闭包损失函数，随机初始化
        closure_loss = torch.rand(1, device=device, dtype=dtype)

        # 定义闭包函数，根据优化器是否需要闭包返回不同的值
        def closure():
            return closure_loss if optim_info.step_requires_closure else None

        # 遍历所有优化器输入
        for optim_input in all_optim_inputs:
            # 从优化器输入中获取参数
            kwargs = optim_input.kwargs
            # 创建优化器对象
            optimizer = optim_cls(params, **optim_input.kwargs)
            # 对优化器执行多次优化步骤
            for _ in range(3):
                optimizer.step(closure)
            
            # 深拷贝优化器的状态字典
            state_dict = deepcopy(optimizer.state_dict())
            
            # 将状态字典中所有张量类型的步数转换为标量值
            for p_state in state_dict["state"].values():
                if "step" in p_state and torch.is_tensor(p_state["step"]):
                    p_state["step"] = p_state["step"].item()
            
            # 加载经过修改的状态字典到优化器中
            optimizer.load_state_dict(state_dict)
            
            # 再次执行优化步骤
            optimizer.step(closure)

    # 用于 CUDA 的装饰器
    @onlyCUDA
    # 用于优化器的装饰器，指定优化器数据库和数据类型为 float32
    @optims(optim_db, dtypes=[torch.float32])
    # 静态方法：预状态字典钩子
    @staticmethod
    def _state_dict_pre_hook(optimizer: Optimizer) -> None:
        # 在优化器状态字典中设置测试键值对
        optimizer.state["test"] = 1

    # 静态方法：后状态字典钩子
    @staticmethod
    def _state_dict_post_hook(
        optimizer: Optimizer, state_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        # 如果状态字典中包含测试键，将其移除，并设置一个标志指示运行了预状态字典钩子
        if "test" in state_dict["state"]:
            state_dict["state"].pop("test")
            state_dict["ran_state_dict_pre_hook"] = True
        else:
            state_dict["ran_state_dict_pre_hook"] = False
        # 返回更新后的状态字典
        return state_dict

    # 用于优化器的装饰器，指定优化器数据库和数据类型为 float32
    @optims(optim_db, dtypes=[torch.float32])
    # 测试状态字典预钩子的方法
    def test_state_dict_pre_hook(self, device, dtype, optim_info):
        # 从 optim_info 中获取优化器类
        optim_cls = optim_info.optim_cls
        # 获取包含全局参数的优化器输入
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        # 遍历所有优化器输入
        for optim_input in all_optim_inputs:
            # 创建一个随机张量参数，形状为 (2, 3)，需要梯度
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            # 创建优化器对象
            optim = optim_cls([param], **optim_input.kwargs)
            # 注册状态字典预钩子
            optim.register_state_dict_pre_hook(self.__class__._state_dict_pre_hook)
            # 获取优化器的状态字典
            state_dict = optim.state_dict()
            # 断言测试键在状态字典中的值为 1
            self.assertEqual(state_dict["state"]["test"], 1)
    # 测试 state_dict_post_hook 方法
    def test_state_dict_post_hook(self, device, dtype, optim_info):
        # 获取优化器类
        optim_cls = optim_info.optim_cls
        # 获取包含全局参数的所有优化器输入
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        # 遍历所有优化器输入
        for optim_input in all_optim_inputs:
            # 创建具有梯度的参数张量
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            # 创建优化器对象
            optim = optim_cls([param], **optim_input.kwargs)
            # 注册 state_dict_post_hook 方法
            optim.register_state_dict_post_hook(self.__class__._state_dict_post_hook)
            # 获取优化器状态字典
            state_dict = optim.state_dict()
            # 断言 state_dict 中不包含 "ran_state_dict_pre_hook" 键
            self.assertFalse(state_dict["ran_state_dict_pre_hook"])

    # 测试 state_dict_pre_post_hook 方法
    @optims(optim_db, dtypes=[torch.float32])
    def test_state_dict_pre_post_hook(self, device, dtype, optim_info):
        # 获取优化器类
        optim_cls = optim_info.optim_cls
        # 获取包含全局参数的所有优化器输入
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        # 遍历所有优化器输入
        for optim_input in all_optim_inputs:
            # 创建具有梯度的参数张量
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            # 创建优化器对象
            optim = optim_cls([param], **optim_input.kwargs)
            # 注册 state_dict_pre_hook 方法
            optim.register_state_dict_pre_hook(self.__class__._state_dict_pre_hook)
            # 注册 state_dict_post_hook 方法
            optim.register_state_dict_post_hook(self.__class__._state_dict_post_hook)
            # 获取优化器状态字典
            state_dict = optim.state_dict()
            # 断言 state_dict["state"] 中不包含 "test" 键
            self.assertFalse("test" in state_dict["state"])
            # 断言 state_dict["ran_state_dict_pre_hook"] 为 True
            self.assertTrue(state_dict["ran_state_dict_pre_hook"])

    # 加载 state_dict_pre_hook1 方法
    @staticmethod
    def _load_state_dict_pre_hook1(
        optimizer: Optimizer, state_dict: Dict[str, Any]
    ) -> None:
        # 修改 state_dict 中第一个参数组的学习率为 0.002
        state_dict["param_groups"][0]["lr"] = 0.002

    # 加载 state_dict_pre_hook2 方法
    @staticmethod
    def _load_state_dict_pre_hook2(
        optimizer: Optimizer, state_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        # 返回修改后的 state_dict，将第一个参数组的学习率修改为 0.003
        my_state_dict = deepcopy(state_dict)
        my_state_dict["param_groups"][0]["lr"] = 0.003
        return my_state_dict

    # 加载 state_dict_post_hook 方法
    @staticmethod
    def _load_state_dict_post_hook(optimizer: Optimizer) -> None:
        # 设置 optimizer 的状态，表示运行了 load_state_dict_pre_hook2 方法
        optimizer.state["ran_load_state_dict_pre_hook2"] = (
            optimizer.param_groups[0]["lr"] == 0.003
        )
        # 设置 optimizer 的状态，表示运行了 load_state_dict_post_hook 方法
        optimizer.state["ran_load_state_dict_post_hook"] = True

    # 测试优化器
    @optims(optim_db, dtypes=[torch.float32])
    # 定义测试方法，用于测试加载状态字典前钩子和前置钩子的功能
    def test_load_state_dict_pre_hook_and_prepend(self, device, dtype, optim_info):
        # 从optim_info获取优化器类
        optim_cls = optim_info.optim_cls
        # 获取包括全局参数的所有优化器输入
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        # 遍历所有优化器输入
        for optim_input in all_optim_inputs:
            # 创建一个带有随机参数的张量
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            # 使用优化器类和参数初始化优化器对象
            optim = optim_cls([param], **optim_input.kwargs)
            # 获取当前优化器的状态字典
            state_dict = optim.state_dict()

            # 注册加载状态字典前钩子，指定为class的_load_state_dict_pre_hook1方法
            optim.register_load_state_dict_pre_hook(
                self.__class__._load_state_dict_pre_hook1
            )
            # 加载状态字典到优化器
            optim.load_state_dict(state_dict)
            # 断言参数组的学习率为0.002
            self.assertEqual(optim.param_groups[0]["lr"], 0.002)

            # 注册加载状态字典前钩子，指定为class的_load_state_dict_pre_hook2方法，并指定在最前面插入
            optim.register_load_state_dict_pre_hook(
                self.__class__._load_state_dict_pre_hook2, prepend=True
            )
            # 再次加载状态字典到优化器
            optim.load_state_dict(state_dict)
            # 因为prepend为True，_load_state_dict_pre_hook2钩子会覆盖之前的设定，所以学习率仍为0.002
            self.assertEqual(optim.param_groups[0]["lr"], 0.002)

    # 使用装饰器定义测试方法，用于测试加载状态字典后钩子的功能
    @optims(optim_db, dtypes=[torch.float32])
    def test_load_state_dict_post_hook(self, device, dtype, optim_info):
        # 从optim_info获取优化器类
        optim_cls = optim_info.optim_cls
        # 获取包括全局参数的所有优化器输入
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        # 遍历所有优化器输入
        for optim_input in all_optim_inputs:
            # 创建一个带有随机参数的张量
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            # 使用优化器类和参数初始化优化器对象
            optim = optim_cls([param], **optim_input.kwargs)

            # 注册加载状态字典后钩子，指定为class的_load_state_dict_post_hook方法
            optim.register_load_state_dict_post_hook(
                self.__class__._load_state_dict_post_hook
            )
            # 加载状态字典到优化器
            optim.load_state_dict(optim.state_dict())
            # 确保ran_load_state_dict_pre_hook2状态为False
            self.assertFalse(optim.state["ran_load_state_dict_pre_hook2"])
            # 确保ran_load_state_dict_post_hook状态为True
            self.assertTrue(optim.state["ran_load_state_dict_post_hook"])

    # 使用装饰器定义测试方法，用于测试加载状态字典前后钩子的功能
    @optims(optim_db, dtypes=[torch.float32])
    def test_load_state_dict_pre_post_hook(self, device, dtype, optim_info):
        # 从optim_info获取优化器类
        optim_cls = optim_info.optim_cls
        # 获取包括全局参数的所有优化器输入
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        # 遍历所有优化器输入
        for optim_input in all_optim_inputs:
            # 创建一个带有随机参数的张量
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            # 使用优化器类和参数初始化优化器对象
            optim = optim_cls([param], **optim_input.kwargs)

            # 注册加载状态字典前钩子，指定为class的_load_state_dict_pre_hook2方法
            optim.register_load_state_dict_pre_hook(
                self.__class__._load_state_dict_pre_hook2
            )
            # 注册加载状态字典后钩子，指定为class的_load_state_dict_post_hook方法
            optim.register_load_state_dict_post_hook(
                self.__class__._load_state_dict_post_hook
            )
            # 加载状态字典到优化器
            optim.load_state_dict(optim.state_dict())
            # 确保ran_load_state_dict_pre_hook2状态为True
            self.assertTrue(optim.state["ran_load_state_dict_pre_hook2"])
            # 确保ran_load_state_dict_post_hook状态为True
            self.assertTrue(optim.state["ran_load_state_dict_post_hook"])

    # 使用装饰器定义测试方法，未提供具体代码块内容
    @optims(optim_db, dtypes=[torch.float32])
    # 定义测试方法，用于测试步骤后钩子的行为
    def test_step_post_hook(self, device, dtype, optim_info):
        # 定义后钩子函数，用于在优化步骤后执行额外操作
        def post_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data  # 使用非局部变量data
            data += 2  # 将data增加2

        params = [torch.tensor([1, 1], device=device, dtype=dtype)]  # 定义张量参数列表

        def dummy_closure():  # 定义一个虚拟闭包函数
            return 1

        # 根据优化信息选择是否使用闭包
        closure = dummy_closure if optim_info.step_requires_closure else None

        # 获取包括全局参数在内的所有优化输入
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )

        # 遍历所有优化输入
        for optim_input in all_optim_inputs:
            # 根据优化器信息和参数初始化优化器对象
            optim = optim_info.optim_cls(params, **optim_input.kwargs)

            data = 2  # 初始化data为2
            # 注册步骤后钩子，并获取钩子句柄
            hook_handle = optim.register_step_post_hook(post_hook)

            optim.step(closure)  # 执行优化步骤
            optim.step(closure)  # 再次执行优化步骤
            # 检查是否成功注册了后钩子，验证data是否为6
            self.assertEqual(data, 6)

            # 移除钩子句柄，再次执行优化步骤，验证钩子是否被正确移除
            hook_handle.remove()
            optim.step(closure)
            self.assertEqual(data, 6)  # data应保持为6，因为钩子已移除

    # 使用装饰器optims标记的测试方法，用于测试步骤前钩子的行为
    @optims(optim_db, dtypes=[torch.float32])
    def test_step_pre_hook(self, device, dtype, optim_info):
        # 定义前钩子函数，用于在优化步骤前执行额外操作
        def pre_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data  # 使用非局部变量data
            data += 2  # 将data增加2

        params = [torch.tensor([1, 1], device=device, dtype=dtype)]  # 定义张量参数列表

        def dummy_closure():  # 定义一个虚拟闭包函数
            return 1

        # 根据优化信息选择是否使用闭包
        closure = dummy_closure if optim_info.step_requires_closure else None

        # 获取包括全局参数在内的所有优化输入
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )

        # 遍历所有优化输入
        for optim_input in all_optim_inputs:
            # 根据优化器信息和参数初始化优化器对象
            optim = optim_info.optim_cls(params, **optim_input.kwargs)

            data = 5  # 初始化data为5
            # 注册步骤前钩子，并获取钩子句柄
            hook_handle = optim.register_step_pre_hook(pre_hook)

            optim.step(closure)  # 执行优化步骤
            optim.step(closure)  # 再次执行优化步骤
            # 检查是否成功注册了前钩子，验证data是否为9
            self.assertEqual(data, 9)

            # 移除钩子句柄，再次执行优化步骤，验证钩子是否被正确移除
            hook_handle.remove()
            optim.step(closure)
            self.assertEqual(data, 9)  # data应保持为9，因为钩子已移除
    def test_step_all_hooks(self, device, dtype, optim_info):
        # 定义全局预处理钩子函数，向数据列表中添加0
        def global_pre_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data.append(0)

        # 定义全局后处理钩子函数，向数据列表中添加5
        def global_post_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data.append(5)

        # 定义局部预处理钩子函数，向数据列表中添加1
        def local_pre_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data.append(1)

        # 定义局部后处理钩子函数，向数据列表中添加2
        def local_post_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data.append(2)

        # 创建包含一个张量参数的列表
        params = [torch.tensor([1, 1], device=device, dtype=dtype)]

        # 定义一个简单的闭包函数
        def dummy_closure():
            return 1

        # 根据条件选择是否使用闭包函数
        closure = dummy_closure if optim_info.step_requires_closure else None

        # 获取包含全局参数的优化器输入列表
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )

        # 对每个优化器输入进行迭代
        for optim_input in all_optim_inputs:
            # 使用优化器类和参数初始化优化器对象
            optim = optim_info.optim_cls(params, **optim_input.kwargs)
            # 创建第二个优化器对象，使用SGD作为默认优化器
            optim2 = SGD(params)
            # 初始化数据列表
            data = []

            # 注册全局预处理钩子到两个优化器
            global_pre_handle = register_optimizer_step_pre_hook(global_pre_hook)
            # 注册全局后处理钩子到两个优化器
            global_post_handle = register_optimizer_step_post_hook(global_post_hook)

            # 注册第一个优化器的局部预处理钩子
            first_pre_handle = optim.register_step_pre_hook(local_pre_hook)
            # 注册第一个优化器的局部后处理钩子
            first_post_handle = optim.register_step_post_hook(local_post_hook)
            # 注册第二个优化器的局部预处理钩子
            second_pre_handle = optim2.register_step_pre_hook(local_pre_hook)
            # 注册第二个优化器的局部后处理钩子
            second_post_handle = optim2.register_step_post_hook(local_post_hook)

            # 执行第一个优化器的步骤，传入闭包函数（如果有）
            optim.step(closure)
            # 断言数据列表的内容符合预期
            self.assertListEqual(data, [0, 1, 2, 5])
            # 执行第二个优化器的步骤，传入闭包函数（如果有）
            optim2.step(closure)
            # 断言数据列表的内容符合预期
            self.assertListEqual(data, [0, 1, 2, 5, 0, 1, 2, 5])
            # 再次执行第一个优化器的步骤，传入闭包函数（如果有）
            optim.step(closure)
            # 断言数据列表的内容符合预期
            self.assertListEqual(data, [0, 1, 2, 5, 0, 1, 2, 5, 0, 1, 2, 5])

            # 移除所有钩子
            global_pre_handle.remove()
            global_post_handle.remove()
            first_pre_handle.remove()
            first_post_handle.remove()
            second_pre_handle.remove()
            second_post_handle.remove()

            # 再次执行两个优化器的步骤，不再注册任何钩子
            optim.step(closure)
            optim2.step(closure)
            # 断言数据列表的内容符合预期
            self.assertListEqual(data, [0, 1, 2, 5, 0, 1, 2, 5, 0, 1, 2, 5])

    @optims(optim_db, dtypes=[torch.float32])
    def test_deepcopy_copies_all_public_attrs(self, device, dtype, optim_info):
        # 获取优化器类
        optim_cls = optim_info.optim_cls

        # 获取所有优化器输入，包括全局的参数
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )

        # 创建参数列表，包含两个形状为 (2, 3) 的参数张量
        params = [
            Parameter(torch.randn(2, 3, device=device, dtype=dtype)) for _ in range(2)
        ]
        # 为每个参数设置梯度为与其同形状的随机张量
        for p in params:
            p.grad = torch.rand_like(p)
            # 如果优化器仅支持稀疏梯度，则将梯度转换为稀疏张量
            if optim_info.only_supports_sparse_grads:
                # 对于这个测试，我们简单地将张量布局转换为稀疏格式，这不是 SparseAdam 等优化器的预期用法！
                p.grad = p.grad.to_sparse()

        # 对于类似 LBFGS 的二阶优化器需要闭包函数
        def closure():
            return 1 if optim_info.step_requires_closure else None

        # 获取对象的所有公共属性（不包括下划线开头的私有属性）
        def getPublicAttrs(obj):
            return {k for k in obj.__dict__ if not k.startswith("_")}

        # 对所有优化器输入执行测试
        for optim_input in all_optim_inputs:
            # 使用给定的参数和优化器输入创建优化器对象
            optimizer = optim_cls(params, **optim_input.kwargs)

            # 创建一些状态
            for _ in range(3):
                if optim_info.step_requires_closure:
                    # 如果优化器要求步骤需要闭包函数，则调用带闭包的步骤函数
                    optimizer.step(closure)
                else:
                    # 否则仅调用闭包函数
                    closure()
                    optimizer.step()

            # 断言深拷贝后的优化器具有相同的公共属性
            self.assertEqual(
                getPublicAttrs(optimizer), getPublicAttrs(deepcopy(optimizer))
            )

    @optims(
        [optim for optim in optim_db if optim.step_requires_closure],
        dtypes=[torch.float32],
    )
    def test_second_order_optims_return_consistent_types(
        self, device, dtype, optim_info
    ):
        # 受 #7586 启发
        # 获取优化器类
        optim_cls = optim_info.optim_cls
        # 创建两个张量参数，形状分别为 (10, 5) 和 (10)
        params = [
            torch.randn(10, 5, device=device, dtype=dtype),
            torch.randn(10, device=device, dtype=dtype),
        ]

        # 定义闭包函数，返回一个张量 [10]
        def closure():
            return torch.tensor([10], device=device, dtype=dtype)

        # 遍历优化器输入函数生成的所有优化器输入
        for optim_input in optim_info.optim_inputs_func(device=device):
            # 目前，唯一的二阶优化器是 LBFGS，因此我们直接修改 "tolerance_grad"，但如果将来添加了其他二阶优化器，则可能不适用
            kwargs = optim_input.kwargs
            kwargs["tolerance_grad"] = math.inf
            optim_inf = optim_cls(params, **kwargs)
            kwargs["tolerance_grad"] = -math.inf
            optim_neg_inf = optim_cls(params, **kwargs)

            # 执行优化器的步骤函数，并断言其返回值类型一致
            res1 = optim_inf.step(closure)
            res2 = optim_neg_inf.step(closure)
            self.assertEqual(type(res1), type(res2))

    @onlyCUDA
    @optims(
        [
            optim
            for optim in optim_db
            if "cpu" in optim.supports_fused_on and "cuda" in optim.supports_fused_on
        ],
        dtypes=floating_types_and(
            torch.bfloat16,
            torch.float16,
        ),
    )
    def test_fused_cpu_matches_cuda(self, device, dtype, optim_info):
        # 使用支持在CPU和CUDA上融合操作的优化器进行测试
        optim_cls = optim_info.optim_cls
        # 获取针对CPU设备的优化器输入
        optim_inputs = optim_info.optim_inputs_func(device="cpu")
        for optim_input in optim_inputs:
            inpts, models, optimizers = [], [], []
            for dev in ("cpu", "cuda"):
                kwargs = optim_input.kwargs
                kwargs["fused"] = True
                # 创建具有指定数据类型和设备的张量
                inpt = torch.tensor(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=dtype, device=dev
                ).reshape(3, 2)

                torch.manual_seed(1)
                # 创建具有指定数据类型和设备的神经网络模型
                model = torch.nn.Sequential(
                    torch.nn.Linear(2, 3),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(3, 1),
                    torch.nn.Sigmoid(),
                )
                model.to(dtype=dtype, device=dev)

                # 为foreach/fused优化器测试使用零大小张量作为最后一个参数
                # 参考：https://github.com/pytorch/pytorch/issues/100701
                empty_param = torch.empty(
                    (), device=dev, dtype=dtype, requires_grad=True
                )
                empty_param.grad = torch.rand_like(empty_param)
                params = list(model.parameters()) + [empty_param]

                # 创建优化器实例
                optimizer = optim_cls(params, **kwargs)
                inpts.append(inpt)
                models.append(model)
                optimizers.append(optimizer)
        # 调用测试方法，比较不同配置下的结果
        self._compare_between(inpts, models, optimizers)

    @onlyCUDA
    @optims(
        [o for o in optim_db if "foreach" in o.supported_impls], dtypes=[torch.float32]
    )
    def test_defaults_changed_to_foreach(self, device, dtype, optim_info):
        # 测试默认优化器实现是否已更改为foreach
        optim_cls = optim_info.optim_cls
        # 创建具有指定数据类型和设备的线性模型
        model = torch.nn.Linear(5, 5)
        model.to(dtype=dtype, device=device)
        # 创建具有指定数据类型和设备的输入张量
        inpt = torch.rand(2, 5, dtype=dtype, device=device)

        import inspect

        # 获取优化器类的模块信息
        module = inspect.getmodule(optim_cls)

        for optim_input in optim_info.optim_inputs_func(device=device):
            # 创建优化器实例
            optim = optim_cls(model.parameters(), **optim_input.kwargs)
            optim.zero_grad()
            output = model(inpt)
            loss = output.sum()
            loss.backward()
            # 使用模拟的foreach实现对象进行操作
            with patch.object(
                module, f"_multi_tensor_{optim_cls.__name__.lower()}"
            ) as mocked_foreach_impl:
                optim.step()
                # 断言模拟的foreach实现已被调用
                self.assertTrue(mocked_foreach_impl.called)

    @optims(optim_db, dtypes=[torch.float32])
    # 测试非空状态的方法，使用指定的设备、数据类型和优化器信息
    def test_non_empty_state(self, device, dtype, optim_info):
        # 存在内部测试以确保状态不为空

        # 获取优化器类
        optim_cls = optim_info.optim_cls
        # 创建一个简单的线性模型
        model = torch.nn.Linear(5, 5)
        # 将模型移动到指定的设备和数据类型
        model.to(dtype=dtype, device=device)
        # 创建输入数据
        inpt = torch.rand(2, 5, dtype=dtype, device=device)

        # 对于优化器输入中的每个配置项
        for optim_input in optim_info.optim_inputs_func(device=device):
            # 使用给定的参数和配置项创建优化器实例
            optim = optim_cls(model.parameters(), **optim_input.kwargs)
            # 清空梯度
            optim.zero_grad()
            # 将输入数据传递给模型，获取输出
            output = model(inpt)
            # 计算损失，这里是输出的总和
            loss = output.sum()
            # 反向传播计算梯度
            loss.backward()

            # 如果优化器仅支持稀疏梯度
            if optim_info.only_supports_sparse_grads:
                # 将模型参数的梯度转换为稀疏格式
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = param.grad.to_sparse()

            # 如果优化器的步骤需要闭包
            if optim_info.step_requires_closure:
                # 使用闭包函数执行优化步骤
                optim.step(lambda: 1.0)
            else:
                # 执行优化步骤
                optim.step()

            # 遍历优化器状态中的每个状态，断言状态长度大于零
            for state in optim.state.values():
                self.assertGreater(len(state), 0)
# 调用函数 instantiate_device_type_tests，用于实例化设备类型测试，将 TestOptimRenewed 类的测试用例添加到全局变量中，允许多处理器系统。
instantiate_device_type_tests(TestOptimRenewed, globals(), allow_mps=True)

# 如果当前脚本作为主程序运行，则调用 run_tests 函数执行测试。
if __name__ == "__main__":
    run_tests()
```