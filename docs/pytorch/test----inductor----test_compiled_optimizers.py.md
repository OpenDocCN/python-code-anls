# `.\pytorch\test\inductor\test_compiled_optimizers.py`

```py
# Owner(s): ["module: inductor"]

# 引入系统、单元测试、弱引用和上下文管理模块
import sys
import unittest
import weakref
from contextlib import ExitStack

# 引入深拷贝和类型提示的命名元组
from copy import deepcopy
from typing import NamedTuple

# 引入PyTorch核心库
import torch

# 引入PyTorch的相关模块
import torch._inductor
import torch._inductor.cudagraph_trees
import torch.optim.lr_scheduler
from torch._inductor import config

# 引入自定义的测试用例
from torch._inductor.test_case import TestCase

# 引入各种优化器
from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    ASGD,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
    SparseAdam,
)

# 引入各种学习率调度器
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    StepLR,
)

# 引入用于设备类型的测试工具
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCUDAIf,
)

# 引入用于优化器的测试工具
from torch.testing._internal.common_optimizers import (
    _get_optim_inputs_including_global_cliquey_kwargs,
    optim_db,
    optims,
)

# 引入通用测试工具
from torch.testing._internal.common_utils import parametrize

# 引入PyTorch测试工具
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA, has_triton
from torch.testing._internal.triton_utils import requires_cuda


# 学习率调度器及其对应的参数字典
LR_SCHEDULER_TO_KWARGS = {
    LambdaLR: {"lr_lambda": lambda x: 10},  # LambdaLR的参数
    MultiplicativeLR: {"lr_lambda": lambda x: 10},  # MultiplicativeLR的参数
    StepLR: {"step_size": 1, "gamma": 100},  # StepLR的参数
    MultiStepLR: {"milestones": [1, 2], "gamma": 100},  # MultiStepLR的参数
    ExponentialLR: {"gamma": 100},  # ExponentialLR的参数
    CosineAnnealingLR: {"T_max": 7},  # CosineAnnealingLR的参数
    # 下面的调度器在eager模式下存在内存泄漏问题
    # SequentialLR: {"schedulers": None, "milestones": [1, 2]},
    # ChainedScheduler: {"schedulers": None},
    CyclicLR: {"base_lr": 0.001, "max_lr": 0.02, "cycle_momentum": False},  # CyclicLR的参数
    CosineAnnealingWarmRestarts: {"T_0": 1},  # CosineAnnealingWarmRestarts的参数
    OneCycleLR: {  # OneCycleLR的参数
        "max_lr": 0.02,
        "cycle_momentum": False,
        "steps_per_epoch": 1,
        "epochs": 10,
    },
    ConstantLR: {"factor": 0.001},  # ConstantLR的参数
    LinearLR: {},  # LinearLR的参数
    ReduceLROnPlateau: {"factor": 0.99, "patience": 1},  # ReduceLROnPlateau的参数
    PolynomialLR: {},  # PolynomialLR的参数
}


# 创建学习率调度器的函数
def create_scheduler(scheduler, optim):
    kwargs = LR_SCHEDULER_TO_KWARGS[scheduler]
    if "schedulers" in kwargs:
        kwargs["schedulers"] = [
            create_scheduler(torch.optim.lr_scheduler.ConstantLR, optim)
            for _ in range(2)
        ] + [create_scheduler(torch.optim.lr_scheduler.LambdaLR, optim)]

    if scheduler == ChainedScheduler:
        return scheduler(**kwargs)
    else:
        return scheduler(optim, **kwargs)


# 命名元组，用于记录多个和单个张量的内核数量
class KernelCounts(NamedTuple):
    multitensor: int
    singletensor: int


# 用于特定测试名称和预期内核数量之间的映射
KERNEL_COUNT_OVERRIDES = {
    "test_rmsprop_foreach_weight_decay_cpu": 12,
}
    {
        "test_nadam_foreach_weight_decay_momentum_decay_cpu": 20,  # 测试案例：nadam优化器，针对权重衰减和动量衰减，运行在CPU上，执行20次
        "test_adamw_amsgrad_capturable_foreach_cuda": 3,  # 测试案例：adamw优化器，amsgrad模式，可捕获，运行在CUDA上，执行3次
        "test_adamw_amsgrad_capturable_cuda": 6,  # 测试案例：adamw优化器，amsgrad模式，可捕获，运行在CUDA上，执行6次
        "test_adamw_tensor_lr_amsgrad_capturable_foreach_cuda": 3,  # 测试案例：adamw优化器，张量学习率，amsgrad模式，可捕获，运行在CUDA上，执行3次
        "test_adamw_tensor_lr_amsgrad_capturable_cuda": 6,  # 测试案例：adamw优化器，张量学习率，amsgrad模式，可捕获，运行在CUDA上，执行6次
        "test_adam_tensor_lr_amsgrad_capturable_cuda": 6,  # 测试案例：adam优化器，张量学习率，amsgrad模式，可捕获，运行在CUDA上，执行6次
        "test_adam_amsgrad_capturable_cuda": 6,  # 测试案例：adam优化器，amsgrad模式，可捕获，运行在CUDA上，执行6次
        "test_adadelta_tensor_lr_capturable_cuda": 6,  # 测试案例：adadelta优化器，张量学习率，可捕获，运行在CUDA上，执行6次
        "test_rmsprop_tensor_lr_capturable_cuda": 6,  # 测试案例：rmsprop优化器，张量学习率，可捕获，运行在CUDA上，执行6次
        "test_adadelta_tensor_lr_capturable_foreach_cuda": 4,  # 测试案例：adadelta优化器，张量学习率，可捕获，运行在CUDA上，执行4次
        "test_adadelta_foreach_weight_decay_maximize_cpu": 12,  # 测试案例：adadelta优化器，针对权重衰减和最大化，运行在CPU上，执行12次
        "test_adadelta_foreach_rho_weight_decay_cpu": 12,  # 测试案例：adadelta优化器，针对rho和权重衰减，运行在CPU上，执行12次
        "test_adadelta_foreach_weight_decay_cpu": 12,  # 测试案例：adadelta优化器，针对权重衰减，运行在CPU上，执行12次
        "test_sgd_foreach_momentum_weight_decay_cpu": 16,  # 测试案例：sgd优化器，针对动量和权重衰减，运行在CPU上，执行16次
        "test_sgd_foreach_momentum_nesterov_weight_decay_cpu": 16,  # 测试案例：sgd优化器，针对动量（nesterov）和权重衰减，运行在CPU上，执行16次
        "test_sgd_momentum_dampening_foreach_cuda": 5,  # 测试案例：sgd优化器，动量衰减，可捕获，运行在CUDA上，执行5次
        "test_sgd_momentum_foreach_cuda": 5,  # 测试案例：sgd优化器，动量，运行在CUDA上，执行5次
        "test_sgd_weight_decay_maximize_cuda": 4,  # 测试案例：sgd优化器，针对权重衰减和最大化，运行在CUDA上，执行4次
        "test_sgd_weight_decay_maximize_cpu": 4,  # 测试案例：sgd优化器，针对权重衰减和最大化，运行在CPU上，执行4次
        "test_sgd_momentum_weight_decay_foreach_cuda": 2,  # 测试案例：sgd优化器，动量和权重衰减，可捕获，运行在CUDA上，执行2次
        "test_sgd_momentum_nesterov_weight_decay_foreach_cuda": 2,  # 测试案例：sgd优化器，动量（nesterov）和权重衰减，可捕获，运行在CUDA上，执行2次
        "test_sgd_cuda": 4,  # 测试案例：sgd优化器，运行在CUDA上，执行4次
        "test_sgd_cpu": 4,  # 测试案例：sgd优化器，运行在CPU上，执行4次
        "test_rmsprop_tensor_lr_capturable_foreach_cuda": 4,  # 测试案例：rmsprop优化器，张量学习率，可捕获，运行在CUDA上，执行4次
        "test_adagrad_initial_accumulator_value_weight_decay_foreach_cuda": 3,  # 测试案例：adagrad优化器，初始累加器值和权重衰减，可捕获，运行在CUDA上，执行3次
        "test_adagrad_lr_decay_weight_decay_foreach_cuda": 3,  # 测试案例：adagrad优化器，学习率衰减和权重衰减，可捕获，运行在CUDA上，执行3次
        "test_adagrad_weight_decay_foreach_cuda": 3,  # 测试案例：adagrad优化器，权重衰减，可捕获，运行在CUDA上，执行3次
        "test_adagrad_weight_decay_maximize_foreach_cuda": 3,  # 测试案例：adagrad优化器，针对权重衰减和最大化，可捕获，运行在CUDA上，执行3次
        "test_adagrad_tensor_lr_cpu": 6,  # 测试案例：adagrad优化器，张量学习率，运行在CPU上，执行6次
        "test_adagrad_tensor_lr_cuda": 6,  # 测试案例：adagrad优化器，张量学习率，运行在CUDA上，执行6次
        "test_adamax_tensor_lr_weight_decay_capturable_cuda": 6,  # 测试案例：adamax优化器，张量学习率和权重衰减，可捕获，运行在CUDA上，执行6次
        "test_asgd_tensor_lr_weight_decay_maximize_capturable_cuda": 8,  # 测试案例：asgd优化器，张量学习率，权重衰减和最大化，可捕获，运行在CUDA上，执行8次
        "test_asgd_tensor_lr_weight_decay_maximize_capturable_foreach_cuda": 4,  # 测试案例：asgd优化器，张量学习率，权重衰减和最大化，可捕获，运行在CUDA上，执行4次
        "test_nadam_tensor_lr_weight_decay_momentum_decay_decoupled_weight_decay_capturable_cuda": 9,  # 测试案例：nadam优化器，张量学习率，权重衰减、动量衰减、分离权重衰减，可捕获，运行在CUDA上，执行9次
        "test_nadam_tensor_lr_weight_decay_momentum_decay_decoupled_weight_decay_capturable_foreach_cuda": 3,  # 测试案例：nadam优化器，张量学习率，权重衰减、动量衰减、分离权重衰减，可捕获，运行在CUDA上，执行3次
        "test_radam_tensor_lr_capturable_weight_decay_decoupled_weight_decay_cuda": 6,  # 测试案例：radam优化器，张量学习率，权重衰减和分离权重衰减，可捕获，运行在CUDA上，执行6次
        "test_radam_tensor_lr_capturable_weight_decay_decoupled_weight_decay_foreach_cuda": 3,
}

# also tracks currently supported optimizers
KERNEL_COUNTS = {
    Adam: KernelCounts(multitensor=2, singletensor=8),  # 定义不同优化器的核心计数，适用于不同的张量设备
    AdamW: KernelCounts(multitensor=2, singletensor=8),
    NAdam: KernelCounts(multitensor=2, singletensor=11),
    Rprop: KernelCounts(multitensor=2, singletensor=8),
    RMSprop: KernelCounts(multitensor=2, singletensor=8),
    Adadelta: KernelCounts(multitensor=2, singletensor=8),
    Adagrad: KernelCounts(multitensor=2, singletensor=8),
    SGD: KernelCounts(multitensor=1, singletensor=8),
    ASGD: KernelCounts(multitensor=2, singletensor=11),
    RAdam: KernelCounts(multitensor=2, singletensor=8),
    Adamax: KernelCounts(multitensor=2, singletensor=8),
}


def build_opt_kwarg_db():
    compiled_opt_db = []
    for optim_info in optim_db:  # 遍历优化器信息列表
        if optim_info.optim_cls not in KERNEL_COUNTS:  # 如果优化器不在核心计数中，则跳过
            continue

        for device in ["cpu", "cuda"]:  # 遍历设备类型
            for optim_inputs in _get_optim_inputs_including_global_cliquey_kwargs(
                device, None, optim_info, skip=("differentiable", "fused")
            ):  # 获取包括全局参数的优化器输入列表
                kwargs = dict(optim_inputs.kwargs)  # 复制优化器参数字典
                name = f"test_{optim_info.optim_cls.__name__.lower()}"  # 构建优化器测试名称

                has_tensor_lr = False  # 是否存在张量类型的学习率

                for key, val in kwargs.items():  # 遍历优化器参数
                    if not key == "lr" and (
                        not isinstance(val, bool) or (isinstance(val, bool) and val)
                    ):  # 如果参数不是学习率且不是布尔型或者是真值
                        name += "_" + key  # 将参数名称添加到测试名称中

                    if key == "lr" and isinstance(kwargs["lr"], torch.Tensor):  # 如果是学习率参数且是张量类型
                        has_tensor_lr = True  # 记录存在张量类型学习率的情况
                        name += "_tensor_lr"  # 在测试名称中添加张量学习率标记

                name += f"_{device}"  # 在测试名称末尾添加设备类型

                kwargs["device"] = device  # 设置优化器参数中的设备类型
                if name in KERNEL_COUNT_OVERRIDES:  # 如果测试名称在核心计数覆盖中
                    kwargs["kernel_count"] = KERNEL_COUNT_OVERRIDES[name]  # 使用覆盖的核心计数
                else:
                    kwargs["kernel_count"] = (
                        KERNEL_COUNTS[optim_info.optim_cls].multitensor
                        if kwargs.get("foreach", False) and device == "cuda"
                        else KERNEL_COUNTS[optim_info.optim_cls].singletensor
                    )  # 根据条件选择适当的核心计数

                if kwargs["kernel_count"] is None or kwargs.get("fused", False):  # 如果核心计数为None或者融合标志为真
                    continue  # 跳过当前循环

                if has_tensor_lr:  # 如果存在张量类型的学习率
                    for scheduler_cls in LR_SCHEDULER_TO_KWARGS.keys():  # 遍历学习率调度器类
                        name_w_scheduler = name + f"_{scheduler_cls.__name__.lower()}"  # 构建包含调度器名称的测试名称
                        compiled_opt_db.append(  # 将优化器信息和参数添加到编译的优化器数据库中
                            (
                                optim_info.optim_cls,
                                name_w_scheduler,
                                kwargs,
                                scheduler_cls,
                            )
                        )
                else:  # 否则
                    compiled_opt_db.append((optim_info.optim_cls, name, kwargs, None))  # 将优化器信息和参数添加到编译的优化器数据库中

    return compiled_opt_db  # 返回编译的优化器数据库


COMPILED_OPT_KWARG_DB = build_opt_kwarg_db()  # 调用函数构建并保存编译的优化器参数数据库

aten = torch.ops.aten  # 引用 torch 的 aten 操作模块
try:
    try:
        from .test_torchinductor import check_model, check_model_cuda
    except ImportError:
        from test_torchinductor import check_model, check_model_cuda
except (unittest.SkipTest, ImportError) as e:
    # 捕获 unittest.SkipTest 或 ImportError 异常，并输出到标准错误流
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        # 如果作为主程序执行，则退出进程，状态码为 0
        sys.exit(0)
    # 否则抛出异常
    raise


def call_scheduler(scheduler):
    # 如果 scheduler 是 ReduceLROnPlateau 类型的对象
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        # 调用 scheduler 的 step 方法，参数为 1.0，用于不降低指标的情况下更新学习率
        scheduler.step(1.0)
    else:
        # 否则直接调用 scheduler 的 step 方法，没有参数
        scheduler.step()


def compile_opt(opt_compiled, closure=None, fullgraph=True):
    # 运行补丁程序，确保 step 方法具有预期的结构
    torch._dynamo.eval_frame.TorchPatcher.patch()

    # 获取 step 方法的实际函数，避免由于 functionalization/no_grad 检测限制导致图形破坏
    # 参见 optimizer.py 中的 [Note on graph break]
    step_fn = opt_compiled.step.__wrapped__.__wrapped__

    # 标记 _opt_called 以避免 LR Scheduler 的警告
    opt_compiled._opt_called = True

    if closure is not None:
        # 如果提供了 closure，则定义一个包装函数 fn，调用 step_fn 时传入 opt_compiled 和 closure
        def fn():
            step_fn(opt_compiled, closure)
    else:
        # 否则定义一个包装函数 fn，调用 step_fn 时仅传入 opt_compiled
        def fn():
            step_fn(opt_compiled)

    # 编译并返回函数 fn，使用后端 "inductor"，是否使用完整图结构取决于 fullgraph 参数
    return torch.compile(fn, backend="inductor", fullgraph=fullgraph)


def check_optim(
    self,
    optim_cls,
    params_eager,
    params_compiled,
    state_eager,
    state_compiled,
    atol=None,
    rtol=None,
):
    # 将参数列表转换为列表形式
    params_eager = list(params_eager)
    params_compiled = list(params_compiled)
    
    # 设置公差参数 atol 和 rtol
    rtol = None
    atol = None
    
    # 如果 optim_cls 是 Adadelta 类型
    if optim_cls is Adadelta:
        # 设置 Adadelta 类型的默认 rtol 和 atol
        rtol = 5.5e-4
        atol = 5e-5

    # 使用 self.assertEqual 检查 params_eager 和 params_compiled 是否相等，使用指定的公差参数
    self.assertEqual(list(params_eager), list(params_compiled), atol=atol, rtol=rtol)

    # 遍历 params_eager 和 params_compiled，使用 self.assertEqual 检查 state_eager 和 state_compiled 对应参数是否相等
    for p_eager, p_compiled in zip(params_eager, params_compiled):
        self.assertEqual(
            state_eager[p_eager],
            state_compiled[p_compiled],
            atol=atol,
            rtol=rtol,
        )


def make_test(
    optim_cls,
    closure=None,
    scheduler_cls=None,
    kernel_count=2,
    device="cuda",
    **kwargs,
):
    def test_fn(self):
        # 使用 ExitStack 确保所有资源在函数结束时被正确关闭
        stack = ExitStack()
        try:
            # 检查是否需要在 CUDA 设备上运行计算图
            run_cudagraphs = device == "cuda" and optim_cls not in (Adagrad, SGD)
            if run_cudagraphs:
                # 如果需要在 CUDA 设备上运行计算图，则设置上下文使 triton.cudagraphs 为 True
                stack.enter_context(config.patch({"triton.cudagraphs": True}))

            # 复制 kwargs，确保不改变原始参数
            kwargs_compiled = deepcopy(kwargs)
            # 如果 lr 参数是 torch.Tensor 类型，则将其移到指定设备上
            if isinstance(kwargs.get("lr", None), torch.Tensor):
                kwargs["lr"] = kwargs["lr"].to(device)
                kwargs_compiled["lr"] = kwargs_compiled["lr"].to(device)

            # 重置 torch._dynamo 和 torch._inductor.metrics 的状态
            torch._dynamo.reset()
            torch._inductor.metrics.reset()

            # 创建输入张量，设备为指定设备
            input = torch.ones([10, 10], device=device)
            # 创建用于即时执行的模型，包含两个线性层
            model_eager = torch.nn.Sequential(
                *[torch.nn.Linear(10, 10, device=device) for _ in range(2)]
            )
            # 执行模型的前向传播、求和、反向传播
            model_eager(input).sum().backward()

            # 再次创建输入张量，设备为指定设备
            input = torch.ones([10, 10], device=device)
            # 深度复制即时执行模型，得到编译执行模型
            model_compiled = deepcopy(model_eager)
            # 执行编译模型的前向传播、求和、反向传播
            model_compiled(input).sum().backward()

            # 使用给定的优化器类创建即时执行模型的优化器
            opt_eager = optim_cls(model_eager.parameters(), **kwargs)
            # 使用给定的优化器类和编译参数创建编译执行模型的优化器
            opt_compiled = optim_cls(model_compiled.parameters(), **kwargs_compiled)
            # 编译执行优化器的单步优化操作
            compiled_step = compile_opt(opt_compiled, closure=closure)

            # 如果有调度器类存在，则创建即时执行和编译执行模型的调度器
            if scheduler_cls:
                scheduler_compiled = create_scheduler(scheduler_cls, opt_compiled)
                scheduler_eager = create_scheduler(scheduler_cls, opt_eager)
                # 有些调度器只有经过至少一个 epoch 后才会生效
                scheduler_compiled.last_epoch = 1
                scheduler_eager.last_epoch = 1

            # 关闭梯度计算
            with torch.set_grad_enabled(False):
                # 执行两次优化迭代
                for i in range(2):
                    compiled_step()  # 执行编译模型的单步优化
                    opt_eager.step()  # 执行即时执行模型的单步优化
                    # 如果有调度器类存在，则调用即时执行和编译执行模型的调度器
                    if scheduler_cls:
                        call_scheduler(scheduler_eager)
                        call_scheduler(scheduler_compiled)

            # 检查优化器状态，比较即时执行和编译执行模型的优化器状态
            check_optim(
                self,
                optim_cls,
                model_eager.parameters(),
                model_compiled.parameters(),
                opt_eager.state,
                opt_compiled.state,
            )

            # 如果需要在 CUDA 设备上运行计算图，则检查计算图是否运行
            if run_cudagraphs:
                self.check_cudagraphs_ran()

            # 如果需要检查内核计数，则进行比较
            if self.check_kernel_count:
                # 当前，我们分别编译单步和其余计算，因为单步是单个元素张量
                # 因此，通常的内核计数为 2
                self.assertEqual(
                    torch._inductor.metrics.generated_kernel_count, kernel_count
                )
        finally:
            # 关闭 ExitStack，确保所有资源被正确释放
            stack.close()

    # 如果设备为 CUDA，则将 test_fn 函数标记为需要 CUDA 的测试函数
    if device == "cuda":
        test_fn = requires_cuda(test_fn)

    # 返回标记后的测试函数
    return test_fn
# 创建一个用于测试优化器重新编译行为的测试函数生成器
def make_recompile_test(optim_cls, closure=None, kernel_count=2, **kwargs):
    # 装饰器，要求测试函数运行在 CUDA 环境下
    @requires_cuda
    def test_fn(self):
        # 重置动态图和归纳器指标
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        
        # 创建一个在 CUDA 设备上的输入张量
        input = torch.ones([10, 10], device="cuda")
        # 构建一个包含多个线性层的序列模型
        model = torch.nn.Sequential(
            *[torch.nn.Linear(10, 10, device="cuda") for _ in range(2)]
        )
        # 对模型进行前向传播、求和并反向传播
        model(input).sum().backward()

        # 使用给定的优化器类和参数初始化优化器
        opt_compiled = optim_cls(model.parameters(), **kwargs)
        # 编译优化器的步骤
        compiled_step = compile_opt(opt_compiled)

        # 检查这里没有重新编译
        with torch.set_grad_enabled(False):
            for _ in range(4):
                compiled_step()

            # 扰动状态以强制重新编译
            # Adagrad 在每个步骤不会重新初始化状态
            # SGD 的状态为空
            if optim_cls in (Adagrad, SGD):
                opt_compiled.param_groups[0]["lr"] = 0.02
            elif optim_cls is Adam:  # 确保我们在状态的数据指针上进行保护
                state_tensor = opt_compiled.state[
                    opt_compiled.param_groups[0]["params"][0]
                ]["exp_avg"]
                opt_compiled.state[opt_compiled.param_groups[0]["params"][0]][
                    "exp_avg"
                ] = torch.zeros_like(state_tensor)
            else:
                opt_compiled.state.clear()

            compiled_step()

        # 如果需要检查核心计数
        if self.check_kernel_count:
            # 目前，我们将步骤和其余计算分开编译
            # 因此，通常的核心计数是2
            # 乘以2以考虑重新编译
            multiplier = 2

            # 断言生成的核心数量是否符合预期的乘数
            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count,
                multiplier * kernel_count,
            )

    return test_fn


class CompiledOptimizerParityTests(TestCase):
    # 如果没有 Triton，则跳过 CUDA 环境下的测试
    @skipCUDAIf(not has_triton(), "torch.compile with cuda requires triton")
    @optims(optim_db, dtypes=[torch.float32])
    @parametrize("use_closure", [True, False])
class CompiledOptimizerTests(TestCase):
    check_model_cuda = check_model_cuda
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        super().setUp()
        # 重置动态图和归纳器指标
        torch._dynamo.reset()
        torch._inductor.metrics.reset()

    def tearDown(self):
        super().tearDown()
        # 重置动态图和归纳器指标
        torch._dynamo.reset()
        torch._inductor.metrics.reset()

    def check_cudagraphs_ran(self):
        # 目前我们运行第零个设备
        manager = torch._inductor.cudagraph_trees.get_container(0).tree_manager
        self.assertIsNotNone(manager)
        self.assertEqual(manager.new_graph_id().id, 1)

    # 为不同的优化器类型创建测试函数，测试其重新编译行为
    test_adam_recompile = make_recompile_test(Adam, lr=0.01)
    test_adamw_recompile = make_recompile_test(AdamW, lr=0.01)
    test_adamax_recompile = make_recompile_test(Adamax, lr=0.01)
    test_nadam_recompile = make_recompile_test(NAdam, lr=0.01)
    # 创建测试用例，使用 make_recompile_test 函数为 Rprop 优化器生成重新编译的测试
    test_rprop_recompile = make_recompile_test(Rprop, lr=0.01)
    # 创建测试用例，使用 make_recompile_test 函数为 RMSprop 优化器生成重新编译的测试
    test_rmsprop_recompile = make_recompile_test(RMSprop, lr=0.01)
    # 创建测试用例，使用 make_recompile_test 函数为 Adadelta 优化器生成重新编译的测试
    test_adadelta_recompile = make_recompile_test(Adadelta, lr=0.01)
    # 创建测试用例，使用 make_recompile_test 函数为 Adagrad 优化器生成重新编译的测试
    test_adagrad_recompile = make_recompile_test(Adagrad, lr=0.01)
    # 创建测试用例，使用 make_recompile_test 函数为 ASGD 优化器生成重新编译的测试（默认设置）
    test_asgd_recompile_default = make_recompile_test(ASGD, lr=0.01)
    # 创建测试用例，使用 make_recompile_test 函数为 ASGD 优化器生成重新编译的测试（自定义设置：kernel_count=11）
    test_asgd_recompile_single = make_recompile_test(
        ASGD, kernel_count=11, lr=0.01, foreach=False
    )
    # 创建测试用例，使用 make_recompile_test 函数为 ASGD 优化器生成重新编译的测试（设置 foreach=True）
    test_asgd_recompile_foreach = make_recompile_test(ASGD, lr=0.01, foreach=True)
    # 创建测试用例，使用 make_recompile_test 函数为 SGD 优化器生成重新编译的测试（自定义设置：kernel_count=4）
    test_sgd_recompile_single = make_recompile_test(
        SGD, kernel_count=4, lr=0.01, foreach=False
    )
    # 创建测试用例，使用 make_recompile_test 函数为 SGD 优化器生成重新编译的测试（设置 foreach=True）
    test_sgd_recompile_foreach = make_recompile_test(
        SGD, kernel_count=1, lr=0.01, foreach=True
    )

    @requires_cuda
    # 测试静态地址终结器功能
    def test_static_address_finalizer(self):
        import gc

        # 禁用垃圾回收器
        gc.disable()
        # 初始化弱引用对象
        p_ref = None

        # 定义内部函数 fn
        def fn():
            nonlocal p_ref
            # 创建一个在 CUDA 设备上的线性神经网络模型
            mod = torch.nn.Linear(10, 10, device="cuda:0", bias=False)
            # 为模型的参数创建随机梯度
            for p in mod.parameters():
                p.grad = torch.rand_like(p)

            # 使用 Adam 优化器来优化模型参数
            opt = torch.optim.Adam(mod.parameters(), lr=0.1)

            # 定义内部函数 fn
            def fn():
                opt.step()

            # 禁用梯度计算上下文
            with torch.set_grad_enabled(False):
                # 编译并执行步骤函数
                step_fn_compiled = torch.compile(fn)
                step_fn_compiled()

            # 创建 p 的弱引用
            p_ref = weakref.ref(p)
            # 断言 p 的弱引用不为 None
            self.assertTrue(p_ref() is not None)

        # 执行 fn 函数
        fn()

        # 断言 p 的弱引用为 None
        self.assertTrue(p_ref() is None)
        # 启用垃圾回收器
        gc.enable()

    # 测试在梯度为 None 时的保护行为
    def test_guard_on_none_grads(self):
        # 定义训练循环函数
        def training_loop():
            # 创建输入张量
            input = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(3, 2)

            # 创建神经网络模型
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 3),
                torch.nn.Sigmoid(),
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid(),
            )

            # 获取模型的参数列表
            params = list(model.parameters())
            # 创建 Adam 优化器
            optimizer = torch.optim.Adam(params)
            # 创建步骤列表
            step_list = []

            # 执行循环 6 次
            for i in range(6):
                optimizer.zero_grad()
                # 当梯度不为 None 时执行以下操作
                if i != 3:
                    output = model(input)
                    loss = output.sum()
                    loss.backward()

                # 执行优化器的步骤
                optimizer.step()
                # 将当前步骤数添加到步骤列表中
                step_list.append(optimizer.state[params[0]]["step"])

            return step_list

        # 编译训练循环函数
        compiled_training_loop = torch._dynamo.optimize("eager")(training_loop)
        # 执行编译后的训练循环函数
        actual_steps = compiled_training_loop()
        # 执行原始训练循环函数
        expected_steps = training_loop()
        # 断言编译后的步骤列表与原始步骤列表相等
        self.assertEqual(actual_steps, expected_steps)

    # 基本的测试用例，用于验证我们在不出错的情况下支持编译各种操作
    @requires_cuda
    def test_basic_shampoo(self):
        # 创建一个形状为 (1024, 128) 的随机张量作为参数缓冲区
        param_buf = torch.rand((1024, 128))
        # 克隆并分离参数缓冲区，形成一个新的张量
        param_buf_c = param_buf.clone().detach()

        # 将参数缓冲区分成两部分，并转置，存入列表 params_c
        params_c = [param_buf_c[0:512, :].t(), param_buf_c[512:, :].t()]
        # 将原始参数缓冲区也分成两部分，并转置，存入列表 params
        params = [param_buf[0:512, :].t(), param_buf[512:, :].t()]

        # 遍历 params 和 params_c 列表中的张量，并为每个张量设置随机梯度
        for p, p_c in zip(params, params_c):
            p.grad = torch.rand_like(p)
            # 克隆并分离原始张量的梯度，存入克隆张量的梯度
            p_c.grad = p.grad.clone().detach()

        # 注意：此处跳过了根逆因为它有很多内部依赖
        # 并且无论如何我们都不会编译它
        @torch.no_grad()
        def shampoo_functional_basic(params):
            step = 1
            weight_decay = 0.1
            # 收集所有参数的梯度到列表 grads 中
            grads = [p.grad for p in params]
            beta1 = 0.9
            beta2 = 1.0
            epsilon = 1e-10
            # 为每个参数创建与其形状相同的零张量，作为 preconditioners
            preconditioners = [torch.zeros_like(p) for p in params]
            lr = 0.01

            # pt2 region 1
            # 权重衰减，将每个参数的梯度加到其本身上
            torch._foreach_add_(grads, params, alpha=weight_decay)

            # 更新 preconditioners
            torch._foreach_addcmul_(preconditioners, grads, grads, value=1.0)

            # 对 grads 中的每个元素乘以 beta1
            torch._foreach_mul_(grads, beta1)
            # 对 grads 中的每个元素进行加权平均，使用 1 - beta1 作为权重
            torch._foreach_add_(
                grads,
                grads,
                alpha=1 - beta1,
            )
            # 计算偏置修正1
            bias_correction1 = 1.0 - beta1**step
            # 将 grads 列表中的每个元素除以 bias_correction1
            grad_list = torch._foreach_div(grads, bias_correction1)

            # pt2 region 2
            # 预条件化（使用 shampoo 分支），不进行嫁接
            bias_correction2 = 1.0 - beta2**step
            # 将 preconditioners 中的每个元素除以 bias_correction2
            bias_corrected_preconditioner_list = torch._foreach_div(
                preconditioners, bias_correction2
            )
            # 对 preconditioners 中的每个元素进行平方根操作
            torch._foreach_sqrt_(bias_corrected_preconditioner_list)
            # 对 preconditioners 中的每个元素加上 epsilon
            torch._foreach_add_(bias_corrected_preconditioner_list, epsilon)
            # 将 grad_list 中的每个元素除以 preconditioners 中对应元素
            search_directions = torch._foreach_div(
                grad_list, bias_corrected_preconditioner_list
            )

            # 将 search_directions 中的每个元素加到 params 中对应的元素上
            torch._foreach_add_(
                search_directions,
                params,
                alpha=weight_decay,
            )

            # 将 search_directions 中的每个元素乘以 -lr
            torch._foreach_mul_(search_directions, -lr)
            # pt2 region 3 更新 params
            torch._foreach_add_(params, search_directions)

            return params, preconditioners, grads

        # 编译 shampoo_functional_basic 函数
        compiled_fn = torch.compile(shampoo_functional_basic)

        # 断言编译后的函数返回的结果与未编译的函数的结果相等
        self.assertEqual(compiled_fn(params_c), shampoo_functional_basic(params))
    # 定义一个测试方法，用于测试闭包图中断的情况
    def test_closure_graph_break(self):
        # 创建一个具有指定属性的随机张量，存储在 CUDA 设备上，并标记需要计算梯度
        param = torch.rand(2, 3, dtype=torch.float32, device="cuda", requires_grad=True)
        # 克隆参数张量，分离计算图并标记需要计算梯度
        param_c = param.clone().detach().requires_grad_(True)

        # 定义闭包函数，设置 param 的梯度为全为2的张量，并返回该梯度
        def closure():
            param.grad = torch.ones_like(param) * 2
            return param.grad

        # 定义闭包函数，设置 param_c 的梯度为全为2的张量，并返回该梯度
        def closure_c():
            param_c.grad = torch.ones_like(param_c) * 2
            return param_c.grad

        # 使用 AdamW 优化器优化 param 参数
        optimizer = torch.optim.AdamW([param])
        # 使用 AdamW 优化器优化 param_c 参数
        optimizer_c = torch.optim.AdamW([param_c])

        # 定义一个循环函数，执行优化步骤
        def loop(opt, c):
            opt.step(c)

        # 使用 Torch._dynamo.optimize("eager") 对循环函数进行编译优化
        compiled_loop = torch._dynamo.optimize("eager")(loop)

        # 执行编译优化后的循环函数，传入 optimizer 和 closure 函数
        compiled_loop(optimizer, closure)
        # 直接调用未编译的循环函数，传入 optimizer_c 和 closure_c 函数
        loop(optimizer_c, closure_c)

        # 断言 param 和 param_c 参数应当相等
        self.assertEqual(param, param_c)

    # 测试从静态地址获取值的函数
    def test_get_value_on_static_address(self):
        # 导入所需模块和函数
        from torch._dynamo.decorators import mark_static_address
        from torch.optim.optimizer import _get_value

        # 编译 _get_value 函数
        compiled = torch.compile(_get_value)

        # 创建一个全为1的 2x2 张量 x
        x = torch.ones(2, 2)
        # 将 x 标记为静态地址
        mark_static_address(x)

        # 调用编译后的 _get_value 函数，获取 x 的返回值
        ret_val = compiled(x)

        # 断言返回值与 x 相等
        self.assertEqual(ret_val, x)

    # 编译一个大型 foreach 操作并验证所花费时间在预期范围内
    @requires_cuda
    def test_compile_time_smoketest(self):
        # 导入时间模块
        import time

        # 创建包含 100 个在 CUDA 设备上的全为1的 2x2 张量 xs 和 ys
        xs = [torch.ones(2, 2, device="cuda") for _ in range(100)]
        ys = [torch.ones(2, 2, device="cuda") for _ in range(100)]

        # 定义一个编译函数 fn，执行 _foreach_add 操作
        @torch.compile
        def fn(xs, ys):
            return torch._foreach_add(xs, ys)

        # 记录测试开始时间
        start = time.perf_counter()
        # 执行编译后的 fn 函数，传入 xs 和 ys
        fn(xs, ys)
        # 记录测试结束时间
        end = time.perf_counter()

        # 断言执行时间应当小于90秒
        self.assertLess(end - start, 90)
# 遍历 COMPILED_OPT_KWARG_DB 中的每个元素，元素结构为 (optim_cls, name, kwargs, scheduler_cls)
for optim_cls, name, kwargs, scheduler_cls in COMPILED_OPT_KWARG_DB:
    # 设置 CompiledOptimizerTests 类的属性，属性名为 name，值为 make_test 函数的返回值
    setattr(
        CompiledOptimizerTests,
        name,
        make_test(optim_cls, scheduler_cls=scheduler_cls, **kwargs),
    )

# 调用 instantiate_device_type_tests 函数，为 CompiledOptimizerParityTests 类生成设备类型相关的测试用例
instantiate_device_type_tests(CompiledOptimizerParityTests, globals())

# 如果脚本作为主程序运行
if __name__ == "__main__":
    # 从 torch._inductor.test_case 模块导入 run_tests 函数
    from torch._inductor.test_case import run_tests

    # 如果当前系统支持 CPU 或 CUDA
    if HAS_CPU or HAS_CUDA:
        # 运行测试，指定需要 "filelock" 支持
        run_tests(needs="filelock")
```