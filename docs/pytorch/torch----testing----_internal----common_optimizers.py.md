# `.\pytorch\torch\testing\_internal\common_optimizers.py`

```
# mypy: ignore-errors

# 导入必要的模块和类
import functools  # 导入 functools 模块
import itertools  # 导入 itertools 模块
import sys  # 导入 sys 模块
import unittest  # 导入 unittest 模块
from copy import deepcopy  # 从 copy 模块导入 deepcopy 函数
from enum import Enum  # 导入 Enum 类
from typing import Any, Dict, List, Tuple, Union  # 导入必要的类型定义

import torch  # 导入 torch 库
from torch import Tensor  # 从 torch 导入 Tensor 类型
from torch.nn import Parameter  # 从 torch.nn 导入 Parameter 类
from torch.optim import (  # 导入各种优化器类
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    ASGD,
    LBFGS,
    NAdam,
    Optimizer,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
    SparseAdam,
)
from torch.optim.lr_scheduler import (  # 导入学习率调度器类
    ConstantLR,
    ExponentialLR,
    LinearLR,
    PolynomialLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.testing._internal.common_device_type import tol, toleranceOverride  # 导入测试相关模块
from torch.testing._internal.common_methods_invocations import DecorateInfo  # 导入测试相关模块
from torch.testing._internal.common_utils import (  # 导入测试相关模块
    _TestParametrizer,
    skipIfMps,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
)
from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices  # 导入测试相关模块

# 定义优化器输入类，包含参数、关键字参数和描述信息
class OptimizerInput:
    """Contains args / kwargs to be passed to an optimizer constructor."""

    __slots__ = ["params", "kwargs", "desc"]

    def __init__(
        self,
        params: Union[List[Parameter], List[Tensor], Dict[Any, Any]],
        kwargs: Dict[str, Any],
        desc: str = "",
    ):
        # params 可以是 Tensor 列表、参数组列表或空列表
        self.params = params
        self.kwargs = kwargs
        self.desc = desc

    def __repr__(self):
        return f"params={self.params}, kwargs={self.kwargs}, desc={self.desc}"


# 定义优化器错误枚举类，列举测试优化器时可能出现的错误
class OptimizerErrorEnum(Enum):
    """Enumerates when an error is raised when testing optimizers."""

    CONSTRUCTION_ERROR = 0  # 构造优化器时出现错误
    STEP_ERROR = 1  # 执行优化器步骤时出现错误


# 定义错误优化器输入类，用于测试时构造会引发错误的优化器输入
class ErrorOptimizerInput:
    """
    An OptimizerInput that will cause the optimizer to throw an error when constructed.
    Includes the type and string of the resulting error.
    """

    __slots__ = ["optimizer_error_input", "error_on", "error_type", "error_regex"]

    def __init__(
        self,
        optimizer_error_input,
        *,
        error_on=OptimizerErrorEnum.CONSTRUCTION_ERROR,
        error_type=RuntimeError,
        error_regex="",
    ):
        self.optimizer_error_input = optimizer_error_input
        self.error_on = error_on
        self.error_type = error_type
        self.error_regex = error_regex


# 定义优化器信息类，用于测试时存储和传递优化器相关信息
class OptimizerInfo:
    """Optimizer information to be used in testing."""
    def __init__(
        self,
        optim_cls: Optimizer,  # Class object for the Optimizer under test
        *,
        optim_inputs_func,  # Function to generate optimizer inputs EXCLUDING params. We delegate params responsibility
                            # to the test using the OptimizerInfo. OptimizerInput.params is likely None.
                            # Can optionally take in device to filter out certain unsupported configs
        scheduler_inputs=(
            [
                lambda opt: StepLR(opt, gamma=0.9, step_size=10),  # Lambda function to create StepLR scheduler with specified parameters
                lambda opt: ReduceLROnPlateau(opt),  # Lambda function to create ReduceLROnPlateau scheduler
            ],
        ),
        supported_impls: Tuple[str] = ("foreach", "differentiable"),  # A subset of the global-cliquey flags (fused, foreach, differentiable) the optimizer
                                                                       # supports. See NOTE: [optimizer kwarg categories] for what global-cliquey means.
        supports_sparse: bool = False,  # The optim supports passing in sparse gradients as well as dense grads
        only_supports_sparse_grads: bool = False,  # The optim only supports one config: sparse grads w/ dense params, see SparseAdam
        metadata_for_sparse=({}, []),  # Tuple of (optimizer kwargs, schedulers_constructors) specifically for sparse tests,
                                       # with especially tuned hyperparameters. These only apply if the optimizer supports
                                       # sparse parameters or grads.
        supports_complex: bool = True,  # The optim supports complex parameters
        step_requires_closure: bool = False,  # Whether the optimizer.step() function requires a closure to be passed
        supports_param_groups: bool = True,  # Whether the optimizer supports per-param options with parameter groups
        supports_multiple_devices: bool = True,  # Whether the optimizer supports parameters on multiple devices
        skips=(),  # Indicates which tests to skip
        decorators=None,  # Additional decorators to apply to generated tests
        optim_error_inputs_func=None,  # Function to generate optim inputs that error
        supports_fused_on: Tuple[str] = (),  # Tuple of supported fused options
    ):
        self.optim_cls = optim_cls
        self.optim_inputs_func = optim_inputs_func
        self.scheduler_inputs = scheduler_inputs
        self.supported_impls = supported_impls
        self.supports_sparse = supports_sparse
        self.metadata_for_sparse = metadata_for_sparse
        self.only_supports_sparse_grads = only_supports_sparse_grads
        self.supports_complex = supports_complex
        self.step_requires_closure = step_requires_closure
        self.supports_param_groups = supports_param_groups
        self.supports_multiple_devices = supports_multiple_devices
        self.decorators = (
            *(decorators if decorators else []),
            *(skips if skips else []),
        )
        self.optim_error_inputs_func = optim_error_inputs_func
        self.supports_fused_on = supports_fused_on


        self.optim_cls = optim_cls
        # 设置优化器类的属性

        self.optim_inputs_func = optim_inputs_func
        # 设置优化器输入函数的属性

        self.scheduler_inputs = scheduler_inputs
        # 设置调度器输入的属性

        self.supported_impls = supported_impls
        # 设置支持的实现方式的属性

        self.supports_sparse = supports_sparse
        # 设置是否支持稀疏张量的属性

        self.metadata_for_sparse = metadata_for_sparse
        # 设置稀疏张量元数据的属性

        self.only_supports_sparse_grads = only_supports_sparse_grads
        # 设置是否仅支持稀疏梯度的属性

        self.supports_complex = supports_complex
        # 设置是否支持复杂类型的属性

        self.step_requires_closure = step_requires_closure
        # 设置步骤是否需要闭包的属性

        self.supports_param_groups = supports_param_groups
        # 设置是否支持参数组的属性

        self.supports_multiple_devices = supports_multiple_devices
        # 设置是否支持多设备的属性

        self.decorators = (
            *(decorators if decorators else []),
            *(skips if skips else []),
        )
        # 设置装饰器的元组属性，包括给定的装饰器和跳过装饰器

        self.optim_error_inputs_func = optim_error_inputs_func
        # 设置优化器错误输入函数的属性

        self.supports_fused_on = supports_fused_on
        # 设置是否支持融合操作的属性


    def get_decorators(self, test_class, test_name, device, dtype, param_kwargs):
        result = []
        for decorator in self.decorators:
            if isinstance(decorator, DecorateInfo):
                if decorator.is_active(
                    test_class, test_name, device, dtype, param_kwargs
                ):
                    result.extend(decorator.decorators)
            else:
                result.append(decorator)
        return result


        # 获取给定测试类、测试名称、设备、数据类型和参数关键字的装饰器列表
        result = []
        # 初始化结果列表
        for decorator in self.decorators:
            # 遍历所有装饰器
            if isinstance(decorator, DecorateInfo):
                # 如果装饰器是DecorateInfo类型的实例
                if decorator.is_active(
                    test_class, test_name, device, dtype, param_kwargs
                ):
                    # 如果装饰器处于活动状态
                    result.extend(decorator.decorators)
                    # 将装饰器的decorators列表扩展到结果列表中
            else:
                result.append(decorator)
                # 否则直接将装饰器添加到结果列表中
        return result
        # 返回最终的装饰器列表作为结果


    @property
    def name(self):
        return self.optim_cls.__name__


        # 返回优化器类的名称作为属性值
        return self.optim_cls.__name__
class optims(_TestParametrizer):
    """Decorator for specifying a list of optimizers over which to run a test."""

    def __init__(self, optim_info_iterable, dtypes=None):
        # 初始化方法，接收一个可迭代的优化器信息列表
        self.optim_info_list = list(optim_info_iterable)

        # 设置优化器的数据类型，默认为 torch.float32，可以通过参数指定
        # 参数可以具有不同的数据类型，不限于一个数据类型
        self.dtypes = dtypes if dtypes is not None else [torch.float32]

    def _parametrize_test(self, test, generic_cls, device_cls):
        if device_cls is None:
            # 如果 device_cls 为 None，则抛出运行时错误，@optims 装饰器应在特定设备上下文中使用
            raise RuntimeError(
                "The @optims decorator is only intended to be used in a device-specific "
                "context; use it with instantiate_device_type_tests() instead of "
                "instantiate_parametrized_tests()"
            )

        # 遍历优化器信息列表和数据类型，生成测试参数
        for optim_info, dtype in itertools.product(self.optim_info_list, self.dtypes):
            # 构造测试名称，设备/数据类型部分在外部处理
            test_name = optim_info.name

            # 构造参数 kwargs 以传递给测试函数
            param_kwargs = {"optim_info": optim_info, "dtype": dtype}

            try:
                # 定义测试函数的包装器，保留原始测试函数的签名和文档字符串
                @functools.wraps(test)
                def test_wrapper(*args, **kwargs):
                    return test(*args, **kwargs)

                # 构造装饰器函数，用于获取优化器的修饰器
                decorator_fn = functools.partial(
                    optim_info.get_decorators,
                    generic_cls.__name__,
                    test.__name__,
                    device_cls.device_type,
                    dtype,
                )

                # 返回生成的测试函数、测试名称、参数 kwargs 和装饰器函数
                yield (test_wrapper, test_name, param_kwargs, decorator_fn)
            except Exception as ex:
                # 在重新抛出异常之前，提供用于调试的错误消息
                print(
                    f"Failed to instantiate {test_name} for module {optim_info.name}!"
                )
                raise ex


# 以下是一个辅助函数，用于生成所有优化器的错误输入，将在下面使用
def get_error_inputs_for_all_optims(device, dtype):
    # 实现函数的具体逻辑，未在此处提供
    # 如果设备是 CPU
    if str(device) == "cpu":
        # 创建一个随机张量作为示例参数，并指定设备和数据类型
        sample_param = Parameter(torch.randn(1, device=device, dtype=dtype))
        # 返回包含三个错误对象的列表，每个对象描述不同的优化器输入错误情况

        return [
            # 第一个错误对象：参数类型无效的错误
            ErrorOptimizerInput(
                OptimizerInput(
                    params=sample_param,
                    kwargs={},
                    desc="invalid param type",
                ),
                error_type=TypeError,
                error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
            ),

            # 第二个错误对象：参数组中包含重复参数的警告
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[sample_param, sample_param],
                    kwargs={},
                    desc="a param group cannot have duplicate parameters",
                ),
                error_type=UserWarning,
                error_regex=".*a parameter group with duplicate parameters.*",
            ),

            # 第三个错误对象：跨参数组存在重复参数的值错误
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[{"params": sample_param}, {"params": sample_param}],
                    kwargs={},
                    desc="duplicate parameters should not occur across param groups either",
                ),
                error_type=ValueError,
                error_regex="some parameters appear in more than one parameter group",
            ),
        ]
    else:
        # 如果设备不是 CPU，返回空列表
        return []
# ------------------------------------------------------------------------------------------
# NOTE: [optimizer kwarg categories]
# We categorize optimizer kwargs as 3 types:
#  1. optimizer-specific flags are like amsgrad or rho or beta, flags that are specific to
#     algorithms and thus only show up for certain optimizers. There are many of these, so I
#     do not bother gathering them all and listing them here. The converse to these would be
#     global flags that every optimizer ideally _should_ support. We break global flags into
#     2 further categories and list them all below.
#  2. global-friendly = ["lr", "weight_decay", "maximize", "capturable"]
#     global-friendly flags are global flags who play nicely with all other global flags,
#     i.e., are mutually exclusive in function. This means that any pair of the following
#     flags can be toggled at once (e.g., maximize and weight_decay). Furthermore, any of the
#     following flags theoretically can be enabled with ANY other global flag, including the
#     cliquey ones (e.g, capturable and foreach).
#  3. global-cliquey = ["foreach", "fused", "differentiable"]
#     global-cliquey flags are global flags that do NOT coexist with other cliquey flags,
#     usually because they contradict each other in function. For example, one should not flip
#     both foreach AND fused to True, because they are two differing performance optimizations
#     in which you can only opt into one.
#
# The following optim_inputs_func_* sampling functions only return constructor combinations of
# optimizer-specific and global-friendly flags. This is because we are confident they would mesh
# well with additional kwargs. On the flip side of the same coin, we reserve setting the
# global-cliquey flags to individual tests and fully expect tests to edit OptimizerInput.kwargs.

def optim_inputs_func_adadelta(device, dtype=None):
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "capturable": True},
            desc="capturable with weight decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={"lr": torch.tensor(0.001), "capturable": True},
            desc="Tensor lr with capturable",
        ),
    ]

    # Default configurations for optimizer inputs, including global-friendly flags
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 0.01}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
        OptimizerInput(
            params=None, kwargs={"rho": 0.95, "weight_decay": 0.9}, desc="rho"
        ),
    ] + (cuda_supported_configs if "cuda" in str(device) else [])
# 为给定设备和数据类型获取所有优化器的错误输入
def optim_error_inputs_func_adadelta(device, dtype):
    # 调用函数获取所有优化器的错误输入
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    # 如果设备为 CPU
    if str(device) == "cpu":
        # 添加一个错误优化器输入到错误输入列表中，用于测试 rho 参数不在有效范围内的情况
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, rho=1.1),
                    desc="rho should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid rho value: 1.1",
            ),
        ]
    # 返回所有错误输入列表
    return error_inputs


# 返回用于 Adam 优化器的不同输入配置
def optim_inputs_func_adagrad(device, dtype=None):
    return [
        # 默认配置的优化器输入
        OptimizerInput(params=None, kwargs={}, desc="default"),
        # 带有非零 weight_decay 的优化器输入
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        # 带有 maximize=True 的优化器输入
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
        # 非默认 lr=0.1 的优化器输入
        OptimizerInput(params=None, kwargs={"lr": 0.1}, desc="non-default lr"),
        # 带有 initial_accumulator_value 参数的优化器输入
        OptimizerInput(
            params=None,
            kwargs={"initial_accumulator_value": 0.1, "weight_decay": 0.1},
            desc="initial_accumulator_value",
        ),
        # 带有 lr_decay 参数的优化器输入
        OptimizerInput(
            params=None,
            kwargs={"lr": 0.1, "lr_decay": 0.5, "weight_decay": 0.1},
            desc="lr_decay",
        ),  # TODO: Move out to testing in param_group?
        # 使用 Tensor 类型的 lr 参数的优化器输入
        OptimizerInput(
            params=None,
            kwargs={"lr": torch.tensor(0.001)},
            desc="Tensor lr",
        ),
    ]


# 为给定设备和数据类型获取所有优化器的错误输入
def optim_error_inputs_func_adagrad(device, dtype):
    # 调用函数获取所有优化器的错误输入
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    # 如果设备为 CPU
    if str(device) == "cpu":
        # 添加一个错误优化器输入到错误输入列表中，用于测试 lr_decay 参数小于零的情况
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, lr_decay=-0.5),
                    desc="lr_decay must be bigger than 0",
                ),
                error_type=ValueError,
                error_regex="Invalid lr_decay value: -0.5",
            ),
        ]
    # 返回所有错误输入列表
    return error_inputs


# 返回用于 Adam 优化器的不同输入配置
# TODO: consider tensor LR! See multi_tensor_optimizer_configs in test_optim.py --> tensor LR should work
# with all implementation code paths...
def optim_inputs_func_adam(device, dtype=None):
    # 支持 CUDA 的配置列表
    cuda_supported_configs = [
        # 支持 capturable 的优化器输入
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        # 支持 capturable 和 amsgrad 的优化器输入
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "amsgrad": True, "capturable": True},
            desc="capturable, amsgrad",
        ),
        # 使用 Tensor 类型的 lr 参数，并支持 capturable 和 amsgrad 的优化器输入
        OptimizerInput(
            params=None,
            kwargs={"lr": torch.tensor(0.001), "amsgrad": True, "capturable": True},
            desc="Tensor lr with capturable and amsgrad",
        ),
    ]
    # 创建一个包含多个 OptimizerInput 对象的列表 total，每个对象都有不同的参数和描述
    total = [
        OptimizerInput(params=None, kwargs={}, desc="default"),  # 默认参数设置
        OptimizerInput(params=None, kwargs={"lr": 0.01}, desc="non-default lr"),  # 设置学习率 lr=0.01
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"  # 设置权重衰减 weight_decay=0.1
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",  # 启用参数最大化操作
        ),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1, "amsgrad": True}, desc="amsgrad"  # 启用 AMSGrad 优化器
        ),
    ] + (cuda_supported_configs if "cuda" in str(device) else [])  # 如果设备支持 CUDA，则添加 cuda_supported_configs 到 total 中

    # 如果 dtype 是 torch.float16 类型
    if dtype in (torch.float16,):
        # 对于 total 列表中的每个 OptimizerInput 对象
        for input in total:
            """
            过小的 eps 值会导致在低精度 dtype 下 denom 为零
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            例如，
            >>> a
            tensor([0.], dtype=torch.float16)
            >>> a + 1e-8
            tensor([0.], dtype=torch.float16)
            """
            # 将 eps 参数设置为 0.1，以避免在低精度 dtype 下出现除零情况
            input.kwargs["eps"] = 0.1

    # 返回填充好参数的 total 列表
    return total
# 根据指定设备和数据类型获取优化器错误输入列表
def optim_error_inputs_func_adam(device, dtype):
    # 调用函数获取适用于所有优化器的错误输入列表
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    
    # 如果设备是 CPU
    if str(device) == "cpu":
        # 添加错误优化器输入：beta1 参数应在 0 和 1 之间
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(1.0, 0.0)),
                    desc="beta1 should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid beta parameter at index 0: 1.0",
            ),
            # 添加错误优化器输入：weight_decay 应大于 0
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, weight_decay=-1),
                    desc="weight_decay should > 0",
                ),
                error_type=ValueError,
                error_regex="Invalid weight_decay value: -1",
            ),
            # 添加错误优化器输入：使用 Tensor 作为 lr 参数不支持 foreach 且不可捕获
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=torch.tensor(0.001), foreach=True),
                    desc="lr as Tensor doesn't work with foreach & not capturable",
                ),
                error_type=ValueError,
                error_regex="lr as a Tensor is not supported for capturable=False and foreach=True",
            ),
        ]
    
    # 如果设备是 CUDA 加速器
    if "cuda" in str(device):
        # 创建一个空的 Tensor 作为示例
        sample_tensor = torch.empty((), device=device, dtype=dtype)
        # 添加错误优化器输入：`fused` 和 `foreach` 不能同时为真
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[sample_tensor],
                    kwargs={"foreach": True, "fused": True},
                    desc="`fused` and `foreach` cannot be `True` together",
                ),
                error_type=RuntimeError,
                error_regex="`fused` and `foreach` cannot be `True` together",
            ),
            # 添加错误优化器输入：`fused` 不支持 `differentiable`
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[sample_tensor],
                    kwargs={"fused": True, "differentiable": True},
                    desc="`fused` does not support `differentiable`",
                ),
                error_type=RuntimeError,
                error_regex="`fused` does not support `differentiable`",
            ),
        ]
    
    # 返回最终的错误优化器输入列表
    return error_inputs


def optim_inputs_func_adamax(device, dtype=None):
    # 定义一个包含多个 OptimizerInput 对象的列表，用于存储不同的优化器配置
    cuda_supported_configs = [
        # 创建一个 OptimizerInput 对象，参数为 None，关键字参数包括 capturable=True，描述为 capturable
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        # 创建一个 OptimizerInput 对象，参数为 None，关键字参数包括 weight_decay=0.9, maximize=True, capturable=True，描述为 capturable, maximize, weight_decay
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.9, "maximize": True, "capturable": True},
            desc="capturable, maximize, weight_decay",
        ),
        # 创建一个 OptimizerInput 对象，参数为 None，关键字参数包括 weight_decay=0, maximize=True, capturable=True，描述为 capturable, maximize
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0, "maximize": True, "capturable": True},
            desc="capturable, maximize",
        ),
        # 创建一个 OptimizerInput 对象，参数为 None，关键字参数包括 weight_decay=0.9, maximize=False, capturable=True，描述为 capturable, weight_decay
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.9, "maximize": False, "capturable": True},
            desc="capturable, weight_decay",
        ),
        # 创建一个 OptimizerInput 对象，参数为 None，关键字参数包括 lr=torch.tensor(0.001), weight_decay=0.9, maximize=False, capturable=True，描述为 capturable, weight_decay, tensor LR
        OptimizerInput(
            params=None,
            kwargs={
                "lr": torch.tensor(0.001),
                "weight_decay": 0.9,
                "maximize": False,
                "capturable": True,
            },
            desc="capturable, weight_decay, tensor LR",
        ),
    ]

    # 返回一个包含多个 OptimizerInput 对象的列表，表示不同优化器配置的默认设置
    return [
        # 创建一个 OptimizerInput 对象，参数为 None，关键字参数为空字典，描述为 default
        OptimizerInput(params=None, kwargs={}, desc="default"),
        # 创建一个 OptimizerInput 对象，参数为 None，关键字参数包括 lr=0.1，描述为 non-default lr
        OptimizerInput(params=None, kwargs={"lr": 0.1}, desc="non-default lr"),
        # 创建一个 OptimizerInput 对象，参数为 None，关键字参数包括 weight_decay=0.1，描述为 nonzero weight_decay
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        # 创建一个 OptimizerInput 对象，参数为 None，关键字参数包括 weight_decay=0.1, maximize=True，描述为 maximize
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
    ] + (cuda_supported_configs if "cuda" in str(device) else [])
    # 如果设备名称中包含字符串 "cuda"，则将 cuda_supported_configs 列表添加到返回列表中，否则返回空列表
# 生成所有优化器错误输入的函数，基于Adamax优化器
def optim_error_inputs_func_adamax(device, dtype):
    # 调用函数获取所有优化器的错误输入
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    # 如果设备为CPU，则添加特定的错误输入
    if str(device) == "cpu":
        error_inputs += [
            # 错误的优化器输入：指定了不合法的beta参数范围
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(0.0, 1.0)),
                    desc="beta2 should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid beta parameter at index 1: 1.0",
            ),
        ]
    # 返回所有错误输入列表
    return error_inputs


# 生成AdamW优化器的输入参数的函数，调用Adam优化器函数
def optim_inputs_func_adamw(device, dtype=None):
    return optim_inputs_func_adam(device, dtype)


# 生成所有AdamW优化器错误输入的函数，调用Adam优化器函数
def optim_error_inputs_func_adamw(device, dtype):
    return optim_error_inputs_func_adam(device, dtype)


# 生成Averaged Stochastic Gradient Descent (ASGD)优化器的输入参数的函数
def optim_inputs_func_asgd(device, dtype=None):
    cuda_supported_configs = [
        # 支持CUDA的配置：可捕获的参数设置
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        # 支持CUDA的配置：最大化和可捕获的参数设置
        OptimizerInput(
            params=None,
            kwargs={"maximize": True, "capturable": True},
            desc="maximize, capturable",
        ),
        # 支持CUDA的配置：权重衰减和可捕获的参数设置
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "capturable": True},
            desc="weight_decay, capturable",
        ),
        # 支持CUDA的配置：最大化、权重衰减和可捕获的参数设置
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True, "capturable": True},
            desc="maximize, weight_decay, capturable",
        ),
        # 支持CUDA的配置：使用张量作为学习率、最大化、权重衰减和可捕获的参数设置
        OptimizerInput(
            params=None,
            kwargs={
                "lr": torch.tensor(0.001),
                "weight_decay": 0.1,
                "maximize": True,
                "capturable": True,
            },
            desc="maximize, weight_decay, capturable, tensor LR",
        ),
    ]
    # 返回ASGD优化器的所有输入参数列表，包括默认和CUDA支持的配置
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lambd": 0.1}, desc="non-default lambd"),
        OptimizerInput(params=None, kwargs={"lr": 0.02}, desc="non-default lr"),
        OptimizerInput(params=None, kwargs={"t0": 100}, desc="t0"),
        OptimizerInput(params=None, kwargs={"maximize": True}, desc="maximize"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize, nonzero weight_decay",
        ),
    ] + (cuda_supported_configs if "cuda" in str(device) else [])


# 生成所有ASGD优化器错误输入的函数，基于ASGD优化器
def optim_error_inputs_func_asgd(device, dtype):
    # 调用函数获取所有优化器的错误输入
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    # 如果设备是 CPU，则执行以下逻辑，将错误输入列表 error_inputs 添加一个元素
    if str(device) == "cpu":
        error_inputs += [
            # 创建 ErrorOptimizerInput 对象，包含以下参数：
            ErrorOptimizerInput(
                # OptimizerInput 对象，参数为 None
                OptimizerInput(
                    params=None,
                    # kwargs 参数包含学习率 lr 为 0.01，权重衰减 weight_decay 为 -0.5
                    kwargs=dict(lr=1e-2, weight_decay=-0.5),
                    # 描述信息说明权重衰减应该大于 0
                    desc="weight_decay should > 0",
                ),
                # 错误类型为 ValueError
                error_type=ValueError,
                # 错误正则表达式为指定权重衰减值无效的错误信息
                error_regex="Invalid weight_decay value: -0.5",
            ),
        ]
    # 返回错误输入列表 error_inputs
    return error_inputs
# 返回 LBFGS 优化器的输入配置列表
def optim_inputs_func_lbfgs(device, dtype=None):
    return [
        # 默认配置，不传入任何参数
        OptimizerInput(params=None, kwargs={}, desc="default"),
        # 设置学习率为 0.01 的非默认配置
        OptimizerInput(params=None, kwargs={"lr": 0.01}, desc="non-default lr"),
        # 设置 tolerance_grad 为 1e-6 的配置
        OptimizerInput(
            params=None, kwargs={"tolerance_grad": 1e-6}, desc="tolerance_grad"
        ),
        # 设置 line_search_fn 为 "strong_wolfe" 的配置
        OptimizerInput(
            params=None,
            kwargs={"line_search_fn": "strong_wolfe"},
            desc="strong_wolfe",
        ),
    ]


# 返回针对 LBFGS 优化器的错误输入配置
def optim_error_inputs_func_lbfgs(device, dtype):
    # 获取适用于所有优化器的错误输入配置
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    return error_inputs


# 返回 Nadam 优化器的输入配置列表
def optim_inputs_func_nadam(device, dtype=None):
    cuda_supported_configs = [
        # 支持 CUDA 的配置，capturable 设置为 True
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        # 支持 CUDA 的配置，设置 weight_decay 和 momentum_decay，capturable 设置为 True
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.9, "momentum_decay": 6e-3, "capturable": True},
            desc="weight_decay, capturable",
        ),
        # 支持 CUDA 的配置，设置 weight_decay、momentum_decay、decoupled_weight_decay，capturable 设置为 True
        OptimizerInput(
            params=None,
            kwargs={
                "weight_decay": 0.9,
                "momentum_decay": 6e-3,
                "decoupled_weight_decay": True,
                "capturable": True,
            },
            desc="decoupled_weight_decay, capturable",
        ),
        # 支持 CUDA 的配置，设置 lr、weight_decay、momentum_decay、decoupled_weight_decay，capturable 设置为 True
        OptimizerInput(
            params=None,
            kwargs={
                "lr": torch.tensor(0.001),
                "weight_decay": 0.9,
                "momentum_decay": 6e-3,
                "decoupled_weight_decay": True,
                "capturable": True,
            },
            desc="decoupled_weight_decay, capturable",
        ),
    ]
    # 返回 Nadam 优化器的默认配置和一些额外配置，如果设备支持 CUDA 则添加 cuda_supported_configs
    return [
        # 默认配置，不传入任何参数
        OptimizerInput(params=None, kwargs={}, desc="default"),
        # 设置学习率为 1e-3 的非默认配置
        OptimizerInput(params=None, kwargs={"lr": 1e-3}, desc="non-default lr"),
        # 设置 momentum_decay 为 6e-3 的非零配置
        OptimizerInput(
            params=None,
            kwargs={"momentum_decay": 6e-3},
            desc="non-zero momentum_decay",
        ),
        # 设置 weight_decay 为 0.1 和 momentum_decay 为 6e-3 的配置
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "momentum_decay": 6e-3},
            desc="weight_decay",
        ),
        # 设置 weight_decay 为 0.1、decoupled_weight_decay 为 True 的配置
        OptimizerInput(
            params=None,
            kwargs={
                "weight_decay": 0.1,
                "momentum_decay": 6e-3,
                "decoupled_weight_decay": True,
            },
            desc="decoupled_weight_decay",
        ),
        # 设置 weight_decay 为 0.1、maximize 为 True 的配置
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
    ] + (cuda_supported_configs if "cuda" in str(device) else [])


# 返回针对 Nadam 优化器的错误输入配置
def optim_error_inputs_func_nadam(device, dtype):
    # 获取适用于所有优化器的错误输入配置
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    return error_inputs
    # 如果设备为 CPU，则执行以下代码块
    if str(device) == "cpu":
        # 向错误输入列表中添加一个错误优化器输入对象
        error_inputs += [
            ErrorOptimizerInput(
                # 创建优化器输入对象，参数为 None
                OptimizerInput(
                    params=None,
                    # 关键字参数设置为学习率为 0.01，beta 参数为 (1.0, 0.0)
                    kwargs=dict(lr=1e-2, betas=(1.0, 0.0)),
                    # 描述信息指出 beta1 应在 0 到 1 之间
                    desc="beta1 should be between 0 and 1",
                ),
                # 错误类型设为 ValueError
                error_type=ValueError,
                # 错误正则表达式指出 beta 参数错误：1.0 超出范围
                error_regex="Invalid beta parameter at index 0: 1.0",
            ),
            # 向错误输入列表中添加另一个错误优化器输入对象
            ErrorOptimizerInput(
                # 创建优化器输入对象，参数为 None
                OptimizerInput(
                    params=None,
                    # 关键字参数设置为学习率为 0.01，动量衰减为 -0.2
                    kwargs=dict(lr=1e-2, momentum_decay=-0.2),
                    # 描述信息指出动量衰减应大于 0
                    desc="momentum_decay should > 0",
                ),
                # 错误类型设为 ValueError
                error_type=ValueError,
                # 错误正则表达式指出动量衰减值错误：-0.2 小于 0
                error_regex="Invalid momentum_decay value: -0.2",
            ),
        ]
    # 返回错误输入列表
    return error_inputs
# 定义一个函数，生成用于 RAdam 优化器的输入配置列表
def optim_inputs_func_radam(device=None, dtype=None):
    # CUDA 支持的配置列表
    cuda_supported_configs = [
        # 创建 OptimizerInput 实例，参数为 None，kwargs 包含 {"capturable": True}，描述为 "capturable"
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        # 创建 OptimizerInput 实例，参数为 None，kwargs 包含 {"capturable": True, "weight_decay": 0.1}，描述为 "capturable, weight_decay"
        OptimizerInput(
            params=None,
            kwargs={
                "capturable": True,
                "weight_decay": 0.1,
            },
            desc="capturable, weight_decay",
        ),
        # 创建 OptimizerInput 实例，参数为 None，kwargs 包含 {"capturable": True, "weight_decay": 0.1, "decoupled_weight_decay": True}，描述为 "capturable, weight_decay, decoupled_weight_decay"
        OptimizerInput(
            params=None,
            kwargs={
                "capturable": True,
                "weight_decay": 0.1,
                "decoupled_weight_decay": True,
            },
            desc="capturable, weight_decay, decoupled_weight_decay",
        ),
        # 创建 OptimizerInput 实例，参数为 None，kwargs 包含 {"lr": torch.tensor(0.001), "capturable": True, "weight_decay": 0.1, "decoupled_weight_decay": True}，描述为 "capturable, weight_decay, decoupled_weight_decay, tensor LR"
        OptimizerInput(
            params=None,
            kwargs={
                "lr": torch.tensor(0.001),
                "capturable": True,
                "weight_decay": 0.1,
                "decoupled_weight_decay": True,
            },
            desc="capturable, weight_decay, decoupled_weight_decay, tensor LR",
        ),
    ]
    # 返回默认的输入配置列表，以及如果设备包含 "cuda" 字符串，则返回 CUDA 支持的配置列表
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 2e-3}, desc="non-default lr"),
        OptimizerInput(params=None, kwargs={"eps": 1e-6}, desc="non-default eps"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "decoupled_weight_decay": True},
            desc="decoupled_weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
    ] + (cuda_supported_configs if "cuda" in str(device) else [])


# 定义一个函数，生成 RAdam 优化器的错误输入配置列表
def optim_error_inputs_func_radam(device, dtype):
    # 调用函数获取适用于所有优化器的错误输入列表
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    # 如果设备是 "cpu"，则添加特定于 RAdam 优化器的错误输入配置
    if str(device) == "cpu":
        error_inputs += [
            # 创建 ErrorOptimizerInput 实例，包含一个 OptimizerInput 实例，kwargs 包含 {"lr": 1e-2, "betas": (1.0, 0.0)}，描述为 "beta1 should be between 0 and 1"
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(1.0, 0.0)),
                    desc="beta1 should be between 0 and 1",
                ),
                error_type=ValueError,  # 错误类型为 ValueError
                error_regex="Invalid beta parameter at index 0: 1.0",  # 错误正则表达式匹配字符串
            ),
            # 创建 ErrorOptimizerInput 实例，包含一个 OptimizerInput 实例，kwargs 包含 {"lr": 1e-2, "weight_decay": -1}，描述为 "weight_decay should > 0"
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, weight_decay=-1),
                    desc="weight_decay should > 0",
                ),
                error_type=ValueError,  # 错误类型为 ValueError
                error_regex="Invalid weight_decay value: -1",  # 错误正则表达式匹配字符串
            ),
        ]
    # 返回所有的错误输入配置列表
    return error_inputs


# 定义一个函数，开始创建 RMSprop 优化器的输入配置列表
def optim_inputs_func_rmsprop(device, dtype=None):
    # 定义一个列表，包含了多个 OptimizerInput 对象，用于配置优化器的参数
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True, "capturable": True},
            desc="capturable, maximize",
        ),
        OptimizerInput(
            params=None,
            kwargs={"lr": torch.tensor(0.001), "capturable": True},
            desc="Tensor lr with capturable",
        ),
    ]

    # 返回一个列表，包含多个 OptimizerInput 对象，根据不同配置描述不同的优化器参数设置
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),  # 默认配置
        OptimizerInput(params=None, kwargs={"lr": 1e-3}, desc="non-default lr"),  # 设置了非默认的学习率
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"  # 设置了非零的 weight_decay
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "centered": True},
            desc="centered",  # 同时设置了 weight_decay 和 centered
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "centered": True, "momentum": 0.1},
            desc="momentum",  # 同时设置了 weight_decay、centered 和 momentum
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "weight_decay": 0.1,
                "centered": True,
                "momentum": 0.1,
                "maximize": True,
            },
            desc="maximize",  # 同时设置了 weight_decay、centered、momentum 和 maximize
        ),
    ] + (cuda_supported_configs if "cuda" in str(device) else [])  # 如果设备名称中包含 "cuda"，则添加 cuda_supported_configs 到返回的列表中
# 根据指定设备和数据类型获取所有优化器的错误输入配置列表
def optim_error_inputs_func_rmsprop(device, dtype):
    # 调用函数获取所有优化器的错误输入配置列表
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    # 如果设备为 CPU，则添加一个错误配置到 error_inputs 列表中
    if str(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, momentum=-1.0),
                    desc="momentum should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid momentum value: -1.0",
            ),
        ]
    # 返回错误输入配置列表
    return error_inputs


# 根据指定设备获取优化器输入配置列表，根据设备是否支持 CUDA 添加不同的配置
def optim_inputs_func_rprop(device, dtype=None):
    # CUDA 支持的配置列表
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={"lr": torch.tensor(0.001), "capturable": True},
            desc="Tensor lr with capturable",
        ),
    ]

    # 返回包含默认及根据设备是否为 CUDA 添加的优化器输入配置列表
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 2e-4}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"etas": (0.5, 1.5)}, desc="non-default etas"
        ),
        OptimizerInput(
            params=None,
            kwargs={"step_sizes": (2e-6, 100)},
            desc="non-default step_sizes",
        ),
        OptimizerInput(params=None, kwargs={"maximize": True}, desc="maximize"),
    ] + (cuda_supported_configs if "cuda" in str(device) else [])


# 根据指定设备和数据类型获取所有优化器的错误输入配置列表
def optim_error_inputs_func_rprop(device, dtype):
    # 调用函数获取所有优化器的错误输入配置列表
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    # 如果设备为 CPU，则添加一个错误配置到 error_inputs 列表中
    if str(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, etas=(1.0, 0.5)),
                    desc="0 < eta1 < 1 < eta2",
                ),
                error_type=ValueError,
                error_regex="Invalid eta values: 1.0, 0.5",
            ),
        ]
    # 返回错误输入配置列表
    return error_inputs


# 根据指定设备获取 SGD 优化器的输入配置列表，根据设备是否支持 CUDA 添加不同的配置
def optim_inputs_func_sgd(device, dtype=None):
    # 返回一个包含多个 OptimizerInput 对象的列表，每个对象都有特定的参数和描述
    return [
        # 创建一个默认参数的 OptimizerInput 对象，kwargs 是空字典，描述为 "default"
        OptimizerInput(params=None, kwargs={}, desc="default"),
        # 创建一个具有非默认学习率参数的 OptimizerInput 对象，学习率为 1e-2，描述为 "non-default lr"
        OptimizerInput(params=None, kwargs={"lr": 1e-2}, desc="non-default lr"),
        # 创建一个具有张量类型学习率参数的 OptimizerInput 对象，学习率为 0.001，描述为 "tensor lr"
        OptimizerInput(
            params=None, kwargs={"lr": torch.tensor(0.001)}, desc="tensor lr"
        ),
        # 创建一个具有动量参数的 OptimizerInput 对象，动量为 0.9，描述为 "momentum"
        OptimizerInput(params=None, kwargs={"momentum": 0.9}, desc="momentum"),
        # 创建一个具有动量和阻尼参数的 OptimizerInput 对象，动量为 0.9，阻尼为 0.5，描述为 "dampening"
        OptimizerInput(
            params=None,
            kwargs={"momentum": 0.9, "dampening": 0.5},
            desc="dampening",
        ),
        # 创建一个具有动量和权重衰减参数的 OptimizerInput 对象，动量为 0.9，权重衰减为 0.1，描述为 "non-zero weight_decay"
        OptimizerInput(
            params=None,
            kwargs={"momentum": 0.9, "weight_decay": 0.1},
            desc="non-zero weight_decay",
        ),
        # 创建一个具有动量、权重衰减和 Nesterov 参数的 OptimizerInput 对象，动量为 0.9，权重衰减为 0.1，启用 Nesterov，描述为 "nesterov"
        OptimizerInput(
            params=None,
            kwargs={"momentum": 0.9, "nesterov": True, "weight_decay": 0.1},
            desc="nesterov",
        ),
        # 创建一个具有权重衰减和最大化标记参数的 OptimizerInput 对象，权重衰减为 0.1，启用最大化，描述为 "maximize"
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
    ]
def optim_error_inputs_func_sgd(device, dtype):
    # 调用函数获取所有优化器的错误输入
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    # 如果设备是 CPU
    if str(device) == "cpu":
        # 添加一个错误优化器输入对象到错误输入列表
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, momentum=-0.5),
                    desc="momentum should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid momentum value: -0.5",
            ),
        ]
    # 返回错误输入列表
    return error_inputs


def optim_inputs_func_sparseadam(device, dtype=None):
    # 返回一个包含多个优化器输入对象的列表
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(
            params=None, kwargs={"lr": 0.01}, desc="non-default lr"
        ),  # TODO: Move out to testing in param_group?
        OptimizerInput(params=None, kwargs={"maximize": True}, desc="maximize"),
    ]


def optim_error_inputs_func_sparseadam(device, dtype):
    # 调用函数获取所有优化器的错误输入
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    # 返回错误输入列表
    return error_inputs
    # 检查设备是否为 CPU，将错误输入列表初始化为一个空列表
    if str(device) == "cpu":
        # 向错误输入列表添加第一个错误优化器输入对象
        error_inputs += [
            ErrorOptimizerInput(
                # 使用空参数，指定学习率和 beta 参数范围的字典
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(1.0, 0.0)),
                    desc="beta1 should be between 0 and 1",
                ),
                # 指定错误类型为 ValueError
                error_type=ValueError,
                # 指定错误正则表达式匹配文本
                error_regex="Invalid beta parameter at index 0: 1.0",
            ),
            # 向错误输入列表添加第二个错误优化器输入对象
            ErrorOptimizerInput(
                # 使用包含一个稀疏张量的参数列表
                OptimizerInput(
                    params=[
                        torch.zeros(
                            3, layout=torch.sparse_coo, device=device, dtype=dtype
                        )
                    ],
                    kwargs={},
                    desc="dense params required",
                ),
                # 指定错误类型为 ValueError
                error_type=ValueError,
                # 指定错误正则表达式匹配文本
                error_regex="SparseAdam requires dense parameter tensors",
            ),
            # 向错误输入列表添加第三个错误优化器输入对象
            ErrorOptimizerInput(
                # 使用包含一个包含稀疏张量的字典参数的参数列表
                OptimizerInput(
                    params=[
                        {
                            "params": [
                                torch.zeros(
                                    3,
                                    layout=torch.sparse_coo,
                                    device=device,
                                    dtype=dtype,
                                )
                            ]
                        }
                    ],
                    kwargs={},
                    desc="dense params required in param_groups",
                ),
                # 指定错误类型为 ValueError
                error_type=ValueError,
                # 指定错误正则表达式匹配文本
                error_regex="SparseAdam requires dense parameter tensors",
            ),
            # 向错误输入列表添加第四个错误优化器输入对象
            ErrorOptimizerInput(
                # 使用包含一个复杂张量的参数列表
                OptimizerInput(
                    params=[torch.rand(2, 3, device=device, dtype=torch.complex64)],
                    kwargs=dict(),
                    desc="complex not supported",
                ),
                # 指定错误类型为 ValueError
                error_type=ValueError,
                # 指定错误正则表达式匹配文本
                error_regex="SparseAdam does not support complex parameters",
            ),
        ]
    # 返回错误输入列表
    return error_inputs
# 返回设备类型的字符串表示，例如 "cpu" 或 "cuda"
def _get_device_type(device: Union[str, torch.device]) -> str:
    if isinstance(device, torch.device):
        # 如果 device 是 torch.device 类型，则将其转换为字符串类型
        device = str(device.type)
    assert isinstance(device, str)
    return device.split(":")[0]  # 返回设备类型的主要部分，去除可能存在的冒号后面的内容


def _get_optim_inputs_including_global_cliquey_kwargs(
    device, dtype, optim_info, skip=()
) -> List[OptimizerInput]:
    """
    返回一个给定优化器的所有配置的 OptimizerInput 列表，
    包括支持的全局 cliquey kwargs（例如 foreach、fused、differentiable），基于 optim_info.supported_impls。

    optim_info.optim_inputs_func(...) 返回的配置 (optim_inputs) 故意不包括全局 cliquey kwargs，
    以便在测试中提供灵活性。例如，轻松测试开启和关闭 foreach 的正确性。
    但有时我们希望测试优化器的所有可能配置，包括所有支持的标志，因此此辅助函数返回所有优化器输入。
    """
    assert all(
        x in ["foreach", "fused", "differentiable"] for x in skip
    ), "skip 必须是 ['foreach', 'fused', 'differentiable'] 的子集"

    optim_inputs = optim_info.optim_inputs_func(device)

    supported_impls = tuple(
        x
        for x in optim_info.supported_impls
        if x not in skip
        and (_get_device_type(device) in optim_info.supports_fused_on or x != "fused")
        and (
            _get_device_type(device) in _get_foreach_kernels_supported_devices()
            or x != "foreach"
        )
    )

    all_optim_inputs = []
    for optim_input in optim_inputs:
        # 添加所有标志都为 False 的基本配置
        base_kwargs = deepcopy(optim_input.kwargs)
        if len(supported_impls) != 0:
            for flag in supported_impls:
                base_kwargs[flag] = False
            all_optim_inputs.append(
                OptimizerInput(params=None, kwargs=base_kwargs, desc=optim_input.desc)
            )
        else:
            all_optim_inputs.append(optim_input)
        # 添加每个全局 cliquey kwargs 为 True 的配置
        # 注意，在 [optimizer kwarg categories] 中，这些 kwargs 是互斥的，因此我们不需要将它们一起使用。
        for flag in supported_impls:
            new_kwargs = deepcopy(base_kwargs)
            new_kwargs[flag] = True
            all_optim_inputs.append(
                OptimizerInput(
                    params=None, kwargs=new_kwargs, desc=f"{optim_input.desc} & {flag}"
                )
            )
    return all_optim_inputs


# 优化器信息条目的数据库，按字母顺序排列。
optim_db: List[OptimizerInfo] = [
    # 创建一个 OptimizerInfo 对象，指定优化器类型为 Adadelta
    OptimizerInfo(
        Adadelta,
        # 指定优化器的输入函数
        optim_inputs_func=optim_inputs_func_adadelta,
        # 指定优化器错误输入的函数
        optim_error_inputs_func=optim_error_inputs_func_adadelta,
        # 指定支持的实现方式
        supported_impls=("foreach", "differentiable"),
        # 定义跳过测试的条件和说明
        skips=(
            DecorateInfo(
                # 如果在 Python 3.8 上失败了固定点断言，参见 issue #97811
                skipIfTorchDynamo("Fails fix point assertion on 3.8, see #97811"),
                "TestOptimRenewed",
                "test_tensor_lr",
                active_if=sys.version_info < (3, 9) and sys.version_info > (3, 7),
            ),
            DecorateInfo(
                # 参见 issue #116028
                skipIfTorchDynamo("See #116028"),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            DecorateInfo(
                # 访问 grad.real 时出现错误，参见 https://github.com/pytorch/pytorch/issues/117184
                skipIfTorchDynamo(
                    "Accessing grad.real errors, see https://github.com/pytorch/pytorch/issues/117184"
                ),
                "TestOptimRenewed",
                "test_complex_2d",
            ),
            # 关于容差的注意事项：
            # test_correctness_Adadelta_cuda_float32
            # 不匹配的元素：10 / 100 (10.0%)
            # 最大的绝对差：4.838220775127411e-05 在索引 (7, 4) 处（允许的最大差为 1e-05）
            # 最大的相对差：0.007270356640219688 在索引 (7, 2) 处（允许的最大差为 1e-05）
            # 这是由于浮点数排序误差和 sqrt 的使用导致的
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.float32: tol(
                            rtol=5.5e-4,
                            atol=5e-5,
                        )
                    }
                ),
                "CompiledOptimizerParityTests",
                "test_correctness",
            ),
            DecorateInfo(
                # 这个测试使用了模拟对象，而 dynamo 不支持
                skipIfTorchDynamo(
                    "This test uses mocks, which dynamo does not support"
                ),
                "TestOptimRenewed",
                "test_defaults_changed_to_foreach",
            ),
        ),
    ),
    OptimizerInfo(
        Ad`
    OptimizerInfo(
        Adagrad,
        optim_inputs_func=optim_inputs_func_adagrad,
        optim_error_inputs_func=optim_error_inputs_func_adagrad,
        supported_impls=("foreach", "differentiable", "fused"),
        supports_fused_on=("cpu",),
        supports_sparse=True,
        metadata_for_sparse=(
            {"lr": 0.1, "weight_decay": 0, "lr_decay": 0},
            [
                lambda opt: StepLR(opt, gamma=1 - 1e-5, step_size=500),
                lambda opt: ReduceLROnPlateau(opt, threshold=1e-4),
            ],
        ),
        decorators=(
            DecorateInfo(
                # Note on tolerances:
                # difference comes from the fact that the non fused kernel have
                # more dtype cast operations. We have another test test_fused_cpu_matches_cuda
                # to make sure there is no discrepancies between cuda fused kernel
                # and cpu fused kernel
                toleranceOverride(
                    {
                        torch.bfloat16: tol(atol=5e-3, rtol=5e-3),
                        torch.float16: tol(atol=5e-3, rtol=5e-3),
                    }
                ),
                "TestOptimRenewed",
                "test_fused_matches_forloop",
            ),
        ),
        skips=(
            DecorateInfo(
                skipIfMps,  # addcdiv doesn't work for non-contiguous, see #118115
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
                active_if=lambda kwargs: not kwargs["contiguous"],
            ),
            DecorateInfo(
                skipIfTorchDynamo("Fails fix point assertion on 3.8, see #97811"),
                "TestOptimRenewed",
                "test_tensor_lr",
                active_if=sys.version_info < (3, 9) and sys.version_info > (3, 7),
            ),
            DecorateInfo(
                skipIfTorchDynamo("See #116028"),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Accessing grad.real errors, see https://github.com/pytorch/pytorch/issues/117184"
                ),
                "TestOptimRenewed",
                "test_complex_2d",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "This test uses mocks, which dynamo does not support"
                ),
                "TestOptimRenewed",
                "test_defaults_changed_to_foreach",
            ),
        ),
    ),


注释：


# 创建一个 OptimizerInfo 实例，指定优化器为 Adagrad
OptimizerInfo(
    Adagrad,
    optim_inputs_func=optim_inputs_func_adagrad,  # 设置优化器输入函数
    optim_error_inputs_func=optim_error_inputs_func_adagrad,  # 设置优化器错误输入函数
    supported_impls=("foreach", "differentiable", "fused"),  # 支持的实现方式，包括 foreach、differentiable 和 fused
    supports_fused_on=("cpu",),  # 支持在 CPU 上进行 fused 操作
    supports_sparse=True,  # 支持稀疏张量
    metadata_for_sparse=(  # 稀疏张量的元数据
        {"lr": 0.1, "weight_decay": 0, "lr_decay": 0},  # 学习率、权重衰减率和学习率衰减率
        [  # 元数据中包含的回调函数列表
            lambda opt: StepLR(opt, gamma=1 - 1e-5, step_size=500),  # StepLR 调度器
            lambda opt: ReduceLROnPlateau(opt,
    OptimizerInfo(
        Adamax,  # 使用 Adamax 优化器
        optim_inputs_func=optim_inputs_func_adamax,  # 设置 Adamax 优化器的输入函数
        optim_error_inputs_func=optim_error_inputs_func_adamax,  # 设置 Adamax 优化器的错误输入函数
        supported_impls=("foreach", "differentiable"),  # 支持的实现方式，包括 foreach 和 differentiable
        skips=(
            DecorateInfo(
                skipIfMps,  # 如果非连续性，addcdiv 无法工作，详见 issue #118115
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
                active_if=lambda kwargs: not kwargs["contiguous"],  # 如果 kwargs 中的 contiguous 为 False 则激活
            ),
            DecorateInfo(
                skipIfTorchDynamo("Fails fix point assertion on 3.8, see #97811"),  # 在 Python 3.8 上无法通过固定点断言，详见 issue #97811
                "TestOptimRenewed",
                "test_tensor_lr",
                active_if=sys.version_info < (3, 9) and sys.version_info > (3, 7),  # 如果 Python 版本在 3.7 到 3.8 之间，则激活
            ),
            DecorateInfo(
                skipIfTorchDynamo("See #116028"),  # 参考 issue #116028
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Accessing grad.real errors, see https://github.com/pytorch/pytorch/issues/117184"
                ),  # 访问 grad.real 时出现错误，参见 issue https://github.com/pytorch/pytorch/issues/117184
                "TestOptimRenewed",
                "test_complex_2d",
            ),
            DecorateInfo(
                unittest.skip("Uses too much memory, even for H100, surprisingly."),  # 使用太多内存，即使对于 H100 也是如此，令人惊讶
                "TestOptimRenewed",
                "test_foreach_large_tensor",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "This test uses mocks, which dynamo does not support"
                ),  # 此测试使用模拟，但 Dynamo 不支持
                "TestOptimRenewed",
                "test_defaults_changed_to_foreach",
            ),
        ),
    ),
    ),
    OptimizerInfo(
        ASGD,  # 使用 ASGD 优化器
        optim_inputs_func=optim_inputs_func_asgd,  # 设置 ASGD 优化器的输入函数
        optim_error_inputs_func=optim_error_inputs_func_asgd,  # 设置 ASGD 优化器的错误输入函数
        supported_impls=("foreach", "differentiable"),  # 支持的实现方式为 foreach 和 differentiable
        skips=(  # 跳过以下测试
            DecorateInfo(
                skipIfTorchDynamo("Fails fix point assertion on 3.8, see #97811"),  # 如果在 Python 3.8 上固定点断言失败，则跳过测试
                "TestOptimRenewed",  # 测试类名
                "test_tensor_lr",  # 测试方法名
                active_if=sys.version_info < (3, 9) and sys.version_info > (3, 7),  # 仅在 Python 版本小于 3.9 且大于 3.7 时激活
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors w/ Global state changed, see https://github.com/pytorch/pytorch/issues/116028"
                ),  # 如果出现全局状态变化错误，则跳过测试
                "TestOptimRenewed",  # 测试类名
                "test_set_default_dtype_works_with_foreach",  # 测试方法名
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Accessing grad.real errors, see https://github.com/pytorch/pytorch/issues/117184"
                ),  # 如果访问 grad.real 出错，则跳过测试
                "TestOptimRenewed",  # 测试类名
                "test_complex_2d",  # 测试方法名
            ),
            DecorateInfo(
                toleranceOverride(  # 设置容差覆盖
                    {
                        torch.float32: tol(atol=1.5e-5, rtol=1e-5),  # 对于 torch.float32 类型设置特定的容差
                    }
                ),
                "TestOptimRenewed",  # 测试类名
                "test_step_is_noop_for_zero_grads",  # 测试方法名
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "This test uses mocks, which dynamo does not support"
                ),  # 如果测试使用模拟对象，而 Dynamo 不支持，则跳过测试
                "TestOptimRenewed",  # 测试类名
                "test_defaults_changed_to_foreach",  # 测试方法名
            ),
            DecorateInfo(
                unittest.skip(
                    "ASGD internally changes the weights even with zero grad"
                ),  # 使用 unittest.skip 标记，说明 ASGD 在零梯度情况下仍会内部改变权重
                "TestOptimRenewed",  # 测试类名
                "test_step_is_noop_for_zero_grads",  # 测试方法名
            ),
        ),
    ),
    OptimizerInfo(
        LBFGS,  # 使用 LBFGS 优化器进行配置
        optim_inputs_func=optim_inputs_func_lbfgs,  # 设置 LBFGS 优化器的输入函数
        optim_error_inputs_func=optim_error_inputs_func_lbfgs,  # 设置 LBFGS 优化器的错误输入函数
        supported_impls=(),  # 不支持任何特定的实现
        step_requires_closure=True,  # 每步优化需要封闭包
        supports_param_groups=False,  # 不支持参数组
        supports_multiple_devices=False,  # 不支持多设备
        skips=(  # 需要跳过的测试装饰器列表
            # 在 MacOS 13.2.1 上失败，详细信息参见 GitHub 上的问题 #117094
            DecorateInfo(
                skipIfMps,  # 跳过条件：MPS 模式下运行
                "TestOptimRenewed",  # 测试类名称
                "test_can_load_older_state_dict"  # 测试方法名称
            ),
            DecorateInfo(
                toleranceOverride(  # 设置容差覆盖
                    {
                        torch.complex64: tol(  # 对于 torch.complex64 类型的容差设置
                            rtol=4.5e-5,  # 相对容差
                            atol=5e-5,    # 绝对容差
                        )
                    }
                ),
                "TestOptimRenewed",  # 测试类名称
                "test_complex_2d"  # 测试方法名称
            ),
            DecorateInfo(
                unittest.skip("Does not support param groups"),  # 标记为不支持参数组的测试跳过
                "TestOptimRenewed",  # 测试类名称
                "test_param_groups_lr"  # 测试方法名称
            ),
            DecorateInfo(
                unittest.skip("Does not support param groups"),  # 标记为不支持参数组的测试跳过
                "TestOptimRenewed",  # 测试类名称
                "test_param_groups_weight_decay"  # 测试方法名称
            ),
            DecorateInfo(
                unittest.skip("LBFGS doesn't support multidevice"),  # 标记为 LBFGS 不支持多设备的测试跳过
                "TestOptimRenewed",  # 测试类名称
                "test_forloop_goes_right_direction_multigpu"  # 测试方法名称
            ),
            DecorateInfo(
                unittest.skip("Does not support param groups"),  # 标记为不支持参数组的测试跳过
                "TestOptimRenewed",  # 测试类名称
                "test_param_group_with_lrscheduler_goes_right_direction"  # 测试方法名称
            ),
            DecorateInfo(
                skipIfTorchDynamo("Fails fix point assertion on 3.8, see #97811"),  # 在 Torch Dynamo 模式下运行的测试跳过
                "TestOptimRenewed",  # 测试类名称
                "test_tensor_lr",  # 测试方法名称
                active_if=sys.version_info < (3, 9) and sys.version_info > (3, 7),  # 激活条件：Python 版本在 3.7 到 3.8 之间
            ),
        ),
    ),
    OptimizerInfo(
        NAdam,  # 使用 NAdam 优化器
        optim_inputs_func=optim_inputs_func_nadam,  # 设置优化器输入函数为 optim_inputs_func_nadam
        optim_error_inputs_func=optim_error_inputs_func_nadam,  # 设置优化器错误输入函数为 optim_error_inputs_func_nadam
        supported_impls=("foreach", "differentiable"),  # 支持的实现方式为 foreach 和 differentiable
        skips=(  # 设置跳过的测试用例列表
            DecorateInfo(
                skipIfMps,  # 如果不是连续的，则跳过（addcdiv 对非连续数据无效，参见 issue #118115）
                "TestOptimRenewed",  # 所属测试类为 TestOptimRenewed
                "test_forloop_goes_right_direction",  # 测试方法为 test_forloop_goes_right_direction
                active_if=lambda kwargs: not kwargs["contiguous"],  # 当参数 kwargs 中的 contiguous 不为 True 时激活
            ),
            DecorateInfo(
                skipIfTorchDynamo("Fails fix point assertion on 3.8, see #97811"),  # 如果在 Python 3.8 下固定点断言失败，参见 issue #97811，则跳过
                "TestOptimRenewed",  # 所属测试类为 TestOptimRenewed
                "test_tensor_lr",  # 测试方法为 test_tensor_lr
                active_if=sys.version_info < (3, 9) and sys.version_info > (3, 7),  # 当 Python 版本在 3.7 到 3.8 之间时激活
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors w/ Global state changed, see https://github.com/pytorch/pytorch/issues/116028"
                ),  # 如果出现全局状态更改的错误，参见 issue https://github.com/pytorch/pytorch/issues/116028，则跳过
                "TestOptimRenewed",  # 所属测试类为 TestOptimRenewed
                "test_set_default_dtype_works_with_foreach",  # 测试方法为 test_set_default_dtype_works_with_foreach
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Accessing grad.real errors, see https://github.com/pytorch/pytorch/issues/117184"
                ),  # 如果访问 grad.real 出现错误，参见 issue https://github.com/pytorch/pytorch/issues/117184，则跳过
                "TestOptimRenewed",  # 所属测试类为 TestOptimRenewed
                "test_complex_2d",  # 测试方法为 test_complex_2d
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors, https://github.com/pytorch/pytorch/issues/117150"
                ),  # 如果出现错误，参见 issue https://github.com/pytorch/pytorch/issues/117150，则跳过
                "TestOptimRenewed",  # 所属测试类为 TestOptimRenewed
                "test_load_nontensor_step",  # 测试方法为 test_load_nontensor_step
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "This test uses mocks, which dynamo does not support"
                ),  # 如果测试使用了模拟对象，而 dynamo 不支持模拟对象，则跳过
                "TestOptimRenewed",  # 所属测试类为 TestOptimRenewed
                "test_defaults_changed_to_foreach",  # 测试方法为 test_defaults_changed_to_foreach
            ),
        ),
    ),
    # 创建一个 OptimizerInfo 对象，用于描述 RAdam 优化器的信息和配置
    OptimizerInfo(
        RAdam,  # 使用 RAdam 优化器
        optim_inputs_func=optim_inputs_func_radam,  # 设置优化器输入函数
        optim_error_inputs_func=optim_error_inputs_func_radam,  # 设置优化器错误输入函数
        supported_impls=("foreach", "differentiable"),  # 支持的实现方式
        skips=(  # 定义一组 DecorateInfo 对象，用于标记需要跳过的测试用例
            DecorateInfo(
                skipIfTorchDynamo("Fails fix point assertion on 3.8, see #97811"),  # 根据条件跳过测试
                "TestOptimRenewed",  # 所属测试类名
                "test_tensor_lr",  # 测试方法名
                active_if=sys.version_info < (3, 9) and sys.version_info > (3, 7),  # 仅在指定条件下激活
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors w/ Global state changed, see https://github.com/pytorch/pytorch/issues/116028"
                ),  # 根据条件跳过测试
                "TestOptimRenewed",  # 所属测试类名
                "test_set_default_dtype_works_with_foreach",  # 测试方法名
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Accessing grad.real errors, see https://github.com/pytorch/pytorch/issues/117184"
                ),  # 根据条件跳过测试
                "TestOptimRenewed",  # 所属测试类名
                "test_complex_2d",  # 测试方法名
            ),
            DecorateInfo(
                toleranceOverride(
                    {
                        # 设置特定数据类型的容忍度
                        torch.float64: tol(atol=1.5e-7, rtol=1.1e-7)
                    }
                ),  # 覆盖默认容忍度
                "TestOptimRenewed",  # 所属测试类名
                "test_foreach_matches_forloop",  # 测试方法名
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "This test uses mocks, which dynamo does not support"
                ),  # 根据条件跳过测试
                "TestOptimRenewed",  # 所属测试类名
                "test_defaults_changed_to_foreach",  # 测试方法名
            ),
        ),
    ),
    # 创建一个 OptimizerInfo 对象，使用 RMSprop 优化器作为参数
    OptimizerInfo(
        RMSprop,
        # 设置优化器输入函数为 optim_inputs_func_rmsprop
        optim_inputs_func=optim_inputs_func_rmsprop,
        # 设置优化器错误输入函数为 optim_error_inputs_func_rmsprop
        optim_error_inputs_func=optim_error_inputs_func_rmsprop,
        # 指定支持的实现方式为 "foreach" 和 "differentiable"
        supported_impls=("foreach", "differentiable"),
        # 定义跳过测试的装饰信息列表
        skips=(
            # 第一个装饰信息，跳过条件为非连续数据，说明在 #118115 中有关 addcdiv 不支持非连续情况
            DecorateInfo(
                skipIfMps,
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
                # 激活条件是 kwargs 中的 contiguous 参数为 False
                active_if=lambda kwargs: not kwargs["contiguous"],
            ),
            # 第二个装饰信息，跳过条件为在 TorchDynamo 环境下，Python 版本不在 3.8 范围内且大于 3.7
            DecorateInfo(
                skipIfTorchDynamo("Fails fix point assertion on 3.8, see #97811"),
                "TestOptimRenewed",
                "test_tensor_lr",
                active_if=sys.version_info < (3, 9) and sys.version_info > (3, 7),
            ),
            # 第三个装饰信息，跳过条件为在 TorchDynamo 环境下，详见 #116028
            DecorateInfo(
                skipIfTorchDynamo("See #116028"),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            # 第四个装饰信息，跳过条件为在 TorchDynamo 环境下，详见 GitHub 问题 #117184
            DecorateInfo(
                skipIfTorchDynamo(
                    "Accessing grad.real errors, see https://github.com/pytorch/pytorch/issues/117184"
                ),
                "TestOptimRenewed",
                "test_complex_2d",
            ),
            # 第五个装饰信息，重写容差设置，仅在 TEST_WITH_TORCHDYNAMO 为真时激活
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.float32: tol(atol=5e-04, rtol=0.01),
                    }
                ),
                "TestOptimRenewed",
                "test_mixed_device_dtype",
                active_if=TEST_WITH_TORCHDYNAMO,
            ),
            # 第六个装饰信息，跳过条件为在 TorchDynamo 环境下，因为不支持使用 mocks
            DecorateInfo(
                skipIfTorchDynamo(
                    "This test uses mocks, which dynamo does not support"
                ),
                "TestOptimRenewed",
                "test_defaults_changed_to_foreach",
            ),
        ),
    ),
    # 定义优化器信息，使用 Rprop 优化器，指定优化器输入和错误输入函数
    OptimizerInfo(
        Rprop,
        optim_inputs_func=optim_inputs_func_rprop,
        optim_error_inputs_func=optim_error_inputs_func_rprop,
        supported_impls=("foreach", "differentiable"),
        skips=(
            # 装饰信息：如果非连续数据，跳过测试
            DecorateInfo(
                skipIfMps,
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
                active_if=lambda kwargs: not kwargs["contiguous"],
            ),
            # 装饰信息：如果在特定 Python 版本中，跳过测试
            DecorateInfo(
                skipIfTorchDynamo("Fails fix point assertion on 3.8, see #97811"),
                "TestOptimRenewed",
                "test_tensor_lr",
                active_if=sys.version_info < (3, 9) and sys.version_info > (3, 7),
            ),
            # 装饰信息：跳过测试，原因是相关问题见 GitHub 上的 issue
            DecorateInfo(
                skipIfTorchDynamo("See #116028"),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            # 装饰信息：跳过测试，因为访问 grad.real 时出现错误
            DecorateInfo(
                skipIfTorchDynamo(
                    "Accessing grad.real errors, see https://github.com/pytorch/pytorch/issues/117184"
                ),
                "TestOptimRenewed",
                "test_complex_2d",
            ),
            # 装饰信息：跳过测试，因为测试使用了 Mock，而 Dynamo 不支持
            DecorateInfo(
                skipIfTorchDynamo(
                    "This test uses mocks, which dynamo does not support"
                ),
                "TestOptimRenewed",
                "test_defaults_changed_to_foreach",
            ),
        ),
    ),
    ),
    OptimizerInfo(
        SparseAdam,  # 使用 SparseAdam 作为优化器
        optim_inputs_func=optim_inputs_func_sparseadam,  # 设置 SparseAdam 的输入函数
        optim_error_inputs_func=optim_error_inputs_func_sparseadam,  # 设置 SparseAdam 的错误输入函数
        supported_impls=(),  # 不支持任何特定的实现
        only_supports_sparse_grads=True,  # 仅支持稀疏梯度
        metadata_for_sparse=({"lr": 4e-2}, []),  # 稀疏梯度的元数据，学习率为 0.04
        supports_complex=False,  # 不支持复杂梯度，参见 issue #118153
        skips=(  # 设置测试跳过的装饰器信息
            DecorateInfo(
                skipIfMps,  # 跳过条件为 skipIfMps，即 SparseAdam 不支持 MPS
                "TestOptimRenewed",  # 跳过的测试类名为 TestOptimRenewed
            ),
            DecorateInfo(
                unittest.skip(
                    "SparseAdam does not support dense gradients, see #116507"
                ),  # 跳过条件为 SparseAdam 不支持密集梯度，参见 issue #116507
                "TestOptimRenewed",  # 跳过的测试类名为 TestOptimRenewed
                "test_state_dict_deterministic",  # 跳过的测试方法名为 test_state_dict_deterministic
            ),
            DecorateInfo(
                skipIfTorchDynamo("cannot call to_sparse on p.grad, see #117184"),  # 跳过条件为 TorchDynamo 下无法调用 to_sparse 函数，参见 issue #117184
                "TestOptimRenewed",  # 跳过的测试类名为 TestOptimRenewed
                "test_param_groups_lr",  # 跳过的测试方法名为 test_param_groups_lr
            ),
            DecorateInfo(
                skipIfTorchDynamo("cannot call to_sparse on p.grad, see #117184"),  # 跳过条件同上
                "TestOptimRenewed",  # 跳过的测试类名为 TestOptimRenewed
                "test_tensor_lr",  # 跳过的测试方法名为 test_tensor_lr
            ),
            DecorateInfo(
                unittest.skip(
                    "SparseAdam does not support dense gradients, see #116507"
                ),  # 跳过条件同上
                "TestOptimRenewed",  # 跳过的测试类名为 TestOptimRenewed
                "test_can_load_older_state_dict",  # 跳过的测试方法名为 test_can_load_older_state_dict
            ),
            DecorateInfo(
                skipIfTorchDynamo("cannot call to_sparse on p.grad, see #117184"),  # 跳过条件同上
                "TestOptimRenewed",  # 跳过的测试类名为 TestOptimRenewed
                "test_load_nontensor_step",  # 跳过的测试方法名为 test_load_nontensor_step
            ),
            DecorateInfo(
                skipIfTorchDynamo("cannot call to_sparse on p.grad, see #117184"),  # 跳过条件同上
                "TestOptimRenewed",  # 跳过的测试类名为 TestOptimRenewed
                "test_forloop_goes_right_direction",  # 跳过的测试方法名为 test_forloop_goes_right_direction
            ),
            DecorateInfo(
                skipIfTorchDynamo("cannot call to_sparse on p.grad, see #117184"),  # 跳过条件同上
                "TestOptimRenewed",  # 跳过的测试类名为 TestOptimRenewed
                "test_forloop_goes_right_direction_multigpu",  # 跳过的测试方法名为 test_forloop_goes_right_direction_multigpu
            ),
            DecorateInfo(
                skipIfTorchDynamo("cannot call to_sparse on p.grad, see #117184"),  # 跳过条件同上
                "TestOptimRenewed",  # 跳过的测试类名为 TestOptimRenewed
                "test_param_group_with_lrscheduler_goes_right_direction",  # 跳过的测试方法名为 test_param_group_with_lrscheduler_goes_right_direction
            ),
            DecorateInfo(
                skipIfTorchDynamo("cannot call to_sparse on p.grad, see #117184"),  # 跳过条件同上
                "TestOptimRenewed",  # 跳过的测试类名为 TestOptimRenewed
                "test_state_dict_with_cuda_params",  # 跳过的测试方法名为 test_state_dict_with_cuda_params
            ),
            DecorateInfo(
                skipIfTorchDynamo("cannot call to_sparse on p.grad, see #117184"),  # 跳过条件同上
                "TestOptimRenewed",  # 跳过的测试类名为 TestOptimRenewed
                "test_deepcopy_copies_all_public_attrs",  # 跳过的测试方法名为 test_deepcopy_copies_all_public_attrs
            ),
        ),
    ),
# TensorTracker 类用于跟踪张量的克隆，并在稍后弹出它们（按顺序），以便在多步计算中进行公平比较。
# 主要用例通常是比较两个被认为相等的计算结果，例如每个都包含多个步骤的优化器步骤，其中数值偏差可能会成倍增加。
# 目标是能够在每个里程碑处比较和对齐数字，以最小化数值差异，因此当测试失败时，很可能是真实问题。

class TensorTracker:
    """
    A utility to track tensor clones in a list, with the expectation of popping them later (in
    order) to make fair comparisons between two multi-step computation. The intended use case is
    usually when comparing two supposed equal computations, such as an optimizer step that each
    individually consists of multiple steps, where numerical deviation could multiply.

    The goal is to be able to compare and align numbers at every milestone so as to minimize
    numerical discrepancies, and so when the test fails, it is likely a real problem.
    """

    def __init__(self, assert_eq_kwargs=None):
        """
        初始化 TensorTracker 实例。
        
        Parameters:
        - assert_eq_kwargs (dict, optional): 传递给断言相等性的关键字参数，默认为 {}。
        """
        if assert_eq_kwargs is None:
            assert_eq_kwargs = {}
        self.assert_eq_kwargs = assert_eq_kwargs
        self.tensors = []

    def add(self, tensor):
        """
        将张量的 clone().detach() 版本添加到跟踪列表中。
        
        Parameters:
        - tensor (Tensor): 要添加的张量。
        """
        self.tensors.append(tensor.clone().detach())

    # pops from beginning, like a queue and not a stack!
    def pop_check_set(self, tensor_to_set, testcase):
        """
        弹出张量跟踪器中的第一个元素，断言弹出的张量与输入张量相等，然后使用 copy_ 将输入张量设置为与弹出的张量相同的值。
        
        Parameters:
        - tensor_to_set (Tensor): 要设置的目标张量。
        - testcase: 测试案例对象，用于执行断言。
        """
        testcase.assertGreater(len(self.tensors), 0, "no tensors to pop")
        ref = self.tensors.pop(0)

        testcase.assertTrue(isinstance(ref, Tensor), f"{type(ref)=}")
        testcase.assertEqual(tensor_to_set, ref, **self.assert_eq_kwargs)

        with torch.no_grad():
            tensor_to_set.copy_(ref)

    def all_popped(self):
        """
        检查张量跟踪器中是否所有张量都已弹出。
        
        Returns:
        - bool: 如果所有张量都已弹出，则返回 True；否则返回 False。
        """
        return len(self.tensors) == 0
```