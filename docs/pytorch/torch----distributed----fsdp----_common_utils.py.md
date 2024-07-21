# `.\pytorch\torch\distributed\fsdp\_common_utils.py`

```py
# mypy: allow-untyped-defs
"""
This file includes private common utilities for FSDP.
"""
import logging                      # 导入 logging 模块，用于日志记录
import traceback                    # 导入 traceback 模块，用于异常追踪
import warnings                     # 导入 warnings 模块，用于发出警告
import weakref                      # 导入 weakref 模块，用于创建弱引用对象
from enum import auto, Enum         # 导入 enum 模块中的 auto 和 Enum 类
from functools import partial       # 导入 functools 模块中的 partial 函数
from typing import (                # 导入 typing 模块中的各种类型注解
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    Iterable,
    List,
    no_type_check,
    Optional,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
)

import torch                        # 导入 torch 模块
import torch.distributed as dist    # 导入 torch.distributed 模块并简称为 dist
import torch.distributed.fsdp._flat_param as flat_param_file  # 导入 flat_param_file 模块
import torch.nn as nn               # 导入 torch.nn 模块并简称为 nn
from torch.distributed._composable_state import _get_module_state, _State  # 导入 _get_module_state 和 _State 函数
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,            # 导入 _CHECKPOINT_PREFIX 常量
)
from torch.distributed.utils import _apply_to_tensors  # 导入 _apply_to_tensors 函数
from torch.utils._mode_utils import no_dispatch         # 导入 no_dispatch 常量

from .api import (                   # 从当前包中导入 api 模块中的各种类和函数
    FullOptimStateDictConfig,
    FullStateDictConfig,
    OptimStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictType,
)

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh        # 导入 DeviceMesh 类（类型检查时导入）
    from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions  # 导入 FSDPExtensions 类（类型检查时导入）
    from ._flat_param import FlatParamHandle                      # 导入 FlatParamHandle 类（类型检查时导入）

FSDP_WRAPPED_MODULE = "_fsdp_wrapped_module"        # 定义常量 FSDP_WRAPPED_MODULE
FSDP_PREFIX = FSDP_WRAPPED_MODULE + "."             # 定义常量 FSDP_PREFIX，用于组成键名前缀
FSDP_FLATTENED = "_fsdp_flattened"                 # 定义常量 FSDP_FLATTENED

# Save a global mapping from module to its input tensor dtype to be populated
# during the forward pre-hook and consumed in the forward post-hook when
# overriding a module's mixed precision
# NOTE: We currently take the last input tensor's dtype in the case of multiple
# floating-point input tensors, which may be incorrect. However, since there is
# not a 1:1 correspondence between input and output tensors, we must use *some*
# heuristic like this to predict the desired output dtype.
_MODULE_TO_INP_DTYPE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # 创建弱引用字典 _MODULE_TO_INP_DTYPE，用于映射模块到其输入张量的数据类型

class _FSDPDeviceHandle:
    """
    This is a simple abstraction for FSDP computing devices,
    which enables custom backends that implement CUDA-like
    semantics to be integrated with FSDP.
    """

    def __init__(self, device: torch.device, backend: Any = None):
        if backend is None:
            try:
                self.__backend = getattr(torch, device.type)  # 尝试获取与设备类型对应的 torch 后端
                self.__device = device                         # 设置设备属性
            except AttributeError as exc:
                raise AttributeError(
                    f"Device '{device}' does not have a corresponding backend registered as 'torch.{device.type}'."
                ) from exc
        else:
            self.__backend = backend  # 使用指定的后端
    # 根据设备返回相应的 _FSDPDeviceHandle 实例
    def from_device(cls, device: torch.device) -> "_FSDPDeviceHandle":
        """
        根据设备返回一个与之对应的设备处理器，通过该处理器可以在设备上执行与 CUDA 类似的操作。
        如果设备是 cuda，则直接返回 torch.cuda，以加快属性访问速度。
        自定义后端必须首先在 torch 上注册一个与设备类型相同名称的模块。
        """
        if device.type == "cuda":
            return cast(_FSDPDeviceHandle, torch.cuda)
        return cls(device)

    # 获取属性的魔术方法，尝试从 self.__backend 中获取相应的属性
    def __getattr__(self, __name: str) -> Any:
        try:
            return getattr(self.__backend, __name)
        except AttributeError as exc:
            # 如果属性不存在，抛出自定义错误信息
            raise AttributeError(
                f"自定义后端 '{self.__device.type}' 未实现 'torch.{self.__device.type}.{__name}'"
            ) from exc
class _UninitializedDeviceHandle(_FSDPDeviceHandle):
    # 这是一个特殊的设备处理类，继承自 _FSDPDeviceHandle
    def __init__(self):
        # 初始化方法，暂时不执行任何操作
        pass

    def __getattribute__(self, __name: str) -> Any:
        # 重载 __getattribute__ 方法，抛出运行时错误，禁止使用未初始化的设备处理对象
        raise RuntimeError("Trying to use an uninitialized device handle.")


class _FSDPState(_State):
    def __init__(self) -> None:
        # 初始化方法，定义了多个实例属性，用于存储 FSDP 状态信息
        # TODO: Move all the attributes to this class to enable typing for
        # FSDP/fully_shard.
        self._ignored_modules: Set[nn.Module] = set()  # 存储被忽略的模块集合
        self._ignored_params: Set[nn.Parameter] = set()  # 存储被忽略的参数集合
        # Buffer names are cleaned (without wrapper prefixes)
        self._ignored_buffer_names: Set[str] = set()  # 存储被忽略的缓冲区名称集合
        self.process_group: Optional[dist.ProcessGroup] = None  # 存储进程组对象，可选类型
        self.rank: int = -1  # 存储进程的排名，默认为 -1
        self.world_size: int = -1  # 存储世界大小，默认为 -1
        self._device_mesh: Optional[DeviceMesh] = None  # 存储设备网格对象，可选类型
        self.sharding_strategy = ShardingStrategy.FULL_SHARD  # 分片策略，默认为完全分片
        self._use_orig_params: bool = False  # 是否使用原始参数的标志
        self.training_state = TrainingState.IDLE  # 训练状态，默认为空闲
        self._unshard_params_ctx: Dict[nn.Module, Generator] = {}  # 存储未分片参数的上下文字典
        self._state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT  # 状态字典类型，默认为完整状态字典
        self._state_dict_config: StateDictConfig = FullStateDictConfig()  # 状态字典配置对象
        self._optim_state_dict_config: OptimStateDictConfig = FullOptimStateDictConfig()  # 优化器状态字典配置对象
        self._is_root: Optional[bool] = None  # 是否为根状态的标志，可选类型
        self._handle: Optional[flat_param_file.FlatParamHandle] = None  # 平坦参数处理对象，可选类型
        self._fully_sharded_module_to_handle: Dict[
            nn.Module, Optional[flat_param_file.FlatParamHandle]
        ] = {}  # 存储全分片模块到处理对象的映射字典
        self.compute_device: Optional[torch.device] = None  # 计算设备对象，可选类型
        self._gradient_predivide_factor: int = 0  # 梯度预分割因子，默认为 0
        self._gradient_postdivide_factor: int = 0  # 梯度后分割因子，默认为 0
        self._comm_hook: Optional[Callable] = None  # 通信钩子函数，可选类型
        self._comm_hook_state: Optional[Any] = None  # 通信钩子状态，可选类型
        self._unshard_event: Optional[torch.cuda.Event] = None  # 未分片事件对象，可选类型
        # Abstract device handle for fsdp compute device. For now,
        # the compute device must implement cuda semantics used by fsdp
        self._device_handle: _FSDPDeviceHandle = _UninitializedDeviceHandle()  # 抽象设备处理对象，默认为未初始化状态
        # All following attributes should only be used for root states:
        # Save these static lists to avoid the repeated tree traversals
        self._all_fsdp_states: List[_FSDPState] = []  # 存储所有 FSDP 状态对象的列表
        self._all_handles: List[flat_param_file.FlatParamHandle] = []  # 存储所有平坦参数处理对象的列表
        self._fsdp_extension: Optional[FSDPExtensions] = None  # FSDP 扩展对象，可选类型


def _get_module_fsdp_state(module: nn.Module) -> Optional[_FSDPState]:
    # 获取指定模块的 FSDP 状态对象，如果不存在或者不是 _FSDPState 类型，则返回 None
    state = _get_module_state(module)
    if state is None or not isinstance(state, _FSDPState):
        return None
    return state


def _get_module_fsdp_state_if_fully_sharded_module(
    module: nn.Module,
) -> Optional[_FSDPState]:
    # 获取指定模块的 FSDP 状态对象，仅在模块是完全分片模块时返回状态对象
    state = _get_module_fsdp_state(module)
    if state is None:
        return None
    if state == module:  # FullyShardedDataParallel module case.
        return state
    if module in state._fully_sharded_module_to_handle:  # fully_shard case.
        return state
    return None


class TrainingState(Enum):
    """
    定义训练状态枚举类
    """
    # 定义一个枚举，表示 ``FullyShardedDataParallel` 实例的状态。

    # 枚举成员：IDLE
    IDLE = auto()
    # 枚举成员：FORWARD_BACKWARD
    FORWARD_BACKWARD = auto()
    # 枚举成员：SUMMON_FULL_PARAMS
    SUMMON_FULL_PARAMS = auto()
# 定义一个枚举类，表示“FlatParamHandle”状态
class HandleTrainingState(Enum):
    """
    An enum that indicates the state of a ``FlatParamHandle`.
    """

    # 空闲状态
    IDLE = auto()
    # 前向传播状态
    FORWARD = auto()
    # 反向传播前处理状态
    BACKWARD_PRE = auto()
    # 反向传播后处理状态
    BACKWARD_POST = auto()
    # 调用所有参数状态
    SUMMON_FULL_PARAMS = auto()


# 定义一个函数，用于检查指定状态是否可组合
def _is_composable(state: _FSDPState):
    # TODO: This is a temporary hack for differentiate between code paths.
    # 检查指定状态是否为 nn.Module 的实例，暂时用于区分代码路径
    return not isinstance(state, nn.Module)


# 定义一个函数，根据FSDP状态和模块返回对应的“FlatParamHandle”
@no_type_check
def _module_handle(state: _FSDPState, module: nn.Module) -> Optional["FlatParamHandle"]:
    """
    Returns the ``FlatParamHandle`` s corresponding to ``module``. This is
    the handle that contains some parameter in ``module``.
    """
    if _is_composable(state):
        # 有效的FSDP状态可能没有受管参数和句柄，因此对应的 `_fully_sharded_module_to_handles` 中没有条目
        if state._handle is None:
            return None
        assert (
            module in state._fully_sharded_module_to_handle
        ), f"Expects a fully sharded module but got {module} on rank {state.rank}"
        return state._fully_sharded_module_to_handle[module]
    else:
        # 注意：这假设“module”是一个“FullyShardedDataParallel”实例。
        return module._handle


# 定义一个函数，用于检查指定模块是否有由FSDP管理的参数
@no_type_check
def _has_fsdp_params(state: _FSDPState, module: nn.Module) -> bool:
    """Returns if ``module`` has parameters managed by FSDP."""
    return _module_handle(state, module) is not None


# 定义一个函数，用于获取句柄的分片策略
def _get_sharding_strategy(handle):
    """
    Returns the sharding strategy of the handle.
    """
    return handle._sharding_strategy if handle else None


# 定义一个函数，清理参数或缓冲名称，去除任何模块包装前缀
def clean_tensor_name(tensor_name: str) -> str:
    """
    Cleans the parameter or buffer name by removing any module wrapper
    prefixes.
    """
    tensor_name = tensor_name.replace(FSDP_PREFIX, "")
    # TODO: Explicitly replacing the checkpoint wrapper prefix is not ideal as
    # it couples `CheckpointWrapper` and FSDP and also does not scale for more
    # module wrappers.
    tensor_name = tensor_name.replace(_CHECKPOINT_PREFIX, "")
    return tensor_name


# 定义一个函数，设置tensor的属性为被FSDP打平
def _set_fsdp_flattened(tensor: torch.Tensor) -> None:
    """
    Sets an attribute on ``tensor`` to mark it as flattened by FSDP. This is to
    avoid re-flattening it during nested construction.
    """
    setattr(tensor, FSDP_FLATTENED, True)


# 定义一个函数，检查tensor是否被标记为被FSDP打平
def _is_fsdp_flattened(tensor: torch.Tensor) -> bool:
    """Returns if ``tensor`` has been marked as flattened by FSDP."""
    return getattr(tensor, FSDP_FLATTENED, False)


# 定义一个函数，获取带有重复项的命名参数
def _named_parameters_with_duplicates(
    module: nn.Module, **kwargs: Any
) -> List[Tuple[str, nn.Parameter]]:
    """
    This API is required as some modules overwrite `named_parameters()` but do not support
    `remove_duplicate`.
    """
    assert (
        "remove_duplicate" not in kwargs
    ), "_named_parameters_with_duplicates cannot be used with `remove_duplicate` argument."
    kwargs["remove_duplicate"] = False
    try:
        ret = list(module.named_parameters(**kwargs))
    # 处理 AssertionError 异常，将 "remove_duplicate" 参数从 kwargs 中移除
    except AssertionError as e:
        kwargs.pop("remove_duplicate")
        # 调用 module 的 named_parameters 方法，使用 kwargs 作为命名参数，返回结果列表
        ret = list(module.named_parameters(**kwargs))
    # 返回 ret 变量作为函数的结果
    return ret
# 定义一个函数 _get_param_to_fqns，接受一个 torch.nn.Module 对象作为参数 model，
# dedup_shared_params 参数默认为 True，用于控制是否去重共享参数的 FQN 列表。
# 返回一个字典，将每个参数映射到其“规范”FQN列表。

def _get_param_to_fqns(
    model: torch.nn.Module,
    dedup_shared_params: bool = True,
) -> Dict[nn.Parameter, List[str]]:
    """
    Constructs a mapping from parameter to a list of its "canonical" FQNs. Here,
    we use canonical to mean the fully-qualified name assigned to the parameter
    based on its position in the original nn.Module hierarchy before any wrapper
    or parallelism has been applied to it. This is in contrast to FQNs that may be
    generated after parallelisms or wrappers have been applied to the model.

    Each normal parameter maps to a singleton list containing its FQN, while each
    ``FlatParameter`` maps to a list of its original parameter FQNs, which may
    have length greater than one.  All FQNs are prefixed starting from ``model``.

    In the case where FSDP was applied with ``use_orig_params=True``, there should be no
    ``FlatParameter`` s registered to the model's modules and this mapping will only
    contain mappings from ``nn.Parameter`` s to singleton FQN lists.

    It is only in the case where FSDP was applied with ``use_orig_params=False`` where
    a ``FlatParameter`` will be registered in place of the original parameters and there
    will be mappings from each ``FlatParameter`` to lists of FQNs corresponding to the
    original parameters.

    Args:
        model (torch.nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance).
        dedup_shared_params (bool): For shared parameters, if ``True``, only
            includes the FQNs corresponding to the first encounter of the
            shared parameter in the module traversal; if ``False``, then
            includes the FQNs across all encounters. (Default: ``True``)
    """
    # 定义一个函数 module_fn，用于遍历模型中的参数并处理它们的全局命名
    def module_fn(module, prefix, tree_level, param_to_fqns):
        # 遍历模型中的参数，获取参数名和参数对象，不递归处理子模块
        for param_name, param in _named_parameters_with_duplicates(module, recurse=False):
            # 根据参数类型确定局部全限定名（local_fqns）
            local_fqns = (
                param._fqns  # 如果参数是 FlatParameter 类型，则使用其 _fqns 属性
                if isinstance(param, flat_param_file.FlatParameter)
                else [param_name]  # 否则使用参数名作为局部全限定名
            )  # 从 `module` 起始添加前缀（prefix）
            # 根据模型的顶级前缀（prefix）和局部全限定名（local_fqns），生成全局全限定名（global_fqns）
            global_fqns = [
                clean_tensor_name(prefix + name) for name in local_fqns
            ]  # 包含 `prefix` 的模型的顶级（即包括 `prefix`）
            # 检查参数是否为共享参数（是否存在于 param_to_fqns 字典中）
            is_shared_param = param in param_to_fqns
            if not is_shared_param:
                # 如果参数不是共享参数，则将其添加到 param_to_fqns 字典中
                param_to_fqns[param] = global_fqns
            else:
                # 如果参数是共享参数，则根据特定条件进行处理
                if isinstance(param, flat_param_file.FlatParameter):
                    # 对于 FlatParameter 类型的参数，特定的处理方式，给出相应警告
                    warnings.warn(
                        "FlatParameter is being traversed more than once. "
                        "This case should only happen when using "
                        "DistributedModelParallel with FullyShardedDataParallel."
                    )
                    # 覆盖 param_to_fqns 中的全限定名列表，确保只获取正确的最后一个
                    param_to_fqns[param] = global_fqns
                elif not dedup_shared_params:
                    # 如果不需要去重共享参数，则扩展 param_to_fqns 中已有的全限定名列表
                    param_to_fqns[param].extend(global_fqns)
    
    # 定义一个函数 return_fn，用于返回参数到全局命名列表的字典 param_to_fqns
    def return_fn(param_to_fqns):
        return param_to_fqns
    
    # 初始化一个空字典 param_to_unflat_param_names，用于存储非扁平化参数名
    param_to_unflat_param_names: Dict[torch.nn.Parameter, List[str]] = {}
    
    # 调用 _apply_to_modules 函数，对模型进行处理，将模型中的参数应用到 module_fn 和 return_fn 函数中
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [key for key, _ in _named_parameters_with_duplicates(model)],  # 获取模型中的参数名列表
        param_to_unflat_param_names,  # 将非扁平化参数名映射字典传递给 _apply_to_modules 函数
    )
@no_type_check
def _log_post_backward_hook(
    state: _FSDPState, handle: "FlatParamHandle", logger: logging.Logger
) -> None:
    """
    在 TORCH_DISTRIBUTED_DEBUG=INFO 下，记录此钩子触发的模块名称。
    在某些激活检查点配置下，可以帮助调试某些钩子不触发的情况。

    Args:
        state (_FSDPState): FSDP 状态对象。
        handle (FlatParamHandle): 平坦参数处理对象。
        logger (logging.Logger): 日志记录器对象。
    """
    if state._use_orig_params and handle._debug_level == dist.DebugLevel.INFO:
        # 如果使用原始参数并且调试级别为 INFO，则记录参数的全限定名
        param_fqns = _get_handle_fqns_from_root(state, handle)
        logger.warning("FSDP firing post-backward hooks for parameters %s", param_fqns)


@no_type_check
def _get_handle_fqns_from_root(
    state: _FSDPState, handle: "FlatParamHandle"
) -> Optional[List[str]]:
    """
    从根节点获取平坦参数处理对象的全限定名列表。

    Args:
        state (_FSDPState): FSDP 状态对象。
        handle (FlatParamHandle): 平坦参数处理对象。

    Returns:
        Optional[List[str]]: 参数的全限定名列表，如果处理对象为 None 则返回 None。
    """
    if handle is None:
        return None
    param_to_fqn = state._exec_order_data.param_to_fqn
    handle_params = handle.flat_param._params  # only populated for use_orig_params
    param_fqns = [
        fqn for fqn_list in [param_to_fqn[p] for p in handle_params] for fqn in fqn_list
    ]
    return param_fqns


def _apply_to_modules(
    root_module: torch.nn.Module,
    module_fn: Callable,
    return_fn: Callable,
    filter_fqns: Optional[List[str]] = None,
    *args,
    **kwargs,
):
    """
    对以 root_module 为根的模块层次结构执行前序遍历，对每个模块应用 module_fn，并最终使用 return_fn 返回一个值。
    遍历构建完整的模块前缀名称（例如，类似于模型状态字典中的 "module.submodule."），并使其在 module_fn 中可用。

    Args:
        root_module (torch.nn.Module): 根模块对象。
        module_fn (Callable): 应用于每个模块的函数。
        return_fn (Callable): 用于最终返回值的函数。
        filter_fqns (Optional[List[str]], optional): 用于过滤模块全限定名的列表。默认为 None。
        *args: 传递给 module_fn 和 return_fn 的位置参数。
        **kwargs: 传递给 module_fn 和 return_fn 的关键字参数。
    """
    def f(module: torch.nn.Module, prefix: str, tree_level: int, *args, **kwargs):
        # 调用模块函数，用于在递归子模块之前执行（前序遍历）
        module_fn(module, prefix, tree_level, *args, **kwargs)
        # 遍历模块的命名子模块
        for submodule_name, submodule in module.named_children():
            # 如果子模块为空，则继续下一个子模块
            if submodule is None:
                continue
            # 构建新的前缀，将当前子模块名添加到前缀中
            new_prefix = prefix + submodule_name + "."
            # 增加树层级
            new_tree_level = tree_level + 1
            # 如果存在过滤全限定名（FQN），则进行过滤检查
            if filter_fqns is not None:
                for fqn in filter_fqns:
                    # 如果 FQN 以新前缀开头，则中断循环
                    if fqn.startswith(new_prefix):
                        break
                else:
                    # 如果当前子模块名为特定名称（"_fsdp_wrapped_module" 或 "_dmp_wrapped_module"）
                    if (
                        submodule_name == "_fsdp_wrapped_module"
                        or submodule_name == "_dmp_wrapped_module"
                    ):
                        # 设置新前缀为当前前缀，以避免特定子模块的干扰
                        new_prefix = prefix
                    # 如果当前子模块名为 "module"
                    elif submodule_name == "module":
                        # 设置新前缀为当前前缀，以避免特定子模块的干扰
                        new_prefix = prefix
            # 递归调用函数 f，处理当前子模块
            f(submodule, new_prefix, new_tree_level, *args, **kwargs)

    # 调用函数 f，从根模块开始，初始前缀为空字符串，初始树层级为 0
    f(root_module, "", 0, *args, **kwargs)
    # 返回执行 return_fn 函数的结果，传入 *args 和 **kwargs
    return return_fn(*args, **kwargs)
# 禁用类型检查的装饰器，标记函数不进行类型检查
@no_type_check
# 检查 FSDP 状态是否在指定的训练状态中
def _assert_in_training_states(
    state: _FSDPState,
    training_states: List[TrainingState],
) -> None:
    """Asserts that FSDP is in the states ``_training_states``."""
    # 使用 `ValueError` 替代 `assert` 来确保这些逻辑断言在禁用 `assert` 的情况下也能运行
    if state.training_state not in training_states:
        msg = (
            f"expected to be in states {training_states} but current state is "
            f"{state.training_state}"
        )
        # 在秩为 0 的进程上打印错误，以防这个函数在反向传播中被调用
        if state.rank == 0:
            if isinstance(state, nn.Module):
                print(f"Asserting FSDP instance is: {state}")
            print(f"ERROR: {msg}")
            traceback.print_stack()
        raise ValueError(msg)


def _get_root_modules(modules: Set[nn.Module]) -> Set[nn.Module]:
    """
    Returns:
        Set[nn.Module]: The subset of ``modules`` that are root modules (i.e.
        parent-less) with respect to the modules in the set itself. In other
        words, these are the modules in ``modules`` that are not the child of
        any other module in ``modules``.
    """
    # 初始化空集合以存储根模块
    root_modules: Set[nn.Module] = set()
    # 创建字典，将每个模块映射到其子模块集合
    module_to_submodules = {module: set(module.modules()) for module in modules}
    # 遍历候选模块，确定哪些是根模块
    for candidate_module in modules:
        is_root_module = True
        # 检查候选模块是否是其他模块的子模块
        for module, submodules in module_to_submodules.items():
            is_child_module = (
                candidate_module is not module and candidate_module in submodules
            )
            if is_child_module:
                is_root_module = False
                break
        # 如果候选模块没有被标记为其他模块的子模块，则将其添加到根模块集合中
        if is_root_module:
            root_modules.add(candidate_module)
    return root_modules


def _override_module_mixed_precision(
    root: torch.nn.Module,
    module_classes_to_override: Iterable[Type[nn.Module]],
    wrap_override_dict: Dict[str, Any] = {"mixed_precision": None},  # noqa: B006
) -> Set[Type[nn.Module]]:
    # 将传入的模块类别转换为集合，并确保唯一性
    module_classes_to_override = tuple(set(module_classes_to_override))
    # 返回一个集合，其中包含实际被覆盖的模块类别
    overridden_module_classes: Set[Type[nn.Module]] = set()
    for mod in root.modules():
        # 遍历根模块的所有子模块
        if isinstance(mod, module_classes_to_override):
            # 如果子模块属于指定需要覆盖的模块类型
            overridden_module_classes.add(type(mod))
            # 将该子模块的类型添加到被覆盖模块类集合中
            mod._wrap_overrides = wrap_override_dict  # type: ignore[assignment]
            # 将 wrap_override_dict 赋值给 mod._wrap_overrides 属性，忽略类型检查

            # TODO: 我们需要在 fp32 中运行此混合精度被忽略的模块，
            # 但是要确保后续的模块（可能也在混合精度下运行）仍然能接收到适当的精度输入，
            # 而不需要用户过多调整混合精度配置。
            # 因此，我们为上/下转换附加前后向钩子。我们应重新审视这个设计。

            def cast_fn(
                dtype: torch.dtype, module: nn.Module, x: torch.Tensor
            ) -> torch.Tensor:
                # 类型转换函数，将张量 x 转换为指定的 dtype 类型
                if not torch.is_floating_point(x) or x.dtype == dtype:
                    return x
                _MODULE_TO_INP_DTYPE[module] = x.dtype
                # 更新模块对应的输入数据类型映射
                return x.to(dtype)

            def forward_pre_hook(module, args):
                # 前向预处理钩子函数，调用 cast_fn 对输入参数 args 进行类型转换
                return _apply_to_tensors(partial(cast_fn, torch.float32, module), args)

            def forward_post_hook(module, args, output):
                # 后向处理钩子函数，如果模块的输入数据类型有变化，则调用 cast_fn 进行类型转换
                # 注意：如果前向传播中没有浮点张量，则不会设置该模块的数据类型，不会进行类型转换。
                if module in _MODULE_TO_INP_DTYPE:
                    old_dtype = _MODULE_TO_INP_DTYPE[module]
                    return _apply_to_tensors(
                        partial(cast_fn, old_dtype, module), output
                    )

            # 有意将这两个钩子函数追加，以便它们在所有其他钩子函数之后运行。
            mod.register_forward_pre_hook(forward_pre_hook, prepend=False)
            # 注册前向预处理钩子函数到 mod，不在前面添加
            mod.register_forward_hook(forward_post_hook, prepend=False)
            # 注册后向处理钩子函数到 mod，不在前面添加
    return overridden_module_classes
# 定义一个函数 _no_dispatch_record_stream，接受两个参数：tensor（torch.Tensor 类型）和 stream（torch.Stream 类型），返回 None
def _no_dispatch_record_stream(tensor: torch.Tensor, stream: torch.Stream) -> None:
    # FIXME record_stream 不支持非 CUDA 张量
    # 检查张量的设备类型是否不是 "cuda" 或私有后端名称，如果是，则直接返回，不执行后续操作
    if tensor.device.type not in ["cuda", torch._C._get_privateuse1_backend_name()]:
        return

    # 检查当前是否处于 TorchDynamo 编译模式，如果是，则直接返回，不执行后续操作
    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        return
        # 以下注释是从 @ezyang 提出的:
        # 在 https://github.com/pytorch/pytorch/pull/88014 中添加了 no_dispatch，
        # 看起来是因为我们实际上不支持 torch dispatch 中的 Stream 参数，所以会出错。
        # 如果 Dynamo 能够回答 "是否有任何 torch dispatch 模式" 激活（应该回答 False），
        # 这里的更好版本应该只是在禁用 dispatch 之前检查是否有任何模式。
        # TODO(voz): 扩展 Dynamo 实用工具以回答上述问题，在这里统一代码路径。
        # 调用 tensor 的 record_stream 方法，记录当前的流
        tensor.record_stream(stream)
    else:
        # 使用 no_dispatch 上下文管理器，禁用当前的 dispatch
        with no_dispatch():
            # 调用 tensor 的 record_stream 方法，记录当前的流
            tensor.record_stream(stream)
```