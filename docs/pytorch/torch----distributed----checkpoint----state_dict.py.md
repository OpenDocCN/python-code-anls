# `.\pytorch\torch\distributed\checkpoint\state_dict.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和库
import contextlib  # 用于上下文管理
import functools   # 用于函数装饰器和其他高阶函数
import gc          # Python 的垃圾回收接口
import warnings    # 用于警告控制

from dataclasses import asdict, dataclass, field  # 数据类相关的装饰器和字段定义
from itertools import chain  # 用于迭代工具
from typing import (  # 引入类型提示
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
    Union,
)

import torch  # PyTorch 深度学习框架
import torch.distributed as dist  # 分布式训练模块
import torch.nn as nn  # 神经网络模块
from torch.distributed._shard.sharded_tensor import ShardedTensor  # 分片张量
from torch.distributed._state_dict_utils import (  # 分布式状态字典相关工具
    _broadcast_state_dict,
    _distribute_state_dict,
    _flatten_state_dict,
    _gather_state_dict,
    _offload_state_dict_to_cpu,
    _unflatten_state_dict,
)
from torch.distributed._tensor import DTensor  # 分布式张量
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # 检查点相关的包装器
    _CHECKPOINT_PREFIX,
)
from torch.distributed.fsdp import (  # FullyShardedDataParallel 模块
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    OptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp._common_utils import (  # FSDP 的通用工具函数
    _get_module_fsdp_state_if_fully_sharded_module,
    FSDP_WRAPPED_MODULE,
)
from torch.nn.modules.module import _IncompatibleKeys  # 模块不兼容键
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式数据并行模块
from torch.utils._pytree import tree_map_only  # 用于仅映射树的工具函数

# 导出的符号列表
__all__ = [
    "FQNS_T",
    "PrimitiveType",
    "ValueType",
    "DictValueType",
    "ListDictValueType",
    "OptimizerStateType",
    "StateDictOptions",
    "get_model_state_dict",
    "get_optimizer_state_dict",
    "get_state_dict",
    "set_model_state_dict",
    "set_optimizer_state_dict",
    "set_state_dict",
]

# 私有变量，用于平铺参数的标志
_FLAT_PARAM = "_flat_param"
# 参数组的标识符
_PG = "param_groups"
# 参数的标识符
_PARAMS = "params"
# 状态的标识符
_STATE = "state"

# 全限定名称集合类型
FQNS_T = Set[str]
# 基本类型的联合类型
PrimitiveType = Union[DTensor, ShardedTensor, torch.Tensor, int, float, str]
# 值的联合类型，可以是基本类型，列表，元组或字典
ValueType = Union[
    PrimitiveType, List[PrimitiveType], Tuple[PrimitiveType], Dict[str, "ValueType"]
]
# 字典值类型，键为字符串，值可以是任何值类型
DictValueType = Dict[str, ValueType]
# 列表字典值类型，列表中的每个元素是字典值类型
ListDictValueType = List[DictValueType]
# 优化器状态类型，字典的键为字符串，值可以是字典值类型或列表字典值类型的联合类型
OptimizerStateType = Dict[str, Union[DictValueType, ListDictValueType]]

# 用于存储已修补状态字典的可调用对象集合
_patched_state_dict: Set[Callable] = set()


@contextlib.contextmanager
def _gc_context():
    # 管理垃圾回收上下文，暂时禁用垃圾回收
    is_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        # 恢复垃圾回收状态
        if is_enabled:
            gc.enable()


@dataclass
class StateDictOptions:
    """
    This dataclass specifies how get_state_dict/set_state_dict will work.

    - ``full_state_dict``: if this is set to True, all the tensors in the
      returned state_dict will be gathered. No ShardedTensor and DTensor
      will be in the returned state_dict.

    - ``cpu_offload``: offload all the tensors to cpu. To prevent CPU OOM, if
      ``full_state_dict`` is also true, then only the rank0 will get the
      state_dict and all other ranks will get empty state_dict.
    """

    full_state_dict: bool  # 是否返回完整状态字典的标志
    cpu_offload: bool  # 是否将所有张量卸载到 CPU 的标志
    # 是否将完整的状态字典传播到所有进程，用于分布式训练
    full_state_dict: bool = False
    
    # 是否在 CPU 上进行张量的卸载（offload），用于优化内存使用
    cpu_offload: bool = False
    
    # 是否忽略冻结参数，即不包含 ``requires_grad`` 为 False 的参数
    ignore_frozen_params: bool = False
    
    # 是否保留子模块前缀（已弃用），当 ``submodules`` 不为空时，决定是否在状态字典键中保留子模块的前缀
    keep_submodule_prefixes: bool = True
    
    # 在调用 ``model.load_state_dict()`` 时的严格模式选项
    strict: bool = True
    
    # 是否从 rank0 广播状态字典张量到其他进程，用于分布式训练；仅支持 DTensor，不支持传统的 ShardedTensor
    broadcast_from_rank0: bool = False
    
    # 是否扁平化优化器状态字典，这在某些优化器中可能会使用，但本处代码未进一步提供细节
    flatten_optimizer_state_dict: bool = False
@dataclass
class _StateDictInfo(StateDictOptions):
    fqn_param_mapping: Dict[
        Union[str, torch.Tensor], Union[FQNS_T, torch.Tensor]
    ] = field(default_factory=dict)  # 字段：参数全限定名映射字典，默认为空字典
    shared_params_mapping: Dict[
        Union[str, torch.Tensor], Union[FQNS_T, torch.Tensor]
    ] = field(default_factory=dict)  # 字段：共享参数映射字典，默认为空字典
    submodule_prefixes: Set[str] = field(default_factory=set)  # 字段：子模块前缀集合，默认为空集合
    handle_model: bool = True  # 字段：处理模型标志，默认为True
    handle_optim: bool = True  # 字段：处理优化器标志，默认为True
    fsdp_context: Callable = contextlib.nullcontext  # 字段：FSDP上下文，默认为nullcontext
    fsdp_modules: List[nn.Module] = field(default_factory=list)  # 字段：FSDP模块列表，默认为空列表


@functools.lru_cache(maxsize=None)
def _get_fqns(
    model: nn.Module,
    name: str,
    skip_ddp_prefix: bool = True,
    skip_compiler_prefix: bool = True,
) -> FQNS_T:
    """
    This API is used to convert the name of a parameter to the FQNs. For FSDP
    without `use_orig_params`, the name of FlatParameter can be mapped to
    multiple original parameters. As a result, the return type of this function
    is `Set[str]`.

    Args:
        module (nn.Module): the root model.
        name (str): the name
        skip_ddp_prefix (bool): whether to skip DDP's `module` prefix

    Returns:
        The canonical FQNs based on the model traversal.
    """
    
    # Remove the checkpoint prefix, if it exists.
    name = name.replace(_CHECKPOINT_PREFIX, "")  # 如果存在检查点前缀，则移除

    if "." not in name:
        return {name}  # 如果名称中不包含点，则直接返回名称作为集合的一个元素

    obj_names = name.split(".")
    fqn_obj_names = []
    curr_obj = model

    for i, curr_obj_name in enumerate(obj_names):
        if isinstance(curr_obj, DDP):
            assert curr_obj_name == "module"
            curr_obj = curr_obj.module
            if not skip_ddp_prefix:
                fqn_obj_names.append(curr_obj_name)
        elif isinstance(curr_obj, FSDP):
            if i < len(obj_names) - 1 and obj_names[i + 1] == _FLAT_PARAM:
                prefix = ".".join(fqn_obj_names)
                flat_param = getattr(curr_obj, _FLAT_PARAM)
                if prefix:
                    prefix = f"{prefix}."
                return {f"{prefix}{fqn}" for fqn in flat_param._fqns}
            curr_obj = getattr(curr_obj, FSDP_WRAPPED_MODULE)
            if curr_obj_name != FSDP_WRAPPED_MODULE:
                fqn_obj_names.append(curr_obj_name)
                curr_obj = getattr(curr_obj, curr_obj_name)
        elif isinstance(curr_obj, torch._dynamo.eval_frame.OptimizedModule):
            assert curr_obj_name == "_orig_mod"
            curr_obj = curr_obj._orig_mod
            if not skip_compiler_prefix:
                fqn_obj_names.append(curr_obj_name)
        else:
            fqn_obj_names.append(curr_obj_name)
            if curr_obj_name == nn.modules.module._EXTRA_STATE_KEY_SUFFIX:
                if i != len(obj_names) - 1:
                    raise RuntimeError("Expect `_extra_state` to be the last obj name")
            else:
                curr_obj = getattr(curr_obj, curr_obj_name)

    return {".".join(fqn_obj_names).replace(_CHECKPOINT_PREFIX, "")}  # 返回最终的全限定名集合，移除检查点前缀


class _EXTRA_STATE:
    pass  # 空类，用于占位，无实际功能
# 定义一个函数 `_iterate_valid_model_state`，用于迭代遍历模型的有效状态
def _iterate_valid_model_state(model):
    # 初始化已访问模块的集合，用于避免重复访问
    visited_modules: Set[nn.Module] = set()

    # 定义递归函数 `recurse`，用于递归地遍历模块及其子模块
    def recurse(module: nn.Module, curr_fqn: str) -> Generator:
        # 将当前模块加入已访问集合
        visited_modules.add(module)

        # 根据当前完全限定名 `curr_fqn` 构造下一级子模块的完全限定名
        curr_fqn = f"{curr_fqn}." if curr_fqn else ""
        
        # 遍历当前模块的子模块
        for name, submodule in module.named_children():
            # 如果子模块已被访问过，则跳过
            if submodule in visited_modules:
                continue
            # 构造新的完全限定名 `new_fqn`
            new_fqn = f"{curr_fqn}{name}"
            # 递归调用 `recurse` 函数，生成子模块的迭代器
            yield from recurse(submodule, new_fqn)

        # 遍历当前模块的缓冲区和参数
        for name, obj in chain(
            module.named_buffers(recurse=False), module.named_parameters(recurse=False)
        ):
            # 如果是非持久缓冲区，则跳过
            if name in module._non_persistent_buffers_set:
                continue
            # 构造新的完全限定名 `new_fqn`
            new_fqn = f"{curr_fqn}{name}"
            # 返回完全限定名和对象的生成器
            yield new_fqn, obj

        # 检查当前模块是否定义了 `get_extra_state` 方法
        if (
            getattr(module.__class__, "get_extra_state", nn.Module.get_extra_state)
            != nn.Module.get_extra_state
        ):
            # 如果定义了，则构造新的完全限定名 `new_fqn` 并生成额外状态对象 `_EXTRA_STATE` 的生成器
            new_fqn = f"{curr_fqn}{nn.modules.module._EXTRA_STATE_KEY_SUFFIX}"
            yield new_fqn, _EXTRA_STATE()

    # 返回整个递归生成器的迭代结果
    yield from recurse(model, "")


# 定义一个函数 `_verify_options`，用于验证用户传入的模型和选项，并生成 `_StateDictInfo`
def _verify_options(
    model: nn.Module,
    optims: Tuple[torch.optim.Optimizer, ...],
    optim_only: bool,
    *,
    submodules: Optional[Set[nn.Module]] = None,
    options: Optional[StateDictOptions] = None,
) -> _StateDictInfo:
    """
    Verify the model and options passed by the user and generates _StateDictInfo.
    """
    # 如果 `submodules` 参数不为空，则发出警告
    if submodules:
        warnings.warn(
            "Getting submodules only model/optim state_dict is deprecated and "
            "will be removed in 2.5. This feature can be achieved by manually "
            "filtering out the state_dict returned from get_state_dict.",
            FutureWarning,
        )
    # 如果 `optim_only` 为真且未传入优化器，则抛出运行时错误
    if optim_only and not optims:
        raise RuntimeError(
            "Optimizers are not passed in but optim_only is set to True."
        )

    # 如果未传入选项对象 `options`，则使用默认的 `StateDictOptions`
    options = options or StateDictOptions()

    # 初始化完全限定名与参数映射的字典 `fqn_param_mapping`
    fqn_param_mapping: Dict[
        Union[str, torch.Tensor], Union[Set[str], torch.Tensor]
    ] = {}
    # 初始化共享参数映射的字典 `shared_params_mapping`
    shared_params_mapping: Dict[
        Union[str, torch.Tensor], Union[Set[str], torch.Tensor]
    ] = {}

    # 遍历 `_iterate_valid_model_state` 函数返回的生成器结果
    for name, param in _iterate_valid_model_state(model):
        # 如果参数是 `_EXTRA_STATE` 类型，则跳过
        if isinstance(param, _EXTRA_STATE):
            continue

        # 获取完全限定名集合 `fqns`
        fqns = _get_fqns(model, name)
        # 获取 `param` 对应的完全限定名集合
        fqn = fqn_param_mapping.get(param, None)
        if fqn is not None:
            # 如果 `param` 已经存在于映射中，则更新其对应的完全限定名集合 `fqns`
            cast(Set[str], fqn_param_mapping[param]).update(fqns)
            shared_params_mapping[param] = fqn_param_mapping[param]
        else:
            # 否则，将完全限定名集合 `fqns` 复制到映射中
            fqn_param_mapping[param] = fqns.copy()
        # 遍历 `fqns`，将每个完全限定名与参数映射到 `fqn_param_mapping` 中
        for fqn in fqns:
            if not isinstance(param, _EXTRA_STATE):
                fqn_param_mapping[fqn] = param

    # 遍历共享参数映射字典 `shared_params_mapping` 中的参数和完全限定名集合，扩展到每个完全限定名
    for param_, fqns_ in list(shared_params_mapping.items()):
        for fqn in fqns_:
            shared_params_mapping[fqn] = cast(torch.Tensor, param_)

    # 初始化子模块前缀集合 `submodule_prefixes`
    submodule_prefixes: Set[str] = set()
    # 如果给定了子模块集合，则转换为集合类型
    if submodules:
        submodules = set(submodules)
        # 遍历模型中所有命名模块的名称和模块对象
        for name, module in model.named_modules():
            # 如果当前模块不在子模块集合中，则跳过
            if module not in submodules:
                continue
            # 获取当前模块的全限定名列表
            fqns = _get_fqns(model, name)
            # 断言全限定名列表长度为1，确保子模块的全限定名只有一个实例
            assert len(fqns) == 1, "Submodule FQN should only have 1 instance"
            # 将每个全限定名加入到子模块前缀集合中，以"."结尾
            submodule_prefixes.update(f"{fqn}." for fqn in fqns)

    # 如果选项中指定了从rank 0广播并且未选择完整状态字典，则抛出值错误异常
    if options.broadcast_from_rank0 and not options.full_state_dict:
        raise ValueError(
            "full_state_dict must be True when broadcast_from_rank0 is True."
        )

    # 获取模型中的FSDP模块列表
    fsdp_modules = FSDP.fsdp_modules(model)
    
    # 定义状态字典配置和优化器状态字典配置变量
    state_dict_config: StateDictConfig
    optim_state_dict_config: OptimStateDictConfig
    fsdp_context: Callable
    
    # 如果存在FSDP模块
    if fsdp_modules:
        # FSDP API只有在至少存在一个FSDP实例时才起作用
        if options.full_state_dict:
            # 使用完整状态字典配置
            state_dict_config = FullStateDictConfig(
                offload_to_cpu=options.cpu_offload, rank0_only=options.cpu_offload
            )
            # 使用完整优化器状态字典配置
            optim_state_dict_config = FullOptimStateDictConfig(
                offload_to_cpu=options.cpu_offload,
                rank0_only=(options.cpu_offload or options.broadcast_from_rank0),
            )
            # 状态字典类型为完整状态字典
            state_dict_type = StateDictType.FULL_STATE_DICT
        else:
            # 使用分片状态字典配置
            state_dict_config = ShardedStateDictConfig(
                offload_to_cpu=options.cpu_offload,
            )
            # 使用分片优化器状态字典配置
            optim_state_dict_config = ShardedOptimStateDictConfig(
                offload_to_cpu=options.cpu_offload,
            )
            # 状态字典类型为分片状态字典
            state_dict_type = StateDictType.SHARDED_STATE_DICT

        # 定义一个上下文管理器，用于处理FSDP状态字典类型而不生成警告
        @contextlib.contextmanager
        def fsdp_state_dict_type_without_warning(
            module,
            state_dict_type,
            state_dict_config,
            optim_state_dict_config,
        ):
            with warnings.catch_warnings():
                # 使用FSDP的状态字典类型上下文管理器
                with FSDP.state_dict_type(
                    module=module,
                    state_dict_type=state_dict_type,
                    state_dict_config=state_dict_config,
                    optim_state_dict_config=optim_state_dict_config,
                ):
                    yield

        # 创建一个部分应用的FSDP状态字典类型上下文管理器
        fsdp_context = functools.partial(
            fsdp_state_dict_type_without_warning,
            module=model,
            state_dict_type=state_dict_type,
            state_dict_config=state_dict_config,
            optim_state_dict_config=optim_state_dict_config,
        )
    else:
        # 如果不存在FSDP模块，则使用空的上下文管理器
        fsdp_context = contextlib.nullcontext

    # 返回_StateDictInfo对象，包括各种选项和映射信息
    return _StateDictInfo(
        **asdict(options),
        fqn_param_mapping=fqn_param_mapping,
        shared_params_mapping=shared_params_mapping,
        submodule_prefixes=submodule_prefixes,
        fsdp_context=fsdp_context,
        fsdp_modules=cast(List[nn.Module], fsdp_modules),
        handle_model=not optim_only,
        handle_optim=(len(optims) > 0),
    )
def _verify_state_dict(
    model_state_dict: Dict[str, ValueType],
    optim_state_dict: OptimizerStateType,
    info: _StateDictInfo,
) -> None:
    # 遍历 FSDP 模块列表，检查每个模块的 FSDP 状态是否存在，若不存在则断言失败
    for module in info.fsdp_modules:
        fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
        assert fsdp_state is not None, "Expected a fsdp_state with a fsdp module."

    # 验证 model_state_dict 和 optim_state_dict 是否有效。如果条件不符合，则抛出 RuntimeError
    if (
        info.handle_model
        and not model_state_dict
        and not info.submodule_prefixes
        and not info.ignore_frozen_params
        and not (info.cpu_offload and info.full_state_dict)
        and info.strict
        and not info.broadcast_from_rank0
    ):
        raise RuntimeError(
            "The option indicates that model state_dict is required to save "
            "or load, but model state_dict is empty."
            f"rank = {dist.get_rank()=}."
        )

    # 如果 info.handle_optim 为真，则检查 optim_state_dict 是否为空。如果为空，则抛出 RuntimeError
    if info.handle_optim:
        if (
            not optim_state_dict
            and not (info.cpu_offload and info.full_state_dict)
            and (not info.broadcast_from_rank0)
        ):
            raise RuntimeError(
                "The option indicates that model state_dict is required to save, "
                f"or load but optim state_dict is empty. {optim_state_dict}"
            )

    # 检查 model_state_dict 的每个键是否包含 "_FLAT_PARAM"。如果有，则抛出 RuntimeError
    for key in model_state_dict.keys():
        if _FLAT_PARAM in key:
            raise RuntimeError(
                f"{key} contains {_FLAT_PARAM}. This can happen if the model "
                "is not the root module."
            )


def _state_dict_fn(obj: Union[nn.Module, torch.optim.Optimizer], api: str) -> Callable:
    # 获取对象 obj 的指定 API 函数，并返回其可调用对象
    call = getattr(obj, api)
    # 如果调用函数在 _patched_state_dict 中，则使用 functools.partial 为其绑定 self 参数
    if call in _patched_state_dict:
        call = functools.partial(getattr(obj.__class__, api), self=obj)
    return call


def _maybe_full_or_cpu_state_dict(
    state_dict: Dict[str, Any], info: _StateDictInfo
) -> Dict[str, Any]:
    # 如果 info.full_state_dict 为真，则根据条件收集 state_dict 中的状态信息
    if info.full_state_dict:
        ranks_only = (
            tuple()
            if (not info.cpu_offload or not torch.distributed.is_initialized())
            else (0,)
        )
        return _gather_state_dict(
            state_dict, cpu_offload=info.cpu_offload, ranks_only=ranks_only
        )
    # 如果 info.full_state_dict 为假且 info.cpu_offload 为真，则将 state_dict 中的状态转移到 CPU
    elif info.cpu_offload:
        return _offload_state_dict_to_cpu(state_dict)
    else:
        return state_dict


def _get_model_state_dict(
    model: nn.Module, info: _StateDictInfo
) -> Dict[str, ValueType]:
    # 如果 info.handle_model 为假，则返回空字典
    if not info.handle_model:
        return {}

    # 使用 info.fsdp_context() 上下文，获取模型 model 的状态字典
    with info.fsdp_context():
        state_dict = _state_dict_fn(model, "state_dict")()
    # 遍历状态字典中的所有键（key）
    for key in list(state_dict.keys()):
        # 获取与当前键关联的全限定名（Fully Qualified Name，FQN）
        fqns = _get_fqns(model, key)
        # 断言当前键只有一个对应的全限定名
        assert len(fqns) == 1, (key, fqns)
        # 从生成的全限定名集合中获取唯一的全限定名
        fqn = next(iter(fqns))
        # 如果全限定名不等于键本身
        if fqn != key:
            # 检验函数，用于验证全限定名是否与键对应
            def verify(key, fqn) -> bool:
                # 如果全限定名长度大于等于键长度，则返回假
                if len(fqn) >= len(key):
                    return False
                # 分割全限定名和键为列表
                fqn_split = fqn.split(".")
                key_split = key.split(".")
                fqn_idx = 0
                # 遍历键的每个部分
                for key_idx, key_name in enumerate(key_split):
                    # 如果当前键部分与全限定名部分匹配
                    if key_name == fqn_split[fqn_idx]:
                        fqn_idx += 1
                        # 如果已经匹配到了全限定名的最后一部分
                        if fqn_idx == len(fqn_split):
                            # 则验证是否已经到达键的最后一部分
                            return key_idx == len(key_split) - 1
                    # 如果键部分是 "module" 或 "_orig_mod"，则继续下一轮循环
                    elif key_name in ("module", "_orig_mod"):
                        continue
                    # 如果有不匹配的部分，则返回假
                    else:
                        return False
                # 如果所有键部分都匹配了全限定名的各个部分，则返回真
                return True

            # 如果验证失败，则抛出运行时错误，指出不期待的键存在以及其全限定名
            if not verify(key, fqn):
                raise RuntimeError(f"An unexpected key, {key}, exists. FQN is {fqn}")
            # 将状态字典中键为 key 的项的值，移动到键为全限定名 fqn 的位置，并删除原键
            state_dict[fqn] = state_dict.pop(key)

    # 如果信息对象指定了子模块前缀
    if info.submodule_prefixes:
        # 创建一个新的状态字典
        new_state_dict: Dict[str, ValueType] = {}
        # 遍历当前状态字典中的所有全限定名
        for fqn in state_dict.keys():
            # 遍历所有指定的子模块前缀
            for prefix in info.submodule_prefixes:
                # 如果当前全限定名不以某个前缀开头，则继续下一轮循环
                if not fqn.startswith(prefix):
                    continue
                # 如果信息对象要求保留子模块前缀
                if info.keep_submodule_prefixes:
                    # 将当前全限定名及其对应的值添加到新状态字典中
                    new_state_dict[fqn] = state_dict[fqn]
                else:
                    # 否则，从全限定名中去掉前缀后，将其及其对应的值添加到新状态字典中
                    new_fqn = fqn[len(prefix) :]
                    new_state_dict[new_fqn] = state_dict[fqn]
        # 将更新后的状态字典赋值给当前状态字典
        state_dict = new_state_dict

    # 如果信息对象要求忽略冻结参数
    if info.ignore_frozen_params:
        # 遍历模型中的所有命名参数
        for key, param in model.named_parameters():
            # 如果参数不要求梯度，则继续下一轮循环
            if param.requires_grad:
                continue
            # 获取与当前参数关联的全限定名集合
            fqns = _get_fqns(model, key)
            # 遍历当前参数关联的所有全限定名
            for fqn in fqns:
                # 从状态字典中删除该全限定名对应的项
                state_dict.pop(fqn)

    # 遍历状态字典中的所有键值对
    for key, p in list(state_dict.items()):
        # 如果当前值是张量并且被标记为元数据
        if torch.is_tensor(p) and p.is_meta:
            # 从状态字典中删除该键值对
            state_dict.pop(key)

    # 返回可能为完整状态字典或CPU状态字典的结果
    return _maybe_full_or_cpu_state_dict(state_dict, info)
def _load_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, ValueType],
    info: _StateDictInfo,
) -> _IncompatibleKeys:
    # 如果不需要处理模型或者状态字典为空且不需要从rank0广播，则返回空的不兼容键对象
    if not info.handle_model or (not state_dict and not info.broadcast_from_rank0):
        return _IncompatibleKeys({}, {})

    # 本地状态字典，用于存储模型的有效状态
    local_state_dict = {}
    # 遍历有效的模型状态
    for key, value in _iterate_valid_model_state(model):
        # 获取当前键的全限定名列表，包括DDP前缀
        fqns = _get_fqns(model, key)
        # 获取当前键的全限定名列表，包括DDP和编译器前缀
        fqns_with_prefix = _get_fqns(
            model, key, skip_ddp_prefix=False, skip_compiler_prefix=False
        )

        # 遍历全限定名及带前缀的全限定名列表，并处理不兼容的键
        for fqn, fqn_with_prefix in zip(fqns, fqns_with_prefix):
            # 如果不需要从rank0广播或当前进程是rank0，并且全限定名与带前缀的全限定名不相同，则将状态字典中对应键的值转移到带前缀的全限定名下
            if (
                not info.broadcast_from_rank0 or dist.get_rank() == 0
            ) and fqn != fqn_with_prefix:
                state_dict[fqn_with_prefix] = state_dict.pop(fqn)
            # 将带前缀的全限定名及其对应的值存入本地状态字典
            local_state_dict[fqn_with_prefix] = value

    # 如果需要从rank0广播，则根据本地状态字典的设备广播状态字典，并更新状态字典
    if info.broadcast_from_rank0:
        device = None
        for key, value in local_state_dict.items():
            if torch.is_tensor(value) and value.dim() > 0:
                if device is None:
                    device = value.device
                else:
                    assert device == value.device
        assert device is not None
        # 使用FSDP上下文广播状态字典，并根据严格模式加载状态字典
        _broadcast_state_dict(
            state_dict, local_state_dict, device=device, strict=info.strict
        )
        # 将本地状态字典中的状态复制到状态字典中
        for fqn, local_state in local_state_dict.items():
            state_dict[fqn] = local_state

    # 在FSDP上下文中加载模型状态字典并返回不兼容的键对象
    with info.fsdp_context():
        return cast(
            _IncompatibleKeys,
            _state_dict_fn(model, "load_state_dict")(
                state_dict=state_dict, strict=info.strict
            ),
        )


def _init_optim_state(optim: torch.optim.Optimizer) -> None:
    """
    Initialize optim states by calling the step() with zero grads.
    """
    # 如果优化器状态已初始化，则直接返回
    if optim.state:
        return

    # 遍历参数组并初始化梯度为零
    for param_group in optim.param_groups:
        for param in param_group[_PARAMS]:
            # 如果参数梯度不为空，则抛出运行时错误
            if param.grad is not None:
                raise RuntimeError(
                    "state_dict can only be used if the optimizer "
                    "states are initialized (usually after one step() with "
                    "gradients) or gradients are None. For the later case, "
                    "state_dict will fake the gradients as zero "
                    "to initialize the optimizer states. However, the "
                    "gradients are not None."
                )
            # 如果参数需要梯度，则将其梯度初始化为与其形状相同的零张量
            if param.requires_grad:
                param.grad = torch.zeros_like(param)

    # 将所有参数组的学习率置零并调用一次优化器的step()方法
    lrs = []
    for param_group in optim.param_groups:
        if "lr" in param_group:
            lrs.append(param_group["lr"])
            param_group["lr"] = 0.0
    optim.step(closure=None)
    # 尽管后续会恢复学习率，但此处恢复学习率是否恢复并不重要
    # 遍历优化器参数组列表
    for param_group in optim.param_groups:
        # 检查参数组是否包含学习率参数
        if "lr" in param_group:
            # 将学习率参数设置为列表 lrs 的第一个元素，并从 lrs 列表中移除该元素
            param_group["lr"] = lrs.pop(0)
    # 通过调用 zero_grad 方法来将优化器中的梯度值清零
    optim.zero_grad(set_to_none=True)
def _flatten_optim_state_dict(state_dict: OptimizerStateType) -> Dict[str, ValueType]:
    """
    将优化器状态字典扁平化，以支持 MPMD（如管道并行）的优化器重新分配。

    未使用该 API 时，原始的优化器状态字典如下所示：
    {
        "state": {
            "layer1.weight": {
                "step": 10, "exp_avg": SomeTensor, "exp_avg_sq": SomeTensor
            },
            "layer2.weight": {
                "step": 10, "exp_avg": SomeTensor, "exp_avg_sq": SomeTensor
            },
        },
        "param_group": [
            {
                "lr": 0.0,
                "betas": (0.9, 0.95), ...,
                "params": ["layer1.weight", "layer2.weight"]
            }
        ]
    }

    使用该 API 后，优化器状态字典变为：
    {
        "state.layer1.weight.step": 10,
        "state.layer2.weight.step": 10,
        "state.layer1.weight.exp_avg": SomeTensor,
        "state.layer2.weight.exp_avg": SomeTensor,
        "state.layer1.weight.exp_avg_sq": SomeTensor,
        "state.layer2.weight.exp_avg_sq": SomeTensor,
        "param_group.layer1.weight.lr" : 0.1,
        "param_group.layer2.weight.lr" : 0.1,
        "param_group.layer1.weight.betas" : (0.9, 0.95),
        "param_group.layer2.weight.betas" : (0.9, 0.95),
    }

    注意，如果任何值是容器，如示例中的 betas，该 API 不会对其进行扁平化。
    """

    def _raise_if_type_not_supported(v):
        """
        检查值的类型是否支持扁平化，如果不支持则抛出异常。
        """
        if not isinstance(v, (torch.Tensor, int, float)):
            raise NotImplementedError(
                "Flattening optimizer state_dict only supports "
                "tensor, int, float states now. "
                f"Type is {type(v)}."
            )

    ret: Dict[str, ValueType] = {}
    for fqn, state in cast(DictValueType, state_dict[_STATE]).items():
        """
        遍历优化器状态字典中的状态部分，并将每个状态扁平化处理。
        """
        for k, v in cast(DictValueType, state).items():
            _raise_if_type_not_supported(v)
            ret[f"{_STATE}.{fqn}.{k}"] = v

    for param_group in cast(ListDictValueType, state_dict[_PG]):
        """
        遍历优化器状态字典中的参数组部分，并将每个参数组扁平化处理。
        """
        fqns = param_group.pop(_PARAMS)
        for fqn in cast(List[str], fqns):
            for k, v in param_group.items():
                ret[f"{_PG}.{fqn}.{k}"] = v
    return ret


def _unflatten_optim_state_dict(
    optim: torch.optim.Optimizer,
    state_dict: Dict[str, ValueType],
    info: _StateDictInfo,
) -> OptimizerStateType:
    """
    此 API 用于将 _flatten_optim_state_dict() 生成的状态字典解扁平化。
    更多详情请参考 _flatten_optim_state_dict() 的文档字符串。
    """
    state: DictValueType = {}
    pg_state: ListDictValueType = []
    return_osd: OptimizerStateType = {_STATE: state, _PG: pg_state}
    # 遍历优化器的参数组列表
    for param_group in optim.param_groups:
        # 在pg_state中追加一个新字典，用于存储参数组的状态信息
        pg_state.append({_PARAMS: []})
        # 遍历当前参数组中的每一个参数
        for param in param_group[_PARAMS]:
            # 遍历与当前参数关联的所有全限定名（Fully Qualified Name）
            for fqn in info.fqn_param_mapping[param]:
                # 获取当前参数组状态列表
                params = pg_state[-1][_PARAMS]
                # 断言params是一个列表（类型注释）
                assert isinstance(params, list)
                # 将全限定名添加到参数组状态列表中
                params.append(fqn)
                # 如果当前参数不需要梯度，则跳过后续操作
                if not param.requires_grad:
                    continue
                # 初始化当前参数的状态信息为空字典
                state[fqn] = {}
                # 遍历优化器状态中与当前参数关联的所有状态名称
                for state_name in optim.state[param].keys():
                    # 将状态值复制到对应的状态字典中
                    cast(DictValueType, state[fqn])[state_name] = state_dict[
                        f"{_STATE}.{fqn}.{state_name}"
                    ]

        # 获取当前参数组第一个参数的全限定名
        first_param_fqn = cast(List[str], pg_state[-1][_PARAMS])[0]
        # 遍历参数组的所有键
        for k in param_group.keys():
            # 如果键是_PARAM则跳过
            if k == _PARAMS:
                continue
            # 获取当前键在状态字典中的值
            value = state_dict[f"{_PG}.{first_param_fqn}.{k}"]
            # 如果当前键不在pg_state中，则将其添加进去
            if k not in pg_state[-1]:
                pg_state[-1][k] = value
            # 否则，如果当前键对应的值与已有值不同，则引发运行时错误
            elif pg_state[-1][k] != value:
                raise RuntimeError(
                    "All the parameters in the same parameter group should have "
                    f"the same saved param_group value. But {first_param_fqn}.{k} "
                    f"is {value} while other(s) is {pg_state[-1][k]}."
                )

    # 返回结果字典return_osd
    return return_osd
# 根据模型、优化器和状态字典信息，返回优化器状态字典
def _get_optim_state_dict(
    model: nn.Module,
    optimizers: Tuple[torch.optim.Optimizer, ...],
    info: _StateDictInfo,
) -> OptimizerStateType:
    # 如果不需要处理优化器状态，则返回空字典
    if not info.handle_optim:
        return {}

    # 初始化优化器状态字典的基本结构
    optim_state_dict: OptimizerStateType = {_STATE: {}, _PG: []}

    # 遍历所有优化器
    for optim in optimizers:
        # 初始化优化器的状态
        _init_optim_state(optim)
        
        # 获取优化器的状态字典
        osd = _state_dict_fn(optim, "state_dict")()
        
        # 如果需要处理FSDP模块
        if info.fsdp_modules:
            # 在FSDP上下文中处理优化器状态
            with info.fsdp_context():
                osd = FSDP.optim_state_dict(model, optim, osd)

            # 特殊处理FlatParameter FSDP
            if not osd:
                continue
            for k in list(osd[_STATE].keys()):
                if "_orig_mod" in k:
                    osd[_STATE][k.replace("_orig_mod.", "")] = osd[_STATE].pop(k)
            for g in osd[_PG]:
                params = [k.replace("_orig_mod.", "") for k in g[_PARAMS]]
                g[_PARAMS] = params
        else:
            # 获取参数列表
            params = list(chain.from_iterable(g[_PARAMS] for g in optim.param_groups))
            param_pid_mapping = dict(zip(params, range(len(params))))
            fqn_pid_mapping = {}
            
            # 遍历模型中的命名参数
            for key, param in model.named_parameters():
                fqns = _get_fqns(model, key)
                assert len(fqns) == 1
                fqn = next(iter(fqns))
                if param not in param_pid_mapping:
                    continue
                pid = param_pid_mapping[param]
                fqn_pid_mapping[fqn] = pid
                fqn_pid_mapping[pid] = fqn

            # 调整状态字典的键名
            for key in list(osd[_STATE].keys()):
                fqn = fqn_pid_mapping[key]
                osd[_STATE][fqn] = osd[_STATE].pop(key)

            # 更新参数组中的参数
            for group in osd[_PG]:
                group[_PARAMS] = [fqn_pid_mapping[pid] for pid in group[_PARAMS]]

        # 如果状态字典为空，则跳过
        if not osd:
            continue

        # 更新优化器状态字典的_STATE部分
        cast(DictValueType, optim_state_dict[_STATE]).update(osd[_STATE])
        # 扩展优化器状态字典的_PG部分
        cast(ListDictValueType, optim_state_dict[_PG]).extend(osd[_PG])

    # 如果需要展平优化器状态字典，则展平处理
    if info.flatten_optimizer_state_dict:
        optim_state_dict = cast(
            OptimizerStateType, _flatten_optim_state_dict(optim_state_dict)
        )

    # 返回可能是完整状态字典或CPU状态字典的结果
    return _maybe_full_or_cpu_state_dict(optim_state_dict, info)


# 从optim_state_dict中提取与给定optim对应的优化器状态字典，并返回结果优化器状态字典
def _split_optim_state_dict(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: OptimizerStateType,
    info: _StateDictInfo,
) -> OptimizerStateType:
    """
    从 ``optim_state_dict`` 中提取与 ``optim`` 对应的优化器状态字典，并返回结果优化器状态字典。
    """
    # 初始化空字典 state 来存储优化器状态字典的键值对
    state: DictValueType = {}
    # 初始化空列表 pg_state 来存储参数组状态字典的列表
    pg_state: ListDictValueType = []
    # 初始化 return_osd 字典，包含 _STATE 和 _PG 两个键
    return_osd: OptimizerStateType = {_STATE: state, _PG: pg_state}
    # 初始化 pg_mapping 字典，用于映射参数组的索引
    pg_mapping: Dict[int, int] = {}

    # 如果 optim_state_dict 的 _STATE 中所有键都是整数，则直接返回 optim_state_dict
    if all(
        isinstance(k, int) for k in cast(DictValueType, optim_state_dict[_STATE]).keys()
    ):
        return optim_state_dict

    # 遍历优化器的 param_groups
    for param_group in optim.param_groups:
        # 在 pg_state 中添加一个空字典，用于存储参数组的参数列表
        pg_state.append({_PARAMS: []})
        # 遍历参数组的参数列表
        for param in param_group[_PARAMS]:
            # 遍历参数的完全限定名（fqn）列表
            for fqn in info.fqn_param_mapping[param]:
                # 如果 fqn 在共享参数映射中，检查其是否在已加载的参数组中
                if fqn in info.shared_params_mapping:
                    in_params = False
                    # 遍历已加载的参数组列表，查找是否有当前 fqn 的存在
                    for loaded_param_group in cast(
                        ListDictValueType, optim_state_dict[_PG]
                    ):
                        if fqn in cast(List[str], loaded_param_group[_PARAMS]):
                            in_params = True
                            break
                else:
                    # 如果 fqn 不在共享参数映射中，则默认为存在
                    in_params = True
                # 如果不在参数列表中，则跳过当前循环
                if not in_params:
                    continue

                # 将当前 fqn 添加到参数组状态的参数列表中
                params = pg_state[-1][_PARAMS]
                assert isinstance(params, list)
                params.append(fqn)
                # 如果参数需要梯度，则从 optim_state_dict 的 _STATE 中复制对应的值到 state 中
                if param.requires_grad:
                    state[fqn] = cast(DictValueType, optim_state_dict[_STATE])[fqn]
                # 更新 pg_mapping，记录加载的参数组的索引
                for loaded_param_group in cast(
                    ListDictValueType, optim_state_dict[_PG]
                ):
                    if fqn in cast(List[str], loaded_param_group[_PARAMS]):
                        pg_mapping[id(loaded_param_group)] = len(return_osd[_PG]) - 1

    # 遍历已加载的参数组列表，根据 pg_mapping 更新 pg_state 中的状态
    for param_group in cast(ListDictValueType, optim_state_dict[_PG]):
        idx = pg_mapping.get(id(param_group), -1)
        if idx == -1:
            continue
        for key, value in param_group.items():
            # 如果键为 _PARAMS，则跳过当前循环
            if key == _PARAMS:
                continue
            # 否则更新 pg_state 中对应索引的值
            pg_state[idx][key] = value

    # 返回更新后的 return_osd 字典，包含优化器的状态信息
    return return_osd
def _load_optim_state_dict(
    model: nn.Module,
    optimizers: Tuple[torch.optim.Optimizer, ...],
    state_dict: OptimizerStateType,
    info: _StateDictInfo,
) -> None:
    if not info.handle_optim:
        return


    # 如果 info 中指示不处理优化器状态，则直接返回，不做任何操作
    if not info.handle_optim:
        return



def get_model_state_dict(
    model: nn.Module,
    *,
    submodules: Optional[Set[nn.Module]] = None,
    options: Optional[StateDictOptions] = None,
) -> Dict[str, ValueType]:
    """
    Return the model state_dict of ``model``.

    See ``get_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        submodules (deprecated): Optional[Set[nn.Module]]: only return the model parameters
            that belong to the submodules.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be returned. See
            `StateDictOptions` for the details.

    Returns:
        The state_dict for ``model``.

    :rtype: typing.Dict[str, ValueType]
    """
    with _gc_context():
        info = _verify_options(
            model,
            tuple(),
            optim_only=False,
            submodules=submodules,
            options=options,
        )
        model_state_dict = _get_model_state_dict(model, info)
        _verify_state_dict(model_state_dict, {}, info)
        return model_state_dict


    # 使用上下文管理器 `_gc_context()` 确保获取模型状态字典期间不会有垃圾对象回收
    with _gc_context():
        # 验证选项和参数，获取与模型相关的信息对象
        info = _verify_options(
            model,
            tuple(),  # 空元组，用于表示没有优化器信息
            optim_only=False,
            submodules=submodules,
            options=options,
        )
        # 获取模型的状态字典
        model_state_dict = _get_model_state_dict(model, info)
        # 验证模型状态字典的正确性
        _verify_state_dict(model_state_dict, {}, info)
        # 返回模型的状态字典
        return model_state_dict



def get_optimizer_state_dict(
    model: nn.Module,
    optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]],
    *,
    submodules: Optional[Set[nn.Module]] = None,
    options: Optional[StateDictOptions] = None,
) -> OptimizerStateType:
    """
    Return the combined state_dict for optimizers.

    See ``get_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[None, Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        submodules (deprecated): Optional[Set[nn.Module]]: only return the model parameters
            that belong to the submodules.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be returned. See
            `StateDictOptions` for the details.

    Returns:
        The state_dict for ``optimizers``.

    :rtype: OptimizerStateType
    """
    with _gc_context():
        # 将单个优化器包装为元组，确保处理时的统一性
        optimizers = (
            (optimizers,)
            if isinstance(optimizers, torch.optim.Optimizer)
            else tuple(optimizers)
        )
        # 验证选项和参数，获取与模型及优化器相关的信息对象
        info = _verify_options(
            model,
            optimizers,
            optim_only=True,
            submodules=submodules,
            options=options,
        )
        # 获取优化器的状态字典
        optim_state_dict = _get_optim_state_dict(model, optimizers, info)
        # 验证优化器状态字典的正确性
        _verify_state_dict({}, optim_state_dict, info)
        # 返回优化器的状态字典
        return optim_state_dict


    # 使用上下文管理器 `_gc_context()` 确保获取优化器状态字典期间不会有垃圾对象回收
    with _gc_context():
        # 如果 optimizers 是单个优化器，则将其包装为元组，保证后续处理的一致性
        optimizers = (
            (optimizers,)
            if isinstance(optimizers, torch.optim.Optimizer)
            else tuple(optimizers)
        )
        # 验证选项和参数，获取与模型及优化器相关的信息对象
        info = _verify_options(
            model,
            optimizers,
            optim_only=True,
            submodules=submodules,
            options=options,
        )
        # 获取优化器的状态字典
        optim_state_dict = _get_optim_state_dict(model, optimizers, info)
        # 验证优化器状态字典的正确性
        _verify_state_dict({}, optim_state_dict, info)
        # 返回优化器的状态字典
        return optim_state_dict



def get_state_dict(
    model: nn.Module,
    optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]],
    *,


    # 函数声明，用于获取模型及其优化器的状态字典
    model: nn.Module,
    optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]],
    *,
    submodules: Optional[Set[nn.Module]] = None,
    options: Optional[StateDictOptions] = None,
def get_state_dict(
    model: nn.Module,
    optimizers: Union[None, Optimizer, Iterable[Optimizer]],
    submodules: Optional[Set[nn.Module]] = None,
    options: StateDictOptions,
) -> Tuple[Dict[str, ValueType], OptimizerStateType]:
    """
    Return the model state_dict and optimizers state_dict.

    ``get_state_dict`` can process any module that is parallelized by PyTorch
    FSDP/fully_shard, DDP/replicate, tensor_parallel/parallelize_module, and any
    combination of these parallelisms. The main functions of ``get_state_dict``
    are: 1.) returning a model and optimizer state_dict that can be resharded
    with a different number of trainers and/or different parallelisms.
    2.) hiding the parallelism-specific state_dict APIs. Users don't have to call
    these APIs.
    3.) sanity checking the result state_dict.

    The keys of the result state dictionary are the canonical FQNs (Fully
    Qualified Names).  A canonical FQN refers to the FQN based on a parameter's
    position in an nn.Module hierarchy. More specifically, a canonical FQN to a
    parameter is the FQN returned by ``module.named_parameters()`` or
    ``module.named_buffers()`` when the module is not distributed by any
    parallelisms. Since the optimizer internally uses parameter IDs to represent
    a parameter, there will be a conversion from the parameter IDs to the
    canonical FQNs when calling this API.

    ``get_state_dict`` can also process a module that is not parallelized. In
    such a case, ``get_state_dict`` only performs one function -- converting the
    optimizer parameter IDs to the canonical FQNs.

    Example:
        >>> # xdoctest: +SKIP
        >>> import torch
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> from torch.distributed.checkpoint.state_dict import get_state_dict

        >>> fsdp_model = FSDP(copy.deepcopy(model))
        >>> fsdp_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> ddp_model = DDP(copy.deepcopy(model))
        >>> ddp_optim = torch.optim.Adam(model.parameters(), lr=1e-3)


        >>> ddp_state_dict, ddp_optim_state_dict = get_state_dict(ddp_model, ddp_optim)
        >>> fsdp_state_dict, fsdp_optim_state_dict = get_state_dict(fsdp_model, fsdp_optim)

        >>> # if we simply call ddp_model.state_dict() and fsdp_model.state_dict(),
        >>> # the asserts will fail.
        >>> assert ddp_state_dict == fsdp_state_dict
        >>> assert ddp_optim_state == fsdp_optim_state_dict


    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[None, Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        submodules (deprecated): Optional[Set[nn.Module]]: only return the model parameters
            that belong to the submodules.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be returned. See
            `StateDictOptions` for the details.

    Returns:
        Tuple[Dict[str, ValueType], OptimizerStateType]: A tuple containing the
        model state dictionary and the optimizers state dictionary.
    """
    # 使用 _gc_context 上下文管理器确保操作后正确的内存回收
    with _gc_context():
        # 如果 optimizers 是单个 Optimizer 对象，则将其转换为包含一个元素的元组
        # 否则，保持其原始类型
        optimizers = (
            (optimizers,)
            if isinstance(optimizers, torch.optim.Optimizer)
            else tuple(optimizers)
        )
        # 调用 _verify_options 函数验证模型和优化器参数的选项
        info = _verify_options(
            model,
            optimizers,
            optim_only=False,
            submodules=submodules,
            options=options,
        )
        # 获取模型的状态字典
        model_state_dict = _get_model_state_dict(model, info)
        # 获取优化器的状态字典
        optim_state_dict = _get_optim_state_dict(model, optimizers, info)
        # 使用 _verify_state_dict 函数验证模型状态字典和优化器状态字典是否匹配
        _verify_state_dict(model_state_dict, optim_state_dict, info)
        # 返回模型状态字典和优化器状态字典的元组作为函数结果
        return model_state_dict, optim_state_dict
def set_optimizer_state_dict(
    optimizer: Optimizer,
    optimizer_state_dict: Dict[str, ValueType],
    *,
    options: Optional[StateDictOptions] = None,
) -> _IncompatibleKeys:
    """Load the optimizer state_dict.

    Load the state_dict of the optimizer. This function is used to load the
    state_dict of the optimizer. The counterpart of ``get_optimizer_state_dict``
    to set the state_dict to the optimizer. See ``set_state_dict`` for detailed usage.

    Args:
        optimizer (Optimizer): the optimizer to load the state_dict.
        optimizer_state_dict (Dict[str, ValueType]):
            the optimizer state_dict to load. If the key of the ``optimizer_state_dict``
            is a str, the key is a parameter group of ``optimizer`` and the value should
            be the state_dict of the parameter group. When loading the state_dict,
            the prefix of the parameter group will be append to the state_dict.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        NamedTuple with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    :type optimizer_state_dict: typing.Dict[str, ValueType]
    """
    optimizer_state_dict: Dict[str, ValueType] = _unflatten_optimizer_state_dict(
        optimizer, optimizer_state_dict
    )
    with _gc_context():
        info = _verify_options(optimizer, tuple(), optim_only=True, options=options)

        _verify_state_dict({}, optimizer_state_dict, info)
        return _load_optimizer_state_dict(optimizer, optimizer_state_dict, info)
    model: nn.Module,
    optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]],
    optim_state_dict: OptimizerStateType,
    *,  # 明确指定下面的参数是关键字参数，不可省略
    options: Optional[StateDictOptions] = None,
# 加载优化器的状态字典。
# 这是 ``get_optimizer_state_dict`` 的对应函数，用于设置优化器的状态字典。
# 查看 ``set_state_dict`` 获取详细使用说明。

def load_optimizer_state_dict(
    model: nn.Module,  # nn.Module 类型的模型
    optimizers: Union[Optimizer, Iterable[Optimizer]],  # 用于优化模型的优化器，可以是单个优化器或优化器的可迭代集合
    optim_state_dict: OptimizerStateType,  # 要加载的优化器状态字典
    options: StateDictOptions  # 控制如何加载模型状态字典和优化器状态字典的选项
) -> None:  # 函数没有返回值

    """Load the optimizers state_dict.

    The counterpart of ``get_optimizer_state_dict`` to set the state_dict to the
    optimizers. See ``set_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        optim_state_dict: OptimizerStateType:
            the optimizer state_dict to load.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        None

    :type optim_state_dict: typing.OptimizerStateType
    """
    
    with _gc_context():  # 使用 _gc_context() 环境管理上下文
        optimizers = (  # 如果 optimizers 是单个优化器，则转换为元组形式
            (optimizers,)
            if isinstance(optimizers, torch.optim.Optimizer)
            else tuple(optimizers)
        )
        info = _verify_options(model, optimizers, optim_only=True, options=options)  # 验证选项，获取相关信息

        _verify_state_dict({}, optim_state_dict, info)  # 验证状态字典，确保符合要求
        _load_optim_state_dict(model, optimizers, optim_state_dict, info)  # 加载优化器状态字典到模型
    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys of the model state_dict.
            * **unexpected_keys** is a list of str containing the unexpected keys of the model state_dict.

    :type model_state_dict: typing.Dict[str, ValueType]
    :type optim_state_dict: typing.OptimizerStateType
    """

    # 将扁平化的模型状态字典重新组织成带有层次结构的字典
    model_state_dict: Dict[str, ValueType] = _unflatten_model_state_dict(
        model, model_state_dict
    )

    # 在垃圾回收上下文中执行以下操作
    with _gc_context():

        # 确保 optimizers 是一个元组
        optimizers = (
            (optimizers,)
            if isinstance(optimizers, torch.optim.Optimizer)
            else tuple(optimizers)
        )

        # 验证模型和优化器的配置选项
        info = _verify_options(
            model, optimizers, optim_only=not model_state_dict, options=options
        )

        # 验证模型状态字典和优化器状态字典的一致性
        _verify_state_dict(model_state_dict, optim_state_dict, info)

        # 加载优化器的状态字典
        _load_optim_state_dict(model, optimizers, optim_state_dict, info)

        # 加载模型的状态字典，并返回加载信息
        return _load_model_state_dict(model, model_state_dict, info)
# TODO: correct the state_dict function signature.
# TODO: this API is not yet fully tested. Make it private
@no_type_check
def _patch_model_state_dict(
    model: nn.Module,
    *,
    options: Optional[StateDictOptions] = None,
) -> None:
    """Patch the ``state_dict`` and ``load_state_dict`` attributes of ``model``.

    Patch the ``state_dict`` and ``load_state_dict`` attributes of ``model`` to
    be a partial function to call ``get_state_dict`` and ``set_state_dict``.

    Example:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.checkpoint.state_dict import patch_model_state_dict

        model = fsdp(model)
        patch_model_state_dict(model)

    Args:
        model (nn.Module): the nn.Module to patch with modified state_dict and load_state_dict.
        options (StateDictOptions): optional options to control state_dict behaviors. See
            `StateDictOptions` for more details.

    Returns:
        None
    """

    _state_dict_call = functools.partial(
        get_model_state_dict,
        model=model,
        options=options,
    )

    def state_dict_call():
        return _state_dict_call()

    # Patch the model's state_dict attribute to invoke _state_dict_call
    model.state_dict = state_dict_call

    _load_state_dict_call = functools.partial(
        set_model_state_dict,
        model=model,
        options=options,
    )

    def load_state_dict_call(state_dict: Dict[str, Any]):
        _load_state_dict_call(model_state_dict=state_dict)

    # Patch the model's load_state_dict attribute to invoke _load_state_dict_call
    model.load_state_dict = load_state_dict_call

    # Add the patched state_dict and load_state_dict functions to _patched_state_dict set
    _patched_state_dict.add(state_dict_call)
    _patched_state_dict.add(load_state_dict_call)


# TODO: correct the load_state_dict function signature.
# TODO: this API is not yet fully tested. Make it private
@no_type_check
def _patch_optimizer_state_dict(
    model: nn.Module,
    *,
    optimizers: Tuple[torch.optim.Optimizer, ...],
    options: Optional[StateDictOptions] = None,
) -> None:
    """Patch the ``state_dict`` and ``load_state_dict`` attributes of ``optimizers``.

    Patch the ``state_dict`` and ``load_state_dict`` attributes of ``optimizers`` to
    be a partial function to call ``get_state_dict`` and ``set_state_dict``.

    Note that if there are multiple optimizers, all of the optimizers will be patched.
    So users only need to call one of the state_dict() to get the full result.

    Example:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.checkpoint.state_dict import patch_model_state_dict

        model = fsdp(model)
        patch_model_state_dict(model)

    Args:
        model (nn.Module): the nn.Module to patch with modified optimizer state_dict and load_state_dict.
        optimizers (Tuple[torch.optim.Optimizer, ...]): tuple of optimizers to patch.
        options (StateDictOptions): optional options to control state_dict behaviors. See
            `StateDictOptions` for more details.

    Returns:
        None
    """
    # 使用 functools.partial 创建一个部分应用函数 _state_dict_call，
    # 该函数用于获取模型和优化器的状态字典，固定了 model、optimizers 和 options 参数
    _state_dict_call = functools.partial(
        get_optimizer_state_dict,
        model=model,
        optimizers=optimizers,
        options=options,
    )

    # 定义 state_dict_call 函数，简单地调用 _state_dict_call 函数并返回其结果
    def state_dict_call():
        return _state_dict_call()

    # 使用 functools.partial 创建一个部分应用函数 _load_state_dict_call，
    # 该函数用于设置模型和优化器的状态字典，固定了 model、optimizers 和 options 参数
    _load_state_dict_call = functools.partial(
        set_optimizer_state_dict,
        model=model,
        optimizers=optimizers,
        options=options,
    )

    # 定义 load_state_dict_call 函数，接受一个名为 state_dict 的字典参数，
    # 并将该字典作为 optim_state_dict 传递给 _load_state_dict_call 函数
    def load_state_dict_call(state_dict: Dict[str, Any]):
        _load_state_dict_call(optim_state_dict=state_dict)

    # 将 state_dict_call 和 load_state_dict_call 函数添加到 _patched_state_dict 集合中
    _patched_state_dict.add(state_dict_call)
    _patched_state_dict.add(load_state_dict_call)

    # 将 optimizers 转换为元组，如果它本身不是 torch.optim.Optimizer 实例的话
    optimizers = (
        (optimizers,)
        if isinstance(optimizers, torch.optim.Optimizer)
        else tuple(optimizers)
    )

    # 遍历 optimizers 元组中的每个优化器对象，并设置其 state_dict 和 load_state_dict 方法
    for optim in optimizers:
        optim.state_dict = state_dict_call  # 设置优化器的状态字典获取方法为 state_dict_call
        optim.load_state_dict = load_state_dict_call  # 设置优化器的状态字典加载方法为 load_state_dict_call
```