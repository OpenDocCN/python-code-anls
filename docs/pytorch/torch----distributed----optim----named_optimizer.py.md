# `.\pytorch\torch\distributed\optim\named_optimizer.py`

```
# mypy: allow-untyped-defs
# 导入 logging 模块
import logging
# 导入 warnings 模块
import warnings
# 从 copy 模块导入 deepcopy 函数
from copy import deepcopy
# 导入类型提示相关模块
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    overload,
    Union,
)

# 导入 PyTorch 相关模块
import torch
import torch.nn as nn
from torch import optim
# 导入分布式相关模块
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 初始化空列表，用于存储导出的名字列表
__all__: List[str] = []

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class _NamedOptimizer(optim.Optimizer):
    """
    ``_NamedOptimizer`` takes a dict of parameters and exposes ``state_dict`` by parameter key.

    We replace the original key (number) in an optim to the
    fully qualified name (FQN) string. User can initialize the optim as they
    initialize a PyTorch optim, the only difference is that they also need to
    pass in the FQN of each parameters.

    Args:
        named_parameters (Mapping[str, Union[torch.Tensor, ShardedTensor]]):
            Mapping from FQN to parameter.
        optimizer_class (optim.Optimizer):
            The class of optimizer to instantiate.
        param_groups (Collection[Mapping[str, Any]]):
            `param_groups` to pass to optimizer if specified.
            The key of the inner map needs to be FQNs.
            Default: None
        module (nn.Module): the module whose parameters to updated
            by the optimizer.
        args: arguments to pass to the optimizer constructor.
        kwargs: arguments to pass to the optimizer constructor.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch import optim
        >>> from torch.distributed.optim import _NamedOptimizer
        >>>
        >>> # Define the named optimizer.
        >>> m = Model(...)
        >>> named_optim = _NamedOptimizer(m.named_parameters(), optim.SGD)
        >>> # Forward pass + backward pass.
        >>> named_optim.step()
        >>> ...
        >>> # Call state_dict for the named optimizer returns a FQN state_dict.
        >>> named_optim.state_dict()

    Warning: This API is still in development and subject to change.

    TODO: Add tutorial for _NamedOptimizer.
    TODO: Add documentation in the docstring for the public attributes
          like self.param_groups and self.named_parameters.
    """

    def __init__(
        self,
        named_parameters: Mapping[str, Union[torch.Tensor, ShardedTensor]],
        optimizer_class: optim.Optimizer,
        param_groups: Optional[Collection[Mapping[str, Any]]] = None,
        module: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        # 调用父类 Optimizer 的初始化方法
        super().__init__()
        # 将传入的参数映射为 FQN 到参数的字典
        self.named_parameters = named_parameters
        # 设置优化器类
        self.optimizer_class = optimizer_class
        # 如果指定了 param_groups，则使用，否则设置为 None
        self.param_groups = param_groups if param_groups is not None else []
        # 设置要更新的模块
        self.module = module
        # 将其他参数和关键字参数传递给优化器的构造函数
        self.args = args
        self.kwargs = kwargs
    ) -> None:
        # 记录一次 API 使用日志，用于分布式优化器命名
        torch._C._log_api_usage_once("torch.distributed.optim._NamedOptimizer")
        
        # 初始化 param_groups，类型为 Collection[Mapping[str, Any]]，用于存储优化器的参数组信息
        self.param_groups: Collection[Mapping[str, Any]] = param_groups  # type: ignore[assignment]
        
        # 检查 param_groups 的有效性
        self._param_groups_check()
        
        # 将 named_parameters 转换为字典形式存储在 self.named_parameters 中
        self.named_parameters = dict(named_parameters)
        
        # 根据传入的 param_groups 确定用于优化器的参数列表
        params_for_optimizer = (
            self.named_parameters.values() if param_groups is None else param_groups
        )
        
        # 创建优化器对象 self._optimizer，根据传入的 optimizer_class 和参数 *args, **kwargs
        self._optimizer = optimizer_class(  # type: ignore[operator]
            params_for_optimizer,
            *args,
            **kwargs,
        )
        
        # 将 module 赋值给 self.module
        self.module = module
        
        # 如果 param_groups 为 None，则使用 self.named_parameters 的键列表作为 ordered_param_keys
        if param_groups is None:
            self.ordered_param_keys = list(self.named_parameters.keys())
        else:
            # 如果传入了 param_groups，则给出警告，因为会忽略 module 的所有参数
            warnings.warn(
                "Since we pass in param_groups, we will use param_groups to "
                "initialize the optimizer, not all parameters of the module."
            )
            
            # 根据 param_groups 构建有序的参数键列表 ordered_param_keys
            param_to_key = {param: key for key, param in self.named_parameters.items()}  # type: ignore[misc, has-type]
            ordered_param_keys = []
            for group in param_groups:
                for param in group["params"]:
                    if param not in param_to_key:
                        raise ValueError(
                            f"Expect param name {param} found in param group but is missing."
                        )
                    ordered_param_keys.append(param_to_key[param])
            self.ordered_param_keys = ordered_param_keys
        
        # 更新 self.param_groups，从 self._optimizer 中获取最新的 param_groups
        self.param_groups = self._optimizer.param_groups

    def _param_groups_check(self):
        # 检查 self.param_groups 的有效性
        if self.param_groups is not None:
            for param_group in self.param_groups:
                assert isinstance(param_group, dict), "param group must be a dict"
                assert "params" in param_group, "param group must contain key params"
                params = param_group["params"]
                
                # 如果 params 是单个张量，则转换为列表形式
                if isinstance(params, torch.Tensor):
                    params = [params]
                    
                params = list(params)
                
                # 检查每个 param 是否为 Tensor 类型
                for param in params:
                    if not isinstance(param, torch.Tensor):
                        raise TypeError(
                            "optimizer can only optimize Tensors, "
                            "but one of the params is " + torch.typename(param)
                        )
                
                # 更新 param_group 中的 params
                param_group["params"] = params
    def state_dict(self) -> Dict[str, Any]:
        """
        Return the ``state_dict`` of the optimizer.

        Instead of using number to index
        parameters, we will use module fully qualified name (FQN) as the key.
        """
        # 获取优化器的状态字典
        state_dict = self._optimizer.state_dict()
        # 获取参数组列表
        param_groups = state_dict["param_groups"]

        # 构建返回的状态字典
        ret_state = {
            self.ordered_param_keys[st_key]: state_val
            for st_key, state_val in state_dict["state"].items()
        }

        # 构建返回的参数组列表
        ret_groups = []
        for group in param_groups:
            param_keys = []
            # 遍历每个参数，使用其对应的键（通过 ordered_param_keys）进行索引
            for param in group["params"]:
                param_keys.append(self.ordered_param_keys[param])
            # 构建每个参数组的返回结构
            ret_group = {"params": sorted(param_keys)}
            for k, v in group.items():
                if k != "params":
                    ret_group[k] = deepcopy(v)
            ret_groups.append(ret_group)

        # 返回优化器状态字典的后处理结果
        return self._post_state_dict({"state": ret_state, "param_groups": ret_groups})

    @overload
    def step(self, closure: None = ...) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        This will call :meth:`torch.optim.Optimizer.step` on the wrapped
        optimizer.
        """
        # 执行单步优化操作，调用包装优化器的 step 方法
        return self._optimizer.step(closure=closure)

    @property
    def state(self) -> Mapping[torch.Tensor, Any]:  # type: ignore[override]
        # 返回优化器的状态属性
        return self._optimizer.state

    def add_param_group(self, param_group: Mapping[str, Any]) -> None:
        """
        Add a param group to the :class:`_NamedOptimizer` s `param_groups`.

        Warning: This API is still in development and subject to change.
        """
        # 断言参数组必须是字典类型
        assert isinstance(param_group, dict), "param group must be a dict"

        # 将参数组的 params 字段转换为列表
        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            param_group["params"] = [params]
        else:
            param_group["params"] = list(params)

        # 创建参数到键的映射
        param_to_key = {param: key for key, param in self.named_parameters.items()}  # type: ignore[misc, has-type]
        # 遍历参数组的每个参数，确保每个参数都在命名参数中
        for param in param_group["params"]:
            if param not in param_to_key:
                raise ValueError("some parameters are not in the module")
            # 将每个参数的键添加到 ordered_param_keys 列表中
            self.ordered_param_keys.append(param_to_key[param])

        # 向包装的优化器中添加参数组
        self._optimizer.add_param_group(param_group)
        # 更新本地的 param_groups 属性为优化器中的 param_groups
        self.param_groups = self._optimizer.param_groups
    # 初始化模型的优化器状态
    def init_state(self) -> None:
        """
        Run a dummy optimizer step, which allows to initialize optimizer state because we do lazy init for most optimizers.

        This allows doing in-place loading of optimizer state from a checkpoint.
        """
        # 遍历模型的所有参数
        for param in self.named_parameters.values():
            # 如果参数需要梯度计算
            if param.requires_grad:
                # 创建一个与参数形状相同的全零张量
                t = torch.zeros_like(param)
                # 将全零张量封装成一个梯度变量，赋给参数的梯度
                param.grad = torch.autograd.Variable(t)
        # 调用模型的 `step` 方法，加载优化器的初始状态
        # 此处调用是为了确保优化器状态的加载
        self.step(closure=None)

    # 准备加载状态字典之前的预处理函数
    def _pre_load_state_dict(self, state_dict) -> Dict[str, Any]:
        # TODO(chienchin): This API should be FSDP agnostic and should support
        # general user hooks.
        # 如果模型的子模块是 FSDP 类型
        if isinstance(self.module, FSDP):
            # 调用 FSDP 类的方法，将优化器状态字典转换为加载状态的格式
            return FSDP.optim_state_dict_to_load(
                self.module, self._optimizer, state_dict, is_named_optimizer=True
            )
        # 如果不是 FSDP 类型，则直接返回原始状态字典
        return state_dict

    # 加载状态字典之后的后处理函数
    def _post_state_dict(self, state_dict) -> Dict[str, Any]:
        # TODO(chienchin): This API should be FSDP agnostic and should support
        # general user hooks.
        # 如果模型的子模块是 FSDP 类型
        if isinstance(self.module, FSDP):
            # 将当前模型的优化器状态字典设置为给定的状态字典
            FSDP.optim_state_dict(self.module, self._optimizer, state_dict)
        # 返回更新后的状态字典
        return state_dict
# 定义一个函数 _gen_param_group_key，用于生成参数组的唯一标识符
def _gen_param_group_key(param_keys: List[str]) -> str:
    # 将传入的参数键列表按字母顺序排序，并用 '/' 连接成一个字符串作为唯一标识符
    return "/".join(sorted(param_keys))
```