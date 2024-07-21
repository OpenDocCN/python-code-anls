# `.\pytorch\torch\distributed\fsdp\_wrap_utils.py`

```py
# 添加类型检查允许未类型化的定义
# 导入标准库模块
import collections
import functools
import inspect
import warnings
# 从 functools 模块导入 partial 函数
from functools import partial
# 导入类型提示相关内容
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union

# 导入 PyTorch 的 nn 模块
import torch.nn as nn
# 从 torch.distributed.fsdp._common_utils 模块中导入特定函数
from torch.distributed.fsdp._common_utils import (
    _get_module_fsdp_state,
    _override_module_mixed_precision,
)
# 从 torch.distributed.fsdp.wrap 模块中导入特定函数
from torch.distributed.fsdp.wrap import (
    _construct_wrap_fn,
    _or_policy,
    _Policy,
    _post_order_apply,
    _recursive_wrap,
    _run_mixed_precision_override_policy,
    _wrap_module_cls_individually,
)

# 定义函数 _auto_wrap，自动根据指定策略对 root_module 中的模块进行包装
def _auto_wrap(
    root_module: nn.Module,
    policy: Union[Callable, _Policy],
    ignored_modules: Set[nn.Module],
    ignored_params: Set[nn.Parameter],
    root_kwargs: Dict[str, Any],
    fsdp_fn: Callable,  # e.g. `FullyShardedDataParallel` or `fully_shard`
):
    """
    Auto wraps modules in ``root_module`` 's tree according to ``policy``
    following a post-order traversal.

    Precondition: ``root_kwargs`` should contain all arguments except
    ``module``. This function accepts the kwargs dict directly since it gets
    forwarded into the post-order traversal function.
    """
    # 从 root_kwargs 中获取 mixed_precision 参数
    mixed_precision = root_kwargs["mixed_precision"]
    # 检查 fsdp_fn 是否是类对象
    is_wrapper = inspect.isclass(fsdp_fn)
    # 检查是否有嵌套包装，如果有则引发异常
    _check_nested_wrapping(root_module)

    # 如果 policy 是 _Policy 的实例
    if isinstance(policy, _Policy):
        # 根据 is_wrapper 设置相应的参数键名
        root_kwargs["auto_wrap_policy" if is_wrapper else "policy"] = None
        # 运行指定策略，获取目标模块到参数的映射
        target_module_to_kwargs = policy._run_policy(
            root_module, ignored_modules, root_kwargs
        )
        # 如果 mixed_precision 不为 None，则运行混合精度模块覆盖策略
        if mixed_precision is not None:
            target_module_to_kwargs = _run_mixed_precision_override_policy(
                root_module,
                mixed_precision._module_classes_to_ignore,
                ignored_modules,
                root_kwargs,
                target_module_to_kwargs,
            )
            # 覆盖指定模块类的混合精度
            overridden_module_classes = _override_module_mixed_precision(
                root_module, mixed_precision._module_classes_to_ignore
            )
            # 在混合精度被覆盖时发出警告
            _warn_on_overridden_mixed_precision(overridden_module_classes)
        
        # 从 root_kwargs 中获取 use_orig_params 参数，默认为 False
        use_orig_params = root_kwargs.get("use_orig_params", False)
        # 验证冻结参数的有效性
        _validate_frozen_params(
            root_module,
            set(target_module_to_kwargs.keys()),
            ignored_params,
            use_orig_params,
        )
        # 构造包装函数，根据目标模块和参数映射以及给定的 fsdp_fn
        wrap_fn = _construct_wrap_fn(root_module, target_module_to_kwargs, fsdp_fn)
        # 对 root_module 进行后序遍历应用包装函数
        _post_order_apply(root_module, wrap_fn)
        return
    
    # 如果 policy 不是 _Policy 的实例，则执行以下代码
    # 配置递归包装所需的参数
    recursive_wrap_kwargs = {
        "module": root_module,
        "auto_wrap_policy": policy,
        "wrapper_cls": fsdp_fn,
        "ignored_modules": ignored_modules,
        "ignored_params": ignored_params,
        "only_wrap_children": True,
    }
    # 如果 mixed_precision 参数不是 None，则执行以下逻辑
    if mixed_precision is not None:
        # 将被忽略类型的模块分别包装，并注册前向钩子，以将其转换为 fp32，然后再转回原始数据类型
        overridden_module_classes = _override_module_mixed_precision(
            root_module, mixed_precision._module_classes_to_ignore
        )
        # 定义一个新的策略函数，结合原有的策略和单独包装被忽略模块类型的逻辑
        policy = functools.partial(
            _or_policy,
            policies=[
                policy,
                partial(
                    _wrap_module_cls_individually,
                    module_classes=mixed_precision._module_classes_to_ignore,
                ),
            ],
        )
        # 将自动包装策略添加到递归包装的参数中
        recursive_wrap_kwargs["auto_wrap_policy"] = policy
        # 在控制台上发出警告，提示有哪些模块被覆盖为混合精度
        _warn_on_overridden_mixed_precision(overridden_module_classes)
    # 执行递归包装的函数，传入递归包装的参数和根参数，忽略类型为 ignore[arg-type] 的类型错误
    _recursive_wrap(**recursive_wrap_kwargs, **root_kwargs)  # type: ignore[arg-type]
# 检查给定根模块下所有命名模块是否已经应用了 FSDP，如果有则抛出错误
def _check_nested_wrapping(root_module: nn.Module):
    # 遍历根模块的命名模块及其名称
    for module_name, module in root_module.named_modules():
        # 检查当前模块是否已经应用了 FSDP
        if _get_module_fsdp_state(module) is not None:
            # 如果已经应用了 FSDP，则抛出值错误
            raise ValueError(
                "FSDP auto wrapping requires modules to not already have "
                f"FSDP applied but found {module_name} in\n{root_module}"
            )


# 在混合精度和自动包装策略都已指定给 FSDP 的情况下，警告有重写的模块类型
def _warn_on_overridden_mixed_precision(
    overridden_module_classes: Set[Type[nn.Module]],
):
    # 如果没有重写的模块类型，则直接返回
    if len(overridden_module_classes) == 0:
        return
    # 发出警告，指出同时指定了混合精度和自动包装策略给 FSDP，以及模块的子模块类型
    warnings.warn(
        "Both mixed precision and an auto_wrap_policy were specified to FSDP, "
        f"where the wrapped module has submodules of type:\n{overridden_module_classes}\n"
        "These modules will be wrapped as separate FSDP instances with mixed "
        "precision disabled."
    )


# 验证冻结参数的有效性
def _validate_frozen_params(
    root_module: nn.Module,
    modules_to_wrap: Set[nn.Module],
    ignored_params: Set[nn.Parameter],
    use_orig_params: bool,
):
    """
    检查给定的 ``modules_to_wrap`` 中每个模块是否统一管理冻结或非冻结的参数。
    对于 ``use_orig_params=False`` 是严格的错误检查，对于 ``use_orig_params=True`` 是强烈建议的用户警告。
    """
    # 获取根模块的后序命名模块列表
    post_order_named_modules = _get_post_order_named_modules(root_module)
    # 记录已访问过的模块集合
    visited_modules: Set[nn.Module] = set()
    # 遍历 post_order_named_modules 中的每个元素，每个元素是一个元组 (module_name, module)
    for module_name, module in post_order_named_modules:
        # 检查当前 module 是否需要包装
        if module in modules_to_wrap:
            # 获取当前 module 的参数到全限定名的映射
            param_to_fqn = _get_managed_param_to_fqn(
                module, ignored_params, visited_modules, module_name
            )
            # 初始化冻结参数的全限定名列表和数量
            frozen_param_fqns: List[str] = []
            frozen_param_numel = 0
            # 初始化非冻结参数的全限定名列表和数量
            nonfrozen_param_fqns: List[str] = []
            nonfrozen_param_numel = 0
            
            # 遍历 param_to_fqn 字典中的每个项，param 是参数，fqn 是全限定名
            for param, fqn in param_to_fqn.items():
                # 检查参数是否需要梯度计算
                if param.requires_grad:
                    # 将非冻结参数的全限定名和元素数量添加到对应列表中
                    nonfrozen_param_fqns.append(fqn)
                    nonfrozen_param_numel += param.numel()
                else:
                    # 将冻结参数的全限定名和元素数量添加到对应列表中
                    frozen_param_fqns.append(fqn)
                    frozen_param_numel += param.numel()
            
            # 如果存在既有 requires_grad=True 又有 requires_grad=False 的参数
            if len(frozen_param_fqns) > 0 and len(nonfrozen_param_fqns) > 0:
                # 构建警告或错误信息字符串
                msg = f"{module_name} has both parameters with requires_grad=True and False."
                # 如果 use_orig_params=True，添加关于内存使用的建议信息
                if use_orig_params:
                    total_param_numel = frozen_param_numel + nonfrozen_param_numel
                    msg += (
                        " We do not recommend wrapping such modules since "
                        "the gradient memory usage will be higher than expected "
                        f"({total_param_numel} numel instead of {nonfrozen_param_numel} numel "
                        "before sharding via reduce-scatter). "
                    )
                # 如果 use_orig_params=False，添加不支持包装的信息
                else:
                    msg += " FSDP does not support wrapping such modules when use_orig_params=False. "
                # 添加关于冻结和非冻结参数的详细信息
                msg += "If possible, wrap the frozen parameters with FSDP separately.\n"
                msg += (
                    f"The following parameters have requires_grad=True:\n{nonfrozen_param_fqns}\n"
                    f"The following parameters have requires_grad=False:\n{frozen_param_fqns}"
                )
                # 如果 use_orig_params=True，发出警告
                if use_orig_params:
                    warnings.warn(msg)
                # 如果 use_orig_params=False，引发值错误
                else:
                    raise ValueError(msg)
def _get_post_order_named_modules(
    root_module: nn.Module,
) -> List[Tuple[str, nn.Module]]:
    """
    返回一个按照后序遍历顺序排列的命名模块列表，这是一种有效的逆拓扑排序方法。
    我们通过使用栈的深度优先搜索的逆序来实现这一点，而不是翻转``root_module.named_modules()``
    的结果，因为前者在模块树的每个级别上按照注册顺序给出模块（与逆序相反），这允许我们在
    第一个违反条件的注册模块时报错/警告。

    例如，考虑以下模块结构：
        M(
          S1(),
          S2(
            SS1(),
            SS2(),
          ),
          S3(),
        )
    逆序的DFS顺序是 [S1, SS1, SS2, S2, S3, M]，而逆``named_modules()``顺序是 [S3, SS2, SS1, S2, S1, M]。
    """
    visited_modules = {root_module}  # 记录已访问的模块，初始为根模块
    stack = [("", root_module)]  # 使用栈进行深度优先搜索，初始将根模块入栈
    reverse_post_order_named_modules: List[Tuple[str, nn.Module]] = []  # 存储逆序的命名模块列表
    while stack:
        module_name, module = stack.pop()  # 弹出栈顶的模块名和模块对象
        reverse_post_order_named_modules.append((module_name, module))  # 将模块名和模块对象添加到逆序列表中
        for child_module_name, child_module in module.named_children():
            if child_module is None:  # 如果子模块为None，则跳过（仅用于`named_children()`的覆盖）
                continue
            if child_module not in visited_modules:
                visited_modules.add(child_module)  # 将子模块添加到已访问集合中
                if module_name != "":
                    child_module_name = module_name + "." + child_module_name  # 更新子模块名，加上父模块名前缀
                stack.append((child_module_name, child_module))  # 将更新后的子模块名和子模块对象入栈
    post_order_named_modules = list(reversed(reverse_post_order_named_modules))  # 将逆序的命名模块列表翻转得到后序命名模块列表
    return post_order_named_modules  # 返回后序命名模块列表


def _get_managed_param_to_fqn(
    module_to_wrap: nn.Module,
    ignored_params: Set[nn.Parameter],
    visited_modules: Set[nn.Module],
    root_prefix: str,
) -> Dict[nn.Parameter, str]:
    """
    返回一个字典，将模块``module_to_wrap``中管理的参数映射到其完全限定名（FQN）。
    字典的键正是该模块中由函数管理的参数，这通过在逆拓扑顺序上调用该函数来实现，破坏性地更新
    ``visited_modules``，并且不会进入那些模块。FQN从根模块通过``root_prefix``前缀更具信息性。

    注意：此函数用于预包装时和在逆拓扑顺序上迭代调用，以覆盖整个模块树。这与``_get_param_to_fqn()``
    函数不同，后者用于后包装并在一个步骤中处理整个模块树。鉴于这些差异，我们不试图统一这两者。
    """
    param_to_fqn: Dict[nn.Parameter, str] = {}  # 初始化参数到FQN的映射字典
    # 运行BFS（或任何树遍历方法都可以）
    queue = collections.deque([(module_to_wrap, root_prefix)])  # 使用双端队列初始化队列，包含根模块和根前缀
    visited_modules.add(module_to_wrap)  # 将根模块添加到已访问集合中
    # 当队列不为空时，循环处理队列中的每个元素
    while queue:
        # 从队列中取出一个模块和其前缀
        module, prefix = queue.popleft()
        
        # 遍历模块中的命名参数，不递归查找子模块的参数
        for param_name, param in module.named_parameters(recurse=False):
            # 如果参数不在被忽略的参数列表中，则处理
            if param not in ignored_params:
                # 构建完全限定名称（Fully Qualified Name, FQN）
                fqn = param_name if prefix == "" else prefix + "." + param_name
                # 将参数和其完全限定名称映射存储到字典中
                param_to_fqn[param] = fqn
        
        # 遍历模块中的子模块
        for child_module_name, child_module in module.named_children():
            # 如果子模块为None，通常用于覆盖`named_children()`方法时
            if child_module is None:
                continue
            # 如果子模块尚未访问过，则处理
            if child_module not in visited_modules:
                # 将子模块添加到已访问模块集合中
                visited_modules.add(child_module)
                # 构建子模块的前缀名称
                child_prefix = (
                    child_module_name
                    if prefix == ""
                    else prefix + "." + child_module_name
                )
                # 将子模块及其前缀作为元组添加到队列中，以便进一步处理
                queue.append((child_module, child_prefix))
    
    # 返回参数到完全限定名称的映射字典
    return param_to_fqn
```