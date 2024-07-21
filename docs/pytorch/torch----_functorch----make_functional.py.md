# `.\pytorch\torch\_functorch\make_functional.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类
import copy  # 导入 copy 模块，用于对象的浅拷贝和深拷贝操作
from typing import (  # 导入 typing 模块，用于类型提示
    Any,  # 任意类型
    Callable,  # 可调用对象
    Dict,  # 字典类型
    Iterable,  # 可迭代对象
    List,  # 列表类型
    NoReturn,  # 永不返回类型
    Sequence,  # 序列类型
    Tuple,  # 元组类型
    Type,  # 类型对象
    Union,  # 联合类型
)

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
from torch import Tensor  # 导入 PyTorch 中的 Tensor 类型
from torch.nn.utils._named_member_accessor import NamedMemberAccessor  # 导入命名成员访问器

# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.

# 定义函数，抛出参数绑定错误的异常
def raise_parameter_tying_error() -> NoReturn:
    raise RuntimeError(
        "make_functional(module): we don't yet support models that "
        "do parameter tying (also sometimes known as weight sharing). "
        "Please try to rewrite your model by replacing all instances of the "
        "tied parameter with another and/or comment your support in "
        "https://github.com/pytorch/functorch/issues/446"
    )


# 创建名字映射的函数
def create_names_map(
    named_params: Union[Dict[str, Tensor], Iterable[Tuple[str, Tensor]]],
    tied_named_params: Union[Dict[str, Tensor], Iterable[Tuple[str, Tensor]]],
) -> Dict[str, List[str]]:
    """
    named_params is a dictionary of tensors: {'A': A, 'B': B}
    tied_named_params is another dictionary of tensors {'A': A, 'B': B, 'B_tied': B}
    with potentially tied (or 'duplicated') tensors

    This function creates a mapping from the names in named_params to the
    names in tied_named_params: {'A': ['A'], 'B': ['B', 'B_tied']}.
    """
    named_params = dict(named_params)  # 转换为字典格式
    tied_named_params = dict(tied_named_params)  # 转换为字典格式

    tensors_dict_keys = set(named_params.keys())  # 获取 named_params 的键集合
    tied_tensors_dict_keys = set(tied_named_params.keys())  # 获取 tied_named_params 的键集合
    assert tensors_dict_keys.issubset(tied_tensors_dict_keys)  # 断言确保 named_params 的所有键都在 tied_named_params 中

    tensor_to_mapping: Dict[Tensor, Tuple[str, List[str]]] = {}  # 初始化存储张量映射关系的字典
    for key, tensor in named_params.items():
        tensor_to_mapping[tensor] = (key, [])  # 将每个张量与其对应的名字初始化为空列表
    for key, tensor in tied_named_params.items():
        assert tensor in tensor_to_mapping  # 断言确保 tied_named_params 中的张量在 tensor_to_mapping 中存在
        tensor_to_mapping[tensor][1].append(key)  # 将 tied_named_params 中的张量名添加到对应张量的列表中
    return dict(tensor_to_mapping.values())  # 返回张量映射关系字典的值列表的字典形式


# 提取模块中的成员函数
def _extract_members(
    mod: nn.Module,
    named_members: Callable[..., Iterable[Tuple[str, Tensor]]],
    subclass: Callable[[Tensor], Tensor],
) -> Tuple[Tuple[Tensor, ...], Tuple[str, ...], Dict[str, List[str]]]:
    all_named_members = tuple(named_members(remove_duplicate=False))  # 获取所有命名成员（未去重）
    unique_named_members = tuple(named_members(remove_duplicate=True))  # 获取唯一命名成员（已去重）
    names_map = create_names_map(unique_named_members, all_named_members)  # 创建命名映射关系

    # Remove all the members in the model
    memo = {}  # 初始化备忘录字典
    accessor = NamedMemberAccessor(mod)  # 使用模块初始化命名成员访问器对象
    # 遍历所有命名成员的列表，其中每个元素是一个元组 (name, p)
    for name, p in all_named_members:
        # 如果当前成员 p 不在 memo 字典中
        if p not in memo:
            # 使用 subclass 函数创建 p 的副本，并存入 memo 字典
            memo[p] = subclass(torch.empty_like(p, device="meta"))
        # 获取 memo 中存储的 p 的副本
        replacement = memo[p]
        # 使用 accessor 对象将名为 name 的张量替换为 replacement
        accessor.set_tensor(name, replacement)

    # 如果 unique_named_members 列表为空
    if len(unique_named_members) == 0:
        # 初始化空的元组 names 和 params
        names, params = (), ()
    else:
        # 从 unique_named_members 中解压出 names 和 params 列表
        names, params = zip(*unique_named_members)  # type: ignore[assignment]
        
    # 返回 params（参数列表）、names（参数名列表）、names_map（参数名映射）
    return params, names, names_map
def extract_weights(
    mod: nn.Module,
) -> Tuple[Tuple[Tensor, ...], Tuple[str, ...], Dict[str, List[str]]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    # 调用 _extract_members 函数提取模型中的参数
    return _extract_members(mod, mod.named_parameters, nn.Parameter)


def extract_buffers(
    mod: nn.Module,
) -> Tuple[Tuple[Tensor, ...], Tuple[str, ...], Dict[str, List[str]]]:
    # 调用 _extract_members 函数提取模型中的缓冲区
    return _extract_members(mod, mod.named_buffers, lambda x: x)


def load_weights(
    mod: nn.Module,
    names: Sequence[str],
    params: Sequence[Tensor],
    as_params: bool = False,
) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    # 创建 NamedMemberAccessor 对象，用于加载权重到模型中
    accessor = NamedMemberAccessor(mod)
    # 如果指定将 params 转换为 nn.Parameter 对象，则进行转换
    if as_params:
        params = [nn.Parameter(p) for p in params]
    # 使用 accessor 将 names 和 params 设置到模型中
    accessor.set_tensors(names, params)


def _swap_state(
    mod: nn.Module, names_map: Dict[str, List[str]], elems: Iterable[Tensor]
) -> List[Tensor]:
    # 创建 NamedMemberAccessor 对象，用于交换模型状态
    result: List[Tensor] = []
    accessor = NamedMemberAccessor(mod)
    # 遍历 names_map 中的每个元素和对应的 elems 中的元素
    for (_, attr_names), elem in zip(names_map.items(), elems):
        # 对于每个 attr_name，根据索引决定是交换还是设置新的值
        for i, attr_name in enumerate(attr_names):
            if i == 0:
                # 对第一个 attr_name 执行交换操作，并将结果添加到 result 中
                result.append(accessor.swap_tensor(attr_name, elem))
            else:
                # 对后续的 attr_name 设置新的 tensor 值
                accessor.set_tensor(attr_name, elem)
    return result


def load_buffers(
    mod: nn.Module,
    names: Sequence[str],
    buffers: Sequence[Tensor],
    as_params: bool = False,
) -> None:
    # 创建 NamedMemberAccessor 对象，用于加载缓冲区到模型中
    accessor = NamedMemberAccessor(mod)
    # 使用 accessor 将 names 和 buffers 设置到模型中
    accessor.set_tensors(names, buffers)


def load_state(
    model: nn.Module,
    weights: Sequence[Tensor],
    weight_names: Sequence[str],
    buffers: Sequence[Tensor] = (),
    buffer_names: Sequence[str] = (),
) -> nn.Module:
    """load_state(model, weights, weight_names, buffers=(), buffer_names=()) -> model

    load_state takes `weights` and `buffers` and assigns them to the model.
    This is the inverse operation of `make_functional_deprecated_v1`.
    """
    # 断言权重和权重名列表长度一致
    assert len(weight_names) == len(weights)
    # 调用 load_weights 将权重加载到模型中
    load_weights(model, weight_names, weights)
    # 如果存在缓冲区，则断言缓冲区和缓冲区名列表长度一致
    if len(buffers) > 0:
        assert len(buffer_names) == len(buffers)
        # 调用 load_buffers 将缓冲区加载到模型中
        load_buffers(model, buffer_names, buffers)
    return model


def make_functional_deprecated_v1(model: nn.Module):
    """make_functional_deprecated_v1(model) -> weights, func, weight_names

    Given an nn.Module, make_functional_deprecated_v1 extracts the state (weights)
    and returns a functional version of the model, `func`. This makes
    it so that it is possible use transforms over the parameters of
    """
    # 函数的实现未完整给出，仅有部分注释
    # 从给定的模型中提取权重、描述符和不需要的信息
    def make_functional_deprecated_v1(model):
        # 获取模型中的所有缓冲区（buffers）
        buffers = list(model.buffers())
        # 如果模型有缓冲区（buffers），则抛出运行时异常
        if len(buffers) > 0:
            raise RuntimeError(
                "make_functional_deprecated_v1(model): `model` has buffers. Please use "
                "make_functional_with_buffers_deprecated_v1(model) instead."
            )
        # 提取模型的权重、描述符和不需要的信息
        weights, descriptors, _ = extract_weights(model)
    
        # 定义一个函数，接受权重和数据作为输入，并返回经过加载权重后的模型的输出
        def fun(weights, data):
            # 深拷贝模型，确保不会修改原始模型的状态
            mutable_model = copy.deepcopy(model)
            # 加载权重到深拷贝的模型中
            load_weights(mutable_model, descriptors, weights)
            # 调用深拷贝模型并传入数据，返回模型的输出
            return mutable_model(*data)
    
        # 返回权重、处理函数和描述符
        return weights, fun, descriptors
def make_functional_with_buffers_deprecated_v1(model: nn.Module):
    """make_functional_with_buffers_deprecated_v1(model) -> weights, buffers, func, weight_names, buffer_names

    Given an nn.Module, make_functional_with_buffers_deprecated_v1 extracts the state (weights and buffers)
    and returns a functional version of the model, `func`.

    `func` can be invoked as follows:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, buffers, func, _, _ = make_functional_with_buffers_deprecated_v1(model)
    func(weights, buffers, (x,))
    ```

    And here is an example of applying the grad transform:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, buffers, func, _, _ = make_functional_with_buffers_deprecated_v1(model)
    func(weights, buffers, (x,))
    grad_weights = grad(func)(weights, buffers, (x,))
    ```

    To put the state back into a model, use `load_state`.
    """
    # Extract weights and their descriptors from the model
    weights, weight_descriptors, _ = extract_weights(model)
    # Extract buffers and their descriptors from the model
    buffers, buf_descriptors, _ = extract_buffers(model)

    # Define a functional version of the model that uses the extracted weights and buffers
    def fun(weights, buffers, data):
        # Create a deep copy of the original model
        mutable_model = copy.deepcopy(model)
        # Load the extracted weights into the mutable model
        load_weights(mutable_model, weight_descriptors, weights)
        # Load the extracted buffers into the mutable model
        load_buffers(mutable_model, buf_descriptors, buffers)
        # Return the output of the mutable model when invoked with the input data
        return mutable_model(*data)

    # Return the extracted weights, buffers, functional model, weight descriptors, and buffer descriptors
    return weights, buffers, fun, weight_descriptors, buf_descriptors


class FunctionalModuleWithBuffers(nn.Module):
    """
    This is the callable object returned by :func:`make_functional_with_buffers`.
    """

    def __init__(
        self,
        stateless_model: nn.Module,
        param_names: Tuple[str, ...],
        buffer_names: Tuple[str, ...],
        param_names_map: Dict[str, List[str]],
        buffer_names_map: Dict[str, List[str]],
    ) -> None:
        super().__init__()
        # Store the stateless model, parameter names, and buffer names
        self.stateless_model = stateless_model
        self.param_names = param_names
        self.buffer_names = buffer_names

        # Create a dictionary mapping all names from both parameters and buffers
        self.all_names_map = dict(param_names_map)
        self.all_names_map.update(buffer_names_map)

    @staticmethod
    def _create_from(
        model: nn.Module, disable_autograd_tracking: bool = False
    ) -> Tuple["FunctionalModuleWithBuffers", Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        # Create a deep copy of the model
        model_copy = copy.deepcopy(model)
        # Extract weights, their names, and their mapping from the copied model
        params, param_names, param_names_map = extract_weights(model_copy)
        # Extract buffers, their names, and their mapping from the copied model
        buffers, buffer_names, buffer_names_map = extract_buffers(model_copy)
        
        # Disable autograd tracking if specified
        if disable_autograd_tracking:
            for param in params:
                param.requires_grad_(False)
        
        # Return a FunctionalModuleWithBuffers instance initialized with the copied model and extracted information
        return (
            FunctionalModuleWithBuffers(
                model_copy, param_names, buffer_names, param_names_map, buffer_names_map
            ),
            params,
            buffers,
        )

    def forward(
        self, params: Iterable[Tensor], buffers: Iterable[Tensor], *args, **kwargs
    ) -> Any:
        # 将状态临时加载到 self.stateless_model 上
        old_state = _swap_state(
            self.stateless_model,
            self.all_names_map,
            tuple(params) + tuple(buffers),
        )
        try:
            # 调用 self.stateless_model 并传递参数
            return self.stateless_model(*args, **kwargs)
        finally:
            # 恢复加载前的状态到 self.stateless_model 上
            _swap_state(self.stateless_model, self.all_names_map, old_state)
# 定义一个可调用对象的类，用于将模型转换为函数式模型
class FunctionalModule(nn.Module):
    """
    This is the callable object returned by :func:`make_functional`.
    """

    def __init__(
        self,
        stateless_model: nn.Module,
        param_names: Tuple[str, ...],
        names_map: Dict[str, List[str]],
    ) -> None:
        super().__init__()
        # 初始化函数式模型对象，包括无状态模型、参数名称元组和名称映射字典
        self.stateless_model = stateless_model
        self.param_names = param_names
        self.names_map = names_map

    @staticmethod
    def _create_from(
        model: nn.Module, disable_autograd_tracking: bool = False
    ) -> Tuple["FunctionalModule", Tuple[Tensor, ...]]:
        # TODO: We don't need to copy the model to create a stateless copy
        # 创建模型的深层副本，以便生成一个无状态模型的副本
        model_copy = copy.deepcopy(model)
        # 提取模型副本的权重参数、参数名称和名称映射
        params, param_names, names_map = extract_weights(model_copy)
        # 如果禁用自动梯度追踪，则关闭参数的梯度追踪
        if disable_autograd_tracking:
            for param in params:
                param.requires_grad_(False)
        # 返回一个包含无状态模型副本、参数元组的元组
        return FunctionalModule(model_copy, param_names, names_map), params

    def forward(self, params: Iterable[Tensor], *args, **kwargs) -> Any:
        # Temporarily load the state back onto self.stateless_model
        # 暂时加载参数状态到 self.stateless_model
        old_state = _swap_state(self.stateless_model, self.names_map, params)
        try:
            # 调用无状态模型，传递参数和额外的参数和关键字参数
            return self.stateless_model(*args, **kwargs)
        finally:
            # Remove the loaded state on self.stateless_model
            # 移除加载到 self.stateless_model 上的参数状态
            _swap_state(self.stateless_model, self.names_map, old_state)


def make_functional(
    model: nn.Module, disable_autograd_tracking: bool = False
) -> Tuple[FunctionalModule, Tuple[Tensor, ...]]:
    """make_functional(model, disable_autograd_tracking=False) -> func, params

    Given a ``torch.nn.Module``, :func:`make_functional` extracts the state
    (params) and returns a functional version of the model, ``func``. This
    makes it so that it is possible use transforms over the parameters of
    ``model``.

    ``func`` can be invoked as follows:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional

        x = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params = make_functional(model)
        func(params, x)

    And here is an example of applying the grad transform over the parameters
    of a model.

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params = make_functional(model)

        def compute_loss(params, x, t):
            y = func(params, x)
            return nn.functional.mse_loss(y, t)

        grad_weights = grad(compute_loss)(params, x, t)

    If the model has any buffers, please use :func:`make_functional_with_buffers` instead.

    """
    # 创建一个函数式模型对象和其参数，用于给定的神经网络模型
    """make_functional(model, disable_autograd_tracking=False) -> func, params

    Given a ``torch.nn.Module``, :func:`make_functional` extracts the state
    (params) and returns a functional version of the model, ``func``. This
    makes it so that it is possible use transforms over the parameters of
    ``model``.

    ``func`` can be invoked as follows:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional

        x = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params = make_functional(model)
        func(params, x)

    And here is an example of applying the grad transform over the parameters
    of a model.

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params = make_functional(model)

        def compute_loss(params, x, t):
            y = func(params, x)
            return nn.functional.mse_loss(y, t)

        grad_weights = grad(compute_loss)(params, x, t)

    If the model has any buffers, please use :func:`make_functional_with_buffers` instead.
    """
    # 提取给定模型的状态（参数）并返回模型的函数式版本和其参数元组
    return FunctionalModule._create_from(model, disable_autograd_tracking)
    # 从输入的模型中获取所有缓冲区并转换为列表
    buffers = list(model.buffers())
    # 如果模型具有任何缓冲区，则引发运行时错误
    if len(buffers) > 0:
        raise RuntimeError(
            "make_functional(model): `model` has buffers. Please use "
            "make_functional_with_buffers(model) instead."
        )
    # 调用 FunctionalModule 的静态方法 _create_from 来创建一个功能模块
    return FunctionalModule._create_from(
        model, disable_autograd_tracking=disable_autograd_tracking
    )
def make_functional_with_buffers(
    model: nn.Module, disable_autograd_tracking: bool = False
) -> Tuple[FunctionalModuleWithBuffers, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    """make_functional_with_buffers(model, disable_autograd_tracking=False) -> func, params, buffers

    Given a ``torch.nn.Module``, make_functional_with_buffers extracts the
    state (params and buffers) and returns a functional version of the model
    ``func`` that can be invoked like a function.

    ``func`` can be invoked as follows:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional_with_buffers

        x = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params, buffers = make_functional_with_buffers(model)
        func(params, buffers, x)

    And here is an example of applying the grad transform over the parameters
    of a model:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional_with_buffers, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params, buffers = make_functional_with_buffers(model)

        def compute_loss(params, buffers, x, t):
            y = func(params, buffers, x)
            return nn.functional.mse_loss(y, t)

        grad_weights = grad(compute_loss)(params, buffers, x, t)

    Args:
        model (torch.nn.Module): Input model.
        disable_autograd_tracking (bool): Flag to disable gradients tracking for output parameters.
            The returned params are unrelated to the set of params from the original model. If False (default),
            the params will have ``requires_grad=True`` on them (aka they will be trackable with regular
            PyTorch autograd), matching the requires_grad-ness of the params from the original model.
            Otherwise, the returned params will have ``requires_grad=False``. Default, False.
            If you plan on using regular PyTorch autograd (e.g., if you want to call ``.backward()`` or
            ``torch.autograd.grad()``, then set ``disable_autograd_tracking=False``.
            Otherwise, if you're only planning on using functorch's gradient transforms,
            then please set ``disable_autograd_tracking=True`` to avoid unnecessarily tracking
            history with PyTorch autograd.

    """
    return FunctionalModuleWithBuffers._create_from(
        model, disable_autograd_tracking=disable_autograd_tracking
    )


def transpose_stack(
    tuple_of_tuple_of_tensors: Tuple[Tuple[Tensor, ...], ...]
) -> Tuple[Tensor, ...]:
    """transpose_stack(tuple_of_tuple_of_tensors) -> Tuple[Tensor, ...]

    Transposes and stacks tensors from a tuple of tuples into a tuple of stacked tensors.

    Args:
        tuple_of_tuple_of_tensors (Tuple[Tuple[Tensor, ...], ...]): Tuple of tuples of tensors.

    Returns:
        Tuple[Tensor, ...]: Tuple of stacked tensors.

    """
    # Transpose the tuple of tuples into a tuple of tuples where each tuple contains tensors from the same index
    tuple_of_tuple_of_tensors = tuple(zip(*tuple_of_tuple_of_tensors))
    # Stack tensors in each tuple and detach them to prevent tracking gradients
    results = tuple(
        torch.stack(shards).detach() for shards in tuple_of_tuple_of_tensors
    )
    return results


def combine_state_for_ensemble(
    models: Sequence[nn.Module],
def combine_state_for_ensemble(models: List[nn.Module]) -> Tuple[FunctionalModuleWithBuffers, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    """
    combine_state_for_ensemble(models) -> func, params, buffers

    Prepares a list of torch.nn.Modules for ensembling with :func:`vmap`.

    Given a list of ``M`` ``nn.Modules`` of the same class, stacks all of their
    parameters and buffers together to make ``params`` and ``buffers``.
    Each parameter and buffer in the result will have an additional dimension
    of size ``M``.

    :func:`combine_state_for_ensemble` also returns ``func``, a functional
    version of one of the models in :attr:`models`. One cannot directly run
    ``func(params, buffers, *args, **kwargs)`` directly, you probably want to
    use ``vmap(func, ...)(params, buffers, *args, **kwargs)``

    Here's an example of how to ensemble over a very simple model:

    .. code-block:: python

        num_models = 5
        batch_size = 64
        in_features, out_features = 3, 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        data = torch.randn(batch_size, 3)

        fmodel, params, buffers = combine_state_for_ensemble(models)
        output = vmap(fmodel, (0, 0, None))(params, buffers, data)

        assert output.shape == (num_models, batch_size, out_features)

    .. warning::
        All of the modules being stacked together must be the same (except for
        the values of their parameters/buffers). For example, they should be in the
        same mode (training vs eval).

        This API is subject to change -- we're investigating better ways to
        create ensembles and would love your feedback how to improve this.
    """
    # 检查模型列表是否为空，若为空则抛出运行时异常
    if len(models) == 0:
        raise RuntimeError(
            "combine_state_for_ensemble: Expected at least one model, got 0."
        )
    # 检查所有模型是否具有相同的训练状态（全部为训练模式或全部为评估模式），否则抛出运行时异常
    if not (all(m.training for m in models) or all(not m.training for m in models)):
        raise RuntimeError(
            "combine_state_for_ensemble: Expected all models to "
            "have the same training/eval mode."
        )
    # 获取第一个模型的类型
    model0_typ = type(models[0])
    # 检查所有模型是否为同一类别的模型，若不是则抛出运行时异常
    if not all(type(m) == model0_typ for m in models):
        raise RuntimeError(
            "combine_state_for_ensemble: Expected all models to be of the same class."
        )
    # 对每个模型调用 make_functional_with_buffers 函数，返回函数、参数和缓冲区的元组
    funcs, params, buffers = zip(
        *[make_functional_with_buffers(model) for model in models]
    )
    # 转置并堆叠所有模型的参数
    params = transpose_stack(params)
    # 转置并堆叠所有模型的缓冲区
    buffers = transpose_stack(buffers)
    # 返回第一个模型的函数、参数和缓冲区
    return funcs[0], params, buffers
    # 定义一个装饰器函数 wrapped，接受任意位置和关键字参数
    def wrapped(*args, **kwargs):
        # 如果 ensemble_shape 的长度大于等于2，则抛出数值错误异常
        if len(ensemble_shape) >= 2:
            raise ValueError("NYI: ensemble_shape with more than 1 element")
        # 如果 ensemble_shape 的长度为0，则创建单个模型并返回其过时的功能
        if len(ensemble_shape) == 0:
            model = model_class(*args, **kwargs).to(device)
            return make_functional_deprecated_v1(model)
        # 获取 ensemble_shape 中的模型数量
        num_models = ensemble_shape[0]  # type: ignore[misc]
        # 如果模型数量小于等于0，则抛出数值错误异常
        if num_models <= 0:
            raise ValueError(f"num_models {num_models} should be > 0")
        # 注意：以下代码段效率不高，更多是概念验证（Proof of Concept，POC）
        # 使用列表推导式创建多个模型，并将它们放到元组中
        models = tuple(
            model_class(*args, **kwargs).to(device) for _ in range(num_models)
        )
        # 调用过时的功能，获取模型的权重张量及其功能和名称
        _, fn, names = make_functional_deprecated_v1(model_class(*args, **kwargs))
        # 使用列表推导式获取模型的权重并放到元组中
        weights = tuple(make_functional_deprecated_v1(model)[0] for model in models)
        # 将权重元组转置，并将每个分片的张量堆叠为一个张量，放到元组中
        weights = tuple(zip(*weights))
        weights = tuple(torch.stack(shards).detach() for shards in weights)
        # 返回权重元组、功能和名称
        return weights, fn, names

    # 返回装饰器函数 wrapped
    return wrapped
# 定义一个函数 functional_init_with_buffers，用于初始化带有缓冲区的函数式模型
def functional_init_with_buffers(
    model_class: Type[nn.Module],
    ensemble_shape: Union[Tuple[()], Tuple[int]] = (),
    device: torch.types.Device = "cpu",
):
    # 定义一个内部函数 wrapped，该函数接收任意参数和关键字参数
    def wrapped(*args, **kwargs):
        # 如果 ensemble_shape 的长度大于等于2，抛出数值错误
        if len(ensemble_shape) >= 2:
            raise ValueError("NYI: ensemble_shape with more than 1 element")
        # 如果 ensemble_shape 的长度为0，创建单个模型并返回其 v1 版本的函数式表示
        if len(ensemble_shape) == 0:
            model = model_class(*args, **kwargs).to(device)
            return make_functional_deprecated_v1(model)
        
        # 获取模型数量 num_models
        num_models = ensemble_shape[0]  # type: ignore[misc]
        # 如果 num_models 小于等于0，抛出数值错误
        if num_models <= 0:
            raise ValueError(f"num_models {num_models} should be > 0")
        
        # 注意：以下部分不是很高效，更像是概念验证（Proof of Concept，POC）
        
        # 创建多个模型并保存在元组 models 中
        models = tuple(
            model_class(*args, **kwargs).to(device) for _ in range(num_models)
        )
        
        # 调用过时的 make_functional_with_buffers_deprecated_v1 函数，并解包返回值
        (
            _,
            _,
            fn,
            weight_names,
            buffer_names,
        ) = make_functional_with_buffers_deprecated_v1(model_class(*args, **kwargs))
        
        # 对每个模型调用 make_functional_with_buffers_deprecated_v1，并解包返回值
        weights, buffers = zip(
            *tuple(
                make_functional_with_buffers_deprecated_v1(model)[:2]
                for model in models
            )
        )
        
        # 对权重和缓冲区进行处理，将它们分别堆叠并分离
        weights = tuple(zip(*weights))
        weights = tuple(torch.stack(shards).detach() for shards in weights)
        buffers = tuple(zip(*buffers))
        buffers = tuple(torch.stack(shards).detach() for shards in buffers)
        
        # 返回处理后的权重、缓冲区、函数、权重名称和缓冲区名称
        return weights, buffers, fn, weight_names, buffer_names

    # 返回内部函数 wrapped
    return wrapped
```