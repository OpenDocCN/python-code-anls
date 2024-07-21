# `.\pytorch\torch\_functorch\functional_call.py`

```
# 引入Counter
    # 检查参数和缓冲区字典是否为字典类型，如果不是则抛出异常
    if isinstance(parameter_and_buffer_dicts, dict):
        # 将参数和缓冲区字典赋值给变量parameters_and_buffers
        parameters_and_buffers = parameter_and_buffer_dicts
    # 如果 parameter_and_buffer_dicts 是一个序列（如列表或元组）
    elif isinstance(parameter_and_buffer_dicts, Sequence):
        # 检查序列中的所有元素是否都是字典类型
        if not all(isinstance(d, dict) for d in parameter_and_buffer_dicts):
            # 如果不是，抛出值错误异常
            raise ValueError(
                "Expected all elements of parameter_and_buffer_dicts to be dictionaries"
            )
        # 获取所有字典中的键，并放入一个列表中
        all_keys = [k for d in parameter_and_buffer_dicts for k in d.keys()]
        # 找出重复出现的键
        repeated_keys = [key for key, n in Counter(all_keys).items() if n > 1]
        # 如果有重复的键存在，抛出值错误异常
        if len(repeated_keys) > 0:
            raise ValueError(
                f"{repeated_keys} appeared in multiple dictionaries; behavior of functional call is ambiguous"
            )
        # 合并所有字典中的键值对到一个新的字典中
        parameters_and_buffers = {
            k: v for d in parameter_and_buffer_dicts for k, v in d.items()
        }
    else:
        # 如果 parameter_and_buffer_dicts 不是字典，也不是序列，则抛出值错误异常
        raise ValueError(
            f"Expected parameter_and_buffer_dicts to be a dict, or a list/tuple of dicts, "
            f"but got {type(parameter_and_buffer_dicts)}"
        )

    # 调用 nn.utils.stateless._functional_call 函数，传入以下参数进行调用
    return nn.utils.stateless._functional_call(
        module,
        parameters_and_buffers,
        args,
        kwargs,
        tie_weights=tie_weights,
        strict=strict,
    )
@exposed_in("torch.func")
def stack_module_state(
    models: List[nn.Module],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """stack_module_state(models) -> params, buffers

    Prepares a list of torch.nn.Modules for ensembling with :func:`vmap`.

    Given a list of ``M`` ``nn.Modules`` of the same class, returns two dictionaries
    that stack all of their parameters and buffers together, indexed by name.
    The stacked parameters are optimizable (i.e. they are new leaf nodes in the
    autograd history that are unrelated to the original parameters and can be
    passed directly to an optimizer).

    Here's an example of how to ensemble over a very simple model:

    .. code-block:: python

        num_models = 5
        batch_size = 64
        in_features, out_features = 3, 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        data = torch.randn(batch_size, 3)

        def wrapper(params, buffers, data):
            return torch.func.functional_call(models[0], (params, buffers), data)

        params, buffers = stack_module_state(models)
        output = vmap(wrapper, (0, 0, None))(params, buffers, data)

        assert output.shape == (num_models, batch_size, out_features)

    When there's submodules, this follows state dict naming conventions

    .. code-block:: python

        import torch.nn as nn
        class Foo(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                hidden = 4
                self.l1 = nn.Linear(in_features, hidden)
                self.l2 = nn.Linear(hidden, out_features)

            def forward(self, x):
                return self.l2(self.l1(x))

        num_models = 5
        in_features, out_features = 3, 3
        models = [Foo(in_features, out_features) for i in range(num_models)]
        params, buffers = stack_module_state(models)
        print(list(params.keys()))  # "l1.weight", "l1.bias", "l2.weight", "l2.bias"

    .. warning::
        All of the modules being stacked together must be the same (except for
        the values of their parameters/buffers). For example, they should be in the
        same mode (training vs eval).
    """
    # 检查输入的模型列表是否为空，如果是则抛出运行时错误
    if len(models) == 0:
        raise RuntimeError("stack_module_state: Expected at least one model, got 0.")
    # 检查所有模型是否具有相同的训练/评估模式，如果不是则抛出运行时错误
    if not (all(m.training for m in models) or all(not m.training for m in models)):
        raise RuntimeError(
            "stack_module_state: Expected all models to have the same training/eval mode."
        )
    # 获取第一个模型的类型，用于后续检查所有模型是否属于同一类
    model0_typ = type(models[0])
    # 检查所有模型是否属于同一类，如果不是则抛出运行时错误
    if not all(type(m) == model0_typ for m in models):
        raise RuntimeError(
            "stack_module_state: Expected all models to be of the same class."
        )
    # 获取所有模型的参数，并以字典形式存储在列表中
    all_params = [dict(model.named_parameters()) for model in models]
    # 使用构造函数 construct_stacked_leaf 将所有模型的同一参数堆叠起来，形成新的优化节点
    params = {
        k: construct_stacked_leaf(tuple(params[k] for params in all_params), k)
        for k in all_params[0]
    }
    # 创建一个包含所有模型缓冲区字典的列表，每个元素是一个模型的命名缓冲区字典
    all_buffers = [dict(model.named_buffers()) for model in models]
    
    # 使用列表推导式构建一个字典 `buffers`，其键为缓冲区的名称 `k`
    # 值为通过 `construct_stacked_leaf` 函数处理后的堆叠叶子（stacked leaf）
    buffers = {
        k: construct_stacked_leaf(tuple(buffers[k] for buffers in all_buffers), k)
        for k in all_buffers[0]  # 使用 all_buffers 列表中第一个模型的缓冲区名称作为迭代键
    }

    # 返回结果 params 和 buffers
    return params, buffers
# 定义一个函数用于构建堆叠的张量，接受一个元组或列表形式的张量输入及其名称，并返回一个张量对象
def construct_stacked_leaf(
    tensors: Union[Tuple[Tensor, ...], List[Tensor]], name: str
) -> Tensor:
    # 检查所有输入张量是否都需要梯度
    all_requires_grad = all(t.requires_grad for t in tensors)
    # 检查所有输入张量是否都不需要梯度
    none_requires_grad = all(not t.requires_grad for t in tensors)
    # 如果有张量部分需要梯度，有张量部分不需要梯度，则抛出运行时异常
    if not all_requires_grad and not none_requires_grad:
        raise RuntimeError(
            f"Expected {name} from each model to have the same .requires_grad"
        )
    # 使用 torch.stack 函数堆叠输入的张量
    result = torch.stack(tensors)
    # 如果所有输入张量都需要梯度，则对结果进行分离（detach）并设置为需要梯度
    if all_requires_grad:
        result = result.detach().requires_grad_()
    # 返回最终的结果张量
    return result
```