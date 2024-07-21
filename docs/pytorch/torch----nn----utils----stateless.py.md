# `.\pytorch\torch\nn\utils\stateless.py`

```py
# mypy: allow-untyped-defs
# 引入
    # 遍历给定的被绑定张量的名称列表
    for given_name in given_names_for_tied_tensors:
        # 获取与当前给定名称相关联的所有绑定名称集合
        tied_names = tied_names_map[given_name]
        
        # 检测是否有多个键指向同一个绑定张量
        if (
            len(tied_names.intersection(given_names_for_tied_tensors)) > 1
            # 只有在用户为同一个绑定张量传递了多个不同值时才会引发错误
            # 如果所有给定的值都相同，则不引发错误
            and len({parameters_and_buffers[tied_name] for tied_name in tied_names})
            != 1
        ):
            # 抛出数值错误，指示存在多个不同的值与同一绑定张量关联
            raise ValueError(
                f"functional_call got multiple values for keys {sorted(tied_names)}, "
                f"which are tied. Consider using tie_weights=False"
            )
    
    # 解绑给定命名张量映射
    # 复制参数和缓冲区字典，以防修改原始字典
    untied_parameters_and_buffers = parameters_and_buffers.copy()
    
    # 遍历每个给定的被绑定张量的名称
    for given_name in given_names_for_tied_tensors:
        # 遍历与当前给定名称相关联的所有绑定名称
        for tied_name in tied_names_map[given_name]:
            # 将每个绑定名称映射到与给定名称相同的值，以解除绑定
            untied_parameters_and_buffers[tied_name] = parameters_and_buffers[given_name]
    
    # 返回解绑后的参数和缓冲区字典
    return untied_parameters_and_buffers
# 定义一个上下文管理器函数，用于重新参数化给定的 torch.nn.Module 对象
@contextlib.contextmanager
def _reparametrize_module(
    module: "torch.nn.Module",
    parameters_and_buffers: Dict[str, Tensor],
    *,
    tie_weights: bool = False,  # 是否要求权重绑定的布尔标志
    strict: bool = False,  # 是否启用严格模式的布尔标志
    stack_weights: bool = False,  # 是否启用权重堆栈的布尔标志
) -> Iterator[None]:  # 返回值为 None 的迭代器

    # 如果 tie_weights 为 True，则解绑参数和缓冲区，返回新的解绑后的字典
    if tie_weights:
        untied_parameters_and_buffers = _untie_named_tensors_map(
            module, parameters_and_buffers
        )
    else:
        untied_parameters_and_buffers = parameters_and_buffers

    # 使用 NamedMemberAccessor 类创建一个访问器对象
    accessor = NamedMemberAccessor(module)

    # 如果 strict 为 True，则进行参数和缓冲区的键检查
    if strict:
        # 检查缺失的键和意外的键，并生成错误消息列表
        missing_keys, unexpected_keys = accessor.check_keys(
            untied_parameters_and_buffers
        )
        error_msgs = []
        if len(unexpected_keys) > 0:
            error_msgs.append(
                f"Unexpected key(s): {', '.join(map(repr, unexpected_keys))}."
            )
        if len(missing_keys) > 0:
            error_msgs.append(f"Missing key(s): {', '.join(map(repr, missing_keys))}.")
        # 如果有错误消息，则抛出 RuntimeError 异常
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in reparametrizing for {}:\n\t{}".format(
                    module._get_name(), "\n\t".join(error_msgs)
                )
            )

    # 声明一个空的原始参数和缓冲区字典
    orig_parameters_and_buffers: Dict[str, Tensor] = {}

    try:
        # 使用 accessor.swap_tensors_dict() 方法替换参数和缓冲区，并允许缺失的键
        orig_parameters_and_buffers, _ = accessor.swap_tensors_dict(
            untied_parameters_and_buffers, allow_missing=True
        )
        # 使用 yield 将控制权传递给调用者
        yield

    finally:
        # 如果 stack_weights 为 True，则以 LIFO（后进先出）顺序恢复权重
        if stack_weights:
            orig_parameters_and_buffers = dict(
                reversed(orig_parameters_and_buffers.items())
            )

        # 再次使用 accessor.swap_tensors_dict() 方法替换参数和缓冲区，并允许缺失的键
        new_parameters_and_buffers, _ = accessor.swap_tensors_dict(
            orig_parameters_and_buffers, allow_missing=True
        )

        # 有些模块可能不是完全无状态的，在 _parameters 和 _buffers 字典上进行原地修改
        # 将更改后的参数和缓冲区写回到原始字典 parameters_and_buffers 中
        parameters_and_buffers.update(
            {
                k: new_parameters_and_buffers[k]
                for k in parameters_and_buffers
                if k in new_parameters_and_buffers
            }
        )


# 使用 @deprecated 装饰器声明一个过时的函数 functional_call
@deprecated(
    "`torch.nn.utils.stateless.functional_call` is deprecated as of PyTorch 2.0 "
    "and will be removed in a future version of PyTorch. "
    "Please use `torch.func.functional_call` instead which is a drop-in replacement.",
    category=FutureWarning,
)
def functional_call(
    module: "torch.nn.Module",
    parameters_and_buffers: Dict[str, Tensor],
    args: Union[Any, Tuple],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    tie_weights: bool = True,  # 是否要求权重绑定的布尔标志，默认为 True
    strict: bool = False,  # 是否启用严格模式的布尔标志，默认为 False
):
    r"""Perform a functional call on the module by replacing the module parameters and buffers with the provided ones.
    
    ```
    # 警告：这个API在PyTorch 2.0中已被弃用，并将在未来的版本中移除。请使用torch.func.functional_call替代，它是这个API的一个完全兼容的替代品。
    .. warning::
    
    # 如果模块具有活跃的参数化，在parameters_and_buffers参数中传递一个与常规参数名相同的值将完全禁用参数化。
    # 如果要将参数化函数应用于传递的值，请将键设置为"{submodule_name}.parametrizations.{parameter_name}.original"。
    .. note:: 如果模块对参数/缓冲执行原地操作，这些操作将反映在parameters_and_buffers输入中。
    
    # 示例：
    # >>> a = {'foo': torch.zeros(())}
    # >>> mod = Foo()  # 执行 self.foo = self.foo + 1
    # >>> print(mod.foo)  # tensor(0.)
    # >>> functional_call(mod, a, torch.ones(()))
    # >>> print(mod.foo)  # tensor(0.)
    # >>> print(a['foo'])  # tensor(1.)
    .. note::
    
    # 如果模块具有绑定的权重，functional_call是否尊重这些绑定取决于tie_weights标志。
    # 示例：
    # >>> a = {'foo': torch.zeros(())}
    # >>> mod = Foo()  # 包括self.foo和self.foo_tied，它们是绑定的。返回x + self.foo + self.foo_tied
    # >>> print(mod.foo)  # tensor(1.)
    # >>> mod(torch.zeros(()))  # tensor(2.)
    # >>> functional_call(mod, a, torch.zeros(()))  # tensor(0.) 因为它会改变self.foo_tied
    # >>> functional_call(mod, a, torch.zeros(()), tie_weights=False)  # tensor(1.)--self.foo_tied没有更新
    # >>> new_a = {'foo': torch.zeros(()), 'foo_tied': torch.zeros(())}
    # >>> functional_call(mod, new_a, torch.zeros()) # tensor(0.)
    .. note::
    # 调用指定的神经网络模块，执行前向计算
    def _forward_call(
        module,  # torch.nn.Module类型，要调用的模块
        parameters_and_buffers,  # 字典，包含用于模块调用的参数和缓冲区
        args,  # 任意类型或元组，传递给模块调用的位置参数。如果不是元组，则被视为单个参数。
        kwargs,  # 字典，传递给模块调用的关键字参数
        tie_weights=False,  # 布尔值，可选参数。如果为True，则原模型中被绑定的参数和缓冲区在重新参数化的版本中也会被视为绑定的。因此，如果为True且为绑定的参数和缓冲区传递了不同的值，则会报错。
        strict=False  # 布尔值，可选参数。如果为True，则传入的参数和缓冲区必须与原始模块中的参数和缓冲区匹配。因此，如果为True且存在任何缺失或意外的键，则会报错。
    ):
        # 调用内部函数 _functional_call 来执行具体的模块调用
        return _functional_call(
            module,
            parameters_and_buffers,
            args,
            kwargs,
            tie_weights=tie_weights,
            strict=strict,
        )
# 定义了一个名为 _functional_call 的函数，用于执行函数式调用的操作
def _functional_call(
    module: "torch.nn.Module",
    parameters_and_buffers: Dict[str, Tensor],
    args: Union[Any, Tuple],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    tie_weights: bool = True,
    strict: bool = False,
):
    # TODO allow kwargs such as unsafe and others for parametrization
    # 检查当前是否处于 Torch 的追踪或脚本状态，或者 module 是否为 Torch 脚本相关对象，若是，则抛出运行时异常
    if (
        torch.jit.is_tracing()
        or torch.jit.is_scripting()
        or isinstance(
            module,
            (
                torch.jit.RecursiveScriptModule,
                torch.jit.ScriptModule,
                torch.jit.ScriptFunction,
            ),
        )
    ):
        raise RuntimeError("The stateless API can't be used with Jitted modules")
    
    # 如果 module 是 torch.nn.DataParallel 类型，则抛出运行时异常，因为无法使用无状态 API
    if isinstance(module, torch.nn.DataParallel):
        raise RuntimeError(
            "The stateless API can't be used with nn.DataParallel module"
        )
    
    # 如果 kwargs 为 None，则将其设置为空字典
    if kwargs is None:
        kwargs = {}
    
    # 如果 args 不是 tuple 类型，则将其转换为单元素 tuple
    if not isinstance(args, tuple):
        args = (args,)
    
    # 使用 _reparametrize_module 上下文管理器，重新参数化 module 中的参数和缓冲区
    with _reparametrize_module(
        module, parameters_and_buffers, tie_weights=tie_weights, strict=strict
    ):
        # 调用 module 对象，传入 args 和 kwargs，返回结果
        return module(*args, **kwargs)
```