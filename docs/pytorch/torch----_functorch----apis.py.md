# `.\pytorch\torch\_functorch\apis.py`

```py
# 允许 mypy 在未定义类型的情况下进行类型推断
# 注意事项：我们允许 Dynamo 通过 torch/_dynamo/trace_rules.py 看到这个文件，以便它可以通过 functorch transforms 进行跟踪。
#       目前，我们不能让 Dynamo 看到 `eager_transforms.py` / `vmap.py`，因为这会破坏很多东西，并且没有一种机制能够选择性地只暴露文件中的某些函数（例如 grad）给 Dynamo。

import functools

from torch._functorch.utils import argnums_t, exposed_in
from torch._functorch.vmap import (
    _check_out_dims_is_int_or_int_pytree,
    _check_randomness_arg,
    _chunked_vmap,
    _process_batched_inputs,
    Callable,
    in_dims_t,
    out_dims_t,
    vmap_impl,
)

# vmap(func)(inputs) wraps all Tensor inputs to be batched in BatchedTensors,
# sends those into func, and then unwraps the output BatchedTensors. Operations
# on BatchedTensors perform the batched operations that the user is asking for.
#
# vmap's randomness behavior differs from JAX's, which would require a PRNG key
# to be passed everywhere.

# 将函数 vmap 注册到 "torch.func" 的命名空间中，使其可以被外部访问
@exposed_in("torch.func")
def vmap(
    func: Callable,
    in_dims: in_dims_t = 0,
    out_dims: out_dims_t = 0,
    randomness: str = "error",
    *,
    chunk_size=None,
) -> Callable:
    """
    vmap is the vectorizing map; ``vmap(func)`` returns a new function that
    maps ``func`` over some dimension of the inputs. Semantically, vmap
    pushes the map into PyTorch operations called by ``func``, effectively
    vectorizing those operations.

    vmap is useful for handling batch dimensions: one can write a function
    ``func`` that runs on examples and then lift it to a function that can
    take batches of examples with ``vmap(func)``. vmap can also be used to
    compute batched gradients when composed with autograd.

    .. note::
        :func:`torch.vmap` is aliased to :func:`torch.func.vmap` for
        convenience. Use whichever one you'd like.
    """
    Args:
        func (function): A Python function that takes one or more arguments.
            Must return one or more Tensors.
            指定一个或多个参数的 Python 函数，必须返回一个或多个张量。
        in_dims (int or nested structure): Specifies which dimension of the
            inputs should be mapped over. ``in_dims`` should have a
            structure like the inputs. If the ``in_dim`` for a particular
            input is None, then that indicates there is no map dimension.
            Default: 0.
            指定输入应该映射的维度。``in_dims`` 应该与输入的结构相似。
            如果特定输入的 ``in_dim`` 是 None，则表示没有映射维度。
            默认值为 0。
        out_dims (int or Tuple[int]): Specifies where the mapped dimension
            should appear in the outputs. If ``out_dims`` is a Tuple, then
            it should have one element per output. Default: 0.
            指定映射维度应该出现在输出的位置。如果 ``out_dims`` 是一个元组，
            则应该有每个输出一个元素。默认值为 0。
        randomness (str): Specifies whether the randomness in this
            vmap should be the same or different across batches. If 'different',
            the randomness for each batch will be different. If 'same', the
            randomness will be the same across batches. If 'error', any calls to
            random functions will error. Default: 'error'. WARNING: this flag
            only applies to random PyTorch operations and does not apply to
            Python's random module or numpy randomness.
            指定此 vmap 中的随机性在批次间是否相同。如果为 'different'，
            每个批次的随机性将不同。如果为 'same'，则随机性在所有批次中相同。
            如果为 'error'，则任何随机函数调用将报错。默认为 'error'。
            警告：此标志仅适用于随机的 PyTorch 操作，不适用于 Python 的 random 模块或 numpy 的随机性。
        chunk_size (None or int): If None (default), apply a single vmap over inputs.
            If not None, then compute the vmap :attr:`chunk_size` samples at a time.
            Note that :attr:`chunk_size=1` is equivalent to computing the vmap with a for-loop.
            If you run into memory issues computing the vmap, please try a non-None chunk_size.
            如果为 None（默认），则对输入应用单个 vmap。
            如果不为 None，则每次计算 :attr:`chunk_size` 个样本的 vmap。
            注意：:attr:`chunk_size=1` 相当于使用 for 循环计算 vmap。
            如果计算 vmap 时遇到内存问题，请尝试非 None 的 chunk_size。

    Returns:
        Returns a new "batched" function. It takes the same inputs as
        ``func``, except each input has an extra dimension at the index
        specified by ``in_dims``. It takes returns the same outputs as
        ``func``, except each output has an extra dimension at the index
        specified by ``out_dims``.
        返回一个新的“批处理”函数。它接受与 ``func`` 相同的输入，除了每个输入在由
        ``in_dims`` 指定的索引处有一个额外的维度。它返回与 ``func`` 相同的输出，
        除了每个输出在由 ``out_dims`` 指定的索引处有一个额外的维度。

    .. warning:
        :func:`vmap` works best with functional-style code. Please do not
        perform any side-effects in ``func``, with the exception of
        in-place PyTorch operations. Examples of side-effects include mutating
        Python data structures and assigning values to variables not captured
        in ``func``.
        :func:`vmap` 最适合函数式风格的代码。请不要在 ``func`` 中执行任何副作用，
        除了就地执行的 PyTorch 操作之外。副作用的示例包括修改 Python 数据结构
        和给未在 ``func`` 中捕获的变量赋值。

    One example of using :func:`vmap` is to compute batched dot products. PyTorch
    doesn't provide a batched ``torch.dot`` API; instead of unsuccessfully
    rummaging through docs, use :func:`vmap` to construct a new function.
    使用 :func:`vmap` 的一个示例是计算批量点积。PyTorch 并没有提供批量 ``torch.dot`` API；
    不要无功而返地查阅文档，而是使用 :func:`vmap` 构建一个新函数。

        >>> torch.dot                            # [D], [D] -> []
        >>> batched_dot = torch.func.vmap(torch.dot)  # [N, D], [N, D] -> [N]
        >>> x, y = torch.randn(2, 5), torch.randn(2, 5)
        >>> batched_dot(x, y)
        
    :func:`vmap` can be helpful in hiding batch dimensions, leading to a simpler
    :func:`vmap` 可以帮助隐藏批量维度，从而导致更简单的```
batched computation code that operates seamlessly across multiple inputs.
批量计算代码，可以在多个输入之间无缝操作。
        >>> batch_size, feature_size = 3, 5
        >>> weights = torch.randn(feature_size, requires_grad=True)
        >>>
        >>> def model(feature_vec):
        >>>     # Very simple linear model with activation
        >>>     return feature_vec.dot(weights).relu()


# 定义批处理大小和特征向量大小，并创建具有随机权重的张量
batch_size, feature_size = 3, 5
weights = torch.randn(feature_size, requires_grad=True)

# 定义模型函数，接收特征向量作为输入，计算线性模型输出并应用ReLU激活函数
def model(feature_vec):
    # 简单的线性模型加上激活函数ReLU
    return feature_vec.dot(weights).relu()



        >>>
        >>> examples = torch.randn(batch_size, feature_size)
        >>> result = torch.vmap(model)(examples)


# 创建示例输入张量，形状为(batch_size, feature_size)，然后使用torch.vmap批处理模型
examples = torch.randn(batch_size, feature_size)
result = torch.vmap(model)(examples)



    :func:`vmap` can also help vectorize computations that were previously difficult
    or impossible to batch. One example is higher-order gradient computation.
    The PyTorch autograd engine computes vjps (vector-Jacobian products).
    Computing a full Jacobian matrix for some function f: R^N -> R^N usually
    requires N calls to ``autograd.grad``, one per Jacobian row. Using :func:`vmap`,
    we can vectorize the whole computation, computing the Jacobian in a single
    call to ``autograd.grad``.


# :func:`vmap`函数有助于将以前难以批处理或不可能批处理的计算向量化。一个例子是高阶梯度计算。
# PyTorch自动求导引擎计算vjps（向量-Jacobian乘积）。
# 计算某个函数f：R^N -> R^N的完整Jacobian矩阵通常需要N次调用``autograd.grad``，每行一个Jacobian。使用:func:`vmap`，
# 我们可以向量化整个计算过程，在单次调用``autograd.grad``中计算Jacobian。



        >>> # Setup
        >>> N = 5
        >>> f = lambda x: x ** 2
        >>> x = torch.randn(N, requires_grad=True)
        >>> y = f(x)
        >>> I_N = torch.eye(N)
        >>>
        >>> # Sequential approach
        >>> jacobian_rows = [torch.autograd.grad(y, x, v, retain_graph=True)[0]
        >>>                  for v in I_N.unbind()]
        >>> jacobian = torch.stack(jacobian_rows)
        >>>
        >>> # vectorized gradient computation
        >>> def get_vjp(v):
        >>>     return torch.autograd.grad(y, x, v)
        >>> jacobian = torch.vmap(get_vjp)(I_N)


# 设置部分
N = 5
f = lambda x: x ** 2
x = torch.randn(N, requires_grad=True)
y = f(x)
I_N = torch.eye(N)

# 顺序方法
jacobian_rows = [torch.autograd.grad(y, x, v, retain_graph=True)[0]
                 for v in I_N.unbind()]
jacobian = torch.stack(jacobian_rows)

# 向量化梯度计算
def get_vjp(v):
    return torch.autograd.grad(y, x, v)
jacobian = torch.vmap(get_vjp)(I_N)



    :func:`vmap` can also be nested, producing an output with multiple batched dimensions


# :func:`vmap`还可以嵌套，生成具有多个批处理维度的输出



        >>> torch.dot                            # [D], [D] -> []
        >>> batched_dot = torch.vmap(torch.vmap(torch.dot))  # [N1, N0, D], [N1, N0, D] -> [N1, N0]
        >>> x, y = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
        >>> batched_dot(x, y) # tensor of size [2, 3]


# 定义torch.dot函数的输入和输出形状
torch.dot                            # [D], [D] -> []

# 使用torch.vmap嵌套调用torch.dot，批处理输入张量x和y的最后两个维度
batched_dot = torch.vmap(torch.vmap(torch.dot))  # [N1, N0, D], [N1, N0, D] -> [N1, N0]
x, y = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
batched_dot(x, y) # tensor of size [2, 3]



        >>> # If the inputs are not batched along the first dimension, ``in_dims`` specifies
        >>> # the dimension that each inputs are batched along as
        >>> torch.dot                            # [N], [N] -> []
        >>> batched_dot = torch.vmap(torch.dot, in_dims=1)  # [N, D], [N, D] -> [D]
        >>> x, y = torch.randn(2, 5), torch.randn(2, 5)
        >>> batched_dot(x, y)   # output is [5] instead of [2] if batched along the 0th dimension


# 如果输入张量的第一个维度没有批处理，``in_dims``参数指定了每个输入张量的批处理维度
torch.dot                            # [N], [N] -> []

# 使用torch.vmap批处理torch.dot函数，指定输入维度为1（第一个维度），批处理输入张量x和y的最后一个维度
batched_dot = torch.vmap(torch.dot, in_dims=1)  # [N, D], [N, D] -> [D]
x, y = torch.randn(2, 5), torch.randn(2, 5)
batched_dot(x, y)   # output is [5] instead of [2] if batched along the 0th dimension



        >>> # If there are multiple inputs each of which is batched along different dimensions,
        >>> # ``in_dims`` must be a tuple with the batch dimension for each input as
        >>> torch.dot                            # [D], [D] -> []
        >>> batched_dot = torch.vmap(torch.dot, in_dims=(0, None))  # [N, D], [D] -> [N]
        >>> x, y = torch.randn(2, 5), torch.randn(5)
        >>> batched_dot(x, y) # second arg doesn't have a batch dim because in_dim[1] was None


# 如果有多个输入，每个输入都批处理在不同的维度上，``in_dims``必须是一个包含每个输入批处理维度的元组
torch.dot                            # [D], [D] -> []

# 使用torch.vmap批处理torch.dot函数，指定输入维度为(0, None)，批处理输入张量x的第一个维度，而y没有批处理维度（因为in_dim[1]为None）
batched_dot = torch.vmap(torch.dot, in_dims=(0, None))  # [N, D], [D] -> [N]
x, y = torch.randn(2, 5), torch.randn(5)
batched_dot(x, y) # second arg doesn't have a batch dim because in_dim[1] was None



        >>> # If the input is a Python struct, ``in_dims`` must be a tuple containing a struct


# 如果输入是Python结构体，``in_dims``必须是一个包含结构体的元组
    """
    from torch._dynamo import is_compiling  # 导入 _dynamo 模块中的 is_compiling 函数

    _check_randomness_arg(randomness)  # 调用 _check_randomness_arg 函数，验证 randomness 参数的有效性
    if not (chunk_size is None or chunk_size > 0):  # 如果 chunk_size 不是 None 且不大于 0，则抛出 ValueError 异常
        raise ValueError(
            f"vmap: chunk_size should be None or greater than 0. (got {chunk_size})"
        )

    def wrapped(*args, **kwargs):
        return vmap_impl(  # 调用 vmap_impl 函数，执行 vmap 的实现
            func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs
        )

    if not is_compiling():  # 如果不处于编译状态，则对 wrapped 函数进行装饰
        wrapped = functools.wraps(func)(wrapped)

    return wrapped  # 返回经装饰的 wrapped 函数
# 检查随机性参数的有效性
_check_randomness_arg(randomness)

# 如果 chunks 等于 1，则直接调用 vmap 函数进行向量化映射
if chunks == 1:
    return vmap(func, in_dims=in_dims, out_dims=out_dims, randomness=randomness)

# 定义内部函数 _get_chunk_flat_args，用于将输入数据分块并扁平化处理
def _get_chunk_flat_args(flat_args_, flat_in_dims_, chunks_):
    # 对每个输入参数进行处理，根据给定的 in_dim 参数分块（如果不为 None）
    flat_args_chunks = tuple(
        t.chunk(chunks_, dim=in_dim)
        if in_dim is not None
        else [
            t,
        ]
        * chunks_
        for t, in_dim in zip(flat_args_, flat_in_dims_)
    )
    # 转置块维度并扁平化结构
    # chunks_flat_args 是扁平化参数列表
    chunks_flat_args = zip(*flat_args_chunks)
    return chunks_flat_args

# 使用 functools.wraps 装饰器，保留原始函数的元数据，使得内部函数具有和原函数一样的属性
@functools.wraps(func)
    # 定义一个装饰器函数 `wrapped_with_chunks`，用于处理带有批处理的函数调用
    def wrapped_with_chunks(*args, **kwargs):
        # 检查输出维度 `out_dims` 是否为整数或整数的 pytree 结构
        _check_out_dims_is_int_or_int_pytree(out_dims, func)
        # 处理批量输入数据，获取处理后的结果：输出维度、扁平化的输入维度、扁平化的参数、参数规范
        _, flat_in_dims, flat_args, args_spec = _process_batched_inputs(
            in_dims, args, func
        )
        # 对扁平化参数进行分块处理，根据扁平化的输入维度和分块数量 `chunks`
        chunks_flat_args = _get_chunk_flat_args(flat_args, flat_in_dims, chunks)

        # 对分块后的参数应用 `vmap`
        return _chunked_vmap(
            func,
            flat_in_dims,
            chunks_flat_args,
            args_spec,
            out_dims,
            randomness,
            **kwargs,
        )

    # 返回装饰后的函数 `wrapped_with_chunks`
    return wrapped_with_chunks
# 将函数标记为在"torch.func"中公开的函数
@exposed_in("torch.func")
# 定义一个名为grad的函数，用于计算给定函数func关于指定参数argnums的梯度
def grad(func: Callable, argnums: argnums_t = 0, has_aux: bool = False) -> Callable:
    """``grad`` operator helps computing gradients of ``func`` with respect to the
    input(s) specified by ``argnums``. This operator can be nested to
    compute higher-order gradients.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. If specified ``has_aux`` equals ``True``,
            function can return a tuple of single-element Tensor and other auxiliary objects:
            ``(output, aux)``.
        argnums (int or Tuple[int]): Specifies arguments to compute gradients with respect to.
            ``argnums`` can be single integer or tuple of integers. Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a tensor and other
            auxiliary objects: ``(output, aux)``. Default: False.

    Returns:
        Function to compute gradients with respect to its inputs. By default, the output of
        the function is the gradient tensor(s) with respect to the first argument.
        If specified ``has_aux`` equals ``True``, tuple of gradients and output auxiliary objects
        is returned. If ``argnums`` is a tuple of integers, a tuple of output gradients with
        respect to each ``argnums`` value is returned.

    Example of using ``grad``:

        >>> # xdoctest: +SKIP
        >>> from torch.func import grad
        >>> x = torch.randn([])
        >>> cos_x = grad(lambda x: torch.sin(x))(x)
        >>> assert torch.allclose(cos_x, x.cos())
        >>>
        >>> # Second-order gradients
        >>> neg_sin_x = grad(grad(lambda x: torch.sin(x)))(x)
        >>> assert torch.allclose(neg_sin_x, -x.sin())

    When composed with ``vmap``, ``grad`` can be used to compute per-sample-gradients:

        >>> # xdoctest: +SKIP
        >>> from torch.func import grad, vmap
        >>> batch_size, feature_size = 3, 5
        >>>
        >>> def model(weights, feature_vec):
        >>>     # Very simple linear model with activation
        >>>     assert feature_vec.dim() == 1
        >>>     return feature_vec.dot(weights).relu()
        >>>
        >>> def compute_loss(weights, example, target):
        >>>     y = model(weights, example)
        >>>     return ((y - target) ** 2).mean()  # MSELoss
        >>>
        >>> weights = torch.randn(feature_size, requires_grad=True)
        >>> examples = torch.randn(batch_size, feature_size)
        >>> targets = torch.randn(batch_size)
        >>> inputs = (weights, examples, targets)
        >>> grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(*inputs)
    """
    pass
    # 导入所需模块和函数，用于避免循环依赖
    import torch._functorch.eager_transforms as eager_transforms
    from torch._dynamo import is_compiling

    # 定义一个包装函数，用于调用梯度计算的实现
    def wrapper(*args, **kwargs):
        return eager_transforms.grad_impl(func, argnums, has_aux, args, kwargs)

    # 如果不处于编译状态，则对包装函数应用 functools.wraps，保留原函数的元数据
    if not is_compiling():
        wrapper = functools.wraps(func)(wrapper)

    # 返回经过包装后的函数
    return wrapper
# 使用 @exposed_in 装饰器将函数暴露在 torch.func 中
@exposed_in("torch.func")
# 定义 grad_and_value 函数，用于计算梯度和原始（或前向）计算的函数
def grad_and_value(
    func: Callable, argnums: argnums_t = 0, has_aux: bool = False
) -> Callable:
    """
    Returns a function to compute a tuple of the gradient and primal, or
    forward, computation.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. If specified ``has_aux``
            equals ``True``, function can return a tuple of single-element
            Tensor and other auxiliary objects: ``(output, aux)``.
        argnums (int or Tuple[int]): Specifies arguments to compute gradients
            with respect to. ``argnums`` can be single integer or tuple of
            integers. Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a tensor and
            other auxiliary objects: ``(output, aux)``. Default: False.

    Returns:
        Function to compute a tuple of gradients with respect to its inputs
        and the forward computation. By default, the output of the function is
        a tuple of the gradient tensor(s) with respect to the first argument
        and the primal computation. If specified ``has_aux`` equals
        ``True``, tuple of gradients and tuple of the forward computation with
        output auxiliary objects is returned. If ``argnums`` is a tuple of
        integers, a tuple of a tuple of the output gradients with respect to
        each ``argnums`` value and the forward computation is returned.

    See :func:`grad` for examples
    """
    # 导入需要的模块和函数
    from torch._dynamo import is_compiling
    from torch._functorch import eager_transforms

    # 定义包装函数 wrapper，用于调用具体的梯度计算和原始计算实现
    def wrapper(*args, **kwargs):
        return eager_transforms.grad_and_value_impl(
            func, argnums, has_aux, args, kwargs
        )

    # 如果不处于编译状态，使用 functools.wraps 封装原始函数
    if not is_compiling():
        wrapper = functools.wraps(func)(wrapper)

    # 返回包装后的函数
    return wrapper
```