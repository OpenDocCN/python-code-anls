# `.\pytorch\torch\_higher_order_ops\map.py`

```
# 引入 torch 库和一些内部模块
import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._functorch.aot_autograd import AOTConfig, create_joint, from_fun
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    reenter_make_fx,
    UnsupportedAliasMutationException,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import (
    disable_functional_mode,
    FunctionalTensor,
)
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.multiprocessing.reductions import StorageWeakRef


# TODO: 为了防止动态跟踪进入 map_wrapper 函数，添加此类作为中间封装器，当 map_wrapper 准备就绪时移除封装器调用。
class MapWrapper(HigherOrderOperator):
    def __call__(self, xs, *args):
        return map_wrapper(xs, *args)


# 创建 MapWrapper 实例
map = MapWrapper("map")
# 创建 HigherOrderOperator 实例
map_impl = HigherOrderOperator("map_impl")

# 创建 AOTConfig 的虚拟实例，用于配置一些参数
dummy_aot_config = AOTConfig(
    fw_compiler=None,  # 前向编译器设置为 None
    bw_compiler=None,  # 反向编译器设置为 None
    partition_fn=None,  # 分区函数设置为 None
    decompositions={},  # 分解设置为空字典
    num_params_buffers=0,  # 参数缓冲区数量设置为 0
    aot_id=0,  # AOT ID 设置为 0
    keep_inference_input_mutations=False,  # 关闭推理输入变异保留
)


# 创建前向和反向图的函数
def create_fw_bw_graph(f, num_mapped_args, *args):
    # 提取前 num_mapped_args 个参数作为 mapped_xs
    mapped_xs = args[:num_mapped_args]
    # 剩余参数作为 pos_args
    pos_args = args[num_mapped_args:]

    # 注意:[HOP create fw_bw graph] 通过暂停所有分发键来创建 "干净" 环境，
    # 以便 make_fx 使用。当前只暂停了 functionalization，但在需要时可以添加更多。
    # 如果不暂停 functionalization 将会遇到两个问题：
    #
    # 1. make_fx 无法捕获输入上的操作：输入被包装为 _to_functional_tensor_wrapper，
    # 但在进入 ProxyTorchDispatchMode 时会被解包。然而，跟踪器创建代理对象是基于外部包装器的。
    # 这会导致跟踪器无法获取输入的代理对象，也无法捕获其上的任何操作。
    #
    # 2. make_fx 无法捕获输出：在 ProxyTorchDispatchMode 之后，输出被进一步包装为 FunctionalTensorWrapper，
    # 在返回后会在 Functionalize 键中包装。然而，跟踪器仅将内部张量与 ProxyTorchDispatchMode 中的代理对象关联。
    # 因此，在创建输出节点时，它无法将包装的张量与其代理对象关联起来。
    # 而是会创建 _tensor_constant 作为输出。

# map_wrapper 函数定义
def map_wrapper(f, xs, *args):
    # 将 xs 展平，并返回展平后的列表和其规范化形式
    flat_xs, xs_spec = pytree.tree_flatten(xs)
    # 如果展平后的所有元素不都是 torch.Tensor 对象，则抛出运行时错误
    if not all(isinstance(t, torch.Tensor) for t in flat_xs):
        raise RuntimeError(f"Mapped xs can only consist of tensors. Got xs {flat_xs}.")

    # 计算 mapped_xs 的数量作为 num_mapped_args
    num_mapped_args = len(flat_xs)
    # 获取每个 flat_xs 中元素的形状，形成形状的列表
    shapes = [xs.shape for xs in flat_xs]
    # 获取第一个元素的 leading dimension 大小
    leading_dim_size = shapes[0][0]
    # 如果 leading dimension 大小为 0，则抛出异常
    if leading_dim_size == 0:
        raise RuntimeError("Leading dimensions of mapped xs cannot be 0.")

    # 检查是否有任何一个元素的 leading dimension 与第一个元素不一致，如果有则抛出异常
    if any(cur_shape[0] != leading_dim_size for cur_shape in shapes):
        raise RuntimeError(
            f"Leading dimensions of mapped xs must be consistent. Got shapes {shapes}."
        )

    # 初始化输出规范为 None
    out_spec = None

    # 定义一个函数 flat_fn，用于处理扁平化的参数并返回扁平化的输出
    def flat_fn(*flat_args):
        # 将 flat_args 的前 num_mapped_args 个参数解析成 xs 结构
        xs = pytree.tree_unflatten(list(flat_args[:num_mapped_args]), xs_spec)
        # 使用 f 函数处理 xs 和其余的 flat_args，得到未扁平化的输出和临时输出规范
        unflattened_out = f(xs, *flat_args[num_mapped_args:])
        # 将未扁平化的输出转换成扁平化的形式，并获取临时输出规范
        flat_out, tmp_out_spec = pytree.tree_flatten(unflattened_out)

        # 使用 nonlocal 更新外部的 out_spec 变量为临时输出规范
        nonlocal out_spec
        out_spec = tmp_out_spec
        # 返回扁平化的输出
        return flat_out

    # 使用 map_impl 函数将 flat_fn 应用到 flat_xs 和 args 上，并返回结果
    return pytree.tree_unflatten(
        map_impl(flat_fn, flat_xs, args), out_spec  # type: ignore[arg-type]
    )
# 定义一个自动微分操作的类 MapAutogradOp，继承自 torch.autograd.Function
class MapAutogradOp(torch.autograd.Function):
    
    # 静态方法：前向传播函数
    @staticmethod
    def forward(ctx, fw_graph, joint_graph, num_mapped_args, *flat_args):
        # 保存用于反向传播的张量到上下文中
        ctx.save_for_backward(*flat_args)
        # 保存联合图和映射参数个数到上下文中
        ctx._joint_graph = joint_graph
        ctx._num_mapped_args = num_mapped_args
        # 使用 torch._C._AutoDispatchBelowAutograd() 上下文管理器
        # 调用 map_impl 函数，将输入的参数分为映射参数和位置参数并执行
        return (
            *map_impl(
                fw_graph, flat_args[:num_mapped_args], flat_args[num_mapped_args:]
            ),
        )

    # 静态方法：反向传播函数
    @staticmethod
    def backward(ctx, *flat_grads):
        # 从上下文中获取保存的张量
        fw_args = ctx.saved_tensors
        # 提取前向传播中的映射参数
        fw_mapped_args = fw_args[: ctx._num_mapped_args]
        # 提取位置参数
        pos_args = fw_args[ctx._num_mapped_args :]

        # 调用 map_impl 函数，计算反向传播的梯度
        grads = map_impl(
            ctx._joint_graph,
            fw_mapped_args + flat_grads,
            pos_args,
        )
        # 返回梯度，None 代表其它梯度未定义
        return None, None, None, *grads


# 定义 trace_map 函数
def trace_map(proxy_mode, func_overload, f, xs, pos_args):
    # 获取输入张量的首个维度大小
    leading_dim_size = xs[0].shape[0]

    # 从输入 xs 中解包并获取示例输入
    example_input = _unstack_pytree(xs)[0]
    # 获取函数 f 的图形体
    body_graph = f

    # 重新创建效果图并执行
    body_graph = reenter_make_fx(body_graph)(*example_input, *pos_args)

    # 使用代理模式跟踪器获取一个新的唯一合格名称
    next_name = proxy_mode.tracer.get_fresh_qualname("body_graph_")

    # 在代理模式的跟踪器中注册模块
    proxy_mode.tracer.root.register_module(next_name, body_graph)

    # 禁用代理模式跟踪
    with disable_proxy_modes_tracing():
        # 执行示例输入和位置参数来获取示例输出
        example_outs = body_graph(*example_input, *pos_args)

        # 定义一个函数，用于扩展张量的维度
        def expand_tensor(t):
            if isinstance(t, torch.Tensor):
                return t.expand(leading_dim_size, *t.shape)
            return t

        # 对示例输出中的每个元素应用 expand_tensor 函数
        expanded_outs = pytree.tree_map(expand_tensor, example_outs)

    # 构建节点参数列表，包括函数图、输入列表和位置参数列表
    node_args = (body_graph, list(xs), list(pos_args))
    # 对节点参数列表中的每个元素应用跟踪器的 unwrap_proxy 函数
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    # 创建一个代理对象，用于跟踪张量树的调用函数
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="map_impl"
    )
    # 跟踪张量树，返回扩展的输出、代理对象及常量等参数
    return track_tensor_tree(
        expanded_outs, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


# 定义私有函数 _unstack_pytree
def _unstack_pytree(xs):
    # 将输入 xs 进行树展平，获取展平列表和结构描述
    flat_xs, inspec = pytree.tree_flatten(xs)
    # 如果展平列表中的所有元素不都是 torch.Tensor，则引发运行时错误
    if not all(isinstance(xs, torch.Tensor) for xs in flat_xs):
        raise RuntimeError(f"Leaves of xs must be Tensor {flat_xs}")

    # 如果展平列表中的所有张量的首个维度大小不一致，则引发运行时错误
    if not all(xs.shape[0] == flat_xs[0].shape[0] for xs in flat_xs):
        raise RuntimeError(
            f"Leaves of xs must have same leading dimension size {[xs.shape for xs in flat_xs]}"
        )

    # 将展平列表进行转置，并将其还原为树形结构
    a = zip(*flat_xs)

    # 创建一个空列表来存储还原后的树形结构
    pytrees = []
    # 对每个元组应用 tree_unflatten 函数，并将结果添加到 pytrees 列表中
    for tuple in a:
        pytrees.append(pytree.tree_unflatten(tuple, inspec))
    # 返回还原后的树形结构列表
    return pytrees


# 定义私有函数 _stack_pytree
def _stack_pytree(pytrees):
    # 创建一个空列表来存储展平的输出和输出结构描述
    flat_out = []
    out_spec = None
    # 对每个输入的树形结构应用 tree_flatten 函数
    for pt in pytrees:
        flat_pt, out_spec = pytree.tree_flatten(pt)
        # 将展平的结果添加到 flat_out 列表中
        flat_out.append(flat_pt)
    # 确保输出结构描述不为空
    assert out_spec is not None
    # 对展平列表进行转置，并将其还原为树形结构
    b = zip(*flat_out)
    # 创建一个空列表来存储还原后的树形结构
    stacked_out = []
    # 遍历列表 b 中的每个元素 leaves
    for leaves in b:
        # 检查 leaves 中的每个 leaf 是否都是 torch.Tensor 类型
        if all(isinstance(leaf, torch.Tensor) for leaf in leaves):
            # 如果是，则将 leaves 中的 tensor 元素堆叠起来并添加到 stacked_out 中
            stacked_out.append(torch.stack(leaves))
        # 如果 leaves 中所有元素都为 None
        elif all(leaf is None for leaf in leaves):
            # 后向图在前向输入不需要梯度时可以返回 None 输出。
            # 当我们急切地执行后向图时，我们需要在其输出上调用 _stack_pytree，
            # 因此我们需要处理 None 输出。
            stacked_out.append(None)  # type: ignore[arg-type]
        else:
            # 如果 leaves 中有非 tensor 且非 None 的元素，则抛出 RuntimeError
            raise RuntimeError(f"Cannot stack {leaves}.")
    # 使用 pytree 模块的 tree_unflatten 函数，将 stacked_out 解压成原始结构并返回
    return pytree.tree_unflatten(stacked_out, out_spec)
# 使用给定的调度键（CompositeExplicitAutograd），将函数映射到密集数据上
@map_impl.py_impl(DispatchKey.CompositeExplicitAutograd)
def map_dense(f, xs, pos_args):
    pytrees = []
    # 对输入数据进行解包，将每个解包后的元素传递给函数 f，并将结果添加到 pytrees 列表中
    for inp in _unstack_pytree(xs):
        pytrees.append(f(*inp, *pos_args))
    # 将处理后的结果重新打包成 PyTree 结构并返回
    return _stack_pytree(pytrees)


# 使用给定的调度键（Autograd），创建前向和反向计算图，并应用到函数 f 上
@map_impl.py_impl(DispatchKey.Autograd)
def map_autograd(f, xs, pos_args):
    num_mapped_args = len(xs)
    # 创建前向和反向计算图，并将其应用到函数 f 上，获取平坦化的输出
    fw_graph, bw_graph = create_fw_bw_graph(f, num_mapped_args, *xs, *pos_args)
    flat_out = MapAutogradOp.apply(fw_graph, bw_graph, num_mapped_args, *xs, *pos_args)
    # 返回平坦化输出结果
    return flat_out


# 使用给定的调度模式（ProxyTorchDispatchMode），如果启用追踪，则调用 trace_map 函数，否则调用 map_impl 函数
@map_impl.py_impl(ProxyTorchDispatchMode)
def map_proxy_torch_dispatch_mode(mode, f, xs, args):
    if mode.enable_tracing:
        return trace_map(mode, map_impl, f, xs, args)
    else:
        return map_impl(f, xs, args)


# 使用给定的调度模式（FakeTensorMode），在模式的上下文中调用 map_dense 函数
@map_impl.py_impl(FakeTensorMode)
def map_fake_tensor_mode(mode, f, xs, args):
    with mode:
        # 在模式的上下文中调用 map_dense 函数
        return map_dense(f, xs, args)


# 使用函数式化的实现（map_functionalize_impl），对上下文进行处理，并调用 map_impl 函数进行映射操作
@map_impl.py_functionalize_impl
def map_functionalize(ctx, f, xs, pos_args):
    # 解包输入的张量数据
    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_args = ctx.unwrap_tensors(pos_args)
    # 使用上下文对象的 functionalize 方法对函数 f 进行包装
    wrapped_fn = ctx.functionalize(f)

    with ctx.redispatch_to_next():
        with disable_proxy_modes_tracing():
            # 构建一个示例输入用于检测输入是否有潜在的分支变异
            example_inputs = (*_unstack_pytree(unwrapped_xs)[0], *unwrapped_args)
        # 检查函数 f 是否可能会变异输入，如果是则抛出异常
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        if _has_potential_branch_input_mutation(
            f, example_inputs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException("torch.map is mutating the input!")

        # 检查函数 f 是否可能会给输入设置别名，如果是则抛出异常
        if _has_potential_branch_input_alias(
            f, example_inputs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException("torch.map is aliasing the input!")

        # 调用 map_impl 函数对处理后的函数进行映射操作，并将结果用上下文对象的 wrap_tensors 方法进行包装
        map_return = map_impl(wrapped_fn, unwrapped_xs, unwrapped_args)
        return ctx.wrap_tensors(map_return)
```