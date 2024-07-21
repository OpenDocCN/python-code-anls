# `.\pytorch\torch\distributed\_spmd\api.py`

```
# mypy: allow-untyped-defs
# 从 abc 模块导入 ABC 和 abstractmethod 装饰器
from abc import ABC, abstractmethod
# 从 contextlib 模块导入 contextmanager 和 nullcontext 上下文管理器
from contextlib import contextmanager, nullcontext
# 从 copy 模块导入 copy 函数
from copy import copy
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 functools 模块导入 partial 和 wraps 装饰器
from functools import partial, wraps
# 从 typing 模块导入各种类型相关的声明
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Union

# 导入 torch 库
import torch
# 导入 torch.distributed 库并命名为 dist
import torch.distributed as dist

# 导入 _functional_collectives 模块以触发操作注册
# 这里导入模块主要是为了注册一些分布式操作
import torch.distributed._functional_collectives
# 导入 nn 模块
import torch.nn as nn
# 导入 torch.utils._pytree 并命名为 pytree
import torch.utils._pytree as pytree
# 从 functorch 模块导入 make_fx 函数
from functorch import make_fx
# 导入 torch.fx 模块
from torch import fx
# 从 torch._decomp.decompositions 导入 native_layer_norm_backward 函数
from torch._decomp.decompositions import native_layer_norm_backward
# 导入 FakeTensorMode 枚举类型
from torch._subclasses.fake_tensor import FakeTensorMode
# 从 torch.distributed._spmd.data_parallel 导入 gradients_tagging 函数
from torch.distributed._spmd.data_parallel import gradients_tagging
# 从 torch.distributed._spmd.parallel_mode 导入 DataParallel, DTensorExpandMode, ParallelMode 类
from torch.distributed._spmd.parallel_mode import (
    DataParallel,
    DTensorExpandMode,
    ParallelMode,
)
# 从 torch.distributed._tensor 导入 Placement 类
from torch.distributed._tensor import Placement
# 从 torch.fx.graph 导入 _PyTreeCodeGen, _PyTreeInfo, CodeGen 类
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo, CodeGen
# 从 torch.nn.utils 导入 stateless 模块
from torch.nn.utils import stateless
# 从 torch.nn.utils._named_member_accessor 导入 NamedMemberAccessor 类
from torch.nn.utils._named_member_accessor import NamedMemberAccessor


# 定义 Override 抽象类，继承自 ABC 类
class Override(ABC):
    r"""Override the tracing and transformation behavior of :meth:`~torch.distributed._spmd.compile`.

    This is useful when any part of the model is not traceable or if you prefer
    to not trace it due to any reason. More specifically, users can implement
    :meth:`torch.distributed._spmd.Override.replacement` to replace an original
    submodule with the return new submodule. The new submodule contains
    operations that users preferred to be traced, which simply be a dummy
    placeholder operator. After tracing, users can implement
    :meth:`torch.distributed._spmd.Override.transform` to transform the traced
    graph, where the dummy placeholder operator serves as an anchor to insert
    new sub-graphs.
    """

    # 抽象方法，用于替换原始模块
    @abstractmethod
    def replacement(self, fqn: str, orig_submodule: torch.nn.Module) -> torch.nn.Module:
        r"""Implement this method to return a new :class:`nn.Module` instance to replace the ``orig_submodule``
        argument in the model.

        This helps if ``orig_submodule`` is not traceable or should not be traced.

        Args:
            fqn (str): fully quantified name of the submodule.
            orig_submodule (class:`nn.Module`): original submodule instance to replace.

        Returns:
            A new :class:`nn.Module` instance to replace the original one.

        """
        pass

    # 抽象方法，用于转换追踪的图形
    @abstractmethod
    def transform(
        self,
        gm: fx.GraphModule,
        flat_state: List[torch.Tensor],
        **kwargs: Any
    ) -> None:
        r"""Implement this method to transform the traced graph.

        Args:
            gm (fx.GraphModule): the graph module containing the traced graph.
            flat_state (List[torch.Tensor]): the flattened state of the model.
            **kwargs (Any): additional keyword arguments.

        """
        pass
    ) -> fx.GraphModule:
        r"""
        给定一个 DTensor 扩展的图和每个节点的分片模式，
        如果需要，对由 :meth:`torch.distributed._spmd.Override.replacement` 返回的 :class:`nn.Module`
        进行子图的额外转换。

        Args:
            gm (:class:`fx.Graph`): DTensor 扩展的图。
            flat_state (List[str, :class:`Tensor`]): 对扁平化状态列表的引用。在图中，前 ``len(flat_state)`` 个占位符与这些元素相对应。
                转换可以向 ``flat_state`` 添加状态或从中删除状态，只要保持 ``flat_state`` 和占位符一致即可。

        Returns:
            转换后的 :class:`fx.Graph`。

        """
        pass
class _PyTreeCodeGenOutputsOnly(_PyTreeCodeGen):
    # 忽略 Pyre 类型检查错误代码[3]
    def process_inputs(self, *args: Any) -> Any:
        # 将输入参数直接返回
        return args

    # 忽略 Pyre 类型检查错误代码[2, 3]
    def gen_fn_def(self, free_vars, maybe_return_annotation):
        # 调用父类 CodeGen 的 gen_fn_def 方法，并返回结果
        return CodeGen.gen_fn_def(self, free_vars, maybe_return_annotation)


def _to_caller_flattened_graph_module(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """将展平输入参数的责任从图模块移交给调用者。

    示例：

        output = gm(my_struct)

        gm = gm(to_caller_flattened_graph_module)

        output = gm(*pytree.flatten(my_struct)[0])

    """
    # 忽略 Pyre 类型检查错误代码[16]
    gm._graph._codegen = _PyTreeCodeGenOutputsOnly(
        pytree_info=_PyTreeInfo(
            # 忽略 Pyre 类型检查错误代码[6]
            orig_args=None,  # 类型: 忽略[arg-type]
            # 忽略 Pyre 类型检查错误代码[6]
            in_spec=None,  # 类型: 忽略[arg-type]
            # 忽略 Pyre 类型检查错误代码[16]
            out_spec=gm._graph._codegen.pytree_info.out_spec,
        )
    )
    gm.recompile()
    return gm


# 为了保留旧有行为并避免破坏现有代码，暂时使用 dtensor 的展开模式
dtensor_expand_mode = DTensorExpandMode()


def _override_placements(t: torch.Tensor, placements: List[Placement]):
    global dtensor_expand_mode
    dtensor_expand_mode._placements_override[id(t)] = placements


@contextmanager
def _rematerialize_optimizer(
    opt: torch.optim.Optimizer,
    named_states: Dict[str, Any],
    params: Dict[str, nn.Parameter],
):
    assert opt is not None

    # 使用代理张量更新 opt.state
    orig_states = copy(opt.state)
    for n in named_states:
        # opt.state 的键类型为字符串，但优化器使用参数作为键
        opt.state[params[n]] = named_states[n]  # 类型: 忽略[index]

    # FIXME: 支持多个参数组
    param_group = opt.param_groups[0]
    orig_params = param_group["params"]
    param_group["params"] = params.values()

    try:
        yield
    finally:
        param_group["params"] = orig_params
        opt.state = orig_states


aten = torch.ops.aten  # 忽略 Pyre


@contextmanager
def _enable_compile():
    # torch._utils.is_compiling 的返回值影响优化器的行为
    # 我们需要该函数返回 True 以便将优化器包含在图中
    # 参考：https://github.com/pytorch/pytorch/blob/a524123c91ab399c9dd6882c1189596dd77e7734/torch/optim/optimizer.py#L41
    def f_true():
        return True

    orig_is_compiling_code = torch._utils.is_compiling.__code__
    torch._utils.is_compiling.__code__ = f_true.__code__
    try:
        yield
    finally:
        torch._utils.is_compiling.__code__ = orig_is_compiling_code


def _foreach_add_decomp(self, other, alpha=1):
    self_updated = aten._foreach_add.List(self, other, alpha=alpha)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_unaop_decomp(op, self):
    self_updated = op(self)
    # 使用 zip 函数同时遍历 self 和 self_updated 这两个对象
    for s, s_u in zip(self, self_updated):
        # 将 s_u 的值复制给 s，即将 self_updated 中的元素值复制到 self 中对应位置的元素
        s.copy_(s_u)
def _foreach_binop_list_decomp(op, self, other):
    # 对 self 和 other 执行二元操作 op，并将结果赋值给 self_updated
    self_updated = op(self, other)
    # 遍历 self 和 self_updated 中的元素，用 self_updated 中的值替换 self 中的值
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_binop_scalar_decomp(op, self, scalar=1):
    # 对 self 和标量 scalar 执行二元操作 op，并将结果赋值给 self_updated
    self_updated = op(self, scalar)
    # 遍历 self 和 self_updated 中的元素，用 self_updated 中的值替换 self 中的值
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_addcop_scalar_decomp(op, self, tensor1, tensor2, scalar=1):
    # 对 self, tensor1 和 tensor2 执行带标量的操作 op，并将结果赋值给 self_updated
    self_updated = op(self, tensor1, tensor2, scalar)
    # 遍历 self 和 self_updated 中的元素，用 self_updated 中的值替换 self 中的值
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _fused_adam_decomp(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr=1,
    beta1=1,
    beta2=1,
    weight_decay=1,
    eps=1,
    amsgrad=True,
    maximize=True,
    grad_scale=None,
    found_inf=None,
):
    # 将输入的参数打包成元组 orig_tuple
    orig_tuple = (self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs)
    # 调用底层的 fused Adam 算子，传递所有输入参数，并将返回值存储在 updated_tuple 中
    updated_tuple = aten._fused_adam.default(
        self,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )

    # 遍历 orig_tuple 和 updated_tuple 的元素
    for idx, (orig, updated) in enumerate(zip(orig_tuple, updated_tuple)):
        # 如果 idx == 1，跳过梯度复制，因为不需要将梯度复制回去
        if idx == 1:
            continue
        # 遍历 orig 和 updated 中的元素，用 updated 中的值替换 orig 中的值
        for o, u in zip(orig, updated):
            o.copy_(u)


SPMD_DECOMP_TABLE = {
    # 定义一系列 aten 操作及其对应的处理函数
    aten._foreach_add_.List: _foreach_add_decomp,
    aten._foreach_add_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_add.Scalar
    ),
    aten._foreach_addcdiv_.Scalar: partial(
        _foreach_addcop_scalar_decomp, aten._foreach_addcdiv.Scalar
    ),
    aten._foreach_addcmul_.Scalar: partial(
        _foreach_addcop_scalar_decomp, aten._foreach_addcmul.Scalar
    ),
    aten._foreach_div_.List: partial(
        _foreach_binop_list_decomp, aten._foreach_div.List
    ),
    aten._foreach_mul_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_mul.Scalar
    ),
    aten._foreach_div_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_div.Scalar
    ),
    aten._foreach_neg_.default: partial(
        _foreach_unaop_decomp, aten._foreach_neg.default
    ),
    aten._foreach_reciprocal_.default: partial(
        _foreach_unaop_decomp, aten._foreach_reciprocal.default
    ),
    aten._foreach_sqrt_.default: partial(
        _foreach_unaop_decomp, aten._foreach_sqrt.default
    ),
    aten._foreach_sub_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_sub.Scalar
    ),
    aten._fused_adam_.default: _fused_adam_decomp,
    aten.native_layer_norm_backward.default: native_layer_norm_backward,
}


DEDUP_TARGETS: Set[torch._ops.OpOverload] = {
    # 定义一组需要去重的 torch 操作
    torch.ops._c10d_functional.all_reduce.default,
    torch.ops._c10d_functional.wait_tensor.default,
}
def _dedup_collectives(gm: fx.GraphModule) -> fx.GraphModule:
    # 创建一个字典，用于存储每个节点的参数元组到节点对象的映射关系
    args_to_node: Dict[Tuple[Any, ...], fx.Node] = {}

    # 遍历计算图中的每个节点
    for node in gm.graph.nodes:
        # 提取节点的所有参数，这些参数可能是树状结构的叶子节点
        args = pytree.arg_tree_leaves(*node.args)

        # 如果节点的目标函数在DEDUP_TARGETS中
        if node.target in DEDUP_TARGETS:
            # 构造参数元组作为键，包括目标函数和参数
            args_key = (node.target, *args)
            # 查找是否已经存在相同参数元组的节点
            unique_node = args_to_node.get(args_key, None)
            if unique_node is None:
                # 第一次遇到这个参数组合，记录下来
                args_to_node[args_key] = node
            else:
                # 当前节点是重复的，用之前记录的节点替换它
                node.replace_all_uses_with(unique_node)
                gm.graph.erase_node(node)

    # 重新编译图模块
    gm.recompile()

    # 返回更新后的图模块
    return gm


@dataclass
class _CompiledResult:
    gm: fx.GraphModule
    mod: nn.Module
    opt: Optional[torch.optim.Optimizer]
    flat_state: List[torch.Tensor]


def _compile(
    func: Callable,
    module_override: Optional[List[Override]],
    parallel_mode: ParallelMode,
    *args: Any,
    **kwargs: Any,
) -> _CompiledResult:
    # 1. 从参数和关键字参数中提取出 nn.Module 和 Optimizer
    # FIXME(@mrshenli): 支持多个 nn.Module 实例
    # FIXME(@mrshenli): 支持多个 Optimizer 实例
    # FIXME(@mrshenli): 需要广播模型以同步参数
    mod, opt = None, None
    for arg in pytree.arg_tree_leaves(*args, **kwargs):
        if isinstance(arg, nn.Module):
            assert mod is None, "目前只支持单个 nn.Module"
            mod = arg
        if isinstance(arg, torch.optim.Optimizer):
            assert opt is None, "目前只支持单个 Optimizer"
            opt = arg

    assert mod is not None, "无法从参数中找到 nn.Module 实例."

    # 2. 使用虚拟替换覆盖目标子模块（例如 MoE）
    if module_override:
        accessor = NamedMemberAccessor(mod)

        def swap(fqn_prefix: str, module: torch.nn.Module) -> None:
            for override in module_override:  # type: ignore[union-attr]
                for name, child in module.named_children():
                    if len(name) == 0:
                        continue
                    fqn = fqn_prefix + "." + name if fqn_prefix != "" else name
                    new_child = override.replacement(fqn, child)
                    if id(new_child) == id(child):
                        swap(fqn, new_child)
                    else:
                        accessor.swap_submodule(fqn, new_child)

        swap("", mod)

    # 3. 跟踪 train_step 的无状态版本
    # 获取命名参数和缓冲区的字典
    params = dict(mod.named_parameters(remove_duplicate=False))
    buffers = dict(mod.named_buffers(remove_duplicate=False))

    named_states = {}
    if opt is not None:
        # 如果优化器不为 None，则传递 named_states 而不是 opt.state 给 stateless_func，
        # 因为后者使用 nn.Parameter 作为键。在跟踪期间，需要确保优化器可以通过代理张量找到状态。
        for n, p in params.items():
            if p in opt.state:
                # opt.state 的键类型是字符串，但优化器使用 Parameter 作为键
                named_states[n] = opt.state[p]  # type: ignore[index]

    is_data_parallel_mode = isinstance(parallel_mode, DataParallel)

    # 将状态和参数提升为函数参数，以便 make_fx 可以跟踪应用于它们的操作。
    def stateless_func(func, params, buffers, named_states, args, kwargs):
        with stateless._reparametrize_module(
            mod, {**params, **buffers}
        ), _rematerialize_optimizer(
            opt, named_states, params
        ) if opt else nullcontext():
            # 对于 DataParallel 模式，首先安装钩子以标记梯度
            with gradients_tagging(params) if is_data_parallel_mode else nullcontext():
                ret = func(*args, **kwargs)

            # 确保更新后的参数被返回
            return ret, list(mod.parameters()), list(named_states.values())  # type: ignore[union-attr]

    # FIXME: 使用符号跟踪来解决在 DTensor 扩展模式中的问题。
    # 否则会出现形状不匹配错误，因为我们使用局部输入来跟踪局部图，并使用 DTensor 扩展运算符，
    # 其中 DTensor 的形状是全局形状。
    tracing_mode = "fake" if is_data_parallel_mode else "symbolic"

    if is_data_parallel_mode:
        fake_mode = FakeTensorMode()
        data_parallel_mode = cast(DataParallel, parallel_mode)

        def _get_full_batch_arg(arg: torch.Tensor) -> torch.Tensor:
            # 因为编译发生在第一次迭代时，我们接收到小批量输入，首先将它们转换为完整批量
            # 为了数据并行分片传播，使用假张量输入
            fake_arg = fake_mode.from_tensor(arg)
            arg_dims = [1] * arg.ndim
            # 在批次维度上将张量扩展到完整批量大小
            arg_dims[data_parallel_mode.input_batch_dim] *= dist.get_world_size()
            return fake_arg.repeat(arg_dims)

        args = pytree.tree_map_only(
            torch.Tensor,
            _get_full_batch_arg,
            args,
        )
        kwargs = pytree.tree_map_only(
            torch.Tensor,
            _get_full_batch_arg,
            kwargs,
        )
    # 启用编译，并检测异常
    with _enable_compile(), torch.autograd.detect_anomaly(check_nan=False):
        # FIXME(@mrshenli): 目前我们的用例中，对于 foreach 操作，功能化尚未生效。
        # 使用显式分解来代替。解决以下问题后移除此处代码。
        # 问题详情：https://github.com/pytorch/pytorch/issues/97852
        gm = make_fx(
            partial(stateless_func, func),  # 使用部分函数应用构造无状态函数
            tracing_mode=tracing_mode,
            decomposition_table=SPMD_DECOMP_TABLE,
            _allow_non_fake_inputs=False,
        )(params, buffers, named_states, args, kwargs)

    # 创建参数和缓冲区的字典
    params_and_buffers: Dict[str, Union[torch.Tensor, nn.Parameter]] = {
        **params,
        **buffers,
    }

    # 4. 并行模式下，将单设备图扩展为分布式图
    gm = parallel_mode.partition(
        gm,
        mod,
        opt,
        params_and_buffers,
        named_states,
        args,
        kwargs,
    )

    # 5. 将展开输入参数的责任从图模块移交给调用者。这样做有两个目的：
    #   - 添加/移除状态的变换需要操纵一个状态容器，该容器按照图占位符中的顺序维护状态张量。
    #   - 减少运行时成本。状态容器只在最开始展开一次。
    flat_state = pytree.tree_leaves([params_and_buffers, named_states])
    gm = _to_caller_flattened_graph_module(gm)

    # 6. 去重通信操作符。
    # 重复可能来自 DTensor 参数和 kwargs 重分发。
    # 假设一个操作符生成部分梯度张量，并且模型参数被复制。在这种情况下，每个使用该部分梯度张量的优化操作都会触发 allreduce。
    # 这是因为 DTensor 只具有单个张量/操作符的本地信息，这不足以检测图中的重复。如果一个参数在前向方法中多次使用，插入 FSDP allgather 也可能导致此类情况发生。
    # TODO(@mrshenli): @yifuwang 建议在追踪器级别进行扩展和去重，以避免多次图遍历。
    gm = _dedup_collectives(gm)

    # 7. 使用真实图替换先前插入的虚拟图。
    if module_override:
        for override in module_override:
            gm = override.transform(gm, flat_state)

    # 返回编译结果对象
    return _CompiledResult(gm, mod, opt, flat_state)
# 定义编译后对象的键名，用于存储编译后的对象
COMPILED_OBJECT_KEY = "_compiled_obj"

# 定义编译函数，用于优化一个可调用对象，例如在训练循环中的训练步骤
def compile(
    module_override: Optional[List[Override]] = None,
    gm_transformation: Optional[Callable[[fx.GraphModule], fx.GraphModule]] = None,
    parallel_mode: Optional[ParallelMode] = None,
):
    r"""Compile and optimize a callable, which can be a train step within a training loop.

    This method will extract :class:`nn.Module` and :class:`torch.optim.Optimizer`
    instances from the input arguments and trace operations applied to their
    parameters and states.

    Args:
        module_override (Optional[List[Override]]): a list of Override instances
            that will be applied to the module in order. The :class:`Override`
            objects provide :class:`nn.Module` replacements during tracing and a
            graph transformation function after tracing. (Default: ``None``)
        gm_transformation (Optional[Callable[fx.GraphModule, fx.GraphModule]]):
            a callback that will be called after the original callable is
            compiled and distributed (usually after the first iteration) to
            transform the compiled GraphModule into a new optimized one.
        parallel_mode (Optional[ParallelMode]): a :class:`ParallelMode` object
            that specifies how to parallelize the callable. Each ParallelMode
            would have its own strategy to partition the model and the captured
            graph (Default: ``None``)

    """
    def inner(func: Callable):
        # 内部函数定义，接受一个可调用对象作为参数
        @wraps(func)
        # 使用 functools.wraps 来保留原始函数的元数据
        def wrapper(*args, **kwargs):
            # 包装函数，接受任意数量的位置参数和关键字参数
            last_train_step = kwargs.pop("last_train_step", False) if kwargs else False
            # 从关键字参数中弹出 "last_train_step"，默认为 False
            first_iter = False
            # 初始化 first_iter 标志为 False
            # 将 COMPILED_OBJECT_KEY 放在 wrapper 函数的属性中，而不是 func 函数中，
            # 因为用户将获取 wrapper 函数。
            compiled_obj = wrapper.__dict__.get(COMPILED_OBJECT_KEY, None)
            # 从 wrapper 函数的字典中获取 COMPILED_OBJECT_KEY 对应的对象
    
            if compiled_obj is None:
                # 如果 compiled_obj 为空，则进行首次迭代的设置
                first_iter = True
                # 设置 first_iter 为 True
                global dtensor_expand_mode
                # 声明全局变量 dtensor_expand_mode
                mode: ParallelMode = (
                    dtensor_expand_mode if parallel_mode is None else parallel_mode
                )
                # 根据 parallel_mode 设置 mode 变量
    
                compiled_obj = _compile(func, module_override, mode, *args, **kwargs)
                # 编译 func 函数，使用给定的 module_override 和 mode 参数
                wrapper.__dict__[COMPILED_OBJECT_KEY] = compiled_obj
                # 将编译后的对象保存在 wrapper 函数的属性中
    
            flat_inps = compiled_obj.flat_state + pytree.arg_tree_leaves(
                *args, **kwargs
            )
            # 构建 flat_inps 列表，包括 compiled_obj 的 flat_state 和参数树叶子节点
    
            with torch.no_grad():
                # 进入 torch 的无梯度上下文
                # 注意：我们不需要 autograd，因为反向传播已经在图中捕获。
    
                if first_iter and gm_transformation:
                    # 如果是第一次迭代且存在 gm_transformation 函数
                    # TODO: SPMD 应该提供一个默认且可配置的转换。
                    compiled_obj.gm = gm_transformation(compiled_obj.gm)
                    # 对 compiled_obj.gm 进行转换操作
    
                if not last_train_step:
                    output = compiled_obj.gm(*flat_inps)[0]
                    # 如果不是最后一个训练步骤，则调用 compiled_obj.gm，并取第一个返回值作为 output
                else:
                    # 如果是最后一个训练步骤
                    # 调用 IterGraphModule.forward()，传入 last_iter 参数，并捕获异常，
                    # 如果 compiled_obj 没有被 IterGraphModule 包装。
                    try:
                        output = compiled_obj.gm(*flat_inps, last_iter=last_train_step)[0]
                    except TypeError as e:
                        if "last_iter" not in str(e):
                            raise e
                        output = compiled_obj.gm(*flat_inps)[0]
                        # 捕获异常并使用没有 last_iter 参数的方式调用 compiled_obj.gm
    
                return output
                # 返回 output 结果
    
        return wrapper
        # 返回 wrapper 函数作为内部函数 inner 的结果
```