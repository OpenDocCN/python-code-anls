# `.\pytorch\torch\_export\non_strict_utils.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类
import contextlib  # 提供上下文管理工具的模块
import inspect  # 提供检查源码的模块
from collections import defaultdict  # 提供默认字典的模块
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union  # 引入类型提示的相关工具

import torch  # 引入PyTorch模块
import torch.utils._pytree as pytree  # 引入PyTorch的_pytree模块
from torch._dynamo.source import (  # 从torch._dynamo.source模块导入多个类
    AttrSource,  # 用于表示属性来源的类
    GetItemSource,  # 用于表示索引来源的类
    LocalSource,  # 用于表示本地变量来源的类
    TensorProperty,  # 表示张量属性的类
    TensorPropertySource,  # 用于表示张量属性来源的类
)
from torch._dynamo.variables.builder import TrackedFake  # 导入TrackedFake类
from torch._export.passes.add_runtime_assertions_for_constraints_pass import InputDim  # 导入InputDim类
from torch._export.passes.lift_constants_pass import ConstantAttrMap  # 导入ConstantAttrMap类
from torch._guards import Source  # 导入Source类
from torch._library.fake_class_registry import FakeScriptObject  # 导入FakeScriptObject类
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode  # 导入FakeTensor和FakeTensorMode类
from torch.export import Constraint  # 导入Constraint类
from torch.export.dynamic_shapes import _tree_map  # 导入_tree_map函数
from torch.export.graph_signature import CustomObjArgument  # 导入CustomObjArgument类
from torch.fx.experimental.symbolic_shapes import (  # 从torch.fx.experimental.symbolic_shapes导入多个类和函数
    ConstraintViolationError,  # 表示约束违规错误的类
    DimDynamic,  # 表示动态维度的枚举
    EqualityConstraint,  # 表示相等约束的类
    ShapeEnv,  # 表示形状环境的类
    StatelessSymbolicContext,  # 表示无状态符号上下文的类
    ValueRanges,  # 表示值范围的类
)

from torch.utils._pytree import (  # 从torch.utils._pytree导入多个类和函数
    GetAttrKey,  # 用于表示获取属性的关键类
    KeyPath,  # 用于表示键路径的类
    MappingKey,  # 用于表示映射键的类
    SequenceKey,  # 用于表示序列键的类
    tree_map_with_path,  # 用于路径映射的函数
)

if TYPE_CHECKING:
    from sympy import Symbol  # 如果是类型检查模式，导入Symbol类


def key_path_to_source(kp: KeyPath) -> Source:
    """
    Given a key path, return the source for the key path.
    """
    # 初始化本地变量来源为"args"
    source: Source = LocalSource("args")
    # 遍历键路径中的每个键
    for k in kp:
        # 根据键的类型选择不同的来源类型
        if isinstance(k, SequenceKey):
            source = GetItemSource(source, k.idx)
        elif isinstance(k, MappingKey):
            source = GetItemSource(source, k.key)
        elif isinstance(k, GetAttrKey):
            source = AttrSource(source, k.name)
        else:
            # 如果键类型未知，则抛出异常
            raise ValueError(f"Unknown KeyEntry {k}")

    return source


def _is_constant_argument(t):
    # 检查参数是否为常量（None、int、float、bool或str类型）
    return t is None or isinstance(t, (int, float, bool, str))


def fakify(
    mode: FakeTensorMode,
    kp: KeyPath,
    t: Any,
    t_constraints: Dict[int, Dict[int, Constraint]],
    sources: Dict[Tuple[int, int], List[Source]],
):
    # 获取键路径对应的来源对象
    source = key_path_to_source(kp)
    # 如果参数是常量或者是torch.ScriptObject，则直接返回参数
    if _is_constant_argument(t) or isinstance(t, torch.ScriptObject):
        return t

    # 如果参数不是torch.Tensor类型，则抛出异常
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"Unsupported input type {type(t)}")

    # 获取张量的维度数
    n_dims = len(t.shape)
    # 创建无状态符号上下文对象，指定动态大小和约束大小的初始值
    symbolic_context = StatelessSymbolicContext(
        dynamic_sizes=[DimDynamic.STATIC] * n_dims,
        constraint_sizes=[None] * n_dims,
    )
    # 获取张量的唯一标识符
    t_id = id(t)
    # 如果张量在约束字典中存在
    if t_id in t_constraints:
        # 遍历张量的约束字典
        for i, constraint in t_constraints[t_id].items():
            # 设置约束大小和动态大小
            symbolic_context.constraint_sizes[i] = constraint.constraint_range
            symbolic_context.dynamic_sizes[i] = DimDynamic.DYNAMIC
            # 创建张量属性来源对象，并将其添加到来源字典中
            src = TensorPropertySource(base=source, prop=TensorProperty.SIZE, idx=i)
            sources[(t_id, i)].append(src)
            # 将来源名称映射到调试名称中
            mode.shape_env.source_name_to_debug_name[src.name()] = constraint.debug_name  # type: ignore[assignment]
    # 使用指定的模式对象从张量 t 中创建一个虚拟值 fake
    fake = mode.from_tensor(t, source=source, symbolic_context=symbolic_context)
    # 将创建的虚拟值 fake 以及其来源 source 和符号上下文 symbolic_context 加入到模式的形状环境的 tracked_fakes 列表中
    mode.shape_env.tracked_fakes.append(TrackedFake(fake, source, symbolic_context))  # type: ignore[union-attr]
    # 返回创建的虚拟值 fake
    return fake
def make_fake_params_buffers(
    fake_mode: FakeTensorMode,
    params_buffers: Dict[str, torch.Tensor],
) -> Dict[str, Union[torch.Tensor, torch.nn.Parameter]]:
    # 初始化一个空字典，用于存储伪造的参数缓冲区
    faked_params_buffers = {}
    # 创建一个空字典 memo 用于存储已经处理过的参数缓冲区的映射关系
    memo: Dict[int, FakeTensor] = {}
    # 遍历输入的参数缓冲区字典
    for key, value in params_buffers.items():
        # 如果当前 value 对象的 id 已经在 memo 中存在，则直接使用 memo 中保存的 FakeTensor
        if id(value) in memo:
            fake_tensor = memo[id(value)]
        else:
            # 否则，使用 fake_mode 从给定的 value 创建一个新的 FakeTensor，并将其存入 memo
            fake_tensor = fake_mode.from_tensor(value, static_shapes=True)
            memo[id(value)] = fake_tensor
        # 将处理后的 FakeTensor 存入 faked_params_buffers 中
        faked_params_buffers[key] = fake_tensor
    # 返回伪造的参数缓冲区字典
    return faked_params_buffers  # type: ignore[return-value]


def make_fake_inputs(
    nn_module,
    args,
    kwargs,
    dynamic_shapes,
    _is_torch_jit_trace=False,
    _allow_complex_guards_as_runtime_asserts=False,
):
    """
    Given an nn module, example inputs, and constraints, return a new fake mode,
    fake inputs created in that mode whose dynamic shape dimensions are constrained
    by the given ranges, and sources for pairs of dynamic shape dimensions that are
    constrained to be equal.
    """
    # TODO(avik): refactor Dynamo to avoid duplication of the following code
    # between non-strict and strict.
    # Specifically, here (non-strict) we do the following pre-tracing steps:
    #   - Fakify inputs.
    #   - Process input shape equalities.
    # In strict, these steps are spread across multiple files:
    #   - output_graph.py fakifies inputs.
    #   - [post-tracing] guards.py processes input shape equalities.

    # 调用 torch.export.dynamic_shapes._process_dynamic_shapes 处理动态形状约束
    constraints = torch.export.dynamic_shapes._process_dynamic_shapes(
        nn_module, args, kwargs, dynamic_shapes, _is_torch_jit_trace=_is_torch_jit_trace
    )
    # 如果 constraints 为 None，则置为一个空列表
    constraints = constraints or []
    # 初始化一个 defaultdict，用于存储约束条件的映射关系
    t_constraints: Dict[int, Dict[int, Constraint]] = defaultdict(dict)
    # 将 constraints 中的每个约束条件按照 t_id 和 dim 存入 t_constraints
    for constraint in constraints:
        t_constraints[constraint.t_id][constraint.dim] = constraint
        # 如果约束条件中存在共享的约束，则也存入相应的 t_constraints
        if constraint.shared is not None:
            t_constraints[constraint.shared.t_id][constraint.shared.dim] = constraint

    # 尝试获取当前的 TracingContext
    context = torch._guards.TracingContext.try_get()
    # 如果 context 不为 None，则表示正在导出过程中已经存在顶层的 TracingContext
    if context is not None:
        # 在这种情况下，已经存在一个顶层的 TracingContext，其中已经有一个 fake_mode
        # 不需要创建新的 fake_mode，并且不应该有任何约束条件存在
        assert (
            len(constraints) == 0
        ), "Found constraints when tracing with a toplevel tracing context."
        fake_mode = context.fake_mode
    elif not _is_torch_jit_trace:
        # 如果不是 Torch JIT 跟踪模式，则获取神经网络模块的前向方法的代码对象
        code = nn_module.forward.__code__
        # 提取代码对象的相关字段，包括函数名、文件名和起始行号
        co_fields = {
            "co_name": code.co_name,
            "co_filename": code.co_filename,
            "co_firstlineno": code.co_firstlineno,
        }
        # 创建一个 FakeTensorMode 对象，使用 ShapeEnv 初始化，传入代码对象的字段以及一些运行时断言选项
        fake_mode = FakeTensorMode(
            shape_env=ShapeEnv(
                tracked_fakes=[],
                co_fields=co_fields,
                prefer_deferred_runtime_asserts_over_guards=_allow_complex_guards_as_runtime_asserts,
                _allow_complex_guards_as_runtime_asserts=_allow_complex_guards_as_runtime_asserts,
            ),
            allow_non_fake_inputs=True,
            export=True,
        )
    else:
        # 如果是 Torch JIT 跟踪模式，则创建一个 FakeTensorMode 对象，仅使用 ShapeEnv 初始化和一些运行时断言选项
        fake_mode = FakeTensorMode(
            shape_env=ShapeEnv(
                tracked_fakes=[],
                prefer_deferred_runtime_asserts_over_guards=_allow_complex_guards_as_runtime_asserts,
                _allow_complex_guards_as_runtime_asserts=_allow_complex_guards_as_runtime_asserts,
            ),
            allow_non_fake_inputs=True,
        )
    # 检查 fake_mode 的 shape_env 属性是否存在以及是否包含 tracked_fakes，若不符合条件则抛出 ValueError 异常
    if fake_mode.shape_env is None or fake_mode.shape_env.tracked_fakes is None:
        raise ValueError(
            "Detected fake_mode does not have a shape_env with tracked fakes. "
            "If you constructed the module under a FakeTensorMode, "
            "please initialize it like: FakeTensorMode(shape_env=ShapeEnv(tracked_fakes=[]))"
        )

    # 使用 fake_mode 进入上下文管理器
    with fake_mode:
        # FIXME(ycao) ScriptMethod doesn't have signature, I am using an empty one to unblock
        # 如果不是 Torch JIT 跟踪模式，则获取神经网络模块前向方法的签名，否则设为 None
        if not _is_torch_jit_trace:
            original_signature = inspect.signature(nn_module.forward)
        else:
            original_signature = None
        
        # 初始化 sources 字典，用于存储路径对应的源信息列表
        sources: Dict[Tuple[int, int], List[Source]] = defaultdict(list)
        
        # 使用 tree_map_with_path 函数，对参数 args 和 kwargs 进行遍历映射并生成 fake_args 和 fake_kwargs
        fake_args, fake_kwargs = tree_map_with_path(
            lambda kp, val: fakify(fake_mode, kp, val, t_constraints, sources),
            (args, kwargs),
        )

        # 初始化 source_pairs、derived_equalities 和 phantom_symbols 用于存储约束信息和符号映射
        source_pairs: List[Tuple[Source, Source]] = []
        derived_equalities: List[Tuple[Source, Union[Source, Symbol], Callable]] = []
        phantom_symbols: Dict[str, Symbol] = {}
        
        # 遍历 constraints 列表，调用 torch.export.dynamic_shapes._process_equalities 处理等式约束
        for constraint in constraints:
            torch.export.dynamic_shapes._process_equalities(
                constraint,
                lambda t_id, dim: sources[(t_id, dim)],
                fake_mode.shape_env,
                source_pairs,
                derived_equalities,
                phantom_symbols,
            )

        # 初始化 equalities_inputs，存储等式约束的输入信息
        equalities_inputs = EqualityConstraint(
            source_pairs=source_pairs,
            derived_equalities=derived_equalities,
            phantom_symbols=list(phantom_symbols.values()),
            warn_only=False,
        )
        
        # 返回 fake_mode 对象、fake_args、fake_kwargs、equalities_inputs 和 original_signature
        return fake_mode, fake_args, fake_kwargs, equalities_inputs, original_signature
def _flatten_dynamic_shapes(
    combined_args: Dict[str, Any],
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any]],
) -> List[Any]:
    # 初始化一个空列表，用于存储扁平化后的形状信息
    flat_shapes = []

    def _tree_map_helper(t, shape):
        nonlocal flat_shapes
        # 将每个形状信息添加到 flat_shapes 列表中
        flat_shapes.append(shape)

    # 调用 _tree_map 函数，对 combined_args 和 dynamic_shapes 进行处理
    _tree_map(_tree_map_helper, combined_args, dynamic_shapes)
    # 返回扁平化后的形状信息列表
    return flat_shapes


def produce_guards_and_solve_constraints(
    fake_mode: FakeTensorMode,
    gm: torch.fx.GraphModule,
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any], None],
    equalities_inputs: EqualityConstraint,
    original_signature: inspect.Signature,
    _disable_forced_specializations: Optional[bool] = False,
    _is_torch_jit_trace=False,
):
    """
    Given a fake mode, sources pairs corresponding to equal dynamic shape dimensions,
    and a graph module, produce guards on the fake mode's shape env (raising constraint
    violations if any), solve (to suggest simplifications or fixes).
    Dynamo already performs this, so this is for non-strict mode.

    Additional inputs:
        equalities_inputs: the equality constraints to use for guards
        original_signature: the signature of the forward method
        _disable_forced_specializations: if True, avoids forced specializations
    """
    # 获取 fake_mode 的形状环境
    shape_env = fake_mode.shape_env
    assert shape_env.tracked_fakes is not None

    # 获取跟踪的虚拟张量信息
    placeholders = [tf.fake for tf in shape_env.tracked_fakes]
    # 获取虚拟张量的来源信息
    sources = [tf.source for tf in shape_env.tracked_fakes]
    # 获取虚拟张量的符号上下文信息
    input_contexts = [tf.symbolic_context for tf in shape_env.tracked_fakes]
    constraint_violation_error = None
    try:
        # 在形状环境中生成保护条件
        shape_env.produce_guards(
            placeholders,
            sources,
            input_contexts=input_contexts,
            equalities_inputs=equalities_inputs,
            ignore_static=False,
            _disable_forced_specializations=_disable_forced_specializations,
        )
    except ConstraintViolationError as e:
        # 捕获约束违规错误并记录
        constraint_violation_error = e

    # 将形状环境标记为冻结状态
    shape_env.frozen = True
    # 获取形状约束
    dim_constraints = shape_env.dim_constraints
    if dim_constraints is None:
        # 如果形状约束为空，则抛出之前捕获到的约束违规错误
        assert constraint_violation_error
        raise constraint_violation_error

    # 解决形状约束
    dim_constraints.solve(
        _disable_forced_specializations=_disable_forced_specializations
    )
    # 移除多余的动态结果
    dim_constraints.remove_redundant_dynamic_results()
    # 获取强制特化信息
    forced_specializations = dim_constraints.forced_specializations()
    if not _is_torch_jit_trace:
        # 生成漂亮的结果消息
        msg = dim_constraints.prettify_results(
            original_signature,
            dynamic_shapes,
            constraint_violation_error,
            forced_specializations,
        )
    else:
        # FIXME(ycao): This is a hack to get around missing signature from ScriptMethod
        # 如果不满足上述条件，执行以下语句：
        #   - 添加一个临时的错误信息，绕过 ScriptMethod 缺失的签名问题
        msg = "dummy constraint violation message"
    if constraint_violation_error:
        # 如果存在约束违规错误对象，则执行以下语句：
        #   - 更新错误对象的参数，将新的消息附加到原始错误消息中
        constraint_violation_error.args = (constraint_violation_error.args[0] + msg,)
    elif forced_specializations:
        # 否则，如果存在强制特化条件：
        #   - 创建一个新的约束违规错误对象，使用当前消息作为其初始化参数
        constraint_violation_error = ConstraintViolationError(msg)
    if constraint_violation_error:
        # 如果最终存在约束违规错误对象，则执行以下语句：
        #   - 抛出约束违规错误对象，中断程序执行
        raise constraint_violation_error
def make_constraints(
    fake_mode: FakeTensorMode,
    gm: torch.fx.GraphModule,
    combined_args: Dict[str, Any],
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any], None],
    num_lifted_inputs: int,
):
    """
    Given a fake mode's shape env and user-specified dynamic shapes,
    return the resulting range constraints and equality constraints.

    Additional args:
        num_lifted_inputs: the number of non-user-input placeholder nodes in the graph
        (used only to enumerate the user-input nodes)
    """

    # 获取虚拟模式的形状环境
    shape_env = fake_mode.shape_env
    # 从图模块的元数据中获取内联约束列表
    inline_constraints = gm.meta.get("inline_constraints", [])
    # 创建包含内联约束的范围约束字典
    range_constraints = {
        symbol: inline_constraints[symbol] for symbol in inline_constraints
    }
    
    # 如果没有动态形状，则直接返回范围约束字典
    if not dynamic_shapes:
        return range_constraints

    # 获取每个输入的动态形状规格
    if not isinstance(dynamic_shapes, dict):
        assert isinstance(dynamic_shapes, (tuple, list))
        # 以动态形状的值创建新的组合参数
        combined_args = type(dynamic_shapes)(combined_args.values())  # type: ignore[assignment, misc]
    # 展平动态形状以便处理
    flat_dynamic_shapes = _flatten_dynamic_shapes(combined_args, dynamic_shapes)

    # 检查形状数量是否与输入数量匹配
    num_placeholders = [node.op == "placeholder" for node in gm.graph.nodes].count(True)
    assert len(flat_dynamic_shapes) == num_placeholders - num_lifted_inputs

    # 初始化输入维度字典和自由符号集合
    input_dims = defaultdict(list)
    free_symbols = set()

    # 遍历图中的节点，处理占位符输入
    for input_index, node in enumerate(gm.graph.nodes):
        if input_index < num_lifted_inputs or node.op != "placeholder":
            continue
        if _is_constant_argument(node.meta["val"]) or isinstance(
            node.meta["val"], CustomObjArgument
        ):
            continue
        # 获取节点的形状规格
        shape_spec = flat_dynamic_shapes[input_index - num_lifted_inputs]
        # 处理节点的每个维度
        for i, d in enumerate(node.meta["val"].shape):
            if isinstance(d, torch.SymInt):
                # 查找与此形状维度对应的符号的范围约束
                # 并存储在以符号表达式索引的范围约束中
                dim = shape_spec[i] if shape_spec else None
                if dim:
                    range_constraints[d.node.expr] = ValueRanges(
                        lower=dim.min, upper=dim.max
                    )
                else:
                    range_constraints[d.node.expr] = shape_env.var_to_range[
                        d.node._expr
                    ]
                # 将输入维度信息添加到相应的符号表达式的列表中
                input_dims[d.node.expr].append(InputDim(input_name=node.name, dim=i))
                # 更新自由符号集合
                free_symbols.update(d.node.expr.free_symbols)
    # 遍历自由符号列表，自由符号通常是指在某个上下文中未定义但被引用的符号
    for symbol in free_symbols:
        # 如果该符号不在已记录的范围约束中
        if symbol not in range_constraints:
            # 对于占位符，可能具有从派生表达式推导出的符号形状
            # 上述代码将为这些符号记录直接的范围约束
            # 以便我们可以在运行时进行断言。此外，对于序列化和反序列化检查
            # 我们希望记录它们根符号的范围约束。
            range_constraints[symbol] = shape_env.var_to_range[symbol]

    # 返回最终记录的范围约束字典
    return range_constraints
def _gather_constant_attrs(m: torch.nn.Module) -> ConstantAttrMap:
    """Search the module hierarchy, gathering up all tensor and ScriptObject constants.

    Returns a dictionary mapping hash(value) to the name of the constant. We
    have to abuse `hash` here unfortunately, see: [ScriptObject hash].
    """
    constants = ConstantAttrMap()  # 创建一个常量属性映射对象

    # 收集模块的缓冲区和参数
    buffers_parameters = set(m.buffers())
    buffers_parameters.update(m.parameters())

    def inner(m: torch.nn.Module, prefix_atoms: List[str], constants):
        for k, v in m.__dict__.items():  # 遍历模块的字典属性
            if isinstance(
                v,
                (
                    torch.Tensor,
                    torch.ScriptObject,
                    FakeScriptObject,
                ),
            ):
                if v in buffers_parameters:
                    # 过滤掉缓冲区和参数，只保留常量
                    continue

                fqn = ".".join(prefix_atoms + [k])  # 构建完全限定名
                constants.add(v, fqn)  # 将常量和其完全限定名添加到常量映射中
        for k, v in m.named_children():  # 遍历命名子模块
            inner(v, prefix_atoms + [k], constants)

    inner(m, [], constants)  # 调用内部递归函数开始收集常量
    return constants  # 返回收集到的常量属性映射


@contextlib.contextmanager
def _fakify_script_objects(
    mod: torch.nn.Module,
    args: Tuple[Any],
    kwargs: Dict[Any, Any],
    fake_mode: torch._subclasses.fake_tensor.FakeTensorMode,
):
    """This context manager is used to fakify script objects into FakeScriptObject.

    Inputs:
      mod: the module to be exported, it (and its recursive submodules)'s script object attrs haven't been fakified.
      args, kwargs: the args and kwargs inputs for mod, script object inputs haven't been fakified.
      fake_mode: the fake mode to be used for fakifying script objects. It's the same mode that fakify input tensors.

    Returns:
      mod: the patched module, its (and its recursive submodules) script object attrs have been fakified.
      fake_args, fake_kwargs: new fakified args and kwargs.
          Script object inputs have been fakified. Don't touch the tensors.
      fake_constant_attrs: a new map from FakeScriptObject to the fqn of the original script object.
      fake_to_real: a mapping between FakeScriptObject and the original script object in order to un-do the patching.
    """

    constant_attrs: ConstantAttrMap = _gather_constant_attrs(mod)  # 收集模块的常量属性
    assert not any(
        isinstance(obj, FakeScriptObject) for obj in constant_attrs.values()
    ), "Mod shouldn't contain any FakeScriptObject."  # 断言：模块中不应该包含任何 FakeScriptObject

    assert not pytree.tree_any(
        lambda obj: isinstance(obj, FakeScriptObject), (args, kwargs)
    ), "args and kwargs shouldn't contain any FakeScriptObject."  # 断言：args 和 kwargs 中不应包含任何 FakeScriptObject

    patched_attr = {}
    fake_constant_attrs = ConstantAttrMap()  # 创建一个新的常量属性映射对象
    fake_to_real = {}  # 创建一个从 FakeScriptObject 到原始 ScriptObject 的映射

    def _maybe_fakify_obj(obj):
        """Helper function to potentially convert an object to its fakified version."""
        fake_obj = torch._library.fake_class_registry.to_fake_obj(fake_mode, obj)  # 将对象转换为其伪造版本
        fake_to_real[fake_obj] = obj  # 记录伪造对象和原始对象的映射
        return fake_obj

    def _leaf_mod_and_attr(
        mod: torch.nn.Module, attr_fqn: str
        # 以下部分未完，不足以构成函数
    ) -> Tuple[torch.nn.Module, str]:
        *prefix_attr, last_attr = attr_fqn.split(".")
        cur_mod = mod
        for attr in prefix_attr:
            cur_mod = getattr(cur_mod, attr)
        return cur_mod, last_attr

这部分代码定义了一个函数签名，接受一个字符串 `attr_fqn`，返回一个元组，包含两个元素：一个是 `torch.nn.Module` 类型的对象 `cur_mod`，另一个是字符串 `last_attr`。


    try:
        for obj, fqns in constant_attrs.items():
            if isinstance(obj, torch.ScriptObject):
                fake_script_obj = _maybe_fakify_obj(obj)
                for fqn in fqns:
                    cur_mod, attr = _leaf_mod_and_attr(mod, fqn)
                    assert obj is getattr(cur_mod, attr)
                    setattr(cur_mod, attr, fake_script_obj)
                    fake_constant_attrs.add(fake_script_obj, fqn)
                    patched_attr[fqn] = obj
            else:
                for fqn in fqns:
                    fake_constant_attrs.add(obj, fqn)

在这个 `try` 块中，首先遍历 `constant_attrs` 字典，其中 `obj` 可能是 `torch.ScriptObject` 类型的对象。如果是这种情况，会对 `obj` 进行一些修改和替换，然后更新 `fake_constant_attrs` 和 `patched_attr`。如果不是 `torch.ScriptObject` 类型，则直接将 `obj` 添加到 `fake_constant_attrs` 中。


        fake_args, fake_kwargs = pytree.tree_map_only(
            torch.ScriptObject, _maybe_fakify_obj, (args, kwargs)
        )

这里调用了 `pytree.tree_map_only` 函数，将 `args` 和 `kwargs` 中的元素都应用 `_maybe_fakify_obj` 函数并替换成 `torch.ScriptObject` 类型的对象，然后分别赋给 `fake_args` 和 `fake_kwargs`。


        yield (mod, fake_args, fake_kwargs, fake_constant_attrs, fake_to_real)

使用 `yield` 关键字返回一个生成器对象，生成一个包含 `mod`, `fake_args`, `fake_kwargs`, `fake_constant_attrs`, `fake_to_real` 的元组。


    finally:
        for fqn, orig_obj in patched_attr.items():
            cur_mod, attr = _leaf_mod_and_attr(mod, fqn)
            setattr(cur_mod, attr, orig_obj)

`finally` 块确保无论是否发生异常，都会执行其中的代码。在这里，遍历 `patched_attr` 字典，将原始对象 `orig_obj` 恢复到 `cur_mod` 的 `attr` 属性上。
```