# `.\pytorch\torch\_higher_order_ops\triton_kernel_wrap.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和库
import dataclasses  # 导入用于数据类的模块
import inspect  # 导入用于对象内省的模块
import logging  # 导入日志记录模块
import threading  # 导入线程相关的模块
from collections import defaultdict  # 导入默认字典
from typing import Any, Dict, List, Optional, Union  # 导入类型提示相关的模块

import torch.utils._pytree as pytree  # 导入PyTorch的私有模块_pytree
from torch import Tensor  # 导入张量类型
from torch._C import DispatchKey  # 导入DispatchKey类
from torch._ops import HigherOrderOperator  # 导入HigherOrderOperator类
from torch._prims_common import clone_preserve_strides  # 导入克隆保留步幅的函数
from torch._subclasses.fake_tensor import FakeTensorMode  # 导入伪张量模式
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,  # 导入禁用代理模式追踪的函数
    ProxyTorchDispatchMode,  # 导入代理Torch调度模式
    track_tensor_tree,  # 导入跟踪张量树的函数
)

# 获取名为 "torch._dynamo" 的日志记录器对象
log = logging.getLogger("torch._dynamo")

###############################################################################
# Kernel Side Table

# Triton内核无法放入FX图中，因为图节点不支持任意函数。
# 使用一个辅助表格。
# 使用两个字典，使得获取内核和ID的操作都是O(1)复杂度。
class KernelSideTable:
    id_to_kernel: Dict[int, Any] = dict()  # 存储ID到内核对象的映射的字典
    kernel_to_id: Dict[Any, int] = dict()  # 存储内核对象到ID的映射的字典
    constant_args: Dict[int, Any] = dict()  # 存储常量参数的字典
    lock = threading.Lock()  # 线程锁对象，用于保证线程安全性

    # 向表中添加内核对象，并返回其索引
    def add_kernel(self, kernel) -> int:
        with self.lock:
            if kernel in self.kernel_to_id:
                return self.kernel_to_id[kernel]

            idx = len(self.id_to_kernel)
            self.id_to_kernel[idx] = kernel
            self.kernel_to_id[kernel] = idx
            return idx

    # 根据索引获取对应的Triton内核
    def get_kernel(self, idx: int):
        # 由于从字典中获取操作是原子性的，因此无需在此加锁
        assert idx in self.id_to_kernel
        return self.id_to_kernel[idx]

    # 将常量参数添加到表中，并返回其索引
    def add_constant_args(self, args) -> int:
        with self.lock:
            idx = len(self.constant_args)
            self.constant_args[idx] = args
            return idx

    # 根据索引获取常量参数
    def get_constant_args(self, idx: int):
        # 由于从字典中获取操作是原子性的，因此无需在此加锁
        assert idx in self.constant_args
        return self.constant_args[idx]

    # 重置表格的内容（仅用于单元测试）
    # 这只在单线程执行时是安全的
    def reset_table(self) -> None:
        self.id_to_kernel = dict()
        self.kernel_to_id = dict()
        self.constant_args = dict()


# 创建 KernelSideTable 类的实例，命名为 kernel_side_table
kernel_side_table = KernelSideTable()


###############################################################################
# Mutation Tracker

# 使用 dataclass 装饰器定义 Param 类，代表参数
@dataclasses.dataclass(frozen=True)
class Param:
    idx: int  # 参数的索引


# 使用 dataclass 装饰器定义 Intermediate 类，代表中间结果
@dataclasses.dataclass(frozen=True)
class Intermediate:
    idx: int  # 中间结果的索引

    # 判断是否为伪中间结果
    def fake(self):
        return self.idx < 0


# 使用 dataclass 装饰器定义 Op 类，代表操作
@dataclasses.dataclass(frozen=True)
class Op:
    name: str  # 操作的名称
    fn_call_name: Optional[str]  # 函数调用的名称（可选）
    args: List[Union[Param, Intermediate]]  # 操作的参数列表，包含 Param 或 Intermediate 对象
    ret: Intermediate = dataclasses.field(repr=False)  # 操作的返回结果，为 Intermediate 类型的对象
    # 初始化对象后处理方法，用于确保对象状态正确
    def __post_init__(self):
        # 如果对象的名称为 "tt.call"
        if self.name == "tt.call":
            # 断言函数调用名称不为空
            assert self.fn_call_name is not None
        else:
            # 断言函数调用名称为空
            assert self.fn_call_name is None
def generate_ttir(kernel, kwargs):
    """
    Uses Triton's internal code generation to create TTIR
    """
    # 导入必要的库和模块
    import sympy
    import triton
    from triton.compiler.compiler import ASTSource
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    import torch
    import torch._inductor.ir
    from torch._subclasses.fake_tensor import FakeTensor

    # 如果 kernel 是 Autotuner 的实例，则使用其配置中的第一个版本的参数
    if isinstance(kernel, Autotuner):
        if len(kernel.configs) > 0:
            kwargs = {**kwargs, **kernel.configs[0].kwargs}
        kernel = kernel.fn

    # 确保 kernel 是 JITFunction 的实例
    assert isinstance(kernel, JITFunction)

    # 检查传入的参数数量是否与 kernel 的参数名列表长度相匹配
    if len(kwargs) != len(kernel.arg_names):
        raise ValueError("Incorrect number of arguments passed to kernel")

    # 为了 TTIR 生成，替换所有 SymExprs 为常规值，替换所有 FakeTensor/TensorBox 为真实的张量
    ordered_args: Dict[str, Any] = {}
    for name in kernel.arg_names:
        a = kwargs[name]
        if isinstance(a, (torch.SymInt, torch.SymFloat, torch.SymBool, sympy.Expr)):
            ordered_args[name] = 2  # 使用常规值代替 SymExprs
        elif isinstance(a, (FakeTensor, torch._inductor.ir.TensorBox)):
            with torch._C._DisableTorchDispatch():
                ordered_args[name] = torch.empty(2, dtype=a.dtype)  # 使用真实张量代替 FakeTensor/TensorBox
        else:
            ordered_args[name] = a

    # 获取所有是 Tensor 的参数名列表
    ordered_tensor_names = [
        name for name, arg in ordered_args.items() if isinstance(arg, Tensor)
    ]

    # 获取 kernel 的特化配置
    specialization = kernel._get_config(*ordered_args.values())

    # 创建常量字典，排除是 Tensor 的参数
    constants = {
        i: arg
        for i, arg in enumerate(ordered_args.values())
        if not isinstance(arg, Tensor)
    }

    # 构建 kernel 签名，不包括 constexpr 参数
    signature = {
        i: kernel._type_of(kernel._key_of(arg))
        for i, arg in enumerate(ordered_args.values())
        if i not in kernel.constexprs
    }

    # 创建 Triton 的上下文和后端
    context = triton._C.libtriton.ir.context()
    target = triton.runtime.driver.active.get_current_target()
    backend = triton.compiler.compiler.make_backend(target)
    options = backend.parse_options(dict())

    # 加载 Triton 的方言和后端
    triton._C.libtriton.ir.load_dialects(context)
    backend.load_dialects(context)

    # 创建 ASTSource 对象
    src = ASTSource(kernel, signature, constants, specialization)

    # 根据 Triton 的版本选择合适的方法生成 TTIR
    if len(inspect.signature(src.make_ir).parameters) == 2:
        ttir_module = src.make_ir(options, context)
    else:
        codegen_fns = backend.get_codegen_implementation()
        ttir_module = src.make_ir(options, codegen_fns, context)

    # 验证生成的 TTIR 模块
    if not ttir_module.verify():
        raise RuntimeError("Verification for TTIR module has failed")

    # 返回生成的 TTIR 模块和有序的 Tensor 名称列表
    return ttir_module, ordered_tensor_names
def ttir_to_functions(ttir_module) -> Dict[str, Dict[Intermediate, List[Op]]]:
    """
    Walk the `ttir_module` bottom up to mine the `functions` from
    the structured MLIR entities representing the Triton kernel
    (mlir::Operation, mlir::Block, mlir::Region).
    """
    # 初始化一个空的函数字典，用于存储函数名到中间表达式和操作列表的映射关系
    functions: Dict[str, Dict[Intermediate, List[Op]]] = {}

    # 用于存储操作堆栈，以及每个块 ID 对应的中间表达式和操作列表的默认字典
    op_stack: Dict[int, Dict[Intermediate, List[Op]]] = defaultdict(
        lambda: defaultdict(list)
    )

    # 初始化用于存储区域 ID 到块 ID 列表的字典
    region_id_to_block_ids: Dict[int, List[int]] = defaultdict(list)

    # 初始化用于存储块 ID 到块参数 ID 列表的字典
    block_id_to_block_arg_ids: Dict[int, List[int]] = {}

    # 初始化用于存储替换操作的字典，将操作 ID 映射到中间表达式或参数对象
    replacements: Dict[int, Union[Intermediate, Param]] = {}

    # 初始化重索引映射，用于将操作 ID 重新映射到新的索引
    reindex_map: Dict[int, int] = {}

    # 下一个虚拟中间表达式的索引
    next_fake_intermediate = 0

    # 定义重索引函数，如果索引尚未被映射，则将其映射到新的索引位置
    def reindex(idx):
        if idx not in reindex_map:
            reindex_map[idx] = len(reindex_map)
        return reindex_map[idx]

    # 使用 mlir_to_functions 函数遍历 ttir_module
    ttir_module.walk(mlir_to_functions)

    # 返回收集到的函数字典
    return functions


class MemoizeWithCycleCheck:
    def __init__(self, fn):
        self.fn = fn
        self.reset()

    def __call__(self, functions, fn_name, num_args):
        # 生成缓存键
        key = (fn_name, num_args)
        # 如果缓存中不存在该键，则调用函数计算结果并缓存
        if key not in self.cache:
            self.cache[key] = None
            self.cache[key] = self.fn(functions, fn_name, num_args)
        # 如果计算结果为 None，则抛出递归运行时错误
        if self.cache[key] is None:
            raise RuntimeError("Recursion is not supported")
        # 返回缓存中的计算结果
        return self.cache[key]

    def reset(self):
        # 重置缓存
        self.cache = {}


@MemoizeWithCycleCheck
def analyze_kernel_mutations(functions, fn_name, num_args):
    """
    Analyzes the graph to detect all sinks from a predefined list of sinks
    by using triton's MemWrite trait list. NOTE: What if triton exposed this?
    From each sink, it traverses the CFG backwards to identify all the input
    pointers that are mutated.
    """
    # 定义具有 MemWrite 特征的变异操作的名称及其变异的参数索引列表
    MUTATION_OPS = {"tt.store": [0], "tt.atomic_cas": [0], "tt.atomic_rmw": [0]}
    
    # 定义需要中止分析的操作名称集合
    UNKNOWN_OPS = {"tt.elementwise_inline_asm"}

    # 初始化操作堆栈
    stack: List[Union[Param, Intermediate]] = []

    # 初始化已访问的操作集合
    visited = set()

    # 从函数字典中获取指定函数的操作列表
    ops = functions[fn_name]

    # 遍历所有操作列表
    for op_list in ops.values():
        for op in op_list:
            # 如果操作名称在未知操作集合中，则抛出运行时错误
            if op.name in UNKNOWN_OPS:
                raise RuntimeError(
                    f"ttir analysis hit an op we do not know how to analyze: {op.name}"
                )

            # 如果操作名称为 "tt.call"，递归分析被调用函数
            if op.name == "tt.call":
                assert op.fn_call_name in functions
                # 分析被调用函数的变异操作，并将结果入栈
                mutations = analyze_kernel_mutations(
                    functions, op.fn_call_name, len(op.args)
                )
                stack.extend(arg for arg, mutated in zip(op.args, mutations) if mutated)
            else:
                # 遍历操作名称对应的变异操作索引列表，将参数入栈
                for idx in MUTATION_OPS.get(op.name, []):
                    stack.append(op.args[idx])
    # 这是一个迭代的深度优先搜索（DFS）算法
    mutated = [False] * num_args  # 创建一个长度为 num_args 的布尔类型列表，表示每个参数是否已经被修改过
    while stack:
        arg = stack.pop()  # 从栈中取出一个参数

        if arg in visited:  # 如果参数已经访问过，则跳过
            continue

        visited.add(arg)  # 将参数标记为已访问

        if isinstance(arg, Param):  # 如果参数是 Param 类型的实例
            if arg.idx >= num_args:
                # 这是在内核中定义的参数，而非传递进来的参数
                continue
            mutated[arg.idx] = True  # 将对应参数位置的 mutated 列表项标记为 True，表示参数已被修改

        elif isinstance(arg, Intermediate) and not arg.fake():  # 如果参数是 Intermediate 类型的实例且不是虚构的
            for op in ops[arg]:  # 遍历与参数相关联的操作列表
                # 跳过名称不为 "tt.load" 的操作
                if op.name != "tt.load":
                    stack.extend(op.args)  # 将操作的参数列表加入栈中继续处理

    return mutated  # 返回标记了被修改参数的列表
# 用于识别变异张量的函数，接收一个 Triton 内核和内核参数作为输入
def identify_mutated_tensors(kernel, kwargs):
    """
    Given a triton kernel and the arguments for this kernel, this function
    1) Retrieves the TTIR converted version of the kernel from Triton's API.
    2) Parses the TTIR and creates a control flow graph
    3) Analyzes the graph to detect all input tensor mutations
    """

    ttir_module = None  # 初始化 TTIR 模块为 None
    functions = None     # 初始化函数集合为 None
    try:
        ttir_module, ordered_tensor_names = generate_ttir(kernel, kwargs)

        # 使用 Triton 代码暴露的 MLIR 绑定从 TTIR 中提取函数
        functions = ttir_to_functions(ttir_module)

        assert functions is not None  # 断言函数集合不为 None
        kernel_name = next(iter(functions.keys()))
        # Triton 代码生成修改了名称
        assert kernel.fn.__name__ in kernel_name
        # 重置缓存以便在顶层调用之间清理
        # 用于分析内核变异的缓存主要用于循环检测，因此每个顶层调用都需要一个干净的缓存
        analyze_kernel_mutations.reset()
        # 分析内核的变异情况，返回变异的张量名称列表
        mutations = analyze_kernel_mutations(
            functions, kernel_name, len(ordered_tensor_names)
        )

        return [
            ordered_tensor_names[i] for i, mutated in enumerate(mutations) if mutated
        ]
    except Exception as e:
        # 遇到异常时记录警告信息，并假定每个输入都发生了变异
        log.warning(
            "Encountered an exception in identify_mutated_tensors, assuming every input is mutated",
            exc_info=True,
        )
        # 如果存在 TTIR 模块，则记录调试信息
        if ttir_module is not None:
            log.debug("TTIR:\n%s", str(ttir_module))
        # 如果存在函数集合，则记录调试信息
        if functions is not None:
            log.debug("functions:")
            for name, fn in functions.items():
                log.debug("===\t%s\t===", name)
                for ret, ops in fn.items():
                    log.debug("%s\t=>\t%s", ret, ops)
        # 返回所有是张量的参数键列表作为默认处理
        return [key for key, value in kwargs.items() if isinstance(value, Tensor)]


###############################################################################
# Triton Kernel Wrappers


# 用于包装 Triton 内核的类
class TritonKernelWrapperMutation(HigherOrderOperator):
    def __init__(self):
        super().__init__("triton_kernel_wrapper_mutation")


triton_kernel_wrapper_mutation = TritonKernelWrapperMutation()


# 用于函数式包装 Triton 内核的类
class TritonKernelWrapperFunctional(HigherOrderOperator):
    def __init__(self):
        super().__init__("triton_kernel_wrapper_functional")


triton_kernel_wrapper_functional = TritonKernelWrapperFunctional()


# 用于在函数式方式下包装 Triton 内核的具体实现
@triton_kernel_wrapper_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_mutation_dense(
    *, kernel_idx, constant_args_idx, grid, kwargs
):
    from torch._inductor.codegen.wrapper import user_defined_kernel_grid_fn_code

    kernel = kernel_side_table.get_kernel(kernel_idx)
    constant_args = kernel_side_table.get_constant_args(constant_args_idx)

    if len(grid) == 1:
        grid_fn = grid[0]
    # 如果不是内置的 kernel，则根据用户定义的 kernel 函数名称、配置和网格生成代码
    fn_name, code = user_defined_kernel_grid_fn_code(
        kernel.fn.__name__, kernel.configs, grid
    )
    # 创建一个空的命名空间字典
    namespace: Dict[str, Any] = {}
    # 在命名空间中执行生成的代码
    exec(code, namespace)
    # 从命名空间中获取用户定义的 kernel 函数
    grid_fn = namespace[fn_name]

    # 调用 kernel 对象的 grid_fn 方法，传入关键字参数和常量参数
    kernel[grid_fn](**kwargs, **constant_args)
# 使用 Triton 框架的装饰器，将函数标记为模拟实现（fake tensor mode）
@triton_kernel_wrapper_mutation.py_impl(FakeTensorMode)
def triton_kernel_wrapper_mutation_fake_tensor_mode(
    mode, *, kernel_idx, constant_args_idx, grid, kwargs
):
    # 进入指定的模式上下文
    with mode:
        # 返回空值
        return None


# 定义一个函数，用于追踪 Triton 框架内核包装器的调用
def trace_triton_kernel_wrapper(proxy_mode, func_overload, node_args):
    # 禁用代理模式的追踪功能
    with disable_proxy_modes_tracing():
        # 调用指定的函数重载并获取结果
        out = func_overload(**node_args)

    # 使用代理模式解包节点参数
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    # 创建一个代理对象，用于追踪函数调用
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        func_overload,
        (),
        proxy_args,
        name=func_overload.__name__ + "_proxy",
    )
    # 返回追踪后的张量树结果
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


# 使用 Triton 框架的装饰器，将函数标记为代理 Torch 调度模式的模拟实现
@triton_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_mutation_proxy_torch_dispatch_mode(
    mode, *, kernel_idx, constant_args_idx, grid, kwargs
):
    # 如果模式启用追踪功能
    if mode.enable_tracing:
        # 调用追踪 Triton 内核包装器函数
        trace_triton_kernel_wrapper(
            mode,
            triton_kernel_wrapper_mutation,
            {
                "kernel_idx": kernel_idx,
                "constant_args_idx": constant_args_idx,
                "grid": grid,
                "kwargs": kwargs,
            },
        )
    else:
        # 否则直接调用 Triton 内核包装器函数
        triton_kernel_wrapper_mutation(
            kernel_idx=kernel_idx,
            constant_args_idx=constant_args_idx,
            grid=grid,
            kwargs=kwargs,
        )

    # 返回空值
    return None


# 使用 Triton 框架的函数化实现装饰器
@triton_kernel_wrapper_mutation.py_functionalize_impl
def triton_kernel_wrapper_mutation_functionalize(
    ctx, kernel_idx, constant_args_idx, grid, kwargs
):
    # 解包张量参数
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    # 获取指定内核的核心对象
    kernel = kernel_side_table.get_kernel(kernel_idx)
    # 获取常量参数对象
    constant_args = kernel_side_table.get_constant_args(constant_args_idx)

    # TODO(oulgen): 已知 bug，如果两个内核输入是彼此的视图，并且一个在内核中被修改，
    # 而稍后另一个被修改，则它们不再相等。在 Dynamo 中早期通过图破坏此条件来修复此问题。
    # 识别在内核中发生变异的张量，并返回需要克隆的张量集合
    tensors_to_clone = identify_mutated_tensors(
        kernel, {**unwrapped_kwargs, **constant_args}
    )

    # 重定向到下一个上下文
    with ctx.redispatch_to_next():
        # 使用 Triton 框架的函数化内核包装器执行
        unwrapped_outputs = triton_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            constant_args_idx=constant_args_idx,
            grid=grid,
            kwargs=unwrapped_kwargs,
            tensors_to_clone=tensors_to_clone,
        )

    # 断言：确保解包后的输出键在原始输入键的子集中
    assert set(unwrapped_outputs.keys()).issubset(set(kwargs.keys()))
    # 遍历解包后的输出键值对
    for key, output_arg in unwrapped_outputs.items():
        # 如果输出参数不是张量，则继续下一次循环
        if not isinstance(output_arg, Tensor):
            continue
        # 获取对应输入参数
        input_arg = kwargs[key]
        # 断言：确保输入参数是张量
        assert isinstance(input_arg, Tensor)

        # 替换输入参数为输出参数
        ctx.replace(input_arg, output_arg)
        # 标记此替换对自动求导不可见
        ctx.mark_mutation_hidden_from_autograd(input_arg)
        # 提交更新
        ctx.commit_update(input_arg)
        # 同步参数
        ctx.sync(input_arg)

    # 返回空值
    return None
@triton_kernel_wrapper_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
# 定义一个函数装饰器，指定该函数实现为 Triton 的函数式编程接口，使用复合显式自动微分的调度键
def triton_kernel_wrapper_functional_dense(
    *, kernel_idx, constant_args_idx, grid, kwargs, tensors_to_clone
):
    # TODO(oulgen): For performance reasons, we want to ensure that these
    # `clone_preserve_strides` calls are never executed at runtime
    # (inductor should always optimize them away).
    # Requires https://github.com/pytorch/pytorch/issues/109240
    # 根据 `tensors_to_clone` 中的键是否在 `kwargs` 中，选择性地对其进行 `clone_preserve_strides` 操作
    kwargs = {
        key: (clone_preserve_strides(val) if key in tensors_to_clone else val)
        for key, val in kwargs.items()
    }
    # 调用另一个函数 `triton_kernel_wrapper_mutation`，传递一些参数进行内核封装的变异处理
    triton_kernel_wrapper_mutation(
        kernel_idx=kernel_idx,
        constant_args_idx=constant_args_idx,
        grid=grid,
        kwargs=kwargs,
    )
    # 返回一个字典，包含那些在 `tensors_to_clone` 中的 `kwargs` 键和对应的值
    return {key: val for key, val in kwargs.items() if key in tensors_to_clone}


@triton_kernel_wrapper_functional.py_impl(FakeTensorMode)
# 定义一个函数装饰器，指定该函数实现为 Triton 的函数式编程接口，使用虚拟张量模式的调度键
def triton_kernel_wrapper_functional_fake_tensor_mode(
    mode, *, kernel_idx, constant_args_idx, grid, kwargs, tensors_to_clone
):
    # TODO(oulgen): For performance reasons, we want to ensure that these
    # `clone_preserve_strides` calls are never executed at runtime
    # (inductor should always optimize them away).
    # Requires https://github.com/pytorch/pytorch/issues/109240
    # 使用虚拟张量模式 `mode` 下的上下文，对 `kwargs` 中需要克隆的张量执行 `clone_preserve_strides` 操作
    with mode:
        return {
            key: clone_preserve_strides(val)
            for key, val in kwargs.items()
            if key in tensors_to_clone
        }


@triton_kernel_wrapper_functional.py_impl(ProxyTorchDispatchMode)
# 定义一个函数装饰器，指定该函数实现为 Triton 的函数式编程接口，使用代理 PyTorch 调度模式的调度键
def triton_kernel_wrapper_functional_proxy_torch_dispatch_mode(
    mode, *, kernel_idx, constant_args_idx, grid, kwargs, tensors_to_clone
):
    if mode.enable_tracing:
        # 如果 `mode` 启用追踪，则调用 `trace_triton_kernel_wrapper` 函数进行追踪封装处理
        return trace_triton_kernel_wrapper(
            mode,
            triton_kernel_wrapper_functional,
            {
                "kernel_idx": kernel_idx,
                "constant_args_idx": constant_args_idx,
                "grid": grid,
                "kwargs": kwargs,
                "tensors_to_clone": tensors_to_clone,
            },
        )
    else:
        # 否则直接调用 `triton_kernel_wrapper_functional` 函数，传递相应参数
        return triton_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            grid=grid,
            kwargs=kwargs,
            tensors_to_clone=tensors_to_clone,
        )


@triton_kernel_wrapper_functional.py_functionalize_impl
# 使用 Triton 的函数式化实现装饰器来修饰函数
def triton_kernel_wrapper_functional_functionalize(
    ctx, kernel_idx, constant_args_idx, grid, kwargs, tensors_to_clone
):
    # 使用 `ctx` 上下文对象，对 `kwargs` 中的张量进行解包操作
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    # 使用 `ctx` 上下文对象重新调度到下一个操作
    with ctx.redispatch_to_next():
        # 调用 `triton_kernel_wrapper_functional` 函数，并包装其输出结果
        outputs = triton_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            constant_args_idx=constant_args_idx,
            grid=grid,
            kwargs=unwrapped_kwargs,
            tensors_to_clone=tensors_to_clone,
        )
        # 返回经过 `ctx` 包装的输出结果
        return ctx.wrap_tensors(outputs)


# 将 `triton_kernel_wrapper_mutation` 函数设置为 `DispatchKey.PythonDispatcher` 下的默认情况
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
# 调用 triton_kernel_wrapper_mutation 对象的 fallthrough 方法，指定 DispatchKey.PythonTLSSnapshot 作为参数
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]

# 调用 triton_kernel_wrapper_mutation 对象的 fallthrough 方法，指定 DispatchKey.ADInplaceOrView 作为参数
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.ADInplaceOrView)

# 调用 triton_kernel_wrapper_mutation 对象的 fallthrough 方法，指定 DispatchKey.BackendSelect 作为参数
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.BackendSelect)

# 调用 triton_kernel_wrapper_mutation 对象的 fallthrough 方法，指定 DispatchKey.AutocastCPU 作为参数
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]

# 调用 triton_kernel_wrapper_mutation 对象的 fallthrough 方法，指定 DispatchKey.AutocastCUDA 作为参数
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]

# 调用 triton_kernel_wrapper_mutation 对象的 fallthrough 方法，指定 DispatchKey.AutogradCUDA 作为参数
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutogradCUDA)

# 调用 triton_kernel_wrapper_mutation 对象的 fallthrough 方法，指定 DispatchKey.AutogradCPU 作为参数
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutogradCPU)

# 调用 triton_kernel_wrapper_functional 对象的 fallthrough 方法，指定 DispatchKey.PythonDispatcher 作为参数
triton_kernel_wrapper_functional.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]

# 调用 triton_kernel_wrapper_functional 对象的 fallthrough 方法，指定 DispatchKey.PythonTLSSnapshot 作为参数
triton_kernel_wrapper_functional.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]

# 调用 triton_kernel_wrapper_functional 对象的 fallthrough 方法，指定 DispatchKey.ADInplaceOrView 作为参数
triton_kernel_wrapper_functional.fallthrough(DispatchKey.ADInplaceOrView)

# 调用 triton_kernel_wrapper_functional 对象的 fallthrough 方法，指定 DispatchKey.BackendSelect 作为参数
triton_kernel_wrapper_functional.fallthrough(DispatchKey.BackendSelect)

# 调用 triton_kernel_wrapper_functional 对象的 fallthrough 方法，指定 DispatchKey.AutocastCPU 作为参数
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]

# 调用 triton_kernel_wrapper_functional 对象的 fallthrough 方法，指定 DispatchKey.AutocastCUDA 作为参数
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]

# 调用 triton_kernel_wrapper_functional 对象的 fallthrough 方法，指定 DispatchKey.AutogradCUDA 作为参数
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutogradCUDA)

# 再次调用 triton_kernel_wrapper_functional 对象的 fallthrough 方法，指定 DispatchKey.AutogradCPU 作为参数
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutogradCPU)
```