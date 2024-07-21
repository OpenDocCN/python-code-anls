# `.\pytorch\torch\_inductor\codegen\cpp.py`

```
# 添加类型注释，允许未定义的函数
def reduction_init(reduction_type: str, dtype: torch.dtype) -> None:
    # 检查是否是 Windows 系统
    _IS_WINDOWS = sys.platform == "win32"
    # 获取名为 schedule 的日志记录器对象
    schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")

    # 定义原生支持的 OpenMP 归约操作类型集合
    NATIVE_OMP_RTYPES = {"+", "*", "^", "||", "min", "max"}
    
    # 定义将 Python 中的归约操作类型映射到对应的 C++ 操作符的字典
    RTYPE_TO_CPP = {
        "sum": "+",
        "prod": "*",
        "xor_sum": "^",
        "min": "min",
        "max": "max",
        "argmin": "argmin",
        "argmax": "argmax",
        "any": "||",
        "welford_reduce": "welford",
        "welford_combine": "welford",
    }
    
    # 定义支持向量化的归约操作类型集合
    VECTORIZABLE_RTYPES = {
        "max",
        "min",
        "sum",
        "prod",
        "xor_sum",
        "welford_reduce",
        "welford_combine",
    }

    # 定义将 Python 类型映射到对应的 C++ 类型的字典
    PYTHON_TO_CPP = {
        "Tensor": "at::Tensor",
        "int": "long",
        "float": "double",
        "bool": "bool",
        "str": "std::string",
        "ScalarType": "c10::ScalarType",
        "MemoryFormat": "at::MemoryFormat",
        "Layout": "at::Layout",
        "Device": "at::Device",
        "number": "at::Scalar",
    }

    # 定义将 Python 容器类型映射到对应的 C++ 容器类型的字典
    CONTAINER_PYTHON_TO_CPP = {
        "List": "std::vector",
        "Optional": "c10::optional",
    }

    # 定义低精度浮点类型的列表
    DTYPE_LOWP_FP = [
        torch.bfloat16,
        torch.float16,
    ]

    # 定义二元比较操作的列表
    BIN_CMP_OPS = ["eq", "ne", "le", "ge", "lt", "gt"]
    # 如果数据类型为低精度浮点数，则将其提升为32位浮点数，以便进行后续计算
    if dtype in DTYPE_LOWP_FP:
        dtype = torch.float32
    # 对于“xor_sum”、“sum”和“any”三种归约类型，直接返回0
    if reduction_type in ("xor_sum", "sum", "any"):
        return 0
    # 对于归约类型为“prod”，直接返回1
    if reduction_type == "prod":
        return 1
    # 对于归约类型为“max”或“argmax”，根据数据类型返回对应的无穷大或最小值
    if reduction_type in {"max", "argmax"}:
        return (
            f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::min()"
        )
    # 对于归约类型为“min”或“argmin”，根据数据类型返回对应的无穷大或最大值
    if reduction_type in {"min", "argmin"}:
        return (
            f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::max()"
        )
    # 如果是Welford归约类型，则返回对应数据类型的Welford对象
    if is_welford_reduction(reduction_type):
        return f"Welford<{DTYPE_TO_CPP[dtype]}>()"
    # 若以上条件均不满足，则抛出断言错误，表明未知的归约类型
    raise AssertionError(reduction_type)
# 根据给定的规约类型和数据类型，返回相应的缩减类型
def reduction_acc_type(reduction_type, dtype):
    # 确保规约类型不是"argmin"或"argmax"
    assert reduction_type not in {"argmin", "argmax"}
    # 根据数据类型查找对应的计算数据类型，再查找对应的标量类型
    scalar_type = DTYPE_TO_CPP[DTYPE_TO_COMPUTATION_DTYPE[dtype]]
    # 如果是 Welford 规约类型，则返回"Welford<标量类型>"
    if is_welford_reduction(reduction_type):
        return f"Welford<{scalar_type}>"
    # 否则返回标量类型
    return scalar_type


# 根据给定的规约类型、变量和下一个值，返回组合后的表达式
def reduction_combine(reduction_type, var, next_value):
    # 根据规约类型返回相应的组合表达式
    if reduction_type == "sum":
        return f"{var} + {next_value}"
    if reduction_type == "prod":
        return f"{var} * {next_value}"
    if reduction_type == "xor_sum":
        return f"{var} ^ {next_value}"
    if reduction_type == "any":
        return f"{var} || {next_value}"
    if reduction_type in ("min", "max"):
        return f"{reduction_type}_propagate_nan({var}, {next_value})"
    if reduction_type == "welford_reduce":
        return f"welford_combine({var}, {next_value})"
    if reduction_type == "welford_combine":
        # 如果下一个值是元组，则解包为均值、m2 和权重
        if isinstance(next_value, tuple):
            mean, m2, weight = next_value
        else:
            mean, m2, weight = reduction_project(reduction_type, next_value)
        return f"welford_combine({var}, {{{mean}, {m2}, {weight}}})"
    # 抛出异常，表示未知的规约类型
    raise AssertionError(reduction_type)


# 根据给定的规约类型和累加器，返回相应的投影结果
def reduction_project(reduction_type, acc):
    # 如果是 Welford 规约类型，则返回均值、m2 和权重
    if is_welford_reduction(reduction_type):
        return f"{acc}.mean", f"{acc}.m2", f"{acc}.weight"
    # 如果是"argmin"或"argmax"规约类型，则返回索引
    elif reduction_type in {"argmin", "argmax"}:
        return f"{acc}.index"
    # 否则直接返回累加器
    return acc


# 判断表达式是否包含低精度数据类型的转换
def is_to_lowp_dtype(expr):
    # 定义低精度数据类型的转换表达式
    to_exprs = ["convert<half>", "convert<bfloat16>"]
    # 判断表达式是否包含低精度数据类型的转换
    return any(to_expr in expr for to_expr in to_exprs)


# 根据给定的低精度变量、数据类型和内核，返回转换为高精度表达式
def get_lowp_to_high_prec_expr(lowp_var, dtype, kernel):
    # 如果内核是 CppVecKernel 类型，则返回向量转换表达式
    if isinstance(kernel, CppVecKernel):
        return f"at::vec::convert<{DTYPE_TO_CPP[dtype]}>({lowp_var})"
    else:
        assert isinstance(kernel, CppKernel)
        return f"c10::convert<{DTYPE_TO_CPP[dtype]}>({lowp_var})"


# 定义全局变量，用于生成唯一的索引值名称
index_value_name_counter = 1


# 根据给定的规约类型、源数据类型和临时变量，返回前缀、并行前缀和本地初始化
def argmax_argmin_prefix(reduction_type, src_dtype, tmpvar):
    global index_value_name_counter
    # 根据配置动态选择线程数或并行线程数
    num_threads = (
        "max_threads" if config.cpp.dynamic_threads else parallel_num_threads()
    )
    struct_name = f"IndexValue_{index_value_name_counter}"
    index_value_name_counter += 1

    # 定义结构体和临时变量的初始化表达式
    prefix = [
        f"struct {struct_name} {{size_t index; {DTYPE_TO_CPP[src_dtype]} value;}};",
        f"{struct_name} {tmpvar}{{0, {reduction_init(reduction_type, src_dtype)}}};",
    ]
    local_init = [
        f"{struct_name} {tmpvar}_local{{0, {reduction_init(reduction_type, src_dtype)}}};",
    ]
    tmpvar_per_thd = f"{tmpvar}_arr[{num_threads}]"
    parallel_prefix = [
        f"{struct_name} {tmpvar_per_thd};",
    ]
    return prefix, parallel_prefix, local_init


# 缓存函数，根据给定的索引和变量，返回步长
@functools.lru_cache
def stride_at(index: sympy.Expr, var: sympy.Symbol):
    # 定义替换规则，将变量替换为变量加一
    replacement = {var: var + 1}
    new_index = sympy_subs(index, replacement)  # type: ignore[arg-type]
    # 计算步长
    return sympy.simplify(new_index - index)


# 缓存函数
@functools.lru_cache
def simplify_index_in_vec_range(index: sympy.Expr, var: sympy.Expr, vec_length: int):
    """
    Simplifies the index expression within the range of a vectorized loop.
    Given a vectorized loop variable `var` in the range of a loop with `vec_length`,
    this function transforms the `index` into an equivalent form. It handles
    simplifications for cases where `var` can be expressed as `vec_length * a + b`,
    where `b` ranges from 0 to `vec_length - 1`. The function reduces occurrences
    of `FloorDiv` and `ModularIndexing` in the `index` with best-effort optimizations.

    NOTE:
    The simplified index expression is intended for analysis purposes only, not
    for code generation. It replaces `FloorDiv` and `ModularIndexing` with free variables
    which are not dependent on the loop variable `var` in the vectorized range. Check
    https://github.com/pytorch/pytorch/pull/117221#discussion_r1449746217 for more details.

    Examples:
    1. If `var` is `x3` and `vec_length` is 16, and `x3 = 16*a + b`, then
       `FloorDiv(x3, div)` or `ModularIndexing(x3, div, mod)` becomes a free variable
       when `div` is divisible by 16.
    2. `ModularIndexing(x3, 1, mod)` can be simplified to `x3 + c` where `c` is a free
       variable when `mod` is divisible by 16.
    """

    div_freevar_id = 0  # 记录生成的自由变量编号，用于替换 FloorDiv
    mod_freevar_id = 0  # 记录生成的自由变量编号，用于替换 ModularIndexing

    def visit_indexing_div(divisor):
        nonlocal div_freevar_id
        result = FloorDiv(var, divisor)
        if sympy.gcd(divisor, vec_length) == vec_length:
            result = sympy.Symbol(f"{var}_div_c{div_freevar_id}")
            div_freevar_id += 1
        return result

    def visit_modular_indexing(divisor, modulus):
        nonlocal mod_freevar_id
        result = ModularIndexing(var, divisor, modulus)
        if sympy.gcd(divisor, vec_length) == vec_length:
            result = sympy.Symbol(f"{var}_mod_c{mod_freevar_id}")
            mod_freevar_id += 1
        elif divisor == 1 and sympy.gcd(modulus, vec_length) == vec_length:
            result = var + sympy.Symbol(f"{var}_mod_c{mod_freevar_id}")
            mod_freevar_id += 1
        return result

    original_index = index

    div = sympy.Wild("divisor", integer=True)
    if index.has(FloorDiv):
        index = index.replace(FloorDiv(var, div), visit_indexing_div(div))

    mod = sympy.Wild("modulus", integer=True)
    if index.has(ModularIndexing):
        index = index.replace(ModularIndexing(var, div, mod), visit_modular_indexing(div, mod))

    index = sympy.simplify(index)  # 简化处理后的索引表达式
    if index != original_index:
        return simplify_index_in_vec_range(index, var, vec_length)

    return index


@functools.lru_cache
def stride_at_vec_range(index: sympy.Expr, var: sympy.Symbol, vec_length: int):
    """
    Calculates the stride at a vectorized range for a given index expression and loop variable.
    Uses the simplified index expression from `simplify_index_in_vec_range` to determine stride.

    Args:
    - index: The simplified index expression.
    - var: The loop variable.
    - vec_length: Length of the vectorized range.

    Returns:
    - The stride value at the vectorized range for the given index.

    Note:
    This function is cached using `functools.lru_cache` for efficiency in repeated calculations.
    """
    index_vec_simplified = simplify_index_in_vec_range(index, var, vec_length)
    return stride_at(index_vec_simplified, var)


class OuterLoopFusedSchedulerNode(FusedSchedulerNode):
    @classmethod
    def fuse(  # type: ignore[override]
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode, outer_loop_fusion_depth
    ):
        # 断言节点1和节点2的调度器相同
        assert node1.scheduler is node2.scheduler
        # 断言节点1和节点2的类型是 OuterLoopFusedSchedulerNode, SchedulerNode 或 FusedSchedulerNode 中的一种
        assert all(
            type(node)
            in (
                OuterLoopFusedSchedulerNode,
                SchedulerNode,
                FusedSchedulerNode,
            )
            for node in (node1, node2)
        )
        # 如果节点1或节点2是 OuterLoopFusedSchedulerNode 类型的，则返回合并后的新实例
        if any(type(node) is OuterLoopFusedSchedulerNode for node in (node1, node2)):
            return cls(
                node1.scheduler,
                (
                    list(node1.get_outer_nodes())
                    if type(node1) is OuterLoopFusedSchedulerNode
                    else [
                        node1,
                    ]
                )
                + (
                    list(node2.get_outer_nodes())
                    if type(node2) is OuterLoopFusedSchedulerNode
                    else [
                        node2,
                    ]
                ),
                outer_loop_fusion_depth,
            )
        else:
            # 否则，返回将节点1和节点2合并的新实例
            return cls(node1.scheduler, [node1, node2], outer_loop_fusion_depth)  # type: ignore[list-item]

    def __init__(
        self,
        scheduler: "Scheduler",
        outer_fused_nodes: List[Union[FusedSchedulerNode, SchedulerNode]],
        outer_loop_fusion_depth,
    ):
        # 初始化方法，设置调度器和外部合并节点列表
        self.outer_fused_nodes: List[
            Union[FusedSchedulerNode, SchedulerNode]
        ] = outer_fused_nodes
        self.outer_loop_fusion_depth = outer_loop_fusion_depth
        # 将外部合并节点列表展开成节点列表
        flatten_snodes = []
        for _node in self.outer_fused_nodes:
            assert isinstance(_node, (SchedulerNode, FusedSchedulerNode))
            flatten_snodes.extend(list(_node.get_nodes()))
        # 调用父类初始化方法，传入调度器和展开后的节点列表
        super().__init__(scheduler, flatten_snodes)  # type: ignore[arg-type]

    def get_outer_nodes(self):
        # 返回外部合并节点列表
        return self.outer_fused_nodes

    def check_outer_fusion_loop_level_attr(
        self, cpp_kernel_proxy_list, outer_loop_fusion_depth
    ):
        # 检查外部循环融合深度的属性
    ):
        # 确保在外部循环融合深度内的每个循环级别应用相同的平铺分割。
        # 在融合阶段，我们只检查具有相同变量和减少操作的节点。
        # 然而，对于具有相同变量和减少操作的节点，循环仍然可能具有不同的平铺分割。
        # 例如（test_expr_vec_non_contiguous in test_cpu_repro.py）：
        #   * buf0 在第二层循环上进行平铺，buf1 在第三层循环上进行平铺。
        # 如果检查失败，我们应该回退到标准的循环代码生成。
        def _inner(
            left_loop_level: LoopLevel,
            right_loop_level: LoopLevel,
            loop_fusion_depth: int,
        ) -> bool:
            # 检查是否具有相同的循环级别属性
            outer_loops_attr_compare_list = [
                "var",
                "size",
                "offset",
                "steps",
            ]
            if not (
                all(
                    getattr(left_loop_level, attr_compare)
                    == getattr(right_loop_level, attr_compare)
                    for attr_compare in outer_loops_attr_compare_list
                )
            ):
                return False

            assert loop_fusion_depth >= 1
            if (loop_fusion_depth := loop_fusion_depth - 1) > 0:
                # 如果下一个循环级别预期进行外部循环融合，
                # 当前循环级别不应该存在内核。
                assert (
                    left_loop_level.kernel is None and right_loop_level.kernel is None
                )
                # 检查下一个循环级别属性
                if any(
                    # 假设在任何外部循环融合深度上没有主/尾循环分割
                    # 对于这种复杂情况没有明显的性能优势
                    len(loop_level.inner) != 1
                    for loop_level in [left_loop_level, right_loop_level]
                ) or not _inner(
                    left_loop_level.inner[0],
                    right_loop_level.inner[0],
                    loop_fusion_depth,
                ):
                    return False

            return True

        for idx in range(len(cpp_kernel_proxy_list) - 1):
            left_loop_nest = cpp_kernel_proxy_list[idx].loop_nest
            right_loop_nest = cpp_kernel_proxy_list[idx + 1].loop_nest
            if any(
                # 假设在任何外部循环融合深度上没有主/尾循环分割
                len(loop_nest.root) != 1
                for loop_nest in [left_loop_nest, right_loop_nest]
            ) or not _inner(
                left_loop_nest.root[0], right_loop_nest.root[0], outer_loop_fusion_depth
            ):
                return False

        return True

    def merge_outer_fusion_kernels(
        self,
        cpp_kernel_proxy_list,
        ):
            # 从 cpp_kernel_proxy_list 中提取每个 kernel 的 loop_nest，构成列表
            loop_nest_list: List[LoopNestWithSplit] = [
                kernel.loop_nest for kernel in cpp_kernel_proxy_list
            ]
            # 将循环嵌套列表的长度添加到 metrics.cpp_outer_loop_fused_inner_counts 中
            metrics.cpp_outer_loop_fused_inner_counts.append(len(loop_nest_list))

            # 获取第一个 cpp_kernel_proxy_list 的 kernel_group
            kernel_group = cpp_kernel_proxy_list[0].kernel_group

            # 定义一个函数 _merge_outer_fusion_loop_levels，用于外部融合循环级别的合并
            def _merge_outer_fusion_loop_levels(
                loop_level_nested_list: List[List["LoopLevel"]],
                outer_loop_fusion_depth,
            ):
                assert outer_loop_fusion_depth >= 1
                # 假设在任何外部循环融合深度上，都没有主/尾部循环分割
                assert all(
                    len(loop_level_list) == 1 for loop_level_list in loop_level_nested_list
                )
                # 如果 outer_loop_fusion_depth >= 1，则继续合并下一个循环级别
                if (outer_loop_fusion_depth := outer_loop_fusion_depth - 1) >= 1:
                    # 进一步合并下一个循环级别
                    next_loop_level_nested_list = [
                        loop_level_list[0].inner
                        for loop_level_list in loop_level_nested_list
                    ]
                    _merge_outer_fusion_loop_levels(
                        next_loop_level_nested_list,
                        outer_loop_fusion_depth,
                    )
                else:
                    # 创建一个外部融合的 kernel 对象
                    outer_loop_fused_kernel = OuterLoopFusedKernel(kernel_group)
                    # 获取第一个 kernel 的循环级别
                    loop_level_of_first_kernel = loop_level_nested_list[0][0]
                    # 将每个 kernel 的第一个循环级别添加到外部融合的 kernel 中
                    for kernel_idx in range(len(loop_level_nested_list)):
                        outer_loop_fused_kernel.inner.append(
                            deepcopy(loop_level_nested_list[kernel_idx][0]),
                        )
                    # 将第一个 kernel 的循环级别设置为空列表，其 kernel 设置为外部融合的 kernel
                    loop_level_of_first_kernel.inner = []
                    loop_level_of_first_kernel.kernel = outer_loop_fused_kernel

            # 合并来自 loop_nest_list 的 List[LoopNestWithSplit] 到 cpp_kernel_proxy_list[0].loop_nest 中
            _merge_outer_fusion_loop_levels(
                [_loop_nest.root for _loop_nest in loop_nest_list],  # type: ignore[misc]
                self.outer_loop_fusion_depth,
            )
            # 返回 cpp_kernel_proxy_list 的第一个元素
            return cpp_kernel_proxy_list[0]
class RecordOptimizationContext:
    # 优化记录上下文的类，用于在特定函数上下文中记录优化信息
    def __init__(self, func_name: str = ""):
        # 初始化函数，设置函数名和当前节点以及优化上下文对象
        self.func_name = func_name
        self.current_node: Optional[torch.fx.Node] = None
        self.opt_ctx: Optional[OptimizationContext] = None

    def __enter__(self):
        # 进入上下文管理器时的操作
        assert V.interpreter  # 确保有解释器对象可用
        assert V.interpreter.current_node  # 确保当前节点存在

        # 将当前节点设置为解释器的当前节点
        self.current_node = V.interpreter.current_node
        assert self.current_node is not None  # 再次确保当前节点存在
        # 如果当前节点的元数据中存在优化上下文的键，则获取该上下文，否则创建一个新的优化上下文
        if OptimizationContext.key in self.current_node.meta:
            self.opt_ctx = self.current_node.meta[OptimizationContext.key]
        else:
            self.opt_ctx = OptimizationContext()
        assert self.opt_ctx is not None  # 确保优化上下文对象存在
        self.opt_ctx.ops_name = self.func_name  # 设置优化上下文的操作名称为当前函数名
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出上下文管理器时的操作
        assert self.current_node  # 确保当前节点存在
        assert self.opt_ctx  # 确保优化上下文对象存在

        # 将优化上下文对象存储回当前节点的元数据中
        self.current_node.meta[OptimizationContext.key] = self.opt_ctx

    def get_opt_ctx(self):
        # 返回当前的优化上下文对象
        return self.opt_ctx

    def get_fx_node(self):
        # 返回当前的FX节点对象
        assert self.current_node  # 确保当前节点存在
        return self.current_node


def get_opt_ctx(node: torch.fx.Node) -> OptimizationContext:
    # 获取给定FX节点的优化上下文对象
    return node.meta.get(OptimizationContext.key, None)


def get_current_node_opt_ctx() -> OptimizationContext:
    # 获取解释器当前节点的优化上下文对象
    assert V.interpreter.current_node  # 确保解释器当前节点存在
    return get_opt_ctx(V.interpreter.current_node)


class CppCSEVariable(CSEVariable):
    # 表示C++中的常量传播变量类，继承自CSEVariable
    def __init__(self, name, bounds: ValueRanges[Any]):
        # 初始化函数，设置名称、值范围、向量标志和数据类型
        super().__init__(name, bounds)
        self.is_vec = False
        self.dtype: Optional[torch.dtype] = None
        self.dependent_itervars: Set[sympy.Symbol] = set()

    def __repr__(self):
        # 返回变量的字符串表示，包括名称、值范围、向量标志、数据类型和依赖迭代变量集合
        return (
            f"CppCSEVariable(name: {self.name}, bounds: {self.bounds}, is_vec: {self.is_vec}, dtype: {self.dtype}, "
            f"dependent_itervars: {self.dependent_itervars})"
        )
    # 根据名称和参数更新对象状态
    def update_on_args(self, name, args, kwargs):
        if name == "load":
            # 如果操作名称为 "load"，则args[1]是索引，用于设置依赖的迭代变量
            self._set_dependent_itervars(args[1])
        else:
            # 如果操作名称不是 "load"，则传播相关的迭代变量和是否为向量的信息从参数中获取
            self.dependent_itervars.update(
                *[
                    arg.dependent_itervars
                    for arg in args
                    if isinstance(arg, CppCSEVariable)
                ]
            )
            if name == "index_expr":
                # 如果操作名称为 "index_expr"，则用参数 args[0] 设置依赖的迭代变量
                self._set_dependent_itervars(args[0])
            # 如果任何参数是向量且为 CppCSEVariable 类型，则设置当前对象的 is_vec 为 True
            if any(arg.is_vec for arg in args if isinstance(arg, CppCSEVariable)):
                self.is_vec = True

        # 注意事项 [CppCSEVariable 的数据类型]
        # 根据当前优化上下文决定数据类型并不总是准确的，因为数据类型是在代码生成开始时初始化的。
        # 可能在当前操作的代码生成过程中调用了其他操作，它们的数据类型与当前操作可能不同。
        # TODO(jgong5): 更准确地决定变量的数据类型的方法是在 `update_on_args` 内部传播数据类型。

        # 如果存在 V.interpreter.current_node 属性并且当前优化上下文不为 None，则使用当前节点的数据类型
        if (
            hasattr(V.interpreter, "current_node")
            and get_current_node_opt_ctx() is not None
        ):
            self.dtype = get_current_node_opt_ctx().dtype

        # 如果操作名称在 BIN_CMP_OPS 中，则将数据类型设置为 torch 的布尔类型
        if name in BIN_CMP_OPS:
            self.dtype = torch.bool

    def _set_dependent_itervars(self, index: sympy.Expr):
        """
        根据 `index` 表达式设置此变量的相关迭代变量。
        这包括直接在 `index` 中使用的迭代变量以及在 `index` 中使用的其他 CSE 变量的相关迭代变量。
        """
        for s in index.free_symbols:
            if s in V.kernel.itervars:
                self.dependent_itervars.add(s)  # 类型提示：忽略[arg-type]
            elif s.name in V.kernel.cse.varname_map:  # 类型提示：忽略[attr-defined]
                self.dependent_itervars.update(
                    V.kernel.cse.varname_map[s.name].dependent_itervars  # 类型提示：忽略[attr-defined]
                )

    def depends_on(self, itervar: sympy.Symbol):
        """
        检查此变量是否依赖于给定的迭代变量 `itervar`。
        """
        return itervar in self.dependent_itervars
    """Map element-wise ops to C++"""
    # 定义一个类，用于将元素操作映射到 C++ 中

    @staticmethod
    def add(a, b):
        # 返回一个字符串，表示 a + b 的 decltype 类型
        return f"decltype({a})({a} + {b})"

    @staticmethod
    def sub(a, b):
        # 返回一个字符串，表示 a - b 的 decltype 类型
        return f"decltype({a})({a} - {b})"

    @staticmethod
    def mul(a, b):
        # 返回一个字符串，表示 a * b 的 decltype 类型
        return f"decltype({a})({a} * {b})"

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None):
        # 断言目标 dtype 必须在 DTYPE_TO_CPP 字典中
        assert dtype in DTYPE_TO_CPP, f"{dtype} missing from {__name__}.DTYPE_TO_CPP"
        # 返回将 x 转换为目标 dtype 的 C++ 表达式
        return f"c10::convert<{DTYPE_TO_CPP[dtype]}>({x})"

    @staticmethod
    def to_dtype_bitcast(x, dtype, src_dtype):
        # 断言目标 dtype 必须在 DTYPE_TO_CPP 字典中
        assert dtype in DTYPE_TO_CPP, f"{dtype} missing from {__name__}.DTYPE_TO_CPP"
        if src_dtype in (torch.float16, torch.bfloat16):
            # 如果源 dtype 是 torch.float16 或 torch.bfloat16，则执行以下转换逻辑
            # 使用 c10::bit_cast 进行位转换，确保保持数据的高精度
            cast_x = f"c10::convert<{DTYPE_TO_CPP[src_dtype]}>({x})"
            cast_x = f"c10::bit_cast<{DTYPE_TO_CPP[dtype]}>({cast_x})"
            return f"c10::convert<{DTYPE_TO_CPP[torch.float32]}>({cast_x})"
        else:
            # 否则，直接使用 c10::bit_cast 进行位转换
            return f"c10::bit_cast<{DTYPE_TO_CPP[dtype]}>({x})"

    @staticmethod
    def abs(x):
        # 返回 x 的绝对值的 C++ 表达式
        return f"std::abs({x})"

    @staticmethod
    def sin(x):
        # 返回 x 的正弦值的 C++ 表达式
        return f"std::sin({x})"

    @staticmethod
    def cos(x):
        # 返回 x 的余弦值的 C++ 表达式
        return f"std::cos({x})"

    @staticmethod
    def neg(x):
        # 返回 x 的负数的 decltype 类型
        return f"decltype({x})(-{x})"

    @staticmethod
    def exp(x):
        # 返回 x 的指数值的 C++ 表达式
        return f"std::exp({x})"

    @staticmethod
    def exp2(x):
        # 返回 x 的 2 的指数值的 C++ 表达式
        return f"std::exp2({x})"

    @staticmethod
    def expm1(x):
        # 返回 exp(x) - 1 的 C++ 表达式
        return f"std::expm1({x})"

    @staticmethod
    def erf(x):
        # 返回 x 的误差函数值的 C++ 表达式
        return f"std::erf({x})"

    @staticmethod
    def erfc(x):
        # 返回 x 的补误差函数值的 C++ 表达式
        return f"std::erfc({x})"

    @staticmethod
    def erfinv(x):
        # 返回计算 x 的逆误差函数的 C++ 表达式
        return f"calc_erfinv({x})"

    @staticmethod
    def sqrt(x):
        # 返回 x 的平方根的 C++ 表达式
        return f"std::sqrt({x})"

    @staticmethod
    def rsqrt(x):
        # 返回 x 的平方根的倒数的 C++ 表达式
        return f"1 / std::sqrt({x})"

    @staticmethod
    def log1p(x):
        # 根据配置选择不同的 log1p 实现
        bug = config.cpp.inject_log1p_bug_TESTING_ONLY
        if bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"std::log1p({x})"
        else:
            raise AssertionError(
                f"unrecognized config cpp.inject_log1p_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def tan(x):
        # 返回 x 的正切值的 C++ 表达式
        return f"std::tan({x})"

    @staticmethod
    def tanh(x):
        # 返回 x 的双曲正切值的 C++ 表达式
        return f"std::tanh({x})"

    @staticmethod
    def signbit(x):
        # 返回 x 的符号位的 C++ 表达式
        return f"std::signbit({x})"
    def pow(a, b):
        # 返回以 a 和 b 为参数的 std::pow 函数调用的字符串表示
        return f"std::pow({a}, {b})"

    @staticmethod
    def log(x):
        # 返回以 x 为参数的 std::log 函数调用的字符串表示
        return f"std::log({x})"

    @staticmethod
    def round(x):
        # 返回以 x 为参数的 std::nearbyint 函数调用的字符串表示
        return f"std::nearbyint({x})"

    @staticmethod
    def floor(x):
        # 返回以 x 为参数的 std::floor 函数调用的字符串表示
        return f"std::floor({x})"

    @staticmethod
    def floordiv(a, b):
        # 返回整数 a 除以 b 的商和余数的条件表达式字符串表示
        # 如果 a 和 b 异号，则根据余数是否为零决定返回商减一或者商
        quot = f"{a} / {b}"
        rem = f"{a} % {b}"
        return f"(({a} < 0) != ({b} < 0) ? ({rem} != 0 ? {quot} - 1 : {quot}) : {quot})"

    @staticmethod
    def ceil(x):
        # 返回以 x 为参数的 std::ceil 函数调用的字符串表示
        return f"std::ceil({x})"

    @staticmethod
    def trunc(x):
        # 返回以 x 为参数的 std::trunc 函数调用的字符串表示
        return f"std::trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # 返回整数 a 除以 b 的商的字符串表示
        return f"{a} / {b}"

    @staticmethod
    def fmod(a, b):
        # 返回以 a 和 b 为参数的 std::fmod 函数调用的字符串表示
        return f"std::fmod({a}, {b})"

    @staticmethod
    def isinf(x):
        # 返回以 x 为参数的 std::isinf 函数调用的字符串表示
        return f"std::isinf({x})"

    @staticmethod
    def isnan(x):
        # 返回以 x 为参数的 std::isnan 函数调用的字符串表示
        return f"std::isnan({x})"

    @staticmethod
    def lgamma(x):
        # 返回以 x 为参数的 std::lgamma 函数调用的字符串表示
        return f"std::lgamma({x})"

    @staticmethod
    def acos(x):
        # 返回以 x 为参数的 std::acos 函数调用的字符串表示
        return f"std::acos({x})"

    @staticmethod
    def acosh(x):
        # 返回以 x 为参数的 std::acosh 函数调用的字符串表示
        return f"std::acosh({x})"

    @staticmethod
    def cosh(x):
        # 返回以 x 为参数的 std::cosh 函数调用的字符串表示
        return f"std::cosh({x})"

    @staticmethod
    def sinh(x):
        # 返回以 x 为参数的 std::sinh 函数调用的字符串表示
        return f"std::sinh({x})"

    @staticmethod
    def asin(x):
        # 返回以 x 为参数的 std::asin 函数调用的字符串表示
        return f"std::asin({x})"

    @staticmethod
    def asinh(x):
        # 返回以 x 为参数的 std::asinh 函数调用的字符串表示
        return f"std::asinh({x})"

    @staticmethod
    def atan2(x, y):
        # 返回以 x 和 y 为参数的 std::atan2 函数调用的字符串表示
        return f"std::atan2({x}, {y})"

    @staticmethod
    def atan(x):
        # 返回以 x 为参数的 std::atan 函数调用的字符串表示
        return f"std::atan({x})"

    @staticmethod
    def atanh(x):
        # 返回以 x 为参数的 std::atanh 函数调用的字符串表示
        return f"std::atanh({x})"

    @staticmethod
    def copysign(x, y):
        # 返回以 x 和 y 为参数的 std::copysign 函数调用的字符串表示
        return f"std::copysign({x}, {y})"

    @staticmethod
    def frexp(x):
        # 返回以 x 为参数的 std::frexp 函数调用结果的字符串表示
        # 如果结果已经被计算过并存储在缓存中，则直接返回缓存结果
        cache_keys = f"frexp({x})[0]", f"frexp({x})[1]"
        if all(cache_key in V.kernel.cse.cache for cache_key in cache_keys):
            return tuple(V.kernel.cse.cache[cache_key] for cache_key in cache_keys)

        # 构建用于计算结果的代码块
        code = BracesBuffer()
        exponent = V.kernel.cse.newvar()
        mantissa = V.kernel.cse.newvar()
        code.writeline(f"int32_t {exponent};")
        code.writeline(f"auto {mantissa} = std::frexp({x}, &{exponent});")
        V.kernel.compute.splice(code)
        cse_vars = (mantissa, exponent)
        # 将计算结果存入缓存
        for cache_key, cse_var in zip(cache_keys, cse_vars):
            V.kernel.cse.cache[cache_key] = cse_var
        return mantissa, exponent

    @staticmethod
    def hypot(x, y):
        # 返回以 x 和 y 为参数的 std::hypot 函数调用的字符串表示
        return f"std::hypot({x}, {y})"

    @staticmethod
    def log10(x):
        # 返回以 x 为参数的 std::log10 函数调用的字符串表示
        return f"std::log10({x})"

    @staticmethod
    def log2(x):
        # 返回以 x 为参数的 std::log2 函数调用的字符串表示
        return f"std::log2({x})"

    @staticmethod
    def nextafter(x, y):
        # 返回以 x 和 y 为参数的 std::nextafter 函数调用的字符串表示
        return f"std::nextafter({x}, {y})"
    def relu(x):
        # 获取配置中的 relu bug 测试设置
        bug = config.cpp.inject_relu_bug_TESTING_ONLY
        # 根据 bug 设置返回不同的字符串，用于测试编译错误和运行时错误
        if bug == "compile_error":
            return "compile error!"
        elif bug == "runtime_error":
            return f"{x}; throw 1"
        elif bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"std::max({x}, decltype({x})(0))"
        else:
            # 如果配置的 bug 值未被识别，则抛出断言错误
            raise AssertionError(
                f"unrecognized config cpp.inject_relu_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def minimum(a, b):
        # 返回两个值的最小值，使用特定的函数处理 NaN 传播
        return f"min_propagate_nan({a}, {b})"

    @staticmethod
    def maximum(a, b):
        # 返回两个值的最大值，使用特定的函数处理 NaN 传播
        return f"max_propagate_nan({a}, {b})"

    @staticmethod
    def where(a, b, c):
        # 返回条件表达式，根据 a 的值选择返回 b 或 c
        return f"{a} ? {b} : {c}"

    @staticmethod
    def mod(a, b):
        # 返回 a 对 b 取模的结果
        return f"mod({a}, {b})"

    @staticmethod
    def constant(val, dtype):
        # 获取当前节点的优化上下文，并确保数据类型已设置
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        assert opt_ctx and opt_ctx.dtype is not None, opt_ctx
        dtype = opt_ctx.dtype
        # 如果数据类型为低精度浮点类型之一，则将常量升级为 float32
        if dtype in DTYPE_LOWP_FP:
            dtype = torch.float32
        # 将常量值转换为 C++ 表达式
        return value_to_cpp(val, DTYPE_TO_CPP[dtype])

    @staticmethod
    def index_expr(expr, dtype):
        # 获取当前节点的优化上下文，并确保数据类型已设置
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        assert opt_ctx and opt_ctx.dtype is not None
        dtype = opt_ctx.dtype

        # 对表达式进行重命名以适应内核计算
        idx_str = cexpr(V.kernel.rename_indexing(expr))
        # 生成一个新的变量用于存储计算结果，并尝试进行公共子表达式消除
        var = V.kernel.cse.generate(
            V.kernel.compute, idx_str, bounds=get_bounds_index_expr(expr)
        )
        # 将结果转换为指定的数据类型并返回
        return ops.to_dtype(var, dtype)

    @staticmethod
    def masked(mask, body, other):
        code = BracesBuffer()

        # 将掩码操作写入 lambda 函数中
        body_var = V.kernel.cse.newvar()
        code.writeline(f"auto {body_var} = [&]")
        with V.kernel.swap_buffers(code), code.indent():
            # 执行 body 函数，并返回结果
            result = body()
            code.writeline(f"return {result};")
        code.writeline(";")
        V.kernel.compute.splice(code)

        # 使用 lambda 返回的类型作为 other 的类型
        other_code = value_to_cpp(other, f"decltype({body_var}())")
        # 返回条件表达式，根据掩码 mask 选择返回 body_var 或 other_code
        return f"{mask} ? {body_var}() : {other_code}"

    @staticmethod
    def logical_and(a, b):
        # 返回逻辑与操作的结果
        return f"{a} && {b}"

    @staticmethod
    def logical_not(a):
        # 返回逻辑非操作的结果
        return f"!{a}"

    @staticmethod
    def logical_or(a, b):
        # 返回逻辑或操作的结果
        return f"{a} || {b}"

    @staticmethod
    def logical_xor(a, b):
        # 返回逻辑异或操作的结果
        return f"{a} != {b}"

    @staticmethod
    def bitwise_and(a, b):
        # 返回按位与操作的结果，并将结果升级为 a 的数据类型
        return f"decltype({a})({a} & {b})"

    @staticmethod
    def bitwise_not(a):
        # 返回按位取反操作的结果，并将结果升级为 a 的数据类型
        return f"decltype({a})(~{a})"

    @staticmethod
    def bitwise_or(a, b):
        # 返回按位或操作的结果，并将结果升级为 a 的数据类型
        return f"decltype({a})({a} | {b})"

    @staticmethod
    def bitwise_xor(a, b):
        # 返回按位异或操作的结果，并将结果升级为 a 的数据类型
        return f"decltype({a})({a} ^ {b})"
    # 返回一个字符串，表示将变量 a 左移 b 位后的结果的类型
    def bitwise_left_shift(a, b):
        return f"decltype({a})({a} << {b})"

    # 返回一个字符串，表示将变量 a 右移 b 位后的结果的类型
    @staticmethod
    def bitwise_right_shift(a, b):
        return f"decltype({a})({a} >> {b})"

    # 返回一个字符串，表示调用 normalized_rand_cpu 函数生成随机数的表达式
    @staticmethod
    def rand(seed: sympy.Expr, offset: sympy.Expr):
        return f"normalized_rand_cpu({seed}, {offset})"

    # 返回一个字符串，表示调用 randn_cpu 函数生成正态分布随机数的表达式
    @staticmethod
    def randn(seed: sympy.Expr, offset: sympy.Expr):
        return f"randn_cpu({seed}, {offset})"

    # 返回一个字符串，表示调用 randint64_cpu 函数生成指定范围随机整数的表达式
    @staticmethod
    def randint64(seed: sympy.Expr, offset: sympy.Expr, low, high):
        return f"randint64_cpu({seed}, {offset}, {low}, {high})"

    # 返回一个字符串，表示计算输入 x 的 sigmoid 函数的表达式
    @staticmethod
    def sigmoid(x):
        return f"decltype({x})(1) / (decltype({x})(1) + std::exp(-{x}))"

    # 返回一个字符串，表示计算输入 x 的符号函数的表达式
    @staticmethod
    def sign(x):
        # 创建一个 BracesBuffer 对象，用于生成复杂表达式
        code = BracesBuffer()
        scalar_zero = f"decltype({x})(0)"
        scalar_one = f"decltype({x})(1)"
        code.writeline("[&]()")
        with code.indent():
            code.writeline(f"auto left = {x} > 0 ? {scalar_one} : {scalar_zero};")
            code.writeline(f"auto right = {x} < 0 ? {scalar_one} : {scalar_zero};")
            code.writeline("return left - right;")
        code.writeline("()")
        return code
CppOverrides._initialize_pointwise_overrides("cpp")
# 调用CppOverrides类的静态方法_initialize_pointwise_overrides，初始化覆盖点操作

class CppVecOverrides(CppOverrides):
    """Map element-wise ops to aten vectorization C++"""

    @staticmethod
    def add(a, b):
        return f"{a} + {b}"
    # 静态方法：返回两个参数a和b的加法运算表达式字符串

    @staticmethod
    def sub(a, b):
        return f"{a} - {b}"
    # 静态方法：返回两个参数a和b的减法运算表达式字符串

    @staticmethod
    def mul(a, b):
        return f"{a} * {b}"
    # 静态方法：返回两个参数a和b的乘法运算表达式字符串

    @staticmethod
    def truediv(a, b):
        return f"{a} / {b}"
    # 静态方法：返回两个参数a和b的除法运算表达式字符串

    @staticmethod
    def abs(x):
        return f"{x}.abs()"
    # 静态方法：返回参数x的绝对值运算表达式字符串

    @staticmethod
    def sin(x):
        return f"{x}.sin()"
    # 静态方法：返回参数x的正弦运算表达式字符串

    @staticmethod
    def cos(x):
        return f"{x}.cos()"
    # 静态方法：返回参数x的余弦运算表达式字符串

    @staticmethod
    def exp(x):
        return f"{x}.exp()"
    # 静态方法：返回参数x的指数函数运算表达式字符串

    @staticmethod
    def exp2(x):
        return f"{x}.exp2()"
    # 静态方法：返回参数x的2的指数运算表达式字符串

    @staticmethod
    def expm1(x):
        # decompose for a better performance
        vec_one = f"decltype({x})(1)"
        return f"{x}.exp() - {vec_one}"
    # 静态方法：返回参数x的expm1运算表达式字符串，先定义一个常量1的变量vec_one，然后计算exp(x) - vec_one

    @staticmethod
    def erf(x):
        return f"{x}.erf()"
    # 静态方法：返回参数x的误差函数运算表达式字符串

    @staticmethod
    def erfc(x):
        return f"{x}.erfc()"
    # 静态方法：返回参数x的余误差函数运算表达式字符串

    @staticmethod
    def erfinv(x):
        return f"{x}.erfinv()"
    # 静态方法：返回参数x的逆误差函数运算表达式字符串

    @staticmethod
    def sqrt(x):
        return f"{x}.sqrt()"
    # 静态方法：返回参数x的平方根运算表达式字符串

    @staticmethod
    def eq(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} == {y})"
    # 静态方法：返回参数x和y的相等比较运算表达式字符串，包含断言检查类型和调用V.kernel._get_mask_type获取掩码类型

    @staticmethod
    def ne(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        if x.dtype == torch.bool:
            assert y.dtype == torch.bool
            x_cast, y_cast = unify_mask_base_type(V.kernel.compute, (x, y))
            return f"{x_cast} != {y_cast}"
        else:
            assert x.dtype is not None
            return f"{V.kernel._get_mask_type(x.dtype)}({x} != {y})"
    # 静态方法：返回参数x和y的不等比较运算表达式字符串，包含断言检查类型和条件判断以及对掩码类型的统一处理

    @staticmethod
    def lt(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} < {y})"
    # 静态方法：返回参数x和y的小于比较运算表达式字符串，包含断言检查类型和调用V.kernel._get_mask_type获取掩码类型

    @staticmethod
    def gt(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} > {y})"
    # 静态方法：返回参数x和y的大于比较运算表达式字符串，包含断言检查类型和调用V.kernel._get_mask_type获取掩码类型

    @staticmethod
    def le(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} <= {y})"
    # 静态方法：返回参数x和y的小于等于比较运算表达式字符串，包含断言检查类型和调用V.kernel._get_mask_type获取掩码类型

    @staticmethod
    def ge(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} >= {y})"
    # 静态方法：返回参数x和y的大于等于比较运算表达式字符串，包含断言检查类型和调用V.kernel._get_mask_type获取掩码类型

    @staticmethod
    def and_(x, y):
        return f"{x} & {y}"
    # 静态方法：返回参数x和y的按位与运算表达式字符串

    @staticmethod
    def rsqrt(x):
        return f"{x}.rsqrt()"
    # 静态方法：返回参数x的平方根的倒数运算表达式字符串

    @staticmethod
    def pow(a, b):
        return f"{a}.pow({b})"
    # 静态方法：返回参数a的b次方运算表达式字符串
    def log(x):
        # 返回一个字符串，格式为 "{x}.log()"
        return f"{x}.log()"

    @staticmethod
    def round(x):
        # 返回一个字符串，格式为 "{x}.round()"
        return f"{x}.round()"

    @staticmethod
    def floor(x):
        # 返回一个字符串，格式为 "{x}.floor()"
        return f"{x}.floor()"

    @staticmethod
    def ceil(x):
        # 返回一个字符串，格式为 "{x}.ceil()"
        return f"{x}.ceil()"

    @staticmethod
    def trunc(x):
        # 返回一个字符串，格式为 "{x}.trunc()"
        return f"{x}.trunc()"

    @staticmethod
    def fmod(a, b):
        # 返回一个字符串，格式为 "{a}.fmod({b})"
        return f"{a}.fmod({b})"

    @staticmethod
    def lgamma(x):
        # 返回一个字符串，格式为 "{x}.lgamma()"
        return f"{x}.lgamma()"

    @staticmethod
    def logical_and(a, b):
        # 返回一个字符串，格式为 "{a} & {b}"
        return f"{a} & {b}"

    @staticmethod
    def logical_not(a):
        # 返回一个字符串，格式为 "~{a}"
        return f"~{a}"

    @staticmethod
    def logical_or(a, b):
        # 返回一个字符串，格式为 "{a} | {b}"
        return f"{a} | {b}"

    @staticmethod
    def logical_xor(a, b):
        # 返回一个字符串，格式为 "{a} ^ {b}"
        return f"{a} ^ {b}"

    @staticmethod
    def bitwise_and(a, b):
        # 返回一个字符串，格式为 "{a} & {b}"
        return f"{a} & {b}"

    @staticmethod
    def bitwise_not(a):
        # 返回一个字符串，格式为 "~{a}"
        return f"~{a}"

    @staticmethod
    def bitwise_or(a, b):
        # 返回一个字符串，格式为 "{a} | {b}"
        return f"{a} | {b}"

    @staticmethod
    def bitwise_xor(a, b):
        # 返回一个字符串，格式为 "{a} ^ {b}"
        return f"{a} ^ {b}"

    @staticmethod
    def bitwise_left_shift(a, b):
        # 返回一个字符串，格式为 "{a} << {b}"
        return f"{a} << {b}"

    @staticmethod
    def bitwise_right_shift(a, b):
        # 返回一个字符串，格式为 "{a} >> {b}"
        return f"{a} >> {b}"

    @staticmethod
    def tan(a):
        # 返回一个字符串，格式为 "{a}.tan()"
        return f"{a}.tan()"

    @staticmethod
    def tanh(a):
        # 返回一个复杂的表达式字符串，计算 tanh(a)
        vec_one = f"decltype({a})(1)"
        vec_two = f"decltype({a})(2)"
        vec_minus_two = f"decltype({a})(-2)"
        return f"{vec_two} / ({vec_one} + ({vec_minus_two} * {a}).exp()) - {vec_one}"

    @staticmethod
    def reciprocal(a):
        # 返回一个字符串，格式为 "{a}.reciprocal()"
        return f"{a}.reciprocal()"

    @staticmethod
    def atan(x):
        # 返回一个字符串，格式为 "{x}.atan()"
        return f"{x}.atan()"

    @staticmethod
    def acos(x):
        # 返回一个字符串，格式为 "{x}.acos()"
        return f"{x}.acos()"

    @staticmethod
    def asin(x):
        # 返回一个字符串，格式为 "{x}.asin()"
        return f"{x}.asin()"

    @staticmethod
    def cosh(x):
        # 返回一个字符串，格式为 "{x}.cosh()"
        return f"{x}.cosh()"

    @staticmethod
    def sinh(x):
        # 返回一个字符串，格式为 "{x}.sinh()"
        return f"{x}.sinh()"

    @staticmethod
    def log10(x):
        # 返回一个字符串，格式为 "{x}.log10()"
        return f"{x}.log10()"

    @staticmethod
    def log2(x):
        # 返回一个字符串，格式为 "{x}.log2()"
        return f"{x}.log2()"

    @staticmethod
    def nextafter(x, y):
        # 返回一个字符串，格式为 "{x}.nextafter({y})"
        return f"{x}.nextafter({y})"

    @staticmethod
    def copysign(a, b):
        # 返回一个字符串，格式为 "{a}.copysign({b})"
        return f"{a}.copysign({b})"

    @staticmethod
    def atan2(a, b):
        # 返回一个字符串，格式为 "{a}.atan2({b})"
        return f"{a}.atan2({b})"

    @staticmethod
    def hypot(a, b):
        # 返回一个字符串，格式为 "{a}.hypot({b})"
        return f"{a}.hypot({b})"

    @staticmethod
    def atanh(x):
        # 返回一个复杂的表达式字符串，计算 atanh(x)
        vec_one = f"decltype({x})(1)"
        vec_one_half = f"decltype({x})(0.5)"
        return f"{vec_one_half} * (({vec_one} + {x})/({vec_one} - {x})).log()"

    @staticmethod
    def asinh(x):
        # 返回一个复杂的表达式字符串，计算 asinh(x)
        vec_one = f"decltype({x})(1)"
        return f"({x} + ({vec_one} + {x}*{x}).sqrt()).log()"

    @staticmethod
    def acosh(x):
        # 返回一个字符串，格式为 "{x}.acosh()"
        return f"{x}.acosh()"
    def relu(x):
        # 获取配置中的 relu 测试标志
        bug = config.cpp.inject_relu_bug_TESTING_ONLY
        # 根据不同的标志返回不同的结果
        if bug == "compile_error":
            return "compile error!"
        elif bug == "runtime_error":
            return f"{x}; throw 1"
        elif bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"at::vec::clamp_min({x}, decltype({x})(0))"
        else:
            # 如果标志未知，则抛出异常
            raise AssertionError(
                f"unrecognized config cpp.inject_relu_bug_TESTING_ONLY = {bug!r}"
            )

    # TODO: this seems to be dead
    @staticmethod
    def sigmoid(x):
        # 返回 sigmoid 函数的表达式
        return f"decltype({x})(1)/(decltype({x})(1) + {x}.neg().exp())"

    @staticmethod
    def neg(x):
        # 返回取负数的表达式
        return f"{x}.neg()"

    @staticmethod
    def floordiv(a, b):
        # a 和 b 都是整数类型
        _t = f"decltype({a})"
        quot = f"{a} / {b}"
        has_rem = f"({a} % {b} != {_t}(0))"
        is_neg = f"(({a} < {_t}(0)) != ({b} < {_t}(0)))"
        # 返回根据余数和符号计算的整数除法结果
        return f"{_t}::blendv({quot}, {quot} - {_t}(1), {has_rem} & {is_neg})"

    @staticmethod
    def truncdiv(a, b):
        # a 和 b 都是整数类型，返回简单的整数除法结果
        return f"{a} / {b}"

    @staticmethod
    def minimum(a, b):
        # 如果 a 是布尔类型，则统一类型后返回按位与的结果
        if a.dtype == torch.bool:
            assert b.dtype == torch.bool
            a_cast, b_cast = unify_mask_base_type(V.kernel.compute, (a, b))
            return f"{a_cast} & {b_cast}"
        else:
            # 否则返回使用向量化操作的最小值计算结果
            return f"at::vec::minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        # 如果 a 是布尔类型，则统一类型后返回按位或的结果
        if a.dtype == torch.bool:
            assert b.dtype == torch.bool
            a_cast, b_cast = unify_mask_base_type(V.kernel.compute, (a, b))
            return f"{a_cast} | {b_cast}"
        else:
            # 否则返回使用向量化操作的最大值计算结果
            return f"at::vec::maximum({a}, {b})"

    @staticmethod
    def square(a):
        # 返回计算平方的表达式
        return f"{a} * {a}"

    @staticmethod
    def where(a, b, c):
        # 确保 V.kernel 是 CppVecKernel 类的实例
        assert isinstance(V.kernel, CppVecKernel)
        # 如果 b 是布尔类型，则统一类型后返回按条件混合的结果
        if b.dtype == torch.bool:
            assert c.dtype == torch.bool
            blendv_a, blendv_b, blendv_c = unify_mask_base_type(
                V.kernel.compute, (a, b, c)
            )
            return f"decltype({blendv_b})::blendv({blendv_c}, {blendv_b}, {blendv_a})"
        else:
            # 否则返回按条件混合的结果，条件基于 V.kernel._get_mask_cast 的返回
            return f"decltype({b})::blendv({c}, {b}, {V.kernel._get_mask_cast(a, b.dtype)})"

    @staticmethod
    def sign(x):
        # 创建代码对象
        code = BracesBuffer()
        vec_zero = f"decltype({x})(0)"
        vec_one = f"decltype({x})(1)"
        # 左右混合操作，根据 x 的符号来决定返回值
        blendv_l = f"decltype({x})::blendv({vec_zero}, {vec_one}, {vec_zero} < {x})"
        blendv_r = f"decltype({x})::blendv({vec_zero}, {vec_one}, {x} < {vec_zero})"
        code.writeline("[&]()")
        with code.indent():
            code.writeline(f"auto left = {blendv_l};")
            code.writeline(f"auto right = {blendv_r};")
            code.writeline("return left - right;")
        code.writeline("()")
        # 返回代码块执行后的结果
        return code
    # 将输入 x 转换为指定的数据类型 dtype
    def to_dtype(x, dtype, src_dtype=None):
        # 断言目标数据类型在支持的范围内，否则抛出异常
        assert dtype in [
            torch.bool,
            torch.float,
            torch.bfloat16,
            torch.float16,
            torch.uint8,
            torch.int8,
            torch.int32,
            torch.int64,
        ], f"{__name__} does not support {dtype}"
        
        # 获取当前节点，应为 torch.fx.Node 类型，用于静态分析
        node: torch.fx.Node = V.interpreter.current_node
        assert node and isinstance(node, torch.fx.Node)
        
        # 获取输入参数 x 的优化上下文
        opt_ctx_x = get_opt_ctx(node.args[1])
        assert opt_ctx_x
        assert opt_ctx_x.dtype is not None
        
        # 断言 V.kernel 是 CppVecKernel 类型的实例
        assert isinstance(V.kernel, CppVecKernel)
        
        # 获取输入数据的源数据类型
        src_dtype = opt_ctx_x.dtype
        
        # 根据源数据类型获取对应的 C++ 类型
        src_cpp_type = DTYPE_TO_CPP[src_dtype]
        
        # 获取源数据类型对应的向量数量
        src_num_vectors = V.kernel._get_num_vectors(src_dtype)
        
        # 根据目标数据类型获取对应的 C++ 类型
        dst_cpp_type = DTYPE_TO_CPP[dtype]
        
        # 获取目标数据类型对应的向量数量
        dst_num_vectors = V.kernel._get_num_vectors(dtype)
        
        # 如果源数据类型不是 torch.bool 且目标数据类型是 torch.bool
        if src_dtype != torch.bool and dtype == torch.bool:
            return f"{V.kernel._get_mask_type(src_dtype)}::from<{src_cpp_type},{src_num_vectors}>({x})"
        
        # 如果优化上下文中的数据类型是 torch.bool 且目标数据类型不是 torch.bool
        if opt_ctx_x.dtype == torch.bool and dtype != torch.bool:
            return f"{x}.to<{dst_cpp_type},{dst_num_vectors}>()"
        
        # 如果源数据类型不等于目标数据类型
        if src_dtype != dtype:
            # 如果源数据类型和目标数据类型的向量数量都是 1
            if src_num_vectors == dst_num_vectors == 1:
                return f"at::vec::convert<{dst_cpp_type}>({x})"
            else:
                return f"at::vec::convert<{dst_cpp_type},{dst_num_vectors},{src_cpp_type},{src_num_vectors}>({x})"
        
        # 默认情况下返回输入 x 的字符串表示
        return f"({x})"

    @staticmethod
    # 静态方法，根据配置选择如何计算 log(1+x) 或返回异常
    def log1p(x):
        bug = config.cpp.inject_log1p_bug_TESTING_ONLY
        if bug == "accuracy":
            return f"{x} + decltype({x})(1)"  # 返回 x + 类型为 x 的 1
        elif bug is None:
            return f"{x}.log1p()"  # 返回 log(1+x) 的计算结果
        else:
            raise AssertionError(
                f"unrecognized config cpp.inject_log1p_bug_TESTING_ONLY = {bug!r}"
            )
    # 定义一个静态方法 `masked`，接受三个参数 `mask`, `body`, `other`
    def masked(mask, body, other):
        # 断言 `V.kernel` 是 `CppVecKernel` 类的实例
        assert isinstance(V.kernel, CppVecKernel)
        # 创建一个代码缓冲区对象 `code`
        code = BracesBuffer()
        # 生成一个新的变量 `var`，用于存储在 `V.kernel.cse` 中的新变量
        var = V.kernel.cse.newvar()
        # 使用 `V.kernel.masked(mask)` 上下文管理器，将 `new_mask` 作为新的掩码
        with V.kernel.masked(mask) as new_mask:
            # 在代码缓冲区 `code` 中写入 lambda 函数定义
            code.writeline(f"auto {var} = [&]")
            # 在 `V.kernel` 中交换缓冲区，将生成的代码添加到 `code` 中
            with V.kernel.swap_buffers(code), code.indent():
                # 调用 `body` 函数并将结果存储到 `result` 中
                result = body()
                # 在 `code` 中写入返回结果的语句
                code.writeline(f"return {result};")
        # 在 `code` 中写入分号
        code.writeline(";")
        # 将 `code` 插入到 `V.kernel.compute` 中
        V.kernel.compute.splice(code)
    
        # 获取 `result` 的数据类型
        dtype = result.dtype
        # 构建 `body_code` 字符串，表示执行 `body` 的代码
        body_code = f"{var}()"
        # 如果 `result` 是向量，直接使用 `body_code`，否则使用标量转向量的类型
        body_code_vec = (
            body_code
            if result.is_vec
            else f"{V.kernel._get_vec_type(dtype)}({body_code})"
        )
        # 将 `other` 转换为对应的 C++ 代码
        other_code = value_to_cpp(other, DTYPE_TO_CPP[dtype])
        # 如果数据类型是 bool，则将 `other_code` 加载为 `VecMask<float, N>` 类型
        other_code_vec = (
            f"{V.kernel._get_mask_type()}::from({other_code})"
            if dtype == torch.bool
            else f"{V.kernel._get_vec_type(dtype)}({other_code})"
        )
        # 断言 `new_mask` 是 `CppCSEVariable` 类的实例
        assert isinstance(new_mask, CppCSEVariable), new_mask
        # 如果 `new_mask` 是向量
        if new_mask.is_vec:
            # 创建一个新的代码缓冲区 `code`
            code = BracesBuffer()
            # 在 `code` 中写入 lambda 函数定义
            code.writeline("[&]")
            # 在 `V.kernel` 中交换缓冲区，将生成的代码添加到 `code` 中
            with V.kernel.swap_buffers(code), code.indent():
                # 在 `code` 中写入条件判断语句，如果 `new_mask` 全部为零则返回 `other_code_vec`
                code.writeline(f"if ({new_mask}.all_zero())")
                with code.indent():
                    code.writeline(f"return {other_code_vec};")
                code.writeline("else")
                with code.indent():
                    # 为 `body_code_vec` 和 `other_code_vec` 创建新的 CSE 变量
                    body_vec_var = V.kernel.cse.generate(
                        V.kernel.compute,
                        body_code_vec,
                    )
                    other_vec_var = V.kernel.cse.generate(
                        V.kernel.compute,
                        other_code_vec,
                    )
                    # 断言 `body_vec_var` 和 `other_vec_var` 都是 `CppCSEVariable` 类的实例
                    assert isinstance(body_vec_var, CppCSEVariable), body_vec_var
                    assert isinstance(other_vec_var, CppCSEVariable), other_vec_var
                    # 更新 `body_vec_var` 和 `other_vec_var` 的数据类型
                    body_vec_var.dtype = dtype
                    other_vec_var.dtype = dtype
                    # 在 `code` 中写入使用 `V.kernel.overrides.where` 的条件语句
                    code.writeline(
                        f"return {V.kernel.overrides.where(new_mask, body_vec_var, other_vec_var)};"
                    )
            # 在 `code` 中写入调用语句并创建一个 CSE 变量 `csevar`
            code.writeline("()")
            csevar = V.kernel.cse.generate(
                V.kernel.compute,
                code,
            )
        # 如果 `result` 是向量
        elif result.is_vec:
            # 创建一个 CSE 变量 `csevar`
            csevar = V.kernel.cse.generate(
                V.kernel.compute, f"{mask} ? {body_code_vec} : {other_code_vec}"
            )
        else:
            # 创建一个 CSE 变量 `csevar`
            csevar = V.kernel.cse.generate(
                V.kernel.compute, f"{mask} ? {body_code} : {other_code}"
            )
        # 更新 `csevar` 的参数，以便正确传播相关的迭代变量和向量化状态
        csevar.update_on_args("masked", (mask, body, other, result), {})
        # 返回最终的 CSE 变量 `csevar`
        return csevar
    # 定义函数 index_expr，用于处理表达式 expr，并返回结果
    def index_expr(expr, dtype):
        # 获取当前优化上下文
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        # 断言确保 opt_ctx 不为空且其数据类型已定义
        assert opt_ctx and opt_ctx.dtype is not None
        # 将 dtype 设置为 opt_ctx 中的数据类型
        dtype = opt_ctx.dtype
        # 断言确保 V.kernel 是 CppVecKernel 类型的实例
        assert isinstance(V.kernel, CppVecKernel)
        
        # 使用 V.kernel 对象的方法重命名表达式的索引
        index = V.kernel.rename_indexing(expr)
        # 获取 V.kernel 对象中的 tiling_idx 对应的迭代变量
        tiling_var = V.kernel.itervars[V.kernel.tiling_idx]
        # 尝试获取 index 相对于 tiling_var 的常量步长
        stride = V.kernel._try_get_const_stride(index, tiling_var)
        
        # 根据步长的不同情况进行处理
        if stride == 0:
            # 如果步长为 0，则调用 CppOverrides 模块的 index_expr 方法处理
            return CppOverrides.index_expr(expr, dtype)
        elif stride is not None:
            # 如果步长不为 None，则生成一个对应的 CSE 变量
            idx = V.kernel.cse.generate(
                V.kernel.compute, cexpr(index), bounds=get_bounds_index_expr(expr)
            )
            # 将生成的索引值转换为指定的数据类型
            value = ops.to_dtype(idx, dtype)
            # 如果转换后的值是 OpsValue 类的实例，则获取其真实值
            if isinstance(value, OpsValue):
                value = value.value
            # 使用 V.kernel 对象的 arange 方法创建一个新的 CSE 变量
            csevar = V.kernel.arange(value, stride)
        else:
            # 如果步长为 None，则调用 V.kernel 对象的 _load_or_store_non_contiguous 方法处理
            csevar = V.kernel._load_or_store_non_contiguous(  # type: ignore[assignment]
                None, index, dtype, V.kernel.compute
            )
        
        # 在生成的 CSE 变量上更新相关参数信息
        csevar.update_on_args("index_expr", (expr, dtype), {})
        # 返回最终生成的 CSE 变量
        return csevar
# 初始化 CppVecOverrides 类的 pointwise overrides，传入参数 "cppvec"
CppVecOverrides._initialize_pointwise_overrides("cppvec")

# 定义 CppTile2DOverrides 类，继承自 CppVecOverrides 类
class CppTile2DOverrides(CppVecOverrides):
    
    # 定义静态方法 index_expr，用于处理表达式和数据类型，确保 V.kernel 是 CppTile2DKernel 的实例
    @staticmethod
    def index_expr(expr, dtype):
        assert isinstance(V.kernel, CppTile2DKernel)
        # 使用 V.kernel.transform_indexing 方法转换表达式
        expr = V.kernel.transform_indexing(expr)
        # 调用 CppVecOverrides 类的 index_expr 方法处理转换后的表达式，并返回结果
        return CppVecOverrides.index_expr(expr, dtype)

# 定义 CppKernel 类，继承自 Kernel 类
class CppKernel(Kernel):
    
    # 设置 CppOverrides 为 CppKernel 类的 overrides 属性，忽略类型检查
    overrides = CppOverrides  # type: ignore[assignment]
    
    # 设置 sexpr 属性为 cexpr
    sexpr = cexpr
    
    # 设置 newvar_prefix 为 "auto "
    newvar_prefix = "auto "
    
    # 设置 suffix 为 ";"
    suffix = ";"

    # 定义初始化方法，接收参数 args 和 num_threads
    def __init__(self, args, num_threads):
        # 调用父类 Kernel 的初始化方法，传入参数 args
        super().__init__(args)
        
        # 初始化 call_ranges 属性为 None，表示调用范围为空
        self.call_ranges: Optional[Tuple[sympy.Expr, ...]] = None
        
        # 初始化 ranges 属性为空列表，用于存储 sympy.Expr 类型的表达式
        self.ranges: List[sympy.Expr] = []
        
        # 初始化 itervars 属性为空列表，用于存储 sympy.Symbol 类型的符号
        self.itervars: List[sympy.Symbol] = []
        
        # 初始化 reduction_depth 为 None，表示归约深度未定义
        self.reduction_depth = None
        
        # 初始化 reduction_prefix 为 IndentedBuffer()，用于存储归约前缀代码的缓冲区
        self.reduction_prefix = IndentedBuffer()
        
        # 初始化 reduction_suffix 为 IndentedBuffer()，用于存储归约后缀代码的缓冲区
        self.reduction_suffix = IndentedBuffer()
        
        # 初始化 parallel_reduction_prefix 为 IndentedBuffer()，用于存储并行归约前缀代码的缓冲区
        self.parallel_reduction_prefix = IndentedBuffer()
        
        # 初始化 parallel_reduction_suffix 为 IndentedBuffer()，用于存储并行归约后缀代码的缓冲区
        self.parallel_reduction_suffix = IndentedBuffer()
        
        # 初始化 local_reduction_init 为 IndentedBuffer()，用于存储局部归约初始化代码的缓冲区
        self.local_reduction_init = IndentedBuffer()
        
        # 初始化 local_reduction_stores 为 IndentedBuffer()，用于存储局部归约存储代码的缓冲区
        self.local_reduction_stores = IndentedBuffer()
        
        # 初始化 is_reduction 为 False，表示不是归约操作
        self.is_reduction = False
        
        # 初始化 non_parallel_reduction_prefix 为 IndentedBuffer()，用于存储非并行归约前缀代码的缓冲区
        self.non_parallel_reduction_prefix = IndentedBuffer()
        
        # 初始化 reduction_cse 为 CSE 类的实例，用于公共子表达式消除
        self.reduction_cse = CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")
        
        # 初始化 preloads 为 IndentedBuffer()，用于存储预加载代码的缓冲区
        self.preloads = IndentedBuffer()
        
        # 初始化 poststores 为 IndentedBuffer()，用于存储后存储代码的缓冲区
        self.poststores = IndentedBuffer()
        
        # 将 num_threads 参数赋值给 num_threads 属性，表示为特定的线程数进行的核函数
        self.num_threads = num_threads
        
        # 初始化 reduction_omp_dec 为空字典，用于存储 OpenMP 归约的声明
        self.reduction_omp_dec: Dict[Tuple[str, str], str] = {}

    # 定义 _gen_parallel_reduction_buffers 方法，用于生成并行归约的缓冲区
    def _gen_parallel_reduction_buffers(
        self,
        acc,
        acc_type,
        reduction_type,
        dtype,
        reduction_combine_fn=reduction_combine,
        reduction_init_fn=reduction_init,
        welford_weight_reciprocal_vec_fn=None,
        ...
        # 省略部分参数说明，具体实现在省略号后继续
    ):
        # 如果配置允许动态线程数，并且并行减少前缀不存在，则向并行减少前缀写入最大线程数获取的语句
        if config.cpp.dynamic_threads and not self.parallel_reduction_prefix:
            self.parallel_reduction_prefix.writeline(
                "int max_threads = omp_get_max_threads();"
            )
        # 根据累加器名称创建本地累加器变量名
        acc_local = f"{acc}_local"
        # 决定使用的线程数，若配置为动态线程，则使用最大线程数，否则调用parallel_num_threads()获取
        num_threads = (
            "max_threads" if config.cpp.dynamic_threads else parallel_num_threads()
        )
        # 创建每个线程对应的累加器数组元素的名称
        acc_per_thread = f"{acc}_arr[{num_threads}]"
        # 创建用于数组中各线程的本地累加器名称
        acc_local_in_array = acc_per_thread.replace(f"[{num_threads}]", "[tid]")
        # 向本地累加器初始化写入初始值的语句
        self.local_reduction_init.writeline(
            f"{acc_type} {acc_local} = {reduction_init_fn(reduction_type, dtype)};"
        )
        # 向并行减少前缀写入每个线程的累加器声明语句
        self.parallel_reduction_prefix.writeline(f"{acc_type} {acc_per_thread};")
        # 向并行减少前缀写入for循环语句，对每个线程的本地累加器进行初始化
        self.parallel_reduction_prefix.writelines(
            [
                f"for (int tid = 0; tid < {num_threads}; tid++)",
                "{",
                f"    {acc_local_in_array} = {reduction_init_fn(reduction_type, dtype)};",
                "}",
            ],
        )
        # 向本地累加器存储行写入存储每个线程本地累加器的语句
        self.local_reduction_stores.writelines(
            [
                f"{acc_local_in_array} = {acc_local};",
            ]
        )
        # 向并行减少后缀写入for循环语句，对每个线程的本地累加器进行最终累加
        self.parallel_reduction_suffix.writelines(
            [
                f"for (int tid = 0; tid < {num_threads}; tid++)",
                "{",
                f"    {acc} = {reduction_combine_fn(reduction_type, acc, acc_local_in_array)};",
                "}",
            ],
        )
        # 如果是Welford减少类型，存在权重向量反数函数以及有权重反数向量范围属性，并且累加器类型字符串中包含'vec'
        if (
            reduction_type == "welford_reduce"
            and welford_weight_reciprocal_vec_fn
            and hasattr(self, "weight_recp_vec_range")
            and "vec" in f"{acc_type}"
        ):
            # 向本地累加器初始化写入权重向量反数函数的调用语句
            self.local_reduction_init.writeline(
                welford_weight_reciprocal_vec_fn(dtype, num_threads)
            )

    # 获取包含并行减少变量模式的方法
    def get_reduction_var_pattern(self, line: str):
        return re.search("tmp_acc[0-9]+", line)

    # 更新带有并行减少的存储行的方法
    def update_stores_with_parallel_reduction(self):
        # 遍历存储行中的每一行
        for i, line in enumerate(self.stores._lines):
            if isinstance(line, str):
                # 在行中查找并行减少变量模式的匹配
                m = self.get_reduction_var_pattern(line)
                if m:
                    # 提取匹配到的变量名
                    var_name = m.group(0)
                    # 将存储行中的变量名替换为本地累加器变量名
                    self.stores._lines[i] = line.replace(var_name, f"{var_name}_local")

    # 使用额外掩码对加载和存储操作添加的上下文管理器
    @contextlib.contextmanager
    def masked(self, mask):
        """Context manager to add an additional mask to loads and stores."""
        # 保存当前的加载掩码
        prior = self._load_mask
        # 如果先前有加载掩码，则与新掩码进行逻辑与操作
        if prior:
            mask = ops.and_(mask, prior)
            # 如果掩码是OpsValue实例，则获取其值，并确保其类型为CppCSEVariable
            if isinstance(mask, OpsValue):
                mask = mask.value
                assert isinstance(mask, CppCSEVariable)
                # 查看NOTE [dtype of CppCSEVariable]，确保掩码的dtype为torch.bool类型
                mask.dtype = torch.bool

        # 设置加载掩码为新掩码
        self._load_mask = mask
        try:
            yield mask
        finally:
            # 恢复先前的加载掩码
            self._load_mask = prior
    def cache_high_prec_cse_var_before_lowp_store(self, var_to_store):
        """
        https://github.com/pytorch/pytorch/issues/115260
        对于 FusedSchedulerNode[node1, node2]，node2 加载 node1 存储的内容，并且缓存是低精度浮点数据类型。
        当 node1 的输出同时作为内核的输出时，节点的结果会与 node1 输出不是内核输出的情况不同（在此情况下我们不需要插入 `to_dtype` 进行合法化）。
        为了解决这个问题，在存储低精度 node1 输出时，我们也将逆数据类型转换添加到高精度数据类型的 cse 缓存中。

        Example (pseudo code):
            node1_output = ...
            node1_output_lowp = to_dtype(node1_output, dtype=torch.bfloat16)
            store(buf, node1_output_lowp)
            node2_input_lowp = load(buf)
            node2_input = to_dtype(node2_input_lowp, dtype=torch.float)

        Without cse cache trick:
            node1_output = ...
            node1_output_lowp = to_dtype(node1_output, dtype=torch.bfloat16)
            store(buf, node1_output_lowp)
            node2_input_lowp = node_output_lowp # hit store cache
            node2_input = to_dtype(node2_input_lowp, dtype=torch.float)

        With cse cache trick:
            node1_output = ...
            node1_output_lowp = to_dtype(node1_output, dtype=torch.bfloat16)
            # also add `to_dtype(node1_input_lowp, dtype=torch.float)` -> `node1_output` to cse cache
            store(buf, node1_output_lowp)
            node2_input_lowp = node_output_lowp # hit store cache
            node2_input = node1_output # hit cse cache
        """

        if var_to_store.dtype not in DTYPE_LOWP_FP:
            # 如果 var_to_store 的数据类型不在低精度浮点数据类型列表中，则只需要缓存 fp32 的 cse 变量
            return

        def find_high_prec_var(var, cache):
            high_prec_cse_var = None
            high_prec_cse_var_name = None
            for expr, cse_var in cache.items():
                if cse_var == var:
                    if is_to_lowp_dtype(expr):
                        m = re.search(r"tmp\d+", expr)
                        if m is not None:
                            high_prec_cse_var_name = m.group()
            if high_prec_cse_var_name:
                for cse_var in cache.values():
                    if cse_var.name == high_prec_cse_var_name:
                        high_prec_cse_var = cse_var
                        break
                assert high_prec_cse_var is not None
            return high_prec_cse_var

        high_prec_var = find_high_prec_var(var_to_store, self.cse.cache)
        if high_prec_var and high_prec_var.dtype in DTYPE_TO_CPP:
            cache_key = get_lowp_to_high_prec_expr(
                var_to_store, high_prec_var.dtype, self
            )
            self.cse.cache[cache_key] = high_prec_var
    def scale_index_with_offset(
        self, index: sympy.Expr, scale=1, itervar_idx=-1, offset=0
    ):
        var = self.itervars[itervar_idx]  # 获取迭代变量列表中指定索引处的变量
        replacement = {var: var * scale + offset}  # 构建替换字典，将变量 var 替换为 var * scale + offset
        new_index = sympy_subs(index, replacement)  # 使用替换字典替换 index 中的符号表达式
        return new_index  # 返回替换后的新索引表达式

    def index_to_str(self, index: sympy.Expr) -> str:
        """
        Convert an index expr to a string that can be used in cpp code.
        e.g. a sympy expression "s2" may actually appear as "ks1" in the cpp kernel.
        """
        return cexpr(self.rename_indexing(index))  # 调用 rename_indexing 方法重命名索引表达式，并将其转换为字符串返回

    def index_indirect_depends_on(self, index: sympy.Expr, itervar: sympy.Symbol):
        """
        Check if an index has free symbol CppCSEVariable that depends on `itervar`.
        """
        return any(
            self.cse.varname_map[s.name].depends_on(itervar)  # 检查索引中的每个自由符号是否依赖于给定的 itervar
            for s in index.free_symbols  # 遍历索引表达式中的自由符号集合
            if s.name in self.cse.varname_map  # 确保符号在 varname_map 中定义
            and isinstance(self.cse.varname_map[s.name], CppCSEVariable)  # 确保符号对应的值是 CppCSEVariable 类型
        )

    def index_depends_on(self, index: sympy.Expr, itervar: sympy.Symbol):
        return itervar in index.free_symbols or self.index_indirect_depends_on(
            index, itervar
        )  # 检查索引表达式是否直接或间接依赖于给定的 itervar 符号

    def var_ranges(self):
        return dict(zip(self.itervars, self.ranges))  # 返回迭代变量和其对应的范围组成的字典

    def check_bounds(
        self,
        expr: sympy.Expr,
        size: sympy.Expr,
        lower: bool,
        upper: bool,
    ):
        if not (lower or upper):  # 如果 lower 和 upper 都为 False，直接返回
            return

        indirect = free_symbol_is_type(expr, SymT.TMP)  # 检查表达式中是否存在间接符号类型为 SymT.TMP
        if indirect:
            # indexing in compute
            csevar = ops.index_expr(expr, torch.int32).value  # 获取索引表达式在 torch.int32 类型下的值
            buffer = V.kernel.compute  # 设置缓冲区为 V.kernel.compute
        else:
            # indexing in loads
            prior_compute = V.kernel.compute  # 备份先前的 V.kernel.compute
            try:
                V.kernel.compute = self.loads  # 设置 V.kernel.compute 为 self.loads
                csevar = ops.index_expr(expr, torch.int32).value  # 获取索引表达式在 torch.int32 类型下的值
            finally:
                V.kernel.compute = prior_compute  # 恢复先前的 V.kernel.compute
            buffer = self.loads  # 设置缓冲区为 self.loads

        size_str = V.kernel.sexpr(self.rename_indexing(size)) if upper else None  # 如果 upper 为 True，则将 size 重命名为字符串形式
                                                                                        # （通过 V.kernel.sexpr 方法）
        line = self.indirect_assert(csevar, "0" if lower else None, size_str)  # 调用 indirect_assert 方法生成指定行内容
        self.cse.generate(buffer, line, assignment=False)  # 生成 CSE（公共子表达式消除）代码，不进行赋值操作

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)  # 获取指定名称的输入变量
        index = self.rename_indexing(index)  # 重命名索引表达式
        line = f"{var}[{cexpr_index(index)}]"  # 构建加载指定变量的代码行
        if V.graph.get_dtype(name) in [torch.float16]:  # 检查指定名称的数据类型是否为 torch.float16
            line = f"static_cast<float>({line})"  # 将加载的行转换为 float 类型
        csevar = self.cse.generate(self.loads, line)  # 生成 CSE 变量，并在 self.loads 中引用该行
        csevar.update_on_args("load", (name, index), {})  # 更新 CSE 变量的参数
        return csevar  # 返回 CSE 变量
    # 定义一个方法用于存储数据到缓冲区，要求缓冲区的名称中必须包含字符串"buf"
    def store(self, name, index, value, mode=None):
        # 断言确保缓冲区名称中包含"buf"
        assert "buf" in name
        # 调用args对象的output方法，获取name对应的输出变量
        var = self.args.output(name)
        # 在进行低精度存储之前，缓存高精度CSE变量
        self.cache_high_prec_cse_var_before_lowp_store(value)
        # 重命名索引操作
        index = self.rename_indexing(index)
        
        # 根据mode的不同选择生成不同的代码行
        if mode is None:
            # 若mode为None，生成赋值语句
            line = f"{var}[{cexpr_index(index)}] = {value};"
        elif mode == "atomic_add":
            # 若mode为"atomic_add"
            if not config.cpp.dynamic_threads and self.num_threads == 1:
                # 如果不使用动态线程且线程数为1，则生成普通加法语句
                line = f"{var}[{cexpr_index(index)}] += {value};"
            else:
                # 否则获取name对应的数据类型
                dtype = V.graph.get_dtype(name)
                # 将value转换为对应类型的C++代码形式，类似于load中的static_cast<float>(...)
                value = f"static_cast<{DTYPE_TO_CPP[dtype]}>({value})"
                # 生成原子加法语句
                line = f"atomic_add(&{var}[{cexpr_index(index)}], {value});"
        else:
            # 若mode既不为None也不为"atomic_add"，抛出未实现的错误
            raise NotImplementedError(f"store mode={mode}")
        
        # 将生成的代码行写入到存储对象stores中
        self.stores.writeline(DeferredLine(name, line))
    # 确定是否为 argmax 或 argmin 形式的降维操作
    argmax_or_argmin = reduction_type in {"argmax", "argmin"}

    # 构建用于缓存优化的降维操作的键
    reduction_key = src_dtype, reduction_type, value
    
    # 如果已经在缓存中找到了相同的降维操作结果，则直接返回缓存中的结果
    if reduction_key in self.reduction_cse.reduction_cache:
        return self.reduction_cse.reduction_cache[reduction_key]

    # 生成一个用于降维操作的临时变量名
    acc = self.reduction_cse.generate(
        self.loads, f"reduction {reduction_key}", write=False
    )

    # 标记当前正在进行降维操作
    self.is_reduction = True

    # 如果是 argmax 或 argmin 形式的降维操作
    if argmax_or_argmin:
        # 生成相应的前缀和并行前缀
        prefix, parallel_prefix, local_init = argmax_argmin_prefix(
            reduction_type, src_dtype, acc
        )

        # 将本地初始化语句写入对应的缓冲区
        self.local_reduction_init.writelines(local_init)
        self.reduction_prefix.writelines(prefix)
        self.parallel_reduction_prefix.writelines(parallel_prefix)

        # 确定比较操作符（根据是 argmax 还是 argmin）
        compare_op = (
            "greater_or_nan" if reduction_type == "argmax" else "less_or_nan"
        )

        # 断言当前的降维深度不为 None
        assert self.reduction_depth is not None

        # 确定当前迭代变量的索引
        index = self.itervars[self.reduction_depth]
        for i in range(self.reduction_depth + 1, len(self.itervars)):
            index = index * self.ranges[i] + self.itervars[i]

        # 将结果写入存储区，根据比较操作选择性更新结果
        self.stores.writelines(
            [
                f"if(!({compare_op}({acc}.value, {value}, {acc}.index, {cexpr_index(index)}))) {{",
                f"    {acc}.index = {cexpr_index(index)}; {acc}.value = {value};",
                "}",
            ]
        )

        # 生成本地变量名和多线程数目
        acc_local = f"{acc}_local"
        num_threads = parallel_num_threads()
        acc_per_thread = f"{acc}_arr[{num_threads}]"
        acc_local_in_array = acc_per_thread.replace(f"[{num_threads}]", "[tid]")

        # 生成并行降维的后缀代码块
        self.parallel_reduction_suffix.writelines(
            [
                f"for (int tid = 0; tid < {num_threads}; tid++)",
                "{",
                f"    if(!({compare_op}({acc}.value, {acc_local_in_array}.value, {acc}.index, {acc_local_in_array}.index))) {{",
                f"        {acc}.index = {acc_local_in_array}.index; {acc}.value = {acc_local_in_array}.value;",
                "    }",
                "}",
            ],
        )

        # 将本地变量写入存储区
        self.local_reduction_stores.writelines(
            [
                f"{acc_local_in_array} = {acc_local};",
            ]
        )
    else:
        # 根据降维操作类型和数据类型生成对应的累加器类型
        acc_type = reduction_acc_type(reduction_type, dtype)

        # 将累加器的初始化语句写入前缀缓冲区
        self.reduction_prefix.writeline(
            f"{acc_type} {acc} = {reduction_init(reduction_type, dtype)};"
        )

        # 将累加器的合并操作写入存储区
        self.stores.writeline(
            f"{acc} = {reduction_combine(reduction_type, acc, value)};"
        )

        # 生成并行降维操作所需的缓冲区
        self._gen_parallel_reduction_buffers(acc, acc_type, reduction_type, dtype)

    # 根据降维操作的类型和累加器生成最终的结果
    result = reduction_project(reduction_type, acc)

    # 将结果缓存起来
    self.reduction_cse.reduction_cache[reduction_key] = result

    # 返回最终结果
    return result
    # 定义一个方法来存储减少操作的代码段
    def store_reduction(self, name, index, value):
        # 将索引重新命名以确保唯一性
        index = self.rename_indexing(index)
        # 使用参数名获取输出变量
        var = self.args.output(name)
        # 向减少后缀对象写入延迟行，以便后续生成的代码会将值赋给指定的变量索引
        self.reduction_suffix.writeline(
            DeferredLine(name, f"{var}[{cexpr_index(index)}] = {value};")
        )

    # 设置调用范围的方法，用于确定代码生成的范围
    def set_ranges(self, lengths, reduction_lengths):
        # 如果已经设置了调用范围，则进行断言验证
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(
                reduction_lengths
            ), f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            # 确保减少深度与长度列表的长度相同
            assert self.reduction_depth == len(lengths)
        else:
            # 如果还未设置调用范围，则进行初始化
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            # 将调用范围中的每个索引重新命名，并存储在ranges属性中
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            # 为每个索引创建迭代变量，使用具有前缀的符号来表示
            self.itervars = [
                sympy_index_symbol_with_prefix(SymT.XBLOCK, n)
                for n in range(len(self.ranges))
            ]
            # 设置减少操作的深度为长度列表的长度
            self.reduction_depth = len(lengths)
        # 返回已设置的调用范围和减少范围
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

    # 返回用于估计代码生成大小的提示
    def size_hint(self):
        return V.graph.sizevars.size_hint(
            sympy_product(self.call_ranges), fallback=8192
        )

    # 根据给定的代码和工作分享参数生成循环嵌套结构
    def codegen_loops(self, code, worksharing):
        loop_nest = LoopNestWithSplit.build(self)
        # 调用内部方法实现循环代码生成
        self.codegen_loops_impl(loop_nest, code, worksharing)

    # 属性装饰器，用于返回断言函数名称的字符串表示
    @property
    def assert_function(self) -> str:
        # 如果处于AOT模式，则返回特定的断言函数名称
        if V.graph.aot_mode:
            # 提示可能会导致某些模型性能下降
            # 与使用TORCH_CHECK的JIT Inductor相比
            return "AOTI_TORCH_CHECK"
        else:
            # 否则返回标准的TORCH_CHECK断言函数名称
            return "TORCH_CHECK"

    # 根据最大并行深度和线程数确定并行深度
    def decide_parallel_depth(self, max_parallel_depth, threads):
        assert self.call_ranges is not None
        # 获取用于并行计算的调用范围
        ranges = self.call_ranges[:max_parallel_depth]
        # 计算调用范围的总大小的估计值
        seq = self.size_hint()
        par = 1
        depth = 0
        # 遍历每个范围表达式以确定最佳的并行深度
        for expr in ranges:
            hint = V.graph.sizevars.size_hint(expr, fallback=8192)
            if par >= 2 * threads or par == threads:
                break
            if seq // threads < config.cpp.min_chunk_size:
                # 如果工作量不足以分配给所有线程，则退出循环
                break
            depth += 1
            par *= hint
            seq /= hint
        # 如果线程数是动态分配的，且深度为0且长度大于0，则至少设置一个并行作用域
        if config.cpp.dynamic_threads and depth == 0 and len(ranges) > 0:
            depth = 1
        # 返回确定的并行深度
        return depth

    # 上下文管理器装饰器，用于定义上下文管理器对象
    @contextlib.contextmanager
    # 定义一个方法用于在后缀中写入内容
    def write_to_suffix(self):
        # 保存当前的加载、计算、存储和公共子表达式的状态
        prior = (self.loads, self.compute, self.stores, self.cse)
        # 创建新的缓冲区对象来存储加载部分
        self.loads = IndentedBuffer()
        # 创建新的缓冲区对象来存储计算部分
        self.compute = IndentedBuffer()
        # 创建新的缓冲区对象来存储存储部分
        self.stores = IndentedBuffer()
        # 克隆当前公共子表达式对象
        self.cse = self.cse.clone()
        # 生成器函数的一部分，用于将加载、计算和存储内容合并到后缀中
        yield
        # 将加载缓冲区内容插入到减少后缀中
        self.reduction_suffix.splice(self.loads)
        # 将计算缓冲区内容插入到减少后缀中
        self.reduction_suffix.splice(self.compute)
        # 将存储缓冲区内容插入到减少后缀中
        self.reduction_suffix.splice(self.stores)
        # 恢复先前保存的加载、计算、存储和公共子表达式状态
        (self.loads, self.compute, self.stores, self.cse) = prior

    # 定义一个方法用于创建公共子表达式变量
    def create_cse_var(self, *args, **kwargs):
        # 返回一个新的CppCSEVariable对象，参数通过args和kwargs传递
        return CppCSEVariable(*args, **kwargs)
    # 定义一个继承自CppKernel的子类CppVecKernel，用于特定的向量化操作
    class CppVecKernel(CppKernel):
        # 设置属性overrides为CppVecOverrides，用于类型提示（ignore[assignment]用于忽略类型检查）
        overrides = CppVecOverrides  # type: ignore[assignment]

        # 初始化方法，接受多个参数并设置实例属性
        def __init__(
            self,
            args,
            num_threads,
            tiling_factor=0,
            tiling_idx=-1,
            tiling_dtype=torch.float,
        ):
            # 调用父类CppKernel的初始化方法，传入args和num_threads参数
            super().__init__(args, num_threads)
            # 使用cpu_vec_isa对象的pick_vec_isa方法获取当前CPU的向量指令集
            self.vec_isa = cpu_vec_isa.pick_vec_isa()
            # 断言确保vec_isa对象已成功获取
            assert self.vec_isa
            # 如果tiling_factor为0，则根据vec_isa的向量元素数量和指定的数据类型确定tiling_factor的值
            if tiling_factor == 0:
                tiling_factor = self.vec_isa.nelements(dtype=tiling_dtype)
            # 将计算得到的tiling_factor赋值给实例属性tiling_factor
            self.tiling_factor = tiling_factor
            # 将传入的tiling_idx参数赋值给实例属性tiling_idx
            self.tiling_idx = tiling_idx

        # 私有方法，尝试获取常量步长，接受index和itervar两个参数
        def _try_get_const_stride(self, index: sympy.Expr, itervar: sympy.Symbol):
            # 如果index间接依赖于itervar，则返回None
            if self.index_indirect_depends_on(index, itervar):
                return None
            # 遍历index中的自由符号，选取类型为SymT.TMP的间接变量
            for indirect_var in (
                self.cse.varname_map[s.name]  # type: ignore[attr-defined]
                for s in index.free_symbols
                if symbol_is_type(s, SymT.TMP)
            ):
                # 断言间接变量是CppCSEVariable类型，并且不是向量化的
                assert isinstance(indirect_var, CppCSEVariable)
                if indirect_var.is_vec:
                    return None
            # 调用stride_at_vec_range方法计算index和itervar的步长，考虑tiling_factor
            stride = stride_at_vec_range(index, itervar, self.tiling_factor)
            # 如果计算得到的stride是数值类型，则返回步长；否则返回None
            return stride if stride.is_number else None

        # 私有方法，获取向量数量，接受参数dtype：torch.dtype，返回一个整数
        def _get_num_vectors(self, dtype: torch.dtype) -> int:
            # 计算向量数量，使用math库的ceil函数，考虑数据类型的字节大小、tiling_factor和向量指令集的位宽
            num_vectors = math.ceil(
                self.tiling_factor * dtype.itemsize * 8 / self.vec_isa.bit_width()
            )
            # 断言确保向量数量至少为1
            assert num_vectors >= 1
            # 返回计算得到的向量数量
            return num_vectors

        # 私有方法，获取向量类型，接受参数dtype：torch.dtype，返回一个字符串
        def _get_vec_type(self, dtype: torch.dtype) -> str:
            # 调用_get_num_vectors方法获取向量数量
            num_vectors = self._get_num_vectors(dtype)
            # 如果向量数量为1，则返回标量的向量化表示
            if num_vectors == 1:
                return f"at::vec::Vectorized<{DTYPE_TO_CPP[dtype]}>"
            else:
                # 否则返回具有指定向量数量的向量化表示
                return f"at::vec::VectorizedN<{DTYPE_TO_CPP[dtype]},{num_vectors}>"

        # 私有方法，获取掩码类型，接受参数dtype：torch.dtype，默认为torch.float，返回一个字符串
        def _get_mask_type(self, dtype: torch.dtype = torch.float) -> str:
            # 如果dtype是torch.bool类型，则返回空字符串
            if dtype == torch.bool:
                return ""
            # 否则调用_get_num_vectors方法获取向量数量，并返回掩码类型的字符串表示
            num_vectors = self._get_num_vectors(dtype)
            return f"at::vec::VecMask<{DTYPE_TO_CPP[dtype]},{num_vectors}>"

        # 私有方法，获取掩码类型转换，接受参数mask：CppCSEVariable类型，dtype：torch.dtype类型，返回一个字符串
        def _get_mask_cast(self, mask: CppCSEVariable, dtype: torch.dtype) -> str:
            # 断言确保mask的数据类型是torch.bool
            assert mask.dtype == torch.bool, repr(mask)
            # 调用_get_num_vectors方法获取向量数量，返回掩码类型的类型转换字符串表示
            num_vectors = self._get_num_vectors(dtype)
            return f"{mask}.template cast<{DTYPE_TO_CPP[dtype]},{num_vectors}>()"

        # 方法，根据输入的行字符串获取规约变量的模式，返回匹配对象
        def get_reduction_var_pattern(self, line: str):
            return re.search("tmp_acc[0-9]+_vec", line)

        # 私有方法，获取向量加载行，接受多个参数：var，index，dtype，load_mask（可选的CppCSEVariable类型），返回一个字符串
        def _get_vec_load_line(
            self,
            var: str,
            index: sympy.Expr,
            dtype: torch.dtype,
            load_mask: Optional[CppCSEVariable] = None,
        ):
            # 省略具体实现细节，根据输入参数生成特定的向量加载行代码
            pass
    ):
        """
        Get a load line str that loads a vector from `var` at `index` of type `dtype`.
        If `load_mask` is not None, we do a masked load accordingly.
        Notes on the `dtype`:
        1. We always load `self.tiling_factor` number of elements regardless of the `dtype`.
           It means we load half of the vector lanes for 16-bit data types and quarter of the
           vector lanes for 8-bit data types.
        2. `torch.bool` and `torch.uint8` could mean masks and we load them as float mask vectors.
        """
        # 获取当前优化上下文
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        assert opt_ctx is not None
        # 根据数据类型获取对应的 C++ 类型
        cpp_type = DTYPE_TO_CPP[dtype]
        # 获取要加载的向量数量
        num_vectors = self._get_num_vectors(dtype)
        load_mask_str = None
        if load_mask:
            if not load_mask.is_vec:
                # 如果加载掩码不是向量，则按照 torch.float 类型加载
                load_mask_str = f"{self._get_mask_type(torch.float)}::from({load_mask})"
            else:
                # 否则，根据加载掩码和 torch.float 进行类型转换
                load_mask_str = f"{self._get_mask_cast(load_mask, torch.float)}"
        # 构造加载缓冲区的字符串表示
        loadbuf = f"{var} + {cexpr_index(index)}" if index != 0 else var
        if dtype == torch.bool:
            # 对于布尔类型，生成加载行字符串
            # TODO: 是否考虑在这里使用加载掩码？
            line = f"{self._get_mask_type()}::from({loadbuf})"
        else:
            # 对于其他数据类型，根据是否有加载掩码来生成不同的加载行字符串
            line = (
                f"{load_mask_str}.template loadu<{cpp_type},{num_vectors}>({loadbuf})"
                if load_mask_str
                else f"{self._get_vec_type(dtype)}::loadu({loadbuf}, {self.tiling_factor})"
            )
        return line

    def _load_or_store_non_contiguous(
        self,
        var: Optional[str],
        index: sympy.Expr,
        dtype: torch.dtype,
        buffer: Optional[IndentedBuffer] = None,
        store_value: Optional[Union[str, CppCSEVariable]] = None,
    ):
        """
        Load or store data non-contiguously based on the provided parameters.
        """
        # 获取当前优化上下文
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        # 如果 var 为 None，则表示存储值
        if var is None:
            return store_value
        # 否则，根据 var、index 和 dtype 进行加载
        line = self.load(var, index)
        if buffer is not None:
            buffer.write(line)
        return line

    def load(self, name: str, index: sympy.Expr):
        """
        Load data from `name` at `index` and return the loaded value or expression.
        """
        # 获取当前优化上下文
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        # 根据名称获取输入变量 var
        var = self.args.input(name)
        # 重命名索引表达式
        index = self.rename_indexing(index)
        # 获取数据类型 dtype
        dtype = V.graph.get_dtype(name)
        # 获取当前迭代变量的 tiling
        tiling_var = self.itervars[self.tiling_idx]
        # 尝试获取索引 index 和 tiling_var 之间的常量步长 stride
        stride = self._try_get_const_stride(index, tiling_var)
        if stride == 0:
            # 如果步长为 0，则加载标量并在需要时延迟广播
            return super().load(name, index)
        elif stride == 1:
            # 如果步长为 1，则连续加载数据
            line = self._get_vec_load_line(var, index, dtype, self._load_mask)
            # 生成 CSE 变量，并将其添加到加载字典中
            csevar = self.cse.generate(self.loads, line)  # type: ignore[assignment]
        else:
            # 否则，根据 var、index 和 dtype 进行非连续加载或存储
            csevar = self._load_or_store_non_contiguous(var, index, dtype)  # type: ignore[assignment]
        # 确保生成的变量是 CppCSEVariable 类型
        assert isinstance(csevar, CppCSEVariable)
        # 更新 CSE 变量的参数和标志
        csevar.update_on_args("load", (name, index), {})
        csevar.is_vec = True
        return csevar

    def _get_store_line(
        self,
        value: Union[str, CppCSEVariable],
        var: str,
        index: sympy.Expr,
        dtype: torch.dtype,
    ):
        """
        Get a store line buffer that stores `value` into `var` at `index` of `dtype`. It handles
        both contiguous and non-contiguous store cases.
        :param value: Vectorized type templaterized on `dtype`.
        :param var: buffer to store into.
        :index: index into the `var`.
        """
        # 当 value 的类型为 str（例如，Welford reduction）时，调用者应确保其是一个向量
        assert isinstance(value, str) or (
            isinstance(value, CppCSEVariable) and value.is_vec
        ), value
        # 获取当前循环变量的名称
        tiling_var = self.itervars[self.tiling_idx]
        # 构建用于存储的表达式，包括变量名和索引
        var_expr = f"{var} + {cexpr_index(index)}"
        # 尝试获取索引处的常量步长
        stride = self._try_get_const_stride(index, tiling_var)
        # 创建一个缓冲区对象来存储生成的代码
        code = IndentedBuffer()
        # 根据步长是否为1进行不同的存储操作
        if stride == 1:
            if dtype == torch.float:
                code.writeline(f"{value}.store({var_expr});")
            else:
                code.writeline(f"{value}.store({var_expr}, {self.tiling_factor});")
        else:
            # 处理非连续存储的情况
            self._load_or_store_non_contiguous(
                var, index, dtype, buffer=code, store_value=value
            )
        # 返回生成的代码缓冲区对象
        return code

    def store(self, name, index, value, mode=None):
        # 确保名称中包含字符串 "buf"
        assert "buf" in name
        # 确保 mode 参数为 None
        assert mode is None
        # 确保 value 是 CppCSEVariable 类型的对象
        assert isinstance(value, CppCSEVariable), value
        # 如果 value 不是向量，则进行广播操作
        if not value.is_vec:
            # 当将标量存储到像 "fill" 这样的向量缓冲区时会发生这种情况
            value = self.broadcast(value)
        # 获取当前优化上下文
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        # 获取输出变量的名称
        var = self.args.output(name)
        # 在低精度存储之前缓存高精度 CSE 变量
        self.cache_high_prec_cse_var_before_lowp_store(value)
        # 重命名索引
        index = self.rename_indexing(index)
        # 获取存储行的代码缓冲区对象
        code = self._get_store_line(value, var, index, V.graph.get_dtype(name))
        # 将生成的代码插入到存储行列表中
        self.stores.splice(code.map(lambda x: DeferredLine(name, x)))

    def store_reduction(self, name, index, value):
        # 重命名索引
        index = self.rename_indexing(index)
        # 获取输出变量的名称
        var = self.args.output(name)
        # 获取输出数据类型
        out_dtype = V.graph.get_dtype(name)
        # 确定数据类型是浮点型还是整型
        dtype = torch.float if out_dtype.is_floating_point else torch.int64
        # 创建代码缓冲区对象
        code = IndentedBuffer()
        if self.tiling_idx >= self.reduction_depth:
            # 水平约简
            code.writeline(
                f"{var}[{cexpr_index(index)}] = static_cast<{DTYPE_TO_CPP[out_dtype]}>({value});"
            )
        else:
            # 垂直约简
            if out_dtype != dtype:
                # 转换值的数据类型以匹配输出数据类型
                converted_value = f"{DTYPE_TO_CPP[out_dtype]}_{value}"
                code.writeline(
                    f"auto {converted_value} = at::vec::convert<{DTYPE_TO_CPP[out_dtype]}>({value});"
                )
                value = converted_value
            # 获取存储行的代码缓冲区对象
            code.splice(self._get_store_line(value, var, index, out_dtype))
        # 将生成的代码插入到约简后缀列表中
        self.reduction_suffix.splice(code.map(lambda x: DeferredLine(name, x)))
    # 将标量变量广播为向量变量
    def broadcast(self, scalar_var: CppCSEVariable`
    # 广播函数，将标量变量转换为向量变量
    def broadcast(self, scalar_var: CppCSEVariable) -> CppCSEVariable:
        # 确保输入不是向量类型
        assert not scalar_var.is_vec
        # 根据数据类型选择合适的向量变量生成方法
        if scalar_var.dtype == torch.bool:
            # 如果数据类型为 bool，使用特定方法生成向量变量
            vec_var = self.cse.generate(
                self.compute, f"{self._get_mask_type()}::from({scalar_var.name})"
            )
        else:
            # 确保数据类型不为空
            assert scalar_var.dtype is not None
            # 根据数据类型生成对应的向量变量
            vec_var = self.cse.generate(
                self.compute,
                f"{self._get_vec_type(scalar_var.dtype)}({scalar_var.name})",
            )
        # 确保生成的变量是 CppCSEVariable 类型
        assert isinstance(vec_var, CppCSEVariable)
        vec_var.dtype = scalar_var.dtype  # 设置数据类型
        vec_var.dependent_itervars = scalar_var.dependent_itervars  # 设置依赖变量
        vec_var.is_vec = True  # 标记为向量类型
        return vec_var  # 返回生成的向量变量

    # 生成一个等差序列的向量变量
    def arange(self, index: CppCSEVariable, stride: sympy.Symbol) -> CppCSEVariable:
        # 确保索引变量不是向量类型
        assert not index.is_vec
        assert index.dtype is not None  # 确保数据类型不为空
        # 生成等差序列向量变量
        csevar = self.cse.generate(
            self.compute,
            f"{self._get_vec_type(index.dtype)}::arange({index}, {stride})",
        )
        # 确保生成的变量是 CppCSEVariable 类型
        assert isinstance(csevar, CppCSEVariable)
        csevar.dtype = index.dtype  # 设置数据类型
        csevar.is_vec = True  # 标记为向量类型
        return csevar  # 返回生成的向量变量

    # 初始化向量的归约操作
    def reduction_init_vec(self, reduction_type, dtype):
        scalar_type = DTYPE_TO_COMPUTATION_DTYPE[dtype]  # 获取标量数据类型
        vec_type = self._get_vec_type(scalar_type)  # 获取向量数据类型

        # 检查是否为 Welford 归约
        if is_welford_reduction(reduction_type):
            return f"Welford<{vec_type}>()"  # 返回 Welford 类型的初始化表达式

        # 获取标量类型的归约初始化表达式
        scalar_init = reduction_init(reduction_type, dtype)
        return f"{vec_type}({scalar_init})"  # 返回向量类型的初始化表达式

    # 获取归约操作的向量类型
    def reduction_acc_type_vec(self, reduction_type, dtype):
        # 确保不是 argmin 或 argmax 归约操作
        assert reduction_type not in {"argmin", "argmax"}
        scalar_type = DTYPE_TO_COMPUTATION_DTYPE[dtype]  # 获取标量数据类型
        vec_type = self._get_vec_type(scalar_type)  # 获取向量数据类型
        # 检查是否为 Welford 归约
        if is_welford_reduction(reduction_type):
            return f"Welford<{vec_type}>"  # 返回 Welford 类型的表达式

        return vec_type  # 返回向量数据类型

    # 生成权重倒数的向量变量表达式
    def welford_weight_reciprocal_vec(self, dtype, num_threads=None):
        # 根据线程数计算向量范围
        vec_num_range_thread = (
            CeilDiv(self.weight_recp_vec_range, num_threads)
            if num_threads
            else self.weight_recp_vec_range
        )
        vec_num_range_thread_expr = cexpr_index(vec_num_range_thread)  # 转换为表达式
        return f"static WeightRecp<{self._get_vec_type(dtype)}> weight_recps({vec_num_range_thread_expr});"  # 返回权重倒数向量表达式

    # 组合归约操作的向量变量
    def reduction_combine_vec(
        self, reduction_type, var, next_value, use_weight_recps=False
    ):
        # 归约操作的向量变量组合逻辑（需要具体实现）
    ):
        # 如果是最大值约简操作，返回对应的 ATen 函数调用字符串
        if reduction_type == "max":
            return f"at::vec::maximum({var}, {next_value})"
        # 如果是最小值约简操作，返回对应的 ATen 函数调用字符串
        elif reduction_type == "min":
            return f"at::vec::minimum({var}, {next_value})"
        # 如果是求和约简操作，返回对应的加法表达式字符串
        elif reduction_type == "sum":
            return f"{var} + {next_value}"
        # 如果是乘积约简操作，返回对应的乘法表达式字符串
        elif reduction_type == "prod":
            return f"{var} * {next_value}"
        # 如果是异或和约简操作，返回对应的异或表达式字符串
        elif reduction_type == "xor_sum":
            return f"{var} ^ {next_value}"
        # 如果是 Welford 归约操作，根据是否使用权重修饰符返回相应的 Welford 结果合并函数调用字符串
        elif reduction_type == "welford_reduce":
            if use_weight_recps:
                return f"welford_combine({var}, {next_value}, &weight_recps)"
            else:
                return f"welford_combine({var}, {next_value})"
        # 如果是 Welford 合并操作，根据 next_value 的类型确定参数格式，返回对应的 Welford 合并函数调用字符串
        elif reduction_type == "welford_combine":
            if isinstance(next_value, tuple):
                # 当从 Inductor IR 读取值时，next_value 是变量名组成的元组
                mean, m2, weight = next_value
            else:
                # 当合并中间累加器时，next_value 是 Welford<T> 结构体
                mean, m2, weight = reduction_project(reduction_type, next_value)
            return f"welford_combine({var}, {{{mean}, {m2}, {weight}}})"
        else:
            # 如果未知的约简操作类型，则抛出未实现的错误
            raise NotImplementedError

    def indirect_assert(self, var, lower, upper, mask=None):
        # 确保不支持间接索引断言中的 mask 参数
        assert not mask, "do not support mask in indirect_indexing assertion"
        # 确保 var 是 CppCSEVariable 的实例，并且具有指定的数据类型
        assert isinstance(var, CppCSEVariable)
        assert var.dtype is not None
        # 如果 var 不是向量类型，则调用父类的 indirect_assert 方法
        if not var.is_vec:
            return super().indirect_assert(var, lower, upper, mask)
        # 将 lower 和 upper 转换为相应的向量类型，如果存在的话
        lower_scalar = lower
        upper_scalar = upper
        if lower:
            lower = f"{self._get_vec_type(var.dtype)}({lower})"
        if upper:
            upper = f"{self._get_vec_type(var.dtype)}({upper})"
        # 根据 lower 和 upper 构建条件字符串和对应的打印信息
        if lower and upper:
            cond = f"({lower} <= {var}) & ({var} < {upper})"
            cond_print = f"{lower_scalar} <= {var} < {upper_scalar}"
        elif lower:
            cond = f"{lower} <= {var}"
            cond_print = f"{lower_scalar} <= {var}"
        else:
            assert upper
            cond = f"{var} < {upper}"
            cond_print = f"{var} < {upper_scalar}"
        # 构建完整的条件判断和打印信息字符串
        cond = f"({self._get_mask_type(var.dtype)}({cond})).all_masked()"
        return f'{self.assert_function}({cond}, "index out of bounds: {cond_print}")'
    """
    CppTile2DKernel类继承自CppVecKernel类，用于处理二维瓦片化（tiling）的向量化内核。
    在内层循环级别使用tiling_factor定义的瓦片大小，并在一个外层循环级别（outer_tiling_idx）上操作。
    当从外部循环轴以连续方式访问数据瓦片时，会对瓦片进行转置，以使从内层循环轴开始的访问连续。
    然后，利用其父类CppVecKernel中相同的向量化逻辑进行加载/存储/计算。转置后的瓦片加载和存储被生成到kernel.preloads和kernel.poststores缓冲区中。

    循环结构如下：
    for ...
      for i_outer ...
        for ...
          for inner_most ...
            // 由CppTile2DKernel生成
            float tmp0[16*16]; at::vec::transpose_mxn<...>(tmp0, in_ptr0 + ..., ...); // 存入kernel.preloads
            float tmp1[16*16]; // 存入kernel.preloads
            for i_inner ... { // 内核内部循环
              向量化加载/计算/存储（例如，加载tmp0，存储tmp1） // 存入kernel.loads/compute/stores
            }
            at::vec::transpose_mxn(out_ptr0 + ..., tmp1, ...) // 存入kernel.poststores
          for inner_most ... (尾部)
            // 由CppVecKernel生成
            ...
      for i_outer ... (尾部)
        for ...
          for ...
            // 由CppKernel生成
            ...
    """

    overrides = CppTile2DOverrides  # type: ignore[assignment]

    def __init__(self, args, num_threads, tiling_factor, tiling_indices, tiling_dtype):
        """
        初始化方法，调用父类的初始化方法并设置瓦片索引。
        
        参数：
        - args: 参数列表
        - num_threads: 线程数
        - tiling_factor: 瓦片因子，用于定义瓦片大小
        - tiling_indices: 瓦片索引，指定瓦片化的轴
        - tiling_dtype: 瓦片的数据类型
        """
        super().__init__(
            args, num_threads, tiling_factor, tiling_indices[1], tiling_dtype
        )
        self.tiling_indices = tiling_indices

    def inner_itervar(self):
        """
        返回内层迭代变量的符号表示。

        返回：
        - 内层迭代变量的符号表示，例如"outer_inner"
        """
        return sympy_index_symbol(f"{self.itervars[self.outer_idx]}_inner")

    def need_vec_transpose(self, index):
        """
        判断是否需要进行向量转置。

        参数：
        - index: 迭代索引

        返回：
        - True：需要进行向量转置
        - False：不需要进行向量转置

        注意：
        此方法检查是否支持带有掩码的转置，并且外部和内部的步长符合转置的条件。
        """
        outer_var = self.itervars[self.outer_idx]
        inner_var = self.itervars[self.tiling_idx]
        outer_stride = stride_at_vec_range(index, outer_var, self.tiling_factor)
        inner_stride = stride_at_vec_range(index, inner_var, self.tiling_factor)
        return (
            self._load_mask is None  # TODO: 支持带有掩码的转置
            and outer_stride == 1
            and index.has(inner_var)
            and not inner_stride.has(inner_var)
            and not inner_stride.has(outer_var)
        )
    def gen_transposed_tile_load_store(self, name, var, index, is_store):
        # 定义一个函数用于生成转置瓦片的加载/存储操作，这些操作在内核内部循环之外执行
        dtype = V.graph.get_dtype(name)
        factor = self.tiling_factor
        src = f"{var} + {cexpr_index(index)}"
        dst = "__place_holder__"  # 初始化目标变量，稍后根据需要进行更新
        ld_src = f"{cexpr_index(stride_at_vec_range(index, self.itervars[self.tiling_idx], self.tiling_factor))}"
        ld_dst = f"{factor}"
        if is_store:
            # 如果是存储操作，交换源和目标
            src, dst = dst, src
            ld_src, ld_dst = ld_dst, ld_src

        need_define = True  # 是否需要定义新的变量
        load_or_store = f"at::vec::transpose_mxn<{DTYPE_TO_CPP[dtype]},{factor},{factor}>({src}, {ld_src}, {dst}, {ld_dst});"
        if is_store:
            # 如果是存储操作，生成一个新的变量名
            tile_var = self.cse.newvar()
        elif load_or_store not in self.cse.cache:
            # 如果加载/存储操作不在缓存中，则生成一个新的变量名
            tile_var = self.cse.generate(self.preloads, load_or_store, write=False)
        else:
            # 否则，从缓存中获取变量名
            need_define = False
            tile_var = self.cse.cache[load_or_store]

        if need_define:
            # 如果需要定义新变量，则生成定义新变量的代码行
            define_line = f"{DTYPE_TO_CPP[dtype]} {tile_var}[{factor}*{factor}] __attribute__ ((aligned ({factor})));"
            self.preloads.writeline(define_line)

        load_or_store = load_or_store.replace("__place_holder__", str(tile_var))
        if is_store:
            # 如果是存储操作，将加载/存储操作加入到后存储操作列表中
            self.poststores.writeline(DeferredLine(name, load_or_store))
        else:
            # 否则，将加载/存储操作加入到预加载操作列表中
            self.preloads.writeline(load_or_store)

        return tile_var

    def load(self, name: str, index: sympy.Expr):
        # 载入指定名称和索引的数据，根据当前优化上下文获取相关的变量和信息
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        var = self.args.input(name)  # 获取名称对应的输入参数
        index = self.rename_indexing(index)  # 重命名索引变量名称

        inner = self.inner_itervar()  # 获取内部迭代变量
        if self.need_vec_transpose(index):
            # 如果需要进行向量转置操作
            tile_var = self.gen_transposed_tile_load_store(
                name, var, index, is_store=False
            )
            # 在内核内部循环中执行向量加载操作
            loadbuf = f"{tile_var} + {cexpr_index(inner * self.tiling_factor)}"
            dtype = V.graph.get_dtype(name)
            line = self._get_vec_load_line(loadbuf, 0, dtype)  # 获取向量加载的代码行
            csevar = self.cse.generate(self.loads, line)
            csevar.update_on_args("load", (name, index), {})  # 更新代码行的参数
            assert isinstance(csevar, CppCSEVariable)
            csevar.is_vec = True  # 标记为向量操作
            return csevar
        else:
            new_index = self.transform_indexing(index)  # 转换索引变量
            return super().load(name, new_index)
    # 在 store 方法中，存储变量到缓冲区中，并生成对应的存储指令
    def store(self, name, index, value, mode=None):
        # 确保变量名包含 "buf"
        assert "buf" in name
        # 获取当前优化上下文
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        # 根据名称获取变量
        var = self.args.output(name)

        # 获取内部迭代变量
        inner = self.inner_itervar()
        # 重命名索引变量
        index = self.rename_indexing(index)
        # 确保 mode 为 None
        assert mode is None

        # 如果需要进行向量转置
        if self.need_vec_transpose(index):
            # 生成转置后的瓦片加载存储变量
            tile_var = self.gen_transposed_tile_load_store(
                name, var, index, is_store=True
            )
            # 在内核内部循环中进行向量存储
            storebuf = f"{tile_var} + {cexpr_index(inner * self.tiling_factor)}"
            # 根据数据类型执行向量存储操作
            if V.graph.get_dtype(name) in DTYPE_LOWP_FP:
                line = f"{value}.store({storebuf}, {self.tiling_factor});"
            elif V.graph.get_dtype(name) in (torch.uint8, torch.int8):
                line = f"{value}.store({storebuf}, {self.tiling_factor});"
            else:
                line = f"{value}.store({storebuf});"
            # 将存储指令写入存储文件中
            self.stores.writeline(DeferredLine(name, line))
        else:
            # 转换索引并调用父类的存储方法
            new_index = self.transform_indexing(index)
            super().store(name, new_index, value, mode)

    # 生成内部循环的代码
    def codegen_inner_loops(self, code):
        # 获取内部迭代变量
        inner = self.inner_itervar()
        # 写入内部循环的代码
        code.writeline(
            f"for (long {inner} = 0; {inner} < {self.tiling_factor}; {inner}++)"
        )

    # 设置变量范围，包括进行垂直归约作为尾部循环
    def set_ranges(self, group, reduction_group):
        # 调用父类方法设置变量范围
        vars = super().set_ranges(group, reduction_group)
        # 决定外部索引和瓦片索引的顺序
        self.outer_idx, self.tiling_idx = (
            self.tiling_indices
            if self.tiling_indices[1] < self.reduction_depth
            else reversed(self.tiling_indices)
        )
        return vars

    # 转换索引表达式
    def transform_indexing(self, index: sympy.Expr) -> sympy.Expr:
        # 使用偏移和迭代变量索引来缩放索引
        return self.scale_index_with_offset(
            index,
            itervar_idx=self.outer_idx,
            offset=self.inner_itervar(),
        )
# 定义一个名为 CppVecKernelChecker 的类，继承自 CppVecKernel 类
class CppVecKernelChecker(CppVecKernel):
    # 初始化方法，接受参数 args、num_threads、tiling_factor 和 tiling_idx，默认为 -1
    def __init__(self, args, num_threads, tiling_factor, tiling_idx=-1):
        # 调用父类 CppVecKernel 的初始化方法
        super().__init__(args, num_threads, tiling_factor, tiling_idx)

        # 减少生成的内核计数，因为此内核仅用于检查而不生成任何代码
        metrics.generated_kernel_count -= 1

        # 用于记录图包装器代码的原始状态，因为 wrapper_code 的状态可能在图运行期间发生变化
        self._orig_wrapper_code = None

        # 设置 simd_vec 属性为 True，表示启用 SIMD 向量化
        self.simd_vec = True

        # 初始化 fast_vec_list 列表，用于存储 CppVecOverrides 类中的静态方法名
        self.fast_vec_list = []
        for k, v in CppVecOverrides.__dict__.items():
            if isinstance(v, staticmethod):
                self.fast_vec_list.append(k)

        # 初始化 exit_stack，用于管理资源的上下文管理器堆栈
        self.exit_stack = contextlib.ExitStack()

        # 缓存所有支持的数据类型到 supported_dtypes 列表中
        self.supported_dtypes: List[torch.dtype] = [
            torch.float,
            torch.bfloat16,
            torch.float16,
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int32,
            torch.int64,
        ]

    # 禁用向量化方法，接受可选参数 msg 用于记录禁用向量化的原因
    def disable_vec(self, msg=None):
        # 如果日志级别为 DEBUG，则记录禁用向量化的消息
        if schedule_log.isEnabledFor(logging.DEBUG):
            schedule_log.debug("Disabled vectorization: %s", msg)
        # 将 simd_vec 属性设置为 False，表示禁用 SIMD 向量化
        self.simd_vec = False

    # 加载方法，接受参数 name（名称）和 index（索引），返回加载的变量 var
    def load(self, name: str, index: sympy.Expr):
        # 使用 RecordOptimizationContext 上下文管理器记录优化上下文
        with RecordOptimizationContext(__name__) as node_ctx:
            # 获取变量的数据类型 load_dtype
            load_dtype = V.graph.get_dtype(name)
            # 获取优化上下文 opt_ctx，并断言确保 opt_ctx 存在
            opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
            assert opt_ctx

            # 设置优化上下文的数据类型为 load_dtype
            opt_ctx.dtype = load_dtype
            # 使用 self.cse.newvar() 创建新的变量 var
            var = self.cse.newvar()

            # 如果没有迭代变量，则禁用向量化并返回 var
            if len(self.itervars) == 0:
                self.disable_vec("not a loop")
                return var

            # 如果加载的数据类型不在支持的数据类型列表中，并且满足特定条件，则禁用向量化并返回 var
            if load_dtype not in self.supported_dtypes and (
                index.has(self.itervars[self.tiling_idx])
                or free_symbol_is_type(index, SymT.TMP)
            ):
                self.disable_vec(f"{load_dtype} not supported by load")
                return var

            # 返回加载的变量 var
            return var

    # 存储方法，接受参数 name（名称）、index（索引）、value（值）、mode（模式），返回 simd_vec 属性
    def store(self, name, index, value, mode=None):
        # 使用 RecordOptimizationContext 上下文管理器记录优化上下文
        with RecordOptimizationContext(__name__) as node_ctx:
            # 如果没有迭代变量，则禁用向量化并返回 simd_vec 属性
            if len(self.itervars) == 0:
                self.disable_vec("not a loop")
                return self.simd_vec

            # 获取存储的数据类型 store_dtype
            store_dtype = V.graph.get_dtype(name)

            # 获取优化上下文 opt_ctx，并断言确保 opt_ctx 存在
            opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
            assert opt_ctx
            # 设置优化上下文的数据类型为 store_dtype
            opt_ctx.dtype = store_dtype

            # 如果存储的数据类型不在支持的数据类型列表中，则禁用向量化并返回 simd_vec 属性
            if store_dtype not in self.supported_dtypes:
                self.disable_vec(f"{store_dtype} not supported by store")
                return self.simd_vec

            # 断言名称中包含 "buf"
            assert "buf" in name
            # 重命名索引
            index = self.rename_indexing(index)

            # 如果存在 mode 参数，则禁用向量化并返回 simd_vec 属性
            if mode:
                self.disable_vec(f"store mode: {mode}")
                return self.simd_vec

            # 返回 simd_vec 属性
            return self.simd_vec
    # 定义一个实例方法 reduction，用于执行降维操作
    def reduction(self, dtype, src_dtype, reduction_type, value):
        # 检查条件，如果不满足特定条件，则禁用向量化操作，并记录日志
        if not (
            (dtype == torch.float and src_dtype == torch.float)
            or (dtype == torch.int64 and src_dtype == torch.int64)
            and reduction_type in VECTORIZABLE_RTYPES
        ):
            self.disable_vec(
                f"reduction: dtype {dtype}, src_dtype {src_dtype}, reduction_type {reduction_type}"
            )
        # 如果是 Welford 形式的降维操作，返回一个由 simd_vec 组成的元组
        if is_welford_reduction(reduction_type):
            return tuple([self.simd_vec] * 3)
        # 否则返回 simd_vec
        return self.simd_vec

    # 定义一个实例方法 check_bounds，用于检查边界
    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ):
        # 返回 simd_vec，用于边界检查
        return self.simd_vec

    # 定义一个实例方法 store_reduction，用于存储降维结果
    def store_reduction(self, name, index, value):
        # 返回 simd_vec，用于存储降维结果
        return self.simd_vec

    # 定义 __exit__ 方法，用于退出上下文管理器
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始的 wrapper_code 到图形对象的属性中
        V.graph.wrapper_code = self._orig_wrapper_code  # type: ignore[assignment]
        # 调用 exit_stack 对象的 __exit__ 方法，以确保上下文管理器正确退出
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)
# 定义一个名为 CppKernelProxy 的类，继承自 CppKernel 类
class CppKernelProxy(CppKernel):
    # 初始化方法，接收 kernel_group 参数
    def __init__(self, kernel_group):
        # 调用父类 CppKernel 的初始化方法，传入 kernel_group.args 和 kernel_group.ws.num_threads 参数
        super().__init__(kernel_group.args, kernel_group.ws.num_threads)
        # 将 kernel_group 参数赋值给实例变量 self.kernel_group
        self.kernel_group = kernel_group
        # 初始化实例变量 self.loop_nest，设为 None
        self.loop_nest = None
        # 初始化实例变量 self.call_ranges，设为 None
        self.call_ranges = None
        # 使用 cpu_vec_isa 模块的 pick_vec_isa 方法，获取一个 cpu_vec_isa.VecISA 类型的实例，并赋值给实例变量 self.picked_vec_isa
        self.picked_vec_isa: cpu_vec_isa.VecISA = cpu_vec_isa.pick_vec_isa()

    # 方法：数据类型传播，接收 nodes 参数
    def data_type_propagation(self, nodes):
        # 遍历 nodes 列表中的每个元素 _node
        for _node in nodes:
            # 断言 _node 是 SchedulerNode 类的实例
            assert isinstance(_node, SchedulerNode)
            # 调用 DataTypePropagation 类的 propagate_scheduler_node 方法，传入 _node 参数，用于数据类型传播
            DataTypePropagation.propagate_scheduler_node(_node)

    # 方法：检查给定调度器节点 scheduler_node 的所有 fx 图节点是否支持 BF16/FP16
    def is_lowp_fp_scheduler(self, scheduler_node: SchedulerNode):
        # 如果 scheduler_node 的 _body 属性不是 ir.LoopBody 类型，则返回 True
        if not isinstance(scheduler_node._body, ir.LoopBody):
            return True

        # 初始化 _lowp_fp_type 变量为 None
        _lowp_fp_type: Optional[torch.dtype] = None

        # 传播数据类型，以检查所有 fx 节点是否为 bf16/fp16
        DataTypePropagation.propagate_scheduler_node(scheduler_node)

        # 获取调度器节点的子块列表，包括根块和所有子块
        sub_blocks = [scheduler_node._body.root_block] + list(
            scheduler_node._body.subblocks.values()
        )
        # 遍历子块列表中的每个子块 sub_block
        for sub_block in sub_blocks:
            # 遍历子块 sub_block 的图节点列表中的每个节点 _node
            for _node in sub_block.graph.nodes:
                # 如果节点的操作是 "placeholder" 或者节点的目标是 "get_index" 或 "index_expr"，则跳过此节点的处理
                if _node.op == "placeholder" or _node.target in (
                    "get_index",
                    "index_expr",
                ):
                    continue

                # 如果节点的目标不在支持 bf16/fp16 的操作列表中，则返回 False
                if _node.target not in [
                    "load",
                    "store",
                    "abs",
                    "neg",
                    "output",
                ]:
                    return False

                # 如果节点有 meta 属性，并且 meta 属性存在
                if hasattr(_node, "meta") and _node.meta:
                    # 断言优化上下文 OptimizationContext.key 在节点的 meta 属性中
                    assert OptimizationContext.key in _node.meta
                    # 获取节点的优化上下文对象，赋值给 opt_ctx 变量
                    opt_ctx: OptimizationContext = _node.meta[OptimizationContext.key]
                    # 如果节点没有指定数据类型或数据类型不在 DTYPE_LOWP_FP 列表中，则返回 False
                    if not opt_ctx.dtype or opt_ctx.dtype not in DTYPE_LOWP_FP:
                        return False
                    # 如果 _lowp_fp_type 已经有值，则断言其与当前节点的数据类型一致
                    if _lowp_fp_type:
                        assert (
                            _lowp_fp_type == opt_ctx.dtype
                        ), "scheduler node do not support bf16/fp16 mix"
                    else:
                        # 否则，将当前节点的数据类型赋值给 _lowp_fp_type
                        _lowp_fp_type = opt_ctx.dtype
                else:
                    return False

        # 将计算出的 _lowp_fp_type 赋值给 scheduler_node 的 _lowp_fp_type 属性
        scheduler_node._lowp_fp_type = _lowp_fp_type  # type: ignore[attr-defined]
        # 返回 True，表示所有节点均支持 bf16/fp16
        return True
    # 对于给定的节点列表进行低精度浮点数据类型的合法化处理
    def legalize_lowp_fp_dtype(self, nodes):
        # 如果所有节点都是 SchedulerNode 类型且符合低精度浮点调度器的条件
        if all(
            isinstance(_node, SchedulerNode) and self.is_lowp_fp_scheduler(_node)
            for _node in nodes
        ):
            # 标记加载节点以加载 bf16/fp16
            for _node in nodes:
                # 获取当前节点及其所有子块的列表
                sub_blocks = [_node._body.root_block] + list(
                    _node._body.subblocks.values()
                )
                # 遍历每个子块的图中的节点
                for sub_block in sub_blocks:
                    for fx_node in sub_block.graph.nodes:
                        # 如果节点的目标是 "load" 或 "store"
                        if fx_node.target in ["load", "store"]:
                            assert fx_node.meta
                            assert OptimizationContext.key in fx_node.meta
                            # 获取优化上下文并确保其数据类型是低精度浮点类型之一
                            opt_ctx: OptimizationContext = fx_node.meta[
                                OptimizationContext.key
                            ]
                            assert opt_ctx.dtype in DTYPE_LOWP_FP

            # 由于内核可以直接运行 bf16/fp16，因此绕过合法化处理
            return

        # 对于每个节点，确保其是 SchedulerNode 类型并且其主体是循环体
        for _node in nodes:
            assert isinstance(_node, SchedulerNode)
            assert isinstance(_node._body, ir.LoopBody)
            node: SchedulerNode = _node

            # 判断节点是否是内存复制调度节点
            def is_memory_copy_scheduler_node(node: SchedulerNode):
                op_counts = node.read_writes.op_counts
                return (
                    len(op_counts) == 2 and "load" in op_counts and "store" in op_counts
                )

            # 如果不是内存复制调度节点，则需要进行合法化处理
            should_legalize = not is_memory_copy_scheduler_node(node)
            if should_legalize:
                # 获取节点的循环主体，并对其进行低精度浮点数据类型的合法化处理
                body: ir.LoopBody = node._body
                self.legalize_lowp_fp_dtype_loopbody(body)

    # 对循环主体列表中的每个循环主体进行低精度浮点数据类型的合法化处理
    def codegen_loop_bodies(self, loop_bodies, var_sizes_list):
        for body in loop_bodies:
            # 对循环主体进行低精度浮点数据类型的合法化处理
            self.legalize_lowp_fp_dtype_loopbody(body)
            # 传播循环主体中的数据类型
            DataTypePropagation.propagate_loopbody(body)
        # 生成循环主体的函数代码
        self.codegen_functions(loop_bodies, var_sizes_list)
    # 为给定节点列表生成代码节点
    def codegen_nodes(self, nodes: List[SchedulerNode]):
        # 将BF16节点合法化，显式添加到_dtype中
        self.legalize_lowp_fp_dtype(nodes)
        # 数据类型传播
        self.data_type_propagation(nodes)

        # 断言节点列表长度至少为1
        assert len(nodes) >= 1
        # 取第一个节点作为参考节点
        first_node = nodes[0]
        # 确定向量数据类型
        vec_dtype = (
            first_node._lowp_fp_type  # type: ignore[attr-defined]
            if all(
                hasattr(_node, "_lowp_fp_type")
                and _node._lowp_fp_type == first_node._lowp_fp_type  # type: ignore[attr-defined]
                for _node in nodes
            )
            else torch.float
        )

        # 定义一个函数，用于处理单个节点和索引变量
        def fn(node, *index_vars):
            # 决定是否进行原地更新
            node.decide_inplace_update()
            # 标记节点已运行
            node.mark_run()
            # 如果V.kernel是NullKernelHandler的实例，则调用节点的_body方法
            if isinstance(V.kernel, NullKernelHandler):
                return node._body(*index_vars)
            else:
                return node.codegen(index_vars)

        # 生成函数列表，每个函数是一个节点的部分函数
        fn_list = [functools.partial(fn, node) for node in nodes]
        # 获取每个节点的变量大小列表
        var_sizes_list = [node.group[1] for node in nodes]
        # 调用代码生成函数来生成代码
        self.codegen_functions(fn_list, var_sizes_list, vec_dtype)

    # 生成循环代码
    def codegen_loops(self, code, worksharing):
        # 调用内部的循环生成实现函数来生成循环代码
        self.codegen_loops_impl(self.loop_nest, code, worksharing)
class OuterLoopFusedKernel(CppKernel):
    # 继承自CppKernel的OuterLoopFusedKernel类，用于表示外层循环融合的内核
    def __init__(self, kernel_group):
        # 调用父类CppKernel的初始化方法，传入内核组的参数和线程数
        super().__init__(kernel_group.args, kernel_group.ws.num_threads)
        # 初始化内部循环列表为空
        self.inner: List[LoopLevel] = []

    def decide_parallel_depth(self, max_parallel_depth, threads) -> int:
        # 存储各内部内核的并行深度
        kernels_parallel_depth = []
        # 获取所有内部循环的内核列表
        nested_kernels: List[List[CppKernel]] = [
            loop.get_kernels() for loop in self.inner
        ]
        for kernels in nested_kernels:
            # 对于任何ScalarKernel、VecKernel或Tile2DKernel，它们应该具有相同的调用范围
            call_ranges = kernels[0].call_ranges
            assert call_ranges is not None
            # 断言所有内核的调用范围都与第一个内核相同
            assert all(kernel.call_ranges == call_ranges for kernel in kernels)
            # 决定第一个内核的并行深度，并将其存入列表
            kernels_parallel_depth.append(
                kernels[0].decide_parallel_depth(len(call_ranges), threads)
            )
        # 返回最小的最大并行深度和所有内核中最大的并行深度之间的较小值
        return min(
            max_parallel_depth,
            max(kernels_parallel_depth),
        )


class ReasonFusedNodes(Enum):
    # 枚举ReasonFusedNodes，用于表示融合节点的不同原因
    SAME_VARS_REDUCE = "same_vars_reduce"
    COMPATIBLE_REDUCTION = "compatible_reduction"
    COMPATIBLE_RANGES_NO_REDUCTION = "compatible_ranges_no_reduction"


class CppScheduling(BaseScheduling):
    # ctypes限制参数数量不超过1024，参考：
    # https://github.com/python/cpython/commit/a285af7e626d1b81cf09f8b2bf7656f100bc1237
    # 在这里设置一个保守的阈值
    MAX_FUSED_KERNEL_ARGS_NUM = 500
    # 后端特性字典，初始化为指定的键，值为None
    backend_features = dict.fromkeys(
        [
            BackendFeature.INPLACE_BUFFERS,
            BackendFeature.REDUCE_TO_SINGLE_ELEMENT,
        ]
    )

    @classmethod
    def get_backend_features(cls, device: torch.device):
        # 返回后端特性字典
        return cls.backend_features

    def __init__(self, scheduler):
        # 调用父类BaseScheduling的初始化方法
        super().__init__()
        # 设置调度器属性
        self.scheduler = scheduler
        # 如果存在调度器，则重置内核组
        if scheduler:
            self.reset_kernel_group()
        # 设置刷新状态为未准备状态
        self._ready_to_flush = False

    def _set_flush_status(self, status: bool):
        # 设置刷新状态
        self._ready_to_flush = status

    def group_fn(self, sizes):
        # 对大小进行分组处理，并返回处理后的元组
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def reset_kernel_group(self):
        # 导入CppWrapperCpu模块的CppWrapperCpu类
        from .cpp_wrapper_cpu import CppWrapperCpu

        # 初始化内核组属性，根据图的包装代码类型选择不同的内核组类型
        self.kernel_group: Union[CppWrapperKernelGroup, KernelGroup]
        if isinstance(V.graph.wrapper_code, CppWrapperCpu):
            self.kernel_group = CppWrapperKernelGroup()
        else:
            self.kernel_group = KernelGroup()
    # 定义一个方法用于融合两个节点，其中self是当前对象实例，node1和node2是待融合的调度节点
    def fuse(self, node1, node2):
        # 如果node1或node2是foreach节点，则调用静态方法ForeachKernelSchedulerNode.fuse来执行融合操作
        if node1.is_foreach() or node2.is_foreach():
            return ForeachKernelSchedulerNode.fuse(node1, node2)
        # 如果node1是模板节点而node2不是，则断言node2不是模板节点，然后调用FusedSchedulerNode.fuse来执行融合操作
        elif node1.is_template():
            assert not node2.is_template()
            return FusedSchedulerNode.fuse(node1, node2)
        else:
            # 如果调用_why_fuse_nodes方法返回COMPATIBLE_RANGES_NO_REDUCTION，则执行以下操作
            if (
                self._why_fuse_nodes(node1, node2)
                == ReasonFusedNodes.COMPATIBLE_RANGES_NO_REDUCTION
            ):
                # 断言node1和node2是SchedulerNode或FusedSchedulerNode的实例
                assert isinstance(node1, (SchedulerNode, FusedSchedulerNode))
                assert isinstance(node2, (SchedulerNode, FusedSchedulerNode))

                # 解构node1和node2的group属性，获取vars1和reduce1以及vars2和reduce2
                _, (vars1, reduce1) = node1.group
                _, (vars2, reduce2) = node2.group
                # 断言reduce1和reduce2都为空元组
                assert reduce1 == () and reduce2 == (), (reduce1, reduce2)

                # 定义一个函数用于获取节点的索引范围表达式和索引表达式集合
                def get_indexing_ranges_exprs(node):
                    # 如果node是FusedSchedulerNode的实例
                    if isinstance(node, FusedSchedulerNode):
                        assert len(node.snodes) > 0, node.snodes
                        var_ranges = None
                        indexing_exprs = set()
                        # 遍历node的snodes属性
                        for snode in node.snodes:
                            v, exprs = get_indexing_ranges_exprs(snode)
                            if var_ranges is None:
                                var_ranges = v
                            assert var_ranges == v, (var_ranges, v, node.snodes)
                            indexing_exprs.update(exprs)
                        return var_ranges, list(indexing_exprs)
                    else:
                        # 断言node是SchedulerNode的实例
                        assert isinstance(node, SchedulerNode)
                        comp_buffer = node.node
                        assert isinstance(comp_buffer, ir.ComputedBuffer)
                        # 获取comp_buffer的默认大小体和主体表达式
                        _, body, _ = comp_buffer.get_default_sizes_body()
                        return body.var_ranges, list(body.indexing_exprs.values())

                # 根据vars1和vars2的长度决定node_to_recomp是node1还是node2
                node_to_recomp = node1 if len(vars1) < len(vars2) else node2
                assert isinstance(node_to_recomp, SchedulerNode)

                # 根据vars1和vars2的长度决定ref_node是node2还是node1
                ref_node = node2 if len(vars1) < len(vars2) else node1

                # 获取额外的索引约束条件
                extra_indexing_constraints = get_indexing_ranges_exprs(ref_node)

                # 调用node_to_recomp的recompute_size_and_body方法重新计算大小和主体
                node_to_recomp.recompute_size_and_body(
                    extra_indexing_constraints=extra_indexing_constraints
                )

                # 再次解构node1和node2的group属性，获取vars1和vars2
                _, (vars1, _) = node1.group
                _, (vars2, _) = node2.group
                # 断言vars1和vars2相等
                assert vars1 == vars2, (vars1, vars2)

                # 调用FusedSchedulerNode.fuse来执行最终的融合操作
                return FusedSchedulerNode.fuse(node1, node2)
            # 如果可以垂直外部循环融合，则调用OuterLoopFusedSchedulerNode.fuse来执行融合操作
            elif self.can_fuse_vertical_outer_loop(node1, node2):
                return OuterLoopFusedSchedulerNode.fuse(
                    node1, node2, self._get_outer_loop_fusion_depth(node1, node2)
                )
            else:
                # 否则，调用FusedSchedulerNode.fuse来执行融合操作
                return FusedSchedulerNode.fuse(node1, node2)
    # 尝试融合两个节点，判断是否可以进行融合并返回原因
    def _why_fuse_nodes(self, node1, node2) -> Optional[ReasonFusedNodes]:
        # 解构节点组，提取变量和约简信息
        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group

        # 检查变量和约简信息是否完全相同，若是则返回相同变量和约简的原因
        if vars1 == vars2 and reduce1 == reduce2:
            return ReasonFusedNodes.SAME_VARS_REDUCE
        # 检查第一个节点约简为空且变量与第二个节点变量加约简相同，若是则返回兼容约简的原因
        if reduce1 == () and vars1 == vars2 + reduce2:
            return ReasonFusedNodes.COMPATIBLE_REDUCTION
        # 检查节点是否具有兼容范围，若是则返回兼容范围但无约简的原因
        if self._can_fuse_nodes_with_compatible_ranges(node1, node2):
            return ReasonFusedNodes.COMPATIBLE_RANGES_NO_REDUCTION
        # TODO(jansel): 允许逐点融合（vars1, ()）后缀？
        # 如果以上条件都不满足，则返回空
        return None

    # 判断是否可以融合具有兼容范围的节点
    def _can_fuse_nodes_with_compatible_ranges(self, node1, node2):
        # 尝试融合具有兼容范围的 SchedulerNode/FusedSchedulerNode
        # 例如 (s0, s1, s2) 和 (s0 * s1 * s2)
        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group

        # 检查约简均为空且变量乘积相等，且至少一个节点的变量数为 1
        c1 = reduce1 == () and reduce2 == ()
        c2 = math.prod(vars1) == math.prod(vars2)
        c3 = len(vars1) == 1 or len(vars2) == 1
        if not (c1 and c2 and c3):
            return False

        # 选择变量较少的节点作为重新计算节点，另一个作为参考节点
        node_to_recomp = node1 if len(vars1) < len(vars2) else node2
        ref_node = node2 if len(vars1) < len(vars2) else node1

        # 对于 FusedSchedulerNode 类型的节点，不允许重新计算大小和主体
        # TODO: 我们可以扩展兼容范围对于 FusedSchedulerNode 的融合支持
        if isinstance(node_to_recomp, FusedSchedulerNode):
            return False

        # 可能出现节点具有相同元素数量但原始范围不同的情况，例如：
        # {d0: s0, d1: s1, d2: s2} vs {d0: s0*s1*s2}
        # 参见 https://github.com/pytorch/pytorch/pull/120077/files#r1500427848 获取更多细节
        # TODO: 如果允许至少 CSE 变量之一，我们可以修复这一点

        # 确保重新计算节点是 SchedulerNode 类型
        assert isinstance(node_to_recomp, SchedulerNode)
        # 对于 TemplateBuffer 类型的节点，不允许重新计算大小和主体
        if isinstance(node_to_recomp.node, ir.TemplateBuffer):
            return False
        assert isinstance(node_to_recomp.node, ir.ComputedBuffer)
        # node.data.get_size() 是 node.get_read_writes().var_ranges 的廉价版本，但不包含变量名称
        ranges2 = node_to_recomp.node.data.get_size()
        ranges1 = None
        # 如果参考节点是 FusedSchedulerNode 类型，则获取其范围并检查是否唯一
        if isinstance(ref_node, FusedSchedulerNode):
            ranges_set = set()
            for snode in ref_node.snodes:
                if isinstance(snode.node, ir.TemplateBuffer):
                    break
                assert isinstance(snode.node, ir.ComputedBuffer)
                ranges_set.add(tuple(snode.node.data.get_size()))

            if len(ranges_set) != 1:
                return False

            ranges1 = list(next(iter(ranges_set)))
        else:
            # 确保参考节点是 SchedulerNode 类型，并且其 node 是 ComputedBuffer 类型
            assert isinstance(ref_node, SchedulerNode)
            assert isinstance(ref_node.node, ir.ComputedBuffer)
            ranges1 = ref_node.node.data.get_size()

        # 检查重新计算节点和参考节点的范围是否相同
        if ranges1 != ranges2:
            return False

        return True
    def _can_fuse_horizontal_impl(self, node1, node2):
        # 确保 node1 和 node2 是 FusedSchedulerNode 或 SchedulerNode 的实例
        assert isinstance(node1, (FusedSchedulerNode, SchedulerNode))
        assert isinstance(node2, (FusedSchedulerNode, SchedulerNode))
        
        # 如果 node1 或 node2 中有任何一个是 OuterLoopFusedSchedulerNode，则无法水平融合
        if any(
            isinstance(node, OuterLoopFusedSchedulerNode) for node in (node1, node2)
        ):
            return False
        
        # 调用内部方法判断为何不能融合 node1 和 node2
        return self._why_fuse_nodes(node1, node2) is not None

    def can_fuse_horizontal(self, node1, node2):
        # 如果 node1 或 node2 是模板节点，则不能进行水平融合
        if node1.is_template() or node2.is_template():
            return False
        
        # 如果 node1 和 node2 的节点总数超过最大水平融合大小限制，则不能水平融合
        if (
            len(node1.get_nodes()) + len(node2.get_nodes())
            > config.cpp.max_horizontal_fusion_size
        ):
            return False
        
        # 调用内部方法判断是否能够水平融合 node1 和 node2
        return self._can_fuse_horizontal_impl(node1, node2)

    def _get_outer_loop_fusion_depth(self, node1, node2):
        DISABLE_OUTER_LOOP_FUSION = 0
        
        # 如果 node1 和 node2 中有任何一个不是 OuterLoopFusedSchedulerNode、FusedSchedulerNode 或 SchedulerNode 的实例，则禁用外部循环融合
        if not all(
            isinstance(node, (OuterLoopFusedSchedulerNode, FusedSchedulerNode, SchedulerNode))
            for node in (node1, node2)
        ):
            return DISABLE_OUTER_LOOP_FUSION
        
        # 确保 _node1 是最外层的 FusedSchedulerNode 或 SchedulerNode
        _node1 = (
            node1.get_outer_nodes()[-1]
            if isinstance(node1, OuterLoopFusedSchedulerNode)
            else node1
        )
        assert isinstance(_node1, (FusedSchedulerNode, SchedulerNode))
        
        # 确保 _node2 是最外层的 FusedSchedulerNode 或 SchedulerNode
        _node2 = (
            node2.get_outer_nodes()[0]
            if isinstance(node2, OuterLoopFusedSchedulerNode)
            else node2
        )
        assert isinstance(_node2, (FusedSchedulerNode, SchedulerNode))
        
        # 获取 _node1 和 _node2 的变量组和归约操作
        _, (vars1, reduce1) = _node1.group
        _, (vars2, reduce2) = _node2.group
        
        # 如果 vars1 和 vars2 都为空，且 reduce1 和 reduce2 都不为空，则只有归约操作，禁用外部循环融合
        if vars1 == () and vars2 == () and reduce1 != () and reduce2 != ():
            return DISABLE_OUTER_LOOP_FUSION
        
        # 如果 node1 和 node2 都是 OuterLoopFusedSchedulerNode，则比较它们的外部循环融合深度
        if all(isinstance(node, OuterLoopFusedSchedulerNode) for node in (node1, node2)):
            return (
                node1.outer_loop_fusion_depth
                if node1.outer_loop_fusion_depth == node2.outer_loop_fusion_depth
                else DISABLE_OUTER_LOOP_FUSION
            )
        
        # 计算外部循环融合深度，取两者变量组长度的最小值
        outer_loop_fusion_depth = min(len(vars1), len(vars2))
        
        # 如果外部循环融合深度大于等于 1，并且前两者变量组的前部分相等，则可能可以进行外部循环融合
        if (
            outer_loop_fusion_depth >= 1
            and vars1[:outer_loop_fusion_depth] == vars2[:outer_loop_fusion_depth]
        ):
            # 如果 node1 或 node2 是 OuterLoopFusedSchedulerNode，则与之前节点的外部循环融合深度相同
            if any(isinstance(node, OuterLoopFusedSchedulerNode) for node in (node1, node2)):
                _compare_node = (
                    node1 if isinstance(node1, OuterLoopFusedSchedulerNode) else node2
                )
                if _compare_node.outer_loop_fusion_depth == outer_loop_fusion_depth:
                    # 与之前节点的外部循环融合深度相同
                    return outer_loop_fusion_depth
                else:
                    return DISABLE_OUTER_LOOP_FUSION
            else:
                # 第一对生成 OuterLoopFusedSchedulerNode 的节点
                return outer_loop_fusion_depth
        
        # 否则禁用外部循环融合
        return DISABLE_OUTER_LOOP_FUSION
    # 检查两个节点是否可以在垂直方向进行外部循环融合
    def can_fuse_vertical_outer_loop(self, node1, node2):
        # 要求两个节点均不是模板节点
        return (
            not node1.is_template()
            and not node2.is_template()
            # 节点1的名称在节点2的祖先集合中，并且不是水平融合并且不是减少操作
            and node1.get_names() & node2.ancestors
            and not (
                self._can_fuse_horizontal_impl(node1, node2)
                and not node1.is_reduction()
            )
            # 外部循环融合深度至少为1
            and self._get_outer_loop_fusion_depth(node1, node2) >= 1
        )

    # 获取节点融合对的优先级
    def get_fusion_pair_priority(self, node1, node2):
        if self.can_fuse_vertical_outer_loop(node1, node2):
            # 如果可以进行垂直外部循环融合，优先级设为1
            return 1
        else:
            # 否则优先级设为0
            return 0

    # 检查两个节点是否可以在垂直方向进行融合
    def can_fuse_vertical(self, node1, node2):
        if node2.is_template():
            # 如果节点2是模板节点，不支持预操作和模板的融合
            return False
        if node1.is_template():
            # 如果节点1是模板节点，要求节点2不是减少操作节点
            return not node2.is_reduction()
        # 否则，节点1和节点2可以进行水平融合并且不是减少操作，或者可以进行垂直外部循环融合
        return (
            self._can_fuse_horizontal_impl(node1, node2) and not node1.is_reduction()
        ) or self.can_fuse_vertical_outer_loop(node1, node2)

    # 为节点生成代码
    def codegen_node(
        self,
        node: Union[OuterLoopFusedSchedulerNode, FusedSchedulerNode, SchedulerNode],
        # 以下省略参数和函数体...
    ):
        """
        Turn an set of pre-fused nodes into a C++ kernel.
        """
        # 获取当前对象的 kernel_group 属性
        kernel_group = self.kernel_group

        # 检查节点是否为 OuterLoopFusedSchedulerNode 类型
        if isinstance(node, OuterLoopFusedSchedulerNode):
            # 创建空列表，用于存储 CppKernelProxy 对象
            cpp_kernel_proxy_list: List[CppKernelProxy] = []
            # 创建空列表，用于存储 SchedulerNode 对象的列表
            nodes_list: List[List[SchedulerNode]] = []

            # 遍历外部节点集合
            for _node in node.get_outer_nodes():
                # 断言 _node 是 FusedSchedulerNode 或 SchedulerNode 类型
                assert isinstance(_node, (FusedSchedulerNode, SchedulerNode))
                # 获取 _node 中的子节点列表
                _nodes: List[SchedulerNode] = _node.get_nodes()  # type: ignore[assignment]
                # 创建 CppKernelProxy 对象，传入 kernel_group
                cpp_kernel_proxy = CppKernelProxy(kernel_group)
                # 为当前节点集合生成代码
                cpp_kernel_proxy.codegen_nodes(_nodes)

                # 将生成的 CppKernelProxy 对象添加到列表中
                cpp_kernel_proxy_list.append(cpp_kernel_proxy)
                # 将当前节点集合添加到列表中
                nodes_list.append(_nodes)

            # 在未来版本中，当每个内核都可以向量化时，选择平铺将更容易进行，
            # 我们将能够在融合阶段把 check_outer_fusion_loop_level_attr 提升到融合时，
            # 避免在融合时对内核进行分组，这些内核“看起来我们可以融合它们”，
            # 但实际上我们不能。
            if node.check_outer_fusion_loop_level_attr(
                cpp_kernel_proxy_list, node.outer_loop_fusion_depth
            ):
                # 将 cpp_kernel_proxy_list 合并成一个 outer_fusion_cpp_kernel_proxy
                outer_fusion_cpp_kernel_proxy = node.merge_outer_fusion_kernels(
                    cpp_kernel_proxy_list,
                )
                # 最终化合并后的内核
                kernel_group.finalize_kernel(
                    outer_fusion_cpp_kernel_proxy,
                    [_node for _nodes in nodes_list for _node in _nodes],
                )
            else:
                # 回退到标准循环代码生成
                for _kernel_proxy, _nodes in zip(cpp_kernel_proxy_list, nodes_list):
                    kernel_group.finalize_kernel(_kernel_proxy, _nodes)
        else:
            # 获取节点中的子节点列表
            nodes: List[SchedulerNode] = node.get_nodes()  # type: ignore[assignment]
            # 创建 CppKernelProxy 对象，传入 kernel_group
            cpp_kernel_proxy = CppKernelProxy(kernel_group)
            # 为当前节点集合生成代码
            cpp_kernel_proxy.codegen_nodes(nodes)
            # 最终化当前内核
            kernel_group.finalize_kernel(cpp_kernel_proxy, nodes)

        # 获取调度后的参数数量
        args_num = self._get_scheduled_num_args()
        # 如果参数数量超过最大融合内核参数数量
        if args_num > CppScheduling.MAX_FUSED_KERNEL_ARGS_NUM:
            # 设置刷新状态为 True
            self._set_flush_status(True)
    ):
        """
        Codegen a CPP template, possibly with fused epilogues
        """
        # 增加计数器以跟踪融合的 CPP 模板后续代码段的数量
        counters["inductor"]["cpp_epilogue_fusion_counter"] += len(epilogue_nodes)
        # 断言传入的模板节点是 CPP 模板缓冲的 SchedulerNode
        assert self.is_cpp_template(
            template_node
        ), "Template node passed to CppScheduler.codegen_template must be a SchedulerNode that wraps a CppTemplateBuffer"
        # 强制类型转换为 SchedulerNode，并获取组的第二个元素的值
        template_node = cast(SchedulerNode, template_node)
        _, (_, rnumel) = template_node.group
        # 断言 rnumel 为空元组
        assert rnumel == ()
        # 强制类型转换为 CppTemplateBuffer，并获取其内部的 epilogue_nodes 的节点
        ctb: ir.CppTemplateBuffer = cast(ir.CppTemplateBuffer, template_node.node)
        epilogue_ir_nodes: List[Optional[ir.Buffer]] = [n.node for n in epilogue_nodes]
        # 断言所有的 epilogue 节点都是 ComputedBuffer 类型
        assert all(
            isinstance(n, ir.ComputedBuffer) for n in epilogue_ir_nodes
        ), "Epilogue nodes must all be instances of ir.ComputedBuffer"
        # 生成 kernel 和 render 函数
        kernel, render = ctb.make_kernel_render(ctb, epilogue_nodes=epilogue_ir_nodes)
        # 在 kernel 内部进行如下操作
        with kernel:
            for node in [template_node, *epilogue_nodes]:
                node.mark_run()  # 标记节点为已运行状态，类型为 ignore[attr-defined]
            # 调用 render 函数获取生成的源代码
            src_code = render()

        # 使用 kernel 作为当前的 kernel 处理程序
        with V.set_kernel_handler(kernel):
            # 设置节点调度顺序为 template_node 和所有 epilogue_nodes
            node_schedule = [template_node, *epilogue_nodes]
            # 定义 kernel 名称，生成 kernel 代码，传入节点调度顺序和 kernel 参数
            kernel_name = self.define_kernel(src_code, node_schedule, kernel.args)
        # 调用 kernel 的具名 kernel_name 函数，传入 ctb 作为参数
        kernel.call_kernel(kernel_name, ctb)
        # 将 kernel 移除的缓冲添加到 removed_buffers
        V.graph.removed_buffers |= kernel.removed_buffers
        # 释放调度器中的缓冲
        self.scheduler.free_buffers()

    def _get_scheduled_num_args(self):
        # 返回 kernel_group 中的参数数量
        return self.kernel_group.get_num_args()

    def ready_to_flush(self):
        # 返回 _ready_to_flush 的值
        return self._ready_to_flush

    def codegen_sync(self):
        # 空函数，用于代码生成同步，没有具体实现
        pass

    def define_kernel(self, src_code, nodes, kernel_args=None):
        # 获取图形的包装器代码
        wrapper = V.graph.wrapper_code
        # 获取融合后内核的名称，如果使用描述性名称，则生成描述性名称
        fused_name = (
            get_fused_kernel_name(nodes, config.cpp.descriptive_names)
            if config.cpp.descriptive_names
            else ""
        )
        # 生成内核名称，使用 cpp 前缀、融合名称和下一个内核后缀
        kernel_name = "_".join(["cpp", fused_name, wrapper.next_kernel_suffix()])
        # 如果 graph.cpp_wrapper 为真，则使用 kernel_name 作为 kernel 声明名称，否则使用 "kernel"
        kernel_decl_name = kernel_name if V.graph.cpp_wrapper else "kernel"
        # 将源代码中的 Placeholder.KERNEL_NAME 替换为 kernel_decl_name
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_decl_name)
        # 将源代码中的 Placeholder.DESCRIPTIVE_NAME 替换为 kernel_name
        src_code = src_code.replace(str(Placeholder.DESCRIPTIVE_NAME), kernel_name)
        # 替换源代码中的 "#pragma CMT" 为 "//"
        src_code = src_code.replace("#pragma CMT", "//")

        # 创建缩进的编译包装器
        compile_wrapper = IndentedBuffer()
        # 如果 kernel_args 为 None，则使用 kernel_group 的 args
        args = self.kernel_group.args if kernel_args is None else kernel_args
        # 获取 args 的 C++ 参数定义
        _, _, arg_types = args.cpp_argdefs()
        # 如果 graph.cpp_wrapper 为假，则添加异步编译 cpp_pybinding 函数调用
        if not V.graph.cpp_wrapper:
            compile_wrapper.writeline(f"async_compile.cpp_pybinding({arg_types!r}, '''")
        # 将源代码插入到编译包装器中
        compile_wrapper.splice(src_code, strip=True)
        # 如果 graph.cpp_wrapper 为假，则添加异步编译结束标记
        if not V.graph.cpp_wrapper:
            compile_wrapper.writeline("''')")
        # 在 wrapper 中定义 kernel，传入 kernel_name、编译包装器的值和 cuda 参数为假
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), cuda=False)
        # 返回 kernel_name
        return kernel_name
    # 定义一个名为 flush 的方法，用于刷新当前对象状态
    def flush(self):
        # 调用 self.kernel_group.codegen_group() 方法生成源代码
        src_code = self.kernel_group.codegen_group()
        # 如果生成的源代码不为空
        if src_code:
            # 调用 self.define_kernel 方法定义一个内核，使用生成的源代码和计划节点作为参数
            kernel_name = self.define_kernel(
                src_code, self.kernel_group.scheduled_nodes
            )
            # 调用 self.kernel_group.call_kernel 方法调用内核，传递 V.graph.wrapper_code 和内核名字作为参数
            self.kernel_group.call_kernel(V.graph.wrapper_code, kernel_name)
        # 调用 self.reset_kernel_group() 方法重置内核组状态
        self.reset_kernel_group()
        # 设置对象的刷新状态为 False
        self._set_flush_status(False)
class KernelGroup:
    # KernelGroup 类的构造函数，初始化一些实例变量和数据结构
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.args = KernelArgs()  # 初始化 KernelArgs 实例给 self.args
        self.loops_code = BracesBuffer()  # 初始化 BracesBuffer 实例给 self.loops_code
        self.ws = WorkSharing(self.loops_code)  # 初始化 WorkSharing 实例给 self.ws，传入 self.loops_code
        self.stack = contextlib.ExitStack()  # 创建一个 ExitStack 实例给 self.stack
        self.stack.enter_context(self.ws)  # 将 self.ws 添加到 self.stack 的上下文中
        self.scheduled_nodes = []  # 初始化一个空列表给 self.scheduled_nodes

    # 创建一个新的 kernel 对象，并返回
    def new_kernel(self, cls, *args):
        return cls(self.args, parallel_num_threads(), *args)

    # 完成 kernel 生成，将 nodes 加入到 scheduled_nodes 中，生成代码并存入 loops_code 和 ws
    def finalize_kernel(self, new_kernel, nodes):
        self.scheduled_nodes += nodes
        code = self.loops_code
        ws = self.ws
        new_kernel.codegen_loops(code, ws)

    # 获取参数个数
    def get_num_args(self):
        arg_defs, call_args, arg_types = self.args.cpp_argdefs()
        args_num = len(arg_defs)
        return args_num

    # 根据操作系统返回导出声明字符串
    def get_export_declaration(self):
        return "__declspec(dllexport)" if _IS_WINDOWS else ""

    # 生成组代码的字符串表示
    def codegen_group(self, name=None) -> str:
        self.stack.close()  # 关闭 self.stack 上下文
        if not self.scheduled_nodes:
            return ""  # 如果 scheduled_nodes 为空，返回空字符串
        code = BracesBuffer()  # 创建一个 BracesBuffer 实例给 code

        # 1. 包含头文件
        # TODO: 支持其他平台上的内核性能分析
        enable_kernel_profile = (
            config.cpp.enable_kernel_profile and sys.platform == "linux"
        )
        if enable_kernel_profile:
            code.writelines(["#include <ATen/record_function.h>"])

        code.writeline(codecache.cpp_prefix())  # 写入 codecache 的 C++ 前缀代码

        # 2. 函数定义
        kernel_decl_name = str(Placeholder.KERNEL_NAME) if name is None else name
        kernel_name = str(Placeholder.DESCRIPTIVE_NAME) if name is None else name
        arg_defs, _, _ = self.args.cpp_argdefs()
        arg_defs = ",\n".ljust(25).join(arg_defs)
        func_export_decl = self.get_export_declaration()
        code.writeline(
            f'extern "C" {func_export_decl} void {kernel_decl_name}({arg_defs})'
        )

        # 3. 函数体
        with code.indent():  # 缩进开始
            if enable_kernel_profile:
                graph_id = V.graph.graph_id
                prefix = "graph_" + str(graph_id) + "_" if graph_id is not None else ""
                code.writelines(
                    [
                        f'RECORD_FUNCTION("{prefix + kernel_name}", c10::ArrayRef<c10::IValue>({{}}));'
                    ]
                )
            for old, new in self.args.aliases():
                code.writeline(f"auto {old} = {new};")
            code.splice(self.loops_code)  # 将 self.loops_code 中的代码插入到 code 中

        return code.getvalue()  # 返回生成的代码字符串

    # 调用 kernel 函数的包装方法
    def call_kernel(self, wrapper, kernel_name):
        _, call_args, arg_types = self.args.cpp_argdefs()
        wrapper.generate_kernel_call(
            kernel_name, call_args, cuda=False, arg_types=arg_types
        )


class CppWrapperKernelGroup(KernelGroup):
    # CppWrapperKernelGroup 类的构造函数，继承自 KernelGroup
    def __init__(self):
        super().__init__()  # 调用父类 KernelGroup 的构造函数
        self.args = CppWrapperKernelArgs()  # 将 CppWrapperKernelArgs 实例赋给 self.args


class WorkSharing:
    # WorkSharing 类的构造函数，初始化实例变量和数据结构
    def __init__(self, code):
        self.code = code  # 将参数 code 赋给 self.code
        self.in_parallel = False  # 初始化 in_parallel 为 False
        self.num_threads = None  # 初始化 num_threads 为 None
        self.stack = contextlib.ExitStack()  # 创建一个 ExitStack 实例给 self.stack
    # 定义一个并行执行方法，接受线程数作为参数
    def parallel(self, threads):
        # 如果已经处于并行状态且线程数与当前线程数不同，则关闭当前上下文
        if self.in_parallel and threads != self.num_threads:
            self.close()
        # 如果当前不在并行状态
        if not self.in_parallel:
            # 设置当前线程数为指定的线程数，并标记为处于并行状态
            self.num_threads = threads
            self.in_parallel = True
            # 根据配置决定是否动态分配线程数，然后在代码中写入相应的 OpenMP 并行指令
            if config.cpp.dynamic_threads:
                self.code.writeline("#pragma omp parallel")
            else:
                self.code.writeline(f"#pragma omp parallel num_threads({threads})")
            # 进入代码块的上下文管理器，并增加缩进
            self.stack.enter_context(self.code.indent())
            # 在并行区域内声明一个整数变量 tid，用于获取当前线程的编号
            self.code.writeline(
                "int tid = omp_get_thread_num();",
            )

    # 定义一个单线程执行方法
    def single(self):
        # 如果当前处于并行状态，则在代码中写入单线程执行的 OpenMP 指令
        if self.in_parallel:
            self.code.writeline("#pragma omp single")
        # 返回当前是否处于并行状态的标志
        return self.in_parallel

    # 定义一个关闭并行区域的方法
    def close(self):
        # 关闭当前代码块的上下文管理器
        self.stack.close()
        # 将并行状态标志设为 False
        self.in_parallel = False

    # 进入对象的上下文管理器
    def __enter__(self):
        self.stack.__enter__()
        return self

    # 退出对象的上下文管理器
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.__exit__(exc_type, exc_val, exc_tb)
# 使用 `dataclasses.dataclass` 装饰器定义一个数据类 `LoopLevel`，用于表示循环层级信息
@dataclasses.dataclass
class LoopLevel:
    # 可选的符号表达式，表示循环变量
    var: Optional[sympy.Expr] = None
    # 可选的符号表达式，表示循环大小
    size: Optional[sympy.Expr] = None
    # 符号表达式，表示循环的偏移量，默认为整数 0
    offset: sympy.Expr = sympy.Integer(0)
    # 符号表达式，表示循环的步长，默认为整数 1
    steps: sympy.Expr = sympy.Integer(1)
    # 整数值，表示并行度，默认为 0
    parallel: int = 0
    # 布尔值，表示是否使用 SIMD 和 OpenMP 并行化，默认为 False
    simd_omp: bool = False
    # 布尔值，表示是否使用 SIMD 向量化，默认为 False
    simd_vec: bool = False
    # 布尔值，表示是否折叠循环，默认为 False
    collapsed: bool = False
    # 布尔值，表示是否为约化循环，默认为 False
    is_reduction: bool = False
    # 可选的 LoopLevel 对象，表示父级循环
    parent: Optional["LoopLevel"] = None
    # 内部循环层级列表，表示下一级内部循环的信息，若为空则表示为最内层循环
    inner: List["LoopLevel"] = dataclasses.field(default_factory=list)
    # 可选的 CppKernel 对象，表示分配给此循环层级的内核，仅当它是叶子节点时有效
    kernel: Optional[CppKernel] = None

    def __post_init__(self):
        # 在 `__post_init__` 方法中初始化 `simd_nelements` 属性，此处描述了 C++/OpenMP 后端的背景信息，
        # 提到 `cpu_vec_isa.pick_vec_isa()` 用于检查向量化指令集，这是一个耗时且一次性的操作。由于装饰器
        # `@dataclasses.dataclass` 会在初始化 `LoopLevel` 时调用 `cpu_vec_isa.pick_vec_isa()` 来初始化
        # `simd_nelements`，可能会增加 Triton 后端的编译开销，因此将 `simd_nelements` 的初始化移到了
        # `__post_init__` 方法中。
        picked_vec_isa: cpu_vec_isa.VecISA = cpu_vec_isa.pick_vec_isa()
        self.simd_nelements: int = picked_vec_isa.nelements() if picked_vec_isa else 0

    def get_kernels(self) -> List[CppKernel]:
        """获取此循环层级下的所有内核对象"""
        if self.kernel:
            return [self.kernel]
        kernels = []
        for loop in self.inner:
            kernels += loop.get_kernels()
        return kernels

    def get_root(self):
        """获取此循环层级的根节点"""
        root = self
        while root.parent:
            root = root.parent
        return root

    def set_kernel(self, kernel: CppKernel):
        """
        设置此循环层级下的内核。不允许在此循环层级下进行分割。
        """
        if not self.inner:
            self.kernel = kernel
            loop: Optional[LoopLevel] = self
            assert loop is not None
            return
        assert len(self.inner) == 1
        self.inner[0].set_kernel(kernel)

    def get_loops_at(self, depth) -> List["LoopLevel"]:
        """获取指定深度处的所有循环层级"""
        if depth == 0:
            return [self]
        else:
            loops = []
            for loop in self.inner:
                loops += loop.get_loops_at(depth - 1)
            return loops
    # 定义一个方法用于根据瓦片化因子和深度分裂循环层次结构
    def split_with_tiling(self, depth, factor):
        
        # 定义一个内部函数，用于克隆当前循环层次结构的内部循环
        def clone_inner():
            inner = []
            if self.inner:
                for loop in self.inner:
                    inner.append(loop.clone())
            return inner
        
        # 定义一个内部函数，实际执行根据瓦片化因子进行分裂的操作
        def do_split_with_tiling():
            # 将瓦片化因子转换为Sympy整数
            sympy_factor = sympy.Integer(factor)

            # 计算偏移量，使其为大小除以瓦片化因子的最大整数倍
            offset = FloorDiv(self.size, sympy_factor) * sympy_factor
            
            # 创建主循环层次结构对象，设定偏移量和步长为瓦片化因子
            main_loop = LoopLevel(self.var, offset)
            main_loop.steps = sympy_factor
            main_loop.parallel = self.parallel
            main_loop.collapsed = False
            main_loop.is_reduction = self.is_reduction
            main_loop.inner = clone_inner()
            if main_loop.inner:
                for loop in main_loop.inner:
                    loop.parent = main_loop
            
            # 创建尾部循环层次结构对象，设定偏移量和并行性与主循环相同
            tail_loop = LoopLevel(self.var, self.size)
            tail_loop.offset = offset
            tail_loop.parallel = self.parallel
            tail_loop.collapsed = False
            tail_loop.is_reduction = self.is_reduction
            tail_loop.inner = clone_inner()
            if tail_loop.inner:
                for loop in tail_loop.inner:
                    loop.parent = tail_loop
            
            return main_loop, tail_loop
        
        # 如果深度为0，则执行瓦片化分裂操作
        if depth == 0:
            main_loop, tail_loop = do_split_with_tiling()
            parent = self.parent
            if parent:
                # 将主循环和尾部循环添加到父级循环的内部循环列表中
                parent.inner = [main_loop, tail_loop]
                main_loop.parent = parent
                tail_loop.parent = parent
            return main_loop, tail_loop
        else:
            # 如果深度不为0，则递归调用当前对象内部循环的瓦片化分裂方法
            assert len(self.inner) == 1
            return self.inner[0].split_with_tiling(depth - 1, factor)

    # 克隆当前循环层次结构对象及其内部循环
    def clone(self):
        loop = copy(self)
        loop.inner = []
        if self.inner:
            for inner_loop in self.inner:
                # 克隆内部循环并设置其父级为当前循环
                inner_loop_clone = inner_loop.clone()
                inner_loop_clone.parent = loop
                loop.inner.append(inner_loop_clone)
        loop.kernel = deepcopy(self.kernel)
        return loop
    def lines(self):
        # 计算偏移量表达式
        offset_expr = cexpr_index(self.offset)
        # 计算大小表达式
        size_expr = cexpr_index(self.size)
        # 如果禁止冗余循环且偏移量表达式等于大小表达式，则返回空
        if config.cpp.no_redundant_loops and offset_expr == size_expr:
            return None
        # 如果支持 SIMD 并且有并行处理且 SIMD 元素数大于 1，则生成相应的 SIMD 表达式
        simd = (
            f"simd simdlen({self.simd_nelements}) "
            if self.simd_omp and self.simd_nelements > 1
            else ""
        )
        # 如果需要并行处理
        if self.parallel:
            # TODO(jansel): 研究分块大小和其他调度方式
            line1 = "#pragma omp for"
            # 如果并行数大于 1，则加入 collapse 子句
            if self.parallel > 1:
                line1 += f" collapse({self.parallel})"
            # 如果需要 SIMD 并行，则修改 for 循环语句中的 for 关键字
            if self.simd_omp:
                line1 = line1.replace(" for ", f" for {simd}")
        elif self.simd_vec:
            # 如果需要 SIMD 向量化，则第一行为空字符串
            line1 = ""
        elif self.simd_omp:
            # 如果需要 SIMD 并行，则生成相应的 #pragma omp SIMD 语句
            line1 = f"#pragma omp {simd}"
        elif not self.is_reduction and cpp_builder.is_gcc():
            # 如果不是归约操作且正在使用 GCC 编译器，则生成相应的 #pragma GCC ivdep 语句
            line1 = "#pragma GCC ivdep"
        else:
            # 其他情况下第一行为空字符串
            line1 = ""
        # 构造 for 循环语句的偏移量字符串
        offset_str = f"{INDEX_TYPE} {self.var}={offset_expr}"
        # 构造 for 循环语句的大小字符串
        size_str = f"{self.var}<{size_expr}"
        # 构造 for 循环语句的步长字符串
        steps_str = f"{self.var}+={cexpr_index(self.steps)}"
        # 构造完整的 for 循环语句
        line2 = f"for({offset_str}; {size_str}; {steps_str})"
        # 如果需要折叠循环或第一行为空，则返回只包含第二行的列表
        if self.collapsed or not line1:
            return [line2]
        # 否则返回包含第一行和第二行的列表
        return [line1, line2]
@dataclasses.dataclass
class LoopNestWithSplit:
    """
    A loop-nest like structure but with some loop level split along
    the loop range into the main tiling loop and the tail. It is built
    with the `build` method as a loop nest and then split with
    `split_with_tiling` at some depth.

    A typical case is for vectorization where we typically split at the inner-most
    loop level. A more complicated case is 2D tiling where we split at
    both inner-most and outer levels.
    """

    root: Optional[List[LoopLevel]] = None
    kernel: Optional[CppKernel] = None

    @staticmethod
    def build(kernel: CppKernel):
        """Build a LoopNest with the given `kernel` as the leaf"""
        # 获取迭代变量和其对应的范围
        itervars = kernel.itervars
        ranges = kernel.ranges
        # 确保减少深度不为空
        reduction_depth = kernel.reduction_depth
        assert reduction_depth is not None

        # 初始化根节点为空列表
        root: List[LoopLevel] = []
        # levels 指向 root
        levels: List[LoopLevel] = root
        loop: Optional[LoopLevel] = None
        # 遍历迭代变量和其对应的范围
        for loop_idx, (var, size) in enumerate(zip(itervars, ranges)):
            # 创建 LoopLevel 对象
            loop = LoopLevel(var, size, parent=loop)
            # 如果循环索引大于等于减少深度，设置循环是否为减少操作的标志
            if loop_idx >= reduction_depth:
                loop.is_reduction = kernel.is_reduction
            levels.append(loop)
            levels = loop.inner
        # 使用根节点创建 LoopNestWithSplit 对象
        loop_nest = LoopNestWithSplit(root)
        # 如果存在 loop 对象，则将 kernel 分配给它，否则分配给 loop_nest 的 kernel
        if loop:
            loop.kernel = kernel
        else:
            loop_nest.kernel = kernel
        return loop_nest

    def __bool__(self):
        return bool(self.root)

    def get_loops_at(self, depth) -> List[LoopLevel]:
        """Get all the loop levels at the given `depth` (most outer loop has depth 0)"""
        # 初始化 loops 列表为空
        loops: List[LoopLevel] = []
        assert self.root is not None
        # 遍历根节点中的每个循环，获取指定深度的所有循环
        for loop in self.root:
            loops += loop.get_loops_at(depth)
        return loops

    @cache_on_self
    def max_parallel_depth(self):
        """
        Maximal allowed depth for parallelism:
        1) Levels without splitting and
        2) All reduction or non-reduction levels
        When the loop is split at the top level, the max depth is 1.
        """
        # 初始化最大深度为 0
        max_depth = 0
        assert self.root is not None
        loops = self.root
        # 如果根节点中循环数量大于 1，则最大深度为 1
        if len(loops) > 1:
            return 1
        # 获取根节点的第一个循环是否为减少操作
        is_reduction = loops[0].is_reduction if loops else False
        # 遍历循环，直到循环不再是单一且减少操作标志与根节点第一个循环的标志一致
        while len(loops) == 1 and loops[0].is_reduction == is_reduction:
            max_depth += 1
            loops = loops[0].inner
        return max_depth

    def is_reduction_only(self):
        """
        Whether all the loops are for reduction. Reduction loops
        are always the inner most ones.
        """
        # 判断根节点是否存在且第一个循环是否为减少操作
        return (
            self.root is not None and len(self.root) > 0 and self.root[0].is_reduction
        )
    # 标记并行性，设置循环的并行深度
    def mark_parallel(self, par_depth):
        # 断言并行深度不超过最大允许的并行深度
        assert (
            par_depth <= self.max_parallel_depth()
        ), "Parallel depth cannot exceed the maximal allowed parallel depth"
        # 断言根节点不为空
        assert self.root is not None
        # 获取根节点的循环
        loops = self.root
        # 遍历每个循环并设置并行深度
        for loop in loops:
            loop.parallel = par_depth
        # 根据并行深度设置循环内部的循环为折叠状态
        for i in range(1, par_depth):
            loops = loops[0].inner
            loops[0].collapsed = True

    # 使用 tiling 方法分割循环
    def split_with_tiling(self, depth, factor):
        """
        Split the loop into main and tail loops at given `depth` so that the range
        of the main loop has range `floor_div(range, factor) * factor` and
        the tail loop handles the remainder. The main loop is tiled
        according to the `factor`.
        """
        # 获取指定深度的循环
        loops = self.get_loops_at(depth)
        # 断言只有一个循环
        assert len(loops) == 1
        # 在指定深度处使用 tiling 方法分割循环
        split_loops = loops[0].split_with_tiling(0, factor)
        # 如果深度为 0，则更新根节点为分割后的循环
        if depth == 0:
            self.root = split_loops
        # 返回分割后的循环
        return split_loops

    # 获取所有在该循环嵌套下的内核对象
    def get_kernels(self) -> List[CppKernel]:
        """Get all kernel objects under this loop nest"""
        # 如果存在内核对象，直接返回该内核对象的列表
        if self.kernel:
            return [self.kernel]
        # 初始化内核列表
        kernels: List[CppKernel] = []
        # 断言根节点不为空
        assert self.root is not None
        # 遍历根节点的每个循环，并获取每个循环的内核对象，添加到内核列表中
        for loop in self.root:
            kernels += loop.get_kernels()
        # 返回所有内核对象的列表
        return kernels
```