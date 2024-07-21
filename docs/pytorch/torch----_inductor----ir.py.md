# `.\pytorch\torch\_inductor\ir.py`

```
log = logging.getLogger(__name__)
# 获取当前模块的日志记录器对象

indent = functools.partial(textwrap.indent, prefix="  ")
# 创建一个文本缩进函数，用于缩进字符串，以两个空格作为前缀

aten = torch.ops.aten
# 导入 torch 的原生操作模块

""" [Note: Inductor IR]

Inductor's IR is produced by executing 'lowering' code (see lowering.py).  Each
lowering is registered to a particular aten operator, and expects inputs that
correspond to the aten schema.  However, in place of torch Tensor inputs, lowerings
expect Inductor TensorBox inputs.

TensorBox IR represents torch tensors.  Tensors are sometimes single objects owning
storage, and sometimes views of another Tensor's storage.  Mutating tensor operations
(such as add_()) affect the underlying storage and any associated views.  Other operations
"""
# 多行字符串注释，介绍 Inductor IR 的生成和使用情况
# 定义一个函数validate_ir，用于验证给定节点或节点集合的结构是否符合指定规范
def validate_ir(node_or_nodes):
    # 定义内部函数_check_tensorbox，用于递归检查节点的类型
    def _check_tensorbox(nodes):
        # 如果节点为None，则直接返回
        if nodes is None:
            pass
        # 如果节点是列表或元组，则逐个检查其中的节点
        elif isinstance(nodes, (list, tuple)):
            for node in nodes:
                _check_tensorbox(node)
        # 如果节点是字典，则逐个检查其值对应的节点
        elif isinstance(nodes, dict):
            for node in nodes.values():
                _check_tensorbox(node)
        # 否则，确保节点的类型是预期的顶层IR节点类型之一
        else:
            assert isinstance(
                nodes,
                (
                    torch._inductor.ir.ExpandView,
                    DynamicScalar,
                    AssertScalar,
                    TensorBox,
                    sympy.logic.boolalg.Boolean,
                    Expr,
                    EffectfulKernel,
                ),
            ), f"Found {type(nodes)}, which is not a supported top level IR node. See [Note: Inductor IR]"

    # 对给定的节点或节点集合进行检查
    _check_tensorbox(node_or_nodes)


# 定义一个函数ops_wrapper，用于生成操作函数的包装器
def ops_wrapper(name):
    # 确保输入的操作名是字符串类型
    assert isinstance(name, str)

    # 定义一个返回指定操作的函数，该函数接受任意参数并调用ops模块中对应的操作
    def fn(*args, **kwargs):
        return getattr(ops, name)(*args, **kwargs)

    return fn


# 定义一个函数inverse_reorder，用于生成反向重排序函数
def inverse_reorder(order):
    # 创建一个映射字典，将给定顺序列表的元素与其索引对应起来
    inv_order = dict(zip(order, range(len(order))))

    # 定义一个重排序函数reindex，根据反向映射字典重新排列索引列表
    def reindex(index):
        # 确保传入的索引列表长度与顺序列表长度一致
        assert len(index) == len(inv_order)
        return [index[inv_order[i]] for i in range(len(index))]

    return reindex


# 定义一个函数same_reorder，用于生成相同顺序重排序函数
def same_reorder(order):
    # 定义一个重排序函数reindex，根据给定的顺序列表重新排列索引列表
    def reindex(index):
        # 确保传入的索引列表长度与顺序列表长度一致
        assert len(index) == len(order)
        return [index[order[i]] for i in range(len(index))]

    return reindex


# 定义一个函数fuse_reindexing，用于组合两个重排序函数
def fuse_reindexing(reindex1, reindex2):
    # 定义一个重排序函数reindex，先对索引列表应用第二个重排序函数，再对结果应用第一个重排序函数
    def reindex(index):
        return reindex1(reindex2(index))

    return reindex


# 定义一个列表NHWC_STRIDE_ORDER，表示NHWC格式的索引顺序
NHWC_STRIDE_ORDER = [3, 0, 2, 1]
# 定义一个列表NHWDC_STRIDE_ORDER，表示NHWDC格式的索引顺序
NHWDC_STRIDE_ORDER = [4, 0, 3, 2, 1]
# 将步幅顺序转换为填充顺序
def stride_order2fill_order(order):
    """
    Convert stride order to fill order
    For channel last format,

    stride order = [3, 0, 2, 1] and fill order = [1, 3, 2, 0]
    """
    # 创建一个从步幅位置到索引的映射字典
    lookup = {pos: idx for idx, pos in enumerate(order)}
    # 使用映射字典创建填充顺序列表
    fill_order = [lookup[i] for i in range(len(order))]
    return fill_order


# 将步幅转换为步幅顺序
def get_stride_order(seq: Sequence[int]) -> List[int]:
    """
    Convert strides to stride order
    """
    # 对步幅进行排序并获取其索引
    sorted_idx: List[int] = argsort(seq)
    # 创建一个初始为0的输出列表
    out = [0 for _ in range(len(seq))]
    # 根据排序后的索引，设置输出列表的值
    for i, elem in enumerate(sorted_idx):
        out[elem] = i
    return out


# 将 IR 节点转换为张量
def ir_node_to_tensor(x, guard_shape=True):
    if x is None:
        return None

    shape_fn: Callable[[Expr], Union[int, Expr]]
    # 根据 guard_shape 决定使用哪个形状函数
    if not guard_shape:
        shape_fn = V.graph.sizevars.size_hint
    else:
        shape_fn = identity
    # 获取节点的尺寸
    size = [shape_fn(s) for s in x.get_size()]
    stride: StrideType
    # 如果节点是存储和布局对象，则获取其步幅；否则计算连续步幅
    if is_storage_and_layout(x):
        stride = [shape_fn(s) for s in x.get_layout().stride]  # type: ignore[misc]
    else:
        stride = FlexibleLayout.contiguous_strides(size)  # type: ignore[arg-type]
    # 获取节点的数据类型和设备
    dtype = x.get_dtype()
    device = x.get_device()
    # 将尺寸和步幅转换为符号整数
    size = convert_shape_to_symint(size)
    stride = convert_shape_to_symint(stride)
    # 创建一个空的张量，并用零填充
    t = torch.empty_strided(
        size=size, stride=stride, dtype=dtype, device=device
    ).zero_()
    return t


# 可能将值转换为可选值
def may_convert_to_optional(value):
    if isinstance(value, list) and not value:
        # [None] 确保 cpp 包装器代码生成时会生成类似 {c10::nullopt} 而不是 {}
        return [None]
    return value


# 获取设备类型
def get_device_type(x):
    if getattr(x, "get_device", None):
        return get_device_type(x.get_device())
    if isinstance(x, torch.device):
        return x.type
    return None


# 判断节点是否在 Triton 上
def is_triton(x):
    return is_gpu(get_device_type(x))


# 判断节点是否在 CPU 上
def is_cpu(x):
    return get_device_type(x) == "cpu"


# IR 节点类
class IRNode:
    _current_origins: ClassVar[Set[Any]] = set()

    @staticmethod
    @contextlib.contextmanager
    # 设置当前的来源节点
    def current_origins(origins: Set[torch.fx.Node]):
        old = IRNode._current_origins
        IRNode._current_origins = old | origins
        try:
            yield
        finally:
            IRNode._current_origins = old

    # 初始化方法
    def __post_init__(self):
        # 将当前的来源设置为实例的来源
        self.origins = set(self._current_origins)
        # 如果启用了调试 IR 的 traceback，则获取调用堆栈
        self.traceback = traceback.format_stack() if config.debug_ir_traceback else None

    # 获取调用堆栈信息
    def get_traceback(self):
        return self.traceback

    # 通用的字符串表示方法
    def common_repr(self, shorten=True):
        origins = f"origins={getattr(self, 'origins', '')}"
        if shorten and len(origins) > 64:
            # 这可能非常长
            origins = f"{origins[:61]}..."
        return [origins]
    # 将给定的 lines 参数与 self.common_repr(shorten) 的结果相加，更新 lines
    lines = lines + self.common_repr(shorten)
    # 将 lines 列表中的每个元素转换为字符串
    lines = list(map(str, lines))
    # 如果 multiline 参数为 True，则将 lines 以逗号分隔并缩进，格式化成多行字符串
    if multiline:
        new_lines = indent(",\n".join(lines))
        return f"{type(self).__name__}(\n{new_lines}\n)"
    else:
        # 如果 multiline 参数为 False，则将 lines 直接作为字符串拼接到类名后面
        return f"{type(self).__name__}({lines})"

    # 检查指定的 name 是否在 self.get_read_names() 返回的集合中
    def is_user_of(self, name):
        return name in self.get_read_names()

    # 使用 @cache_on_self 装饰器缓存结果，返回一个集合，包含通过 self.get_reads() 获取的每个 dep 的 name 属性
    @cache_on_self
    def get_read_names(self):
        return {dep.name for dep in self.get_reads()}

    # 返回对象的 dtype 属性
    def get_dtype(self):
        return self.dtype

    # 抽象方法，抛出未实现错误，指示子类应该实现该方法
    def get_layout(self):
        raise NotImplementedError(f"get_layout() is not implemented by {type(self)}!")

    # 抽象方法，抛出未实现错误，指示子类应该实现该方法
    def get_size(self):
        raise NotImplementedError(f"get_size() is not implemented by {type(self)}!")

    # @property 装饰器，返回对象的 shape，实际上调用 self.get_size() 方法
    @property
    def shape(self):
        return self.get_size()

    # 返回通过 sympy_product 计算得到的对象尺寸的元素数量
    def get_numel(self):
        return sympy_product(self.get_size())

    # 判断对象的元素数量是否为零，使用 V.graph.sizevars.is_expr_static_and_true 判断是否静态等于 0
    def is_zero_elements(self):
        return V.graph.sizevars.is_expr_static_and_true(sympy.Eq(self.get_numel(), 0))  # type: ignore[arg-type]

    # 抽象方法，抛出未实现错误，指示子类应该实现该方法
    def realize(self):
        raise NotImplementedError(f"realize NYI on {type(self)}")

    # 抽象方法，抛出未实现错误，指示子类应该实现该方法
    def codegen_reference(self, writer=None):
        raise NotImplementedError(f"codegen_reference NYI on {type(self)}")

    # 下面的抽象方法声明用于确保 mypy 确定所有 IRNode 实例都具有这些函数定义，但在运行时不会产生任何效果
    get_device: Callable[[], torch.device]
    dtype: torch.dtype
    get_name: Callable[[], str]
    get_reads: Callable[[], Any]
    get_stride: Callable[[], Any]
    get_storage_numel: Callable[[], Any]
    has_exceeded_max_reads: Callable[[], bool]
    make_loader: Callable[[], Callable[[Any], Any]]
    make_indexer: Callable[[], Callable[[Any], Any]]
    mark_reuse: Callable[[int], None]
    realize_hint: Callable[[], None]
    get_unbacked_symbol_uses: Callable[[], Set[sympy.Symbol]]
@dataclasses.dataclass
class Loops(IRNode):
    device: torch.device  # 设备对象，表示代码运行的设备
    dtype: torch.dtype  # 数据类型对象，表示张量的数据类型
    inner_fn: Callable[..., Any]  # 可调用对象，表示内部函数
    ranges: List[Expr]  # 表达式列表，表示循环的范围

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        # 返回未支持的符号集合，包括范围表达式中的自由未支持符号和内部函数的自由未支持符号
        return set().union(
            *(free_unbacked_symbols(e) for e in self.ranges),
            self.inner_fn_free_unbacked_symbols(),
        )

    def __str__(self, names=("ranges",)):
        # 返回对象的字符串表示，包括设备类型、数据类型、内部函数的字符串表示以及其他指定名称的属性
        return self.str_helper(
            [
                f"'{self.device.type}'",  # 设备类型字符串
                str(self.dtype),  # 数据类型字符串
                self.inner_fn_str(),  # 内部函数的字符串表示
            ]
            + [f"{name}={getattr(self, name)}" for name in names]  # 其他指定名称的属性字符串
            + [f"origin_node={self.origin_node!r}"]  # 原始节点字符串表示
        )

    def __post_init__(self):
        super().__post_init__()
        self.origin_node = None  # 初始化原始节点为 None

    __repr__ = __str__  # __repr__ 方法与 __str__ 方法一致

    def get_device(self):
        # 返回对象的设备属性
        return self.device

    def get_origin_node(self):
        # 返回对象的原始节点属性
        return self.origin_node

    def get_size(self):
        # 返回对象的范围属性
        return self.ranges

    def get_pointwise_size(self):
        # 返回对象的范围属性（用于某些特定情况）
        return self.ranges

    def is_extern(self):
        # 返回 False，表示对象不是外部对象
        return False

    @classmethod
    def create(cls, *args, **kwargs):
        # 创建对象的类方法，支持传入额外的原始节点和回溯信息，返回创建的对象
        origin_node = kwargs.pop("origin_node", None)  # 弹出原始节点参数
        tb = kwargs.pop("traceback", None)  # 弹出回溯信息参数
        r = cls(*args, **kwargs)  # 使用剩余参数创建对象
        r.origin_node = origin_node  # 设置原始节点属性
        r.traceback = (
            tb or traceback.format_stack() if config.debug_ir_traceback else None
        )  # 设置回溯信息属性（如果启用调试模式）
        return TensorBox.create(r)  # 返回创建的对象的 TensorBox 实例

    @staticmethod
    def _index(ranges, prefix=SymT.INDEX):
        # 静态方法，根据范围和前缀创建索引列表
        return [
            sympy.Integer(0) if s == 1 else sympy_index_symbol_with_prefix(prefix, n)
            for n, s in enumerate(ranges)
        ]

    @cache_on_self
    def inner_fn_opcount(self):
        # 缓存装饰器，计算内部函数操作数
        opcounter = OpCounterCSE(V.MockHandler())  # 创建操作计数器
        with V.set_ops_handler(opcounter), patch.object(
            FlexibleLayout, "allow_indexing", True
        ):
            self.inner_fn(*self.inner_fn_args())  # 调用内部函数
            return opcounter.op_count  # 返回操作计数

    def inner_fn_args(self):
        # 返回内部函数的参数（索引列表）
        return (self._index(self.ranges),)

    def inner_fn_str(self):
        # 返回内部函数的字符串表示
        return V.KernelFormatterHandler.ir_to_string(
            self.inner_fn, *self.inner_fn_args()
        )

    def has_large_inner_fn(self):
        # 判断内部函数的操作数是否超过阈值
        return self.inner_fn_opcount() > config.realize_opcount_threshold

    def inner_fn_free_unbacked_symbols(self):
        # 返回内部函数中自由未支持符号的集合
        index = self._index(self.ranges)
        return extract_free_unbacked_symbols(self.inner_fn, index)

    def get_reads(self):
        # 获取对象的读操作集合
        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.get_reduction_type():
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                    self.get_reduction_size(),
                ).reads
            else:
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                ).reads
    # 抛出未实现错误，指示子类应该实现这个方法
    def get_reduction_size(self):
        raise NotImplementedError(
            f"get_reduction_size() is not implemented by {type(self)}!"
        )

    # 抛出未实现错误，指示子类应该实现这个方法
    def get_reduction_type(self):
        raise NotImplementedError(
            f"get_reduction_type() is not implemented by {type(self)}!"
        )

    # 抛出未实现错误，指示子类应该实现这个方法
    def constant_to_device(self, device):
        raise NotImplementedError(
            f"constant_to_device() is not implemented by {type(self)}!"
        )
# 定义一个函数nop_loader_fn，根据索引idx和数据类型dtype创建常数张量
def nop_loader_fn(idx, *, dtype):
    # 如果数据类型是浮点型，返回一个值为NaN的常数张量
    if dtype.is_floating_point:
        return ops.constant(float("nan"), dtype)
    else:
        # 否则返回一个值为0的常数张量
        return ops.constant(0, dtype)


# 定义一个类Pointwise，继承自Loops
class Pointwise(Loops):
    # 实现make_loader方法
    def make_loader(self):
        # 如果当前对象的循环为零元素，则返回一个nop_loader_fn函数的偏函数，其中dtype为当前对象的数据类型
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.dtype)

        # 否则返回当前对象的inner_fn函数
        return self.inner_fn

    # 实现get_reduction_size方法，返回一个空列表
    def get_reduction_size(self):
        return []

    # 实现get_reduction_type方法，返回None
    def get_reduction_type(self):
        return None

    # 实现store_output方法，用于存储输出
    def store_output(self, output_name, indexer, vars):
        # 调用make_loader方法获取一个加载器
        loader = self.make_loader()
        # 返回调用ops.store函数的结果，存储output_name，使用indexer(vars)进行索引，loader(vars)作为加载器
        return ops.store(output_name, indexer(vars), loader(vars))

    # 实现constant_to_device方法，将对象移动到指定设备
    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        # 调用make_loader方法获取一个加载器
        loader = self.make_loader()
        # 使用patch.object方法，将ConstantBuffer的override_device方法设为device，并应用于loader
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        # 返回一个新的Pointwise对象，设备为device，数据类型为self.dtype，加载器为loader，范围为self.ranges
        return Pointwise(device, self.dtype, loader, self.ranges)


# 定义一个类Scatter，继承自Pointwise
@dataclasses.dataclass
class Scatter(Pointwise):
    output_indexer: Callable[[List[Expr]], Expr]
    scatter_mode: Optional[str] = None

    # 实现constant_to_device方法，将对象移动到指定设备
    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        # 调用make_loader方法获取一个加载器
        loader = self.make_loader()
        # 使用patch.object方法，将ConstantBuffer的override_device方法设为device，并应用于loader
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        # 返回一个新的Scatter对象，设备为device，数据类型为self.dtype，加载器为loader，范围为self.ranges，
        # 输出索引器为self.output_indexer，散布模式为self.scatter_mode
        return Scatter(
            device,
            self.dtype,
            loader,
            self.ranges,
            self.output_indexer,
            self.scatter_mode,
        )

    # 实现store_output方法，用于存储输出
    def store_output(self, output_name, indexer, vars):
        # 调用make_loader方法获取一个加载器
        loader = self.make_loader()
        # 返回调用ops.store函数的结果，存储output_name，使用indexer(self.output_indexer(vars))进行索引，
        # loader(vars)作为加载器，模式为self.scatter_mode
        return ops.store(
            output_name,
            indexer(self.output_indexer(vars)),
            loader(vars),
            mode=self.scatter_mode,
        )


# 定义一个字典REDUCTION_COMBINE_FN，包含不同类型的汇总函数
REDUCTION_COMBINE_FN = {
    "any": ops_wrapper("logical_or"),
    "max": ops_wrapper("maximum"),
    "min": ops_wrapper("minimum"),
    "prod": ops_wrapper("mul"),
    "sum": ops_wrapper("add"),
    "xor_sum": ops_wrapper("bitwise_xor"),
}


# 定义函数get_reduction_combine_fn，根据汇总类型reduction_type和数据类型dtype获取汇总函数
def get_reduction_combine_fn(reduction_type, dtype, arg_break_ties_left=True):
    # 如果reduction_type在REDUCTION_COMBINE_FN字典中
    if reduction_type in REDUCTION_COMBINE_FN:
        # 获取对应的汇总函数
        combine_fn = REDUCTION_COMBINE_FN[reduction_type]
    # 如果 reduction_type 是 {"argmax", "argmin"} 中的一个
    elif reduction_type in {"argmax", "argmin"}:
        
        # 定义一个函数 combine_fn，用于合并两个元素
        def combine_fn(a, b):
            # 解构元组 a，得到值和索引
            a_value, a_index = a
            # 解构元组 b，得到值和索引
            b_value, b_index = b

            # 根据 reduction_type 判断条件掩码 mask
            if reduction_type == "argmin":
                mask = ops.lt(a_value, b_value)  # 如果是 argmin，则比较值的大小
            else:
                mask = ops.gt(a_value, b_value)  # 如果是 argmax，则比较值的大小

            # 判断相等的条件 equal
            equal = ops.eq(a_value, b_value)

            # 如果 dtype 是浮点类型，处理 NaN 值
            if is_float_dtype(dtype):
                a_isnan = ops.ne(a_value, a_value)  # 判断 a_value 是否为 NaN
                b_isnan = ops.ne(b_value, b_value)  # 判断 b_value 是否为 NaN
                mask = ops.logical_or(mask, ops.gt(a_isnan, b_isnan))  # NaN 值的比较
                equal = ops.logical_or(equal, ops.logical_and(a_isnan, b_isnan))  # NaN 值的相等比较

            # 处理 tie-breaking 策略
            tie = (
                ops.lt(a_index, b_index) if arg_break_ties_left else ops.gt(a_index, b_index)
            )
            mask = ops.logical_or(mask, ops.logical_and(equal, tie))  # 组合 mask 条件

            # 返回合并后的结果元组
            return (
                ops.where(mask, a_value, b_value),  # 根据 mask 选择值
                ops.where(mask, a_index, b_index),  # 根据 mask 选择索引
            )

    # 如果 reduction_type 是 "welford_combine"
    elif reduction_type == "welford_combine":
        
        # 定义一个函数 combine_fn，用于合并两个元素
        def combine_fn(a, b):
            # 解构元组 a，得到均值、m2 和权重
            a_mean, a_m2, a_weight = a
            # 解构元组 b，得到均值、m2 和权重
            b_mean, b_m2, b_weight = b

            # 计算均值之间的差异
            delta = b_mean - a_mean
            # 计算新的权重
            new_weight = a_weight + b_weight
            # 计算权重比例
            w2_over_w = b_weight / new_weight
            # 返回合并后的结果元组
            return (
                a_mean + delta * w2_over_w,  # 计算新的均值
                a_m2 + b_m2 + delta * delta * a_weight * w2_over_w,  # 计算新的 m2
                new_weight,  # 返回新的权重
            )

    # 如果 reduction_type 不在已知的处理类型中
    else:
        raise NotImplementedError(f"unknown reduction_type={reduction_type}")

    # 返回合适的 combine_fn 函数
    return combine_fn
# 使用 dataclasses 模块的 dataclass 装饰器，定义一个名为 Reduction 的类，它继承自 Loops
@dataclasses.dataclass
class Reduction(Loops):
    # reduction_ranges 是一个表达式列表，用于表示归约操作的范围
    reduction_ranges: List[Expr]
    # reduction_type 表示归约操作的类型，是一个字符串
    reduction_type: str
    # src_dtype 表示源数据类型，是一个 torch 的数据类型
    src_dtype: torch.dtype
    # reduction_hint 表示归约操作的提示类型
    reduction_hint: ReductionHint

    # 重写 __str__ 方法，返回继承类 Loops 的字符串表示形式
    def __str__(self):
        return Loops.__str__(  # type: ignore[call-arg]
            self, names=("ranges", "reduction_ranges", "reduction_type")
        )

    # 重写 __repr__ 方法，返回对象的字符串表示形式
    def __repr__(self):
        return self.__str__()

    # 获取未备份符号使用的方法，返回所有归约范围中的未备份符号集合
    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        return super().get_unbacked_symbol_uses() | set().union(
            *(free_unbacked_symbols(e) for e in self.reduction_ranges)
        )

    # 获取归约操作的大小，返回归约范围列表
    def get_reduction_size(self):
        return self.reduction_ranges

    # 获取归约操作的类型，返回归约类型字符串
    def get_reduction_type(self):
        return self.reduction_type

    # 存储归约结果的方法，返回归约操作的存储结果
    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        # 使用 ops 模块进行归约操作，生成存储结果
        value = ops.reduction(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
        )
        return ops.store_reduction(output_name, indexer(vars), value)

    # 计算索引长度的方法，返回 ranges 和 reduction_ranges 的总长度
    def index_length(self):
        return len(self.ranges) + len(self.reduction_ranges)

    # 获取内部函数参数的方法，返回 ranges 和 reduction_ranges 的索引
    def inner_fn_args(self):
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.RINDEX)
        return (index, rindex)

    # 获取内部函数自由未备份符号的方法，返回内部函数在 ranges 和 reduction_ranges 中的未备份符号集合
    def inner_fn_free_unbacked_symbols(self):
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.RINDEX)
        return extract_free_unbacked_symbols(self.inner_fn, index, rindex)

    # 将常量移动到指定设备的方法，要求所有读取的内容都是常量
    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        # 创建 loader 对象，将其设定为指定设备上的常量缓冲区，并返回新的 Reduction 对象
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Reduction(
            device,
            self.dtype,
            loader,
            self.ranges,
            self.reduction_ranges,
            self.reduction_type,
            self.src_dtype,
            ReductionHint.DEFAULT,
        )

    # 静态方法：计算归约操作的分割数
    @staticmethod
    def num_splits(
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        reduction_type,
        reduction_numel,
        input_node: Optional[IRNode] = None,
    ):
    def _unroll_reduction_fn(inner_fn, reduction_ranges, reduction_type, src_dtype):
        """Convert inner_fn from a reduction to a pointwise operation."""
        # 将 reduction_ranges 转换为静态形状列表
        reduction_ranges = [
            V.graph.sizevars.evaluate_static_shape(x) for x in reduction_ranges
        ]

        # 获取指定 reduction_type 和 src_dtype 的合并函数
        combine_fn = get_reduction_combine_fn(reduction_type, src_dtype)

        # 定义一个函数 fn，接受一个索引并执行指定的内部函数
        def fn(index):
            return functools.reduce(
                combine_fn,
                (
                    # 生成器表达式，调用 value_fn 以生成值
                    value_fn(index, rindex)
                    for rindex in itertools.product(
                        *[range(x) for x in reduction_ranges]
                    )
                ),
            )

        # 如果 reduction_type 是 "argmin" 或 "argmax"
        if reduction_type in ("argmin", "argmax"):
            # 创建一个固定布局的 flatten_index 函数
            flatten_index = FixedLayout(
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                reduction_ranges,
                FlexibleLayout.contiguous_strides(reduction_ranges),
            ).make_indexer()

            # 定义 value_fn 函数，用于返回值和 flatten_index 的索引
            def value_fn(index, rindex):
                rindex = [sympy.expand(i) for i in rindex]
                return (
                    inner_fn(index, rindex),
                    ops.index_expr(flatten_index(rindex), torch.int64),
                )

            # 返回一个 lambda 函数，该函数返回 fn(index) 的第二个元素
            return lambda index: fn(index)[1]
        else:
            # 如果不是 "argmin" 或 "argmax"，直接使用 inner_fn 作为 value_fn
            value_fn = inner_fn
            return fn

    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: List[Expr],
        reduction_ranges: List[Expr],
        reduction_type: str,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        input_node: Optional[IRNode] = None,
    ):
        # 方法：创建一个新的类实例，指定设备、目标数据类型、源数据类型、内部函数等参数
        pass

    @staticmethod
    def default_accumulator(reduction_type, dtype):
        # 静态方法：返回指定 reduction_type 和 dtype 的默认累加器值
        if reduction_type in {"max", "argmax"}:
            if is_float_dtype(dtype):
                return float("-inf")
            elif is_boolean_dtype(dtype):
                return 0
            else:
                return torch.iinfo(dtype).min
        if reduction_type in {"min", "argmin"}:
            if is_float_dtype(dtype):
                return float("inf")
            elif is_boolean_dtype(dtype):
                return 1
            else:
                return torch.iinfo(dtype).max

        # 返回根据 reduction_type 确定的默认值
        return {
            "sum": 0,
            "prod": 1,
            "xor_sum": 0,
            "any": 0,
            "welford_reduce": (0, 0, 0),
            "welford_combine": (0, 0, 0),
        }[reduction_type]

    @staticmethod
    def default_value(reduction_type, dtype):
        # 静态方法：返回指定 reduction_type 和 dtype 的默认值
        if reduction_type == "welford_reduce":
            return 0
        return Reduction.default_accumulator(reduction_type, dtype)

    @staticmethod
    def _multilayer_second_step_hint(
        split: int, numel_hint: int, reduction_hint: ReductionHint
    ):
        # 静态方法：提供多层次操作的第二步提示
        pass
    ) -> ReductionHint:
        # 如果 split 参数为 -1，则直接返回 reduction_hint
        if split == -1:
            return reduction_hint
        # 如果 split 小于等于 512，并且 numel_hint 也小于等于 512，并且 reduction_hint 为 ReductionHint.OUTER，则返回 ReductionHint.OUTER_TINY
        if (
            split <= 512
            and numel_hint <= 512
            and reduction_hint == ReductionHint.OUTER
        ):
            return ReductionHint.OUTER_TINY
        # 如果 split 小于等于 1024，并且 numel_hint 小于等于 256，并且 reduction_hint 为 ReductionHint.OUTER，则返回 ReductionHint.OUTER_TINY
        if (
            split <= 1024
            and numel_hint <= 256
            and reduction_hint == ReductionHint.OUTER
        ):
            return ReductionHint.OUTER_TINY

        # 否则返回 reduction_hint
        return reduction_hint

    @classmethod
    def _multilayer_wrap_loader(
        cls,
        loader,
        reduction_ranges,
        reduction_numel,
        split,
        block_size,
        default,
    ):
        # 使用 View.dynamic_reshape_indexer 方法重新索引 reduction_ranges，保证 reduction_numel 为静态表达式且等于 split 的倍数时返回 False
        need_mask = not V.graph.sizevars.is_expr_static_and_true(
            sympy.Eq(reduction_numel % split, 0)  # type: ignore[arg-type]
        )

        def wrapper_fn(index, reduction_index):
            (reduction_index,) = reduction_index
            *new_index, reduction_block = index
            # 计算 indices
            indices = block_size * reduction_block + reduction_index

            def body():
                # 调用 loader 函数，传入新索引和重新索引后的 indices
                return loader(new_index, reindex([indices]))

            # 如果需要 mask
            if need_mask:
                # 创建掩码 mask
                mask = ops.lt(
                    ops.index_expr(indices, torch.int32),
                    ops.index_expr(reduction_numel, torch.int32),
                )
                # 返回经过掩码的操作结果，否则直接返回 body 的执行结果
                return ops.masked(mask, body, default)
            else:
                return body()

        return wrapper_fn

    @classmethod
    def _multilayer_wrap_loader_existing_ranges(
        cls,
        loader,
        original_ranges,
        original_reduction_ranges,
        new_ranges,
        new_reduction_ranges,
        default,
    ):
        # 断言所有的 original_ranges 元素均为 1，用于确保 numel_hint == 1
        assert all(
            r == 1 for r in original_ranges
        ), f"Only enabled for numel_hint == 1, found {original_ranges=}"
        # 使用 View.dynamic_reshape_indexer 方法重新索引 original_reduction_ranges，合并 new_ranges 和 new_reduction_ranges
        reindex = View.dynamic_reshape_indexer(
            original_reduction_ranges, tuple(new_ranges) + tuple(new_reduction_ranges)
        )

        def wrapper_fn(merged_index, new_reduction_index):
            # 提取出原始索引和新的索引
            original_idx = merged_index[: len(original_ranges)]
            new_index = merged_index[len(original_ranges) :]
            # 调用 loader 函数，传入原始索引和重新索引后的新索引和新的 reduction_index
            return loader(
                original_idx,
                reindex(tuple(new_index) + tuple(new_reduction_index)),
            )

        return wrapper_fn

    @classmethod
    def create_multilayer_helper(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        wrapper_fn: Callable[..., Any],
        original_ranges: List[Expr],
        original_reduction_ranges: List[Expr],
        new_ranges: List[Expr],
        new_reduction_ranges: List[Expr],
        reduction_type: str,
        split: int,
        reduction_hint: ReductionHint,
    ):
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        # triton will automatically compute reductions in fp32 if reducing over fp16/bf16
        # within the kernel. keep the intermediate in fp32 so as to keep the whole reduction
        # in fp32 and not reduce precision by breaking up the kernel into multiple layers
        intermediate_dtype = (
            dst_dtype
            if dst_dtype not in (torch.float16, torch.bfloat16)
            else torch.float
        )
        # Create a Reduction object with specified parameters
        intermediate = Reduction.create(
            device,
            intermediate_dtype,
            src_dtype,
            wrapper_fn,
            new_ranges,
            new_reduction_ranges,
            reduction_type,
            reduction_hint,
        )
        intermediate.realize()  # Realize the intermediate reduction
        intermediate_loader = intermediate.make_loader()  # Create a loader for the intermediate reduction

        def intermediate_fn(index, reduction_index):
            return intermediate_loader([*index, *reduction_index])

        numel_hint = V.graph.sizevars.size_hint(sympy_product(original_ranges))
        reduction_hint = cls._multilayer_second_step_hint(
            split, numel_hint, reduction_hint
        )

        assert original_ranges == new_ranges[: len(original_ranges)]
        # Create a TensorBox with the intermediate Reduction object
        return TensorBox.create(
            Reduction(
                device,
                dst_dtype,
                intermediate_fn,
                original_ranges,
                new_ranges[len(original_ranges) :],
                reduction_type,
                src_dtype,
                reduction_hint,
            )
        )

    @classmethod
    def create_multilayer(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: List[Expr],
        reduction_ranges: List[Expr],
        reduction_type: str,
        split: int,
        reduction_hint: ReductionHint,
    ):
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        # TODO(jansel): realize the reduction so we can do dynamic indexing
        # Calculate the number of elements in the reduction
        reduction_numel = sympy_product(reduction_ranges)
        # Determine the block size for the reduction
        block_size = FloorDiv(reduction_numel + (split - 1), split)
        # Get the default value for the reduction type and destination dtype
        default = cls.default_value(reduction_type, dst_dtype)
        # Create a wrapper function for the inner function with specified parameters
        wrapper_fn = cls._multilayer_wrap_loader(
            inner_fn, reduction_ranges, reduction_numel, split, block_size, default
        )

        # Call create_multilayer_helper to create the multilayer reduction
        return cls.create_multilayer_helper(
            device,
            dst_dtype,
            src_dtype,
            wrapper_fn,
            ranges,
            reduction_ranges,
            [*ranges, split],  # type: ignore[list-item]
            [block_size],
            reduction_type,
            split,
            reduction_hint,
        )

    @classmethod
    def create_multilayer_existing_ranges(
        cls,  # 类方法，接受类作为第一个参数
        device: torch.device,  # 设备类型参数
        dst_dtype: torch.dtype,  # 目标数据类型参数
        src_dtype: torch.dtype,  # 源数据类型参数
        inner_fn: Callable[..., Any],  # 可调用对象，内部函数
        original_ranges: List[Expr],  # 列表，原始范围表达式
        original_reduction_ranges: List[Expr],  # 列表，原始减少范围表达式
        new_ranges: List[Expr],  # 列表，新范围表达式
        new_reduction_ranges: List[Expr],  # 列表，新减少范围表达式
        reduction_type: str,  # 字符串，减少类型
        reduction_hint: ReductionHint,  # 减少提示对象
    ):
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        # 使用指定减少类型和目标数据类型获取默认值
        default = cls.default_value(reduction_type, dst_dtype)
        # 调用类方法获取包装后的加载器函数
        wrapper_fn = cls._multilayer_wrap_loader_existing_ranges(
            inner_fn,
            original_ranges,
            original_reduction_ranges,
            new_ranges,
            new_reduction_ranges,
            default,
        )
        # 调用辅助方法创建多层辅助器，传递多个参数
        return cls.create_multilayer_helper(
            device,
            dst_dtype,
            src_dtype,
            wrapper_fn,
            original_ranges,
            original_reduction_ranges,
            [*original_ranges, *new_ranges],
            new_reduction_ranges,
            reduction_type,
            -1,
            reduction_hint,
        )
# 定义一个函数，根据给定的 reduction_type 返回输出的数量，如果 reduction_type 包含 "welford"，则返回 3，否则返回 1
def num_reduction_outputs(reduction_type):
    return 3 if "welford" in reduction_type else 1


# 定义一个类 WelfordReduction，继承自 Reduction 类
class WelfordReduction(Reduction):
    output_index: int

    # 初始化方法
    def __init__(
        self,
        device,
        dtype,
        inner_fns,
        ranges,
        reduction_ranges,
        reduction_type,
        reduction_hint,
        output_index,
    ):
        # 如果 inner_fns 只有一个函数，则直接赋值给 loader
        if len(inner_fns) == 1:
            loader = inner_fns[0]
        else:
            # 如果 inner_fns 包含多个函数，则定义一个 loader 函数，该函数将调用 inner_fns 中所有函数，并返回结果元组
            def loader(idx, reduction_idx):
                return tuple(fn(idx, reduction_idx) for fn in inner_fns)

        # 调用父类 Reduction 的初始化方法
        super().__init__(
            device,
            dtype,
            loader,  # 将 loader 函数作为 loader 参数传入父类初始化方法
            ranges,
            reduction_ranges,
            reduction_type,
            dtype,  # 两次传入 dtype 参数，第一个用于父类，第二个用于自身的初始化
            reduction_hint,
        )
        # 设置对象的 output_index 属性
        self.output_index = output_index

    # 存储规约结果的方法
    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        # 调用 ops.reduction 方法计算规约结果的值
        values = ops.reduction(
            self.dtype,
            self.src_dtype,  # 源数据类型
            self.reduction_type,  # 规约类型
            self.inner_fn(vars, reduction_vars),  # 调用 inner_fn 方法计算规约函数
        )
        # 获取指定输出索引处的值
        value = values[self.output_index]
        # 调用 ops.store_reduction 方法存储规约结果
        return ops.store_reduction(output_name, indexer(vars), value)

    # 类方法，返回默认的规约数值
    @classmethod
    def default_value(cls, reduction_type, dtype):
        return (0, 0, 0)  # 返回一个包含三个零的元组作为默认值

    # 类方法，创建多层结构的 WelfordReduction 实例
    @classmethod
    def create_multilayer(cls, device, dtype, inner_fns, ranges, reduction_ranges, reduction_type, split, reduction_hint):
        # 实现略，未提供完整代码段
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        # 计算归约操作的总元素个数
        reduction_numel = sympy_product(reduction_ranges)
        # 检查是否需要进行掩码操作，当归约元素个数不能整除分割数时需要掩码
        need_mask = not V.graph.sizevars.is_expr_static_and_true(
            sympy.Eq(reduction_numel % split, 0)  # type: ignore[arg-type]
        )

        # 如果需要掩码并且归约类型不是"welford_combine"
        if need_mask and reduction_type != "welford_combine":
            # 如果需要掩码，则"welford_reduce"不适用，因为掩码输入不应计入welford权重

            def constant(idx, reduction_idx, value):
                return ops.constant(value, dtype)

            # 创建多层归约对象
            return cls.create_multilayer(
                device=device,
                dtype=dtype,
                inner_fns=(
                    inner_fns[0],  # 使用第一个内部函数
                    partial(constant, value=0),  # 创建值为0的常数函数
                    partial(constant, value=1),  # 创建值为1的常数函数
                ),
                ranges=ranges,
                reduction_ranges=reduction_ranges,
                reduction_type="welford_combine",  # 使用"welford_combine"归约类型
                split=split,
                reduction_hint=reduction_hint,
            )

        # 计算分块大小
        block_size = FloorDiv(reduction_numel + (split - 1), split)
        # 创建Welford归约对象
        intermediates = WelfordReduction.create(
            device,
            dtype,
            tuple(
                cls._multilayer_wrap_loader(
                    loader,
                    reduction_ranges,
                    reduction_numel,
                    split,
                    block_size,
                    default=0,
                )
                for loader in inner_fns
            ),
            [*ranges, split],  # 创建用于Welford归约的范围
            [block_size],
            reduction_type,  # 归约类型
            reduction_hint,
        )
        # 实例化中间结果对象
        for i in intermediates:
            i.realize()

        # 创建中间结果的加载器列表
        i_loaders = [i.make_loader() for i in intermediates]

        # 定义中间加载器函数
        def intermediate_loader_fn(index, reduction_index, loader):
            return loader([*index, *reduction_index])

        # 计算输入的总元素个数的提示值
        numel_hint = V.graph.sizevars.size_hint(sympy_product(ranges))
        # 计算归约操作的提示值
        reduction_hint = cls._multilayer_second_step_hint(
            split, numel_hint, reduction_hint
        )
        # 返回Welford归约对象，用于第二步操作
        return WelfordReduction.create(
            device,
            dtype,
            tuple(
                partial(intermediate_loader_fn, loader=i.make_loader())
                for i in intermediates
            ),
            ranges,
            [split],  # 创建用于Welford归约的分割数
            # welford_reduce将一个输入转换为三个输出，这些输出使用welford_combine进行组合
            "welford_combine",
            reduction_hint,
        )
@dataclasses.dataclass
class Scan(Loops):
    # Represents a class `Scan` inheriting from `Loops`, containing:
    scan_ranges: List[Expr]  # List of expressions defining scan ranges
    size: List[Expr]  # List of expressions defining size
    combine_fn: Callable[[Tuple[Any, ...], Tuple[Any, ...]], Tuple[Any, ...]]
        # Callable function for combining tuples of any types
    reindex: Callable[[List[Expr], List[Expr]], List[Expr]]
        # Callable function for reindexing lists of expressions
    reduction_hint: ReductionHint  # Reduction hint object
    output_index: int  # Index referring to output tuple
    # output_index indexes the following tuples
    dtypes: Tuple[torch.dtype, ...]  # Tuple of Torch data types
    inner_fns: Tuple[Callable[..., Any], ...]  # Tuple of inner functions

    # HACK we mimick reduction

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        # TODO: Can combine_fn/reindex close over unbacked symbols? If so, we
        # need to explicitly represent the closure so we can pull out unbacked
        # symbols here
        return (
            super().get_unbacked_symbol_uses()
            | set().union(*(free_unbacked_symbols(e) for e in self.scan_ranges))
            | set().union(*(free_unbacked_symbols(e) for e in self.size))
        )

    def __post_init__(self):
        assert len(self.ranges) + len(self.scan_ranges) == len(self.size)
        super().__post_init__()

    def store_reduction(self, output_name, indexer, vars, scan_vars):
        # Store reduction operation
        idx = self.reindex(vars, scan_vars)  # Reindex variables
        values = [inner_fn(idx) for inner_fn in self.inner_fns]  # Compute inner function values
        result = ops.scan(self.dtypes, self.combine_fn, values)  # Perform scan operation
        return ops.store(output_name, indexer(idx), result[self.output_index])  # Store result

    def get_reduction_type(self):
        # return self.scan_op
        return "custom"

    def get_reduction_size(self):
        return self.scan_ranges  # Return scan ranges

    def get_size(self):
        return self.size  # Return size expressions

    def get_pointwise_size(self):
        return self.ranges  # Return ranges

    def index_length(self):
        return len(self.ranges) + len(self.scan_ranges)  # Length of ranges and scan_ranges

    def inner_fn_args(self):
        index = self._index(self.ranges)  # Compute index from ranges
        rindex = self._index(self.scan_ranges, SymT.RINDEX)  # Compute index from scan_ranges
        idx = self.reindex(index, rindex)  # Reindex using reindex function
        return (idx,)  # Return tuple of indices

    def inner_fn_free_unbacked_symbols(self):
        index = self._index(self.ranges)  # Compute index from ranges
        rindex = self._index(self.scan_ranges, SymT.RINDEX)  # Compute index from scan_ranges
        idx = self.reindex(index, rindex)  # Reindex using reindex function
        return extract_free_unbacked_symbols(self.inner_fn, idx)  # Extract unbacked symbols for inner function

    @classmethod
    def create(
        cls,
        device: torch.device,
        dtypes: Tuple[torch.dtype, ...],
        inner_fns: Tuple[Callable[[List[Expr]], Any], ...],
        size: List[Expr],
        axis: int,
        combine_fn: Callable[[Tuple[Any, ...], Tuple[Any, ...]], Tuple[Any, ...]],
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        **kwargs,
        # Create method for constructing Scan objects
        ) -> List[Optional[TensorBox]]:
        # 定义一个函数，接受一些参数并返回一个 Optional[TensorBox] 类型的列表

        pointwise_ranges = [*size[:axis], *size[axis + 1 :]]
        # 从 size 列表中提取 pointwise_ranges，排除指定的 axis

        scan_ranges = [size[axis]]
        # 从 size 列表中获取指定的 axis 的大小，作为 scan_ranges 列表的唯一元素

        if not V.graph.has_feature(device, BackendFeature.SCAN):
            # 如果在 V.graph 中未找到指定的 device 和 BackendFeature.SCAN 特性
            return [None] * len(dtypes)
            # 返回一个长度为 dtypes 的列表，每个元素为 None

        if len(dtypes) > 1 and not V.graph.has_feature(
            device, BackendFeature.TUPLE_REDUCTION
        ):
            # 如果 dtypes 的长度大于 1，并且在 V.graph 中未找到指定的 device 和 BackendFeature.TUPLE_REDUCTION 特性
            return [None] * len(dtypes)
            # 返回一个长度为 dtypes 的列表，每个元素为 None

        sizevars = V.graph.sizevars
        # 从 V.graph 中获取 sizevars

        scan_numel = sizevars.simplify(sympy_product(scan_ranges))
        # 使用 sizevars 对 scan_ranges 中的大小进行简化，得到 scan_numel

        assert len(dtypes) == len(inner_fns)
        # 断言 dtypes 和 inner_fns 的长度相等

        # 如果 scan_numel 小于等于 1，则进行以下操作
        if sizevars.is_expr_static_and_true(sympy.Le(scan_numel, 1)):  # type: ignore[arg-type]
            return [
                Pointwise.create(
                    device=device,
                    dtype=dtypes[output_index],
                    inner_fn=inner_fns[output_index],
                    ranges=size,
                )
                for output_index in range(len(dtypes))
            ]
            # 返回一个由 Pointwise.create() 函数生成的列表，每个元素是一个对象，包括设备、数据类型、内部函数和范围

        reduction_hint, num_splits = cls.num_splits(
            device=device,
            dtype=dtypes[0],
            inner_fn=inner_fns[0],
            axis=axis,
            pointwise_ranges=pointwise_ranges,
            scan_ranges=scan_ranges,
            combine_fn=combine_fn,
            scan_numel=scan_numel,
        )
        # 调用 cls.num_splits() 方法计算 reduction_hint 和 num_splits

        scan_type = Scan if num_splits <= 1 else SplitScan
        # 根据 num_splits 的值选择 Scan 或 SplitScan 类型

        if num_splits > 1 and torch.version.hip is not None:
            # 如果 num_splits 大于 1 并且当前环境支持 HIP（AMD GPU 的运行时平台）
            return [None] * len(dtypes)
            # 返回一个长度为 dtypes 的列表，每个元素为 None

        if num_splits > 1 and len(dtypes) > 1:
            # 如果 num_splits 大于 1 并且 dtypes 的长度大于 1
            return [None] * len(dtypes)
            # 返回一个长度为 dtypes 的列表，每个元素为 None

        def reindex(index, scan_index):
            # 定义一个 reindex 函数，接受 index 和 scan_index 作为参数
            assert len(scan_index) == len(scan_ranges)
            # 断言 scan_index 的长度与 scan_ranges 的长度相等
            assert len(index) == len(pointwise_ranges)
            # 断言 index 的长度与 pointwise_ranges 的长度相等
            return [*index[:axis], *scan_index, *index[axis:]]
            # 返回重新索引后的列表

        results = [
            TensorBox.create(
                scan_type(
                    device=device,
                    dtype=dtypes[output_index],
                    dtypes=dtypes,
                    inner_fn=inner_fns[output_index],
                    inner_fns=inner_fns,
                    size=size,
                    ranges=pointwise_ranges,
                    scan_ranges=scan_ranges,
                    combine_fn=combine_fn,
                    reindex=reindex,
                    reduction_hint=reduction_hint,
                    output_index=output_index,
                    **kwargs,
                )
            )
            for output_index in range(len(dtypes))
        ]
        # 使用列表推导式创建 results 列表，每个元素是 TensorBox.create() 返回的对象

        for result in results:
            result.realize()
            # 对 results 中的每个对象调用 realize() 方法

        return results
        # 返回 results 列表
    # 定义一个类方法用于计算分割数量
    def num_splits(
        cls,
        device: torch.device,
        dtype: torch.dtype,
        inner_fn: Callable[[List[Expr]], Any],
        axis: int,
        pointwise_ranges: List[Expr],
        scan_ranges: List[Expr],
        combine_fn: Callable[[Tuple[Any, ...], Tuple[Any, ...]], Tuple[Any, ...]],
        scan_numel: Expr,
    ):
        # TODO: custom splitting heuristic for scan
        # 定义一个包装函数，用于重新组织索引以供内部函数调用
        def wrapper_fn(idx, reduction_idx):
            return inner_fn([*idx[:axis], *reduction_idx, *idx[axis:]])

        # 调用 Reduction 类的 num_splits 方法，计算并返回分割数量
        return Reduction.num_splits(
            device=device,
            dst_dtype=dtype,
            src_dtype=dtype,
            inner_fn=wrapper_fn,
            ranges=pointwise_ranges,
            reduction_ranges=scan_ranges,
            reduction_type="sum",
            reduction_numel=scan_numel,
        )
# 表示一个需要在 CUDA 上通过 TritonSplitScanKernel 代码生成进行扫描操作的类。
@dataclasses.dataclass
class SplitScan(Scan):
    pass

# 表示一个排序操作的类，继承自 Loops 类。
@dataclasses.dataclass
class Sort(Loops):
    # 排序的范围表达式列表
    sort_ranges: List[Expr]
    # 大小表达式列表
    size: List[Expr]
    # 重新索引函数，接受两个表达式列表参数，返回一个表达式列表
    reindex: Callable[[List[Expr], List[Expr]], List[Expr]]
    # 减少提示对象
    reduction_hint: ReductionHint
    # 输出索引
    output_index: int
    # 下面元组的索引
    dtypes: Tuple[torch.dtype, ...]
    # 内部函数元组，每个元素是一个接受任意参数的可调用对象
    inner_fns: Tuple[Callable[..., Any], ...]

    # 稳定性标志
    stable: bool
    # 降序标志
    descending: bool

    # HACK we mimick reduction

    # 获取未支持符号使用的方法重写
    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        return (
            super().get_unbacked_symbol_uses()
            | set().union(*(free_unbacked_symbols(e) for e in self.sort_ranges))
            | set().union(*(free_unbacked_symbols(e) for e in self.size))
        )

    # 初始化后的处理
    def __post_init__(self):
        assert len(self.ranges) + len(self.sort_ranges) == len(self.size)
        super().__post_init__()

    # 存储减少结果的方法
    def store_reduction(self, output_name, indexer, vars, sort_vars):
        idx = self.reindex(vars, sort_vars)
        values = [inner_fn(idx) for inner_fn in self.inner_fns]
        result = ops.sort(self.dtypes, values, self.stable, self.descending)
        return ops.store(output_name, indexer(idx), result[self.output_index])

    # 获取减少类型的方法
    def get_reduction_type(self):
        return "sort"

    # 获取减少大小的方法
    def get_reduction_size(self):
        return self.sort_ranges

    # 获取大小的方法
    def get_size(self):
        return self.size

    # 获取逐点大小的方法
    def get_pointwise_size(self):
        return self.ranges

    # 索引长度的方法
    def index_length(self):
        return len(self.ranges) + len(self.sort_ranges)

    # 内部函数参数的方法
    def inner_fn_args(self):
        index = self._index(self.ranges)
        rindex = self._index(self.sort_ranges, SymT.RINDEX)
        idx = self.reindex(index, rindex)
        return (idx,)

    # 内部自由未支持符号的函数方法
    def inner_fn_free_unbacked_symbols(self):
        index = self._index(self.ranges)
        rindex = self._index(self.sort_ranges, SymT.RINDEX)
        idx = self.reindex(index, rindex)
        return extract_free_unbacked_symbols(self.inner_fn, idx)

    @classmethod
    def create(
        cls,
        device: torch.device,
        dtypes: Tuple[torch.dtype, ...],
        inner_fns: Tuple[Callable[[List[Expr]], Any], ...],
        size: List[Expr],
        axis: int,
        stable: bool,
        descending: bool,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        **kwargs,
        ) -> List[Optional[TensorBox]]:
        # 定义 pointwise_ranges 变量，它是一个包含了当前轴之前和之后所有尺寸的列表
        pointwise_ranges = [*size[:axis], *size[axis + 1 :]]
        # 定义 sort_ranges 变量，它是一个包含了当前轴尺寸的列表
        sort_ranges = [size[axis]]

        # 检查图形 V.graph 是否具有指定的设备和后端特性 SORT
        if not V.graph.has_feature(device, BackendFeature.SORT):
            # 如果不支持 SORT 特性，返回一个与 dtypes 列表长度相同的空列表
            return [None] * len(dtypes)

        # 获取图形 V.graph 的大小变量集合
        sizevars = V.graph.sizevars
        # 计算 sort_numel 作为简化后的排序范围内元素数量
        sort_numel = sizevars.simplify(sympy_product(sort_ranges))

        # 根据启发式方法，确定 triton 通常优于 aten.sort 的最小 rblock
        # 同时也不受带宽限制，融合也不太可能提高性能
        max_rblock = 256
        # 判断是否持久化内核计算
        is_persistent_kernel = (
            config.triton.persistent_reductions
            and sizevars.is_expr_static_and_true(sympy.Le(sort_numel, max_rblock))
        )
        if not is_persistent_kernel:
            # 如果不是持久化内核，返回一个与 dtypes 列表长度相同的空列表
            return [None] * len(dtypes)

        # 确保 dtypes 和 inner_fns 长度相等
        assert len(dtypes) == len(inner_fns)

        # 如果 sort_numel 小于等于 1，则排序结果是简单的复制操作
        if sizevars.is_expr_static_and_true(sympy.Le(sort_numel, 1)):  # type: ignore[arg-type]
            # 返回一个生成器表达式，创建 Pointwise 对象列表
            return [
                Pointwise.create(
                    device=device,
                    dtype=dtypes[output_index],
                    inner_fn=inner_fns[output_index],
                    ranges=size,
                )
                for output_index in range(len(dtypes))
            ]

        # 定义 reindex 函数，用于重新索引操作
        def reindex(index, sort_index):
            assert len(sort_index) == len(sort_ranges)
            assert len(index) == len(pointwise_ranges)
            return [*index[:axis], *sort_index, *index[axis:]]

        # 创建包含 TensorBox 对象的列表 results
        results = [
            TensorBox.create(
                Sort(
                    device=device,
                    dtype=dtypes[output_index],
                    dtypes=dtypes,
                    inner_fn=inner_fns[output_index],
                    inner_fns=inner_fns,
                    size=size,
                    ranges=pointwise_ranges,
                    sort_ranges=sort_ranges,
                    reindex=reindex,
                    reduction_hint=reduction_hint,
                    output_index=output_index,
                    stable=stable,
                    descending=descending,
                    **kwargs,
                )
            )
            for output_index in range(len(dtypes))
        ]

        # 对 results 中的每个 TensorBox 对象执行实现操作
        for result in results:
            result.realize()

        # 返回 results 列表，其中包含了经过实现的 TensorBox 对象
        return results
# 判断对象 x 是否具有存储框 StorageBox 和布局 Layout
def is_storage_and_layout(x):
    try:
        # 调用 as_storage_and_layout 函数尝试获取 StorageBox 和 Layout，不冻结对象
        as_storage_and_layout(x, freeze=False)
        # 如果成功获取，返回 True
        return True
    except NotImplementedError:
        # 如果抛出 NotImplementedError 异常，返回 False
        return False


# 判断对象 x 是否具有连续的存储框 StorageBox 和布局 Layout
def is_contiguous_storage_and_layout(x):
    try:
        # 调用 as_storage_and_layout 函数尝试获取 StorageBox 和 Layout，不冻结对象
        buffer, layout = as_storage_and_layout(x, freeze=False)
        # 如果布局 Layout 需要填充步幅（strides），则进行填充
        if layout.should_pad_strides():
            layout.pad_strides()
        # 返回布局是否连续
        return layout.is_contiguous()
    except NotImplementedError:
        # 如果抛出 NotImplementedError 异常，返回 False
        return False


# 将对象 x 简化为 StorageBox 和 Layout
def as_storage_and_layout(
    x, freeze=True, want_contiguous=False, stride_order=None, allow_padding=False
):
    """
    尝试将 x 简化为 StorageBox 和 Layout。

    allow_padding 只影响 stride_order 的应用方式。当 allow_padding 为 True 时，
    我们可以在应用 stride_order 时自由添加填充。
    """
    # 如果 x 是 TensorBox 类型的对象，递归调用 as_storage_and_layout 处理其 data 属性
    if isinstance(x, TensorBox):
        return as_storage_and_layout(
            x.data,
            freeze=freeze,
            want_contiguous=want_contiguous,
            stride_order=stride_order,
            allow_padding=allow_padding,
        )
    # 如果 x 是 StorageBox 类型且其 data 属性是 Buffer 类型
    if isinstance(x, StorageBox) and isinstance(x.data, Buffer):
        # 如果需要冻结对象
        if freeze:
            # 如果需要连续布局
            if want_contiguous:
                # 冻结布局并断言其连续性
                x.data.freeze_layout()
                assert x.data.layout.is_contiguous()
            # 如果提供了 stride_order 参数
            elif stride_order is not None:
                # 使用 stride_order 冻结布局
                x.data.freeze_layout_with_stride_order(
                    stride_order, allow_padding=allow_padding
                )
            else:
                # 决定布局
                x.data.decide_layout()
        # 返回 StorageBox 对象和其布局 Layout
        return x, x.data.layout
    # 如果 x 是 ReinterpretView 类型的对象
    if isinstance(x, ReinterpretView):
        # 使 x 的基础对象变为连续布局或按步幅排序，但不一定使 ReinterpretView 变为连续布局
        buffer, _ = as_storage_and_layout(
            x.data,
            freeze=freeze,
        )
        # 返回 Buffer 对象和 ReinterpretView 的布局 Layout
        return buffer, x.layout
    # 抛出 NotImplementedError 异常，表示未实现对 x 的处理
    raise NotImplementedError


# 创建一个带有 want_contiguous=True 参数的 as_storage_and_layout 函数的偏函数
as_contiguous_storage_and_layout = functools.partial(
    as_storage_and_layout, want_contiguous=True
)


# 判断对象 x 的布局是否按给定的步幅排序
def is_stride_order_storage_and_layout(x, stride_order):
    try:
        # 尝试获取对象 x 的 StorageBox 和 Layout，不冻结对象
        buffer, layout = as_storage_and_layout(x, freeze=False)
        # 返回布局是否按给定步幅排序
        return layout.is_stride_ordered(stride_order)
    except NotImplementedError:
        # 如果抛出 NotImplementedError 异常，返回 False
        return False


# BaseView 类，继承自 IRNode 类，用于表示基本视图
@dataclasses.dataclass
class BaseView(IRNode):
    # 数据属性，类型为 IRNode
    data: IRNode

    # 获取未备份符号使用情况的方法
    def get_unbacked_symbol_uses(self):
        return self.data.get_unbacked_symbol_uses()

    # 创建重新索引器的抽象方法，抛出 NotImplementedError 异常
    def make_reindexer(self):
        raise NotImplementedError(f"make_reindexer NYI on {self}")

    # 创建索引器的抽象方法
    def make_indexer(self):
        # 调用 data 对象的 make_indexer 方法
        inner = self.data.make_indexer()
        # 获取重新索引器
        reindex = self.make_reindexer()

        # 定义索引器函数，接受 idx 参数
        def indexer(idx):
            # 返回内部索引器函数 inner 的结果，传入重新索引器 reindex 处理的索引 idx
            return inner(reindex(idx))

        # 返回索引器函数
        return indexer

    # 创建加载器的抽象方法
    def make_loader(self):
        # 调用 data 对象的 make_loader 方法
        inner = self.data.make_loader()
        # 获取重新索引器
        reindex = self.make_reindexer()

        # 定义加载器函数，接受 idx 参数
        def loader(idx):
            # 返回内部加载器函数 inner 的结果，传入重新索引器 reindex 处理的索引 idx
            return inner(reindex(idx))

        # 返回加载器函数
        return loader

    # BaseView 类的属性
    @property
    # 返回 self.data 的数据类型
    def dtype(self):
        return self.data.dtype

    # 调用 self.data 的 get_layout 方法并返回结果
    def get_layout(self):
        return self.data.get_layout()

    # 调用 self.data 的 get_device 方法并返回结果
    def get_device(self):
        return self.data.get_device()

    # 返回 None，表示没有 origin node
    def get_origin_node(self):
        return None

    # 调用 self.data 的 get_name 方法并返回结果
    def get_name(self):
        return self.data.get_name()

    # 调用 self.get_size 方法并返回结果，此处与 self.get_pointwise_size 方法等价
    def get_pointwise_size(self):
        return self.get_size()

    # 调用 self.data 的 mark_reuse 方法，并传入参数 users
    def mark_reuse(self, users):
        return self.data.mark_reuse(users)

    # 调用 self.data 的 has_exceeded_max_reads 方法，并返回其结果
    def has_exceeded_max_reads(self):
        return self.data.has_exceeded_max_reads()

    # 调用 self.data 的 realize 方法，并返回其结果
    def realize(self):
        return self.data.realize()

    # 调用 self.data 的 realize_hint 方法，并返回其结果
    def realize_hint(self):
        return self.data.realize_hint()

    # 调用 self.data 的 get_storage_numel 方法，并返回其结果
    def get_storage_numel(self):
        return self.data.get_storage_numel()

    # 调用 self.data 的 is_extern 方法，并返回其结果
    def is_extern(self):
        return self.data.is_extern()  # type: ignore[attr-defined]

    # 调用 self.data 的 is_module_buffer 方法，并返回其结果
    def is_module_buffer(self):
        return self.data.is_module_buffer()  # type: ignore[attr-defined]

    # 使用 patch.object 方法设定 FlexibleLayout 类的 allow_indexing 属性为 True，
    # 然后调用 extract_read_writes 方法，并传入 self.make_loader() 和 self.get_size() 的结果，返回其中的 reads
    def get_reads(self):
        with patch.object(FlexibleLayout, "allow_indexing", True):
            return extract_read_writes(
                self.make_loader(),
                self.get_size(),
            ).reads

    # 如果 x 是 BaseView 类的实例，则将 x 赋值为 x.data，直到 x 不再是 BaseView 的实例为止，返回 x
    def unwrap_view(self):
        x: IRNode = self
        while isinstance(x, BaseView):
            x = x.data
        return x

    # 将数据移到指定设备 device，要求所有读取操作都是对常量的读取
    def constant_to_device(self, device):
        loader = self.make_loader()
        # 使用 patch.object 方法设定 ConstantBuffer 类的 override_device 属性为 device
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        # 返回一个新的 Pointwise 对象，使用指定的设备、数据类型、加载器 loader 和尺寸 self.get_size()
        return Pointwise(device, self.get_dtype(), loader, self.get_size())
# 使用 dataclasses 装饰器创建数据类 ExpandView，继承自 BaseView
@dataclasses.dataclass
class ExpandView(BaseView):
    # 类属性：表示视图的大小，是一个表达式列表
    size: List[Expr]

    # 静态方法：用于规范化视图的大小，将 -1 替换为正确的大小
    @staticmethod
    def _normalize_size(x, new_size):
        """Replace `-1` with correct sizes"""
        # 获取图形中定义的大小变量
        sizevars = V.graph.sizevars
        # 对新的大小列表中的每个元素应用 sympy 的 expand 函数
        new_size = list(map(sympy.expand, new_size))
        # 获取当前视图 x 的大小
        old_size = x.get_size()
        # 如果新大小比旧大小长，前面补 None
        old_size = [None] * (len(new_size) - len(old_size)) + list(old_size)
        # 断言新旧大小列表的长度相同
        assert len(new_size) == len(old_size)
        # 遍历大小列表，处理 -1 和广播兼容性
        for i in range(len(new_size)):
            if new_size[i] == -1:
                # 如果新大小是 -1，则使用对应位置的旧大小
                assert old_size[i] is not None
                new_size[i] = old_size[i]
            elif old_size[i] is None or old_size[i] == 1:
                # 如果旧大小是 None 或者 1，则跳过
                pass
            else:
                # 合理性检查：期望广播兼容性
                #
                # 注意：预期 new_size[i] == old_size[i] 应该已经被守卫，
                # 因为预计元公式已经教导了我们这个相等关系。
                assert (
                    sizevars.size_hint(new_size[i] - old_size[i], fallback=0) == 0
                ), "Broadcast failed in ExpandView({x.get_size()}, {new_size}) on dimension {i}"
        # 返回规范化后的新大小列表
        return new_size

    # 类方法：创建 ExpandView 实例，处理视图 x 和新大小 new_size
    @classmethod
    def create(cls, x, new_size):
        # 规范化新的大小列表
        new_size = cls._normalize_size(x, new_size)

        # 如果 x 是存储和布局对象
        if is_storage_and_layout(x):
            # 将 x 解释为存储和布局对象的存储和旧布局
            storage, old_layout = as_storage_and_layout(x)
            # 计算跳过的维度数
            skip = len(new_size) - len(old_layout.size)
            # 断言跳过的维度数大于等于 0
            assert skip >= 0
            # 创建新的步长列表，将旧布局的步长转换为新的步长
            new_stride = [sympy.Integer(0)] * skip
            for stride, size in zip(old_layout.stride, old_layout.size):
                new_stride.append(stride if size != 1 else sympy.Integer(0))
            # 创建新的固定布局对象
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                list(new_size),
                new_stride,
                old_layout.offset,
            )
            # 返回重新解释的视图对象，使用新的存储和布局
            return ReinterpretView(storage, new_layout)

        # 如果不是存储和布局对象，则返回创建的 ExpandView 对象
        return ExpandView(x, new_size)

    # 实例方法：获取视图的大小
    def get_size(self):
        # 返回视图的大小列表
        return self.size

    # 实例方法：创建重新索引器函数
    def make_reindexer(self):
        # 获取目标大小和实际大小
        target = self.get_size()
        actual = self.data.get_size()
        # 计算跳过的维度数
        skip = len(target) - len(actual)

        # 内部函数：用于重新索引
        def reindex(index):
            # 裁剪索引以匹配实际大小的长度
            index = list(index[skip:])
            # 断言索引的长度等于实际大小的长度
            assert len(index) == len(actual)
            # 遍历实际大小列表，处理广播维度
            for i in range(len(actual)):
                if actual[i] == 1:
                    # 将广播维度置为 0
                    index[i] = sympy.Integer(0)
            # 返回重新索引后的索引列表
            return index

        # 返回内部的重新索引器函数
        return reindex


# 使用 dataclasses 装饰器创建数据类 PermuteView，继承自 BaseView
@dataclasses.dataclass
class PermuteView(BaseView):
    # 类属性：表示视图的维度，是一个表达式列表
    dims: List[Expr]

    # 类方法：
    def create(cls, x, dims):
        # 转换负数索引为正数索引
        dims = cls._map_neg_dims(dims)
        # 确保dims是一个从0到len(dims)-1的完整集合
        assert set(dims) == set(range(len(dims)))

        # 如果输入x是存储对象并且有特定布局
        if is_storage_and_layout(x):
            # 将x解析为存储和布局对象
            storage, old_layout = as_storage_and_layout(x)
            # 创建新的布局对象，使用指定维度的尺寸和步幅
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                [old_layout.size[i] for i in dims],
                [old_layout.stride[i] for i in dims],
                old_layout.offset,
            )
            # 返回重新解释视图对象，使用新的存储和布局
            return ReinterpretView(storage, new_layout)

        # 如果不是特定布局的存储对象，则返回维度置换视图对象
        return PermuteView(x, dims)

    @classmethod
    def _map_neg_dims(cls, dims):
        # 将负数索引映射为正数索引
        return [dim if dim >= 0 else len(dims) + dim for dim in dims]

    def get_size(self):
        # 确保使用的维度索引是从0到len(self.dims)-1的完整集合
        assert set(self._map_neg_dims(self.dims)) == set(range(len(self.dims)))
        # 获取数据的尺寸，并返回按当前维度顺序排列的尺寸列表
        size = self.data.get_size()
        return [size[i] for i in self.dims]

    def make_reindexer(self):
        # 创建维度索引到逆索引的映射字典
        inv = {j: i for i, j in enumerate(self.dims)}
        # 创建逆索引列表，确保包含从0到len(self.dims)-1的完整集合
        inv = [inv[i] for i in range(len(self.dims))]  # type: ignore[index]
        assert set(inv) == set(range(len(self.dims)))

        # 定义并返回一个函数，用于根据逆索引重新排列输入的索引列表
        def reindex(index):
            return [index[i] for i in inv]

        return reindex
class SqueezeView(BaseView):
    @classmethod
    def create(cls, x, *, dim=None):
        # 检查输入的 x 是否符合存储和布局要求
        if is_storage_and_layout(x):
            # 将 x 转换为存储和布局对象
            storage, old_layout = as_storage_and_layout(x)
            new_size = []
            new_stride = []
            if dim is not None:
                # 如果指定了维度 dim，则验证其为整数
                assert isinstance(dim, int), "expected integer dim argument"
                # 并且在合理范围内
                assert 0 <= dim and dim < len(old_layout.size)

            # 遍历旧布局对象的尺寸和步长
            for i, (size, stride) in enumerate(zip(old_layout.size, old_layout.stride)):
                if dim is None:
                    # 如果未指定维度，保留尺寸不为1的维度信息
                    if size != 1:
                        new_size.append(size)
                        new_stride.append(stride)
                else:
                    # 如果指定了维度，保留除指定维度外的尺寸信息
                    if i != dim:
                        new_size.append(size)
                        new_stride.append(stride)
                    else:
                        # 验证指定维度的尺寸为1
                        assert size == 1, "expected squeezed size to be 1"

            # 根据新的尺寸和步长创建新的布局对象
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset,
            )
            # 返回重新解释视图对象，使用新的存储和布局对象
            return ReinterpretView(storage, new_layout)

        # 如果未符合存储和布局要求，且未指定维度，则重定向到通用视图的创建方法
        if dim is None:
            return View.create(x, [s for s in x.get_size() if s != 1])
        else:
            # 否则，验证指定维度的尺寸为1，并调用通用视图的创建方法
            assert x.get_size()[dim] == 1
            return View.create(x, [s for i, s in enumerate(x.get_size()) if i != dim])

    @staticmethod
    def squeezer(size: Tuple[sympy.Expr, ...]):
        # 从给定的尺寸中筛选出不为1的尺寸
        new_size = [s for s in size if s != 1]
        # 记录不为1的尺寸在原列表中的索引
        not_one = [i for i, s in enumerate(size) if s != 1]
        length = len(size)

        # 定义内部函数，用于重新索引操作
        def reindex(index: List[sympy.Expr]) -> Tuple[sympy.Expr, ...]:
            # 确保传入的索引长度与不为1的尺寸长度一致
            assert len(index) == len(not_one), f"{index} {not_one}"
            new_index = [sympy.Integer(0)] * length
            # 根据不为1的尺寸索引，重新组织索引值
            for idx, s in zip(not_one, index):
                new_index[idx] = s
            return tuple(new_index)

        # 返回不为1的尺寸列表和重新索引函数
        return new_size, reindex

    def __init__(self, data):
        # 禁止直接实例化 SqueezeView 对象，需使用 SqueezeView.create() 方法
        raise AssertionError("use SqueezeView.create()")


@dataclasses.dataclass
class GenericView(BaseView):
    size: List[Expr]
    reindex: Callable[..., Any]

    def make_reindexer(self):
        # 返回当前对象的重新索引方法
        return self.reindex

    def reindex_str(self):
        # 生成重新索引的字符串表示形式
        index_old = [
            sympy_index_symbol_with_prefix(SymT.INDEX, n) for n in range(len(self.size))
        ]
        index_new = list(self.reindex(index_old))
        return f"lambda {', '.join(map(str, index_old))}: {index_new}"

    def __str__(self):
        # 返回对象的字符串表示形式
        return self.str_helper(
            [self.data, f"size={self.size}", f"reindex={self.reindex_str()}"]
        )

    __repr__ = __str__

    @classmethod
    def create(cls, x, new_size, reindex):
        # 使用给定的参数创建 GenericView 对象
        return cls(x, list(new_size), reindex)

    def get_size(self):
        # 返回对象的尺寸信息
        return self.size


@dataclasses.dataclass
class View(GenericView):
    @staticmethod
    def handle_negative_index(idx, size):
        # 对索引进行展开，以便处理复杂的数学表达式
        idx = sympy.expand(idx)
        # 对尺寸进行展开，以便处理复杂的数学表达式
        size = sympy.expand(size)
        # 获取图的大小变量的形状环境中的表达式评估函数
        evaluate_expr = V.graph.sizevars.shape_env.evaluate_expr
        # 如果索引小于零，则将其转换为正索引
        if evaluate_expr(sympy.Lt(idx, 0)):
            idx = idx + size
        # 返回处理后的索引
        return idx

    @classmethod
    def create(cls, x, new_size):
        # 确保新尺寸是元组或列表类型
        assert isinstance(new_size, (tuple, list))
        # 解析并处理输入的尺寸，处理负数尺寸情况
        old_size, new_size = cls.resolve_negative_size(x.get_size(), new_size)

        # 如果旧尺寸与新尺寸相同，则返回原始对象
        if V.graph.sizevars.statically_known_list_equals(old_size, new_size):
            return x

        # 检查尺寸中是否包含未支持的符号
        unbacked_symbols_in_sizes = False
        if (
            len(free_unbacked_symbols(old_size)) > 0
            or len(free_unbacked_symbols(new_size)) > 0
        ):
            unbacked_symbols_in_sizes = True

        # 如果新尺寸中包含零，则返回一个虚假的重新索引函数
        if 0 in new_size:

            def fake_reindex(index):
                return tuple([0] * len(old_size))

            return cls(x, list(new_size), fake_reindex)

        # TODO: 创建一个新的 FixedTransferLayout 类，其输出布局受输入布局限制
        elif is_contiguous_storage_and_layout(x) or unbacked_symbols_in_sizes:
            # 如果尺寸中包含未支持的符号且不是连续存储和布局，则实现输入对象
            if unbacked_symbols_in_sizes and (not is_contiguous_storage_and_layout(x)):
                x = ExternKernel.realize_input(x)

            # 获取连续存储和布局
            storage, old_layout = as_contiguous_storage_and_layout(x)
            # 创建新布局对象
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                FlexibleLayout.contiguous_strides(new_size),
                old_layout.offset,
            )
            # 返回重新解释的视图对象
            return ReinterpretView(storage, new_layout)

        # 动态创建重塑索引器
        reindex = cls.dynamic_reshape_indexer(old_size, new_size)
        # 返回重塑后的对象
        return cls(x, list(new_size), reindex)

    @staticmethod
    def resolve_negative_size(old_size, new_size):
        # 简化新旧尺寸中的每个表达式
        new_size = [V.graph.sizevars.simplify(x) for x in new_size]
        old_size = [V.graph.sizevars.simplify(x) for x in old_size]

        # 处理新尺寸中的负数，将其转换为正数并进行适当的清除除法
        new_size = list(new_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                new_size[i] = sympy.Integer(1)
                new_size[i] = CleanDiv(sympy_product(old_size), sympy_product(new_size))
                break

        # 确保图的大小变量相等
        V.graph.sizevars.guard_equals(sympy_product(old_size), sympy_product(new_size))
        # 返回处理后的旧尺寸和新尺寸
        return old_size, new_size
    # 定义一个静态方法 dynamic_reshape_indexer，用于动态重塑索引器
    def dynamic_reshape_indexer(cls, old_size, new_size):
        try:
            # 尝试调用类方法 _dynamic_reshape_indexer 处理重塑索引
            reindex = cls._dynamic_reshape_indexer(old_size, new_size)
        except (AssertionError, IndexError):
            # 如果遇到断言错误或索引错误，采用备用方案
            # 计算 old_size 的乘积并组成一个列表 flat
            flat = [sympy_product(old_size)]
            # 尝试用 flat 作为新的 old_size 重新调用 _dynamic_reshape_indexer
            reindex1 = cls._dynamic_reshape_indexer(old_size, flat)
            # 尝试用 flat 和 new_size 调用 _dynamic_reshape_indexer
            reindex2 = cls._dynamic_reshape_indexer(flat, new_size)
            # 合并 reindex1 和 reindex2 的结果
            reindex = fuse_reindexing(reindex1, reindex2)
        # 返回最终的重塑索引 reindex
        return reindex
    def _dynamic_reshape_indexer(old_size, new_size):
        """
        Perform a reshape entirely by modifying indexing math
        """
        size_hint = V.graph.sizevars.size_hint
        # TODO: These symbols may not escape, if they don't assert so and
        # treat them as temporary
        vars = [
            sympy_index_symbol_with_prefix(SymT.VIEW, i) for i in range(len(new_size))
        ]

        stack_new = list(zip(vars, new_size))
        stack_old = list(old_size)

        view_expr = []
        while stack_new and stack_old:
            size_old = stack_old.pop()
            var, size_new = stack_new.pop()
            if size_old == 1:
                # Append 0 for dimensions that collapse to size 1
                view_expr.append(sympy.Integer(0))
                stack_new.append((var, size_new))  # re-add to stack
            elif size_new == 1:
                stack_old.append(size_old)  # re-add to stack
            elif size_hint(size_new) == size_hint(size_old):
                # Directly use the variable for matching sizes
                view_expr.append(var)
                V.graph.sizevars.guard_equals(size_new, size_old)
            elif size_hint(size_new) < size_hint(size_old):
                # Handle cases where new size hint is smaller
                while size_hint(size_new) < size_hint(size_old):
                    var2, size_new2 = stack_new.pop()
                    var = var2 * size_new + var
                    size_new = size_new * size_new2
                view_expr.append(var)
                V.graph.sizevars.guard_equals(size_new, size_old)
            elif size_hint(size_new) > size_hint(size_old):
                # Handle cases where new size hint is larger
                divisor = sympy.Integer(1)
                modulus = size_old
                view_expr.append(ModularIndexing(var, divisor, modulus))
                divisor = divisor * modulus
                while size_hint(size_new) > size_hint(size_old):
                    modulus = stack_old.pop()
                    view_expr.append(ModularIndexing(var, divisor, modulus))
                    divisor = divisor * modulus
                    size_old = size_old * modulus
                V.graph.sizevars.guard_equals(size_new, size_old)
            else:
                raise AssertionError

        # Handle any remaining dimensions in the old size stack
        while stack_old:
            size_old = stack_old.pop()
            V.graph.sizevars.guard_equals(size_old, 1)  # type: ignore[arg-type]
            view_expr.append(sympy.Integer(0))

        # Handle any remaining dimensions in the new size stack
        while stack_new:
            var, size_new = stack_new.pop()
            V.graph.sizevars.guard_equals(size_new, 1)  # type: ignore[arg-type]

        view_expr.reverse()
        assert len(view_expr) == len(old_size)

        def reindex(index):
            assert len(index) == len(vars), (len(index), len(vars))
            replacements = dict(zip(vars, index))
            return tuple(sympy_subs(x, replacements) for x in view_expr)  # type: ignore[arg-type]

        return reindex


注释完成。
@dataclasses.dataclass
class ReinterpretView(BaseView):
    """Pretend our storage has a different layout"""

    layout: Layout  # 接收一个 Layout 对象作为参数，用于指定数据的布局

    def __post_init__(self):
        super().__post_init__()  # 调用父类的初始化方法
        if isinstance(self.data, BaseView):
            self.data = self.data.unwrap_view()  # 如果 data 是 BaseView 的实例，则解包为其原始视图

    def __str__(self):
        return self.str_helper(
            [
                self.data,
                self.layout,
            ]
        )  # 返回对象的字符串表示形式，包括 data 和 layout 的信息

    __repr__ = __str__  # 将 __repr__ 方法与 __str__ 方法一致，返回相同的字符串表示形式

    def get_name(self):
        return self.data.get_name()  # 获取数据视图的名称

    def get_device(self):
        return self.layout.device  # 获取布局对象的设备信息

    def get_origin_node(self):
        return None  # 返回 None，表示没有原始节点信息

    @property
    def dtype(self):
        return self.layout.dtype  # 获取布局对象的数据类型

    def get_size(self):
        return list(self.layout.size)  # 获取布局对象的尺寸信息并返回为列表

    def get_stride(self):
        return list(self.layout.stride)  # 获取布局对象的步长信息并返回为列表

    def make_loader(self):
        def loader(index):
            indexer = self.layout.make_indexer()
            return ops.load(self.get_name(), indexer(index))

        return loader  # 返回一个加载器函数，用于根据索引加载数据

    def make_indexer(self):
        return self.layout.make_indexer()  # 返回布局对象的索引器

    def get_layout(self):
        return self.layout  # 返回对象的布局信息

    def freeze_layout(self):
        pass  # 空方法，暂无实现内容

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        return (
            free_unbacked_symbols(self.layout.size)  # 获取布局尺寸中未支持的符号集合
            | free_unbacked_symbols(self.layout.stride)  # 获取布局步长中未支持的符号集合
            | free_unbacked_symbols(self.layout.offset)  # 获取布局偏移量中未支持的符号集合
        )

    def codegen_reference(self, writer=None):
        # reinterpret_tensor is similar to as_strided except:
        # - offset is added to the existing offset (rather than replacing it)
        # - view tracking is disabled similar to unsafe_view
        return V.graph.wrapper_code.codegen_reinterpret_view(
            self.data,  # 传递数据对象
            self.layout.size,  # 传递布局尺寸
            self.layout.stride,  # 传递布局步长
            self.layout.offset,  # 传递布局偏移量
            writer,  # 传递写入器对象
        )


class SliceView(View):
    @classmethod
    def normalize_start_end(cls, x, dim, start, end):
        """
        Normalize start and end such that both are in the range
        [0, x.get_size()[dim]] and start <= end.
        """
        sizevars = V.graph.sizevars
        dim_size = x.get_size()[dim]  # 获取维度 dim 的尺寸信息

        if any(free_unbacked_symbols(x) for x in (start, end, dim_size)):

            def clamp(x, lower, upper):
                return sympy.Min(sympy.Max(x, lower), upper)
        else:

            def clamp(x, lower, upper):
                return sizevars.evaluate_min(sizevars.evaluate_max(x, lower), upper)

        def clamp_wrap(val, lower, upper, default):
            if val is None:
                return default
            val = cls.handle_negative_index(val, dim_size)  # 处理负索引情况
            return clamp(val, lower, upper)

        start = clamp_wrap(start, 0, dim_size, 0)  # 规范化起始位置
        end = clamp_wrap(end, start, dim_size, dim_size)  # 规范化结束位置
        return start, end  # 返回规范化后的起始和结束位置
    def create(cls, x, dim, start, end, step=1, clamp=True):
        # 将步长表达式展开为 Sympy 表达式
        step = sympy.expand(step)
        # 断言步长是 Sympy 表达式或者大于零的数值
        assert isinstance(step, sympy.Expr) or step > 0
        try:
            # 如果起始为0且结束大于等于2的63次方减1且步长为1，则直接返回输入张量 x
            if start == 0 and end >= 2**63 - 1 and step == 1:
                return x
        except TypeError:
            pass

        # 获取图的大小变量集合
        sizevars = V.graph.sizevars
        # 复制输入张量的大小
        new_size = list(x.get_size())

        # 注意：通常我们默认启用夹紧功能
        # 仅对 split_with_sizes 不进行夹紧处理。对于 split_with_sizes，大小应已经是有效的，
        # 在此情况下失败是可以接受的，因为无效的大小可能会触发静默错误。
        if clamp:
            # 规范化起始和结束索引
            start, end = cls.normalize_start_end(x, dim, start, end)

        # 计算新的维度大小
        new_size[dim] = FloorDiv(end - start + (step - 1), step)

        if is_storage_and_layout(x):
            # 快速路径
            # 将输入张量转换为存储和布局对象
            storage, old_layout = as_storage_and_layout(x)
            # 复制旧布局的步长信息
            new_stride = list(old_layout.stride)
            # 更新指定维度的步长
            new_stride[dim] = new_stride[dim] * step
            # 创建新的布局对象
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset + old_layout.stride[dim] * start,
            )
            # 返回重新解释后的视图对象
            return ReinterpretView(storage, new_layout)

        def reindex(index):
            # 断言索引的维度与新大小一致
            assert len(index) == len(new_size), f"wrong ndim {index} {new_size}"
            # 复制索引
            index = list(index)
            # 更新指定维度的索引
            index[dim] = index[dim] * step + start
            return index

        # 重定向到通用视图对象
        return SliceView(x, size=new_size, reindex=reindex)
# 定义一个继承自 IRNode 的基础常量类 BaseConstant
class BaseConstant(IRNode):
    # 属性：数据类型，torch.dtype 对象
    dtype: torch.dtype
    # 属性：设备类型，torch.device 对象
    device: torch.device

    # 获取大小的方法，返回一个空元组
    def get_size(self):
        return ()

    # 获取设备的方法，返回设备对象
    def get_device(self):
        return self.device

    # 获取原始节点的方法，始终返回 None
    def get_origin_node(self):
        return None

    # 标记重用的方法，接受用户参数但什么也不做
    def mark_reuse(self, users):
        pass

    # 判断是否超过最大读取次数的方法，始终返回 False
    def has_exceeded_max_reads(self):
        return False

    # 获取读取内容的方法，返回一个空元组
    def get_reads(self):
        return ()

    # 判断是否为外部的方法，始终返回 False
    def is_extern(self):
        return False


# 使用 @dataclasses.dataclass 装饰的常量类 Constant，继承自 BaseConstant
@dataclasses.dataclass
class Constant(BaseConstant):
    # 属性：值，可以是任何类型
    value: Any
    # 属性：数据类型，torch.dtype 对象
    dtype: torch.dtype
    # 属性：设备类型，torch.device 对象
    device: torch.device

    # 创建加载器的方法，返回一个闭包函数 loader，用于返回常量值
    def make_loader(self):
        def loader(index):
            return ops.constant(self.value, self.dtype)

        return loader

    # 实现常量的方法，什么也不做
    def realize(self):
        pass

    # 将常量移动到指定设备的方法，返回一个新的 Constant 实例
    def constant_to_device(self, device):
        return Constant(self.value, self.dtype, device)


# 使用 @dataclasses.dataclass 装饰的索引常量类 IndexingConstant，继承自 BaseConstant
@dataclasses.dataclass
class IndexingConstant(BaseConstant):
    # 属性：索引，可以是任何类型
    index: Any
    # 属性：数据类型，torch.dtype 对象
    dtype: torch.dtype
    # 属性：设备类型，torch.device 对象
    device: torch.device

    # 创建加载器的方法，返回一个闭包函数 loader，用于返回索引表达式
    def make_loader(self):
        def loader(index):
            return ops.index_expr(self.index, self.dtype)

        return loader

    # 将索引常量移动到指定设备的方法，返回一个新的 IndexingConstant 实例
    def constant_to_device(self, device):
        return IndexingConstant(self.index, self.dtype, device)


# 检查给定的步长 stride 是否适合给定形状 shape 的连续布局
def is_contiguous_strides_for_shape(stride, shape):
    return all(
        size == 1 or left == right
        for left, right, size in zip(
            stride, FlexibleLayout.contiguous_strides(shape), shape
        )
    )


# 根据数据类型 dtype 计算对齐值
def get_align_for_dtype(dtype):
    """
    CUDA 最大内存事务大小为一个 warp 的 128 字节。
    我们选择 `128 // dtype.itemsize` 作为对齐值，以便 GPU 可以执行协同内存访问。
    """
    return 128 // dtype.itemsize


# 使用 @dataclasses.dataclass 装饰的布局类 Layout，继承自 IRNode
@dataclasses.dataclass
class Layout(IRNode):
    # 构造方法，接受设备、数据类型、大小列表、步长（可选）、偏移量（默认为 0）
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: List[Expr],
        stride: Optional[Sequence[Union[Expr, int]]],
        offset: Expr = Integer(0),
    ):
        # 断言：如果步长不为 None，则大小列表和步长列表长度必须相等
        assert stride is None or len(size) == len(
            stride
        ), f"size={size}, stride={stride}"
        # 设置设备属性
        self.device = device
        # 设置数据类型属性
        self.dtype = dtype
        # 断言：确保大小列表中的每个元素都是 Expr 或 int 类型
        assert all(isinstance(s, (Expr, int)) for s in size)
        # 设置大小列表属性
        self.size = size
        # 设置步长属性
        self._stride = stride
        # 设置偏移量属性
        self.offset = offset

    # 获取步长的属性方法
    @property
    def stride(self):
        return self._stride

    # 对象转换为字符串的方法
    def __str__(self):
        offset = ""
        if self.offset != 0:
            offset = f", offset={self.offset}"
        return (
            f"{type(self).__name__}('{self.device.type}', {self.dtype}, "
            f"size={self.size}, stride={self.stride}{offset})"
        )

    # __repr__ 方法与 __str__ 方法相同
    __repr__ = __str__

    # 判断布局是否连续的方法
    def is_contiguous(self):
        return is_contiguous_strides_for_shape(self.stride, self.size)

    # 静态方法：获取灵活布局的连续步长
    @staticmethod
    def contiguous_strides(shape):
        # 省略实现细节，由调用者自行实现
        pass
    def is_channels_last_contiguous(shape, strides):
        # 确定维度数
        ndim = len(shape)
        # 如果维度数不是4或5，或者第二维的大小为1，则不是channels-last布局
        if ndim not in [4, 5] or shape[1] == 1:
            return False
        # 检查每个维度上的步长是否符合channels-last布局
        for left, right, size in zip(
            strides, make_channels_last_strides_for(shape), shape  # type: ignore[arg-type]
        ):
            if size != 1 and left != right:
                return False
        return True

    def is_transposed(self):
        # 检查步长数组是否表示转置布局
        for left, right, size in zip(
            self.stride,
            reversed(FlexibleLayout.contiguous_strides(list(reversed(self.size)))),
            self.size,
        ):
            if size != 1 and left != right:
                return False
        return True

    def is_stride_ordered(self, order):
        assert len(self.stride) == len(order)

        # 忽略尺寸为1的维度，它们不影响布局
        non_1_indices = [
            i
            for i, dim in enumerate(self.size)
            if V.graph.sizevars.size_hint(dim, fallback=2) != 1
        ]

        # 提取非1尺寸维度的步长和指定的顺序
        stride = [self.stride[i] for i in non_1_indices]
        order = [order[i] for i in non_1_indices]

        def sorted_indices(arr):
            sorted_arr = sorted(arr)
            return [sorted_arr.index(element) for element in arr]

        # 重新排序和重新索引顺序数组
        order = sorted_indices(order)

        # 根据指定顺序重新排列步长
        stride_ordered = [-1] * len(order)
        for i in range(len(order)):
            stride_ordered[order[i]] = V.graph.sizevars.size_hint(stride[i])
        # 检查步长是否按升序排列
        for i in range(len(order) - 1):
            if stride_ordered[i] > stride_ordered[i + 1]:
                return False
        return True

    def is_channels_last_stride_ordered(self):
        # 创建channels-last的顺序（NCHW, NCDHW，C是第一维度）
        order = [0] + list(reversed(range(1, len(self.stride) - 1)))
        order = [len(order)] + order
        # 检查步长是否按指定的顺序排列
        return self.is_stride_ordered(order)

    @staticmethod
    def pad_strides(self):
        assert isinstance(self, FlexibleLayout)
        assert self._stride is not None
        # 对步长进行填充，以确保合适的布局
        self._stride = self._pad_strides(self._stride, self.size, self.dtype)

    def should_pad_strides(self):
        # 检查是否应该对步长进行填充
        return config.comprehensive_padding and isinstance(self, FlexibleLayout)

    def as_fixed(self):
        if isinstance(self, FixedLayout):
            return self

        if self.should_pad_strides():
            self.pad_strides()
        # 将当前布局转换为FixedLayout
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            self.stride,
            self.offset,
        )

    def make_indexer(self):
        assert (
            FlexibleLayout.allow_indexing
        ), f"convert {type(self).__name__} to FixedLayout first"
        # 将当前布局转换为FixedLayout，并生成相应的索引器
        return self.as_fixed().make_indexer()
    # 定义一个特殊方法 __eq__，用于比较两个对象是否相等
    def __eq__(self, other) -> bool:
        # 检查对象的设备是否相等
        return (
            self.device == other.device
            # 检查对象的数据类型是否相等
            and self.dtype == other.dtype
            # 检查对象的大小是否相等
            and self.size == other.size
            # 检查对象的步幅是否相等
            and self.stride == other.stride
            # 检查对象的偏移量是否相等
            and self.offset == other.offset
        )

    # 定义一个方法 storage_size，返回一个 sympy 表达式
    def storage_size(self) -> sympy.Expr:
        # 调用 compute_required_storage_length 函数计算所需存储长度
        return compute_required_storage_length(self.size, self.stride, self.offset)  # type: ignore[arg-type, return-value]
# 定义一个固定布局的张量布局类，继承自Layout类
class FixedLayout(Layout):
    """A Tensor layout we cannot change"""

    # 初始化方法，接受设备、数据类型、尺寸、步长和偏移量等参数
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: Union[List[Expr], List[int]],
        stride: Optional[Sequence[Union[Expr, int]]] = None,
        offset: Union[Expr, int] = Integer(0),
    ):
        # 如果步长未提供，则使用FlexibleLayout类的contiguous_strides方法生成
        if stride is None:
            stride = FlexibleLayout.contiguous_strides(size)
        # 调用父类Layout的初始化方法
        super().__init__(
            device,
            dtype,
            size,  # type: ignore[arg-type]
            stride,
            offset,  # type: ignore[arg-type]
        )

    # 创建一个闭包函数，用于计算给定索引处元素的偏移量
    def make_indexer(self):
        """A closure containing math to read a given element"""

        def indexer(index):
            # 断言索引长度与步长和尺寸长度相同
            assert len(index) == len(self.stride)
            assert len(index) == len(self.size)
            result = self.offset
            # 根据索引计算偏移量
            for idx, stride, sz in zip(index, self.stride, self.size):
                if sz != 1:
                    result = result + idx * stride
            return result

        return indexer


class FlexibleLayout(Layout):
    """A Tensor layout we are allowed to change"""

    # 允许索引的标志，默认为False
    allow_indexing = False

    # 静态方法，生成连续排列的步长
    # 注意：此方法无法正确处理零大小的张量
    @staticmethod
    def contiguous_strides(sizes):
        if len(sizes) == 0:
            return []
        reversed_strides = [sympy.Integer(1)]
        # 生成反向排列的步长
        for size in reversed(sizes[1:]):
            reversed_strides.append(size * reversed_strides[-1])
        return list(reversed(reversed_strides))

    # 静态方法，根据给定的填充顺序创建步长
    @staticmethod
    def fill_ordered(sizes, order):
        """
        Create a stride based on the order the dimensions should be filled in.

        In this format, channels last would be:
            [1, 3, 2, 0]
        """
        # 断言维度的顺序与尺寸的顺序集合相同
        assert set(range(len(sizes))) == set(order), (sizes, order)
        next_stride = sympy.Integer(1)
        strides = [None] * len(order)

        # 根据给定的顺序填充步长数组
        for i in order:
            strides[i] = next_stride
            next_stride = next_stride * sizes[i]
        return strides

    # 静态方法，根据给定的排序创建步长
    @staticmethod
    def stride_ordered(sizes, order):
        """
        Create a stride based on the sorted order of a permuted range.

        In this format, channels last would be:
            [3, 0, 2, 1]
        """
        # 断言维度的顺序与给定顺序集合相同
        assert set(range(len(sizes))) == set(order)
        # 获取填充顺序
        fill_order = stride_order2fill_order(order)
        # 使用填充顺序生成步长
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    @staticmethod
    def stride_ordered_for_memory_format(sizes, memory_format):
        """
        根据内存格式创建一个基于步长的顺序。

        将内存格式转换为步长顺序，
        因此 channels_last 等同于:
            FlexibleLayout.stride_ordered(sizes, [3, 0, 2, 1])

        此接口不支持内存格式 `torch.preserve_format`，
        应使用它来从其他来源推断格式。
        """
        if memory_format == torch.channels_last:
            return FlexibleLayout.stride_ordered(sizes, NHWC_STRIDE_ORDER)
        elif memory_format == torch.channels_last_3d:
            return FlexibleLayout.stride_ordered(sizes, NHWDC_STRIDE_ORDER)
        elif memory_format == torch.contiguous_format:
            return FlexibleLayout.contiguous_strides(sizes)
        else:
            log.debug(
                "stride_ordered_for_memory_format, unsuppored memory_format: %s",
                memory_format,
            )
            raise NotImplementedError

    @staticmethod
    def same_ordered(sizes, stride):
        """
        创建一个具有与给定步长相同顺序的步长数组。

        例如，如果给定的步长是 [1000, 1, 100, 10]，
        则填充顺序应为 [1, 3, 2, 0]。
        """
        assert len(sizes) == len(stride)
        stride = [V.graph.sizevars.size_hint(x) for x in stride]
        fill_order = sorted(range(len(stride)), key=stride.__getitem__)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    def as_stride_order(self, order, allow_padding=False):
        new_stride = self.stride_ordered(self.size, order)
        if self.should_pad_strides() and allow_padding:
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)

        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
        )

    def as_fill_order(self, order):
        new_stride = self.fill_ordered(self.size, order)
        if self.should_pad_strides():
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
        )

    def as_same_order(self, stride):
        """
        将对象的步长调整为与给定步长相同的顺序。

        如果应该填充步长，则使用 `_pad_strides` 方法填充。
        """
        new_stride = self.same_ordered(self.size, stride)
        if self.should_pad_strides():
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
        )

    def __init__(self, device, dtype, size, stride_order=None):
        """
        初始化方法，根据给定的设备、数据类型、尺寸和步长顺序创建对象。

        如果指定了步长顺序，则使用 `fill_ordered` 方法填充；
        否则使用 `contiguous_strides` 方法。
        """
        if stride_order:
            strides = FlexibleLayout.fill_ordered(size, stride_order)
        else:
            strides = FlexibleLayout.contiguous_strides(size)
        super().__init__(device, dtype, size, strides)
class NonOwningLayout(Layout):
    """Is a view into the storage of another tensor"""

    def __init__(self, view: Union[BaseView, TensorBox]):
        # 获取视图的布局信息
        layout = view.get_layout()
        # 调用父类构造函数，传入布局相关信息
        super().__init__(
            layout.device,
            layout.dtype,
            layout.size,
            layout.stride,
        )
        # 保存视图对象
        self.view = view

    def make_indexer(self):
        # 将当前对象转换为固定布局后调用其索引生成器
        return self.as_fixed().make_indexer()

    def maybe_guard_aligned(self):
        # 获取视图的偏移量
        offset = self.view.get_layout().offset
        # 如果偏移量为0，则返回True
        if offset == 0:
            return True
        # 否则从.compile_fx模块导入ALIGNMENT常量，并检查偏移量是否为ALIGNMENT的静态倍数
        from .compile_fx import ALIGNMENT

        return V.graph.sizevars.statically_known_multiple_of(offset, ALIGNMENT)  # type: ignore[arg-type]


class NoneLayout(IRNode):
    # This is janky, I figured out what fields to populate by just running
    # the model I was interested in and adding properties/methods as needed.
    # This doesn't inherit from Layout because Layout assumes you have stuff
    # like sizes, but I don't really have anything here.
    #
    # If you have an ir.Node with NoneLayout, you probably need to setup
    # dependencies manually in scheduler

    def __init__(self, device):
        # 初始化设备属性和空的尺寸、步幅
        self.device = device
        self.size = [0]
        self.stride = [0]

    def storage_size(self):
        # 返回存储大小为0
        return 0

    def as_fixed(self):
        # 返回自身，因为没有尺寸和步幅等概念
        return self


class MutationLayoutSHOULDREMOVE(Layout):
    def __init__(self, target: IRNode):
        # 调用父类构造函数，传入目标的设备、数据类型、大小和空的步幅
        super().__init__(
            target.get_device(),
            target.get_dtype(),
            target.get_size(),
            None,
        )
        # 保存目标对象
        self.target = target
        # 获取关联缓冲区的名称，并标记为已变异
        name = self.get_buffer().get_name()
        V.graph.mark_buffer_mutated(name)

    @Layout.stride.getter  # type: ignore[attr-defined]
    def stride(self):
        # 获取实际布局对象的步幅
        return self.real_layout().stride

    def storage_size(self) -> sympy.Expr:
        # 返回实际布局对象的存储大小
        return self.real_layout().storage_size()

    def get_buffer(self) -> Buffer:
        # 递归展开视图，直到找到关联的缓冲区对象
        def unwrap_views(target):
            if isinstance(target, MutationLayoutSHOULDREMOVE):
                return unwrap_views(target.target)
            if isinstance(target, BaseView):
                return unwrap_views(target.unwrap_view())
            if isinstance(target, MutableBox):
                return unwrap_views(target.data)
            return target

        result = unwrap_views(self.target)
        # 确保结果是一个缓冲区对象
        assert isinstance(
            result, Buffer
        ), "MutationLayoutSHOULDREMOVE must refer to a buffer"
        return result

    def real_layout(self):
        # 获取关联缓冲区的实际布局对象
        return self.get_buffer().layout

    @classmethod
    # 将目标对象 dst 实现（materialize）
    dst.realize()

    # 注意：在实现 src 之前，我们必须先实现（materialize）dst 的用户，
    # 因为实现顺序决定调度顺序。否则，src 的变更会在现有 dst 用户之前调度！
    V.graph.mark_buffer_mutated(dst.get_name())

    # 如果 src 是 TensorBox 的实例，则使用其数据作为 src
    if isinstance(src, TensorBox):
        src = src.data

    # 将 src 的内容复制到 dst 中。在大多数情况下，调度器应该将其融合为单个内核。
    # 注意：我们不能更改 src 的布局来直接修改 dst，因为这会将 src 别名为 dst，
    # 这是不正确的，因为对 dst 的进一步变更会影响到 src 的用户。但是，如果没有
    # 更多的 dst 用户，我们可以将 src 别名为 dst。
    src.realize_hint()

    # 如果不允许不安全的别名（unsafe_alias），则创建 Pointwise 对象，用于创建
    # 一个新的数据副本，保留了原始数据的设备、数据类型及其它属性，但其形状与 dst 的
    # 形状相匹配。
    if not unsafe_alias:
        src = Pointwise.create(
            device=src.get_device(),
            dtype=src.get_dtype(),
            inner_fn=src.make_loader(),
            ranges=[
                V.graph.sizevars.guard_equals(a, b)
                for a, b in zip(src.get_size(), dst.get_size())
            ],
        ).data

    # 实现 src
    src.realize()

    # 断言：确保 src 的数据布局是 FlexibleLayout
    assert isinstance(src.data.layout, FlexibleLayout)

    # 设置 src 的数据布局为 MutationLayoutSHOULDREMOVE(dst)，意图移除 dst 的变更布局
    src.data.layout = MutationLayoutSHOULDREMOVE(dst)

    # 返回 src 的数据
    return src.data
@dataclasses.dataclass
class Buffer(IRNode):
    # Name is sometimes None; e.g., ForceInPlace, where there isn't
    # a meaningful name
    # 缓冲区的名称，有时可能为 None，例如 ForceInPlace 中没有有意义的名称
    name: Optional[str]
    layout: Layout

    # Multi-output buffers will define 'outputs: List[Buffer]'. Confusingly,
    # MultiOutput does NOT define this!
    # 多输出缓冲区将定义 'outputs: List[Buffer]'。令人困惑的是，MultiOutput 并不定义这个属性！

    def __post_init__(self):
        super().__post_init__()
        # Initialize origin_node to None after post initialization
        # 在后期初始化后将 origin_node 初始化为 None
        self.origin_node = None

    def make_indexer(self):
        # Delegate to layout to create an indexer
        # 委托给 layout 创建一个索引器
        return self.layout.make_indexer()

    def get_name(self) -> str:
        assert self.name, self
        # Return the name of the buffer; assertion ensures name is not None
        # 返回缓冲区的名称；断言确保名称不是 None
        return self.name

    def get_device(self):
        # Return the device associated with the layout
        # 返回与布局关联的设备
        return self.layout.device

    def get_origin_node(self):
        # Return the origin_node attribute
        # 返回 origin_node 属性
        return self.origin_node

    @property
    def dtype(self):
        # Return the data type of the buffer from layout if available
        # 如果可用，从布局中返回缓冲区的数据类型
        return getattr(self.layout, "dtype", None)

    def get_size(self):
        # Return the size of the buffer as a list
        # 返回缓冲区的大小作为列表
        return list(self.layout.size)

    def get_stride(self):
        # Return the stride of the buffer as a list
        # 返回缓冲区的步长作为列表
        return list(self.layout.stride)

    def get_offset(self):
        # Return the offset of the buffer from layout
        # 从布局中返回缓冲区的偏移量
        return self.layout.offset

    def get_layout(self):
        # Return the layout object associated with the buffer
        # 返回与缓冲区关联的布局对象
        return self.layout

    def get_storage_numel(self):
        # Return the number of elements in the buffer
        # 返回缓冲区中的元素数
        return self.get_numel()

    def is_extern(self):
        # Buffers are not external by default
        # 缓冲区默认不是外部的
        return False

    def freeze_layout(self):
        if not isinstance(self.layout, (MultiOutputLayout, NonOwningLayout)):
            # Convert the layout to a fixed layout if it's not already fixed
            # 如果布局不是固定的，则将其转换为固定布局
            self.layout = self.layout.as_fixed()

    def freeze_layout_with_stride_order(self, order, allow_padding=False):
        assert isinstance(self.layout, FlexibleLayout)
        # Convert the layout to a stride-ordered layout with optional padding
        # 将布局转换为按步长排序的布局，可选填充
        self.layout = self.layout.as_stride_order(order, allow_padding=allow_padding)

    def freeze_layout_with_fill_order(self, order):
        assert isinstance(self.layout, FlexibleLayout)
        # Convert the layout to a fill-ordered layout
        # 将布局转换为按填充顺序排序的布局
        self.layout = self.layout.as_fill_order(order)

    def freeze_layout_with_same_order(self, stride):
        assert isinstance(self.layout, FlexibleLayout)
        # Convert the layout to have the same order as specified stride
        # 将布局转换为与指定步长相同顺序的布局
        self.layout = self.layout.as_same_order(stride)

    def is_zero_elements(self):
        return V.graph.sizevars.is_expr_static_and_true(sympy.Eq(self.get_numel(), 0))  # type: ignore[arg-type]
        # Check if the buffer has zero elements using symbolic expression evaluation
        # 使用符号表达式评估检查缓冲区是否有零元素

    def make_loader(self):
        # Loading from a zero-element buffer is a no-op
        if self.is_zero_elements():
            # Return a no-operation loader function for zero-element buffers
            # 对于零元素缓冲区，返回一个空操作的加载函数
            return partial(nop_loader_fn, dtype=self.get_dtype())

        def loader(index):
            # Create a loader function that loads data from the buffer using indexer
            # 创建一个加载器函数，使用索引器从缓冲区加载数据
            indexer = self.layout.make_indexer()
            return ops.load(self.name, indexer(index))

        return loader

    def is_no_op(self):
        # Buffers are not considered as no-operation by default
        # 缓冲区默认不被视为无操作
        return False

    def codegen_reference(self, writer=None):
        # Generate code reference for the buffer, optionally using a writer
        # 生成缓冲区的代码引用，可选择使用写入器
        return self.get_name()

    def decide_layout(self):
        # Placeholder method to decide on the layout strategy; currently does nothing
        # 决定布局策略的占位符方法；当前什么也不做
        pass

    def get_inputs_that_alias_output(self):
        if isinstance(self.layout, NonOwningLayout):
            # Return the name of the view that aliases the output if layout is NonOwningLayout
            # 如果布局是 NonOwningLayout，则返回别名输出的视图名称
            return [self.layout.view.get_name()]
        return ()
        # Return an empty tuple if no input aliases the output
        # 如果没有输入与输出别名，则返回空元组

    def get_mutation_names(self):
        if isinstance(self.layout, MutationLayoutSHOULDREMOVE):
            # Return the name of the target for mutation if layout is MutationLayoutSHOULDREMOVE
            # 如果布局是 MutationLayoutSHOULDREMOVE，则返回变异目标的名称
            return [self.layout.target.get_name()]
        return ()
        # Return an empty tuple if no mutation names are defined
        # 如果没有定义变异名称，则返回空元组
    # 使用 `patch.object` 来设置 `FlexibleLayout` 类的 `allow_indexing` 属性为 `True`
    with patch.object(FlexibleLayout, "allow_indexing", True):
        # 调用 `extract_read_writes` 函数，传入 `make_loader()` 和 `get_size()` 方法的结果作为参数，
        # 并返回其结果
        return extract_read_writes(
            self.make_loader(),
            self.get_size(),
        )

    # 返回 `get_read_writes()` 方法返回结果的 `reads` 属性
    return self.get_read_writes().reads

    # 返回一个空集合，表示没有未支持的符号定义
    return set()

    # 返回一个空集合，表示没有未支持的符号使用
    return set()

    # 空方法，表示没有实现具体的实现逻辑
    pass

    # 返回额外的全局内存大小，通常为 0，表示不需要额外的内存
    return 0

    # 默认情况下返回 False，表示不应该分配额外的资源
    return False
class InputBuffer(Buffer):
    pass

# InputBuffer 类继承自 Buffer 类，表示输入缓冲区。

class ConstantBuffer(InputBuffer):
    override_device: Optional[torch.device] = None

    def make_loader(self):
        # 定义内部函数 loader，接受一个索引参数 index
        def loader(index):
            # 创建索引器对象
            indexer = self.layout.make_indexer()
            # 调用 ops.load 函数，加载常量数据
            return ops.load(
                V.graph.constant_name(self.get_name(), self.override_device),
                indexer(index),
            )

        # 返回内部函数 loader
        return loader

    def constant_to_device(self, device):
        # 返回一个新的 ConstantBuffer 对象，用于特定设备上的常量加载
        return ConstantBuffer(
            V.graph.constant_name(self.get_name(), device), self.layout
        )


class NoneAsConstantBuffer(IRNode):
    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        # 返回一个空集合，表示未支持的符号使用
        return set()

    def codegen_reference(self, writer=None):
        # 返回 V.graph.wrapper_code.none_str，表示对应的代码生成器输出为 none_str
        return V.graph.wrapper_code.none_str


class ShapeAsConstantBuffer(IRNode):
    def __init__(self, shape):
        super().__init__()
        self._shape = shape

    @property
    def shape(self):
        # 返回保存的形状数据
        return self._shape

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        # 返回与形状相关的未支持符号使用集合
        return free_unbacked_symbols(self.shape)

    def codegen_reference(self, writer=None):
        # 调用 V.graph.sizevars.simplify 函数简化形状表达式后，返回其对应的代码生成器输出
        return V.graph.wrapper_code.expr_printer(V.graph.sizevars.simplify(self.shape))


@dataclasses.dataclass
class ComputedBuffer(Buffer):
    data: Loops

    def get_computed_buffer_name(self):
        """
        Returns self.name if it exists, otherwise returns the name of the data node if that exists.
        If neither exist, returns None.
        """
        # 如果 self.name 存在，则返回 self.name
        if self.name is not None:
            return self.name
        # 如果 self.data 具有 "name" 属性，则返回 self.data.name
        if hasattr(self.data, "name"):
            return self.data.name
        # 否则返回 None
        return None

    @cache_on_self
    def num_reads(self):
        # 返回读取操作的数量，通过获取读写操作的 reads 成员的长度实现
        return len(self.get_read_writes().reads)

    def get_read_writes(self):
        # 使用 patch.object 临时允许 FlexibleLayout 的索引操作
        with patch.object(FlexibleLayout, "allow_indexing", True):
            # 如果 self.data 具有减少类型，则使用其点对点大小和减少大小进行读写操作提取
            if self.data.get_reduction_type():
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_pointwise_size(),
                    self.data.get_reduction_size(),
                )
            # 否则，使用其大小进行读写操作提取
            else:
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_size(),
                )
    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        # 通常情况下，我们希望只查看参数列表，
        # 但是ComputedBuffers没有参数列表。
        #
        # 从道义上讲，这段逻辑需要与KernelArgs.size调用同步，
        # 后者负责将符号作为内核参数传递（确切地说，传递其中一个符号会建立一个依赖关系）。
        # 然而，我们还没有开始代码生成，所以无法直接重用那段逻辑。
        #
        # 目前，我只是随意地使用缓冲区的大小。不确定是否足够。
        #
        # 你可能会想这是否足够用于表示一个关于i0的ComputedBuffer的约简操作。
        # 从经验上看，足够了，但原因不寻常：我们只需要对item()调用有准确的依赖关系，
        # 但是如果没有常规的非约简缓冲区，就不可能从item()调用中得到关于i0的约简。
        return (
            free_unbacked_symbols(self.get_size())  # 获取大小中的自由未支持符号
            | free_unbacked_symbols(self.get_stride())  # 获取步幅中的自由未支持符号
            | free_unbacked_symbols(self.get_offset())  # 获取偏移中的自由未支持符号
            | self.data.get_unbacked_symbol_uses()  # 获取数据中的自由未支持符号使用情况
        )

    def make_loader(self):
        # 内联常量和索引表达式
        if (
            hasattr(self.data, "make_loader")  # 如果self.data具有"make_loader"属性
            and self.name not in V.graph.mutated_buffers  # 并且self.name不在V.graph.mutated_buffers中
            and self.num_reads() == 0  # 并且self.num_reads()为0
        ):
            # 可以内联
            return self.data.make_loader()  # 调用self.data的make_loader方法
        return super().make_loader()  # 否则调用父类的make_loader方法

    def get_store_function(self):
        indexer = self.layout.as_fixed().make_indexer()  # 获取索引器，使用layout的固定版本生成
        if isinstance(self.data, (Reduction, Scan, Sort)):  # 如果self.data是Reduction、Scan或Sort的实例
            return partial(self.data.store_reduction, self.name, indexer)  # 返回self.data的store_reduction方法的偏函数
        else:
            assert isinstance(self.data, Pointwise)  # 否则，断言self.data是Pointwise的实例
            return partial(self.data.store_output, self.name, indexer)  # 返回self.data的store_output方法的偏函数
    def get_fill_order(self):
        """
        如果布局仍然灵活，尝试根据读取的步长顺序确定填充顺序。

        TODO(jansel): 这里可以使用更好的算法，考虑这个值的下游使用者，并尝试进行全局图级布局优化。
                      这也是一个很好的自动调优的对象。
        """
        if isinstance(self.layout, FlexibleLayout):
            # 获取索引变量和约简变量，并从中获取索引变量和约简变量的压缩
            (index_vars, reduction_vars), _ = dependencies.index_vars_squeeze(
                self.data.get_pointwise_size(), self.data.get_reduction_size()
            )
            # 获取所有的读取操作
            reads = self.get_read_writes().reads
            # 获取读取操作对应的缓冲区列表
            reads_bufs = [
                V.graph.name_to_buffer[r.name]
                if r.name in V.graph.name_to_buffer.keys()
                else None
                for r in reads
            ]
            # 只考虑大小相同的读取到缓冲区的情况
            # 忽略 StarDeps 因为它们不提供步长信息
            assert all(
                isinstance(r, (dependencies.StarDep, dependencies.MemoryDep))
                for r in reads
            )
            # 对于内存依赖的读取操作，进行符号替换
            reads = [
                sympy_subs(
                    r.index, {v: sympy.Integer(0) for v in reduction_vars if v != 0}
                )
                for r in reads
                if isinstance(r, dependencies.MemoryDep)
            ]

            if reads:
                # 如果有读取操作，根据具体的数据类型重新索引或者排序
                if isinstance(self.data, (Scan, Sort)):
                    indices = self.data.reindex(index_vars, reduction_vars)
                else:
                    indices = index_vars
                # 计算每个读取操作的步长提示
                stride_lengths = [
                    V.graph.sizevars.stride_hints(expr, indices) for expr in reads  # type: ignore[arg-type]
                ]
                from .scheduler import pick_loop_order

                # 选择循环顺序
                return pick_loop_order(stride_lengths, self.get_size())

        # 如果没有填充顺序可确定，则返回 None
        return None

    def decide_layout(self):
        if isinstance(self.layout, FlexibleLayout):
            # 获取填充顺序
            order = self.get_fill_order()
            if order:
                # 使用给定的填充顺序冻结布局
                self.freeze_layout_with_fill_order(order)
            else:
                # 如果没有可用的填充顺序，则简单冻结布局
                self.freeze_layout()

    @cache_on_self
    def get_default_sizes_body(self):
        # 获取点式大小和约简大小作为参数，以及它们的变量范围
        args, var_ranges = dependencies.index_vars_squeeze(
            self.data.get_pointwise_size(), self.data.get_reduction_size(), prefix="q"
        )
        # 使用当前对象的设备进行常量缓冲区的设备重写
        with patch.object(ConstantBuffer, "override_device", self.get_device()):
            # 创建循环体对象，使用存储函数和参数进行初始化
            body = LoopBody(
                self.get_store_function(),
                (args if self.get_reduction_type() else args[:1]),
                var_ranges,
            )
        index_vars = []
        reduce_vars: List[Any] = []
        index_size = []
        reduce_size = []
        # 根据变量范围将变量分类为索引变量和约简变量
        for v, s in var_ranges.items():
            if v in args[0]:
                assert not reduce_vars
                index_vars.append(v)
                index_size.append(s)
            else:
                assert v in args[1]
                reduce_vars.append(v)
                reduce_size.append(s)
        # 返回索引大小、约简大小、循环体对象以及索引变量和约简变量
        return (index_size, reduce_size), body, (index_vars, reduce_vars)

    def simplify_and_reorder(
        self,
        extra_indexing_constraints: Optional[Tuple[Dict[Any, Any], List[Any]]] = None,
    ):
        """
        Shuffle the order of loops around to hopefully improve performance.
        """
        # 导入循环调度器，用于选择循环顺序
        from .scheduler import pick_loop_order

        # 如果未提供优先索引，则设置为空列表
        if priority_idx is None:
            priority_idx = []

        try:
            # 计算内存地址的步长提示，用于决定循环顺序
            strides = [
                V.graph.sizevars.stride_hints(expr, index_vars, support_vars)
                for expr in memory_addrs
            ]
            # 确保步长和内存地址的长度一致，并且每个步长都有对应的索引变量
            assert len(strides) == len(memory_addrs) and len(strides[0]) == len(
                index_vars
            )
            # 选择最佳的循环顺序，并倒序排列
            order = list(reversed(pick_loop_order(strides, sizes, priority_idx)))
        except Exception:
            # 如果出现异常，并且处于调试模式，则记录警告日志
            if config.debug:
                log.warning(
                    "Did not simplify complex index:\n%s\n%s",
                    dict(zip(index_vars, sizes)),
                    memory_addrs,
                )
            # 否则，按默认顺序生成索引
            order = list(range(len(sizes)))
        # 根据选择的顺序重新排序大小，并生成相同顺序和逆序的重排列表
        sizes = [sizes[i] for i in order]
        return sizes, same_reorder(order), inverse_reorder(order)

    def get_reduction_size(self):
        # 返回当前数据对象的约简大小
        return self.data.get_reduction_size()

    def get_reduction_type(self):
        # 返回当前数据对象的约简类型
        return self.data.get_reduction_type()

    def is_no_op(self):
        # 检查当前数据对象是否具有零元素
        return self.data.is_zero_elements()

    def should_allocate(self):
        # 始终返回True，表示始终要分配资源
        return True

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        # 将数据对象移动到指定的设备上，要求所有读取都是对常量的读取
        return self.data.constant_to_device(device)
class TemplateBuffer(Buffer):
    """
    Represents a Triton (in the future other type) of template operator
    that we can fuse an epilogue onto.
    """

    def __init__(self, layout, inputs, make_kernel_render):
        # 调用父类构造函数初始化模板缓冲区
        super().__init__(name=None, layout=layout)
        # 解包输入并将其存储为成员变量
        self.inputs = InputsKernel.unwrap_storage(inputs)
        # 存储 make_kernel_render 用于后续使用
        self.make_kernel_render = make_kernel_render
        # 在图形注册缓冲区，并存储其名称
        self.name = V.graph.register_buffer(self)

    def get_read_writes(self):
        # 返回标准化后的读写依赖关系
        return self.normalized_read_writes()

    def normalized_read_writes(self):
        # 获取模板名称和索引器
        name = self.get_name()
        indexer = self.layout.make_indexer()

        def dummy(index, rindex):
            # 断言读索引为空，返回虚拟的写操作
            assert len(rindex) == 0
            return ops.store(name, indexer(index), "fake")

        # 提取标准化的读写依赖关系
        deps = dependencies.extract_read_writes(
            dummy, self.get_size(), (), normalize=True
        )
        # 将输入标记为读取依赖
        deps.reads = {dependencies.StarDep(x.get_name()) for x in self.inputs}
        return deps

    def get_reduction_size(self):
        # 返回减少的大小
        return 1

    def get_reduction_type(self):
        # 返回减少的类型
        return None

    def is_no_op(self):
        # 检查是否为无操作
        return False

    def should_allocate(self):
        # 确定是否应该分配资源
        return True

    def simplify_and_reorder(
        self,
        extra_indexing_constraints: Optional[Tuple[Dict[Any, Any], List[Any]]] = None,
    ):
        # 简化和重新排序操作
        return (
            (
                self.get_size(),
                (),
            ),
            None,
        )


class TritonTemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        layout,
        inputs,
        make_kernel_render,
        debug_extra=None,
        mutated_inputs: Optional[Iterable[IRNode]] = None,
    ):
        """
        NOTE:[TritonTemplates with multiple outputs]
        We want the ability for TritonTemplates to output multiple tensors. Triton
        kernels have no notion of outputs and this is done by creating tensors that
        are then mutated by the kernel. Currenlty our STORE_OUTPUT codegen doesn't
        support creating multinode outputs for triton templates.
        We work around this by creating an extra input buffer during the lowering
        and we mark them as mutated inputs.
        """
        # 调用父类构造函数初始化 Triton 模板缓冲区
        super().__init__(layout, inputs, make_kernel_render)
        # 存储调试额外信息
        self.debug_extra = debug_extra
        # 存储被突变输入
        self.mutated_inputs = mutated_inputs
        if mutated_inputs is not None:
            # 确保被突变输入仅允许特定节点使用
            allowed_set = {
                torch.ops.higher_order.flex_attention,
                torch.ops.higher_order.flex_attention_backward,
            }
            current_node = V.graph.current_node.target
            # 断言当前节点在允许集合中
            assert (
                current_node in allowed_set
            ), f"Mutated inputs are only allowed for {allowed_set} but got {current_node}"
            # 标记节点为变异输入
            mark_node_as_mutating(self, *mutated_inputs)
    # 定义一个特殊方法 `__str__()`，用于返回对象的字符串表示形式
    def __str__(self):
        # 创建一个字符串 `out`，包含对象的布局信息和调试额外信息
        out = f"TritonTemplateBuffer(layout={self.layout}, {self.debug_extra})"
        # 返回字符串表示形式
        return out
PrimitiveInfoType = Union[int, float, bool, str, List[Union[int, str, float, bool]]]
# 定义一个类型别名 PrimitiveInfoType，可以是 int、float、bool、str 或它们的列表

class ChoiceCaller:
    """
    Represents a possible choice used in autotune_process.py.
    During autotuning, self.benchmark() is first called to get benchmark result,
    and if this choice is selected, self.output_node() is called to get the output_node.

    Children classes: TritonTemplateCaller, CUDATemplateCaller.
    """
    
    def __init__(self, name, input_nodes, layout):
        super().__init__()
        self.name = name
        self.layout = layout
        self.input_nodes = input_nodes
        # 初始化 ChoiceCaller 对象的名称、布局和输入节点列表

    def benchmark(self, *args, out) -> float:
        algo = self.to_callable()
        return do_bench(algo, args, {"out": out})
        # 使用 self.to_callable() 转换为算法，执行基准测试，并返回结果

    def call_name(self) -> str:
        raise NotImplementedError
        # 抽象方法，返回调用名称的字符串，子类需要实现

    def to_callable(self):
        raise NotImplementedError
        # 抽象方法，返回可调用对象，子类需要实现

    def hash_key(self) -> str:
        raise NotImplementedError
        # 抽象方法，返回哈希键的字符串表示，子类需要实现

    def output_node(self) -> TensorBox:
        raise NotImplementedError
        # 抽象方法，返回输出节点的 TensorBox 对象，子类需要实现

    def info_dict(self) -> Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        return {}
        # 返回一个空字典，记录到自动调优日志文件中的信息

class TritonTemplateCallerBase(ChoiceCaller):
    def get_make_kernel_render(self) -> Any:
        raise NotImplementedError
        # 抽象方法，返回任意类型的对象，子类需要实现

class MultiTemplateBuffer(TritonTemplateBuffer):
    """
    Represents a Buffer with multiple backing implementation choices.

    Choices can be TritonTemplates or ExternKernels. During scheduling if there is a potential
    epilogue we will benchmark each of the choices with the epilogue to determine an implementation.
    Otherwise, the fastest base choice will be chosen.
    """

    def __init__(
        self,
        layout: Layout,
        inputs: List[IRNode],
        choice_timings: Callable[[], Dict[ChoiceCaller, float]],
    ):
        super().__init__(layout=layout, inputs=inputs, make_kernel_render=None)
        self._choice_timings_fn = choice_timings
        self._choice_timings: Optional[Dict[ChoiceCaller, float]] = None
        self.original_inputs = inputs
        # 初始化 MultiTemplateBuffer 对象的布局、输入节点列表和选择计时的回调函数

    @property
    def choice_timings(self) -> Dict[ChoiceCaller, float]:
        if self._choice_timings is None:
            self._choice_timings = self._choice_timings_fn()
        return self._choice_timings
        # 返回选择调度时的时间字典，如果为空则通过回调函数获取

    @contextlib.contextmanager
    def swap_as_triton_caller(self, caller: TritonTemplateCallerBase):
        assert isinstance(caller, torch._inductor.select_algorithm.TritonTemplateCaller)
        assert self.layout == caller.layout

        render = self.make_kernel_render
        self.make_kernel_render = caller.get_make_kernel_render()
        try:
            yield
        finally:
            self.make_kernel_render = render
        # 上下文管理器，用于将 MultiTemplateBuffer 对象作为 TritonTemplateCallerBase 对象调用
    # 将当前对象的状态设置为 Triton 调用者的最终状态
    def finalize_as_triton_caller(self, caller: TritonTemplateCallerBase):
        # 断言调用者对象是 TritonTemplateCaller 的实例
        assert isinstance(caller, torch._inductor.select_algorithm.TritonTemplateCaller)
        # 断言当前对象的布局大小与调用者的布局大小相同
        assert self.layout.size == caller.layout.size
        # 断言当前对象的布局步长与调用者的布局步长相同
        assert self.layout.stride == caller.layout.stride
        # 获取调用者的内核渲染方法，并将其设置为当前对象的内核渲染方法
        self.make_kernel_render = caller.get_make_kernel_render()

    # 获取选择时间列表中时间最小的选择调用者及其时间
    def get_min_choice(self) -> Tuple[ChoiceCaller, float]:
        # 从选择时间字典中找到值最小的键（即时间最小的选择调用者）
        min_choice = min(self.choice_timings, key=self.choice_timings.get)  # type: ignore[arg-type]
        # 返回时间最小的选择调用者及其对应的时间
        return (min_choice, self.choice_timings[min_choice])
class CUDATemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        layout,
        inputs,
        make_kernel_render,
        workspace_size: int,
        template: CUDATemplate,  # type: ignore[name-defined]  # noqa: F821
    ):
        # 调用父类的构造函数，初始化布局、输入、渲染函数
        super().__init__(layout, inputs, make_kernel_render)
        # 设置模板需要的全局内存大小（字节数）
        self.workspace_size = workspace_size
        # 设置 CUDA 模板
        self.template = template

    def get_workspace_size(self):
        # 返回模板需要的全局内存大小，如果未设置则返回 0
        return self.workspace_size if self.workspace_size is not None else 0


class CppTemplateBuffer(TemplateBuffer):
    def __init__(self, layout, inputs, make_kernel_render, template, choice):
        # 调用父类的构造函数，初始化布局、输入、渲染函数
        super().__init__(layout, inputs, make_kernel_render)
        # 设置 C++ 模板和选择
        self.template = template
        self.choice = choice


@dataclasses.dataclass
class InputsKernel(Buffer):
    inputs: List[Buffer]

    def get_read_writes_input(self, x):
        # 返回输入对象 x 的读写依赖
        return dependencies.StarDep(x.get_name())

    def get_read_writes(self):
        # 获取所有输入的读写依赖
        star_dep = []
        for input in self.inputs:
            if isinstance(input, list):
                star_dep.extend([self.get_read_writes_input(x) for x in input])
            else:
                star_dep.append(self.get_read_writes_input(input))

        return dependencies.ReadWrites(
            set(star_dep),
            {dependencies.StarDep(self.get_name())},
            set(),
            [],
            None,
            op_counts=collections.Counter(),
        )

    @classmethod
    def unwrap_storage_for_input(cls, x):
        # 根据输入 x 类型，递归解开其存储封装，直到获得底层对象
        if isinstance(x, TensorBox):
            x = x.data
        if isinstance(x, StorageBox):
            x = x.data
        if isinstance(x, BaseView) and not isinstance(x, ReinterpretView):
            x = ExternKernel.realize_input(x)
        if isinstance(x, TensorBox):
            # 当在上述 realize_input 调用中转换为 ReinterpretView 失败时，
            # 结果会作为 TensorBox / StorageBox 对被包装，因此需要递归解包
            return cls.unwrap_storage_for_input(x)
        if isinstance(x, TorchBindObject):
            return x
        assert isinstance(x, (Buffer, ReinterpretView)), x
        return x

    @staticmethod
    def unwrap_storage(inputs):
        # 解开输入列表中的所有存储封装
        inputs_new = []
        for x in inputs:
            if isinstance(x, list):
                x = [InputsKernel.unwrap_storage_for_input(i) for i in x]
            else:
                x = InputsKernel.unwrap_storage_for_input(x)
            inputs_new.append(x)
        return inputs_new

    def is_extern(self):
        # 标识此内核是否为外部内核
        return True


class NopKernel(InputsKernel):
    def is_no_op(self):
        # 标识此内核是否为无操作内核
        return True


class ConcatKernel(NopKernel):
    """
    There isn't actually a real kernel for concat, we just change the
    storage for the upstream data.
    """

    @classmethod
    @classmethod  # 这里可能是多余的修饰符，需要确认是否是错误
    def can_realize_into_without_copy(cls, src):
        # 检查是否是 TensorBox 类型的对象
        if isinstance(src, TensorBox):
            # 如果是 TensorBox，解包并递归调用 can_realize_into_without_copy
            return cls.can_realize_into_without_copy(src.data)

        # 检查 src.data 是否具有 FlexibleLayout 并且不是 ExternKernelAlloc 类型
        return isinstance(src.data.layout, FlexibleLayout) and not isinstance(
            src.data, ExternKernelAlloc
        )

    @classmethod
    def realize_into(cls, src, dst):
        # 尝试将 src 转换为 ReinterpretView，而不是抛出异常。
        # 这在布局方面做了些让步，因为 as_storage_and_layout
        # 可能会使我们从灵活布局转换为固定布局。
        if not isinstance(dst, ReinterpretView):
            if is_storage_and_layout(dst):
                storage, layout = as_storage_and_layout(dst)
                dst = ReinterpretView(storage, layout)
        assert isinstance(dst, ReinterpretView), dst

        # 如果 src 是 TensorBox 类型的对象，则解包并递归调用 realize_into
        if isinstance(src, TensorBox):
            return cls.realize_into(src.data, dst)

        # 如果 src 是 StorageBox 类型的对象
        if isinstance(src, StorageBox):
            src.realize()  # 实现 StorageBox 中的数据
            # ExternKernelAlloc 对输出布局有特定要求，应创建一个副本
            assert hasattr(src.data, "layout")
            if cls.can_realize_into_without_copy(src):
                src.data.layout = NonOwningLayout(dst)
                return src.data

        # 引入一个复制操作
        pw = Pointwise.create(
            device=src.get_device(),
            dtype=src.get_dtype(),
            inner_fn=src.make_loader(),
            ranges=[
                V.graph.sizevars.guard_equals(a, b)
                for a, b in zip(src.get_size(), dst.get_size())
            ],
        )
        return cls.realize_into(pw, dst)

    def should_allocate(self):
        # 始终返回 True 表示应分配资源
        return True
# 定义一个函数，用于从给定的 torch._ops.OpOverload 对象中获取对应的 ATen C++ 内核名称
def get_aten_cpp_kernel_name(kernel):
    # 如果 kernel 不是 torch._ops.OpOverload 的实例，或者其命名空间不是 "aten"，则返回 None
    if not isinstance(kernel, torch._ops.OpOverload) or kernel.namespace != "aten":
        return None
    # 获取操作符名称，处理默认重载名称的情况，将 "." 替换为 "_" 以适配 C++ 命名约定
    opname = (
        kernel.__name__.split(".")[0]
        if kernel._overloadname == "default"
        else kernel.__name__.replace(".", "_")
    )
    # 构造并返回 ATen C++ 内核函数的调用名称
    return f"at::_ops::{opname}::call"


# 定义一个数据类 ExternKernel，继承自 InputsKernel
@dataclasses.dataclass
class ExternKernel(InputsKernel):
    # 定义常量参数元组 constant_args，默认为空元组
    constant_args: Tuple[Any, ...] = ()
    # 定义关键字参数字典 kwargs，默认为空字典
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # 定义输出视图 output_view，默认为 None
    output_view: Optional[ReinterpretView] = None
    # 定义 Python 内核名称 python_kernel_name，默认为 None
    python_kernel_name: Optional[str] = None
    # 定义 C++ 内核名称 cpp_kernel_name，默认为 None
    cpp_kernel_name: Optional[str] = None
    # FIXME: 在某些情况下，我们仍需要显式传入 ordered_kwargs_for_cpp_kernel
    # 我们不应该这样做，因为这些信息可以从 op_overload._schema 中获取。
    # 定义用于 C++ 内核的有序关键字参数列表 ordered_kwargs_for_cpp_kernel，默认为空列表
    ordered_kwargs_for_cpp_kernel: Iterable[str] = dataclasses.field(
        default_factory=list
    )
    # 定义操作重载 op_overload，默认为 None
    op_overload: Optional[
        Union[torch._ops.OpOverload, torch._ops.HigherOrderOperator]
    ] = None
    # 定义参数属性列表 arg_properties，默认为 None
    arg_properties: Optional[List[Dict[str, Any]]] = None
    # 定义关键字参数属性字典 kwarg_properties，默认为 None
    kwarg_properties: Optional[Dict[str, Dict[str, Any]]] = None
    # 定义未支持的绑定字典 unbacked_bindings，默认为空字典
    unbacked_bindings: Dict[sympy.Symbol, pytree.KeyPath] = dataclasses.field(
        default_factory=dict
    )

    # 初始化方法，接受多个参数，并调用父类的初始化方法
    def __init__(
        self,
        name,
        layout,
        inputs,
        constant_args=(),
        kwargs=None,
        output_view=None,
        python_kernel_name=None,
        cpp_kernel_name=None,
        ordered_kwargs_for_cpp_kernel=(),
        op_overload=None,
    ):
        super().__init__(
            name,
            layout,
            inputs,
        )
        # 设置常量参数
        self.constant_args = constant_args
        # 设置关键字参数，如果未提供则为空字典
        self.kwargs = kwargs if kwargs else {}
        # 设置输出视图
        self.output_view = output_view
        # 设置 Python 内核名称
        self.python_kernel_name = python_kernel_name
        # 如果 cpp_kernel_name 为 None，则尝试从 op_overload 中构造它
        self.cpp_kernel_name = cpp_kernel_name or get_aten_cpp_kernel_name(op_overload)
        # 设置用于 C++ 内核的有序关键字参数列表
        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel
        # 设置操作重载对象
        self.op_overload = op_overload
        # 收集参数和关键字参数属性
        self.collect_arg_kwarg_properties()
        # 初始化未支持的绑定字典
        self.unbacked_bindings = {}
        # 获取当前节点的 FX 图节点
        self.fx_node = V.graph.current_node

    # 定义一个方法，返回未支持的符号定义的集合
    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()
    def collect_arg_kwarg_properties(self):
        # 如果 self.op_overload 是 torch._ops.OpOverload 类型，可以使用其 schema 收集参数和关键字参数的额外信息，
        # 例如类型和默认值，以帮助生成 CPP 包装器代码
        self.arg_properties = (
            [
                {
                    "name": x.name,
                    "type": x.real_type,
                    "default_value": x.default_value,
                }
                for x in self.op_overload._schema.arguments
                if not x.kwarg_only
            ]
            if isinstance(self.op_overload, torch._ops.OpOverload)
            else [{} for i in range(len(self.inputs))]
        )
        # 如果 self.op_overload 是 torch._ops.OpOverload 类型，则使用其 schema 收集所有参数的信息
        self.allarg_properties = (
            {
                x.name: {"type": x.real_type, "default_value": x.default_value}
                for x in self.op_overload._schema.arguments
            }
            if isinstance(self.op_overload, torch._ops.OpOverload)
            else {}
        )
        # FIXME: self.kwargs 并不总是与 schema 中定义的 kwargs 匹配，因此有时会显式地传入 ordered_kwargs_for_cpp_kernel。
        # 如果 self.op_overload 是 torch._ops.OpOverload 类型，并且 ordered_kwargs_for_cpp_kernel 未设置，
        # 则根据 schema 中的定义，收集仅关键字参数的顺序列表。
        if (
            isinstance(self.op_overload, torch._ops.OpOverload)
            and not self.ordered_kwargs_for_cpp_kernel
        ):
            self.ordered_kwargs_for_cpp_kernel = [
                x.name for x in self.op_overload._schema.arguments if x.kwarg_only
            ]
    def fill_non_provided_args(self, args, kwargs, convert_val_to_str=False):
        # 确保 args 是列表或元组类型
        assert isinstance(args, (list, tuple))
        # 如果 args 是元组，转换为列表
        if isinstance(args, tuple):
            args = list(args)
        # 断言 ExternKernel.arg_properties 不为空
        assert self.arg_properties, "ExternKernel.arg_properties should not be empty"

        # 获取传入的位置参数个数和 ExternKernel.arg_properties 的个数
        n_args = len(args)
        n_pos_args = len(self.arg_properties)

        # 对于 cpp 包装器，如果某些位置参数未提供，需要检查它们是否在关键字参数中或者使用它们的默认值
        if n_args < n_pos_args:
            log.debug(
                "%s has %d unprovided positional arguments. "
                "Will check if they are in the keyword arguments or will use default values.",
                self.op_overload,
                n_pos_args - n_args,
            )
            # 补充缺失的位置参数
            for i in range(n_args, n_pos_args):
                arg_name = self.arg_properties[i]["name"]
                args.append(
                    kwargs[arg_name]
                    if arg_name in kwargs
                    else self.arg_properties[i]["default_value"]
                )

        # 返回补充后的参数列表
        return args

    def decide_layout(self):
        # 如果 self.layout 是 FlexibleLayout 类型，则应用约束并冻结布局
        if isinstance(self.layout, FlexibleLayout):
            self.apply_constraint()
            self.freeze_layout()

    def codegen_comment(self, wrapper):
        # 获取内核元数据和详细的内核元数据字符串
        origin_str, detailed_origin_str = get_kernel_metadata(self, wrapper)
        # 如果存在 origin_str，则将其写入包装器
        if origin_str:
            wrapper.writeline(origin_str)

    def codegen(self, wrapper):
        # 抛出未实现错误，子类需要实现这个方法
        raise NotImplementedError

    def get_kernel_name(self):
        # 根据配置返回内核名称，根据是否有 cpp_wrapper 决定是使用 C 语言包装函数名还是 Python 内核名称
        return (
            (
                V.graph.wrapper_code.get_c_shim_func_name(self.cpp_kernel_name)  # type: ignore[attr-defined]
                if config.abi_compatible
                else self.cpp_kernel_name
            )
            if V.graph.cpp_wrapper
            else self.python_kernel_name
        )

    @staticmethod
    def copy_input(x):
        # 创建 Pointwise 实例，并复制输入 x 的相关属性
        pw = Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=x.make_loader(),
            ranges=x.get_size(),
            origin_node=x.get_origin_node(),
            traceback=x.get_traceback(),
        )
        pw.realize()  # 实现 Pointwise 实例
        return pw

    @classmethod
    def process_kernel(
        cls, kernel, *args, **kwargs
    ) -> Tuple[
        Any,
        List[Any],
        List[Any],
        Callable[[Any, Any], Any],
        Optional[Dict[sympy.Symbol, pytree.KeyPath]],
    # 定义函数签名，接受一个参数并返回一个元组，包含不同类型的数据结构作为返回值的标识
    @classmethod
    def convert_to_reinterpret_view(cls, x):
        """
        In order to pass this to an extern kernel we need a
        ReinterpretView not a View.  This allows us to avoid some
        unneeded copies.
        """
        # 断言输入参数 x 是 BaseView 类的实例
        assert isinstance(x, BaseView)
        # 如果 x 已经是 ReinterpretView 的实例，则直接返回 x
        if isinstance(x, ReinterpretView):
            return x

        # 注意：在这里不使用 extract_read_writes，因为在 make_loader() 内联计算时会失败

        # 获取 x 的未包装视图（unwrap_view）并获取其原始节点的名称
        x_unwrap_view = x.unwrap_view()
        x_unwrap_view_fx_node = V.graph.get_buffer(
            x_unwrap_view.get_name()
        ).get_origin_node()

        # 根据 eager 模式设置，优先使用 channels last 的格式
        if (
            x_unwrap_view_fx_node is not None
            and "val" in x_unwrap_view_fx_node.meta
            and isinstance(x_unwrap_view.layout, FlexibleLayout)
            and (
                x_unwrap_view_fx_node.meta["val"].is_contiguous(
                    memory_format=torch.channels_last
                )
                or x_unwrap_view_fx_node.meta["val"].is_contiguous(
                    memory_format=torch.channels_last_3d
                )
            )
        ):
            # 使用相同顺序创建 channels last 的步长
            x_unwrap_view.freeze_layout_with_same_order(
                make_channels_last_strides_for(x_unwrap_view.get_size())
            )
        else:
            # 冻结当前布局
            x_unwrap_view.freeze_layout()

        # 根据 x 的大小获取索引参数和变量范围
        index_args, var_ranges = dependencies.index_vars_squeeze(
            x.get_size(), prefix="r"
        )
        range_vars = index_args[0]
        # 使用 x 的索引方法创建索引
        index = x.make_indexer()(range_vars)

        # 简化索引并结合变量范围
        index = V.graph.sizevars.simplify_with_ranges(index, var_ranges)
        # 获取索引的步长
        strides = V.graph.sizevars.stride_vars(index, range_vars)
        # 获取偏移量
        offset = V.graph.sizevars.offset_var(index, range_vars)
        # 预期的索引计算结果
        expected = sympy_dot(range_vars, strides) + offset

        # 如果计算的索引与预期不符合，则抛出 NotImplementedError
        if index != expected:
            log.debug(
                "convert_to_reinterpret_view failed: stride=%s offset=%s index=%s",
                strides,
                offset,
                index,
            )
            raise NotImplementedError

        # 返回一个新的 ReinterpretView 对象，使用指定的数据和布局参数
        return ReinterpretView(
            data=x.data,
            layout=FixedLayout(
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=x.get_size(),
                stride=strides,
                offset=offset,
            ),
        )

    @classmethod
    def realize_input(cls, x):
        # 如果输入为 None，则返回 NoneAsConstantBuffer 对象
        if x is None:
            return NoneAsConstantBuffer()
        # 如果输入是 sympy.Expr、sympy.logic.boolalg.Boolean 或整数，则返回 ShapeAsConstantBuffer 对象
        if isinstance(x, (sympy.Expr, sympy.logic.boolalg.Boolean, int)):
            return ShapeAsConstantBuffer(x)
        # 如果输入是 Constant 对象，则使用其值创建 tensor，并添加到图中，返回结果
        if isinstance(x, Constant):
            return V.graph.add_tensor_constant(
                torch.tensor(x.value, dtype=x.get_dtype(), device=x.get_device())
            )
        # 如果输入是 ConstantBuffer 对象，则直接返回
        if isinstance(x, ConstantBuffer):
            return x
        # 如果输入是 TensorBox 对象，则递归调用 realize_input 处理其数据
        if isinstance(x, TensorBox):
            return cls.realize_input(x.data)
        # 如果输入是 ReinterpretView 对象，则递归调用 realize_input 处理其数据，并按指定布局重新解释视图
        if isinstance(x, ReinterpretView):
            return ReinterpretView(cls.realize_input(x.data), x.get_layout())
        # 如果输入是 BaseView 对象，则实现视图，并检查其是否满足存储和布局条件
        if isinstance(x, BaseView):
            x.realize()
            if is_storage_and_layout(x.unwrap_view()):
                try:
                    return cls.convert_to_reinterpret_view(x)
                except NotImplementedError:
                    pass
        # 如果输入是 StorageBox 对象，则实现存储，并返回其本身
        if isinstance(x, StorageBox):
            # TODO(jansel): impose layout preference on realized buffer
            x.realize()
            return x
        # 如果输入是 TorchBindObject 对象，则直接返回
        if isinstance(x, TorchBindObject):
            return x
        # 其他情况下，复制输入对象并返回
        return cls.copy_input(x)

    @classmethod
    def require_stride1(cls, x):
        # 如果输入对象满足存储和布局条件，并且其步长为 1，则返回该对象
        if is_storage_and_layout(x):
            if len(x.get_stride()) == 0:
                return x
            for stride in x.get_stride():
                if stride == 1:
                    return x
        # 否则，复制输入对象并返回
        return cls.copy_input(x)

    @classmethod
    def require_channels_last(cls, x):
        # 要求输入对象的布局为 NHWC_STRIDE_ORDER
        return cls.require_stride_order(x, NHWC_STRIDE_ORDER)

    @classmethod
    def require_channels_last_3d(cls, x):
        # 要求输入对象的布局为 NHWDC_STRIDE_ORDER
        return cls.require_stride_order(x, NHWDC_STRIDE_ORDER)

    @classmethod
    def require_contiguous(cls, x):
        # 要求输入对象的布局为输入维度的逆序
        return cls.require_stride_order(x, list(reversed(range(len(x.get_size())))))

    def apply_constraint(self):
        # 应用约束，但函数体为空，不执行任何操作
        pass

    def codegen_const_args(self):
        # 如果 V.graph.cpp_wrapper 存在，则生成常量参数的包装代码
        if V.graph.cpp_wrapper:
            result = []
            for i, x in enumerate(self.constant_args):
                idx = len(self.inputs) + i
                # 获取参数类型信息，如果存在则获取类型字符串，否则为 None
                type_ = (
                    self.arg_properties[i].get("type")
                    if self.arg_properties and idx < len(self.arg_properties)
                    else None
                )
                # 调用 V.graph.wrapper_code.val_to_arg_str 将值转换为参数字符串，并添加到结果列表中
                result.append(
                    V.graph.wrapper_code.val_to_arg_str(x, type_)  # type: ignore[arg-type]
                )
            return result
        else:
            # 如果 V.graph.cpp_wrapper 不存在，则直接将 constant_args 中的每个值转换为参数字符串，并返回生成的结果
            return map(V.graph.wrapper_code.val_to_arg_str, self.constant_args)
    # 生成函数调用所需的位置参数列表
    def codegen_args(self):
        args = []
        # 遍历输入列表的索引和元素
        for i, x in enumerate(self.inputs):
            # 如果元素是列表
            if isinstance(x, list):
                # 生成列表内每个元素的代码引用，并拼接成字符串
                names = [i.codegen_reference() for i in x]
                codegen_reference = f'[{", ".join(names)}]'
                args.append(codegen_reference)
            else:
                # 如果元素不是列表
                if V.graph.cpp_wrapper:
                    # 断言确保索引有效，访问外部内核属性
                    assert self.arg_properties and i < len(
                        self.arg_properties
                    ), "Invalid access to ExternKernel.arg_properties"
                    # 获取参数的类型
                    type_ = self.arg_properties[i].get("type")
                    # 转换参数值为C++代码字符串表示形式
                    args.append(
                        V.graph.wrapper_code.val_to_arg_str(  # type: ignore[arg-type]
                            x, type_
                        )
                    )
                else:
                    # 生成参数的代码引用
                    args.append(x.codegen_reference())
        # 添加常量参数的代码引用
        args.extend(self.codegen_const_args())
        return args

    # 获取指定关键字参数的值
    def get_kwargs_value(self, arg_name):
        if arg_name in self.kwargs:
            return self.kwargs.get(arg_name)
        # 如果关键字参数不存在于kwargs中，则获取其默认值
        if self.allarg_properties and self.allarg_properties.get(arg_name):
            return self.allarg_properties.get(arg_name).get("default_value")  # type: ignore[union-attr]
        else:
            # 如果找不到指定的关键字参数，则引发断言错误
            raise AssertionError(f"{arg_name} not in self.allarg_properties")

    # 生成函数调用所需的关键字参数列表
    def codegen_kwargs(self, skip_out=False):
        if V.graph.cpp_wrapper:
            kwargs = []
            # 遍历用于C++内核的排序后的关键字参数列表
            for arg_name in self.ordered_kwargs_for_cpp_kernel:
                # 如果跳过输出参数并且当前参数是"out"，则继续下一个循环
                if skip_out and arg_name == "out":
                    # ExternKernelOut 对于插入输出参数有自己的逻辑处理
                    continue

                # 获取关键字参数的值
                v = self.get_kwargs_value(arg_name)
                # 如果值是 sympy.Expr 类型，则直接添加
                if isinstance(v, sympy.Expr):
                    kwargs.append(v)
                else:
                    # 否则，获取参数的类型并将其转换为C++代码字符串表示形式
                    type_ = (
                        self.allarg_properties.get(arg_name).get("type")  # type: ignore[union-attr]
                        if self.allarg_properties and arg_name in self.allarg_properties
                        else None
                    )
                    kwargs.append(
                        V.graph.wrapper_code.val_to_arg_str(  # type: ignore[arg-type]
                            v, type_
                        )
                    )
        else:
            # 如果不是C++包装器，则生成关键字参数的字符串表示形式列表
            kwargs = [
                f"{k}={V.graph.wrapper_code.val_to_arg_str(v)}"  # type: ignore[misc]
                for k, v in self.kwargs.items()
            ]
        return kwargs
    def codegen_size_asserts(self, wrapper):
        # 检查是否启用尺寸断言且不是使用 C++ 封装器
        if config.size_asserts and not V.graph.cpp_wrapper:
            # 对于大小为 0 的张量，比较步长是棘手的，暂时忽略它们
            if sympy_product(self.get_size()) == 0:
                return
            # 生成尺寸和步长的代码形式
            size = V.graph.wrapper_code.codegen_shape_tuple(self.get_size())
            stride = V.graph.wrapper_code.codegen_shape_tuple(self.get_stride())
            # 写入断言语句
            wrapper.writeline(
                f"assert_size_stride({self.get_name()}, {size}, {stride})"
            )

    def get_group_stride(self):
        """
        获取输出大小和步长，用于模板代码生成
        """
        _size = self.get_size()
        _stride = self.get_stride()
        # iter_ranges = 输出张量的大小，reduce_range = [] 因为没有减少
        return [_size, []], _stride

    def canonicalize(self):
        """
        手动获取输出索引的规范化
        """
        # 手动为卷积生成索引公式
        sizevars = V.graph.sizevars
        sizes = self.get_size()
        strides = self.get_stride()
        strides = [sizevars.size_hint(x) for x in strides]
        # TODO: 我无法确定这里的符号是临时的
        index_vars = [sympy_index_symbol(f"d{i}") for i in range(len(sizes))]
        # 根据步长重新排序索引变量
        index_order = sorted(range(len(strides)), key=strides.__getitem__, reverse=True)
        lookup = {pos: idx for idx, pos in enumerate(index_order)}
        order = [lookup[i] for i in range(len(lookup))]
        index_vars = [index_vars[i] for i in order]
        indexer = self.make_indexer()
        index = indexer(index_vars)

        new_sizes, reindex, prune = V.graph.sizevars._simplify_loops(
            index_vars, sizes, [index]
        )

        # 为每个维度分配新变量以处理编号不匹配
        # d0, d1, d2 可能会变成 d0, d2 —— 与 d0, d1 不匹配
        _, add_var = var_builder("c")
        replacement = dict(zip(index_vars, reindex([add_var(x) for x in new_sizes])))

        index = sympy_subs(sympy.expand(index), replacement)  # type: ignore[arg-type]
        return index, tuple(new_sizes)

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        # 注意：不需要检查常规输入，因为我们自动有它们的依赖关系
        r = set()
        for arg in self.constant_args:
            r |= maybe_free_unbacked_symbols(arg)
        for arg in self.kwargs.values():
            r |= maybe_free_unbacked_symbols(arg)
        return r
    # 定义对象的字符串表示方法，用于返回对象的描述信息
    def __str__(self):
        # 获取对象的 python_kernel_name 属性，如果不存在则设为 None
        kernel_name = getattr(self, "python_kernel_name", None)
        # 创建描述信息的列表，包括 python_kernel_name 属性的值
        lines = [
            f"python_kernel_name={kernel_name!r}",
        ]
        # 遍历数据类中所有字段，生成每个字段的名称和对应的属性值的描述信息
        lines += [
            f"{field.name}={getattr(self, field.name)}"
            for field in dataclasses.fields(self)
        ]
        # 添加 origin_node 属性的描述信息到列表末尾
        lines.append(f"origin_node={self.origin_node!r}")
        # 调用辅助方法 str_helper 返回描述信息的字符串表示
        return self.str_helper(lines)

    # 将 __repr__ 方法设置为与 __str__ 方法相同的方法实现
    __repr__ = __str__
@dataclasses.dataclass
class ExternKernelOut(ExternKernel):
    # ExternKernelOut 类继承自 ExternKernel，用于表示带输出的外部内核对象

    def codegen(self, wrapper):
        # 生成外部内核的代码
        self.codegen_comment(wrapper)  # 调用内部方法生成注释
        args = [*self.codegen_args(), *self.codegen_kwargs(skip_out=True)]  # 获取生成的参数列表，包括跳过输出的参数
        wrapper.generate_extern_kernel_out(
            self.get_kernel_name(),  # 获取内核名称
            self.codegen_reference(),  # 生成内核引用的代码
            self.output_view.codegen_reference() if self.output_view else None,  # 生成输出视图的引用代码（如果存在的话）
            args,  # 传递参数列表给外部内核生成器
        )

    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        kwargs=None,
        output_view=None,
        python_kernel_name=None,
        cpp_kernel_name=None,
        ordered_kwargs_for_cpp_kernel=(),
        op_overload=None,
    ):
        super().__init__(
            None,
            layout,
            self.unwrap_storage(inputs),  # 将输入解封装为存储对象
            constant_args,
            kwargs or {},  # 处理关键字参数
            None,
            python_kernel_name,
            cpp_kernel_name,
            ordered_kwargs_for_cpp_kernel,
            op_overload,
        )
        self.name = V.graph.register_buffer(self)  # 注册缓冲区对象到图中

    def should_allocate(self):
        return True  # 始终返回 True，表示应该分配资源


class RandomSeeds(ExternKernelOut):
    def __init__(self, count: int, device: torch.device):
        limits = torch.iinfo(torch.int64)
        super().__init__(
            layout=FixedLayout(
                device=device,
                dtype=torch.int64,
                size=[count],
            ),  # 使用固定的布局参数初始化
            inputs=[],  # 没有输入
            constant_args=[limits.min, limits.max, [count]],  # 常数参数列表
            python_kernel_name="aten.randint.low_out",  # Python 内核名称
            # FIXME: Ideally we should only use at::_ops::randint_low_out::call here,
            # but the signature is different from is at::randint_out. Again,
            # we can simplify the code when only keeping an ABI-compatible version.
            cpp_kernel_name="at::_ops::randint_low_out::call"  # C++ 内核名称，根据 ABI 兼容性选择不同的版本
            if config.abi_compatible
            else "at::randint_out",  # 如果 ABI 兼容，选择简化的版本
            op_overload=aten.randint.low_out,  # 操作重载函数
        )


class ExternKernelAlloc(ExternKernel):
    def codegen(self, wrapper):
        # 生成外部内核的代码
        self.codegen_comment(wrapper)  # 调用内部方法生成注释
        args = [*self.codegen_args(), *self.codegen_kwargs()]  # 获取生成的参数列表，包括关键字参数
        V.graph.wrapper_code.generate_extern_kernel_alloc(self, args)  # 使用图的包装代码生成外部内核的分配部分
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)  # 如果布局是 Layout 类的实例，生成大小断言的代码

    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        kwargs=None,
        python_kernel_name=None,
        cpp_kernel_name=None,
        ordered_kwargs_for_cpp_kernel=(),
        op_overload=None,
    ):
        super().__init__(
            None,
            layout,
            self.unwrap_storage(inputs),  # 将输入解封装为存储对象
            constant_args,
            kwargs or {},  # 处理关键字参数
            None,
            python_kernel_name,
            cpp_kernel_name,
            ordered_kwargs_for_cpp_kernel,
            op_overload,
        )
        self.name = V.graph.register_buffer(self)  # 注册缓冲区对象到图中
    # 返回 False，表示不需要分配资源
    def should_allocate(self):
        return False

    # 抛出 NotImplementedError 异常，表示子类需要实现该方法
    def apply_constraint(self):
        raise NotImplementedError
class UserDefinedTritonKernel(ExternKernel):
    # 从 Triton 运行时的 autotuner 模块导入 Autotuner 类
    def get_kernel_and_configs(self):
        from triton.runtime.autotuner import Autotuner
        
        # 从 torch._higher_order_ops.triton_kernel_wrap 模块导入 kernel_side_table
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        
        # 根据 kernel_idx 从 kernel_side_table 中获取特定的内核
        kernel = kernel_side_table.get_kernel(self.kernel_idx)
        configs = []
        if isinstance(kernel, Autotuner):
            # 如果 kernel 是 Autotuner 的实例，则获取其配置列表，并将 kernel 替换为其函数
            configs = kernel.configs
            kernel = kernel.fn
        return kernel, configs

    # 生成代码的方法，接受一个 wrapper 对象作为参数
    def codegen(self, wrapper):
        kernel, configs = self.get_kernel_and_configs()

        # 定义内核的新名称和 Triton 元数据
        new_name, triton_meta = wrapper.define_user_defined_triton_kernel(
            kernel, configs, self.kwargs
        )

        # 获取代码生成的关键字参数
        args = self.codegen_kwargs()
        raw_args = list(self.kwargs.values())

        if V.graph.cpp_wrapper:
            # 如果在 C++ 包装器中，则不传递 constexpr 参数，
            # 因为它们不会作为 PTX 代码的参数传递给用户定义的 Triton 内核
            args = [arg for i, arg in enumerate(args) if i not in kernel.constexprs]
            # 在 cpp 包装器和 python 包装器之间统一 raw_args 计算方式
            raw_args = []
            for i, arg_name in enumerate(self.ordered_kwargs_for_cpp_kernel):
                if i not in kernel.constexprs:
                    raw_args.append(self.get_kwargs_value(arg_name))

        # 调用内核方法
        self.codegen_comment(wrapper)
        wrapper.generate_user_defined_triton_kernel(
            new_name, self.grid, configs, args, triton_meta, raw_args
        )

    # 是否应该分配资源的方法，始终返回 False
    def should_allocate(self):
        return False

    # 是否具有副作用的方法，始终返回 True
    def has_side_effects(self):
        # UserDefinedTritonKernel 不返回任何内容，而是在原地修改输入，
        # 不允许其被 DCE（死代码消除）处理
        return True

    # 获取未支持的符号用途的方法，返回由 grid 使用的未支持符号和 kwargs 使用的符号的并集
    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        # 将 ExternKernel 类的 get_unbacked_symbol_uses 方法返回的符号集合与 grid 使用的自由未支持符号的集合进行合并
        return super().get_unbacked_symbol_uses() | free_unbacked_symbols(self.grid)

    # 获取未支持的符号定义的方法，始终返回空集合
    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    # 获取变异名称的方法，返回空列表
    def get_mutation_names(self):
        # 注意事项：Inductor 只允许节点变异 0 或 1 个缓冲区。
        # 为了绕过此限制，我们创建 MutationOutputs，标记其分配的输入为可变的，从而遵循 Inductor 的约束。
        return []
    # 初始化方法，接受关键字参数 kernel_idx, grid, kernel_args
    def __init__(self, *, kernel_idx, grid, kernel_args):
        # 初始化空列表用于存储输入
        inputs = []
        # 初始化空字典用于存储参数
        kwargs = dict()
        # 初始化空列表用于存储常量参数
        constant_args = []
        
        # 遍历 kernel_args 中的键值对
        for k, v in kernel_args.items():
            # 如果值 v 是 TensorBox 类型的实例
            if isinstance(v, TensorBox):
                # 调用 realize_input 方法处理 v，并获取其存储的输入数据
                t = InputsKernel.unwrap_storage_for_input(self.realize_input(v))
                # 将处理后的输入数据 t 添加到 inputs 列表中
                inputs.append(t)
                # 将 t 存储到 kwargs 字典中的键 k 下
                kwargs[k] = t
            else:
                # 如果 v 不是 TensorBox 类型的实例，将其作为常量参数添加到 constant_args 列表中
                constant_args.append(v)
                # 将 v 存储到 kwargs 字典中的键 k 下
                kwargs[k] = v
        
        # 断言确保 inputs 列表中至少有一个输入
        assert len(inputs) != 0
        # 获取 inputs 列表中第一个元素的设备信息
        device = inputs[0].get_device()

        # 调用父类的初始化方法，传入以下参数
        super().__init__(
            None,  # 传入 None
            NoneLayout(device),  # 创建一个与设备相关的布局对象，用于存储空间分配
            inputs,  # 输入列表
            tuple(constant_args),  # 常量参数的元组
            kwargs,  # 所有参数的字典
        )
        
        # 在 V.graph 中注册当前实例并存储到 self.name 属性中
        self.name = V.graph.register_buffer(self)
        # 存储 kernel_idx 到 self.kernel_idx 属性
        self.kernel_idx = kernel_idx
        # 存储 grid 到 self.grid 属性

        self.grid = grid

        # 调用 get_kernel_and_configs 方法获取 kernel 和 configs
        kernel, configs = self.get_kernel_and_configs()
        
        # 如果正在进行自动调优，可能不会传递所有参数
        # 根据 kernel.arg_names 和 kernel_args 确定有序的参数列表
        self.ordered_kwargs_for_cpp_kernel = [
            arg for arg in kernel.arg_names if arg in kernel_args
        ]

        # 导入 identify_mutated_tensors 方法
        from torch._higher_order_ops.triton_kernel_wrap import identify_mutated_tensors
        
        # 如果 configs 长度大于 0，则获取第一个配置的 kwargs，否则使用空字典
        autotuned_kwargs = configs[0].kwargs if len(configs) > 0 else {}
        
        # 根据 kernel 和合并后的 kernel_args 和 autotuned_kwargs 确定可变参数列表
        self.mutable_args = [
            kernel_args[key]
            for key in identify_mutated_tensors(
                kernel, {**kernel_args, **autotuned_kwargs}
            )
        ]
        
        # 调用 mark_node_as_mutating 方法，标记当前对象以及其可变参数为变异状态
        mark_node_as_mutating(self, *self.mutable_args)

    # 返回可变参数 mutable_args 中每个元素的名称列表
    def get_inputs_that_alias_output(self):
        return [i.get_name() for i in self.mutable_args]
# 将当前缓冲区标记为具有变异性，同时指示调度器这些操作依赖于当前缓冲区
def mark_node_as_mutating(cur_buffer, *mutated_nodes: IRNode):
    """
    允许标记mutated_nodes中的操作为被变异，并告知调度器这些操作依赖于cur_buffer。

    注意：请使用这个函数而不是直接构造MutationOutput对象
    """
    for node in mutated_nodes:
        assert isinstance(
            node, IRNode
        ), f"{node} node is type {type(node)} and is not an IRNode"
        # 标记图中的缓冲区为已变异
        V.graph.mark_buffer_mutated(node.get_name())
        # 创建MutationOutput对象，表示发生变异
        MutationOutput(node.get_layout(), node, cur_buffer)


class MutationOutput(ExternKernel):
    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def __init__(self, layout, mutated_node, node_doing_mutating):
        """
        创建MutationOutput对象，用于表示变异操作的输出

        注意：请不要直接构造这个对象，使用`mark_node_as_mutating`
        """
        super().__init__(None, layout, [mutated_node, node_doing_mutating], ())
        self.node_doing_mutating = node_doing_mutating
        self.name = V.graph.register_buffer(self)

    def should_allocate(self):
        return False

    def is_no_op(self):
        return True

    def has_side_effects(self):
        return True

    def get_inputs_that_alias_output(self):
        return [self.inputs[0].get_name()]


class InplaceBernoulliFallback(ExternKernel):
    """
    处理变异的自定义类
    """

    def codegen(self, wrapper):
        (x,) = (t.codegen_reference() for t in self.inputs)

        if V.graph.cpp_wrapper and config.abi_compatible:
            # 对于cpp包装器，Generator参数总是NULL，需要显式生成
            wrapper.writeline(
                f"{self.get_kernel_name()}({x}, {', '.join(map(repr, self.constant_args))}, NULL){wrapper.ending}"
            )
        else:
            wrapper.writeline(
                f"{self.get_kernel_name()}({x}, {', '.join(map(repr, self.constant_args))}){wrapper.ending}"
            )

    def should_allocate(self):
        return False

    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    def __init__(self, op_overload, x, *constant_args):
        super().__init__(
            None,
            NoneLayout(x.get_device()),  # 为x设备选择NoneLayout
            self.unwrap_storage([x]),    # 解封装存储的输入x
            constant_args,
            op_overload=op_overload,
        )
        self.name = V.graph.register_buffer(self)
        self.python_kernel_name = "aten.bernoulli_"
        if not config.abi_compatible:
            # 当我们完全切换到ABI兼容时，应简化此处逻辑
            self.cpp_kernel_name = "at::native::bernoulli_"
        # 标记x为发生变异的节点
        mark_node_as_mutating(self, x)


# 用于处理torch.complex类型
class InplaceCopyFallback(ExternKernel):
    """
    处理变异的自定义类
    """
    # 生成代码的方法，用于生成调用特定内核的代码片段
    def codegen(self, wrapper):
        # 获取代码生成的参数：目标地址(dst)，源地址(src)，是否非阻塞(non_blocking)
        (dst, src, non_blocking) = self.codegen_args()
        # 在包装器中写入生成的代码行，调用内核函数并添加结尾符号
        wrapper.writeline(
            f"{self.get_kernel_name()}({dst}, {src}, {non_blocking}){wrapper.ending}"
        )

    # 判断是否需要分配资源，始终返回 False
    def should_allocate(self):
        return False

    # 获取变异名称列表，返回第一个输入的名称
    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    # 获取未支持符号定义的集合，始终返回空集合
    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    # 初始化方法，设置布局、输入、常量参数等，并注册缓冲区
    def __init__(
        self,
        layout,
        inputs,
        constant_args,
    ):
        super().__init__(
            None,
            layout,
            inputs,
            constant_args,
            python_kernel_name="aten.copy_",
            cpp_kernel_name=(
                "aoti_torch_copy_" if config.abi_compatible else "at::_ops::copy_::call"
            ),
        )
        # 在图形中注册当前对象为缓冲区
        self.name = V.graph.register_buffer(self)

    # 类方法，创建一个 InplaceCopyFallback 实例
    @classmethod
    def create(cls, dst, src, non_blocking: bool = False):
        # 实例化输入对象列表
        inputs = [cls.realize_input(t) for t in [dst, src]]
        # 设置常量参数为非阻塞标志
        constant_args = (non_blocking,)
        # 创建并返回 InplaceCopyFallback 实例
        result = InplaceCopyFallback(
            NoneLayout(dst.get_device()),  # type: ignore[arg-type]
            inputs,
            constant_args,
        )
        # 标记结果节点为具有变异性质，并影响目标地址的状态
        mark_node_as_mutating(result, dst)
        return result
class MutatingFirstArgExternKernel(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """

    def codegen(self, wrapper):
        # 生成代码，将输入参数和常量参数转换为代码引用
        argrefs = [
            *(t.codegen_reference() for t in self.inputs),  # 输入参数的代码引用
            *map(repr, self.constant_args),  # 常量参数的字符串表示
        ]
        # 在包装器中写入生成的函数调用代码
        wrapper.writeline(
            f"{self.get_kernel_name()}({', '.join(argrefs)}){wrapper.ending}"
        )

    def should_allocate(self):
        # 不需要分配额外内存空间
        return False

    def get_mutation_names(self):
        # 返回可能发生变异的参数名称列表
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        # 返回未支持的符号定义的集合
        return set()

    def has_side_effects(self):
        # 具有副作用，可能会修改输入参数
        return True


class ResizeStorageBytes(MutatingFirstArgExternKernel):
    def __init__(self, variable, new_size):
        assert isinstance(new_size, int), "TODO: dynamic shapes"
        super().__init__(
            None,
            NoneLayout(variable.get_device()),  # 使用NoneLayout初始化布局
            self.unwrap_storage([variable]),  # 解封装变量存储
            constant_args=(new_size,),  # 常量参数是新的大小
        )
        V.graph.mark_buffer_mutated(variable.get_name())  # 标记变量缓冲区已变异
        self.name = V.graph.register_buffer(self)  # 注册缓冲区名称到图中
        self.python_kernel_name = "inductor_ops.resize_storage_bytes_"  # Python内核名称
        self.cpp_kernel_name = "torch::inductor::resize_storage_bytes_"  # C++内核名称
        V.graph.never_reuse_buffers.add(variable.data.get_name())  # 标记永不重用的变量数据名称
        mark_node_as_mutating(self, variable)  # 标记节点为变异，应用于变量


class SetSourceTensorKernel(ExternKernelAlloc):
    def __init__(self, self_tensor, storage_tensor):
        self_tensor.freeze_layout()  # 冻结自身张量的布局
        super().__init__(
            self_tensor.get_layout(),  # 获取自身张量的布局
            [self_tensor, storage_tensor],  # 输入参数为自身张量和存储张量
            python_kernel_name="torch.ops.aten.set_.source_Tensor",  # Python内核名称
        )
        V.graph.never_reuse_buffers.add(self_tensor.data.get_name())  # 标记永不重用的自身张量数据名称
        V.graph.never_reuse_buffers.add(storage_tensor.get_name())  # 标记永不重用的存储张量名称
        V.graph.never_reuse_buffers.add(self.get_name())  # 标记永不重用的自身名称
        mark_node_as_mutating(self, self_tensor, storage_tensor)  # 标记节点为变异，应用于自身张量和存储张量

    def get_inputs_that_alias_output(self):
        # 返回输入参数的名称列表，这些参数别名为输出
        return [self.inputs[0].get_name(), self.inputs[1].get_name()]

    def get_mutation_names(self):
        # 返回可能发生变异的存储张量名称列表
        return [self.inputs[1].get_name()]

    def has_side_effects(self):
        # 具有副作用，可能会修改自身张量和存储张量
        return True


class ScatterFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly.
    This class handles both aten.scatter_ and aten.scatter_reduce_.
    It also handle the case `src` being a scalar properly.
    """
    # 定义一个名为 codegen 的方法，接受一个参数 wrapper
    def codegen(self, wrapper):
        # 从 self.kwargs 中获取 reduce 参数
        reduce = self.kwargs["reduce"]
        
        # 如果 V.graph.cpp_wrapper 存在，则进行以下操作
        if V.graph.cpp_wrapper:
            # 定义一个字典 get_operator_enum，映射 reduce 参数到对应的字符串操作符
            get_operator_enum = {"add": "sum", "multiply": "prod"}
            # 如果 reduce 存在于 get_operator_enum 中，则将 reduce 替换为对应的字符串操作符
            if reduce in get_operator_enum:
                reduce = get_operator_enum[reduce]

        # 如果 self.src_is_tensor 为真，则分别从 self.inputs 中获取 x, index, src 的 codegen_reference
        if self.src_is_tensor:
            (x, index, src) = (t.codegen_reference() for t in self.inputs)
        else:
            # 否则从 self.inputs 中获取 x, index 的 codegen_reference，并从 self.constant_args 中获取 src
            (x, index) = (t.codegen_reference() for t in self.inputs)
            src = self.constant_args[1]
        
        # 调用 wrapper 的 generate_scatter_fallback 方法，传入相应的参数
        wrapper.generate_scatter_fallback(
            x,
            [x, self.constant_args[0], index, src],
            self.cpp_kernel_name,
            self.python_kernel_name,
            self.src_is_tensor,
            reduce,
            self.codegen_kwargs(),
        )

    # 定义一个名为 should_allocate 的方法，返回 False
    def should_allocate(self):
        return False

    # 定义一个名为 get_mutation_names 的方法，返回一个包含 self.inputs[0] 名称的列表
    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    # 定义一个名为 get_unbacked_symbol_defs 的方法，返回一个空集合
    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    # 定义一个构造函数 __init__，接受多个参数，并进行初始化操作
    def __init__(
        self,
        op_overload,
        x,
        dim: int,
        index,
        src,
        *,
        reduce: Optional[str] = None,
        include_self: bool = True,
    ):
        # 判断 src 是否为 TensorBox 类型，将结果保存到 self.src_is_tensor 中
        self.src_is_tensor = isinstance(src, TensorBox)

        # 根据 self.src_is_tensor 的值，选择性地实例化 tensors 和 constant_args
        constant_args: Tuple[Any, ...]
        if self.src_is_tensor:
            tensors = [self.realize_input(t) for t in [x, index, src]]
            constant_args = (dim,)
        else:
            tensors = [self.realize_input(t) for t in [x, index]]
            constant_args = (dim, src)

        # 调用父类的构造函数进行初始化
        super().__init__(
            None,
            NoneLayout(x.get_device()),  # 设置布局为 NoneLayout，并传入 x 的设备信息
            self.unwrap_storage(tensors),  # 对 tensors 进行解包并获取其存储信息
            constant_args,  # 传入 constant_args
            {"reduce": reduce, "include_self": include_self},  # 设置关键字参数字典
            python_kernel_name=str(op_overload),  # 设置 Python 内核名称为 op_overload 的字符串表示
            ordered_kwargs_for_cpp_kernel=["reduce", "include_self"],  # 指定传递给 C++ 内核的关键字参数顺序
            op_overload=op_overload,  # 设置操作重载
        )
        
        # 获取并设置 C++ 内核名称
        self.cpp_kernel_name = get_aten_cpp_kernel_name(op_overload)
        
        # 将自身注册为 V.graph 的缓冲对象，并获取其名称保存到 self.name 中
        self.name = V.graph.register_buffer(self)
        
        # 将自身标记为会改变 x 的节点
        mark_node_as_mutating(self, x)
class IndexPutFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation and indices properly
    """

    def codegen(self, wrapper):
        # 生成输入的代码引用
        (x, values, *valid_indices) = (t.codegen_reference() for t in self.inputs)
        # 初始化空列表用于存放有效索引
        indices = []
        # 创建有效索引的迭代器
        iter_valid_indices = iter(valid_indices)
        # 遍历索引列表
        for i, _ in enumerate(self.indices):
            # 如果当前索引不为None，则使用下一个有效索引
            if self.indices[i] is not None:
                indices.append(next(iter_valid_indices))
            # 否则使用默认的none_str
            else:
                indices.append(V.graph.wrapper_code.none_str)

        # 调用包装器的生成索引放置回退方法
        wrapper.generate_index_put_fallback(
            self.get_kernel_name(), x, indices, values, *self.codegen_const_args()
        )

    def should_allocate(self):
        # 不需要分配额外的空间
        return False

    def get_mutation_names(self):
        # 返回输入的第一个对象的名称作为突变名称列表
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        # 返回一个空集合，表示没有未支持的符号定义
        return set()

    def __init__(self, op_overload, x, indices, values, accumulate):
        # 初始化函数，接受运算重载、输入张量、索引、数值和累加器作为参数
        self.indices = indices
        # 从索引中筛选出非空的有效索引
        valid_indices = [i for i in indices if i is not None]
        # 实现输入的张量
        tensors = [self.realize_input(x) for x in [x, values, *valid_indices]]
        # 根据ABI兼容性选择CPP内核名称
        cpp_kernel_name = (
            "aoti_torch_index_put_out" if config.abi_compatible else "at::index_put_out"
        )
        # 调用父类的初始化方法，设置Python和CPP内核名称、操作重载等
        super().__init__(
            None,
            NoneLayout(x.get_device()),  # 忽略参数类型检查
            self.unwrap_storage(tensors),
            (accumulate,),
            python_kernel_name="aten.index_put_",
            cpp_kernel_name=cpp_kernel_name,
            op_overload=op_overload,
        )
        # 在图中注册缓冲并设置名称
        self.name = V.graph.register_buffer(self)
        # 标记节点为突变
        mark_node_as_mutating(self, x)


class DeviceCopy(ExternKernelOut):
    @classmethod
    def create(cls, x, device):
        # 如果输入不是外部内存、所有读取都是常量且未启用运行时常量折叠，则常量复制到设备
        if (
            not x.is_extern()
            and all(
                (r.name in V.graph.constants and isinstance(r, dependencies.MemoryDep))
                for r in x.get_reads()
            )
            and not config.aot_inductor.use_runtime_constant_folding
        ):
            return x.constant_to_device(device)

        # 向图中添加设备信息
        V.graph.add_device_info(device)
        V.graph.add_device_info(x.get_device())

        # 发出开发者警告
        developer_warning("DeviceCopy in input program")
        # 返回设备复制对象
        return DeviceCopy(
            FlexibleLayout(
                device=device,
                dtype=x.get_dtype(),
                size=x.get_size(),
            ),
            [cls.realize_input(x)],
        )

    def codegen(self, wrapper):
        # 生成代码的参数
        args = self.codegen_args()
        assert len(args) == 1
        # 如果有输出视图，则调用包装器的设备复制方法
        if self.output_view:
            wrapper.codegen_device_copy(args[0], self.output_view.codegen_reference())
        else:
            # 否则调用包装器的设备复制方法
            wrapper.codegen_device_copy(args[0], self.codegen_reference())


class DynamicScalar(ExternKernel):
    """
    The result of a call to aten._local_scalar_dense.
    """

    def get_reads(self):
        # 返回空元组，表示没有读取操作
        return ()

    def should_allocate(self):
        # 不需要分配额外的空间
        return False
    # 初始化方法，接受符号(sym)、密钥路径(keypath)和数据(data)参数
    def __init__(self, sym, keypath, data):
        # 对数据进行实现（可能是某种数据结构的实现操作）
        data.realize()
        # 调用父类的初始化方法，使用空值和空布局，设备为CPU，并传入数据的存储信息（解包数据）
        super().__init__(None, NoneLayout(torch.device("cpu")), self.unwrap_storage([data]))  # type: ignore[arg-type]
        # 将符号(sym)和密钥路径(keypath)存储在对象中
        self.sym = sym
        self.keypath = keypath

    # 返回一个集合，其中包含对象持有的符号(sym)
    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return {self.sym}

    # 在给定的包装器(wrapper)上调用动态标量代码生成
    def codegen(self, wrapper):
        wrapper.codegen_dynamic_scalar(self)
class AssertScalar(ExternKernel):
    """
    The result of a call to aten._assert_scalar
    """

    def get_reads(self):
        # 返回一个空元组，表示没有读取任何东西
        return ()

    def should_allocate(self):
        # 指示不需要分配任何内存
        return False

    def __init__(self, scalar, msg):
        super().__init__(
            # 调用父类的构造函数，传入空缓冲区和空布局
            None,
            NoneLayout(torch.device("cpu")),  # type: ignore[arg-type]
            # 输入为一个空列表
            [],
        )  # type: ignore[arg-type]
        self.scalar = scalar  # 设置实例变量 scalar
        self.msg = msg  # 设置实例变量 msg

    def has_side_effects(self):
        # 表示该函数有副作用
        return True

    def get_unbacked_symbol_uses(self):
        # 返回与 scalar 相关的自由未支持符号的使用情况
        return free_unbacked_symbols(self.scalar)

    def codegen(self, wrapper):
        if V.graph.cpp_wrapper:
            pass
        else:
            # 注意：非常重要的一点是，在这里不要对标量进行简化，
            # 因为简化是基于运行时的断言进行的。所以如果在运行时断言中有 "u0 == 0"，
            # 如果随后尝试简化(simplify) "u0 == 0"，会得到 True（因为我们已经断言它为真）。
            # 但是我们在这里生成的是实际的运行时断言代码！
            wrapper.writeline(
                f"if not {V.graph.wrapper_code.codegen_python_sizevar(self.scalar, simplify=False)}:"
            )
            wrapper.writeline(f"    raise RuntimeError({repr(self.msg)})")
            # 没有人应该使用这个缓冲区，但为了统一性
            # 定义这个变量并将其赋值为 None
            wrapper.writeline(f"{self.get_name()} = None")
            

@dataclasses.dataclass
class ExternKernelNode:
    name: str
    node: export_schema.Node


has_c_shim = {
    aten._embedding_bag.default,
    aten._fft_c2c.default,
    aten._scaled_dot_product_efficient_attention.default,
    aten._scaled_dot_product_flash_attention.default,
    aten._scaled_mm.default,
    aten.addmm.out,
    aten.bmm.out,
    aten.copy_.default,
    aten.mm.out,
    aten.repeat_interleave.Tensor,
    aten.nonzero.default,
    aten.view.dtype,
    aten.view_as_real.default,
}

class FallbackKernel(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        kwargs=None,
        *,
        unbacked_bindings=None,
    def codegen_unbacked_symbol_defs(self, wrapper):
        # 如果对象没有属性"unbacked_bindings"，直接返回，不进行后续操作
        if not hasattr(self, "unbacked_bindings"):
            return

        # 解析未支持的绑定，使用图形大小变量的形状环境和对象的未支持绑定
        unbacked_bindings = resolve_unbacked_bindings(
            V.graph.sizevars.shape_env, self.unbacked_bindings
        )

        # 如果没有未支持的绑定，直接返回，不进行后续操作
        if not unbacked_bindings:
            return

        # 遍历未支持的绑定，其中s为键，keypath为值
        for s, keypath in unbacked_bindings.items():

            # 定义递归函数go，用于生成符号定义的代码
            def go(expr, keypath):
                # 如果keypath为空元组，返回当前表达式
                if keypath == ():
                    return expr

                # 根据不同类型的keypath进行不同处理
                if (
                    len(keypath) >= 2
                    and isinstance(keypath[0], CallMethodKey)
                    and isinstance(keypath[1], pytree.SequenceKey)
                ):
                    # 处理调用方法的情况，例如expr.method(index)
                    return go(
                        f"{expr}.{keypath[0].name}({keypath[1].idx})", keypath[2:]
                    )
                elif isinstance(keypath[0], CallMethodKey):
                    # 处理单个方法调用的情况，例如expr.method()
                    return go(f"{expr}.{keypath[0].name}()", keypath[1:])
                elif isinstance(keypath[0], pytree.SequenceKey):
                    # 处理序列索引的情况，例如expr[index]
                    return go(f"{expr}[{keypath[0].idx}]", keypath[1:])
                elif isinstance(keypath[0], DivideByKey):
                    # TODO: 需要断言可除性
                    # TODO: 这是无效的 C++ 代码生成
                    # 处理除法操作的情况，例如expr.__floordiv__(divisor)
                    return go(f"{expr}.__floordiv__({keypath[0].divisor})", keypath[1:])
                else:
                    # 抛出断言错误，指示未识别的keypath类型
                    raise AssertionError(f"unrecognized keypath {keypath}")

            # 定义外部函数go_outer，根据特定条件生成符号定义的代码
            def go_outer():
                if V.graph.cpp_wrapper and config.abi_compatible:
                    # 对于顶层缓冲区访问的特殊处理
                    # 如果只有一个输出，则直接调用go生成代码
                    if len(self.outputs) == 1:
                        return go(self.outputs[0].get_name(), keypath)
                    else:
                        # 否则，根据序列键的索引调用go生成代码
                        assert isinstance(keypath[0], pytree.SequenceKey)
                        return go(self.outputs[keypath[0].idx].get_name(), keypath[1:])
                else:
                    # 否则，调用go生成对象的名称的代码
                    return go(self.get_name(), keypath)

            # 将生成的符号定义代码写入包装器
            wrapper.writeline(
                f"{wrapper.codegen_unbacked_symbol_decl(s)} = {go_outer()}{wrapper.ending}"
            )

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        # 获取未支持符号定义的集合
        if unbacked_bindings := getattr(self, "unbacked_bindings", None):
            return resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, unbacked_bindings
            ).keys()
        else:
            return set()
    # 设置 C++ 内核的方法，接受一个内核对象作为参数
    def set_cpp_kernel(self, kernel):
        # 导入获取 C++ 操作模式架构的函数
        from .codegen.wrapper import get_cpp_op_schema
        
        # 断言内核对象的架构不可变，否则抛出异常
        assert (
            not kernel._schema.is_mutable
        ), f"mutable {kernel.__name__} is not supported with cpp_wrapper"

        # 检查内核参数列表中是否有写入操作的参数
        def is_not_write(arg):
            return arg.alias_info is None or not arg.alias_info.is_write
        
        # 断言所有参数都不是写入操作的参数，否则抛出异常
        assert all(
            is_not_write(x) for x in kernel._schema.arguments
        ), f"{kernel.__name__} with alias_info arguments is not supported with cpp_wrapper"
        
        # 断言所有返回值都不是写入操作的返回值，否则抛出异常
        assert all(
            is_not_write(x) for x in kernel._schema.returns
        ), f"{kernel.__name__} with alias_info returns is not supported with cpp_wrapper"

        # 设置对象的 C++ 内核名称和重载名称
        self.cpp_kernel_name = kernel._schema.name
        self.cpp_kernel_overload_name = kernel._schema.overload_name
        
        # 构建用于唯一标识内核的键值
        self.cpp_kernel_key = f"{self.cpp_kernel_name.replace('::', '_')}_{self.cpp_kernel_overload_name}"  # type: ignore[union-attr]

        # 获取 C++ 操作模式架构并存储到对象属性中
        self.cpp_op_schema = get_cpp_op_schema(kernel)

    # 生成代码所需的参数处理方法
    def codegen_args(self):
        # 定义用于包装参数的数据类 Shim
        @dataclasses.dataclass
        class Shim:
            ref: Any
            
            def __repr__(self):
                return self.ref
        
        # 生成输入张量的参数列表
        tensor_args = [Shim(x.codegen_reference()) for x in self.inputs]
        
        # 解包参数和常量参数
        args, kwargs = self.unflatten_args(tensor_args, self.constant_args)
        
        # 如果启用了 C++ 封装且操作重载是 torch._ops.OpOverload 类型
        if V.graph.cpp_wrapper and isinstance(self.op_overload, torch._ops.OpOverload):
            # 填充未提供的参数
            args = self.fill_non_provided_args(args, kwargs)
            
            # 将参数转换为 C++ 参数字符串格式
            args = [
                V.graph.wrapper_code.val_to_arg_str(x, param.real_type)
                for param, x in zip(self.op_overload._schema.arguments, args)
            ]
        else:
            # 否则，将参数转换为默认的封装器参数字符串格式
            args = [V.graph.wrapper_code.val_to_arg_str(x) for x in args]

        # 更新关键字参数
        self.kwargs.update(kwargs)
        
        # 返回生成的参数列表
        return args

    # 查找张量参数的设备信息
    @staticmethod
    def find_device(tensor_args, example_output):
        # 如果存在张量参数，则返回第一个张量的设备信息
        if tensor_args:
            devices = [arg.get_device() for arg in tensor_args if arg.get_device()]
            return devices[0]
        
        # 如果示例输出是 torch.Tensor 类型，则返回其设备信息
        if isinstance(example_output, torch.Tensor):
            return example_output.device
        
        # 如果示例输出是列表或元组，则递归查找设备信息并返回
        if isinstance(example_output, (list, tuple)):
            device_set = {FallbackKernel.find_device(None, x) for x in example_output}
            # 去除 None
            devices = [device for device in device_set if device]
            # 如果只有一个设备，则返回该设备
            if len(devices) == 1:
                return devices[0]
            # 否则，优先返回 GPU 设备
            for device in devices:
                if is_gpu(device.type):
                    return device
            # 最后返回任意一个设备
            return devices[0]
        
        # 默认情况下返回 None
        return None

    # 判断操作是否具有副作用
    def has_side_effects(self):
        # 如果操作重载是 torch._ops.HigherOrderOperator 类型，则没有副作用
        if isinstance(self.op_overload, torch._ops.HigherOrderOperator):
            return False
        
        # 否则，调用函数获取操作重载的架构信息，并判断是否可变
        return get_schema_info(self.op_overload).is_mutable()
    # 返回 self.alias_names，即与输出存在别名关系的输入名称列表
    def get_inputs_that_alias_output(self):
        return self.alias_names

    # 返回 self.mutation_names，确保其长度不超过1
    def get_mutation_names(self):
        assert len(self.mutation_names) <= 1
        return self.mutation_names

    # ProxyExecutor 设计说明
    # 我们将 ExternFallbackNodes（用于自定义操作）导出到一个序列化文件中，
    # 并使用主机侧的代理执行器来解决 ABI 问题
    # 目前仅在 fbcode 中实现了此功能。最终，我们还将使其在 OSS 上工作。
    # 详细的设计文档可以在以下链接找到：
    # https://docs.google.com/document/d/1wC4DOZFaYym2t1Esz0X5yxlLI3RDnSiyRbUus3bkJ64/edit?usp=sharing
    # 导出外部内核节点的方法，确保当前对象是FallbackKernel的实例
    def export_extern_kernel_node(self):
        assert isinstance(self, FallbackKernel)
        # 将输入参数和常量参数展开为args和kwargs
        args, kwargs = self.unflatten_args(self.inputs, self.constant_args)
        # 填充缺失的参数
        args = self.fill_non_provided_args(args, kwargs)
        # 根据顺序化的关键字参数列表，获取对应的参数值
        ordered_kwargs = [
            kwargs.get(key, None) for key in self.ordered_kwargs_for_cpp_kernel
        ]
        # 如果不是在图形模式下（AOT模式），则直接返回args和ordered_kwargs
        if not V.graph.aot_mode:
            # 在cpp包装的JIT模式下不需要序列化
            return [*args, *ordered_kwargs]

        # 创建一个GraphModuleSerializer对象进行序列化
        serializer = GraphModuleSerializer(None, None)  # type: ignore[arg-type]
        # 使用serializer将操作重载、args和kwargs序列化为命名参数
        named_arguments = serializer.serialize_inputs(self.op_overload, args, kwargs)  # type: ignore[arg-type]

        # 序列化输出
        def handle_single_output(return_type, output):
            if isinstance(return_type, torch.TensorType):
                # 对于单个张量
                out = output
                if isinstance(output, (list, tuple)):
                    assert len(output) == 1
                    out = output[0]
                # 创建一个TensorArgument对象作为输出参数
                return export_schema.Argument.create(
                    as_tensor=export_schema.TensorArgument(name=out.get_name())
                )
            elif isinstance(return_type, torch.ListType) and isinstance(
                return_type.getElementType(), torch.TensorType
            ):
                # 对于单个张量列表
                # 创建多个TensorArgument对象作为输出参数
                return export_schema.Argument.create(
                    as_tensors=[
                        export_schema.TensorArgument(name=out.get_name())
                        for out in output
                    ]
                )
            else:
                raise RuntimeError(f"Unsupported return type {type(return_type)}")

        # 获取操作重载对象
        target = self.op_overload
        # 获取操作重载对象的返回类型列表
        returns = target._schema.returns  # type: ignore[union-attr]
        # 如果只有一个返回值
        if len(returns) == 1:
            return_type = returns[0].real_type
            # 处理单个输出参数
            output_arguments = [handle_single_output(return_type, self.outputs)]
        else:
            # 对于元组返回，例如 "-> (Tensor, Tensor)" 或 "-> (Tesnor, Tensor[])"
            assert isinstance(self.outputs, tuple)
            assert len(returns) == len(self.outputs)
            # 处理多个输出参数
            output_arguments = [
                handle_single_output(return_schema.real_type, output)
                for return_schema, output in zip(returns, self.outputs)
            ]

        # 创建一个ExternKernelNode节点对象
        node = ExternKernelNode(
            name=self.get_name(),
            # 创建一个Node对象，描述操作重载、输入、输出和元数据
            node=export_schema.Node(
                target=self.op_overload.name(),  # type: ignore[union-attr]
                inputs=named_arguments,
                outputs=output_arguments,
                metadata={},
            ),
        )

        # 将创建的ExternKernelNode节点添加到图形对象的extern_kernel_nodes列表中
        V.graph.extern_kernel_nodes.append(node)

        # 返回args和ordered_kwargs组成的列表
        return [*args, *ordered_kwargs]
    def codegen(self, wrapper):
        # 获取当前操作的内核对象
        kernel = self.op_overload
        # 检查内核对象的命名空间是否为 "aten"
        if kernel.namespace == "aten":  # type: ignore[union-attr]
            # Aten Fallback Ops（ATen 回退操作）
            assert isinstance(kernel, torch._ops.OpOverload)
            # 如果当前环境支持 C++ 封装
            if V.graph.cpp_wrapper:
                # 在 FB Code 环境下，并且当前内核不在已有的 C shim 列表中
                if (
                    config.is_fbcode()
                    and kernel not in has_c_shim
                    # C shim v2 是通过 torchgen 生成的，应该覆盖所有 aten 操作
                    # 如果确实缺少某个操作，请更新 gen_aoti_c_shim.py
                    and config.c_shim_version == "1"
                ):
                    # 记录警告信息，表示缺少 C-shim 实现，使用代理执行器作为后备
                    log.warning(
                        "%s is missing a c-shim implementation, using proxy executor as fallback",
                        kernel,
                    )
                    # 启用运行时分派
                    self.use_runtime_dispatch = True
                    # 设置 C++ 内核
                    self.set_cpp_kernel(kernel)
            else:
                # 设置 Python 内核名称
                self.python_kernel_name = str(kernel)
        # 如果内核对象的命名空间为 "_quantized"
        elif kernel.namespace == "_quantized":  # type: ignore[union-attr]
            # 内部量化回退操作
            assert isinstance(kernel, torch._ops.OpOverload)
            # 如果当前环境支持 C++ 封装
            if V.graph.cpp_wrapper:
                # 设置 C++ 内核
                self.set_cpp_kernel(kernel)
                # 如果不是 ABI 兼容
                if not config.abi_compatible:
                    # 启用运行时分派
                    self.use_runtime_dispatch = True
            else:
                # 设置 Python 内核名称
                self.python_kernel_name = str(kernel)
        # 如果内核对象是 torch._ops.HigherOrderOperator 的实例
        elif isinstance(kernel, torch._ops.HigherOrderOperator):
            # 设置 Python 内核名称为高阶操作的路径
            self.python_kernel_name = f"torch.ops.higher_order.{kernel.__name__}"
        else:
            # 对于非 aten 的 OpOverload，即自定义操作
            # 设置 Python 内核名称为完整路径
            self.python_kernel_name = f"{kernel.__module__.replace('._ops.', '.ops.')}.{kernel.__name__}"  # type: ignore[union-attr]
            # 如果当前环境支持 C++ 封装
            if V.graph.cpp_wrapper:
                # 启用运行时分派
                self.use_runtime_dispatch = True
                # 设置 C++ 内核
                self.set_cpp_kernel(kernel)

        # 如果需要运行时分派
        if self.use_runtime_dispatch:
            # 添加代码生成的注释
            self.codegen_comment(wrapper)

            exported_args = None
            args = None
            # 如果 ABI 兼容
            if config.abi_compatible:
                # 导出外部内核节点的参数
                exported_args = self.export_extern_kernel_node()
            else:
                # 生成参数列表
                args = [*self.codegen_args(), *self.codegen_kwargs()]

            # 生成外部内核的分配和查找模式（如果需要的话）
            wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
                self.get_name(),
                self.python_kernel_name,
                self.cpp_kernel_name,
                args,
                self.cpp_op_schema,
                self.cpp_kernel_key,
                self.cpp_kernel_overload_name,
                self.op_overload,
                exported_args,
                self.outputs,
            )
        else:
            # 添加代码生成的注释
            self.codegen_comment(wrapper)
            # 生成参数列表
            args = [*self.codegen_args(), *self.codegen_kwargs()]
            # 生成回退内核
            V.graph.wrapper_code.generate_fallback_kernel(self, args)
            # 如果布局是 Layout 的实例
            if isinstance(self.layout, Layout):
                # 生成大小断言
                self.codegen_size_asserts(wrapper)

        # 生成未支持符号定义的代码
        self.codegen_unbacked_symbol_defs(wrapper)
    # 定义一个静态方法，将 PyTorch 张量转换为固定布局对象
    @staticmethod
    def tensor_to_layout(output: torch.Tensor):
        return FixedLayout(
            output.device,
            output.dtype,
            convert_shape_to_inductor(output.size()),
            convert_shape_to_inductor(output.stride()),
        )
    
    # 类方法，根据给定的内核创建对象实例
    @classmethod
    def create(cls, kernel, *args, **kwargs):
        # 模拟不正确内核的集合
        fake_incorrect_kernels = (aten._fused_moving_avg_obs_fq_helper_functional,)
        # 如果内核不在上述集合中，使用 fake_mode 上下文，否则使用 nullcontext
        context = (
            V.graph.fake_mode if kernel not in fake_incorrect_kernels else nullcontext()
        )
        with context:
            # 处理内核，获取示例输出、张量参数、非张量参数、展平参数、未支持的绑定
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings,
            ) = cls.process_kernel(kernel, *args, **kwargs)
    
        # 查找设备信息
        device = cls.find_device(tensor_args, example_output)
        # 如果示例输出为空，创建空布局对象
        if example_output is None:
            packed = cls(
                NoneLayout(device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings=unbacked_bindings,
            )
        else:
            # 否则，创建多输出布局对象
            assert device, "Not sure where to find device info"
            packed = cls(
                MultiOutputLayout(device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings=unbacked_bindings,
            )
    
        # 生成输出函数，递归处理多种类型的输出
        def generate_output(output, indices):
            if isinstance(output, (list, tuple)):
                return type(output)(
                    generate_output(output[i], indices + [(type(output), i)])
                    for i in range(len(output))
                )
            elif isinstance(output, dict):
                return {
                    key: generate_output(val, indices + [(type(output), key)])
                    for key, val in output.items()
                }
            elif isinstance(output, torch.Tensor):
                return MultiOutput(
                    cls.tensor_to_layout(output),
                    packed,
                    indices,
                )
            elif isinstance(output, int):
                return output
            elif isinstance(output, torch.SymInt):
                return output.node.expr
            else:
                # 断言如果输出为 None，则抛出异常
                assert (
                    output is None
                ), f"FallbackKernel output type {type(output)} is not supported"
                return None
    
        # 生成实际的输出结果
        outputs = generate_output(example_output, [])
        # 如果输出是列表、元组或字典，直接赋值给 packed.outputs
        if isinstance(outputs, (list, tuple, dict)):
            packed.outputs = outputs  # type: ignore[assignment]
        else:
            # 否则，包装成列表
            packed.outputs = [outputs]
        return outputs
    
    # 应用约束方法，调用父类的 apply_constraint 方法
    def apply_constraint(self):
        return super().apply_constraint()
@dataclasses.dataclass
class ComplexView(FallbackKernel):
    """View a complex number as two dtyped numbers or vice versa"""

    def should_allocate(self):
        return False

    def get_inputs_that_alias_output(self):
        # Signal to codegen that our output buffer isn't safe to reuse
        # 返回一个列表，包含当前对象输入中可能与输出别名的对象名称
        return [self.inputs[0].get_name()]

    def __init__(
        self,
        layout,
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        *,
        unbacked_bindings=None,
    ):
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            unbacked_bindings=unbacked_bindings,
        )


@dataclasses.dataclass
class MultiOutputLayout(IRNode):
    device: torch.device


class MultiOutput(ExternKernel):
    # Given an input MultiOutputLayout buffer, indexes out an actual buffer
    # from that result.  This doesn't actually produce multiple outputs,
    # that's MultiOutputLayout!
    def codegen_list_tuple_access(self, basename, indices):
        if len(indices) > 0:
            itype, i = indices[0]
            if issubclass(itype, list):
                # 递归调用，处理列表索引的访问
                return self.codegen_list_tuple_access(f"{basename}[{i}]", indices[1:])
            elif issubclass(itype, tuple):
                # 对于元组索引，生成 CPP 包装器代码，使用 std::get<> 访问元素
                tuple_access = V.graph.wrapper_code.codegen_tuple_access(
                    basename, self.get_name(), str(i)
                )
                return self.codegen_list_tuple_access(tuple_access, indices[1:])
            elif issubclass(itype, dict):
                # 处理字典索引的访问
                return self.codegen_list_tuple_access(f"{basename}['{i}']", indices[1:])
            else:
                raise AssertionError("non supported index type: ", itype)
        else:
            return basename

    def codegen(self, wrapper):
        # 生成多输出的代码
        wrapper.codegen_multi_output(
            self.get_name(),
            self.codegen_list_tuple_access(self.inputs[0].get_name(), self.indices),
        )

    def __init__(self, layout, input, indices: List[Tuple[Any, ...]]):
        super().__init__(None, layout, [input], ())
        self.name = V.graph.register_buffer(self)
        self.indices = indices

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        # 获取未支持符号使用的集合
        return self.inputs[0].get_unbacked_symbol_uses()

    def should_allocate(self):
        return False

    def get_inputs_that_alias_output(self):
        # 返回一个列表，包含当前对象输入中可能与输出别名的对象名称
        return [
            inp.get_name()
            for inp in self.inputs
            if isinstance(inp, FallbackKernel)
            and len(inp.get_inputs_that_alias_output()) > 0
        ]


@dataclasses.dataclass
class MutableBox(IRNode):
    """
    TensorBox / StorageBox allow in-place mutation of Tensors
    """

    data: IRNode
    # 当尝试获取不存在的属性时，尝试从self.data中获取对应的属性
    def __getattr__(self, name):
        fn = getattr(self.data, name)
        # 如果获取到的属性是可调用的，则返回该属性
        if callable(fn):
            return fn
        # 否则抛出属性错误，指明该属性不可调用
        raise AttributeError(f"{type(self.data).__name__}.{name} not callable")

    # 实现实例的实现方法，委托给self.data的实现方法
    def realize(self):
        return self.data.realize()

    # 返回self.data中未备份的符号使用集合
    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        return self.data.get_unbacked_symbol_uses()

    # 生成引用代码的方法，可以选择将结果写入writer
    def codegen_reference(self, writer=None):
        return self.data.codegen_reference(writer)

    # layout属性的装饰器，返回self.data的布局信息
    @property
    def layout(self):
        return self.data.get_layout()

    # 获取布局信息的方法，委托给self.layout
    def get_layout(self):
        return self.layout

    # 获取self.data的大小信息
    def get_size(self):
        return self.data.get_size()

    # dtype属性的装饰器，返回self.data的数据类型
    @property
    def dtype(self):
        return self.data.dtype

    # 自定义实例的字符串表示形式
    def __str__(self):
        # 如果self.data是MutableBox的实例
        if isinstance(self.data, MutableBox):
            line0 = f"{type(self).__name__}({type(self.data).__name__}("
            endl = "))"
            inner = self.data.data
        else:
            line0 = f"{type(self).__name__}("
            inner = self.data
            endl = ")"

        # 构建实例的字符串表示形式
        lines = [
            line0,
            indent(str(inner)),  # 使用缩进函数将inner的字符串表示形式缩进
            endl,
        ]
        return "\n".join(lines)

    # 将__repr__方法指向__str__，以便在打印实例时使用相同的字符串表示形式
    __repr__ = __str__
class TensorBox(MutableBox):
    # TensorBox 类，继承自 MutableBox 类

    @staticmethod
    def create(data):
        # 静态方法 create，接受一个参数 data
        return TensorBox(StorageBox(data))
        # 创建并返回一个 TensorBox 对象，传入一个 StorageBox 对象作为参数


class StorageBox(MutableBox):
    # StorageBox 类，继承自 MutableBox 类

    def is_input_buffer(self):
        # 判断当前实例是否为输入缓冲区
        if isinstance(self.data, (InputBuffer, ReinterpretView)):
            return self.data.get_name() in V.graph.graph_inputs
            # 返回当前数据对象的名称是否在图输入列表中
        return False
        # 若不是输入缓冲区类型，则返回 False

    def is_module_buffer(self):
        # 判断当前实例是否为模块缓冲区
        return (
            isinstance(self.data, (ConstantBuffer))
            and self.data.get_name() in V.graph.constants
        )
        # 返回当前数据对象是否为常量缓冲区并且其名称在图常量列表中

    def realize(self):
        # 实现方法，用于实例化当前对象
        if isinstance(
            self.data,
            (
                ComputedBuffer,
                InputsKernel,
                InputBuffer,
                ReinterpretView,
                TemplateBuffer,
            ),
        ):
            return self.data.get_name()
            # 如果当前数据对象属于某些类型，则返回其名称
        assert isinstance(self.data, (Pointwise, Reduction, Scan, Sort)), type(
            self.data
        )
        # 断言当前数据对象为 Pointwise、Reduction、Scan 或 Sort 类型
        origin_node = self.data.get_origin_node()
        traceback = self.data.get_traceback()
        self.data = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=self.data.get_device(),
                dtype=self.data.get_dtype(),
                size=self.data.get_size(),
            ),
            data=self.data,
        )
        # 创建 ComputedBuffer 对象，并设置其属性
        self.data.name = V.graph.register_buffer(self.data)
        self.data.origins = self.origins
        self.data.origin_node = origin_node
        self.data.traceback = traceback
        return self.data.name
        # 返回 ComputedBuffer 对象的名称

    def realize_hint(self):
        """
        Called on buffers we expect to be forced to realize later.
        """
        # 提示方法，用于标记需要稍后强制实例化的缓冲区
        if (
            isinstance(self.data, (Pointwise, Reduction))
            and self.num_reads() > 1
            and self.is_pointwise_non_scalar_tensor_num_reads_larger_than_one()
        ):
            self.realize()
            # 如果数据对象为 Pointwise 或 Reduction 类型，并且读取次数大于 1，则实例化它

    def has_exceeded_max_reads(self):
        # 判断当前数据对象的读取次数是否超过最大限制
        return isinstance(self.data, Pointwise) and (
            self.num_reads() > config.realize_acc_reads_threshold
            or self.has_large_inner_fn()
        )
        # 返回当前数据对象是否为 Pointwise 类型，并且读取次数超过配置的阈值或具有大型内部函数

    def mark_reuse(self, users):
        """
        A heuristic to decide if we should realize a tensor
        that is used multiple times.
        """
        # 标记重用的方法，用于决定是否需要实例化一个被多次使用的张量
        def should_realize_on_cpu(loops: Union[Pointwise, Reduction]):
            """
            The heuristic for realizing reused result of heavy ops on cpu
            """
            heavy_ops = ["exp", "sigmoid"]  # a list of heavy ops
            fn_str = loops.inner_fn_str()
            return any((op + "(") in fn_str for op in heavy_ops)
            # 判断内部函数字符串是否包含重操作的任何一个，用于在 CPU 上实例化重复使用的结果

        if (
            users > 1
            and isinstance(self.data, (Pointwise, Reduction))
            and (
                self.num_reads() > config.realize_reads_threshold
                or self.has_large_inner_fn()
                or (is_cpu(self.data) and should_realize_on_cpu(self.data))
            )
        ):
            self.realize()
            # 如果用户数大于 1，并且数据对象为 Pointwise 或 Reduction 类型，并且满足一定条件，则实例化它

    @cache_on_self
    # 缓存装饰器，用于将该方法的结果缓存到对象自身
    # 返回对象自身的数据内容
    def num_reads(self):
        data = self.data  # 获取对象的数据属性
        # 如果数据是 InputsKernel、InputBuffer 或 ReinterpretView 类型的实例，返回 1
        if isinstance(data, (InputsKernel, InputBuffer, ReinterpretView)):
            return 1
        # 如果数据是 ComputedBuffer 类型的实例，获取其读写信息
        if isinstance(data, ComputedBuffer):
            read_writes = data.get_read_writes()
        else:
            # 否则，确保数据是 Pointwise 或 Reduction 类型的实例
            assert isinstance(data, (Pointwise, Reduction)), type(data)
            # 创建一个 ComputedBuffer 对象，用于获取数据的读写信息
            read_writes = ComputedBuffer(
                name=None,
                layout=FlexibleLayout(
                    device=data.get_device(),
                    dtype=data.get_dtype(),
                    size=data.get_size(),
                ),
                data=data,
            ).get_read_writes()
        # 返回读取操作的次数
        return len(read_writes.reads)

    # 对象方法的缓存装饰器，用于判断是否是 Pointwise 非标量张量且读取操作大于一次
    @cache_on_self
    def is_pointwise_non_scalar_tensor_num_reads_larger_than_one(self):
        # 如果数据是 Pointwise 类型的实例，并且所有读取操作均不是 StarDep 的依赖关系
        return (
            (sum(read.index != 0 for read in self.data.get_reads()) > 1)
            if isinstance(self.data, Pointwise)
            and all(
                not isinstance(read, dependencies.StarDep)
                for read in self.data.get_reads()
            )
            # 否则，返回 True
            else True
        )
@dataclasses.dataclass
class Subgraph(IRNode):
    name: str
    graph_module: torch.fx.GraphModule
    graph: Optional[GraphLowering] = None


# 定义了一个子图类Subgraph，继承自IRNode，表示一个计算图的子部分
# 包含属性：
# - name：子图的名称，类型为字符串
# - graph_module：torch.fx.GraphModule对象，表示子图的计算模块
# - graph：可选的GraphLowering对象，表示子图的降阶图，如果有的话


def _has_aliased_buffers(buffers):
    buffers = [
        buffer.unwrap_view() if isinstance(buffer, ReinterpretView) else buffer
        for buffer in buffers
    ]
    # assuming the same buffer is represented by the same IRNode object
    return len({id(buffer) for buffer in buffers}) < len(buffers)


# 检查是否存在别名缓冲区的函数_has_aliased_buffers
# 参数:
# - buffers: 缓冲区列表，可能包含ReinterpretView类型的对象
# 返回值:
# - 如果存在相同缓冲区被同一个IRNode对象表示，则返回True；否则返回False


@dataclasses.dataclass
class Conditional(ExternKernel):
    predicate: Optional[IRNode] = None
    operands: Optional[List[TensorBox]] = None
    true_subgraph: Optional[Subgraph] = None
    false_subgraph: Optional[Subgraph] = None
    outputs: Optional[List[MultiOutput]] = None

    def __init__(
        self,
        predicate: IRNode,
        operands: List[TensorBox],
        true_subgraph: Subgraph,
        false_subgraph: Subgraph,
        layout: MultiOutputLayout,
    ):
        self.predicate = predicate
        self.operands = operands
        self.true_subgraph = true_subgraph
        self.false_subgraph = false_subgraph

        inputs = []
        if not isinstance(predicate, ShapeAsConstantBuffer):
            inputs.append(predicate)
        inputs.extend(operands)

        super().__init__(
            name=None,
            layout=layout,  # type: ignore[arg-type]
            inputs=inputs,  # type: ignore[list-item]
        )

        self.name = V.graph.register_buffer(self)

    @classmethod
    def create(
        cls,
        predicate: TensorBox,
        true_fn: Subgraph,
        false_fn: Subgraph,
        operands: List[TensorBox],


# 表示一个条件操作的类Conditional，继承自ExternKernel
# 包含属性：
# - predicate: 可选的IRNode，表示条件谓词
# - operands: 可选的TensorBox列表，表示操作数列表
# - true_subgraph: 可选的Subgraph，表示条件为真时执行的子图
# - false_subgraph: 可选的Subgraph，表示条件为假时执行的子图
# - outputs: 可选的MultiOutput列表，表示操作的输出结果列表
# 方法：
# - __init__: 初始化函数，设置各个属性并调用父类的构造函数初始化
# - create: 类方法，用于创建Conditional对象，接受条件谓词、真子图、假子图和操作数列表作为参数


    def codegen(self, wrapper):
        wrapper.codegen_conditional(self)


@dataclasses.dataclass
class WhileLoop(ExternKernel):
    carried_inputs: Optional[List[TensorBox]] = None
    additional_inputs: Optional[List[TensorBox]] = None
    cond_subgraph: Optional[Subgraph] = None
    body_subgraph: Optional[Subgraph] = None
    outputs: Optional[List[MultiOutput]] = None

    def __init__(
        self,
        carried_inputs: List[TensorBox],
        additional_inputs: List[TensorBox],
        cond_subgraph: Subgraph,
        body_subgraph: Subgraph,
        layout: MultiOutputLayout,
    ):
        self.carried_inputs = carried_inputs
        self.additional_inputs = additional_inputs
        self.cond_subgraph = cond_subgraph
        self.body_subgraph = body_subgraph

        super().__init__(
            name=None,
            layout=layout,  # type: ignore[arg-type]
            inputs=carried_inputs + additional_inputs,  # type: ignore[list-item]
        )

        self.name = V.graph.register_buffer(self)

    @classmethod
    def create(
        cls,
        cond_fn: Subgraph,
        body_fn: Subgraph,
        carried_inputs: List[TensorBox],
        additional_inputs: List[TensorBox],


# 表示一个while循环操作的类WhileLoop，继承自ExternKernel
# 包含属性：
# - carried_inputs: 可选的TensorBox列表，表示携带的输入
# - additional_inputs: 可选的TensorBox列表，表示额外的输入
# - cond_subgraph: 可选的Subgraph，表示条件子图
# - body_subgraph: 可选的Subgraph，表示循环体子图
# - outputs: 可选的MultiOutput列表，表示操作的输出结果列表
# 方法：
# - __init__: 初始化函数，设置各个属性并调用父类的构造函数初始化
# - create: 类方法，用于创建WhileLoop对象，接受条件子图、循环体子图、携带输入和额外输入作为参数


    def codegen(self, wrapper):
        wrapper.codegen_while_loop(self)


class EffectfulKernel(FallbackKernel):


# 表示一个有副作用的内核操作的类EffectfulKernel，继承自FallbackKernel
# 这部分代码缺失了，没有提供完整的类定义和实现
    # 初始化方法，用于创建一个新的实例对象
    def __init__(
        self,
        layout,
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        kwargs=None,
        *,
        unbacked_bindings=None,
    ):
        # 调用父类的初始化方法
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=None,  # kwargs 参数传递为 None
            unbacked_bindings=unbacked_bindings,  # 设置 unbacked_bindings 参数
        )

        # 导入 get_effect_key 函数，用于获取特定操作的效果键值
        from torch._higher_order_ops.effects import get_effect_key

        # 根据 kernel、tensor_args、nontensor_args 和 kwargs 获取效果类型键值
        effect_type = get_effect_key(kernel, (*nontensor_args, *tensor_args), kwargs)
        # 断言效果类型不为 None
        assert effect_type is not None
        # 将效果类型存储在实例对象中
        self.effect_type = effect_type
        # 获取上一个效果缓冲区（如果存在）
        self.prev_effect_buffer = V.graph.effectful_ops.get(effect_type, None)
        # 将当前对象存储到效果操作的全局字典中
        V.graph.effectful_ops[effect_type] = self

    # 获取读写操作方法
    def get_read_writes(self):
        # 调用父类的 get_read_writes 方法获取读写操作
        read_writes = super().get_read_writes()

        # 如果存在上一个效果缓冲区，将其读操作添加到读写操作集合中
        if self.prev_effect_buffer is not None:
            read_writes.reads.add(
                dependencies.StarDep(self.prev_effect_buffer.get_name())
            )

        # 返回读写操作集合
        return read_writes

    # 判断当前对象是否具有副作用
    def has_side_effects(self):
        return True  # 总是返回 True，表示具有副作用
@dataclasses.dataclass
class TorchBindObject(IRNode):
    name: str
    value: torch._C.ScriptObject

    def get_name(self):
        return self.name  # 返回对象的名称

    def get_device(self):
        return None  # is there a device?? 返回 None 表示设备未知

    def codegen_reference(self, writer=None):
        return self.name  # 返回对象的名称作为代码生成的引用


class InterpreterShim(torch.fx.Interpreter):
    @staticmethod
    @functools.lru_cache(None)
    def _dummy_gm():
        return torch.fx.symbolic_trace(identity)  # 返回一个经过符号化追踪的 identity 函数的结果

    def __init__(self, graph, submodules):
        # 使用 _dummy_gm() 的结果作为超类初始化的参数，避免昂贵的 GraphModule 构造
        super().__init__(self._dummy_gm(), garbage_collect_values=False)
        self.module = self  # 设置 self.module 为当前对象的引用，忽略类型检查
        self.graph = graph  # 设置对象的图属性
        self.submodules = submodules  # 设置对象的子模块属性
        self.extra_traceback = False  # 设置额外的回溯标志为 False
        self.fetch_attr = submodules.__getitem__  # 设置 fetch_attr 方法为获取子模块的方法
        self.current_node = None  # 初始化当前节点为 None

    def run_node(self, n: torch.fx.Node) -> Any:
        self.current_node = n  # 设置当前节点为参数 n
        return super().run_node(n)  # 调用超类的 run_node 方法执行节点 n

    def run(self, *args, **kwargs):
        with V.set_interpreter_handler(self):  # 使用 V.set_interpreter_handler(self) 上下文
            return super().run(*args, **kwargs)  # 调用超类的 run 方法执行给定参数


class LoopBody:
    """
    Captures the body of a Loops subclass into an FX graph.  Persists any
    indexing simplifications and makes it easier to analyze loop bodies.
    """

    def __init__(self, fn, args, var_ranges):
        super().__init__()  # 调用超类初始化方法
        self.var_ranges = var_ranges  # 设置变量范围
        self.indexing_exprs = {}  # 初始化索引表达式字典
        self.indexing_exprs_name = {}  # 初始化索引表达式名称字典
        self.reads = []  # 初始化读操作列表
        self.writes = []  # 初始化写操作列表
        self.reads_name2expr = {}  # 初始化读操作名称到表达式字典
        self.writes_name2expr = {}  # 初始化写操作名称到表达式字典
        self.other = []  # 初始化其他操作列表
        self.submodules = {"get_index": self.get_index}  # 设置子模块字典，包含 get_index 方法
        self.subblocks = {}  # 初始化子块字典
        self.indirect_vars = []  # 初始化间接变量列表
        self.root_block = LoopBodyBlock(self, fn, args)  # 初始化根块属性为 LoopBodyBlock 对象
        self.indexing = None  # 初始化索引属性为 None

    @cache_on_self
    def get_nodes(self):
        all_graphs = itertools.chain(
            (self.root_block.graph,),
            (block.graph for block in self.subblocks.values()),
        )
        return [node for graph in all_graphs for node in graph.nodes]  # 返回所有图中的节点列表

    @cache_on_self
    def bounds(self):
        # 局部导入 BoundVars 类避免在此处导入所有代码
        from .bounds import BoundVars
        return BoundVars(self)  # 返回当前 LoopBody 对象的 BoundVars 对象

    def debug_str(self):
        lines = [f"var_ranges = {dict(self.var_ranges)}"]  # 变量范围的调试信息
        lines.extend([f"{name} = {val}" for name, val in self.indexing_exprs.items()])  # 扩展索引表达式的调试信息
        lines.extend(
            [
                block.debug_str(name)
                for name, block in itertools.chain(
                    [("body", self.root_block)], self.subblocks.items()
                )
            ]
        )
        return "\n".join(lines)  # 返回调试信息字符串
    # 将表达式添加到指定类别的列表中
    def add_index_expr(self, expr: sympy.Expr, category, buf_name):
        getattr(self, category).append(expr)  # 将表达式添加到指定类别的列表中
        if buf_name is not None:
            getattr(self, f"{category}_name2expr")[buf_name] = expr  # 如果 buf_name 不为 None，则将表达式与其关联存储
        if expr not in self.indexing_exprs_name:
            # 如果表达式不在索引表达式名称字典中，则为其生成一个唯一的名称
            name = f"index{len(self.indexing_exprs)}"
            self.indexing_exprs_name[expr] = name  # 将表达式与生成的名称关联存储
            self.indexing_exprs[name] = expr  # 将表达式与其名称关联存储到索引表达式字典中
        return self.indexing_exprs_name[expr]  # 返回表达式的名称

    # 向生成的代码中添加子模块，映射到 FX call_module 操作码
    def add_submodule(self, block, prefix):
        """Not actually for nn.Modules, but subblocks in generated code are mapped to FX call_module opcodes"""
        if prefix[-1].isnumeric() and prefix not in self.submodules:
            name = prefix  # 如果前缀以数字结尾且不在子模块中，则直接使用前缀作为名称
        else:
            name = f"{prefix}{len(self.submodules)}"  # 否则，使用前缀加上当前子模块数量作为名称
        self.submodules[name] = block  # 将子模块与其名称关联存储
        return name  # 返回添加的子模块的名称

    # 添加间接索引变量，并返回其生成的符号
    def add_indirect(self, size):
        var = sympy_index_symbol_with_prefix(SymT.INDIRECT, len(self.indirect_vars))  # 使用特定前缀和索引变量数量生成符号
        self.indirect_vars.append(var)  # 将生成的间接索引变量符号添加到列表中
        return var  # 返回生成的间接索引变量符号

    # 替换用于间接索引的旧变量为新变量
    def replace_indirect(self, old, new):
        """Swap in a variable used in indirect indexing"""
        if str(old) == str(new):
            return  # 如果新旧变量相同，则无需替换
        assert self.indexing is not None
        # 对所有索引表达式中使用的旧变量进行替换为新变量
        self.indexing = {k: sympy_subs(v, {old: new}) for k, v in self.indexing.items()}

    # 根据名称获取索引表达式
    def get_index(self, name):
        assert self.indexing is not None  # 确保索引表达式不为空
        return self.indexing[name]  # 返回指定名称的索引表达式

    # 根据传入的索引生成完整的索引表达式字典
    def indexing_from_args(self, indices):
        index = [*itertools.chain.from_iterable(indices)]  # 将嵌套索引展开为一维列表
        assert len(index) == len(self.var_ranges), (index, self.var_ranges)  # 确保索引长度与变量范围字典长度相同
        assert all(v not in self.var_ranges for v in index)  # 确保索引中的所有元素不在变量范围字典中
        replacements = dict(zip(self.var_ranges.keys(), index))  # 创建变量名到索引值的映射字典
        return {
            name: sympy_subs(expr, replacements)  # 使用映射字典替换索引表达式中的变量
            for name, expr in self.indexing_exprs.items()
        }

    # 在调用实例时，根据传入的索引生成索引表达式，并执行根块操作，最后清空索引表达式
    def __call__(self, *indices):
        self.indexing = self.indexing_from_args(indices)  # 根据传入的索引生成完整的索引表达式字典
        result = self.root_block()  # 执行根块操作
        self.indexing = None  # 清空索引表达式字典
        return result  # 返回操作结果
class LoopBodyBlock:
    """
    Captures the body of a Loops subclass into an FX graph.
    In normal cases there will be a 1:1 mapping between LoopBody and
    LoopBodyBlock, hower in the case of ops.masked() the masked out
    operations will manifest as an extra LoopBodyBlock.
    """

    def __call__(self):
        # 获取当前对象的图形表示
        graph = self.graph
        # 获取当前对象的子模块列表
        submodules = self.body.submodules

        # 使用InterpreterShim运行图形并返回结果
        return InterpreterShim(graph, submodules).run(V.get_ops_handler())

    def debug_str(self, name="block"):
        # 使用torch.fx.GraphModule创建图模块并提取其代码
        code = torch.fx.GraphModule(self.body.submodules, self.graph).code
        return re.sub(
            # 去除代码中以`; del var0`结尾的部分，美化输出
            r";[^\n]*",
            "",
            code.strip().replace("def forward(", f"def {name}("),
        )


class _CollectiveKernel(FallbackKernel):
    def should_allocate(self):
        # 永远返回False，指示不需要分配资源
        return False

    def has_side_effects(self):
        # 永远返回True，指示存在副作用
        return True

    # 此方法与FallbackKernel.set_cpp_kernel()完全相同，仅省略了针对输入别名和突变的检查部分
    def set_cpp_kernel(self, kernel):
        from .codegen.wrapper import get_cpp_op_schema

        # 设置C++内核相关属性
        self.cpp_kernel_name = kernel._schema.name
        self.cpp_kernel_overload_name = kernel._schema.overload_name
        self.cpp_kernel_key = f"{self.cpp_kernel_name.replace('::', '_')}_{self.cpp_kernel_overload_name}"  # type: ignore[union-attr]

        # 获取C++操作的模式
        self.cpp_op_schema = get_cpp_op_schema(kernel)
        # 获取C++内核的关键字参数列表
        self.ordered_kwargs_for_cpp_kernel = [
            x.name for x in kernel._schema.arguments if x.kwarg_only
        ]

    # NOTE: [In-Place Collective Safety]
    # 在原地集合操作的开始和结束之间，输入缓冲区会经历易变的读写操作。
    # 它们不能被另一个内核读取、写入或重用。为了确保这些约束条件，我们将集合操作
    # 模拟为两步变异：集合 -> 等待张量。
    @classmethod
    def create_inplace(
        cls, kernel, inputs: Union[TensorBox, List[TensorBox]], *args, **kwargs
    ) -> None:
        cpp_kernel_name = kernel._name
        python_kernel_name = cpp_kernel_name.replace("::", ".")
        # 在虚拟图形模式下处理内核和输入参数
        with V.graph.fake_mode:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings,
            ) = cls.process_kernel(kernel, inputs, *args, **kwargs)
        # 断言确保没有未支持的绑定
        assert not unbacked_bindings, f"{kernel} {unbacked_bindings}"
        # 对于所有张量参数，实现它们
        for tensor_arg in tensor_args:
            tensor_arg.realize()

        # 创建_CollectiveKernel实例
        packed = cls(
            NoneLayout(tensor_args[0].get_device()),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
        )
        # 设置C++内核和Python内核名称
        packed.cpp_kernel_name = cpp_kernel_name
        packed.python_kernel_name = python_kernel_name

        # 标记节点为突变，用于所有输入参数
        mark_node_as_mutating(packed, *pytree.tree_leaves(inputs))
    # NOTE: [Out-of-Place Collective Safety]
    # Between the initiation and completion of an out-of-place collective:
    #
    # Input buffers:
    # - Are subject to volatile reads
    # - Can be read by another kernel
    # - Must not be written to or reused by another kernel
    #
    # Output buffers:
    # - Are subject to volatile writes
    # - Must not be read, written to or reused by another kernel
    #
    # To ensure the safety of input buffers without sacrificing read
    # availability, we add input buffers as read deps of wait_tensor kernels.
    #
    # To ensure the safety of output buffers, we model wait_tensor as a
    # mutation to the output buffer. Note we also assumes the user program being
    # correct and the output buffer is not consumed by kernels other than
    # wait_tensor.
    #
    # TODO(yifu): add a pre-grad pass to validate the correctness of collective
    # usage in the user program.
    @classmethod
    def create_out_of_place(
        cls, kernel, inputs: Union[TensorBox, List[TensorBox]], *args, **kwargs
    ):
        # 获取 C++ 内核名称
        cpp_kernel_name = kernel._name
        # 将 C++ 内核名称转换为 Python 风格
        python_kernel_name = cpp_kernel_name.replace("::", ".")
        # 进入虚拟图模式
        with V.graph.fake_mode:
            # 处理内核，获取处理后的输出、张量参数、非张量参数、展开参数、未支持的绑定
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings,
            ) = cls.process_kernel(kernel, inputs, *args, **kwargs)
        # 断言没有未支持的绑定
        assert not unbacked_bindings, f"{kernel}, {unbacked_bindings}"
        # 对每个张量参数执行实现
        for tensor_arg in tensor_args:
            tensor_arg.realize()

        # 如果示例输出是列表
        if isinstance(example_output, list):
            # 查找设备
            device = cls.find_device(tensor_args, example_output)
            # 创建输出对象，并传入设备、内核、张量参数、非张量参数、展开参数
            packed = cls(
                MultiOutputLayout(device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )
            packed.cpp_kernel_name = cpp_kernel_name
            packed.python_kernel_name = python_kernel_name
            # 将每个张量转换为输出布局对象，并创建多输出对象列表
            packed.outputs = [
                MultiOutput(
                    cls.tensor_to_layout(tensor),
                    packed,
                    [(list, i)],
                )
                for i, tensor in enumerate(example_output)
            ]
            return packed.outputs
        else:
            # 否则，创建输出对象，传入示例输出的布局、内核、张量参数、非张量参数、展开参数
            packed = cls(
                cls.tensor_to_layout(example_output),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )
            packed.cpp_kernel_name = cpp_kernel_name
            packed.python_kernel_name = python_kernel_name
            packed.outputs = [packed]
            return packed
# 定义一个名为 _WaitKernel 的类，继承自 _CollectiveKernel 类
class _WaitKernel(_CollectiveKernel):
    # 获取此核函数的易变读取项
    def get_volatile_reads(self):
        # 获取输入的第一个元素
        inp = self.inputs[0]
        # 如果输入是 _CollectiveKernel 类型
        if isinstance(inp, _CollectiveKernel):
            # 表示是非就地操作的单一输出
            return [inp.inputs[0]]
        # 如果输入是 MultiOutput 类型
        elif isinstance(inp, MultiOutput):
            # 这里可能有两种情况：
            # 1. 非就地操作的多输出集合
            # 2. 就地操作的集合，其输入来自另一个 MultiOutput
            # 获取集合中的第一个元素
            coll = inp.inputs[0]
            # 如果是情况1
            if isinstance(coll, _CollectiveKernel):
                # 获取索引信息
                _, idx = inp.indices[0]
                # 返回集合中指定索引的输入
                return [coll.inputs[idx]]
            # 如果是情况2
            return []
        else:
            # 对于就地操作，易变读取不需要额外的依赖处理
            # 因为输入会被改变
            return []

    # 创建一个等待操作的静态方法
    @classmethod
    def create_wait(cls, kernel, inp: TensorBox) -> None:
        # 在虚拟图模式下执行以下操作
        with V.graph.fake_mode:
            # 处理核函数和输入，返回一系列处理结果
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings,
            ) = cls.process_kernel(kernel, inp)
        # 断言确保没有未支持的绑定
        assert not unbacked_bindings, f"{kernel} {unbacked_bindings}"
        # 创建一个 _WaitKernel 对象
        packed = cls(
            NoneLayout(inp.get_device()),  # 使用 NoneLayout 初始化设备
            kernel,  # 核函数
            tensor_args,  # 张量参数
            non_tensor_args,  # 非张量参数
            unflatten_args,  # 未展开的参数
        )
        # 标记节点为改变状态
        mark_node_as_mutating(packed, inp)

    # 获取读写操作的方法
    def get_read_writes(self):
        # 调用父类方法获取读写操作
        read_writes = super().get_read_writes()
        # 获取易变读取项
        volatile_reads = self.get_volatile_reads()
        # 遍历易变读取项，添加到读写操作的读取集合中
        for vr in volatile_reads:
            read_writes.reads.add(dependencies.StarDep(vr.get_name()))
        # 返回读写操作对象
        return read_writes


# 注意：这里递归结构反映了 val_to_arg_str，避免在不支持的类型上调用 free_unbacked_symbols
# 这里也包括不要在 "奇异" 类型上调用 free_unbacked_symbols
def maybe_free_unbacked_symbols(s):
    # 如果 s 是 SymTypes 或 sympy.Expr 类型
    if isinstance(s, (SymTypes, sympy.Expr)):
        # 在返回位置，不可能进入此分支
        return free_unbacked_symbols(s)
    # 如果 s 是 tuple 或 list 类型
    elif isinstance(s, (tuple, list)):
        # 初始化一个集合
        r = set()
        # 遍历元组或列表中的每个元素，合并结果集合
        for t in s:
            r |= maybe_free_unbacked_symbols(t)
        # 返回结果集合
        return r
    # 如果 s 是 torch.Tensor 类型
    elif isinstance(s, torch.Tensor):
        # 在常量参数位置，不可能进入此分支
        return free_unbacked_symbols(s)
    else:
        # 返回一个空集合，表示不需要释放未支持符号
        return set()
```