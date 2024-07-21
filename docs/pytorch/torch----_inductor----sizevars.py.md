# `.\pytorch\torch\_inductor\sizevars.py`

```py
# mypy: allow-untyped-defs
# 导入需要的模块和函数
import functools
import itertools
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import sympy  # 导入sympy库
from sympy import Expr  # 导入sympy表达式类

# 导入相关符号形状环境和运行时工具函数
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.symbol import symbol_is_type, SymT
from torch.utils._sympy.value_ranges import bound_sympy

# 导入本地定义的函数和类
from .runtime.runtime_utils import is_power_of_2
from .utils import (
    sympy_index_symbol,
    sympy_index_symbol_with_prefix,
    sympy_subs,
    VarRanges,
)
from .virtualized import V  # 导入虚拟化类V

# 设置日志记录器
log = logging.getLogger(__name__)

# This class is a little awkward, because ShapeEnv is doing most of the heavy
# lifting and in some cases we should be directly passing through to ShapeEnv,
# but there is some extra inductor logic that needs to be handled here
# SizeVarAllocator类负责管理尺寸变量的分配
class SizeVarAllocator:
    def __init__(self, shape_env=None):
        super().__init__()  # 调用父类的构造方法
        if shape_env is None:
            shape_env = ShapeEnv()  # 如果未提供符号形状环境，则创建一个新的
        self.shape_env = shape_env  # 设置符号形状环境
        self.var_to_val = self.shape_env.var_to_val  # 将变量到值的映射设置为符号形状环境的映射
        self.replacements: Dict[sympy.Symbol, Expr] = self.shape_env.replacements  # 设置符号替换的字典
        # Maps of dynamic sizes that have to be precomputed on the host to the kernel args.
        # The basic idea is if we have some complicated sympy expression
        # f(s0), we may choose to precompute it on the host and then replace
        # all occurrences of that sympy expression with ps0, so that when we
        # codegen we simply reference ps0 directly without repeating
        # f(s0).  Unlike regular size variables, ps variables cannot be
        # guarded upon; so if we are asked to guard on a Sympy expression
        # which potentially could have already had a precomputed replacement
        # on it, we are obligated to invert the precomputed replacements
        # (inv_precomputed_replacements).
        self.precomputed_replacements: Dict[Expr, sympy.Symbol] = dict()  # 预计算的替换映射
        self.inv_precomputed_replacements: Dict[sympy.Symbol, Expr] = dict()  # 反向预计算的替换映射
        self.stride_vars = self.make_stride_vars_cache()  # 创建步长变量缓存
        self.simplify_with_ranges = self.make_simplify_with_ranges_cache()  # 创建带有范围简化的缓存
        self._simplify_loops = self.make_simplify_loops_cache()  # 创建简化循环的缓存

    # 使用符号替换和扩展来简化给定的表达式
    def simplify(self, expr: Expr):
        return sympy.expand(expr).xreplace(self.replacements)
    def make_simplify_with_ranges_cache(self) -> Callable[[Expr, VarRanges], Expr]:
        """
        self._simplify_with_ranges() can be expensive, cache its results
        """
        # 定义一个空的字典作为缓存
        cache: Dict[Tuple[Any, ...], Expr] = dict()
        # 获取当前替换操作的数量
        replacement_count = len(self.replacements)

        # 定义一个内部函数 simplify_with_ranges，接受表达式和变量范围作为参数，并返回简化后的表达式
        def simplify_with_ranges(expr: Expr, var_ranges: VarRanges) -> Expr:
            nonlocal replacement_count
            # 如果替换操作的数量发生了变化，则清空缓存
            if replacement_count != len(self.replacements):
                # 新的替换操作会使得缓存失效
                cache.clear()
                replacement_count = len(self.replacements)
            # 构建用于缓存的键值，包含表达式和所有变量范围的项
            key = (expr, *var_ranges.items())
            # 尝试从缓存中获取结果
            result = cache.get(key, None)
            # 如果缓存中没有找到，则进行实际的简化操作，并将结果存入缓存
            if result is None:
                result = self._simplify_with_ranges(expr, var_ranges)
                cache[key] = result
            return result

        return simplify_with_ranges

    def make_simplify_loops_cache(self):
        """
        self._simplify_with_ranges() can be expensive, cache its results
        """
        # 定义一个空的字典作为缓存
        cache: Dict[Tuple[Any, ...], Any] = dict()
        # 获取当前替换操作的数量
        replacement_count = len(self.replacements)

        # 定义一个内部函数 simplify_loops，接受索引变量、大小和索引公式作为参数，并返回简化后的结果
        def simplify_loops(index_vars, sizes, index_formulas):
            nonlocal replacement_count
            # 如果替换操作的数量发生了变化，则清空缓存
            if replacement_count != len(self.replacements):
                # 新的替换操作会使得缓存失效
                cache.clear()
                replacement_count = len(self.replacements)
            # 构建用于缓存的键值，包含所有索引变量、大小和索引公式的项
            key = (*index_vars, *sizes, *index_formulas)
            # 尝试从缓存中获取结果
            result = cache.get(key, None)
            # 如果缓存中没有找到，则进行实际的简化操作，并将结果存入缓存
            if result is None:
                result = self._simplify_loops_impl(index_vars, sizes, index_formulas)
                cache[key] = result
            return result

        return simplify_loops

    def _simplify_loops_impl(
        self, index_vars: List[sympy.Symbol], sizes, index_formulas
        # 实际执行循环简化的内部方法，接受索引变量列表、大小和索引公式作为参数
    ):
        """
        Try to remove as many axis from loop iterations as possible, by:
            1) removing size==1 dimensions
            2) fuse contiguous dimensions into a single loop
            If channel_last = True, we will prevent the last dim fused with other dims
        """
        # 使用 self.simplify 对 sizes 列表中的每个元素进行简化处理
        sizes = list(map(self.simplify, sizes))

        # 针对 index_formulas 列表中的每个元素 x，如果 x 是 sympy.Expr 类型，则调用 self.stride_vars(x, index_vars)，否则返回长度为 len(index_vars) 的零列表
        strides = [
            self.stride_vars(x, index_vars)
            if isinstance(x, sympy.Expr)
            else [0] * len(index_vars)
            for x in index_formulas
        ]
        # 断言 sizes 和 strides[0] 的长度相同
        assert len(sizes) == len(strides[0]), (len(sizes), len(strides[0]))

        # 遍历 sizes 列表，如果 sizes[i] == 1，则将其置为 None，以移除大小为 1 的维度
        for i in range(len(sizes)):
            if sizes[i] == 1:
                sizes[i] = None

        # 定义函数 can_merge_dims(a, b)，用于检测是否可以合并维度 a 和 b
        def can_merge_dims(a, b):
            # 遍历 strides 列表
            for k in range(len(strides)):
                # 如果 strides[k][a] * sizes[a] 等于 strides[k][b]，则进行进一步测试
                if self.simplify(strides[k][a] * sizes[a]) == self.simplify(
                    strides[k][b]
                ):
                    # 近似测试通过，尝试进行更严格的声音版本测试
                    va = index_vars[a]
                    vb = index_vars[b]
                    v = sympy_index_symbol("_merge_tester")
                    expr1 = sympy_subs(index_formulas[k], {va: v * sizes[a], vb: 0})
                    expr2 = sympy_subs(index_formulas[k], {va: 0, vb: v})
                    # 如果 expr1 和 expr2 经过简化后相等，则可以合并维度
                    if self.simplify(expr1) == self.simplify(expr2):
                        continue
                return False
            return True

        # 初始化循环变量 changed 为 True
        changed = True
        # 当 changed 为 True 时，执行循环
        while changed:
            changed = False
            # 使用 itertools.product 生成 sizes 列表中元素的逆序排列的所有组合 (i, j)
            for i, j in itertools.product(
                reversed(range(len(sizes))), reversed(range(len(sizes)))
            ):
                # 如果 i == j 或者 sizes[i] 或 sizes[j] 为 None，则继续下一次循环
                if i == j or sizes[i] is None or sizes[j] is None:
                    continue
                # 如果可以合并维度 i 和 j，则执行合并操作，并将 changed 置为 True
                if can_merge_dims(i, j):
                    changed = True
                    sizes[i] = sizes[i] * sizes[j]
                    sizes[j] = None

        # 定义函数 reindex(index)，用于重新索引
        def reindex(index):
            it = list(reversed(index))
            new_index = []
            # 遍历 sizes 列表，根据 sizes 中的每个元素生成新的索引列表 new_index
            for size in sizes:
                if size is None:
                    new_index.append(sympy.Integer(0))
                else:
                    new_index.append(it.pop())
            # 断言 it 列表为空
            assert not it
            return new_index

        # 定义函数 prune(index)，用于修剪索引
        def prune(index):
            assert len(index) == len(sizes)
            # 返回 index 列表中那些对应 sizes 列表元素不为 None 的元素
            return [i for i, s in zip(index, sizes) if s is not None]

        # 返回 sizes 列表中不为 None 的元素列表、reindex 函数和 prune 函数
        return [x for x in sizes if x is not None], reindex, prune

    # Note - [On Statically Known]
    #
    # The statically_known_* family of functions below replaces a prior system, called maybe_guard_*. The prior system
    # See Note - [On Statically Known]
    def is_expr_static_and_true(self, expr: Union[sympy.Basic, bool]) -> bool:
        # 如果表达式是布尔值 True 或 False，则直接返回其布尔值
        if expr in (True, False):
            return bool(expr)
    
        try:
            # 尝试简化表达式以评估其静态性
            simplified = self.shape_env._maybe_evaluate_static(expr)
            # 如果能够简化表达式并得到非空结果，则返回其布尔值
            if simplified is not None:
                return bool(simplified)
        except Exception:
            # 捕获可能出现的异常情况，并记录调试信息
            log.debug("Could not simplify %s", expr)
    
        # 默认情况下，返回 False，表示无法确定表达式的静态真值
        return False
    
    # See Note - [On Statically Known]
    def statically_known_equals(self, left: Expr, right: Union[Expr, int]) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left and right are equal.
        """
        # 判断左右表达式是否静态等值，即左表达式是否等于右表达式
        return self.is_expr_static_and_true(sympy.Eq(left, right))  # type: ignore[arg-type]
    
    # See Note - [On Statically Known]
    def statically_known_list_equals(self, left: List[Expr], right: List[Expr]) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left and right lists are equal.
        """
        # 如果左右列表长度不相等，则直接返回 False
        if len(left) != len(right):
            return False
        # 检查左右列表中对应位置的元素是否静态等值
        if all(self.statically_known_equals(l, r) for l, r in zip(left, right)):
            return True
        return False
    
    # See Note - [On Statically Known]
    def statically_known_leq(self, left: Expr, right: Union[Expr, int]) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is less than or equal to right.
        """
        # 创建左表达式小于等于右表达式的比较表达式
        expr = left <= right
        # 判断这个比较表达式是否是静态真值
        return self.is_expr_static_and_true(expr)
    
    # See Note - [On Statically Known]
    def statically_known_geq(self, left: Expr, right: Union[Expr, int]) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is greater than or equal to right.
        """
        # 创建左表达式大于等于右表达式的比较表达式
        expr = left >= right
        # 判断这个比较表达式是否是静态真值
        return self.is_expr_static_and_true(expr)
    # 返回一个布尔值，指示是否可以安全地优化为左边小于右边
    def statically_known_lt(self, left: Expr, right: Union[Expr, int]) -> bool:
        expr = left < right  # 创建比较表达式 left < right
        return self.is_expr_static_and_true(expr)  # 调用方法判断表达式是否静态为真

    # See Note - [On Statically Known]
    # 返回一个布尔值，指示是否可以安全地优化为左边大于右边
    def statically_known_gt(self, left: Expr, right: Union[Expr, int]) -> bool:
        expr = left > right  # 创建比较表达式 left > right
        return self.is_expr_static_and_true(expr)  # 调用方法判断表达式是否静态为真

    # See Note - [On Statically Known]
    # 返回一个布尔值，指示是否可以优化为 numerator 是 denominator 的倍数
    def statically_known_multiple_of(
        self, numerator: Expr, denominator: Union[Expr, int]
    ) -> bool:
        expr = sympy.Eq(numerator % denominator, 0)  # 创建表达式 numerator % denominator == 0
        return self.is_expr_static_and_true(expr)  # 调用方法判断表达式是否静态为真

    # See Note - [On Statically Known]
    # 返回一个布尔值，指示表达式是否已知为 2 的幂次方
    def statically_known_power_of_2(self, expr: Expr) -> bool:
        return isinstance(expr, sympy.Integer) and is_power_of_2(int(expr))  # 判断表达式是否为整数且是否为 2 的幂次方

    # The guard functions require you to ALREADY KNOW that a particular
    # condition holds.  If you don't know (you want to guard on an expression
    # being a particular value, and then get access to that value), use
    # the evaluate functions.

    # 返回 left 和 right 相等的表达式，如果 left 或 right 是符号表达式，则进行替换
    def guard_equals(self, left: Expr, right: Expr) -> Expr:
        if isinstance(left, Expr):
            left = sympy_subs(left, self.inv_precomputed_replacements)  # 替换 left 中的符号表达式
        if isinstance(right, Expr):
            right = sympy_subs(right, self.inv_precomputed_replacements)  # 替换 right 中的符号表达式
        assert self.shape_env.evaluate_expr(sympy.Eq(left, right))  # 断言 left 等于 right
        return left  # 返回替换后的 left

    # 断言 left 小于 right
    def guard_leq(self, left: Expr, right: Expr) -> None:
        return self.guard_lt(left, right + 1)  # 调用 guard_lt 方法，确保 left 小于 right + 1

    # 断言 left 小于 right
    def guard_lt(self, left: Expr, right: Expr) -> None:
        assert self.shape_env.evaluate_expr(sympy.Lt(left, right))  # 断言 left 小于 right

    # 返回一个序列的顺序作为 range(len(seq)) 的一个排列，并保证顺序不变
    def guarded_order(self, seq):
        seq = [*map(self.remove_precomputed_replacements, seq)]  # 移除预计算替换的元素
        seq = [(self.size_hint(var), orig_idx, var) for orig_idx, var in enumerate(seq)]  # 为序列中的每个元素创建元组 (size_hint, 原始索引, 元素)
        seq.sort()  # 根据元组的第一个元素对序列进行排序
        order = [-1] * len(seq)  # 创建与序列长度相同的初始顺序列表
        last_var = None  # 初始化上一个变量
        for new_index, (_, orig_index, var) in enumerate(seq):
            order[orig_index] = new_index  # 更新顺序列表中的顺序
            if last_var is not None:
                self.guard_leq(last_var, var)  # 确保上一个变量小于等于当前变量
            last_var = var  # 更新上一个变量为当前变量
        return order  # 返回顺序列表

    # The evaluate functions evaluate some symbolic sympy expression
    # (NB: not necessarily an Expr) and return what the concrete result
    # 判断表达式是否为其结果保护的对象

    # 注意：应当使用 evaluate_expr(sympy.Lt(a, b)) 而不是 evaluate_expr(a < b)
    # 这样可以确保您实际上拥有一个 sympy 化的表达式，
    # 并且可以防止您不正确地编写 evaluate_expr(a == b)，
    # 如果 a 或 b 是 sympy 表达式的话，这将产生错误的结果
    def evaluate_expr(self, left: Union[Expr, sympy.logic.boolalg.Boolean]) -> bool:
        assert isinstance(left, (Expr, sympy.logic.boolalg.Boolean)), type(left)
        return self.shape_env.evaluate_expr(sympy.sympify(left))

    def evaluate_min(self, left: Expr, right: Expr) -> Expr:
        """返回左右两个表达式中较小的一个，并对该选择进行保护"""
        if isinstance(left, Expr):
            left = sympy_subs(left, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        if isinstance(right, Expr):
            right = sympy_subs(right, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        try:
            lv = self.size_hint(left)
            rv = self.size_hint(right)
        except TypeError:  # 对于未支持的符号整数
            if left == right or self.statically_known_leq(left, right):
                return left
            if self.statically_known_leq(right, left):
                return right
            gcd = sympy.gcd(left, right)
            if left == gcd:  # 处理 `min(10*u0, u0)` 等情况
                return left
            if right == gcd:
                return right
            raise TypeError(
                f"evaluate_min({left}, {right}) with unbacked symints"
            ) from None
        if lv <= rv:
            self.guard_leq(left, right)
            return left
        else:
            self.guard_leq(right, left)
            return right

    def evaluate_max(self, left: Expr, right: Expr) -> Expr:
        """返回左右两个表达式中较大的一个，并对该选择进行保护"""
        # 始终选择与 eval min 相反的结果以保持一致性
        # 这意味着 min(a, b) 和 max(a, b) 会产生相同的保护
        min_val = self.evaluate_min(left, right)
        return right if min_val is left else left

    def evaluate_static_shape(self, left: Expr) -> int:
        right = self.size_hint(left)
        self.guard_equals(left, sympy.Integer(right))
        return int(right)

    def evaluate_static_shapes(self, left: List[Expr]) -> List[int]:
        return [self.evaluate_static_shape(x) for x in left]

    def remove_precomputed_replacements(self, expr: Expr) -> Expr:
        if any(symbol_is_type(s, SymT.PRECOMPUTED_SIZE) for s in expr.free_symbols):  # type: ignore[attr-defined]
            return sympy_subs(expr, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        return expr
    def symbolic_hint(self, expr: Expr) -> Union[Expr, int]:
        # 对给定表达式进行简化处理
        expr = self.simplify(expr)
        # 如果表达式不是 Expr 类型，返回整数表达式
        if not isinstance(expr, Expr):
            assert isinstance(expr, int)
            return expr
        # 获取表达式中的自由符号
        free_symbols = expr.free_symbols
        # 如果表达式中没有自由符号，尝试将表达式转换为整数，若失败则返回原表达式
        if not free_symbols:
            try:
                return int(expr)  # type: ignore[return-value]
            except TypeError:
                return expr  # inf/nan/I
        # 移除预计算的替换内容
        expr = self.remove_precomputed_replacements(expr)
        # 对表达式进行符号替换，使用 self.var_to_val 中的映射
        return sympy_subs(expr, self.var_to_val)

    def size_hint(self, expr: Expr, *, fallback: Optional[int] = None) -> int:
        # 获取符号提示
        out = self.symbolic_hint(expr)
        # 如果符号提示不是整数或 sympy.Integer 类型，并且提供了回退值，则进行处理
        if not isinstance(out, (int, sympy.Integer)) and fallback is not None:
            # 使用提供的启发式回退提示
            unbacked_sym_vrs = {
                s: self.shape_env.var_to_range.get(s, None) for s in out.free_symbols
            }
            # 如果所有未支持的符号变量都有值，则使用 bound_sympy 计算回退值
            if all(vr is not None for vr in unbacked_sym_vrs.values()):
                hint_vr = bound_sympy(out, unbacked_sym_vrs)  # type: ignore[arg-type]
                if isinstance(hint_vr.lower, (int, sympy.Integer)):
                    fallback = max(fallback, int(hint_vr.lower))
                if isinstance(hint_vr.upper, (int, sympy.Integer)):
                    fallback = min(fallback, int(hint_vr.upper))
            return fallback

        try:
            return int(out)
        except Exception:
            # 记录调试日志并抛出异常
            log.debug("failed on: %s", out)
            raise

    def size_hints(
        self,
        exprs: Iterable[Expr],
        *,
        fallback: Optional[int] = None,
    ) -> Tuple[int, ...]:
        # 返回表达式集合的大小提示，使用给定的回退值
        return tuple(self.size_hint(x, fallback=fallback) for x in exprs)

    def _lru_cache(self, fn, maxsize=None):
        """
        使用 functools.lru_cache 包装函数 fn，并在替换无效时进行清理。
        """
        fn_cache = functools.lru_cache(maxsize)(fn)
        prior_len = len(self.replacements)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal prior_len
            # 如果替换发生变化，则清除缓存
            if prior_len != len(self.replacements):
                prior_len = len(self.replacements)
                fn_cache.cache_clear()
            return fn_cache(*args, **kwargs)

        return wrapper

    def make_stride_vars_cache(self):
        # 使用 _lru_cache 方法创建缓存函数
        cache = self._lru_cache(self._stride_vars)

        def stride_vars(
            index: Expr,
            vars: Sequence[sympy.Symbol],
            support_vars: Optional[Sequence[sympy.Symbol]] = None,
        ) -> List[Expr]:
            # 如果支持变量未提供，则使用 vars 自身作为支持变量
            if not support_vars:
                support_vars = vars
            return cache(index, tuple(vars), tuple(support_vars))

        return stride_vars

    def _stride_vars(
        self,
        index: Expr,
        vars: Sequence[sympy.Symbol],
        support_vars: Sequence[sympy.Symbol],
    def stride_hints(
        self,
        index: Expr,
        vars: Sequence[sympy.Symbol],
        support_vars: Optional[Sequence[sympy.Symbol]] = None,
    ) -> List[int]:
        # 遍历索引表达式中的自由符号
        for v in index.free_symbols:
            # 检查符号是否是间接类型，如果是，则将其替换为0
            if symbol_is_type(v, SymT.INDIRECT):  # type: ignore[attr-defined]
                index = sympy_subs(index, {v: 0})  # type: ignore[dict-item]

        result = []
        # 获得索引表达式的步长提示
        for s in self.stride_vars(index, vars, support_vars):
            try:
                # 尝试获取步长对应的大小提示
                result.append(self.size_hint(s))
            except TypeError:
                # 如果出现类型错误，则将大小提示置为0
                result.append(0)
        return result

    def stride_order(self, index: Expr, vars: List[sympy.Symbol]) -> List[int]:
        # 获得索引表达式的步长值的绝对值元组
        strides = tuple(map(abs, self.stride_hints(index, vars)))
        # 创建步长的排序顺序
        order = list(range(len(strides)))
        # 根据步长值的大小进行排序，零值步长排在最后
        order.sort(key=lambda x: (strides[x] == 0, strides[x]))
        return order
    def lookup_precomputed_size(self, expr: Expr) -> Expr:
        # 如果表达式是整数、符号或数字，直接返回表达式本身
        if (
            isinstance(expr, (int, sympy.Symbol, sympy.Number))
            or expr.is_number
            or expr.is_symbol
        ):
            return expr
        # 移除表达式中的预计算替换
        expr = self.remove_precomputed_replacements(expr)
        # 如果表达式不在预计算替换字典中，则进行处理
        if expr not in self.precomputed_replacements:
            # 生成一个新的符号，并将其与表达式关联存储
            sym = sympy_index_symbol_with_prefix(
                SymT.PRECOMPUTED_SIZE, len(self.precomputed_replacements)
            )
            self.precomputed_replacements[expr] = sym
            self.inv_precomputed_replacements[sym] = expr
        # 返回预计算替换字典中与表达式关联的符号
        return self.precomputed_replacements[expr]

    def free_symbols(self) -> Set[sympy.Symbol]:
        # 返回变量值字典中的符号集合，排除已替换的符号集合
        return set(self.var_to_val.keys()) - set(self.replacements.keys())

    def combine_modular_indexing_pairs(self, index: sympy.Expr) -> sympy.Expr:
        """
        A pair of special ModularIndexing can be combined.

        E.g. ModularIndexing(ModularIndexing(x, 1, a), 1, b)
        We can simplify this to ModuleIndexing(x, 1, b), if
        1. x is non negative integer
        2. a and b are positive integers
        3. a is a multiple of b.
        """

        def _check_args(x, div, mod, is_first):
            # 检查除数和模数是否为整数
            if not isinstance(div, sympy.Integer) or not isinstance(mod, sympy.Integer):
                return False
            # 除数必须为1
            if div != 1:
                return False
            # 模数必须大于0
            if mod <= 0:
                return False

            if is_first:
                # 第一个 ModularIndexing 应包含嵌套的 ModularIndexing
                if not isinstance(x, ModularIndexing):
                    return False
            else:
                # 第二个 ModularIndexing 应包含非负符号
                if not isinstance(x, sympy.Symbol) or not self.statically_known_geq(
                    x, 0
                ):
                    return False
            return True

        # 如果 index 是 ModularIndexing 类型
        if isinstance(index, ModularIndexing):
            x, div, mod = index.args

            # 检查第一个 ModularIndexing 的参数
            if not _check_args(x, div, mod, True):
                return index

            x2, div2, mod2 = x.args

            # 检查第二个 ModularIndexing 的参数
            if not _check_args(x2, div2, mod2, False):
                return index

            # 检查第一个 ModularIndexing 的模数是否是第二个 ModularIndexing 的整数倍
            if mod2 % mod != 0:
                return index

            # 返回简化后的 ModularIndexing 对象
            return ModularIndexing(x2, 1, mod)

        # 如果 index 不是 ModularIndexing 类型，则直接返回 index
        return index
# 定义函数 join_dimensions，接受一个 sympy.Expr 类型参数 expr，并返回一个 sympy.Expr 类型的结果
def join_dimensions(expr: Expr) -> Expr:
    # 如果 expr 不是 sympy.Add 类型或者不包含 ModularIndexing，直接返回 expr，快速退出路径
    if not isinstance(expr, sympy.Add) or not expr.has(ModularIndexing):
        return expr  # fast exit path
    
    # 调用 _join_dimensions_cached 函数处理 expr，并返回结果
    return _join_dimensions_cached(expr)


# 使用 functools.lru_cache(256) 装饰器定义 _join_dimensions_cached 函数，接受一个 sympy.Expr 类型参数 expr，并返回一个 sympy.Expr 类型的结果
@functools.lru_cache(256)
def _join_dimensions_cached(expr: Expr) -> Expr:
    """
    ModularIndexing(i0, 1, 32) + 32 * ModularIndexing(i0, 32, 4)
    becomes
    ModularIndexing(i0, 1, 128)
    ModularIndexing(i0, 1, 32) + 32 * FloorDiv(i0, 32)
    becomes i0

    This type of pattern can come from view operations
    """
    # 断言 expr 是 sympy.Add 类型
    assert isinstance(expr, sympy.Add)

    # 定义多个 Wild 匹配模式：scale, base, divisor, mod1, mod2
    scale = sympy.Wild("scale", exclude=[0], integer=True)
    base = sympy.Wild("base", integer=True)
    divisor = sympy.Wild("divisor", integer=True)
    mod1 = sympy.Wild("modulus", integer=True)
    mod2 = sympy.Wild("modulus2", integer=True)

    # 遍历 expr 的每个子项 term1
    for term1 in expr.args:
        # 尝试匹配 scale * ModularIndexing(base, divisor, mod1) 形式的表达式
        m1 = term1.match(scale * ModularIndexing(base, divisor, mod1))
        if m1:
            # 再次遍历 expr 的每个子项 term2
            for term2 in expr.args:
                # 尝试匹配 m1[scale] * m1[mod1] * ModularIndexing(m1[base], m1[divisor] * m1[mod1], mod2) 形式的表达式
                m2 = term2.match(
                    m1[scale]
                    * m1[mod1]
                    * ModularIndexing(m1[base], m1[divisor] * m1[mod1], mod2)
                )
                # 如果匹配成功且 term1 不等于 term2
                if m2 and term1 != term2:
                    # 更新 expr，并递归调用 join_dimensions 处理结果
                    expr = join_dimensions(
                        expr
                        - term1
                        - term2
                        + m1[scale]
                        * ModularIndexing(m1[base], m1[divisor], m1[mod1] * m2[mod2])
                    )
                    return expr

    # 再次遍历 expr 的每个子项 term1
    for term1 in expr.args:
        # 尝试匹配 scale * ModularIndexing(base, divisor, mod1) 形式的表达式
        m1 = term1.match(scale * ModularIndexing(base, divisor, mod1))
        if m1:
            # 再次遍历 expr 的每个子项 term2
            for term2 in expr.args:
                # 尝试匹配 m1[scale] * m1[mod1] * FloorDiv(m1[base], m1[divisor] * m1[mod1]) 形式的表达式
                m2 = term2.match(
                    m1[scale] * m1[mod1] * FloorDiv(m1[base], m1[divisor] * m1[mod1])
                )
                # 如果匹配成功
                if m2 is not None:  # in case of success we get an empty dict here
                    # 更新 expr，并递归调用 join_dimensions 处理结果
                    expr = join_dimensions(
                        expr
                        - term1
                        - term2
                        + m1[scale] * FloorDiv(m1[base], m1[divisor])
                    )
                    return expr
    
    # 如果以上匹配均未成功，直接返回原始的 expr
    return expr


# 定义类 SimplifyIndexing，继承自 V.WrapperHandler 类型
class SimplifyIndexing(V.WrapperHandler):  # type: ignore[name-defined]
    """
    A wrapper around .virtualize.ops that uses var range information to
    simplify ModularIndexing/FloorDiv.
    """

    # 初始化方法，接受 inner 和 var_ranges 两个参数
    def __init__(self, inner, var_ranges: VarRanges):
        super().__init__(inner)
        # 设置实例属性 name 为 "SimplifyIndexing"
        self.name = "SimplifyIndexing"
        # 使用 var_ranges 初始化实例属性 _simplify，是一个 lambda 函数
        self._simplify: Callable[
            [Expr], Expr
        ] = lambda index: V.graph.sizevars.simplify_with_ranges(index, var_ranges)

    # 方法 load，接受 name 和 index 两个参数，返回调用 _inner.load 处理后的结果
    def load(self, name: str, index: sympy.Expr):
        return self._inner.load(name, self._simplify(index))

    # 方法 store，接受 name、index、value 和 mode（可选）四个参数，返回调用 _inner.store 处理后的结果
    def store(self, name, index, value, mode=None):
        return self._inner.store(name, self._simplify(index), value, mode=mode)

    # 方法 store_reduction，接受 name、index 和 value 三个参数，返回调用 _inner.store_reduction 处理后的结果
    def store_reduction(self, name, index, value):
        return self._inner.store_reduction(name, self._simplify(index), value)
    # 调用内部对象的索引表达式方法，传入简化后的索引和数据类型参数，返回结果
    def index_expr(self, index, dtype):
        return self._inner.index_expr(self._simplify(index), dtype)

    # 调用内部对象的边界检查方法，传入简化后的索引、大小、下界和上界参数，返回结果
    def check_bounds(self, index, size, lower, upper):
        return self._inner.check_bounds(self._simplify(index), size, lower, upper)
```