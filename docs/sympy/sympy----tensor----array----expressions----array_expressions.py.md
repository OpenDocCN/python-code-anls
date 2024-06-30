# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\array_expressions.py`

```
# 导入标准库中的模块和函数
import collections.abc
import operator
# 从 collections 模块中导入 defaultdict 和 Counter 类
from collections import defaultdict, Counter
# 从 functools 模块中导入 reduce 函数
from functools import reduce
# 导入 itertools 模块，用于生成迭代器
import itertools
# 从 itertools 模块中导入 accumulate 函数
from itertools import accumulate
# 导入类型提示相关的模块和类
from typing import Optional, List, Tuple as tTuple

# 导入 sympy 库中的具体数值、关系等模块
import typing
from sympy.core.numbers import Integer
from sympy.core.relational import Equality
# 导入特殊张量函数相关的模块
from sympy.functions.special.tensor_functions import KroneckerDelta
# 导入 sympy 核心基础类
from sympy.core.basic import Basic
# 导入 sympy 核心容器类
from sympy.core.containers import Tuple
# 导入 sympy 核心表达式类
from sympy.core.expr import Expr
# 导入 sympy 核心函数类，包括函数和 Lambda 函数
from sympy.core.function import (Function, Lambda)
# 导入 sympy 核心乘法类
from sympy.core.mul import Mul
# 导入 sympy 核心单例类
from sympy.core.singleton import S
# 导入 sympy 核心排序类
from sympy.core.sorting import default_sort_key
# 导入 sympy 核心符号类
from sympy.core.symbol import (Dummy, Symbol)
# 导入 sympy 矩阵基类
from sympy.matrices.matrixbase import MatrixBase
# 导入 sympy 矩阵表达式类
from sympy.matrices.expressions.diagonal import diagonalize_vector
# 导入 sympy 矩阵表达式类
from sympy.matrices.expressions.matexpr import MatrixExpr
# 导入 sympy 张量数组操作类
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensordiagonal, tensorproduct)
# 导入 sympy 稠密 N 维数组类
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
# 导入 sympy N 维数组类
from sympy.tensor.array.ndim_array import NDimArray
# 导入 sympy 索引和索引基类
from sympy.tensor.indexed import (Indexed, IndexedBase)
# 导入 sympy 矩阵表达式类
from sympy.matrices.expressions.matexpr import MatrixElement
# 导入 sympy 张量数组表达式工具类
from sympy.tensor.array.expressions.utils import _apply_recursively_over_nested_lists, _sort_contraction_indices, \
    _get_mapping_from_subranks, _build_push_indices_up_func_transformation, _get_contraction_links, \
    _build_push_indices_down_func_transformation
# 导入 sympy 排列组合类
from sympy.combinatorics import Permutation
# 导入 sympy 排列组合类中的反转函数
from sympy.combinatorics.permutations import _af_invert
# 导入 sympy 核心转换函数
from sympy.core.sympify import _sympify

# 定义一个 _ArrayExpr 类，继承自 Expr 类
class _ArrayExpr(Expr):
    # shape 属性，表示数组的形状
    shape: tTuple[Expr, ...]

    # 定义 __getitem__ 方法，实现数组元素的获取
    def __getitem__(self, item):
        # 如果 item 不是可迭代对象，则转为包含 item 的元组
        if not isinstance(item, collections.abc.Iterable):
            item = (item,)
        # 检查数组元素的形状是否合法
        ArrayElement._check_shape(self, item)
        # 返回数组中指定元素的值
        return self._get(item)

    # 定义 _get 方法，用于获取数组中指定元素的值
    def _get(self, item):
        return _get_array_element_or_slice(self, item)

# 定义一个 ArraySymbol 类，继承自 _ArrayExpr 类
class ArraySymbol(_ArrayExpr):
    """
    Symbol representing an array expression
    """

    # 构造函数，创建一个数组符号对象
    def __new__(cls, symbol, shape: typing.Iterable) -> "ArraySymbol":
        # 如果 symbol 是字符串，则转为 Symbol 对象
        if isinstance(symbol, str):
            symbol = Symbol(symbol)
        # 将 shape 转为元组，其中每个元素都经过 _sympify 处理
        shape = Tuple(*map(_sympify, shape))
        # 调用 Expr 类的构造函数，创建对象
        obj = Expr.__new__(cls, symbol, shape)
        return obj

    # 返回数组符号的名称
    @property
    def name(self):
        return self._args[0]

    # 返回数组符号的形状
    @property
    def shape(self):
        return self._args[1]

    # 将数组符号转换为显式表示的数组
    def as_explicit(self):
        # 如果数组的形状中包含符号，则无法转换为显式数组，抛出 ValueError
        if not all(i.is_Integer for i in self.shape):
            raise ValueError("cannot express explicit array with symbolic shape")
        # 使用 itertools 生成器生成所有可能的索引组合，获取数组的所有元素值
        data = [self[i] for i in itertools.product(*[range(j) for j in self.shape])]
        # 使用获取到的数据创建一个 ImmutableDenseNDimArray 对象，并将其形状重新调整为原形状
        return ImmutableDenseNDimArray(data).reshape(*self.shape)


# 定义一个 ArrayElement 类，继承自 Expr 类
class ArrayElement(Expr):
    """
    An element of an array.
    """

    # 可微分标志
    _diff_wrt = True
    # 符号性质标志
    is_symbol = True
    # 可交换性标志
    is_commutative = True
    # 定义一个新的类方法__new__，用于创建新的对象实例
    def __new__(cls, name, indices):
        # 如果name是字符串，则将其转换为Symbol对象
        if isinstance(name, str):
            name = Symbol(name)
        # 对name进行符号化处理
        name = _sympify(name)
        # 如果indices不是可迭代对象，则转换为包含单个元素的元组
        if not isinstance(indices, collections.abc.Iterable):
            indices = (indices,)
        # 对indices中的每个元素进行符号化处理，并转换为元组
        indices = _sympify(tuple(indices))
        # 调用类方法_check_shape，检查name和indices的形状是否匹配
        cls._check_shape(name, indices)
        # 调用父类Expr的__new__方法，创建一个新的对象实例obj
        obj = Expr.__new__(cls, name, indices)
        return obj

    # 类方法_check_shape，用于检查数组的形状是否匹配
    @classmethod
    def _check_shape(cls, name, indices):
        # 将indices转换为元组
        indices = tuple(indices)
        # 如果name具有"shape"属性，则进行下一步检查
        if hasattr(name, "shape"):
            # 如果indices的长度与数组形状的长度不匹配，则引发IndexError异常
            index_error = IndexError("number of indices does not match shape of the array")
            if len(indices) != len(name.shape):
                raise index_error
            # 如果任何索引超出了数组的形状范围，则引发ValueError异常
            if any((i >= s) == True for i, s in zip(indices, name.shape)):
                raise ValueError("shape is out of bounds")
        # 如果indices中包含任何负值，则引发ValueError异常
        if any((i < 0) == True for i in indices):
            raise ValueError("shape contains negative values")

    # name属性，返回对象的第一个参数作为名称
    @property
    def name(self):
        return self._args[0]

    # indices属性，返回对象的第二个参数作为索引
    @property
    def indices(self):
        return self._args[1]

    # 私有方法_eval_derivative，用于计算导数
    def _eval_derivative(self, s):
        # 如果s不是ArrayElement类型，则返回零
        if not isinstance(s, ArrayElement):
            return S.Zero

        # 如果s与当前对象相同，则返回1
        if s == self:
            return S.One

        # 如果s的名称与当前对象的名称不同，则返回零
        if s.name != self.name:
            return S.Zero

        # 返回KroneckerDelta函数生成器的乘积，用于比较当前对象的索引和s的索引
        return Mul.fromiter(KroneckerDelta(i, j) for i, j in zip(self.indices, s.indices))
class ZeroArray(_ArrayExpr):
    """
    Symbolic array of zeros. Equivalent to ``ZeroMatrix`` for matrices.
    """

    def __new__(cls, *shape):
        # 如果没有给定形状，则返回零
        if len(shape) == 0:
            return S.Zero
        # 将形状中的每个元素转换为 SymPy 表达式
        shape = map(_sympify, shape)
        # 创建一个新的 ZeroArray 对象
        obj = Expr.__new__(cls, *shape)
        return obj

    @property
    def shape(self):
        # 返回当前数组的形状
        return self._args

    def as_explicit(self):
        # 如果数组形状中有符号变量，则抛出错误
        if not all(i.is_Integer for i in self.shape):
            raise ValueError("Cannot return explicit form for symbolic shape.")
        # 返回一个具体形式的零数组
        return ImmutableDenseNDimArray.zeros(*self.shape)

    def _get(self, item):
        # 获取数组中的元素，此处为零
        return S.Zero


class OneArray(_ArrayExpr):
    """
    Symbolic array of ones.
    """

    def __new__(cls, *shape):
        # 如果没有给定形状，则返回一
        if len(shape) == 0:
            return S.One
        # 将形状中的每个元素转换为 SymPy 表达式
        shape = map(_sympify, shape)
        # 创建一个新的 OneArray 对象
        obj = Expr.__new__(cls, *shape)
        return obj

    @property
    def shape(self):
        # 返回当前数组的形状
        return self._args

    def as_explicit(self):
        # 如果数组形状中有符号变量，则抛出错误
        if not all(i.is_Integer for i in self.shape):
            raise ValueError("Cannot return explicit form for symbolic shape.")
        # 返回一个具体形式的包含全为一的数组
        return ImmutableDenseNDimArray([S.One for i in range(reduce(operator.mul, self.shape))]).reshape(*self.shape)

    def _get(self, item):
        # 获取数组中的元素，此处为一
        return S.One


class _CodegenArrayAbstract(Basic):

    @property
    def subranks(self):
        """
        Returns the ranks of the objects in the uppermost tensor product inside
        the current object.  In case no tensor products are contained, return
        the atomic ranks.

        Examples
        ========

        >>> from sympy.tensor.array import tensorproduct, tensorcontraction
        >>> from sympy import MatrixSymbol
        >>> M = MatrixSymbol("M", 3, 3)
        >>> N = MatrixSymbol("N", 3, 3)
        >>> P = MatrixSymbol("P", 3, 3)

        Important: do not confuse the rank of the matrix with the rank of an array.

        >>> tp = tensorproduct(M, N, P)
        >>> tp.subranks
        [2, 2, 2]

        >>> co = tensorcontraction(tp, (1, 2), (3, 4))
        >>> co.subranks
        [2, 2, 2]
        """
        # 返回当前对象中最上层张量积的秩（或原子秩）
        return self._subranks[:]

    def subrank(self):
        """
        The sum of ``subranks``.
        """
        # 返回所有 subranks 的总和
        return sum(self.subranks)

    @property
    def shape(self):
        # 返回当前数组的形状
        return self._shape

    def doit(self, **hints):
        # 获取深度提示，如果为真则深度规范化所有参数
        deep = hints.get("deep", True)
        if deep:
            # 对所有参数执行 doit 操作并规范化结果
            return self.func(*[arg.doit(**hints) for arg in self.args])._canonicalize()
        else:
            # 对当前对象进行规范化
            return self._canonicalize()


class ArrayTensorProduct(_CodegenArrayAbstract):
    r"""
    Class to represent the tensor product of array-like objects.
    """
    # 重写 __new__ 方法，用于创建新的实例
    def __new__(cls, *args, **kwargs):
        # 对传入的参数进行符号化处理
        args = [_sympify(arg) for arg in args]

        # 从关键字参数中取出 canonicalize，并移除它，如果不存在则默认为 False
        canonicalize = kwargs.pop("canonicalize", False)

        # 获取所有参数的秩（rank）
        ranks = [get_rank(arg) for arg in args]

        # 使用父类的 __new__ 方法创建基础对象实例
        obj = Basic.__new__(cls, *args)
        # 将计算得到的秩列表赋值给对象的 _subranks 属性
        obj._subranks = ranks
        # 获取所有参数的形状（shape）
        shapes = [get_shape(i) for i in args]

        # 如果任意一个参数的形状为 None，则设置对象的 _shape 属性为 None；否则将所有形状拼接为元组赋给 _shape
        if any(i is None for i in shapes):
            obj._shape = None
        else:
            obj._shape = tuple(j for i in shapes for j in i)
        
        # 如果 canonicalize 参数为 True，则返回对象的规范化结果
        if canonicalize:
            return obj._canonicalize()
        
        # 否则返回对象本身
        return obj

    @classmethod
    def _flatten(cls, args):
        # 将 args 中的所有元素展平成一个列表，对于类型为 cls 的元素，展开其 args 属性
        args = [i for arg in args for i in (arg.args if isinstance(arg, cls) else [arg])]
        return args

    # 将对象转换为显式表示的张量积形式
    def as_explicit(self):
        # 对于参数列表中具有 as_explicit 方法的对象，递归调用其 as_explicit 方法；否则直接使用参数本身
        return tensorproduct(*[arg.as_explicit() if hasattr(arg, "as_explicit") else arg for arg in self.args])
class ArrayAdd(_CodegenArrayAbstract):
    r"""
    Class for elementwise array additions.
    """

    def __new__(cls, *args, **kwargs):
        # 将传入的参数符号化（若未符号化），确保每个参数都被符号化
        args = [_sympify(arg) for arg in args]
        # 获取每个参数的秩（rank）
        ranks = [get_rank(arg) for arg in args]
        # 将秩去重，确保所有数组的秩相同
        ranks = list(set(ranks))
        # 如果秩的数量不为1，则抛出错误
        if len(ranks) != 1:
            raise ValueError("summing arrays of different ranks")
        # 获取每个参数的形状（shape）
        shapes = [arg.shape for arg in args]
        # 如果形状中包含不同的非空形状，抛出错误
        if len({i for i in shapes if i is not None}) > 1:
            raise ValueError("mismatching shapes in addition")

        # 弹出可选参数“canonicalize”，默认为 False
        canonicalize = kwargs.pop("canonicalize", False)

        # 使用 Basic 类的 __new__ 方法创建对象
        obj = Basic.__new__(cls, *args)
        # 存储参数的秩信息
        obj._subranks = ranks
        # 如果所有参数都有定义的形状，则存储其形状
        if all(i is not None for i in shapes):
            obj._shape = shapes[0]
        else:
            obj._shape = None
        # 如果 canonicalize 参数为 True，则返回规范化后的对象
        if canonicalize:
            return obj._canonicalize()
        # 否则返回对象本身
        return obj

    def _canonicalize(self):
        # 获取对象的参数列表
        args = self.args

        # 扁平化参数列表（将嵌套的 ArrayAdd 对象展开）
        args = self._flatten_args(args)

        # 获取每个参数的形状
        shapes = [get_shape(arg) for arg in args]
        # 过滤掉参数列表中的 ZeroArray 和 ZeroMatrix 对象
        args = [arg for arg in args if not isinstance(arg, (ZeroArray, ZeroMatrix))]
        # 如果参数列表为空
        if len(args) == 0:
            # 如果有未定义形状的参数，则抛出错误
            if any(i for i in shapes if i is None):
                raise NotImplementedError("cannot handle addition of ZeroMatrix/ZeroArray and undefined shape object")
            # 返回一个形状为 shapes[0] 的 ZeroArray 对象
            return ZeroArray(*shapes[0])
        # 如果参数列表中只有一个参数，则直接返回该参数
        elif len(args) == 1:
            return args[0]
        # 否则使用 func 方法重新构造对象，不进行 canonicalize
        return self.func(*args, canonicalize=False)

    @classmethod
    def _flatten_args(cls, args):
        # 创建一个新的参数列表
        new_args = []
        # 遍历原始参数列表
        for arg in args:
            # 如果参数是 ArrayAdd 对象，则展开其 args 列表中的元素
            if isinstance(arg, ArrayAdd):
                new_args.extend(arg.args)
            else:
                new_args.append(arg)
        return new_args

    def as_explicit(self):
        # 将对象的参数列表逐个求和，使用 reduce 函数和 operator.add
        return reduce(
            operator.add,
            [arg.as_explicit() if hasattr(arg, "as_explicit") else arg for arg in self.args])


class PermuteDims(_CodegenArrayAbstract):
    r"""
    Class to represent permutation of axes of arrays.

    Examples
    ========

    >>> from sympy.tensor.array import permutedims
    >>> from sympy import MatrixSymbol
    >>> M = MatrixSymbol("M", 3, 3)
    >>> cg = permutedims(M, [1, 0])

    The object ``cg`` represents the transposition of ``M``, as the permutation
    ``[1, 0]`` will act on its indices by switching them:

    `M_{ij} \Rightarrow M_{ji}`

    This is evident when transforming back to matrix form:

    >>> from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    >>> convert_array_to_matrix(cg)
    M.T

    >>> N = MatrixSymbol("N", 3, 2)
    >>> cg = permutedims(N, [1, 0])
    >>> cg.shape
    (2, 3)

    There are optional parameters that can be used as alternative to the permutation:

    >>> from sympy.tensor.array.expressions import ArraySymbol, PermuteDims
    >>> M = ArraySymbol("M", (1, 2, 3, 4, 5))
    >>> expr = PermuteDims(M, index_order_old="ijklm", index_order_new="kijml")
    >>> expr
    PermuteDims(M, (0 2 1)(3 4))
    # 检查张量的形状属性
    >>> expr.shape
    (3, 1, 2, 5, 4)

    # 张量积的排列顺序被简化为达到标准形式：
    Permutations of tensor products are simplified in order to achieve a
    standard form:

    # 导入张量积相关的模块
    >>> from sympy.tensor.array import tensorproduct
    # 创建一个符号矩阵 M，形状为 4x5
    >>> M = MatrixSymbol("M", 4, 5)
    # 计算张量积 M ⊗ N
    >>> tp = tensorproduct(M, N)
    # 检查张量积的形状属性
    >>> tp.shape
    (4, 5, 3, 2)
    # 对张量积进行重新排列
    >>> perm1 = permutedims(tp, [2, 3, 1, 0])

    # 参数 ``(M, N)`` 已经被排序，排列已被简化，表达式等效：
    The args ``(M, N)`` have been sorted and the permutation has been
    simplified, the expression is equivalent:

    # 检查重新排列后的表达式的参数
    >>> perm1.expr.args
    (N, M)
    # 检查重新排列后的张量的形状属性
    >>> perm1.shape
    (3, 2, 5, 4)
    # 检查重新排列的顺序
    >>> perm1.permutation
    (2 3)

    # 数组形式的排列顺序已从 ``[2, 3, 1, 0]`` 简化为 ``[0, 1, 3, 2]``，因为张量积的参数 `M` 和 `N` 被交换：
    The permutation in its array form has been simplified from
    ``[2, 3, 1, 0]`` to ``[0, 1, 3, 2]``, as the arguments of the tensor
    product `M` and `N` have been switched:

    # 检查数组形式的重新排列顺序
    >>> perm1.permutation.array_form
    [0, 1, 3, 2]

    # 可以嵌套第二次重新排列：
    We can nest a second permutation:

    # 对 perm1 进行第二次重新排列
    >>> perm2 = permutedims(perm1, [1, 0, 2, 3])
    # 检查第二次重新排列后的张量的形状属性
    >>> perm2.shape
    (2, 3, 5, 4)
    # 检查第二次重新排列后的数组形式的排列顺序
    >>> perm2.permutation.array_form
    [1, 0, 3, 2]
    """
    # 定义一个类方法 `_PermuteDims_denestarg_ArrayTensorProduct`，接受三个参数：cls（类本身）、expr（表达式）、permutation（排列）
    def _PermuteDims_denestarg_ArrayTensorProduct(cls, expr, permutation):
        # 获取排列的映像形式：
        perm_image_form = _af_invert(permutation.array_form)
        # 将表达式的参数转换为列表
        args = list(expr.args)
        # 计算每个参数的起始全局位置累积和
        cumul = list(accumulate([0] + expr.subranks))
        # 将 `perm_image_form` 按参数的索引范围拆分为子列表
        perm_image_form_in_components = [perm_image_form[cumul[i]:cumul[i+1]] for i in range(len(args))]
        # 创建一个索引和目标位置键的数组
        ps = [(i, sorted(comp)) for i, comp in enumerate(perm_image_form_in_components)]
        # 根据目标位置键对数组进行排序：
        # 这样，我们定义了根据排列来排序参数的一种标准方式。
        ps.sort(key=lambda x: x[1])
        # 读取参数的逆排列（即映像形式）
        perm_args_image_form = [i[0] for i in ps]
        # 将参数的排列应用于 `args`
        args_sorted = [args[i] for i in perm_args_image_form]
        # 将排列应用于 `expr` 的轴的数组形式
        perm_image_form_sorted_args = [perm_image_form_in_components[i] for i in perm_args_image_form]
        # 创建一个新的排列对象，其数组形式为排列的轴的逆
        new_permutation = Permutation(_af_invert([j for i in perm_image_form_sorted_args for j in i]))
        # 返回 `_array_tensor_product` 函数应用于排序后的参数及新的排列对象
        return _array_tensor_product(*args_sorted), new_permutation

    @classmethod
    # 定义类方法 `_PermuteDims_denestarg_ArrayContraction`，用于处理数组压缩操作的维度重排
    def _PermuteDims_denestarg_ArrayContraction(cls, expr, permutation):
        # 如果 `expr` 不是 `ArrayContraction` 类型，则直接返回 `expr` 和 `permutation`
        if not isinstance(expr, ArrayContraction):
            return expr, permutation
        # 如果 `expr.expr` 不是 `ArrayTensorProduct` 类型，则直接返回 `expr` 和 `permutation`
        if not isinstance(expr.expr, ArrayTensorProduct):
            return expr, permutation
        # 获取 `ArrayTensorProduct` 的参数列表
        args = expr.expr.args
        # 获取每个参数的秩
        subranks = [get_rank(arg) for arg in expr.expr.args]

        # 获取压缩指标
        contraction_indices = expr.contraction_indices
        # 将压缩指标展平为一维列表
        contraction_indices_flat = [j for i in contraction_indices for j in i]
        # 计算每个参数的累积索引
        cumul = list(accumulate([0] + subranks))

        # 将排列在其数组形式中的排列按照自由指标在对应的张量积参数中展开：
        permutation_array_blocks_up = []
        # 获取排列的图像形式
        image_form = _af_invert(permutation.array_form)
        counter = 0
        for i, e in enumerate(subranks):
            current = []
            for j in range(cumul[i], cumul[i+1]):
                if j in contraction_indices_flat:
                    continue
                current.append(image_form[counter])
                counter += 1
            permutation_array_blocks_up.append(current)

        # 获取每个张量积参数的轴重新定位映射：
        index_blocks = [list(range(cumul[i], cumul[i+1])) for i, e in enumerate(expr.subranks)]
        index_blocks_up = expr._push_indices_up(expr.contraction_indices, index_blocks)
        # 计算逆排列
        inverse_permutation = permutation**(-1)
        # 对上推的索引块进行排列
        index_blocks_up_permuted = [[inverse_permutation(j) for j in i if j is not None] for i in index_blocks_up]

        # 排序键是元组的列表，第一个元素是 `args` 的索引，第二个元组元素是对 `args` 进行排序的排序键：
        sorting_keys = list(enumerate(index_blocks_up_permuted))
        sorting_keys.sort(key=lambda x: x[1])

        # 现在可以在其图像形式上对排列在 `args` 上的排列进行操作：
        new_perm_image_form = [i[0] for i in sorting_keys]
        # 将参数级别的排列应用于各个元素：
        new_index_blocks = [index_blocks[i] for i in new_perm_image_form]
        new_index_perm_array_form = _af_invert([j for i in new_index_blocks for j in i])
        new_args = [args[i] for i in new_perm_image_form]
        new_contraction_indices = [tuple(new_index_perm_array_form[j] for j in i) for i in contraction_indices]
        # 创建新的 `ArrayContraction` 表达式
        new_expr = _array_contraction(_array_tensor_product(*new_args), *new_contraction_indices)
        # 创建新的排列对象
        new_permutation = Permutation(_af_invert([j for i in [permutation_array_blocks_up[k] for k in new_perm_image_form] for j in i]))
        return new_expr, new_permutation

    @classmethod
    # 类方法修饰符，表示下面的方法是类方法
    @classmethod
    def _check_if_there_are_closed_cycles(cls, expr, permutation):
        # 将表达式的参数列表转换为列表
        args = list(expr.args)
        # 获取表达式的子秩列表
        subranks = expr.subranks
        # 获取置换的循环形式
        cyclic_form = permutation.cyclic_form
        # 计算累积子秩，从0开始累加子秩列表
        cumulative_subranks = [0] + list(accumulate(subranks))
        # 计算每个循环的最小索引
        cyclic_min = [min(i) for i in cyclic_form]
        # 计算每个循环的最大索引
        cyclic_max = [max(i) for i in cyclic_form]
        # 存储需要保留的循环
        cyclic_keep = []
        # 遍历每个循环
        for i, cycle in enumerate(cyclic_form):
            flag = True
            # 遍历累积子秩列表
            for j in range(len(cumulative_subranks) - 1):
                # 判断是否找到可下沉的循环
                if cyclic_min[i] >= cumulative_subranks[j] and cyclic_max[i] < cumulative_subranks[j+1]:
                    # 找到可下沉的循环
                    args[j] = _permute_dims(args[j], Permutation([[k - cumulative_subranks[j] for k in cyclic_form[i]]]))
                    flag = False
                    break
            if flag:
                # 没找到可下沉的循环，将其加入保留列表
                cyclic_keep.append(cyclic_form[i])
        # 返回数组张量积和新的置换对象
        return _array_tensor_product(*args), Permutation(cyclic_keep, size=permutation.size)

    def nest_permutation(self):
        r"""
        DEPRECATED.
        """
        # 调用内部方法 `_nest_permutation` 处理表达式和置换
        ret = self._nest_permutation(self.expr, self.permutation)
        # 如果返回结果为空，返回自身对象；否则返回处理后的结果
        if ret is None:
            return self
        return ret

    @classmethod
    def _nest_permutation(cls, expr, permutation):
        # 如果表达式是 ArrayTensorProduct 类型
        if isinstance(expr, ArrayTensorProduct):
            # 调用 `_check_if_there_are_closed_cycles` 处理封闭循环，然后进行维度置换
            return _permute_dims(*cls._check_if_there_are_closed_cycles(expr, permutation))
        # 如果表达式是 ArrayContraction 类型
        elif isinstance(expr, ArrayContraction):
            # 将树形层次结构反转：将收缩操作放在顶层
            cycles = permutation.cyclic_form
            # 转换外部索引为内部索引
            newcycles = ArrayContraction._convert_outer_indices_to_inner_indices(expr, *cycles)
            newpermutation = Permutation(newcycles)
            # 构建新的收缩索引
            new_contr_indices = [tuple(newpermutation(j) for j in i) for i in expr.contraction_indices]
            # 执行数组收缩操作，并应用维度置换
            return _array_contraction(PermuteDims(expr.expr, newpermutation), *new_contr_indices)
        # 如果表达式是 ArrayAdd 类型
        elif isinstance(expr, ArrayAdd):
            # 对每个参数执行维度置换
            return _array_add(*[PermuteDims(arg, permutation) for arg in expr.args])
        # 其他情况返回空值
        return None

    def as_explicit(self):
        # 获取表达式对象，并尝试调用其 `as_explicit` 方法
        expr = self.expr
        if hasattr(expr, "as_explicit"):
            expr = expr.as_explicit()
        # 返回经过维度置换后的表达式
        return permutedims(expr, self.permutation)

    @classmethod
    def _get_permutation_from_arguments(cls, permutation, index_order_old, index_order_new, dim):
        # 如果未提供置换参数
        if permutation is None:
            # 确保 index_order_new 和 index_order_old 都未定义
            if index_order_new is None or index_order_old is None:
                raise ValueError("Permutation not defined")
            # 根据旧的和新的索引顺序获取置换对象
            return PermuteDims._get_permutation_from_index_orders(index_order_old, index_order_new, dim)
        else:
            # 如果提供了置换参数，但同时也提供了 index_order_new 或 index_order_old，则报错
            if index_order_new is not None:
                raise ValueError("index_order_new cannot be defined with permutation")
            if index_order_old is not None:
                raise ValueError("index_order_old cannot be defined with permutation")
            # 直接返回提供的置换对象
            return permutation
    # 定义一个类方法，用于根据旧的索引顺序和新的索引顺序获取排列顺序
    def _get_permutation_from_index_orders(cls, index_order_old, index_order_new, dim):
        # 检查新索引顺序中的索引数是否正确
        if len(set(index_order_new)) != dim:
            raise ValueError("wrong number of indices in index_order_new")
        # 检查旧索引顺序中的索引数是否正确
        if len(set(index_order_old)) != dim:
            raise ValueError("wrong number of indices in index_order_old")
        # 检查新旧索引顺序是否包含相同的索引
        if len(set.symmetric_difference(set(index_order_new), set(index_order_old))) > 0:
            raise ValueError("index_order_new and index_order_old must have the same indices")
        # 根据新索引顺序在旧索引顺序中查找索引位置，生成排列顺序列表
        permutation = [index_order_old.index(i) for i in index_order_new]
        # 返回排列顺序列表
        return permutation
# ArrayDiagonal 类，继承自 _CodegenArrayAbstract 类，用于表示对角线操作符。
class ArrayDiagonal(_CodegenArrayAbstract):
    r"""
    Class to represent the diagonal operator.

    Explanation
    ===========

    In a 2-dimensional array it returns the diagonal, this looks like the
    operation:

    `A_{ij} \rightarrow A_{ii}`

    The diagonal over axes 1 and 2 (the second and third) of the tensor product
    of two 2-dimensional arrays `A \otimes B` is

    `\Big[ A_{ab} B_{cd} \Big]_{abcd} \rightarrow \Big[ A_{ai} B_{id} \Big]_{adi}`

    In this last example the array expression has been reduced from
    4-dimensional to 3-dimensional. Notice that no contraction has occurred,
    rather there is a new index `i` for the diagonal, contraction would have
    reduced the array to 2 dimensions.

    Notice that the diagonalized out dimensions are added as new dimensions at
    the end of the indices.
    """

    # 构造函数，创建 ArrayDiagonal 类的实例。
    def __new__(cls, expr, *diagonal_indices, **kwargs):
        # 符号化输入的表达式 expr
        expr = _sympify(expr)
        # 对角线索引进行排序并封装成 Tuple 类型的列表
        diagonal_indices = [Tuple(*sorted(i)) for i in diagonal_indices]
        # 获取是否规范化的标志，默认为 False
        canonicalize = kwargs.get("canonicalize", False)

        # 获取表达式的形状
        shape = get_shape(expr)
        # 如果形状不为 None，则进行验证和位置形状的获取
        if shape is not None:
            cls._validate(expr, *diagonal_indices, **kwargs)
            # 获取新的形状和位置信息
            positions, shape = cls._get_positions_shape(shape, diagonal_indices)
        else:
            positions = None
        
        # 如果没有指定对角线索引，直接返回原始表达式
        if len(diagonal_indices) == 0:
            return expr
        
        # 创建 Basic 类的新实例 obj
        obj = Basic.__new__(cls, expr, *diagonal_indices)
        # 设置对象的位置属性
        obj._positions = positions
        # 获取表达式的子秩
        obj._subranks = _get_subranks(expr)
        # 设置对象的形状属性
        obj._shape = shape
        
        # 如果需要进行规范化，则调用 _canonicalize 方法
        if canonicalize:
            return obj._canonicalize()
        
        # 返回创建的对象实例
        return obj
    def _canonicalize(self):
        # 获取表达式和对角线索引
        expr = self.expr
        diagonal_indices = self.diagonal_indices
        
        # 找出长度为1的对角线索引
        trivial_diags = [i for i in diagonal_indices if len(i) == 1]
        
        # 如果存在长度为1的对角线索引
        if len(trivial_diags) > 0:
            # 创建长度为1的对角线索引的位置映射字典
            trivial_pos = {e[0]: i for i, e in enumerate(diagonal_indices) if len(e) == 1}
            # 创建大于1的对角线索引的位置映射字典
            diag_pos = {e: i for i, e in enumerate(diagonal_indices) if len(e) > 1}
            # 过滤掉长度为1的对角线索引，生成新的对角线索引列表
            diagonal_indices_short = [i for i in diagonal_indices if len(i) > 1]
            
            # 获取表达式的秩和对角线索引的长度
            rank1 = get_rank(self)
            rank2 = len(diagonal_indices)
            rank3 = rank1 - rank2
            
            # 初始化反置排列列表
            inv_permutation = []
            counter1 = 0
            
            # 将对角线索引下推并更新索引位置
            indices_down = ArrayDiagonal._push_indices_down(diagonal_indices_short, list(range(rank1)), get_rank(expr))
            
            # 遍历更新后的索引位置
            for i in indices_down:
                if i in trivial_pos:
                    inv_permutation.append(rank3 + trivial_pos[i])
                elif isinstance(i, (Integer, int)):
                    inv_permutation.append(counter1)
                    counter1 += 1
                else:
                    inv_permutation.append(rank3 + diag_pos[i])
            
            # 计算置换
            permutation = _af_invert(inv_permutation)
            
            # 如果存在长度大于0的对角线索引，则对表达式进行维度置换和对角化
            if len(diagonal_indices_short) > 0:
                return _permute_dims(_array_diagonal(expr, *diagonal_indices_short), permutation)
            else:
                return _permute_dims(expr, permutation)
        
        # 如果表达式是ArrayAdd类型，则展开该类型
        if isinstance(expr, ArrayAdd):
            return self._ArrayDiagonal_denest_ArrayAdd(expr, *diagonal_indices)
        
        # 如果表达式是ArrayDiagonal类型，则展开该类型
        if isinstance(expr, ArrayDiagonal):
            return self._ArrayDiagonal_denest_ArrayDiagonal(expr, *diagonal_indices)
        
        # 如果表达式是PermuteDims类型，则展开该类型
        if isinstance(expr, PermuteDims):
            return self._ArrayDiagonal_denest_PermuteDims(expr, *diagonal_indices)
        
        # 如果表达式是ZeroArray或ZeroMatrix类型，则返回对应形状的ZeroArray
        if isinstance(expr, (ZeroArray, ZeroMatrix)):
            positions, shape = self._get_positions_shape(expr.shape, diagonal_indices)
            return ZeroArray(*shape)
        
        # 否则，调用原始函数处理表达式和对角线索引
        return self.func(expr, *diagonal_indices, canonicalize=False)

    @staticmethod
    def _validate(expr, *diagonal_indices, **kwargs):
        # 检查对角化操作是否在维度不匹配的索引上进行
        shape = get_shape(expr)
        for i in diagonal_indices:
            if any(j >= len(shape) for j in i):
                raise ValueError("index is larger than expression shape")
            if len({shape[j] for j in i}) != 1:
                raise ValueError("diagonalizing indices of different dimensions")
            if not kwargs.get("allow_trivial_diags", False) and len(i) <= 1:
                raise ValueError("need at least two axes to diagonalize")
            if len(set(i)) != len(i):
                raise ValueError("axis index cannot be repeated")

    @staticmethod
    def _remove_trivial_dimensions(shape, *diagonal_indices):
        # 去除形状中长度为1的维度的对角线索引
        return [tuple(j for j in i) for i in diagonal_indices if shape[i[0]] != 1]

    @property
    def expr(self):
        # 返回对象的第一个参数作为表达式
        return self.args[0]

    @property
    def diagonal_indices(self):
        # 返回对象的对角线索引参数
        return self.args[1:]
    # 返回除第一个参数外的所有参数，即去除第一个参数后的其余参数元组
    def diagonal_indices(self):
        return self.args[1:]

    @staticmethod
    # 将表达式的对角线索引展平，处理外部对角线索引参数
    def _flatten(expr, *outer_diagonal_indices):
        # 获取表达式的内部对角线索引
        inner_diagonal_indices = expr.diagonal_indices
        # 将所有内部对角线索引放入一个列表并排序
        all_inner = [j for i in inner_diagonal_indices for j in i]
        all_inner.sort()
        # TODO: 添加总秩和累积秩的 API:
        # 计算总秩
        total_rank = _get_subrank(expr)
        # 计算内部对角线的秩
        inner_rank = len(all_inner)
        # 计算外部对角线的秩
        outer_rank = total_rank - inner_rank
        # 初始化位移列表
        shifts = [0 for i in range(outer_rank)]
        counter = 0
        pointer = 0
        # 遍历外部对角线的每一个维度
        for i in range(outer_rank):
            # 更新计数器和指针，直到指针超过内部对角线索引或计数器大于当前指针位置
            while pointer < inner_rank and counter >= all_inner[pointer]:
                counter += 1
                pointer += 1
            # 更新位移列表的当前维度
            shifts[i] += pointer
            counter += 1
        # 更新外部对角线索引参数
        outer_diagonal_indices = tuple(tuple(shifts[j] + j for j in i) for i in outer_diagonal_indices)
        # 组合内外对角线索引参数
        diagonal_indices = inner_diagonal_indices + outer_diagonal_indices
        # 返回数组的对角线元素
        return _array_diagonal(expr.expr, *diagonal_indices)

    @classmethod
    # 处理数组对角线元素中的数组加法
    def _ArrayDiagonal_denest_ArrayAdd(cls, expr, *diagonal_indices):
        # 对每个参数中的表达式应用对角线操作，并进行数组加法
        return _array_add(*[_array_diagonal(arg, *diagonal_indices) for arg in expr.args])

    @classmethod
    # 处理数组对角线元素中的数组对角线操作
    def _ArrayDiagonal_denest_ArrayDiagonal(cls, expr, *diagonal_indices):
        # 调用 _flatten 方法展平对角线索引
        return cls._flatten(expr, *diagonal_indices)

    @classmethod
    # 处理数组对角线元素中的维度置换操作
    def _ArrayDiagonal_denest_PermuteDims(cls, expr: PermuteDims, *diagonal_indices):
        # 反向计算对角线索引的排列维度
        back_diagonal_indices = [[expr.permutation(j) for j in i] for i in diagonal_indices]
        # 获取非对角线索引
        nondiag = [i for i in range(get_rank(expr)) if not any(i in j for j in diagonal_indices)]
        # 获取非对角线索引的排列
        back_nondiag = [expr.permutation(i) for i in nondiag]
        # 创建非对角线索引的重排映射
        remap = {e: i for i, e in enumerate(sorted(back_nondiag))}
        # 生成新的排列
        new_permutation1 = [remap[i] for i in back_nondiag]
        shift = len(new_permutation1)
        # 生成对角块排列
        diag_block_perm = [i + shift for i in range(len(back_diagonal_indices))]
        # 构造新的排列
        new_permutation = new_permutation1 + diag_block_perm
        # 应用维度置换和数组对角线操作
        return _permute_dims(
            _array_diagonal(
                expr.expr,
                *back_diagonal_indices
            ),
            new_permutation
        )

    # 将非静态方法中的索引向下推送
    def _push_indices_down_nonstatic(self, indices):
        # 定义转换函数，根据位置列表返回索引对应位置或 None
        transform = lambda x: self._positions[x] if x < len(self._positions) else None
        # 递归地应用转换函数到嵌套列表中的每个元素
        return _apply_recursively_over_nested_lists(transform, indices)

    # 将非静态方法中的索引向上推送
    def _push_indices_up_nonstatic(self, indices):
        # 定义转换函数，根据位置列表和索引返回对应的位置
        def transform(x):
            for i, e in enumerate(self._positions):
                if (isinstance(e, int) and x == e) or (isinstance(e, tuple) and x in e):
                    return i

        # 递归地应用转换函数到嵌套列表中的每个元素
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    # 将对角线索引推向较低维度的方法
    def _push_indices_down(cls, diagonal_indices, indices, rank):
        # 获取非对角线索引的位置和形状
        positions, shape = cls._get_positions_shape(range(rank), diagonal_indices)
        # 定义转换函数，将输入的索引映射到新的位置
        transform = lambda x: positions[x] if x < len(positions) else None
        # 递归地应用转换函数到嵌套列表中的所有索引
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    # 将对角线索引推向较高维度的方法
    def _push_indices_up(cls, diagonal_indices, indices, rank):
        # 获取所有索引的位置和形状
        positions, shape = cls._get_positions_shape(range(rank), diagonal_indices)

        def transform(x):
            # 对于每个索引 x，在 positions 中查找其位置 i
            for i, e in enumerate(positions):
                # 如果 positions 中的元素 e 是整数且与 x 相等，或者是元组/列表且 x 在其中，则返回 i
                if (isinstance(e, int) and x == e) or (isinstance(e, (tuple, Tuple)) and (x in e)):
                    return i

        # 递归地应用 transform 函数到嵌套列表中的所有索引
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    # 获取位置和形状的方法
    def _get_positions_shape(cls, shape, diagonal_indices):
        # 获取非对角线索引的位置和形状
        data1 = tuple((i, shp) for i, shp in enumerate(shape) if not any(i in j for j in diagonal_indices))
        pos1, shp1 = zip(*data1) if data1 else ((), ())
        # 获取对角线索引的位置和形状
        data2 = tuple((i, shape[i[0]]) for i in diagonal_indices)
        pos2, shp2 = zip(*data2) if data2 else ((), ())
        # 合并非对角线和对角线索引的位置和形状
        positions = pos1 + pos2
        shape = shp1 + shp2
        # 返回位置和形状元组
        return positions, shape

    # 将对象表达式转换为明确的表达式的方法
    def as_explicit(self):
        expr = self.expr
        # 如果对象表达式有 as_explicit 方法，则递归地调用它，直到获取明确的表达式
        if hasattr(expr, "as_explicit"):
            expr = expr.as_explicit()
        # 使用 tensordiagonal 函数创建一个新的 tensordiagonal 对象，传入表达式和对角线索引
        return tensordiagonal(expr, *self.diagonal_indices)
    # ArrayElementwiseApplyFunc 类，用于表示对数组进行逐元素应用的函数对象
    class ArrayElementwiseApplyFunc(_CodegenArrayAbstract):

        # 构造函数，初始化对象
        def __new__(cls, function, element):
            # 如果 function 不是 Lambda 类型，则创建一个虚拟符号 d，并使用其创建 Lambda 函数对象
            if not isinstance(function, Lambda):
                d = Dummy('d')
                function = Lambda(d, function(d))

            # 调用父类的构造函数来创建对象
            obj = _CodegenArrayAbstract.__new__(cls, function, element)
            # 获取元素的子秩
            obj._subranks = _get_subranks(element)
            return obj

        # 返回函数对象
        @property
        def function(self):
            return self.args[0]

        # 返回表达式对象
        @property
        def expr(self):
            return self.args[1]

        # 返回表达式的形状
        @property
        def shape(self):
            return self.expr.shape

        # 获取函数的一阶偏导数函数对象
        def _get_function_fdiff(self):
            d = Dummy("d")
            function = self.function(d)
            # 对函数进行 d 变量的一阶偏导数计算
            fdiff = function.diff(d)
            if isinstance(fdiff, Function):
                fdiff = type(fdiff)
            else:
                fdiff = Lambda(d, fdiff)
            return fdiff

        # 将表达式转换为显式形式
        def as_explicit(self):
            expr = self.expr
            # 如果表达式具有 as_explicit 方法，则调用该方法
            if hasattr(expr, "as_explicit"):
                expr = expr.as_explicit()
            # 对表达式应用当前对象的函数
            return expr.applyfunc(self.function)


    # ArrayContraction 类，用于表示数组的收缩操作，以便于代码打印处理
    class ArrayContraction(_CodegenArrayAbstract):
        """
        This class is meant to represent contractions of arrays in a form easily
        processable by the code printers.
        """

        # 构造函数，初始化对象
        def __new__(cls, expr, *contraction_indices, **kwargs):
            # 对收缩索引进行排序
            contraction_indices = _sort_contraction_indices(contraction_indices)
            # 符号化表达式 expr
            expr = _sympify(expr)

            # 获取可选参数中的 canonicalize，若不存在则设为 False
            canonicalize = kwargs.get("canonicalize", False)

            # 调用父类的构造函数来创建对象
            obj = Basic.__new__(cls, expr, *contraction_indices)
            # 获取表达式的子秩
            obj._subranks = _get_subranks(expr)
            # 根据子秩获取映射关系
            obj._mapping = _get_mapping_from_subranks(obj._subranks)

            # 创建自由索引到位置的映射
            free_indices_to_position = {i: i for i in range(sum(obj._subranks)) if all(i not in cind for cind in contraction_indices)}
            obj._free_indices_to_position = free_indices_to_position

            # 获取表达式的形状，并在收缩索引存在时对其进行调整
            shape = get_shape(expr)
            cls._validate(expr, *contraction_indices)
            if shape:
                shape = tuple(shp for i, shp in enumerate(shape) if not any(i in j for j in contraction_indices))
            obj._shape = shape

            # 如果 canonicalize 为 True，则规范化对象
            if canonicalize:
                return obj._canonicalize()
            return obj
    # 将表达式规范化，根据已定义的规则对数组进行约简
    def _canonicalize(self):
        # 获取表达式和收缩指数
        expr = self.expr
        contraction_indices = self.contraction_indices

        # 如果没有收缩指数，直接返回原始表达式
        if len(contraction_indices) == 0:
            return expr

        # 根据表达式的类型进行不同的约简操作
        if isinstance(expr, ArrayContraction):
            return self._ArrayContraction_denest_ArrayContraction(expr, *contraction_indices)

        if isinstance(expr, (ZeroArray, ZeroMatrix)):
            return self._ArrayContraction_denest_ZeroArray(expr, *contraction_indices)

        if isinstance(expr, PermuteDims):
            return self._ArrayContraction_denest_PermuteDims(expr, *contraction_indices)

        if isinstance(expr, ArrayTensorProduct):
            # 对张量积进行排序和降阶处理
            expr, contraction_indices = self._sort_fully_contracted_args(expr, contraction_indices)
            expr, contraction_indices = self._lower_contraction_to_addends(expr, contraction_indices)
            # 如果没有剩余的收缩指数，返回处理后的表达式
            if len(contraction_indices) == 0:
                return expr

        if isinstance(expr, ArrayDiagonal):
            return self._ArrayContraction_denest_ArrayDiagonal(expr, *contraction_indices)

        if isinstance(expr, ArrayAdd):
            return self._ArrayContraction_denest_ArrayAdd(expr, *contraction_indices)

        # 检查单索引收缩在一维轴上的情况
        # 这里过滤掉长度为1或在表达式形状中对应位置不为1的索引
        contraction_indices = [i for i in contraction_indices if len(i) > 1 or get_shape(expr)[i[0]] != 1]
        if len(contraction_indices) == 0:
            return expr

        # 调用函数的特定方法进行表达式的约简处理，关闭规范化选项
        return self.func(expr, *contraction_indices, canonicalize=False)

    # 重载乘法运算符，处理乘以标量1的情况
    def __mul__(self, other):
        if other == 1:
            return self
        else:
            raise NotImplementedError("Product of N-dim arrays is not uniquely defined. Use another method.")

    # 重载右乘法运算符，处理右乘标量1的情况
    def __rmul__(self, other):
        if other == 1:
            return self
        else:
            raise NotImplementedError("Product of N-dim arrays is not uniquely defined. Use another method.")

    # 静态方法：验证表达式和收缩指数的有效性
    @staticmethod
    def _validate(expr, *contraction_indices):
        # 获取表达式的形状
        shape = get_shape(expr)
        if shape is None:
            return

        # 检查当形状不匹配时，没有收缩发生的情况
        for i in contraction_indices:
            if len({shape[j] for j in i if shape[j] != -1}) != 1:
                raise ValueError("contracting indices of different dimensions")

    # 类方法：将收缩指数推向下方
    @classmethod
    def _push_indices_down(cls, contraction_indices, indices):
        # 展平收缩指数并排序
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        flattened_contraction_indices.sort()
        # 构建推送指数向下的转换函数，并递归应用于嵌套列表
        transform = _build_push_indices_down_func_transformation(flattened_contraction_indices)
        return _apply_recursively_over_nested_lists(transform, indices)
    # 类方法：将收缩指标向上推移
    def _push_indices_up(cls, contraction_indices, indices):
        # 展平收缩指标列表并排序
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        flattened_contraction_indices.sort()
        # 构建将指标向上推移的转换函数
        transform = _build_push_indices_up_func_transformation(flattened_contraction_indices)
        # 递归应用转换函数到嵌套列表中的指标
        return _apply_recursively_over_nested_lists(transform, indices)

    # 类方法：将收缩转为加法项
    def _lower_contraction_to_addends(cls, expr, contraction_indices):
        # 如果表达式是 ArrayAdd 类型，则抛出未实现异常
        if isinstance(expr, ArrayAdd):
            raise NotImplementedError()
        # 如果表达式不是 ArrayTensorProduct 类型，则直接返回表达式和收缩指标
        if not isinstance(expr, ArrayTensorProduct):
            return expr, contraction_indices
        # 计算子秩和累积秩
        subranks = expr.subranks
        cumranks = list(accumulate([0] + subranks))
        contraction_indices_remaining = []
        contraction_indices_args = [[] for i in expr.args]
        backshift = set()
        # 遍历收缩指标组
        for contraction_group in contraction_indices:
            for j in range(len(expr.args)):
                # 如果表达式参数不是 ArrayAdd 类型，则跳过
                if not isinstance(expr.args[j], ArrayAdd):
                    continue
                # 检查收缩组是否在正确的子秩范围内，并更新参数列表和回移集合
                if all(cumranks[j] <= k < cumranks[j+1] for k in contraction_group):
                    contraction_indices_args[j].append([k - cumranks[j] for k in contraction_group])
                    backshift.update(contraction_group)
                    break
            else:
                contraction_indices_remaining.append(contraction_group)
        # 如果剩余的收缩指标组数量和原来相同，则直接返回表达式和收缩指标
        if len(contraction_indices_remaining) == len(contraction_indices):
            return expr, contraction_indices
        # 计算总秩并生成位移列表
        total_rank = get_rank(expr)
        shifts = list(accumulate([1 if i in backshift else 0 for i in range(total_rank)]))
        # 更新剩余的收缩指标组并调整指标
        contraction_indices_remaining = [Tuple.fromiter(j - shifts[j] for j in i) for i in contraction_indices_remaining]
        # 调用 _array_tensor_product 函数生成结果
        ret = _array_tensor_product(*[
            _array_contraction(arg, *contr) for arg, contr in zip(expr.args, contraction_indices_args)
        ])
        return ret, contraction_indices_remaining

    # 方法：扁平化对角收缩
    def flatten_contraction_of_diagonal(self):
        # 如果表达式不是 ArrayDiagonal 类型，则直接返回自身
        if not isinstance(self.expr, ArrayDiagonal):
            return self
        # 调用 ArrayDiagonal 类的 _push_indices_down 方法推移收缩指标
        contraction_down = self.expr._push_indices_down(self.expr.diagonal_indices, self.contraction_indices)
        new_contraction_indices = []
        diagonal_indices = self.expr.diagonal_indices[:]
        # 遍历推移后的结果
        for i in contraction_down:
            contraction_group = list(i)
            # 处理对角指标和收缩组的交集
            for j in i:
                diagonal_with = [k for k in diagonal_indices if j in k]
                contraction_group.extend([l for k in diagonal_with for l in k])
                diagonal_indices = [k for k in diagonal_indices if k not in diagonal_with]
            new_contraction_indices.append(sorted(set(contraction_group)))

        # 调用 ArrayDiagonal 类的 _push_indices_up 方法推移对角指标
        new_contraction_indices = ArrayDiagonal._push_indices_up(diagonal_indices, new_contraction_indices)
        # 构造新的 _array_contraction 结果
        return _array_contraction(
            _array_diagonal(
                self.expr.expr,
                *diagonal_indices
            ),
            *new_contraction_indices
        )
    @staticmethod
    def _get_free_indices_to_position_map(free_indices, contraction_indices):
        # 创建一个空字典，用于存储自由指标到位置的映射关系
        free_indices_to_position = {}
        # 将收缩指标列表展开为一维列表
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        # 计数器初始化为0，用于跟踪位置
        counter = 0
        # 遍历自由指标列表
        for ind in free_indices:
            # 跳过所有在收缩指标位置上的索引
            while counter in flattened_contraction_indices:
                counter += 1
            # 将自由指标与当前位置的映射关系存入字典
            free_indices_to_position[ind] = counter
            # 更新位置计数器
            counter += 1
        # 返回自由指标到位置的映射字典
        return free_indices_to_position

    @staticmethod
    def _get_index_shifts(expr):
        """
        获取收缩发生前索引位置的映射。

        Examples
        ========

        >>> from sympy.tensor.array import tensorproduct, tensorcontraction
        >>> from sympy import MatrixSymbol
        >>> M = MatrixSymbol("M", 3, 3)
        >>> N = MatrixSymbol("N", 3, 3)
        >>> cg = tensorcontraction(tensorproduct(M, N), [1, 2])
        >>> cg._get_index_shifts(cg)
        [0, 2]

        实际上，``cg`` 在收缩后有两个维度，0 和 1。它们需要通过0和2的偏移来获取收缩之前的对应位置（即0和3）。
        """
        # 获取内部收缩的索引
        inner_contraction_indices = expr.contraction_indices
        # 将所有内部索引合并并排序
        all_inner = [j for i in inner_contraction_indices for j in i]
        all_inner.sort()
        # 计算总秩和内部秩
        total_rank = _get_subrank(expr)  # TODO: 添加总秩和累积秩的API
        inner_rank = len(all_inner)
        outer_rank = total_rank - inner_rank
        # 初始化位移列表
        shifts = [0 for i in range(outer_rank)]
        counter = 0
        pointer = 0
        # 遍历外部秩
        for i in range(outer_rank):
            # 跳过所有在内部索引位置上的索引
            while pointer < inner_rank and counter >= all_inner[pointer]:
                counter += 1
                pointer += 1
            # 记录偏移量
            shifts[i] += pointer
            counter += 1
        # 返回索引位移列表
        return shifts

    @staticmethod
    def _convert_outer_indices_to_inner_indices(expr, *outer_contraction_indices):
        # 获取索引位移
        shifts = ArrayContraction._get_index_shifts(expr)
        # 将外部收缩索引转换为内部收缩索引
        outer_contraction_indices = tuple(tuple(shifts[j] + j for j in i) for i in outer_contraction_indices)
        # 返回内部收缩索引
        return outer_contraction_indices

    @staticmethod
    def _flatten(expr, *outer_contraction_indices):
        # 获取内部和外部收缩索引
        inner_contraction_indices = expr.contraction_indices
        outer_contraction_indices = ArrayContraction._convert_outer_indices_to_inner_indices(expr, *outer_contraction_indices)
        # 合并所有收缩索引，并进行数组收缩操作
        contraction_indices = inner_contraction_indices + outer_contraction_indices
        return _array_contraction(expr.expr, *contraction_indices)

    @classmethod
    def _ArrayContraction_denest_ArrayContraction(cls, expr, *contraction_indices):
        # 调用_flatten方法展开ArrayContraction
        return cls._flatten(expr, *contraction_indices)
    @classmethod
    # 类方法：将数组压缩（消除）操作展开，处理零数组情况
    def _ArrayContraction_denest_ZeroArray(cls, expr, *contraction_indices):
        # 将传入的压缩索引展平成一维列表
        contraction_indices_flat = [j for i in contraction_indices for j in i]
        # 根据表达式的形状，构造一个零数组
        shape = [e for i, e in enumerate(expr.shape) if i not in contraction_indices_flat]
        return ZeroArray(*shape)

    @classmethod
    # 类方法：将数组压缩（消除）操作展开，处理数组相加情况
    def _ArrayContraction_denest_ArrayAdd(cls, expr, *contraction_indices):
        # 对表达式的每个子项递归调用数组压缩展开，并进行数组相加操作
        return _array_add(*[_array_contraction(i, *contraction_indices) for i in expr.args])

    @classmethod
    # 类方法：将数组压缩（消除）操作展开，处理维度置换情况
    def _ArrayContraction_denest_PermuteDims(cls, expr, *contraction_indices):
        # 获取表达式的置换信息和置换后的压缩索引
        permutation = expr.permutation
        plist = permutation.array_form
        new_contraction_indices = [tuple(permutation(j) for j in i) for i in contraction_indices]
        # 过滤掉新的置换列表中已包含的索引
        new_plist = [i for i in plist if not any(i in j for j in new_contraction_indices)]
        # 将新的压缩索引推到置换列表中的合适位置
        new_plist = cls._push_indices_up(new_contraction_indices, new_plist)
        # 调用函数进行维度置换，并返回结果
        return _permute_dims(
            _array_contraction(expr.expr, *new_contraction_indices),
            Permutation(new_plist)
        )

    @classmethod
    # 类方法：将数组压缩（消除）操作展开，处理对角线数组情况
    def _ArrayContraction_denest_ArrayDiagonal(cls, expr: 'ArrayDiagonal', *contraction_indices):
        # 获取对角线索引并将压缩索引向下推
        diagonal_indices = list(expr.diagonal_indices)
        down_contraction_indices = expr._push_indices_down(expr.diagonal_indices, contraction_indices, get_rank(expr.expr))
        # 将对角线压缩索引展平
        down_contraction_indices = [[k for j in i for k in (j if isinstance(j, (tuple, Tuple)) else [j])] for i in down_contraction_indices]
        new_contraction_indices = []
        # 处理新的压缩索引，合并对角线索引
        for contr_indgrp in down_contraction_indices:
            ind = contr_indgrp[:]
            for j, diag_indgrp in enumerate(diagonal_indices):
                if diag_indgrp is None:
                    continue
                if any(i in diag_indgrp for i in contr_indgrp):
                    ind.extend(diag_indgrp)
                    diagonal_indices[j] = None
            new_contraction_indices.append(sorted(set(ind)))

        # 过滤掉处理后的对角线索引
        new_diagonal_indices_down = [i for i in diagonal_indices if i is not None]
        # 将新的对角线索引推到上级，调用对角线数组函数
        return _array_diagonal(
            _array_contraction(expr.expr, *new_contraction_indices),
            *new_diagonal_indices
        )
    # 对象方法：静态方法，用于排序完全收缩参数
    def _sort_fully_contracted_args(cls, expr, contraction_indices):
        # 如果表达式的形状为空，则直接返回表达式和收缩索引
        if expr.shape is None:
            return expr, contraction_indices
        # 计算每个子张量的累积索引范围
        cumul = list(accumulate([0] + expr.subranks))
        # 创建索引块列表，每个块对应一个子张量的索引范围
        index_blocks = [list(range(cumul[i], cumul[i+1])) for i in range(len(expr.args))]
        # 将收缩索引展开为扁平集合
        contraction_indices_flat = {j for i in contraction_indices for j in i}
        # 检查每个子张量是否完全被收缩
        fully_contracted = [all(j in contraction_indices_flat for j in range(cumul[i], cumul[i+1])) for i, arg in enumerate(expr.args)]
        # 根据子张量的完全收缩情况进行排序，生成新的位置列表
        new_pos = sorted(range(len(expr.args)), key=lambda x: (0, default_sort_key(expr.args[x])) if fully_contracted[x] else (1,))
        # 根据新的位置列表重新排列子张量
        new_args = [expr.args[i] for i in new_pos]
        # 将索引块展开为扁平列表
        new_index_blocks_flat = [j for i in new_pos for j in index_blocks[i]]
        # 使用 _af_invert 函数生成索引排列数组形式
        index_permutation_array_form = _af_invert(new_index_blocks_flat)
        # 将收缩元组转换为新的收缩索引，保持排序
        new_contraction_indices = [tuple(index_permutation_array_form[j] for j in i) for i in contraction_indices]
        # 对新的收缩索引进行排序
        new_contraction_indices = _sort_contraction_indices(new_contraction_indices)
        # 返回新的数组张量乘积和新的收缩索引
        return _array_tensor_product(*new_args), new_contraction_indices

    # 对象方法：返回收缩元组的列表
    def _get_contraction_tuples(self):
        """
        Return tuples containing the argument index and position within the
        argument of the index position.

        Examples
        ========

        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> from sympy.tensor.array import tensorproduct, tensorcontraction
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)

        >>> cg = tensorcontraction(tensorproduct(A, B), (1, 2))
        >>> cg._get_contraction_tuples()
        [[(0, 1), (1, 0)]]

        Notes
        =====

        Here the contraction pair `(1, 2)` meaning that the 2nd and 3rd indices
        of the tensor product `A\otimes B` are contracted, has been transformed
        into `(0, 1)` and `(1, 0)`, identifying the same indices in a different
        notation. `(0, 1)` is the second index (1) of the first argument (i.e.
        0 or `A`). `(1, 0)` is the first index (i.e. 0) of the second
        argument (i.e. 1 or `B`).
        """
        # 获取索引映射字典
        mapping = self._mapping
        # 返回收缩索引对应的参数索引和位置的列表
        return [[mapping[j] for j in i] for i in self.contraction_indices]

    # 静态方法：将收缩元组转换为收缩索引
    @staticmethod
    def _contraction_tuples_to_contraction_indices(expr, contraction_tuples):
        # TODO: check that `expr` has `.subranks`:
        # 获取表达式的秩列表
        ranks = expr.subranks
        # 计算累积秩列表
        cumulative_ranks = [0] + list(accumulate(ranks))
        # 将收缩元组转换为收缩索引列表
        return [tuple(cumulative_ranks[j]+k for j, k in i) for i in contraction_tuples]

    # 对象属性：返回自由索引的副本
    @property
    def free_indices(self):
        return self._free_indices[:]

    # 对象属性：返回自由索引到位置的字典副本
    @property
    def free_indices_to_position(self):
        return dict(self._free_indices_to_position)

    # 对象属性：返回表达式的第一个参数
    @property
    def expr(self):
        return self.args[0]

    # 对象属性：返回收缩索引的所有参数（除了第一个参数）
    @property
    def contraction_indices(self):
        return self.args[1:]
    # 将表达式中的张量积的收缩指标转换为对应的组件位置
    def _contraction_indices_to_components(self):
        expr = self.expr  # 获取类实例中的表达式
        if not isinstance(expr, ArrayTensorProduct):  # 检查表达式是否为 ArrayTensorProduct 类型
            raise NotImplementedError("only for contractions of tensor products")  # 抛出未实现错误
        ranks = expr.subranks  # 获取张量积的子秩列表
        mapping = {}  # 初始化空字典，用于存放收缩指标到组件位置的映射
        counter = 0  # 初始化计数器
        for i, rank in enumerate(ranks):  # 遍历子秩列表
            for j in range(rank):  # 遍历每个子秩中的元素个数
                mapping[counter] = (i, j)  # 将收缩指标映射到组件位置
                counter += 1  # 计数器加一，更新收缩指标
        return mapping  # 返回收缩指标到组件位置的映射字典

    # 按照变量名称的字典序对张量积中的参数进行排序
    def sort_args_by_name(self):
        """
        Sort arguments in the tensor product so that their order is lexicographical.

        Examples
        ========

        >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> C = MatrixSymbol("C", N, N)
        >>> D = MatrixSymbol("D", N, N)

        >>> cg = convert_matrix_to_array(C*D*A*B)
        >>> cg
        ArrayContraction(ArrayTensorProduct(A, D, C, B), (0, 3), (1, 6), (2, 5))
        >>> cg.sort_args_by_name()
        ArrayContraction(ArrayTensorProduct(A, D, B, C), (0, 3), (1, 4), (2, 7))
        """
        expr = self.expr  # 获取类实例中的表达式
        if not isinstance(expr, ArrayTensorProduct):  # 检查表达式是否为 ArrayTensorProduct 类型
            return self  # 若不是，则返回当前实例
        args = expr.args  # 获取张量积中的参数列表
        sorted_data = sorted(enumerate(args), key=lambda x: default_sort_key(x[1]))  # 按照参数名称的字典序对参数进行排序
        pos_sorted, args_sorted = zip(*sorted_data)  # 解压排序后的数据
        # 创建重排序映射，用于更新收缩元组中的索引
        reordering_map = {i: pos_sorted.index(i) for i, arg in enumerate(args)}
        # 获取收缩元组并使用重排序映射更新其中的索引
        contraction_tuples = self._get_contraction_tuples()
        contraction_tuples = [[(reordering_map[j], k) for j, k in i] for i in contraction_tuples]
        # 构建新的张量积，使用排序后的参数列表
        c_tp = _array_tensor_product(*args_sorted)
        # 将收缩元组转换为收缩指标，并返回新的数组收缩对象
        new_contr_indices = self._contraction_tuples_to_contraction_indices(
                c_tp,
                contraction_tuples
        )
        return _array_contraction(c_tp, *new_contr_indices)
    def _get_contraction_links(self):
        r"""
        返回一个字典，其中包含张量积中正在收缩的参数之间的链接。

        详见示例以了解值的解释。

        Examples
        ========

        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> C = MatrixSymbol("C", N, N)
        >>> D = MatrixSymbol("D", N, N)

        矩阵乘法是相邻矩阵之间的成对收缩：

        `A_{ij} B_{jk} C_{kl} D_{lm}`

        >>> cg = convert_matrix_to_array(A*B*C*D)
        >>> cg
        ArrayContraction(ArrayTensorProduct(B, C, A, D), (0, 5), (1, 2), (3, 6))

        >>> cg._get_contraction_links()
        {0: {0: (2, 1), 1: (1, 0)}, 1: {0: (0, 1), 1: (3, 0)}, 2: {1: (0, 0)}, 3: {0: (1, 1)}}

        此字典解释如下：位置0处的参数（即矩阵 `A`）其第二个索引（即1）与 `(1, 0)` 收缩，即位置1处的参数（矩阵 `B`）的第一个索引槽（即 `j` 来自 `A`）。

        位置1处的参数（即矩阵 `B`）有两个收缩，分别由索引 `j` 和 `k` 提供，分别是子字典中的第一个和第二个索引（0 和 1）。链接分别为 `(0, 1)` 和 `(2, 0)`。`(0, 1)` 是位置0处参数的索引槽1（第二个），即 `A_{\ldot j}`，依此类推。

        """
        args, dlinks = _get_contraction_links([self], self.subranks, *self.contraction_indices)
        return dlinks

    def as_explicit(self):
        expr = self.expr
        if hasattr(expr, "as_explicit"):
            expr = expr.as_explicit()
        return tensorcontraction(expr, *self.contraction_indices)
class Reshape(_CodegenArrayAbstract):
    """
    Reshape the dimensions of an array expression.

    Examples
    ========

    >>> from sympy.tensor.array.expressions import ArraySymbol, Reshape
    >>> A = ArraySymbol("A", (6,))
    >>> A.shape
    (6,)
    >>> Reshape(A, (3, 2)).shape
    (3, 2)

    Check the component-explicit forms:

    >>> A.as_explicit()
    [A[0], A[1], A[2], A[3], A[4], A[5]]
    >>> Reshape(A, (3, 2)).as_explicit()
    [[A[0], A[1]], [A[2], A[3]], [A[4], A[5]]]

    """

    def __new__(cls, expr, shape):
        # 将表达式符号化（转换为符号表达式）
        expr = _sympify(expr)
        # 如果形状不是 Tuple 类型，则转换为 Tuple
        if not isinstance(shape, Tuple):
            shape = Tuple(*shape)
        # 检查形状是否匹配，如果不匹配则抛出 ValueError
        if Equality(Mul.fromiter(expr.shape), Mul.fromiter(shape)) == False:
            raise ValueError("shape mismatch")
        # 创建新的表达式对象
        obj = Expr.__new__(cls, expr, shape)
        obj._shape = tuple(shape)
        obj._expr = expr
        return obj

    @property
    def shape(self):
        # 返回对象的形状属性
        return self._shape

    @property
    def expr(self):
        # 返回对象的表达式属性
        return self._expr

    def doit(self, *args, **kwargs):
        # 如果 kwargs 中有 "deep"，且为 True，则递归进行 doit 操作
        if kwargs.get("deep", True):
            expr = self.expr.doit(*args, **kwargs)
        else:
            expr = self.expr
        # 如果表达式是 MatrixBase 或 NDimArray 类型，则进行 reshape 操作
        if isinstance(expr, (MatrixBase, NDimArray)):
            return expr.reshape(*self.shape)
        # 否则返回一个新的 Reshape 对象
        return Reshape(expr, self.shape)

    def as_explicit(self):
        # 获取表达式属性
        ee = self.expr
        # 如果表达式具有 as_explicit 方法，则调用该方法
        if hasattr(ee, "as_explicit"):
            ee = ee.as_explicit()
        # 如果表达式是 MatrixBase 类型，则转换为 Array 类型
        if isinstance(ee, MatrixBase):
            from sympy import Array
            ee = Array(ee)
        elif isinstance(ee, MatrixExpr):
            return self
        # 对表达式进行 reshape 操作
        return ee.reshape(*self.shape)


class _ArgE:
    """
    The ``_ArgE`` object contains references to the array expression
    (``.element``) and a list containing the information about index
    contractions (``.indices``).

    Index contractions are numbered and contracted indices show the number of
    the contraction. Uncontracted indices have ``None`` value.

    For example:
    ``_ArgE(M, [None, 3])``
    This object means that expression ``M`` is part of an array contraction
    and has two indices, the first is not contracted (value ``None``),
    the second index is contracted to the 4th (i.e. number ``3``) group of the
    array contraction object.
    """
    indices: List[Optional[int]]

    def __init__(self, element, indices: Optional[List[Optional[int]]] = None):
        # 初始化 _ArgE 对象，包含元素表达式和索引收缩信息列表
        self.element = element
        # 如果索引信息为 None，则创建具有与元素维度相同数量 None 的列表
        if indices is None:
            self.indices = [None for i in range(get_rank(element))]
        else:
            self.indices = indices

    def __str__(self):
        # 返回对象的字符串表示形式
        return "_ArgE(%s, %s)" % (self.element, self.indices)

    __repr__ = __str__


class _IndPos:
    """
    Index position, requiring two integers in the constructor:

    - arg: the position of the argument in the tensor product,
    - rel: the relative position of the index inside the argument.
    """
    # 初始化方法，用于设置对象的属性
    def __init__(self, arg: int, rel: int):
        # 将参数 arg 和 rel 分别赋给对象的属性 self.arg 和 self.rel
        self.arg = arg
        self.rel = rel

    # 字符串表示方法，返回对象的字符串表示形式
    def __str__(self):
        # 返回格式化后的字符串，显示对象的 arg 和 rel 属性的取值
        return "_IndPos(%i, %i)" % (self.arg, self.rel)

    # __repr__ 方法与 __str__ 方法相同
    __repr__ = __str__

    # 迭代器方法，使对象可迭代
    def __iter__(self):
        # 使用生成器将对象的 arg 和 rel 属性放入迭代器中
        yield from [self.arg, self.rel]
class _EditArrayContraction:
    """
    Utility class to help manipulate array contraction objects.

    This class takes as input an ``ArrayContraction`` object and turns it into
    an editable object.

    The field ``args_with_ind`` of this class is a list of ``_ArgE`` objects
    which can be used to easily edit the contraction structure of the
    expression.

    Once editing is finished, the ``ArrayContraction`` object may be recreated
    by calling the ``.to_array_contraction()`` method.
    """
    # 初始化方法，接受一个基础数组作为参数，可以是 ArrayContraction、ArrayDiagonal 或 ArrayTensorProduct 的实例
    def __init__(self, base_array: typing.Union[ArrayContraction, ArrayDiagonal, ArrayTensorProduct]):

        # 表达式的类型为 Basic
        expr: Basic
        # 对角化后的元组的类型为 tTuple[tTuple[int, ...], ...]
        diagonalized: tTuple[tTuple[int, ...], ...]
        # 收缩索引的列表类型为 List[tTuple[int]]
        contraction_indices: List[tTuple[int]]

        # 如果 base_array 是 ArrayContraction 类型
        if isinstance(base_array, ArrayContraction):
            # 从 subranks 获取映射关系
            mapping = _get_mapping_from_subranks(base_array.subranks)
            # 表达式为 base_array 的表达式
            expr = base_array.expr
            # 收缩索引为 base_array 的收缩索引
            contraction_indices = base_array.contraction_indices
            # 对角化为空元组
            diagonalized = ()

        # 如果 base_array 是 ArrayDiagonal 类型
        elif isinstance(base_array, ArrayDiagonal):

            # 如果 base_array 的表达式是 ArrayContraction 类型
            if isinstance(base_array.expr, ArrayContraction):
                # 从 base_array.expr.subranks 获取映射关系
                mapping = _get_mapping_from_subranks(base_array.expr.subranks)
                # 表达式为 base_array.expr 的表达式
                expr = base_array.expr.expr
                # 对角化为调用 ArrayContraction._push_indices_down 的结果
                diagonalized = ArrayContraction._push_indices_down(base_array.expr.contraction_indices, base_array.diagonal_indices)
                # 收缩索引为 base_array.expr 的收缩索引
                contraction_indices = base_array.expr.contraction_indices
            # 如果 base_array 的表达式是 ArrayTensorProduct 类型
            elif isinstance(base_array.expr, ArrayTensorProduct):
                # 映射为空字典
                mapping = {}
                # 表达式为 base_array.expr
                expr = base_array.expr
                # 对角化为 base_array 的对角索引
                diagonalized = base_array.diagonal_indices
                # 收缩索引为空列表
                contraction_indices = []
            else:
                # 映射为空字典
                mapping = {}
                # 表达式为 base_array 的表达式
                expr = base_array.expr
                # 对角化为 base_array 的对角索引
                diagonalized = base_array.diagonal_indices
                # 收缩索引为空列表
                contraction_indices = []

        # 如果 base_array 是 ArrayTensorProduct 类型
        elif isinstance(base_array, ArrayTensorProduct):
            # 表达式为 base_array
            expr = base_array
            # 收缩索引为空列表
            contraction_indices = []
            # 对角化为空元组
            diagonalized = ()
        else:
            # 抛出未实现的错误
            raise NotImplementedError()

        # 如果 expr 是 ArrayTensorProduct 类型，则 args 是 expr.args 的列表形式，否则是包含 expr 的列表
        if isinstance(expr, ArrayTensorProduct):
            args = list(expr.args)
        else:
            args = [expr]

        # args_with_ind 是 args 的 _ArgE 对象列表
        args_with_ind: List[_ArgE] = [_ArgE(arg) for arg in args]

        # 遍历收缩索引列表中的每个收缩元组
        for i, contraction_tuple in enumerate(contraction_indices):
            # 遍历每个收缩元组中的索引
            for j in contraction_tuple:
                # 获取映射中的相对位置
                arg_pos, rel_pos = mapping[j]
                # 将收缩索引设置到 args_with_ind[arg_pos] 的 indices[rel_pos] 中
                args_with_ind[arg_pos].indices[rel_pos] = i

        # 将 args_with_ind 设置为 self 的 args_with_ind 属性
        self.args_with_ind: List[_ArgE] = args_with_ind
        # 将收缩索引的数量设置为 len(contraction_indices)
        self.number_of_contraction_indices: int = len(contraction_indices)
        # 将 _track_permutation 设置为 None
        self._track_permutation: Optional[List[List[int]]] = None

        # 从 base_array.subranks 获取映射关系
        mapping = _get_mapping_from_subranks(base_array.subranks)

        # 将对角化的索引作为负索引添加到编辑器对象中
        for i, e in enumerate(diagonalized):
            for j in e:
                # 获取映射中的相对位置
                arg_pos, rel_pos = mapping[j]
                # 将 -1 - i 设置到 args_with_ind[arg_pos] 的 indices[rel_pos] 中
                self.args_with_ind[arg_pos].indices[rel_pos] = -1 - i

    # 在指定 arg 后插入新的 arg
    def insert_after(self, arg: _ArgE, new_arg: _ArgE):
        # 获取 arg 在 args_with_ind 中的位置
        pos = self.args_with_ind.index(arg)
        # 在 pos + 1 处插入 new_arg
        self.args_with_ind.insert(pos + 1, new_arg)

    # 获取新的收缩索引
    def get_new_contraction_index(self):
        # 增加收缩索引的数量
        self.number_of_contraction_indices += 1
        # 返回新增的收缩索引
        return self.number_of_contraction_indices - 1
    # 更新索引的方法，将所有参数中的索引进行更新
    def refresh_indices(self):
        # 初始化一个空字典用于存储更新后的索引值
        updates = {}
        # 遍历每一个带有索引的参数
        for arg_with_ind in self.args_with_ind:
            # 对每个参数中的索引进行更新，将索引映射为-1，如果索引为None则不更新
            updates.update({i: -1 for i in arg_with_ind.indices if i is not None})
        
        # 对更新后的索引进行排序并重新编号
        for i, e in enumerate(sorted(updates)):
            updates[e] = i
        
        # 记录更新后的收缩索引的数量
        self.number_of_contraction_indices = len(updates)
        
        # 更新每个参数中的索引值
        for arg_with_ind in self.args_with_ind:
            arg_with_ind.indices = [updates.get(i, None) for i in arg_with_ind.indices]

    # 合并标量的方法
    def merge_scalars(self):
        # 初始化一个空列表，用于存储标量参数
        scalars = []
        
        # 遍历每一个带有索引的参数
        for arg_with_ind in self.args_with_ind:
            # 如果参数的索引列表为空，则将其视为标量，加入到标量列表中
            if len(arg_with_ind.indices) == 0:
                scalars.append(arg_with_ind)
        
        # 从参数列表中移除所有的标量参数
        for i in scalars:
            self.args_with_ind.remove(i)
        
        # 将所有标量参数组合成一个新的乘积表达式
        scalar = Mul.fromiter([i.element for i in scalars])
        
        # 如果参数列表为空，则将乘积表达式作为唯一的参数加入列表
        if len(self.args_with_ind) == 0:
            self.args_with_ind.append(_ArgE(scalar))
        else:
            # 否则，将乘积表达式与参数列表中的第一个元素进行张量积操作
            from sympy.tensor.array.expressions.from_array_to_matrix import _a2m_tensor_product
            self.args_with_ind[0].element = _a2m_tensor_product(scalar, self.args_with_ind[0].element)
    # 将对象转换为数组缩并表达式的形式
    def to_array_contraction(self):

        # 计算参数的秩数：
        counter = 0
        # 创建新对角线索引的收集器：
        diag_indices = defaultdict(list)

        # 统计索引频率的计数器：
        count_index_freq = Counter()
        for arg_with_ind in self.args_with_ind:
            count_index_freq.update(Counter(arg_with_ind.indices))

        # 计算自由索引的数量：
        free_index_count = count_index_freq[None]

        # 构造反置排列：
        inv_perm1 = []
        inv_perm2 = []
        # 跟踪已处理的对角线索引：
        done = set()

        # 对角线索引的计数器：
        counter4 = 0

        for arg_with_ind in self.args_with_ind:
            # 如果移除了某些对角化轴，应按顺序排列它们以保持排列。
            # 在此处添加排列
            counter2 = 0  # 索引计数器
            for i in arg_with_ind.indices:
                if i is None:
                    inv_perm1.append(counter4)
                    counter2 += 1
                    counter4 += 1
                    continue
                if i >= 0:
                    continue
                # 重建对角线索引：
                diag_indices[-1 - i].append(counter + counter2)
                if count_index_freq[i] == 1 and i not in done:
                    inv_perm1.append(free_index_count - 1 - i)
                    done.add(i)
                elif i not in done:
                    inv_perm2.append(free_index_count - 1 - i)
                    done.add(i)
                counter2 += 1
            # 移除负索引以恢复正确的编辑对象：
            arg_with_ind.indices = [i if i is not None and i >= 0 else None for i in arg_with_ind.indices]
            counter += len([i for i in arg_with_ind.indices if i is None or i < 0])

        # 计算逆置排列和排列：
        inverse_permutation = inv_perm1 + inv_perm2
        permutation = _af_invert(inverse_permutation)

        # 在表达式中检测到HadamardProduct后获取对角线索引：
        diag_indices_filtered = [tuple(v) for v in diag_indices.values() if len(v) > 1]

        # 合并标量，刷新索引：
        self.merge_scalars()
        self.refresh_indices()

        # 提取参数列表，并获取缩并索引：
        args = [arg.element for arg in self.args_with_ind]
        contraction_indices = self.get_contraction_indices()

        # 构建数组缩并和张量积表达式：
        expr = _array_contraction(_array_tensor_product(*args), *contraction_indices)

        # 对表达式进行对角线操作：
        expr2 = _array_diagonal(expr, *diag_indices_filtered)

        # 如果有追踪排列，执行维度排列：
        if self._track_permutation is not None:
            permutation2 = _af_invert([j for i in self._track_permutation for j in i])
            expr2 = _permute_dims(expr2, permutation2)

        # 执行最终的维度排列：
        expr3 = _permute_dims(expr2, permutation)
        return expr3
    # 返回包含收缩指数的列表，每个收缩指数的位置列表包含在一个单独的子列表中
    def get_contraction_indices(self) -> List[List[int]]:
        # 创建一个包含空列表的列表，数量与收缩指数的数量相同
        contraction_indices: List[List[int]] = [[] for i in range(self.number_of_contraction_indices)]
        # 当前位置的初始化
        current_position: int = 0
        # 遍历每个带有指数的参数
        for arg_with_ind in self.args_with_ind:
            # 遍历当前参数的指数
            for j in arg_with_ind.indices:
                # 如果指数不为None，则将当前位置添加到对应的收缩指数列表中
                if j is not None:
                    contraction_indices[j].append(current_position)
                # 增加当前位置
                current_position += 1
        # 返回收缩指数列表
        return contraction_indices

    # 返回给定指数的映射列表，包含了所有具有该指数的参数位置
    def get_mapping_for_index(self, ind) -> List[_IndPos]:
        # 如果指数超出了收缩指数的范围，则引发值错误异常
        if ind >= self.number_of_contraction_indices:
            raise ValueError("index value exceeding the index range")
        # 初始化位置列表
        positions: List[_IndPos] = []
        # 遍历每个带有指数的参数及其索引
        for i, arg_with_ind in enumerate(self.args_with_ind):
            for j, arg_ind in enumerate(arg_with_ind.indices):
                # 如果找到匹配的指数，则将其位置添加到位置列表中
                if ind == arg_ind:
                    positions.append(_IndPos(i, j))
        # 返回位置列表
        return positions

    # 返回包含收缩指数相对位置信息的列表，每个收缩指数的位置对象列表包含在一个单独的子列表中
    def get_contraction_indices_to_ind_rel_pos(self) -> List[List[_IndPos]]:
        # 创建一个包含空列表的列表，数量与收缩指数的数量相同
        contraction_indices: List[List[_IndPos]] = [[] for i in range(self.number_of_contraction_indices)]
        # 遍历每个带有指数的参数及其索引
        for i, arg_with_ind in enumerate(self.args_with_ind):
            for j, ind in enumerate(arg_with_ind.indices):
                # 如果指数不为None，则将位置对象添加到对应的收缩指数列表中
                if ind is not None:
                    contraction_indices[ind].append(_IndPos(i, j))
        # 返回收缩指数及其位置对象的列表
        return contraction_indices

    # 计算具有给定指数的参数的数量
    def count_args_with_index(self, index: int) -> int:
        """
        Count the number of arguments that have the given index.
        """
        # 计数器初始化
        counter: int = 0
        # 遍历每个带有指数的参数
        for arg_with_ind in self.args_with_ind:
            # 如果指定的指数在当前参数的指数列表中，则增加计数器
            if index in arg_with_ind.indices:
                counter += 1
        # 返回具有给定指数的参数数量
        return counter

    # 返回具有给定指数的参数列表
    def get_args_with_index(self, index: int) -> List[_ArgE]:
        """
        Get a list of arguments having the given index.
        """
        # 初始化结果列表，包含所有具有指定指数的参数
        ret: List[_ArgE] = [i for i in self.args_with_ind if index in i.indices]
        # 返回结果列表
        return ret

    # 返回对角线指数的数量
    @property
    def number_of_diagonal_indices(self):
        # 初始化一个集合用于存储对角线指数
        data = set()
        # 遍历每个带有指数的参数
        for arg in self.args_with_ind:
            # 将所有小于0且不为None的指数添加到集合中
            data.update({i for i in arg.indices if i is not None and i < 0})
        # 返回对角线指数的数量
        return len(data)

    # 跟踪排列的起始点
    def track_permutation_start(self):
        # 初始化排列和对角线排列的空列表以及计数器
        permutation = []
        perm_diag = []
        counter = 0
        counter2 = -1
        # 遍历每个带有指数的参数
        for arg_with_ind in self.args_with_ind:
            # 初始化当前参数的排列列表
            perm = []
            # 遍历当前参数的指数
            for i in arg_with_ind.indices:
                # 如果指数不为None
                if i is not None:
                    # 如果指数小于0，则将其添加到对角线排列中，并递减计数器2
                    if i < 0:
                        perm_diag.append(counter2)
                        counter2 -= 1
                    continue
                # 将当前位置添加到排列列表中，并增加计数器
                perm.append(counter)
                counter += 1
            # 将当前参数的排列列表添加到排列列表中
            permutation.append(perm)
        # 计算最大索引，并生成对角线排列列表
        max_ind = max(max(i) if i else -1 for i in permutation) if permutation else -1
        perm_diag = [max_ind - i for i in perm_diag]
        # 将排列列表和对角线排列列表合并为单个列表，并赋值给成员变量
        self._track_permutation = permutation + [perm_diag]
    # 将指定目标元素的置换轨迹扩展到来源元素的置换轨迹中
    def track_permutation_merge(self, destination: _ArgE, from_element: _ArgE):
        # 获取目标元素在 args_with_ind 中的索引位置
        index_destination = self.args_with_ind.index(destination)
        # 获取来源元素在 args_with_ind 中的索引位置
        index_element = self.args_with_ind.index(from_element)
        # 将来源元素的置换轨迹数据合并到目标元素的置换轨迹中
        self._track_permutation[index_destination].extend(self._track_permutation[index_element]) # type: ignore
        # 移除来源元素的置换轨迹数据
        self._track_permutation.pop(index_element) # type: ignore

    # 获取参数 arg 在所有自由索引中的绝对位置范围
    def get_absolute_free_range(self, arg: _ArgE) -> typing.Tuple[int, int]:
        """
        返回参数 arg 在所有自由索引中的绝对位置范围。
        """
        counter = 0
        for arg_with_ind in self.args_with_ind:
            # 计算当前参数 arg_with_ind 中空索引的数量
            number_free_indices = len([i for i in arg_with_ind.indices if i is None])
            # 如果找到参数 arg，返回其绝对位置范围
            if arg_with_ind == arg:
                return counter, counter + number_free_indices
            counter += number_free_indices
        # 如果未找到对应的参数 arg，抛出索引错误异常
        raise IndexError("argument not found")

    # 获取参数 arg 的绝对索引范围，忽略虚拟索引
    def get_absolute_range(self, arg: _ArgE) -> typing.Tuple[int, int]:
        """
        返回参数 arg 的绝对索引范围，忽略虚拟索引。
        """
        counter = 0
        for arg_with_ind in self.args_with_ind:
            # 获取当前参数 arg_with_ind 的索引数量
            number_indices = len(arg_with_ind.indices)
            # 如果找到参数 arg，返回其绝对索引范围
            if arg_with_ind == arg:
                return counter, counter + number_indices
            counter += number_indices
        # 如果未找到对应的参数 arg，抛出索引错误异常
        raise IndexError("argument not found")
# 返回表达式的秩（rank）：
# 如果表达式是 MatrixExpr 或 MatrixElement 类型，则秩为 2
# 如果表达式是 _CodegenArrayAbstract 类型，则秩为其形状的长度
# 如果表达式是 NDimArray 类型，则使用其 rank() 方法获取秩
# 如果表达式是 Indexed 类型，则返回其 rank 属性
# 如果表达式是 IndexedBase 类型，则获取其形状，若形状为 None 则返回 -1，否则返回形状的长度
# 如果表达式具有 "shape" 属性，则返回其形状的长度
# 其它情况返回 0
def get_rank(expr):
    if isinstance(expr, (MatrixExpr, MatrixElement)):
        return 2
    if isinstance(expr, _CodegenArrayAbstract):
        return len(expr.shape)
    if isinstance(expr, NDimArray):
        return expr.rank()
    if isinstance(expr, Indexed):
        return expr.rank
    if isinstance(expr, IndexedBase):
        shape = expr.shape
        if shape is None:
            return -1
        else:
            return len(shape)
    if hasattr(expr, "shape"):
        return len(expr.shape)
    return 0


# 返回表达式的子秩（subrank）：
# 如果表达式是 _CodegenArrayAbstract 类型，则调用其 subrank() 方法返回子秩
# 否则调用 get_rank() 返回其秩
def _get_subrank(expr):
    if isinstance(expr, _CodegenArrayAbstract):
        return expr.subrank()
    return get_rank(expr)


# 返回表达式的子秩列表（subranks）：
# 如果表达式是 _CodegenArrayAbstract 类型，则返回其 subranks 属性
# 否则返回一个包含表达式秩的列表
def _get_subranks(expr):
    if isinstance(expr, _CodegenArrayAbstract):
        return expr.subranks
    else:
        return [get_rank(expr)]


# 返回表达式的形状（shape）：
# 如果表达式具有 "shape" 属性，则返回其形状
# 否则返回空元组 ()
def get_shape(expr):
    if hasattr(expr, "shape"):
        return expr.shape
    return ()


# 如果表达式是 PermuteDims 类型，则调用其 nest_permutation() 方法返回嵌套排列
# 否则直接返回表达式
def nest_permutation(expr):
    if isinstance(expr, PermuteDims):
        return expr.nest_permutation()
    else:
        return expr


# 返回一个 ArrayTensorProduct 对象：
# 根据传入的参数创建 ArrayTensorProduct 实例，并使用 canonicalize=True 进行规范化
def _array_tensor_product(*args, **kwargs):
    return ArrayTensorProduct(*args, canonicalize=True, **kwargs)


# 返回一个 ArrayContraction 对象：
# 根据传入的参数创建 ArrayContraction 实例，并使用 canonicalize=True 进行规范化
def _array_contraction(expr, *contraction_indices, **kwargs):
    return ArrayContraction(expr, *contraction_indices, canonicalize=True, **kwargs)


# 返回一个 ArrayDiagonal 对象：
# 根据传入的参数创建 ArrayDiagonal 实例，并使用 canonicalize=True 进行规范化
def _array_diagonal(expr, *diagonal_indices, **kwargs):
    return ArrayDiagonal(expr, *diagonal_indices, canonicalize=True, **kwargs)


# 返回一个 PermuteDims 对象：
# 根据传入的参数创建 PermuteDims 实例，并使用 canonicalize=True 进行规范化
def _permute_dims(expr, permutation, **kwargs):
    return PermuteDims(expr, permutation, canonicalize=True, **kwargs)


# 返回一个 ArrayAdd 对象：
# 根据传入的参数创建 ArrayAdd 实例，并使用 canonicalize=True 进行规范化
def _array_add(*args, **kwargs):
    return ArrayAdd(*args, canonicalize=True, **kwargs)


# 返回一个 ArrayElement 对象：
# 根据传入的表达式和索引创建 ArrayElement 实例
def _get_array_element_or_slice(expr, indices):
    return ArrayElement(expr, indices)
```