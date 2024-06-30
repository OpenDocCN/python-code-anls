# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\from_indexed_to_array.py`

```
from collections import defaultdict  # 导入 defaultdict 类，用于创建默认字典

from sympy import Function  # 导入 Function 类，用于定义符号函数
from sympy.combinatorics.permutations import _af_invert  # 导入 _af_invert 函数，用于排列的反演
from sympy.concrete.summations import Sum  # 导入 Sum 类，用于表示求和表达式
from sympy.core.add import Add  # 导入 Add 类，用于表示加法表达式
from sympy.core.mul import Mul  # 导入 Mul 类，用于表示乘法表达式
from sympy.core.numbers import Integer  # 导入 Integer 类，用于表示整数
from sympy.core.power import Pow  # 导入 Pow 类，用于表示幂运算
from sympy.core.sorting import default_sort_key  # 导入 default_sort_key 函数，用于排序
from sympy.functions.special.tensor_functions import KroneckerDelta  # 导入 KroneckerDelta 函数，用于表示克罗内克 δ 函数
from sympy.tensor.array.expressions import ArrayElementwiseApplyFunc  # 导入 ArrayElementwiseApplyFunc 类，用于数组元素级函数应用
from sympy.tensor.indexed import (Indexed, IndexedBase)  # 导入 Indexed 和 IndexedBase 类，用于表示索引和索引基
from sympy.combinatorics import Permutation  # 导入 Permutation 类，用于表示置换
from sympy.matrices.expressions.matexpr import MatrixElement  # 导入 MatrixElement 类，用于表示矩阵元素
from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal, \
    get_shape, ArrayElement, _array_tensor_product, _array_diagonal, _array_contraction, _array_add, \
    _permute_dims, OneArray, ArrayAdd  # 导入各种数组操作的函数和类

from sympy.tensor.array.expressions.utils import _get_argindex, _get_diagonal_indices  # 导入数组表达式的辅助函数

def convert_indexed_to_array(expr, first_indices=None):
    r"""
    Parse indexed expression into a form useful for code generation.

    Examples
    ========

    >>> from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
    >>> from sympy import MatrixSymbol, Sum, symbols

    >>> i, j, k, d = symbols("i j k d")
    >>> M = MatrixSymbol("M", d, d)
    >>> N = MatrixSymbol("N", d, d)

    Recognize the trace in summation form:

    >>> expr = Sum(M[i, i], (i, 0, d-1))
    >>> convert_indexed_to_array(expr)
    ArrayContraction(M, (0, 1))

    Recognize the extraction of the diagonal by using the same index `i` on
    both axes of the matrix:

    >>> expr = M[i, i]
    >>> convert_indexed_to_array(expr)
    ArrayDiagonal(M, (0, 1))

    This function can help perform the transformation expressed in two
    different mathematical notations as:

    `\sum_{j=0}^{N-1} A_{i,j} B_{j,k} \Longrightarrow \mathbf{A}\cdot \mathbf{B}`

    Recognize the matrix multiplication in summation form:

    >>> expr = Sum(M[i, j]*N[j, k], (j, 0, d-1))
    >>> convert_indexed_to_array(expr)
    ArrayContraction(ArrayTensorProduct(M, N), (1, 2))

    Specify that ``k`` has to be the starting index:

    >>> convert_indexed_to_array(expr, first_indices=[k])
    ArrayContraction(ArrayTensorProduct(N, M), (0, 3))
    """

    result, indices = _convert_indexed_to_array(expr)  # 将索引表达式转换为数组表达式形式，并获取索引列表

    if any(isinstance(i, (int, Integer)) for i in indices):
        result = ArrayElement(result, indices)  # 如果索引中包含整数，则将结果转换为数组元素
        indices = []

    if not first_indices:
        return result  # 如果没有指定首要索引，直接返回结果

    def _check_is_in(elem, indices):
        if elem in indices:
            return True
        if any(elem in i for i in indices if isinstance(i, frozenset)):
            return True
        return False

    repl = {j: i for i in indices if isinstance(i, frozenset) for j in i}  # 创建替换字典，处理冻结集合类型的索引
    first_indices = [repl.get(i, i) for i in first_indices]  # 使用替换字典对首要索引进行处理
    for i in first_indices:
        if not _check_is_in(i, indices):
            first_indices.remove(i)  # 移除不在索引列表中的首要索引
    # 扩展 first_indices 列表，添加所有不在 first_indices 中的索引
    first_indices.extend([i for i in indices if not _check_is_in(i, first_indices)])
    
    # 定义一个函数 _get_pos(elem, indices)，用于获取元素在 indices 中的位置
    def _get_pos(elem, indices):
        # 如果 elem 在 indices 中，直接返回其索引位置
        if elem in indices:
            return indices.index(elem)
        # 否则，遍历 indices，查找 elem 在其中的位置
        for i, e in enumerate(indices):
            # 如果当前元素不是 frozenset 类型，则继续下一个循环
            if not isinstance(e, frozenset):
                continue
            # 如果 elem 存在于当前 frozenset 中，返回其索引位置
            if elem in e:
                return i
        # 如果 elem 未找到，抛出 ValueError 异常
        raise ValueError("not found")
    
    # 根据 indices 中元素的位置重新排列，生成 permutation 列表
    permutation = _af_invert([_get_pos(i, first_indices) for i in indices])
    
    # 如果 result 是 ArrayAdd 类型的实例
    if isinstance(result, ArrayAdd):
        # 对 result.args 中的每个参数执行 _permute_dims 函数，使用 permutation 进行维度置换，并返回结果
        return _array_add(*[_permute_dims(arg, permutation) for arg in result.args])
    else:
        # 否则，对 result 执行 _permute_dims 函数，使用 permutation 进行维度置换，并返回结果
        return _permute_dims(result, permutation)
# 将一个 Indexed 类型的表达式转换为数组表示形式
def _convert_indexed_to_array(expr):
    # 如果表达式是乘法表达式
    if isinstance(expr, Mul):
        # 分解表达式的参数和索引
        args, indices = zip(*[_convert_indexed_to_array(arg) for arg in expr.args])
        
        # 检查是否存在 KroneckerDelta 对象：
        kronecker_delta_repl = {}
        for arg in args:
            if not isinstance(arg, KroneckerDelta):
                continue
            # 对两个索引进行对角化处理：
            i, j = arg.indices
            kindices = set(arg.indices)
            if i in kronecker_delta_repl:
                kindices.update(kronecker_delta_repl[i])
            if j in kronecker_delta_repl:
                kindices.update(kronecker_delta_repl[j])
            kindices = frozenset(kindices)
            for index in kindices:
                kronecker_delta_repl[index] = kindices
        
        # 移除 KroneckerDelta 对象，它们的关系应由 ArrayDiagonal 处理：
        newargs = []
        newindices = []
        for arg, loc_indices in zip(args, indices):
            if isinstance(arg, KroneckerDelta):
                continue
            newargs.append(arg)
            newindices.append(loc_indices)
        
        # 展开索引并获取对角线索引和返回索引
        flattened_indices = [kronecker_delta_repl.get(j, j) for i in newindices for j in i]
        diagonal_indices, ret_indices = _get_diagonal_indices(flattened_indices)
        
        # 对新参数进行数组张量积操作
        tp = _array_tensor_product(*newargs)
        
        # 如果存在对角线索引，则返回数组对角线化结果及返回索引；否则返回数组张量积结果及返回索引
        if diagonal_indices:
            return _array_diagonal(tp, *diagonal_indices), ret_indices
        else:
            return tp, ret_indices
    
    # 如果表达式是矩阵元素
    if isinstance(expr, MatrixElement):
        indices = expr.args[1:]
        diagonal_indices, ret_indices = _get_diagonal_indices(indices)
        
        # 如果存在对角线索引，则返回数组对角线化结果及返回索引；否则返回矩阵元素及返回索引
        if diagonal_indices:
            return _array_diagonal(expr.args[0], *diagonal_indices), ret_indices
        else:
            return expr.args[0], ret_indices
    
    # 如果表达式是数组元素
    if isinstance(expr, ArrayElement):
        indices = expr.indices
        diagonal_indices, ret_indices = _get_diagonal_indices(indices)
        
        # 如果存在对角线索引，则返回数组对角线化结果及返回索引；否则返回数组元素名称及返回索引
        if diagonal_indices:
            return _array_diagonal(expr.name, *diagonal_indices), ret_indices
        else:
            return expr.name, ret_indices
    
    # 如果表达式是 Indexed 类型
    if isinstance(expr, Indexed):
        indices = expr.indices
        diagonal_indices, ret_indices = _get_diagonal_indices(indices)
        
        # 如果存在对角线索引，则返回数组对角线化结果及返回索引；否则返回 Indexed 对象基础部分及返回索引
        if diagonal_indices:
            return _array_diagonal(expr.base, *diagonal_indices), ret_indices
        else:
            return expr.args[0], ret_indices
    
    # 如果表达式是 IndexedBase 类型，暂未实现处理方式
    if isinstance(expr, IndexedBase):
        raise NotImplementedError
    
    # 如果表达式是 KroneckerDelta 类型，则直接返回该对象及其索引
    if isinstance(expr, KroneckerDelta):
        return expr, expr.indices
    # 如果表达式是加法表达式（Add类的实例）
    if isinstance(expr, Add):
        # 对表达式的每个参数进行转换成数组，并获取它们的索引
        args, indices = zip(*[_convert_indexed_to_array(arg) for arg in expr.args])
        args = list(args)
        
        # 检查所有的索引是否兼容，否则扩展维度
        index0 = []
        shape0 = []
        for arg, arg_indices in zip(args, indices):
            arg_indices_set = set(arg_indices)
            # 找出缺失的索引
            arg_indices_missing = arg_indices_set.difference(index0)
            # 将缺失的索引添加到index0中
            index0.extend([i for i in arg_indices if i in arg_indices_missing])
            # 获取参数的形状
            arg_shape = get_shape(arg)
            # 根据缺失的索引获取相应的形状
            shape0.extend([arg_shape[i] for i, e in enumerate(arg_indices) if e in arg_indices_missing])
        
        # 遍历每个参数及其索引
        for i, (arg, arg_indices) in enumerate(zip(args, indices)):
            # 如果参数的索引长度小于index0的长度
            if len(arg_indices) < len(index0):
                # 找出缺失的索引位置
                missing_indices_pos = [i for i, e in enumerate(index0) if e not in arg_indices]
                # 获取缺失的形状
                missing_shape = [shape0[i] for i in missing_indices_pos]
                # 将缺失的索引添加到arg_indices中，并在前面添加缺失的形状
                arg_indices = tuple(index0[j] for j in missing_indices_pos) + arg_indices
                # 对缺失的维度进行张量积
                args[i] = _array_tensor_product(OneArray(*missing_shape), args[i])
            
            # 创建索引的排列
            permutation = Permutation([arg_indices.index(j) for j in index0])
            # 对参数执行索引置换
            args[i] = _permute_dims(args[i], permutation)
        
        # 返回数组的加法和索引
        return _array_add(*args), tuple(index0)
    
    # 如果表达式是幂次方表达式（Pow类的实例）
    if isinstance(expr, Pow):
        # 将基础表达式转换为数组
        subexpr, subindices = _convert_indexed_to_array(expr.base)
        # 如果指数是整数或Integer类型
        if isinstance(expr.exp, (int, Integer)):
            # 创建对角线的位置
            diags = zip(*[(2*i, 2*i + 1) for i in range(expr.exp)])
            # 对子表达式执行张量积和对角线操作
            arr = _array_diagonal(_array_tensor_product(*[subexpr for i in range(expr.exp)]), *diags)
            return arr, subindices
    
    # 如果表达式是函数表达式（Function类的实例）
    if isinstance(expr, Function):
        # 将函数的参数转换为数组
        subexpr, subindices = _convert_indexed_to_array(expr.args[0])
        # 应用逐元素函数操作到子表达式
        return ArrayElementwiseApplyFunc(type(expr), subexpr), subindices
    
    # 默认情况下返回表达式和空的索引元组
    return expr, ()
```