# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\from_array_to_matrix.py`

```
# 导入模块 itertools，用于高效的迭代工具函数和生成器
import itertools
# 导入模块 defaultdict，用于创建默认值为列表的字典
from collections import defaultdict
# 导入类型别名 tTuple, tUnion, FrozenSet, tDict, List, Optional
from typing import Tuple as tTuple, Union as tUnion, FrozenSet, Dict as tDict, List, Optional
# 导入函数 singledispatch，用于创建泛型函数的装饰器
from functools import singledispatch
# 导入函数 accumulate，用于累积计算迭代器中的值
from itertools import accumulate

# 导入 sympy 中的数学表达式和矩阵运算相关模块和类
from sympy import MatMul, Basic, Wild, KroneckerProduct
# 导入 sympy 的假设模块 Q 和 ask 函数，用于符号逻辑推理
from sympy.assumptions.ask import (Q, ask)
# 导入 sympy 中的数学表达式 Mul
from sympy.core.mul import Mul
# 导入 sympy 中的单例模块 S
from sympy.core.singleton import S
# 导入 sympy 中矩阵表达式的对角矩阵类 DiagMatrix
from sympy.matrices.expressions.diagonal import DiagMatrix
# 导入 sympy 中哈达玛积的计算函数 hadamard_product 和类 HadamardPower
from sympy.matrices.expressions.hadamard import hadamard_product, HadamardPower
# 导入 sympy 中矩阵表达式的基类 MatrixExpr
from sympy.matrices.expressions.matexpr import MatrixExpr
# 导入 sympy 中特殊矩阵表达式类 Identity, ZeroMatrix, OneMatrix
from sympy.matrices.expressions.special import (Identity, ZeroMatrix, OneMatrix)
# 导入 sympy 中矩阵表达式的迹类 Trace
from sympy.matrices.expressions.trace import Trace
# 导入 sympy 中矩阵表达式的转置类 Transpose
from sympy.matrices.expressions.transpose import Transpose
# 导入 sympy 中排列组合的逆类和 Permutation 类
from sympy.combinatorics.permutations import _af_invert, Permutation
# 导入 sympy 中矩阵基类 MatrixBase
from sympy.matrices.matrixbase import MatrixBase
# 导入 sympy 中矩阵表达式的逐元素应用函数类 ElementwiseApplyFunction
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
# 导入 sympy 中矩阵表达式的元素访问类 MatrixElement
from sympy.matrices.expressions.matexpr import MatrixElement
# 导入 sympy 中张量数组表达式相关类 PermuteDims, ArrayDiagonal, ArrayTensorProduct, OneArray, get_rank 等
from sympy.tensor.array.expressions.array_expressions import PermuteDims, ArrayDiagonal, \
    ArrayTensorProduct, OneArray, get_rank, _get_subrank, ZeroArray, ArrayContraction, \
    ArrayAdd, _CodegenArrayAbstract, get_shape, ArrayElementwiseApplyFunc, _ArrayExpr, _EditArrayContraction, _ArgE, \
    ArrayElement, _array_tensor_product, _array_contraction, _array_diagonal, _array_add, _permute_dims
# 导入 sympy 中张量数组表达式工具类 _get_mapping_from_subranks
from sympy.tensor.array.expressions.utils import _get_mapping_from_subranks


# 定义函数 _get_candidate_for_matmul_from_contraction，用于从收缩操作中获取 MatMul 的候选项
def _get_candidate_for_matmul_from_contraction(scan_indices: List[Optional[int]], remaining_args: List[_ArgE]) -> tTuple[Optional[_ArgE], bool, int]:
    # 提取所有非空的索引值为整数列表 scan_indices_int
    scan_indices_int: List[int] = [i for i in scan_indices if i is not None]
    # 若没有有效的索引，返回空的候选项、False 标志和 -1 的元组
    if len(scan_indices_int) == 0:
        return None, False, -1

    # 初始化变量 transpose 为 False，candidate 为 None，candidate_index 为 -1
    transpose: bool = False
    candidate: Optional[_ArgE] = None
    candidate_index: int = -1

    # 遍历剩余参数列表 remaining_args
    for arg_with_ind2 in remaining_args:
        # 如果元素不是 MatrixExpr 类型，则跳过本次循环
        if not isinstance(arg_with_ind2.element, MatrixExpr):
            continue
        # 遍历扫描索引列表 scan_indices_int
        for index in scan_indices_int:
            # 如果已选择候选索引且当前索引不同于候选索引，则继续下一次循环
            if candidate_index != -1 and candidate_index != index:
                continue
            # 如果当前索引在参数的索引列表中
            if index in arg_with_ind2.indices:
                # 如果参数的索引列表只包含当前索引，说明索引重复了两次
                if set(arg_with_ind2.indices) == {index}:
                    candidate = None
                    break
                # 如果候选项为空，则设置为当前参数
                if candidate is None:
                    candidate = arg_with_ind2
                    candidate_index = index
                    # 判断是否需要转置
                    transpose = (index == arg_with_ind2.indices[1])
                else:
                    # 否则说明索引重复超过两次，候选项置空，跳出循环
                    candidate = None
                    break
    # 返回候选项、是否需要转置和候选索引的元组
    return candidate, transpose, candidate_index
# 将候选项插入到编辑器中
def _insert_candidate_into_editor(editor: _EditArrayContraction, arg_with_ind: _ArgE, candidate: _ArgE, transpose1: bool, transpose2: bool):
    other = candidate.element  # 获取候选项的元素
    other_index: Optional[int]
    if transpose2:
        other = Transpose(other)  # 如果需要转置，对候选项元素进行转置操作
        other_index = candidate.indices[0]  # 获取候选项的第一个索引
    else:
        other_index = candidate.indices[1]  # 获取候选项的第二个索引
    # 计算新元素，根据需要进行转置
    new_element = (Transpose(arg_with_ind.element) if transpose1 else arg_with_ind.element) * other
    editor.args_with_ind.remove(candidate)  # 从编辑器中移除候选项
    new_arge = _ArgE(new_element)  # 创建新的 _ArgE 对象
    return new_arge, other_index  # 返回新的 _ArgE 对象和其他索引


# 识别第一类支持函数
def _support_function_tp1_recognize(contraction_indices, args):
    if len(contraction_indices) == 0:
        return _a2m_tensor_product(*args)  # 如果缩并索引为空，返回张量积结果

    ac = _array_contraction(_array_tensor_product(*args), *contraction_indices)  # 执行数组缩并操作
    editor = _EditArrayContraction(ac)  # 创建编辑器对象
    editor.track_permutation_start()  # 开始追踪置换操作

    while True:
        flag_stop = True  # 停止标志设为真
        for i, arg_with_ind in enumerate(editor.args_with_ind):
            if not isinstance(arg_with_ind.element, MatrixExpr):
                continue  # 如果元素不是 MatrixExpr 类型，则继续下一次循环

            first_index = arg_with_ind.indices[0]  # 获取第一个索引
            second_index = arg_with_ind.indices[1]  # 获取第二个索引

            first_frequency = editor.count_args_with_index(first_index)  # 计算第一个索引出现的频率
            second_frequency = editor.count_args_with_index(second_index)  # 计算第二个索引出现的频率

            # 如果第一个索引和第二个索引都不为空，并且它们相等且出现频率为 1
            if first_index is not None and first_frequency == 1 and first_index == second_index:
                flag_stop = False  # 停止标志设为假
                arg_with_ind.element = Trace(arg_with_ind.element)._normalize()  # 对元素执行迹运算并归一化
                arg_with_ind.indices = []  # 清空索引
                break  # 跳出循环

            scan_indices = []
            if first_frequency == 2:
                scan_indices.append(first_index)  # 如果第一个索引出现频率为 2，加入扫描索引列表
            if second_frequency == 2:
                scan_indices.append(second_index)  # 如果第二个索引出现频率为 2，加入扫描索引列表

            # 从缩并操作的候选项中获取矩阵乘法的候选项及其相关信息
            candidate, transpose, found_index = _get_candidate_for_matmul_from_contraction(scan_indices, editor.args_with_ind[i+1:])
            if candidate is not None:
                flag_stop = False  # 停止标志设为假
                editor.track_permutation_merge(arg_with_ind, candidate)  # 追踪合并操作
                transpose1 = found_index == first_index  # 判断是否为第一个索引
                new_arge, other_index = _insert_candidate_into_editor(editor, arg_with_ind, candidate, transpose1, transpose)  # 插入候选项到编辑器中
                if found_index == first_index:
                    new_arge.indices = [second_index, other_index]  # 设置新的索引组合
                else:
                    new_arge.indices = [first_index, other_index]  # 设置新的索引组合
                set_indices = set(new_arge.indices)  # 创建索引集合
                if len(set_indices) == 1 and set_indices != {None}:
                    # 如果索引集合长度为 1 且不为 {None}，表示这是一个迹运算
                    new_arge.element = Trace(new_arge.element)._normalize()  # 对元素执行迹运算并归一化
                    new_arge.indices = []  # 清空索引
                editor.args_with_ind[i] = new_arge  # 更新编辑器中的元素
                # TODO: 是否需要这个 break？
                break  # 跳出循环

        if flag_stop:
            break  # 如果停止标志为真，则跳出循环

    editor.refresh_indices()  # 刷新索引
    return editor.to_array_contraction()  # 返回编辑器的数组缩并结果
# 在给定的 ArrayTensorProduct 表达式中查找并重写包含平凡矩阵（即形状为 (1, 1) 的矩阵）的情况，
# 尝试查找适当的非平凡 MatMul，以便将表达式插入其中。

# 例如，如果 "a" 的形状为 (1, 1)，"b" 的形状为 (k, 1)，那么表达式 "_array_tensor_product(a, b*b.T)" 可以重写为 "b*a*b.T"

def _find_trivial_matrices_rewrite(expr: ArrayTensorProduct):
    trivial_matrices = []   # 存储找到的平凡矩阵
    pos: Optional[int] = None   # 插入非平凡 MatMul 的位置
    first: Optional[MatrixExpr] = None   # MatMul 的第一部分
    second: Optional[MatrixExpr] = None  # MatMul 的第二部分
    removed: List[int] = []   # 已移除的位置列表
    counter: int = 0   # 辅助计数器
    args: List[Optional[Basic]] = list(expr.args)   # 表达式的参数列表

    # 遍历表达式的每个参数
    for i, arg in enumerate(expr.args):
        if isinstance(arg, MatrixExpr):
            if arg.shape == (1, 1):
                trivial_matrices.append(arg)   # 将平凡矩阵加入列表
                args[i] = None   # 将当前位置的参数设为 None，表示移除
                removed.extend([counter, counter+1])   # 记录已移除的位置范围
            elif pos is None and isinstance(arg, MatMul):
                margs = arg.args
                # 寻找是否存在合适的非平凡 MatMul
                for j, e in enumerate(margs):
                    if isinstance(e, MatrixExpr) and e.shape[1] == 1:
                        pos = i
                        first = MatMul.fromiter(margs[:j+1])
                        second = MatMul.fromiter(margs[j+1:])
                        break
        counter += get_rank(arg)   # 更新计数器

    # 如果未找到合适的非平凡 MatMul，则返回原始表达式和空移除列表
    if pos is None:
        return expr, []

    # 构造新的表达式并返回
    args[pos] = (first * MatMul.fromiter(i for i in trivial_matrices) * second).doit()
    return _array_tensor_product(*[i for i in args if i is not None]), removed


# 在给定的 ArrayTensorProduct 表达式中查找并处理 Kronecker 乘积的广播情况
def _find_trivial_kronecker_products_broadcast(expr: ArrayTensorProduct):
    newargs: List[Basic] = []   # 存储处理后的参数列表
    removed = []   # 已移除的位置列表
    count_dims = 0   # 维度计数器

    # 遍历表达式的每个参数
    for arg in expr.args:
        count_dims += get_rank(arg)   # 更新维度计数

        shape = get_shape(arg)   # 获取当前参数的形状
        current_range = [count_dims-i for i in range(len(shape), 0, -1)]   # 当前参数在表达式中的位置范围

        # 处理形状为 (1, 1) 的参数，并且前一个参数不是形状为 (1, 1) 的矩阵表达式时
        if (shape == (1, 1) and len(newargs) > 0 and 1 not in get_shape(newargs[-1]) and
            isinstance(newargs[-1], MatrixExpr) and isinstance(arg, MatrixExpr)):
            # 使用 KroneckerProduct 对象进行广播处理
            newargs[-1] = KroneckerProduct(newargs[-1], arg)
            removed.extend(current_range)   # 记录已移除的位置范围

        # 处理前一个参数形状为 (1, 1) 并且当前参数不是形状为 (1, 1) 的情况
        elif 1 not in shape and len(newargs) > 0 and get_shape(newargs[-1]) == (1, 1):
            # 使用 KroneckerProduct 对象进行广播处理
            newargs[-1] = KroneckerProduct(newargs[-1], arg)
            prev_range = [i for i in range(min(current_range)) if i not in removed]
            removed.extend(prev_range[-2:])   # 记录已移除的位置范围

        else:
            newargs.append(arg)   # 其他情况直接添加参数到新列表中

    # 返回处理后的新表达式和已移除的位置列表
    return _array_tensor_product(*newargs), removed


# 处理 ZeroArray 类型对象，将其转换为 ZeroMatrix 类型对象（如果维度为 2）
# 否则直接返回原对象
@_array2matrix.register(ZeroArray)
def _(expr: ZeroArray):
    if get_rank(expr) == 2:
        return ZeroMatrix(*expr.shape)
    else:
        return expr


# 处理 ArrayTensorProduct 类型对象，递归地将其参数转换为矩阵类型对象
@_array2matrix.register(ArrayTensorProduct)
def _(expr: ArrayTensorProduct):
    return _a2m_tensor_product(*[_array2matrix(arg) for arg in expr.args])

# 后续还有其他处理 ArrayContraction 的代码，但这里省略
# 定义函数 _，处理 ArrayContraction 类型的表达式
def _(expr: ArrayContraction):
    # 对表达式进行扁平化处理对角线缩并
    expr = expr.flatten_contraction_of_diagonal()
    # 识别并移除可移除的单位矩阵
    expr = identify_removable_identity_matrices(expr)
    # 分割多重缩并
    expr = expr.split_multiple_contractions()
    # 识别哈达玛积
    expr = identify_hadamard_products(expr)
    
    # 如果 expr 不是 ArrayContraction 类型，则将其转换为矩阵表达式
    if not isinstance(expr, ArrayContraction):
        return _array2matrix(expr)
    
    # 获取子表达式和缩并的索引
    subexpr = expr.expr
    contraction_indices: tTuple[tTuple[int]] = expr.contraction_indices
    
    # 检查缩并的情况，特别处理 ((0,), (1,)) 或者 ((0,),) 和 subexpr.shape[1] == 1 或者 ((1,),) 和 subexpr.shape[0] == 1 的情况
    if contraction_indices == ((0,), (1,)) or (
        contraction_indices == ((0,),) and subexpr.shape[1] == 1
    ) or (
        contraction_indices == ((1,),) and subexpr.shape[0] == 1
    ):
        # 获取子表达式的形状
        shape = subexpr.shape
        # 将子表达式转换为矩阵表达式
        subexpr = _array2matrix(subexpr)
        # 如果转换后是 MatrixExpr 类型，则返回对应的矩阵乘积
        if isinstance(subexpr, MatrixExpr):
            return OneMatrix(1, shape[0])*subexpr*OneMatrix(shape[1], 1)
    
    # 如果子表达式是 ArrayTensorProduct 类型
    if isinstance(subexpr, ArrayTensorProduct):
        # 将子表达式转换为矩阵后进行缩并操作
        newexpr = _array_contraction(_array2matrix(subexpr), *contraction_indices)
        # 更新缩并的索引
        contraction_indices = newexpr.contraction_indices
        # 如果任意子秩大于2，则处理加法项
        if any(i > 2 for i in newexpr.subranks):
            # 计算加法项的和
            addends = _array_add(*[_a2m_tensor_product(*j) for j in itertools.product(*[i.args if isinstance(i,
                                                                                                             ArrayAdd) else [i] for i in expr.expr.args])])
            # 对加法项进行缩并操作
            newexpr = _array_contraction(addends, *contraction_indices)
        # 如果新表达式是 ArrayAdd 类型，则转换为矩阵表达式
        if isinstance(newexpr, ArrayAdd):
            ret = _array2matrix(newexpr)
            return ret
        # 断言新表达式是 ArrayContraction 类型
        assert isinstance(newexpr, ArrayContraction)
        # 识别并返回新表达式
        ret = _support_function_tp1_recognize(contraction_indices, list(newexpr.expr.args))
        return ret
    
    # 如果子表达式不是 _CodegenArrayAbstract 类型
    elif not isinstance(subexpr, _CodegenArrayAbstract):
        # 将子表达式转换为矩阵表达式
        ret = _array2matrix(subexpr)
        # 如果转换后是 MatrixExpr 类型，则断言缩并的索引为 ((0, 1),)，并返回对应的矩阵迹
        if isinstance(ret, MatrixExpr):
            assert expr.contraction_indices == ((0, 1),)
            return _a2m_trace(ret)
        # 否则，进行缩并操作并返回结果
        else:
            return _array_contraction(ret, *expr.contraction_indices)


# 注册 ArrayDiagonal 类型的 _array2matrix 函数
@_array2matrix.register(ArrayDiagonal)
def _(expr: ArrayDiagonal):
    # 将表达式转换为矩阵表达式，并处理哈达玛积
    pexpr = _array_diagonal(_array2matrix(expr.expr), *expr.diagonal_indices)
    pexpr = identify_hadamard_products(pexpr)
    # 如果表达式是 ArrayDiagonal 类型，则将其转换为对角矩阵
    if isinstance(pexpr, ArrayDiagonal):
        pexpr = _array_diag2contr_diagmatrix(pexpr)
    # 如果原表达式等于处理后的表达式，则返回原表达式，否则继续处理
    if expr == pexpr:
        return expr
    return _array2matrix(pexpr)


# 注册 PermuteDims 类型的 _array2matrix 函数
@_array2matrix.register(PermuteDims)
def _(expr: PermuteDims):
    # 如果排列形式是 [1, 0]，则对表达式进行转置并转换为矩阵表达式
    if expr.permutation.array_form == [1, 0]:
        return _a2m_transpose(_array2matrix(expr.expr))
    # 如果表达式的子表达式是 ArrayTensorProduct 类型
    elif isinstance(expr.expr, ArrayTensorProduct):
        # 获取子表达式的各个张量的秩
        ranks = expr.expr.subranks
        # 计算表达式的逆置换
        inv_permutation = expr.permutation**(-1)
        # 生成新的索引范围，根据逆置换重新排列
        newrange = [inv_permutation(i) for i in range(sum(ranks))]
        # 初始化新的位置列表
        newpos = []
        # 计数器初始化
        counter = 0
        # 遍历各个张量的秩
        for rank in ranks:
            # 将新的位置范围添加到新位置列表中
            newpos.append(newrange[counter:counter+rank])
            # 更新计数器
            counter += rank
        # 初始化新的参数列表、新的置换列表和标量列表
        newargs = []
        newperm = []
        scalars = []
        # 遍历新位置列表和表达式的参数
        for pos, arg in zip(newpos, expr.expr.args):
            # 如果位置为空，将参数转换为矩阵并添加到标量列表中
            if len(pos) == 0:
                scalars.append(_array2matrix(arg))
            # 如果位置已排序，将参数和位置的第一个元素添加到新参数列表中，并扩展新置换列表
            elif pos == sorted(pos):
                newargs.append((_array2matrix(arg), pos[0]))
                newperm.extend(pos)
            # 如果位置长度为2，将参数转置为矩阵并添加到新参数列表中，并逆序扩展新置换列表
            elif len(pos) == 2:
                newargs.append((_a2m_transpose(_array2matrix(arg)), pos[0]))
                newperm.extend(reversed(pos))
            # 否则，抛出未实现的错误
            else:
                raise NotImplementedError()
        # 从新参数列表中获取矩阵参数
        newargs = [i[0] for i in newargs]
        # 对标量和新参数进行张量积，然后对新置换进行反转
        return _permute_dims(_a2m_tensor_product(*scalars, *newargs), _af_invert(newperm))
    # 如果表达式的子表达式是 ArrayContraction 类型
    elif isinstance(expr.expr, ArrayContraction):
        # 将表达式转换为矩阵乘法行
        mat_mul_lines = _array2matrix(expr.expr)
        # 如果不是 ArrayTensorProduct 类型，则根据给定的置换进行维度置换
        if not isinstance(mat_mul_lines, ArrayTensorProduct):
            return _permute_dims(mat_mul_lines, expr.permutation)
        # 计划：假设所有参数都是矩阵，但这可能不是情况：
        # 计算新的置换，将其乘以表达式的给定置换
        permutation = Permutation(2*len(mat_mul_lines.args)-1)*expr.permutation
        # 根据新的置换生成重新排序后的索引
        permuted = [permutation(i) for i in range(2*len(mat_mul_lines.args))]
        # 初始化参数数组
        args_array = [None for i in mat_mul_lines.args]
        # 遍历参数数组
        for i in range(len(mat_mul_lines.args)):
            # 获取第一个和第二个置换后的索引
            p1 = permuted[2*i]
            p2 = permuted[2*i+1]
            # 如果两个索引除以2不相等，则根据整个置换重新排列维度
            if p1 // 2 != p2 // 2:
                return _permute_dims(mat_mul_lines, permutation)
            # 如果第一个索引大于第二个索引，则将参数转置为矩阵并存储在参数数组中
            if p1 > p2:
                args_array[i] = _a2m_transpose(mat_mul_lines.args[p1 // 2])
            else:
                args_array[i] = mat_mul_lines.args[p1 // 2]
        # 对参数数组执行张量积
        return _a2m_tensor_product(*args_array)
    # 如果以上条件都不满足，则返回原始表达式
    else:
        return expr
# 注册 `_array2matrix` 的特定处理函数，用于处理 `ArrayAdd` 类型的表达式
@_array2matrix.register(ArrayAdd)
def _(expr: ArrayAdd):
    # 递归地将表达式中的每个参数转换为对应的矩阵表达式
    addends = [_array2matrix(arg) for arg in expr.args]
    # 调用 `_a2m_add` 函数对所有加数进行矩阵加法操作
    return _a2m_add(*addends)


# 注册 `_array2matrix` 的特定处理函数，用于处理 `ArrayElementwiseApplyFunc` 类型的表达式
@_array2matrix.register(ArrayElementwiseApplyFunc)
def _(expr: ArrayElementwiseApplyFunc):
    # 递归地将表达式中的子表达式转换为对应的矩阵表达式
    subexpr = _array2matrix(expr.expr)
    # 如果子表达式是矩阵表达式
    if isinstance(subexpr, MatrixExpr):
        # 如果子表达式的形状不是 (1, 1)
        if subexpr.shape != (1, 1):
            # 获取函数的符号及匹配相关表达式
            d = expr.function.bound_symbols[0]
            w = Wild("w", exclude=[d])
            p = Wild("p", exclude=[d])
            m = expr.function.expr.match(w*d**p)
            # 如果匹配成功，返回相应的 HadamardPower 表达式
            if m is not None:
                return m[w]*HadamardPower(subexpr, m[p])
        # 如果不满足条件，返回原始的 ElementwiseApplyFunction 表达式
        return ElementwiseApplyFunction(expr.function, subexpr)
    else:
        # 如果子表达式不是矩阵表达式，返回原始的 ArrayElementwiseApplyFunc 表达式
        return ArrayElementwiseApplyFunc(expr.function, subexpr)


# 注册 `_array2matrix` 的特定处理函数，用于处理 `ArrayElement` 类型的表达式
@_array2matrix.register(ArrayElement)
def _(expr: ArrayElement):
    # 尝试将表达式中的名称转换为对应的矩阵表达式
    ret = _array2matrix(expr.name)
    # 如果转换后得到的是矩阵表达式，返回对应的 MatrixElement 表达式
    if isinstance(ret, MatrixExpr):
        return MatrixElement(ret, *expr.indices)
    # 否则返回原始的 ArrayElement 表达式
    return ArrayElement(ret, expr.indices)


# 定义 `_remove_trivial_dims` 的单分派函数，用于处理移除表达式中的平凡维度
@singledispatch
def _remove_trivial_dims(expr):
    return expr, []


# 注册 `_remove_trivial_dims` 的特定处理函数，用于处理 `ArrayTensorProduct` 类型的表达式
@_remove_trivial_dims.register(ArrayTensorProduct)
def _(expr: ArrayTensorProduct):
    # 识别形状为 (k, 1, k, 1) 的表达式 `[x, y]`，将其视为 `x*y.T` 的形式
    # 矩阵表达式必须等价于矩阵的张量积，并且要删除平凡维度（即 dim=1）
    # 因此，需要对平凡维度进行收缩操作：

    # 初始化变量
    removed = []    # 记录已移除的维度
    newargs = []    # 存储新的参数列表
    cumul = list(accumulate([0] + [get_rank(arg) for arg in expr.args]))    # 累积参数的秩信息
    pending = None  # 待处理的维度
    prev_i = None   # 前一个索引
    # 遍历表达式中的每个参数及其索引
    for i, arg in enumerate(expr.args):
        # 根据累计计数确定当前参数的范围
        current_range = list(range(cumul[i], cumul[i+1]))
        # 如果参数是 OneArray 类型，则将当前范围加入移除列表并继续下一个参数
        if isinstance(arg, OneArray):
            removed.extend(current_range)
            continue
        # 如果参数不是 MatrixExpr 或 MatrixBase 类型
        if not isinstance(arg, (MatrixExpr, MatrixBase)):
            # 移除参数中的微小维度，并将移除的范围加入移除列表
            rarg, rem = _remove_trivial_dims(arg)
            removed.extend(rem)
            # 将处理后的参数加入新参数列表，并继续下一个参数处理
            newargs.append(rarg)
            continue
        # 如果参数具有 is_Identity 属性且形状为 (1, 1)
        elif getattr(arg, "is_Identity", False) and arg.shape == (1, 1):
            # 忽略形状为 (1, 1) 的单位矩阵，因为它等效于标量 1
            if arg.shape == (1, 1):
                removed.extend(current_range)
            continue
        # 如果参数的形状为 (1, 1)
        elif arg.shape == (1, 1):
            # 移除参数中微小的维度
            arg, _ = _remove_trivial_dims(arg)
            # 矩阵等效于标量：
            if len(newargs) == 0:
                newargs.append(arg)
            elif 1 in get_shape(newargs[-1]):
                # 如果最后一个参数的形状包含 1
                if newargs[-1].shape[1] == 1:
                    newargs[-1] = newargs[-1] * arg
                else:
                    newargs[-1] = arg * newargs[-1]
                removed.extend(current_range)
            else:
                newargs.append(arg)
        # 如果参数的形状包含 1
        elif 1 in arg.shape:
            k = [i for i in arg.shape if i != 1][0]
            # 如果有待处理的参数
            if pending is None:
                pending = k
                prev_i = i
                newargs.append(arg)
            elif pending == k:
                prev = newargs[-1]
                if prev.shape[0] == 1:
                    d1 = cumul[prev_i]
                    prev = _a2m_transpose(prev)
                else:
                    d1 = cumul[prev_i] + 1
                if arg.shape[1] == 1:
                    d2 = cumul[i] + 1
                    arg = _a2m_transpose(arg)
                else:
                    d2 = cumul[i]
                newargs[-1] = prev * arg
                pending = None
                removed.extend([d1, d2])
            else:
                newargs.append(arg)
                pending = k
                prev_i = i
        else:
            # 如果参数不符合以上条件，则直接加入新参数列表
            newargs.append(arg)
            pending = None
    # 对新参数进行张量积运算，得到新的表达式和更新后的移除列表
    newexpr, newremoved = _a2m_tensor_product(*newargs), sorted(removed)
    # 如果新表达式是 ArrayTensorProduct 类型
    if isinstance(newexpr, ArrayTensorProduct):
        # 查找和重写微小矩阵
        newexpr, newremoved2 = _find_trivial_matrices_rewrite(newexpr)
        newremoved = _combine_removed(-1, newremoved, newremoved2)
    # 如果新表达式是 ArrayTensorProduct 类型
    if isinstance(newexpr, ArrayTensorProduct):
        # 查找和重写微小 Kronecker 乘积的广播
        newexpr, newremoved2 = _find_trivial_kronecker_products_broadcast(newexpr)
        newremoved = _combine_removed(-1, newremoved, newremoved2)
    # 返回最终的新表达式和移除列表
    return newexpr, newremoved
# 注册函数，用于处理 ArrayAdd 类型的表达式
@_remove_trivial_dims.register(ArrayAdd)
def _(expr: ArrayAdd):
    # 递归调用 _remove_trivial_dims 处理表达式中的每个参数，返回列表 rec
    rec = [_remove_trivial_dims(arg) for arg in expr.args]
    # 将 rec 中的结果拆分成 newargs 和 removed
    newargs, removed = zip(*rec)
    # 检查 newargs 中的各参数的形状是否不同
    if len({get_shape(i) for i in newargs}) > 1:
        # 如果形状不同，返回原始表达式和空列表
        return expr, []
    # 检查是否没有移除任何元素
    if len(removed) == 0:
        # 如果没有移除元素，返回原始表达式和 removed
        return expr, removed
    # 获取第一个移除的元素
    removed1 = removed[0]
    # 调用 _a2m_add 函数，将 newargs 中的参数相加
    return _a2m_add(*newargs), removed1


# 注册函数，用于处理 PermuteDims 类型的表达式
@_remove_trivial_dims.register(PermuteDims)
def _(expr: PermuteDims):
    # 递归调用 _remove_trivial_dims 处理表达式中的子表达式和被移除元素列表
    subexpr, subremoved = _remove_trivial_dims(expr.expr)
    # 获取置换数组和其逆的引用
    p = expr.permutation.array_form
    pinv = _af_invert(expr.permutation.array_form)
    # 根据 subremoved 创建 shift 列表
    shift = list(accumulate([1 if i in subremoved else 0 for i in range(len(p))]))
    # 根据 pinv 和 subremoved 创建 premoved 列表
    premoved = [pinv[i] for i in subremoved]
    # 创建新的置换数组 p2，移除 subremoved 中的元素
    p2 = [e - shift[e] for e in p if e not in subremoved]
    # TODO: 检查是否应该对 subremoved 进行置换...
    # 使用新的子表达式和 p2 执行 _permute_dims 函数
    newexpr = _permute_dims(subexpr, p2)
    # 对 premoved 列表进行排序
    premoved = sorted(premoved)
    # 如果新表达式不等于原始表达式 expr
    if newexpr != expr:
        # 对新表达式执行 _array2matrix 函数，并合并 removed1
        newexpr, removed2 = _remove_trivial_dims(_array2matrix(newexpr))
        premoved = _combine_removed(-1, premoved, removed2)
    return newexpr, premoved


# 注册函数，用于处理 ArrayContraction 类型的表达式
@_remove_trivial_dims.register(ArrayContraction)
def _(expr: ArrayContraction):
    # 将 ArrayContraction 类型的表达式转换为对角多重单位矩阵
    new_expr, removed0 = _array_contraction_to_diagonal_multiple_identity(expr)
    # 如果转换后的表达式与原始表达式不同
    if new_expr != expr:
        # 对新表达式执行 _array2matrix 函数，并合并 removed0 和 removed1
        new_expr2, removed1 = _remove_trivial_dims(_array2matrix(new_expr))
        removed = _combine_removed(-1, removed0, removed1)
        return new_expr2, removed
    # 获取表达式的秩 rank1
    rank1 = get_rank(expr)
    # 移除表达式中的单位矩阵
    expr, removed1 = remove_identity_matrices(expr)
    # 如果表达式不是 ArrayContraction 类型
    if not isinstance(expr, ArrayContraction):
        # 递归调用 _remove_trivial_dims 处理表达式，并合并 removed1 和 removed2
        expr2, removed2 = _remove_trivial_dims(expr)
        return expr2, _combine_removed(rank1, removed1, removed2)
    # 获取表达式的子表达式和其被移除的元素列表
    newexpr, removed2 = _remove_trivial_dims(expr.expr)
    # 根据 removed2 创建 shifts 列表
    shifts = list(accumulate([1 if i in removed2 else 0 for i in range(get_rank(expr.expr))]))
    # 创建新的收缩指标列表 new_contraction_indices
    new_contraction_indices = [tuple(j for j in i if j not in removed2) for i in expr.contraction_indices]
    # 删除可能的空元组 "()"：
    new_contraction_indices = [i for i in new_contraction_indices if len(i) > 0]
    # 将收缩指标列表扁平化为 contraction_indices_flat
    contraction_indices_flat = [j for i in expr.contraction_indices for j in i]
    # 从 removed2 中移除 contraction_indices_flat 中的元素
    removed2 = [i for i in removed2 if i not in contraction_indices_flat]
    # 根据 shifts 调整 new_contraction_indices 中的元素
    new_contraction_indices = [tuple(j - shifts[j] for j in i) for i in new_contraction_indices]
    # 移除 removed2 中的元素
    removed2 = ArrayContraction._push_indices_up(expr.contraction_indices, removed2)
    # 合并 removed0、removed1 和 removed2
    removed = _combine_removed(rank1, removed1, removed2)
    # 使用新的收缩指标列表创建 _array_contraction 表达式，并返回结果和 removed 列表
    return _array_contraction(newexpr, *new_contraction_indices), list(removed)


# 私有函数，用于移除 ArrayDiagonal 类型表达式中的对角化单位矩阵
def _remove_diagonalized_identity_matrices(expr: ArrayDiagonal):
    # 断言 expr 是 ArrayDiagonal 类型
    assert isinstance(expr, ArrayDiagonal)
    # 使用 _EditArrayContraction 编辑器处理表达式
    editor = _EditArrayContraction(expr)
    # 创建映射 mapping，将每个索引映射到包含该索引的参数列表
    mapping = {i: {j for j in editor.args_with_ind if i in j.indices} for i in range(-1, -1-editor.number_of_diagonal_indices, -1)}
    # 初始化移除列表 removed
    removed = []
    # 计数器初始化为 0
    counter: int = 0
    # 遍历 editor.args_with_ind 列表，同时获取索引 i 和元素 arg_with_ind
    for i, arg_with_ind in enumerate(editor.args_with_ind):
        # 将 counter 增加 arg_with_ind.indices 的长度
        counter += len(arg_with_ind.indices)
        
        # 检查 arg_with_ind.element 是否是 Identity 类的实例
        if isinstance(arg_with_ind.element, Identity):
            # 检查 arg_with_ind.indices 中是否包含 None，并且是否有负数索引
            if None in arg_with_ind.indices and any(i is not None and (i < 0) == True for i in arg_with_ind.indices):
                # 找到第一个非 None 的索引 diag_ind
                diag_ind = [j for j in arg_with_ind.indices if j is not None][0]
                # 在 mapping[diag_ind] 中找到不等于 arg_with_ind 的元素 other
                other = [j for j in mapping[diag_ind] if j != arg_with_ind][0]
                
                # 如果 other.element 不是 MatrixExpr 类型，则跳过当前循环
                if not isinstance(other.element, MatrixExpr):
                    continue
                # 如果 other.element 的形状中不包含 1，则跳过当前循环
                if 1 not in other.element.shape:
                    continue
                # 如果 other.indices 中没有 None，则跳过当前循环
                if None not in other.indices:
                    continue
                
                # 将 editor.args_with_ind[i].element 设为 None
                editor.args_with_ind[i].element = None
                # 找到 other.indices 中的第一个 None 的索引 none_index
                none_index = other.indices.index(None)
                # 将 other.element 替换为 DiagMatrix 类型的对象
                other.element = DiagMatrix(other.element)
                # 获取 other 的绝对范围 other_range
                other_range = editor.get_absolute_range(other)
                # 将 removed 扩展，加入 other_range[0] + none_index
                removed.extend([other_range[0] + none_index])
    
    # 重新赋值 editor.args_with_ind，仅保留 element 不为 None 的元素
    editor.args_with_ind = [i for i in editor.args_with_ind if i.element is not None]
    
    # 调用 ArrayDiagonal._push_indices_up 方法，将 expr.diagonal_indices 中的 removed 元素移除，并根据 expr.expr 的秩进行操作
    removed = ArrayDiagonal._push_indices_up(expr.diagonal_indices, removed, get_rank(expr.expr))
    
    # 返回 editor 转换为数组缩并的结果以及处理后的 removed 列表
    return editor.to_array_contraction(), removed
# 注册一个函数 `_remove_trivial_dims`，用于处理 `ArrayDiagonal` 类型的表达式
@_remove_trivial_dims.register(ArrayDiagonal)
def _(expr: ArrayDiagonal):
    # 递归调用 `_remove_trivial_dims` 函数，处理 `expr` 中的表达式和移除的维度
    newexpr, removed = _remove_trivial_dims(expr.expr)
    # 计算移位量，用于调整维度索引
    shifts = list(accumulate([0] + [1 if i in removed else 0 for i in range(get_rank(expr.expr))]))
    # 创建新的对角线索引映射表，移除已经对角化的单一轴
    new_diag_indices_map = {i: tuple(j for j in i if j not in removed) for i in expr.diagonal_indices}
    for old_diag_tuple, new_diag_tuple in new_diag_indices_map.items():
        if len(new_diag_tuple) == 1:
            removed = [i for i in removed if i not in old_diag_tuple]
    # 根据新的对角线索引映射表和移位量，生成调整后的对角线索引
    new_diag_indices = [tuple(j - shifts[j] for j in i) for i in new_diag_indices_map.values()]
    # 获取表达式的秩
    rank = get_rank(expr.expr)
    # 将对角线索引中的移除维度向上推，调整排序
    removed = ArrayDiagonal._push_indices_up(expr.diagonal_indices, removed, rank)
    removed = sorted(set(removed))
    # 如果仍有需要对角化的单一轴，说明其对应的维度已被移除，无需再进行对角化
    new_diag_indices = [i for i in new_diag_indices if len(i) > 0]
    # 如果仍有对角线索引需要处理，则调用 `_array_diagonal` 函数对新的表达式进行对角化
    if len(new_diag_indices) > 0:
        newexpr2 = _array_diagonal(newexpr, *new_diag_indices, allow_trivial_diags=True)
    else:
        newexpr2 = newexpr
    # 如果新的表达式仍然是 `ArrayDiagonal` 类型，则继续处理移除对角化的恒等矩阵
    if isinstance(newexpr2, ArrayDiagonal):
        newexpr3, removed2 = _remove_diagonalized_identity_matrices(newexpr2)
        # 合并移除的维度信息
        removed = _combine_removed(-1, removed, removed2)
        return newexpr3, removed
    else:
        return newexpr2, removed


# 注册一个函数 `_remove_trivial_dims`，用于处理 `ElementwiseApplyFunction` 类型的表达式
@_remove_trivial_dims.register(ElementwiseApplyFunction)
def _(expr: ElementwiseApplyFunction):
    # 递归调用 `_remove_trivial_dims` 函数，处理 `expr` 中的表达式和移除的维度
    subexpr, removed = _remove_trivial_dims(expr.expr)
    # 如果子表达式的形状为 (1, 1)，则直接应用函数并更新移除的维度信息
    if subexpr.shape == (1, 1):
        # TODO: move this to ElementwiseApplyFunction
        return expr.function(subexpr), removed + [0, 1]
    # 否则，返回新的 `ElementwiseApplyFunction` 表达式和空的移除维度信息
    return ElementwiseApplyFunction(expr.function, subexpr), []


# 注册一个函数 `_remove_trivial_dims`，用于处理 `ArrayElementwiseApplyFunc` 类型的表达式
@_remove_trivial_dims.register(ArrayElementwiseApplyFunc)
def _(expr: ArrayElementwiseApplyFunc):
    # 递归调用 `_remove_trivial_dims` 函数，处理 `expr` 中的表达式和移除的维度
    subexpr, removed = _remove_trivial_dims(expr.expr)
    # 返回新的 `ArrayElementwiseApplyFunc` 表达式和移除的维度信息
    return ArrayElementwiseApplyFunc(expr.function, subexpr), []


# 定义函数 `convert_array_to_matrix`，用于将数组表达式转换为矩阵表达式
def convert_array_to_matrix(expr):
    r"""
    Recognize matrix expressions in codegen objects.

    If more than one matrix multiplication line have been detected, return a
    list with the matrix expressions.

    Examples
    ========

    >>> from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
    >>> from sympy.tensor.array import tensorcontraction, tensorproduct
    >>> from sympy import MatrixSymbol, Sum
    >>> from sympy.abc import i, j, k, l, N
    >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
    >>> from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    >>> A = MatrixSymbol("A", N, N)
    >>> B = MatrixSymbol("B", N, N)
    >>> C = MatrixSymbol("C", N, N)
    >>> D = MatrixSymbol("D", N, N)

    >>> expr = Sum(A[i, j]*B[j, k], (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B
    """
    # Convert indexed expression `expr` into an array representation and then into a matrix representation.
    def convert_indexed_to_array(expr, first_indices=None):
        # If `first_indices` is provided, convert `expr` to an array using those indices.
        cg = _array2matrix(expr, first_indices)
        # Remove trivial dimensions from the converted array representation.
        rec, removed = _remove_trivial_dims(cg)
        # Return the processed matrix representation after conversion.
        return rec
# 将 ArrayDiagonal 类型的表达式转换为对角线对应的对角矩阵
def _array_diag2contr_diagmatrix(expr: ArrayDiagonal):
    # 检查表达式中的子表达式是否为 ArrayTensorProduct 类型
    if isinstance(expr.expr, ArrayTensorProduct):
        # 将表达式中的参数和对角线索引分别转换为列表
        args = list(expr.expr.args)
        diag_indices = list(expr.diagonal_indices)
        # 根据子表达式的秩生成映射关系
        mapping = _get_mapping_from_subranks([_get_subrank(arg) for arg in args])
        # 根据映射关系更新对角线索引中的绝对位置
        tuple_links = [[mapping[j] for j in i] for i in diag_indices]
        contr_indices = []
        # 获取表达式的总秩
        total_rank = get_rank(expr)
        replaced = [False for arg in args]
        # 遍历每一个对角线索引及其映射
        for i, (abs_pos, rel_pos) in enumerate(zip(diag_indices, tuple_links)):
            # 只处理长度为 2 的对角线索引
            if len(abs_pos) != 2:
                continue
            # 获取映射后的位置信息
            (pos1_outer, pos1_inner), (pos2_outer, pos2_inner) = rel_pos
            arg1 = args[pos1_outer]
            arg2 = args[pos2_outer]
            # 如果 arg1 或 arg2 的秩不为 2，则跳过
            if get_rank(arg1) != 2 or get_rank(arg2) != 2:
                # 如果已经替换过其中一个参数，则将对角线索引置为 None
                if replaced[pos1_outer]:
                    diag_indices[i] = None
                if replaced[pos2_outer]:
                    diag_indices[i] = None
                continue
            # 计算在 arg1 或 arg2 中的另一维度
            pos1_in2 = 1 - pos1_inner
            pos2_in2 = 1 - pos2_inner
            # 如果 arg1 的其中一维度为 1，则生成对应的对角矩阵 darg1
            if arg1.shape[pos1_in2] == 1:
                if arg1.shape[pos1_inner] != 1:
                    darg1 = DiagMatrix(arg1)
                else:
                    darg1 = arg1
                # 将 darg1 添加到参数列表中，并更新收缩索引
                args.append(darg1)
                contr_indices.append(((pos2_outer, pos2_inner), (len(args)-1, pos1_inner)))
                total_rank += 1
                # 将对角线索引置为 None，表示已经处理过
                diag_indices[i] = None
                # 将 arg1 的另一维度设置为 OneArray 形式
                args[pos1_outer] = OneArray(arg1.shape[pos1_in2])
                replaced[pos1_outer] = True
            # 类似地，如果 arg2 的其中一维度为 1，则生成对应的对角矩阵 darg2
            elif arg2.shape[pos2_in2] == 1:
                if arg2.shape[pos2_inner] != 1:
                    darg2 = DiagMatrix(arg2)
                else:
                    darg2 = arg2
                # 将 darg2 添加到参数列表中，并更新收缩索引
                args.append(darg2)
                contr_indices.append(((pos1_outer, pos1_inner), (len(args)-1, pos2_inner)))
                total_rank += 1
                # 将对角线索引置为 None，表示已经处理过
                diag_indices[i] = None
                # 将 arg2 的另一维度设置为 OneArray 形式
                args[pos2_outer] = OneArray(arg2.shape[pos2_in2])
                replaced[pos2_outer] = True
        # 过滤掉已处理的对角线索引中的 None 值
        diag_indices_new = [i for i in diag_indices if i is not None]
        # 计算参数列表中每个参数的累积秩
        cumul = list(accumulate([0] + [get_rank(arg) for arg in args]))
        # 根据累积秩和收缩索引，进行数组收缩操作
        contr_indices2 = [tuple(cumul[a] + b for a, b in i) for i in contr_indices]
        # 对数组进行收缩操作，并返回对角线的结果
        tc = _array_contraction(
            _array_tensor_product(*args), *contr_indices2
        )
        td = _array_diagonal(tc, *diag_indices_new)
        return td
    # 如果表达式不是 ArrayTensorProduct 类型，则直接返回原表达式
    return expr


# 将多个数组张量的乘积转换为矩阵乘法或数组收缩操作
def _a2m_mul(*args):
    # 如果所有参数均不是 _CodegenArrayAbstract 类型的实例
    if not any(isinstance(i, _CodegenArrayAbstract) for i in args):
        # 导入矩阵乘法模块，并对参数进行矩阵乘法操作
        from sympy.matrices.expressions.matmul import MatMul
        return MatMul(*args).doit()
    else:
        # 否则，对参数进行数组张量积和数组收缩操作
        return _array_contraction(
            _array_tensor_product(*args),
            *[(2*i-1, 2*i) for i in range(1, len(args))]
        )


# 将多个数组张量的张量积转换为数组收缩操作
def _a2m_tensor_product(*args):
    scalars = []
    arrays = []
    # 遍历参数列表中的每个参数
    for arg in args:
        # 检查参数是否是 MatrixExpr、_ArrayExpr 或 _CodegenArrayAbstract 的实例
        if isinstance(arg, (MatrixExpr, _ArrayExpr, _CodegenArrayAbstract)):
            # 如果是数组表达式或者代码生成数组抽象对象，则将其添加到数组列表中
            arrays.append(arg)
        else:
            # 如果不是上述类型的实例，则将其视为标量，添加到标量列表中
            scalars.append(arg)
    
    # 将所有标量元素累乘成一个 Mul 对象
    scalar = Mul.fromiter(scalars)
    
    # 如果数组列表为空，则直接返回累乘后的标量
    if len(arrays) == 0:
        return scalar
    
    # 如果累乘后的标量不等于 1
    if scalar != 1:
        # 如果第一个数组对象是 _CodegenArrayAbstract 类型的实例
        if isinstance(arrays[0], _CodegenArrayAbstract):
            # 将累乘后的标量作为第一个元素插入数组列表中
            arrays = [scalar] + arrays
        else:
            # 否则，将累乘后的标量乘以第一个数组对象
            arrays[0] *= scalar
    
    # 返回数组列表的张量积结果
    return _array_tensor_product(*arrays)
# 定义一个函数 _a2m_add，接受任意数量参数
def _a2m_add(*args):
    # 如果参数中没有任何一个是 _CodegenArrayAbstract 类型的实例
    if not any(isinstance(i, _CodegenArrayAbstract) for i in args):
        # 导入 MatAdd 类，并将参数传递给 MatAdd 对象的构造函数，然后执行 doit() 方法
        from sympy.matrices.expressions.matadd import MatAdd
        return MatAdd(*args).doit()
    else:
        # 如果参数中有 _CodegenArrayAbstract 类型的实例，则调用 _array_add 函数处理参数
        return _array_add(*args)


# 定义一个函数 _a2m_trace，接受一个参数 arg
def _a2m_trace(arg):
    # 如果参数 arg 是 _CodegenArrayAbstract 类型的实例
    if isinstance(arg, _CodegenArrayAbstract):
        # 调用 _array_contraction 函数，对 arg 执行 (0, 1) 的缩并操作
        return _array_contraction(arg, (0, 1))
    else:
        # 如果参数 arg 不是 _CodegenArrayAbstract 类型的实例，导入 Trace 类，并返回 Trace(arg) 对象
        from sympy.matrices.expressions.trace import Trace
        return Trace(arg)


# 定义一个函数 _a2m_transpose，接受一个参数 arg
def _a2m_transpose(arg):
    # 如果参数 arg 是 _CodegenArrayAbstract 类型的实例
    if isinstance(arg, _CodegenArrayAbstract):
        # 调用 _permute_dims 函数，对 arg 进行维度置换，置换顺序为 [1, 0]
        return _permute_dims(arg, [1, 0])
    else:
        # 如果参数 arg 不是 _CodegenArrayAbstract 类型的实例，导入 Transpose 类，并返回 Transpose(arg).doit() 的结果
        from sympy.matrices.expressions.transpose import Transpose
        return Transpose(arg).doit()


# 定义一个函数 identify_hadamard_products，接受一个参数 expr，类型为 tUnion[ArrayContraction, ArrayDiagonal]
def identify_hadamard_products(expr: tUnion[ArrayContraction, ArrayDiagonal]):

    # 创建一个 _EditArrayContraction 对象 editor，用于处理参数 expr
    editor: _EditArrayContraction = _EditArrayContraction(expr)

    # 创建一个默认字典 map_contr_to_args，键为 frozenset(arg_with_ind.indices)，值为对应的 _ArgE 对象列表
    map_contr_to_args: tDict[FrozenSet, List[_ArgE]] = defaultdict(list)

    # 创建一个默认字典 map_ind_to_inds，键为可选的 int 值，值为 int 类型的计数器，默认值为 0
    map_ind_to_inds: tDict[Optional[int], int] = defaultdict(int)

    # 遍历 editor 的 args_with_ind 属性中的每个 arg_with_ind 对象
    for arg_with_ind in editor.args_with_ind:
        # 遍历 arg_with_ind.indices 中的每个索引 ind
        for ind in arg_with_ind.indices:
            # 增加 map_ind_to_inds 中 ind 对应的计数
            map_ind_to_inds[ind] += 1
        # 如果 arg_with_ind.indices 中包含 None，则继续下一次循环
        if None in arg_with_ind.indices:
            continue
        # 将 arg_with_ind.indices 转换为 frozenset，并将 arg_with_ind 添加到 map_contr_to_args 对应的列表中
        map_contr_to_args[frozenset(arg_with_ind.indices)].append(arg_with_ind)

    # 声明 k 为 FrozenSet[int] 类型，v 为 List[_ArgE] 类型
    k: FrozenSet[int]
    v: List[_ArgE]
    # 遍历 map_contr_to_args 字典中的每对键值对 (k, v)
    for k, v in map_contr_to_args.items():
        # 默认不生成迹(trace)
        make_trace: bool = False
        
        # 如果 k 的长度为 1，且 k 的唯一元素大于等于 0，并且只有一个索引在 map_contr_to_args 中出现过：
        if len(k) == 1 and next(iter(k)) >= 0 and sum(next(iter(k)) in i for i in map_contr_to_args) == 1:
            # 这是一个迹(trace)：参数完全收缩，只有一个索引，并且这个索引没有在其他地方使用：
            make_trace = True
            first_element = S.One
        
        # 如果 k 的长度不等于 2，跳过此次循环：
        elif len(k) != 2:
            # 只有矩阵才能定义哈达玛积：
            continue
        
        # 如果 v 的长度为 1，跳过此次循环：
        if len(v) == 1:
            # 只有一个参数的哈达玛积没有意义：
            continue
        
        # 遍历 k 中的索引：
        for ind in k:
            # 如果 map_ind_to_inds 中索引对应的值小于等于 2，跳过此次循环：
            if map_ind_to_inds[ind] <= 2:
                # 没有其他的收缩，跳过：
                continue
        
        # 定义检查转置函数：
        def check_transpose(x):
            # 将 x 中小于 0 的元素替换为其相反数
            x = [i if i >= 0 else -1-i for i in x]
            # 判断 x 是否是排序后的相同元素
            return x == sorted(x)
        
        # 检查表达式是否是迹(trace)：
        if all(map_ind_to_inds[j] == len(v) and j >= 0 for j in k) and all(j >= 0 for j in k):
            # 这是一个迹(trace)
            make_trace = True
            first_element = v[0].element
            # 如果第一个元素的索引不是按顺序排列的，则取其转置
            if not check_transpose(v[0].indices):
                first_element = first_element.T
            # 哈达玛积因子为除第一个元素外的其余元素
            hadamard_factors = v[1:]
        else:
            # 否则哈达玛积因子为 v
            hadamard_factors = v
        
        # 这是一个哈达玛积：
        hp = hadamard_product(*[i.element if check_transpose(i.indices) else Transpose(i.element) for i in hadamard_factors])
        hp_indices = v[0].indices
        # 如果第一个元素的索引不是按顺序排列的，则将 hp_indices 反转
        if not check_transpose(hadamard_factors[0].indices):
            hp_indices = list(reversed(hp_indices))
        
        # 如果是迹(trace)，则对哈达玛积进行处理：
        if make_trace:
            hp = Trace(first_element*hp.T)._normalize()
            hp_indices = []
        
        # 在编辑器中插入 _ArgE(hp, hp_indices)，并从 editor.args_with_ind 中移除 v 中的每个元素
        editor.insert_after(v[0], _ArgE(hp, hp_indices))
        for i in v:
            editor.args_with_ind.remove(i)
    
    # 返回编辑器的数组收缩表示
    return editor.to_array_contraction()
# 根据给定表达式识别可移除的单位矩阵
def identify_removable_identity_matrices(expr):
    # 创建 _EditArrayContraction 对象以编辑表达式
    editor = _EditArrayContraction(expr)

    # 设定循环标志位
    flag = True
    while flag:
        # 初始化标志位为 False
        flag = False
        
        # 遍历所有具有索引的参数
        for arg_with_ind in editor.args_with_ind:
            # 检查元素是否为单位矩阵
            if isinstance(arg_with_ind.element, Identity):
                # 获取单位矩阵的维度
                k = arg_with_ind.element.shape[0]
                
                # 判断是否是可以移除的候选单位矩阵
                if arg_with_ind.indices == [None, None]:
                    # 自由单位矩阵，将会被 _remove_trivial_dims 清除
                    continue
                elif None in arg_with_ind.indices:
                    # 若其中一个索引为 None
                    ind = [j for j in arg_with_ind.indices if j is not None][0]
                    # 计算具有该索引的参数数量
                    counted = editor.count_args_with_index(ind)
                    
                    # 判断是否只有一个单位矩阵与自身的一个索引收缩
                    if counted == 1:
                        # 转换为 OneArray(k) 元素
                        editor.insert_after(arg_with_ind, OneArray(k))
                        editor.args_with_ind.remove(arg_with_ind)
                        flag = True
                        break
                    elif counted > 2:
                        # 当 counted > 2 时，是多次收缩的情况，跳过处理
                        continue
                elif arg_with_ind.indices[0] == arg_with_ind.indices[1]:
                    # 若两个索引相同，表示为迹运算
                    ind = arg_with_ind.indices[0]
                    # 计算具有该索引的参数数量
                    counted = editor.count_args_with_index(ind)
                    
                    # 若该索引出现次数大于 1
                    if counted > 1:
                        editor.args_with_ind.remove(arg_with_ind)
                        flag = True
                        break
                    else:
                        # 是迹运算，跳过，因为在其他地方会被识别
                        pass
            # 若元素被认为是对角元素
            elif ask(Q.diagonal(arg_with_ind.element)):
                # 判断索引情况
                if arg_with_ind.indices == [None, None]:
                    continue
                elif None in arg_with_ind.indices:
                    pass
                elif arg_with_ind.indices[0] == arg_with_ind.indices[1]:
                    ind = arg_with_ind.indices[0]
                    counted = editor.count_args_with_index(ind)
                    
                    # 当 counted == 3 时，执行下列操作
                    if counted == 3:
                        # 转换为 A_ai D_ij B_bj 形式
                        ind_new = editor.get_new_contraction_index()
                        other_args = [j for j in editor.args_with_ind if j != arg_with_ind]
                        other_args[1].indices = [ind_new if j == ind else j for j in other_args[1].indices]
                        arg_with_ind.indices = [ind, ind_new]
                        flag = True
                        break

    # 返回编辑后的数组收缩结果
    return editor.to_array_contraction()
    # 创建一个 _EditArrayContraction 类的实例，用于编辑表达式
    editor = _EditArrayContraction(expr)
    
    # 存储被移除的索引的列表
    removed: List[int] = []
    
    # 存储排列映射的空字典
    permutation_map = {}
    
    # 计算每个参数的自由索引数量并构建自由索引映射
    free_indices = list(accumulate([0] + [sum(i is None for i in arg.indices) for arg in editor.args_with_ind]))
    free_map = dict(zip(editor.args_with_ind, free_indices[:-1]))
    
    # 存储更新对的空字典
    update_pairs = {}
    
    # 遍历每一个收缩索引
    for ind in range(editor.number_of_contraction_indices):
        # 获取与当前收缩索引相关的参数
        args = editor.get_args_with_index(ind)
        # 找到参数中的单位矩阵
        identity_matrices = [i for i in args if isinstance(i.element, Identity)]
        number_identity_matrices = len(identity_matrices)
        
        # 如果收缩涉及非单位矩阵和多个单位矩阵：
        if number_identity_matrices != len(args) - 1 or number_identity_matrices == 0:
            continue
        
        # 获取非单位元素
        non_identity = [i for i in args if not isinstance(i.element, Identity)][0]
        
        # 检查所有单位矩阵是否至少有一个自由索引
        if any(None not in i.indices for i in identity_matrices):
            continue
        
        # 标记要移除的单位矩阵
        for i in identity_matrices:
            i.element = None
            removed.extend(range(free_map[i], free_map[i] + len([j for j in i.indices if j is None])))
        
        # 弹出最后一个被移除的索引
        last_removed = removed.pop(-1)
        update_pairs[last_removed, ind] = non_identity.indices[:]
        
        # 由于收缩不再存在，从非单位矩阵中移除相应的索引
        non_identity.indices = [None if i == ind else i for i in non_identity.indices]

    # 对移除的索引进行排序
    removed.sort()
    
    # 计算移位量，用于更新排列映射
    shifts = list(accumulate([1 if i in removed else 0 for i in range(get_rank(expr))]))
    
    # 更新排列映射
    for (last_removed, ind), non_identity_indices in update_pairs.items():
        pos = [free_map[non_identity] + i for i, e in enumerate(non_identity_indices) if e == ind]
        assert len(pos) == 1
        for j in pos:
            permutation_map[j] = last_removed
    
    # 过滤掉已被标记为 None 的参数
    editor.args_with_ind = [i for i in editor.args_with_ind if i.element is not None]
    
    # 获取数组收缩后的表达式
    ret_expr = editor.to_array_contraction()
    
    # 初始化排列和计数器
    permutation = []
    counter = 0
    counter2 = 0
    
    # 根据移除的索引和排列映射生成最终的排列
    for j in range(get_rank(expr)):
        if j in removed:
            continue
        if counter2 in permutation_map:
            target = permutation_map[counter2]
            permutation.append(target - shifts[target])
            counter2 += 1
        else:
            while counter in permutation_map.values():
                counter += 1
            permutation.append(counter)
            counter += 1
            counter2 += 1
    
    # 使用排列对表达式进行维度重排
    ret_expr2 = _permute_dims(ret_expr, _af_invert(permutation))
    
    # 返回重排后的表达式和被移除的索引列表
    return ret_expr2, removed
# 合并移除操作，以便组合为一个列表，这些操作由 _remove_trivial_dims 函数执行
def _combine_removed(dim: int, removed1: List[int], removed2: List[int]) -> List[int]:
    # 对 removed1 和 removed2 列表进行排序
    removed1 = sorted(removed1)
    removed2 = sorted(removed2)
    i = 0
    j = 0
    removed = []
    # 合并两个排序后的列表，确保结果的顺序性
    while True:
        # 如果 removed2 列表已经处理完毕，则将 removed1 剩余部分加入结果列表
        if j >= len(removed2):
            while i < len(removed1):
                removed.append(removed1[i])
                i += 1
            break
        # 如果 removed1 还有剩余元素，并且该元素小于等于 i + removed2[j]
        elif i < len(removed1) and removed1[i] <= i + removed2[j]:
            removed.append(removed1[i])
            i += 1
        # 否则，将 i + removed2[j] 加入结果列表
        else:
            removed.append(i + removed2[j])
            j += 1
    # 返回合并后的移除操作列表
    return removed


def _array_contraction_to_diagonal_multiple_identity(expr: ArrayContraction):
    # 创建 _EditArrayContraction 实例进行编辑
    editor = _EditArrayContraction(expr)
    # 开始跟踪排列的起始点
    editor.track_permutation_start()
    # 存储被移除的索引列表
    removed: List[int] = []
    # 对角线索引计数器
    diag_index_counter: int = 0
    # 遍历收缩表达式中的每一个收缩索引
    for i in range(editor.number_of_contraction_indices):
        identities = []
        args = []
        # 遍历每一个带有索引的参数
        for j, arg in enumerate(editor.args_with_ind):
            # 如果当前索引不在参数的索引列表中，则继续下一个参数
            if i not in arg.indices:
                continue
            # 如果参数是 Identity 类型，则加入 identities 列表
            if isinstance(arg.element, Identity):
                identities.append(arg)
            else:
                args.append(arg)
        # 如果没有找到 Identity 参数，则继续下一个收缩索引
        if len(identities) == 0:
            continue
        # 如果 identities 和 args 的总数小于 3，则继续下一个收缩索引
        if len(args) + len(identities) < 3:
            continue
        # 创建新的对角线索引
        new_diag_ind = -1 - diag_index_counter
        diag_index_counter += 1
        # 控制是否跳过这个收缩集合的标志变量 "flag"
        flag: bool = True
        for i1, id1 in enumerate(identities):
            # 如果 Identity 的索引列表中没有 None，则设置 flag 为 True 并跳出循环
            if None not in id1.indices:
                flag = True
                break
            # 获取自由位置的第一个索引并跟踪排列
            free_pos = list(range(*editor.get_absolute_free_range(id1)))[0]
            editor._track_permutation[-1].append(free_pos)  # type: ignore
            id1.element = None
            flag = False
            break
        # 如果 flag 为 True，则跳过当前收缩集合
        if flag:
            continue
        # 将 identities 列表中的其余 Identity 参数设置为 None，并将其对应的索引加入 removed 列表
        for arg in identities[:i1] + identities[i1+1:]:
            arg.element = None
            removed.extend(range(*editor.get_absolute_free_range(arg)))
        # 将 args 中的参数索引更新为新的对角线索引
        for arg in args:
            arg.indices = [new_diag_ind if j == i else j for j in arg.indices]
    # 将编辑中 element 为 None 的参数对应的排列设置为 None
    for j, e in enumerate(editor.args_with_ind):
        if e.element is None:
            editor._track_permutation[j] = None  # type: ignore
    # 清理掉所有值为 None 的排列
    editor._track_permutation = [i for i in editor._track_permutation if i is not None]  # type: ignore
    # 对删除位置重新编号排列数组形式
    remap = {e: i for i, e in enumerate(sorted({k for j in editor._track_permutation for k in j}))}
    editor._track_permutation = [[remap[j] for j in i] for i in editor._track_permutation]
    # 清理掉 element 为 None 的参数
    editor.args_with_ind = [i for i in editor.args_with_ind if i.element is not None]
    # 返回新的收缩表达式和移除操作列表
    new_expr = editor.to_array_contraction()
    return new_expr, removed
```