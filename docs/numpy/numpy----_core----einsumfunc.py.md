# `.\numpy\numpy\_core\einsumfunc.py`

```
"""
Implementation of optimized einsum.

"""
import itertools  # 导入 itertools 模块，用于生成迭代器和循环的工具
import operator   # 导入 operator 模块，提供了对内置操作符的函数接口

from numpy._core.multiarray import c_einsum   # 导入 numpy 中的 c_einsum 函数
from numpy._core.numeric import asanyarray, tensordot   # 导入 numpy 中的数组处理函数
from numpy._core.overrides import array_function_dispatch   # 导入 numpy 中的函数装饰器

__all__ = ['einsum', 'einsum_path']   # 设置模块的公开接口

# importing string for string.ascii_letters would be too slow
# the first import before caching has been measured to take 800 µs (#23777)
einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'   # 定义字符串，表示可能出现的全部字母作为 einsum 的符号
einsum_symbols_set = set(einsum_symbols)   # 将所有可能的符号转化为集合，以供快速查找


def _flop_count(idx_contraction, inner, num_terms, size_dictionary):
    """
    Computes the number of FLOPS in the contraction.

    Parameters
    ----------
    idx_contraction : iterable
        The indices involved in the contraction
    inner : bool
        Does this contraction require an inner product?
    num_terms : int
        The number of terms in a contraction
    size_dictionary : dict
        The size of each of the indices in idx_contraction

    Returns
    -------
    flop_count : int
        The total number of FLOPS required for the contraction.

    Examples
    --------

    >>> _flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})
    30

    >>> _flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})
    60

    """
    overall_size = _compute_size_by_dict(idx_contraction, size_dictionary)   # 计算给定指标集合的总大小
    op_factor = max(1, num_terms - 1)   # 计算操作因子，考虑到合并操作的次数
    if inner:
        op_factor += 1   # 如果需要内积，增加操作因子

    return overall_size * op_factor   # 返回总的 FLOPS 数量


def _compute_size_by_dict(indices, idx_dict):
    """
    Computes the product of the elements in indices based on the dictionary
    idx_dict.

    Parameters
    ----------
    indices : iterable
        Indices to base the product on.
    idx_dict : dictionary
        Dictionary of index sizes

    Returns
    -------
    ret : int
        The resulting product.

    Examples
    --------
    >>> _compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})
    90

    """
    ret = 1   # 初始化结果为1
    for i in indices:
        ret *= idx_dict[i]   # 根据索引字典计算给定索引集合的乘积
    return ret   # 返回计算结果


def _find_contraction(positions, input_sets, output_set):
    """
    Finds the contraction for a given set of input and output sets.

    Parameters
    ----------
    positions : iterable
        Integer positions of terms used in the contraction.
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript

    Returns
    -------
    new_result : set
        The indices of the resulting contraction
    remaining : list
        List of sets that have not been contracted, the new set is appended to
        the end of this list
    idx_removed : set
        Indices removed from the entire contraction
    idx_contraction : set
        The indices used in the current contraction

    Examples
    --------

    # A simple dot product test case
    >>> pos = (0, 1)
    >>> isets = [set('ab'), set('bc')]
    >>> oset = set('ac')

    """
    # 初始化空集合，用于记录收缩的索引集合
    idx_contract = set()
    # 复制输出集合，以确保不改变原始数据
    idx_remain = output_set.copy()
    # 初始化空列表，用于存储剩余的集合
    remaining = []
    
    # 遍历输入集合的索引和值
    for ind, value in enumerate(input_sets):
        # 如果当前索引在收缩位置集合中
        if ind in positions:
            # 将当前集合并入收缩的索引集合中
            idx_contract |= value
        else:
            # 否则将当前集合加入剩余集合列表中
            remaining.append(value)
            # 并将当前集合并入剩余的索引集合中
            idx_remain |= value
    
    # 计算新的结果集合，为剩余索引集合和收缩索引集合的交集
    new_result = idx_remain & idx_contract
    # 计算从收缩索引集合中移除的元素
    idx_removed = (idx_contract - new_result)
    # 将新结果集合添加到剩余集合列表中
    remaining.append(new_result)
    
    # 返回四个结果：新的结果集合、剩余集合列表、移除的索引集合、收缩的索引集合
    return (new_result, remaining, idx_removed, idx_contract)
def _optimal_path(input_sets, output_set, idx_dict, memory_limit):
    """
    Computes all possible pair contractions, sieves the results based
    on ``memory_limit`` and returns the lowest cost path. This algorithm
    scales factorial with respect to the elements in the list ``input_sets``.

    Parameters
    ----------
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    idx_dict : dictionary
        Dictionary of index sizes
    memory_limit : int
        The maximum number of elements in a temporary array

    Returns
    -------
    path : list
        The optimal contraction order within the memory limit constraint.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set()
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> _optimal_path(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """

    # Initialize with the full set of input sets
    full_results = [(0, [], input_sets)]

    # Iterate through all possible contractions
    for iteration in range(len(input_sets) - 1):
        iter_results = []

        # Compute all unique pairs
        for curr in full_results:
            cost, positions, remaining = curr
            for con in itertools.combinations(
                range(len(input_sets) - iteration), 2
            ):

                # Find the contraction and associated details
                cont = _find_contraction(con, remaining, output_set)
                new_result, new_input_sets, idx_removed, idx_contract = cont

                # Calculate size of new result and check against memory limit
                new_size = _compute_size_by_dict(new_result, idx_dict)
                if new_size > memory_limit:
                    continue

                # Calculate total cost of this contraction path
                total_cost = cost + _flop_count(
                    idx_contract, idx_removed, len(con), idx_dict
                )
                new_pos = positions + [con]
                iter_results.append((total_cost, new_pos, new_input_sets))

        # Update full results with new iteration results
        if iter_results:
            full_results = iter_results
        else:
            # If no valid contractions found, return best path plus remaining contractions
            path = min(full_results, key=lambda x: x[0])[1]
            path += [tuple(range(len(input_sets) - iteration))]
            return path

    # If no valid contractions found throughout all iterations, return single einsum contraction
    if len(full_results) == 0:
        return [tuple(range(len(input_sets)))]

    # Return the optimal path found
    path = min(full_results, key=lambda x: x[0])[1]
    return path


def _parse_possible_contraction(
        positions, input_sets, output_set, idx_dict, 
        memory_limit, path_cost, naive_cost
    ):
    """Compute the cost (removed size + flops) and resultant indices for
    performing the contraction specified by ``positions``.

    Parameters
    ----------
    positions : list
        List of pairs specifying contraction positions
    input_sets : list
        List of sets representing lhs side of the einsum subscript
    output_set : set
        Set representing rhs side of the einsum subscript
    idx_dict : dict
        Dictionary of index sizes
    memory_limit : int
        Maximum number of elements in a temporary array
    path_cost : int
        Current cost of the contraction path
    naive_cost : int
        Naive cost estimate

    Returns
    -------
    tuple
        Tuple containing the resultant indices, input sets after contraction,
        indices removed, and indices contracted
    """
    # positions: tuple of int
    #     The locations of the proposed tensors to contract.
    # input_sets: list of sets
    #     The indices found on each tensor.
    # output_set: set
    #     The output indices of the expression.
    # idx_dict: dict
    #     Mapping of each index to its size.
    # memory_limit: int
    #     The total allowed size for an intermediary tensor.
    # path_cost: int
    #     The contraction cost so far.
    # naive_cost: int
    #     The cost of the unoptimized expression.
    # Returns
    # -------
    # cost: (int, int)
    #     A tuple containing the size of any indices removed, and the flop cost.
    # positions: tuple of int
    #     The locations of the proposed tensors to contract.
    # new_input_sets: list of sets
    #     The resulting new list of indices if this proposed contraction
    #     is performed.
    
    # Find the contraction based on provided positions and input/output sets
    contract = _find_contraction(positions, input_sets, output_set)
    idx_result, new_input_sets, idx_removed, idx_contract = contract

    # Compute the size of the resulting tensor based on idx_result and idx_dict
    new_size = _compute_size_by_dict(idx_result, idx_dict)
    
    # Check if the computed size exceeds the memory limit; if so, return None
    if new_size > memory_limit:
        return None

    # Calculate the sizes of the tensors before contraction
    old_sizes = (
        _compute_size_by_dict(input_sets[p], idx_dict) for p in positions
    )
    removed_size = sum(old_sizes) - new_size

    # Compute the flop count based on the contraction indices and idx_dict
    cost = _flop_count(idx_contract, idx_removed, len(positions), idx_dict)
    
    # Create a tuple for sorting purposes based on removed_size and cost
    sort = (-removed_size, cost)

    # Check if the total path cost plus the current cost exceeds naive_cost; if so, return None
    if (path_cost + cost) > naive_cost:
        return None

    # Return the result of the contraction as a list containing the sort tuple, positions, and new_input_sets
    return [sort, positions, new_input_sets]
def _update_other_results(results, best):
    """Update the positions and provisional input_sets of ``results``
    based on performing the contraction result ``best``. Remove any
    involving the tensors contracted.
    
    Parameters
    ----------
    results : list
        List of contraction results produced by 
        ``_parse_possible_contraction``.
    best : list
        The best contraction of ``results`` i.e. the one that
        will be performed.
        
    Returns
    -------
    mod_results : list
        The list of modified results, updated with outcome of
        ``best`` contraction.
    """

    best_con = best[1]  # Extract the contraction indices from the best contraction result
    bx, by = best_con    # Unpack the indices into bx and by
    mod_results = []     # Initialize an empty list to store modified results

    for cost, (x, y), con_sets in results:
        # Ignore results involving tensors just contracted
        if x in best_con or y in best_con:
            continue
        
        # Update the input_sets by removing contracted indices
        del con_sets[by - int(by > x) - int(by > y)]
        del con_sets[bx - int(bx > x) - int(bx > y)]
        con_sets.insert(-1, best[2][-1])  # Insert the result of the best contraction at the end of con_sets
        
        # Update the position indices based on the performed contraction
        mod_con = x - int(x > bx) - int(x > by), y - int(y > bx) - int(y > by)
        mod_results.append((cost, mod_con, con_sets))  # Append the modified result to mod_results list

    return mod_results  # Return the list of modified results


def _greedy_path(input_sets, output_set, idx_dict, memory_limit):
    """
    Finds the path by contracting the best pair until the input list is
    exhausted. The best pair is found by minimizing the tuple
    ``(-prod(indices_removed), cost)``.  What this amounts to is prioritizing
    matrix multiplication or inner product operations, then Hadamard like
    operations, and finally outer operations. Outer products are limited by
    ``memory_limit``. This algorithm scales cubically with respect to the
    number of elements in the list ``input_sets``.
    
    Parameters
    ----------
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    idx_dict : dictionary
        Dictionary of index sizes
    memory_limit : int
        The maximum number of elements in a temporary array
    
    Returns
    -------
    path : list
        The greedy contraction order within the memory limit constraint.
    
    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set()
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> _greedy_path(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """
    
    # Handle trivial cases that leaked through
    if len(input_sets) == 1:
        return [(0,)]
    elif len(input_sets) == 2:
        return [(0, 1)]
    
    # Build up a naive cost
    contract = _find_contraction(
        range(len(input_sets)), input_sets, output_set
    )
    idx_result, new_input_sets, idx_removed, idx_contract = contract
    naive_cost = _flop_count(
        idx_contract, idx_removed, len(input_sets), idx_dict
    )
    
    # Initially iterate over all pairs
    # 生成输入集合中元素的所有两两组合的迭代器
    comb_iter = itertools.combinations(range(len(input_sets)), 2)
    # 已知的可以收缩的对的列表
    known_contractions = []

    # 路径的总成本和路径的列表初始化
    path_cost = 0
    path = []

    # 对输入集合进行迭代，执行 len(input_sets) - 1 次
    for iteration in range(len(input_sets) - 1):

        # 在第一步遍历所有对，在后续步骤只遍历先前找到的对
        for positions in comb_iter:

            # 如果两个集合是不相交的，则忽略
            if input_sets[positions[0]].isdisjoint(input_sets[positions[1]]):
                continue

            # 调用函数尝试解析可能的收缩操作
            result = _parse_possible_contraction(
                positions, input_sets, output_set, idx_dict,
                memory_limit, path_cost, naive_cost
            )
            # 如果成功找到了收缩操作，则将其加入已知的收缩列表中
            if result is not None:
                known_contractions.append(result)

        # 如果没有找到内部收缩操作，则重新扫描包括外部乘积的对
        if len(known_contractions) == 0:

            # 检查包括外部乘积的所有对
            for positions in itertools.combinations(
                range(len(input_sets)), 2
            ):
                result = _parse_possible_contraction(
                    positions, input_sets, output_set, idx_dict,
                    memory_limit, path_cost, naive_cost
                )
                if result is not None:
                    known_contractions.append(result)

            # 如果仍然没有找到剩余的收缩操作，则默认回到 einsum 的行为方式
            if len(known_contractions) == 0:
                # 将整个集合作为一个路径元素添加到路径中
                path.append(tuple(range(len(input_sets))))
                break

        # 根据第一个索引排序已知的收缩操作列表
        best = min(known_contractions, key=lambda x: x[0])

        # 将尽可能多未使用的收缩操作传播到下一次迭代
        known_contractions = _update_other_results(known_contractions, best)

        # 下一次迭代仅计算新张量参与的收缩操作
        # 所有其他的收缩操作已经被处理过了
        input_sets = best[2]
        new_tensor_pos = len(input_sets) - 1
        # 生成新的两两组合迭代器，只包含新张量与其他张量的组合
        comb_iter = ((i, new_tensor_pos) for i in range(new_tensor_pos))

        # 更新路径和总成本
        path.append(best[1])
        path_cost += best[0][1]

    # 返回最终计算出的路径
    return path
# 定义名为 `_can_dot` 的函数，用于判断是否可以使用 BLAS（如 `np.tensordot`）进行张量点积操作，并且判断其是否有效。
def _can_dot(inputs, result, idx_removed):
    """
    Checks if we can use BLAS (np.tensordot) call and its beneficial to do so.

    Parameters
    ----------
    inputs : list of str
        Specifies the subscripts for summation.
    result : str
        Resulting summation.
    idx_removed : set
        Indices that are removed in the summation

    Returns
    -------
    type : bool
        Returns true if BLAS should and can be used, else False

    Notes
    -----
    If the operations is BLAS level 1 or 2 and is not already aligned
    we default back to einsum as the memory movement to copy is more
    costly than the operation itself.

    Examples
    --------

    # Standard GEMM operation
    >>> _can_dot(['ij', 'jk'], 'ik', set('j'))
    True

    # Can use the standard BLAS, but requires odd data movement
    >>> _can_dot(['ijj', 'jk'], 'ik', set('j'))
    False

    # DDOT where the memory is not aligned
    >>> _can_dot(['ijk', 'ikj'], '', set('ijk'))
    False

    """

    # 检查所有 `dot` 调用是否移除了索引
    if len(idx_removed) == 0:
        return False

    # BLAS 只能处理两个操作数
    if len(inputs) != 2:
        return False

    # 分别获取左右两个输入字符串
    input_left, input_right = inputs

    # 遍历输入字符串中的字符集合
    for c in set(input_left + input_right):
        # 不能处理同一输入中的重复索引或超过两个总数的情况
        nl, nr = input_left.count(c), input_right.count(c)
        if (nl > 1) or (nr > 1) or (nl + nr > 2):
            return False

        # 不能进行隐式求和或维度折叠，例如：
        # "ab,bc->c"（隐式求和 'a'）
        # "ab,ca->ca"（取 'a' 的对角线）
        if nl + nr - 1 == int(c in result):
            return False

    # 创建几个临时集合
    set_left = set(input_left)
    set_right = set(input_right)
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed
    rs = len(idx_removed)

    # 判断是否为 DOT、GEMV 或 GEMM 操作

    # 对齐数据的 DDOT 操作
    if input_left == input_right:
        return True

    # 不对齐数据的 DDOT 操作（建议使用 einsum）
    if set_left == set_right:
        return False

    # 处理四种可能的（对齐的）GEMV 或 GEMM 情况

    # GEMM 或 GEMV 操作，不转置
    if input_left[-rs:] == input_right[:rs]:
        return True

    # GEMM 或 GEMV 操作，两者都转置
    if input_left[:rs] == input_right[-rs:]:
        return True

    # GEMM 或 GEMV 操作，右侧转置
    if input_left[-rs:] == input_right[-rs:]:
        return True

    # GEMM 或 GEMV 操作，左侧转置
    if input_left[:rs] == input_right[:rs]:
        return True

    # 如果需要复制数据，则 einsum 比 GEMV 更快
    if not keep_left or not keep_right:
        return False

    # 我们是矩阵-矩阵乘积，但需要复制数据
    return True
    input_strings : str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the numpy contraction

    Examples
    --------
    The operand list is simplified to reduce printing:

    >>> np.random.seed(123)
    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> _parse_einsum_input(('...a,...a->...', a, b))
    ('za,xza', 'xz', [a, b]) # may vary

    >>> _parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('za,xza', 'xz', [a, b]) # may vary
    """

    # Check if there are no input operands, raise an error
    if len(operands) == 0:
        raise ValueError("No input operands")

    # If the first operand is a string, parse subscripts and validate characters
    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = [asanyarray(v) for v in operands[1:]]

        # Ensure all characters in subscripts are valid
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)

    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []

        # Parse each pair of operand and its subscript
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = [asanyarray(v) for v in operand_list]
        subscripts = ""
        last = len(subscript_list) - 1

        # Construct the subscripts string based on the parsed subscripts
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += "..."
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError(
                            "For this input type lists must contain "
                            "either int or Ellipsis"
                        ) from e
                    subscripts += einsum_symbols[s]
            if num != last:
                subscripts += ","

        # If there is an output list, append it to the subscripts string
        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "..."
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError(
                            "For this input type lists must contain "
                            "either int or Ellipsis"
                        ) from e
                    subscripts += einsum_symbols[s]

    # Check for proper '->' syntax in subscripts
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    # 如果子脚本中包含"."，则去除"."、","和"->"后，得到使用过的字符集合
    used = subscripts.replace(".", "").replace(",", "").replace("->", "")
    # 计算未使用的字符集合，即 einsum_symbols_set 减去已使用的字符集合
    unused = list(einsum_symbols_set - set(used))
    # 将未使用的字符集合转换为字符串
    ellipse_inds = "".join(unused)
    # 初始化最长 ellipsis 的长度为 0
    longest = 0

    # 如果子脚本包含"->"，则按"->"分割输入和输出子脚本
    if "->" in subscripts:
        input_tmp, output_sub = subscripts.split("->")
        split_subscripts = input_tmp.split(",")
        out_sub = True
    else:
        split_subscripts = subscripts.split(',')
        out_sub = False

    # 遍历分割后的子脚本列表
    for num, sub in enumerate(split_subscripts):
        # 如果子脚本中包含"."，并且不符合 ellipsis 的规则，则引发 ValueError
        if "." in sub:
            if (sub.count(".") != 3) or (sub.count("...") != 1):
                raise ValueError("Invalid Ellipses.")

            # 计算 ellipsis 的长度，考虑操作数的形状
            if operands[num].shape == ():
                ellipse_count = 0
            else:
                ellipse_count = max(operands[num].ndim, 1)
                ellipse_count -= (len(sub) - 3)

            # 更新最长 ellipsis 的长度
            if ellipse_count > longest:
                longest = ellipse_count

            # 检查 ellipsis 的长度是否合法
            if ellipse_count < 0:
                raise ValueError("Ellipses lengths do not match.")
            elif ellipse_count == 0:
                split_subscripts[num] = sub.replace('...', '')
            else:
                # 替换 ellipsis 为未使用字符集合中对应长度的字符
                rep_inds = ellipse_inds[-ellipse_count:]
                split_subscripts[num] = sub.replace('...', rep_inds)

    # 将更新后的子脚本列表重新拼接为字符串
    subscripts = ",".join(split_subscripts)

    # 根据最长 ellipsis 的长度确定输出 ellipsis 字符串
    if longest == 0:
        out_ellipse = ""
    else:
        out_ellipse = ellipse_inds[-longest:]

    # 如果有输出子脚本，则根据最长 ellipsis 更新输出子脚本
    if out_sub:
        subscripts += "->" + output_sub.replace("...", out_ellipse)
    else:
        # 处理无输出子脚本时的特殊情况
        output_subscript = ""
        tmp_subscripts = subscripts.replace(",", "")
        # 检查并构建正常的输出子脚本
        for s in sorted(set(tmp_subscripts)):
            if s not in (einsum_symbols):
                raise ValueError("Character %s is not a valid symbol." % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s
        # 从正常的输出子脚本中删除输出 ellipsis 的字符，得到正常的索引字符
        normal_inds = ''.join(sorted(set(output_subscript) - set(out_ellipse)))
        # 更新子脚本字符串，包括输出 ellipsis 和正常的索引字符
        subscripts += "->" + out_ellipse + normal_inds

# 如果子脚本中包含"->"，则按"->"分割输入子脚本和输出子脚本
if "->" in subscripts:
    input_subscripts, output_subscript = subscripts.split("->")
else:
    # 如果没有输出子脚本，则将整个子脚本作为输入子脚本
    input_subscripts = subscripts

# 确保输出子脚本中的字符都在输入子脚本中存在
    # 遍历输出下标字符串中的每个字符
    for char in output_subscript:
        # 检查字符在输出下标字符串中出现的次数，应该为1次，否则引发值错误异常
        if output_subscript.count(char) != 1:
            raise ValueError("Output character %s appeared more than once in "
                             "the output." % char)
        # 检查字符是否存在于输入下标字符串中，否则引发值错误异常
        if char not in input_subscripts:
            raise ValueError("Output character %s did not appear in the input"
                             % char)

    # 确保输入下标字符串的逗号分隔的子字符串数量等于操作数的数量
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the "
                         "number of operands.")

    # 返回元组，包含输入下标字符串、输出下标字符串和操作数列表
    return (input_subscripts, output_subscript, operands)
# 根据传入的参数来分发 einsum_path 函数调用，选择合适的优化方法和路径
def _einsum_path_dispatcher(*operands, optimize=None, einsum_call=None):
    # 注意：从技术上讲，我们应该只对类数组的参数进行分发，而不是对子脚本（以字符串形式给出）进行分发。
    # 但是，将操作数分开为数组/子脚本有些复杂/慢（考虑到 einsum 的两种支持签名），
    # 所以作为一种实际上的快捷方式，我们对所有内容进行分发。
    # 字符串将被忽略以用于分发，因为它们不定义 __array_function__。
    return operands


# 使用 numpy 提供的 array_function_dispatch 装饰器，将 _einsum_path_dispatcher 函数注册为 einsum_path 的分发函数
@array_function_dispatch(_einsum_path_dispatcher, module='numpy')
def einsum_path(*operands, optimize='greedy', einsum_call=False):
    """
    einsum_path(subscripts, *operands, optimize='greedy')

    根据考虑中间数组的创建来评估 einsum 表达式的最低成本收缩顺序。

    Parameters
    ----------
    subscripts : str
        指定求和的下标。
    *operands : list of array_like
        这些是操作的数组。
    optimize : {bool, list, tuple, 'greedy', 'optimal'}
        选择路径的类型。如果提供了一个元组，假定第二个参数是所创建的最大中间大小。
        如果只提供了一个参数，则使用最大输入或输出数组大小作为最大中间大小。

        * 如果给定以 ``einsum_path`` 开头的列表，将其用作收缩路径
        * 如果为 False，不进行优化
        * 如果为 True，默认使用 'greedy' 算法
        * 'optimal' 一种算法，通过组合地探索所有可能的张量收缩方式并选择成本最低的路径。随着收缩项数的增加呈指数级增长。
        * 'greedy' 一种算法，在每一步选择最佳的对子收缩。实际上，此算法在每一步搜索最大的内部、哈达玛尔乘积，然后外积。随着收缩项数的增加呈立方级增长。对于大多数收缩，与 'optimal' 路径等效。

        默认为 'greedy'。

    Returns
    -------
    path : list of tuples
        einsum 路径的列表表示。
    string_repr : str
        einsum 路径的可打印表示。

    Notes
    -----
    结果路径指示应首先收缩输入收缩的哪些项，然后将此收缩的结果附加到收缩列表的末尾。
    然后可以遍历此列表，直到所有中间收缩完成。

    See Also
    --------
    einsum, linalg.multi_dot

    Examples
    --------

    我们可以从一个链式点乘的例子开始。在这种情况下，首先收缩 ``b`` 和 ``c`` 张量是最优的，由路径的第一个元素 ``(1, 2)`` 表示。
    收缩的结果张量被添加到末尾
    """
    """
    of the contraction and the remaining contraction ``(0, 1)`` is then
    completed.

    >>> np.random.seed(123)
    >>> a = np.random.rand(2, 2)
    >>> b = np.random.rand(2, 5)
    >>> c = np.random.rand(5, 2)
    >>> path_info = np.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')
    >>> print(path_info[0])
    ['einsum_path', (1, 2), (0, 1)]
    >>> print(path_info[1])
      Complete contraction:  ij,jk,kl->il # may vary
             Naive scaling:  4
         Optimized scaling:  3
          Naive FLOP count:  1.600e+02
      Optimized FLOP count:  5.600e+01
       Theoretical speedup:  2.857
      Largest intermediate:  4.000e+00 elements
    -------------------------------------------------------------------------
    scaling                  current                                remaining
    -------------------------------------------------------------------------
       3                   kl,jk->jl                                ij,jl->il
       3                   jl,ij->il                                   il->il


    A more complex index transformation example.

    >>> I = np.random.rand(10, 10, 10, 10)
    >>> C = np.random.rand(10, 10)
    >>> path_info = np.einsum_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C,
    ...                            optimize='greedy')

    >>> print(path_info[0])
    ['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)]
    >>> print(path_info[1]) 
      Complete contraction:  ea,fb,abcd,gc,hd->efgh # may vary
             Naive scaling:  8
         Optimized scaling:  5
          Naive FLOP count:  8.000e+08
      Optimized FLOP count:  8.000e+05
       Theoretical speedup:  1000.000
      Largest intermediate:  1.000e+04 elements
    --------------------------------------------------------------------------
    scaling                  current                                remaining
    --------------------------------------------------------------------------
       5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh
       5               bcde,fb->cdef                         gc,hd,cdef->efgh
       5               cdef,gc->defg                            hd,defg->efgh
       5               defg,hd->efgh                               efgh->efgh
    """

    # Figure out what the path really is
    path_type = optimize  # Assign the value of 'optimize' to 'path_type'

    if path_type is True:  # Check if 'path_type' is exactly True
        path_type = 'greedy'  # Set 'path_type' to 'greedy' if it is True

    if path_type is None:  # Check if 'path_type' is exactly None
        path_type = False  # Set 'path_type' to False if it is None

    explicit_einsum_path = False  # Initialize a boolean flag 'explicit_einsum_path' to False
    memory_limit = None  # Initialize 'memory_limit' to None

    # No optimization or a named path algorithm
    if (path_type is False) or isinstance(path_type, str):  # Check if 'path_type' is False or a string
        pass  # If so, do nothing

    # Given an explicit path
    elif len(path_type) and (path_type[0] == 'einsum_path'):  # Check if 'path_type' has length and starts with 'einsum_path'
        explicit_einsum_path = True  # Set 'explicit_einsum_path' to True

    # Path tuple with memory limit
    elif ((len(path_type) == 2) and isinstance(path_type[0], str) and
            isinstance(path_type[1], (int, float))):  # Check if 'path_type' is a tuple with specific types and length
        memory_limit = int(path_type[1])  # Set 'memory_limit' to the integer value of the second element in 'path_type'
        path_type = path_type[0]  # Update 'path_type' to the first element in 'path_type'
    ```
    else:
        raise TypeError("Did not understand the path: %s" % str(path_type))
    # 如果路径类型不被理解，则抛出类型错误异常

    # Hidden option, only einsum should call this
    einsum_call_arg = einsum_call
    # 隐藏选项，只有 einsum 应该调用这个

    # Python side parsing
    input_subscripts, output_subscript, operands = (
        _parse_einsum_input(operands)
    )
    # 解析输入，获取输入子标记、输出子标记以及操作数

    # Build a few useful list and sets
    input_list = input_subscripts.split(',')
    # 将输入子标记拆分为列表
    input_sets = [set(x) for x in input_list]
    # 创建输入子标记的集合列表
    output_set = set(output_subscript)
    # 创建输出子标记的集合
    indices = set(input_subscripts.replace(',', ''))
    # 获取所有索引的集合

    # Get length of each unique dimension and ensure all dimensions are correct
    dimension_dict = {}
    # 创建维度字典
    broadcast_indices = [[] for x in range(len(input_list))]
    # 创建广播索引的列表

    for tnum, term in enumerate(input_list):
        sh = operands[tnum].shape
        # 获取操作数的形状
        if len(sh) != len(term):
            raise ValueError("Einstein sum subscript %s does not contain the "
                             "correct number of indices for operand %d."
                             % (input_subscripts[tnum], tnum))
        # 如果子标记中的索引数量与操作数的维度不匹配，则引发值错误异常
        for cnum, char in enumerate(term):
            dim = sh[cnum]
            # 获取维度

            # Build out broadcast indices
            if dim == 1:
                broadcast_indices[tnum].append(char)
            # 如果维度为1，则添加到广播索引列表中

            if char in dimension_dict.keys():
                # For broadcasting cases we always want the largest dim size
                if dimension_dict[char] == 1:
                    dimension_dict[char] = dim
                elif dim not in (1, dimension_dict[char]):
                    raise ValueError("Size of label '%s' for operand %d (%d) "
                                     "does not match previous terms (%d)."
                                     % (char, tnum, dimension_dict[char], dim))
            else:
                dimension_dict[char] = dim
            # 更新维度字典中的值

    # Convert broadcast inds to sets
    broadcast_indices = [set(x) for x in broadcast_indices]
    # 将广播索引转换为集合

    # Compute size of each input array plus the output array
    size_list = [_compute_size_by_dict(term, dimension_dict)
                 for term in input_list + [output_subscript]]
    # 计算每个输入数组及输出数组的大小列表
    max_size = max(size_list)
    # 获取最大大小

    if memory_limit is None:
        memory_arg = max_size
    else:
        memory_arg = memory_limit
    # 设置内存参数

    # Compute naive cost
    # This isn't quite right, need to look into exactly how einsum does this
    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    # 计算内积
    naive_cost = _flop_count(
        indices, inner_product, len(input_list), dimension_dict
    )
    # 计算天真的成本

    # Compute the path
    if explicit_einsum_path:
        path = path_type[1:]
    elif (
        (path_type is False)
        or (len(input_list) in [1, 2])
        or (indices == output_set)
    ):
        # Nothing to be optimized, leave it to einsum
        path = [tuple(range(len(input_list)))]
    elif path_type == "greedy":
        path = _greedy_path(
            input_sets, output_set, dimension_dict, memory_arg
        )
    # 计算路径的选择逻辑
    elif path_type == "optimal":
        # 如果路径类型是 "optimal"，则调用 _optimal_path 函数计算最优路径
        path = _optimal_path(
            input_sets, output_set, dimension_dict, memory_arg
        )
    else:
        # 如果路径类型不是 "optimal"，则抛出 KeyError 异常
        raise KeyError("Path name %s not found", path_type)

    # 初始化空列表用于存储成本、缩放、大小和收缩元组
    cost_list, scale_list, size_list, contraction_list = [], [], [], []

    # 构建收缩元组 (位置、gemm、einsum_str、剩余)
    for cnum, contract_inds in enumerate(path):
        # 确保从右到左移除 inds
        contract_inds = tuple(sorted(list(contract_inds), reverse=True))

        # 查找并返回收缩的结果以及相关信息
        contract = _find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract

        # 计算 FLOP 数量
        cost = _flop_count(
            idx_contract, idx_removed, len(contract_inds), dimension_dict
        )
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(_compute_size_by_dict(out_inds, dimension_dict))

        bcast = set()
        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))
            bcast |= broadcast_indices.pop(x)

        new_bcast_inds = bcast - idx_removed

        # 如果有广播操作，不使用 BLAS
        if not len(idx_removed & bcast):
            do_blas = _can_dot(tmp_inputs, out_inds, idx_removed)
        else:
            do_blas = False

        # 最后一个收缩
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            idx_result = "".join([x[1] for x in sorted(sort_result)])

        input_list.append(idx_result)
        broadcast_indices.append(new_bcast_inds)
        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        # 构建收缩元组并添加到列表中
        contraction = (
            contract_inds, idx_removed, einsum_str, input_list[:], do_blas
        )
        contraction_list.append(contraction)

    # 计算优化成本
    opt_cost = sum(cost_list) + 1

    if len(input_list) != 1:
        # 如果输入列表长度不为1，抛出 RuntimeError 异常
        raise RuntimeError(
            "Invalid einsum_path is specified: {} more operands has to be "
            "contracted.".format(len(input_list) - 1))

    if einsum_call_arg:
        # 如果 einsum_call_arg 为真，返回操作数和收缩列表
        return (operands, contraction_list)

    # 返回路径以及漂亮的字符串表示
    overall_contraction = input_subscripts + "->" + output_subscript
    header = ("scaling", "current", "remaining")

    # 计算速度提升比例和最大大小
    speedup = naive_cost / opt_cost
    max_i = max(size_list)

    # 构建打印路径的字符串
    path_print = "  Complete contraction:  %s\n" % overall_contraction
    path_print += "         Naive scaling:  %d\n" % len(indices)
    path_print += "     Optimized scaling:  %d\n" % max(scale_list)
    path_print += "      Naive FLOP count:  %.3e\n" % naive_cost
    path_print += "  Optimized FLOP count:  %.3e\n" % opt_cost
    # 将理论加速比格式化为字符串，并添加到 path_print 中
    path_print += "   Theoretical speedup:  %3.3f\n" % speedup
    # 将最大中间元素数格式化为科学计数法字符串，并添加到 path_print 中
    path_print += "  Largest intermediate:  %.3e elements\n" % max_i
    # 添加一行分隔线到 path_print 中
    path_print += "-" * 74 + "\n"
    # 使用格式化字符串将表头添加到 path_print 中
    path_print += "%6s %24s %40s\n" % header
    # 添加一行长的分隔线到 path_print 中
    path_print += "-" * 74

    # 遍历收缩列表 contraction_list
    for n, contraction in enumerate(contraction_list):
        # 解包收缩操作的各个部分
        inds, idx_rm, einsum_str, remaining, blas = contraction
        # 构造剩余指标的字符串表示形式，形如 "i,j,k->abc"
        remaining_str = ",".join(remaining) + "->" + output_subscript
        # 构造当前路径的运行信息
        path_run = (scale_list[n], einsum_str, remaining_str)
        # 将当前路径的运行信息格式化并添加到 path_print 中
        path_print += "\n%4d    %24s %40s" % path_run

    # 构造路径列表，并将 'einsum_path' 字符串添加到开头
    path = ['einsum_path'] + path
    # 返回路径列表和打印输出的字符串
    return (path, path_print)
# 定义一个生成器函数 _einsum_dispatcher，用于分派参数
def _einsum_dispatcher(*operands, out=None, optimize=None, **kwargs):
    # 根据注释，此处应该是解释为什么会分派更多参数，参见 _einsum_path_dispatcher 的说明。
    # 生成器函数通过 yield 语句产生操作数及输出对象
    yield from operands
    yield out


# 使用 array_function_dispatch 装饰器将 _einsum_dispatcher 作为分派函数，指定它属于 numpy 模块
@array_function_dispatch(_einsum_dispatcher, module='numpy')
# 定义 einsum 函数，用于处理不同的情况
def einsum(*operands, out=None, optimize=False, **kwargs):
    """
    einsum(subscripts, *operands, out=None, dtype=None, order='K',
           casting='safe', optimize=False)

    根据爱因斯坦求和约定在操作数上执行求和。

    使用爱因斯坦求和约定，许多常见的多维线性代数数组操作可以用简单的方式表示。
    在 *implicit* 模式下，`einsum` 计算这些值。

    在 *explicit* 模式下，`einsum` 提供了更多的灵活性，可以计算其他可能不是经典爱因斯坦求和操作的数组操作，
    通过禁用或强制求和指定的下标标签。

    请参阅注释和示例以获取更多说明。

    Parameters
    ----------
    subscripts : str
        指定求和下标的字符串，以逗号分隔的下标标签列表。除非明确指定输出形式，否则将执行隐式（经典爱因斯坦求和）计算。
    operands : list of array_like
        这些是操作的数组。
    out : ndarray, optional
        如果提供，则将计算结果存入此数组。
    dtype : {data-type, None}, optional
        如果提供，则强制使用指定的数据类型进行计算。
        请注意，可能需要更宽松的 `casting` 参数来允许转换。默认为 None。
    order : {'C', 'F', 'A', 'K'}, optional
        控制输出的内存布局。'C' 表示应为 C 连续，'F' 表示应为 Fortran 连续，
        'A' 表示如果输入全部为 'F'，则输出也应为 'F'，否则为 'C'。
        'K' 表示应尽可能接近输入的布局，包括任意排列的轴。
        默认为 'K'。
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        控制可能发生的数据转换类型。不推荐设置为 'unsafe'，因为可能会对累积产生不利影响。

        * 'no' 表示根本不应进行数据类型转换。
        * 'equiv' 表示只允许字节顺序的改变。
        * 'safe' 表示只允许可以保留值的转换。
        * 'same_kind' 表示只允许安全转换或同一种类内的转换，例如 float64 到 float32。
        * 'unsafe' 表示可以进行任何数据转换。

        默认为 'safe'。
    optimize : bool, optional
        是否应该优化 einsum 路径。默认为 False。
    """
    # 函数文档字符串提供了 einsum 函数的详细说明，包括它的参数和使用方法。
    # 函数本身实现了根据爱因斯坦求和约定执行多维数组操作的功能。
    pass  # 函数体暂未提供，实际实现中会根据 subscripts 和 operands 执行相应的计算
    optimize : {False, True, 'greedy', 'optimal'}, optional
        # 控制是否进行中间优化。如果为False，则不进行优化；如果为True，默认使用'greedy'算法。
        # 也可以接受来自np.einsum_path函数的显式缩并列表。详见np.einsum_path获取更多信息。默认为False。

    Returns
    -------
    output : ndarray
        # 基于爱因斯坦求和约定进行的计算结果。

    See Also
    --------
    einsum_path, dot, inner, outer, tensordot, linalg.multi_dot
    einsum:
        # `einsum`函数提供了一种简洁的方式来表示多维线性代数数组操作。

    Notes
    -----
    .. versionadded:: 1.6.0

    The Einstein summation convention can be used to compute
    many multi-dimensional, linear algebraic array operations. `einsum`
    provides a succinct way of representing these.
    # 爱因斯坦求和约定可用于计算许多多维线性代数数组操作。`einsum`提供了一种简洁的表示方式。

    A non-exhaustive list of these operations,
    which can be computed by `einsum`, is shown below along with examples:
    # 下面列出了一些可通过`einsum`计算的操作，以及相应的示例：

    * Trace of an array, :py:func:`numpy.trace`.
    * Return a diagonal, :py:func:`numpy.diag`.
    * Array axis summations, :py:func:`numpy.sum`.
    * Transpositions and permutations, :py:func:`numpy.transpose`.
    * Matrix multiplication and dot product, :py:func:`numpy.matmul`
        :py:func:`numpy.dot`.
    * Vector inner and outer products, :py:func:`numpy.inner`
        :py:func:`numpy.outer`.
    * Broadcasting, element-wise and scalar multiplication,
        :py:func:`numpy.multiply`.
    * Tensor contractions, :py:func:`numpy.tensordot`.
    * Chained array operations, in efficient calculation order,
        :py:func:`numpy.einsum_path`.
    # 列出了一些可以通过`einsum`计算的操作，以及相应的示例，包括数组的迹、对角线、轴的求和、转置、矩阵乘法、向量内积外积、广播、张量收缩以及连锁数组操作。

    The subscripts string is a comma-separated list of subscript labels,
    where each label refers to a dimension of the corresponding operand.
    # 子脚本字符串是逗号分隔的子脚本标签列表，其中每个标签都引用相应操作数的一个维度。

    In *implicit mode*, the chosen subscripts are important
    since the axes of the output are reordered alphabetically.  This
    means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while
    # 在隐式模式下，所选择的子脚本很重要，因为输出的轴会按字母顺序重新排序。
    # 这意味着`np.einsum('ij', a)`不会影响一个2D数组，
    # 使用 `np.einsum` 函数可以根据爱因斯坦求和约定计算张量的乘积、对角线元素、轴上的和等操作。
    
    # 当操作符只有一个时，不会进行轴上的求和，也不会返回输出参数，而是返回操作数的视图。
    
    # `einsum` 还提供了另一种方式来提供子脚本和操作数，即 `einsum(op0, sublist0, op1, sublist1, ..., [sublistout])`。如果在这种格式中没有提供输出形状，`einsum` 将以隐式模式计算，否则将显式执行。
    
    # 从版本 1.10.0 开始，`einsum` 返回的视图现在在输入数组可写时也是可写的。
    
    # 版本 1.12.0 中新增了 `optimize` 参数，它可以优化 `einsum` 表达式的收缩顺序。对于三个或更多操作数的收缩，这可以大幅提高计算效率，但在计算过程中会增加内存占用。
    Typically a 'greedy' algorithm is applied which empirical tests have shown
    returns the optimal path in the majority of cases. In some cases 'optimal'
    will return the superlative path through a more expensive, exhaustive
    search. For iterative calculations it may be advisable to calculate
    the optimal path once and reuse that path by supplying it as an argument.
    An example is given below.

    See :py:func:`numpy.einsum_path` for more details.


    # 通常使用“贪婪”算法，经验测试表明在大多数情况下返回最优路径。在某些情况下，“optimal”
    # 会通过更昂贵、详尽的搜索返回卓越路径。对于迭代计算，建议计算一次最优路径并通过参数重复使用该路径。
    # 下面给出一个示例。
    
    # 查看更多详细信息，请参阅 :py:func:`numpy.einsum_path`。


```    
    Examples
    --------
    >>> a = np.arange(25).reshape(5,5)
    >>> b = np.arange(5)
    >>> c = np.arange(6).reshape(2,3)

    Trace of a matrix:


    # 创建示例数组
    a = np.arange(25).reshape(5,5)
    b = np.arange(5)
    c = np.arange(6).reshape(2,3)

    # 计算矩阵的迹



    >>> np.einsum('ii', a)
    60
    >>> np.einsum(a, [0,0])
    60
    >>> np.trace(a)
    60


    # 使用einsum计算矩阵对角线元素之和
    >>> np.einsum('ii', a)
    60
    >>> np.einsum(a, [0,0])
    60
    >>> np.trace(a)
    60



    Extract the diagonal (requires explicit form):


    # 提取对角线元素（需要显式形式）



    >>> np.einsum('ii->i', a)
    array([ 0,  6, 12, 18, 24])
    >>> np.einsum(a, [0,0], [0])
    array([ 0,  6, 12, 18, 24])
    >>> np.diag(a)
    array([ 0,  6, 12, 18, 24])


    # 使用einsum提取对角线元素到一维数组
    >>> np.einsum('ii->i', a)
    array([ 0,  6, 12, 18, 24])
    >>> np.einsum(a, [0,0], [0])
    array([ 0,  6, 12, 18, 24])
    >>> np.diag(a)
    array([ 0,  6, 12, 18, 24])



    Sum over an axis (requires explicit form):


    # 沿着指定轴求和（需要显式形式）



    >>> np.einsum('ij->i', a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [0,1], [0])
    array([ 10,  35,  60,  85, 110])
    >>> np.sum(a, axis=1)
    array([ 10,  35,  60,  85, 110])


    # 使用einsum沿着行求和并返回结果数组
    >>> np.einsum('ij->i', a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [0,1], [0])
    array([ 10,  35,  60,  85, 110])
    >>> np.sum(a, axis=1)
    array([ 10,  35,  60,  85, 110])



    For higher dimensional arrays summing a single axis can be done
    with ellipsis:


    # 对于更高维度的数组，可以使用省略号来沿着单个轴求和



    >>> np.einsum('...j->...', a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [Ellipsis,1], [Ellipsis])
    array([ 10,  35,  60,  85, 110])


    # 使用einsum沿着列（最后一个轴）求和并返回结果数组
    >>> np.einsum('...j->...', a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [Ellipsis,1], [Ellipsis])
    array([ 10,  35,  60,  85, 110])



    Compute a matrix transpose, or reorder any number of axes:


    # 计算矩阵转置或重新排列任意数量的轴



    >>> np.einsum('ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum('ij->ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum(c, [1,0])
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.transpose(c)
    array([[0, 3],
           [1, 4],
           [2, 5]])


    # 使用einsum计算矩阵转置或重排轴的操作及结果
    >>> np.einsum('ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum('ij->ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum(c, [1,0])
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.transpose(c)
    array([[0, 3],
           [1, 4],
           [2, 5]])



    Vector inner products:


    # 向量的内积运算



    >>> np.einsum('i,i', b, b)
    30
    >>> np.einsum(b, [0], b, [0])
    30
    >>> np.inner(b,b)
    30


    # 使用einsum计算向量的内积并返回结果
    >>> np.einsum('i,i', b, b)
    30
    >>> np.einsum(b, [0], b, [0])
    30
    >>> np.inner(b,b)
    30



    Matrix vector multiplication:


    # 矩阵与向量的乘法运算



    >>> np.einsum('ij,j', a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum(a, [0,1], b, [1])
    array([ 30,  80, 130, 180, 230])
    >>> np.dot(a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum('...j,j', a, b)
    array([ 30,  80, 130, 180, 230])


    # 使用einsum计算矩阵与向量的乘积并返回结果数组
    >>> np.einsum('ij,j', a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum(a, [0,1], b, [1])
    array([ 30,  80, 130, 180, 230])
    >>> np.dot(a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum('...j,j', a, b)
    array([ 30,  80, 130, 180, 230])



    Broadcasting and
    # 如果指定了输出（out），则特殊处理
    specified_out = out is not None
    
    # 如果不进行优化，使用纯粹的 einsum 运算
    if optimize is False:
        # 如果指定了输出，将其设为关键字参数中的输出
        if specified_out:
            kwargs['out'] = out
        # 返回通过 c_einsum 函数计算得到的结果，使用给定的操作数和关键字参数
        return c_einsum(*operands, **kwargs)
    
    # 检查关键字参数，避免在稍后出现更难理解的错误，同时不必在此处重复默认值
    valid_einsum_kwargs = ['dtype', 'order', 'casting']
    # 检查是否存在未知的关键字参数
    unknown_kwargs = [k for (k, v) in kwargs.items() if
                      k not in valid_einsum_kwargs]
    # 如果存在未知参数，抛出类型错误异常
    if len(unknown_kwargs):
        raise TypeError("Did not understand the following kwargs: %s"
                        % unknown_kwargs)

    # 构建收缩列表和操作数
    operands, contraction_list = einsum_path(*operands, optimize=optimize,
                                             einsum_call=True)

    # 处理输出数组的顺序关键字参数
    output_order = kwargs.pop('order', 'K')
    # 如果输出顺序是 'A'，并且所有操作数都是 F 连续的，则设定输出顺序为 'F'；否则设定为 'C'
    if output_order.upper() == 'A':
        if all(arr.flags.f_contiguous for arr in operands):
            output_order = 'F'
        else:
            output_order = 'C'

    # 开始收缩循环
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, blas = contraction
        tmp_operands = [operands.pop(x) for x in inds]

        # 是否需要处理输出？
        handle_out = specified_out and ((num + 1) == len(contraction_list))

        # 如果是 BLAS 收缩
        if blas:
            # 已经处理过检查
            input_str, results_index = einsum_str.split('->')
            input_left, input_right = input_str.split(',')

            # 构建收缩后的结果张量
            tensor_result = input_left + input_right
            for s in idx_rm:
                tensor_result = tensor_result.replace(s, "")

            # 查找需要收缩的索引位置
            left_pos, right_pos = [], []
            for s in sorted(idx_rm):
                left_pos.append(input_left.find(s))
                right_pos.append(input_right.find(s))

            # 执行张量收缩
            new_view = tensordot(
                *tmp_operands, axes=(tuple(left_pos), tuple(right_pos))
            )

            # 如果需要构建新视图或者处理输出，调用 c_einsum 函数
            if (tensor_result != results_index) or handle_out:
                if handle_out:
                    kwargs["out"] = out
                new_view = c_einsum(
                    tensor_result + '->' + results_index, new_view, **kwargs
                )

        # 如果不是 BLAS 收缩，调用 einsum 函数
        else:
            # 如果指定了输出参数 out
            if handle_out:
                kwargs["out"] = out

            # 执行收缩操作
            new_view = c_einsum(einsum_str, *tmp_operands, **kwargs)

        # 将新视图添加到操作数列表中，并删除临时操作数和新视图引用
        operands.append(new_view)
        del tmp_operands, new_view

    # 如果指定了输出 out，返回 out；否则将第一个操作数转换为数组并返回，按指定的顺序
    if specified_out:
        return out
    else:
        return asanyarray(operands[0], order=output_order)
```