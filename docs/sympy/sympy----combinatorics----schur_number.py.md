# `D:\src\scipysrc\sympy\sympy\combinatorics\schur_number.py`

```
"""
The Schur number S(k) is the largest integer n for which the interval [1,n]
can be partitioned into k sum-free sets.(https://mathworld.wolfram.com/SchurNumber.html)
"""
# 导入必要的数学库
import math
# 导入 sympy 库的相关模块
from sympy.core import S
from sympy.core.basic import Basic
from sympy.core.function import Function
from sympy.core.numbers import Integer

# 定义一个函数类 SchurNumber，继承自 Function 类
class SchurNumber(Function):
    r"""
    This function creates a SchurNumber object
    which is evaluated for `k \le 5` otherwise only
    the lower bound information can be retrieved.

    Examples
    ========

    >>> from sympy.combinatorics.schur_number import SchurNumber

    Since S(3) = 13, hence the output is a number
    >>> SchurNumber(3)
    13

    We do not know the Schur number for values greater than 5, hence
    only the object is returned
    >>> SchurNumber(6)
    SchurNumber(6)

    Now, the lower bound information can be retrieved using lower_bound()
    method
    >>> SchurNumber(6).lower_bound()
    536

    """

    @classmethod
    # 类方法 eval，用于计算 Schur number 的值
    def eval(cls, k):
        # 检查 k 是否为数值
        if k.is_Number:
            # 若 k 为无穷大，则返回无穷大
            if k is S.Infinity:
                return S.Infinity
            # 若 k 为零，则返回零
            if k.is_zero:
                return S.Zero
            # 若 k 不是正整数或者为负数，则抛出异常
            if not k.is_integer or k.is_negative:
                raise ValueError("k should be a positive integer")
            # 已知的 Schur number 的初始值
            first_known_schur_numbers = {1: 1, 2: 4, 3: 13, 4: 44, 5: 160}
            # 若 k <= 5，则返回对应的 Schur number
            if k <= 5:
                return Integer(first_known_schur_numbers[k])

    # 定义一个方法 lower_bound，返回 Schur number 的下界信息
    def lower_bound(self):
        # 获取参数列表中的第一个参数
        f_ = self.args[0]
        # 对于特定的 f_ 值，返回改进的下界值
        if f_ == 6:
            return Integer(536)
        if f_ == 7:
            return Integer(1680)
        # 对于其他情况，使用一般的表达式计算下界
        if f_.is_Integer:
            return 3*self.func(f_ - 1).lower_bound() - 1
        return (3**f_ - 1)/2

# 定义一个函数 _schur_subsets_number，计算给定 n 的最小 Schur number 下界
def _schur_subsets_number(n):
    # 若 n 为无穷大，则抛出异常
    if n is S.Infinity:
        raise ValueError("Input must be finite")
    # 若 n 小于等于零，则抛出异常
    if n <= 0:
        raise ValueError("n must be a non-zero positive integer.")
    elif n <= 3:
        min_k = 1
    else:
        # 计算大于 2n + 1 的 3 的对数的向上取整值作为最小 k
        min_k = math.ceil(math.log(2*n + 1, 3))

    return Integer(min_k)

# 定义函数 schur_partition，返回在给定 Schur number 下界条件下的最小和自由子集的分割
def schur_partition(n):
    """

    This function returns the partition in the minimum number of sum-free subsets
    according to the lower bound given by the Schur Number.

    Parameters
    ==========

    n: a number
        n is the upper limit of the range [1, n] for which we need to find and
        return the minimum number of free subsets according to the lower bound
        of schur number

    Returns
    =======

    List of lists
        List of the minimum number of sum-free subsets

    Notes
    =====

    It is possible for some n to make the partition into less
    subsets since the only known Schur numbers are:
    S(1) = 1, S(2) = 4, S(3) = 13, S(4) = 44.
    e.g for n = 44 the lower bound from the function above is 5 subsets but it has been proven
    that can be done with 4 subsets.

    Examples
    ========

    """
    For n = 1, 2, 3 the answer is the set itself

    >>> from sympy.combinatorics.schur_number import schur_partition
    >>> schur_partition(2)
    [[1, 2]]

    For n > 3, the answer is the minimum number of sum-free subsets:

    >>> schur_partition(5)
    [[3, 2], [5], [1, 4]]

    >>> schur_partition(8)
    [[3, 2], [6, 5, 8], [1, 4, 7]]
    """

    # 如果输入的 n 不是一个数值类型，抛出值错误异常
    if isinstance(n, Basic) and not n.is_Number:
        raise ValueError("Input value must be a number")

    # 计算 n 对应的 Schur 数，即 sum-free 子集的数量
    number_of_subsets = _schur_subsets_number(n)
    
    # 根据不同的 n 值，初始化 sum-free 子集列表
    if n == 1:
        sum_free_subsets = [[1]]
    elif n == 2:
        sum_free_subsets = [[1, 2]]
    elif n == 3:
        sum_free_subsets = [[1, 2, 3]]
    else:
        sum_free_subsets = [[1, 4], [2, 3]]

    # 循环直到 sum-free 子集的数量达到 Schur 数
    while len(sum_free_subsets) < number_of_subsets:
        # 生成下一个 sum-free 子集列表
        sum_free_subsets = _generate_next_list(sum_free_subsets, n)
        # 计算出未包含的元素，加入到最后一个子集中
        missed_elements = [3*k + 1 for k in range(len(sum_free_subsets), (n-1)//3 + 1)]
        sum_free_subsets[-1] += missed_elements

    # 返回计算得到的 sum-free 子集列表
    return sum_free_subsets
# 定义一个函数，生成下一个列表，基于给定的当前列表和上限n
def _generate_next_list(current_list, n):
    # 创建一个新的空列表，用于存放生成的新列表
    new_list = []

    # 遍历当前列表中的每个项
    for item in current_list:
        # 生成一个临时列表temp_1，包含当前项中每个数的三倍，且小于等于n
        temp_1 = [number*3 for number in item if number*3 <= n]
        # 生成一个临时列表temp_2，包含当前项中每个数的三倍减一，且小于等于n
        temp_2 = [number*3 - 1 for number in item if number*3 - 1 <= n]
        # 将temp_1和temp_2合并成新的项，并添加到新列表中
        new_item = temp_1 + temp_2
        new_list.append(new_item)

    # 生成最后一个列表，包含形如3*k + 1的项，直到大于n为止
    last_list = [3*k + 1 for k in range(len(current_list)+1) if 3*k + 1 <= n]
    # 将最后一个列表添加到新列表的末尾
    new_list.append(last_list)
    # 更新当前列表为生成的新列表
    current_list = new_list

    # 返回更新后的当前列表
    return current_list
```