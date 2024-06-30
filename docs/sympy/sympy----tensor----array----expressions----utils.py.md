# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\utils.py`

```
import bisect  # 导入 bisect 模块，用于处理有序列表的插入和搜索
from collections import defaultdict  # 导入 defaultdict 类，用于创建默认值为字典的字典容器

from sympy.combinatorics import Permutation  # 导入 Permutation 类，用于处理置换操作
from sympy.core.containers import Tuple  # 导入 Tuple 类，用于处理元组操作
from sympy.core.numbers import Integer  # 导入 Integer 类，用于处理整数操作


def _get_mapping_from_subranks(subranks):
    mapping = {}  # 创建空字典 mapping，用于存储映射关系
    counter = 0  # 初始化计数器 counter 为 0
    for i, rank in enumerate(subranks):
        for j in range(rank):
            mapping[counter] = (i, j)  # 将 counter 映射到元组 (i, j)
            counter += 1  # 计数器递增
    return mapping  # 返回构建的映射字典


def _get_contraction_links(args, subranks, *contraction_indices):
    mapping = _get_mapping_from_subranks(subranks)  # 调用 _get_mapping_from_subranks 函数获取映射关系
    contraction_tuples = [[mapping[j] for j in i] for i in contraction_indices]  # 构建收缩索引的元组列表
    dlinks = defaultdict(dict)  # 创建嵌套字典 defaultdict，用于存储对应关系

    for links in contraction_tuples:
        if len(links) == 2:
            (arg1, pos1), (arg2, pos2) = links  # 解包收缩索引对
            dlinks[arg1][pos1] = (arg2, pos2)  # 在 dlinks 中建立反向连接
            dlinks[arg2][pos2] = (arg1, pos1)  # 在 dlinks 中建立反向连接

    return args, dict(dlinks)  # 返回参数和构建的对应关系字典


def _sort_contraction_indices(pairing_indices):
    pairing_indices = [Tuple(*sorted(i)) for i in pairing_indices]  # 对每个收缩索引进行排序并封装为 Tuple
    pairing_indices.sort(key=lambda x: min(x))  # 根据每个 Tuple 中最小的值进行排序
    return pairing_indices  # 返回排序后的收缩索引列表


def _get_diagonal_indices(flattened_indices):
    axes_contraction = defaultdict(list)  # 创建嵌套字典 defaultdict，用于存储坐标收缩信息

    for i, ind in enumerate(flattened_indices):
        if isinstance(ind, (int, Integer)):
            # 如果索引是整数，则不能进行对角化操作，直接跳过
            continue
        axes_contraction[ind].append(i)  # 将索引添加到对应的坐标收缩列表中

    axes_contraction = {k: v for k, v in axes_contraction.items() if len(v) > 1}  # 只保留存在多于一个索引的收缩
    ret_indices = [i for i in flattened_indices if i not in axes_contraction]  # 从扁平化索引中排除对角索引
    diag_indices = list(axes_contraction)  # 提取对角索引列表
    diag_indices.sort(key=lambda x: flattened_indices.index(x))  # 按照在原列表中的顺序排序对角索引
    diagonal_indices = [tuple(axes_contraction[i]) for i in diag_indices]  # 构建对角索引的元组列表
    ret_indices += diag_indices  # 将对角索引添加到结果索引列表末尾
    ret_indices = tuple(ret_indices)  # 将结果索引列表转换为元组
    return diagonal_indices, ret_indices  # 返回对角索引列表和结果索引元组


def _get_argindex(subindices, ind):
    for i, sind in enumerate(subindices):
        if ind == sind:
            return i  # 如果找到索引，返回其位置
        if isinstance(sind, (set, frozenset)) and ind in sind:
            return i  # 如果索引在集合中，返回其位置
    raise IndexError("%s not found in %s" % (ind, subindices))  # 如果索引不存在于 subindices 中，抛出异常


def _apply_recursively_over_nested_lists(func, arr):
    if isinstance(arr, (tuple, list, Tuple)):  # 如果 arr 是元组、列表或 Tuple 类型
        return tuple(_apply_recursively_over_nested_lists(func, i) for i in arr)  # 递归地应用函数到每个元素
    elif isinstance(arr, Tuple):  # 如果 arr 是 Tuple 类型
        return Tuple.fromiter(_apply_recursively_over_nested_lists(func, i) for i in arr)  # 从迭代器构建 Tuple
    else:
        return func(arr)  # 对于其他类型的数据，直接应用函数


def _build_push_indices_up_func_transformation(flattened_contraction_indices):
    shifts = {0: 0}  # 创建初始的偏移量字典，起始位置为 0
    i = 0  # 初始化索引 i 为 0
    cumulative = 0  # 初始化累积计数为 0

    while i < len(flattened_contraction_indices):
        j = 1  # 初始化步长 j 为 1
        while i+j < len(flattened_contraction_indices):
            if flattened_contraction_indices[i] + j != flattened_contraction_indices[i+j]:
                break  # 如果发现不连续的索引，跳出内层循环
            j += 1  # 步长递增
        cumulative += j  # 更新累积计数
        shifts[flattened_contraction_indices[i]] = cumulative  # 记录每个索引的累积偏移量
        i += j  # 更新外层循环索引位置
    # 对 shifts 字典的键进行排序，返回排序后的键列表
    shift_keys = sorted(shifts.keys())

    # 定义函数 func，接受一个索引 idx，返回 shifts 字典中与 idx 最接近且不大于 idx 的键对应的值
    def func(idx):
        return shifts[shift_keys[bisect.bisect_right(shift_keys, idx) - 1]]

    # 定义函数 transform，接受一个整数 j，根据条件返回 j 减去 func(j) 的结果或者 None
    def transform(j):
        # 如果 j 在 flattened_contraction_indices 中，返回 None
        if j in flattened_contraction_indices:
            return None
        else:
            # 否则返回 j 减去 func(j) 的值
            return j - func(j)

    # 返回 transform 函数作为结果
    return transform
# 构建一个函数来创建“向下推送索引”的变换函数，根据给定的扁平化压缩索引。
def _build_push_indices_down_func_transformation(flattened_contraction_indices):
    # 计算 N，即扁平化压缩索引中最大值加上 2
    N = flattened_contraction_indices[-1] + 2
    
    # 计算所有不在扁平化压缩索引中的数字，并存储在 shifts 列表中
    shifts = [i for i in range(N) if i not in flattened_contraction_indices]
    
    # 定义内部函数 transform，用于将给定的索引 j 进行变换
    def transform(j):
        # 如果 j 小于 shifts 列表的长度，则直接返回 shifts[j]
        if j < len(shifts):
            return shifts[j]
        # 否则，返回 j 加上 shifts 列表中最后一个元素再减去 shifts 长度后的结果
        else:
            return j + shifts[-1] - len(shifts) + 1
    
    # 返回内部定义的 transform 函数
    return transform


# 应用给定的置换 perm 到目标列表 target_list 上
def _apply_permutation_to_list(perm: Permutation, target_list: list):
    """
    根据给定的置换 perm 对列表进行重新排列。
    """
    # 创建一个与目标列表同长度的新列表 new_list
    new_list = [None for i in range(perm.size)]
    
    # 遍历目标列表 target_list 的元素及其索引 i
    for i, e in enumerate(target_list):
        # 将目标列表中索引 i 处的元素 e，按照 perm 函数映射到新列表中对应的位置
        new_list[perm(i)] = e
    
    # 返回重新排列后的新列表 new_list
    return new_list
```