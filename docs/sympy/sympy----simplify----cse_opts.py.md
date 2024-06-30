# `D:\src\scipysrc\sympy\sympy\simplify\cse_opts.py`

```
`
""" Optimizations of the expression tree representation for better CSE
opportunities.
"""
# 从 sympy.core 中导入 Add, Basic, Mul 类
from sympy.core import Add, Basic, Mul
# 从 sympy.core.singleton 中导入 S 单例
from sympy.core.singleton import S
# 从 sympy.core.sorting 中导入 default_sort_key 函数
from sympy.core.sorting import default_sort_key
# 从 sympy.core.traversal 中导入 preorder_traversal 函数
from sympy.core.traversal import preorder_traversal


# 定义函数 sub_pre，用于优化表达式树以实现更好的公共子表达式消除 (CSE) 机会
def sub_pre(e):
    """ Replace y - x with -(x - y) if -1 can be extracted from y - x.
    """
    # 查找所有可以提取负号的 Add 类对象
    adds = [a for a in e.atoms(Add) if a.could_extract_minus_sign()]
    # 初始化替换字典和忽略集合
    reps = {}
    ignore = set()
    # 遍历每个可以提取负号的 Add 类对象
    for a in adds:
        # 构造其相反数
        na = -a
        # 如果相反数是一个乘法表达式，例如 MatExpr，则将其加入忽略集合并跳过
        if na.is_Mul:
            ignore.add(a)
            continue
        # 构造替换后的表达式，例如 -(x - y)
        reps[a] = Mul._from_args([S.NegativeOne, na])

    # 使用替换字典替换原始表达式中的对象
    e = e.xreplace(reps)

    # 对于仍然存在的 Add 类对象，重复处理以添加前导的 1 和 -1
    # 例如 y - x -> 1*-1*(x - y)
    if isinstance(e, Basic):
        # 初始化新的替换字典
        negs = {}
        # 按默认排序遍历每个 Add 类对象
        for a in sorted(e.atoms(Add), key=default_sort_key):
            # 如果对象在忽略集合中，则跳过
            if a in ignore:
                continue
            # 如果对象已经在替换字典中，则将其添加到 negs 中
            if a in reps:
                negs[a] = reps[a]
            # 否则，如果对象可以提取负号，则构造替换后的表达式并添加到 negs 中
            elif a.could_extract_minus_sign():
                negs[a] = Mul._from_args([S.One, S.NegativeOne, -a])
        # 使用新的替换字典替换原始表达式中的对象
        e = e.xreplace(negs)
    # 返回优化后的表达式
    return e


# 定义函数 sub_post，用于后处理优化表达式树
def sub_post(e):
    """ Replace 1*-1*x with -x.
    """
    # 初始化替换列表
    replacements = []
    # 使用前序遍历遍历每个节点
```