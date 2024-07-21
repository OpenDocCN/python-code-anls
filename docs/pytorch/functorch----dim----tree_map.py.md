# `.\pytorch\functorch\dim\tree_map.py`

```
# 导入 functorch._C 模块中的 dim 对象
from functorch._C import dim

# 将 dim.tree_flatten 赋值给 tree_flatten，用于后续函数调用
tree_flatten = dim.tree_flatten

# 定义函数 tree_map，接受两个参数 fn 和 tree，用于映射操作
def tree_map(fn, tree):
    # 调用 tree_flatten 函数，将 tree 展平为值列表 vs 和反展平函数 unflatten
    vs, unflatten = tree_flatten(tree)
    # 对 vs 中的每个值 v 应用 fn 函数，并使用 unflatten 重新构造成原始结构
    return unflatten(fn(v) for v in vs)
```