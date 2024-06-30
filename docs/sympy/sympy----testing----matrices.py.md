# `D:\src\scipysrc\sympy\sympy\testing\matrices.py`

```
# 定义一个函数用于比较两个可迭代对象 A 和 B 是否全部接近
def allclose(A, B, rtol=1e-05, atol=1e-08):
    # 如果 A 和 B 的长度不相等，则它们不可能全部接近，直接返回 False
    if len(A) != len(B):
        return False

    # 使用 zip 函数同时迭代 A 和 B 中的元素 x 和 y
    for x, y in zip(A, B):
        # 计算 x 和 y 的绝对误差
        abs_diff = abs(x - y)
        # 计算容忍误差，包括绝对误差和相对误差的影响
        tolerance = atol + rtol * max(abs(x), abs(y))
        # 如果 x 和 y 的绝对误差超过容忍误差，则它们不接近，返回 False
        if abs_diff > tolerance:
            return False
    
    # 如果所有元素对比都通过，则返回 True，表示所有元素均接近
    return True
```