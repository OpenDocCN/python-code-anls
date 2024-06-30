# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\tests\__init__.py`

```
# 定义一个名为 calculate_fibonacci 的函数，接受一个整数参数 n
def calculate_fibonacci(n):
    # 如果 n 小于或等于 0，直接返回空列表
    if n <= 0:
        return []
    # 如果 n 等于 1，返回包含一个元素 0 的列表
    elif n == 1:
        return [0]
    # 否则，初始化一个列表 fib，包含前两个斐波那契数列的值 0 和 1
    else:
        fib = [0, 1]
        # 使用循环计算并追加斐波那契数列的值，直到达到第 n 个数
        for i in range(2, n):
            fib.append(fib[-1] + fib[-2])
        # 返回完整的斐波那契数列列表
        return fib
```