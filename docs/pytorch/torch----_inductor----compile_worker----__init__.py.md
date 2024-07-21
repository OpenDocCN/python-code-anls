# `.\pytorch\torch\_inductor\compile_worker\__init__.py`

```
# 定义一个名为calculate_fibonacci的函数，用于计算斐波那契数列
def calculate_fibonacci(n):
    # 如果n小于等于0，则直接返回0
    if n <= 0:
        return 0
    # 如果n等于1，则返回1
    elif n == 1:
        return 1
    else:
        # 使用递归调用来计算斐波那契数列中第n个数字的值
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
```