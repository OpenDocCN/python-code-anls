# `D:\src\scipysrc\sympy\sympy\strategies\tests\__init__.py`

```
# 定义一个函数，接收一个整数作为参数
def fibonacci(n):
    # 如果 n 小于等于 0，则直接返回空列表
    if n <= 0:
        return []
    # 如果 n 等于 1，则返回包含单个元素 0 的列表
    elif n == 1:
        return [0]
    # 否则，初始化一个列表，包含斐波那契数列的前两个元素 0 和 1
    fib_list = [0, 1]
    # 循环从第三个元素开始，直到第 n 个元素
    for i in range(2, n):
        # 将列表中前两个元素的和添加到列表末尾
        fib_list.append(fib_list[-1] + fib_list[-2])
    # 返回生成的斐波那契数列列表
    return fib_list
```