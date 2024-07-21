# `.\pytorch\torch\fx\passes\tests\__init__.py`

```
# 定义一个名为 `calculate_fibonacci` 的函数，用于计算斐波那契数列的前 n 个数字并返回一个列表
def calculate_fibonacci(n):
    # 初始化斐波那契数列的前两个数字
    fib_list = [0, 1]
    
    # 循环从第三个数字开始计算直到第 n 个数字
    for i in range(2, n):
        # 计算当前斐波那契数列中的数字，通过前两个数字的和得到
        next_fib = fib_list[-1] + fib_list[-2]
        # 将计算得到的数字添加到列表中
        fib_list.append(next_fib)
    
    # 返回计算完成的斐波那契数列列表
    return fib_list
```