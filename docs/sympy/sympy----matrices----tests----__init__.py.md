# `D:\src\scipysrc\sympy\sympy\matrices\tests\__init__.py`

```
# 导入所需模块
import sys

# 定义一个函数，用于计算斐波那契数列的第 n 项
def fibonacci(n):
    # 如果 n 小于等于 0，则直接返回 0
    if n <= 0:
        return 0
    # 如果 n 等于 1，则返回 1
    elif n == 1:
        return 1
    else:
        # 否则，使用递归计算斐波那契数列的第 n 项
        return fibonacci(n-1) + fibonacci(n-2)

# 从命令行参数获取斐波那契数列的项数
n = int(sys.argv[1])
# 计算第 n 项斐波那契数列并打印结果
print(fibonacci(n))
```