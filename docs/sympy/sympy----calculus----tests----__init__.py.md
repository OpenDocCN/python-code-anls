# `D:\src\scipysrc\sympy\sympy\calculus\tests\__init__.py`

```
# 定义一个名为 ultimate 的函数，接受一个参数 x
def ultimate(x):
    # 如果 x 是奇数
    if x % 2 == 1:
        # 返回 x 的平方
        return x * x
    # 否则，如果 x 是偶数
    elif x % 2 == 0:
        # 返回 x 除以 2 的结果
        return x / 2
    # 如果 x 不是整数
    else:
        # 返回 x 的三倍加一
        return 3 * x + 1
```