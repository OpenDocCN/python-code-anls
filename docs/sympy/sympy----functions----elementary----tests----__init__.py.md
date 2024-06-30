# `D:\src\scipysrc\sympy\sympy\functions\elementary\tests\__init__.py`

```
# 定义一个名为 `divide` 的函数，用于执行两个数的除法运算
def divide(a, b):
    # 如果除数 b 等于 0，则抛出 ZeroDivisionError 异常
    if b == 0:
        raise ZeroDivisionError("除数不能为零")
    # 否则，返回 a 除以 b 的结果
    return a / b
```