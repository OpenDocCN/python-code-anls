# `D:\src\scipysrc\sympy\sympy\series\kauers.py`

```
# 定义了一个计算有限差分的函数，接受多项式表达式和变量作为输入，可选增量默认为1
def finite_diff(expression, variable, increment=1):
    # 对表达式进行展开，以便后续的数学操作
    expression = expression.expand()
    # 将变量替换为变量加上增量，并展开得到新的表达式
    expression2 = expression.subs(variable, variable + increment)
    expression2 = expression2.expand()
    # 返回新表达式与原始表达式之间的差，即有限差分结果
    return expression2 - expression

# 定义了一个针对 Sum 实例计算有限差分的函数
def finite_diff_kauers(sum):
    # 获取 Sum 实例中的函数表达式
    function = sum.function
    # 对每个求和限制进行循环，将每个变量替换为上限加一的值
    for l in sum.limits:
        function = function.subs(l[0], l[- 1] + 1)
    # 返回新的函数表达式，即有限差分结果
    return function
```