# `.\pytorch\torch\utils\bottleneck\__init__.py`

```py
# 定义一个名为 fibonacci 的生成器函数
def fibonacci():
    a, b = 0, 1  # 初始化两个变量 a 和 b，分别为斐波那契数列的前两个值
    while True:  # 无限循环，直到手动终止
        yield a  # 使用 yield 返回当前斐波那契数列的值
        a, b = b, a + b  # 更新斐波那契数列的值，a 变成原来的 b，b 变成原来的 a+b
```