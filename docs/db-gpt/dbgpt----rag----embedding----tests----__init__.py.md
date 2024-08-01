# `.\DB-GPT-src\dbgpt\rag\embedding\tests\__init__.py`

```py
# 定义一个名为 fibonacci 的生成器函数，用于生成斐波那契数列
def fibonacci():
    a, b = 0, 1  # 初始化斐波那契数列的前两个数字
    while True:  # 无限循环，用于生成数列中的下一个数字
        yield a  # 返回当前斐波那契数列的数字
        a, b = b, a + b  # 计算斐波那契数列的下一个数字

# 使用生成器函数 fibonacci() 创建一个生成器对象 fib
fib = fibonacci()

# 打印前 10 个斐波那契数列的数字
for i in range(10):
    print(next(fib))
```