# `.\pytorch\tools\pyi\__init__.py`

```
# 定义一个名为 square 的函数，接受一个参数 num
def square(num):
    # 返回 num 的平方
    return num ** 2

# 定义一个名为 nums 的列表，包含整数 1 到 5
nums = [1, 2, 3, 4, 5]

# 使用列表推导式对 nums 中的每个元素进行平方操作，生成一个新列表 squares
squares = [square(n) for n in nums]

# 打印新生成的 squares 列表
print(squares)
```