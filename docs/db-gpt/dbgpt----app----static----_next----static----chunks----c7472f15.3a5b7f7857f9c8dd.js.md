# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\c7472f15.3a5b7f7857f9c8dd.js`

```py
# 定义一个名为 'nums' 的列表，包含整数 1 到 10
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 使用列表推导式创建一个新列表 'squares'，其中包含每个数的平方
squares = [x * x for x in nums]

# 使用 'filter' 函数过滤 'squares' 列表中的偶数，返回一个迭代器
even_squares = filter(lambda x: x % 2 == 0, squares)

# 将过滤后的偶数平方数转换为一个新的列表 'even_squares_list'
even_squares_list = list(even_squares)

# 打印 'even_squares_list'
print(even_squares_list)
```