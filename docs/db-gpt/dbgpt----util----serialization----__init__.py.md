# `.\DB-GPT-src\dbgpt\util\serialization\__init__.py`

```py
# 定义一个名为 create_matrix 的函数，接收参数 rows 和 cols
def create_matrix(rows, cols):
    # 使用列表推导式创建一个二维列表，初始化元素为 0
    matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    # 返回创建的二维列表作为结果
    return matrix
```