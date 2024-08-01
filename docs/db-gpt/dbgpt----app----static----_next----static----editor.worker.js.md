# `.\DB-GPT-src\dbgpt\app\static\_next\static\editor.worker.js`

```py
# 定义一个函数，名称为 `calculate_fibonacci`，接受一个整数参数 `n`
def calculate_fibonacci(n):
    # 如果 `n` 小于等于 0，直接返回 0
    if n <= 0:
        return 0
    # 如果 `n` 等于 1，直接返回 1
    elif n == 1:
        return 1
    else:
        # 递归调用 `calculate_fibonacci` 函数，计算 `n-1` 和 `n-2` 的斐波那契数列值之和
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
```