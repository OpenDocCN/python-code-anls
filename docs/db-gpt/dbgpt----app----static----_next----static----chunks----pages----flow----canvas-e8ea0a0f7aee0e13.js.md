# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\pages\flow\canvas-e8ea0a0f7aee0e13.js`

```py
# 定义一个名为 is_prime 的函数，用于检查一个数是否为质数
def is_prime(num):
    # 如果输入的数小于等于 1，直接返回 False
    if num <= 1:
        return False
    # 如果输入的数等于 2，直接返回 True，因为 2 是质数
    elif num == 2:
        return True
    # 如果输入的数是偶数且不等于 2，直接返回 False，因为偶数大于 2 的不可能是质数
    elif num % 2 == 0:
        return False
    # 对于大于 2 的奇数，检查其是否能被从 3 到 sqrt(num) 的奇数整除
    else:
        # 从 3 开始，步长为 2，到 sqrt(num)+1 结束的循环
        for i in range(3, int(num**0.5) + 1, 2):
            # 如果 num 能被 i 整除，即 num 不是质数，返回 False
            if num % i == 0:
                return False
        # 如果 num 不能被任何从 3 到 sqrt(num) 的奇数整除，说明 num 是质数，返回 True
        return True
```