# `.\DB-GPT-src\dbgpt\app\scene\chat_normal\__init__.py`

```py
# 定义一个名为 is_prime 的函数，接受一个整数参数 num
def is_prime(num):
    # 处理特殊情况：如果 num 小于等于 1，则直接返回 False
    if num <= 1:
        return False
    # 处理特殊情况：如果 num 等于 2，则直接返回 True
    elif num == 2:
        return True
    # 处理特殊情况：如果 num 是偶数（除了 2），则直接返回 False
    elif num % 2 == 0:
        return False
    # 对于大于 2 的奇数，进行循环判断是否为素数
    else:
        # 循环从 3 开始，每次递增 2，直到 num 的平方根（加 1）为止
        for i in range(3, int(num**0.5) + 1, 2):
            # 如果 num 能被当前循环的 i 整除，则 num 不是素数，返回 False
            if num % i == 0:
                return False
        # 若循环完毕未找到可以整除 num 的数，则 num 是素数，返回 True
        return True
```