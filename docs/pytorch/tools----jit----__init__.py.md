# `.\pytorch\tools\jit\__init__.py`

```py
# 定义一个名为 is_prime 的函数，用于检查一个整数是否为素数
def is_prime(num):
    # 素数大于 1
    if num > 1:
        # 检查从 2 到 num-1 的每个整数
        for i in range(2, num):
            # 如果 num 能被任何 i 整除，则 num 不是素数
            if (num % i) == 0:
                return False
        # 如果没有找到能整除 num 的数，则 num 是素数
        return True
    else:
        # 小于或等于 1 的数字不是素数
        return False
```