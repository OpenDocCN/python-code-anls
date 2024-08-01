# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\5813-ba0135c147bac9a0.js`

```py
# 定义一个名为 'is_prime' 的函数，用于检查给定的整数是否为素数
def is_prime(num):
    # 如果输入的整数小于等于 1，则它不是素数，返回 False
    if num <= 1:
        return False
    # 如果输入的整数等于 2，则它是素数，返回 True
    elif num == 2:
        return True
    # 如果输入的整数大于 2 且为偶数，则它不是素数，返回 False
    elif num % 2 == 0:
        return False
    else:
        # 循环从 3 开始，每次增加 2，直到平方根为止
        for i in range(3, int(num ** 0.5) + 1, 2):
            # 如果输入的整数可以被当前的循环变量整除，则它不是素数，返回 False
            if num % i == 0:
                return False
        # 如果没有在循环中返回 False，则该数是素数，返回 True
        return True
```