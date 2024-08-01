# `.\DB-GPT-src\dbgpt\model\cluster\controller\tests\__init__.py`

```py
# 定义一个函数，接受一个整数参数 num
def is_prime(num):
    # 如果 num 小于等于 1，直接返回 False
    if num <= 1:
        return False
    # 对于从 2 到 num-1 的每一个整数 i
    for i in range(2, num):
        # 如果 num 能被 i 整除，说明 num 不是质数，返回 False
        if num % i == 0:
            return False
    # 如果经过上述循环都没有返回 False，说明 num 是质数，返回 True
    return True
```