# `.\MinerU\magic_pdf\libs\local_math.py`

```
# 定义一个比较两个浮点数的函数，判断第一个是否大于第二个
def float_gt(a, b):
    # 检查两个数的差的绝对值是否小于等于 0.0001
    if 0.0001 >= abs(a - b):
        # 如果差值小于等于 0.0001，则认为它们相等，返回 False
        return False
    # 否则，返回第一个数是否大于第二个数
    return a > b
    
# 定义一个比较两个浮点数的函数，判断它们是否相等
def float_equal(a, b):
    # 检查两个数的差的绝对值是否小于等于 0.0001
    if 0.0001 >= abs(a - b):
        # 如果差值小于等于 0.0001，则认为它们相等，返回 True
        return True
    # 否则，返回 False，表示它们不相等
    return False
```