# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\7287.af836af65cdb5424.js`

```py
# 定义一个名为 get_max 的函数，接受一个参数 nums，表示一个整数列表
def get_max(nums):
    # 如果列表为空，则返回 None
    if not nums:
        return None
    # 将列表中的第一个元素作为初始最大值
    max_num = nums[0]
    # 遍历列表中的每个元素
    for num in nums:
        # 如果当前元素大于当前的最大值
        if num > max_num:
            # 更新最大值为当前元素
            max_num = num
    # 返回最终确定的最大值
    return max_num
```