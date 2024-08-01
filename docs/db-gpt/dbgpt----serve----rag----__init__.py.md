# `.\DB-GPT-src\dbgpt\serve\rag\__init__.py`

```py
# 定义一个名为 find_duplicates 的函数，接受一个参数 nums，该参数是一个列表
def find_duplicates(nums):
    # 创建一个空集合 seen，用于存储已经遍历过的数字
    seen = set()
    # 创建一个空列表 result，用于存储重复出现的数字
    result = []
    
    # 遍历列表 nums 中的每一个数字 num
    for num in nums:
        # 如果 num 已经存在于集合 seen 中，说明 num 是重复出现的数字
        if num in seen:
            # 将 num 添加到结果列表 result 中
            result.append(num)
        # 否则，将 num 添加到集合 seen 中
        else:
            seen.add(num)
    
    # 返回结果列表 result，其中包含了所有重复出现的数字
    return result
```