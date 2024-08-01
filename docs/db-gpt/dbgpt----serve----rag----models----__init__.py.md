# `.\DB-GPT-src\dbgpt\serve\rag\models\__init__.py`

```py
# 定义一个名为 `find_duplicates` 的函数，接受一个名为 `nums` 的列表参数
def find_duplicates(nums):
    # 创建一个空集合 `seen` 用于存储已经遍历过的数字
    seen = set()
    # 创建一个空列表 `duplicates` 用于存储找到的重复数字
    duplicates = []
    
    # 遍历列表 `nums` 中的每一个数字
    for num in nums:
        # 如果 `num` 已经存在于集合 `seen` 中，说明它是重复出现的数字
        if num in seen:
            # 将重复的数字 `num` 添加到 `duplicates` 列表中
            duplicates.append(num)
        else:
            # 如果 `num` 不在集合 `seen` 中，将其添加到集合 `seen` 中
            seen.add(num)
    
    # 返回找到的所有重复数字的列表 `duplicates`
    return duplicates
```