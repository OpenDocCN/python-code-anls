# `.\pytorch\torch\_dynamo\backends\__init__.py`

```
# 定义一个名为 find_duplicates 的函数，接受一个参数 nums
def find_duplicates(nums):
    # 创建一个空集合 seen 用来存储已经出现过的数字
    seen = set()
    # 创建一个空列表 result 用来存储找到的重复数字
    result = []
    
    # 遍历 nums 列表中的每一个数字
    for num in nums:
        # 如果当前数字 num 已经在集合 seen 中出现过
        if num in seen:
            # 将该数字 num 添加到结果列表 result 中
            result.append(num)
        # 否则，将当前数字 num 添加到集合 seen 中
        else:
            seen.add(num)
    
    # 返回找到的重复数字列表 result
    return result
```