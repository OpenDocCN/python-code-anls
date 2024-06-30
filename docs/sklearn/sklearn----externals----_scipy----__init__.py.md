# `D:\src\scipysrc\scikit-learn\sklearn\externals\_scipy\__init__.py`

```
# 定义一个名为 find_duplicates 的函数，接受一个参数 nums，这里 nums 是一个列表
def find_duplicates(nums):
    # 创建一个空集合 seen，用来存储已经遇到过的元素
    seen = set()
    # 创建一个空列表 duplicates，用来存储找到的重复元素
    duplicates = []
    
    # 遍历列表 nums 中的每个元素
    for num in nums:
        # 如果当前元素 num 已经在集合 seen 中，说明它是重复出现的
        if num in seen:
            # 将重复元素 num 添加到列表 duplicates 中
            duplicates.append(num)
        # 否则，将当前元素 num 添加到集合 seen 中
        else:
            seen.add(num)
    
    # 返回存储重复元素的列表 duplicates
    return duplicates
```