# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\data\openml\id_40675\__init__.py`

```
# 定义一个名为 "find_duplicates" 的函数，接收一个列表参数 "nums"
def find_duplicates(nums):
    # 创建一个空集合 "seen" 用来存储已经遍历过的元素
    seen = set()
    # 创建一个空列表 "duplicates" 用来存储找到的重复元素
    duplicates = []
    
    # 遍历列表 "nums" 中的每个元素
    for num in nums:
        # 如果当前元素 "num" 已经存在于集合 "seen" 中
        if num in seen:
            # 将当前元素 "num" 添加到列表 "duplicates" 中
            duplicates.append(num)
        # 否则，将当前元素 "num" 添加到集合 "seen" 中
        else:
            seen.add(num)
    
    # 返回找到的重复元素列表 "duplicates"
    return duplicates
```