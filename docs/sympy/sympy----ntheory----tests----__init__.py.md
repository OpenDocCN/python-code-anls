# `D:\src\scipysrc\sympy\sympy\ntheory\tests\__init__.py`

```
# 定义一个名为 find_duplicates 的函数，接受一个参数 nums，该参数是一个整数列表
def find_duplicates(nums):
    # 创建一个空的集合，用于存储出现过的数字
    seen = set()
    # 创建一个空列表，用于存储重复的数字
    duplicates = []
    
    # 遍历列表 nums 中的每一个数字
    for num in nums:
        # 如果当前数字 num 已经存在于集合 seen 中，说明它是重复的
        if num in seen:
            # 将重复的数字 num 添加到列表 duplicates 中
            duplicates.append(num)
        else:
            # 否则，将当前数字 num 添加到集合 seen 中，表示已经看到过这个数字
            seen.add(num)
    
    # 返回存储重复数字的列表 duplicates
    return duplicates
```