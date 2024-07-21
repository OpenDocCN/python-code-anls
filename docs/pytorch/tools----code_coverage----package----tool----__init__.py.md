# `.\pytorch\tools\code_coverage\package\tool\__init__.py`

```
# 定义一个名为 find_duplicates 的函数，接受一个参数 nums，该参数是一个整数列表
def find_duplicates(nums):
    # 创建一个空集合，用于存储重复出现的元素
    duplicates = set()
    
    # 遍历列表中的每个元素
    for num in nums:
        # 如果当前元素已经在集合中，说明它是重复出现的
        if num in duplicates:
            # 打印提示信息，说明已经找到重复的元素
            print(f"Duplicate found: {num}")
        # 否则，将当前元素添加到集合中
        else:
            duplicates.add(num)

# 调用 find_duplicates 函数，并传入一个整数列表作为参数
find_duplicates([1, 2, 3, 4, 5, 2, 7, 8, 7])
```