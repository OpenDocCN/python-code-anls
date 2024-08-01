# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\3585.6c90607f461f61d8.js`

```py
# 定义一个名为 find_duplicates 的函数，用于查找列表中的重复项并返回一个包含重复项的集合
def find_duplicates(lst):
    # 创建一个空集合，用于存储找到的重复项
    duplicates = set()
    # 创建一个空集合，用于存储已经遍历过的元素
    seen = set()
    
    # 遍历列表中的每个元素
    for item in lst:
        # 如果元素已经存在于 seen 集合中，则将其添加到 duplicates 集合中
        if item in seen:
            duplicates.add(item)
        # 否则将元素添加到 seen 集合中
        else:
            seen.add(item)
    
    # 返回包含重复项的集合
    return duplicates
```