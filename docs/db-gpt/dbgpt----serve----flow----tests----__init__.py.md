# `.\DB-GPT-src\dbgpt\serve\flow\tests\__init__.py`

```py
# 定义一个名为 find_duplicates 的函数，接收一个列表参数 lst
def find_duplicates(lst):
    # 创建一个空集合 unique 和一个空列表 duplicates
    unique = set()
    duplicates = []
    
    # 遍历列表 lst 中的每个元素 ele
    for ele in lst:
        # 如果元素 ele 已经在集合 unique 中，则将其添加到 duplicates 列表中
        if ele in unique:
            duplicates.append(ele)
        # 否则将元素 ele 添加到集合 unique 中
        else:
            unique.add(ele)
    
    # 返回找到的重复元素列表 duplicates
    return duplicates
```