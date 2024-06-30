# `D:\src\scipysrc\seaborn\seaborn\_core\__init__.py`

```
# 定义一个名为 find_duplicates 的函数，接受一个列表参数 lst
def find_duplicates(lst):
    # 创建一个空集合，用于存储重复项
    seen = set()
    # 创建一个空列表，用于存储找到的重复项
    dupes = []
    
    # 遍历列表 lst 中的每一个元素 e
    for e in lst:
        # 如果 e 已经存在于集合 seen 中，说明 e 是重复项
        if e in seen:
            # 将 e 添加到重复项列表 dupes 中
            dupes.append(e)
        else:
            # 否则将 e 添加到集合 seen 中，表示第一次见到 e
            seen.add(e)
    
    # 返回找到的所有重复项列表 dupes
    return dupes
```