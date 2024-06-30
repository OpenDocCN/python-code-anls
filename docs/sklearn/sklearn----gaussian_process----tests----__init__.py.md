# `D:\src\scipysrc\scikit-learn\sklearn\gaussian_process\tests\__init__.py`

```
# 定义一个名为 search 的函数，接受两个参数：lst 为列表，item 为要查找的元素
def search(lst, item):
    # 使用一个 for 循环遍历列表 lst 中的每一个元素
    for i in range(len(lst)):
        # 如果当前元素等于要查找的 item
        if lst[i] == item:
            # 返回当前元素的索引 i
            return i
    # 如果未找到要查找的 item，则返回 -1 表示未找到
    return -1
```