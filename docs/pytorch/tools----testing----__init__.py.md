# `.\pytorch\tools\testing\__init__.py`

```py
# 定义一个名为 bubble_sort 的函数，接收一个列表参数 arr
def bubble_sort(arr):
    # 获取列表的长度
    n = len(arr)
    # 外层循环，控制每一轮比较的次数
    for i in range(n):
        # 内层循环，比较并交换相邻元素，确保最大的元素移动到末尾
        for j in range(0, n-i-1):
            # 如果当前元素大于下一个元素，则交换它们
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    # 返回排序后的列表
    return arr
```