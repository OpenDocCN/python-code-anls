# `.\benchmark\__init__.py`

```
# 定义一个名为 bubble_sort 的函数，接受一个列表参数 arr
def bubble_sort(arr):
    # 获取列表的长度，用于确定需要比较的次数
    n = len(arr)
    # 外层循环，控制比较的轮数，总共需要比较 n-1 轮
    for i in range(n - 1):
        # 内层循环，每轮比较相邻的元素并交换顺序
        for j in range(0, n - i - 1):
            # 如果前一个元素比后一个元素大，则交换它们的位置
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    # 函数执行完成后，返回排序后的列表
    return arr
```