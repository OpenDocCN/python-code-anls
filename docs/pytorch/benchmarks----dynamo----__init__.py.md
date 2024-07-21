# `.\pytorch\benchmarks\dynamo\__init__.py`

```
# 定义一个名为 bubble_sort 的函数，用于对输入的列表进行冒泡排序
def bubble_sort(arr):
    # 获取列表的长度
    n = len(arr)
    # 外层循环控制需要进行 n-1 轮比较和交换操作
    for i in range(n-1):
        # 内层循环控制每一轮比较和交换操作
        for j in range(0, n-i-1):
            # 如果前一个元素大于后一个元素，则交换它们的位置
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    # 返回排序后的列表
    return arr
```