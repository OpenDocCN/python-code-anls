# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\__init__.py`

```
# 定义一个名为 bubble_sort 的函数，用于实现冒泡排序算法
def bubble_sort(arr):
    # 获取数组的长度
    n = len(arr)
    # 外层循环控制每一轮的比较次数
    for i in range(n):
        # 内层循环用于比较相邻元素并进行交换，确保每轮结束最大元素位于正确位置
        for j in range(0, n-i-1):
            # 如果前面的元素大于后面的元素，则交换它们
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    # 返回排序后的数组
    return arr
```