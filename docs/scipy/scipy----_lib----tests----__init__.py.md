# `D:\src\scipysrc\scipy\scipy\_lib\tests\__init__.py`

```
# 定义一个名为 `bubble_sort` 的函数，用于实现冒泡排序算法
def bubble_sort(arr):
    # 获取数组的长度
    n = len(arr)
    # 外层循环，控制每一轮比较的次数
    for i in range(n):
        # 内层循环，从第一个元素到倒数第二个元素，依次比较相邻两个元素的大小
        for j in range(0, n-i-1):
            # 如果前面的元素大于后面的元素，则交换它们的位置
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    # 排序完成后，返回排序后的数组
    return arr
```