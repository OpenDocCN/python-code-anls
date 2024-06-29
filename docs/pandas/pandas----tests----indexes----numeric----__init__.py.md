# `D:\src\scipysrc\pandas\pandas\tests\indexes\numeric\__init__.py`

```
# 定义一个名为bubble_sort的函数，用于对传入的列表进行冒泡排序
def bubble_sort(arr):
    # 获取列表的长度
    n = len(arr)
    # 外部循环，控制比较轮数
    for i in range(n):
        # 内部循环，逐个比较并交换相邻元素，确保当前轮次最大元素移动到最后
        for j in range(0, n-i-1):
            # 如果相邻元素顺序错误，进行交换
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    # 返回排序后的列表
    return arr
```