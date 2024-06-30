# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\data\openml\id_62\__init__.py`

```
# 定义一个名为 `bubble_sort` 的函数，用于实现冒泡排序算法
def bubble_sort(arr):
    # 获取数组的长度
    n = len(arr)
    # 外层循环，控制每次冒泡的范围
    for i in range(n):
        # 内层循环，执行一轮冒泡操作
        for j in range(0, n-i-1):
            # 如果相邻的两个元素顺序错误，就交换它们
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    # 排序完成后返回排序后的数组
    return arr
```