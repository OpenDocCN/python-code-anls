# `D:\src\scipysrc\sympy\sympy\codegen\tests\__init__.py`

```
# 定义一个名为 bubble_sort 的函数，接受一个列表参数 arr
def bubble_sort(arr):
    # 获取列表的长度
    n = len(arr)
    # 外层循环，遍历整个列表长度减一次
    for i in range(n-1):
        # 内层循环，遍历当前未排序部分的元素
        for j in range(0, n-i-1):
            # 如果当前元素大于下一个元素
            if arr[j] > arr[j+1]:
                # 交换它们的位置
                arr[j], arr[j+1] = arr[j+1], arr[j]
    # 返回排序后的列表
    return arr
```