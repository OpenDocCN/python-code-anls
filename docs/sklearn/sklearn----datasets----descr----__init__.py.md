# `D:\src\scipysrc\scikit-learn\sklearn\datasets\descr\__init__.py`

```
# 定义一个名为 bubble_sort 的函数，接受一个列表参数 lst，用于进行冒泡排序
def bubble_sort(lst):
    # 获取列表的长度，决定需要进行多少次排序操作
    n = len(lst)
    # 外层循环，控制排序的轮数
    for i in range(n):
        # 内层循环，对相邻的元素进行比较和交换，将较大的元素逐步移动到列表的末尾
        for j in range(0, n-i-1):
            # 如果前面的元素大于后面的元素，则交换它们的位置
            if lst[j] > lst[j+1]:
                lst[j], lst[j+1] = lst[j+1], lst[j]
    # 返回排序后的列表
    return lst
```