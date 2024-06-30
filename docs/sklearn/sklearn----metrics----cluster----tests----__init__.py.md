# `D:\src\scipysrc\scikit-learn\sklearn\metrics\cluster\tests\__init__.py`

```
# 定义一个名为merge_sort的函数，用于对传入的列表进行归并排序
def merge_sort(arr):
    # 检查列表长度是否小于等于1，如果是，则直接返回该列表，因为无需排序
    if len(arr) <= 1:
        return arr
    
    # 计算列表中间位置的索引
    mid = len(arr) // 2
    
    # 递归调用merge_sort函数，对列表左半部分进行排序
    left = merge_sort(arr[:mid])
    # 递归调用merge_sort函数，对列表右半部分进行排序
    right = merge_sort(arr[mid:])
    
    # 调用merge函数，将排好序的左右两部分列表合并成一个有序的列表，并返回结果
    return merge(left, right)

# 定义一个名为merge的函数，用于将两个有序列表合并成一个有序列表
def merge(left, right):
    # 创建一个空列表，用于存放合并后的结果
    result = []
    # 初始化左右两个列表的起始索引
    i = 0
    j = 0
    
    # 比较左右两个列表的元素，将较小的元素添加到结果列表中，直到其中一个列表为空
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 将剩余的元素直接添加到结果列表的末尾
    result.extend(left[i:])
    result.extend(right[j:])
    
    # 返回合并后的有序列表
    return result
```