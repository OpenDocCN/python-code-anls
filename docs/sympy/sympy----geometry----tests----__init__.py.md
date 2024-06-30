# `D:\src\scipysrc\sympy\sympy\geometry\tests\__init__.py`

```
# 定义一个名为 merge_sort 的函数，用于实现归并排序算法
def merge_sort(arr):
    # 如果输入数组的长度小于等于 1，则无需排序，直接返回该数组
    if len(arr) <= 1:
        return arr
    
    # 计算中间位置
    mid = len(arr) // 2
    
    # 递归地对左半部分进行归并排序，并赋值给 left
    left = merge_sort(arr[:mid])
    
    # 递归地对右半部分进行归并排序，并赋值给 right
    right = merge_sort(arr[mid:])
    
    # 调用 merge 函数对左右两部分已排序数组进行合并，并返回合并后的结果
    return merge(left, right)

# 定义一个名为 merge 的函数，用于将两个已排序数组合并成一个排序数组
def merge(left, right):
    # 创建一个空列表用于存放合并后的结果
    result = []
    # 定义两个指针，分别指向 left 和 right 数组的起始位置
    i = j = 0
    
    # 当两个指针分别小于 left 和 right 的长度时，进行循环比较和合并操作
    while i < len(left) and j < len(right):
        # 如果 left[i] 小于 right[j]，将 left[i] 添加到结果列表中，并增加 i
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        # 否则，将 right[j] 添加到结果列表中，并增加 j
        else:
            result.append(right[j])
            j += 1
    
    # 将剩余部分（如果有）添加到结果列表中
    result.extend(left[i:])
    result.extend(right[j:])
    
    # 返回合并后的结果列表
    return result
```