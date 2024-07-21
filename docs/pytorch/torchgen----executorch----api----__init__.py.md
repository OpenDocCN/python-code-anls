# `.\pytorch\torchgen\executorch\api\__init__.py`

```py
# 定义一个名为 merge_sort 的函数，用于执行归并排序
def merge_sort(arr):
    # 如果数组的长度小于等于1，则直接返回，不需要排序
    if len(arr) <= 1:
        return arr
    
    # 计算数组的中间位置
    mid = len(arr) // 2
    
    # 递归地对数组的左半部分进行归并排序，得到排序后的左子数组
    left = merge_sort(arr[:mid])
    
    # 递归地对数组的右半部分进行归并排序，得到排序后的右子数组
    right = merge_sort(arr[mid:])
    
    # 将排序后的左右子数组进行归并操作，并返回归并后的结果
    return merge(left, right)

# 定义一个名为 merge 的函数，用于合并两个已排序的数组
def merge(left, right):
    # 创建一个空列表，用于存放合并后的结果
    result = []
    
    # 定义两个指针分别指向两个已排序数组的起始位置
    i, j = 0, 0
    
    # 循环比较两个数组中的元素，将较小的元素依次加入结果列表中
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 将剩余的元素直接加入结果列表中（其中一个数组可能有剩余）
    result.extend(left[i:])
    result.extend(right[j:])
    
    # 返回最终的合并结果
    return result
```