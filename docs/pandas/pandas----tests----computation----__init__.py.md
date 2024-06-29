# `D:\src\scipysrc\pandas\pandas\tests\computation\__init__.py`

```
# 定义一个名为 `merge_sort` 的函数，用于执行归并排序
def merge_sort(arr):
    # 如果输入的数组长度小于等于 1，直接返回该数组，不需要排序
    if len(arr) <= 1:
        return arr
    
    # 计算数组的中间位置
    mid = len(arr) // 2
    
    # 递归调用 `merge_sort` 函数，对数组的左半部分进行排序
    left_half = merge_sort(arr[:mid])
    
    # 递归调用 `merge_sort` 函数，对数组的右半部分进行排序
    right_half = merge_sort(arr[mid:])
    
    # 调用 `merge` 函数，将排好序的左右两部分数组进行合并
    return merge(left_half, right_half)

# 定义一个名为 `merge` 的函数，用于合并两个已排序的数组
def merge(left, right):
    # 创建一个空列表 `result` 用于存储合并后的结果
    result = []
    # 定义 `i` 和 `j` 作为左右数组的索引，初始值都为 0
    i = j = 0
    
    # 当左右数组都还有元素未被合并时，进行循环比较
    while i < len(left) and j < len(right):
        # 如果左数组当前元素小于右数组当前元素
        if left[i] < right[j]:
            # 将左数组当前元素添加到 `result` 中
            result.append(left[i])
            # 左数组索引 `i` 后移一位
            i += 1
        else:
            # 否则，将右数组当前元素添加到 `result` 中
            result.append(right[j])
            # 右数组索引 `j` 后移一位
            j += 1
    
    # 将左数组剩余的元素（如果有）添加到 `result` 中
    result.extend(left[i:])
    # 将右数组剩余的元素（如果有）添加到 `result` 中
    result.extend(right[j:])
    
    # 返回合并后的结果数组 `result`
    return result
```