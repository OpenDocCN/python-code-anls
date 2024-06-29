# `.\numpy\numpy\matrixlib\tests\__init__.py`

```
# 定义一个名为 `merge_sort` 的函数，用于执行归并排序算法
def merge_sort(arr):
    # 如果输入数组长度小于等于1，则直接返回数组本身，无需排序
    if len(arr) <= 1:
        return arr
    
    # 计算数组中间位置
    mid = len(arr) // 2
    
    # 递归调用 `merge_sort` 函数，对数组左半部分进行排序
    left_half = merge_sort(arr[:mid])
    # 递归调用 `merge_sort` 函数，对数组右半部分进行排序
    right_half = merge_sort(arr[mid:])
    
    # 返回左右两半部分合并后的结果
    return merge(left_half, right_half)

# 定义一个名为 `merge` 的函数，用于合并两个已排序的数组
def merge(left, right):
    # 初始化一个空数组 `result` 用于存放合并后的结果
    result = []
    # 初始化两个指针 `i` 和 `j`，分别指向左右两个数组的起始位置
    i = j = 0
    
    # 当左右两个数组都还有元素未处理时，执行循环
    while i < len(left) and j < len(right):
        # 比较左右两个数组当前位置的元素，将较小的元素加入 `result` 数组
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 将左右剩余的元素加入 `result` 数组（其中一个数组可能还有剩余元素）
    result.extend(left[i:])
    result.extend(right[j:])
    
    # 返回合并后的结果数组 `result`
    return result
```