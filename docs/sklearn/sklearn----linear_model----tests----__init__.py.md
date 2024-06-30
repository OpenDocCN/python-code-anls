# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\__init__.py`

```
# 定义一个名为 'merge_sort' 的函数，用于执行归并排序算法
def merge_sort(arr):
    # 若数组长度小于等于1，无需排序，直接返回数组
    if len(arr) <= 1:
        return arr
    
    # 计算数组中间位置
    mid = len(arr) // 2
    
    # 递归地对数组左半部分进行归并排序，并将结果存储在 'left' 中
    left = merge_sort(arr[:mid])
    
    # 递归地对数组右半部分进行归并排序，并将结果存储在 'right' 中
    right = merge_sort(arr[mid:])
    
    # 合并已排序的左右两个子数组，结果存储在 'result' 中
    result = merge(left, right)
    
    # 返回最终排序结果
    return result

# 定义一个名为 'merge' 的函数，用于合并两个已排序的数组 'left' 和 'right'
def merge(left, right):
    # 创建一个空数组 'result' 用于存储合并后的结果
    result = []
    
    # 初始化左右两个数组的索引变量 'i' 和 'j'，初始值为 0
    i = j = 0
    
    # 循环比较左右两个数组的元素，并将较小的元素添加到 'result' 中
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 将左右数组中剩余的元素依次添加到 'result' 中
    result.extend(left[i:])
    result.extend(right[j:])
    
    # 返回合并后的排序结果数组 'result'
    return result
```