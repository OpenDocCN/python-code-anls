# `.\DB-GPT-src\dbgpt\core\awel\tests\__init__.py`

```py
# 定义一个名为 merge_sort 的函数，用于实现归并排序算法
def merge_sort(arr):
    # 如果数组长度小于等于1，则不需要排序，直接返回数组本身
    if len(arr) <= 1:
        return arr
    
    # 计算数组的中间位置
    mid = len(arr) // 2
    
    # 递归地对数组的左半部分进行归并排序，得到左半部分排序后的结果
    left = merge_sort(arr[:mid])
    
    # 递归地对数组的右半部分进行归并排序，得到右半部分排序后的结果
    right = merge_sort(arr[mid:])
    
    # 将左半部分和右半部分排序后的结果进行合并，得到最终排序后的结果
    return merge(left, right)

# 定义一个名为 merge 的函数，用于合并两个已排序的数组
def merge(left, right):
    # 创建一个空列表，用于存储合并后的结果
    result = []
    # 定义两个指针，分别指向左右两个数组的起始位置
    i = j = 0
    
    # 循环比较左右两个数组的元素，将较小的元素添加到结果列表中
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 将左边数组剩余的元素添加到结果列表中
    result.extend(left[i:])
    # 将右边数组剩余的元素添加到结果列表中
    result.extend(right[j:])
    
    # 返回最终合并后的结果列表
    return result
```