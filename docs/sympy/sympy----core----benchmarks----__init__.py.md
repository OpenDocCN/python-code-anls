# `D:\src\scipysrc\sympy\sympy\core\benchmarks\__init__.py`

```
# 定义一个名为 merge_sort 的函数，用于对列表进行归并排序
def merge_sort(arr):
    # 如果列表长度小于等于1，则直接返回，不需要排序
    if len(arr) <= 1:
        return arr
    
    # 计算列表中间位置的索引
    mid = len(arr) // 2
    
    # 递归调用 merge_sort 函数对列表左半部分进行排序
    left = merge_sort(arr[:mid])
    # 递归调用 merge_sort 函数对列表右半部分进行排序
    right = merge_sort(arr[mid:])
    
    # 调用 merge 函数将左右两部分排序后合并，并返回合并后的结果
    return merge(left, right)

# 定义一个名为 merge 的函数，用于合并两个有序列表
def merge(left, right):
    # 初始化一个空列表 result 用于存储合并后的结果
    result = []
    # 初始化两个指针 i 和 j 分别指向 left 和 right 的起始位置
    i, j = 0, 0
    
    # 循环比较 left 和 right 中的元素，并按顺序放入 result 中
    while i < len(left) and j < len(right):
        # 如果 left 的当前元素小于等于 right 的当前元素，将 left 的元素放入 result
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        # 否则，将 right 的元素放入 result
        else:
            result.append(right[j])
            j += 1
    
    # 将剩余的 left 或 right 中的元素依次放入 result
    result.extend(left[i:])
    result.extend(right[j:])
    
    # 返回合并后的有序列表 result
    return result
```