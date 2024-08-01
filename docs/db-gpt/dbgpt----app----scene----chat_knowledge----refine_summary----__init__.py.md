# `.\DB-GPT-src\dbgpt\app\scene\chat_knowledge\refine_summary\__init__.py`

```py
# 定义一个名为merge_sort的函数，接收一个名为arr的列表参数，用于执行归并排序并返回排序后的列表
def merge_sort(arr):
    # 如果传入的列表长度小于等于1，直接返回该列表，因为已经是有序的
    if len(arr) <= 1:
        return arr
    
    # 找到列表的中间位置
    mid = len(arr) // 2
    
    # 递归调用merge_sort函数对左半部分进行排序
    left_half = merge_sort(arr[:mid])
    # 递归调用merge_sort函数对右半部分进行排序
    right_half = merge_sort(arr[mid:])
    
    # 将排好序的左半部分和右半部分合并
    return merge(left_half, right_half)

# 定义一个名为merge的函数，接收两个名为left和right的列表参数，将两个已排序的列表合并为一个新的有序列表并返回
def merge(left, right):
    # 初始化一个空列表用于存放合并后的结果
    result = []
    # 初始化两个指针，分别指向left和right列表的起始位置
    i = 0
    j = 0
    
    # 当两个指针都没有超过各自列表的长度时，比较它们指向的元素，将较小的元素添加到result列表中，并将对应指针向后移动一位
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 将剩余的部分（如果有）直接添加到result列表的末尾
    result.extend(left[i:])
    result.extend(right[j:])
    
    # 返回合并后的有序列表
    return result
```