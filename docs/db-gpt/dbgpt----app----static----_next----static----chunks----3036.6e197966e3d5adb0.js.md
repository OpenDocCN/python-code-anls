# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\3036.6e197966e3d5adb0.js`

```py
# 定义一个名为 `merge_sort` 的函数，实现归并排序算法
def merge_sort(arr):
    # 如果输入数组的长度小于等于1，直接返回该数组，不需要排序
    if len(arr) <= 1:
        return arr
    
    # 计算数组的中间位置
    mid = len(arr) // 2
    
    # 递归地对数组的左半部分进行归并排序，并将结果赋给 left
    left = merge_sort(arr[:mid])
    # 递归地对数组的右半部分进行归并排序，并将结果赋给 right
    right = merge_sort(arr[mid:])
    
    # 调用 merge 函数，将排好序的左右两部分合并起来，并返回合并后的结果
    return merge(left, right)

# 定义一个名为 `merge` 的函数，实现将两个已排序数组合并成一个有序数组
def merge(left, right):
    # 创建一个空数组 `result` 用来存放合并后的结果
    result = []
    # 定义两个指针 `i` 和 `j` 分别指向左右数组的起始位置
    i, j = 0, 0
    
    # 循环比较左右数组中的元素，直到某个数组的元素全部处理完毕
    while i < len(left) and j < len(right):
        # 如果左数组当前位置的元素小于右数组当前位置的元素
        if left[i] < right[j]:
            # 将左数组当前位置的元素加入到 `result` 中，并将左数组指针向右移动一位
            result.append(left[i])
            i += 1
        else:
            # 否则将右数组当前位置的元素加入到 `result` 中，并将右数组指针向右移动一位
            result.append(right[j])
            j += 1
    
    # 将左数组剩余的元素加入到 `result` 中
    result.extend(left[i:])
    # 将右数组剩余的元素加入到 `result` 中
    result.extend(right[j:])
    
    # 返回合并后的有序数组 `result`
    return result
```