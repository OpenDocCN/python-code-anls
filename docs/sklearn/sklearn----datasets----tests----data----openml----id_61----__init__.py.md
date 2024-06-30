# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\data\openml\id_61\__init__.py`

```
# 定义一个名为 merge_sort 的函数，用于实现归并排序算法
def merge_sort(arr):
    # 如果数组长度小于等于1，直接返回该数组，不需要排序
    if len(arr) <= 1:
        return arr
    
    # 计算数组的中间位置
    mid = len(arr) // 2
    
    # 递归地对数组的左半部分进行归并排序，并将结果赋给 left
    left = merge_sort(arr[:mid])
    
    # 递归地对数组的右半部分进行归并排序，并将结果赋给 right
    right = merge_sort(arr[mid:])
    
    # 调用 merge 函数，将左右两个已排序的子数组进行合并，并将结果赋给 sorted_arr
    sorted_arr = merge(left, right)
    
    # 返回合并后的已排序数组
    return sorted_arr

# 定义一个名为 merge 的函数，用于将两个已排序的数组合并成一个已排序的数组
def merge(left, right):
    # 创建一个空数组 result 用于存放合并后的结果
    result = []
    # 定义左右两个指针 i 和 j，初始值都为 0
    i = j = 0
    
    # 当左右两个数组的指针 i 和 j 都没有超出其数组长度时，执行循环
    while i < len(left) and j < len(right):
        # 如果左边数组当前元素小于右边数组当前元素，则将左边数组当前元素加入 result 中，并将左边指针 i 后移一位
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        # 否则，将右边数组当前元素加入 result 中，并将右边指针 j 后移一位
        else:
            result.append(right[j])
            j += 1
    
    # 将左边数组剩余部分（如果有）加入 result 中
    result.extend(left[i:])
    # 将右边数组剩余部分（如果有）加入 result 中
    result.extend(right[j:])
    
    # 返回合并后的结果数组 result
    return result
```