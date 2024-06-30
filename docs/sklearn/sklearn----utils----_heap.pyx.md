# `D:\src\scipysrc\scikit-learn\sklearn\utils\_heap.pyx`

```
# 导入 Cython 模块中的 floating 类型
from cython cimport floating

# 从 _typedefs 模块中导入 intp_t 类型
from ._typedefs cimport intp_t

# 定义一个内联函数 heap_push，用于向固定大小的最大堆中推入元组 (val, val_idx)
# 函数声明中使用 noexcept 和 nogil 用于指明函数不会抛出异常且不涉及 GIL
cdef inline int heap_push(
    # values 是存储堆数据的数组
    floating* values,
    # indices 是存储每个值索引的数组
    intp_t* indices,
    # 堆的当前大小
    intp_t size,
    # 要推入堆的值
    floating val,
    # 要推入堆的值的索引
    intp_t val_idx,
) noexcept nogil:
    """Push a tuple (val, val_idx) onto a fixed-size max-heap.

    The max-heap is represented as a Structure of Arrays where:
     - values is the array containing the data to construct the heap with
     - indices is the array containing the indices (meta-data) of each value

    Notes
    -----
    Arrays are manipulated via a pointer to there first element and their size
    as to ease the processing of dynamically allocated buffers.

    For instance, in pseudo-code:

        values = [1.2, 0.4, 0.1],
        indices = [42, 1, 5],
        heap_push(
            values=values,
            indices=indices,
            size=3,
            val=0.2,
            val_idx=4,
        )

    will modify values and indices inplace, giving at the end of the call:

        values  == [0.4, 0.2, 0.1]
        indices == [1, 4, 5]

    """
    # 声明局部变量
    cdef:
        intp_t current_idx, left_child_idx, right_child_idx, swap_idx

    # 检查是否需要将 val 推入堆中
    if val >= values[0]:
        return 0

    # 将 val 插入到堆的第一个位置
    values[0] = val
    indices[0] = val_idx

    # 下沉操作，交换值直到满足最大堆的条件
    current_idx = 0
    while True:
        left_child_idx = 2 * current_idx + 1
        right_child_idx = left_child_idx + 1

        # 如果左子节点超出堆的大小，则停止
        if left_child_idx >= size:
            break
        # 如果只有左子节点或者右子节点的值比 val 大，则进行交换
        elif right_child_idx >= size:
            if values[left_child_idx] > val:
                swap_idx = left_child_idx
            else:
                break
        # 如果左右子节点都存在，选择较大的子节点进行交换
        elif values[left_child_idx] >= values[right_child_idx]:
            if val < values[left_child_idx]:
                swap_idx = left_child_idx
            else:
                break
        else:
            if val < values[right_child_idx]:
                swap_idx = right_child_idx
            else:
                break

        # 将当前位置的值与 swap_idx 处的值进行交换
        values[current_idx] = values[swap_idx]
        indices[current_idx] = indices[swap_idx]

        # 更新当前位置为 swap_idx
        current_idx = swap_idx

    # 将 val 和 val_idx 放置到最终位置
    values[current_idx] = val
    indices[current_idx] = val_idx

    # 返回操作成功
    return 0
```