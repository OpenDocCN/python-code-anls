# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\common.pxd`

```
# 从utils._typedefs模块中导入特定类型的C语言类型定义
from ...utils._typedefs cimport float32_t, float64_t, intp_t, uint8_t, uint32_t

# 定义C语言类型别名，对应Python中的数据类型
ctypedef float64_t X_DTYPE_C
ctypedef uint8_t X_BINNED_DTYPE_C
ctypedef float64_t Y_DTYPE_C
ctypedef float32_t G_H_DTYPE_C
ctypedef uint32_t BITSET_INNER_DTYPE_C
ctypedef BITSET_INNER_DTYPE_C[8] BITSET_DTYPE_C

# 定义C语言中的结构体hist_struct，用于处理直方图的视图声明
cdef packed struct hist_struct:
    # 总梯度和总Hessian值，与直方图的数据类型相同，需要保证字节对齐
    Y_DTYPE_C sum_gradients
    Y_DTYPE_C sum_hessians
    # 计数器，用于记录直方图中的条目数量
    unsigned int count

# 定义C语言中的结构体node_struct，与PREDICTOR_RECORD_DTYPE等价，用于内存视图
cdef packed struct node_struct:
    # 节点的预测值，计数器和特征索引
    Y_DTYPE_C value
    unsigned int count
    intp_t feature_idx
    X_DTYPE_C num_threshold
    # 表示遗漏值是否进入左侧子节点的标志
    unsigned char missing_go_to_left
    # 左右子节点的索引
    unsigned int left
    unsigned int right
    # 节点的增益和深度
    Y_DTYPE_C gain
    unsigned int depth
    # 表示是否为叶子节点的标志
    unsigned char is_leaf
    X_BINNED_DTYPE_C bin_threshold
    # 表示位集在预测器的位集数组中的索引，仅当is_categorical为True时使用
    unsigned int bitset_idx

# 定义C语言中的枚举MonotonicConstraint，表示单调性约束
cpdef enum MonotonicConstraint:
    NO_CST = 0
    POS = 1
    NEG = -1
```