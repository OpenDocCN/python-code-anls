# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\_partition_nodes.pxd`

```
# 导入 Cython 模块中的 floating 类型
from cython cimport floating
# 导入当前模块的 utils 子模块中的 _typedefs 模块，引入其中的 float64_t 和 intp_t 类型
from ..utils._typedefs cimport float64_t, intp_t

# 定义一个 Cython 的函数，用于在给定数据中进行节点索引的分割
cdef int partition_node_indices(
        # 数据数组，用于存储浮点数类型的数据
        const floating *data,
        # 节点索引数组，存储节点分割后的索引
        intp_t *node_indices,
        # 分割维度索引，指示在哪个维度上进行分割
        intp_t split_dim,
        # 分割点索引，表示在指定维度上的具体分割位置
        intp_t split_index,
        # 特征数，表示数据的特征维度
        intp_t n_features,
        # 点数，表示数据点的数量
        intp_t n_points) except -1:
    # 函数主体部分未提供，功能应该是根据给定条件对节点索引进行分割
```