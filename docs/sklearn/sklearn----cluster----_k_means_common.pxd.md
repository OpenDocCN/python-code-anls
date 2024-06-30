# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_k_means_common.pxd`

```
# 导入 Cython 模块中的 floating 类型
from cython cimport floating

# 定义 Cython 函数 _euclidean_dense_dense，计算两个密集型数组之间的欧几里德距离
cdef floating _euclidean_dense_dense(
    const floating*,      # 第一个密集型数组的指针
    const floating*,      # 第二个密集型数组的指针
    int,                  # 数组长度
    bint                 # 布尔值，表示是否启用 GIL 线程
) noexcept nogil         # 禁用 GIL 的声明

# 定义 Cython 函数 _euclidean_sparse_dense，计算稀疏数组和密集数组之间的欧几里德距离
cdef floating _euclidean_sparse_dense(
    const floating[::1],  # 稀疏数组的数据
    const int[::1],       # 稀疏数组的索引
    const floating[::1],  # 密集数组的数据
    floating,             # 浮点数常量
    bint                 # 布尔值，表示是否启用 GIL 线程
) noexcept nogil         # 禁用 GIL 的声明

# 定义 Cython 函数 _relocate_empty_clusters_dense，重定位空聚类中心，处理密集型数据
cpdef void _relocate_empty_clusters_dense(
    const floating[:, ::1],  # 二维密集型数组，表示数据点和聚类中心的距离
    const floating[::1],     # 一维密集型数组，表示聚类中心的数据
    const floating[:, ::1],  # 二维密集型数组，表示当前聚类中心的位置
    floating[:, ::1],        # 二维密集型数组，用于更新聚类中心的新位置
    floating[::1],           # 一维密集型数组，表示每个聚类的权重
    const int[::1]           # 一维整型数组，表示每个数据点的聚类分配
)

# 定义 Cython 函数 _relocate_empty_clusters_sparse，重定位空聚类中心，处理稀疏型数据
cpdef void _relocate_empty_clusters_sparse(
    const floating[::1],     # 一维稀疏型数组，表示数据点和聚类中心的距离
    const int[::1],          # 一维整型数组，表示数据点的索引
    const int[::1],          # 一维整型数组，表示聚类中心的索引
    const floating[::1],     # 一维稀疏型数组，表示聚类中心的数据
    const floating[:, ::1],  # 二维密集型数组，表示当前聚类中心的位置
    floating[:, ::1],        # 二维密集型数组，用于更新聚类中心的新位置
    floating[::1],           # 一维密集型数组，表示每个聚类的权重
    const int[::1]           # 一维整型数组，表示每个数据点的聚类分配
)

# 定义 Cython 函数 _average_centers，计算聚类中心的平均值
cdef void _average_centers(
    floating[:, ::1],  # 二维密集型数组，表示聚类中心的位置
    const floating[::1]  # 一维密集型数组，表示每个聚类的权重
)

# 定义 Cython 函数 _center_shift，计算聚类中心的偏移量
cdef void _center_shift(
    const floating[:, ::1],  # 二维密集型数组，表示原始聚类中心的位置
    const floating[:, ::1],  # 二维密集型数组，表示更新后的聚类中心的位置
    floating[::1]            # 一维密集型数组，表示每个聚类的偏移量
)
```