# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\_gradient_boosting.pyx`

```
# 作者：Nicolas Hug

# 导入并行处理的prange函数
from cython.parallel import prange
# 导入NumPy库，并使用np作为别名
import numpy as np
# 从当前目录的common模块中导入Y_DTYPE常量
from .common import Y_DTYPE
# 从当前目录的common模块中的C扩展中导入Y_DTYPE_C类型
from .common cimport Y_DTYPE_C

# 定义函数 _update_raw_predictions，用于更新raw_predictions数组
def _update_raw_predictions(
        Y_DTYPE_C [::1] raw_predictions,  # OUT，raw_predictions的输出数组类型为Y_DTYPE_C
        grower,  # grower对象，用于生成决策树
        n_threads,  # 指定的线程数
):
    """Update raw_predictions with the predictions of the newest tree.

    This is equivalent to (and much faster than):
        raw_predictions += last_estimator.predict(X_train)

    It's only possible for data X_train that is used to train the trees (it
    isn't usable for e.g. X_val).
    """
    # 定义Cython变量
    cdef:
        unsigned int [::1] starts  # 每个叶子在分区中的起始位置
        unsigned int [::1] stops  # 每个叶子在分区中的结束位置
        Y_DTYPE_C [::1] values  # 每个叶子的值
        const unsigned int [::1] partition = grower.splitter.partition  # 分区数组
        list leaves  # 叶子节点列表

    # 获取决策树的最终叶子节点
    leaves = grower.finalized_leaves
    # 获取叶子节点的起始位置数组
    starts = np.array([leaf.partition_start for leaf in leaves],
                      dtype=np.uint32)
    # 获取叶子节点的结束位置数组
    stops = np.array([leaf.partition_stop for leaf in leaves],
                     dtype=np.uint32)
    # 获取叶子节点的值数组
    values = np.array([leaf.value for leaf in leaves], dtype=Y_DTYPE)

    # 调用Cython中的辅助函数，更新raw_predictions数组
    _update_raw_predictions_helper(raw_predictions, starts, stops, partition,
                                   values, n_threads)


# 定义Cython中的内联函数 _update_raw_predictions_helper，用于辅助更新raw_predictions数组
cdef inline void _update_raw_predictions_helper(
        Y_DTYPE_C [::1] raw_predictions,  # OUT，raw_predictions的输出数组类型为Y_DTYPE_C
        const unsigned int [::1] starts,  # 叶子节点起始位置数组
        const unsigned int [::1] stops,  # 叶子节点结束位置数组
        const unsigned int [::1] partition,  # 分区数组
        const Y_DTYPE_C [::1] values,  # 叶子节点的值数组
        int n_threads,  # 指定的线程数
):

    # 定义Cython变量
    cdef:
        unsigned int position  # 当前位置
        int leaf_idx  # 叶子节点索引
        int n_leaves = starts.shape[0]  # 叶子节点的数量

    # 使用prange函数并行遍历叶子节点
    for leaf_idx in prange(n_leaves, schedule='static', nogil=True,
                           num_threads=n_threads):
        # 遍历当前叶子节点的位置范围
        for position in range(starts[leaf_idx], stops[leaf_idx]):
            # 更新raw_predictions数组中对应位置的值
            raw_predictions[partition[position]] += values[leaf_idx]
```