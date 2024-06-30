# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\_validation.py`

```
# 导入 NumPy 库，并使用 np 别名
# 导入稀疏矩阵和相关工具函数
# 导入用于处理稀疏图的工具函数

DTYPE = np.float64  # 设置默认的数据类型为 np.float64

def validate_graph(csgraph, directed, dtype=DTYPE,
                   csr_output=True, dense_output=True,
                   copy_if_dense=False, copy_if_sparse=False,
                   null_value_in=0, null_value_out=np.inf,
                   infinity_null=True, nan_null=True):
    """Routine for validation and conversion of csgraph inputs"""
    # 如果不选择输出为 CSR 或者 dense，则抛出错误
    if not (csr_output or dense_output):
        raise ValueError("Internal: dense or csr output must be true")

    accept_fv = [null_value_in]  # 接受的空值列表包括 null_value_in
    if infinity_null:
        accept_fv.append(np.inf)  # 如果 infinity_null 为真，添加 np.inf 到接受的空值列表
    if nan_null:
        accept_fv.append(np.nan)  # 如果 nan_null 为真，添加 np.nan 到接受的空值列表
    csgraph = convert_pydata_sparse_to_scipy(csgraph, accept_fv=accept_fv)  # 将输入转换为 SciPy 稀疏矩阵

    # 如果是无向图且存储格式为 csc，则原地转置比后续转换为 csr 更快
    if (not directed) and issparse(csgraph) and csgraph.format == "csc":
        csgraph = csgraph.T  # 转置稀疏矩阵

    if issparse(csgraph):  # 如果是稀疏矩阵
        if csr_output:
            csgraph = csr_matrix(csgraph, dtype=DTYPE, copy=copy_if_sparse)  # 转换为 CSR 格式
        else:
            csgraph = csgraph_to_dense(csgraph, null_value=null_value_out)  # 转换为密集矩阵
    elif np.ma.isMaskedArray(csgraph):  # 如果是掩码数组
        if dense_output:
            mask = csgraph.mask
            csgraph = np.array(csgraph.data, dtype=DTYPE, copy=copy_if_dense)  # 转换为密集矩阵
            csgraph[mask] = null_value_out  # 将掩码位置设置为空值
        else:
            csgraph = csgraph_from_masked(csgraph)  # 从掩码数组创建稀疏矩阵
    else:  # 如果是密集矩阵
        if dense_output:
            csgraph = csgraph_masked_from_dense(csgraph,
                                                copy=copy_if_dense,
                                                null_value=null_value_in,
                                                nan_null=nan_null,
                                                infinity_null=infinity_null)  # 从密集矩阵创建稀疏矩阵
            mask = csgraph.mask
            csgraph = np.asarray(csgraph.data, dtype=DTYPE)  # 将数据数组化
            csgraph[mask] = null_value_out  # 将掩码位置设置为空值
        else:
            csgraph = csgraph_from_dense(csgraph, null_value=null_value_in,
                                         infinity_null=infinity_null,
                                         nan_null=nan_null)  # 从密集矩阵创建稀疏矩阵

    if csgraph.ndim != 2:  # 如果稀疏矩阵维度不为 2
        raise ValueError("compressed-sparse graph must be 2-D")

    if csgraph.shape[0] != csgraph.shape[1]:  # 如果稀疏矩阵形状不是 (N, N)
        raise ValueError("compressed-sparse graph must be shape (N, N)")

    return csgraph  # 返回处理后的稀疏矩阵
```