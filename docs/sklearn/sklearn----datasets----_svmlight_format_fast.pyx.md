# `D:\src\scipysrc\scikit-learn\sklearn\datasets\_svmlight_format_fast.pyx`

```
    # 导入必要的库和模块
    import array  # 导入标准库中的array模块
    from cpython cimport array  # 从cpython中导入array模块，用于Cython的特定优化
    cimport cython  # 导入Cython的cimport，用于Cython的编译指令
    from libc.string cimport strchr  # 从libc中的string模块导入strchr函数，用于Cython扩展中的字符串操作

    # 导入numpy，并将其重命名为np，以便在代码中使用
    import numpy as np


    cdef bytes COMMA = u','.encode('ascii')  # 定义一个字节串，用于表示ASCII编码的逗号
    cdef bytes COLON = u':'.encode('ascii')  # 定义一个字节串，用于表示ASCII编码的冒号


    def _load_svmlight_file(f, dtype, bint multilabel, bint zero_based,
                            bint query_id, long long offset, long long length):
        # 声明Cython变量
        cdef array.array data, indices, indptr
        cdef bytes line  # 声明一个字节串变量，用于存储读取的每一行数据
        cdef char *hash_ptr  # 声明一个指向字符的指针，用于处理字符操作
        cdef char *line_cstr  # 声明一个指向字符的指针，用于处理C字符串
        cdef int idx, prev_idx  # 声明整型变量idx和prev_idx，用于索引操作
        cdef Py_ssize_t i  # 声明Py_ssize_t类型的变量i，用于表示Python对象的大小
        cdef bytes qid_prefix = b'qid'  # 声明一个字节串变量，存储表示查询ID的前缀
        cdef Py_ssize_t n_features  # 声明Py_ssize_t类型的变量n_features，用于表示特征数目
        cdef long long offset_max = offset + length if length > 0 else -1  # 计算最大偏移量

        # 如果dtype为np.float32，则使用单精度浮点数类型创建数组data；否则使用双精度浮点数类型
        if dtype == np.float32:
            data = array.array("f")
        else:
            dtype = np.float64
            data = array.array("d")

        indices = array.array("q")  # 创建一个数组indices，用于存储长整型数据
        indptr = array.array("q", [0])  # 创建一个数组indptr，初始化为包含单个0的长整型数组
        query = np.arange(0, dtype=np.int64)  # 创建一个包含0到n-1整数的numpy数组，数据类型为64位整数

        # 如果multilabel为True，则创建一个空列表labels；否则创建一个双精度浮点数数组labels
        if multilabel:
            labels = []
        else:
            labels = array.array("d")

        if offset > 0:
            f.seek(offset)  # 定位文件对象到指定的偏移量
            # 跳过当前行，因为它可能被截断，并将由另一个调用获取
            f.readline()
    for line in f:
        # 逐行读取文件 f
        line_cstr = line
        # 将当前行赋给 line_cstr
        hash_ptr = strchr(line_cstr, 35)  # ASCII 值为 '#' 的字符的位置
        # 在 line_cstr 中查找 '#' 的位置，返回指针 hash_ptr
        if hash_ptr != NULL:
            # 如果找到了 '#' 字符
            line = line[:hash_ptr - line_cstr]
            # 截取行，去掉 '#' 之后的内容（注释部分）

        line_parts = line.split()
        # 将当前行按空格分割，得到列表 line_parts
        if len(line_parts) == 0:
            # 如果分割后的列表为空，则跳过当前循环
            continue

        target, features = line_parts[0], line_parts[1:]
        # 将第一个元素作为 target，其余元素作为 features

        if multilabel:
            # 如果是多标签情况
            if COLON in target:
                # 如果 target 包含冒号
                target, features = [], line_parts[0:]
                # 将 target 置为空列表，features 使用整个 line_parts
            else:
                target = [float(y) for y in target.split(COMMA)]
                # 否则，将 target 按逗号分割成浮点数列表
            target.sort()
            # 对 target 列表排序
            labels.append(tuple(target))
            # 将排序后的 target 添加为元组，并加入 labels 列表中
        else:
            array.resize_smart(labels, len(labels) + 1)
            # 调整 labels 数组大小，增加一个元素位置
            labels[len(labels) - 1] = float(target)
            # 在最后一个位置上添加 float 类型的 target

        prev_idx = -1
        # 初始化前一个索引为 -1
        n_features = len(features)
        # 计算 features 的长度
        if n_features and features[0].startswith(qid_prefix):
            # 如果 features 不为空且第一个元素以 qid_prefix 开头
            _, value = features[0].split(COLON, 1)
            # 分割第一个元素，获取值部分
            if query_id:
                query.resize(len(query) + 1)
                # 调整 query 的大小，增加一个元素位置
                query[len(query) - 1] = np.int64(value)
                # 在最后一个位置上添加 np.int64 类型的 value
            features.pop(0)
            # 弹出 features 的第一个元素
            n_features -= 1

        for i in range(0, n_features):
            idx_s, value = features[i].split(COLON, 1)
            # 分割 features 中的每个元素，获取索引和值
            idx = int(idx_s)
            # 将索引转换为整数类型
            if idx < 0 or not zero_based and idx == 0:
                # 如果索引小于 0 或者不是从零开始且索引为 0
                raise ValueError(
                    "Invalid index %d in SVMlight/LibSVM data file." % idx)
                # 抛出数值错误，索引无效
            if idx <= prev_idx:
                # 如果当前索引小于等于前一个索引
                raise ValueError("Feature indices in SVMlight/LibSVM data "
                                 "file should be sorted and unique.")
                # 抛出数值错误，特征索引应该是排序且唯一的

            array.resize_smart(indices, len(indices) + 1)
            # 调整 indices 数组大小，增加一个元素位置
            indices[len(indices) - 1] = idx
            # 在最后一个位置上添加 idx

            array.resize_smart(data, len(data) + 1)
            # 调整 data 数组大小，增加一个元素位置
            data[len(data) - 1] = float(value)
            # 在最后一个位置上添加 float 类型的 value

            prev_idx = idx
            # 更新 prev_idx 为当前 idx

        # 增加索引指针数组的大小
        array.resize_smart(indptr, len(indptr) + 1)
        # 调整 indptr 数组大小，增加一个元素位置
        indptr[len(indptr) - 1] = len(data)
        # 在最后一个位置上添加 data 数组的长度

        if offset_max != -1 and f.tell() > offset_max:
            # 如果 offset_max 不为 -1 并且当前文件指针位置大于 offset_max
            # 停止并让另一个调用处理后续内容
            break

    return (dtype, data, indices, indptr, labels, query)
    # 返回元组包含 dtype、data、indices、indptr、labels、query
# 定义一个融合类型，可以使用所有可能的参数组合。
ctypedef fused int_or_float:
    cython.integral  # 整数类型
    cython.floating  # 浮点数类型
    signed long long  # 有符号长长整型

# 定义一个融合类型，包含双精度浮点数和有符号长长整型。
ctypedef fused double_or_longlong:
    double  # 双精度浮点数
    signed long long  # 有符号长长整型

# 定义一个融合类型，可以使用整数或有符号长长整型。
ctypedef fused int_or_longlong:
    cython.integral  # 整数类型
    signed long long  # 有符号长长整型


# 定义函数get_dense_row_string，接受稠密矩阵和相关参数，返回格式化后的稠密行字符串。
def get_dense_row_string(
    const int_or_float[:, :] X,  # 输入的稠密矩阵 X，类型为 int_or_float
    Py_ssize_t[:] x_inds,  # 稀疏索引数组 x_inds，类型为 Py_ssize_t
    double_or_longlong[:] x_vals,  # 稀疏值数组 x_vals，类型为 double_or_longlong
    Py_ssize_t row,  # 要处理的行索引
    str value_pattern,  # 值的格式化字符串模式
    bint one_based,  # 布尔值，指示是否基于 1 的索引
):
    cdef:
        Py_ssize_t row_length = X.shape[1]  # 行长度为 X 的列数
        Py_ssize_t x_nz_used = 0  # 非零值计数器
        Py_ssize_t k  # 循环变量
        int_or_float val  # 存储当前值的变量

    # 遍历行中的每个元素
    for k in range(row_length):
        val = X[row, k]  # 获取当前元素的值
        if val == 0:  # 如果值为零，则跳过
            continue
        x_inds[x_nz_used] = k  # 将非零元素的列索引存入 x_inds
        x_vals[x_nz_used] = <double_or_longlong> val  # 将值转换为 double_or_longlong 类型存入 x_vals
        x_nz_used += 1  # 非零值计数器加一

    # 构建格式化后的稠密行字符串列表
    reprs = [
        value_pattern % (x_inds[i] + one_based, x_vals[i])
        for i in range(x_nz_used)
    ]

    return " ".join(reprs)  # 返回用空格连接的字符串


# 定义函数get_sparse_row_string，接受稀疏矩阵和相关参数，返回格式化后的稀疏行字符串。
def get_sparse_row_string(
    int_or_float[:] X_data,  # 输入的稀疏数据数组 X_data，类型为 int_or_float
    int[:] X_indptr,  # 指示行开始和结束的索引指针数组 X_indptr，类型为 int
    int[:] X_indices,  # 列索引数组 X_indices，类型为 int
    Py_ssize_t row,  # 要处理的行索引
    str value_pattern,  # 值的格式化字符串模式
    bint one_based,  # 布尔值，指示是否基于 1 的索引
):
    cdef:
        Py_ssize_t row_start = X_indptr[row]  # 当前行在 X_data 和 X_indices 中的起始位置
        Py_ssize_t row_end = X_indptr[row + 1]  # 当前行在 X_data 和 X_indices 中的结束位置

    # 构建格式化后的稀疏行字符串列表
    reprs = [
        value_pattern % (X_indices[i] + one_based, X_data[i])
        for i in range(row_start, row_end)
    ]

    return " ".join(reprs)  # 返回用空格连接的字符串


# 定义函数_dump_svmlight_file，实现将数据写入 SVMLight 文件格式。
def _dump_svmlight_file(
    X,  # 输入数据矩阵 X
    y,  # 标签数组 y
    f,  # 文件对象 f，用于写入数据
    bint multilabel,  # 布尔值，指示是否为多标签分类
    bint one_based,  # 布尔值，指示是否基于 1 的索引
    int_or_longlong[:] query_id,  # 查询 ID 数组 query_id，类型为 int_or_longlong
    bint X_is_sp,  # 布尔值，指示输入数据 X 是否为稀疏矩阵
    bint y_is_sp,  # 布尔值，指示标签数组 y 是否为稀疏数组
):
    cdef bint X_is_integral  # 布尔值，指示输入数据 X 是否为整数类型
    cdef bint query_id_is_not_empty = query_id.size > 0  # 布尔值，指示查询 ID 数组是否非空
    X_is_integral = X.dtype.kind == "i"  # 判断 X 是否为整数类型
    if X_is_integral:
        value_pattern = "%d:%d"  # 值的格式化字符串模式（整数类型）
    else:
        value_pattern = "%d:%.16g"  # 值的格式化字符串模式（浮点数类型）
    if y.dtype.kind == "i":
        label_pattern = "%d"  # 标签的格式化字符串模式（整数类型）
    else:
        label_pattern = "%.16g"  # 标签的格式化字符串模式（浮点数类型）

    line_pattern = "%s"  # 行的格式化字符串模式
    if query_id_is_not_empty:
        line_pattern += " qid:%d"  # 如果查询 ID 数组非空，则在行模式中加入查询 ID

    line_pattern += " %s\n"  # 在行模式中加入标签部分，并添加换行符

    cdef:
        Py_ssize_t num_labels = y.shape[1]  # 标签的数量
        Py_ssize_t x_len = X.shape[0]  # 数据的行数
        Py_ssize_t row_length = X.shape[1]  # 数据的列数
        Py_ssize_t i  # 循环变量
        Py_ssize_t j  # 循环变量
        Py_ssize_t col_start  # 列开始索引
        Py_ssize_t col_end  # 列结束索引
        Py_ssize_t[:] x_inds = np.empty(row_length, dtype=np.intp)  # 稀疏索引数组 x_inds
        signed long long[:] x_vals_int  # 整数类型的稀疏值数组 x_vals_int
        double[:] x_vals_float  # 浮点数类型的稀疏值数组 x_vals_float

    if not X_is_sp:  # 如果输入数据 X 不是稀疏矩阵
        if X_is_integral:
            x_vals_int = np.zeros(row_length, dtype=np.longlong)  # 初始化整数类型的稀疏值数组
        else:
            x_vals_float = np.zeros(row_length, dtype=np.float64)  # 初始化浮点数类型的稀疏值数组
    # 对输入数据集中每一行进行迭代处理
    for i in range(x_len):
        # 检查是否数据集 X 是稀疏矩阵
        if not X_is_sp:
            # 如果 X 是稠密矩阵且是整数类型
            if X_is_integral:
                # 调用函数获取当前行稠密格式的字符串表示
                s = get_dense_row_string(X, x_inds, x_vals_int, i, value_pattern, one_based)
            else:
                # 调用函数获取当前行稠密格式的字符串表示（浮点数类型）
                s = get_dense_row_string(X, x_inds, x_vals_float, i, value_pattern, one_based)
        else:
            # 调用函数获取当前行稀疏格式的字符串表示
            s = get_sparse_row_string(X.data, X.indptr, X.indices, i, value_pattern, one_based)
    
        # 如果处理的是多标签分类问题
        if multilabel:
            # 如果标签数据 y 是稀疏矩阵
            if y_is_sp:
                # 获取当前行标签的起始和结束位置
                col_start = y.indptr[i]
                col_end = y.indptr[i+1]
                # 使用生成器表达式构建标签字符串，跳过值为零的项
                labels_str = ','.join(tuple(label_pattern % y.indices[j] for j in range(col_start, col_end) if y.data[j] != 0))
            else:
                # 使用生成器表达式构建标签字符串，跳过值为零的项
                labels_str = ','.join(label_pattern % j for j in range(num_labels) if y[i, j] != 0)
        else:
            # 如果标签数据 y 是稀疏矩阵
            if y_is_sp:
                # 使用格式化字符串获取当前行的标签
                labels_str = label_pattern % y.data[i]
            else:
                # 使用格式化字符串获取当前行的标签
                labels_str = label_pattern % y[i, 0]
    
        # 如果查询 ID 非空
        if query_id_is_not_empty:
            # 组装特征元组（包含标签字符串、查询 ID 和当前行数据的字符串表示）
            feat = (labels_str, query_id[i], s)
        else:
            # 组装特征元组（包含标签字符串和当前行数据的字符串表示）
            feat = (labels_str, s)
    
        # 将特征元组格式化为字符串，并写入文件（编码为 UTF-8）
        f.write((line_pattern % feat).encode("utf-8"))
```