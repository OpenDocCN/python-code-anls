# `D:\src\scipysrc\scikit-learn\sklearn\datasets\_svmlight_format_io.py`

```
    f,
        # 数据源，可以是文件名（字符串）、路径对象或者有读取方法的对象
        [
            str,  # 文件名字符串
            Interval(Integral, 0, None, closed="left"),  # 整数区间，表示文件描述符（file descriptor）
            os.PathLike,  # 路径对象
            HasMethods("read"),  # 具有读取方法的对象
        ],
    n_features:
        # 特征数量，必须是正整数
        [Interval(Integral, 1, None, closed="left"), None],
    dtype:
        # 数据类型，由 numpy 自行验证
        "no_validation",
    multilabel:
        # 是否多标签数据，布尔值
        ["boolean"],
    zero_based:
        # 特征索引是否从0开始，可以是布尔值或者字符串选项 {"auto"}
        ["boolean", StrOptions({"auto"})],
    query_id:
        # 是否考虑对查询 ID 的约束，布尔值
        ["boolean"],
    offset:
        # 偏移量，必须是非负整数
        [Interval(Integral, 0, None, closed="left")],
    length:
        # 数据集长度，必须是整数
        [Integral],
    ----------
    f : str, path-like, file-like or int
        (Path to) a file to load. If a path ends in ".gz" or ".bz2", it will
        be uncompressed on the fly. If an integer is passed, it is assumed to
        be a file descriptor. A file-like or file descriptor will not be closed
        by this function. A file-like object must be opened in binary mode.
        
        .. versionchanged:: 1.2
           Path-like objects are now accepted.
           
        文件路径或文件描述符，用于加载数据。若路径以".gz"或".bz2"结尾，将会动态解压缩。
        若传入整数，将被视为文件描述符。文件对象或文件描述符不会在函数内部关闭。
        文件对象必须以二进制模式打开。

    n_features : int, default=None
        The number of features to use. If None, it will be inferred. This
        argument is useful to load several files that are subsets of a
        bigger sliced dataset: each subset might not have examples of
        every feature, hence the inferred shape might vary from one
        slice to another.
        n_features is only required if ``offset`` or ``length`` are passed a
        non-default value.
        
        要使用的特征数量。如果为None，将会推断。此参数对加载大型切片数据集中的子集非常有用：
        每个子集可能没有所有特征的示例，因此推断的形状可能会从一个切片变化到另一个切片。
        当传入``offset``或``length``的非默认值时，才需要指定n_features。

    dtype : numpy data type, default=np.float64
        Data type of dataset to be loaded. This will be the data type of the
        output numpy arrays ``X`` and ``y``.
        
        要加载数据集的数据类型。这将是输出numpy数组``X``和``y``的数据类型。

    multilabel : bool, default=False
        Samples may have several labels each (see
        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html).
        
        样本可能有多个标签。参见https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html。

    zero_based : bool or "auto", default="auto"
        Whether column indices in f are zero-based (True) or one-based
        (False). If column indices are one-based, they are transformed to
        zero-based to match Python/NumPy conventions.
        If set to "auto", a heuristic check is applied to determine this from
        the file contents. Both kinds of files occur "in the wild", but they
        are unfortunately not self-identifying. Using "auto" or True should
        always be safe when no ``offset`` or ``length`` is passed.
        If ``offset`` or ``length`` are passed, the "auto" mode falls back
        to ``zero_based=True`` to avoid having the heuristic check yield
        inconsistent results on different segments of the file.
        
        列索引在f中是否以零为基础（True）还是以一为基础（False）。如果列索引是以一为基础的，
        它们将被转换为以零为基础，以匹配Python/NumPy的约定。
        如果设置为"auto"，将应用启发式检查来从文件内容中确定。这两种类型的文件都“在野外”中存在，
        但不幸的是它们不能自我识别。在没有传递``offset``或``length``时，使用"auto"或True应始终安全。
        如果传递了``offset``或``length``，"auto"模式将退回到``zero_based=True``，
        以避免启发式检查在文件的不同段上产生不一致的结果。

    query_id : bool, default=False
        If True, will return the query_id array for each file.
        
        如果为True，则会为每个文件返回查询ID数组。

    offset : int, default=0
        Ignore the offset first bytes by seeking forward, then
        discarding the following bytes up until the next new line
        character.
        
        忽略前置的偏移字节，然后向前搜索，直到下一个换行符为止丢弃以下字节。

    length : int, default=-1
        If strictly positive, stop reading any new line of data once the
        position in the file has reached the (offset + length) bytes threshold.
        
        如果严格为正数，则一旦文件位置达到（偏移量+长度）字节阈值，
        停止读取任何新行数据。

    Returns
    -------
    X : scipy.sparse matrix of shape (n_samples, n_features)
        The data matrix.
        
        数据矩阵，形状为（n_samples，n_features）的稀疏矩阵。

    y : ndarray of shape (n_samples,), or a list of tuples of length n_samples
        The target. It is a list of tuples when ``multilabel=True``, else a
        ndarray.
        
        目标。当``multilabel=True``时，它是一个长度为n_samples的元组列表，否则是一个ndarray。

    query_id : array of shape (n_samples,)
       The query_id for each sample. Only returned when query_id is set to
       True.
       
       每个样本的查询ID。仅在query_id设置为True时返回。

    See Also
    --------
    
    # load_svmlight_files 函数类似于加载多个文件的功能，这些文件以特定格式存储，并确保它们具有相同数量的特征/列。
    
    Examples
    --------
    使用 joblib.Memory 缓存 svmlight 文件：
    
        from joblib import Memory
        from sklearn.datasets import load_svmlight_file
        mem = Memory("./mycache")
    
        @mem.cache
        def get_data():
            data = load_svmlight_file("mysvmlightfile")
            return data[0], data[1]
    
        X, y = get_data()
    """
    将单个文件 `f` 传递给 load_svmlight_files 函数，同时传递其他参数进行配置，例如：
    - n_features：特征的数量
    - dtype：数据类型
    - multilabel：是否多标签
    - zero_based：是否以零为基础索引
    - query_id：查询 ID
    - offset：偏移量
    - length：数据长度
    
    函数将返回一个元组，包含加载后的数据。
    """
    return tuple(
        load_svmlight_files(
            [f],
            n_features=n_features,
            dtype=dtype,
            multilabel=multilabel,
            zero_based=zero_based,
            query_id=query_id,
            offset=offset,
            length=length,
        )
    )
# 定义一个函数_gen_open，用于根据输入的文件类型打开文件并返回文件对象
def _gen_open(f):
    if isinstance(f, int):  # 如果输入是文件描述符
        return open(f, "rb", closefd=False)  # 以二进制只读方式打开文件
    elif isinstance(f, os.PathLike):  # 如果输入是路径对象
        f = os.fspath(f)  # 将路径对象转换为字符串路径
    elif not isinstance(f, str):  # 如果输入既不是整数也不是路径对象，抛出类型错误
        raise TypeError("expected {str, int, path-like, file-like}, got %s" % type(f))

    _, ext = os.path.splitext(f)  # 获取文件路径的扩展名
    if ext == ".gz":  # 如果文件是gzip压缩文件
        import gzip

        return gzip.open(f, "rb")  # 使用gzip模块打开文件并以二进制只读方式返回文件对象
    elif ext == ".bz2":  # 如果文件是bz2压缩文件
        from bz2 import BZ2File

        return BZ2File(f, "rb")  # 使用bz2模块打开文件并以二进制只读方式返回文件对象
    else:  # 如果文件既不是gzip也不是bz2格式
        return open(f, "rb")  # 直接以二进制只读方式打开文件并返回文件对象


# 定义函数_open_and_load，用于打开文件并加载数据
def _open_and_load(f, dtype, multilabel, zero_based, query_id, offset=0, length=-1):
    if hasattr(f, "read"):  # 如果文件对象f具有read方法（即已经是可读取的文件对象）
        # 调用_load_svmlight_file函数加载数据
        actual_dtype, data, ind, indptr, labels, query = _load_svmlight_file(
            f, dtype, multilabel, zero_based, query_id, offset, length
        )
    else:  # 如果文件对象f不具有read方法
        with closing(_gen_open(f)) as f:  # 使用_gen_open函数打开文件并使用closing确保文件关闭
            # 调用_load_svmlight_file函数加载数据
            actual_dtype, data, ind, indptr, labels, query = _load_svmlight_file(
                f, dtype, multilabel, zero_based, query_id, offset, length
            )

    # 将labels数据从array.array转换为正确的数据类型（如果不是多标签数据）
    if not multilabel:
        labels = np.frombuffer(labels, np.float64)
    data = np.frombuffer(data, actual_dtype)  # 将data数据从缓冲区转换为指定的数据类型
    indices = np.frombuffer(ind, np.longlong)  # 将ind数据从缓冲区转换为64位整数类型
    indptr = np.frombuffer(indptr, dtype=np.longlong)  # 将indptr数据从缓冲区转换为64位整数类型（不会为空）
    query = np.frombuffer(query, np.int64)  # 将query数据从缓冲区转换为64位整数类型

    data = np.asarray(data, dtype=dtype)  # 将data数据转换为指定的数据类型（对于float{32,64}来说是无操作）
    return data, indices, indptr, labels, query


# 装饰器validate_params用于验证函数load_svmlight_files的参数是否合法
@validate_params(
    {
        "files": [
            "array-like",  # files参数应为类似数组的对象
            str,  # 或者字符串类型
            os.PathLike,  # 或者PathLike对象
            HasMethods("read"),  # 或者具有read方法的对象
            Interval(Integral, 0, None, closed="left"),  # 或者是大于等于0的整数区间左闭
        ],
        "n_features": [Interval(Integral, 1, None, closed="left"), None],  # n_features参数应为大于等于1的整数区间左闭，或者为None
        "dtype": "no_validation",  # dtype参数交由numpy进行验证
        "multilabel": ["boolean"],  # multilabel参数应为布尔值
        "zero_based": ["boolean", StrOptions({"auto"})],  # zero_based参数应为布尔值或者"auto"
        "query_id": ["boolean"],  # query_id参数应为布尔值
        "offset": [Interval(Integral, 0, None, closed="left")],  # offset参数应为大于等于0的整数区间左闭
        "length": [Integral],  # length参数应为整数
    },
    prefer_skip_nested_validation=True,  # 偏好跳过嵌套验证
)
# load_svmlight_files函数用于从多个SVMlight格式的文件中加载数据集
def load_svmlight_files(
    files,
    *,
    n_features=None,  # 特征数量，默认为None
    dtype=np.float64,  # 数据类型，默认为np.float64
    multilabel=False,  # 是否多标签，默认为False
    zero_based="auto",  # 数据是否从0开始，默认为"auto"
    query_id=False,  # 是否包含查询ID，默认为False
    offset=0,  # 偏移量，默认为0
    length=-1,  # 数据长度，默认为-1
):
    """Load dataset from multiple files in SVMlight format.

    This function is equivalent to mapping load_svmlight_file over a list of
    files, except that the results are concatenated into a single, flat list
    and the samples vectors are constrained to all have the same number of
    features.

    In case the file contains a pairwise preference constraint (known
    as "qid" in the svmlight format) these are ignored unless the
    query_id parameter is set to True. These pairwise preference
    constraints can be used to constraint the combination of samples
    when using pairwise loss functions (as is the case in some
    learning to rank problems) so that only pairs with the same
    """
    # 函数内部逻辑在文档字符串中有详细描述，这里不进行重复注释
    query_id value are considered.

    Parameters
    ----------
    files : array-like, dtype=str, path-like, file-like or int
        (Paths of) files to load. If a path ends in ".gz" or ".bz2", it will
        be uncompressed on the fly. If an integer is passed, it is assumed to
        be a file descriptor. File-likes and file descriptors will not be
        closed by this function. File-like objects must be opened in binary
        mode.

        .. versionchanged:: 1.2
           Path-like objects are now accepted.

    n_features : int, default=None
        The number of features to use. If None, it will be inferred from the
        maximum column index occurring in any of the files.

        This can be set to a higher value than the actual number of features
        in any of the input files, but setting it to a lower value will cause
        an exception to be raised.

    dtype : numpy data type, default=np.float64
        Data type of dataset to be loaded. This will be the data type of the
        output numpy arrays ``X`` and ``y``.

    multilabel : bool, default=False
        Samples may have several labels each (see
        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html).

    zero_based : bool or "auto", default="auto"
        Whether column indices in f are zero-based (True) or one-based
        (False). If column indices are one-based, they are transformed to
        zero-based to match Python/NumPy conventions.
        If set to "auto", a heuristic check is applied to determine this from
        the file contents. Both kinds of files occur "in the wild", but they
        are unfortunately not self-identifying. Using "auto" or True should
        always be safe when no offset or length is passed.
        If offset or length are passed, the "auto" mode falls back
        to zero_based=True to avoid having the heuristic check yield
        inconsistent results on different segments of the file.

    query_id : bool, default=False
        If True, will return the query_id array for each file.

    offset : int, default=0
        Ignore the offset first bytes by seeking forward, then
        discarding the following bytes up until the next new line
        character.

    length : int, default=-1
        If strictly positive, stop reading any new line of data once the
        position in the file has reached the (offset + length) bytes threshold.

    Returns
    -------
    [X1, y1, ..., Xn, yn] or [X1, y1, q1, ..., Xn, yn, qn]: list of arrays
        Each (Xi, yi) pair is the result from load_svmlight_file(files[i]).
        If query_id is set to True, this will return instead (Xi, yi, qi)
        triplets.

    See Also
    --------
    load_svmlight_file: Similar function for loading a single file in this
        format.

    Notes
    -----
    When fitting a model to a matrix X_train and evaluating it against a
    matrix X_test, it is essential that X_train and X_test have the same
    # 如果偏移量不为零或长度大于零，并且未显式指定 zero_based 参数时，默认启用零基准
    if (offset != 0 or length > 0) and zero_based == "auto":
        # 禁用启发式搜索，以避免在文件的不同段上得到不一致的结果
        zero_based = True

    # 如果指定了偏移量或长度，但未提供 n_features 参数，则抛出数值错误异常
    if (offset != 0 or length > 0) and n_features is None:
        raise ValueError("n_features is required when offset or length is specified.")

    # 打开并加载多个文件，每个文件对应调用 _open_and_load 函数
    r = [
        _open_and_load(
            f,
            dtype,
            multilabel,
            bool(zero_based),
            bool(query_id),
            offset=offset,
            length=length,
        )
        for f in files
    ]

    # 如果 zero_based 参数为 False，或者为 "auto" 且所有结果中的最小索引大于零
    if (
        zero_based is False
        or zero_based == "auto"
        and all(len(tmp[1]) and np.min(tmp[1]) > 0 for tmp in r)
    ):
        # 将所有结果中的索引减一，使其符合零基准假设
        for _, indices, _, _, _ in r:
            indices -= 1

    # 计算所有结果中的最大特征数目，用于确定最终数据集的特征数目
    n_f = max(ind[1].max() if len(ind[1]) else 0 for ind in r) + 1

    # 如果未指定 n_features 参数，则将其设置为计算出的特征数目
    if n_features is None:
        n_features = n_f
    # 否则，如果指定的 n_features 小于计算出的特征数目，则抛出数值错误异常
    elif n_features < n_f:
        raise ValueError(
            "n_features was set to {}, but input file contains {} features".format(
                n_features, n_f
            )
        )

    # 初始化结果列表
    result = []
    # 遍历每个文件加载的数据
    for data, indices, indptr, y, query_values in r:
        # 根据加载的数据、索引、指针、以及可能的查询值构建稀疏矩阵
        shape = (indptr.shape[0] - 1, n_features)
        X = sp.csr_matrix((data, indices, indptr), shape)
        # 对索引进行排序
        X.sort_indices()
        # 将构建好的特征矩阵 X 和标签 y 添加到结果列表中
        result += X, y
        # 如果有查询标识，则将查询值也添加到结果列表中
        if query_id:
            result.append(query_values)

    # 返回最终构建的结果列表
    return result
# 将数据集以 svmlight / libsvm 格式存储到文件中

@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # X 是训练数据，可以是稠密数组或稀疏矩阵
        "y": ["array-like", "sparse matrix"],  # y 是目标值，可以是稠密数组或稀疏矩阵
        "f": [str, HasMethods(["write"])],  # f 是文件路径或支持 write 方法的文件对象
        "zero_based": ["boolean"],  # zero_based 表示列索引是否从 0 开始
        "comment": [str, bytes, None],  # comment 是要插入到文件顶部的注释，可以是字符串或字节串
        "query_id": ["array-like", None],  # query_id 是一维数组，包含配对偏好约束
        "multilabel": ["boolean"],  # multilabel 表示是否多标签分类
    },
    prefer_skip_nested_validation=True,
)
def dump_svmlight_file(
    X,
    y,
    f,
    *,
    zero_based=True,  # 是否使用零基索引
    comment=None,  # 要插入的文件顶部注释
    query_id=None,  # 配对偏好约束数组
    multilabel=False,  # 是否多标签
):
    """Dump the dataset in svmlight / libsvm file format.

    This format is a text-based format, with one sample per line. It does
    not store zero valued features hence is suitable for sparse dataset.

    The first element of each line can be used to store a target variable
    to predict.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vectors, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : {array-like, sparse matrix}, shape = (n_samples,) or (n_samples, n_labels)
        Target values. Class labels must be an
        integer or float, or array-like objects of integer or float for
        multilabel classifications.

    f : str or file-like in binary mode
        If string, specifies the path that will contain the data.
        If file-like, data will be written to f. f should be opened in binary
        mode.

    zero_based : bool, default=True
        Whether column indices should be written zero-based (True) or one-based
        (False).

    comment : str or bytes, default=None
        Comment to insert at the top of the file. This should be either a
        Unicode string, which will be encoded as UTF-8, or an ASCII byte
        string.
        If a comment is given, then it will be preceded by one that identifies
        the file as having been dumped by scikit-learn. Note that not all
        tools grok comments in SVMlight files.

    query_id : array-like of shape (n_samples,), default=None
        Array containing pairwise preference constraints (qid in svmlight
        format).
    """
    if comment:
        f.write(
            (
                "# Generated by dump_svmlight_file from scikit-learn %s\n" % __version__
            ).encode()
        )  # 写入由 scikit-learn 生成的注释，包含版本信息

        f.write(
            ("# Column indices are %s-based\n" % ["zero", "one"][one_based]).encode()
        )  # 写入列索引基于零或一的说明

        f.write(b"#\n")  # 写入空行
        f.writelines(b"# %s\n" % line for line in comment.splitlines())  # 写入用户提供的注释

    X_is_sp = sp.issparse(X)  # 判断 X 是否为稀疏矩阵
    y_is_sp = sp.issparse(y)  # 判断 y 是否为稀疏矩阵

    if not multilabel and not y_is_sp:
        y = y[:, np.newaxis]  # 如果不是多标签分类且 y 不是稀疏矩阵，则增加一个维度

    _dump_svmlight_file(
        X,
        y,
        f,
        multilabel,
        one_based,
        query_id,
        X_is_sp,
        y_is_sp,
    )  # 调用实际的数据转换函数 _dump_svmlight_file
    if comment is not None:
        # 如果提供了注释，将注释字符串转换成UTF-8编码的字节列表。
        # 如果传入的是字节串，则检查是否为ASCII编码；
        # 如果用户希望更复杂，他们需要自行解码。
        if isinstance(comment, bytes):
            comment.decode("ascii")  # 仅用于异常处理
        else:
            comment = comment.encode("utf-8")
        # 如果注释字符串中包含空字节（NUL byte），则抛出值错误异常。
        if b"\0" in comment:
            raise ValueError("comment string contains NUL byte")

    # 检查y的格式，接受稀疏矩阵csr格式，并确保是二维的。
    yval = check_array(y, accept_sparse="csr", ensure_2d=False)
    if sp.issparse(yval):
        # 如果y是稀疏矩阵，并且不是多标签模式，检查其形状是否为(n_samples, 1)。
        if yval.shape[1] != 1 and not multilabel:
            raise ValueError(
                "expected y of shape (n_samples, 1), got %r" % (yval.shape,)
            )
    else:
        # 如果y不是稀疏矩阵，并且不是多标签模式，检查其维度是否为1。
        if yval.ndim != 1 and not multilabel:
            raise ValueError("expected y of shape (n_samples,), got %r" % (yval.shape,))

    # 检查X的格式，接受稀疏矩阵csr格式。
    Xval = check_array(X, accept_sparse="csr")
    # 检查X和y的样本数是否相等。
    if Xval.shape[0] != yval.shape[0]:
        raise ValueError(
            "X.shape[0] and y.shape[0] should be the same, got %r and %r instead."
            % (Xval.shape[0], yval.shape[0])
        )

    # 我们曾经遇到过CSR矩阵未排序索引的问题（例如#1501），
    # 因此在此处对它们进行排序，但首先确保不修改用户的X。
    # TODO: 我们可以更便宜地做到这一点；sorted_indices会复制整个矩阵。
    if yval is y and hasattr(yval, "sorted_indices"):
        y = yval.sorted_indices()
    else:
        y = yval
        if hasattr(y, "sort_indices"):
            y.sort_indices()

    if Xval is X and hasattr(Xval, "sorted_indices"):
        X = Xval.sorted_indices()
    else:
        X = Xval
        if hasattr(X, "sort_indices"):
            X.sort_indices()

    if query_id is None:
        # 注意：query_id被传递给Cython函数，使用query_id的融合类型。
        # 然而，从Cython>=3.0开始，内存视图不能为None，否则运行时将不知道将Python调用分派给哪个具体实现。
        # TODO: 在_svm_light_format_fast.pyx中简化接口和实现。
        query_id = np.array([], dtype=np.int32)
    else:
        # 将query_id转换为NumPy数组。
        query_id = np.asarray(query_id)
        # 如果query_id的样本数不等于y的样本数，则抛出值错误异常。
        if query_id.shape[0] != y.shape[0]:
            raise ValueError(
                "expected query_id of shape (n_samples,), got %r" % (query_id.shape,)
            )

    # 根据zero_based的值确定one_based是True还是False。
    one_based = not zero_based
    # 检查对象 f 是否具有 write 方法，即判断 f 是否可写入
    if hasattr(f, "write"):
        # 若 f 可写入，则调用 _dump_svmlight 函数将数据以 SVMLight 格式写入 f 中
        _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id)
    else:
        # 若 f 不可写入，则以二进制写入模式打开文件 f
        with open(f, "wb") as f:
            # 在文件 f 中调用 _dump_svmlight 函数将数据以 SVMLight 格式写入
            _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id)
```