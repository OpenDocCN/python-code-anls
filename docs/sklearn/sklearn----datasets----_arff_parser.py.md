# `D:\src\scipysrc\scikit-learn\sklearn\datasets\_arff_parser.py`

```
# 导入所需的模块和包
"""Implementation of ARFF parsers: via LIAC-ARFF and pandas."""
import itertools  # 导入 itertools 模块，用于迭代工具函数
import re  # 导入 re 模块，用于正则表达式操作
from collections import OrderedDict  # 导入 OrderedDict 类，用于有序字典
from collections.abc import Generator  # 导入 Generator 类，用于抽象基类的生成器类型
from typing import List  # 导入 List 类型提示

import numpy as np  # 导入 NumPy 库，用于数值计算
import scipy as sp  # 导入 SciPy 库，用于科学计算

from ..externals import _arff  # 从外部模块导入 _arff
from ..externals._arff import ArffSparseDataType  # 从外部模块导入 ArffSparseDataType 类型
from ..utils._chunking import chunk_generator, get_chunk_n_rows  # 从工具模块导入 chunk_generator 和 get_chunk_n_rows 函数
from ..utils._optional_dependencies import check_pandas_support  # 从工具模块导入 check_pandas_support 函数
from ..utils.fixes import pd_fillna  # 从修复模块导入 pd_fillna 函数


def _split_sparse_columns(
    arff_data: ArffSparseDataType, include_columns: List
) -> ArffSparseDataType:
    """Obtains several columns from sparse ARFF representation. Additionally,
    the column indices are re-labelled, given the columns that are not
    included. (e.g., when including [1, 2, 3], the columns will be relabelled
    to [0, 1, 2]).

    Parameters
    ----------
    arff_data : tuple
        A tuple of three lists of equal size; first list indicating the value,
        second the x coordinate and the third the y coordinate.

    include_columns : list
        A list of columns to include.

    Returns
    -------
    arff_data_new : tuple
        Subset of arff data with only the include columns indicated by the
        include_columns argument.
    """
    # 初始化新的稀疏数据集合
    arff_data_new: ArffSparseDataType = (list(), list(), list())
    # 重新索引列索引，确保包含列的索引从 0 开始递增
    reindexed_columns = {
        column_idx: array_idx for array_idx, column_idx in enumerate(include_columns)
    }
    # 遍历原始稀疏数据，只选择包含在 include_columns 中的列
    for val, row_idx, col_idx in zip(arff_data[0], arff_data[1], arff_data[2]):
        if col_idx in include_columns:
            arff_data_new[0].append(val)
            arff_data_new[1].append(row_idx)
            arff_data_new[2].append(reindexed_columns[col_idx])
    return arff_data_new


def _sparse_data_to_array(
    arff_data: ArffSparseDataType, include_columns: List
) -> np.ndarray:
    # 将稀疏数据转换回数组（无法使用 toarray() 函数，因为它仅适用于数值数据）
    num_obs = max(arff_data[1]) + 1  # 计算观测数量
    y_shape = (num_obs, len(include_columns))  # 确定输出数组形状
    # 重新索引列索引，确保包含列的索引从 0 开始递增
    reindexed_columns = {
        column_idx: array_idx for array_idx, column_idx in enumerate(include_columns)
    }
    # 创建空的 NumPy 数组用于存储转换后的稀疏数据
    y = np.empty(y_shape, dtype=np.float64)
    # 遍历原始稀疏数据，将其填充到 NumPy 数组中
    for val, row_idx, col_idx in zip(arff_data[0], arff_data[1], arff_data[2]):
        if col_idx in include_columns:
            y[row_idx, reindexed_columns[col_idx]] = val
    return y


def _post_process_frame(frame, feature_names, target_names):
    """Post process a dataframe to select the desired columns in `X` and `y`.

    Parameters
    ----------
    frame : dataframe
        The dataframe to split into `X` and `y`.

    feature_names : list of str
        The list of feature names to populate `X`.

    target_names : list of str
        The list of target names to populate `y`.

    Returns
    -------
    X : dataframe
        The dataframe containing the features.
    """
    # 未提供此函数的实现，因此没有进一步的注释
    # y : {series, dataframe} or None
    # 定义函数参数 y，可以是 series、dataframe 类型或者 None

    X = frame[feature_names]
    # 从数据框 frame 中选取特征名列表 feature_names 所对应的列，赋值给 X

    if len(target_names) >= 2:
        # 如果目标名称列表 target_names 的长度大于等于 2
        y = frame[target_names]
        # 则将 frame 中 target_names 列的数据赋值给 y，此时 y 是一个 dataframe

    elif len(target_names) == 1:
        # 否则，如果目标名称列表 target_names 的长度为 1
        y = frame[target_names[0]]
        # 则将 frame 中 target_names[0] 列的数据赋值给 y，此时 y 是一个 series

    else:
        # 否则，如果目标名称列表 target_names 的长度为 0
        y = None
        # 将 y 设置为 None

    return X, y
    # 返回 X 和 y，其中 X 是特征数据，y 是目标数据（可能是 series、dataframe 或 None）
    def _liac_arff_parser(
        gzip_file,
        output_arrays_type,
        openml_columns_info,
        feature_names_to_select,
        target_names_to_select,
        shape=None,
    ):
        """ARFF parser using the LIAC-ARFF library coded purely in Python.

        This parser is quite slow but consumes a generator. Currently it is needed
        to parse sparse datasets. For dense datasets, it is recommended to instead
        use the pandas-based parser, although it does not always handles the
        dtypes exactly the same.

        Parameters
        ----------
        gzip_file : GzipFile instance
            The file compressed to be read.

        output_arrays_type : {"numpy", "sparse", "pandas"}
            The type of the arrays that will be returned. The possibilities are:

            - `"numpy"`: both `X` and `y` will be NumPy arrays;
            - `"sparse"`: `X` will be sparse matrix and `y` will be a NumPy array;
            - `"pandas"`: `X` will be a pandas DataFrame and `y` will be either a
              pandas Series or DataFrame.

        columns_info : dict
            The information provided by OpenML regarding the columns of the ARFF
            file.

        feature_names_to_select : list of str
            A list of the feature names to be selected.

        target_names_to_select : list of str
            A list of the target names to be selected.

        Returns
        -------
        X : {ndarray, sparse matrix, dataframe}
            The data matrix.

        y : {ndarray, dataframe, series}
            The target.

        frame : dataframe or None
            A dataframe containing both `X` and `y`. `None` if
            `output_array_type != "pandas"`.

        categories : list of str or None
            The names of the features that are categorical. `None` if
            `output_array_type == "pandas"`.
        """

        def _io_to_generator(gzip_file):
            # Convert GzipFile stream to a generator yielding lines as UTF-8 strings.
            for line in gzip_file:
                yield line.decode("utf-8")

        # Convert GzipFile to a generator of lines
        stream = _io_to_generator(gzip_file)

        # Determine the type of ARFF data structure based on output_arrays_type
        return_type = _arff.COO if output_arrays_type == "sparse" else _arff.DENSE_GEN
        
        # Determine if nominal attributes should be encoded with numerical values
        encode_nominal = not (output_arrays_type == "pandas")

        # Load ARFF data using LIAC-ARFF library
        arff_container = _arff.load(
            stream, return_type=return_type, encode_nominal=encode_nominal
        )

        # Select columns that are specified in feature_names_to_select and target_names_to_select
        columns_to_select = feature_names_to_select + target_names_to_select

        # Create a dictionary of categorical features from arff_container attributes
        categories = {
            name: cat
            for name, cat in arff_container["attributes"]
            if isinstance(cat, list) and name in columns_to_select
        }
    # 如果输出类型为 "pandas"
    if output_arrays_type == "pandas":
        # 检查 Pandas 支持情况，提示信息为 "fetch_openml with as_frame=True"
        pd = check_pandas_support("fetch_openml with as_frame=True")

        # 从 arff_container 中获取列信息，并保持有序
        columns_info = OrderedDict(arff_container["attributes"])
        # 提取列名列表
        columns_names = list(columns_info.keys())

        # 计算第一行的内存使用量
        first_row = next(arff_container["data"])
        # 创建包含第一行数据的 Pandas DataFrame，使用列名列表作为列名
        first_df = pd.DataFrame([first_row], columns=columns_names, copy=False)

        # 计算每个数据块的大小
        row_bytes = first_df.memory_usage(deep=True).sum()
        chunksize = get_chunk_n_rows(row_bytes)

        # 选择需要保留的列
        columns_to_keep = [col for col in columns_names if col in columns_to_select]
        # 初始化数据块列表，包含第一个 DataFrame
        dfs = [first_df[columns_to_keep]]

        # 遍历数据生成器，读取数据块并添加到数据块列表中
        for data in chunk_generator(arff_container["data"], chunksize):
            dfs.append(
                # 创建包含数据块的 Pandas DataFrame，使用列名列表作为列名，并仅保留需要的列
                pd.DataFrame(data, columns=columns_names, copy=False)[columns_to_keep]
            )

        # 如果数据块列表中有多于一个数据块，使用第二个数据块的类型配置第一个数据块的类型
        if len(dfs) >= 2:
            dfs[0] = dfs[0].astype(dfs[1].dtypes)

        # liac-arff 解析器不依赖于 NumPy，并使用 None 表示缺失值。
        # 为了与 Pandas 解析器一致，我们将 None 替换为 np.nan
        frame = pd.concat(dfs, ignore_index=True)
        frame = pd_fillna(pd, frame)
        del dfs, first_df

        # 配置数据框中的列类型
        dtypes = {}
        for name in frame.columns:
            # 从 openml_columns_info 中获取列的数据类型
            column_dtype = openml_columns_info[name]["data_type"]
            if column_dtype.lower() == "integer":
                # 使用 Pandas 扩展数组 "Int64" 替代 np.int64，以支持缺失值
                dtypes[name] = "Int64"
            elif column_dtype.lower() == "nominal":
                # 将名义型数据配置为 Pandas 的 "category" 类型
                dtypes[name] = "category"
            else:
                # 其他情况下保留原始数据框中的数据类型
                dtypes[name] = frame.dtypes[name]
        # 使用配置后的类型重新设置数据框的列类型
        frame = frame.astype(dtypes)

        # 对数据框进行后处理，获取特征和目标列
        X, y = _post_process_frame(
            frame, feature_names_to_select, target_names_to_select
        )
    else:
        # 获取 ARFF 数据
        arff_data = arff_container["data"]

        # 提取要选择的特征列的索引
        feature_indices_to_select = [
            int(openml_columns_info[col_name]["index"])
            for col_name in feature_names_to_select
        ]

        # 提取要选择的目标列的索引
        target_indices_to_select = [
            int(openml_columns_info[col_name]["index"])
            for col_name in target_names_to_select
        ]

        # 如果 ARFF 数据是生成器
        if isinstance(arff_data, Generator):
            # 如果未提供形状，则抛出值错误异常
            if shape is None:
                raise ValueError(
                    "shape must be provided when arr['data'] is a Generator"
                )
            # 如果形状的第一个维度为-1，则计数设置为-1
            if shape[0] == -1:
                count = -1
            else:
                count = shape[0] * shape[1]
            # 将生成器数据转换为 NumPy 数组
            data = np.fromiter(
                itertools.chain.from_iterable(arff_data),
                dtype="float64",
                count=count,
            )
            # 根据提供的形状重新调整数据的形状
            data = data.reshape(*shape)
            # 从数据中选择特征列和目标列
            X = data[:, feature_indices_to_select]
            y = data[:, target_indices_to_select]
        # 如果 ARFF 数据是元组形式
        elif isinstance(arff_data, tuple):
            # 将稀疏列拆分为 COO 格式的矩阵
            arff_data_X = _split_sparse_columns(arff_data, feature_indices_to_select)
            num_obs = max(arff_data[1]) + 1
            X_shape = (num_obs, len(feature_indices_to_select))
            # 创建 COO 稀疏矩阵
            X = sp.sparse.coo_matrix(
                (arff_data_X[0], (arff_data_X[1], arff_data_X[2])),
                shape=X_shape,
                dtype=np.float64,
            )
            # 将 COO 矩阵转换为 CSR 格式
            X = X.tocsr()
            # 将稀疏数据转换为数组
            y = _sparse_data_to_array(arff_data, target_indices_to_select)
        else:
            # 如果不属于预期的 ARFF 数据类型，则抛出值错误异常
            raise ValueError(
                f"Unexpected type for data obtained from arff: {type(arff_data)}"
            )

        # 确定是否是分类问题
        is_classification = {
            col_name in categories for col_name in target_names_to_select
        }
        
        # 如果不是分类问题，则无需处理目标数据
        if not is_classification:
            # 没有目标数据
            pass
        # 如果所有目标都是分类问题
        elif all(is_classification):
            # 将分类数据合并到 y 中
            y = np.hstack(
                [
                    np.take(
                        np.asarray(categories.pop(col_name), dtype="O"),
                        y[:, i : i + 1].astype(int, copy=False),
                    )
                    for i, col_name in enumerate(target_names_to_select)
                ]
            )
        # 如果有混合分类和非分类的目标数据
        elif any(is_classification):
            # 目前不支持混合分类和非分类目标数据，抛出值错误异常
            raise ValueError(
                "Mix of nominal and non-nominal targets is not currently supported"
            )

        # 如果 y 的形状只有一列，则将其重新调整为 1-D 数组
        # 如果 y 没有列，则将其设为 None
        if y.shape[1] == 1:
            y = y.reshape((-1,))
        elif y.shape[1] == 0:
            y = None

    # 如果输出类型为 pandas，则返回 X, y, frame 和 None
    if output_arrays_type == "pandas":
        return X, y, frame, None
    # 否则返回 X, y, None 和 categories
    return X, y, None, categories
    # 读取 gzip 文件直到数据部分，跳过 ARFF 元数据头部
    for line in gzip_file:
        if line.decode("utf-8").lower().startswith("@data"):
            break

    # 定义数据类型字典，用于存储每个列的数据类型信息
    dtypes = {}

    # 遍历从 OpenML 获取的列信息
    for name in openml_columns_info:
        # 获取列的数据类型
        column_dtype = openml_columns_info[name]["data_type"]

        # 如果数据类型是整数
        if column_dtype.lower() == "integer":
            # 使用 Int64 类型以便从数据中推断缺失值
            dtypes[name] = "Int64"

        # 如果数据类型是名义（nominal）
        elif column_dtype.lower() == "nominal":
            # 将数据类型设置为分类类型
            dtypes[name] = "category"

    # 由于在读取 ARFF 文件时不会传递 `names` 参数，因此需要将 `dtypes` 从列名转换为列索引
    dtypes_positional = {
        col_idx: dtypes[name]
        for col_idx, name in enumerate(openml_columns_info)
        if name in dtypes
    }
    # 默认的读取 CSV 文件的参数设置
    default_read_csv_kwargs = {
        "header": None,  # 不使用文件的第一行作为列名
        "index_col": False,  # 强制 pandas 不使用第一列作为索引
        "na_values": ["?"],  # 将问号字符视为缺失值
        "keep_default_na": False,  # 只有 `?` 被视为缺失值，符合 ARFF 规范
        "comment": "%",  # 跳过以 `%` 开头的行，因为它们是注释
        "quotechar": '"',  # 用于引用字符串的定界符
        "skipinitialspace": True,  # 跳过分隔符后的空格，以符合 ARFF 规范
        "escapechar": "\\",  # 转义字符
        "dtype": dtypes_positional,  # 数据类型，从外部指定
    }
    # 合并默认的参数和传入的参数（如果有的话）
    read_csv_kwargs = {**default_read_csv_kwargs, **(read_csv_kwargs or {})}
    # 使用 pandas 读取 gzip 压缩的 CSV 文件，应用上述参数
    frame = pd.read_csv(gzip_file, **read_csv_kwargs)
    try:
        # 在读取文件时设置列名，选择前 N 列，避免 ParserError。如果列数与 OpenML 给出的
        # 元数据列数不匹配，则会引发 ValueError。
        frame.columns = [name for name in openml_columns_info]
    except ValueError as exc:
        # 如果列数不匹配，则抛出 ParserError 异常
        raise pd.errors.ParserError(
            "The number of columns provided by OpenML does not match the number of "
            "columns inferred by pandas when reading the file."
        ) from exc

    # 需要保留的列名列表，包括特征和目标列
    columns_to_select = feature_names_to_select + target_names_to_select
    # 保留在数据帧中存在于 columns_to_select 列表中的列
    columns_to_keep = [col for col in frame.columns if col in columns_to_select]
    frame = frame[columns_to_keep]

    # `pd.read_csv` 自动处理双引号来引用非数值的 CSV 单元格值。与 LIAC-ARFF 不同，
    # `pd.read_csv` 不能同时配置单引号和双引号作为有效的引用字符，因为在常规
    # （非 ARFF）CSV 文件中这种情况不会发生。为了模仿 LIAC-ARFF 解析器的行为，
    # 如果需要，我们手动去除类别变量中的单引号作为后处理步骤。
    #
    # 注意，我们故意不尝试对（非类别）字符串类型的列进行此类手动后处理，
    # 因为我们无法解决包含嵌套引用的 CSV 单元格值的歧义，例如 `"'some string value'"`。
    single_quote_pattern = re.compile(r"^'(?P<contents>.*)'$")

    def strip_single_quotes(input_string):
        match = re.search(single_quote_pattern, input_string)
        if match is None:
            return input_string

        return match.group("contents")

    # 所有类别变量的列名列表，这些列具有 pd.CategoricalDtype 类型
    categorical_columns = [
        name
        for name, dtype in frame.dtypes.items()
        if isinstance(dtype, pd.CategoricalDtype)
    ]
    # 对每个类别变量列进行单引号去除操作
    for col in categorical_columns:
        frame[col] = frame[col].cat.rename_categories(strip_single_quotes)

    # 对数据帧进行后处理，返回处理后的特征矩阵 X 和目标向量 y
    X, y = _post_process_frame(frame, feature_names_to_select, target_names_to_select)

    # 如果输出数组类型为 pandas，则返回 X、y、frame 和 None
    if output_arrays_type == "pandas":
        return X, y, frame, None
    else:
        # 如果输入的数据不是numpy数组，则转换为numpy数组
        X, y = X.to_numpy(), y.to_numpy()

    # 创建一个字典categories，用于存储DataFrame中所有列的分类数据
    categories = {
        name: dtype.categories.tolist()
        for name, dtype in frame.dtypes.items()  # 遍历DataFrame的列名和对应的数据类型
        if isinstance(dtype, pd.CategoricalDtype)  # 仅选择类型为pd.CategoricalDtype的列
    }
    # 返回转换后的X和y数组，以及一个空对象和categories字典
    return X, y, None, categories
# 加载一个经过压缩的 ARFF 文件，使用指定的解析器进行解析

def load_arff_from_gzip_file(
    gzip_file,  # GzipFile 的实例，待读取的压缩文件
    parser,  # 字符串，指定用于解析 ARFF 文件的解析器。推荐使用 "pandas"，但仅支持加载密集型数据集。
    output_type,  # 字符串，指定返回的数组类型，可以是 "numpy"、"sparse" 或 "pandas"
    openml_columns_info,  # 字典，包含来自 OpenML 的关于 ARFF 文件列的信息
    feature_names_to_select,  # 字符串列表，选择要使用的特征名称
    target_names_to_select,  # 字符串列表，选择要使用的目标名称
    shape=None,  # 可选的元组，指定数据的形状
    read_csv_kwargs=None,  # 字典，默认为 None，传递给 `pandas.read_csv` 的关键字参数，允许覆盖默认选项
):
    """Load a compressed ARFF file using a given parser.

    Parameters
    ----------
    gzip_file : GzipFile instance
        The file compressed to be read.

    parser : {"pandas", "liac-arff"}
        The parser used to parse the ARFF file. "pandas" is recommended
        but only supports loading dense datasets.

    output_type : {"numpy", "sparse", "pandas"}
        The type of the arrays that will be returned. The possibilities are:

        - `"numpy"`: both `X` and `y` will be NumPy arrays;
        - `"sparse"`: `X` will be sparse matrix and `y` will be a NumPy array;
        - `"pandas"`: `X` will be a pandas DataFrame and `y` will be either a
          pandas Series or DataFrame.

    openml_columns_info : dict
        The information provided by OpenML regarding the columns of the ARFF
        file.

    feature_names_to_select : list of str
        A list of the feature names to be selected.

    target_names_to_select : list of str
        A list of the target names to be selected.

    read_csv_kwargs : dict, default=None
        Keyword arguments to pass to `pandas.read_csv`. It allows to overwrite
        the default options.

    Returns
    -------
    X : {ndarray, sparse matrix, dataframe}
        The data matrix.

    y : {ndarray, dataframe, series}
        The target.

    frame : dataframe or None
        A dataframe containing both `X` and `y`. `None` if
        `output_array_type != "pandas"`.

    categories : list of str or None
        The names of the features that are categorical. `None` if
        `output_array_type == "pandas"`.
    """

    if parser == "liac-arff":
        # 使用 liac-arff 解析器解析 ARFF 文件
        return _liac_arff_parser(
            gzip_file,
            output_type,
            openml_columns_info,
            feature_names_to_select,
            target_names_to_select,
            shape,
        )
    elif parser == "pandas":
        # 使用 pandas 解析器解析 ARFF 文件
        return _pandas_arff_parser(
            gzip_file,
            output_type,
            openml_columns_info,
            feature_names_to_select,
            target_names_to_select,
            read_csv_kwargs,
        )
    else:
        # 如果解析器不是 "liac-arff" 或 "pandas"，抛出错误
        raise ValueError(
            f"Unknown parser: '{parser}'. Should be 'liac-arff' or 'pandas'."
        )
```