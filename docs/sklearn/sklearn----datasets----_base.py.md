# `D:\src\scipysrc\scikit-learn\sklearn\datasets\_base.py`

```
"""
Base IO code for all datasets
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
import csv  # 导入处理 CSV 文件的模块
import gzip  # 导入处理 gzip 压缩文件的模块
import hashlib  # 导入用于计算哈希值的模块
import os  # 导入操作系统相关功能的模块
import shutil  # 导入文件和目录操作的高级模块
import time  # 导入时间相关的模块
import warnings  # 导入警告处理相关的模块
from collections import namedtuple  # 导入命名元组用于创建命名的数据结构
from importlib import resources  # 导入用于管理资源的模块
from numbers import Integral  # 导入用于数值类型检查的模块
from os import environ, listdir, makedirs  # 从 os 模块导入环境变量和目录操作相关函数
from os.path import expanduser, isdir, join, splitext  # 导入路径操作相关的函数
from pathlib import Path  # 导入操作路径的模块
from urllib.error import URLError  # 导入处理 URL 错误的模块
from urllib.request import urlretrieve  # 导入下载文件的模块

import numpy as np  # 导入数值计算库 numpy

from ..preprocessing import scale  # 导入预处理模块中的数据缩放函数
from ..utils import Bunch, check_random_state  # 导入实用工具中的数据结构和随机状态检查函数
from ..utils._optional_dependencies import check_pandas_support  # 导入检查 pandas 支持的函数
from ..utils._param_validation import Interval, StrOptions, validate_params  # 导入参数验证相关的函数和类

DATA_MODULE = "sklearn.datasets.data"  # 定义数据模块名称
DESCR_MODULE = "sklearn.datasets.descr"  # 定义描述模块名称
IMAGES_MODULE = "sklearn.datasets.images"  # 定义图像模块名称

RemoteFileMetadata = namedtuple("RemoteFileMetadata", ["filename", "url", "checksum"])  # 定义远程文件元数据的命名元组


@validate_params(
    {
        "data_home": [str, os.PathLike, None],  # 参数验证器，验证 data_home 参数是否为字符串或路径对象或 None
    },
    prefer_skip_nested_validation=True,
)
def get_data_home(data_home=None) -> str:
    """Return the path of the scikit-learn data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data directory is set to a folder named 'scikit_learn_data' in the
    user home folder.

    Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str or path-like, default=None
        The path to scikit-learn data directory. If `None`, the default path
        is `~/scikit_learn_data`.

    Returns
    -------
    data_home: str
        The path to scikit-learn data directory.

    Examples
    --------
    >>> import os
    >>> from sklearn.datasets import get_data_home
    >>> data_home_path = get_data_home()
    >>> os.path.exists(data_home_path)
    True
    """
    if data_home is None:
        data_home = environ.get("SCIKIT_LEARN_DATA", join("~", "scikit_learn_data"))  # 如果 data_home 为 None，则设置默认路径
    data_home = expanduser(data_home)  # 展开用户目录的路径
    makedirs(data_home, exist_ok=True)  # 如果目录不存在，则创建目录
    return data_home  # 返回数据目录的路径


@validate_params(
    {
        "data_home": [str, os.PathLike, None],  # 参数验证器，验证 data_home 参数是否为字符串或路径对象或 None
    },
    prefer_skip_nested_validation=True,
)
def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.

    Parameters
    ----------
    data_home : str or path-like, default=None
        The path to scikit-learn data directory. If `None`, the default path
        is `~/scikit_learn_data`.

    Examples
    --------
    >>> from sklearn.datasets import clear_data_home
    >>> clear_data_home()  # doctest: +SKIP
    """
    data_home = get_data_home(data_home)  # 获取数据目录的路径
    shutil.rmtree(data_home)  # 递归删除目录及其内容


def _convert_data_dataframe(
    # 定义函数参数：
    # caller_name: 调用方的名称或标识符
    # data: 输入数据，通常是特征数据
    # target: 目标数据，通常是分类或回归的目标值
    # feature_names: 特征的名称列表或标识符
    # target_names: 目标值的名称列表或标识符
    # sparse_data: 是否使用稀疏数据表示（默认为 False）
):
    # 检查是否支持 Pandas 库，并在不支持时返回相应错误信息
    pd = check_pandas_support("{} with as_frame=True".format(caller_name))
    # 如果不是稀疏数据，则使用给定的数据和特征名创建数据帧
    if not sparse_data:
        data_df = pd.DataFrame(data, columns=feature_names, copy=False)
    else:
        # 如果是稀疏数据，则使用稀疏矩阵创建稀疏数据帧
        data_df = pd.DataFrame.sparse.from_spmatrix(data, columns=feature_names)

    # 使用目标数据和目标名创建目标数据帧
    target_df = pd.DataFrame(target, columns=target_names)
    # 将数据帧和目标数据帧合并为一个组合数据帧
    combined_df = pd.concat([data_df, target_df], axis=1)
    # 提取特征数据（X）和目标数据（y）从组合数据帧中
    X = combined_df[feature_names]
    y = combined_df[target_names]
    # 如果目标数据是二维的，则将其转换为一维
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    # 返回组合数据帧、特征数据和目标数据
    return combined_df, X, y


@validate_params(
    {
        "container_path": [str, os.PathLike],
        "description": [str, None],
        "categories": [list, None],
        "load_content": ["boolean"],
        "shuffle": ["boolean"],
        "encoding": [str, None],
        "decode_error": [StrOptions({"strict", "ignore", "replace"})],
        "random_state": ["random_state"],
        "allowed_extensions": [list, None],
    },
    prefer_skip_nested_validation=True,
)
# 加载文件函数，从给定路径加载文本文件，以其子文件夹作为分类名称
def load_files(
    container_path,
    *,
    description=None,
    categories=None,
    load_content=True,
    shuffle=True,
    encoding=None,
    decode_error="strict",
    random_state=0,
    allowed_extensions=None,
):
    """Load text files with categories as subfolder names.

    Individual samples are assumed to be files stored a two levels folder
    structure such as the following:

        container_folder/
            category_1_folder/
                file_1.txt
                file_2.txt
                ...
                file_42.txt
            category_2_folder/
                file_43.txt
                file_44.txt
                ...

    The folder names are used as supervised signal label names. The individual
    file names are not important.

    This function does not try to extract features into a numpy array or scipy
    sparse matrix. In addition, if load_content is false it does not try to
    load the files in memory.

    To use text files in a scikit-learn classification or clustering algorithm,
    you will need to use the :mod:`~sklearn.feature_extraction.text` module to
    build a feature extraction transformer that suits your problem.

    If you set load_content=True, you should also specify the encoding of the
    text using the 'encoding' parameter. For many modern text files, 'utf-8'
    will be the correct encoding. If you leave encoding equal to None, then the
    content will be made of bytes instead of Unicode, and you will not be able
    to use most functions in :mod:`~sklearn.feature_extraction.text`.

    Similar feature extractors should be built for other kind of unstructured
    data input such as images, audio, video, ...

    If you want files with a specific file extension (e.g. `.txt`) then you
    can pass a list of those file extensions to `allowed_extensions`.

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    """
    # container_path : str
    #     主文件夹路径，其中每个子文件夹代表一个类别。

    # description : str, default=None
    #     描述数据集特征的段落：数据集的来源、引用等。

    # categories : list of str, default=None
    #     如果为 None（默认），加载所有类别。如果不为 None，则加载指定的类别名称，忽略其他类别。

    # load_content : bool, default=True
    #     是否加载不同文件的内容。如果为 True，则返回的数据结构中包含一个 'data' 属性，其中包含文本信息。
    #     如果为 False，则返回的数据结构中的 'filenames' 属性给出文件的路径。

    # shuffle : bool, default=True
    #     是否对数据进行洗牌：对于那些假定样本独立同分布（i.i.d.）的模型（例如随机梯度下降），这可能很重要。

    # encoding : str, default=None
    #     如果为 None，则不尝试解码文件内容（例如图像或其他非文本内容）。如果不为 None，
    #     则用于解码文本文件为 Unicode 的编码，如果 load_content 为 True。

    # decode_error : {'strict', 'ignore', 'replace'}, default='strict'
    #     如果给定的编码包含不属于指定编码的字符，对字节序列进行分析时应执行的操作指令。
    #     作为关键字参数 'errors' 传递给 bytes.decode。

    # random_state : int, RandomState instance or None, default=0
    #     确定数据集洗牌时的随机数生成。传递一个整数以实现多次函数调用间的可重现输出。
    #     参见“术语表”。

    # allowed_extensions : list of str, default=None
    #     想要过滤加载的文件的所需文件扩展名列表。

    target = []
    target_names = []
    filenames = []

    # 获取主文件夹下的所有文件夹列表
    folders = [
        f for f in sorted(listdir(container_path)) if isdir(join(container_path, f))
    ]

    # 如果指定了 categories，只保留其中指定的类别文件夹
    if categories is not None:
        folders = [f for f in folders if f in categories]

    # 如果指定了 allowed_extensions，将其转化为 frozenset 提高查找效率
    # 对每个文件夹进行遍历，使用enumerate函数获取索引和文件夹名
    for label, folder in enumerate(folders):
        # 将文件夹名添加到目标名称列表中
        target_names.append(folder)
        # 构建当前文件夹的完整路径
        folder_path = join(container_path, folder)
        # 获取文件夹下所有文件的列表，并按字母顺序排序
        files = sorted(listdir(folder_path))
        
        # 如果指定了允许的文件扩展名，则仅选择符合条件的文件路径
        if allowed_extensions is not None:
            documents = [
                join(folder_path, file)
                for file in files
                if os.path.splitext(file)[1] in allowed_extensions
            ]
        else:
            # 否则选择所有文件路径
            documents = [join(folder_path, file) for file in files]
        
        # 将目标数组扩展，使其与当前文件夹中的文件数量相匹配
        target.extend(len(documents) * [label])
        # 将文件路径列表扩展到整体文件名列表中
        filenames.extend(documents)

    # 将文件名列表转换为NumPy数组，以便进行高级索引操作
    filenames = np.array(filenames)
    # 将目标数组也转换为NumPy数组
    target = np.array(target)

    # 如果需要打乱数据集
    if shuffle:
        # 检查随机数生成器状态
        random_state = check_random_state(random_state)
        # 创建包含所有文件名索引的数组
        indices = np.arange(filenames.shape[0])
        # 使用随机状态对象对索引数组进行洗牌操作
        random_state.shuffle(indices)
        # 使用洗牌后的索引对文件名和目标数组进行重排
        filenames = filenames[indices]
        target = target[indices]

    # 如果需要加载文件内容
    if load_content:
        # 创建空的数据列表，用于存储文件内容
        data = []
        # 遍历所有文件名
        for filename in filenames:
            # 读取文件的二进制内容并添加到数据列表中
            data.append(Path(filename).read_bytes())
        # 如果指定了编码方式，则根据编码方式对数据进行解码
        if encoding is not None:
            data = [d.decode(encoding, decode_error) for d in data]
        # 返回一个Bunch对象，包含数据、文件名列表、目标名称列表、目标数组和描述信息
        return Bunch(
            data=data,
            filenames=filenames,
            target_names=target_names,
            target=target,
            DESCR=description,
        )

    # 如果不需要加载文件内容，则返回一个Bunch对象，包含文件名列表、目标名称列表、目标数组和描述信息
    return Bunch(
        filenames=filenames, target_names=target_names, target=target, DESCR=description
    )
def load_csv_data(
    data_file_name,
    *,
    data_module=DATA_MODULE,
    descr_file_name=None,
    descr_module=DESCR_MODULE,
    encoding="utf-8",
):
    """Loads `data_file_name` from `data_module` with `importlib.resources`.

    Parameters
    ----------
    data_file_name : str
        Name of csv file to be loaded from `data_module/data_file_name`.
        For example `'wine_data.csv'`.

    data_module : str or module, default='sklearn.datasets.data'
        Module where data lives. The default is `'sklearn.datasets.data'`.

    descr_file_name : str, default=None
        Name of rst file to be loaded from `descr_module/descr_file_name`.
        For example `'wine_data.rst'`. See also :func:`load_descr`.
        If not None, also returns the corresponding description of
        the dataset.

    descr_module : str or module, default='sklearn.datasets.descr'
        Module where `descr_file_name` lives. See also :func:`load_descr`.
        The default is `'sklearn.datasets.descr'`.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features)
        A 2D array with each row representing one sample and each column
        representing the features of a given sample.

    target : ndarry of shape (n_samples,)
        A 1D array holding target variables for all the samples in `data`.
        For example target[0] is the target variable for data[0].

    target_names : ndarry of shape (n_samples,)
        A 1D array containing the names of the classifications. For example
        target_names[0] is the name of the target[0] class.

    descr : str, optional
        Description of the dataset (the content of `descr_file_name`).
        Only returned if `descr_file_name` is not None.

    encoding : str, optional
        Text encoding of the CSV file.

        .. versionadded:: 1.4
    """
    # 构建数据文件的路径
    data_path = resources.files(data_module) / data_file_name
    # 使用指定编码打开 CSV 文件
    with data_path.open("r", encoding="utf-8") as csv_file:
        # 创建 CSV 读取器
        data_file = csv.reader(csv_file)
        # 读取并解析文件的第一行数据
        temp = next(data_file)
        # 提取样本数量和特征数量
        n_samples = int(temp[0])
        n_features = int(temp[1])
        # 提取目标变量的名称数组
        target_names = np.array(temp[2:])
        # 创建空的数据和目标变量数组
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)

        # 逐行读取数据文件并填充数据和目标变量数组
        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=int)

    # 如果未提供描述文件名，则返回数据、目标变量和目标变量名称数组
    if descr_file_name is None:
        return data, target, target_names
    else:
        # 否则，加载描述文件并返回数据、目标变量、目标变量名称数组和描述信息
        assert descr_module is not None
        descr = load_descr(descr_module=descr_module, descr_file_name=descr_file_name)
        return data, target, target_names, descr


def load_gzip_compressed_csv_data(
    data_file_name,
    *,
    data_module=DATA_MODULE,
    descr_file_name=None,
    descr_module=DESCR_MODULE,
    encoding="utf-8",
    **kwargs,
):
    """Loads gzip-compressed with `importlib.resources`.

    1) Open resource file with `importlib.resources.open_binary`
    # 使用 gzip.open 解压文件对象
    compressed_file = gzip.open(compressed_file, mode="rt", encoding=encoding)
    # 使用 np.loadtxt 加载解压后的数据文件，传递所有的关键字参数
    data = np.loadtxt(compressed_file, **kwargs)

if descr_file_name is None:
    # 如果没有提供描述文件名，返回加载的数据
    return data
else:
    # 否则，确保描述模块不为空，并加载描述文件
    assert descr_module is not None
    descr = load_descr(descr_module=descr_module, descr_file_name=descr_file_name)
    # 返回加载的数据和对应的描述信息
    return data, descr
def load_descr(descr_file_name, *, descr_module=DESCR_MODULE, encoding="utf-8"):
    """Load `descr_file_name` from `descr_module` with `importlib.resources`.

    Parameters
    ----------
    descr_file_name : str, default=None
        Name of rst file to be loaded from `descr_module/descr_file_name`.
        For example `'wine_data.rst'`. See also :func:`load_descr`.
        If not None, also returns the corresponding description of
        the dataset.

    descr_module : str or module, default='sklearn.datasets.descr'
        Module where `descr_file_name` lives. See also :func:`load_descr`.
        The default  is `'sklearn.datasets.descr'`.

    encoding : str, default="utf-8"
        Name of the encoding that `descr_file_name` will be decoded with.
        The default is 'utf-8'.

        .. versionadded:: 1.4

    Returns
    -------
    fdescr : str
        Content of `descr_file_name`.
    """
    # 构建文件路径，使用 importlib.resources 读取 `descr_file_name` 的内容
    path = resources.files(descr_module) / descr_file_name
    # 返回文件内容作为字符串
    return path.read_text(encoding=encoding)


@validate_params(
    {
        "return_X_y": ["boolean"],
        "as_frame": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_wine(*, return_X_y=False, as_frame=False):
    """Load and return the wine dataset (classification).

    .. versionadded:: 0.18

    The wine dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class        [59,71,48]
    Samples total                  178
    Dimensionality                  13
    Features            real, positive
    =================   ==============

    The copy of UCI ML Wine Data Set dataset is downloaded and modified to fit
    standard format from:
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

    Read more in the :ref:`User Guide <wine_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    """
    # 省略部分代码，不输出其它东西
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (178, 13)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, Series} of shape (178,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        frame: DataFrame of shape (178, 14)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        DESCR: str
            The full description of the dataset.
        
        This section describes the attributes of the `data` object, including
        data matrices, target information, feature names, and related details.

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarrays by default. The first contains a 2D array of shape
        (178, 13) with each row representing one sample and each column representing
        the features. The second array of shape (178,) contains the target samples.
        
        Specifies the return value format when `return_X_y` is True, providing
        data and target arrays as a tuple.

    Examples
    --------
    Let's say you are interested in the samples 10, 80, and 140, and want to
    know their class name.

    >>> from sklearn.datasets import load_wine
    >>> data = load_wine()
    >>> data.target[[10, 80, 140]]
    array([0, 1, 2])
    >>> list(data.target_names)
    ['class_0', 'class_1', 'class_2']
    """

    # Load data and metadata from CSV and RST files
    data, target, target_names, fdescr = load_csv_data(
        data_file_name="wine_data.csv", descr_file_name="wine_data.rst"
    )

    # Define the feature names corresponding to the dataset columns
    feature_names = [
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od280/od315_of_diluted_wines",
        "proline",
    ]

    # Initialize frame as None and define target_columns
    frame = None
    target_columns = [
        "target",
    ]

    # Convert data and target to a DataFrame if `as_frame=True`
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_wine", data, target, feature_names, target_columns
        )

    # Return data and target as a tuple if `return_X_y=True`
    if return_X_y:
        return data, target

    # Return a Bunch object containing all relevant data and metadata
    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR=fdescr,
        feature_names=feature_names,
    )
# 使用 @validate_params 装饰器验证参数，确保 return_X_y 和 as_frame 是布尔类型
@validate_params(
    {"return_X_y": ["boolean"], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
# 定义 load_iris 函数，加载并返回鸢尾花数据集（分类）
def load_iris(*, return_X_y=False, as_frame=False):
    """Load and return the iris dataset (classification).

    The iris dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============

    Read more in the :ref:`User Guide <iris_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (150, 4)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, Series} of shape (150,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        frame: DataFrame of shape (150, 5)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.

            .. versionadded:: 0.20

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarray. The first containing a 2D array of shape
        (n_samples, n_features) with each row representing one sample and
        each column representing the features. The second ndarray of shape
        (n_samples,) containing the target samples.

        .. versionadded:: 0.18

    Notes
    -----
        .. versionchanged:: 0.20
            Fixed two wrong data points according to Fisher's paper.
            The new version is the same as in R, but not as in the UCI
            Machine Learning Repository.

    Examples
    --------
    Let's say you are interested in the samples 10, 25, and 50, and want to
    know their class name.

    >>> from sklearn.datasets import load_iris
    # 加载 iris 数据集
    data_file_name = "iris.csv"
    # 调用 load_csv_data 函数加载 CSV 数据和相关描述信息
    data, target, target_names, fdescr = load_csv_data(
        data_file_name=data_file_name, descr_file_name="iris.rst"
    )

    # 定义特征名列表
    feature_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    # 初始化数据框架为空
    frame = None
    # 目标列名称列表
    target_columns = [
        "target",
    ]
    # 如果需要返回数据框架（DataFrame）
    if as_frame:
        # 将数据转换为数据框架（DataFrame）
        frame, data, target = _convert_data_dataframe(
            "load_iris", data, target, feature_names, target_columns
        )

    # 如果需要返回特征数据和目标数据
    if return_X_y:
        return data, target

    # 返回包含数据集各项信息的 Bunch 对象
    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file_name,
        data_module=DATA_MODULE,
    )
# 使用装饰器 validate_params 对 load_breast_cancer 函数进行参数验证
@validate_params(
    {"return_X_y": ["boolean"], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
# 定义 load_breast_cancer 函数，加载并返回乳腺癌数据集（分类）
def load_breast_cancer(*, return_X_y=False, as_frame=False):
    """Load and return the breast cancer wisconsin dataset (classification).

    The breast cancer dataset is a classic and very easy binary classification
    dataset.

    =================   ==============
    Classes                          2
    Samples per class    212(M),357(B)
    Samples total                  569
    Dimensionality                  30
    Features            real, positive
    =================   ==============

    The copy of UCI ML Breast Cancer Wisconsin (Diagnostic) dataset is
    downloaded from:
    https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

    Read more in the :ref:`User Guide <breast_cancer_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (569, 30)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target : {ndarray, Series} of shape (569,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names : ndarray of shape (30,)
            The names of the dataset columns.
        target_names : ndarray of shape (2,)
            The names of target classes.
        frame : DataFrame of shape (569, 31)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        DESCR : str
            The full description of the dataset.
        filename : str
            The path to the location of the data.

            .. versionadded:: 0.20

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarrays by default. The first contains a 2D ndarray of
        shape (569, 30) with each row representing one sample and each column
        representing the features. The second ndarray of shape (569,) contains
        the target samples.  If `as_frame=True`, both arrays are pandas objects,
        i.e. `X` a dataframe and `y` a series.

        .. versionadded:: 0.18

    Examples
    --------

    """
    # 假设你对乳腺癌数据集的样本编号为10、50和85感兴趣，并想知道它们的类别名称。
    
    # 导入 load_breast_cancer 函数从 sklearn.datasets 中
    from sklearn.datasets import load_breast_cancer
    # 载入乳腺癌数据集
    data = load_breast_cancer()
    # 获取样本索引为10、50和85的目标变量（类别标签）
    data.target[[10, 50, 85]]
    # 输出为：array([0, 1, 0])
    
    # 获取目标变量的所有可能类别名称
    list(data.target_names)
    # 输出为：['malignant', 'benign']
    """
    data_file_name = "breast_cancer.csv"
    # 载入 CSV 格式的数据和其对应的目标变量、目标变量名称以及数据描述信息
    data, target, target_names, fdescr = load_csv_data(
        data_file_name=data_file_name, descr_file_name="breast_cancer.rst"
    )
    
    # 定义特征名称列表，包含乳腺癌数据集的所有特征名称
    feature_names = np.array([
        "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
        "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
        "mean fractal dimension", "radius error", "texture error", "perimeter error",
        "area error", "smoothness error", "compactness error", "concavity error",
        "concave points error", "symmetry error", "fractal dimension error", "worst radius",
        "worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness",
        "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
    ])
    
    # 初始化 frame 为 None
    frame = None
    # 定义目标变量列名列表，包含 "target"
    target_columns = ["target"]
    
    # 如果 as_frame 参数为 True，则将数据和目标变量转换为 DataFrame 格式
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_breast_cancer", data, target, feature_names, target_columns
        )
    
    # 如果 return_X_y 参数为 True，则返回数据和目标变量
    if return_X_y:
        return data, target
    
    # 返回一个 Bunch 对象，包含数据集的各种信息和内容
    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file_name,
        data_module=DATA_MODULE,
    )
# 使用装饰器 @validate_params 对 load_digits 函数进行参数验证，确保参数符合以下规定
@validate_params(
    {
        "n_class": [Interval(Integral, 1, 10, closed="both")],  # n_class 参数必须为介于 1 到 10 之间的整数，包括端点
        "return_X_y": ["boolean"],  # return_X_y 参数必须为布尔值
        "as_frame": ["boolean"],  # as_frame 参数必须为布尔值
    },
    prefer_skip_nested_validation=True,  # 优先跳过嵌套验证
)
def load_digits(*, n_class=10, return_X_y=False, as_frame=False):
    """Load and return the digits dataset (classification).

    Each datapoint is a 8x8 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class             ~180
    Samples total                 1797
    Dimensionality                  64
    Features             integers 0-16
    =================   ==============

    This is a copy of the test set of the UCI ML hand-written digits datasets
    https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

    Read more in the :ref:`User Guide <digits_dataset>`.

    Parameters
    ----------
    n_class : int, default=10
        The number of classes to return. Between 0 and 10.

    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (1797, 64)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1797,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.

            .. versionadded:: 0.20

        frame: DataFrame of shape (1797, 65)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        images: {ndarray} of shape (1797, 8, 8)
            The raw image data.
        DESCR: str
            The full description of the dataset.
    """
    (data, target) : tuple if ``return_X_y`` is True
        返回一个元组，当 ``return_X_y`` 为 True 时。默认情况下，第一个元素是一个形状为 (1797, 64) 的二维 ndarray，每行表示一个样本，每列表示特征。第二个元素是形状为 (1797,) 的一维 ndarray，包含目标样本值。如果 `as_frame=True`，则两个数组都是 pandas 对象，即 `X` 是一个 dataframe，`y` 是一个 series。

        .. versionadded:: 0.18

    Examples
    --------
    To load the data and visualize the images::

        >>> from sklearn.datasets import load_digits
        >>> digits = load_digits()
        >>> print(digits.data.shape)
        (1797, 64)
        >>> import matplotlib.pyplot as plt
        >>> plt.gray()
        >>> plt.matshow(digits.images[0])
        <...>
        >>> plt.show()
    """

    # 加载经过 gzip 压缩的 CSV 数据和对应的描述文件
    data, fdescr = load_gzip_compressed_csv_data(
        data_file_name="digits.csv.gz", descr_file_name="digits.rst", delimiter=","
    )

    # 从加载的数据中提取目标变量，并转换为整数类型
    target = data[:, -1].astype(int, copy=False)

    # 提取除了目标变量外的所有特征数据
    flat_data = data[:, :-1]

    # 将特征数据展示为图像格式的 ndarray
    images = flat_data.view()
    images.shape = (-1, 8, 8)

    # 如果指定的类别数 n_class 小于 10，则只保留目标变量对应类别小于 n_class 的数据
    if n_class < 10:
        idx = target < n_class
        flat_data, target = flat_data[idx], target[idx]
        images = images[idx]

    # 创建特征名称列表，格式为 "pixel_row_idx_col_idx"
    feature_names = [
        "pixel_{}_{}".format(row_idx, col_idx)
        for row_idx in range(8)
        for col_idx in range(8)
    ]

    frame = None
    target_columns = [
        "target",
    ]
    
    # 如果 as_frame=True，将数据转换为 pandas dataframe 格式
    if as_frame:
        frame, flat_data, target = _convert_data_dataframe(
            "load_digits", flat_data, target, feature_names, target_columns
        )

    # 如果 return_X_y=True，则返回特征数据和目标变量
    if return_X_y:
        return flat_data, target

    # 否则，返回一个 Bunch 对象，包含特征数据、目标变量、DataFrame、特征名称、目标名称和图像数据
    return Bunch(
        data=flat_data,
        target=target,
        frame=frame,
        feature_names=feature_names,
        target_names=np.arange(10),
        images=images,
        DESCR=fdescr,
    )
# 使用装饰器对函数参数进行验证，确保参数的类型正确
@validate_params(
    {"return_X_y": ["boolean"], "as_frame": ["boolean"], "scaled": ["boolean"]},
    prefer_skip_nested_validation=True,
)
# 定义加载糖尿病数据集的函数，返回用于回归的数据集
def load_diabetes(*, return_X_y=False, as_frame=False, scaled=True):
    """Load and return the diabetes dataset (regression).

    ==============   ==================
    Samples total    442
    Dimensionality   10
    Features         real, -.2 < x < .2
    Targets          integer 25 - 346
    ==============   ==================

    .. note::
       每个特征的含义（如 `feature_names`）可能不明确，特别是原始数据集的文档
       并未明确说明。我们根据科学文献提供了看似正确的信息。

    详细信息请参阅：:ref:`用户指南 <diabetes_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        如果为 True，则返回 ``(data, target)`` 而不是一个 Bunch 对象。
        有关 `data` 和 `target` 对象的更多信息，请参见下文。

        .. versionadded:: 0.18

    as_frame : bool, default=False
        如果为 True，则数据是包含适当数据类型列（数值类型）的 pandas DataFrame。
        target 是一个 pandas DataFrame 或 Series，具体取决于目标列的数量。
        如果 `return_X_y` 为 True，则 (`data`, `target`) 将会是 pandas
        DataFrames 或 Series，如下文所述。

        .. versionadded:: 0.23

    scaled : bool, default=True
        如果为 True，则特征变量将进行均值中心化，并按照标准差乘以样本数的平方根进行缩放。
        如果为 False，则返回原始数据作为特征变量。

        .. versionadded:: 1.1

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        类似于字典的对象，具有以下属性。

        data : {ndarray, dataframe} of shape (442, 10)
            数据矩阵。如果 `as_frame=True`，则 `data` 将是一个 pandas DataFrame。
        target: {ndarray, Series} of shape (442,)
            回归目标。如果 `as_frame=True`，则 `target` 将是一个 pandas Series。
        feature_names: list
            数据集列的名称。
        frame: DataFrame of shape (442, 11)
            当 `as_frame=True` 时才存在。包含 `data` 和 `target` 的 DataFrame。

            .. versionadded:: 0.23
        DESCR: str
            数据集的完整描述。
        data_filename: str
            数据文件的路径。
        target_filename: str
            目标文件的路径。

    (data, target) : tuple if ``return_X_y`` is True
        如果 ``return_X_y`` 为 True，则返回两个 ndarray，形状为 (n_samples, n_features)
        每行代表一个样本，每列代表一个样本的特征和/或目标。

        .. versionadded:: 0.18

    Examples
    --------
    """
    data_filename = "diabetes_data_raw.csv.gz"
    target_filename = "diabetes_target.csv.gz"
    # 载入经过 gzip 压缩的 CSV 数据文件，分别为特征数据和目标数据
    data = load_gzip_compressed_csv_data(data_filename)
    target = load_gzip_compressed_csv_data(target_filename)

    if scaled:
        # 若需要进行数据缩放，则对特征数据进行缩放处理
        data = scale(data, copy=False)
        data /= data.shape[0] ** 0.5

    # 载入描述性文件
    fdescr = load_descr("diabetes.rst")

    feature_names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

    frame = None
    target_columns = [
        "target",
    ]
    if as_frame:
        # 如果需要返回 DataFrame 格式的数据集，则转换数据和目标为 DataFrame
        frame, data, target = _convert_data_dataframe(
            "load_diabetes", data, target, feature_names, target_columns
        )

    if return_X_y:
        # 如果需要返回特征数据和目标数据，则直接返回它们
        return data, target

    # 否则，返回一个命名元组 Bunch，包含特征数据、目标数据、DataFrame、描述性文件等信息
    return Bunch(
        data=data,
        target=target,
        frame=frame,
        DESCR=fdescr,
        feature_names=feature_names,
        data_filename=data_filename,
        target_filename=target_filename,
        data_module=DATA_MODULE,
    )
@validate_params(
    {
        "return_X_y": ["boolean"],   # 参数验证装饰器，验证 return_X_y 应为布尔类型
        "as_frame": ["boolean"],     # 参数验证装饰器，验证 as_frame 应为布尔类型
    },
    prefer_skip_nested_validation=True,  # 参数验证装饰器的设置，优先跳过嵌套验证
)
def load_linnerud(*, return_X_y=False, as_frame=False):
    """Load and return the physical exercise Linnerud dataset.

    This dataset is suitable for multi-output regression tasks.

    ==============   ============================
    Samples total    20
    Dimensionality   3 (for both data and target)
    Features         integer
    Targets          integer
    ==============   ============================

    Read more in the :ref:`User Guide <linnerrud_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (20, 3)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, dataframe} of shape (20, 3)
            The regression targets. If `as_frame=True`, `target` will be
            a pandas DataFrame.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of the target columns.
        frame: DataFrame of shape (20, 6)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        DESCR: str
            The full description of the dataset.
        data_filename: str
            The path to the location of the data.
        target_filename: str
            The path to the location of the target.

            .. versionadded:: 0.20

    (data, target) : tuple if ``return_X_y`` is True
        Returns a tuple of two ndarrays or dataframe of shape
        `(20, 3)`. Each row represents one sample and each column represents the
        features in `X` and a target in `y` of a given sample.

        .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.datasets import load_linnerud
    >>> linnerud = load_linnerud()
    >>> linnerud.data.shape
    (20, 3)
    >>> linnerud.target.shape
    (20, 3)
    """
    data_filename = "linnerud_exercise.csv"    # 数据文件名
    target_filename = "linnerud_physiological.csv"   # 目标文件名

    data_module_path = resources.files(DATA_MODULE)   # 获取数据模块的文件路径
    # Read header and data   # 读取文件头和数据
    data_path = data_module_path / data_filename
    # 构建数据文件的完整路径

    with data_path.open("r", encoding="utf-8") as f:
        # 打开数据文件，并指定编码为 UTF-8

        header_exercise = f.readline().split()
        # 读取文件的第一行，将其按空格分割为列表，作为运动数据的列名

        f.seek(0)  # reset file obj
        # 将文件指针移到文件开头，准备重新读取文件数据

        data_exercise = np.loadtxt(f, skiprows=1)
        # 使用 NumPy 加载文件中的数据，跳过第一行（已经作为列名处理过）

    target_path = data_module_path / target_filename
    # 构建目标文件的完整路径

    with target_path.open("r", encoding="utf-8") as f:
        # 打开目标文件，并指定编码为 UTF-8

        header_physiological = f.readline().split()
        # 读取文件的第一行，将其按空格分割为列表，作为生理数据的列名

        f.seek(0)  # reset file obj
        # 将文件指针移到文件开头，准备重新读取文件数据

        data_physiological = np.loadtxt(f, skiprows=1)
        # 使用 NumPy 加载文件中的数据，跳过第一行（已经作为列名处理过）

    fdescr = load_descr("linnerud.rst")
    # 载入与数据集相关的描述信息

    frame = None
    if as_frame:
        # 如果需要返回 DataFrame 格式数据

        (frame, data_exercise, data_physiological) = _convert_data_dataframe(
            "load_linnerud",
            data_exercise,
            data_physiological,
            header_exercise,
            header_physiological,
        )
        # 调用函数将数据转换为 DataFrame 格式

    if return_X_y:
        return data_exercise, data_physiological
        # 如果只需返回原始数据数组，则直接返回运动数据和生理数据

    return Bunch(
        data=data_exercise,
        # 返回处理后的运动数据

        feature_names=header_exercise,
        # 返回运动数据的列名

        target=data_physiological,
        # 返回处理后的生理数据

        target_names=header_physiological,
        # 返回生理数据的列名

        frame=frame,
        # 返回 DataFrame 格式的数据（如果有）

        DESCR=fdescr,
        # 返回数据集描述信息

        data_filename=data_filename,
        # 返回数据文件名

        target_filename=target_filename,
        # 返回目标文件名

        data_module=DATA_MODULE,
        # 返回数据模块名
    )
    # 返回一个命名元组 Bunch，包含所有数据集相关的信息
def load_sample_images():
    """Load sample images for image manipulation.

    Loads both, ``china`` and ``flower``.

    Read more in the :ref:`User Guide <sample_images>`.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        images : list of ndarray of shape (427, 640, 3)
            The two sample image.
        filenames : list
            The filenames for the images.
        DESCR : str
            The full description of the dataset.

    Examples
    --------
    To load the data and visualize the images:

    >>> from sklearn.datasets import load_sample_images
    >>> dataset = load_sample_images()     #doctest: +SKIP
    >>> len(dataset.images)                #doctest: +SKIP
    2
    >>> first_img_data = dataset.images[0] #doctest: +SKIP
    >>> first_img_data.shape               #doctest: +SKIP
    (427, 640, 3)
    >>> first_img_data.dtype               #doctest: +SKIP
    dtype('uint8')
    """
    try:
        from PIL import Image  # 尝试导入 PIL 库中的 Image 模块
    except ImportError:
        raise ImportError(
            "The Python Imaging Library (PIL) is required to load data "
            "from jpeg files. Please refer to "
            "https://pillow.readthedocs.io/en/stable/installation.html "
            "for installing PIL."
        )

    # 使用 load_descr 函数加载 README.txt 中的描述信息
    descr = load_descr("README.txt", descr_module=IMAGES_MODULE)

    filenames, images = [], []

    # 获取 IMAGES_MODULE 中所有的 .jpg 文件路径并按名称排序
    jpg_paths = sorted(
        resource
        for resource in resources.files(IMAGES_MODULE).iterdir()
        if resource.is_file() and resource.match("*.jpg")
    )

    # 遍历所有 .jpg 文件路径
    for path in jpg_paths:
        filenames.append(str(path))  # 将文件路径转换为字符串并加入 filenames 列表中
        with path.open("rb") as image_file:
            pil_image = Image.open(image_file)  # 使用 PIL 打开图像文件
            image = np.asarray(pil_image)  # 将 PIL 图像转换为 numpy 数组
        images.append(image)  # 将转换后的 numpy 数组加入 images 列表中

    # 返回一个 Bunch 对象，包含加载的图像数据、文件名列表和描述信息
    return Bunch(images=images, filenames=filenames, DESCR=descr)


@validate_params(
    {
        "image_name": [StrOptions({"china.jpg", "flower.jpg"})],
    },
    prefer_skip_nested_validation=True,
)
def load_sample_image(image_name):
    """Load the numpy array of a single sample image.

    Read more in the :ref:`User Guide <sample_images>`.

    Parameters
    ----------
    image_name : {`china.jpg`, `flower.jpg`}
        The name of the sample image loaded.

    Returns
    -------
    img : 3D array
        The image as a numpy array: height x width x color.

    Examples
    --------

    >>> from sklearn.datasets import load_sample_image
    >>> china = load_sample_image('china.jpg')   # doctest: +SKIP
    >>> china.dtype                              # doctest: +SKIP
    dtype('uint8')
    >>> china.shape                              # doctest: +SKIP
    (427, 640, 3)
    >>> flower = load_sample_image('flower.jpg') # doctest: +SKIP
    >>> flower.dtype                             # doctest: +SKIP
    dtype('uint8')
    >>> flower.shape                             # doctest: +SKIP
    (427, 640, 3)
    """
    # 载入示例图片数据集
    images = load_sample_images()
    # 初始化索引变量为 None
    index = None
    # 遍历示例图片数据集中的文件名列表，同时获取索引 i 和文件名 filename
    for i, filename in enumerate(images.filenames):
        # 检查当前文件名是否以指定的图像名称 image_name 结尾
        if filename.endswith(image_name):
            # 如果是，将当前索引 i 赋值给 index，并结束循环
            index = i
            break
    # 如果未找到匹配的图像名称，则抛出属性错误异常
    if index is None:
        raise AttributeError("Cannot find sample image: %s" % image_name)
    # 返回示例图片数据集中匹配的图像数据
    return images.images[index]
# 返回适用于 Python 3 pickle 的文件名
def _pkl_filepath(*args, **kwargs):
    """
    args[-1] 是期望的 ".pkl" 文件名。为了与旧版 scikit-learn 兼容，在扩展名前插入后缀。

    _pkl_filepath('/path/to/folder', 'filename.pkl') 返回
    '/path/to/folder/filename_py3.pkl'
    """
    py3_suffix = kwargs.get("py3_suffix", "_py3")  # 获取可选参数 py3_suffix，默认为 "_py3"
    basename, ext = splitext(args[-1])  # 拆分文件路径的基本名称和扩展名
    basename += py3_suffix  # 在基本名称后添加后缀
    new_args = args[:-1] + (basename + ext,)  # 构建新的文件路径参数
    return join(*new_args)  # 返回连接后的新文件路径


# 计算给定路径文件的 SHA256 哈希值
def _sha256(path):
    """计算路径处文件的 SHA256 哈希值."""
    sha256hash = hashlib.sha256()  # 创建 SHA256 哈希对象
    chunk_size = 8192  # 每次读取文件的块大小为 8192 字节
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)  # 读取文件块
            if not buffer:
                break
            sha256hash.update(buffer)  # 更新哈希对象
    return sha256hash.hexdigest()  # 返回十六进制表示的哈希值


# 辅助函数，用于下载远程数据集到指定路径并验证其完整性
def _fetch_remote(remote, dirname=None, n_retries=3, delay=1):
    """
    辅助函数，下载远程数据集到路径 dirname

    从 remote 的 URL 获取指定数据集，以 remote 的文件名保存到路径，
    根据下载文件的 SHA256 校验和验证其完整性。

    Parameters
    ----------
    remote : RemoteFileMetadata
        包含远程数据集的元信息的命名元组: url, filename 和 checksum

    dirname : str
        要保存文件的目录。

    n_retries : int, 默认为 3
        在遇到 HTTP 错误时的重试次数。

        .. versionadded:: 1.5

    delay : int, 默认为 1
        重试之间的延迟时间（秒）。

        .. versionadded:: 1.5

    Returns
    -------
    file_path: str
        创建文件的完整路径。
    """

    file_path = remote.filename if dirname is None else join(dirname, remote.filename)  # 设置文件的完整路径
    while True:
        try:
            urlretrieve(remote.url, file_path)  # 下载文件到指定路径
            break
        except (URLError, TimeoutError):
            if n_retries == 0:
                # 如果没有更多的重试次数，重新引发捕获的异常。
                raise
            warnings.warn(f"Retry downloading from url: {remote.url}")  # 发出警告，尝试重新下载
            n_retries -= 1
            time.sleep(delay)  # 等待一段时间后重试

    checksum = _sha256(file_path)  # 计算下载文件的 SHA256 校验和
    if remote.checksum != checksum:
        raise OSError(
            "{} 的 SHA256 校验和 ({}) "
            "与期望的不同 ({}), "
            "文件可能已损坏。".format(file_path, checksum, remote.checksum)
        )  # 如果校验和不匹配，则抛出异常表示文件可能已损坏
    return file_path  # 返回创建文件的完整路径
```