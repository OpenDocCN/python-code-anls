# `D:\src\scipysrc\scikit-learn\sklearn\datasets\_california_housing.py`

```
"""California housing dataset.

The original database is available from StatLib

    http://lib.stat.cmu.edu/datasets/

The data contains 20,640 observations on 9 variables.

This dataset contains the average house value as target variable
and the following input variables (features): average income,
housing average age, average rooms, average bedrooms, population,
average occupation, latitude, and longitude in that order.

References
----------

Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
Statistics and Probability Letters, 33 (1997) 291-297.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import logging                # 导入日志模块，用于记录程序运行时的日志信息
import tarfile                # 导入tarfile模块，用于处理tar压缩文件
from numbers import Integral, Real  # 从numbers模块导入Integral和Real类，用于参数验证
from os import PathLike, makedirs, remove  # 导入PathLike, makedirs, remove函数，用于文件和路径操作
from os.path import exists    # 从os.path模块导入exists函数，用于检查文件是否存在

import joblib                 # 导入joblib模块，用于对象持久化（序列化）
import numpy as np            # 导入NumPy库，用于数组操作

from ..utils import Bunch     # 导入自定义的Bunch类，用于封装数据集
from ..utils._param_validation import Interval, validate_params  # 导入参数验证相关的函数和类
from . import get_data_home   # 从当前包中导入get_data_home函数
from ._base import (          # 从当前包中导入多个函数和类
    RemoteFileMetadata,       # 远程文件元数据类
    _convert_data_dataframe,  # 数据帧转换函数
    _fetch_remote,             # 远程数据获取函数
    _pkl_filepath,            # 返回pickle文件路径函数
    load_descr,               # 加载描述信息函数
)

# The original data can be found at:
# https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz
ARCHIVE = RemoteFileMetadata(
    filename="cal_housing.tgz",  # 定义远程文件元数据对象，包含文件名、URL和校验和
    url="https://ndownloader.figshare.com/files/5976036",
    checksum="aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681",
)

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@validate_params(  # 参数验证装饰器，用于验证fetch_california_housing函数的输入参数
    {
        "data_home": [str, PathLike, None],  # data_home参数接受字符串、PathLike对象或None
        "download_if_missing": ["boolean"],  # download_if_missing参数接受布尔值
        "return_X_y": ["boolean"],           # return_X_y参数接受布尔值
        "as_frame": ["boolean"],             # as_frame参数接受布尔值
        "n_retries": [Interval(Integral, 1, None, closed="left")],  # n_retries参数接受整数，范围为1到无穷大
        "delay": [Interval(Real, 0.0, None, closed="neither")],     # delay参数接受实数，范围为大于0的任意值
    },
    prefer_skip_nested_validation=True,  # 设置为True，跳过嵌套验证
)
def fetch_california_housing(
    *,
    data_home=None,                    # 数据集缓存路径，默认为None
    download_if_missing=True,          # 如果数据集不存在是否下载，默认为True
    return_X_y=False,                  # 是否返回(X, y)形式数据，默认为False
    as_frame=False,                    # 是否返回数据帧格式，默认为False
    n_retries=3,                       # 下载重试次数，默认为3
    delay=1.0,                         # 下载延迟时间，默认为1.0秒
):
    """Load the California housing dataset (regression).

    ==============   ==============
    Samples total             20640
    Dimensionality                8
    Features                   real
    Target           real 0.15 - 5.
    ==============   ==============

    Read more in the :ref:`User Guide <california_housing_dataset>`.

    Parameters
    ----------
    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : bool, default=False
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

        .. versionadded:: 0.20
    """
    # 如果设置为True，则返回的数据包含在pandas DataFrame中，包括适当的数据类型（数值、字符串或分类）。
    # 如果目标有多个列，则返回的目标是pandas的DataFrame或Series。
    # 这是在版本0.23中添加的功能。
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target_columns.

        .. versionadded:: 0.23

    # 在遇到HTTP错误时的重试次数。
    # 这是在版本1.5中添加的功能。
    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.

        .. versionadded:: 1.5

    # 每次重试之间的延迟秒数。
    # 这是在版本1.5中添加的功能。
    delay : float, default=1.0
        Number of seconds between retries.

        .. versionadded:: 1.5

    # 返回一个:class:`~sklearn.utils.Bunch`对象，类似于字典，具有以下属性。
    # data: ndarray, shape (20640, 8)
    # 每行对应于8个特征值，按顺序排列。
    # 如果`as_frame`为True，则`data`是一个pandas对象。
    # target: 形状为(20640,)的numpy数组
    # 每个值对应于以100,000单位计算的平均房价。
    # 如果`as_frame`为True，则`target`是一个pandas对象。
    # feature_names: 长度为8的名称列表
    # 数据集中使用的有序特征名称数组。
    # DESCR: str
    # California房屋数据集的描述。
    # frame: pandas DataFrame
    # 仅在`as_frame=True`时存在。包含`data`和`target`的DataFrame。
    # 这是在版本0.23中添加的功能。
    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray, shape (20640, 8)
            Each row corresponding to the 8 feature values in order.
            If ``as_frame`` is True, ``data`` is a pandas object.
        target : numpy array of shape (20640,)
            Each value corresponds to the average
            house value in units of 100,000.
            If ``as_frame`` is True, ``target`` is a pandas object.
        feature_names : list of length 8
            Array of ordered feature names used in the dataset.
        DESCR : str
            Description of the California housing dataset.
        frame : pandas DataFrame
            Only present when `as_frame=True`. DataFrame with ``data`` and
            ``target``.

            .. versionadded:: 0.23

    # 如果`return_X_y`为True，则返回一个元组`(data, target)`，其中包含两个ndarray。
    # 第一个包含形状为(n_samples, n_features)的2D数组，每行表示一个样本，每列表示特征。
    # 第二个ndarray的形状为(n_samples,)，包含目标样本。
    # 这是在版本0.20中添加的功能。
    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarray. The first containing a 2D array of
        shape (n_samples, n_features) with each row representing one
        sample and each column representing the features. The second
        ndarray of shape (n_samples,) containing the target samples.

        .. versionadded:: 0.20

    # 数据集的说明和示例
    Notes
    -----

    This dataset consists of 20,640 samples and 9 features.

    Examples
    --------
    >>> from sklearn.datasets import fetch_california_housing
    >>> housing = fetch_california_housing()
    >>> print(housing.data.shape, housing.target.shape)
    (20640, 8) (20640,)
    >>> print(housing.feature_names[0:6])
    ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    """
    # 获取数据的主目录路径
    data_home = get_data_home(data_home=data_home)
    # 如果主目录不存在，则创建它
    if not exists(data_home):
        makedirs(data_home)

    # 构建保存数据文件的完整路径
    filepath = _pkl_filepath(data_home, "cal_housing.pkz")
    # 如果指定的文件路径不存在
    if not exists(filepath):
        # 如果设置了 download_if_missing 为 False，则抛出 OSError 异常
        if not download_if_missing:
            raise OSError("Data not found and `download_if_missing` is False")

        # 记录信息，下载 California housing 数据集到指定的 data_home 目录
        logger.info(
            "Downloading Cal. housing from {} to {}".format(ARCHIVE.url, data_home)
        )

        # 下载远程文件并返回下载后的本地路径
        archive_path = _fetch_remote(
            ARCHIVE,
            dirname=data_home,
            n_retries=n_retries,
            delay=delay,
        )

        # 使用 gzip 模式打开下载的 tar 文件
        with tarfile.open(mode="r:gz", name=archive_path) as f:
            # 从 tar 文件中提取并加载 California housing 数据集的内容
            cal_housing = np.loadtxt(
                f.extractfile("CaliforniaHousing/cal_housing.data"), delimiter=","
            )
            # 调整数据集的列顺序，使其与之前在 lib.stat.cmu.edu 上的资源相匹配
            columns_index = [8, 7, 2, 3, 4, 5, 6, 1, 0]
            cal_housing = cal_housing[:, columns_index]

            # 将处理后的数据集使用 joblib 存储到指定的 filepath，压缩等级为 6
            joblib.dump(cal_housing, filepath, compress=6)
        
        # 删除已下载的 tar 文件
        remove(archive_path)

    else:
        # 如果指定的文件路径存在，则加载已存储的 California housing 数据集
        cal_housing = joblib.load(filepath)

    # 定义 California housing 数据集的特征名称
    feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    # 将目标值和特征数据分别赋值给 target 和 data
    target, data = cal_housing[:, 0], cal_housing[:, 1:]

    # 计算平均房间数：总房间数 / 家庭数
    data[:, 2] /= data[:, 5]

    # 计算平均卧室数：总卧室数 / 家庭数
    data[:, 3] /= data[:, 5]

    # 计算平均占用率：人口数 / 家庭数
    data[:, 5] = data[:, 4] / data[:, 5]

    # 目标值转换为以 100,000 为单位
    target = target / 100000.0

    # 加载 California housing 数据集的描述信息
    descr = load_descr("california_housing.rst")

    # 将数据和目标值分别赋值给 X 和 y
    X = data
    y = target

    # 如果设置了 as_frame，则将数据和目标值转换为 DataFrame 格式
    frame = None
    target_names = [
        "MedHouseVal",
    ]
    if as_frame:
        frame, X, y = _convert_data_dataframe(
            "fetch_california_housing", data, target, feature_names, target_names
        )

    # 如果设置了 return_X_y，则返回特征数据 X 和目标值 y
    if return_X_y:
        return X, y

    # 否则，返回一个 Bunch 对象，包含数据集的各项信息
    return Bunch(
        data=X,
        target=y,
        frame=frame,
        target_names=target_names,
        feature_names=feature_names,
        DESCR=descr,
    )
```