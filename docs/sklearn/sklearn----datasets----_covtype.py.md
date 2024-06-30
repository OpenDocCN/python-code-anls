# `D:\src\scipysrc\scikit-learn\sklearn\datasets\_covtype.py`

```
# Forest covertype dataset.
# 
# A classic dataset for classification benchmarks, featuring categorical and
# real-valued features.
# 
# The dataset page is available from UCI Machine Learning Repository
# 
#     https://archive.ics.uci.edu/ml/datasets/Covertype
# 
# Courtesy of Jock A. Blackard and Colorado State University.

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Importing necessary libraries and modules
import logging
import os
from gzip import GzipFile
from numbers import Integral, Real
from os.path import exists, join
from tempfile import TemporaryDirectory

import joblib
import numpy as np

# Importing utility functions and classes from scikit-learn
from ..utils import Bunch, check_random_state
from ..utils._param_validation import Interval, validate_params
from . import get_data_home
from ._base import (
    RemoteFileMetadata,
    _convert_data_dataframe,
    _fetch_remote,
    _pkl_filepath,
    load_descr,
)

# The original data can be found in:
# https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz

# Metadata about the dataset's archived file, including filename, URL, and checksum
ARCHIVE = RemoteFileMetadata(
    filename="covtype.data.gz",
    url="https://ndownloader.figshare.com/files/5976039",
    checksum="614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771",
)

# Setting up logging for this module
logger = logging.getLogger(__name__)

# Column names reference:
# https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info

# List of feature names based on the dataset's columns
FEATURE_NAMES = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
FEATURE_NAMES += [f"Wilderness_Area_{i}" for i in range(4)]
FEATURE_NAMES += [f"Soil_Type_{i}" for i in range(40)]

# Target column name
TARGET_NAMES = ["Cover_Type"]

# Decorator function to validate parameters for fetching the dataset
@validate_params(
    {
        "data_home": [str, os.PathLike, None],
        "download_if_missing": ["boolean"],
        "random_state": ["random_state"],
        "shuffle": ["boolean"],
        "return_X_y": ["boolean"],
        "as_frame": ["boolean"],
        "n_retries": [Interval(Integral, 1, None, closed="left")],
        "delay": [Interval(Real, 0.0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
# Function to fetch the covertype dataset
def fetch_covtype(
    *,
    data_home=None,
    download_if_missing=True,
    random_state=None,
    shuffle=False,
    return_X_y=False,
    as_frame=False,
    n_retries=3,
    delay=1.0,
):
    """Load the covertype dataset (classification).

    Download it if necessary.

    =================   ============
    Classes                        7
    Samples total             581012
    Dimensionality                54
    Features                     int
    =================   ============

    Read more in the :ref:`User Guide <covtype_dataset>`.

    Parameters
    ----------
    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.
    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    shuffle : bool, default=False
        Whether to shuffle dataset.

    return_X_y : bool, default=False
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

        .. versionadded:: 0.20

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is a pandas DataFrame or
        Series depending on the number of target columns. If `return_X_y` is
        True, then (`data`, `target`) will be pandas DataFrames or Series as
        described below.

        .. versionadded:: 0.24

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.

        .. versionadded:: 1.5

    delay : float, default=1.0
        Number of seconds between retries.

        .. versionadded:: 1.5

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray of shape (581012, 54)
            Each row corresponds to the 54 features in the dataset.
        target : ndarray of shape (581012,)
            Each value corresponds to one of
            the 7 forest covertypes with values
            ranging between 1 to 7.
        frame : dataframe of shape (581012, 55)
            Only present when `as_frame=True`. Contains `data` and `target`.
        DESCR : str
            Description of the forest covertype dataset.
        feature_names : list
            The names of the dataset columns.
        target_names: list
            The names of the target columns.

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarray. The first containing a 2D array of
        shape (n_samples, n_features) with each row representing one
        sample and each column representing the features. The second
        ndarray of shape (n_samples,) containing the target samples.

        .. versionadded:: 0.20

    Examples
    --------
    >>> from sklearn.datasets import fetch_covtype
    >>> cov_type = fetch_covtype()
    >>> cov_type.data.shape
    (581012, 54)
    >>> cov_type.target.shape
    (581012,)
    >>> # Let's check the 4 first feature names
    >>> cov_type.feature_names[:4]
    ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology']
    """
    # 获取数据存储的根目录
    data_home = get_data_home(data_home=data_home)
    # 构建覆盖类型数据集的路径
    covtype_dir = join(data_home, "covertype")
    # 获取样本数据文件的路径
    samples_path = _pkl_filepath(covtype_dir, "samples")
    # 获取目标数据文件的路径
    targets_path = _pkl_filepath(covtype_dir, "targets")
    # 检查数据是否可用，需同时存在样本路径和目标路径
    available = exists(samples_path) and exists(targets_path)

    # 如果需要下载且数据不可用，则开始下载数据
    if download_if_missing and not available:
        # 创建covtype_dir目录，如果不存在则创建
        os.makedirs(covtype_dir, exist_ok=True)

        # 在temp_dir中创建临时目录，作为目标目录的直接子目录，确保在同一文件系统上
        # 这样可以使用os.rename原子性地将数据文件移动到其目标位置
        with TemporaryDirectory(dir=covtype_dir) as temp_dir:
            logger.info(f"Downloading {ARCHIVE.url}")
            # 下载远程数据文件到temp_dir，并获取其路径
            archive_path = _fetch_remote(
                ARCHIVE, dirname=temp_dir, n_retries=n_retries, delay=delay
            )
            # 使用GzipFile打开archive_path，读取数据并解析为numpy数组Xy
            Xy = np.genfromtxt(GzipFile(filename=archive_path), delimiter=",")

            # 将Xy数组切片，分离出特征X和目标y
            X = Xy[:, :-1]
            y = Xy[:, -1].astype(np.int32, copy=False)

            # 将X序列化为pickle格式，保存到samples_tmp_path
            samples_tmp_path = _pkl_filepath(temp_dir, "samples")
            joblib.dump(X, samples_tmp_path, compress=9)
            # 使用原子操作将samples_tmp_path移动到samples_path
            os.rename(samples_tmp_path, samples_path)

            # 将y序列化为pickle格式，保存到targets_tmp_path
            targets_tmp_path = _pkl_filepath(temp_dir, "targets")
            joblib.dump(y, targets_tmp_path, compress=9)
            # 使用原子操作将targets_tmp_path移动到targets_path
            os.rename(targets_tmp_path, targets_path)

    # 如果数据不可用且不允许下载，则抛出OSError异常
    elif not available and not download_if_missing:
        raise OSError("Data not found and `download_if_missing` is False")

    try:
        # 尝试获取X和y变量，如果未定义则从文件加载
        X, y
    except NameError:
        X = joblib.load(samples_path)
        y = joblib.load(targets_path)

    # 如果需要对数据进行洗牌，则重新排列X和y的顺序
    if shuffle:
        ind = np.arange(X.shape[0])  # 创建索引数组
        rng = check_random_state(random_state)  # 检查随机状态
        rng.shuffle(ind)  # 使用随机状态随机排列索引
        X = X[ind]  # 根据索引重排X
        y = y[ind]  # 根据索引重排y

    # 加载covtype数据集的描述信息
    fdescr = load_descr("covtype.rst")

    frame = None
    # 如果需要返回数据帧，则将数据转换为DataFrame格式
    if as_frame:
        frame, X, y = _convert_data_dataframe(
            caller_name="fetch_covtype",
            data=X,
            target=y,
            feature_names=FEATURE_NAMES,
            target_names=TARGET_NAMES,
        )

    # 如果需要仅返回X和y，则直接返回X和y
    if return_X_y:
        return X, y

    # 否则，返回一个命名元组（Bunch），包含数据X、目标y、数据帧frame以及其他元数据
    return Bunch(
        data=X,
        target=y,
        frame=frame,
        target_names=TARGET_NAMES,
        feature_names=FEATURE_NAMES,
        DESCR=fdescr,
    )
```