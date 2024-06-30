# `D:\src\scipysrc\scikit-learn\sklearn\datasets\_olivetti_faces.py`

```
# 导入必要的模块和函数
from numbers import Integral, Real  # 导入数字类型的验证工具
from os import PathLike, makedirs, remove  # 导入文件系统相关函数和路径类型
from os.path import exists  # 导入判断路径是否存在的函数

import joblib  # 导入用于序列化和反序列化的工具
import numpy as np  # 导入数值计算库 numpy
from scipy.io import loadmat  # 导入用于加载 MATLAB 文件的工具函数

# 导入工具函数和类
from ..utils import Bunch, check_random_state  # 导入自定义的工具函数和类
from ..utils._param_validation import Interval, validate_params  # 导入参数验证相关工具
from . import get_data_home  # 导入获取数据存储路径的函数
from ._base import RemoteFileMetadata, _fetch_remote, _pkl_filepath, load_descr  # 导入远程文件元数据和相关函数

# 定义远程数据文件的元数据
FACES = RemoteFileMetadata(
    filename="olivettifaces.mat",  # 文件名
    url="https://ndownloader.figshare.com/files/5976027",  # 下载地址
    checksum="b612fb967f2dc77c9c62d3e1266e0c73d5fca46a4b8906c18e454d41af987794",  # 文件校验和
)

# 参数验证装饰器，用于检查和验证函数参数
@validate_params(
    {
        "data_home": [str, PathLike, None],  # 数据存储路径可以是字符串、PathLike 对象或者 None
        "shuffle": ["boolean"],  # 是否打乱数据集顺序的布尔值
        "random_state": ["random_state"],  # 随机数生成器的种子，用于确保结果的可复现性
        "download_if_missing": ["boolean"],  # 如果数据缺失是否尝试下载的布尔值
        "return_X_y": ["boolean"],  # 是否返回 (data, target) 格式的数据
        "n_retries": [Interval(Integral, 1, None, closed="left")],  # 下载重试次数的区间设定
        "delay": [Interval(Real, 0.0, None, closed="neither")],  # 下载失败后的延迟时间区间设定
    },
    prefer_skip_nested_validation=True,  # 设置跳过嵌套验证
)
def fetch_olivetti_faces(
    *,
    data_home=None,  # 数据存储路径，默认为 None
    shuffle=False,  # 是否打乱数据集顺序，默认为 False
    random_state=0,  # 随机数种子，默认为 0
    download_if_missing=True,  # 如果数据缺失是否尝试下载，默认为 True
    return_X_y=False,  # 是否返回 (data, target) 格式的数据，默认为 False
    n_retries=3,  # 下载重试次数，默认为 3
    delay=1.0,  # 下载失败后的延迟时间，默认为 1.0
):
    """Load the Olivetti faces data-set from AT&T (classification).

    Download it if necessary.

    =================   =====================
    Classes                                40
    Samples total                         400
    Dimensionality                       4096
    Features            real, between 0 and 1
    =================   =====================

    Read more in the :ref:`User Guide <olivetti_faces_dataset>`.

    Parameters
    ----------
    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    shuffle : bool, default=False
        If True the order of the dataset is shuffled to avoid having
        images of the same person grouped.

    random_state : int, RandomState instance or None, default=0
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : bool, default=False
        If True, returns (data, target) instead of a Bunch object.

    n_retries : int, default=3
        Number of times to retry remote file download.

    delay : float, default=1.0
        Delay in seconds between retries.
    """
    return_X_y : bool, default=False
        # 控制返回值是否为 `(data, target)` 对，而非 `Bunch` 对象。有关 `data` 和 `target` 对象的更多信息，请参见下文。

        .. versionadded:: 0.22
            # 添加于版本 0.22

    n_retries : int, default=3
        # 在遇到 HTTP 错误时的重试次数。

        .. versionadded:: 1.5
            # 添加于版本 1.5

    delay : float, default=1.0
        # 重试之间的延迟秒数。

        .. versionadded:: 1.5
            # 添加于版本 1.5

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        # 类似字典的对象，具有以下属性。

        data: ndarray, shape (400, 4096)
            # 每行对应于一个大小为 64 x 64 像素的展平后的人脸图像。
        images : ndarray, shape (400, 64, 64)
            # 每行是一个人脸图像，对应数据集中的一个主体。
        target : ndarray, shape (400,)
            # 每个人脸图像相关联的标签。
            # 这些标签从 0 到 39，对应于主体 ID。
        DESCR : str
            # 修改后的 Olivetti Faces 数据集的描述。

    (data, target) : tuple if `return_X_y=True`
        # 如果 `return_X_y=True`，返回 `data` 和 `target` 对象的元组。

        .. versionadded:: 0.22
            # 添加于版本 0.22

    Examples
    --------
    >>> from sklearn.datasets import fetch_olivetti_faces
    >>> olivetti_faces = fetch_olivetti_faces()
    >>> olivetti_faces.data.shape
    (400, 4096)
    >>> olivetti_faces.target.shape
    (400,)
    >>> olivetti_faces.images.shape
    (400, 64, 64)
    """
    # 获取数据存储路径
    data_home = get_data_home(data_home=data_home)
    # 如果路径不存在，则创建
    if not exists(data_home):
        makedirs(data_home)
    # 构建数据文件的路径
    filepath = _pkl_filepath(data_home, "olivetti.pkz")
    # 如果数据文件不存在
    if not exists(filepath):
        # 如果不允许在数据丢失时下载
        if not download_if_missing:
            raise OSError("Data not found and `download_if_missing` is False")

        # 打印下载信息
        print("downloading Olivetti faces from %s to %s" % (FACES.url, data_home))
        # 下载远程数据文件
        mat_path = _fetch_remote(
            FACES, dirname=data_home, n_retries=n_retries, delay=delay
        )
        # 加载 .mat 文件数据
        mfile = loadmat(file_name=mat_path)
        # 删除原始的 .mat 数据文件
        remove(mat_path)

        # 复制人脸数据并存储到指定路径
        faces = mfile["faces"].T.copy()
        joblib.dump(faces, filepath, compress=6)
        del mfile
    else:
        # 如果数据文件已经存在，则加载数据
        faces = joblib.load(filepath)

    # 将数据类型转换为浮点型，使用 float32 即可（原始的 uint8 数据只有一个字节的精度）
    faces = np.float32(faces)
    # 数据归一化处理
    faces = faces - faces.min()
    faces /= faces.max()
    # 重塑数据形状为 (400, 64, 64)，并进行轴交换
    faces = faces.reshape((400, 64, 64)).transpose(0, 2, 1)
    # 每类有 10 张图像，总共 400 张图像，每类连续
    target = np.array([i // 10 for i in range(400)])
    # 如果需要打乱数据顺序
    if shuffle:
        random_state = check_random_state(random_state)
        order = random_state.permutation(len(faces))
        faces = faces[order]
        target = target[order]
    # 将图像展平为向量
    faces_vectorized = faces.reshape(len(faces), -1)
    # 载入说明文件 "olivetti_faces.rst" 的内容，返回给变量 fdescr
    fdescr = load_descr("olivetti_faces.rst")

    # 如果 return_X_y 为 True，则返回 faces_vectorized 和 target
    if return_X_y:
        return faces_vectorized, target

    # 否则，返回一个 Bunch 对象，包括 faces_vectorized（数据）、faces（图像）、target（目标值）和描述信息 fdescr
    return Bunch(data=faces_vectorized, images=faces, target=target, DESCR=fdescr)
```