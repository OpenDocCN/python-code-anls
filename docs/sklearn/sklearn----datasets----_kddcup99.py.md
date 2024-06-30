# `D:\src\scipysrc\scikit-learn\sklearn\datasets\_kddcup99.py`

```
# 引入必要的模块和库
import errno  # 引入 errno 模块，用于处理操作系统相关的错误码
import logging  # 引入 logging 模块，用于记录日志信息
import os  # 引入 os 模块，提供了与操作系统交互的功能
from gzip import GzipFile  # 从 gzip 模块中导入 GzipFile 类，用于处理 gzip 压缩文件
from numbers import Integral, Real  # 从 numbers 模块中导入 Integral 和 Real 类型，用于数字类型检查
from os.path import exists, join  # 从 os.path 模块中导入 exists 和 join 函数，用于检查路径是否存在和路径拼接

import joblib  # 引入 joblib 库，用于高效地保存和加载 Python 对象
import numpy as np  # 引入 numpy 库，并使用 np 别名，用于科学计算

from ..utils import Bunch, check_random_state  # 从相对路径中的 utils 模块导入 Bunch 和 check_random_state 函数
from ..utils import shuffle as shuffle_method  # 从 utils 模块中导入 shuffle 函数，并使用 shuffle_method 别名
from ..utils._param_validation import Interval, StrOptions, validate_params  # 从 utils 模块中导入参数验证相关的函数和类
from . import get_data_home  # 从当前目录下的 get_data_home 模块中导入 get_data_home 函数
from ._base import (  # 从当前目录下的 _base 模块中导入以下函数和类
    RemoteFileMetadata,  # 远程文件元数据类
    _convert_data_dataframe,  # 数据帧转换函数
    _fetch_remote,  # 远程数据获取函数
    load_descr,  # 加载描述信息函数
)

# kddcup99 数据集原始数据可以在以下链接找到：
# https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data.gz
ARCHIVE = RemoteFileMetadata(
    filename="kddcup99_data",
    url="https://ndownloader.figshare.com/files/5976045",
    checksum="3b6c942aa0356c0ca35b7b595a26c89d343652c9db428893e7494f837b274292",
)

# kddcup99 10% 数据集原始数据可以在以下链接找到：
# https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data_10_percent.gz
ARCHIVE_10_PERCENT = RemoteFileMetadata(
    filename="kddcup99_10_data",
    url="https://ndownloader.figshare.com/files/5976042",
    checksum="8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561",
)

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


@validate_params(
    {
        "subset": [StrOptions({"SA", "SF", "http", "smtp"}), None],  # 参数验证：subset 可以是指定的子集或者 None
        "data_home": [str, os.PathLike, None],  # 参数验证：data_home 可以是字符串、路径类型或者 None
        "shuffle": ["boolean"],  # 参数验证：shuffle 必须是布尔类型
        "random_state": ["random_state"],  # 参数验证：random_state 必须是随机状态对象
        "percent10": ["boolean"],  # 参数验证：percent10 必须是布尔类型
        "download_if_missing": ["boolean"],  # 参数验证：download_if_missing 必须是布尔类型
        "return_X_y": ["boolean"],  # 参数验证：return_X_y 必须是布尔类型
        "as_frame": ["boolean"],  # 参数验证：as_frame 必须是布尔类型
        "n_retries": [Interval(Integral, 1, None, closed="left")],  # 参数验证：n_retries 必须是整数且大于等于 1
        "delay": [Interval(Real, 0.0, None, closed="neither")],  # 参数验证：delay 必须是实数且大于 0
    },
    prefer_skip_nested_validation=True,  # 首选跳过嵌套验证
)
def fetch_kddcup99(
    *,
    subset=None,
    data_home=None,
    shuffle=False,
    random_state=None,
    percent10=True,
    download_if_missing=True,
    return_X_y=False,
    as_frame=False,
    n_retries=3,
    delay=1.0,
):
    """Load the kddcup99 dataset (classification).

    Download it if necessary.

    =================   ====================================
    Classes                                               23
    Samples total                                    4898431
    Dimensionality                                        41
    Features            discrete (int) or continuous (float)
    =================   ====================================

    Read more in the :ref:`User Guide <kddcup99_dataset>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    subset : {'SA', 'SF', 'http', 'smtp'}, default=None
        To return the corresponding classical subsets of kddcup 99.
        If None, return the entire kddcup 99 dataset.
    """
    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

        .. versionadded:: 0.19
        指定另一个用于下载和缓存数据集的文件夹路径。默认情况下，所有scikit-learn数据存储在'~/scikit_learn_data'子文件夹中。

    shuffle : bool, default=False
        Whether to shuffle dataset.
        是否对数据集进行洗牌。

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset shuffling and for
        selection of abnormal samples if `subset='SA'`. Pass an int for
        reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

        .. versionadded:: 0.19
        确定用于数据集洗牌和（如果`subset='SA'`）选择异常样本的随机数生成。传递整数以在多个函数调用之间生成可重现的输出。

    percent10 : bool, default=True
        Whether to load only 10 percent of the data.
        是否只加载数据的10%。

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.
        如果为False，则在本地数据不可用时引发OSError，而不是尝试从源站点下载数据。

    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` object.

        .. versionadded:: 0.20
        如果为True，则返回`(data, target)`而不是Bunch对象。有关`data`和`target`对象的更多信息，请参见下文。

    as_frame : bool, default=False
        If `True`, returns a pandas Dataframe for the ``data`` and ``target``
        objects in the `Bunch` returned object; `Bunch` return object will also
        have a ``frame`` member.

        .. versionadded:: 0.24
        如果为True，则返回Bunch返回对象中`data`和`target`对象的pandas DataFrame；Bunch返回对象还将具有`frame`成员。

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.

        .. versionadded:: 1.5
        在遇到HTTP错误时的重试次数。

    delay : float, default=1.0
        Number of seconds between retries.

        .. versionadded:: 1.5
        重试之间的秒数。

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (494021, 41)
            The data matrix to learn. If `as_frame=True`, `data` will be a
            pandas DataFrame.
            用以下属性的类字典对象。

            data: {ndarray, dataframe}，形状为（494021，41）
                用于学习的数据矩阵。如果`as_frame=True`，则`data`将是一个pandas DataFrame。

        target : {ndarray, series} of shape (494021,)
            The regression target for each sample. If `as_frame=True`, `target`
            will be a pandas Series.
            每个样本的回归目标。如果`as_frame=True`，则`target`将是一个pandas Series。

        frame : dataframe of shape (494021, 42)
            Only present when `as_frame=True`. Contains `data` and `target`.
            当`as_frame=True`时存在。包含`data`和`target`。

        DESCR : str
            The full description of the dataset.
            数据集的完整描述。

        feature_names : list
            The names of the dataset columns
            数据集列的名称列表。

        target_names: list
            The names of the target columns
            目标列的名称列表。

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarray. The first containing a 2D array of
        shape (n_samples, n_features) with each row representing one
        sample and each column representing the features. The second
        ndarray of shape (n_samples,) containing the target samples.

        .. versionadded:: 0.20
        如果`return_X_y`为True，则返回两个ndarray的元组。第一个包含形状为（n_samples，n_features）的2D数组，每行代表一个样本，每列代表特征。第二个形状为（n_samples，）的ndarray，包含目标样本。
    # 使用 _fetch_brute_kddcup99 函数获取 KDD Cup 99 数据集的 Bunch 对象
    kddcup99 = _fetch_brute_kddcup99(
        data_home=data_home,
        percent10=percent10,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )

    # 从 kddcup99 对象中获取数据集和目标值
    data = kddcup99.data
    target = kddcup99.target
    feature_names = kddcup99.feature_names
    target_names = kddcup99.target_names

    # 如果选择的子集为 "SA"，则处理正常和异常样本
    if subset == "SA":
        # 筛选出正常样本和异常样本
        s = target == b"normal."
        t = np.logical_not(s)
        normal_samples = data[s, :]
        normal_targets = target[s]
        abnormal_samples = data[t, :]
        abnormal_targets = target[t]

        # 随机选择部分异常样本进行处理
        n_samples_abnormal = abnormal_samples.shape[0]
        random_state = check_random_state(random_state)
        r = random_state.randint(0, n_samples_abnormal, 3377)
        abnormal_samples = abnormal_samples[r]
        abnormal_targets = abnormal_targets[r]

        # 合并处理后的数据集
        data = np.r_[normal_samples, abnormal_samples]
        target = np.r_[normal_targets, abnormal_targets]

    # 如果选择的子集为 "SF", "http", 或 "smtp"，则进一步处理数据
    if subset == "SF" or subset == "http" or subset == "smtp":
        # 筛选出具有正 logged_in 属性的所有样本
        s = data[:, 11] == 1
        data = np.c_[data[s, :11], data[s, 12:]]
        feature_names = feature_names[:11] + feature_names[12:]
        target = target[s]

        # 对特定列进行对数转换
        data[:, 0] = np.log((data[:, 0] + 0.1).astype(float, copy=False))
        data[:, 4] = np.log((data[:, 4] + 0.1).astype(float, copy=False))
        data[:, 5] = np.log((data[:, 5] + 0.1).astype(float, copy=False))

        # 根据子集进一步筛选数据
        if subset == "http":
            s = data[:, 2] == b"http"
            data = data[s]
            target = target[s]
            data = np.c_[data[:, 0], data[:, 4], data[:, 5]]
            feature_names = [feature_names[0], feature_names[4], feature_names[5]]

        if subset == "smtp":
            s = data[:, 2] == b"smtp"
            data = data[s]
            target = target[s]
            data = np.c_[data[:, 0], data[:, 4], data[:, 5]]
            feature_names = [feature_names[0], feature_names[4], feature_names[5]]

        if subset == "SF":
            data = np.c_[data[:, 0], data[:, 2], data[:, 4], data[:, 5]]
            feature_names = [
                feature_names[0],
                feature_names[2],
                feature_names[4],
                feature_names[5],
            ]

    # 如果 shuffle 参数为 True，则对数据和目标值进行打乱顺序操作
    if shuffle:
        data, target = shuffle_method(data, target, random_state=random_state)

    # 载入描述文件
    fdescr = load_descr("kddcup99.rst")

    frame = None
    # 如果 as_frame 参数为 True，则将数据转换为 DataFrame 格式
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "fetch_kddcup99", data, target, feature_names, target_names
        )

    # 如果 return_X_y 参数为 True，则返回数据集和目标值
    if return_X_y:
        return data, target

    # 返回 Bunch 对象包含的数据集和相关信息
    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        feature_names=feature_names,
        DESCR=fdescr,
    )
# 确定数据集存储路径，默认为 '~/scikit_learn_data' 子文件夹下
data_home = get_data_home(data_home=data_home)
# 设置数据集文件名后缀，针对 Python 3 版本
dir_suffix = "-py3"

# 根据 percent10 参数决定是否加载数据集的 10% 版本
if percent10:
    # 构建 10% 版本数据集的完整路径
    kddcup_dir = join(data_home, "kddcup99_10" + dir_suffix)
    # 指定使用 10% 版本的数据集存档文件名
    archive = ARCHIVE_10_PERCENT
else:
    # 构建完整数据集的路径
    kddcup_dir = join(data_home, "kddcup99" + dir_suffix)
    # 指定使用完整数据集的存档文件名
    archive = ARCHIVE

# 构建样本文件和目标文件的具体路径
samples_path = join(kddcup_dir, "samples")
targets_path = join(kddcup_dir, "targets")
# 检查样本路径是否存在，返回布尔值
available = exists(samples_path)
    dt = [
        ("duration", int),  # 定义数据类型列表，包含特征名称和对应的数据类型
        ("protocol_type", "S4"),  # 网络协议类型，使用4字节的字符串表示
        ("service", "S11"),  # 服务类型，使用11字节的字符串表示
        ("flag", "S6"),  # 标志，使用6字节的字符串表示
        ("src_bytes", int),  # 源字节数，整数类型
        ("dst_bytes", int),  # 目的字节数，整数类型
        ("land", int),  # 是否是从同一个主机/端口发送的连接标志，整数类型
        ("wrong_fragment", int),  # 错误分段数量，整数类型
        ("urgent", int),  # 紧急包数量，整数类型
        ("hot", int),  # 访问系统资源的频率，整数类型
        ("num_failed_logins", int),  # 失败登录尝试次数，整数类型
        ("logged_in", int),  # 是否登录成功，整数类型
        ("num_compromised", int),  # 受损的计算机数量，整数类型
        ("root_shell", int),  # 是否获取了root shell，整数类型
        ("su_attempted", int),  # 是否尝试su命令，整数类型
        ("num_root", int),  # root访问次数，整数类型
        ("num_file_creations", int),  # 文件创建次数，整数类型
        ("num_shells", int),  # shell命令次数，整数类型
        ("num_access_files", int),  # 访问文件数量，整数类型
        ("num_outbound_cmds", int),  # 发出的命令数量，整数类型
        ("is_host_login", int),  # 是否是主机登录标志，整数类型
        ("is_guest_login", int),  # 是否是guest登录标志，整数类型
        ("count", int),  # 连接到同一主机的连接数，整数类型
        ("srv_count", int),  # 连接到相同服务的连接数，整数类型
        ("serror_rate", float),  # 错误连接率，浮点数类型
        ("srv_serror_rate", float),  # 服务错误连接率，浮点数类型
        ("rerror_rate", float),  # 拒绝连接率，浮点数类型
        ("srv_rerror_rate", float),  # 服务拒绝连接率，浮点数类型
        ("same_srv_rate", float),  # 相同服务连接率，浮点数类型
        ("diff_srv_rate", float),  # 不同服务连接率，浮点数类型
        ("srv_diff_host_rate", float),  # 服务不同主机连接率，浮点数类型
        ("dst_host_count", int),  # 目的主机数量，整数类型
        ("dst_host_srv_count", int),  # 目的主机服务数量，整数类型
        ("dst_host_same_srv_rate", float),  # 目的主机相同服务连接率，浮点数类型
        ("dst_host_diff_srv_rate", float),  # 目的主机不同服务连接率，浮点数类型
        ("dst_host_same_src_port_rate", float),  # 目的主机相同源端口连接率，浮点数类型
        ("dst_host_srv_diff_host_rate", float),  # 目的主机服务不同主机连接率，浮点数类型
        ("dst_host_serror_rate", float),  # 目的主机错误连接率，浮点数类型
        ("dst_host_srv_serror_rate", float),  # 目的主机服务错误连接率，浮点数类型
        ("dst_host_rerror_rate", float),  # 目的主机拒绝连接率，浮点数类型
        ("dst_host_srv_rerror_rate", float),  # 目的主机服务拒绝连接率，浮点数类型
        ("labels", "S16"),  # 标签，使用16字节的字符串表示
    ]

    column_names = [c[0] for c in dt]  # 提取特征名列表
    target_names = column_names[-1]  # 提取标签名称
    feature_names = column_names[:-1]  # 提取特征名称列表

    if available:  # 如果数据可用
        try:
            X = joblib.load(samples_path)  # 加载样本数据
            y = joblib.load(targets_path)  # 加载目标数据
        except Exception as e:
            raise OSError(
                "The cache for fetch_kddcup99 is invalid, please delete "
                f"{str(kddcup_dir)} and run the fetch_kddcup99 again"
            ) from e

    elif download_if_missing:  # 如果数据缺失且可以下载
        _mkdirp(kddcup_dir)  # 创建目录
        logger.info("Downloading %s" % archive.url)  # 记录日志，下载文件
        _fetch_remote(archive, dirname=kddcup_dir, n_retries=n_retries, delay=delay)  # 从远程获取文件
        DT = np.dtype(dt)  # 定义数据类型
        logger.debug("extracting archive")  # 记录日志，解压文件
        archive_path = join(kddcup_dir, archive.filename)  # 构建存档路径
        file_ = GzipFile(filename=archive_path, mode="r")  # 打开gzip文件
        Xy = []
        for line in file_.readlines():  # 逐行读取文件内容
            line = line.decode()  # 解码为字符串
            Xy.append(line.replace("\n", "").split(","))  # 移除换行符并按逗号分割，添加到Xy列表
        file_.close()  # 关闭文件
        logger.debug("extraction done")  # 记录日志，解压完成
        os.remove(archive_path)  # 删除解压后的存档文件

        Xy = np.asarray(Xy, dtype=object)  # 将Xy转换为NumPy数组，数据类型为对象
        for j in range(42):  # 遍历特征数量
            Xy[:, j] = Xy[:, j].astype(DT[j])  # 按照定义的数据类型转换每列数据类型

        X = Xy[:, :-1]  # 提取特征数据
        y = Xy[:, -1]  # 提取标签数据
        # XXX bug when compress!=0:
        # (error: 'Incorrect data length while decompressing[...] the file
        #  could be corrupted.')

        joblib.dump(X, samples_path, compress=0)  # 将特征数据保存到文件中，不使用压缩
        joblib.dump(y, targets_path, compress=0)  # 将标签数据保存到文件中，不使用压缩
    else:
        raise OSError("Data not found and `download_if_missing` is False")  # 如果数据不可用且不可下载，则抛出异常
    # 创建一个包含多个属性的对象 Bunch 并返回
    return Bunch(
        # 设置对象的 data 属性为 X，通常用于存储数据集的特征数据
        data=X,
        # 设置对象的 target 属性为 y，通常用于存储数据集的目标值或标签
        target=y,
        # 设置对象的 feature_names 属性为 feature_names，通常用于存储特征名称列表
        feature_names=feature_names,
        # 设置对象的 target_names 属性为 [target_names]，通常用于存储目标类别名称列表
        target_names=[target_names],
    )
# 确保目录 d 存在，类似于 Unix 上的 mkdir -p 功能
# 没有保证目录是否可写。
def _mkdirp(d):
    try:
        # 尝试递归创建目录 d，如果目录已存在则会抛出 OSError 错误
        os.makedirs(d)
    except OSError as e:
        # 如果错误码不是 EEXIST（目录已存在），则重新抛出该异常
        if e.errno != errno.EEXIST:
            raise
```