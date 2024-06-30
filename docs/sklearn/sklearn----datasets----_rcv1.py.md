# `D:\src\scipysrc\scikit-learn\sklearn\datasets\_rcv1.py`

```
"""RCV1 dataset.

The dataset page is available at

    http://jmlr.csail.mit.edu/papers/volume5/lewis04a/
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入所需的模块和函数
import logging                  # 导入 logging 模块用于记录日志
from gzip import GzipFile       # 导入 GzipFile 类用于处理 gzip 压缩文件
from numbers import Integral, Real  # 导入 Integral 和 Real 类型，用于类型检查
from os import PathLike, makedirs, remove  # 导入 PathLike 类和 os 模块的函数
from os.path import exists, join   # 导入 exists 和 join 函数，用于操作文件路径

import joblib                  # 导入 joblib 库，用于数据持久化
import numpy as np             # 导入 NumPy 库，用于数值计算
import scipy.sparse as sp      # 导入 SciPy 的稀疏矩阵模块

from ..utils import Bunch      # 从相对路径导入 Bunch 类
from ..utils import shuffle as shuffle_  # 从相对路径导入 shuffle 函数并重命名为 shuffle_
from ..utils._param_validation import Interval, StrOptions, validate_params  # 导入参数验证相关的类和函数
from . import get_data_home    # 从当前包导入 get_data_home 函数
from ._base import RemoteFileMetadata, _fetch_remote, _pkl_filepath, load_descr  # 导入远程文件元数据相关的类和函数
from ._svmlight_format_io import load_svmlight_files  # 导入加载 SVMLight 格式文件的函数

# The original vectorized data can be found at:
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt0.dat.gz
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt1.dat.gz
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt2.dat.gz
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt3.dat.gz
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_train.dat.gz
# while the original stemmed token files can be found
# in the README, section B.12.i.:
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm

XY_METADATA = (
    RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/5976069",
        checksum="ed40f7e418d10484091b059703eeb95ae3199fe042891dcec4be6696b9968374",
        filename="lyrl2004_vectors_test_pt0.dat.gz",
    ),
    RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/5976066",
        checksum="87700668ae45d45d5ca1ef6ae9bd81ab0f5ec88cc95dcef9ae7838f727a13aa6",
        filename="lyrl2004_vectors_test_pt1.dat.gz",
    ),
    RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/5976063",
        checksum="48143ac703cbe33299f7ae9f4995db49a258690f60e5debbff8995c34841c7f5",
        filename="lyrl2004_vectors_test_pt2.dat.gz",
    ),
    RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/5976060",
        checksum="dfcb0d658311481523c6e6ca0c3f5a3e1d3d12cde5d7a8ce629a9006ec7dbb39",
        filename="lyrl2004_vectors_test_pt3.dat.gz",
    ),
    RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/5976057",
        checksum="5468f656d0ba7a83afc7ad44841cf9a53048a5c083eedc005dcdb5cc768924ae",
        filename="lyrl2004_vectors_train.dat.gz",
    ),
)

# The original data can be found at:
# http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz
TOPICS_METADATA = RemoteFileMetadata(
    url="https://ndownloader.figshare.com/files/5976048",
    checksum="2a98e5e5d8b770bded93afc8930d88299474317fe14181aee1466cc754d0d1c1",
    filename="rcv1v2.topics.qrels.gz",


# 定义一个变量 filename 并赋值为 "rcv1v2.topics.qrels.gz"
# 导入日志模块中的 getLogger 函数，用于获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义 fetch_rcv1 函数，使用装饰器 validate_params 对函数参数进行验证
@validate_params(
    {
        "data_home": [str, PathLike, None],  # data_home 参数可以是字符串、路径对象或者 None
        "subset": [StrOptions({"train", "test", "all"})],  # subset 参数必须是 {'train', 'test', 'all'} 中的一个
        "download_if_missing": ["boolean"],  # download_if_missing 参数必须是布尔值
        "random_state": ["random_state"],  # random_state 参数可以是整数、RandomState 实例或者 None
        "shuffle": ["boolean"],  # shuffle 参数必须是布尔值
        "return_X_y": ["boolean"],  # return_X_y 参数必须是布尔值
        "n_retries": [Interval(Integral, 1, None, closed="left")],  # n_retries 参数是大于等于1的整数
        "delay": [Interval(Real, 0.0, None, closed="neither")],  # delay 参数是大于0的实数
    },
    prefer_skip_nested_validation=True,  # 设置优先跳过嵌套验证
)
# fetch_rcv1 函数：加载 RCV1 多标签数据集（分类），如果需要的话进行下载
def fetch_rcv1(
    *,
    data_home=None,  # 数据集的下载和缓存文件夹，默认为 None
    subset="all",  # 要加载的数据集子集，默认为 'all'
    download_if_missing=True,  # 如果数据不存在，是否下载，默认为 True
    random_state=None,  # 随机数生成器的种子，默认为 None
    shuffle=False,  # 是否对数据集进行洗牌，默认为 False
    return_X_y=False,  # 如果为 True，则返回 (dataset.data, dataset.target)，而不是 Bunch 对象
    n_retries=3,  # 遇到 HTTP 错误时的重试次数，默认为 3
    delay=1.0,  # 每次重试之间的延迟秒数，默认为 1.0
):
    """Load the RCV1 multilabel dataset (classification).

    Download it if necessary.

    Version: RCV1-v2, vectors, full sets, topics multilabels.

    =================   =====================
    Classes                               103
    Samples total                      804414
    Dimensionality                      47236
    Features            real, between 0 and 1
    =================   =====================

    Read more in the :ref:`User Guide <rcv1_dataset>`.

    .. versionadded:: 0.17

    Parameters
    ----------
    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    subset : {'train', 'test', 'all'}, default='all'
        Select the dataset to load: 'train' for the training set
        (23149 samples), 'test' for the test set (781265 samples),
        'all' for both, with the training samples first if shuffle is False.
        This follows the official LYRL2004 chronological split.

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
        If True, returns ``(dataset.data, dataset.target)`` instead of a Bunch
        object. See below for more information about the `dataset.data` and
        `dataset.target` object.

        .. versionadded:: 0.20

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.

        .. versionadded:: 1.5

    delay : float, default=1.0
        Number of seconds between retries.

        .. versionadded:: 1.5

    Returns
    -------

"""
    N_SAMPLES = 804414
    N_FEATURES = 47236
    N_CATEGORIES = 103
    N_TRAIN = 23149

    data_home = get_data_home(data_home=data_home)
    rcv1_dir = join(data_home, "RCV1")
    if download_if_missing:
        if not exists(rcv1_dir):
            makedirs(rcv1_dir)

    samples_path = _pkl_filepath(rcv1_dir, "samples.pkl")
    sample_id_path = _pkl_filepath(rcv1_dir, "sample_id.pkl")
    sample_topics_path = _pkl_filepath(rcv1_dir, "sample_topics.pkl")
    topics_path = _pkl_filepath(rcv1_dir, "topics_names.pkl")

    # load data (X) and sample_id
    if download_if_missing and (not exists(samples_path) or not exists(sample_id_path)):
        # 构造下载文件列表
        files = []
        for each in XY_METADATA:
            logger.info("Downloading %s" % each.url)
            # 下载数据文件并添加到文件列表
            file_path = _fetch_remote(
                each, dirname=rcv1_dir, n_retries=n_retries, delay=delay
            )
            files.append(GzipFile(filename=file_path))

        # 加载数据文件并创建稀疏矩阵 X
        Xy = load_svmlight_files(files, n_features=N_FEATURES)

        # 将训练数据按特定顺序堆叠成 CSR 格式稀疏矩阵 X
        X = sp.vstack([Xy[8], Xy[0], Xy[2], Xy[4], Xy[6]]).tocsr()

        # 合并并转换样本 ID 数据类型为 uint32
        sample_id = np.hstack((Xy[9], Xy[1], Xy[3], Xy[5], Xy[7]))
        sample_id = sample_id.astype(np.uint32, copy=False)

        # 将 X 和 sample_id 保存到 pickle 文件
        joblib.dump(X, samples_path, compress=9)
        joblib.dump(sample_id, sample_id_path, compress=9)

        # 删除下载的压缩文件
        for f in files:
            f.close()
            remove(f.name)
    else:
        # 如果数据文件已存在，则直接加载 X 和 sample_id
        X = joblib.load(samples_path)
        sample_id = joblib.load(sample_id_path)

    # load target (y), categories, and sample_id_bis
    if download_if_missing and (
        not exists(sample_topics_path) or not exists(topics_path)
    ):
    ):
        logger.info("Downloading %s" % TOPICS_METADATA.url)
        topics_archive_path = _fetch_remote(
            TOPICS_METADATA, dirname=rcv1_dir, n_retries=n_retries, delay=delay
        )

        # parse the target file
        n_cat = -1
        n_doc = -1
        doc_previous = -1
        y = np.zeros((N_SAMPLES, N_CATEGORIES), dtype=np.uint8)
        sample_id_bis = np.zeros(N_SAMPLES, dtype=np.int32)
        category_names = {}
        with GzipFile(filename=topics_archive_path, mode="rb") as f:
            for line in f:
                line_components = line.decode("ascii").split(" ")
                if len(line_components) == 3:
                    cat, doc, _ = line_components
                    if cat not in category_names:
                        # Increment category count and assign it to current category
                        n_cat += 1
                        category_names[cat] = n_cat

                    # Convert doc to integer
                    doc = int(doc)
                    if doc != doc_previous:
                        # Increment document count and update sample_id_bis
                        doc_previous = doc
                        n_doc += 1
                        sample_id_bis[n_doc] = doc
                    # Set the corresponding position in y matrix to 1
                    y[n_doc, category_names[cat]] = 1

        # delete archive
        remove(topics_archive_path)

        # Samples in X are ordered with sample_id,
        # whereas in y, they are ordered with sample_id_bis.
        permutation = _find_permutation(sample_id_bis, sample_id)
        y = y[permutation, :]

        # save category names in a list, with same order than y
        categories = np.empty(N_CATEGORIES, dtype=object)
        for k in category_names.keys():
            categories[category_names[k]] = k

        # reorder categories in lexicographic order
        order = np.argsort(categories)
        categories = categories[order]
        y = sp.csr_matrix(y[:, order])

        # Save y matrix and categories list using joblib compression
        joblib.dump(y, sample_topics_path, compress=9)
        joblib.dump(categories, topics_path, compress=9)
    else:
        # Load precomputed data if available
        y = joblib.load(sample_topics_path)
        categories = joblib.load(topics_path)

    # Adjust data subset based on the subset parameter
    if subset == "all":
        pass
    elif subset == "train":
        X = X[:N_TRAIN, :]
        y = y[:N_TRAIN, :]
        sample_id = sample_id[:N_TRAIN]
    elif subset == "test":
        X = X[N_TRAIN:, :]
        y = y[N_TRAIN:, :]
        sample_id = sample_id[N_TRAIN:]
    else:
        # Raise error for unknown subset parameter
        raise ValueError(
            "Unknown subset parameter. Got '%s' instead of one"
            " of ('all', 'train', test')" % subset
        )

    # Shuffle data if specified
    if shuffle:
        X, y, sample_id = shuffle_(X, y, sample_id, random_state=random_state)

    # Load description of the dataset
    fdescr = load_descr("rcv1.rst")

    # Return X, y arrays if requested
    if return_X_y:
        return X, y

    # Return Bunch object containing dataset information
    return Bunch(
        data=X, target=y, sample_id=sample_id, target_names=categories, DESCR=fdescr
    )
# 返回一个数组的逆置置换
def _inverse_permutation(p):
    # 获取置换数组的大小
    n = p.size
    # 创建一个大小为 n 的零数组，用来存放结果
    s = np.zeros(n, dtype=np.int32)
    # 创建一个包含 0 到 n-1 的数组
    i = np.arange(n, dtype=np.int32)
    # 将 i 按照 p 中的顺序放置在 s 中，即 s[p] = i
    np.put(s, p, i)
    # 返回逆置换数组 s
    return s


# 找到将数组 a 转换为数组 b 的置换
def _find_permutation(a, b):
    # 对数组 a 进行排序并返回其索引
    t = np.argsort(a)
    # 对数组 b 进行排序并返回其索引
    u = np.argsort(b)
    # 计算数组 u 的逆置换
    u_ = _inverse_permutation(u)
    # 返回将数组 a 转换为数组 b 的置换
    return t[u_]
```