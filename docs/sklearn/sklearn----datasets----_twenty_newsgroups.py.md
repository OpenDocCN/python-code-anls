# `D:\src\scipysrc\scikit-learn\sklearn\datasets\_twenty_newsgroups.py`

```
# 导入必要的库和模块
import codecs  # 用于编码解码文件
import logging  # 日志记录模块
import os  # 系统操作相关功能
import pickle  # 用于序列化和反序列化 Python 对象
import re  # 正则表达式模块
import shutil  # 文件和目录操作模块
import tarfile  # 用于操作 tar 文件的模块
from contextlib import suppress  # 上下文管理工具，用于忽略特定的异常
from numbers import Integral, Real  # 数字相关的基类

import joblib  # 用于并行执行任务的工具
import numpy as np  # 数值计算库 NumPy
import scipy.sparse as sp  # 稀疏矩阵库 SciPy

from .. import preprocessing  # 导入自定义模块 preprocessing
from ..feature_extraction.text import CountVectorizer  # 文本特征提取模块中的计数向量化器
from ..utils import Bunch, check_random_state  # 自定义工具函数 Bunch 和随机状态检查函数
from ..utils._param_validation import Interval, StrOptions, validate_params  # 参数验证相关的工具
from ..utils.fixes import tarfile_extractall  # 修复的 tarfile.extractall 方法
from . import get_data_home, load_files  # 导入当前包中的函数 get_data_home 和 load_files
from ._base import (
    RemoteFileMetadata,  # 远程文件的元数据
    _convert_data_dataframe,  # 数据框转换函数
    _fetch_remote,  # 远程数据获取函数
    _pkl_filepath,  # pickle 文件路径函数
    load_descr,  # 加载描述函数
)

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

# 数据集的原始文件地址
ARCHIVE = RemoteFileMetadata(
    filename="20news-bydate.tar.gz",
    url="https://ndownloader.figshare.com/files/5975967",
    checksum="8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610",
)

CACHE_NAME = "20news-bydate.pkz"  # 缓存文件的名称
TRAIN_FOLDER = "20news-bydate-train"  # 训练集文件夹名称
TEST_FOLDER = "20news-bydate-test"  # 测试集文件夹名称


def _download_20newsgroups(target_dir, cache_path, n_retries, delay):
    """Download the 20 newsgroups data and stored it as a zipped pickle."""
    # 构建训练集和测试集的完整路径
    train_path = os.path.join(target_dir, TRAIN_FOLDER)
    test_path = os.path.join(target_dir, TEST_FOLDER)

    # 创建目标文件夹，如果不存在则创建
    os.makedirs(target_dir, exist_ok=True)

    # 记录下载进度及文件大小信息
    logger.info("Downloading dataset from %s (14 MB)", ARCHIVE.url)
    # 下载数据集并返回存储路径
    archive_path = _fetch_remote(
        ARCHIVE, dirname=target_dir, n_retries=n_retries, delay=delay
    )

    # 调试信息：解压缩下载的数据集文件
    logger.debug("Decompressing %s", archive_path)
    with tarfile.open(archive_path, "r:gz") as fp:
        # 解压缩文件到目标文件夹
        tarfile_extractall(fp, path=target_dir)

    # 安静模式：如果存在文件则尝试删除
    with suppress(FileNotFoundError):
        os.remove(archive_path)

    # 存储压缩的 pickle 文件
    # （未完成的部分，代码未提供具体实现）
    # 创建一个缓存字典，包含训练集和测试集数据
    cache = dict(
        train=load_files(train_path, encoding="latin1"),  # 从指定路径加载训练集文件，使用Latin-1编码
        test=load_files(test_path, encoding="latin1"),    # 从指定路径加载测试集文件，使用Latin-1编码
    )
    # 将缓存字典序列化为pickle格式，然后用zlib压缩编码
    compressed_content = codecs.encode(pickle.dumps(cache), "zlib_codec")
    # 打开指定路径的文件，以二进制写模式写入压缩后的内容
    with open(cache_path, "wb") as f:
        f.write(compressed_content)

    # 递归地删除目标目录及其所有内容
    shutil.rmtree(target_dir)
    # 返回生成的缓存字典
    return cache
# 导入标准库 re 和 os 中的 PathLike 类型
import re
import os

# 定义函数 strip_newsgroup_header，用于移除新闻格式中的头部信息
def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.

    Parameters
    ----------
    text : str
        The text from which to remove the signature block.
    """
    # 使用空行分割文本，返回分割后的第一部分和第二部分
    _before, _blankline, after = text.partition("\n\n")
    return after


# 编译正则表达式，用于匹配引用标记或引用前缀的行
_QUOTE_RE = re.compile(
    r"(writes in|writes:|wrote:|says:|said:" r"|^In article|^Quoted from|^\||^>)"
)

# 定义函数 strip_newsgroup_quoting，用于移除新闻格式中的引用部分
def strip_newsgroup_quoting(text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)

    Parameters
    ----------
    text : str
        The text from which to remove the signature block.
    """
    # 筛选出不包含引用标记的行，形成新的行列表
    good_lines = [line for line in text.split("\n") if not _QUOTE_RE.search(line)]
    return "\n".join(good_lines)


# 定义函数 strip_newsgroup_footer，用于移除新闻格式中的尾部签名块
def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.

    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).

    Parameters
    ----------
    text : str
        The text from which to remove the signature block.
    """
    # 去除文本两端的空白符，并按行分割形成行列表
    lines = text.strip().split("\n")
    # 逆向遍历行列表，找到最后一个空白行或由连字符组成的行
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip("-") == "":
            break

    # 如果找到了签名块的分隔行，则返回分隔行之前的部分作为新的文本内容
    if line_num > 0:
        return "\n".join(lines[:line_num])
    else:
        return text


# 定义函数 fetch_20newsgroups，用于加载并返回 20 newsgroups 数据集的文件名和数据
@validate_params(
    {
        "data_home": [str, os.PathLike, None],
        "subset": [StrOptions({"train", "test", "all"})],
        "categories": ["array-like", None],
        "shuffle": ["boolean"],
        "random_state": ["random_state"],
        "remove": [tuple],
        "download_if_missing": ["boolean"],
        "return_X_y": ["boolean"],
        "n_retries": [Interval(Integral, 1, None, closed="left")],
        "delay": [Interval(Real, 0.0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def fetch_20newsgroups(
    *,
    data_home=None,
    subset="train",
    categories=None,
    shuffle=True,
    random_state=42,
    remove=(),
    download_if_missing=True,
    return_X_y=False,
    n_retries=3,
    delay=1.0,
):
    """Load the filenames and data from the 20 newsgroups dataset \
(classification).

    Download it if necessary.

    =================   ==========
    Classes                     20
    Samples total            18846
    Dimensionality               1
    Features                  text
    =================   ==========

    Read more in the :ref:`User Guide <20newsgroups_dataset>`.

    Parameters
    ----------

    """
    # 函数文档字符串，描述了函数的作用和数据集的相关信息
    pass
    # 数据集的下载和缓存文件夹路径，如果为 None，则存储在 '~/scikit_learn_data' 子文件夹中
    data_home : str or path-like, default=None

    # 选择要加载的数据集类型：'train' 表示训练集，'test' 表示测试集，'all' 表示全部数据集（顺序已打乱）
    subset : {'train', 'test', 'all'}, default='train'

    # 要加载的类别列表，如果为 None（默认），加载所有类别；如果不为 None，则加载列表中指定的类别
    categories : array-like, dtype=str, default=None

    # 是否对数据进行洗牌（shuffle），对于假设样本独立同分布（i.i.d.）的模型（如随机梯度下降），这可能很重要
    shuffle : bool, default=True

    # 用于数据洗牌的随机数生成器种子，设定一个整数可保证多次调用函数时输出可复现
    random_state : int, RandomState instance or None, default=42

    # 包含元组 ('headers', 'footers', 'quotes') 中的任何子集，用于从新闻组帖子中检测并移除特定类型的文本
    # 'headers' 移除新闻组标题，'footers' 移除看起来像签名的帖子尾部块，'quotes' 移除引用其他帖子的行
    # 'headers' 遵循一个精确的标准；其他过滤器不总是正确
    remove : tuple, default=()

    # 如果数据在本地不可用，是否尝试从源站点下载数据；若为 False，则遇到数据不可用时抛出 OSError
    download_if_missing : bool, default=True

    # 如果为 True，则返回 `(data.data, data.target)` 而不是 Bunch 对象
    return_X_y : bool, default=False
        .. versionadded:: 0.22

    # 在遇到 HTTP 错误时的重试次数
    n_retries : int, default=3
        .. versionadded:: 1.5

    # 每次重试之间的延迟秒数
    delay : float, default=1.0
        .. versionadded:: 1.5

    # 返回值
    bunch : :class:`~sklearn.utils.Bunch`
        类似字典的对象，具有以下属性：

        data : list of shape (n_samples,)
            用于学习的数据列表
        target: ndarray of shape (n_samples,)
            目标标签
        filenames: list of shape (n_samples,)
            数据位置的路径
        DESCR: str
            数据集的完整描述
        target_names: list of shape (n_classes,)
            目标类别的名称
    # 获取数据存储的根目录路径
    data_home = get_data_home(data_home=data_home)
    # 构建缓存文件的路径
    cache_path = _pkl_filepath(data_home, CACHE_NAME)
    # 构建存放 20newsgroups 数据集的目录路径
    twenty_home = os.path.join(data_home, "20news_home")
    # 初始化缓存为 None
    cache = None
    # 如果缓存文件存在，则尝试加载缓存内容
    if os.path.exists(cache_path):
        try:
            # 以二进制读取缓存文件内容
            with open(cache_path, "rb") as f:
                compressed_content = f.read()
            # 解压缩缓存内容
            uncompressed_content = codecs.decode(compressed_content, "zlib_codec")
            # 反序列化解压后的内容为缓存对象
            cache = pickle.loads(uncompressed_content)
        except Exception as e:
            # 若加载缓存失败，则打印错误信息
            print(80 * "_")
            print("Cache loading failed")
            print(80 * "_")
            print(e)

    # 如果缓存为空，根据下载选项决定是否下载 20newsgroups 数据集
    if cache is None:
        if download_if_missing:
            # 提示开始下载数据集
            logger.info("Downloading 20news dataset. This may take a few minutes.")
            # 下载并缓存数据集
            cache = _download_20newsgroups(
                target_dir=twenty_home,
                cache_path=cache_path,
                n_retries=n_retries,
                delay=delay,
            )
        else:
            # 若不允许下载且缓存为空，则抛出异常
            raise OSError("20Newsgroups dataset not found")

    # 根据指定的数据子集加载数据
    if subset in ("train", "test"):
        data = cache[subset]
    elif subset == "all":
        # 初始化空列表和数组以存储所有数据
        data_lst = list()
        target = list()
        filenames = list()
        # 遍历训练集和测试集
        for subset in ("train", "test"):
            data = cache[subset]
            # 合并数据
            data_lst.extend(data.data)
            target.extend(data.target)
            filenames.extend(data.filenames)

        # 更新数据对象的内容
        data.data = data_lst
        data.target = np.array(target)
        data.filenames = np.array(filenames)

    # 载入数据集的描述信息
    fdescr = load_descr("twenty_newsgroups.rst")

    # 将数据集的描述信息保存到数据对象中
    data.DESCR = fdescr

    # 根据移除选项处理数据集中的特定部分
    if "headers" in remove:
        data.data = [strip_newsgroup_header(text) for text in data.data]
    if "footers" in remove:
        data.data = [strip_newsgroup_footer(text) for text in data.data]
    if "quotes" in remove:
        data.data = [strip_newsgroup_quoting(text) for text in data.data]
    # 如果给定了分类列表
    if categories is not None:
        # 为每个分类确定对应的标签和分类名称，并按标签排序
        labels = [(data.target_names.index(cat), cat) for cat in categories]
        labels.sort()  # 对标签排序以保持标签的顺序
        # 解压标签和分类名称为两个列表
        labels, categories = zip(*labels)
        
        # 创建一个布尔掩码，用于仅保留目标数据中存在于labels中的项
        mask = np.isin(data.target, labels)
        # 根据掩码筛选文件名和目标标签
        data.filenames = data.filenames[mask]
        data.target = data.target[mask]
        
        # 使用searchsorted函数将目标标签映射到连续的标签值
        data.target = np.searchsorted(labels, data.target)
        # 将分类名称列表赋值给目标数据的目标名称属性
        data.target_names = list(categories)
        
        # 使用对象数组进行数据洗牌，以避免内存复制
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[mask]
        data.data = data_lst.tolist()

    # 如果需要洗牌数据集
    if shuffle:
        # 使用给定的随机状态对象初始化随机状态生成器
        random_state = check_random_state(random_state)
        # 创建一个索引数组，表示目标数组的索引范围
        indices = np.arange(data.target.shape[0])
        # 对索引数组进行洗牌操作
        random_state.shuffle(indices)
        
        # 根据洗牌后的索引数组重新排列文件名和目标标签数组
        data.filenames = data.filenames[indices]
        data.target = data.target[indices]
        
        # 使用对象数组进行数据洗牌，以避免内存复制
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[indices]
        data.data = data_lst.tolist()

    # 如果需要返回特征数据和目标标签
    if return_X_y:
        # 返回经过处理后的特征数据和目标标签
        return data.data, data.target

    # 返回经过所有处理的数据对象
    return data
# 使用装饰器 @validate_params 进行参数验证，确保参数满足指定的类型和取值范围
@validate_params(
    {
        "subset": [StrOptions({"train", "test", "all"})],  # subset 参数应为 {'train', 'test', 'all'} 中的一个字符串
        "remove": [tuple],  # remove 参数应为元组类型
        "data_home": [str, os.PathLike, None],  # data_home 参数可以是字符串、os.PathLike 类型或 None
        "download_if_missing": ["boolean"],  # download_if_missing 参数应为布尔类型
        "return_X_y": ["boolean"],  # return_X_y 参数应为布尔类型
        "normalize": ["boolean"],  # normalize 参数应为布尔类型
        "as_frame": ["boolean"],  # as_frame 参数应为布尔类型
        "n_retries": [Interval(Integral, 1, None, closed="left")],  # n_retries 参数应为大于等于 1 的整数
        "delay": [Interval(Real, 0.0, None, closed="neither")],  # delay 参数应为大于 0 的实数
    },
    prefer_skip_nested_validation=True,  # 设置优先跳过嵌套验证
)
def fetch_20newsgroups_vectorized(
    *,
    subset="train",  # 默认为 'train'，指定数据集加载的子集
    remove=(),  # 默认为空元组，用于指定要从新闻组帖子中移除的内容类型
    data_home=None,  # 默认为 None，指定数据集的下载和缓存文件夹
    download_if_missing=True,  # 默认为 True，如果数据不可用，则尝试从源站点下载数据
    return_X_y=False,  # 默认为 False，如果为 True，则返回 (data.data, data.target) 而不是 Bunch 对象
    normalize=True,  # 默认为 True，如果为 True，则对结果进行归一化处理
    as_frame=False,  # 默认为 False，如果为 True，则返回 pandas DataFrame 格式数据
    n_retries=3,  # 默认为 3，指定下载时的重试次数
    delay=1.0,  # 默认为 1.0，指定下载请求之间的延迟时间
):
    """Load and vectorize the 20 newsgroups dataset (classification).

    Download it if necessary.

    This is a convenience function; the transformation is done using the
    default settings for
    :class:`~sklearn.feature_extraction.text.CountVectorizer`. For more
    advanced usage (stopword filtering, n-gram extraction, etc.), combine
    fetch_20newsgroups with a custom
    :class:`~sklearn.feature_extraction.text.CountVectorizer`,
    :class:`~sklearn.feature_extraction.text.HashingVectorizer`,
    :class:`~sklearn.feature_extraction.text.TfidfTransformer` or
    :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.

    The resulting counts are normalized using
    :func:`sklearn.preprocessing.normalize` unless normalize is set to False.

    =================   ==========
    Classes                     20
    Samples total            18846
    Dimensionality          130107
    Features                  real
    =================   ==========

    Read more in the :ref:`User Guide <20newsgroups_dataset>`.

    Parameters
    ----------
    subset : {'train', 'test', 'all'}, default='train'
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    remove : tuple, default=()
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        newsgroup posts, preventing classifiers from overfitting on
        metadata.

        'headers' removes newsgroup headers, 'footers' removes blocks at the
        ends of posts that look like signatures, and 'quotes' removes lines
        that appear to be quoting another post.

    data_home : str or path-like, default=None
        Specify an download and cache folder for the datasets. If None,
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : bool, default=False
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

        .. versionadded:: 0.20
    normalize : bool, default=True
        If True, normalizes each document's feature vector to unit norm using
        :func:`sklearn.preprocessing.normalize`.
        # 是否进行归一化处理，默认为True，使用sklearn.preprocessing.normalize函数将每个文档的特征向量归一化为单位范数。

        .. versionadded:: 0.22
        # 版本新增功能：0.22版本添加了此选项。

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string, or categorical). The target is
        a pandas DataFrame or Series depending on the number of
        `target_columns`.
        # 是否返回DataFrame格式的数据，默认为False。如果为True，返回的数据将是一个包含合适数据类型列（数值、字符串或分类）的pandas DataFrame。根据`target_columns`的数量，目标值可以是pandas的DataFrame或Series。

        .. versionadded:: 0.24
        # 版本新增功能：0.24版本添加了此选项。

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.
        # 在遇到HTTP错误时的重试次数，默认为3次。

        .. versionadded:: 1.5
        # 版本新增功能：1.5版本添加了此选项。

    delay : float, default=1.0
        Number of seconds between retries.
        # 每次重试之间的延迟时间（秒），默认为1.0秒。

        .. versionadded:: 1.5
        # 版本新增功能：1.5版本添加了此选项。

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data: {sparse matrix, dataframe} of shape (n_samples, n_features)
            The input data matrix. If ``as_frame`` is `True`, ``data`` is
            a pandas DataFrame with sparse columns.
        target: {ndarray, series} of shape (n_samples,)
            The target labels. If ``as_frame`` is `True`, ``target`` is a
            pandas Series.
        target_names: list of shape (n_classes,)
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        frame: dataframe of shape (n_samples, n_features + 1)
            Only present when `as_frame=True`. Pandas DataFrame with ``data``
            and ``target``.
            # 当`as_frame=True`时才存在。包含`data`和`target`的Pandas DataFrame，形状为（n_samples, n_features + 1）。

            .. versionadded:: 0.24
            # 版本新增功能：0.24版本添加了此选项。

    (data, target) : tuple if ``return_X_y`` is True
        `data` and `target` would be of the format defined in the `Bunch`
        description above.
        # 如果`return_X_y`为True，则返回一个元组`(data, target)`，格式与上面`Bunch`描述中定义的格式相同。

        .. versionadded:: 0.20
        # 版本新增功能：0.20版本添加了此选项。

    Examples
    --------
    >>> from sklearn.datasets import fetch_20newsgroups_vectorized
    >>> newsgroups_vectorized = fetch_20newsgroups_vectorized(subset='test')
    >>> newsgroups_vectorized.data.shape
    (7532, 130107)
    >>> newsgroups_vectorized.target.shape
    (7532,)
    """
    data_home = get_data_home(data_home=data_home)
    # 获取数据存储目录，使用get_data_home函数，默认使用data_home参数作为目录。

    filebase = "20newsgroup_vectorized"
    if remove:
        filebase += "remove-" + "-".join(remove)
    # 如果remove不为空，修改文件基础名以包含remove内容。

    target_file = _pkl_filepath(data_home, filebase + ".pkl")
    # 构建目标文件路径，使用_pkl_filepath函数，包含data_home和filebase加上.pkl后缀。

    # we shuffle but use a fixed seed for the memoization
    data_train = fetch_20newsgroups(
        data_home=data_home,
        subset="train",
        categories=None,
        shuffle=True,
        random_state=12,
        remove=remove,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )
    # 获取训练数据集，使用fetch_20newsgroups函数，设置参数并根据需要下载数据。

    data_test = fetch_20newsgroups(
        data_home=data_home,
        subset="test",
        categories=None,
        shuffle=True,
        random_state=12,
        remove=remove,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )
    # 获取测试数据集，使用fetch_20newsgroups函数，设置参数并根据需要下载数据。
    # 检查目标文件是否存在
    if os.path.exists(target_file):
        try:
            # 尝试从目标文件中加载训练数据、测试数据和特征名
            X_train, X_test, feature_names = joblib.load(target_file)
        except ValueError as e:
            # 如果加载失败，抛出值错误，指示可能是因为使用了不兼容的 scikit-learn 版本
            raise ValueError(
                f"The cached dataset located in {target_file} was fetched "
                "with an older scikit-learn version and it is not compatible "
                "with the scikit-learn version imported. You need to "
                f"manually delete the file: {target_file}."
            ) from e
    else:
        # 如果目标文件不存在，则创建一个新的向量化器
        vectorizer = CountVectorizer(dtype=np.int16)
        # 对训练数据进行向量化，并转换成压缩稀疏行格式
        X_train = vectorizer.fit_transform(data_train.data).tocsr()
        # 对测试数据进行向量化，并转换成压缩稀疏行格式
        X_test = vectorizer.transform(data_test.data).tocsr()
        # 获取特征名列表
        feature_names = vectorizer.get_feature_names_out()

        # 将训练数据、测试数据和特征名保存到目标文件中，使用压缩级别 9
        joblib.dump((X_train, X_test, feature_names), target_file, compress=9)

    # 如果需要进行归一化操作
    if normalize:
        # 将训练数据和测试数据类型转换为 np.float64
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        # 对训练数据进行归一化处理，inplace 操作
        preprocessing.normalize(X_train, copy=False)
        # 对测试数据进行归一化处理，inplace 操作
        preprocessing.normalize(X_test, copy=False)

    # 获取训练数据的目标类别名称列表
    target_names = data_train.target_names

    # 根据参数 subset 的值选择数据集
    if subset == "train":
        # 如果 subset 为 "train"，则选取训练数据集和对应的目标值
        data = X_train
        target = data_train.target
    elif subset == "test":
        # 如果 subset 为 "test"，则选取测试数据集和对应的目标值
        data = X_test
        target = data_test.target
    elif subset == "all":
        # 如果 subset 为 "all"，则合并训练数据集和测试数据集，生成一个整体数据集
        data = sp.vstack((X_train, X_test)).tocsr()
        target = np.concatenate((data_train.target, data_test.target))

    # 加载数据集的描述信息
    fdescr = load_descr("twenty_newsgroups.rst")

    # 初始化 DataFrame 为 None
    frame = None
    # 设置目标名称为单一的 "category_class" 列表
    target_name = ["category_class"]

    # 如果需要返回 DataFrame 格式的数据
    if as_frame:
        # 调用内部函数将数据转换为 DataFrame 格式
        frame, data, target = _convert_data_dataframe(
            "fetch_20newsgroups_vectorized",
            data,
            target,
            feature_names,
            target_names=target_name,
            sparse_data=True,
        )

    # 如果需要返回 X 和 y，直接返回数据和目标值
    if return_X_y:
        return data, target

    # 返回 Bunch 对象，包含数据、目标值、DataFrame、目标名称、特征名称和描述信息
    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        feature_names=feature_names,
        DESCR=fdescr,
    )
```