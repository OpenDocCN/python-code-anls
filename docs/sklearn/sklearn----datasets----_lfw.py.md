# `D:\src\scipysrc\scikit-learn\sklearn\datasets\_lfw.py`

```
"""Labeled Faces in the Wild (LFW) dataset

This dataset is a collection of JPEG pictures of famous people collected
over the internet, all details are available on the official website:

    http://vis-www.cs.umass.edu/lfw/
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import logging
from numbers import Integral, Real
from os import PathLike, listdir, makedirs, remove
from os.path import exists, isdir, join

import numpy as np
from joblib import Memory

from ..utils import Bunch
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.fixes import tarfile_extractall
from ._base import (
    RemoteFileMetadata,
    _fetch_remote,
    get_data_home,
    load_descr,
)

logger = logging.getLogger(__name__)

# The original data can be found in:
# http://vis-www.cs.umass.edu/lfw/lfw.tgz
ARCHIVE = RemoteFileMetadata(
    filename="lfw.tgz",
    url="https://ndownloader.figshare.com/files/5976018",
    checksum="055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0",
)

# The original funneled data can be found in:
# http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
FUNNELED_ARCHIVE = RemoteFileMetadata(
    filename="lfw-funneled.tgz",
    url="https://ndownloader.figshare.com/files/5976015",
    checksum="b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a",
)

# The original target data can be found in:
# http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt',
# http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt',
# http://vis-www.cs.umass.edu/lfw/pairs.txt',
TARGETS = (
    RemoteFileMetadata(
        filename="pairsDevTrain.txt",
        url="https://ndownloader.figshare.com/files/5976012",
        checksum="1d454dada7dfeca0e7eab6f65dc4e97a6312d44cf142207be28d688be92aabfa",
    ),
    RemoteFileMetadata(
        filename="pairsDevTest.txt",
        url="https://ndownloader.figshare.com/files/5976009",
        checksum="7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c",
    ),
    RemoteFileMetadata(
        filename="pairs.txt",
        url="https://ndownloader.figshare.com/files/5976006",
        checksum="ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592",
    ),
)


#
# Common private utilities for data fetching from the original LFW website
# local disk caching, and image decoding.
#

def _check_fetch_lfw(
    data_home=None, funneled=True, download_if_missing=True, n_retries=3, delay=1.0
):
    """Helper function to download any missing LFW data"""

    # Determine the directory to store LFW data, using the provided or default data home
    data_home = get_data_home(data_home=data_home)
    lfw_home = join(data_home, "lfw_home")

    # If the directory for LFW data does not exist, create it
    if not exists(lfw_home):
        makedirs(lfw_home)
    # 遍历目标列表 TARGETS
    for target in TARGETS:
        # 构建目标文件的完整路径
        target_filepath = join(lfw_home, target.filename)
        # 检查目标文件是否存在
        if not exists(target_filepath):
            # 如果文件不存在，并且设置了下载选项，则下载目标文件
            if download_if_missing:
                # 记录日志，显示正在下载 LFW 元数据
                logger.info("Downloading LFW metadata: %s", target.url)
                # 调用 _fetch_remote 函数下载文件
                _fetch_remote(
                    target, dirname=lfw_home, n_retries=n_retries, delay=delay
                )
            else:
                # 如果文件不存在且不允许下载，则抛出 OSError 异常
                raise OSError("%s is missing" % target_filepath)

    # 根据 funneled 变量决定数据文件夹路径和压缩包的变量
    if funneled:
        # 如果 funneled 为 True，则数据文件夹路径为 lfw_home 下的 "lfw_funneled"
        data_folder_path = join(lfw_home, "lfw_funneled")
        # 压缩包变量为 FUNNELED_ARCHIVE
        archive = FUNNELED_ARCHIVE
    else:
        # 如果 funneled 为 False，则数据文件夹路径为 lfw_home 下的 "lfw"
        data_folder_path = join(lfw_home, "lfw")
        # 压缩包变量为 ARCHIVE
        archive = ARCHIVE

    # 检查数据文件夹路径是否存在
    if not exists(data_folder_path):
        # 构建压缩包路径
        archive_path = join(lfw_home, archive.filename)
        # 检查压缩包路径是否存在
        if not exists(archive_path):
            # 如果压缩包不存在，并且设置了下载选项，则下载压缩包
            if download_if_missing:
                # 记录日志，显示正在下载 LFW 数据（大约 200MB）
                logger.info("Downloading LFW data (~200MB): %s", archive.url)
                # 调用 _fetch_remote 函数下载压缩包
                _fetch_remote(
                    archive, dirname=lfw_home, n_retries=n_retries, delay=delay
                )
            else:
                # 如果压缩包不存在且不允许下载，则抛出 OSError 异常
                raise OSError("%s is missing" % archive_path)

        import tarfile

        # 记录调试信息，显示正在解压数据到 data_folder_path
        logger.debug("Decompressing the data archive to %s", data_folder_path)
        # 使用 tarfile 打开压缩包
        with tarfile.open(archive_path, "r:gz") as fp:
            # 解压所有文件到 lfw_home 下
            tarfile_extractall(fp, path=lfw_home)
        # 删除解压后的压缩包
        remove(archive_path)

    # 返回 LFW 数据主目录和数据文件夹路径
    return lfw_home, data_folder_path
# 执行内部用于加载图像的函数
def _load_imgs(file_paths, slice_, color, resize):
    """Internally used to load images"""
    try:
        from PIL import Image  # 尝试导入PIL库，用于处理图像文件
    except ImportError:
        raise ImportError(
            "The Python Imaging Library (PIL) is required to load data "
            "from jpeg files. Please refer to "
            "https://pillow.readthedocs.io/en/stable/installation.html "
            "for installing PIL."
        )

    # 计算根据调用者提供的slice_参数加载图像的部分
    default_slice = (slice(0, 250), slice(0, 250))  # 默认切片大小
    if slice_ is None:
        slice_ = default_slice  # 如果未提供slice_参数，则使用默认切片
    else:
        slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))

    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)  # 计算高度
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)  # 计算宽度

    if resize is not None:
        resize = float(resize)
        h = int(resize * h)  # 根据resize参数调整高度
        w = int(resize * w)  # 根据resize参数调整宽度

    # 分配一些连续内存来存储解码后的图像切片
    n_faces = len(file_paths)  # 文件路径列表的长度即为面部图像数量
    if not color:
        faces = np.zeros((n_faces, h, w), dtype=np.float32)  # 如果不是彩色图像，分配灰度图像数组内存
    else:
        faces = np.zeros((n_faces, h, w, 3), dtype=np.float32)  # 如果是彩色图像，分配彩色图像数组内存

    # 迭代所有收集的文件路径，将JPEG文件加载为NumPy数组
    for i, file_path in enumerate(file_paths):
        if i % 1000 == 0:
            logger.debug("Loading face #%05d / %05d", i + 1, n_faces)  # 每加载1000个图像输出调试信息

        # 检查JPEG读取是否成功。更多详情请参考问题＃3594。
        pil_img = Image.open(file_path)  # 使用PIL库打开图像文件
        pil_img = pil_img.crop(
            (w_slice.start, h_slice.start, w_slice.stop, h_slice.stop)
        )  # 根据切片参数裁剪图像
        if resize is not None:
            pil_img = pil_img.resize((w, h))  # 根据resize参数调整图像大小
        face = np.asarray(pil_img, dtype=np.float32)  # 将PIL图像转换为NumPy数组

        if face.ndim == 0:
            raise RuntimeError(
                "Failed to read the image file %s, "
                "Please make sure that libjpeg is installed" % file_path
            )  # 如果图像维度为0，则抛出运行时错误

        face /= 255.0  # 将uint8编码的颜色值缩放到[0.0, 1.0]范围内
        if not color:
            face = face.mean(axis=2)  # 如果不是彩色图像，则计算灰度值

        faces[i, ...] = face  # 将处理后的图像数据存入faces数组

    return faces  # 返回加载的图像数据数组


#
# Task #1: 人脸识别图片带名称
#
    # 遍历数据文件夹中的人名列表，按字母顺序排序
    for person_name in sorted(listdir(data_folder_path)):
        # 构建当前人名对应的文件夹路径
        folder_path = join(data_folder_path, person_name)
        # 如果路径不是文件夹，则跳过当前循环
        if not isdir(folder_path):
            continue
        # 获取当前文件夹下所有文件的路径，并按字母顺序排序
        paths = [join(folder_path, f) for f in sorted(listdir(folder_path))]
        # 计算当前文件夹下的图片数量
        n_pictures = len(paths)
        # 如果图片数量大于等于指定的最小人脸数
        if n_pictures >= min_faces_per_person:
            # 将人名中的下划线替换为空格，并将该人名重复 n_pictures 次加入到 person_names 列表中
            person_name = person_name.replace("_", " ")
            person_names.extend([person_name] * n_pictures)
            # 将当前文件夹中所有图片的路径添加到 file_paths 列表中
            file_paths.extend(paths)

    # 统计获取到的所有图片文件的总数
    n_faces = len(file_paths)
    # 如果没有获取到任何图片文件，则抛出 ValueError 异常
    if n_faces == 0:
        raise ValueError(
            "min_faces_per_person=%d is too restrictive" % min_faces_per_person
        )

    # 使用 numpy 获取唯一的目标人名列表
    target_names = np.unique(person_names)
    # 将 person_names 列表中的每个人名映射为其在 target_names 中的索引，构成 target 数组
    target = np.searchsorted(target_names, person_names)

    # 调用 _load_imgs 函数加载所有图片数据
    faces = _load_imgs(file_paths, slice_, color, resize)

    # 使用确定性随机数生成器打乱 indices 数组，以避免相邻的图片属于同一个人，从而破坏某些交叉验证和学习算法（如 SGD 和在线 k-means）的 IID 假设
    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    # 根据打乱后的索引重新排列 faces 和 target 数组
    faces, target = faces[indices], target[indices]

    # 返回加载的所有图片数据、对应的目标标签和唯一的目标人名列表
    return faces, target, target_names
@validate_params(
    {
        "data_home": [str, PathLike, None],  # 数据集存储路径，可以是字符串、路径对象或者None
        "funneled": ["boolean"],  # 是否使用变形数据集的布尔值标志
        "resize": [Interval(Real, 0, None, closed="neither"), None],  # 图像缩放比例，实数且大于0，或者None
        "min_faces_per_person": [Interval(Integral, 0, None, closed="left"), None],  # 每个人至少要包含的面部图像数量，整数且不小于0
        "color": ["boolean"],  # 是否保留彩色通道的布尔值标志
        "slice_": [tuple, Hidden(None)],  # 自定义的二维切片元组，用于提取感兴趣的图像部分
        "download_if_missing": ["boolean"],  # 如果本地缺少数据，是否尝试下载的布尔值标志
        "return_X_y": ["boolean"],  # 是否返回数据集的特征矩阵X和目标向量y的布尔值标志
        "n_retries": [Interval(Integral, 1, None, closed="left")],  # 下载时的重试次数，整数且不小于1
        "delay": [Interval(Real, 0.0, None, closed="neither")],  # 下载重试之间的延迟时间，实数且大于等于0
    },
    prefer_skip_nested_validation=True,
)
def fetch_lfw_people(
    *,
    data_home=None,  # 数据集存储路径，默认为None
    funneled=True,  # 是否使用变形数据集，默认为True
    resize=0.5,  # 图像缩放比例，默认为0.5
    min_faces_per_person=0,  # 每个人至少包含的面部图像数量，默认为0
    color=False,  # 是否保留彩色通道，默认为False
    slice_=(slice(70, 195), slice(78, 172)),  # 自定义的二维切片元组，默认为(slice(70, 195), slice(78, 172))
    download_if_missing=True,  # 如果本地缺少数据是否尝试下载，默认为True
    return_X_y=False,  # 是否返回数据集的特征矩阵X和目标向量y，默认为False
    n_retries=3,  # 下载时的重试次数，默认为3
    delay=1.0,  # 下载重试之间的延迟时间，默认为1.0
):
    """Load the Labeled Faces in the Wild (LFW) people dataset \
(classification).

    Download it if necessary.

    =================   =======================
    Classes                                5749
    Samples total                         13233
    Dimensionality                         5828
    Features            real, between 0 and 255
    =================   =======================

    For a usage example of this dataset, see
    :ref:`sphx_glr_auto_examples_applications_plot_face_recognition.py`.

    Read more in the :ref:`User Guide <labeled_faces_in_the_wild_dataset>`.

    Parameters
    ----------
    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    funneled : bool, default=True
        Download and use the funneled variant of the dataset.

    resize : float or None, default=0.5
        Ratio used to resize the each face picture. If `None`, no resizing is
        performed.

    min_faces_per_person : int, default=None
        The extracted dataset will only retain pictures of people that have at
        least `min_faces_per_person` different pictures.

    color : bool, default=False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than the shape with color = False.

    slice_ : tuple of slice, default=(slice(70, 195), slice(78, 172))
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.
    """
    return_X_y : bool, default=False
        如果为True，则返回`(dataset.data, dataset.target)`，而不是一个Bunch对象。
        关于`dataset.data`和`dataset.target`对象的更多信息，请参见下文。

        .. versionadded:: 0.20
            新版本添加的功能说明

    n_retries : int, default=3
        在遇到HTTP错误时的重试次数。

        .. versionadded:: 1.5
            新版本添加的功能说明

    delay : float, default=1.0
        重试之间的延迟时间，单位为秒。

        .. versionadded:: 1.5
            新版本添加的功能说明

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        类似字典的对象，具有以下属性。

        data : numpy array of shape (13233, 2914)
            每行对应一个大小为62 x 47像素的平坦化脸部图像。
            更改`slice_`或resize参数将改变输出的形状。
        images : numpy array of shape (13233, 62, 47)
            每行是一个脸部图像，对应数据集中的一个人物。
            更改`slice_`或resize参数将改变输出的形状。
        target : numpy array of shape (13233,)
            每个脸部图像对应的标签。
            这些标签范围从0到5748，对应于人物的ID。
        target_names : numpy array of shape (5749,)
            数据集中所有人物的名称。
            数组中的位置对应于目标数组中的人物ID。
        DESCR : str
            Labeled Faces in the Wild（LFW）数据集的描述。

    (data, target) : tuple if ``return_X_y`` is True
        如果`return_X_y`为True，则返回两个ndarray的元组。
        第一个包含形状为(n_samples, n_features)的二维数组，每行代表一个样本，每列代表特征。
        第二个ndarray的形状为(n_samples,)，包含目标样本。

        .. versionadded:: 0.20
            新版本添加的功能说明

    Examples
    --------
    >>> from sklearn.datasets import fetch_lfw_people
    >>> lfw_people = fetch_lfw_people()
    >>> lfw_people.data.shape
    (13233, 2914)
    >>> lfw_people.target.shape
    (13233,)
    >>> for name in lfw_people.target_names[:5]:
    ...    print(name)
    AJ Cook
    AJ Lamas
    Aaron Eckhart
    Aaron Guiel
    Aaron Patterson

    """
    lfw_home, data_folder_path = _check_fetch_lfw(
        data_home=data_home,
        funneled=funneled,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )
    logger.debug("Loading LFW people faces from %s", lfw_home)

    # wrap the loader in a memoizing function that will return memmaped data
    # arrays for optimal memory usage
    # 将加载器包装在一个记忆函数中，以便以最佳内存使用返回memmaped数据数组
    m = Memory(location=lfw_home, compress=6, verbose=0)
    load_func = m.cache(_fetch_lfw_people)

    # load and memoize the pairs as np arrays
    # 加载并使用np数组作为记忆
    # 调用加载函数 `load_func`，加载人脸数据集
    faces, target, target_names = load_func(
        data_folder_path,           # 数据文件夹路径
        resize=resize,              # 是否调整大小
        min_faces_per_person=min_faces_per_person,  # 每个人脸最小样本数
        color=color,                # 是否彩色
        slice_=slice_,              # 数据切片
    )

    # 将 faces 数组重新形状为二维数组，第一维是样本数，第二维是特征数
    X = faces.reshape(len(faces), -1)

    # 载入人脸数据集的描述信息
    fdescr = load_descr("lfw.rst")

    # 如果需要返回 X 和 target 数组，则直接返回它们
    if return_X_y:
        return X, target

    # 将结果打包为一个 Bunch 实例并返回
    return Bunch(
        data=X,                     # 特征数据数组
        images=faces,               # 原始人脸图像数组
        target=target,              # 目标数组，标识每张图像对应的人物类别
        target_names=target_names,  # 人物类别名称列表
        DESCR=fdescr                # 数据集描述信息
    )
# Task #2:  Face Verification on pairs of face pictures
#

# 定义一个函数用于获取 LFW 数据集的配对数据集
def _fetch_lfw_pairs(
    index_file_path, data_folder_path, slice_=None, color=False, resize=None
):
    """Perform the actual data loading for the LFW pairs dataset

    This operation is meant to be cached by a joblib wrapper.
    """
    # 打开索引文件并解析内容，获取配对的元数据行数
    with open(index_file_path, "rb") as index_file:
        split_lines = [ln.decode().strip().split("\t") for ln in index_file]
    
    # 从解析后的行中筛选出有效的配对规范
    pair_specs = [sl for sl in split_lines if len(sl) > 2]
    # 计算配对的数量
    n_pairs = len(pair_specs)

    # 初始化目标标签数组，用于存储每对配对的标签（同人或不同人）
    target = np.zeros(n_pairs, dtype=int)
    # 存储每个文件的路径
    file_paths = list()
    
    # 遍历每对配对的元数据行，找到文件名并加载到内存中
    for i, components in enumerate(pair_specs):
        if len(components) == 3:
            # 如果有三个组件，则代表同一个人的两张图片
            target[i] = 1
            pair = (
                (components[0], int(components[1]) - 1),
                (components[0], int(components[2]) - 1),
            )
        elif len(components) == 4:
            # 如果有四个组件，则代表不同人的两张图片
            target[i] = 0
            pair = (
                (components[0], int(components[1]) - 1),
                (components[2], int(components[3]) - 1),
            )
        else:
            # 如果组件数量不符合预期，则引发错误
            raise ValueError("invalid line %d: %r" % (i + 1, components))
        
        # 遍历每对图片的名称和索引，获取对应的文件路径
        for j, (name, idx) in enumerate(pair):
            try:
                person_folder = join(data_folder_path, name)
            except TypeError:
                person_folder = join(data_folder_path, str(name, "UTF-8"))
            filenames = list(sorted(listdir(person_folder)))
            file_path = join(person_folder, filenames[idx])
            file_paths.append(file_path)

    # 调用 _load_imgs 函数加载图片数据，返回加载后的图片数据和形状信息
    pairs = _load_imgs(file_paths, slice_, color, resize)
    shape = list(pairs.shape)
    n_faces = shape.pop(0)
    shape.insert(0, 2)
    shape.insert(0, n_faces // 2)
    pairs.shape = shape

    # 返回加载好的图片数据对、目标标签和标签说明信息
    return pairs, target, np.array(["Different persons", "Same person"])


# 使用装饰器 validate_params 验证参数，并定义函数 fetch_lfw_pairs 用于获取 LFW 数据集的配对数据集
@validate_params(
    {
        "subset": [StrOptions({"train", "test", "10_folds"})],
        "data_home": [str, PathLike, None],
        "funneled": ["boolean"],
        "resize": [Interval(Real, 0, None, closed="neither"), None],
        "color": ["boolean"],
        "slice_": [tuple, Hidden(None)],
        "download_if_missing": ["boolean"],
        "n_retries": [Interval(Integral, 1, None, closed="left")],
        "delay": [Interval(Real, 0.0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
# 加载 LFW 数据集的配对数据集，支持设置多个参数以自定义加载行为
def fetch_lfw_pairs(
    *,
    subset="train",
    data_home=None,
    funneled=True,
    resize=0.5,
    color=False,
    slice_=(slice(70, 195), slice(78, 172)),
    download_if_missing=True,
    n_retries=3,
    delay=1.0,
):
    """Load the Labeled Faces in the Wild (LFW) pairs dataset (classification).

    Download it if necessary.

    =================   =======================
    # 类文档和数据集基本信息的说明，列出了数据集的基本特征和链接到官方 README.txt 的描述
    Classes                                   2
    Samples total                         13233
    Dimensionality                         5828
    Features            real, between 0 and 255
    =================   =======================
    
    # 官方 README.txt 中将此任务描述为“Restricted”任务。因为不确定如何正确实现“Unrestricted”变体，所以目前不支持该变体。
    # 这里提供了 README.txt 的链接，以供进一步阅读
    In the official `README.txt`_ this task is described as the
    "Restricted" task.  As I am not sure as to implement the
    "Unrestricted" variant correctly, I left it as unsupported for now.
    
      .. _`README.txt`: http://vis-www.cs.umass.edu/lfw/README.txt
    
    # 原始图像尺寸为 250 x 250 像素，但默认的切片和调整大小参数将其减小到 62 x 47 像素。
    The original images are 250 x 250 pixels, but the default slice and resize
    arguments reduce them to 62 x 47.
    
    # 在用户指南中可以找到更多有关“labeled_faces_in_the_wild_dataset”数据集的信息
    Read more in the :ref:`User Guide <labeled_faces_in_the_wild_dataset>`.
    
    # 参数部分开始，详细描述了函数可以接受的各种参数及其默认值和功能
    
    Parameters
    ----------
    subset : {'train', 'test', '10_folds'}, default='train'
        Select the dataset to load: 'train' for the development training
        set, 'test' for the development test set, and '10_folds' for the
        official evaluation set that is meant to be used with a 10-folds
        cross validation.
    
    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By
        default all scikit-learn data is stored in '~/scikit_learn_data'
        subfolders.
    
    funneled : bool, default=True
        Download and use the funneled variant of the dataset.
    
    resize : float, default=0.5
        Ratio used to resize the each face picture.
    
    color : bool, default=False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than the shape with color = False.
    
    slice_ : tuple of slice, default=(slice(70, 195), slice(78, 172))
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background.
    
    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.
    
    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.
    
        .. versionadded:: 1.5
    
    delay : float, default=1.0
        Number of seconds between retries.
    
        .. versionadded:: 1.5
    
    # 返回值部分未提供，可能被省略或者在函数的其他部分进一步描述
    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray of shape (2200, 5828). Shape depends on ``subset``.
            Each row corresponds to 2 ravel'd face images
            of original size 62 x 47 pixels.
            Changing the ``slice_``, ``resize`` or ``subset`` parameters
            will change the shape of the output.
        pairs : ndarray of shape (2200, 2, 62, 47). Shape depends on ``subset``
            Each row has 2 face images corresponding
            to same or different person from the dataset
            containing 5749 people. Changing the ``slice_``,
            ``resize`` or ``subset`` parameters will change the shape of the
            output.
        target : numpy array of shape (2200,). Shape depends on ``subset``.
            Labels associated to each pair of images.
            The two label values being different persons or the same person.
        target_names : numpy array of shape (2,)
            Explains the target values of the target array.
            0 corresponds to "Different person", 1 corresponds to "same person".
        DESCR : str
            Description of the Labeled Faces in the Wild (LFW) dataset.

    Examples
    --------
    >>> from sklearn.datasets import fetch_lfw_pairs
    >>> lfw_pairs_train = fetch_lfw_pairs(subset='train')
    >>> list(lfw_pairs_train.target_names)
    ['Different persons', 'Same person']
    >>> lfw_pairs_train.pairs.shape
    (2200, 2, 62, 47)
    >>> lfw_pairs_train.data.shape
    (2200, 5828)
    >>> lfw_pairs_train.target.shape
    (2200,)
    """
    # 检查和获取 LFW 数据集的主目录路径
    lfw_home, data_folder_path = _check_fetch_lfw(
        data_home=data_home,
        funneled=funneled,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )
    # 记录调试信息，显示正在加载的 LFW 数据集子集和主目录路径
    logger.debug("Loading %s LFW pairs from %s", subset, lfw_home)

    # 将加载器封装在一个记忆函数中，以便返回最优内存使用的 memmaped 数据数组
    m = Memory(location=lfw_home, compress=6, verbose=0)
    load_func = m.cache(_fetch_lfw_pairs)

    # 根据请求的子集选择正确的元数据文件
    label_filenames = {
        "train": "pairsDevTrain.txt",
        "test": "pairsDevTest.txt",
        "10_folds": "pairs.txt",
    }
    if subset not in label_filenames:
        raise ValueError(
            "subset='%s' is invalid: should be one of %r"
            % (subset, list(sorted(label_filenames.keys())))
        )
    index_file_path = join(lfw_home, label_filenames[subset])

    # 加载并缓存为 np 数组的成对数据和目标标签
    pairs, target, target_names = load_func(
        index_file_path, data_folder_path, resize=resize, color=color, slice_=slice_
    )

    # 加载数据集描述文件
    fdescr = load_descr("lfw.rst")

    # 将结果打包为 Bunch 实例
    # 返回一个 Bunch 对象，该对象包含以下属性：
    # - data: 将 pairs 变量重新整形为 len(pairs) 行，每行的列数为 -1（自动计算列数）
    # - pairs: 包含数据的原始数组或矩阵
    # - target: 目标数组或矢量，通常是监督学习中的目标变量
    # - target_names: 目标变量的名称或标签
    # - DESCR: 描述性字符串或文本，通常用于描述数据集的详细信息
    return Bunch(
        data=pairs.reshape(len(pairs), -1),
        pairs=pairs,
        target=target,
        target_names=target_names,
        DESCR=fdescr,
    )
```