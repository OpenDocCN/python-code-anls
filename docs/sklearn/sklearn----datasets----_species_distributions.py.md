# `D:\src\scipysrc\scikit-learn\sklearn\datasets\_species_distributions.py`

```
# 导入日志模块，用于记录程序运行时的信息
import logging
# 导入 BytesIO 模块，用于在内存中操作二进制数据流
from io import BytesIO
# 导入数字类型的数据类型验证模块
from numbers import Integral, Real
# 导入 PathLike 模块和相关函数，用于处理文件路径
from os import PathLike, makedirs, remove
# 导入 exists 函数，用于检查文件或目录是否存在
from os.path import exists
# 导入 joblib 模块，用于并行执行任务
import joblib
# 导入 numpy 模块，用于科学计算
import numpy as np

# 导入 Bunch 类，用于将一组数据打包成一个对象
from ..utils import Bunch
# 导入参数验证模块中的 Interval 和 validate_params 函数
from ..utils._param_validation import Interval, validate_params
# 导入 get_data_home 函数，用于获取数据存储路径
from . import get_data_home
# 导入 _base 模块中的相关函数和类
from ._base import RemoteFileMetadata, _fetch_remote, _pkl_filepath

# 定义远程数据文件的元数据 RemoteFileMetadata 对象 SAMPLES
SAMPLES = RemoteFileMetadata(
    filename="samples.zip",
    url="https://ndownloader.figshare.com/files/5976075",
    checksum="abb07ad284ac50d9e6d20f1c4211e0fd3c098f7f85955e89d321ee8efe37ac28",
)

# 定义远程数据文件的元数据 RemoteFileMetadata 对象 COVERAGES
COVERAGES = RemoteFileMetadata(
    filename="coverages.zip",
    url="https://ndownloader.figshare.com/files/5976078",
    checksum="4d862674d72e79d6cee77e63b98651ec7926043ba7d39dcb31329cf3f6073807",
)

# 定义数据归档文件名
DATA_ARCHIVE_NAME = "species_coverage.pkz"

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)


def _load_coverage(F, header_length=6, dtype=np.int16):
    """从打开的文件对象中加载覆盖文件数据。

    Parameters
    ----------
    F : file object
        打开的文件对象。
    header_length : int, optional
        头部长度，默认为 6。
    dtype : numpy data type, optional
        返回的数组数据类型，默认为 np.int16。

    Returns
    -------
    M : numpy.ndarray
        指定数据类型的 numpy 数组，包含加载的数据。
    """
    # 读取指定行数的文件头信息，并构建为字典
    header = [F.readline() for _ in range(header_length)]
    make_tuple = lambda t: (t.split()[0], float(t.split()[1]))
    header = dict([make_tuple(line) for line in header])

    # 从文件中加载数据到 numpy 数组 M
    M = np.loadtxt(F, dtype=dtype)

    # 获取无数据值，并将其替换为指定值
    nodata = int(header[b"NODATA_value"])
    if nodata != -9999:
        M[nodata] = -9999
    return M


def _load_csv(F):
    """加载 CSV 文件。

    Parameters
    ----------
    F : file object
        打开的字节模式 CSV 文件对象。

    Returns
    -------
    rec : np.ndarray
        表示数据的记录数组。
    """
    # 读取 CSV 文件的列名，并以逗号分隔
    names = F.readline().decode("ascii").strip().split(",")

    # 加载 CSV 文件数据到 numpy 的记录数组 rec
    rec = np.loadtxt(F, skiprows=0, delimiter=",", dtype="S22,f4,f4")
    rec.dtype.names = names
    # 返回递归函数 rec 的执行结果
    return rec
# 构建地图网格从批处理对象中

def construct_grids(batch):
    """Construct the map grid from the batch object

    Parameters
    ----------
    batch : Batch object
        The object returned by :func:`fetch_species_distributions`

    Returns
    -------
    (xgrid, ygrid) : 1-D arrays
        The grid corresponding to the values in batch.coverages
    """
    # x,y coordinates for corner cells
    xmin = batch.x_left_lower_corner + batch.grid_size
    xmax = xmin + (batch.Nx * batch.grid_size)
    ymin = batch.y_left_lower_corner + batch.grid_size
    ymax = ymin + (batch.Ny * batch.grid_size)

    # x coordinates of the grid cells
    xgrid = np.arange(xmin, xmax, batch.grid_size)
    # y coordinates of the grid cells
    ygrid = np.arange(ymin, ymax, batch.grid_size)

    return (xgrid, ygrid)


@validate_params(
    {
        "data_home": [str, PathLike, None],
        "download_if_missing": ["boolean"],
        "n_retries": [Interval(Integral, 1, None, closed="left")],
        "delay": [Interval(Real, 0.0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def fetch_species_distributions(
    *,
    data_home=None,
    download_if_missing=True,
    n_retries=3,
    delay=1.0,
):
    """Loader for species distribution dataset from Phillips et. al. (2006).

    Read more in the :ref:`User Guide <species_distribution_dataset>`.

    Parameters
    ----------
    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.

        .. versionadded:: 1.5

    delay : float, default=1.0
        Number of seconds between retries.

        .. versionadded:: 1.5

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        coverages : array, shape = [14, 1592, 1212]
            These represent the 14 features measured
            at each point of the map grid.
            The latitude/longitude values for the grid are discussed below.
            Missing data is represented by the value -9999.
        train : record array, shape = (1624,)
            The training points for the data.  Each point has three fields:

            - train['species'] is the species name
            - train['dd long'] is the longitude, in degrees
            - train['dd lat'] is the latitude, in degrees
        test : record array, shape = (620,)
            The test points for the data.  Same format as the training data.
        Nx, Ny : integers
            The number of longitudes (x) and latitudes (y) in the grid
        x_left_lower_corner, y_left_lower_corner : floats
            The (x,y) position of the lower-left corner, in degrees
        grid_size : float
            The spacing between points of the grid, in degrees

    Notes
    -----

    This dataset represents the geographic distribution of species.
    The dataset is provided by Phillips et. al. (2006).

    The two species are:

    - `"Bradypus variegatus"
      <http://www.iucnredlist.org/details/3038/0>`_ ,
      the Brown-throated Sloth.

    - `"Microryzomys minutus"
      <http://www.iucnredlist.org/details/13408/0>`_ ,
      also known as the Forest Small Rice Rat, a rodent that lives in Peru,
      Colombia, Ecuador, Peru, and Venezuela.

    - For an example of using this dataset with scikit-learn, see
      :ref:`examples/applications/plot_species_distribution_modeling.py
      <sphx_glr_auto_examples_applications_plot_species_distribution_modeling.py>`.

    References
    ----------

    * `"Maximum entropy modeling of species geographic distributions"
      <http://rob.schapire.net/papers/ecolmod.pdf>`_
      S. J. Phillips, R. P. Anderson, R. E. Schapire - Ecological Modelling,
      190:231-259, 2006.

    Examples
    --------
    >>> from sklearn.datasets import fetch_species_distributions
    >>> species = fetch_species_distributions()
    >>> species.train[:5]
    array([(b'microryzomys_minutus', -64.7   , -17.85  ),
           (b'microryzomys_minutus', -67.8333, -16.3333),
           (b'microryzomys_minutus', -67.8833, -16.3   ),
           (b'microryzomys_minutus', -67.8   , -16.2667),
           (b'microryzomys_minutus', -67.9833, -15.9   )],
          dtype=[('species', 'S22'), ('dd long', '<f4'), ('dd lat', '<f4')])
    """

    # 获取数据存储的路径，确保路径存在，若不存在则创建
    data_home = get_data_home(data_home)
    if not exists(data_home):
        makedirs(data_home)

    # Define parameters for the data files.  These should not be changed
    # unless the data model changes.  They will be saved in the npz file
    # with the downloaded data.
    # 定义额外的参数字典，包含地理网格的左下角坐标、网格数量以及网格大小
    extra_params = dict(
        x_left_lower_corner=-94.8,
        Nx=1212,
        y_left_lower_corner=-56.05,
        Ny=1592,
        grid_size=0.05,
    )
    # 设置数据类型为 16 位整数
    dtype = np.int16

    # 构建存档文件路径
    archive_path = _pkl_filepath(data_home, DATA_ARCHIVE_NAME)

    # 如果存档文件不存在
    if not exists(archive_path):
        # 如果允许下载缺失数据
        if not download_if_missing:
            raise OSError("Data not found and `download_if_missing` is False")
        
        # 下载物种数据文件并保存到本地
        logger.info("Downloading species data from %s to %s" % (SAMPLES.url, data_home))
        samples_path = _fetch_remote(
            SAMPLES, dirname=data_home, n_retries=n_retries, delay=delay
        )
        with np.load(samples_path) as X:  # samples.zip is a valid npz
            # 遍历压缩文件中的数据文件
            for f in X.files:
                fhandle = BytesIO(X[f])
                # 如果文件名中包含'train'，加载为训练数据
                if "train" in f:
                    train = _load_csv(fhandle)
                # 如果文件名中包含'test'，加载为测试数据
                if "test" in f:
                    test = _load_csv(fhandle)
        # 删除下载的临时样本数据文件
        remove(samples_path)

        # 下载覆盖率数据文件并保存到本地
        logger.info(
            "Downloading coverage data from %s to %s" % (COVERAGES.url, data_home)
        )
        coverages_path = _fetch_remote(
            COVERAGES, dirname=data_home, n_retries=n_retries, delay=delay
        )
        with np.load(coverages_path) as X:  # coverages.zip is a valid npz
            coverages = []
            # 遍历覆盖率数据文件
            for f in X.files:
                fhandle = BytesIO(X[f])
                # 记录日志，表示正在转换当前文件
                logger.debug(" - converting {}".format(f))
                # 加载覆盖率数据并追加到列表中
                coverages.append(_load_coverage(fhandle))
            # 将覆盖率数据转换为指定的数据类型
            coverages = np.asarray(coverages, dtype=dtype)
        # 删除下载的临时覆盖率数据文件
        remove(coverages_path)

        # 创建一个包含各种数据的命名空间对象
        bunch = Bunch(coverages=coverages, test=test, train=train, **extra_params)
        # 使用高压缩比将对象保存到存档文件中
        joblib.dump(bunch, archive_path, compress=9)
    else:
        # 如果存档文件已存在，直接加载存档文件中的数据对象
        bunch = joblib.load(archive_path)

    # 返回最终的数据对象
    return bunch
```