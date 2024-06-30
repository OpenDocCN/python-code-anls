# `D:\src\scipysrc\scipy\scipy\datasets\_fetchers.py`

```
# 从 numpy 库导入 array, frombuffer, load 函数
# numpy 是一个专门用于科学计算的库，提供了高效的多维数组对象和相关工具
from numpy import array, frombuffer, load
# 从 _registry 模块导入 registry, registry_urls
# 这些变量可能是数据集的注册信息和其对应的下载链接
from ._registry import registry, registry_urls

try:
    # 尝试导入 pooch 库，用于数据下载和缓存管理
    import pooch
except ImportError:
    # 如果导入失败，则设置 pooch 和 data_fetcher 为 None
    pooch = None
    data_fetcher = None
else:
    # 如果成功导入 pooch 库，则创建一个 data_fetcher 对象
    data_fetcher = pooch.create(
        # 使用操作系统的默认缓存文件夹
        # Pooch 使用 appdirs 库来选择适当的平台缓存目录
        path=pooch.os_cache("scipy-data"),
        # 远程数据存储在 Github 上
        # base_url 是一个必需的参数，尽管我们在注册表中覆盖了这个设置
        base_url="https://github.com/scipy/",
        # 注册表变量是数据集的名称和对应的下载地址的字典
        registry=registry,
        urls=registry_urls
    )

# fetch_data 函数用于从远程下载数据集文件
def fetch_data(dataset_name, data_fetcher=data_fetcher):
    if data_fetcher is None:
        # 如果 data_fetcher 为 None，则抛出 ImportError 异常
        raise ImportError("Missing optional dependency 'pooch' required "
                          "for scipy.datasets module. Please use pip or "
                          "conda to install 'pooch'.")
    # 调用 data_fetcher 的 fetch 方法来获取数据集的完整路径
    return data_fetcher.fetch(dataset_name)

# ascent 函数返回一个 512x512 的 8-bit 灰度图像的 ndarray 数组
def ascent():
    import pickle
    
    # 使用 fetch_data 函数下载 ascent.dat 数据集文件
    fname = fetch_data("ascent.dat")
    # 打开数据集文件，加载其中的数据并转换为 ndarray 数组
    with open(fname, 'rb') as f:
        ascent = array(pickle.load(f))
    return ascent

# electrocardiogram 函数加载一个心电图信号的示例数据
def electrocardiogram():
    """
    Load an electrocardiogram as an example for a 1-D signal.

    The returned signal is a 5 minute long electrocardiogram (ECG), a medical
    recording of the heart's electrical activity, sampled at 360 Hz.

    Returns
    -------
    ecg : ndarray
        The electrocardiogram in millivolt (mV) sampled at 360 Hz.

    Notes
    -----
    The provided signal is an excerpt (19:35 to 24:35) from the `record 208`_
    (lead MLII) provided by the MIT-BIH Arrhythmia Database [1]_ on
    PhysioNet [2]_. The excerpt includes noise induced artifacts, typical
    heartbeats as well as pathological changes.

    .. _record 208: https://physionet.org/physiobank/database/html/mitdbdir/records.htm#208

    .. versionadded:: 1.1.0

    References
    ----------
    """
    # 这个函数包含详细的文档字符串，描述了加载心电图数据的用途和返回结果
    """
    从指定的数据源获取 ECG 数据文件的文件名
    """
    fname = fetch_data("ecg.dat")
    
    """
    以加载模式打开数据文件
    """
    with load(fname) as file:
        """
        从文件对象中提取名为 "ecg" 的数据，并将其转换为整数类型
        """
        ecg = file["ecg"].astype(int)  # np.uint16 -> int
        
    """
    将原始 ADC 输出转换为毫伏（mV）单位：(ecg - adc_zero) / adc_gain
    """
    ecg = (ecg - 1024) / 200.0
    
    """
    返回处理后的 ECG 数据
    """
    return ecg
def face(gray=False):
    """
    Get a 1024 x 768, color image of a raccoon face.

    The image is derived from
    https://pixnio.com/fauna-animals/raccoons/raccoon-procyon-lotor

    Parameters
    ----------
    gray : bool, optional
        If True return 8-bit grey-scale image, otherwise return a color image

    Returns
    -------
    face : ndarray
        image of a raccoon face

    Examples
    --------
    >>> import scipy.datasets
    >>> face = scipy.datasets.face()
    >>> face.shape
    (768, 1024, 3)
    >>> face.max()
    255
    >>> face.dtype
    dtype('uint8')

    >>> import matplotlib.pyplot as plt
    >>> plt.gray()
    >>> plt.imshow(face)
    >>> plt.show()

    """
    import bz2  # 导入 bz2 模块，用于数据解压缩
    fname = fetch_data("face.dat")  # 调用 fetch_data 函数获取文件名 "face.dat"
    with open(fname, 'rb') as f:
        rawdata = f.read()  # 读取文件内容到 rawdata 变量中
    face_data = bz2.decompress(rawdata)  # 使用 bz2 解压缩 rawdata，得到面部图像数据
    face = frombuffer(face_data, dtype='uint8')  # 将解压缩后的数据转换为 uint8 类型的 ndarray
    face.shape = (768, 1024, 3)  # 设置图像数组的形状为 768x1024x3，即高度、宽度和通道数
    if gray is True:
        # 如果 gray 参数为 True，则将彩色图像转换为灰度图像
        face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
                0.07 * face[:, :, 2]).astype('uint8')
    return face  # 返回生成的面部图像数组
```