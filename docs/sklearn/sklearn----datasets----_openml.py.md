# `D:\src\scipysrc\scikit-learn\sklearn\datasets\_openml.py`

```
# 导入需要的模块
import gzip  # 提供 GZIP 文件压缩与解压功能
import hashlib  # 提供哈希算法函数
import json  # 处理 JSON 格式数据
import os  # 提供与操作系统相关的功能
import shutil  # 提供高级文件操作功能
import time  # 提供时间相关功能
from contextlib import closing  # 提供上下文管理器的辅助函数
from functools import wraps  # 提供创建装饰器的工具
from os.path import join  # 提供路径拼接功能
from tempfile import TemporaryDirectory  # 提供临时目录的创建和清理
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 提供类型提示功能

from urllib.error import HTTPError, URLError  # 处理 URL 相关的错误
from urllib.request import Request, urlopen  # 发送 HTTP 请求和获取响应
from warnings import warn  # 发出警告消息

import numpy as np  # 提供数值计算功能，如数组操作

from ..utils import Bunch  # 导入自定义工具类 Bunch
from ..utils._optional_dependencies import check_pandas_support  # 导入 Pandas 支持检查函数
from ..utils._param_validation import (  # 导入参数验证相关功能
    Integral,  # 整数验证
    Interval,  # 区间验证
    Real,  # 实数验证
    StrOptions,  # 字符串选项验证
    validate_params,  # 参数验证函数
)
from . import get_data_home  # 导入获取数据存储路径函数
from ._arff_parser import load_arff_from_gzip_file  # 导入从 GZIP 文件中加载 ARFF 数据的函数

__all__ = ["fetch_openml"]  # 模块的公开接口列表

_OPENML_PREFIX = "https://api.openml.org/"  # OpenML API 的基础 URL
_SEARCH_NAME = "api/v1/json/data/list/data_name/{}/limit/2"  # 根据数据集名称搜索数据集的 API 路径模板
_DATA_INFO = "api/v1/json/data/{}"  # 获取数据集详细信息的 API 路径模板
_DATA_FEATURES = "api/v1/json/data/features/{}"  # 获取数据集特征信息的 API 路径模板
_DATA_QUALITIES = "api/v1/json/data/qualities/{}"  # 获取数据集质量信息的 API 路径模板
_DATA_FILE = "data/v1/download/{}"  # 下载数据集文件的 API 路径模板

OpenmlQualitiesType = List[Dict[str, str]]  # OpenML 数据集质量信息的类型定义
OpenmlFeaturesType = List[Dict[str, str]]  # OpenML 数据集特征信息的类型定义


def _get_local_path(openml_path: str, data_home: str) -> str:
    """根据数据集路径和数据存储路径获取本地文件路径"""
    return os.path.join(data_home, "openml.org", openml_path + ".gz")


def _retry_with_clean_cache(
    openml_path: str,
    data_home: Optional[str],
    no_retry_exception: Optional[Exception] = None,
) -> Callable:
    """如果首次调用装饰的函数失败，删除本地缓存文件并再次调用。
    如果 `data_home` 为 `None`，则只调用一次。
    可以提供一个特定异常类给 `no_retry_exception` 参数，遇到该异常时不重试。
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kw):
            if data_home is None:
                return f(*args, **kw)
            try:
                return f(*args, **kw)
            except URLError:
                raise
            except Exception as exc:
                if no_retry_exception is not None and isinstance(
                    exc, no_retry_exception
                ):
                    raise
                warn("Invalid cache, redownloading file", RuntimeWarning)
                local_path = _get_local_path(openml_path, data_home)
                if os.path.exists(local_path):
                    os.unlink(local_path)
                return f(*args, **kw)

        return wrapper

    return decorator


def _retry_on_network_error(
    n_retries: int = 3, delay: float = 1.0, url: str = ""
) -> Callable:
    """如果函数调用导致网络错误，最多重试 `n_retries` 次，每次重试间隔 `delay` 秒。
    如果错误的状态码为 412，则不再重试，因为这是特定的 OpenML 错误。
    `url` 参数用于向用户提供关于错误的更多信息。
    """
    # 定义一个装饰器函数 `decorator`，接受一个函数 `f` 作为参数
    def decorator(f):
        # 使用 functools 库中的 wraps 装饰器，保留被装饰函数的元数据
        @wraps(f)
        # 定义内部函数 `wrapper`，接受任意位置参数 `*args` 和关键字参数 `**kwargs`
        def wrapper(*args, **kwargs):
            # 设置重试计数器为预设的重试次数 `n_retries`
            retry_counter = n_retries
            # 循环执行直到成功或达到最大重试次数
            while True:
                try:
                    # 调用被装饰的函数 `f`，并返回其结果
                    return f(*args, **kwargs)
                except (URLError, TimeoutError) as e:
                    # 捕获 `URLError` 或 `TimeoutError` 异常，表示网络错误
                    # 412 是 OpenML 的特定错误码，不进行重试，直接抛出异常
                    if isinstance(e, HTTPError) and e.code == 412:
                        raise
                    # 如果重试次数耗尽，则抛出异常
                    if retry_counter == 0:
                        raise
                    # 发出警告，指示发生网络错误，正在重试
                    warn(
                        f"A network error occurred while downloading {url}. Retrying..."
                    )
                    # 减少重试计数器
                    retry_counter -= 1
                    # 等待一段时间后再次尝试
                    time.sleep(delay)

        # 返回内部函数 `wrapper`，作为装饰器的返回结果
        return wrapper
# 定义一个函数，用于从 OpenML.org 获取资源并根据需要缓存到 data_home

def _open_openml_url(
    openml_path: str,  # OpenML 资源的路径
    data_home: Optional[str],  # 缓存文件的目录路径，如果为 None，则不缓存
    n_retries: int = 3,  # 遇到 HTTP 错误时的重试次数，默认为 3
    delay: float = 1.0  # 每次重试之间的延迟时间，默认为 1.0 秒
):
    """
    Returns a resource from OpenML.org. Caches it to data_home if required.

    Parameters
    ----------
    openml_path : str
        OpenML URL that will be accessed. This will be prefixes with
        _OPENML_PREFIX.

    data_home : str
        Directory to which the files will be cached. If None, no caching will
        be applied.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered. Error with status
        code 412 won't be retried as they represent OpenML generic errors.

    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    result : stream
        A stream to the OpenML resource.
    """

    # 检查文件是否使用 gzip 编码
    def is_gzip_encoded(_fsrc):
        return _fsrc.info().get("Content-Encoding", "") == "gzip"

    # 创建一个 Request 对象，设置接受 gzip 编码
    req = Request(_OPENML_PREFIX + openml_path)
    req.add_header("Accept-encoding", "gzip")

    if data_home is None:
        # 如果不需要缓存，则直接从网络获取资源
        fsrc = _retry_on_network_error(n_retries, delay, req.full_url)(urlopen)(req)
        if is_gzip_encoded(fsrc):
            return gzip.GzipFile(fileobj=fsrc, mode="rb")
        return fsrc

    # 计算本地缓存文件路径
    local_path = _get_local_path(openml_path, data_home)
    dir_name, file_name = os.path.split(local_path)

    if not os.path.exists(local_path):
        # 如果本地文件不存在，则下载并缓存文件
        os.makedirs(dir_name, exist_ok=True)
        try:
            # 创建一个临时目录，下载成功后将文件移动到该目录的子文件夹中，确保缓存操作的原子性
            with TemporaryDirectory(dir=dir_name) as tmpdir:
                with closing(
                    _retry_on_network_error(n_retries, delay, req.full_url)(urlopen)(
                        req
                    )
                ) as fsrc:
                    # 根据文件是否使用 gzip 编码，选择不同的打开方式
                    opener: Callable
                    if is_gzip_encoded(fsrc):
                        opener = open
                    else:
                        opener = gzip.GzipFile
                    # 将下载的文件复制到临时目录中
                    with opener(os.path.join(tmpdir, file_name), "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                # 将临时文件移动到最终缓存位置
                shutil.move(fdst.name, local_path)
        except Exception:
            # 下载或缓存过程中出现异常，删除已下载的文件并抛出异常
            if os.path.exists(local_path):
                os.unlink(local_path)
            raise

    # 第一次访问时，不需要解压缩（使用 fsrc），但仍会执行解压缩操作
    return gzip.GzipFile(local_path, "rb")
    Loads json data from the openml api.

    Parameters
    ----------
    url : str
        The URL to load from. Should be an official OpenML endpoint.

    error_message : str or None
        The error message to raise if an acceptable OpenML error is thrown
        (acceptable error is, e.g., data id not found. Other errors, like 404's
        will throw the native error message).

    data_home : str or None
        Location to cache the response. None if no cache is required.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered. Error with status
        code 412 won't be retried as they represent OpenML generic errors.

    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    json_data : json
        the json result from the OpenML server if the call was successful.
        An exception otherwise.
    """
    # 使用装饰器 `_retry_with_clean_cache` 包装 `_load_json` 函数，以处理重试和缓存
    @_retry_with_clean_cache(url, data_home=data_home)
    def _load_json():
        # 打开 OpenML URL 并获取响应，使用 `closing` 确保响应关闭
        with closing(
            _open_openml_url(url, data_home, n_retries=n_retries, delay=delay)
        ) as response:
            # 解码 JSON 响应数据并返回
            return json.loads(response.read().decode("utf-8"))

    try:
        # 尝试调用 `_load_json` 函数并返回结果
        return _load_json()
    except HTTPError as error:
        # 如果捕获到 HTTPError 异常
        # 412 表示 OpenML 的特定错误，例如数据未找到
        if error.code != 412:
            # 如果错误码不是 412，则重新抛出原始异常
            raise error

    # 如果出现 412 错误，则通过抛出 OpenMLError 异常返回错误消息
    # 用于更友好的异常追踪
    raise OpenMLError(error_message)
# 根据数据集名称和版本从 OpenML 数据集列表 API 中获取数据集信息
def _get_data_info_by_name(
    name: str,
    version: Union[int, str],
    data_home: Optional[str],
    n_retries: int = 3,
    delay: float = 1.0,
):
    """
    Utilizes the openml dataset listing api to find a dataset by
    name/version
    OpenML api function:
    https://www.openml.org/api_docs#!/data/get_data_list_data_name_data_name

    Parameters
    ----------
    name : str
        数据集的名称

    version : int or str
        如果 version 是整数，则从 OpenML 中获取确切的名称/版本。如果 version 是字符串
        （值为 "active"），则从 OpenML 中获取第一个标记为活动的版本。除了 "active"
        外的任何其他字符串值都被视为整数。

    data_home : str or None
        响应缓存的位置。如果不需要缓存，则为 None。

    n_retries : int, default=3
        遇到 HTTP 错误时重试的次数。状态码为 412 的错误不会重试，因为它们表示 OpenML 的通用错误。

    delay : float, default=1.0
        重试之间的秒数。

    Returns
    -------
    first_dataset : json
        符合搜索条件的第一个数据集对象的 JSON 表示

    """
    if version == "active":
        # 当 version 为 "active" 时，返回最旧的活动版本情况
        url = _SEARCH_NAME.format(name) + "/status/active/"
        error_msg = "No active dataset {} found.".format(name)
        json_data = _get_json_content_from_openml_api(
            url,
            error_msg,
            data_home=data_home,
            n_retries=n_retries,
            delay=delay,
        )
        res = json_data["data"]["dataset"]
        if len(res) > 1:
            first_version = version = res[0]["version"]
            warning_msg = (
                "Multiple active versions of the dataset matching the name"
                f" {name} exist. Versions may be fundamentally different, "
                f"returning version {first_version}. "
                "Available versions:\n"
            )
            for r in res:
                warning_msg += f"- version {r['version']}, status: {r['status']}\n"
                warning_msg += (
                    f"  url: https://www.openml.org/search?type=data&id={r['did']}\n"
                )
            warn(warning_msg)
        return res[0]

    # 提供了整数版本
    url = (_SEARCH_NAME + "/data_version/{}").format(name, version)
    try:
        json_data = _get_json_content_from_openml_api(
            url,
            error_message=None,
            data_home=data_home,
            n_retries=n_retries,
            delay=delay,
        )
    # 捕获 OpenMLError 异常
    except OpenMLError:
        # 如果 OpenML 不需要指定数据集状态（例如，返回具有给定名称/版本的数据集，而不考虑活动状态、停用状态等），则可以在一个函数调用中完成此操作
        # TODO: OpenML 的功能请求
        # 在 URL 后面添加 "/status/deactivated"，用于指定数据集状态为停用状态
        url += "/status/deactivated"
        # 格式化错误消息，包含数据集名称和版本号
        error_msg = "Dataset {} with version {} not found.".format(name, version)
        # 从 OpenML API 获取 JSON 数据
        json_data = _get_json_content_from_openml_api(
            url,
            error_msg,
            data_home=data_home,
            n_retries=n_retries,
            delay=delay,
        )

    # 返回 JSON 数据中的第一个数据集
    return json_data["data"]["dataset"][0]
def _get_data_description_by_id(
    data_id: int,
    data_home: Optional[str],
    n_retries: int = 3,
    delay: float = 1.0,
) -> Dict[str, Any]:
    # 构建获取数据描述信息的 OpenML API URL
    url = _DATA_INFO.format(data_id)
    # 当数据未找到时的错误信息
    error_message = "Dataset with data_id {} not found.".format(data_id)
    # 调用 OpenML API 获取 JSON 格式的数据描述信息
    json_data = _get_json_content_from_openml_api(
        url,
        error_message,
        data_home=data_home,
        n_retries=n_retries,
        delay=delay,
    )
    # 返回数据集描述信息部分
    return json_data["data_set_description"]


def _get_data_features(
    data_id: int,
    data_home: Optional[str],
    n_retries: int = 3,
    delay: float = 1.0,
) -> OpenmlFeaturesType:
    # 构建获取数据特征信息的 OpenML API URL
    url = _DATA_FEATURES.format(data_id)
    # 当数据未找到时的错误信息
    error_message = "Dataset with data_id {} not found.".format(data_id)
    # 调用 OpenML API 获取 JSON 格式的数据特征信息
    json_data = _get_json_content_from_openml_api(
        url,
        error_message,
        data_home=data_home,
        n_retries=n_retries,
        delay=delay,
    )
    # 返回数据集特征信息部分
    return json_data["data_features"]["feature"]


def _get_data_qualities(
    data_id: int,
    data_home: Optional[str],
    n_retries: int = 3,
    delay: float = 1.0,
) -> OpenmlQualitiesType:
    # 构建获取数据质量信息的 OpenML API URL
    url = _DATA_QUALITIES.format(data_id)
    # 当数据未找到时的错误信息
    error_message = "Dataset with data_id {} not found.".format(data_id)
    # 调用 OpenML API 获取 JSON 格式的数据质量信息
    json_data = _get_json_content_from_openml_api(
        url,
        error_message,
        data_home=data_home,
        n_retries=n_retries,
        delay=delay,
    )
    # 尝试获取数据质量信息，若无法获取则返回空列表
    return json_data.get("data_qualities", {}).get("quality", [])


def _get_num_samples(data_qualities: OpenmlQualitiesType) -> int:
    """从数据质量信息中获取样本数。

    Parameters
    ----------
    data_qualities : list of dict
        用于获取数据集中实例（样本）数量的数据质量信息。

    Returns
    -------
    n_samples : int
        数据集中的样本数，若数据质量信息不可用则返回 -1。
    """
    # 若数据质量信息不可用，则返回 -1
    default_n_samples = -1

    # 提取数据质量信息中的每个指标名称和对应数值，以字典形式存储
    qualities = {d["name"]: d["value"] for d in data_qualities}
    # 将样本数转换为整数，并返回
    return int(float(qualities.get("NumberOfInstances", default_n_samples)))


def _load_arff_response(
    url: str,
    data_home: Optional[str],
    parser: str,
    output_type: str,
    openml_columns_info: dict,
    feature_names_to_select: List[str],
    target_names_to_select: List[str],
    shape: Optional[Tuple[int, int]],
    md5_checksum: str,
    n_retries: int = 3,
    delay: float = 1.0,
    read_csv_kwargs: Optional[Dict] = None,
):
    """加载与 OpenML URL 关联的 ARFF 数据。

    除了加载数据外，此函数还将检查
    # 使用指定的 URL 和参数从 OpenML 下载数据文件，返回一个文件对象
    gzip_file = _open_openml_url(url, data_home, n_retries=n_retries, delay=delay)
    
    # 使用 `closing` 上下文确保文件对象在使用后被正确关闭
    with closing(gzip_file):
        # 创建一个 MD5 对象用于计算文件的 MD5 校验和
        md5 = hashlib.md5()
        
        # 以每次读取 4096 字节的方式迭代文件内容，并更新 MD5 校验和
        for chunk in iter(lambda: gzip_file.read(4096), b""):
            md5.update(chunk)
        
        # 计算实际文件的 MD5 校验和的十六进制表示
        actual_md5_checksum = md5.hexdigest()
    
    # 检查计算出的 MD5 校验和是否与预期的 MD5 校验和相符
    if actual_md5_checksum != md5_checksum:
        # 如果校验和不符合预期，则抛出值错误，提示文件可能已被修改或损坏
        raise ValueError(
            f"md5 checksum of local file for {url} does not match description: "
            f"expected: {md5_checksum} but got {actual_md5_checksum}. "
            "Downloaded file could have been modified / corrupted, clean cache "
            "and retry..."
        )
    def _open_url_and_load_gzip_file(url, data_home, n_retries, delay, arff_params):
        # 使用提供的 URL 打开 OpenML 数据集的 gzip 文件，并返回打开的文件对象
        gzip_file = _open_openml_url(url, data_home, n_retries=n_retries, delay=delay)
        # 使用 closing 上下文管理器确保在函数结束时关闭 gzip 文件
        with closing(gzip_file):
            # 调用函数加载 gzip 文件中的 ARFF 数据，并传递额外的参数 arff_params
            return load_arff_from_gzip_file(gzip_file, **arff_params)

    # 设置 ARFF 数据加载函数的参数字典
    arff_params: Dict = dict(
        parser=parser,
        output_type=output_type,
        openml_columns_info=openml_columns_info,
        feature_names_to_select=feature_names_to_select,
        target_names_to_select=target_names_to_select,
        shape=shape,
        read_csv_kwargs=read_csv_kwargs or {},
    )
    try:
        # 尝试从 URL 加载 gzip 文件中的数据集，获取数据集的 X, y, frame 和 categories
        X, y, frame, categories = _open_url_and_load_gzip_file(
            url, data_home, n_retries, delay, arff_params
        )
    except Exception as exc:
        # 如果出现异常
        if parser != "pandas":
            # 如果加载函数不是针对 pandas 进行的，则直接抛出异常
            raise

        from pandas.errors import ParserError

        # 如果异常不是 ParserError 类型的，再次抛出异常
        if not isinstance(exc, ParserError):
            raise

        # 如果出现 ParserError，尝试修改 read_csv_kwargs 的参数并重新加载数据集
        # 这里假设错误可能是由于错误的引号字符引起的，默认使用双引号，现在尝试使用单引号再次加载
        arff_params["read_csv_kwargs"].update(quotechar="'")
        # 重新加载数据集的 X, y, frame 和 categories
        X, y, frame, categories = _open_url_and_load_gzip_file(
            url, data_home, n_retries, delay, arff_params
        )

    # 返回加载后的数据集的 X, y, frame 和 categories
    return X, y, frame, categories
# 定义函数 `_download_data_to_bunch`，用于从 OpenML 下载 ARFF 数据并加载到 Bunch 结构中
def _download_data_to_bunch(
    url: str,
    sparse: bool,
    data_home: Optional[str],
    *,
    as_frame: bool,
    openml_columns_info: List[dict],
    data_columns: List[str],
    target_columns: List[str],
    shape: Optional[Tuple[int, int]],
    md5_checksum: str,
    n_retries: int = 3,
    delay: float = 1.0,
    parser: str,
    read_csv_kwargs: Optional[Dict] = None,
):
    """Download ARFF data, load it to a specific container and create to Bunch.

    This function has a mechanism to retry/cache/clean the data.

    Parameters
    ----------
    url : str
        The URL of the ARFF file on OpenML.

    sparse : bool
        Whether the dataset is expected to use the sparse ARFF format.

    data_home : str
        The location where to cache the data.

    as_frame : bool
        Whether or not to return the data into a pandas DataFrame.

    openml_columns_info : list of dict
        The information regarding the columns provided by OpenML for the
        ARFF dataset. The information is stored as a list of dictionaries.

    data_columns : list of str
        The list of the features to be selected.

    target_columns : list of str
        The list of the target variables to be selected.

    shape : tuple or None
        With `parser="liac-arff"`, when using a generator to load the data,
        one needs to provide the shape of the data beforehand.

    md5_checksum : str
        The MD5 checksum provided by OpenML to check the data integrity.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered. Error with status
        code 412 won't be retried as they represent OpenML generic errors.

    delay : float, default=1.0
        Number of seconds between retries.

    parser : {"liac-arff", "pandas"}
        The parser used to parse the ARFF file.

    read_csv_kwargs : dict, default=None
        Keyword arguments to pass to `pandas.read_csv` when using the pandas parser.
        It allows to overwrite the default options.

        .. versionadded:: 1.3

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        X : {ndarray, sparse matrix, dataframe}
            The data matrix.
        y : {ndarray, dataframe, series}
            The target.
        frame : dataframe or None
            A dataframe containing both `X` and `y`. `None` if
            `output_array_type != "pandas"`.
        categories : list of str or None
            The names of the features that are categorical. `None` if
            `output_array_type == "pandas"`.
    """
    
    # 准备用于 X 和 y 的列名和数据类型信息的字典
    features_dict = {feature["name"]: feature for feature in openml_columns_info}

    # 根据参数设定输出类型：稀疏矩阵、pandas DataFrame 或 numpy 数组
    if sparse:
        output_type = "sparse"
    elif as_frame:
        output_type = "pandas"
    else:
        output_type = "numpy"

    # XXX: 目标列应全部为分类或全部为数值型
    # 使用特定函数验证特征字典和目标列的数据类型是否符合要求
    _verify_target_data_type(features_dict, target_columns)
    
    # 遍历目标列，检查每列的缺失值情况
    for name in target_columns:
        # 获取特征字典中目标列的信息
        column_info = features_dict[name]
        # 获取该列的缺失值数量并转换为整数
        n_missing_values = int(column_info["number_of_missing_values"])
        # 如果存在缺失值，则抛出 ValueError 异常
        if n_missing_values > 0:
            raise ValueError(
                f"Target column '{column_info['name']}' has {n_missing_values} missing "
                "values. Missing values are not supported for target columns."
            )
    
    # 初始化无需重试的异常变量为 None
    no_retry_exception = None
    if parser == "pandas":
        # 如果使用 pandas 解析器，导入 ParserError
        from pandas.errors import ParserError
        # 将 no_retry_exception 设置为 ParserError，表示遇到此异常时不重试
        no_retry_exception = ParserError
    
    # 调用 _retry_with_clean_cache 函数，尝试加载 ARFF 数据
    X, y, frame, categories = _retry_with_clean_cache(
        url, data_home, no_retry_exception
    )(_load_arff_response)(
        # 传递参数加载 ARFF 数据
        url,
        data_home,
        parser=parser,
        output_type=output_type,
        openml_columns_info=features_dict,
        feature_names_to_select=data_columns,
        target_names_to_select=target_columns,
        shape=shape,
        md5_checksum=md5_checksum,
        n_retries=n_retries,
        delay=delay,
        read_csv_kwargs=read_csv_kwargs,
    )

    # 返回一个 Bunch 对象，包含加载的数据和元信息
    return Bunch(
        data=X,
        target=y,
        frame=frame,
        categories=categories,
        feature_names=data_columns,
        target_names=target_columns,
    )
# 验证目标数据类型是否符合预期，用于多目标情况
def _verify_target_data_type(features_dict, target_columns):
    # 检查目标列是否为列表类型，否则抛出数值错误
    if not isinstance(target_columns, list):
        raise ValueError("target_column should be list, got: %s" % type(target_columns))
    
    # 用于存储发现的数据类型集合
    found_types = set()
    
    # 遍历目标列
    for target_column in target_columns:
        # 如果特征字典中没有目标列，则抛出键错误
        if target_column not in features_dict:
            raise KeyError(f"Could not find target_column='{target_column}'")
        
        # 根据数据类型字段判断目标列的数据类型是数值型还是其他类型，分别加入集合
        if features_dict[target_column]["data_type"] == "numeric":
            found_types.add(np.float64)
        else:
            found_types.add(object)

        # 注意：这里是与字符串进行比较，而不是布尔值
        # 如果目标列的 is_ignore 标志为 "true"，则发出警告
        if features_dict[target_column]["is_ignore"] == "true":
            warn(f"target_column='{target_column}' has flag is_ignore.")
        
        # 如果目标列的 is_row_identifier 标志为 "true"，则发出警告
        if features_dict[target_column]["is_row_identifier"] == "true":
            warn(f"target_column='{target_column}' has flag is_row_identifier.")
    
    # 如果发现的数据类型集合超过一个元素，则抛出数值错误
    if len(found_types) > 1:
        raise ValueError(
            "Can only handle homogeneous multi-target datasets, "
            "i.e., all targets are either numeric or "
            "categorical."
        )


# 验证有效的数据列名
def _valid_data_column_names(features_list, target_columns):
    # 用于存储有效的数据列名列表
    valid_data_column_names = []
    
    # 遍历特征列表中的每个特征
    for feature in features_list:
        # 如果特征名不在目标列中，并且既不是忽略标志也不是行标识符标志，则加入有效数据列名列表
        if (
            feature["name"] not in target_columns
            and feature["is_ignore"] != "true"
            and feature["is_row_identifier"] != "true"
        ):
            valid_data_column_names.append(feature["name"])
    
    # 返回有效的数据列名列表
    return valid_data_column_names
    # 设置一个浮点型的延迟时间，默认为1.0秒
    delay: float = 1.0,
    # 指定解析器的类型，默认为自动选择
    parser: str = "auto",
    # 用于读取CSV文件的额外参数，可以是一个字典类型或者None
    read_csv_kwargs: Optional[Dict] = None,
):
    """Fetch dataset from openml by name or dataset id.

    Datasets are uniquely identified by either an integer ID or by a
    combination of name and version (i.e. there might be multiple
    versions of the 'iris' dataset). Please give either name or data_id
    (not both). In case a name is given, a version can also be
    provided.

    Read more in the :ref:`User Guide <openml>`.

    .. versionadded:: 0.20

    .. note:: EXPERIMENTAL

        The API is experimental (particularly the return value structure),
        and might have small backward-incompatible changes without notice
        or warning in future releases.

    Parameters
    ----------
    name : str, default=None
        String identifier of the dataset. Note that OpenML can have multiple
        datasets with the same name.

    version : int or 'active', default='active'
        Version of the dataset. Can only be provided if also ``name`` is given.
        If 'active' the oldest version that's still active is used. Since
        there may be more than one active version of a dataset, and those
        versions may fundamentally be different from one another, setting an
        exact version is highly recommended.

    data_id : int, default=None
        OpenML ID of the dataset. The most specific way of retrieving a
        dataset. If data_id is not given, name (and potential version) are
        used to obtain a dataset.

    data_home : str or path-like, default=None
        Specify another download and cache folder for the data sets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    target_column : str, list or None, default='default-target'
        Specify the column name in the data to use as target. If
        'default-target', the standard target column a stored on the server
        is used. If ``None``, all columns are returned as data and the
        target is ``None``. If list (of strings), all columns with these names
        are returned as multi-target (Note: not all scikit-learn classifiers
        can handle all types of multi-output combinations).

    cache : bool, default=True
        Whether to cache the downloaded datasets into `data_home`.

    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` objects.
    """
    # 在OpenML上根据数据集名称或数据集ID获取数据集

    # 如果提供了data_id，则使用最具体的方式检索数据集
    # 如果未提供data_id，则使用名称（和可能的版本）来获取数据集

    # 根据参数设置是否缓存下载的数据集到指定的data_home文件夹中
    # 默认情况下，所有scikit-learn数据集存储在'~/scikit_learn_data'子文件夹中

    # 如果设置了return_X_y为True，则返回(data, target)而不是一个Bunch对象
    pass
    # as_frame 参数控制返回的数据类型，如果为 True，则返回的数据将是一个 pandas DataFrame，
    # 包含适当的数据类型列（数值型、字符串或分类型）。目标数据是一个 pandas DataFrame 或 Series，
    # 取决于目标列的数量。返回的 Bunch 对象会包含一个名为 "frame" 的属性，其中包含数据和目标。
    # 如果 return_X_y 参数为 True，则返回的数据将是 pandas 的 DataFrame 或 Series，具体取决于上述描述。

    # 如果 as_frame 设置为 'auto'，数据和目标将被转换为 DataFrame 或 Series，就好像 as_frame 设置为 True 一样，
    # 除非数据集以稀疏格式存储。

    # 如果 as_frame 设置为 False，数据和目标将是 NumPy 数组，且当 parser="liac-arff" 时，
    # 数据仅包含数值。当 parser="pandas" 时，不进行序数编码。

    # 从版本 0.24 开始，默认的 as_frame 值由 False 更改为 'auto'。

    # n_retries 参数指定在遇到 HTTP 错误或网络超时时的重试次数。
    # 状态码为 412 的错误不会重试，因为它们代表 OpenML 的通用错误。

    # delay 参数指定重试之间的秒数间隔。

    # parser 参数指定用于加载 ARFF 文件的解析器。当前实现了两种解析器：
    # - "pandas"：这是最高效的解析器，但需要安装 pandas，并且只能打开密集的数据集。
    # - "liac-arff"：这是一个纯 Python 的 ARFF 解析器，内存和 CPU 效率远不如前者，但能处理稀疏的 ARFF 数据集。

    # 如果 parser 设置为 "auto"，则根据数据集的稀疏性自动选择解析器："liac-arff" 用于稀疏 ARFF 数据集，否则选择 "pandas"。

    # 从版本 1.2 开始添加了 parser 参数。
    # 从版本 1.4 开始，parser 的默认值从 "liac-arff" 更改为 "auto"。

    # read_csv_kwargs 参数是传递给 pandas.read_csv 函数的关键字参数，用于在加载 ARFF 文件并使用 pandas 解析器时覆盖一些默认参数。

    # 从版本 1.3 开始添加了 read_csv_kwargs 参数。

    # 函数没有具体的返回说明，返回值将根据具体实现来决定。
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : np.array, scipy.sparse.csr_matrix of floats, or pandas DataFrame
            The feature matrix. Categorical features are encoded as ordinals.
        target : np.array, pandas Series or DataFrame
            The regression target or classification labels, if applicable.
            Dtype is float if numeric, and object if categorical. If
            ``as_frame`` is True, ``target`` is a pandas object.
        DESCR : str
            The full description of the dataset.
        feature_names : list
            The names of the dataset columns.
        target_names: list
            The names of the target columns.

        .. versionadded:: 0.22

        categories : dict or None
            Maps each categorical feature name to a list of values, such
            that the value encoded as i is ith in the list. If ``as_frame``
            is True, this is None.
        details : dict
            More metadata from OpenML.
        frame : pandas DataFrame
            Only present when `as_frame=True`. DataFrame with ``data`` and
            ``target``.

    (data, target) : tuple if ``return_X_y`` is True

        .. note:: EXPERIMENTAL

            This interface is **experimental** and subsequent releases may
            change attributes without notice (although there should only be
            minor changes to ``data`` and ``target``).

        Missing values in the 'data' are represented as NaN's. Missing values
        in 'target' are represented as NaN's (numerical target) or None
        (categorical target).

    Notes
    -----
    The `"pandas"` and `"liac-arff"` parsers can lead to different data types
    in the output. The notable differences are the following:

    - The `"liac-arff"` parser always encodes categorical features as `str` objects.
      To the contrary, the `"pandas"` parser instead infers the type while
      reading and numerical categories will be casted into integers whenever
      possible.
    - The `"liac-arff"` parser uses float64 to encode numerical features
      tagged as 'REAL' and 'NUMERICAL' in the metadata. The `"pandas"`
      parser instead infers if these numerical features corresponds
      to integers and uses panda's Integer extension dtype.
    - In particular, classification datasets with integer categories are
      typically loaded as such `(0, 1, ...)` with the `"pandas"` parser while
      `"liac-arff"` will force the use of string encoded class labels such as
      `"0"`, `"1"` and so on.
    - The `"pandas"` parser will not strip single quotes - i.e. `'` - from
      string columns. For instance, a string `'my string'` will be kept as is
      while the `"liac-arff"` parser will strip the single quotes. For
      categorical columns, the single quotes are stripped from the values.

    In addition, when `as_frame=False` is used, the `"liac-arff"` parser
    # 如果不缓存数据，设定数据主目录为 None
    if cache is False:
        data_home = None
    else:
        # 否则，获取数据主目录路径
        data_home = get_data_home(data_home=data_home)
        # 将数据主目录路径拼接为 "openml" 子目录
        data_home = join(str(data_home), "openml")

    # 检查函数参数的有效性。必须提供 data_id 或者 (name, version) 之一
    if name is not None:
        # OpenML 对大小写不敏感，但缓存机制对大小写敏感
        # 将数据集名称转换为小写
        name = name.lower()
        if data_id is not None:
            # 如果同时提供了 data_id 和 name，则引发 ValueError
            raise ValueError(
                "Dataset data_id={} and name={} passed, but you can only "
                "specify a numeric data_id or a name, not "
                "both.".format(data_id, name)
            )
        # 根据数据集名称和版本获取数据信息
        data_info = _get_data_info_by_name(
            name, version, data_home, n_retries=n_retries, delay=delay
        )
        # 获取数据集的唯一标识符
        data_id = data_info["did"]
    elif data_id is not None:
        # 根据前一个条件，已确定 name 为 None
        if version != "active":
            # 如果同时提供了 data_id 和 version，则引发 ValueError
            raise ValueError(
                "Dataset data_id={} and version={} passed, but you can only "
                "specify a numeric data_id or a version, not "
                "both.".format(data_id, version)
            )
    else:
        # 如果既未提供 name，也未提供 data_id，则引发 ValueError
        raise ValueError(
            "Neither name nor data_id are provided. Please provide name or data_id."
        )

    # 根据数据集的唯一标识符获取数据集的详细描述信息
    data_description = _get_data_description_by_id(data_id, data_home)
    # 检查数据集状态是否为活跃，如果不是则发出警告
    if data_description["status"] != "active":
        warn(
            "Version {} of dataset {} is inactive, meaning that issues have "
            "been found in the dataset. Try using a newer version from "
            "this URL: {}".format(
                data_description["version"],
                data_description["name"],
                data_description["url"],
            )
        )
    
    # 如果数据集描述中包含错误信息，则发出警告
    if "error" in data_description:
        warn(
            "OpenML registered a problem with the dataset. It might be "
            "unusable. Error: {}".format(data_description["error"])
        )
    
    # 如果数据集描述中包含警告信息，则发出警告
    if "warning" in data_description:
        warn(
            "OpenML raised a warning on the dataset. It might be "
            "unusable. Warning: {}".format(data_description["warning"])
        )

    # 确定数据格式是否为稀疏 ARFF，并设置返回值类型
    return_sparse = data_description["format"].lower() == "sparse_arff"
    
    # 根据参数设置是否返回 DataFrame
    as_frame = not return_sparse if as_frame == "auto" else as_frame
    
    # 确定数据解析器的类型
    if parser == "auto":
        parser_ = "liac-arff" if return_sparse else "pandas"
    else:
        parser_ = parser
    
    # 如果解析器为 pandas，则检查 pandas 支持情况
    if parser_ == "pandas":
        try:
            check_pandas_support("`fetch_openml`")
        except ImportError as exc:
            # 抛出导入错误，提示安装 pandas 或更改参数设置
            if as_frame:
                err_msg = (
                    "Returning pandas objects requires pandas to be installed. "
                    "Alternatively, explicitly set `as_frame=False` and "
                    "`parser='liac-arff'`."
                )
            else:
                err_msg = (
                    f"Using `parser={parser!r}` wit dense data requires pandas to be "
                    "installed. Alternatively, explicitly set `parser='liac-arff'`."
                )
            raise ImportError(err_msg) from exc
    
    # 如果返回值为稀疏 ARFF，则进行额外的参数检查和错误抛出
    if return_sparse:
        if as_frame:
            raise ValueError(
                "Sparse ARFF datasets cannot be loaded with as_frame=True. "
                "Use as_frame=False or as_frame='auto' instead."
            )
        if parser_ == "pandas":
            raise ValueError(
                f"Sparse ARFF datasets cannot be loaded with parser={parser!r}. "
                "Use parser='liac-arff' or parser='auto' instead."
            )

    # 下载数据特征列表和关于列类型的元信息
    features_list = _get_data_features(data_id, data_home)

    # 如果不返回 DataFrame，则检查特征列表中的每个特征
    if not as_frame:
        for feature in features_list:
            # 跳过被标记为忽略或行标识符的特征
            if "true" in (feature["is_ignore"], feature["is_row_identifier"]):
                continue
            # 如果特征数据类型为字符串，则抛出错误
            if feature["data_type"] == "string":
                raise ValueError(
                    "STRING attributes are not supported for "
                    "array representation. Try as_frame=True"
                )
    if target_column == "default-target":
        # 如果目标列是"default-target"，根据数据特征结果确定默认的目标列
        # （目前这比数据描述更可靠；参见问题：https://github.com/openml/OpenML/issues/768）
        target_columns = [
            feature["name"]
            for feature in features_list
            if feature["is_target"] == "true"
        ]
    elif isinstance(target_column, str):
        # 如果目标列是字符串类型，为了简化代码，默认将目标列转换为列表形式
        target_columns = [target_column]
    elif target_column is None:
        # 如果目标列为None，则设置目标列为空列表
        target_columns = []
    else:
        # 如果目标列已经是列表类型，则直接使用
        target_columns = target_column
    # 获取有效的数据列名
    data_columns = _valid_data_column_names(features_list, target_columns)

    shape: Optional[Tuple[int, int]]
    # 确定返回的ARFF编码方式
    if not return_sparse:
        # 如果不返回稀疏数据，需要包括被忽略的特征，以保持正确的索引
        data_qualities = _get_data_qualities(data_id, data_home)
        shape = _get_num_samples(data_qualities), len(features_list)
    else:
        # 如果返回稀疏数据，则形状为None
        shape = None

    # 获取数据
    url = _DATA_FILE.format(data_description["file_id"])
    bunch = _download_data_to_bunch(
        url,
        return_sparse,
        data_home,
        as_frame=bool(as_frame),
        openml_columns_info=features_list,
        shape=shape,
        target_columns=target_columns,
        data_columns=data_columns,
        md5_checksum=data_description["md5_checksum"],
        n_retries=n_retries,
        delay=delay,
        parser=parser_,
        read_csv_kwargs=read_csv_kwargs,
    )

    if return_X_y:
        # 如果需要返回X和y，则返回数据和目标列
        return bunch.data, bunch.target

    # 更新数据集对象的描述和详情
    description = "{}\n\nDownloaded from openml.org.".format(
        data_description.pop("description")
    )
    bunch.update(
        DESCR=description,
        details=data_description,
        url="https://www.openml.org/d/{}".format(data_id),
    )

    # 返回更新后的数据集对象
    return bunch
```