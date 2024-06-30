# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\test_openml.py`

```
# 导入必要的库
import gzip  # 用于处理 gzip 压缩文件
import json  # 用于处理 JSON 数据
import os  # 提供与操作系统相关的功能
import re  # 提供正则表达式操作
from functools import partial  # 导入 functools 库中的 partial 函数，用于创建 partial 函数应用
from importlib import resources  # 导入 importlib 库中的 resources 模块，用于访问包资源
from io import BytesIO  # 导入 io 库中的 BytesIO 类，用于在内存中读写二进制数据
from urllib.error import HTTPError  # 导入 urllib 库中的 HTTPError 类，用于处理 HTTP 错误

import numpy as np  # 导入 numpy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试
import scipy.sparse  # 导入 scipy 库中的 sparse 模块，用于稀疏矩阵处理

import sklearn  # 导入 sklearn 库，用于机器学习
from sklearn import config_context  # 导入 sklearn 库中的 config_context 模块
from sklearn.datasets import fetch_openml as fetch_openml_orig  # 从 sklearn.datasets 中导入 fetch_openml 函数
from sklearn.datasets._openml import (
    _OPENML_PREFIX,  # 导入 _OPENML_PREFIX 常量，表示 openml 数据集前缀
    _get_local_path,  # 导入 _get_local_path 函数，用于获取本地路径
    _open_openml_url,  # 导入 _open_openml_url 函数，用于打开 openml URL
    _retry_with_clean_cache,  # 导入 _retry_with_clean_cache 函数，用于清理缓存并重试
)
from sklearn.utils import Bunch  # 导入 sklearn.utils 中的 Bunch 类，用于封装数据集
from sklearn.utils._optional_dependencies import check_pandas_support  # 导入 check_pandas_support 函数，用于检查 pandas 支持
from sklearn.utils._testing import (
    SkipTest,  # 导入 SkipTest 异常类，用于跳过测试
    assert_allclose,  # 导入 assert_allclose 函数，用于检查数组近似相等
    assert_array_equal,  # 导入 assert_array_equal 函数，用于检查数组完全相等
)

OPENML_TEST_DATA_MODULE = "sklearn.datasets.tests.data.openml"  # 定义 openml 测试数据模块

# 如果为 True，则将 urlopen 功能修改为仅使用本地文件
test_offline = True


class _MockHTTPResponse:
    def __init__(self, data, is_gzip):
        self.data = data
        self.is_gzip = is_gzip

    def read(self, amt=-1):
        return self.data.read(amt)

    def close(self):
        self.data.close()

    def info(self):
        if self.is_gzip:
            return {"Content-Encoding": "gzip"}  # 如果是 gzip 压缩格式，则返回内容编码为 gzip

        return {}  # 否则返回空字典

    def __iter__(self):
        return iter(self.data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# 在测试 fetch_openml 函数时禁用基于磁盘的缓存：
# sklearn/datasets/tests/data/openml/ 中的模拟数据可能与 openml.org 上的版本不一致。
# 如果在测试之外加载数据集，可能导致数据不一致的问题。
fetch_openml = partial(fetch_openml_orig, data_home=None)


def _monkey_patch_webbased_functions(context, data_id, gzip_response):
    # monkey patch 替换 urlopen 函数。重要提示：不要与常规缓存目录一起使用，
    # 因为缓存的文件不应该与真实的 openml 数据集混合在一起。
    url_prefix_data_description = "https://api.openml.org/api/v1/json/data/"
    url_prefix_data_features = "https://api.openml.org/api/v1/json/data/features/"
    url_prefix_download_data = "https://api.openml.org/data/v1/"
    url_prefix_data_list = "https://api.openml.org/api/v1/json/data/list/"

    path_suffix = ".gz"  # 定义文件路径后缀为 .gz
    read_fn = gzip.open  # 设置读取函数为 gzip.open

    data_module = OPENML_TEST_DATA_MODULE + "." + f"id_{data_id}"
    def _file_name(url, suffix):
        # 根据给定的 URL 和后缀生成一个文件名，将非单词字符替换为连字符
        output = (
            re.sub(r"\W", "-", url[len("https://api.openml.org/") :])
            + suffix
            + path_suffix
        )
        # 缩短文件名以更好地兼容 Windows 10 和文件名超过 260 字符的情况
        return (
            output.replace("-json-data-list", "-jdl")
            .replace("-json-data-features", "-jdf")
            .replace("-json-data-qualities", "-jdq")
            .replace("-json-data", "-jd")
            .replace("-data_name", "-dn")
            .replace("-download", "-dl")
            .replace("-limit", "-l")
            .replace("-data_version", "-dv")
            .replace("-status", "-s")
            .replace("-deactivated", "-dact")
            .replace("-active", "-act")
        )

    def _mock_urlopen_shared(url, has_gzip_header, expected_prefix, suffix):
        # 确保 URL 以指定的前缀开头
        assert url.startswith(expected_prefix)

        # 根据 URL 和后缀生成数据文件名
        data_file_name = _file_name(url, suffix)
        # 拼接数据文件路径
        data_file_path = resources.files(data_module) / data_file_name

        # 使用二进制模式打开数据文件
        with data_file_path.open("rb") as f:
            if has_gzip_header and gzip_response:
                # 如果有 gzip 头且指定需要 gzip 响应，则读取数据文件并返回压缩后的模拟 HTTP 响应
                fp = BytesIO(f.read())
                return _MockHTTPResponse(fp, True)
            else:
                # 否则，按照指定的读取函数读取数据文件，再将数据存入 BytesIO 对象并返回模拟 HTTP 响应
                decompressed_f = read_fn(f, "rb")
                fp = BytesIO(decompressed_f.read())
                return _MockHTTPResponse(fp, False)

    def _mock_urlopen_data_description(url, has_gzip_header):
        # 调用共享的模拟 HTTP 请求方法处理数据描述 URL
        return _mock_urlopen_shared(
            url=url,
            has_gzip_header=has_gzip_header,
            expected_prefix=url_prefix_data_description,
            suffix=".json",
        )

    def _mock_urlopen_data_features(url, has_gzip_header):
        # 调用共享的模拟 HTTP 请求方法处理数据特征 URL
        return _mock_urlopen_shared(
            url=url,
            has_gzip_header=has_gzip_header,
            expected_prefix=url_prefix_data_features,
            suffix=".json",
        )

    def _mock_urlopen_download_data(url, has_gzip_header):
        # 调用共享的模拟 HTTP 请求方法处理数据下载 URL
        return _mock_urlopen_shared(
            url=url,
            has_gzip_header=has_gzip_header,
            expected_prefix=url_prefix_download_data,
            suffix=".arff",
        )
    # 定义一个函数用于模拟通过 URL 访问数据列表资源时的行为，接受两个参数：URL 和是否包含 gzip 头信息
    def _mock_urlopen_data_list(url, has_gzip_header):
        # 断言 URL 是否以指定的数据列表 URL 前缀开头，用于验证输入的 URL 参数
        assert url.startswith(url_prefix_data_list)

        # 生成数据文件名，通过指定的函数获取文件名后缀 ".json"
        data_file_name = _file_name(url, ".json")
        # 生成数据文件的完整路径，结合指定的数据模块路径
        data_file_path = resources.files(data_module) / data_file_name

        # 打开数据文件以二进制只读模式，用于模拟 HTTP 错误的加载行为
        with data_file_path.open("rb") as f:
            # 调用 read_fn 函数处理文件对象，以二进制只读模式读取解压后的文件对象
            decompressed_f = read_fn(f, "rb")
            # 读取解码后的字符串数据，假设使用 UTF-8 编码
            decoded_s = decompressed_f.read().decode("utf-8")
            # 将解码后的 JSON 字符串转换为 Python 对象
            json_data = json.loads(decoded_s)

        # 如果 JSON 数据中包含 "error" 键，模拟抛出 HTTPError 异常
        if "error" in json_data:
            raise HTTPError(
                url=None, code=412, msg="Simulated mock error", hdrs=None, fp=BytesIO()
            )

        # 再次打开数据文件以二进制只读模式
        with data_file_path.open("rb") as f:
            # 如果请求包含 gzip 头信息
            if has_gzip_header:
                # 将文件内容读取到 BytesIO 对象中并返回模拟的 HTTP 响应对象
                fp = BytesIO(f.read())
                return _MockHTTPResponse(fp, True)
            else:
                # 否则，调用 read_fn 处理文件对象，以二进制只读模式读取解压后的文件对象
                decompressed_f = read_fn(f, "rb")
                # 将解压后的文件对象内容读取到 BytesIO 对象中并返回模拟的 HTTP 响应对象
                fp = BytesIO(decompressed_f.read())
                return _MockHTTPResponse(fp, False)

    # 定义一个函数用于模拟通过 urllib.urlopen 访问 URL 的行为，接受 request 对象和额外的参数
    def _mock_urlopen(request, *args, **kwargs):
        # 获取请求的完整 URL
        url = request.get_full_url()
        # 判断请求是否包含 gzip 头信息
        has_gzip_header = request.get_header("Accept-encoding") == "gzip"

        # 根据 URL 的前缀判断应该调用哪个模拟函数来处理该 URL 请求
        if url.startswith(url_prefix_data_list):
            return _mock_urlopen_data_list(url, has_gzip_header)
        elif url.startswith(url_prefix_data_features):
            return _mock_urlopen_data_features(url, has_gzip_header)
        elif url.startswith(url_prefix_download_data):
            return _mock_urlopen_download_data(url, has_gzip_header)
        elif url.startswith(url_prefix_data_description):
            return _mock_urlopen_data_description(url, has_gzip_header)
        else:
            # 如果 URL 不匹配任何已知的模拟 URL 前缀，则抛出 ValueError 异常
            raise ValueError("Unknown mocking URL pattern: %s" % url)

    # 标记：全局变量
    # 如果测试离线模式被启用，则设置 sklearn.datasets._openml.urlopen 函数为 _mock_urlopen 函数
    if test_offline:
        context.setattr(sklearn.datasets._openml, "urlopen", _mock_urlopen)
###############################################################################
# Test the behaviour of `fetch_openml` depending of the input parameters.

# 使用 pytest 的标记参数化测试数据，定义多组测试参数：
@pytest.mark.parametrize(
    "data_id, dataset_params, n_samples, n_features, n_targets",
    [
        # iris 数据集
        (61, {"data_id": 61}, 150, 4, 1),
        (61, {"name": "iris", "version": 1}, 150, 4, 1),
        # anneal 数据集
        (2, {"data_id": 2}, 11, 38, 1),
        (2, {"name": "anneal", "version": 1}, 11, 38, 1),
        # cpu 数据集
        (561, {"data_id": 561}, 209, 7, 1),
        (561, {"name": "cpu", "version": 1}, 209, 7, 1),
        # emotions 数据集
        (40589, {"data_id": 40589}, 13, 72, 6),
        # adult-census 数据集
        (1119, {"data_id": 1119}, 10, 14, 1),
        (1119, {"name": "adult-census"}, 10, 14, 1),
        # miceprotein 数据集
        (40966, {"data_id": 40966}, 7, 77, 1),
        (40966, {"name": "MiceProtein"}, 7, 77, 1),
        # titanic 数据集
        (40945, {"data_id": 40945}, 1309, 13, 1),
    ],
)

# 继续使用 pytest 的参数化，定义解析器参数：
@pytest.mark.parametrize("parser", ["liac-arff", "pandas"])

# 继续使用 pytest 的参数化，定义 gzip_response 参数：
@pytest.mark.parametrize("gzip_response", [True, False])
def test_fetch_openml_as_frame_true(
    monkeypatch,
    data_id,
    dataset_params,
    n_samples,
    n_features,
    n_targets,
    parser,
    gzip_response,
):
    """Check the behaviour of `fetch_openml` with `as_frame=True`.

    Fetch by ID and/or name (depending if the file was previously cached).
    """
    pd = pytest.importorskip("pandas")

    # 使用 monkeypatch 修改 webbased 函数的行为，模拟对应的数据 ID 和 gzip_response 参数：
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=gzip_response)
    
    # 调用 fetch_openml 函数，获取数据集 bunch：
    bunch = fetch_openml(
        as_frame=True,
        cache=False,
        parser=parser,
        **dataset_params,
    )

    # 断言数据集的 ID 符合预期的 data_id：
    assert int(bunch.details["id"]) == data_id
    # 断言 bunch 是 Bunch 类型的实例：
    assert isinstance(bunch, Bunch)

    # 断言 bunch.frame 是 pandas 的 DataFrame 类型，并且形状符合预期：
    assert isinstance(bunch.frame, pd.DataFrame)
    assert bunch.frame.shape == (n_samples, n_features + n_targets)

    # 断言 bunch.data 是 pandas 的 DataFrame 类型，并且形状符合预期：
    assert isinstance(bunch.data, pd.DataFrame)
    assert bunch.data.shape == (n_samples, n_features)

    # 根据 n_targets 的数量不同，断言 bunch.target 是 pandas 的 Series 或 DataFrame，并且形状符合预期：
    if n_targets == 1:
        assert isinstance(bunch.target, pd.Series)
        assert bunch.target.shape == (n_samples,)
    else:
        assert isinstance(bunch.target, pd.DataFrame)
        assert bunch.target.shape == (n_samples, n_targets)

    # 断言 bunch.categories 为 None：
    assert bunch.categories is None
    [
        # iris 数据集的元组: (数据集 ID 61, 字典 {"data_id": 61}, 样本数 150, 特征数 4, 目标数 1)
        (61, {"data_id": 61}, 150, 4, 1),
        # iris 数据集的元组: (数据集 ID 61, 字典 {"name": "iris", "version": 1}, 样本数 150, 特征数 4, 目标数 1)
        (61, {"name": "iris", "version": 1}, 150, 4, 1),
        # anneal 数据集的元组: (数据集 ID 2, 字典 {"data_id": 2}, 样本数 11, 特征数 38, 目标数 1)
        (2, {"data_id": 2}, 11, 38, 1),
        # anneal 数据集的元组: (数据集 ID 2, 字典 {"name": "anneal", "version": 1}, 样本数 11, 特征数 38, 目标数 1)
        (2, {"name": "anneal", "version": 1}, 11, 38, 1),
        # cpu 数据集的元组: (数据集 ID 561, 字典 {"data_id": 561}, 样本数 209, 特征数 7, 目标数 1)
        (561, {"data_id": 561}, 209, 7, 1),
        # cpu 数据集的元组: (数据集 ID 561, 字典 {"name": "cpu", "version": 1}, 样本数 209, 特征数 7, 目标数 1)
        (561, {"name": "cpu", "version": 1}, 209, 7, 1),
        # emotions 数据集的元组: (数据集 ID 40589, 字典 {"data_id": 40589}, 样本数 13, 特征数 72, 目标数 6)
        (40589, {"data_id": 40589}, 13, 72, 6),
        # adult-census 数据集的元组: (数据集 ID 1119, 字典 {"data_id": 1119}, 样本数 10, 特征数 14, 目标数 1)
        (1119, {"data_id": 1119}, 10, 14, 1),
        # adult-census 数据集的元组: (数据集 ID 1119, 字典 {"name": "adult-census"}, 样本数 10, 特征数 14, 目标数 1)
        (1119, {"name": "adult-census"}, 10, 14, 1),
        # miceprotein 数据集的元组: (数据集 ID 40966, 字典 {"data_id": 40966}, 样本数 7, 特征数 77, 目标数 1)
        (40966, {"data_id": 40966}, 7, 77, 1),
        # miceprotein 数据集的元组: (数据集 ID 40966, 字典 {"name": "MiceProtein"}, 样本数 7, 特征数 77, 目标数 1)
        (40966, {"name": "MiceProtein"}, 7, 77, 1),
    ],
# 使用 pytest 的 mark 来参数化测试函数，允许多次运行并使用不同的参数
@pytest.mark.parametrize("parser", ["liac-arff", "pandas"])
def test_fetch_openml_as_frame_false(
    monkeypatch,             # 传入 monkeypatch 对象，用于模拟补丁和数据
    data_id,                 # 数据集的 ID
    dataset_params,          # 其他数据集参数
    n_samples,               # 样本数量
    n_features,              # 特征数量
    n_targets,               # 目标数量
    parser,                  # 使用的解析器（liac-arff 或 pandas）
):
    """Check the behaviour of `fetch_openml` with `as_frame=False`.

    Fetch both by ID and/or name + version.
    """
    pytest.importorskip("pandas")  # 如果导入 pandas 失败，跳过测试

    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)  # 模拟网络功能的补丁
    bunch = fetch_openml(
        as_frame=False,        # 不返回 Pandas DataFrame，而是返回 Bunch 对象
        cache=False,           # 禁用缓存
        parser=parser,         # 指定使用的数据解析器
        **dataset_params,      # 其他数据集相关参数
    )
    assert int(bunch.details["id"]) == data_id  # 断言数据集 ID 匹配
    assert isinstance(bunch, Bunch)             # 断言 bunch 是 Bunch 对象的实例

    assert bunch.frame is None  # 断言没有返回 Pandas DataFrame

    assert isinstance(bunch.data, np.ndarray)  # 断言数据集数据是 NumPy 数组
    assert bunch.data.shape == (n_samples, n_features)  # 断言数据集数据形状正确

    assert isinstance(bunch.target, np.ndarray)  # 断言数据集目标是 NumPy 数组
    if n_targets == 1:
        assert bunch.target.shape == (n_samples,)  # 断言单目标数据的形状
    else:
        assert bunch.target.shape == (n_samples, n_targets)  # 断言多目标数据的形状

    assert isinstance(bunch.categories, dict)  # 断言数据集的类别是字典类型


@pytest.mark.parametrize("data_id", [61, 1119, 40945])
def test_fetch_openml_consistency_parser(monkeypatch, data_id):
    """Check the consistency of the LIAC-ARFF and pandas parsers."""
    pd = pytest.importorskip("pandas")  # 如果导入 pandas 失败，跳过测试

    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)  # 模拟网络功能的补丁
    bunch_liac = fetch_openml(
        data_id=data_id,       # 使用数据集的 ID 来获取数据
        as_frame=True,         # 返回 Pandas DataFrame
        cache=False,           # 禁用缓存
        parser="liac-arff",    # 使用 LIAC-ARFF 解析器
    )
    bunch_pandas = fetch_openml(
        data_id=data_id,       # 使用数据集的 ID 来获取数据
        as_frame=True,         # 返回 Pandas DataFrame
        cache=False,           # 禁用缓存
        parser="pandas",       # 使用 Pandas 解析器
    )

    # 数据框架的输入特征应该匹配到某些数值类型的转换（例如 float64 <=> Int64），这是由于 LIAC-ARFF 解析器的限制。
    data_liac, data_pandas = bunch_liac.data, bunch_pandas.data

    def convert_numerical_dtypes(series):
        pandas_series = data_pandas[series.name]
        if pd.api.types.is_numeric_dtype(pandas_series):
            return series.astype(pandas_series.dtype)
        else:
            return series

    # 应用数值数据类型的转换后，LIAC-ARFF 的数据集应该与 Pandas 的数据集匹配
    data_liac_with_fixed_dtypes = data_liac.apply(convert_numerical_dtypes)
    pd.testing.assert_frame_equal(data_liac_with_fixed_dtypes, data_pandas)

    # 同时检查 .frame 属性是否匹配
    frame_liac, frame_pandas = bunch_liac.frame, bunch_pandas.frame

    # 注意，.frame 属性是 .data 属性的超集：
    pd.testing.assert_frame_equal(frame_pandas[bunch_pandas.feature_names], data_pandas)

    # 然而，剩余的列，通常是目标（targets），由于 LIAC-ARFF 解析器的限制，可能不会由两个解析器相似地转换数据类型。
    # 因此，这些列需要额外的数据类型转换：
    # 定义函数，用于将数据框中指定系列的数据类型转换为与 Pandas 数据框中相同的类型
    def convert_numerical_and_categorical_dtypes(series):
        # 从 Pandas 数据框中提取指定系列的数据
        pandas_series = frame_pandas[series.name]
        
        # 检查数据类型是否为数值型
        if pd.api.types.is_numeric_dtype(pandas_series):
            # 如果是数值型，则将当前系列转换为相同的数据类型
            return series.astype(pandas_series.dtype)
        # 检查数据类型是否为分类型
        elif isinstance(pandas_series.dtype, pd.CategoricalDtype):
            # 如果是分类型，则比较分类特征
            # LIAC 使用字符串表示类别，我们重命名类别使其可以与 Pandas 解析器比较。
            # 修复 LIAC-ARFF 中的这种行为将允许在未来检查一致性，但我们不计划长期维护 LIAC-ARFF。
            return series.cat.rename_categories(pandas_series.cat.categories)
        else:
            # 如果不是数值型也不是分类型，则直接返回当前系列
            return series

    # 应用函数 convert_numerical_and_categorical_dtypes 到 frame_liac 的每一列
    frame_liac_with_fixed_dtypes = frame_liac.apply(
        convert_numerical_and_categorical_dtypes
    )

    # 断言 frame_liac_with_fixed_dtypes 和 frame_pandas 在内容上相等，用于验证转换的准确性
    pd.testing.assert_frame_equal(frame_liac_with_fixed_dtypes, frame_pandas)
# 使用 pytest 的 parametrize 装饰器来多次运行这个测试函数，每次提供不同的 parser 参数
@pytest.mark.parametrize("parser", ["liac-arff", "pandas"])
def test_fetch_openml_equivalence_array_dataframe(monkeypatch, parser):
    """Check the equivalence of the dataset when using `as_frame=False` and
    `as_frame=True`.
    """
    # 导入 pandas 库，如果导入失败则跳过这个测试
    pytest.importorskip("pandas")

    # 数据集 ID
    data_id = 61
    # 使用 monkeypatch 修改 webbased 函数，返回 gzip 压缩的响应数据
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)

    # 使用 fetch_openml 获取数据集，as_frame=True 的情况
    bunch_as_frame_true = fetch_openml(
        data_id=data_id,
        as_frame=True,
        cache=False,
        parser=parser,
    )

    # 使用 fetch_openml 获取数据集，as_frame=False 的情况
    bunch_as_frame_false = fetch_openml(
        data_id=data_id,
        as_frame=False,
        cache=False,
        parser=parser,
    )

    # 断言两种获取方式的数据部分应该相等
    assert_allclose(bunch_as_frame_false.data, bunch_as_frame_true.data)
    # 断言两种获取方式的目标部分应该相等
    assert_array_equal(bunch_as_frame_false.target, bunch_as_frame_true.target)


# 使用 pytest 的 parametrize 装饰器来多次运行这个测试函数，每次提供不同的 parser 参数
@pytest.mark.parametrize("parser", ["liac-arff", "pandas"])
def test_fetch_openml_iris_pandas(monkeypatch, parser):
    """Check fetching on a numerical only dataset with string labels."""
    # 导入 pandas 库，如果导入失败则跳过这个测试
    pd = pytest.importorskip("pandas")
    # 导入 pandas 的 CategoricalDtype 类型
    CategoricalDtype = pd.api.types.CategoricalDtype

    # 数据集 ID
    data_id = 61
    # 数据集的形状
    data_shape = (150, 4)
    # 目标列的形状
    target_shape = (150,)
    # DataFrame 的形状
    frame_shape = (150, 5)

    # 目标数据类型，包含三种 Iris 类别
    target_dtype = CategoricalDtype(
        ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    )
    # 数据列的数据类型为四个 np.float64 类型
    data_dtypes = [np.float64] * 4
    # 数据列的名称
    data_names = ["sepallength", "sepalwidth", "petallength", "petalwidth"]
    # 目标列的名称
    target_name = "class"

    # 使用 monkeypatch 修改 webbased 函数，返回 True 表示成功
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)

    # 使用 fetch_openml 获取数据集
    bunch = fetch_openml(
        data_id=data_id,
        as_frame=True,
        cache=False,
        parser=parser,
    )

    # 获取数据集的数据部分
    data = bunch.data
    # 获取数据集的目标部分
    target = bunch.target
    # 获取数据集的完整帧（frame）
    frame = bunch.frame

    # 断言数据部分是 pandas 的 DataFrame 类型
    assert isinstance(data, pd.DataFrame)
    # 断言数据部分的数据类型符合预期
    assert np.all(data.dtypes == data_dtypes)
    # 断言数据部分的形状符合预期
    assert data.shape == data_shape
    # 断言数据部分的列名符合预期
    assert np.all(data.columns == data_names)
    # 断言数据集对象的特征名称与数据列名一致
    assert np.all(bunch.feature_names == data_names)
    # 断言数据集对象的目标名称与目标列名一致
    assert bunch.target_names == [target_name]

    # 断言目标部分是 pandas 的 Series 类型
    assert isinstance(target, pd.Series)
    # 断言目标部分的数据类型符合预期
    assert target.dtype == target_dtype
    # 断言目标部分的形状符合预期
    assert target.shape == target_shape
    # 断言目标部分的列名符合预期
    assert target.name == target_name
    # 断言目标部分的索引是唯一的
    assert target.index.is_unique

    # 断言帧（frame）部分是 pandas 的 DataFrame 类型
    assert isinstance(frame, pd.DataFrame)
    # 断言帧（frame）部分的形状符合预期
    assert frame.shape == frame_shape
    # 断言帧（frame）部分的数据类型符合预期，包括数据列和目标列
    assert np.all(frame.dtypes == data_dtypes + [target_dtype])
    # 断言帧（frame）部分的索引是唯一的


# 使用 pytest 的 parametrize 装饰器来多次运行这个测试函数，每次提供不同的 parser 参数
@pytest.mark.parametrize("parser", ["liac-arff", "pandas"])
@pytest.mark.parametrize("target_column", ["petalwidth", ["petalwidth", "petallength"]])
def test_fetch_openml_forcing_targets(monkeypatch, parser, target_column):
    """Check that we can force the target to not be the default target."""
    # 导入 pandas 库，如果导入失败则跳过这个测试
    pd = pytest.importorskip("pandas")

    # 数据集 ID
    data_id = 61
    # 使用 monkeypatch 修改 webbased 函数，返回 True 表示成功
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)

    # 使用 fetch_openml 获取数据集，指定目标列的情况
    bunch_forcing_target = fetch_openml(
        data_id=data_id,
        as_frame=True,
        cache=False,
        target_column=target_column,
        parser=parser,
    )
    # 使用 fetch_openml 函数获取数据集的默认配置
    bunch_default = fetch_openml(
        data_id=data_id,   # 数据集的ID，用于标识数据集
        as_frame=True,     # 返回数据框形式的数据集
        cache=False,       # 禁用缓存，确保每次都重新获取数据
        parser=parser,     # 数据解析器的选项，可能用于特定数据格式的处理
    )

    # 检查两个数据集对象的数据框是否相等
    pd.testing.assert_frame_equal(bunch_forcing_target.frame, bunch_default.frame)

    # 如果目标列是一个列表
    if isinstance(target_column, list):
        # 检查目标数据集的列索引是否与目标列列表相等
        pd.testing.assert_index_equal(
            bunch_forcing_target.target.columns, pd.Index(target_column)
        )
        # 确保目标数据集的数据形状是 (150, 3)
        assert bunch_forcing_target.data.shape == (150, 3)
    else:
        # 检查目标数据集的目标列名称是否与目标列相等
        assert bunch_forcing_target.target.name == target_column
        # 确保目标数据集的数据形状是 (150, 4)
        assert bunch_forcing_target.data.shape == (150, 4)
@pytest.mark.parametrize("data_id", [61, 2, 561, 40589, 1119])
@pytest.mark.parametrize("parser", ["liac-arff", "pandas"])
def test_fetch_openml_equivalence_frame_return_X_y(monkeypatch, data_id, parser):
    """Check the behaviour of `return_X_y=True` when `as_frame=True`."""
    pd = pytest.importorskip("pandas")

    # Monkey patch web-based functions to simulate data fetching
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)
    
    # Fetch dataset with `as_frame=True` and `return_X_y=False`
    bunch = fetch_openml(
        data_id=data_id,
        as_frame=True,
        cache=False,
        return_X_y=False,
        parser=parser,
    )
    
    # Fetch dataset with `as_frame=True` and `return_X_y=True`
    X, y = fetch_openml(
        data_id=data_id,
        as_frame=True,
        cache=False,
        return_X_y=True,
        parser=parser,
    )

    # Assert that the fetched data matches expected pandas objects
    pd.testing.assert_frame_equal(bunch.data, X)
    if isinstance(y, pd.Series):
        pd.testing.assert_series_equal(bunch.target, y)
    else:
        pd.testing.assert_frame_equal(bunch.target, y)


@pytest.mark.parametrize("data_id", [61, 561, 40589, 1119])
@pytest.mark.parametrize("parser", ["liac-arff", "pandas"])
def test_fetch_openml_equivalence_array_return_X_y(monkeypatch, data_id, parser):
    """Check the behaviour of `return_X_y=True` when `as_frame=False`."""
    pytest.importorskip("pandas")

    # Monkey patch web-based functions to simulate data fetching
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)
    
    # Fetch dataset with `as_frame=False` and `return_X_y=False`
    bunch = fetch_openml(
        data_id=data_id,
        as_frame=False,
        cache=False,
        return_X_y=False,
        parser=parser,
    )
    
    # Fetch dataset with `as_frame=False` and `return_X_y=True`
    X, y = fetch_openml(
        data_id=data_id,
        as_frame=False,
        cache=False,
        return_X_y=True,
        parser=parser,
    )

    # Assert that the fetched data arrays match expected values
    assert_array_equal(bunch.data, X)
    assert_array_equal(bunch.target, y)


def test_fetch_openml_difference_parsers(monkeypatch):
    """Check the difference between liac-arff and pandas parser."""
    pytest.importorskip("pandas")

    data_id = 1119
    # Monkey patch web-based functions to simulate data fetching
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)
    
    # Fetch dataset with `as_frame=False` and different parsers
    # liac-arff parser
    bunch_liac_arff = fetch_openml(
        data_id=data_id,
        as_frame=False,
        cache=False,
        parser="liac-arff",
    )
    # pandas parser
    bunch_pandas = fetch_openml(
        data_id=data_id,
        as_frame=False,
        cache=False,
        parser="pandas",
    )

    # Assert that data type expectations are met for both parsers
    assert bunch_liac_arff.data.dtype.kind == "f"  # Floating-point type
    assert bunch_pandas.data.dtype == "O"  # Object type


###############################################################################
# Test the ARFF parsing on several dataset to check if detect the correct
# types (categories, integers, floats).


@pytest.fixture(scope="module")
def datasets_column_names():
    """Returns the columns names for each dataset."""
    pass


@pytest.fixture(scope="module")
def datasets_missing_values():
    pass
    # 返回一个字典对象，包含多个键值对
    return {
        # 键为 61，值为空字典
        61: {},
        # 键为 2，值为包含多个属性及其值的字典
        2: {
            "family": 11,  # 属性 "family" 的值为 11
            "temper_rolling": 9,  # 属性 "temper_rolling" 的值为 9
            "condition": 2,  # 属性 "condition" 的值为 2
            "formability": 4,  # 属性 "formability" 的值为 4
            "non-ageing": 10,  # 属性 "non-ageing" 的值为 10
            "surface-finish": 11,  # 属性 "surface-finish" 的值为 11
            "enamelability": 11,  # 属性 "enamelability" 的值为 11
            "bc": 11,  # 属性 "bc" 的值为 11
            "bf": 10,  # 属性 "bf" 的值为 10
            "bt": 11,  # 属性 "bt" 的值为 11
            "bw%2Fme": 8,  # 属性 "bw%2Fme" 的值为 8
            "bl": 9,  # 属性 "bl" 的值为 9
            "m": 11,  # 属性 "m" 的值为 11
            "chrom": 11,  # 属性 "chrom" 的值为 11
            "phos": 11,  # 属性 "phos" 的值为 11
            "cbond": 10,  # 属性 "cbond" 的值为 10
            "marvi": 11,  # 属性 "marvi" 的值为 11
            "exptl": 11,  # 属性 "exptl" 的值为 11
            "ferro": 11,  # 属性 "ferro" 的值为 11
            "corr": 11,  # 属性 "corr" 的值为 11
            "blue%2Fbright%2Fvarn%2Fclean": 11,  # 属性 "blue%2Fbright%2Fvarn%2Fclean" 的值为 11
            "lustre": 8,  # 属性 "lustre" 的值为 8
            "jurofm": 11,  # 属性 "jurofm" 的值为 11
            "s": 11,  # 属性 "s" 的值为 11
            "p": 11,  # 属性 "p" 的值为 11
            "oil": 10,  # 属性 "oil" 的值为 10
            "packing": 11,  # 属性 "packing" 的值为 11
        },
        # 键为 561，值为空字典
        561: {},
        # 键为 40589，值为空字典
        40589: {},
        # 键为 1119，值为空字典
        1119: {},
        # 键为 40966，值为包含 "BCL2_N" 属性及其值的字典
        40966: {"BCL2_N": 7},
        # 键为 40945，值为包含多个属性及其值的字典
        40945: {
            "age": 263,  # 属性 "age" 的值为 263
            "fare": 1,  # 属性 "fare" 的值为 1
            "cabin": 1014,  # 属性 "cabin" 的值为 1014
            "embarked": 2,  # 属性 "embarked" 的值为 2
            "boat": 823,  # 属性 "boat" 的值为 823
            "body": 1188,  # 属性 "body" 的值为 1188
            "home.dest": 564,  # 属性 "home.dest" 的值为 564
        },
    }
@pytest.mark.parametrize(
    "data_id, parser, expected_n_categories, expected_n_floats, expected_n_ints",
    [
        # iris dataset
        (61, "liac-arff", 1, 4, 0),
        (61, "pandas", 1, 4, 0),
        # anneal dataset
        (2, "liac-arff", 33, 6, 0),
        (2, "pandas", 33, 2, 4),
        # cpu dataset
        (561, "liac-arff", 1, 7, 0),
        (561, "pandas", 1, 0, 7),
        # emotions dataset
        (40589, "liac-arff", 6, 72, 0),
        (40589, "pandas", 6, 69, 3),
        # adult-census dataset
        (1119, "liac-arff", 9, 6, 0),
        (1119, "pandas", 9, 0, 6),
        # miceprotein
        (40966, "liac-arff", 1, 77, 0),
        (40966, "pandas", 1, 77, 0),
        # titanic
        (40945, "liac-arff", 3, 6, 0),
        (40945, "pandas", 3, 3, 3),
    ],
)
@pytest.mark.parametrize("gzip_response", [True, False])
def test_fetch_openml_types_inference(
    monkeypatch,
    data_id,
    parser,
    expected_n_categories,
    expected_n_floats,
    expected_n_ints,
    gzip_response,
    datasets_column_names,
    datasets_missing_values,
):
    """
    测试`fetch_openml`函数推断分类、整数和浮点数的数量是否正确。
    """
    pd = pytest.importorskip("pandas")
    CategoricalDtype = pd.api.types.CategoricalDtype

    # 使用monkeypatch修改webbased函数的行为，根据参数data_id和gzip_response
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=gzip_response)

    # 调用fetch_openml函数获取数据集
    bunch = fetch_openml(
        data_id=data_id,
        as_frame=True,
        cache=False,
        parser=parser,
    )
    frame = bunch.frame

    # 计算数据框架中的分类变量数量
    n_categories = len(
        [dtype for dtype in frame.dtypes if isinstance(dtype, CategoricalDtype)]
    )
    # 计算数据框架中的浮点数变量数量
    n_floats = len([dtype for dtype in frame.dtypes if dtype.kind == "f"])
    # 计算数据框架中的整数变量数量
    n_ints = len([dtype for dtype in frame.dtypes if dtype.kind == "i"])

    # 断言分类变量的数量符合预期
    assert n_categories == expected_n_categories
    # 断言浮点数变量的数量符合预期
    assert n_floats == expected_n_floats
    # 断言整数变量的数量符合预期
    assert n_ints == expected_n_ints

    # 断言数据框架的列名列表与预期的列名列表相同
    assert frame.columns.tolist() == datasets_column_names[data_id]

    # 计算每个特征列中缺失值的数量，并与预期值进行比较
    frame_feature_to_n_nan = frame.isna().sum().to_dict()
    for name, n_missing in frame_feature_to_n_nan.items():
        expected_missing = datasets_missing_values[data_id].get(name, 0)
        assert n_missing == expected_missing


###############################################################################
# Test some more specific behaviour


@pytest.mark.parametrize(
    "params, err_msg",
    [
        (
            {"parser": "unknown"},
            "The 'parser' parameter of fetch_openml must be a str among",
        ),
        (
            {"as_frame": "unknown"},
            "The 'as_frame' parameter of fetch_openml must be an instance",
        ),
    ],
)
def test_fetch_openml_validation_parameter(monkeypatch, params, err_msg):
    data_id = 1119
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    with pytest.raises(ValueError, match=err_msg):
        # 调用fetch_openml函数，并断言捕获到指定的参数验证错误
        fetch_openml(data_id=data_id, **params)


@pytest.mark.parametrize(
    "params",
    [
        # 第一个字典，指定了参数 "as_frame" 为 True，"parser" 为 "auto"
        {"as_frame": True, "parser": "auto"},
        # 第二个字典，指定了参数 "as_frame" 为 "auto"（字符串），"parser" 为 "auto"
        {"as_frame": "auto", "parser": "auto"},
        # 第三个字典，指定了参数 "as_frame" 为 False，"parser" 为 "pandas"
        {"as_frame": False, "parser": "pandas"},
        # 第四个字典，指定了参数 "as_frame" 为 False，"parser" 为 "auto"
        {"as_frame": False, "parser": "auto"},
    ],
)
# 定义测试函数，检查在需要 pandas 时是否引发正确的错误
def test_fetch_openml_requires_pandas_error(monkeypatch, params):
    """Check that we raise the proper errors when we require pandas."""
    # 设置数据集 ID
    data_id = 1119
    try:
        # 检查 pandas 支持情况
        check_pandas_support("test_fetch_openml_requires_pandas")
    except ImportError:
        # 如果缺少 pandas，则进行 monkey patching
        _monkey_patch_webbased_functions(monkeypatch, data_id, True)
        # 设置错误信息
        err_msg = "requires pandas to be installed. Alternatively, explicitly"
        # 确保调用 fetch_openml 时抛出 ImportError，并匹配错误信息
        with pytest.raises(ImportError, match=err_msg):
            fetch_openml(data_id=data_id, **params)
    else:
        # 如果不缺少 pandas，则跳过测试
        raise SkipTest("This test requires pandas to not be installed.")


# 标记忽略警告信息：“Version 1 of dataset Australian is inactive”
@pytest.mark.filterwarnings("ignore:Version 1 of dataset Australian is inactive")
# 参数化测试用例
@pytest.mark.parametrize(
    "params, err_msg",
    [
        (
            {"parser": "pandas"},
            "Sparse ARFF datasets cannot be loaded with parser='pandas'",
        ),
        (
            {"as_frame": True},
            "Sparse ARFF datasets cannot be loaded with as_frame=True.",
        ),
        (
            {"parser": "pandas", "as_frame": True},
            "Sparse ARFF datasets cannot be loaded with as_frame=True.",
        ),
    ],
)
# 定义测试函数，检查对稀疏 ARFF 数据集和不兼容参数组合的预期错误
def test_fetch_openml_sparse_arff_error(monkeypatch, params, err_msg):
    """Check that we raise the expected error for sparse ARFF datasets and
    a wrong set of incompatible parameters.
    """
    # 导入 pandas，如果缺少则跳过测试
    pytest.importorskip("pandas")
    # 设置数据集 ID
    data_id = 292

    # 进行 monkey patching
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    # 确保调用 fetch_openml 时抛出 ValueError，并匹配错误信息
    with pytest.raises(ValueError, match=err_msg):
        fetch_openml(
            data_id=data_id,
            cache=False,
            **params,
        )


# 标记忽略警告信息：“Version 1 of dataset Australian is inactive”
@pytest.mark.filterwarnings("ignore:Version 1 of dataset Australian is inactive")
# 参数化测试用例
@pytest.mark.parametrize(
    "data_id, data_type",
    [
        (61, "dataframe"),  # iris dataset version 1
        (292, "sparse"),  # Australian dataset version 1
    ],
)
# 定义测试函数，检查 `fetch_openml` 的自动模式
def test_fetch_openml_auto_mode(monkeypatch, data_id, data_type):
    """Check the auto mode of `fetch_openml`."""
    # 导入 pandas，如果缺少则跳过测试
    pd = pytest.importorskip("pandas")

    # 进行 monkey patching
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    # 调用 fetch_openml 获取数据，根据数据类型进行断言
    data = fetch_openml(data_id=data_id, as_frame="auto", cache=False)
    klass = pd.DataFrame if data_type == "dataframe" else scipy.sparse.csr_matrix
    assert isinstance(data.data, klass)


# 定义测试函数，检查使用 LIAC-ARFF 解析器时对工作内存的警告
def test_convert_arff_data_dataframe_warning_low_memory_pandas(monkeypatch):
    """Check that we raise a warning regarding the working memory when using
    LIAC-ARFF parser."""
    # 导入 pandas，如果缺少则跳过测试
    pytest.importorskip("pandas")

    # 设置数据集 ID
    data_id = 1119
    # 进行 monkey patching
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)

    # 设置警告信息
    msg = "Could not adhere to working_memory config."
    # 确保在特定工作内存配置下调用 fetch_openml 时引发 UserWarning，并匹配警告信息
    with pytest.warns(UserWarning, match=msg):
        with config_context(working_memory=1e-6):
            fetch_openml(
                data_id=data_id,
                as_frame=True,
                cache=False,
                parser="liac-arff",
            )


# 参数化测试用例
@pytest.mark.parametrize("gzip_response", [True, False])
# 测试函数，检查当存在多个版本且未请求特定版本时是否会引发警告
def test_fetch_openml_iris_warn_multiple_version(monkeypatch, gzip_response):
    data_id = 61  # 数据集的ID
    data_name = "iris"  # 数据集的名称

    # 使用 monkeypatch 修改 webbased 函数
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)

    # 准备匹配的警告消息，用于检查警告是否包含正确的信息
    msg = re.escape(
        "Multiple active versions of the dataset matching the name"
        " iris exist. Versions may be fundamentally different, "
        "returning version 1. Available versions:\n"
        "- version 1, status: active\n"
        "  url: https://www.openml.org/search?type=data&id=61\n"
        "- version 3, status: active\n"
        "  url: https://www.openml.org/search?type=data&id=969\n"
    )

    # 使用 pytest 的 warns 方法检查是否会引发 UserWarning，并且匹配预期的警告消息
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(
            name=data_name,
            as_frame=False,
            cache=False,
            parser="liac-arff",
        )


# 参数化测试函数，检查获取不带目标的数据集是否正常工作
@pytest.mark.parametrize("gzip_response", [True, False])
def test_fetch_openml_no_target(monkeypatch, gzip_response):
    data_id = 61  # 数据集的ID
    target_column = None  # 没有指定目标列
    expected_observations = 150  # 预期的观测值数量
    expected_features = 5  # 预期的特征数量

    # 使用 monkeypatch 修改 webbased 函数
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)

    # 获取数据集，并断言数据集的形状和目标值是否为 None
    data = fetch_openml(
        data_id=data_id,
        target_column=target_column,
        cache=False,
        as_frame=False,
        parser="liac-arff",
    )
    assert data.data.shape == (expected_observations, expected_features)
    assert data.target is None


# 参数化测试函数，检查 Pandas 中分类数据中的缺失值是否兼容
@pytest.mark.parametrize("gzip_response", [True, False])
@pytest.mark.parametrize("parser", ["liac-arff", "pandas"])
def test_missing_values_pandas(monkeypatch, gzip_response, parser):
    """检查分类数据中的缺失值是否与 Pandas 的分类数据兼容"""
    pytest.importorskip("pandas")  # 如果没有安装 Pandas，则跳过测试

    data_id = 42585  # 数据集的ID
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=gzip_response)

    # 获取数据集，并获取性别列的类型
    penguins = fetch_openml(
        data_id=data_id,
        cache=False,
        as_frame=True,
        parser=parser,
    )

    # 检查性别列中是否存在缺失值，并断言分类类型的分类是否符合预期
    cat_dtype = penguins.data.dtypes["sex"]
    assert penguins.data["sex"].isna().any()
    assert_array_equal(cat_dtype.categories, ["FEMALE", "MALE", "_"])


# 参数化测试函数，检查获取已停用数据集时是否会引发警告
@pytest.mark.parametrize("gzip_response", [True, False])
@pytest.mark.parametrize(
    "dataset_params",
    [
        {"data_id": 40675},  # 使用数据集ID获取数据
        {"data_id": None, "name": "glass2", "version": 1},  # 指定名称和版本获取数据
    ],
)
def test_fetch_openml_inactive(monkeypatch, gzip_response, dataset_params):
    """检查当数据集已停用时是否会引发警告"""
    data_id = 40675  # 数据集的ID
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)

    # 准备匹配的警告消息，用于检查警告是否包含正确的信息
    msg = "Version 1 of dataset glass2 is inactive,"

    # 使用 pytest 的 warns 方法检查是否会引发 UserWarning，并且匹配预期的警告消息
    with pytest.warns(UserWarning, match=msg):
        glass2 = fetch_openml(
            cache=False, as_frame=False, parser="liac-arff", **dataset_params
        )
    assert glass2.data.shape == (163, 9)  # 断言获取的数据集形状是否符合预期
    # 断言语句，用于验证变量 `glass2.details["id"]` 的值是否等于字符串 "40675"
    assert glass2.details["id"] == "40675"
@pytest.mark.parametrize("gzip_response", [True, False])
# 使用 pytest 的 parametrize 装饰器，为 gzip_response 参数分别传入 True 和 False 运行测试
@pytest.mark.parametrize(
    "data_id, params, err_type, err_msg",
    [
        (40675, {"name": "glass2"}, ValueError, "No active dataset glass2 found"),
        (
            61,
            {"data_id": 61, "target_column": ["sepalwidth", "class"]},
            ValueError,
            "Can only handle homogeneous multi-target datasets",
        ),
        (
            40945,
            {"data_id": 40945, "as_frame": False},
            ValueError,
            (
                "STRING attributes are not supported for array representation. Try"
                " as_frame=True"
            ),
        ),
        (
            2,
            {"data_id": 2, "target_column": "family", "as_frame": True},
            ValueError,
            "Target column 'family'",
        ),
        (
            2,
            {"data_id": 2, "target_column": "family", "as_frame": False},
            ValueError,
            "Target column 'family'",
        ),
        (
            61,
            {"data_id": 61, "target_column": "undefined"},
            KeyError,
            "Could not find target_column='undefined'",
        ),
        (
            61,
            {"data_id": 61, "target_column": ["undefined", "class"]},
            KeyError,
            "Could not find target_column='undefined'",
        ),
    ],
)
# 使用 parametrize 装饰器设置多个输入参数组合，用于测试 fetch_openml 函数的异常情况
@pytest.mark.parametrize("parser", ["liac-arff", "pandas"])
# 使用 parametrize 装饰器为 parser 参数传入 "liac-arff" 和 "pandas" 两个值进行测试
def test_fetch_openml_error(
    monkeypatch, gzip_response, data_id, params, err_type, err_msg, parser
):
    # 调用 _monkey_patch_webbased_functions 函数，模拟对 webbased 函数的 monkey patch
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    # 如果 params 中的 "as_frame" 值为 True 或者 parser 为 "pandas"，则导入 pytest
    if params.get("as_frame", True) or parser == "pandas":
        pytest.importorskip("pandas")
    # 使用 pytest.raises 断言捕获特定类型（err_type）和消息（err_msg）的异常
    with pytest.raises(err_type, match=err_msg):
        fetch_openml(cache=False, parser=parser, **params)


@pytest.mark.parametrize(
    "params, err_type, err_msg",
    [
        (
            {"data_id": -1, "name": None, "version": "version"},
            ValueError,
            "The 'version' parameter of fetch_openml must be an int in the range",
        ),
        (
            {"data_id": -1, "name": "nAmE"},
            ValueError,
            "The 'data_id' parameter of fetch_openml must be an int in the range",
        ),
        (
            {"data_id": -1, "name": "nAmE", "version": "version"},
            ValueError,
            "The 'version' parameter of fetch_openml must be an int",
        ),
        (
            {},
            ValueError,
            "Neither name nor data_id are provided. Please provide name or data_id.",
        ),
    ],
)
# 使用 parametrize 装饰器设置多个输入参数组合，用于测试 fetch_openml 函数的非法参数情况
def test_fetch_openml_raises_illegal_argument(params, err_type, err_msg):
    # 使用 pytest.raises 断言捕获特定类型（err_type）和消息（err_msg）的异常
    with pytest.raises(err_type, match=err_msg):
        fetch_openml(**params)


@pytest.mark.parametrize("gzip_response", [True, False])
# 使用 pytest 的 parametrize 装饰器，为 gzip_response 参数分别传入 True 和 False 运行测试
def test_warn_ignore_attribute(monkeypatch, gzip_response):
    # 设置 data_id 变量为 40966
    data_id = 40966
    # 设置期望的错误消息模板
    expected_row_id_msg = "target_column='{}' has flag is_row_identifier."
    # 设置预期的忽略消息模板，用于格式化列名
    expected_ignore_msg = "target_column='{}' has flag is_ignore."
    # 对 webbased 函数进行 monkey patch，传入参数包括 monkeypatch、data_id 和 gzip_response
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    # 单列测试
    # 指定目标列为 "MouseID"
    target_col = "MouseID"
    # 格式化预期的行ID消息，使用指定的目标列
    msg = expected_row_id_msg.format(target_col)
    # 断言捕获 UserWarning 并匹配预期消息
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(
            data_id=data_id,
            target_column=target_col,
            cache=False,
            as_frame=False,
            parser="liac-arff",
        )
    # 再次指定目标列为 "Genotype"
    target_col = "Genotype"
    # 格式化预期的忽略消息，使用指定的目标列
    msg = expected_ignore_msg.format(target_col)
    # 断言捕获 UserWarning 并匹配预期消息
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(
            data_id=data_id,
            target_column=target_col,
            cache=False,
            as_frame=False,
            parser="liac-arff",
        )
    # 多列测试
    # 再次指定目标列为 "MouseID"
    target_col = "MouseID"
    # 格式化预期的行ID消息，使用指定的目标列
    msg = expected_row_id_msg.format(target_col)
    # 断言捕获 UserWarning 并匹配预期消息
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(
            data_id=data_id,
            target_column=[target_col, "class"],
            cache=False,
            as_frame=False,
            parser="liac-arff",
        )
    # 再次指定目标列为 "Genotype"
    target_col = "Genotype"
    # 格式化预期的忽略消息，使用指定的目标列
    msg = expected_ignore_msg.format(target_col)
    # 断言捕获 UserWarning 并匹配预期消息
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(
            data_id=data_id,
            target_column=[target_col, "class"],
            cache=False,
            as_frame=False,
            parser="liac-arff",
        )
@pytest.mark.parametrize("gzip_response", [True, False])
# 使用 pytest 提供参数化测试，测试 gzip_response 为 True 和 False 两种情况
def test_dataset_with_openml_error(monkeypatch, gzip_response):
    # 设置数据集 ID
    data_id = 1
    # 使用 monkeypatch 修改 webbased 函数
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    # 设置警告消息
    msg = "OpenML registered a problem with the dataset. It might be unusable. Error:"
    # 确保在执行过程中会发出 UserWarning 并匹配特定消息
    with pytest.warns(UserWarning, match=msg):
        # 调用 fetch_openml 函数获取数据集，设置参数为 cache=False, as_frame=False, parser="liac-arff"
        fetch_openml(data_id=data_id, cache=False, as_frame=False, parser="liac-arff")


@pytest.mark.parametrize("gzip_response", [True, False])
# 使用 pytest 提供参数化测试，测试 gzip_response 为 True 和 False 两种情况
def test_dataset_with_openml_warning(monkeypatch, gzip_response):
    # 设置数据集 ID
    data_id = 3
    # 使用 monkeypatch 修改 webbased 函数
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    # 设置警告消息
    msg = "OpenML raised a warning on the dataset. It might be unusable. Warning:"
    # 确保在执行过程中会发出 UserWarning 并匹配特定消息
    with pytest.warns(UserWarning, match=msg):
        # 调用 fetch_openml 函数获取数据集，设置参数为 cache=False, as_frame=False, parser="liac-arff"
        fetch_openml(data_id=data_id, cache=False, as_frame=False, parser="liac-arff")


def test_fetch_openml_overwrite_default_params_read_csv(monkeypatch):
    """Check that we can overwrite the default parameters of `read_csv`."""
    # 如果存在 pandas 库，则导入，否则跳过测试
    pytest.importorskip("pandas")
    # 设置数据集 ID
    data_id = 1590
    # 使用 monkeypatch 修改 webbased 函数
    _monkey_patch_webbased_functions(monkeypatch, data_id=data_id, gzip_response=False)

    common_params = {
        "data_id": data_id,
        "as_frame": True,
        "cache": False,
        "parser": "pandas",
    }

    # 默认情况下，跳过初始空格。我们检查将 `skipinitialspace` 参数设置为 False 会产生效果。
    # 获取包含初始空格的数据集
    adult_without_spaces = fetch_openml(**common_params)
    # 获取不包含初始空格的数据集，通过 read_csv_kwargs 设置参数 `skipinitialspace=False`
    adult_with_spaces = fetch_openml(
        **common_params, read_csv_kwargs={"skipinitialspace": False}
    )
    # 断言所有分类的类别中都以空格开头
    assert all(
        cat.startswith(" ") for cat in adult_with_spaces.frame["class"].cat.categories
    )
    # 断言所有分类的类别中没有以空格开头的
    assert not any(
        cat.startswith(" ")
        for cat in adult_without_spaces.frame["class"].cat.categories
    )


###############################################################################
# Test cache, retry mechanisms, checksum, etc.


@pytest.mark.parametrize("gzip_response", [True, False])
# 使用 pytest 提供参数化测试，测试 gzip_response 为 True 和 False 两种情况
def test_open_openml_url_cache(monkeypatch, gzip_response, tmpdir):
    # 设置数据集 ID
    data_id = 61

    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    # 获取数据集在本地缓存中的路径
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    # 设置缓存目录
    cache_directory = str(tmpdir.mkdir("scikit_learn_data"))
    # 首先填充缓存
    response1 = _open_openml_url(openml_path, cache_directory)
    # 断言文件存在
    location = _get_local_path(openml_path, cache_directory)
    assert os.path.isfile(location)
    # 重新下载以利用缓存
    response2 = _open_openml_url(openml_path, cache_directory)
    assert response1.read() == response2.read()


@pytest.mark.parametrize("write_to_disk", [True, False])
# 使用 pytest 提供参数化测试，测试 write_to_disk 为 True 和 False 两种情况
def test_open_openml_url_unlinks_local_path(monkeypatch, tmpdir, write_to_disk):
    # 设置数据集 ID
    data_id = 61
    # 获取数据集在 OpenML 中的路径
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    # 设置缓存目录
    cache_directory = str(tmpdir.mkdir("scikit_learn_data"))
    # 获取本地路径，根据给定的 openml_path 和缓存目录 cache_directory
    location = _get_local_path(openml_path, cache_directory)

    # 定义一个用于模拟 urlopen 的函数 _mock_urlopen，接受 request 参数和其他参数
    def _mock_urlopen(request, *args, **kwargs):
        # 如果 write_to_disk 为真，则尝试打开 location，并写入空字符串
        if write_to_disk:
            with open(location, "w") as f:
                f.write("")
        # 抛出值错误，指示请求无效
        raise ValueError("Invalid request")

    # 使用 monkeypatch 设置 sklearn.datasets._openml.urlopen 的模拟函数为 _mock_urlopen
    monkeypatch.setattr(sklearn.datasets._openml, "urlopen", _mock_urlopen)

    # 使用 pytest 的 assertRaises 上下文确保在打开 openml_path 时抛出值错误，并匹配 "Invalid request"
    with pytest.raises(ValueError, match="Invalid request"):
        _open_openml_url(openml_path, cache_directory)

    # 断言 location 路径不存在
    assert not os.path.exists(location)
# 定义一个测试函数，用于测试带有缓存清理功能的重试装饰器
def test_retry_with_clean_cache(tmpdir):
    # 设置数据 ID
    data_id = 61
    # 构建 OpenML 数据文件路径
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    # 创建缓存目录
    cache_directory = str(tmpdir.mkdir("scikit_learn_data"))
    # 获取本地路径
    location = _get_local_path(openml_path, cache_directory)
    # 创建存储文件所在目录
    os.makedirs(os.path.dirname(location))

    # 在位置上创建一个空文件
    with open(location, "w") as f:
        f.write("")

    # 定义带有重试和清理缓存功能的函数装饰器
    @_retry_with_clean_cache(openml_path, cache_directory)
    def _load_data():
        # 如果位置已存在文件，则抛出异常
        if os.path.exists(location):
            raise Exception("File exist!")
        return 1

    # 定义警告消息
    warn_msg = "Invalid cache, redownloading file"
    # 断言在重试过程中会产生 RuntimeWarning 警告，并获得预期结果值
    with pytest.warns(RuntimeWarning, match=warn_msg):
        result = _load_data()
    assert result == 1


# 定义测试函数，用于测试带有缓存清理功能的重试装饰器，处理 HTTP 错误情况
def test_retry_with_clean_cache_http_error(tmpdir):
    # 设置数据 ID
    data_id = 61
    # 构建 OpenML 数据文件路径
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    # 创建缓存目录
    cache_directory = str(tmpdir.mkdir("scikit_learn_data"))

    # 定义带有重试和清理缓存功能的函数装饰器
    @_retry_with_clean_cache(openml_path, cache_directory)
    def _load_data():
        # 模拟抛出 HTTPError
        raise HTTPError(
            url=None, code=412, msg="Simulated mock error", hdrs=None, fp=BytesIO()
        )

    # 定义错误消息
    error_msg = "Simulated mock error"
    # 断言函数调用时会抛出预期的 HTTPError 异常
    with pytest.raises(HTTPError, match=error_msg):
        _load_data()


# 使用参数化测试，验证 fetch_openml 函数的缓存和下载行为
@pytest.mark.parametrize("gzip_response", [True, False])
def test_fetch_openml_cache(monkeypatch, gzip_response, tmpdir):
    # 定义一个模拟函数，用于在 urlopen 被访问时抛出 ValueError
    def _mock_urlopen_raise(request, *args, **kwargs):
        raise ValueError(
            "This mechanism intends to test correct cache"
            "handling. As such, urlopen should never be "
            "accessed. URL: %s" % request.get_full_url()
        )

    # 设置数据 ID
    data_id = 61
    # 创建缓存目录
    cache_directory = str(tmpdir.mkdir("scikit_learn_data"))
    # 使用 Monkeypatch 修改 webbased 函数的行为
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    
    # 获取数据集 X 和标签 y
    X_fetched, y_fetched = fetch_openml(
        data_id=data_id,
        cache=True,
        data_home=cache_directory,
        return_X_y=True,
        as_frame=False,
        parser="liac-arff",
    )

    # 在 Monkeypatch 中设置 urlopen 函数行为为 _mock_urlopen_raise
    monkeypatch.setattr(sklearn.datasets._openml, "urlopen", _mock_urlopen_raise)

    # 再次获取数据集 X 和标签 y
    X_cached, y_cached = fetch_openml(
        data_id=data_id,
        cache=True,
        data_home=cache_directory,
        return_X_y=True,
        as_frame=False,
        parser="liac-arff",
    )

    # 断言两次获取的数据集 X 相等
    np.testing.assert_array_equal(X_fetched, X_cached)
    # 断言两次获取的标签 y 相等
    np.testing.assert_array_equal(y_fetched, y_cached)


# 使用参数化测试，验证 fetch_openml 函数的 checksum 功能
@pytest.mark.parametrize(
    "as_frame, parser",
    [
        (True, "liac-arff"),
        (False, "liac-arff"),
        (True, "pandas"),
        (False, "pandas"),
    ],
)
def test_fetch_openml_verify_checksum(monkeypatch, as_frame, parser, tmpdir):
    """Check that the checksum is working as expected."""
    # 如果 as_frame 为 True 或 parser 是 pandas，则导入 pandas 库，否则跳过该测试
    if as_frame or parser == "pandas":
        pytest.importorskip("pandas")

    # 设置数据 ID
    data_id = 2
    # 使用 Monkeypatch 修改 webbased 函数的行为
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)

    # 创建一个临时修改后的 arff 文件
    # 构建原始数据模块的完整路径，形如 OPENML_TEST_DATA_MODULE.id_{data_id}
    original_data_module = OPENML_TEST_DATA_MODULE + "." + f"id_{data_id}"
    # 指定原始数据文件名
    original_data_file_name = "data-v1-dl-1666876.arff.gz"
    # 构建原始数据文件的完整路径，使用 resources.files 函数
    original_data_path = resources.files(original_data_module) / original_data_file_name
    # 指定损坏拷贝文件的路径，位于临时目录 tmpdir 下
    corrupt_copy_path = tmpdir / "test_invalid_checksum.arff"

    # 打开原始数据文件，以二进制读取模式
    with original_data_path.open("rb") as orig_file:
        # 使用 gzip 解压原始数据
        orig_gzip = gzip.open(orig_file, "rb")
        # 读取原始数据为字节数组
        data = bytearray(orig_gzip.read())
        # 修改数据的最后一个字节为 ASCII 值 37

    # 使用 gzip 创建一个新的 gzip 文件，以二进制写入模式
    with gzip.GzipFile(corrupt_copy_path, "wb") as modified_gzip:
        # 将修改后的数据写入新的 gzip 文件中
        modified_gzip.write(data)

    # 使用 monkeypatch.setattr 替换 sklearn.datasets._openml.urlopen 函数为 swap_file_mock
    # 该函数用于模拟对 OpenML 的 HTTP 请求，特别处理文件下载请求
    mocked_openml_url = sklearn.datasets._openml.urlopen

    def swap_file_mock(request, *args, **kwargs):
        # 获取请求的完整 URL
        url = request.get_full_url()
        # 如果 URL 是以 "data/v1/download/1666876" 结尾，则返回损坏拷贝文件的内容
        if url.endswith("data/v1/download/1666876"):
            with open(corrupt_copy_path, "rb") as f:
                corrupted_data = f.read()
            return _MockHTTPResponse(BytesIO(corrupted_data), is_gzip=True)
        else:
            # 否则，调用原始的 mocked_openml_url 处理请求
            return mocked_openml_url(request)

    # 使用 monkeypatch.setattr 替换 sklearn.datasets._openml.urlopen 函数
    monkeypatch.setattr(sklearn.datasets._openml, "urlopen", swap_file_mock)

    # 使用 pytest.raises 捕获 ValueError 异常
    with pytest.raises(ValueError) as exc:
        # 调用 sklearn.datasets.fetch_openml 获取数据，验证失败的校验和
        sklearn.datasets.fetch_openml(
            data_id=data_id, cache=False, as_frame=as_frame, parser=parser
        )
    # 断言异常消息中包含文件路径 "1666876"
    assert exc.match("1666876")
# 定义测试函数，模拟在网络错误时重试打开 OpenML URL 的行为
def test_open_openml_url_retry_on_network_error(monkeypatch):
    # 定义模拟函数，抛出 HTTPError 异常，模拟网络错误
    def _mock_urlopen_network_error(request, *args, **kwargs):
        raise HTTPError(
            url=None, code=404, msg="Simulated network error", hdrs=None, fp=BytesIO()
        )

    # 使用 monkeypatch 替换 sklearn.datasets._openml.urlopen 方法为模拟函数
    monkeypatch.setattr(
        sklearn.datasets._openml, "urlopen", _mock_urlopen_network_error
    )

    # 定义无效的 OpenML URL
    invalid_openml_url = "invalid-url"

    # 使用 pytest.warns 检查是否发出 UserWarning，匹配特定的错误信息
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "A network error occurred while downloading"
            f" {_OPENML_PREFIX + invalid_openml_url}. Retrying..."
        ),
    ) as record:
        # 使用 pytest.raises 检查是否抛出 HTTPError 异常，匹配特定的错误信息
        with pytest.raises(HTTPError, match="Simulated network error"):
            _open_openml_url(invalid_openml_url, None, delay=0)
        # 断言记录的警告数量为 3
        assert len(record) == 3


###############################################################################
# 非回归测试


# 使用 pytest.mark.parametrize 定义多个参数化测试用例
@pytest.mark.parametrize("gzip_response", [True, False])
@pytest.mark.parametrize("parser", ("liac-arff", "pandas"))
def test_fetch_openml_with_ignored_feature(monkeypatch, gzip_response, parser):
    """Check that we can load the "zoo" dataset.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/14340
    """
    # 如果 parser 为 "pandas"，则要求导入 pandas 库，否则跳过该测试
    if parser == "pandas":
        pytest.importorskip("pandas")
    # 数据集 ID 为 62
    data_id = 62
    # 使用 _monkey_patch_webbased_functions 函数模拟 webbased 函数
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)

    # 调用 sklearn.datasets.fetch_openml 函数获取数据集
    dataset = sklearn.datasets.fetch_openml(
        data_id=data_id, cache=False, as_frame=False, parser=parser
    )
    # 断言数据集不为空
    assert dataset is not None
    # 数据集包含 17 个特征，其中 1 个特征被忽略 (animal)
    # 所以断言在最终的 Bunch 中不包含被忽略的特征
    assert dataset["data"].shape == (101, 16)
    assert "animal" not in dataset["feature_names"]


# 定义测试函数，检查在作为字符串分隔符时是否去掉单引号
def test_fetch_openml_strip_quotes(monkeypatch):
    """Check that we strip the single quotes when used as a string delimiter.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/23381
    """
    # 导入 pandas 库，如果不存在则跳过该测试
    pd = pytest.importorskip("pandas")
    # 数据集 ID 为 40966
    data_id = 40966
    # 使用 _monkey_patch_webbased_functions 函数模拟 webbased 函数
    _monkey_patch_webbased_functions(monkeypatch, data_id=data_id, gzip_response=False)

    # 定义通用参数字典
    common_params = {"as_frame": True, "cache": False, "data_id": data_id}
    # 使用 fetch_openml 函数获取数据集，parser 参数为 "pandas"
    mice_pandas = fetch_openml(parser="pandas", **common_params)
    # 使用 fetch_openml 函数获取数据集，parser 参数为 "liac-arff"
    mice_liac_arff = fetch_openml(parser="liac-arff", **common_params)
    # 断言 pandas 版本和 liac-arff 版本的 target 列相等
    pd.testing.assert_series_equal(mice_pandas.target, mice_liac_arff.target)
    # 断言 pandas 版本的 target 列中不包含以单引号开头的任何值
    assert not mice_pandas.target.str.startswith("'").any()
    # 断言 pandas 版本的 target 列中不包含以单引号结尾的任何值
    assert not mice_pandas.target.str.endswith("'").any()

    # 当列不是目标时，应观察类似的行为
    # 使用 fetch_openml 函数获取数据集，parser 参数为 "pandas"，target_column 参数为 "NUMB_N"
    mice_pandas = fetch_openml(parser="pandas", target_column="NUMB_N", **common_params)
    # 使用 fetch_openml 函数获取数据集，parser 参数为 "liac-arff"，target_column 参数为 "NUMB_N"
    mice_liac_arff = fetch_openml(
        parser="liac-arff", target_column="NUMB_N", **common_params
    )
    # 断言 pandas 版本和 liac-arff 版本的 "class" 列相等
    pd.testing.assert_series_equal(
        mice_pandas.frame["class"], mice_liac_arff.frame["class"]
    )
    # 断言检查：确保所有 "class" 列中的字符串都不以单引号开头
    assert not mice_pandas.frame["class"].str.startswith("'").any()
    
    # 断言检查：确保所有 "class" 列中的字符串都不以单引号结尾
    assert not mice_pandas.frame["class"].str.endswith("'").any()
# 检查是否能在 pandas 解析器中去除 leading whitespace。
# 这是一个非回归测试，针对以下问题：
# https://github.com/scikit-learn/scikit-learn/issues/25311
def test_fetch_openml_leading_whitespace(monkeypatch):
    # 导入 pytest，并跳过如果未安装的话
    pd = pytest.importorskip("pandas")
    # 数据集 ID
    data_id = 1590
    # 使用 monkeypatch 对象，修改 webbased 函数，设置 data_id 和禁用 gzip 响应
    _monkey_patch_webbased_functions(monkeypatch, data_id=data_id, gzip_response=False)

    # 共同的参数
    common_params = {"as_frame": True, "cache": False, "data_id": data_id}
    # 使用 pandas 解析器获取数据集
    adult_pandas = fetch_openml(parser="pandas", **common_params)
    # 使用 liac-arff 解析器获取相同的数据集
    adult_liac_arff = fetch_openml(parser="liac-arff", **common_params)
    # 断言两个数据框的 "class" 列是否相等
    pd.testing.assert_series_equal(
        adult_pandas.frame["class"], adult_liac_arff.frame["class"]
    )


# 检查是否能处理 escapechar 和单/双引号字符。
# 这是一个非回归测试，针对以下问题：
# https://github.com/scikit-learn/scikit-learn/issues/25478
def test_fetch_openml_quotechar_escapechar(monkeypatch):
    # 导入 pytest，并跳过如果未安装的话
    pd = pytest.importorskip("pandas")
    # 数据集 ID
    data_id = 42074
    # 使用 monkeypatch 对象，修改 webbased 函数，设置 data_id 和禁用 gzip 响应
    _monkey_patch_webbased_functions(monkeypatch, data_id=data_id, gzip_response=False)

    # 共同的参数
    common_params = {"as_frame": True, "cache": False, "data_id": data_id}
    # 使用 pandas 解析器获取数据集
    adult_pandas = fetch_openml(parser="pandas", **common_params)
    # 使用 liac-arff 解析器获取相同的数据集
    adult_liac_arff = fetch_openml(parser="liac-arff", **common_params)
    # 断言两个数据框是否相等
    pd.testing.assert_frame_equal(adult_pandas.frame, adult_liac_arff.frame)
```