# `D:\src\scipysrc\scikit-learn\doc\conftest.py`

```
import os
import warnings  # 导入警告模块
from os import environ  # 导入环境变量模块
from os.path import exists, join  # 导入路径相关函数

import pytest  # 导入pytest测试框架
from _pytest.doctest import DoctestItem  # 导入pytest的DoctestItem类

from sklearn.datasets import get_data_home  # 导入获取数据主目录的函数
from sklearn.datasets._base import _pkl_filepath  # 导入用于处理数据文件路径的函数
from sklearn.datasets._twenty_newsgroups import CACHE_NAME  # 导入缓存名称
from sklearn.utils._testing import SkipTest, check_skip_network  # 导入跳过测试相关功能
from sklearn.utils.fixes import np_base_version, parse_version  # 导入版本相关函数


def setup_labeled_faces():
    data_home = get_data_home()  # 获取数据主目录
    if not exists(join(data_home, "lfw_home")):  # 检查数据集是否存在
        raise SkipTest("Skipping dataset loading doctests")  # 抛出跳过测试的异常信息


def setup_rcv1():
    check_skip_network()  # 检查网络连接是否可用
    # 如果数据集目录不存在，则抛出跳过测试的异常信息
    rcv1_dir = join(get_data_home(), "RCV1")
    if not exists(rcv1_dir):
        raise SkipTest("Download RCV1 dataset to run this test.")


def setup_twenty_newsgroups():
    cache_path = _pkl_filepath(get_data_home(), CACHE_NAME)  # 获取缓存文件路径
    if not exists(cache_path):  # 检查缓存文件是否存在
        raise SkipTest("Skipping dataset loading doctests")  # 抛出跳过测试的异常信息


def setup_working_with_text_data():
    check_skip_network()  # 检查网络连接是否可用
    cache_path = _pkl_filepath(get_data_home(), CACHE_NAME)  # 获取缓存文件路径
    if not exists(cache_path):  # 检查缓存文件是否存在
        raise SkipTest("Skipping dataset loading doctests")  # 抛出跳过测试的异常信息


def setup_loading_other_datasets():
    try:
        import pandas  # 尝试导入pandas库
    except ImportError:
        raise SkipTest("Skipping loading_other_datasets.rst, pandas not installed")  # 抛出跳过测试的异常信息

    # 检查环境变量SKLEARN_SKIP_NETWORK_TESTS是否为0，以确定是否运行网络测试
    run_network_tests = environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") == "0"
    if not run_network_tests:
        raise SkipTest(
            "Skipping loading_other_datasets.rst, tests can be "
            "enabled by setting SKLEARN_SKIP_NETWORK_TESTS=0"
        )


def setup_compose():
    try:
        import pandas  # 尝试导入pandas库
    except ImportError:
        raise SkipTest("Skipping compose.rst, pandas not installed")  # 抛出跳过测试的异常信息


def setup_impute():
    try:
        import pandas  # 尝试导入pandas库
    except ImportError:
        raise SkipTest("Skipping impute.rst, pandas not installed")  # 抛出跳过测试的异常信息


def setup_grid_search():
    try:
        import pandas  # 尝试导入pandas库
    except ImportError:
        raise SkipTest("Skipping grid_search.rst, pandas not installed")  # 抛出跳过测试的异常信息


def setup_preprocessing():
    try:
        import pandas  # 尝试导入pandas库

        # 如果pandas版本低于1.1.0，则抛出跳过测试的异常信息
        if parse_version(pandas.__version__) < parse_version("1.1.0"):
            raise SkipTest("Skipping preprocessing.rst, pandas version < 1.1.0")
    except ImportError:
        raise SkipTest("Skipping preprocessing.rst, pandas not installed")  # 抛出跳过测试的异常信息


def setup_unsupervised_learning():
    try:
        import skimage  # 尝试导入scikit-image库
    except ImportError:
        raise SkipTest("Skipping unsupervised_learning.rst, scikit-image not installed")  # 抛出跳过测试的异常信息
    # 忽略来自scipy.misc.face的弃用警告
    warnings.filterwarnings(
        "ignore", "The binary mode of fromstring", DeprecationWarning
    )


def skip_if_matplotlib_not_installed(fname):
    try:
        import matplotlib  # 尝试导入matplotlib库
    except ImportError:
        raise SkipTest("Skipping plot_tests.rst, matplotlib not installed")  # 抛出跳过测试的异常信息
    except ImportError:
        # 如果导入错误（ImportError），表示找不到所需的模块或库
        # 获取文件名的基本名称（不含路径部分）
        basename = os.path.basename(fname)
        # 抛出 SkipTest 异常，提示跳过对应的文档测试，说明 matplotlib 没有安装
        raise SkipTest(f"Skipping doctests for {basename}, matplotlib not installed")
def skip_if_cupy_not_installed(fname):
    try:
        import cupy  # noqa
    except ImportError:
        # 获取文件名的基本名称
        basename = os.path.basename(fname)
        # 抛出跳过测试的异常，并提供相应的消息
        raise SkipTest(f"Skipping doctests for {basename}, cupy not installed")


def pytest_runtest_setup(item):
    fname = item.fspath.strpath
    # 将文件名标准化，使用正斜杠以便在后续处理中在Windows上更容易处理
    fname = fname.replace(os.sep, "/")

    is_index = fname.endswith("datasets/index.rst")
    # 根据不同的文件名后缀进行不同的测试设置
    if fname.endswith("datasets/labeled_faces.rst") or is_index:
        setup_labeled_faces()
    elif fname.endswith("datasets/rcv1.rst") or is_index:
        setup_rcv1()
    elif fname.endswith("datasets/twenty_newsgroups.rst") or is_index:
        setup_twenty_newsgroups()
    elif fname.endswith("modules/compose.rst") or is_index:
        setup_compose()
    elif fname.endswith("datasets/loading_other_datasets.rst"):
        setup_loading_other_datasets()
    elif fname.endswith("modules/impute.rst"):
        setup_impute()
    elif fname.endswith("modules/grid_search.rst"):
        setup_grid_search()
    elif fname.endswith("modules/preprocessing.rst"):
        setup_preprocessing()
    elif fname.endswith("statistical_inference/unsupervised_learning.rst"):
        setup_unsupervised_learning()

    # 需要 matplotlib 的 rst 文件列表
    rst_files_requiring_matplotlib = [
        "modules/partial_dependence.rst",
        "modules/tree.rst",
    ]
    # 对于需要 matplotlib 的文件，如果文件名以列表中的任何一个结尾，则跳过测试
    for each in rst_files_requiring_matplotlib:
        if fname.endswith(each):
            skip_if_matplotlib_not_installed(fname)

    # 如果文件名以 array_api.rst 结尾，则检查是否需要跳过测试
    if fname.endswith("array_api.rst"):
        skip_if_cupy_not_installed(fname)


def pytest_configure(config):
    # 在测试期间使用 matplotlib 的 agg 后端
    try:
        import matplotlib

        matplotlib.use("agg")
    except ImportError:
        pass


def pytest_collection_modifyitems(config, items):
    """Called after collect is completed.

    Parameters
    ----------
    config : pytest config
    items : list of collected items
    """
    skip_doctests = False
    if np_base_version >= parse_version("2"):
        # 当使用 numpy 2 时跳过 doctests 测试
        reason = "Due to NEP 51 numpy scalar repr has changed in numpy 2"
        skip_doctests = True

    # 对于 doctest 类型的项目，设置全局变量为空字典以移除模块的作用域
    for item in items:
        if isinstance(item, DoctestItem):
            item.dtest.globs = {}

    # 如果需要跳过 doctests 测试，则标记为跳过
    if skip_doctests:
        skip_marker = pytest.mark.skip(reason=reason)

        for item in items:
            if isinstance(item, DoctestItem):
                item.add_marker(skip_marker)
```