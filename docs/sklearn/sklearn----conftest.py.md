# `D:\src\scipysrc\scikit-learn\sklearn\conftest.py`

```
# 导入内置模块 builtins、platform 和 sys，以及从 contextlib 中导入 suppress
# 从 functools 中导入 wraps
# 从 os 中导入 environ
# 从 unittest 中导入 SkipTest
import builtins
import platform
import sys
from contextlib import suppress
from functools import wraps
from os import environ
from unittest import SkipTest

# 导入第三方库
import joblib
import numpy as np
import pytest
from _pytest.doctest import DoctestItem
from threadpoolctl import threadpool_limits

# 导入 scikit-learn 相关模块和函数
from sklearn import config_context, set_config
from sklearn._min_dependencies import PYTEST_MIN_VERSION
from sklearn.datasets import (
    fetch_20newsgroups,
    fetch_20newsgroups_vectorized,
    fetch_california_housing,
    fetch_covtype,
    fetch_kddcup99,
    fetch_lfw_pairs,
    fetch_lfw_people,
    fetch_olivetti_faces,
    fetch_rcv1,
    fetch_species_distributions,
)
from sklearn.utils._testing import get_pytest_filterwarning_lines
from sklearn.utils.fixes import (
    _IS_32BIT,
    np_base_version,
    parse_version,
    sp_version,
)

# 检查 pytest 版本是否符合要求
if parse_version(pytest.__version__) < parse_version(PYTEST_MIN_VERSION):
    raise ImportError(
        f"Your version of pytest is too old. Got version {pytest.__version__}, you"
        f" should have pytest >= {PYTEST_MIN_VERSION} installed."
    )

# 检查是否需要网络访问来获取 SciPy 数据集
scipy_datasets_require_network = sp_version >= parse_version("1.10")


# 创建一个 pytest fixture，用于启用 SLEP006
@pytest.fixture
def enable_slep006():
    """Enable SLEP006 for all tests."""
    with config_context(enable_metadata_routing=True):
        yield


# 定义一个函数，根据 SciPy 版本是否需要网络访问来获取数据集
def raccoon_face_or_skip():
    # 如果需要网络访问
    if scipy_datasets_require_network:
        # 检查环境变量 SKLEARN_SKIP_NETWORK_TESTS 是否允许运行网络测试
        run_network_tests = environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") == "0"
        if not run_network_tests:
            raise SkipTest("test is enabled when SKLEARN_SKIP_NETWORK_TESTS=0")

        try:
            import pooch  # noqa
        except ImportError:
            raise SkipTest("test requires pooch to be installed")

        # 从 scipy.datasets 中导入 face 函数
        from scipy.datasets import face
    else:
        # 如果不需要网络访问，从 scipy.misc 中导入 face 函数
        from scipy.misc import face

    # 返回灰度的 face 数据
    return face(gray=True)


# 定义一个字典，将数据集名称映射到对应的 fetch 函数
dataset_fetchers = {
    "fetch_20newsgroups_fxt": fetch_20newsgroups,
    "fetch_20newsgroups_vectorized_fxt": fetch_20newsgroups_vectorized,
    "fetch_california_housing_fxt": fetch_california_housing,
    "fetch_covtype_fxt": fetch_covtype,
    "fetch_kddcup99_fxt": fetch_kddcup99,
    "fetch_lfw_pairs_fxt": fetch_lfw_pairs,
    "fetch_lfw_people_fxt": fetch_lfw_people,
    "fetch_olivetti_faces_fxt": fetch_olivetti_faces,
    "fetch_rcv1_fxt": fetch_rcv1,
    "fetch_species_distributions_fxt": fetch_species_distributions,
}

# 如果需要网络访问，添加 raccoon_face_fxt 数据集到 dataset_fetchers 字典中
if scipy_datasets_require_network:
    dataset_fetchers["raccoon_face_fxt"] = raccoon_face_or_skip

# 定义一个 pytest mark，用于标记跳过特定条件下的测试
_SKIP32_MARK = pytest.mark.skipif(
    environ.get("SKLEARN_RUN_FLOAT32_TESTS", "0") != "1",
    reason="Set SKLEARN_RUN_FLOAT32_TESTS=1 to run float32 dtype tests",
)


# 定义全局 fixture，参数化运行测试的数据类型为 np.float32 和 np.float64
@pytest.fixture(params=[pytest.param(np.float32, marks=_SKIP32_MARK), np.float64])
def global_dtype(request):
    yield request.param


# 定义一个私有函数，用于下载并返回数据集（如果环境需要）
def _fetch_fixture(f):
    """Fetch dataset (download if missing and requested by environment)."""
    # 从环境变量中获取是否跳过网络测试的设置，默认为 False
    download_if_missing = environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") == "0"
    
    # 定义装饰器函数 `wrapped`，用来包装原函数 `f`
    @wraps(f)
    def wrapped(*args, **kwargs):
        # 将 `download_if_missing` 参数传递给被装饰的函数
        kwargs["download_if_missing"] = download_if_missing
        try:
            # 调用被装饰的函数 `f`，并返回其结果
            return f(*args, **kwargs)
        except OSError as e:
            # 捕获 OSError 异常
            # 如果异常消息不是 "Data not found and `download_if_missing` is False"，则重新抛出异常
            if str(e) != "Data not found and `download_if_missing` is False":
                raise
            # 如果异常消息符合预期，则使用 pytest 跳过当前测试
            pytest.skip("test is enabled when SKLEARN_SKIP_NETWORK_TESTS=0")
    
    # 返回一个 pytest 的 fixture，该 fixture 每次返回 `wrapped` 函数对象
    return pytest.fixture(lambda: wrapped)
# 添加用于获取数据的测试夹具
fetch_20newsgroups_fxt = _fetch_fixture(fetch_20newsgroups)
fetch_20newsgroups_vectorized_fxt = _fetch_fixture(fetch_20newsgroups_vectorized)
fetch_california_housing_fxt = _fetch_fixture(fetch_california_housing)
fetch_covtype_fxt = _fetch_fixture(fetch_covtype)
fetch_kddcup99_fxt = _fetch_fixture(fetch_kddcup99)
fetch_lfw_pairs_fxt = _fetch_fixture(fetch_lfw_pairs)
fetch_lfw_people_fxt = _fetch_fixture(fetch_lfw_people)
fetch_olivetti_faces_fxt = _fetch_fixture(fetch_olivetti_faces)
fetch_rcv1_fxt = _fetch_fixture(fetch_rcv1)
fetch_species_distributions_fxt = _fetch_fixture(fetch_species_distributions)
# 使用 pytest 的 fixture 装饰器为 raccoon_face_or_skip 创建夹具
raccoon_face_fxt = pytest.fixture(raccoon_face_or_skip)


def pytest_collection_modifyitems(config, items):
    """Called after collect is completed.

    Parameters
    ----------
    config : pytest config
        Pytest 的配置对象
    items : list of collected items
        收集到的测试项列表
    """
    # 根据环境变量 SKLEARN_SKIP_NETWORK_TESTS 判断是否运行网络测试
    run_network_tests = environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") == "0"
    # 创建用于跳过网络测试的 pytest marker
    skip_network = pytest.mark.skip(
        reason="test is enabled when SKLEARN_SKIP_NETWORK_TESTS=0"
    )

    # 在收集测试项时下载数据集，以避免与 pytest-xdist 并行运行时的线程不安全行为
    dataset_features_set = set(dataset_fetchers)
    datasets_to_download = set()

    for item in items:
        if isinstance(item, DoctestItem) and "fetch_" in item.name:
            fetcher_function_name = item.name.split(".")[-1]
            dataset_fetchers_key = f"{fetcher_function_name}_fxt"
            dataset_to_fetch = set([dataset_fetchers_key]) & dataset_features_set
        elif not hasattr(item, "fixturenames"):
            continue
        else:
            item_fixtures = set(item.fixturenames)
            dataset_to_fetch = item_fixtures & dataset_features_set

        if not dataset_to_fetch:
            continue

        if run_network_tests:
            datasets_to_download |= dataset_to_fetch
        else:
            # 网络测试被跳过
            item.add_marker(skip_network)

    # 只在 pytest-xdist 的第一个工作进程上下载数据集，以避免线程不安全行为
    # 如果未使用 pytest-xdist，仍在测试运行前下载数据集
    worker_id = environ.get("PYTEST_XDIST_WORKER", "gw0")
    if worker_id == "gw0" and run_network_tests:
        for name in datasets_to_download:
            with suppress(SkipTest):
                dataset_fetchers[name]()

    for item in items:
        # 在 ARM64 平台上已知 GradientBoostingClassifier 存在问题
        if (
            item.name.endswith("GradientBoostingClassifier")
            and platform.machine() == "aarch64"
        ):
            marker = pytest.mark.xfail(
                reason=(
                    "know failure. See "
                    "https://github.com/scikit-learn/scikit-learn/issues/17797"  # noqa
                )
            )
            item.add_marker(marker)

    # 是否跳过 doctest 测试
    skip_doctests = False
    # 尝试导入 matplotlib 库，如果失败则跳过后续的文档测试
    try:
        import matplotlib  # noqa
    except ImportError:
        skip_doctests = True
        reason = "matplotlib is required to run the doctests"

    # 如果运行环境为 32 位系统，跳过文档测试，并指定跳过原因
    if _IS_32BIT:
        reason = "doctest are only run when the default numpy int is 64 bits."
        skip_doctests = True
    # 如果运行环境为 Windows，跳过文档测试，并指定跳过原因
    elif sys.platform.startswith("win32"):
        reason = (
            "doctests are not run for Windows because numpy arrays "
            "repr is inconsistent across platforms."
        )
        skip_doctests = True

    # 如果 numpy 的基础版本大于等于 2，跳过文档测试，并指定跳过原因
    if np_base_version >= parse_version("2"):
        reason = "Due to NEP 51 numpy scalar repr has changed in numpy 2"
        skip_doctests = True

    # 通常 doctest 具有整个模块的作用域。在这里将全局命名空间设置为空字典，
    # 以移除模块的作用域：https://docs.python.org/3/library/doctest.html#what-s-the-execution-context
    for item in items:
        if isinstance(item, DoctestItem):
            item.dtest.globs = {}

    # 如果需要跳过文档测试，则为相关的测试项添加跳过标记
    if skip_doctests:
        skip_marker = pytest.mark.skip(reason=reason)

        for item in items:
            if isinstance(item, DoctestItem):
                # 解决在上下文管理器中向 doctest 添加跳过标记时的内部错误，
                # 参考 https://github.com/pytest-dev/pytest/issues/8796 获取更多细节。
                if item.name != "sklearn._config.config_context":
                    item.add_marker(skip_marker)

    # 尝试导入 PIL 库，如果失败则将 pillow_installed 设为 False
    try:
        import PIL  # noqa

        pillow_installed = True
    except ImportError:
        pillow_installed = False

    # 如果 pillow (或 PIL) 未安装，则为特定测试项添加跳过标记
    if not pillow_installed:
        skip_marker = pytest.mark.skip(reason="pillow (or PIL) not installed!")
        for item in items:
            if item.name in [
                "sklearn.feature_extraction.image.PatchExtractor",
                "sklearn.feature_extraction.image.extract_patches_2d",
            ]:
                item.add_marker(skip_marker)
@pytest.fixture(scope="function")
def pyplot():
    """
    Setup and teardown fixture for matplotlib.

    This fixture checks if we can import matplotlib. If not, the tests will be
    skipped. Otherwise, we close the figures before and after running the
    functions.

    Returns
    -------
    pyplot : module
        The ``matplotlib.pyplot`` module.
    """
    # 尝试导入 matplotlib.pyplot 模块，如果导入失败则跳过测试
    pyplot = pytest.importorskip("matplotlib.pyplot")
    # 关闭所有已存在的图形窗口，确保每个测试函数运行前后都是干净的状态
    pyplot.close("all")
    yield pyplot
    # 再次关闭所有图形窗口，确保测试结束后的清理工作
    pyplot.close("all")


def pytest_generate_tests(metafunc):
    """
    Parametrization of global_random_seed fixture based on the
    SKLEARN_TESTS_GLOBAL_RANDOM_SEED environment variable.

    The goal of this fixture is to prevent tests that use it to be sensitive
    to a specific seed value while still being deterministic by default.

    See the documentation for the SKLEARN_TESTS_GLOBAL_RANDOM_SEED
    variable for instructions on how to use this fixture.

    https://scikit-learn.org/dev/computing/parallelism.html#sklearn-tests-global-random-seed
    """
    # 在使用 pytest-xdist 插件时，此函数会在 xdist 的工作进程中调用。
    # 我们依赖于 SKLEARN_TESTS_GLOBAL_RANDOM_SEED 环境变量，该变量在运行 pytest 之前设置，并且在 xdist 工作进程中可用，因为它们是子进程。
    RANDOM_SEED_RANGE = list(range(100))  # 所有在 [0, 99] 范围内的种子都是有效的。
    random_seed_var = environ.get("SKLEARN_TESTS_GLOBAL_RANDOM_SEED")

    default_random_seeds = [42]

    if random_seed_var is None:
        random_seeds = default_random_seeds
    elif random_seed_var == "all":
        random_seeds = RANDOM_SEED_RANGE
    else:
        if "-" in random_seed_var:
            start, stop = random_seed_var.split("-")
            random_seeds = list(range(int(start), int(stop) + 1))
        else:
            random_seeds = [int(random_seed_var)]

        if min(random_seeds) < 0 or max(random_seeds) > 99:
            raise ValueError(
                "The value(s) of the environment variable "
                "SKLEARN_TESTS_GLOBAL_RANDOM_SEED must be in the range [0, 99] "
                f"(or 'all'), got: {random_seed_var}"
            )

    # 如果测试函数中包含了 global_random_seed 这个参数，就使用上面计算得到的随机种子列表来参数化该参数。
    if "global_random_seed" in metafunc.fixturenames:
        metafunc.parametrize("global_random_seed", random_seeds)


def pytest_configure(config):
    """
    Use matplotlib agg backend during the tests including doctests.

    Also, adjust the allowed parallelism based on the number of CPU cores and
    xdist workers to prevent oversubscription.

    """
    # 尝试导入 matplotlib 模块，使用 agg 后端进行测试
    try:
        import matplotlib

        matplotlib.use("agg")
    except ImportError:
        pass

    # 获取系统的物理 CPU 核心数作为允许的并行度
    allowed_parallelism = joblib.cpu_count(only_physical_cores=True)
    xdist_worker_count = environ.get("PYTEST_XDIST_WORKER_COUNT")
    if xdist_worker_count is not None:
        # 根据 xdist 使用的工作进程数量，设置 OpenMP 和 BLAS 的线程数，以避免过度订阅。
        allowed_parallelism = max(allowed_parallelism // int(xdist_worker_count), 1)
    # 设置线程池的并行度上限
    threadpool_limits(allowed_parallelism)
    # 检查环境变量中名为 "SKLEARN_WARNINGS_AS_ERRORS" 的值是否不为 "0"
    if environ.get("SKLEARN_WARNINGS_AS_ERRORS", "0") != "0":
        # 如果条件成立，说明需要将警告视为错误处理
        # 这似乎是唯一以编程方式更改配置 filterwarnings 的方法
        # 此方法建议参考自 https://github.com/pytest-dev/pytest/issues/3311#issuecomment-373177592
        # 遍历获取 pytest_filterwarning_lines 返回的行，并添加到 pytest 的配置中
        for line in get_pytest_filterwarning_lines():
            config.addinivalue_line("filterwarnings", line)
@pytest.fixture
def hide_available_pandas(monkeypatch):
    """
    Fixture用于模拟未安装pandas的情况。

    Args:
        monkeypatch: pytest提供的用于在测试运行时修改行为的工具

    Returns:
        None
    """
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        """
        自定义的导入函数，用于模拟导入库时的行为。

        Args:
            name (str): 导入的库的名称
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            module: 导入的模块对象或者抛出ImportError异常
        """
        if name == "pandas":
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    # 将 monkeypatch 应用到 builtins 模块的 __import__ 方法上，替换为 mocked_import
    monkeypatch.setattr(builtins, "__import__", mocked_import)


@pytest.fixture
def print_changed_only_false():
    """
    Fixture用于将全局配置中的 `print_changed_only` 设置为 False，在测试期间保持不变。

    Yields:
        None

    Notes:
        在测试运行时，yield之前的代码段被用于设置配置，yield之后的代码段用于恢复默认配置。
    """
    set_config(print_changed_only=False)
    yield
    set_config(print_changed_only=True)  # reset to default
```