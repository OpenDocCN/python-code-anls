# `D:\src\scipysrc\scipy\scipy\conftest.py`

```
# Pytest customization

# 导入所需的模块和库
import json
import os
import warnings
import tempfile
from contextlib import contextmanager

# 导入 NumPy 和相关的测试模块
import numpy as np
import numpy.testing as npt

# 导入 Pytest 和 Hypothesis
import pytest
import hypothesis

# 导入 SciPy 内部库
from scipy._lib._fpumode import get_fpu_mode
from scipy._lib._testutils import FPUModeChangeWarning
from scipy._lib._array_api import SCIPY_ARRAY_API, SCIPY_DEVICE

# 尝试导入 scipy_doctest 的配置，标记是否成功导入
try:
    from scipy_doctest.conftest import dt_config
    HAVE_SCPDT = True
except ModuleNotFoundError:
    HAVE_SCPDT = False


# Pytest 的配置函数，用于设置各种自定义选项
def pytest_configure(config):
    # 添加自定义标记到 pytest 配置中
    config.addinivalue_line("markers",
        "slow: Tests that are very slow.")
    config.addinivalue_line("markers",
        "xslow: mark test as extremely slow (not run unless explicitly requested)")
    config.addinivalue_line("markers",
        "xfail_on_32bit: mark test as failing on 32-bit platforms")

    # 尝试导入 pytest_timeout 模块，如果失败则添加超时标记
    try:
        import pytest_timeout  # noqa:F401
    except Exception:
        config.addinivalue_line(
            "markers", 'timeout: mark a test for a non-default timeout')

    # 尝试导入 pytest_fail_slow 模块，如果失败则添加失败慢速标记
    try:
        from pytest_fail_slow import parse_duration  # type: ignore[import-not-found] # noqa:F401,E501
    except Exception:
        config.addinivalue_line(
            "markers", 'fail_slow: mark a test for a non-default timeout failure')

    # 添加关于 `skip_xp_backends` 的自定义标记说明
    config.addinivalue_line("markers",
        "skip_xp_backends(*backends, reasons=None, np_only=False, cpu_only=False): "
        "mark the desired skip configuration for the `skip_xp_backends` fixture.")


# Pytest 的测试运行设置函数，用于在运行测试之前进行各种检查和配置
def pytest_runtest_setup(item):
    # 检查是否存在 "xslow" 标记，如果存在并且未设置环境变量 SCIPY_XSLOW=1，则跳过测试
    mark = item.get_closest_marker("xslow")
    if mark is not None:
        try:
            v = int(os.environ.get('SCIPY_XSLOW', '0'))
        except ValueError:
            v = False
        if not v:
            pytest.skip("very slow test; "
                        "set environment variable SCIPY_XSLOW=1 to run it")

    # 检查是否存在 "xfail_on_32bit" 标记，并且当前平台为 32 位，标记测试为预期失败
    mark = item.get_closest_marker("xfail_on_32bit")
    if mark is not None and np.intp(0).itemsize < 8:
        pytest.xfail(f'Fails on our 32-bit test platform(s): {mark.args[0]}')

    # 检查 threadpoolctl 的旧版本问题，可能会导致该警告被触发，参考 GitHub issue gh-14441
    # 使用 npt 库的 suppress_warnings 上下文管理器来抑制特定的警告信息
    with npt.suppress_warnings() as sup:
        # 过滤 pytest.PytestUnraisableExceptionWarning 警告
        sup.filter(pytest.PytestUnraisableExceptionWarning)

        # 尝试导入 threadpoolctl 库
        try:
            from threadpoolctl import threadpool_limits

            # 设置标志，表示 threadpoolctl 库已成功导入
            HAS_THREADPOOLCTL = True
        except Exception:  # 捕获可能的 ImportError 或 AttributeError 异常
            # threadpoolctl 库是可选依赖项，如果导入失败，设置标志为 False
            HAS_THREADPOOLCTL = False

        # 如果成功导入 threadpoolctl 库
        if HAS_THREADPOOLCTL:
            # 根据环境变量 PYTEST_XDIST_WORKER_COUNT 获取 xdist 的工作进程数
            try:
                xdist_worker_count = int(os.environ['PYTEST_XDIST_WORKER_COUNT'])
            except KeyError:
                # 当环境变量未设置时，说明 pytest-xdist 未安装，直接返回
                return

            # 如果未设置环境变量 OMP_NUM_THREADS
            if not os.getenv('OMP_NUM_THREADS'):
                # 计算最大的 OpenMP 线程数，使用物理核心数的一半
                max_openmp_threads = os.cpu_count() // 2
                # 计算每个工作进程的线程数，确保至少为 1
                threads_per_worker = max(max_openmp_threads // xdist_worker_count, 1)
                try:
                    # 使用 threadpool_limits 函数限制每个工作进程的线程数，调用的 API 为 'blas'
                    threadpool_limits(threads_per_worker, user_api='blas')
                except Exception:
                    # 捕获任何异常，确保程序的健壮性
                    return
@pytest.fixture(scope="function", autouse=True)
def check_fpu_mode(request):
    """
    Check FPU mode was not changed during the test.
    """
    # 获取当前的 FPU 模式
    old_mode = get_fpu_mode()
    # 让出执行权，执行测试函数
    yield
    # 获取测试后的新的 FPU 模式
    new_mode = get_fpu_mode()

    # 检查 FPU 模式是否在测试期间发生变化，并发出警告
    if old_mode != new_mode:
        warnings.warn(f"FPU mode changed from {old_mode:#x} to {new_mode:#x} during "
                      "the test",
                      category=FPUModeChangeWarning, stacklevel=0)


# Array API backend handling
# 支持的数组 API 后端字典，初始包含 NumPy
xp_available_backends = {'numpy': np}

if SCIPY_ARRAY_API and isinstance(SCIPY_ARRAY_API, str):
    # 填充后端字典，根据可用库添加支持的数组 API 后端
    try:
        import array_api_strict
        xp_available_backends.update({'array_api_strict': array_api_strict})
    except ImportError:
        pass

    try:
        import torch  # type: ignore[import-not-found]
        xp_available_backends.update({'pytorch': torch})
        # 可以使用 `mps` 或 `cpu`
        torch.set_default_device(SCIPY_DEVICE)
    except ImportError:
        pass

    try:
        import cupy  # type: ignore[import-not-found]
        xp_available_backends.update({'cupy': cupy})
    except ImportError:
        pass

    try:
        import jax.numpy  # type: ignore[import-not-found]
        xp_available_backends.update({'jax.numpy': jax.numpy})
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_device", jax.devices(SCIPY_DEVICE)[0])
    except ImportError:
        pass

    # 默认情况下，使用所有可用的后端
    if SCIPY_ARRAY_API.lower() not in ("1", "true"):
        SCIPY_ARRAY_API_ = json.loads(SCIPY_ARRAY_API)

        if 'all' in SCIPY_ARRAY_API_:
            pass  # 和 True 一样
        else:
            # 根据过滤后的字典选择一部分后端
            try:
                xp_available_backends = {
                    backend: xp_available_backends[backend]
                    for backend in SCIPY_ARRAY_API_
                }
            except KeyError:
                msg = f"'--array-api-backend' must be in {xp_available_backends.keys()}"
                raise ValueError(msg)

if 'cupy' in xp_available_backends:
    SCIPY_DEVICE = 'cuda'

# 用于参数化测试的标记，将所有可用的后端作为参数传递
array_api_compatible = pytest.mark.parametrize("xp", xp_available_backends.values())

# 当使用 SCIPY_ARRAY_API 时，跳过测试的标记
skip_xp_invalid_arg = pytest.mark.skipif(SCIPY_ARRAY_API,
    reason=('Test involves masked arrays, object arrays, or other types '
            'that are not valid input when `SCIPY_ARRAY_API` is used.'))


@pytest.fixture
def skip_xp_backends(xp, request):
    """
    Skip based on the ``skip_xp_backends`` marker.

    Parameters
    ----------
    *backends : tuple
        Backends to skip, e.g. ``("array_api_strict", "torch")``.
        These are overriden when ``np_only`` is ``True``, and are not
        necessary to provide for non-CPU backends when ``cpu_only`` is ``True``.

    """
    reasons : list, optional
        # 用于存储每个跳过的原因的列表。当 np_only 为 True 时，应该是一个单元素列表。否则，应该是一个与 backends 对应的原因列表。
        # 如果未提供，则使用默认原因。注意，无法通过 cpu_only 指定自定义原因。默认值为 None。
    np_only : bool, optional
        # 当为 True 时，跳过除默认的 NumPy 后端之外的所有后端的测试。在这种情况下，不需要提供任何 backends。要指定原因，请传递一个单元素列表给 reasons。默认值为 False。
    cpu_only : bool, optional
        # 当为 True 时，在非 CPU 设备上跳过测试。在这种情况下，不需要提供任何 backends，但是任何 backends 也会在 CPU 上跳过。默认值为 False。
    """
    # 如果测试用例不包含 "skip_xp_backends" 关键字，则直接返回，不执行跳过逻辑
    if "skip_xp_backends" not in request.keywords:
        return
    
    # 获取 "skip_xp_backends" 关键字的参数和关键字参数
    backends = request.keywords["skip_xp_backends"].args
    kwargs = request.keywords["skip_xp_backends"].kwargs
    
    # 获取 np_only 和 cpu_only 的值，默认为 False
    np_only = kwargs.get("np_only", False)
    cpu_only = kwargs.get("cpu_only", False)
    
    # 如果 np_only 为 True，则根据 xp 模块的名称判断是否为 NumPy，如果不是则跳过测试
    if np_only:
        reasons = kwargs.get("reasons", ["do not run with non-NumPy backends."])
        reason = reasons[0]
        if xp.__name__ != 'numpy':
            pytest.skip(reason=reason)
        return
    
    # 如果 cpu_only 为 True，则根据条件判断是否在不允许的情况下运行测试
    if cpu_only:
        reason = "do not run with `SCIPY_ARRAY_API` set and not on CPU"
        if SCIPY_ARRAY_API and SCIPY_DEVICE != 'cpu':
            if xp.__name__ == 'cupy':
                pytest.skip(reason=reason)
            elif xp.__name__ == 'torch':
                if 'cpu' not in xp.empty(0).device.type:
                    pytest.skip(reason=reason)
            elif xp.__name__ == 'jax.numpy':
                for d in xp.empty(0).devices():
                    if 'cpu' not in d.device_kind:
                        pytest.skip(reason=reason)
    
    # 如果 backends 不为 None，则根据 xp 模块的名称和给定的 backends 列表跳过测试
    if backends is not None:
        reasons = kwargs.get("reasons", False)
        for i, backend in enumerate(backends):
            if xp.__name__ == backend:
                if not reasons:
                    reason = f"do not run with array API backend: {backend}"
                else:
                    reason = reasons[i]
                pytest.skip(reason=reason)
# 按照 NumPy 的 conftest.py 的方法...
# 使用已知且持久的临时目录用于 hypothesis 的缓存，这些缓存可以由操作系统或用户自动清除。
hypothesis.configuration.set_hypothesis_home_dir(
    os.path.join(tempfile.gettempdir(), ".hypothesis")
)

# 注册两个自定义的 SciPy 配置文件 - 更多细节请参见 https://hypothesis.readthedocs.io/en/latest/settings.html
# 第一个配置文件设计用于我们自己的 CI 运行；后者还强制确定性并设计用于通过 scipy.test() 使用。
hypothesis.settings.register_profile(
    name="nondeterministic", deadline=None, print_blob=True,
)
hypothesis.settings.register_profile(
    name="deterministic",
    deadline=None, print_blob=True, database=None, derandomize=True,
    suppress_health_check=list(hypothesis.HealthCheck),
)

# 当前配置文件由环境变量 `SCIPY_HYPOTHESIS_PROFILE` 设置
# 未来可能将选择集成到 dev.py 中。
SCIPY_HYPOTHESIS_PROFILE = os.environ.get("SCIPY_HYPOTHESIS_PROFILE",
                                          "deterministic")
hypothesis.settings.load_profile(SCIPY_HYPOTHESIS_PROFILE)


############################################################################
# doctesting stuff

if HAVE_SCPDT:

    # FIXME: populate the dict once
    @contextmanager
    # 将 dt_config.user_context_mgr 设置为 warnings_errors_and_rng 的上下文管理器
    dt_config.user_context_mgr = warnings_errors_and_rng
    # 设置 dt_config.skiplist，跳过以下函数的 doctest 测试
    dt_config.skiplist = set([
        'scipy.linalg.LinAlgError',     # 来自 numpy
        'scipy.fftpack.fftshift',       # fftpack 也来自 numpy
        'scipy.fftpack.ifftshift',
        'scipy.fftpack.fftfreq',
        'scipy.special.sinc',           # sinc 函数来自 numpy
        'scipy.optimize.show_options',  # 不需要进行 doctest
        'scipy.signal.normalize',       # 操作警告 (XXX 暂时跳过)
        'scipy.sparse.linalg.norm',     # XXX 暂时跳过
    ])

    # 这些受 NumPy 2.0 标量表示影响：依赖于字符串比较
    if np.__version__ < "2":
        # 更新 dt_config.skiplist，跳过以下函数的 doctest 测试
        dt_config.skiplist.update(set([
            'scipy.io.hb_read',
            'scipy.io.hb_write',
            'scipy.sparse.csgraph.connected_components',
            'scipy.sparse.csgraph.depth_first_order',
            'scipy.sparse.csgraph.shortest_path',
            'scipy.sparse.csgraph.floyd_warshall',
            'scipy.sparse.csgraph.dijkstra',
            'scipy.sparse.csgraph.bellman_ford',
            'scipy.sparse.csgraph.johnson',
            'scipy.sparse.csgraph.yen',
            'scipy.sparse.csgraph.breadth_first_order',
            'scipy.sparse.csgraph.reverse_cuthill_mckee',
            'scipy.sparse.csgraph.structural_rank',
            'scipy.sparse.csgraph.construct_dist_matrix',
            'scipy.sparse.csgraph.reconstruct_path',
            'scipy.ndimage.value_indices',
            'scipy.stats.mstats.describe',
    ]))

    # 帮助 pytest 进行集合：这些名称要么是私有的（分布），要么只是不需要 doctest。
    # 定义 pytest 测试时需要忽略的额外模块列表
    dt_config.pytest_extra_ignore = [
        "scipy.stats.distributions",          # 忽略 scipy.stats.distributions 模块
        "scipy.optimize.cython_optimize",    # 忽略 scipy.optimize.cython_optimize 模块
        "scipy.test",                        # 忽略 scipy.test 模块
        "scipy.show_config",                 # 忽略 scipy.show_config 模块
        # 等同于运行 "pytest --ignore=path/to/file"
        "scipy/special/_precompute",         # 忽略 scipy/special/_precompute 文件
        "scipy/interpolate/_interpnd_info.py",  # 忽略 scipy/interpolate/_interpnd_info.py 文件
        "scipy/_lib/array_api_compat",       # 忽略 scipy/_lib/array_api_compat 模块
        "scipy/_lib/highs",                  # 忽略 scipy/_lib/highs 模块
        "scipy/_lib/unuran",                 # 忽略 scipy/_lib/unuran 模块
        "scipy/_lib/_gcutils.py",            # 忽略 scipy/_lib/_gcutils.py 文件
        "scipy/_lib/doccer.py",              # 忽略 scipy/_lib/doccer.py 文件
        "scipy/_lib/_uarray",                # 忽略 scipy/_lib/_uarray 模块
    ]
    
    # 定义 pytest 测试时的预期失败情况字典
    dt_config.pytest_extra_xfail = {
        # 模块名: 原因
        "io.rst": "",                                # io.rst 模块预期失败
        "ND_regular_grid.rst": "ReST parser limitation",  # ND_regular_grid.rst 模块因 ReST 解析器限制而预期失败
        "extrapolation_examples.rst": "ReST parser limitation",  # extrapolation_examples.rst 模块因 ReST 解析器限制而预期失败
        "sampling_pinv.rst": "__cinit__ unexpected argument",  # sampling_pinv.rst 模块因 __cinit__ 参数意外而预期失败
        "sampling_srou.rst": "nan in scalar_power",   # sampling_srou.rst 模块因 scalar_power 中有 NaN 而预期失败
        "probability_distributions.rst": "integration warning",  # probability_distributions.rst 模块因积分警告而预期失败
    }
    
    # tutorials
    # 定义伪代码相关的文件集合
    dt_config.pseudocode = set(['integrate.nquad(func,'])
    
    # 定义本地资源与其关联的文件字典
    dt_config.local_resources = {'io.rst': ["octave_a.mat"]}
# 导入pandas库，用于数据处理和分析
import pandas as pd

# 定义一个名为data的列表，包含三个字典，每个字典都有两个键值对
data = [
    {'A': 1, 'B': 2},
    {'A': 5, 'B': 10, 'C': 20},
    {'B': 3, 'C': 4}
]

# 使用pandas库的DataFrame函数，将data列表转换为数据框（DataFrame）
df = pd.DataFrame(data)

# 输出打印数据框df的内容，显示在控制台
print(df)
```