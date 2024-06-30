# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_config.py`

```
# 导入内置模块 builtins，用于处理内建函数和异常
import builtins
# 导入时间模块，用于处理时间相关操作
import time
# 导入线程池执行器，用于并发执行任务
from concurrent.futures import ThreadPoolExecutor

# 导入 pytest 模块，用于编写和执行测试用例
import pytest

# 导入 sklearn 库及其子模块
import sklearn
# 从 sklearn 中导入 config_context、get_config、set_config 函数
from sklearn import config_context, get_config, set_config
# 从 sklearn.utils.fixes 导入 _IS_WASM，用于修复兼容性问题
from sklearn.utils.fixes import _IS_WASM
# 从 sklearn.utils.parallel 导入 Parallel 和 delayed，用于并行处理
from sklearn.utils.parallel import Parallel, delayed


# 定义测试函数 test_config_context，测试 sklearn 的配置上下文管理功能
def test_config_context():
    # 断言当前的配置与预期的默认配置相匹配
    assert get_config() == {
        "assume_finite": False,
        "working_memory": 1024,
        "print_changed_only": True,
        "display": "diagram",
        "array_api_dispatch": False,
        "pairwise_dist_chunk_size": 256,
        "enable_cython_pairwise_dist": True,
        "transform_output": "default",
        "enable_metadata_routing": False,
        "skip_parameter_validation": False,
    }

    # 使用 config_context 函数修改 assume_finite 配置为 True，但并不影响全局配置
    config_context(assume_finite=True)
    assert get_config()["assume_finite"] is False

    # 使用 config_context 函数作为上下文管理器，修改 assume_finite 配置为 True，并检查配置
    with config_context(assume_finite=True):
        assert get_config() == {
            "assume_finite": True,
            "working_memory": 1024,
            "print_changed_only": True,
            "display": "diagram",
            "array_api_dispatch": False,
            "pairwise_dist_chunk_size": 256,
            "enable_cython_pairwise_dist": True,
            "transform_output": "default",
            "enable_metadata_routing": False,
            "skip_parameter_validation": False,
        }
    assert get_config()["assume_finite"] is False

    # 嵌套使用 config_context 函数，演示配置的继承和覆盖效果
    with config_context(assume_finite=True):
        with config_context(assume_finite=None):
            assert get_config()["assume_finite"] is True

        assert get_config()["assume_finite"] is True

        with config_context(assume_finite=False):
            assert get_config()["assume_finite"] is False

            with config_context(assume_finite=None):
                assert get_config()["assume_finite"] is False

                # 使用 set_config 函数修改全局配置 assume_finite 为 True
                set_config(assume_finite=True)
                assert get_config()["assume_finite"] is True

            assert get_config()["assume_finite"] is False

        assert get_config()["assume_finite"] is True

    # 最终检查配置是否恢复到默认值
    assert get_config() == {
        "assume_finite": False,
        "working_memory": 1024,
        "print_changed_only": True,
        "display": "diagram",
        "array_api_dispatch": False,
        "pairwise_dist_chunk_size": 256,
        "enable_cython_pairwise_dist": True,
        "transform_output": "default",
        "enable_metadata_routing": False,
        "skip_parameter_validation": False,
    }

    # 测试 config_context 函数不接受位置参数的情况
    with pytest.raises(TypeError):
        config_context(True)

    # 测试 config_context 函数不接受未知参数的情况
    with pytest.raises(TypeError):
        config_context(do_something_else=True).__enter__()


# 定义测试函数 test_config_context_exception，测试配置上下文管理中的异常情况
def test_config_context_exception():
    # 断言当前的 assume_finite 配置为 False
    assert get_config()["assume_finite"] is False
    # 尝试进入配置上下文，设置 assume_finite 为 True
    try:
        # 进入配置上下文，并设定 assume_finite=True
        with config_context(assume_finite=True):
            # 断言当前配置中 assume_finite 应为 True
            assert get_config()["assume_finite"] is True
            # 触发 ValueError 异常，用于后续处理
            raise ValueError()
    # 捕获到 ValueError 异常后执行以下代码
    except ValueError:
        # 仅处理异常，不做具体操作
        pass
    # 断言当前配置中 assume_finite 应为 False
    assert get_config()["assume_finite"] is False
# 测试设定配置函数 `set_config` 的功能
def test_set_config():
    # 断言初始配置中的 "assume_finite" 属性为 False
    assert get_config()["assume_finite"] is False
    # 将 "assume_finite" 设置为 None，但不改变当前配置
    set_config(assume_finite=None)
    # 再次断言 "assume_finite" 仍然为 False
    assert get_config()["assume_finite"] is False
    # 将 "assume_finite" 设置为 True
    set_config(assume_finite=True)
    # 断言 "assume_finite" 现在为 True
    assert get_config()["assume_finite"] is True
    # 将 "assume_finite" 再次设置为 None，但不改变当前配置
    set_config(assume_finite=None)
    # 再次断言 "assume_finite" 仍然为 True
    assert get_config()["assume_finite"] is True
    # 将 "assume_finite" 设置为 False
    set_config(assume_finite=False)
    # 最后断言 "assume_finite" 现在为 False
    assert get_config()["assume_finite"] is False

    # 使用 pytest 检查是否有未知的参数
    # 预期会抛出 TypeError 异常
    with pytest.raises(TypeError):
        set_config(do_something_else=True)


# 设置 assume_finite 的函数，等待指定的睡眠时间后返回 assume_finite 的值
def set_assume_finite(assume_finite, sleep_duration):
    """Return the value of assume_finite after waiting `sleep_duration`."""
    # 使用上下文管理器设置 assume_finite 的值
    with config_context(assume_finite=assume_finite):
        # 等待指定的睡眠时间
        time.sleep(sleep_duration)
        # 返回当前的 assume_finite 值
        return get_config()["assume_finite"]


# 使用不同的 joblib 后端并行测试全局配置是否线程安全
@pytest.mark.parametrize("backend", ["loky", "multiprocessing", "threading"])
def test_config_threadsafe_joblib(backend):
    """Test that the global config is threadsafe with all joblib backends.
    Two jobs are spawned and sets assume_finite to two different values.
    When the job with a duration 0.1s completes, the assume_finite value
    should be the same as the value passed to the function. In other words,
    it is not influenced by the other job setting assume_finite to True.
    """
    # 定义 assume_finite 和 sleep_durations 数组
    assume_finites = [False, True, False, True]
    sleep_durations = [0.1, 0.2, 0.1, 0.2]

    # 使用 Parallel 并行执行任务
    items = Parallel(backend=backend, n_jobs=2)(
        delayed(set_assume_finite)(assume_finite, sleep_dur)
        for assume_finite, sleep_dur in zip(assume_finites, sleep_durations)
    )

    # 断言执行结果与预期相符合
    assert items == [False, True, False, True]


# 使用 ThreadPoolExecutor 直接测试全局配置是否线程安全
@pytest.mark.xfail(_IS_WASM, reason="cannot start threads")
def test_config_threadsafe():
    """Uses threads directly to test that the global config does not change
    between threads. Same test as `test_config_threadsafe_joblib` but with
    `ThreadPoolExecutor`."""

    # 定义 assume_finite 和 sleep_durations 数组
    assume_finites = [False, True, False, True]
    sleep_durations = [0.1, 0.2, 0.1, 0.2]

    # 使用 ThreadPoolExecutor 并行执行任务
    with ThreadPoolExecutor(max_workers=2) as e:
        items = [
            output
            for output in e.map(set_assume_finite, assume_finites, sleep_durations)
        ]

    # 断言执行结果与预期相符合
    assert items == [False, True, False, True]


# 测试在未安装 array_api_compat 时，是否会引发错误
def test_config_array_api_dispatch_error(monkeypatch):
    """Check error is raised when array_api_compat is not installed."""

    # 隐藏 array_api_compat 的导入
    orig_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == "array_api_compat":
            raise ImportError
        return orig_import(name, *args, **kwargs)

    # 使用 monkeypatch 修改 __import__ 函数行为
    monkeypatch.setattr(builtins, "__import__", mocked_import)

    # 使用 pytest 检查是否抛出预期的 ImportError 异常
    with pytest.raises(ImportError, match="array_api_compat is required"):
        with config_context(array_api_dispatch=True):
            pass

    # 再次使用 pytest 检查是否抛出预期的 ImportError 异常
    with pytest.raises(ImportError, match="array_api_compat is required"):
        set_config(array_api_dispatch=True)


# 继续下一个测试函数的注释
def test_config_array_api_dispatch_error_numpy(monkeypatch):
    """Check error when NumPy is too old"""
    # 定义一个模拟导入函数，用于替换内置的 __import__ 函数
    orig_import = builtins.__import__

    # 定义一个模拟导入函数的函数体，用于特定模块名的模拟导入
    def mocked_import(name, *args, **kwargs):
        # 如果导入的模块名为 "array_api_compat"，返回一个空对象
        if name == "array_api_compat":
            return object()
        # 否则调用原始的 __import__ 函数进行导入
        return orig_import(name, *args, **kwargs)

    # 使用 pytest 的 monkeypatch 功能，替换内置 __import__ 函数为自定义的 mocked_import 函数
    monkeypatch.setattr(builtins, "__import__", mocked_import)
    
    # 设置 sklearn.utils._array_api.numpy 模块的版本号为 "1.20"
    monkeypatch.setattr(sklearn.utils._array_api.numpy, "__version__", "1.20")

    # 使用 pytest.raises 检测是否会抛出 ImportError 异常，并且匹配异常信息中包含 "NumPy must be 1.21 or newer"
    with pytest.raises(ImportError, match="NumPy must be 1.21 or newer"):
        # 在 array_api_dispatch=True 的配置上下文中执行测试代码
        with config_context(array_api_dispatch=True):
            pass

    # 使用 pytest.raises 检测是否会抛出 ImportError 异常，并且匹配异常信息中包含 "NumPy must be 1.21 or newer"
    with pytest.raises(ImportError, match="NumPy must be 1.21 or newer"):
        # 设置全局配置中的 array_api_dispatch=True，并检测是否抛出异常
        set_config(array_api_dispatch=True)
```