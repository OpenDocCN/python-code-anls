# `D:\src\scipysrc\pandas\pandas\tests\plotting\test_backend.py`

```
# 导入系统模块
import sys
# 导入 types 模块，用于创建新的模块对象
import types

# 导入 pytest 测试框架
import pytest

# 导入 pandas 库中的测试装饰器模块
import pandas.util._test_decorators as td

# 导入 pandas 库
import pandas


# 定义 pytest 的 fixture，创建一个名为 dummy_backend 的虚拟后端模块
@pytest.fixture
def dummy_backend():
    # 创建一个名为 pandas_dummy_backend 的新模块对象
    db = types.ModuleType("pandas_dummy_backend")
    # 在新模块中设置一个名为 plot 的 lambda 函数，用于返回字符串 "used_dummy"
    setattr(db, "plot", lambda *args, **kwargs: "used_dummy")
    return db


# 定义 pytest 的 fixture，用于恢复绘图后端为 matplotlib
@pytest.fixture
def restore_backend():
    """Restore the plotting backend to matplotlib"""
    # 使用 pandas 的 option_context 将绘图后端设置为 matplotlib
    with pandas.option_context("plotting.backend", "matplotlib"):
        # 使用 yield 返回上下文管理器
        yield


# 测试函数：检查设置错误的绘图后端是否引发 ValueError 异常，并且绘图后端被重置为 matplotlib
def test_backend_is_not_module():
    msg = "Could not find plotting backend 'not_an_existing_module'."
    with pytest.raises(ValueError, match=msg):
        pandas.set_option("plotting.backend", "not_an_existing_module")

    assert pandas.options.plotting.backend == "matplotlib"


# 测试函数：检查设置正确的绘图后端是否成功，并且返回的后端与预期的 dummy_backend 匹配
def test_backend_is_correct(monkeypatch, restore_backend, dummy_backend):
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)

    pandas.set_option("plotting.backend", "pandas_dummy_backend")
    assert pandas.get_option("plotting.backend") == "pandas_dummy_backend"
    assert (
        pandas.plotting._core._get_plot_backend("pandas_dummy_backend") is dummy_backend
    )


# 测试函数：检查在绘图调用中设置绘图后端是否有效，确保使用 dummy_backend 后端的 plot 方法返回 "used_dummy"
def test_backend_can_be_set_in_plot_call(monkeypatch, restore_backend, dummy_backend):
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)
    df = pandas.DataFrame([1, 2, 3])

    assert pandas.get_option("plotting.backend") == "matplotlib"
    assert df.plot(backend="pandas_dummy_backend") == "used_dummy"


# 测试函数：验证注册的 entry point 是否正确地将 pandas_dummy_backend 注册为 my_ep_backend，并可以成功使用
def test_register_entrypoint(restore_backend, tmp_path, monkeypatch, dummy_backend):
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)

    dist_info = tmp_path / "my_backend-0.0.0.dist-info"
    dist_info.mkdir()
    # 写入 entry_points.txt 文件，指定 pandas_plotting_backends 的 entry point 为 my_ep_backend
    (dist_info / "entry_points.txt").write_bytes(
        b"[pandas_plotting_backends]\nmy_ep_backend = pandas_dummy_backend\n"
    )

    # 确认 my_ep_backend 已成功注册为 dummy_backend
    assert pandas.plotting._core._get_plot_backend("my_ep_backend") is dummy_backend

    # 使用 option_context 将绘图后端设置为 my_ep_backend，确认设置成功
    with pandas.option_context("plotting.backend", "my_ep_backend"):
        assert pandas.plotting._core._get_plot_backend() is dummy_backend


# 测试函数：在未安装 matplotlib 的情况下，验证设置未知的绘图后端是否引发 ValueError 异常
@td.skip_if_installed("matplotlib")
def test_no_matplotlib_ok():
    msg = (
        'matplotlib is required for plotting when the default backend "matplotlib" is '
        "selected."
    )
    # 使用 pytest 模块来测试是否会引发 ImportError 异常，并且异常信息需要与给定的 msg 变量匹配
    with pytest.raises(ImportError, match=msg):
        # 调用 pandas 模块中的 plotting._core._get_plot_backend 函数，尝试获取 matplotlib 绘图后端
        pandas.plotting._core._get_plot_backend("matplotlib")
def test_extra_kinds_ok(monkeypatch, restore_backend, dummy_backend):
    # 设置 monkeypatch，用于模拟修改 pandas 的模块字典，添加 dummy_backend
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)
    # 设置 pandas 绘图后端选项为 pandas_dummy_backend
    pandas.set_option("plotting.backend", "pandas_dummy_backend")
    # 创建一个包含列"A"的 DataFrame，数据为 [1, 2, 3]
    df = pandas.DataFrame({"A": [1, 2, 3]})
    # 使用 DataFrame 的 plot 方法，尝试使用一种并不存在的绘图类型 "not a real kind"
    df.plot(kind="not a real kind")
```