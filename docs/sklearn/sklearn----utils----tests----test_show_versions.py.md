# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_show_versions.py`

```
# 导入 threadpool_info 函数，用于获取线程池信息
from threadpoolctl import threadpool_info

# 从 sklearn.utils._show_versions 模块中导入 _get_deps_info, _get_sys_info, show_versions 函数
from sklearn.utils._show_versions import _get_deps_info, _get_sys_info, show_versions

# 从 sklearn.utils._testing 模块中导入 ignore_warnings 函数
from sklearn.utils._testing import ignore_warnings


# 定义测试函数 test_get_sys_info
def test_get_sys_info():
    # 调用 _get_sys_info 函数，获取系统信息
    sys_info = _get_sys_info()

    # 断言确保返回的 sys_info 字典中包含指定的键
    assert "python" in sys_info
    assert "executable" in sys_info
    assert "machine" in sys_info


# 定义测试函数 test_get_deps_info
def test_get_deps_info():
    # 使用 ignore_warnings 上下文管理器，调用 _get_deps_info 函数，获取依赖信息
    with ignore_warnings():
        deps_info = _get_deps_info()

    # 断言确保返回的 deps_info 字典中包含指定的依赖项
    assert "pip" in deps_info
    assert "setuptools" in deps_info
    assert "sklearn" in deps_info
    assert "numpy" in deps_info
    assert "scipy" in deps_info
    assert "Cython" in deps_info
    assert "pandas" in deps_info
    assert "matplotlib" in deps_info
    assert "joblib" in deps_info


# 定义测试函数 test_show_versions，使用 capsys 参数捕获标准输出和错误
def test_show_versions(capsys):
    # 使用 ignore_warnings 上下文管理器
    with ignore_warnings():
        # 调用 show_versions 函数，打印版本信息
        show_versions()
        # 读取捕获的标准输出和错误
        out, err = capsys.readouterr()

    # 断言确保标准输出中包含指定的内容
    assert "python" in out
    assert "numpy" in out

    # 获取当前线程池信息
    info = threadpool_info()
    # 如果有线程池信息，则断言标准输出中包含特定线程池信息的提示
    if info:
        assert "threadpoolctl info:" in out
```