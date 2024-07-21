# `.\pytorch\test\torch_np\conftest.py`

```
# Owner(s): ["module: dynamo"]

# 导入系统模块
import sys

# 导入 pytest 模块
import pytest

# 导入 torch._numpy 模块，作为 tnp 别名
import torch._numpy as tnp


# pytest 配置函数，用于配置 pytest
def pytest_configure(config):
    # 添加自定义标记 "slow"，用于标记非常慢的测试用例
    config.addinivalue_line("markers", "slow: very slow tests")


# pytest 参数添加函数，用于添加命令行参数
def pytest_addoption(parser):
    # 添加 "--runslow" 参数，用于运行慢速测试用例
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    # 添加 "--nonp" 参数，用于在访问 NumPy 时产生错误
    parser.addoption("--nonp", action="store_true", help="error when NumPy is accessed")


# 用于模拟不可访问的 NumPy 对象的类
class Inaccessible:
    # 覆盖 __getattribute__ 方法，抛出错误信息，指示非法访问
    def __getattribute__(self, attr):
        raise RuntimeError(f"Using --nonp but accessed np.{attr}")


# pytest 会话启动钩子函数，用于在 pytest 会话开始时执行操作
def pytest_sessionstart(session):
    # 如果传入了 "--nonp" 参数
    if session.config.getoption("--nonp"):
        # 将 numpy 模块替换为不可访问的对象
        sys.modules["numpy"] = Inaccessible()


# pytest 参数化测试用例的钩子函数
def pytest_generate_tests(metafunc):
    """
    Hook to parametrize test cases
    See https://docs.pytest.org/en/6.2.x/parametrize.html#pytest-generate-tests

    The logic here allows us to test with both NumPy-proper and torch._numpy.
    Normally we'd just test torch._numpy, e.g.

        import torch._numpy as np
        ...
        def test_foo():
            np.array([42])
            ...

    but this hook allows us to test NumPy-proper as well, e.g.

        def test_foo(np):
            np.array([42])
            ...

    np is a pytest parameter, which is either NumPy-proper or torch._numpy. This
    allows us to sanity check our own tests, so that tested behaviour is
    consistent with NumPy-proper.

    pytest will have test names respective to the library being tested, e.g.

        $ pytest --collect-only
        test_foo[torch._numpy]
        test_foo[numpy]

    """
    # 初始化 numpy 参数列表，包含 torch._numpy 模块
    np_params = [tnp]

    try:
        # 尝试导入 numpy 模块
        import numpy as np
    except ImportError:
        pass
    else:
        # 如果 numpy 成功导入且不是 Inaccessible 类型，则将其添加到参数列表中
        if not isinstance(np, Inaccessible):  # i.e. --nonp was used
            np_params.append(np)

    # 如果测试用例需要 "np" 参数，则使用参数化方式提供 np_params
    if "np" in metafunc.fixturenames:
        metafunc.parametrize("np", np_params)


# pytest 测试集合修改函数，用于根据命令行选项修改测试项
def pytest_collection_modifyitems(config, items):
    # 如果未传入 "--runslow" 参数，则跳过所有标记为 "slow" 的测试用例
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="slow test, use --runslow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
```