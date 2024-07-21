# `.\pytorch\benchmarks\fastrnns\conftest.py`

```
import pytest  # noqa: F401
# 引入 pytest 模块，禁止 Flake8 检查时引发 F401 未使用警告

default_rnns = [
    "cudnn",
    "aten",
    "jit",
    "jit_premul",
    "jit_premul_bias",
    "jit_simple",
    "jit_multilayer",
    "py",
]
# 默认的循环神经网络名称列表

default_cnns = ["resnet18", "resnet18_jit", "resnet50", "resnet50_jit"]
# 默认的卷积神经网络名称列表

all_nets = default_rnns + default_cnns
# 所有网络的名称列表，包括默认的循环神经网络和卷积神经网络


def pytest_generate_tests(metafunc):
    # pytest 生成测试用例的函数，根据 metafunc 参数定制测试
    if metafunc.cls.__name__ == "TestBenchNetwork":
        metafunc.parametrize("net_name", all_nets, scope="class")
        # 参数化网络名称，作用域为整个类
        metafunc.parametrize(
            "executor", [metafunc.config.getoption("executor")], scope="class"
        )
        # 参数化执行器选项，从配置中获取，作用域为整个类
        metafunc.parametrize(
            "fuser", [metafunc.config.getoption("fuser")], scope="class"
        )
        # 参数化融合器选项，从配置中获取，作用域为整个类


def pytest_addoption(parser):
    parser.addoption("--fuser", default="old", help="fuser to use for benchmarks")
    # 添加选项 --fuser，用于指定基准测试中使用的融合器，默认为 "old"
    parser.addoption(
        "--executor", default="legacy", help="executor to use for benchmarks"
    )
    # 添加选项 --executor，用于指定基准测试中使用的执行器，默认为 "legacy"
```