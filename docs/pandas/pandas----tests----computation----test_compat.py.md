# `D:\src\scipysrc\pandas\pandas\tests\computation\test_compat.py`

```
import pytest  # 导入 pytest 测试框架

from pandas.compat._optional import VERSIONS  # 导入 pandas 兼容性版本信息

import pandas as pd  # 导入 pandas 库
from pandas.core.computation import expr  # 导入 pandas 计算表达式模块
from pandas.core.computation.engines import ENGINES  # 导入 pandas 计算引擎列表
from pandas.util.version import Version  # 导入 pandas 版本管理工具


def test_compat():
    # 测试当前环境是否与 numexpr 版本兼容

    from pandas.core.computation.check import NUMEXPR_INSTALLED  # 导入 numexpr 安装检查模块

    ne = pytest.importorskip("numexpr")  # 导入并检查 numexpr 库

    ver = ne.__version__  # 获取 numexpr 库的版本号
    if Version(ver) < Version(VERSIONS["numexpr"]):  # 比较 numexpr 版本与 pandas 所需版本
        assert not NUMEXPR_INSTALLED  # 如果版本过低，断言 numexpr 未安装
    else:
        assert NUMEXPR_INSTALLED  # 如果版本符合要求，断言 numexpr 已安装


@pytest.mark.parametrize("engine", ENGINES)  # 参数化测试引擎列表中的各个引擎
@pytest.mark.parametrize("parser", expr.PARSERS)  # 参数化测试解析器列表中的各个解析器
def test_invalid_numexpr_version(engine, parser):
    if engine == "numexpr":
        pytest.importorskip("numexpr")  # 如果引擎为 numexpr，导入并检查 numexpr 库
    a, b = 1, 2  # noqa: F841  # 定义变量 a 和 b，这里禁止未使用变量的警告
    res = pd.eval("a + b", engine=engine, parser=parser)  # 使用指定引擎和解析器计算表达式结果
    assert res == 3  # 断言计算结果为 3
```