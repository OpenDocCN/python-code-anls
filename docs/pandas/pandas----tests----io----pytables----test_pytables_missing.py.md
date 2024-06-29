# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_pytables_missing.py`

```
# 导入 pytest 库，用于单元测试
import pytest

# 导入 pandas 库中的测试装饰器模块 _test_decorators
import pandas.util._test_decorators as td

# 导入 pandas 库，并使用 pd 别名
import pandas as pd

# 导入 pandas 库中的测试工具模块 _testing
import pandas._testing as tm

# 使用测试装饰器，如果 "tables" 已安装则跳过测试
@td.skip_if_installed("tables")
# 定义一个测试函数，用于测试 pytables 是否会抛出异常
def test_pytables_raises():
    # 创建一个简单的 DataFrame 对象
    df = pd.DataFrame({"A": [1, 2]})
    # 使用 pytest 来确保抛出 ImportError 异常，并匹配消息中是否包含 "tables"
    with pytest.raises(ImportError, match="tables"):
        # 在确保环境干净的情况下，使用 foo.h5 文件路径作为上下文管理器的路径
        with tm.ensure_clean("foo.h5") as path:
            # 将 DataFrame 写入 HDF5 文件中的指定键
            df.to_hdf(path, key="df")
```