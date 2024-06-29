# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\common.py`

```
# 从 collections.abc 模块中导入 Generator 类型
from collections.abc import Generator
# 从 contextlib 模块中导入 contextmanager 装饰器
from contextlib import contextmanager
# 导入 pathlib 模块，用于处理路径
import pathlib
# 导入 tempfile 模块，用于创建临时文件和目录
import tempfile

# 导入 pytest 模块
import pytest

# 从 pandas.io.pytables 模块中导入 HDFStore 类
from pandas.io.pytables import HDFStore

# 导入 tables 模块，如果不存在则跳过
tables = pytest.importorskip("tables")
# 设置 tables 模块的一些参数，限制线程数量以避免文件共享问题
tables.parameters.MAX_NUMEXPR_THREADS = 1
tables.parameters.MAX_BLOSC_THREADS = 1
tables.parameters.MAX_THREADS = 1


# 定义一个安全关闭 HDFStore 的函数
def safe_close(store):
    try:
        # 如果 store 不为 None，则关闭它
        if store is not None:
            store.close()
    except OSError:
        # 捕获 OSError 异常，不做处理
        pass


# 定义一个上下文管理器，确保文件在使用后被清理
@contextmanager
def ensure_clean_store(
    path, mode="a", complevel=None, complib=None, fletcher32=False
) -> Generator[HDFStore, None, None]:
    # 使用 tempfile 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdirname:
        # 构造临时文件的完整路径
        tmp_path = pathlib.Path(tmpdirname, path)
        # 使用 HDFStore 打开临时文件，作为上下文管理器的一部分
        with HDFStore(
            tmp_path,
            mode=mode,
            complevel=complevel,
            complib=complib,
            fletcher32=fletcher32,
        ) as store:
            yield store  # 返回 HDFStore 实例


# 定义一个函数，用于尝试删除 HDFStore 中指定的键
def _maybe_remove(store, key):
    """
    For tests using tables, try removing the table to be sure there is
    no content from previous tests using the same table name.
    """
    try:
        store.remove(key)  # 尝试移除指定键对应的内容
    except (ValueError, KeyError):
        # 捕获 ValueError 或 KeyError 异常，不做处理
        pass
```