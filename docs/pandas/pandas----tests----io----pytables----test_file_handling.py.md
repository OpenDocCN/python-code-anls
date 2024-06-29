# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_file_handling.py`

```
# 导入必要的库和模块
import os  # 导入操作系统相关功能的模块

import numpy as np  # 导入数值计算库NumPy
import pytest  # 导入测试框架pytest

from pandas.compat import (  # 从pandas.compat模块中导入多个函数和变量
    PY311,  # Python版本兼容性相关
    is_ci_environment,  # 判断是否处于CI环境
    is_platform_linux,  # 判断是否运行在Linux平台
    is_platform_little_endian,  # 判断系统是否小端存储
    is_platform_mac,  # 判断是否运行在Mac平台
)
from pandas.errors import (  # 从pandas.errors模块导入异常类
    ClosedFileError,  # 文件关闭异常
    PossibleDataLossError,  # 可能的数据丢失异常
)

from pandas import (  # 从pandas库中导入多个对象和函数
    DataFrame,  # 数据框对象
    HDFStore,  # HDF格式数据存储对象
    Index,  # 索引对象
    Series,  # 序列对象
    _testing as tm,  # 测试相关的私有模块
    date_range,  # 日期范围生成函数
    read_hdf,  # 读取HDF格式数据的函数
)
from pandas.tests.io.pytables.common import (  # 从pandas.tests.io.pytables.common模块导入多个函数和对象
    _maybe_remove,  # 可能的文件移除函数
    ensure_clean_store,  # 确保清理数据存储函数
    tables,  # pytables相关对象
)

from pandas.io import pytables  # 从pandas.io模块中导入pytables
from pandas.io.pytables import Term  # 从pandas.io.pytables模块中导入Term对象

pytestmark = pytest.mark.single_cpu  # 设置pytest的标记为单CPU执行


@pytest.mark.parametrize("mode", ["r", "r+", "a", "w"])
def test_mode(setup_path, tmp_path, mode):
    # 创建一个随机数据的DataFrame
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    msg = r"[\S]* does not exist"  # 错误信息的正则表达式
    path = tmp_path / setup_path  # 设置文件路径

    # 构造函数测试
    if mode in ["r", "r+"]:
        # 在读取模式下，使用pytest检查是否抛出预期的OSError异常
        with pytest.raises(OSError, match=msg):
            HDFStore(path, mode=mode)
    else:
        # 在其他模式下，打开HDFStore并验证模式
        with HDFStore(path, mode=mode) as store:
            assert store._handle.mode == mode

    path = tmp_path / setup_path  # 重置文件路径

    # 上下文管理器测试
    if mode in ["r", "r+"]:
        # 在读取模式下，使用pytest检查是否抛出预期的OSError异常
        with pytest.raises(OSError, match=msg):
            with HDFStore(path, mode=mode) as store:
                pass
    else:
        # 在其他模式下，打开HDFStore并验证模式
        with HDFStore(path, mode=mode) as store:
            assert store._handle.mode == mode

    path = tmp_path / setup_path  # 重置文件路径

    # 写入转换测试
    if mode in ["r", "r+"]:
        # 在读取模式下，使用pytest检查是否抛出预期的OSError异常
        with pytest.raises(OSError, match=msg):
            df.to_hdf(path, key="df", mode=mode)
        # 以"w"模式写入数据
        df.to_hdf(path, key="df", mode="w")
    else:
        # 在其他模式下，写入数据到HDF文件
        df.to_hdf(path, key="df", mode=mode)

    # 读取转换测试
    if mode in ["w"]:
        # 如果模式为"w"，则验证是否抛出预期的值错误异常
        msg = (
            "mode w is not allowed while performing a read. "
            r"Allowed modes are r, r\+ and a."
        )
        with pytest.raises(ValueError, match=msg):
            read_hdf(path, "df", mode=mode)
    else:
        # 在其他模式下，读取HDF文件并验证数据一致性
        result = read_hdf(path, "df", mode=mode)
        tm.assert_frame_equal(result, df)


def test_default_mode(tmp_path, setup_path):
    # 默认模式下的测试：使用read_hdf默认模式
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    path = tmp_path / setup_path  # 设置文件路径
    df.to_hdf(path, key="df", mode="w")  # 写入数据到HDF文件
    result = read_hdf(path, "df")  # 读取数据
    tm.assert_frame_equal(result, df)  # 验证数据一致性


def test_reopen_handle(tmp_path, setup_path):
    path = tmp_path / setup_path  # 设置文件路径

    store = HDFStore(path, mode="a")  # 打开HDFStore对象，以追加模式
    store["a"] = Series(  # 向存储中添加Series数据
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )

    msg = (
        r"Re-opening the file \[[\S]*\] with mode \[a\] will delete the "
        "current file!"
    )
    # 无效模式更改测试
    # 使用 pytest 的上下文管理器检查是否抛出 PossibleDataLossError 异常，并验证异常消息是否匹配 msg
    with pytest.raises(PossibleDataLossError, match=msg):
        # 尝试以写入模式打开 store 对象
        store.open("w")

    # 关闭 store 对象
    store.close()
    # 断言 store 对象已关闭
    assert not store.is_open

    # 在此处进行截断操作是可以接受的
    # 重新以写入模式打开 store 对象
    store.open("w")
    # 断言 store 对象已打开
    assert store.is_open
    # 断言 store 对象中的数据长度为 0
    assert len(store) == 0
    # 关闭 store 对象
    store.close()
    # 断言 store 对象已关闭
    assert not store.is_open

    # 使用 HDFStore 创建一个新的存储对象，以追加模式打开
    store = HDFStore(path, mode="a")
    # 向 store 对象中存储一个 Series 对象
    store["a"] = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )

    # 重新以读取模式打开 store 对象
    store.open("r")
    # 断言 store 对象已打开
    assert store.is_open
    # 断言 store 对象中的数据长度为 1
    assert len(store) == 1
    # 断言 store 对象的模式为 "r"
    assert store._mode == "r"
    # 关闭 store 对象
    store.close()
    # 断言 store 对象已关闭
    assert not store.is_open

    # 重新以追加模式打开 store 对象
    store.open("a")
    # 断言 store 对象已打开
    assert store.is_open
    # 断言 store 对象中的数据长度为 1
    assert len(store) == 1
    # 断言 store 对象的模式为 "a"
    assert store._mode == "a"
    # 关闭 store 对象
    store.close()
    # 断言 store 对象已关闭
    assert not store.is_open

    # 再次以追加模式打开 store 对象
    store.open("a")
    # 断言 store 对象已打开
    assert store.is_open
    # 断言 store 对象中的数据长度为 1
    assert len(store) == 1
    # 断言 store 对象的模式为 "a"
    assert store._mode == "a"
    # 关闭 store 对象
    store.close()
    # 断言 store 对象已关闭
    assert not store.is_open
# 定义一个测试函数，测试 HDFStore 打开文件的不同参数设置
def test_open_args(setup_path):
    # 使用 ensure_clean 上下文管理器，确保测试环境的清洁性，并获取文件路径
    with tm.ensure_clean(setup_path) as path:
        # 创建一个 DataFrame 对象，包含特定的数据和索引
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )

        # 创建一个内存中的 HDFStore 对象
        store = HDFStore(
            path, mode="a", driver="H5FD_CORE", driver_core_backing_store=0
        )
        # 将 DataFrame 对象写入 HDFStore 中的键 'df'
        store["df"] = df
        # 追加 DataFrame 对象到 HDFStore 中的键 'df2'
        store.append("df2", df)

        # 断言 HDFStore 中的 'df' 与原始 DataFrame df 相等
        tm.assert_frame_equal(store["df"], df)
        # 断言 HDFStore 中的 'df2' 与原始 DataFrame df 相等
        tm.assert_frame_equal(store["df2"], df)

        # 关闭 HDFStore 对象
        store.close()

    # 断言文件路径 path 实际上并未被写入文件系统
    assert not os.path.exists(path)


# 定义一个测试函数，测试 HDFStore 对象的 flush 方法
def test_flush(setup_path):
    # 使用 ensure_clean_store 上下文管理器，确保测试环境的清洁性，并获取 HDFStore 对象
    with ensure_clean_store(setup_path) as store:
        # 向 HDFStore 对象写入 Series 对象到键 'a'
        store["a"] = Series(range(5))
        # 调用 HDFStore 对象的 flush 方法，将数据刷新到文件
        store.flush()
        # 使用 fsync=True 参数调用 HDFStore 对象的 flush 方法，强制数据写入磁盘
        store.flush(fsync=True)


# 定义一个测试函数，测试 HDF 文件的默认压缩库设置
def test_complibs_default_settings(tmp_path, setup_path):
    # 创建一个 DataFrame 对象，包含特定的数据和索引
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )

    # 在临时路径下创建 HDF 文件，将 DataFrame df 写入其中，设置压缩级别为 9
    tmpfile = tmp_path / setup_path
    df.to_hdf(tmpfile, key="df", complevel=9)
    # 读取 HDF 文件中键为 'df' 的数据，与原始 DataFrame df 进行比较断言
    result = read_hdf(tmpfile, "df")
    tm.assert_frame_equal(result, df)

    # 使用 tables 打开 HDF 文件，遍历所有 Leaf 类型的节点
    with tables.open_file(tmpfile, mode="r") as h5file:
        for node in h5file.walk_nodes(where="/df", classname="Leaf"):
            # 断言节点的压缩级别为 9
            assert node.filters.complevel == 9
            # 断言节点的压缩库为 'zlib'
            assert node.filters.complib == "zlib"

    # 再次创建 HDF 文件，将 DataFrame df 写入其中，设置压缩库为 'zlib'
    tmpfile = tmp_path / setup_path
    df.to_hdf(tmpfile, key="df", complib="zlib")
    # 读取 HDF 文件中键为 'df' 的数据，与原始 DataFrame df 进行比较断言
    result = read_hdf(tmpfile, "df")
    tm.assert_frame_equal(result, df)

    # 使用 tables 打开 HDF 文件，遍历所有 Leaf 类型的节点
    with tables.open_file(tmpfile, mode="r") as h5file:
        for node in h5file.walk_nodes(where="/df", classname="Leaf"):
            # 断言节点的压缩级别为 0（即不压缩）
            assert node.filters.complevel == 0
            # 断言节点的压缩库为 None（即不使用压缩库）
            assert node.filters.complib is None

    # 再次创建 HDF 文件，将 DataFrame df 写入其中，不设置压缩库或压缩级别
    tmpfile = tmp_path / setup_path
    df.to_hdf(tmpfile, key="df")
    # 读取 HDF 文件中键为 'df' 的数据，与原始 DataFrame df 进行比较断言
    result = read_hdf(tmpfile, "df")
    tm.assert_frame_equal(result, df)

    # 使用 tables 打开 HDF 文件，遍历所有 Leaf 类型的节点
    with tables.open_file(tmpfile, mode="r") as h5file:
        for node in h5file.walk_nodes(where="/df", classname="Leaf"):
            # 断言节点的压缩级别为 0（即不压缩）
            assert node.filters.complevel == 0
            # 断言节点的压缩库为 None（即不使用压缩库）
            assert node.filters.complib is None


# 定义一个测试函数，测试 HDF 文件的默认压缩库设置能否被覆盖
def test_complibs_default_settings_override(tmp_path, setup_path):
    # 创建一个 DataFrame 对象，包含特定的数据和索引
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    # 在临时路径下创建 HDF 文件
    tmpfile = tmp_path / setup_path
    # 创建 HDFStore 对象，将 DataFrame df 写入其中，设置压缩级别为 9，压缩库为 'blosc'
    store = HDFStore(tmpfile)
    store.append("dfc", df, complevel=9, complib="blosc")
    # 将 DataFrame df 写入 HDFStore 对象，键名为 'df'
    store.append("df", df)
    # 关闭 HDFStore 对象
    store.close()
    # 使用 `tables` 模块打开 HDF5 文件 `tmpfile`，以只读模式
    with tables.open_file(tmpfile, mode="r") as h5file:
        # 遍历 HDF5 文件中路径为 "/df" 的所有 Leaf 节点
        for node in h5file.walk_nodes(where="/df", classname="Leaf"):
            # 断言当前节点的压缩级别为 0
            assert node.filters.complevel == 0
            # 断言当前节点的压缩库为 None
            assert node.filters.complib is None
    
        # 继续遍历 HDF5 文件中路径为 "/dfc" 的所有 Leaf 节点
        for node in h5file.walk_nodes(where="/dfc", classname="Leaf"):
            # 断言当前节点的压缩级别为 9
            assert node.filters.complevel == 9
            # 断言当前节点的压缩库为 "blosc"
            assert node.filters.complib == "blosc"
@pytest.mark.parametrize("lvl", range(10))
# 使用参数化标记定义 'lvl' 参数，范围是 0 到 9
@pytest.mark.parametrize("lib", tables.filters.all_complibs)
# 使用参数化标记定义 'lib' 参数，从 tables.filters.all_complibs 中取值
@pytest.mark.filterwarnings("ignore:object name is not a valid")
# 忽略特定警告信息，该警告消息包含字符串 "object name is not a valid"
@pytest.mark.skipif(
    not PY311 and is_ci_environment() and is_platform_linux(),
    reason="Segfaulting in a CI environment",
    # 在 CI 环境中遇到段错误时，跳过测试，因为可能会导致 UnicodeDecodeError
)
def test_complibs(tmp_path, lvl, lib, request):
    # GH14478
    # 如果满足条件：PY311 并且在 Linux 平台且 lib 是 "blosc2" 且 lvl 不等于 0
    if PY311 and is_platform_linux() and lib == "blosc2" and lvl != 0:
        # 应用 xfail 标记，标记为预期失败，并提供失败的原因
        request.applymarker(
            pytest.mark.xfail(reason=f"Fails for {lib} on Linux and PY > 3.11")
        )
    # 创建一个 DataFrame 对象，包含 30 行和 4 列，所有值初始化为 1
    df = DataFrame(
        np.ones((30, 4)), columns=list("ABCD"), index=np.arange(30).astype(np.str_)
    )

    # 如果平台不支持 lzo 压缩库，跳过测试
    if not tables.which_lib_version("lzo"):
        pytest.skip("lzo not available")
    # 如果平台不支持 bzip2 压缩库，跳过测试
    if not tables.which_lib_version("bzip2"):
        pytest.skip("bzip2 not available")

    # 在临时路径下创建一个 HDF5 文件
    tmpfile = tmp_path / f"{lvl}_{lib}.h5"
    # 定义组名
    gname = f"{lvl}_{lib}"

    # 将 DataFrame 对象写入 HDF5 文件，并指定压缩参数
    df.to_hdf(tmpfile, key=gname, complib=lib, complevel=lvl)
    # 读取 HDF5 文件中的数据，返回结果
    result = read_hdf(tmpfile, gname)
    # 断言读取结果与原始 DataFrame 对象相等
    tm.assert_frame_equal(result, df)

    # 判断当前是否为 macOS 系统
    is_mac = is_platform_mac()

    # 打开 HDF5 文件，检查元数据以确保压缩参数设置正确
    with tables.open_file(tmpfile, mode="r") as h5table:
        # 遍历 HDF5 文件中指定路径下的所有 Leaf 节点
        for node in h5table.walk_nodes(where="/" + gname, classname="Leaf"):
            # 断言节点的压缩级别与预期一致
            assert node.filters.complevel == lvl
            if lvl == 0:
                # 如果压缩级别为 0，断言压缩库为 None
                assert node.filters.complib is None
            elif is_mac and lib == "blosc2":
                # 如果是 macOS 系统且使用的是 "blosc2" 压缩库，检查实际使用的压缩库
                res = node.filters.complib
                assert res in [lib, "blosc2:blosclz"], res
            else:
                # 其他情况下，断言压缩库与预期一致
                assert node.filters.complib == lib


@pytest.mark.skipif(
    not is_platform_little_endian(), reason="reason platform is not little endian"
)
# 如果当前平台不是小端字节序，跳过测试
def test_encoding(setup_path):
    # 在确保存储区域为空的情况下，创建 DataFrame 对象并写入存储
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含 5 行数据的 DataFrame 对象，包含两列 "A" 和 "B"
        df = DataFrame({"A": "foo", "B": "bar"}, index=range(5))
        # 修改 DataFrame 中的部分数据为 NaN
        df.loc[2, "A"] = np.nan
        df.loc[3, "B"] = np.nan
        # 如果存在名为 "df" 的数据对象，删除它
        _maybe_remove(store, "df")
        # 将 DataFrame 对象追加到存储中，指定编码为 "ascii"
        store.append("df", df, encoding="ascii")
        # 断言存储中的数据与原始 DataFrame 对象相等
        tm.assert_frame_equal(store["df"], df)

        # 期望的结果是仅包含列 "A" 的 DataFrame 对象
        expected = df.reindex(columns=["A"])
        # 从存储中选择数据，仅选择编码为 "ascii" 的列 "A"
        result = store.select("df", Term("columns=A", encoding="ascii"))
        # 断言选择结果与期望的结果相等
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "val",
    [
        [b"E\xc9, 17", b"", b"a", b"b", b"c"],
        [b"E\xc9, 17", b"a", b"b", b"c"],
        [b"EE, 17", b"", b"a", b"b", b"c"],
        [b"E\xc9, 17", b"\xf8\xfc", b"a", b"b", b"c"],
        [b"", b"a", b"b", b"c"],
        [b"\xf8\xfc", b"a", b"b", b"c"],
        [b"A\xf8\xfc", b"", b"a", b"b", b"c"],
        [np.nan, b"", b"b", b"c"],
        [b"A\xf8\xfc", np.nan, b"", b"b", b"c"],
    ],
)
# 使用参数化标记定义 'val' 参数，对应多个测试用例
@pytest.mark.parametrize("dtype", ["category", object])
# 使用参数化标记定义 'dtype' 参数，对应多种数据类型
# 定义一个测试函数，用于测试使用 Latin-1 编码的情况下的数据处理和存储
def test_latin_encoding(tmp_path, setup_path, dtype, val):
    # 设置编码格式为 Latin-1
    enc = "latin-1"
    # 定义空值替代符号
    nan_rep = ""
    # 定义存储的关键字
    key = "data"

    # 如果数据类型是 bytes，则将其解码为字符串，否则保持不变
    val = [x.decode(enc) if isinstance(x, bytes) else x for x in val]
    # 创建一个 Series 对象
    ser = Series(val, dtype=dtype)

    # 设置存储路径
    store = tmp_path / setup_path
    # 将 Series 对象写入 HDF 文件中
    ser.to_hdf(store, key=key, format="table", encoding=enc, nan_rep=nan_rep)
    # 从 HDF 文件中读取数据
    retr = read_hdf(store, key)

    # 如果数据类型是 category，则处理空值的情况
    if dtype == "category":
        if nan_rep in ser.cat.categories:
            # 移除包含空值替代符号的类别
            s_nan = ser.cat.remove_categories([nan_rep])
        else:
            # 否则保持不变
            s_nan = ser
    else:
        # 将空值替代符号替换为 NaN
        s_nan = ser.replace(nan_rep, np.nan)

    # 断言处理后的 Series 和从 HDF 文件读取的数据是否相等
    tm.assert_series_equal(s_nan, retr)


# 定义一个测试函数，用于测试多次打开和关闭 HDFStore 的情况
def test_multiple_open_close(tmp_path, setup_path):
    # 设置文件路径
    path = tmp_path / setup_path

    # 创建一个 DataFrame 对象
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    # 将 DataFrame 对象写入 HDF 文件中
    df.to_hdf(path, key="df", mode="w", format="table")

    # 单次打开和关闭操作
    store = HDFStore(path)
    assert "CLOSED" not in store.info()
    assert store.is_open

    store.close()
    assert "CLOSED" in store.info()
    assert not store.is_open

    # 重新设置文件路径
    path = tmp_path / setup_path

    # 如果 PyTables 的文件打开策略是 strict
    if pytables._table_file_open_policy_is_strict:
        # 多次打开和关闭操作，应该会抛出 ValueError 异常
        store1 = HDFStore(path)
        msg = (
            r"The file [\S]* is already opened\.  Please close it before "
            r"reopening in write mode\."
        )
        with pytest.raises(ValueError, match=msg):
            HDFStore(path)

        store1.close()
    else:
        # 多次打开和关闭操作
        store1 = HDFStore(path)
        store2 = HDFStore(path)

        assert "CLOSED" not in store1.info()
        assert "CLOSED" not in store2.info()
        assert store1.is_open
        assert store2.is_open

        store1.close()
        assert "CLOSED" in store1.info()
        assert not store1.is_open
        assert "CLOSED" not in store2.info()
        assert store2.is_open

        store2.close()
        assert "CLOSED" in store1.info()
        assert "CLOSED" in store2.info()
        assert not store1.is_open
        assert not store2.is_open

        # 嵌套关闭操作
        store = HDFStore(path, mode="w")
        store.append("df", df)

        store2 = HDFStore(path)
        store2.append("df2", df)
        store2.close()
        assert "CLOSED" in store2.info()
        assert not store2.is_open

        store.close()
        assert "CLOSED" in store.info()
        assert not store.is_open

        # 双重关闭操作
        store = HDFStore(path, mode="w")
        store.append("df", df)

        store2 = HDFStore(path)
        store.close()
        assert "CLOSED" in store.info()
        assert not store.is_open

        store2.close()
        assert "CLOSED" in store2.info()
        assert not store2.is_open
    # 创建临时路径，用于存储 HDF 文件
    path = tmp_path / setup_path

    # 创建一个包含特定数据的 DataFrame 对象
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    # 将 DataFrame 对象写入 HDF 文件
    df.to_hdf(path, key="df", mode="w", format="table")

    # 打开创建的 HDF 文件并获取 HDFStore 对象
    store = HDFStore(path)
    # 关闭 HDFStore 对象
    store.close()

    # 定义用于匹配异常消息的正则表达式
    msg = r"[\S]* file is not open!"
    # 测试在 HDFStore 对象关闭后访问属性会抛出 ClosedFileError 异常
    with pytest.raises(ClosedFileError, match=msg):
        store.keys()

    with pytest.raises(ClosedFileError, match=msg):
        "df" in store

    with pytest.raises(ClosedFileError, match=msg):
        len(store)

    with pytest.raises(ClosedFileError, match=msg):
        store["df"]

    with pytest.raises(ClosedFileError, match=msg):
        store.select("df")

    with pytest.raises(ClosedFileError, match=msg):
        store.get("df")

    with pytest.raises(ClosedFileError, match=msg):
        store.append("df2", df)

    with pytest.raises(ClosedFileError, match=msg):
        store.put("df3", df)

    with pytest.raises(ClosedFileError, match=msg):
        store.get_storer("df2")

    with pytest.raises(ClosedFileError, match=msg):
        store.remove("df2")

    with pytest.raises(ClosedFileError, match=msg):
        store.select("df")

    # 测试在 HDFStore 对象关闭后访问不存在的属性会抛出 AttributeError 异常
    msg = "'HDFStore' object has no attribute 'df'"
    with pytest.raises(AttributeError, match=msg):
        store.df
# 定义一个函数 test_fspath，用于测试文件路径处理
def test_fspath():
    # 使用 tm.ensure_clean 上下文管理器创建并确保 "foo.h5" 文件不存在
    with tm.ensure_clean("foo.h5") as path:
        # 使用 HDFStore 打开指定路径的 HDF 文件，并使用 store 作为其上下文管理对象
        with HDFStore(path) as store:
            # 断言 HDFStore 对象 store 的文件路径等效于 str(path)
            assert os.fspath(store) == str(path)
```