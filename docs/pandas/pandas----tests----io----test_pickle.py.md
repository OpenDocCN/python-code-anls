# `D:\src\scipysrc\pandas\pandas\tests\io\test_pickle.py`

```
"""
管理旧版 pickle 测试

如何添加 pickle 测试:

1. 安装生成 pickle 文件所需的 pandas 版本。

2. 执行 "generate_legacy_storage_files.py" 以创建 pickle 文件。
$ python generate_legacy_storage_files.py <output_dir> pickle

3. 将生成的 pickle 移动到 "data/legacy_pickle/<version>" 目录下。
"""

# 导入必要的模块和库
from __future__ import annotations

import bz2
import datetime
import functools
from functools import partial
import gzip
import io
import os
from pathlib import Path
import pickle
import shutil
import tarfile
from typing import Any
import uuid
import zipfile

import numpy as np
import pytest

# 导入 pandas 库及其组件
import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    period_range,
)
import pandas._testing as tm
from pandas.tests.io.generate_legacy_storage_files import create_pickle_data

import pandas.io.common as icom
from pandas.tseries.offsets import (
    Day,
    MonthEnd,
)

# ---------------------
# 比较函数
# ---------------------
def compare_element(result, expected, typ):
    # 如果期望值是索引类型，则比较索引是否相等
    if isinstance(expected, Index):
        tm.assert_index_equal(expected, result)
        return

    # 根据类型选择相应的比较方式
    if typ.startswith("sp_"):
        tm.assert_equal(result, expected)
    elif typ == "timestamp":
        if expected is pd.NaT:
            assert result is pd.NaT
        else:
            assert result == expected
    else:
        comparator = getattr(tm, f"assert_{typ}_equal", tm.assert_almost_equal)
        comparator(result, expected)


# ---------------------
# 测试
# ---------------------

def test_pickles(datapath):
    # 跳过测试，如果缺少 pytz 库
    pytest.importorskip("pytz")
    # 如果不是小端平台，则跳过测试
    if not is_platform_little_endian():
        pytest.skip("known failure on non-little endian")

    # 兼容 --strict-data-files 的 for 循环
    # 遍历指定路径下所有符合通配符"data/legacy_pickle/*/*.p*kl*"的遗留pickle文件
    for legacy_pickle in Path(__file__).parent.glob("data/legacy_pickle/*/*.p*kl*"):
        # 将遗留pickle文件的路径转换为标准化路径
        legacy_pickle = datapath(legacy_pickle)

        # 从pickle文件中读取数据
        data = pd.read_pickle(legacy_pickle)

        # 遍历数据字典中的每个项目类型(typ)和数据值(dv)
        for typ, dv in data.items():
            # 遍历每个数据值(dv)中的日期类型(dt)和对应的结果(result)
            for dt, result in dv.items():
                # 获取预期的结果(expected)
                expected = data[typ][dt]

                # 如果类型为"series"并且日期为"ts"
                if typ == "series" and dt == "ts":
                    # 进行系列数据的相等性断言
                    # GH 7748
                    tm.assert_series_equal(result, expected)
                    # 断言结果的索引频率相等
                    assert result.index.freq == expected.index.freq
                    # 断言结果的索引频率未经过标准化
                    assert not result.index.freq.normalize
                    # 对结果进行大于零的系列断言
                    tm.assert_series_equal(result > 0, expected > 0)

                    # GH 9291
                    # 检查结果的索引频率
                    freq = result.index.freq
                    assert freq + Day(1) == Day(2)

                    # 计算频率加上一天的结果
                    res = freq + pd.Timedelta(hours=1)
                    # 断言结果是一个时间增量对象
                    assert isinstance(res, pd.Timedelta)
                    # 断言时间增量是一天加一小时
                    assert res == pd.Timedelta(days=1, hours=1)

                    # 计算频率加上一纳秒的结果
                    res = freq + pd.Timedelta(nanoseconds=1)
                    # 断言结果是一个时间增量对象
                    assert isinstance(res, pd.Timedelta)
                    # 断言时间增量是一天加一纳秒
                    assert res == pd.Timedelta(days=1, nanoseconds=1)
                
                # 如果类型为"index"并且日期为"period"
                elif typ == "index" and dt == "period":
                    # 进行索引数据的相等性断言
                    tm.assert_index_equal(result, expected)
                    # 断言结果的频率是月末
                    assert isinstance(result.freq, MonthEnd)
                    assert result.freq == MonthEnd()
                    # 断言结果的频率字符串为"M"
                    assert result.freqstr == "M"
                    # 断言结果向前移动两步后的索引与预期结果向前移动两步后的索引相等
                    tm.assert_index_equal(result.shift(2), expected.shift(2))
                
                # 如果类型为"series"并且日期为"dt_tz"或"cat"
                elif typ == "series" and dt in ("dt_tz", "cat"):
                    # 进行系列数据的相等性断言
                    tm.assert_series_equal(result, expected)
                
                # 如果类型为"frame"并且日期为"dt_mixed_tzs"、"cat_onecol"或"cat_and_float"
                elif typ == "frame" and dt in (
                    "dt_mixed_tzs",
                    "cat_onecol",
                    "cat_and_float",
                ):
                    # 进行数据框的相等性断言
                    tm.assert_frame_equal(result, expected)
                
                # 对于其它未列出的情况，调用比较函数compare_element进行比较
                else:
                    compare_element(result, expected, typ)
# 将对象序列化并保存到指定路径的文件中，使用最高兼容性协议（protocol=-1）
def python_pickler(obj, path):
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=-1)


# 从指定路径的文件中反序列化对象
def python_unpickler(path):
    with open(path, "rb") as fh:
        fh.seek(0)
        return pickle.load(fh)


# 将嵌套字典数据扁平化为元组列表
def flatten(data: dict) -> list[tuple[str, Any]]:
    """Flatten create_pickle_data"""
    return [
        (typ, example)
        for typ, examples in data.items()
        for example in examples.values()
    ]


# 参数化测试，使用不同的 pickler 函数和协议进行序列化
@pytest.mark.parametrize(
    "pickle_writer",
    [
        pytest.param(python_pickler, id="python"),
        pytest.param(pd.to_pickle, id="pandas_proto_default"),
        pytest.param(
            functools.partial(pd.to_pickle, protocol=pickle.HIGHEST_PROTOCOL),
            id="pandas_proto_highest",
        ),
        pytest.param(functools.partial(pd.to_pickle, protocol=4), id="pandas_proto_4"),
        pytest.param(
            functools.partial(pd.to_pickle, protocol=5),
            id="pandas_proto_5",
        ),
    ],
)
@pytest.mark.parametrize("writer", [pd.to_pickle, python_pickler])
@pytest.mark.parametrize("typ, expected", flatten(create_pickle_data()))
def test_round_trip_current(typ, expected, pickle_writer, writer):
    with tm.ensure_clean() as path:
        # 使用每个 pickler 函数进行序列化并写入文件
        pickle_writer(expected, path)

        # 使用 pandas 的 read_pickle 函数读取文件，并进行比较
        result = pd.read_pickle(path)
        compare_element(result, expected, typ)

        # 使用自定义的 python_unpickler 函数读取文件，并进行比较
        result = python_unpickler(path)
        compare_element(result, expected, typ)

        # 测试文件对象的情况（GH 35679）
        with open(path, mode="wb") as handle:
            writer(expected, path)
            handle.seek(0)  # 确保不关闭文件句柄
        with open(path, mode="rb") as handle:
            result = pd.read_pickle(handle)
            handle.seek(0)  # 确保不关闭文件句柄
        compare_element(result, expected, typ)


# 测试使用 pathlib 路径的 pickle 和 unpickle 操作
def test_pickle_path_pathlib():
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    result = tm.round_trip_pathlib(df.to_pickle, pd.read_pickle)
    tm.assert_frame_equal(df, result)


# ---------------------
# 测试 pickle 压缩
# ---------------------


# 生成一个随机的 pickle 文件路径的 fixture
@pytest.fixture
def get_random_path():
    return f"__{uuid.uuid4()}__.pickle"


# 测试 pickle 压缩相关功能的类
class TestCompression:
    _extension_to_compression = icom.extension_to_compression
    # 定义一个方法用于压缩文件，根据指定的压缩算法将源文件压缩到目标路径
    def compress_file(self, src_path, dest_path, compression):
        # 如果未指定压缩算法，则直接复制文件到目标路径
        if compression is None:
            shutil.copyfile(src_path, dest_path)
            return

        # 根据指定的压缩算法执行相应的压缩操作
        if compression == "gzip":
            f = gzip.open(dest_path, "w")
        elif compression == "bz2":
            f = bz2.BZ2File(dest_path, "w")
        elif compression == "zip":
            # 使用 ZIP 压缩算法创建一个压缩文件对象
            with zipfile.ZipFile(dest_path, "w", compression=zipfile.ZIP_DEFLATED) as f:
                # 将源文件添加到压缩文件中，使用其基本名称作为文件名
                f.write(src_path, os.path.basename(src_path))
        elif compression == "tar":
            # 使用 TAR 压缩算法打开目标路径，准备写入源文件内容
            with open(src_path, "rb") as fh:
                with tarfile.open(dest_path, mode="w") as tar:
                    # 获取源文件的 TAR 信息，并将其添加到 TAR 文件中
                    tarinfo = tar.gettarinfo(src_path, os.path.basename(src_path))
                    tar.addfile(tarinfo, fh)
        elif compression == "xz":
            # 使用 XZ 压缩算法创建一个压缩文件对象
            import lzma
            f = lzma.LZMAFile(dest_path, "w")
        elif compression == "zstd":
            # 使用 Zstandard 压缩算法打开目标路径，准备写入二进制数据
            f = import_optional_dependency("zstandard").open(dest_path, "wb")
        else:
            # 如果压缩类型未识别，则抛出 ValueError 异常
            msg = f"Unrecognized compression type: {compression}"
            raise ValueError(msg)

        # 对于非 ZIP 和 TAR 压缩类型，将源文件内容写入压缩文件中
        if compression not in ["zip", "tar"]:
            with open(src_path, "rb") as fh:
                with f:
                    f.write(fh.read())
    def test_write_infer(self, compression_ext, get_random_path):
        base = get_random_path
        path1 = base + compression_ext
        path2 = base + ".raw"
        compression = self._extension_to_compression.get(compression_ext.lower())

        # 使用 tm.ensure_clean 上下文管理器清理文件路径 path1 和 path2
        with tm.ensure_clean(path1) as p1, tm.ensure_clean(path2) as p2:
            # 创建一个 DataFrame 对象 df
            df = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=Index([f"i-{i}" for i in range(30)], dtype=object),
            )

            # 将 DataFrame df 写入为经推断的压缩格式文件 p1
            df.to_pickle(p1)

            # 解压缩文件 p1
            with tm.decompress_file(p1, compression=compression) as f:
                # 将解压缩后的内容写入文件 p2
                with open(p2, "wb") as fh:
                    fh.write(f.read())

            # 从解压缩后的文件 p2 中读取数据到 DataFrame df2
            df2 = pd.read_pickle(p2, compression=None)

            # 断言 df 和 df2 相等
            tm.assert_frame_equal(df, df2)

    def test_read_explicit(self, compression, get_random_path):
        base = get_random_path
        path1 = base + ".raw"
        path2 = base + ".compressed"

        # 使用 tm.ensure_clean 上下文管理器清理文件路径 path1 和 path2
        with tm.ensure_clean(path1) as p1, tm.ensure_clean(path2) as p2:
            # 创建一个 DataFrame 对象 df
            df = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=Index([f"i-{i}" for i in range(30)], dtype=object),
            )

            # 将 DataFrame df 写入为未压缩格式文件 p1
            df.to_pickle(p1, compression=None)

            # 压缩文件 p1
            self.compress_file(p1, p2, compression=compression)

            # 从压缩后的文件 p2 中读取数据到 DataFrame df2
            df2 = pd.read_pickle(p2, compression=compression)

            # 断言 df 和 df2 相等
            tm.assert_frame_equal(df, df2)

    def test_read_infer(self, compression_ext, get_random_path):
        base = get_random_path
        path1 = base + ".raw"
        path2 = base + compression_ext
        compression = self._extension_to_compression.get(compression_ext.lower())

        # 使用 tm.ensure_clean 上下文管理器清理文件路径 path1 和 path2
        with tm.ensure_clean(path1) as p1, tm.ensure_clean(path2) as p2:
            # 创建一个 DataFrame 对象 df
            df = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=Index([f"i-{i}" for i in range(30)], dtype=object),
            )

            # 将 DataFrame df 写入为未压缩格式文件 p1
            df.to_pickle(p1, compression=None)

            # 压缩文件 p1
            self.compress_file(p1, p2, compression=compression)

            # 从经推断的压缩格式文件 p2 中读取数据到 DataFrame df2
            df2 = pd.read_pickle(p2)

            # 断言 df 和 df2 相等
            tm.assert_frame_equal(df, df2)
# ---------------------
# test pickle compression
# ---------------------

# 定义一个测试类 TestProtocol
class TestProtocol:
    
    # 使用 pytest 的参数化装饰器，参数为协议版本号 -1, 0, 1, 2
    @pytest.mark.parametrize("protocol", [-1, 0, 1, 2])
    # 定义一个测试方法 test_read，接受协议版本号和一个随机路径
    def test_read(self, protocol, get_random_path):
        # 使用 ensure_clean 上下文管理器，确保路径干净
        with tm.ensure_clean(get_random_path) as path:
            # 创建一个 DataFrame 对象 df，包含 30 行 4 列的数据
            df = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=Index([f"i-{i}" for i in range(30)], dtype=object),
            )
            # 将 df 对象保存为 pickle 文件到指定路径，使用给定的协议版本
            df.to_pickle(path, protocol=protocol)
            # 从 pickle 文件中读取数据，保存到 df2
            df2 = pd.read_pickle(path)
            # 断言 df 和 df2 相等
            tm.assert_frame_equal(df, df2)


# 定义一个测试方法 test_pickle_buffer_roundtrip
def test_pickle_buffer_roundtrip():
    # 使用 ensure_clean 上下文管理器，确保路径干净
    with tm.ensure_clean() as path:
        # 创建一个 DataFrame 对象 df，包含 30 行 4 列的数据
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        # 将 df 对象保存为 pickle 格式到文件对象 fh 中
        with open(path, "wb") as fh:
            df.to_pickle(fh)
        # 从文件对象 fh 中读取 pickle 数据，保存到 result
        with open(path, "rb") as fh:
            result = pd.read_pickle(fh)
        # 断言 df 和 result 相等
        tm.assert_frame_equal(df, result)


# ---------------------
# tests for URL I/O
# ---------------------

# 使用 pytest 的参数化装饰器，mockurl 参数为 ["http://url.com", "ftp://test.com", "http://gzip.com"]
@pytest.mark.parametrize(
    "mockurl", ["http://url.com", "ftp://test.com", "http://gzip.com"]
)
# 定义一个测试方法 test_pickle_generalurl_read，接受 monkeypatch 和 mockurl 参数
def test_pickle_generalurl_read(monkeypatch, mockurl):
    
    # 定义一个函数 python_pickler，用于将对象序列化为 pickle 格式并保存到指定路径
    def python_pickler(obj, path):
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=-1)

    # 定义一个 MockReadResponse 类，模拟读取 response 数据
    class MockReadResponse:
        def __init__(self, path) -> None:
            self.file = open(path, "rb")
            # 根据路径内容判断是否为 gzip 压缩文件，设置相应的 headers
            if "gzip" in path:
                self.headers = {"Content-Encoding": "gzip"}
            else:
                self.headers = {"Content-Encoding": ""}

        # 进入上下文管理器，返回自身对象
        def __enter__(self):
            return self

        # 退出上下文管理器时关闭文件
        def __exit__(self, *args):
            self.close()

        # 读取文件内容
        def read(self):
            return self.file.read()

        # 关闭文件
        def close(self):
            return self.file.close()

    # 使用 ensure_clean 上下文管理器，确保路径干净
    with tm.ensure_clean() as path:
        
        # 定义一个 mock_urlopen_read 函数，模拟 urllib.request.urlopen 方法读取数据
        def mock_urlopen_read(*args, **kwargs):
            return MockReadResponse(path)

        # 创建一个 DataFrame 对象 df，包含 30 行 4 列的数据
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        # 使用 python_pickler 函数将 df 对象保存为 pickle 格式到指定路径
        python_pickler(df, path)
        # 使用 monkeypatch 设置 urllib.request.urlopen 方法为 mock_urlopen_read 函数
        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen_read)
        # 从 mockurl 中读取 pickle 数据，保存到 result
        result = pd.read_pickle(mockurl)
        # 断言 df 和 result 相等
        tm.assert_frame_equal(df, result)


# 定义一个测试方法 test_pickle_fsspec_roundtrip
def test_pickle_fsspec_roundtrip():
    # 导入 fsspec 模块，如果不存在则跳过测试
    pytest.importorskip("fsspec")
    # 使用 ensure_clean 上下文管理器，确保路径干净
    with tm.ensure_clean():
        # 定义一个 mockurl，用于测试内存中的 mockfile
        mockurl = "memory://mockfile"
        # 创建一个 DataFrame 对象 df，包含 30 行 4 列的数据
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        # 将 df 对象保存为 pickle 格式到 mockurl
        df.to_pickle(mockurl)
        # 从 mockurl 中读取 pickle 数据，保存到 result
        result = pd.read_pickle(mockurl)
        # 断言 df 和 result 相等
        tm.assert_frame_equal(df, result)


# 定义一个空的类 MyTz，继承自 datetime.tzinfo
class MyTz(datetime.tzinfo):
    pass
    # 定义一个构造函数 __init__，该函数是类的特殊方法，用于初始化对象
    def __init__(self) -> None:
        # pass 是 Python 中的占位符语句，表示什么都不做，保证函数的语法正确性
        pass
def test_read_pickle_with_subclass():
    # GH 12163
    # 定义期望值为包含对象类型数据的 Series 和 MyTz 实例
    expected = Series(dtype=object), MyTz()
    # 对期望值进行 pickle 往返操作，返回结果
    result = tm.round_trip_pickle(expected)

    # 断言往返后的 Series 相等
    tm.assert_series_equal(result[0], expected[0])
    # 断言 result[1] 是 MyTz 类型的实例
    assert isinstance(result[1], MyTz)


def test_pickle_binary_object_compression(compression):
    """
    Read/write from binary file-objects w/wo compression.

    GH 26237, GH 29054, and GH 29570
    """
    # 创建一个 DataFrame 对象
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )

    # 用于压缩的参考数据
    with tm.ensure_clean() as path:
        # 将 DataFrame 对象以指定的压缩方式写入到文件
        df.to_pickle(path, compression=compression)
        # 读取文件的字节内容作为参考
        reference = Path(path).read_bytes()

    # 写入到内存缓冲区
    buffer = io.BytesIO()
    df.to_pickle(buffer, compression=compression)
    buffer.seek(0)

    # 对比内存缓冲区中的内容和之前的参考内容是否相同，或者压缩方式为 gzip、zip 或 tar
    assert buffer.getvalue() == reference or compression in ("gzip", "zip", "tar")

    # 从缓冲区中读取数据
    read_df = pd.read_pickle(buffer, compression=compression)
    buffer.seek(0)
    # 断言读取的 DataFrame 和原始的 DataFrame 相等
    tm.assert_frame_equal(df, read_df)


def test_pickle_dataframe_with_multilevel_index(
    multiindex_year_month_day_dataframe_random_data,
    multiindex_dataframe_random_data,
):
    # 获取多级索引的 DataFrame 对象
    ymd = multiindex_year_month_day_dataframe_random_data
    frame = multiindex_dataframe_random_data

    def _test_roundtrip(frame):
        # 进行 pickle 往返操作，并断言结果与原始的 DataFrame 相等
        unpickled = tm.round_trip_pickle(frame)
        tm.assert_frame_equal(frame, unpickled)

    # 分别对 DataFrame 和其转置进行往返测试
    _test_roundtrip(frame)
    _test_roundtrip(frame.T)
    # 对多级索引 DataFrame 进行往返测试
    _test_roundtrip(ymd)
    _test_roundtrip(ymd.T)


def test_pickle_timeseries_periodindex():
    # GH#2891
    # 创建一个周期索引的时间序列
    prng = period_range("1/1/2011", "1/1/2012", freq="M")
    ts = Series(np.random.default_rng(2).standard_normal(len(prng)), prng)
    # 对时间序列进行 pickle 往返操作
    new_ts = tm.round_trip_pickle(ts)
    # 断言新的时间序列的频率是 "M"
    assert new_ts.index.freqstr == "M"


@pytest.mark.parametrize(
    "name", [777, 777.0, "name", datetime.datetime(2001, 11, 11), (1, 2)]
)
def test_pickle_preserve_name(name):
    # 对具有不同名称的 Series 对象进行 pickle 往返操作，并断言名称得到保留
    unpickled = tm.round_trip_pickle(Series(np.arange(10, dtype=np.float64), name=name))
    assert unpickled.name == name


def test_pickle_datetimes(datetime_series):
    # 对日期时间序列进行 pickle 往返操作，并断言结果相等
    unp_ts = tm.round_trip_pickle(datetime_series)
    tm.assert_series_equal(unp_ts, datetime_series)


def test_pickle_strings(string_series):
    # 对字符串序列进行 pickle 往返操作，并断言结果相等
    unp_series = tm.round_trip_pickle(string_series)
    tm.assert_series_equal(unp_series, string_series)


def test_pickle_preserves_block_ndim():
    # GH#37631
    # 创建一个分类类型的 Series 对象
    ser = Series(list("abc")).astype("category").iloc[[0]]
    # 对其进行 pickle 往返操作
    res = tm.round_trip_pickle(ser)

    # 断言 pickle 往返后的对象的维度为 1
    assert res._mgr.blocks[0].ndim == 1
    # 断言 pickle 往返后的对象的形状为 (1,)
    assert res._mgr.blocks[0].shape == (1,)

    # GH#37631 的问题与索引有关，但核心问题是 pickle
    # 断言对 res[[True]] 的操作与原始的 Series 对象相等
    tm.assert_series_equal(res[[True]], ser)


@pytest.mark.parametrize("protocol", [pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL])
def test_pickle_big_dataframe_compression(protocol, compression):
    # 参数化测试，测试不同协议下以及不同压缩方式的大型 DataFrame 的 pickle 操作
    # GH#39002
    # 创建一个包含100000个整数的DataFrame对象
    df = DataFrame(range(100000))
    # 使用round_trip_pathlib函数测试路径操作的往返正确性，
    # 使用df.to_pickle方法部分应用protocol和compression参数来序列化DataFrame，
    # 使用pd.read_pickle方法部分应用compression参数来反序列化数据
    result = tm.round_trip_pathlib(
        partial(df.to_pickle, protocol=protocol, compression=compression),
        partial(pd.read_pickle, compression=compression),
    )
    # 断言序列化前后的DataFrame对象和反序列化后的结果DataFrame对象相等
    tm.assert_frame_equal(df, result)
# 定义函数 test_pickle_frame_v124_unpickle_130，用于测试从旧版本（1.2.x）反序列化数据框在新版本（1.3.x）中的行为
def test_pickle_frame_v124_unpickle_130(datapath):
    # GH#42345 DataFrame created in 1.2.x, unpickle in 1.3.x
    # 构造文件路径，使用 datapath 函数获取数据路径，从指定的 pickle 文件中加载数据
    path = datapath(
        Path(__file__).parent,  # 当前文件的父目录路径
        "data",                  # 数据目录
        "legacy_pickle",         # 存放旧版本 pickle 文件的目录
        "1.2.4",                 # 版本号 1.2.4
        "empty_frame_v1_2_4-GH#42345.pkl",  # 文件名，包含特定标识 GH#42345
    )
    # 打开指定路径的 pickle 文件，读取其中的 DataFrame 对象
    with open(path, "rb") as fd:
        df = pickle.load(fd)

    # 期望的 DataFrame 是一个空的数据框，即只有索引和列名，但没有任何数据
    expected = DataFrame(index=[], columns=[])
    # 使用测试框架中的 assert_frame_equal 方法来比较 df 和 expected 是否相同
    tm.assert_frame_equal(df, expected)
```