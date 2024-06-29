# `D:\src\scipysrc\pandas\pandas\tests\io\test_http_headers.py`

```
"""
Tests for the pandas custom headers in http(s) requests
"""

# 导入所需的模块和库
from functools import partial  # 导入partial函数，用于创建带有预设参数的可调用对象
import gzip  # 导入gzip模块，用于gzip压缩和解压缩
from io import BytesIO  # 导入BytesIO类，用于在内存中操作二进制数据流

import pytest  # 导入pytest测试框架

import pandas.util._test_decorators as td  # 导入测试装饰器相关的模块

import pandas as pd  # 导入pandas库，并使用pd作为别名
import pandas._testing as tm  # 导入pandas测试相关的模块

# 定义pytest标记
pytestmark = [
    pytest.mark.single_cpu,  # 单CPU环境测试标记
    pytest.mark.network,  # 需要网络连接的测试标记
    pytest.mark.filterwarnings(  # 过滤警告信息的标记
        "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
    ),
]


# 定义一个函数，将字节数据gzip压缩后返回
def gzip_bytes(response_bytes):
    with BytesIO() as bio:  # 使用BytesIO创建内存中的二进制流
        with gzip.GzipFile(fileobj=bio, mode="w") as zipper:  # 创建gzip压缩文件对象
            zipper.write(response_bytes)  # 将数据写入gzip压缩文件对象
        return bio.getvalue()  # 返回压缩后的二进制数据


# 定义一个函数，将DataFrame转换为CSV格式的字节数据
def csv_responder(df):
    return df.to_csv(index=False).encode("utf-8")  # 将DataFrame转换为CSV格式并编码为utf-8


# 定义一个函数，将DataFrame转换为gzip压缩的CSV格式字节数据
def gz_csv_responder(df):
    return gzip_bytes(csv_responder(df))  # 使用gzip压缩字节数据


# 定义一个函数，将DataFrame转换为JSON格式的字节数据
def json_responder(df):
    return df.to_json().encode("utf-8")  # 将DataFrame转换为JSON格式并编码为utf-8


# 定义一个函数，将DataFrame转换为gzip压缩的JSON格式字节数据
def gz_json_responder(df):
    return gzip_bytes(json_responder(df))  # 使用gzip压缩字节数据


# 定义一个函数，将DataFrame转换为HTML格式的字节数据
def html_responder(df):
    return df.to_html(index=False).encode("utf-8")  # 将DataFrame转换为HTML格式并编码为utf-8


# 定义一个函数，将DataFrame转换为parquet格式数据（使用pyarrow引擎）
def parquetpyarrow_reponder(df):
    return df.to_parquet(index=False, engine="pyarrow")  # 将DataFrame转换为parquet格式


# 定义一个函数，将DataFrame转换为parquet格式数据（使用fastparquet引擎）
def parquetfastparquet_responder(df):
    # fastparquet引擎不喜欢直接写入到缓冲区
    # 可以通过适当设置open_with函数来实现
    # 然而它会自动调用close方法并清空缓冲区
    # 因此在这个实例中，只需覆盖该属性即可
    # 在相关测试中通过importorskip保护
    import fsspec  # 导入fsspec模块，用于文件系统相关操作

    df.to_parquet(
        "memory://fastparquet_user_agent.parquet",  # 定义parquet文件存储路径
        index=False,  # 不保存索引
        engine="fastparquet",  # 使用fastparquet引擎
        compression=None,  # 不压缩
    )
    with fsspec.open("memory://fastparquet_user_agent.parquet", "rb") as f:  # 使用fsspec打开parquet文件
        return f.read()  # 返回文件内容的字节数据


# 定义一个函数，将DataFrame转换为pickle格式的字节数据
def pickle_respnder(df):
    with BytesIO() as bio:  # 使用BytesIO创建内存中的二进制流
        df.to_pickle(bio)  # 将DataFrame转换为pickle格式并写入到二进制流
        return bio.getvalue()  # 返回二进制流的内容


# 定义一个函数，将DataFrame转换为Stata格式的字节数据
def stata_responder(df):
    with BytesIO() as bio:  # 使用BytesIO创建内存中的二进制流
        df.to_stata(bio, write_index=False)  # 将DataFrame转换为Stata格式并写入到二进制流
        return bio.getvalue()  # 返回二进制流的内容


# 使用pytest的parametrize装饰器定义测试参数化
@pytest.mark.parametrize(
    "responder, read_method",  # 参数化的两个参数：响应函数和读取函数
    [
        (csv_responder, pd.read_csv),  # CSV响应函数和对应的读取函数
        (json_responder, pd.read_json),  # JSON响应函数和对应的读取函数
        (
            html_responder,  # HTML响应函数
            lambda *args, **kwargs: pd.read_html(*args, **kwargs)[0],  # HTML读取函数
        ),
        pytest.param(
            parquetpyarrow_reponder,  # 使用pyarrow引擎的parquet响应函数
            partial(pd.read_parquet, engine="pyarrow"),  # pyarrow引擎的parquet读取函数
            marks=td.skip_if_no("pyarrow"),  # 标记：如果没有pyarrow则跳过测试
        ),
        pytest.param(
            parquetfastparquet_responder,  # 使用fastparquet引擎的parquet响应函数
            partial(pd.read_parquet, engine="fastparquet"),  # fastparquet引擎的parquet读取函数
            marks=[
                td.skip_if_no("fastparquet"),  # 标记：如果没有fastparquet则跳过测试
                td.skip_if_no("fsspec"),  # 标记：如果没有fsspec则跳过测试
            ],
        ),
        (pickle_respnder, pd.read_pickle),  # pickle响应函数和对应的读取函数
        (stata_responder, pd.read_stata),  # Stata响应函数和对应的读取函数
        (gz_csv_responder, pd.read_csv),  # gzip压缩CSV响应函数和对应的读取函数
        (gz_json_responder, pd.read_json),  # gzip压缩JSON响应函数和对应的读取函数
    ],
)
@pytest.mark.parametrize(
    "storage_options",  # 参数化的存储选项参数
    [
        None,  # 不指定存储选项
        {"User-Agent": "foo"},  # 指定User-Agent为"foo"
        {"User-Agent": "foo", "Auth": "bar"},  # 指定User-Agent为"foo"和Auth为"bar"
    ],
)
# 定义一个测试函数，用于验证 HTTP 请求的头部信息
def test_request_headers(responder, read_method, httpserver, storage_options):
    # 创建一个预期的 DataFrame 对象，包含一个字段 'a'，值为 ['b']
    expected = pd.DataFrame({"a": ["b"]})
    # 默认的 HTTP 头部信息列表
    default_headers = ["Accept-Encoding", "Host", "Connection", "User-Agent"]
    
    # 如果 responder 函数名中包含 'gz'，说明响应内容经过 gzip 压缩
    if "gz" in responder.__name__:
        # 设置额外的 HTTP 头部信息，指定内容编码为 gzip
        extra = {"Content-Encoding": "gzip"}
        # 如果未指定 storage_options，将额外的头部信息赋给 storage_options
        if storage_options is None:
            storage_options = extra
        else:
            # 否则，将额外的头部信息与已有的 storage_options 合并
            storage_options |= extra
    else:
        # 如果不是 gzip 压缩，设置额外信息为 None
        extra = None
    
    # 预期的 HTTP 头部信息包括默认的头部信息以及 storage_options 中的键
    expected_headers = set(default_headers).union(
        storage_options.keys() if storage_options else []
    )
    
    # 使用 httpserver 提供的 responder 函数生成内容并发送，带有额外的 HTTP 头部信息
    httpserver.serve_content(content=responder(expected), headers=extra)
    
    # 使用 read_method 从 httpserver 中读取数据，传入 storage_options
    result = read_method(httpserver.url, storage_options=storage_options)
    
    # 使用 pytest 比较读取结果和预期的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # 获取 HTTP 请求的头部信息，并转换为字典格式
    request_headers = dict(httpserver.requests[0].headers)
    
    # 遍历预期的 HTTP 头部信息
    for header in expected_headers:
        # 弹出实际请求中对应的头部信息值
        exp = request_headers.pop(header)
        # 如果 storage_options 存在且当前头部信息在 storage_options 中
        if storage_options and header in storage_options:
            # 断言实际值与 storage_options 中的值相等
            assert exp == storage_options[header]
    
    # 断言所有预期的头部信息已经检查完毕，实际请求中没有多余的头部信息
    # 意味着没有额外的头部信息添加
    assert not request_headers


# 使用 pytest 的 parametrize 装饰器进行多参数化测试
@pytest.mark.parametrize(
    "engine",
    [
        "pyarrow",
        "fastparquet",
    ],
)
# 定义一个测试函数，用于测试将 DataFrame 对象写入 Parquet 格式文件
def test_to_parquet_to_disk_with_storage_options(engine):
    # 设置 HTTP 头部信息
    headers = {
        "User-Agent": "custom",
        "Auth": "other_custom",
    }
    
    # 如果未能导入指定的 engine，跳过该测试
    pytest.importorskip(engine)
    
    # 创建一个真实的 DataFrame 对象，包含一个字段 'column_name'，值为 ['column_value']
    true_df = pd.DataFrame({"column_name": ["column_value"]})
    
    # 定义错误消息，用于测试时的异常匹配
    msg = (
        "storage_options passed with file object or non-fsspec file path|"
        "storage_options passed with buffer, or non-supported URL"
    )
    
    # 使用 pytest 的 raises 断言，验证传递 storage_options 时的异常情况
    with pytest.raises(ValueError, match=msg):
        true_df.to_parquet("/tmp/junk.parquet", storage_options=headers, engine=engine)
```