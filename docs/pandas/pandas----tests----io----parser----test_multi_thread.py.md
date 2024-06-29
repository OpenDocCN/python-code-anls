# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_multi_thread.py`

```
"""
Tests multithreading behaviour for reading and
parsing files for each parser defined in parsers.py
"""

# 导入所需的库
from contextlib import ExitStack             # 用于管理上下文的工具
from io import BytesIO                       # 用于处理二进制数据的流
from multiprocessing.pool import ThreadPool  # 多线程池
import numpy as np                           # 数值计算库
import pytest                                # 测试框架

import pandas as pd                          # 数据分析库
from pandas import DataFrame                 # 数据分析库的数据结构
import pandas._testing as tm                 # 测试工具
from pandas.util.version import Version      # 版本管理工具

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")

# We'll probably always skip these for pyarrow
# Maybe we'll add our own tests for pyarrow too
# 定义 pytest 的标记列表
pytestmark = [
    pytest.mark.single_cpu,  # 单 CPU 环境下运行标记
    pytest.mark.slow,         # 标记为慢速测试
]

# 忽略警告的装饰器
@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
def test_multi_thread_string_io_read_csv(all_parsers, request):
    # see gh-11786
    parser = all_parsers

    # 如果使用 pyarrow 引擎，检查版本是否符合要求
    if parser.engine == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        if Version(pa.__version__) < Version("16.0"):
            # 对不支持的情况添加标记
            request.applymarker(
                pytest.mark.xfail(reason="# ValueError: Found non-unique column index")
            )

    max_row_range = 100
    num_files = 10

    # 生成多个文件的内容，每个文件内容为一行数字，编码为字节流
    bytes_to_df = (
        "\n".join([f"{i:d},{i:d},{i:d}" for i in range(max_row_range)]).encode()
        for _ in range(num_files)
    )

    # 在多线程环境中读取所有文件
    with ExitStack() as stack:
        files = [stack.enter_context(BytesIO(b)) for b in bytes_to_df]

        # 创建线程池
        pool = stack.enter_context(ThreadPool(8))

        # 使用线程池并行读取文件并解析
        results = pool.map(parser.read_csv, files)
        first_result = results[0]

        # 断言所有结果与第一个结果相等
        for result in results:
            tm.assert_frame_equal(first_result, result)


def _generate_multi_thread_dataframe(parser, path, num_rows, num_tasks):
    """
    Generate a DataFrame via multi-thread.

    Parameters
    ----------
    parser : BaseParser
        The parser object to use for reading the data.
    path : str
        The location of the CSV file to read.
    num_rows : int
        The number of rows to read per task.
    num_tasks : int
        The number of tasks to use for reading this DataFrame.

    Returns
    -------
    df : DataFrame
    """

    def reader(arg):
        """
        Create a reader for part of the CSV.

        Parameters
        ----------
        arg : tuple
            A tuple of the following:

            * start : int
                The starting row to start for parsing CSV
            * nrows : int
                The number of rows to read.

        Returns
        -------
        df : DataFrame
        """
        start, nrows = arg

        # 根据参数读取部分 CSV 数据
        if not start:
            return parser.read_csv(
                path, index_col=0, header=0, nrows=nrows, parse_dates=["date"]
            )

        return parser.read_csv(
            path,
            index_col=0,
            header=None,
            skiprows=int(start) + 1,
            nrows=nrows,
            parse_dates=[9],
        )

    # 生成任务列表，每个任务定义一段要读取的数据范围
    tasks = [
        (num_rows * i // num_tasks, num_rows // num_tasks) for i in range(num_tasks)
    ]
    # 使用线程池来并发执行任务，数量由 num_tasks 指定
    with ThreadPool(processes=num_tasks) as pool:
        # 并发地对任务列表 tasks 中的每个任务应用 reader 函数，返回结果列表
        results = pool.map(reader, tasks)
    
    # 获取第一个结果的列名作为标准列名
    header = results[0].columns
    
    # 将其他结果的列名调整为第一个结果的列名，保持一致性
    for r in results[1:]:
        r.columns = header
    
    # 将所有结果按行连接成一个最终的数据框
    final_dataframe = pd.concat(results)
    
    # 返回最终的数据框作为函数的输出
    return final_dataframe
@xfail_pyarrow  # 标记测试为预期失败，因为 'nrows' 选项不被支持
def test_multi_thread_path_multipart_read_csv(all_parsers):
    # 设置任务数为4，行数为48
    num_tasks = 4
    num_rows = 48

    # 使用所有解析器中的一个进行测试
    parser = all_parsers

    # 文件名
    file_name = "__thread_pool_reader__.csv"

    # 创建一个包含随机数据的 DataFrame
    df = DataFrame(
        {
            "a": np.random.default_rng(2).random(num_rows),
            "b": np.random.default_rng(2).random(num_rows),
            "c": np.random.default_rng(2).random(num_rows),
            "d": np.random.default_rng(2).random(num_rows),
            "e": np.random.default_rng(2).random(num_rows),
            "foo": ["foo"] * num_rows,
            "bar": ["bar"] * num_rows,
            "baz": ["baz"] * num_rows,
            "date": pd.date_range("20000101 09:00:00", periods=num_rows, freq="s"),
            "int": np.arange(num_rows, dtype="int64"),
        }
    )

    # 确保文件路径干净，并在上下文中使用
    with tm.ensure_clean(file_name) as path:
        # 将 DataFrame 写入 CSV 文件
        df.to_csv(path)

        # 调用多线程函数生成 DataFrame 结果
        result = _generate_multi_thread_dataframe(parser, path, num_rows, num_tasks)

    # 预期的结果 DataFrame
    expected = df[:]
    expected["date"] = expected["date"].astype("M8[s]")

    # 断言生成的结果与预期结果相等
    tm.assert_frame_equal(result, expected)
```