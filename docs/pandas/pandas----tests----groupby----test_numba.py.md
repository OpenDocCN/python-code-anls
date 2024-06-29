# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_numba.py`

```
import pytest  # 导入 pytest 模块

from pandas import (  # 从 pandas 库导入 DataFrame、Series 和 option_context
    DataFrame,
    Series,
    option_context,
)
import pandas._testing as tm  # 导入 pandas 内部测试工具模块

pytestmark = pytest.mark.single_cpu  # 定义 pytest 的标记为 single_cpu

pytest.importorskip("numba")  # 如果没有安装 numba 库，则跳过该测试用例


@pytest.mark.filterwarnings("ignore")
# 定义一个测试类 TestEngine，用于测试引擎功能
# Filter warnings when parallel=True and the function can't be parallelized by Numba
class TestEngine:
    def test_cython_vs_numba_frame(
        self, sort, nogil, parallel, nopython, numba_supported_reductions
    ):
        func, kwargs = numba_supported_reductions
        df = DataFrame({"a": [3, 2, 3, 2], "b": range(4), "c": range(1, 5)})
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        gb = df.groupby("a", sort=sort)
        result = getattr(gb, func)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        expected = getattr(gb, func)(**kwargs)
        tm.assert_frame_equal(result, expected)  # 断言结果与预期相等

    def test_cython_vs_numba_getitem(
        self, sort, nogil, parallel, nopython, numba_supported_reductions
    ):
        func, kwargs = numba_supported_reductions
        df = DataFrame({"a": [3, 2, 3, 2], "b": range(4), "c": range(1, 5)})
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        gb = df.groupby("a", sort=sort)["c"]
        result = getattr(gb, func)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        expected = getattr(gb, func)(**kwargs)
        tm.assert_series_equal(result, expected)  # 断言结果与预期相等

    def test_cython_vs_numba_series(
        self, sort, nogil, parallel, nopython, numba_supported_reductions
    ):
        func, kwargs = numba_supported_reductions
        ser = Series(range(3), index=[1, 2, 1], name="foo")
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        gb = ser.groupby(level=0, sort=sort)
        result = getattr(gb, func)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        expected = getattr(gb, func)(**kwargs)
        tm.assert_series_equal(result, expected)  # 断言结果与预期相等

    def test_as_index_false_unsupported(self, numba_supported_reductions):
        func, kwargs = numba_supported_reductions
        df = DataFrame({"a": [3, 2, 3, 2], "b": range(4), "c": range(1, 5)})
        gb = df.groupby("a", as_index=False)
        with pytest.raises(NotImplementedError, match="as_index=False"):
            getattr(gb, func)(engine="numba", **kwargs)  # 断言引发 NotImplementedError 异常

    def test_no_engine_doesnt_raise(self):
        # GH55520
        df = DataFrame({"a": [3, 2, 3, 2], "b": range(4), "c": range(1, 5)})
        gb = df.groupby("a")
        # Make sure behavior of functions w/out engine argument don't raise
        # when the global use_numba option is set
        with option_context("compute.use_numba", True):
            res = gb.agg({"b": "first"})  # 使用 Numba 引擎执行聚合操作
        expected = gb.agg({"b": "first"})  # 期望的聚合结果
        tm.assert_frame_equal(res, expected)  # 断言结果与预期相等
```