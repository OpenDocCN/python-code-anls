# `D:\src\scipysrc\pandas\pandas\tests\window\test_online.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于单元测试

from pandas import (  # 从 Pandas 库中导入 DataFrame 和 Series 类
    DataFrame,
    Series,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

pytestmark = pytest.mark.single_cpu  # 设置 Pytest 的 marker，表明只在单 CPU 环境下运行测试用例

pytest.importorskip("numba")  # 如果没有安装 Numba 库，则跳过这些测试

@pytest.mark.filterwarnings("ignore")
# 当使用 parallel=True 且函数无法被 Numba 并行化时，忽略相应的警告
class TestEWM:
    def test_invalid_update(self):
        df = DataFrame({"a": range(5), "b": range(5)})  # 创建一个包含两列的 DataFrame
        online_ewm = df.head(2).ewm(0.5).online()  # 对 DataFrame 的前两行应用指数加权移动平均，online() 方法可能返回在线计算的对象
        with pytest.raises(
            ValueError,
            match="Must call mean with update=None first before passing update",
        ):
            online_ewm.mean(update=df.head(1))  # 测试在线计算对象在未正确调用 mean 方法前更新的情况

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "obj", [DataFrame({"a": range(5), "b": range(5)}), Series(range(5), name="foo")]
    )
    def test_online_vs_non_online_mean(
        self, obj, nogil, parallel, nopython, adjust, ignore_na
    ):
        expected = obj.ewm(0.5, adjust=adjust, ignore_na=ignore_na).mean()  # 计算标准的指数加权移动平均值作为对照
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        online_ewm = (
            obj.head(2)
            .ewm(0.5, adjust=adjust, ignore_na=ignore_na)
            .online(engine_kwargs=engine_kwargs)  # 在线计算指数加权移动平均值，使用给定的引擎参数
        )
        # 测试重置一次
        for _ in range(2):
            result = online_ewm.mean()
            tm.assert_equal(result, expected.head(2))  # 断言在线计算的平均值与预期结果一致

            result = online_ewm.mean(update=obj.tail(3))  # 更新在线计算的对象，并计算平均值
            tm.assert_equal(result, expected.tail(3))  # 断言更新后的结果与预期一致

            online_ewm.reset()  # 重置在线计算对象的状态

    @pytest.mark.xfail(raises=NotImplementedError)
    @pytest.mark.parametrize(
        "obj", [DataFrame({"a": range(5), "b": range(5)}), Series(range(5), name="foo")]
    )
    def test_update_times_mean(
        self, obj, nogil, parallel, nopython, adjust, ignore_na, halflife_with_times
    ):
        times = Series(
            np.array(
                ["2020-01-01", "2020-01-05", "2020-01-07", "2020-01-17", "2020-01-21"],
                dtype="datetime64[ns]",
            )
        )
        expected = obj.ewm(
            0.5,
            adjust=adjust,
            ignore_na=ignore_na,
            times=times,
            halflife=halflife_with_times,
        ).mean()  # 根据给定的时间序列计算带权的指数移动平均值作为预期结果

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        online_ewm = (
            obj.head(2)
            .ewm(
                0.5,
                adjust=adjust,
                ignore_na=ignore_na,
                times=times.head(2),
                halflife=halflife_with_times,
            )
            .online(engine_kwargs=engine_kwargs)  # 使用指定的引擎参数进行在线计算
        )
        # 测试重置一次
        for _ in range(2):
            result = online_ewm.mean()
            tm.assert_equal(result, expected.head(2))  # 断言在线计算的平均值与预期结果一致

            result = online_ewm.mean(update=obj.tail(3), update_times=times.tail(3))  # 更新在线计算的对象和时间序列，并计算平均值
            tm.assert_equal(result, expected.tail(3))  # 断言更新后的结果与预期一致

            online_ewm.reset()  # 重置在线计算对象的状态
    # 使用 pytest 的参数化装饰器标记这个测试方法，测试多个方法名称
    @pytest.mark.parametrize("method", ["aggregate", "std", "corr", "cov", "var"])
    # 定义测试函数，测试 ewm 方法抛出 NotImplementedError 异常
    def test_ewm_notimplementederror_raises(self, method):
        # 创建一个包含 0 到 9 的 Series 对象
        ser = Series(range(10))
        # 初始化一个空的关键字参数字典
        kwargs = {}
        # 如果 method 是 "aggregate"，设置 func 参数为一个恒等函数 lambda x: x
        if method == "aggregate":
            kwargs["func"] = lambda x: x

        # 使用 pytest 的断言检查，期望在执行下列代码时抛出 NotImplementedError 异常，
        # 异常信息要匹配 ".* is not implemented." 的正则表达式
        with pytest.raises(NotImplementedError, match=".* is not implemented."):
            # 获取 ser 对象调用 ewm(1).online() 返回的对象，再调用 method 对应的方法，
            # 传入 kwargs 字典作为参数
            getattr(ser.ewm(1).online(), method)(**kwargs)
```