# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_pipe.py`

```
import pytest  # 导入 pytest 库

from pandas import (  # 导入 pandas 库中的 DataFrame 和 Series
    DataFrame,
    Series,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestPipe:
    def test_pipe(self, frame_or_series):
        obj = DataFrame({"A": [1, 2, 3]})  # 创建 DataFrame 对象
        expected = DataFrame({"A": [1, 4, 9]})  # 创建预期的 DataFrame 对象
        if frame_or_series is Series:  # 如果参数 frame_or_series 是 Series 类型
            obj = obj["A"]  # 取出 DataFrame 中的 "A" 列，转为 Series 对象
            expected = expected["A"]  # 取出预期 DataFrame 中的 "A" 列，转为 Series 对象

        f = lambda x, y: x**y  # 定义一个函数 f，计算 x 的 y 次方
        result = obj.pipe(f, 2)  # 使用 pipe 方法对 obj 应用函数 f，并传入参数 2
        tm.assert_equal(result, expected)  # 使用 pandas 测试模块的 assert_equal 方法比较 result 和 expected

    def test_pipe_tuple(self, frame_or_series):
        obj = DataFrame({"A": [1, 2, 3]})  # 创建 DataFrame 对象
        obj = tm.get_obj(obj, frame_or_series)  # 使用测试模块的 get_obj 方法获取 obj

        f = lambda x, y: y  # 定义一个函数 f，返回第二个参数 y
        result = obj.pipe((f, "y"), 0)  # 使用 pipe 方法传入元组 (f, "y") 和参数 0 对 obj 进行处理
        tm.assert_equal(result, obj)  # 使用 pandas 测试模块的 assert_equal 方法比较 result 和 obj

    def test_pipe_tuple_error(self, frame_or_series):
        obj = DataFrame({"A": [1, 2, 3]})  # 创建 DataFrame 对象
        obj = tm.get_obj(obj, frame_or_series)  # 使用测试模块的 get_obj 方法获取 obj

        f = lambda x, y: y  # 定义一个函数 f，返回第二个参数 y

        msg = "y is both the pipe target and a keyword argument"  # 设置异常消息

        # 使用 pytest 的 raises 方法捕获 ValueError 异常，并验证异常消息
        with pytest.raises(ValueError, match=msg):
            obj.pipe((f, "y"), x=1, y=0)
```