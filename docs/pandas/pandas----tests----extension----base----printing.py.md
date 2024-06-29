# `D:\src\scipysrc\pandas\pandas\tests\extension\base\printing.py`

```
# 导入必要的库
import io  # 导入io模块，用于处理文件流
import pytest  # 导入pytest模块，用于编写和运行测试
import pandas as pd  # 导入pandas库，用于数据处理和分析


class BasePrintingTests:
    """Tests checking the formatting of your EA when printed."""

    @pytest.mark.parametrize("size", ["big", "small"])
    def test_array_repr(self, data, size):
        # 根据参数size选择数据的部分或复制数据多次
        if size == "small":
            data = data[:5]  # 如果size为"small"，则只选择前5个数据
        else:
            data = type(data)._concat_same_type([data] * 5)  # 如果size为"big"，则复制数据5次

        # 获取数据的字符串表示
        result = repr(data)
        # 断言数据类型的名称在结果中
        assert type(data).__name__ in result
        # 断言结果中包含数据长度的信息
        assert f"Length: {len(data)}" in result
        # 断言结果中包含数据类型的字符串表示
        assert str(data.dtype) in result
        # 如果size为"big"，则断言结果中包含省略号
        if size == "big":
            assert "..." in result

    def test_array_repr_unicode(self, data):
        # 获取数据的Unicode字符串表示
        result = str(data)
        # 断言结果是字符串类型
        assert isinstance(result, str)

    def test_series_repr(self, data):
        # 创建一个Pandas Series对象
        ser = pd.Series(data)
        # 断言Series对象的字符串表示中包含数据类型的名称
        assert data.dtype.name in repr(ser)

    def test_dataframe_repr(self, data):
        # 创建一个Pandas DataFrame对象，列名为"A"，数据为输入的data
        df = pd.DataFrame({"A": data})
        # 获取DataFrame对象的字符串表示
        repr(df)

    def test_dtype_name_in_info(self, data):
        # 创建一个StringIO对象，用于捕获DataFrame的info()方法的输出
        buf = io.StringIO()
        # 调用DataFrame的info()方法，将输出写入到buf中
        pd.DataFrame({"A": data}).info(buf=buf)
        # 获取info()方法的输出结果
        result = buf.getvalue()
        # 断言输出结果中包含数据类型的名称
        assert data.dtype.name in result
```