# `D:\src\scipysrc\seaborn\tests\_core\test_properties.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
import matplotlib as mpl  # 导入 Matplotlib 库的顶层模块
from matplotlib.colors import same_color, to_rgb, to_rgba  # 导入 Matplotlib 的颜色处理函数
from matplotlib.markers import MarkerStyle  # 导入 Matplotlib 的标记样式

import pytest  # 导入 Pytest 测试框架
from numpy.testing import assert_array_equal  # 导入 NumPy 测试工具，用于数组相等性检查

from seaborn._core.rules import categorical_order  # 导入 Seaborn 中的分类顺序规则
from seaborn._core.scales import Nominal, Continuous, Boolean  # 导入 Seaborn 中的标度类型
from seaborn._core.properties import (  # 导入 Seaborn 中的图形属性类
    Alpha,
    Color,
    Coordinate,
    EdgeWidth,
    Fill,
    LineStyle,
    LineWidth,
    Marker,
    PointSize,
)
from seaborn._compat import get_colormap  # 导入 Seaborn 中的颜色映射兼容性函数
from seaborn.palettes import color_palette  # 导入 Seaborn 中的调色板函数


class DataFixtures:  # 数据夹具类，用于提供测试数据

    @pytest.fixture  # 定义 Pytest 的测试夹具
    def num_vector(self, long_df):  # 数值向量夹具，从长数据框中获取数值列
        return long_df["s"]

    @pytest.fixture  # 定义 Pytest 的测试夹具
    def num_order(self, num_vector):  # 数值顺序夹具，从数值向量获取分类顺序
        return categorical_order(num_vector)

    @pytest.fixture  # 定义 Pytest 的测试夹具
    def cat_vector(self, long_df):  # 分类向量夹具，从长数据框中获取分类列
        return long_df["a"]

    @pytest.fixture  # 定义 Pytest 的测试夹具
    def cat_order(self, cat_vector):  # 分类顺序夹具，从分类向量获取分类顺序
        return categorical_order(cat_vector)

    @pytest.fixture  # 定义 Pytest 的测试夹具
    def dt_num_vector(self, long_df):  # 数据类型为数值的时间向量夹具，从长数据框中获取时间数值列
        return long_df["t"]

    @pytest.fixture  # 定义 Pytest 的测试夹具
    def dt_cat_vector(self, long_df):  # 数据类型为分类的时间向量夹具，从长数据框中获取时间分类列
        return long_df["d"]

    @pytest.fixture  # 定义 Pytest 的测试夹具
    def bool_vector(self, long_df):  # 布尔向量夹具，从长数据框中获取大于10的布尔列
        return long_df["x"] > 10

    @pytest.fixture  # 定义 Pytest 的测试夹具
    def vectors(self, num_vector, cat_vector, bool_vector):  # 向量夹具，提供数值向量、分类向量和布尔向量
        return {"num": num_vector, "cat": cat_vector, "bool": bool_vector}


class TestCoordinate(DataFixtures):  # 坐标测试类，继承自数据夹具类

    def test_bad_scale_arg_str(self, num_vector):  # 测试不良的坐标标度参数为字符串类型
        err = "Unknown magic arg for x scale: 'xxx'."  # 错误信息
        with pytest.raises(ValueError, match=err):  # 断言抛出值错误且匹配特定错误信息
            Coordinate("x").infer_scale("xxx", num_vector)

    def test_bad_scale_arg_type(self, cat_vector):  # 测试不良的坐标标度参数为列表类型
        err = "Magic arg for x scale must be str, not list."  # 错误信息
        with pytest.raises(TypeError, match=err):  # 断言抛出类型错误且匹配特定错误信息
            Coordinate("x").infer_scale([1, 2, 3], cat_vector)


class TestColor(DataFixtures):  # 颜色测试类，继承自数据夹具类

    def assert_same_rgb(self, a, b):  # 断言两个 RGB 颜色数组相同
        assert_array_equal(a[:, :3], b[:, :3])

    def test_nominal_default_palette(self, cat_vector, cat_order):  # 测试标称型默认调色板
        m = Color().get_mapping(Nominal(), cat_vector)  # 获取标称型映射
        n = len(cat_order)  # 分类顺序的长度
        actual = m(np.arange(n))  # 根据映射和范围创建实际颜色
        expected = color_palette(None, n)  # 使用默认调色板生成期望颜色
        for have, want in zip(actual, expected):  # 遍历实际和期望颜色
            assert same_color(have, want)  # 断言实际颜色与期望颜色相同

    def test_nominal_default_palette_large(self):  # 测试大型标称型默认调色板
        vector = pd.Series(list("abcdefghijklmnopqrstuvwxyz"))  # 创建包含字母的系列向量
        m = Color().get_mapping(Nominal(), vector)  # 获取标称型映射
        actual = m(np.arange(26))  # 根据映射和范围创建实际颜色
        expected = color_palette("husl", 26)  # 使用 HUSL 调色板生成期望颜色
        for have, want in zip(actual, expected):  # 遍历实际和期望颜色
            assert same_color(have, want)  # 断言实际颜色与期望颜色相同
    # 测试使用命名的调色板生成映射
    def test_nominal_named_palette(self, cat_vector, cat_order):
        # 设定调色板名称为 "Blues"
        palette = "Blues"
        # 调用 Color 类的 get_mapping 方法，生成从分类数据到颜色映射的对象 m
        m = Color().get_mapping(Nominal(palette), cat_vector)
        # 获取分类顺序的长度
        n = len(cat_order)
        # 计算实际颜色映射值
        actual = m(np.arange(n))
        # 获取预期颜色映射值，使用 color_palette 函数根据 palette 名称生成
        expected = color_palette(palette, n)
        # 遍历实际与预期的颜色值，并断言它们相同
        for have, want in zip(actual, expected):
            assert same_color(have, want)

    # 测试使用列表形式的调色板生成映射
    def test_nominal_list_palette(self, cat_vector, cat_order):
        # 使用 color_palette 函数生成指定长度的调色板 "Reds"
        palette = color_palette("Reds", len(cat_order))
        # 调用 Color 类的 get_mapping 方法，生成从分类数据到颜色映射的对象 m
        m = Color().get_mapping(Nominal(palette), cat_vector)
        # 计算实际颜色映射值
        actual = m(np.arange(len(palette)))
        # 预期颜色映射值即为生成的调色板
        expected = palette
        # 遍历实际与预期的颜色值，并断言它们相同
        for have, want in zip(actual, expected):
            assert same_color(have, want)

    # 测试使用字典形式的调色板生成映射
    def test_nominal_dict_palette(self, cat_vector, cat_order):
        # 使用 color_palette 函数生成颜色列表 "Greens"
        colors = color_palette("Greens")
        # 创建一个字典，将分类顺序与颜色列表中的颜色一一对应
        palette = dict(zip(cat_order, colors))
        # 调用 Color 类的 get_mapping 方法，生成从分类数据到颜色映射的对象 m
        m = Color().get_mapping(Nominal(palette), cat_vector)
        # 获取分类顺序的长度
        n = len(cat_order)
        # 计算实际颜色映射值
        actual = m(np.arange(n))
        # 预期颜色映射值即为颜色列表 "Greens"
        expected = colors
        # 遍历实际与预期的颜色值，并断言它们相同
        for have, want in zip(actual, expected):
            assert same_color(have, want)

    # 测试使用具有缺失键的字典形式的调色板生成映射
    def test_nominal_dict_with_missing_keys(self, cat_vector, cat_order):
        # 使用 color_palette 函数生成颜色列表 "Purples"
        palette = dict(zip(cat_order[1:], color_palette("Purples")))
        # 使用 pytest 的 raises 断言检测 ValueError 异常，并匹配特定的错误信息
        with pytest.raises(ValueError, match="No entry in color dict"):
            # 调用 Color 类的 get_mapping 方法，生成从分类数据到颜色映射的对象
            Color("color").get_mapping(Nominal(palette), cat_vector)

    # 测试使用过短列表形式的调色板生成映射
    def test_nominal_list_too_short(self, cat_vector, cat_order):
        # 计算调色板长度，比分类顺序少一
        n = len(cat_order) - 1
        # 使用 color_palette 函数生成指定长度的调色板 "Oranges"
        palette = color_palette("Oranges", n)
        # 构造警告消息，指出调色板列表长度比所需长度少一
        msg = rf"The edgecolor list has fewer values \({n}\) than needed \({n + 1}\)"
        # 使用 pytest 的 warns 断言捕获 UserWarning 警告，并匹配特定的警告消息
        with pytest.warns(UserWarning, match=msg):
            # 调用 Color 类的 get_mapping 方法，生成从分类数据到颜色映射的对象
            Color("edgecolor").get_mapping(Nominal(palette), cat_vector)

    # 测试使用过长列表形式的调色板生成映射
    def test_nominal_list_too_long(self, cat_vector, cat_order):
        # 计算调色板长度，比分类顺序多一
        n = len(cat_order) + 1
        # 使用 color_palette 函数生成指定长度的调色板 "Oranges"
        palette = color_palette("Oranges", n)
        # 构造警告消息，指出调色板列表长度比所需长度多一
        msg = rf"The edgecolor list has more values \({n}\) than needed \({n - 1}\)"
        # 使用 pytest 的 warns 断言捕获 UserWarning 警告，并匹配特定的警告消息
        with pytest.warns(UserWarning, match=msg):
            # 调用 Color 类的 get_mapping 方法，生成从分类数据到颜色映射的对象
            Color("edgecolor").get_mapping(Nominal(palette), cat_vector)

    # 测试使用默认的连续调色板生成映射
    def test_continuous_default_palette(self, num_vector):
        # 使用 color_palette 函数生成连续调色板 "ch:"，并设定为 cmap
        cmap = color_palette("ch:", as_cmap=True)
        # 调用 Color 类的 get_mapping 方法，生成从连续数据到颜色映射的对象 m
        m = Color().get_mapping(Continuous(), num_vector)
        # 断言生成的颜色映射与连续调色板 cmap 的颜色映射相同
        self.assert_same_rgb(m(num_vector), cmap(num_vector))

    # 测试使用命名的连续调色板生成映射
    def test_continuous_named_palette(self, num_vector):
        # 设定调色板名称为 "flare"
        pal = "flare"
        # 使用 color_palette 函数生成连续调色板 "flare"，并设定为 cmap
        cmap = color_palette(pal, as_cmap=True)
        # 调用 Color 类的 get_mapping 方法，生成从连续数据到颜色映射的对象 m
        m = Color().get_mapping(Continuous(pal), num_vector)
        # 断言生成的颜色映射与连续调色板 cmap 的颜色映射相同
        self.assert_same_rgb(m(num_vector), cmap(num_vector))

    # 测试使用元组形式的连续调色板生成映射
    def test_continuous_tuple_palette(self, num_vector):
        # 设定颜色元组值
        vals = ("blue", "red")
        # 使用 color_palette 函数生成混合的连续调色板 "blend:blue,red"，并设定为 cmap
        cmap = color_palette("blend:" + ",".join(vals), as_cmap=True)
        # 调用 Color 类的 get_mapping 方法，生成从连续数据到颜色映射的对象 m
        m = Color().get_mapping(Continuous(vals), num_vector)
        # 断言生成的颜色映射与混合调色板 cmap 的颜色映射相同
        self.assert_same_rgb(m(num_vector), cmap(num_vector))
    # 测试连续型调色板的功能
    def test_continuous_callable_palette(self, num_vector):
        # 获取名为"viridis"的颜色映射
        cmap = get_colormap("viridis")
        # 使用颜色映射创建 Color 对象，并获取连续型映射
        m = Color().get_mapping(Continuous(cmap), num_vector)
        # 断言 m(num_vector) 和 cmap(num_vector) 的 RGB 值相同
        self.assert_same_rgb(m(num_vector), cmap(num_vector))

    # 测试连续型映射中的缺失值处理
    def test_continuous_missing(self):
        # 创建包含缺失值的 pandas Series 对象 x
        x = pd.Series([1, 2, np.nan, 4])
        # 使用默认连续型映射创建 Color 对象，并获取映射结果
        m = Color().get_mapping(Continuous(), x)
        # 断言 m(x)[2] 中所有值为 NaN
        assert np.isnan(m(x)[2]).all()

    # 测试连续型映射中不合法的颜色比例值
    def test_bad_scale_values_continuous(self, num_vector):
        # 使用包含非法颜色值的列表创建 Color 对象，并断言引发 TypeError 异常
        with pytest.raises(TypeError, match="Scale values for color with a Continuous"):
            Color().get_mapping(Continuous(["r", "g", "b"]), num_vector)

    # 测试名义型映射中不合法的颜色比例值
    def test_bad_scale_values_nominal(self, cat_vector):
        # 使用 "viridis" 调色板创建名义型映射的 Color 对象，并断言引发 TypeError 异常
        with pytest.raises(TypeError, match="Scale values for color with a Nominal"):
            Color().get_mapping(Nominal(get_colormap("viridis")), cat_vector)

    # 测试推断映射时的不合法参数
    def test_bad_inference_arg(self, cat_vector):
        # 使用整数参数调用推断映射方法，并断言引发 TypeError 异常
        with pytest.raises(TypeError, match="A single scale argument for color"):
            Color().infer_scale(123, cat_vector)

    # 测试默认映射功能
    @pytest.mark.parametrize(
        "data_type,scale_class",
        [("cat", Nominal), ("num", Continuous), ("bool", Boolean)]
    )
    def test_default(self, data_type, scale_class, vectors):
        # 获取默认的颜色映射
        scale = Color().default_scale(vectors[data_type])
        # 断言 scale 类型为 scale_class
        assert isinstance(scale, scale_class)

    # 测试默认映射对数值分类数据类型的处理
    def test_default_numeric_data_category_dtype(self, num_vector):
        # 将数值类型的 Series 转换为分类类型，并获取默认映射
        scale = Color().default_scale(num_vector.astype("category"))
        # 断言 scale 类型为名义型
        assert isinstance(scale, Nominal)

    # 测试默认映射对二元数据的处理
    def test_default_binary_data(self):
        # 创建包含二元数据的 Series 对象 x
        x = pd.Series([0, 0, 1, 0, 1], dtype=int)
        # 获取默认映射
        scale = Color().default_scale(x)
        # 断言 scale 类型为连续型
        assert isinstance(scale, Continuous)

    # 测试映射推断功能
    @pytest.mark.parametrize(
        "values,data_type,scale_class",
        [
            ("viridis", "cat", Nominal),  # 基于变量类型
            ("viridis", "num", Continuous),  # 基于变量类型
            ("viridis", "bool", Boolean),  # 基于变量类型
            ("muted", "num", Nominal),  # 基于质性调色板
            (["r", "g", "b"], "num", Nominal),  # 基于列表调色板
            ({2: "r", 4: "g", 8: "b"}, "num", Nominal),  # 基于字典调色板
            (("r", "b"), "num", Continuous),  # 基于元组 / 变量类型
            (("g", "m"), "cat", Nominal),  # 基于元组 / 变量类型
            (("c", "y"), "bool", Boolean),  # 基于元组 / 变量类型
            (get_colormap("inferno"), "num", Continuous),  # 基于可调用对象
        ]
    )
    def test_inference(self, values, data_type, scale_class, vectors):
        # 根据给定的 values 推断映射类型，并获取映射对象
        scale = Color().infer_scale(values, vectors[data_type])
        # 断言 scale 类型为 scale_class
        assert isinstance(scale, scale_class)
        # 断言 scale 的值与给定的 values 相等
        assert scale.values == values
    # 定义一个测试方法，用于测试颜色标准化函数的正确性
    def test_standardization(self):
        
        # 获取颜色标准化函数的引用
        f = Color().standardize
        
        # 断言颜色字符串 "C3" 经过标准化后等于 RGB 表示形式 "C3"
        assert f("C3") == to_rgb("C3")
        
        # 断言颜色字符串 "dodgerblue" 经过标准化后等于其对应的 RGB 表示形式
        assert f("dodgerblue") == to_rgb("dodgerblue")
        
        # 断言 RGB 元组 (.1, .2, .3) 经过标准化后仍等于自身
        assert f((.1, .2, .3)) == (.1, .2, .3)
        
        # 断言 RGBA 元组 (.1, .2, .3, .4) 经过标准化后仍等于自身
        assert f((.1, .2, .3, .4)) == (.1, .2, .3, .4)
        
        # 断言十六进制颜色字符串 "#123456" 经过标准化后等于其对应的 RGB 表示形式
        assert f("#123456") == to_rgb("#123456")
        
        # 断言十六进制颜色字符串 "#12345678" 经过标准化后等于其对应的 RGBA 表示形式
        assert f("#12345678") == to_rgba("#12345678")
        
        # 断言简写十六进制颜色字符串 "#123" 经过标准化后等于其对应的 RGB 表示形式
        assert f("#123") == to_rgb("#123")
        
        # 断言简写十六进制颜色字符串 "#1234" 经过标准化后等于其对应的 RGBA 表示形式
        assert f("#1234") == to_rgba("#1234")
# 定义一个类 ObjectPropertyBase，继承自 DataFixtures 类
class ObjectPropertyBase(DataFixtures):

    # 定义一个断言方法 assert_equal，用于比较两个参数 a 和 b 是否相等
    def assert_equal(self, a, b):
        assert self.unpack(a) == self.unpack(b)

    # 定义一个方法 unpack，用于返回其参数 x，未经处理直接返回
    def unpack(self, x):
        return x

    # 使用 pytest.mark.parametrize 装饰器标记参数化测试用例，data_type 可选值为 "cat", "num", "bool"
    def test_default(self, data_type, vectors):
        # 调用 self.prop() 的 default_scale 方法，传入 vectors 中对应 data_type 的值，返回 scale
        scale = self.prop().default_scale(vectors[data_type])
        # 断言 scale 的类型，如果 data_type 是 "bool" 则为 Boolean，否则为 Nominal
        assert isinstance(scale, Boolean if data_type == "bool" else Nominal)

    # 参数化测试用例，data_type 可选值为 "cat", "num", "bool"
    def test_inference_list(self, data_type, vectors):
        # 调用 self.prop() 的 infer_scale 方法，传入 self.values 和 vectors[data_type]，返回 scale
        scale = self.prop().infer_scale(self.values, vectors[data_type])
        # 断言 scale 的类型，如果 data_type 是 "bool" 则为 Boolean，否则为 Nominal
        assert isinstance(scale, Boolean if data_type == "bool" else Nominal)
        # 断言 scale 的值等于 self.values
        assert scale.values == self.values

    # 参数化测试用例，data_type 可选值为 "cat", "num", "bool"
    def test_inference_dict(self, data_type, vectors):
        # 获取 vectors[data_type] 并生成与 self.values 组成的字典 values
        x = vectors[data_type]
        values = dict(zip(categorical_order(x), self.values))
        # 调用 self.prop() 的 infer_scale 方法，传入 values 和 x，返回 scale
        scale = self.prop().infer_scale(values, x)
        # 断言 scale 的类型，如果 data_type 是 "bool" 则为 Boolean，否则为 Nominal
        assert isinstance(scale, Boolean if data_type == "bool" else Nominal)
        # 断言 scale 的值等于 values
        assert scale.values == values

    # 定义一个方法 test_dict_missing，测试处理缺失字典条目的情况
    def test_dict_missing(self, cat_vector):
        # 获取 cat_vector 的分类顺序 levels，生成 values 字典，生成 Nominal 类型的 scale
        levels = categorical_order(cat_vector)
        values = dict(zip(levels, self.values[:-1]))
        scale = Nominal(values)
        # 获取属性名小写的字符串 name，生成错误信息 msg
        name = self.prop.__name__.lower()
        msg = f"No entry in {name} dictionary for {repr(levels[-1])}"
        # 使用 pytest.raises 断言引发 ValueError 异常，并匹配错误信息 msg
        with pytest.raises(ValueError, match=msg):
            self.prop().get_mapping(scale, cat_vector)

    # 参数化测试用例，data_type 可选值为 "cat", "num"
    def test_mapping_default(self, data_type, vectors):
        # 获取 vectors[data_type]，调用 self.prop() 的 get_mapping 方法，传入 Nominal() 和 x，返回 mapping
        x = vectors[data_type]
        mapping = self.prop().get_mapping(Nominal(), x)
        # 获取 x 的唯一值数量 n，使用 for 循环比较 self.prop()._default_values(n) 和 mapping([i])
        n = x.nunique()
        for i, expected in enumerate(self.prop()._default_values(n)):
            actual, = mapping([i])
            self.assert_equal(actual, expected)

    # 参数化测试用例，data_type 可选值为 "cat", "num"
    def test_mapping_from_list(self, data_type, vectors):
        # 获取 vectors[data_type]，生成 Nominal 类型的 scale，调用 self.prop() 的 get_mapping 方法，返回 mapping
        x = vectors[data_type]
        scale = Nominal(self.values)
        mapping = self.prop().get_mapping(scale, x)
        # 使用 for 循环比较 self.standardized_values 和 mapping([i])
        for i, expected in enumerate(self.standardized_values):
            actual, = mapping([i])
            self.assert_equal(actual, expected)

    # 参数化测试用例，data_type 可选值为 "cat", "num"
    def test_mapping_from_dict(self, data_type, vectors):
        # 获取 vectors[data_type] 的分类顺序 levels，生成与 self.values 和 self.standardized_values 相反的字典 values 和 standardized_values
        x = vectors[data_type]
        levels = categorical_order(x)
        values = dict(zip(levels, self.values[::-1]))
        standardized_values = dict(zip(levels, self.standardized_values[::-1]))
        # 生成 Nominal 类型的 scale，调用 self.prop() 的 get_mapping 方法，返回 mapping
        scale = Nominal(values)
        mapping = self.prop().get_mapping(scale, x)
        # 使用 for 循环比较 standardized_values[level] 和 mapping([i])
        for i, level in enumerate(levels):
            actual, = mapping([i])
            expected = standardized_values[level]
            self.assert_equal(actual, expected)
    # 测试用例：使用空值测试映射功能
    def test_mapping_with_null_value(self, cat_vector):
        # 获取映射，使用属性对象的方法获取从名义数据到类别向量的映射
        mapping = self.prop().get_mapping(Nominal(self.values), cat_vector)
        # 调用映射，传入包含空值的数组进行测试
        actual = mapping(np.array([0, np.nan, 2]))
        # 获取标准化后的值中的零、空值、二的值
        v0, _, v2 = self.standardized_values
        # 期望的输出应该是一个列表，包含标准化后的零、属性对象的空值、二
        expected = [v0, self.prop.null_value, v2]
        # 遍历实际输出和期望输出进行断言比较
        for a, b in zip(actual, expected):
            self.assert_equal(a, b)

    # 测试用例：测试具有大数量的唯一默认值
    def test_unique_default_large_n(self):
        # 设定测试数据的数量
        n = 24
        # 创建一个包含0到n-1的序列
        x = pd.Series(np.arange(n))
        # 获取映射，使用属性对象的方法获取从名义数据到x的映射
        mapping = self.prop().get_mapping(Nominal(), x)
        # 断言映射后的唯一值数量等于n
        assert len({self.unpack(x_i) for x_i in mapping(x)}) == n

    # 测试用例：测试不良比例值
    def test_bad_scale_values(self, cat_vector):
        # 获取属性对象名称的小写形式作为变量名
        var_name = self.prop.__name__.lower()
        # 使用pytest断言捕获特定类型错误，验证比例值是否适用于变量名称的变量
        with pytest.raises(TypeError, match=f"Scale values for a {var_name} variable"):
            # 获取映射，使用属性对象的方法获取从名义数据到类别向量的映射
            self.prop().get_mapping(Nominal(("o", "s")), cat_vector)
# TestMarker 类，继承自 ObjectPropertyBase 类
class TestMarker(ObjectPropertyBase):

    # 类属性 prop 被赋值为 Marker 类型
    prop = Marker
    # 类属性 values 包含字符串 "o"、元组 (5, 2, 0) 和 MarkerStyle("^")
    values = ["o", (5, 2, 0), MarkerStyle("^")]
    # 类属性 standardized_values 通过列表推导式将 values 中的每个元素转换为 MarkerStyle 对象
    standardized_values = [MarkerStyle(x) for x in values]

    # 定义 assert_equal 方法，用于比较两个对象的路径、连接样式、变换值和填充样式是否相等
    def assert_equal(self, a, b):
        # 获取对象 a 和 b 的路径对象
        a_path, b_path = a.get_path(), b.get_path()
        # 断言两个路径对象的顶点数组 vertices 和代码数组 codes 相等
        assert_array_equal(a_path.vertices, b_path.vertices)
        assert_array_equal(a_path.codes, b_path.codes)
        # 断言路径对象的简化阈值 simplify_threshold 和是否应该简化 should_simplify 相等
        assert a_path.simplify_threshold == b_path.simplify_threshold
        assert a_path.should_simplify == b_path.should_simplify

        # 断言对象 a 和 b 的连接样式相等
        assert a.get_joinstyle() == b.get_joinstyle()
        # 断言对象 a 和 b 的变换值相等
        assert a.get_transform().to_values() == b.get_transform().to_values()
        # 断言对象 a 和 b 的填充样式相等
        assert a.get_fillstyle() == b.get_fillstyle()

    # 定义 unpack 方法，用于获取对象 x 的路径、连接样式、变换值和填充样式，并返回为元组
    def unpack(self, x):
        return (
            x.get_path(),
            x.get_joinstyle(),
            x.get_transform().to_values(),
            x.get_fillstyle(),
        )


# TestLineStyle 类，继承自 ObjectPropertyBase 类
class TestLineStyle(ObjectPropertyBase):

    # 类属性 prop 被赋值为 LineStyle 类型
    prop = LineStyle
    # 类属性 values 包含字符串 "solid"、字符串 "--" 和元组 (1, .5)
    values = ["solid", "--", (1, .5)]
    # 类属性 standardized_values 通过列表推导式调用 LineStyle._get_dash_pattern 方法，将 values 中的每个元素转换为对应的线型对象
    standardized_values = [LineStyle._get_dash_pattern(x) for x in values]

    # 定义 test_bad_type 方法，测试当输入非期望类型时是否引发 TypeError 异常
    def test_bad_type(self):
        p = LineStyle()
        with pytest.raises(TypeError, match="^Linestyle must be .+, not list.$"):
            p.standardize([1, 2])

    # 定义 test_bad_style 方法，测试当输入非期望线型样式时是否引发 ValueError 异常
    def test_bad_style(self):
        p = LineStyle()
        with pytest.raises(ValueError, match="^Linestyle string must be .+, not 'o'.$"):
            p.standardize("o")

    # 定义 test_bad_dashes 方法，测试当输入非法的虚线模式时是否引发 TypeError 异常
    def test_bad_dashes(self):
        p = LineStyle()
        with pytest.raises(TypeError, match="^Invalid dash pattern"):
            p.standardize((1, 2, "x"))


# TestFill 类，继承自 DataFixtures 类
class TestFill(DataFixtures):

    # 定义 vectors 方法，作为 pytest 的 fixture，返回一个包含不同类型数据的字典
    @pytest.fixture
    def vectors(self):
        return {
            "cat": pd.Series(["a", "a", "b"]),
            "num": pd.Series([1, 1, 2]),
            "bool": pd.Series([True, True, False])
        }

    # 定义 cat_vector 方法，作为 pytest 的 fixture，返回 vectors 中键为 "cat" 的值
    @pytest.fixture
    def cat_vector(self, vectors):
        return vectors["cat"]

    # 定义 num_vector 方法，作为 pytest 的 fixture，返回 vectors 中键为 "num" 的值
    @pytest.fixture
    def num_vector(self, vectors):
        return vectors["num"]

    # 使用 pytest.mark.parametrize 装饰器，定义 test_default 方法，测试 Fill().default_scale 方法的默认行为
    @pytest.mark.parametrize("data_type", ["cat", "num", "bool"])
    def test_default(self, data_type, vectors):
        x = vectors[data_type]
        # 调用 Fill().default_scale 方法，获取数据 x 的标准化比例尺，并断言其类型和数值是否符合预期
        scale = Fill().default_scale(x)
        assert isinstance(scale, Boolean if data_type == "bool" else Nominal)

    # 使用 pytest.mark.parametrize 装饰器，定义 test_inference_list 方法，测试 Fill().infer_scale 方法对列表数据的推断能力
    @pytest.mark.parametrize("data_type", ["cat", "num", "bool"])
    def test_inference_list(self, data_type, vectors):
        x = vectors[data_type]
        # 调用 Fill().infer_scale 方法，推断列表 [True, False] 对数据 x 的标准化比例尺，并断言其类型和数值是否符合预期
        scale = Fill().infer_scale([True, False], x)
        assert isinstance(scale, Boolean if data_type == "bool" else Nominal)
        assert scale.values == [True, False]

    # 使用 pytest.mark.parametrize 装饰器，定义 test_inference_dict 方法，测试 Fill().infer_scale 方法对字典数据的推断能力
    @pytest.mark.parametrize("data_type", ["cat", "num", "bool"])
    def test_inference_dict(self, data_type, vectors):
        x = vectors[data_type]
        # 创建由数据 x 唯一值和 [True, False] 组成的字典，调用 Fill().infer_scale 方法，推断其对数据 x 的标准化比例尺
        values = dict(zip(x.unique(), [True, False]))
        scale = Fill().infer_scale(values, x)
        assert isinstance(scale, Boolean if data_type == "bool" else Nominal)
        assert scale.values == values
    # 测试用例：对分类数据进行映射测试
    def test_mapping_categorical_data(self, cat_vector):
        # 使用Fill类的get_mapping方法获取分类数据的映射
        mapping = Fill().get_mapping(Nominal(), cat_vector)
        # 断言映射结果与预期结果一致
        assert_array_equal(mapping([0, 1, 0]), [True, False, True])
    
    # 测试用例：对数值数据进行映射测试
    def test_mapping_numeric_data(self, num_vector):
        # 使用Fill类的get_mapping方法获取数值数据的映射
        mapping = Fill().get_mapping(Nominal(), num_vector)
        # 断言映射结果与预期结果一致
        assert_array_equal(mapping([0, 1, 0]), [True, False, True])
    
    # 测试用例：对列表进行映射测试
    def test_mapping_list(self, cat_vector):
        # 使用Fill类的get_mapping方法获取列表的映射，指定Nominal对象的映射值为[False, True]
        mapping = Fill().get_mapping(Nominal([False, True]), cat_vector)
        # 断言映射结果与预期结果一致
        assert_array_equal(mapping([0, 1, 0]), [False, True, False])
    
    # 测试用例：对真值列表进行映射测试
    def test_mapping_truthy_list(self, cat_vector):
        # 使用Fill类的get_mapping方法获取真值列表的映射，指定Nominal对象的映射值为[0, 1]
        mapping = Fill().get_mapping(Nominal([0, 1]), cat_vector)
        # 断言映射结果与预期结果一致
        assert_array_equal(mapping([0, 1, 0]), [False, True, False])
    
    # 测试用例：对字典进行映射测试
    def test_mapping_dict(self, cat_vector):
        # 根据cat_vector的唯一值和指定的映射字典构建映射值
        values = dict(zip(cat_vector.unique(), [False, True]))
        # 使用Fill类的get_mapping方法获取字典映射
        mapping = Fill().get_mapping(Nominal(values), cat_vector)
        # 断言映射结果与预期结果一致
        assert_array_equal(mapping([0, 1, 0]), [False, True, False])
    
    # 测试用例：检查循环警告是否触发
    def test_cycle_warning(self):
        # 创建包含字符串的Series对象x
        x = pd.Series(["a", "b", "c"])
        # 使用pytest的warns上下文管理器检查是否触发UserWarning并匹配给定的警告信息
        with pytest.warns(UserWarning, match="The variable assigned to fill"):
            # 调用Fill类的get_mapping方法，传入Nominal对象和Series对象x
            Fill().get_mapping(Nominal(), x)
    
    # 测试用例：检查数值错误是否引发异常
    def test_values_error(self):
        # 创建包含字符串的Series对象x
        x = pd.Series(["a", "b"])
        # 使用pytest的raises上下文管理器检查是否引发TypeError并匹配给定的异常信息
        with pytest.raises(TypeError, match="Scale values for fill must be"):
            # 调用Fill类的get_mapping方法，传入Nominal对象和Series对象x
            Fill().get_mapping(Nominal("bad_values"), x)
class IntervalBase(DataFixtures):
    # IntervalBase 类继承自 DataFixtures，用于测试数据修正相关的间隔操作

    def norm(self, x):
        # 归一化函数，对输入 x 进行归一化处理
        return (x - x.min()) / (x.max() - x.min())

    @pytest.mark.parametrize("data_type,scale_class", [
        ("cat", Nominal),
        ("num", Continuous),
        ("bool", Boolean),
    ])
    def test_default(self, data_type, scale_class, vectors):
        # 测试默认设置功能

        # 获取对应数据类型的向量 x
        x = vectors[data_type]
        # 调用 prop 方法获取默认的数据缩放器 scale
        scale = self.prop().default_scale(x)
        # 断言 scale 是指定的 scale_class 类型
        assert isinstance(scale, scale_class)

    @pytest.mark.parametrize("arg,data_type,scale_class", [
        ((1, 3), "cat", Nominal),
        ((1, 3), "num", Continuous),
        ((1, 3), "bool", Boolean),
        ([1, 2, 3], "cat", Nominal),
        ([1, 2, 3], "num", Nominal),
        ([1, 3], "bool", Boolean),
        ({"a": 1, "b": 3, "c": 2}, "cat", Nominal),
        ({2: 1, 4: 3, 8: 2}, "num", Nominal),
        ({True: 4, False: 2}, "bool", Boolean),
    ])
    def test_inference(self, arg, data_type, scale_class, vectors):
        # 测试推断功能

        # 获取对应数据类型的向量 x
        x = vectors[data_type]
        # 调用 prop 方法进行推断数据缩放器 scale
        scale = self.prop().infer_scale(arg, x)
        # 断言 scale 是指定的 scale_class 类型
        assert isinstance(scale, scale_class)
        # 断言 scale 的值等于参数 arg
        assert scale.values == arg

    def test_mapped_interval_numeric(self, num_vector):
        # 测试数值型数据的映射间隔

        # 获取使用 Continuous 类型的映射器 mapping
        mapping = self.prop().get_mapping(Continuous(), num_vector)
        # 断言 mapping([0, 1]) 等于默认的数据范围 default_range
        assert_array_equal(mapping([0, 1]), self.prop().default_range)

    def test_mapped_interval_categorical(self, cat_vector):
        # 测试分类数据的映射间隔

        # 获取使用 Nominal 类型的映射器 mapping
        mapping = self.prop().get_mapping(Nominal(), cat_vector)
        # 获取分类向量的唯一值数量 n
        n = cat_vector.nunique()
        # 断言 mapping([n - 1, 0]) 等于默认的数据范围 default_range
        assert_array_equal(mapping([n - 1, 0]), self.prop().default_range)

    def test_bad_scale_values_numeric_data(self, num_vector):
        # 测试数值型数据不良的缩放值情况

        # 获取属性名并转换为小写形式
        prop_name = self.prop.__name__.lower()
        # 构建错误提示信息前缀
        err_stem = (
            f"Values for {prop_name} variables with Continuous scale must be 2-tuple"
        )

        # 使用 pytest 引发 TypeError 异常，并匹配错误信息
        with pytest.raises(TypeError, match=f"{err_stem}; not <class 'str'>."):
            self.prop().get_mapping(Continuous("abc"), num_vector)

        # 使用 pytest 引发 TypeError 异常，并匹配错误信息
        with pytest.raises(TypeError, match=f"{err_stem}; not 3-tuple."):
            self.prop().get_mapping(Continuous((1, 2, 3)), num_vector)

    def test_bad_scale_values_categorical_data(self, cat_vector):
        # 测试分类数据不良的缩放值情况

        # 获取属性名并转换为小写形式
        prop_name = self.prop.__name__.lower()
        # 构建错误提示信息
        err_text = f"Values for {prop_name} variables with Nominal scale"
        
        # 使用 pytest 引发 TypeError 异常，并匹配错误信息
        with pytest.raises(TypeError, match=err_text):
            self.prop().get_mapping(Nominal("abc"), cat_vector)


class TestAlpha(IntervalBase):
    # 测试 Alpha 属性的间隔基类

    prop = Alpha


class TestLineWidth(IntervalBase):
    # 测试 LineWidth 属性的间隔基类

    prop = LineWidth

    def test_rcparam_default(self):
        # 测试 rc 参数默认设置

        # 使用 mpl.rc_context 上下文设置 {"lines.linewidth": 2}
        with mpl.rc_context({"lines.linewidth": 2}):
            # 断言 prop().default_range 等于 (1, 4)
            assert self.prop().default_range == (1, 4)


class TestEdgeWidth(IntervalBase):
    # 测试 EdgeWidth 属性的间隔基类

    prop = EdgeWidth

    def test_rcparam_default(self):
        # 测试 rc 参数默认设置

        # 使用 mpl.rc_context 上下文设置 {"patch.linewidth": 2}
        with mpl.rc_context({"patch.linewidth": 2}):
            # 断言 prop().default_range 等于 (1, 4)
            assert self.prop().default_range == (1, 4)


class TestPointSize(IntervalBase):
    # 测试 PointSize 属性的间隔基类

    prop = PointSize
    # 定义一个测试方法，用于测试数值类型的区域缩放映射
    def test_areal_scaling_numeric(self, num_vector):
        
        # 设置数值限制范围
        limits = 5, 10
        # 创建一个连续型的缩放对象
        scale = Continuous(limits)
        # 调用 self.prop() 获取属性，然后使用 scale 和 num_vector 获取映射
        mapping = self.prop().get_mapping(scale, num_vector)
        # 生成一个包含 6 个元素的等间距数值数组
        x = np.linspace(0, 1, 6)
        # 根据数值限制计算预期的映射结果
        expected = np.sqrt(np.linspace(*np.square(limits), num=len(x)))
        # 断言映射函数 mapping(x) 的输出与预期结果 expected 相等
        assert_array_equal(mapping(x), expected)

    # 定义一个测试方法，用于测试分类类型的区域缩放映射
    def test_areal_scaling_categorical(self, cat_vector):
        
        # 设置分类限制范围
        limits = (2, 4)
        # 创建一个名义型的缩放对象
        scale = Nominal(limits)
        # 调用 self.prop() 获取属性，然后使用 scale 和 cat_vector 获取映射
        mapping = self.prop().get_mapping(scale, cat_vector)
        # 断言映射函数 mapping(np.arange(3)) 的输出与预期结果数组相等
        assert_array_equal(mapping(np.arange(3)), [4, np.sqrt(10), 2])
```