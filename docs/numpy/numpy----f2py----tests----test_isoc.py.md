# `.\numpy\numpy\f2py\tests\test_isoc.py`

```
# 从当前目录导入util模块
from . import util
# 导入numpy库，并重命名为np
import numpy as np
# 导入pytest库
import pytest
# 从numpy.testing模块导入assert_allclose函数
from numpy.testing import assert_allclose

# 定义一个测试类TestISOC，继承自util.F2PyTest
class TestISOC(util.F2PyTest):
    # 定义sources属性，包含一个源文件路径的列表
    sources = [
        util.getpath("tests", "src", "isocintrin", "isoCtests.f90"),
    ]

    # 标记为gh-24553的测试方法，用于测试c_double函数
    @pytest.mark.slow
    def test_c_double(self):
        # 调用self.module.coddity.c_add方法，计算1加2的结果
        out = self.module.coddity.c_add(1, 2)
        # 预期输出为3
        exp_out = 3
        # 断言计算结果与预期输出相等
        assert out == exp_out

    # 标记为gh-9693的测试方法，用于测试bindc_function函数
    def test_bindc_function(self):
        # 调用self.module.coddity.wat方法，计算1和20的结果
        out = self.module.coddity.wat(1, 20)
        # 预期输出为8
        exp_out = 8
        # 断言计算结果与预期输出相等
        assert out == exp_out

    # 标记为gh-25207的测试方法，用于测试bindc_kinds函数
    def test_bindc_kinds(self):
        # 调用self.module.coddity.c_add_int64方法，计算1和20的结果
        out = self.module.coddity.c_add_int64(1, 20)
        # 预期输出为21
        exp_out = 21
        # 断言计算结果与预期输出相等
        assert out == exp_out

    # 标记为gh-25207的测试方法，用于测试bindc_add_arr函数
    def test_bindc_add_arr(self):
        # 创建一个包含1, 2, 3的numpy数组a
        a = np.array([1,2,3])
        # 创建一个包含1, 2, 3的numpy数组b
        b = np.array([1,2,3])
        # 调用self.module.coddity.add_arr方法，将数组a和b相加
        out = self.module.coddity.add_arr(a, b)
        # 预期输出为a的每个元素乘以2
        exp_out = a * 2
        # 使用assert_allclose函数断言计算结果与预期输出在允许误差范围内相等
        assert_allclose(out, exp_out)


# 定义测试函数test_process_f2cmap_dict
def test_process_f2cmap_dict():
    # 从numpy.f2py.auxfuncs模块导入process_f2cmap_dict函数
    from numpy.f2py.auxfuncs import process_f2cmap_dict

    # 定义f2cmap_all字典，包含一个键为"integer"，值为{"8": "rubbish_type"}的字典
    f2cmap_all = {"integer": {"8": "rubbish_type"}}
    # 定义new_map字典，包含一个键为"INTEGER"，值为{"4": "int"}的字典
    new_map = {"INTEGER": {"4": "int"}}
    # 定义c2py_map字典，包含一个键为"int"，值为"int"，一个键为"rubbish_type"，值为"long"的字典
    c2py_map = {"int": "int", "rubbish_type": "long"}

    # 定义期望的exp_map和exp_maptyp结果
    exp_map, exp_maptyp = ({"integer": {"8": "rubbish_type", "4": "int"}}, ["int"])

    # 调用process_f2cmap_dict函数，传入f2cmap_all、new_map和c2py_map作为参数，获取结果
    res_map, res_maptyp = process_f2cmap_dict(f2cmap_all, new_map, c2py_map)

    # 使用assert断言确保计算结果与期望输出相等
    assert res_map == exp_map
    assert res_maptyp == exp_maptyp
```