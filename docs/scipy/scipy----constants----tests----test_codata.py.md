# `D:\src\scipysrc\scipy\scipy\constants\tests\test_codata.py`

```
# 导入所需模块和函数
from scipy.constants import find, value, ConstantWarning, c, speed_of_light
from numpy.testing import (assert_equal, assert_, assert_almost_equal,
                           suppress_warnings)
import scipy.constants._codata as _cd

# 定义测试函数，用于测试 find 函数的不同输入情况
def test_find():
    # 测试找到关键词 'weak mixing' 的结果是否正确
    keys = find('weak mixing', disp=False)
    assert_equal(keys, ['weak mixing angle'])

    # 测试找到关键词 'qwertyuiop' 的结果是否为空列表
    keys = find('qwertyuiop', disp=False)
    assert_equal(keys, [])

    # 测试找到关键词 'natural unit' 的结果是否包含特定自然单位
    keys = find('natural unit', disp=False)
    assert_equal(keys, sorted(['natural unit of velocity',
                                'natural unit of action',
                                'natural unit of action in eV s',
                                'natural unit of mass',
                                'natural unit of energy',
                                'natural unit of energy in MeV',
                                'natural unit of momentum',
                                'natural unit of momentum in MeV/c',
                                'natural unit of length',
                                'natural unit of time']))

# 定义测试函数，用于测试 value 函数的使用情况
def test_basic_table_parse():
    # 检查 speed of light 的值是否与常量 c 相等
    c_s = 'speed of light in vacuum'
    assert_equal(value(c_s), c)
    assert_equal(value(c_s), speed_of_light)

# 定义测试函数，用于测试 _cd 模块中的 unit 函数和 c 常量的使用情况
def test_basic_lookup():
    # 检查 speed of light 的数值和单位是否正确
    assert_equal('%d %s' % (_cd.c, _cd.unit('speed of light in vacuum')),
                 '299792458 m s^-1')

# 定义测试函数，用于测试 find 函数在不指定关键词的情况下返回结果的数量
def test_find_all():
    assert_(len(find(disp=False)) > 300)

# 定义测试函数，用于测试 find 函数返回单个结果的情况
def test_find_single():
    # 检查找到关键词 'Wien freq' 的第一个结果是否正确
    assert_equal(find('Wien freq', disp=False)[0],
                 'Wien frequency displacement law constant')

# 定义测试函数，用于测试 value 函数返回精确值的情况
def test_2002_vs_2006():
    # 检查 'magn. flux quantum' 在不同版本中返回的值是否相等
    assert_almost_equal(value('magn. flux quantum'),
                        value('mag. flux quantum'))

# 定义测试函数，用于检查更新存储的精确值是否成功
def test_exact_values():
    # 使用 suppress_warnings 确保不打印常量警告
    with suppress_warnings() as sup:
        sup.filter(ConstantWarning)
        # 遍历 _cd.exact_values 中的所有键，检查更新后的精确值是否正确
        for key in _cd.exact_values:
            assert_((_cd.exact_values[key][0] - value(key)) / value(key) == 0)
```