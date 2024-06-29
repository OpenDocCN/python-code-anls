# `.\numpy\numpy\lib\tests\test__version.py`

```py
"""Tests for the NumpyVersion class.

"""
# 从numpy.testing中导入assert_和assert_raises函数
from numpy.testing import assert_, assert_raises
# 从numpy.lib中导入NumpyVersion类
from numpy.lib import NumpyVersion


# 定义测试函数test_main_versions
def test_main_versions():
    # 断言NumpyVersion('1.8.0')等于字符串'1.8.0'
    assert_(NumpyVersion('1.8.0') == '1.8.0')
    # 遍历列表，断言NumpyVersion('1.8.0')小于列表中的版本字符串
    for ver in ['1.9.0', '2.0.0', '1.8.1', '10.0.1']:
        assert_(NumpyVersion('1.8.0') < ver)
    # 遍历列表，断言NumpyVersion('1.8.0')大于列表中的版本字符串
    for ver in ['1.7.0', '1.7.1', '0.9.9']:
        assert_(NumpyVersion('1.8.0') > ver)


# 定义测试函数test_version_1_point_10
def test_version_1_point_10():
    # regression test for gh-2998.
    # 断言NumpyVersion('1.9.0')小于字符串'1.10.0'
    assert_(NumpyVersion('1.9.0') < '1.10.0')
    # 断言NumpyVersion('1.11.0')小于字符串'1.11.1'
    assert_(NumpyVersion('1.11.0') < '1.11.1')
    # 断言NumpyVersion('1.11.0')等于字符串'1.11.0'
    assert_(NumpyVersion('1.11.0') == '1.11.0')
    # 断言NumpyVersion('1.99.11')小于字符串'1.99.12'
    assert_(NumpyVersion('1.99.11') < '1.99.12')


# 定义测试函数test_alpha_beta_rc
def test_alpha_beta_rc():
    # 断言NumpyVersion('1.8.0rc1')等于字符串'1.8.0rc1'
    assert_(NumpyVersion('1.8.0rc1') == '1.8.0rc1')
    # 遍历列表，断言NumpyVersion('1.8.0rc1')小于列表中的版本字符串
    for ver in ['1.8.0', '1.8.0rc2']:
        assert_(NumpyVersion('1.8.0rc1') < ver)
    # 遍历列表，断言NumpyVersion('1.8.0rc1')大于列表中的版本字符串
    for ver in ['1.8.0a2', '1.8.0b3', '1.7.2rc4']:
        assert_(NumpyVersion('1.8.0rc1') > ver)
    # 断言NumpyVersion('1.8.0b1')大于字符串'1.8.0a2'
    assert_(NumpyVersion('1.8.0b1') > '1.8.0a2')


# 定义测试函数test_dev_version
def test_dev_version():
    # 断言NumpyVersion('1.9.0.dev-Unknown')小于字符串'1.9.0'
    assert_(NumpyVersion('1.9.0.dev-Unknown') < '1.9.0')
    # 遍历列表，断言NumpyVersion('1.9.0.dev-f16acvda')小于列表中的版本字符串
    for ver in ['1.9.0', '1.9.0a1', '1.9.0b2', '1.9.0b2.dev-ffffffff']:
        assert_(NumpyVersion('1.9.0.dev-f16acvda') < ver)
    # 断言NumpyVersion('1.9.0.dev-f16acvda')等于字符串'1.9.0.dev-11111111'
    assert_(NumpyVersion('1.9.0.dev-f16acvda') == '1.9.0.dev-11111111')


# 定义测试函数test_dev_a_b_rc_mixed
def test_dev_a_b_rc_mixed():
    # 断言NumpyVersion('1.9.0a2.dev-f16acvda')等于字符串'1.9.0a2.dev-11111111'
    assert_(NumpyVersion('1.9.0a2.dev-f16acvda') == '1.9.0a2.dev-11111111')
    # 断言NumpyVersion('1.9.0a2.dev-6acvda54')小于字符串'1.9.0a2'
    assert_(NumpyVersion('1.9.0a2.dev-6acvda54') < '1.9.0a2')


# 定义测试函数test_dev0_version
def test_dev0_version():
    # 断言NumpyVersion('1.9.0.dev0+Unknown')小于字符串'1.9.0'
    assert_(NumpyVersion('1.9.0.dev0+Unknown') < '1.9.0')
    # 遍历列表，断言NumpyVersion('1.9.0.dev0+f16acvda')小于列表中的版本字符串
    for ver in ['1.9.0', '1.9.0a1', '1.9.0b2', '1.9.0b2.dev0+ffffffff']:
        assert_(NumpyVersion('1.9.0.dev0+f16acvda') < ver)
    # 断言NumpyVersion('1.9.0.dev0+f16acvda')等于字符串'1.9.0.dev0+11111111'
    assert_(NumpyVersion('1.9.0.dev0+f16acvda') == '1.9.0.dev0+11111111')


# 定义测试函数test_dev0_a_b_rc_mixed
def test_dev0_a_b_rc_mixed():
    # 断言NumpyVersion('1.9.0a2.dev0+f16acvda')等于字符串'1.9.0a2.dev0+11111111'
    assert_(NumpyVersion('1.9.0a2.dev0+f16acvda') == '1.9.0a2.dev0+11111111')
    # 断言NumpyVersion('1.9.0a2.dev0+6acvda54')小于字符串'1.9.0a2'
    assert_(NumpyVersion('1.9.0a2.dev0+6acvda54') < '1.9.0a2')


# 定义测试函数test_raises
def test_raises():
    # 遍历列表，对于每个版本字符串，断言调用NumpyVersion(ver)会引发ValueError异常
    for ver in ['1.9', '1,9.0', '1.7.x']:
        assert_raises(ValueError, NumpyVersion, ver)
```