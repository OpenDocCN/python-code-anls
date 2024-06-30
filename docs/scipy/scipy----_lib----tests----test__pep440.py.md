# `D:\src\scipysrc\scipy\scipy\_lib\tests\test__pep440.py`

```
# 导入 pytest 模块中的 raises 函数并命名为 assert_raises
from pytest import raises as assert_raises
# 从 scipy._lib._pep440 模块导入 Version 类和 parse 函数
from scipy._lib._pep440 import Version, parse


# 定义测试函数 test_main_versions
def test_main_versions():
    # 断言两个 Version 对象相等
    assert Version('1.8.0') == Version('1.8.0')
    # 循环遍历版本列表，断言给定版本小于循环变量表示的版本
    for ver in ['1.9.0', '2.0.0', '1.8.1']:
        assert Version('1.8.0') < Version(ver)

    # 循环遍历版本列表，断言给定版本大于循环变量表示的版本
    for ver in ['1.7.0', '1.7.1', '0.9.9']:
        assert Version('1.8.0') > Version(ver)


# 定义测试函数 test_version_1_point_10
def test_version_1_point_10():
    # regression test for gh-2998.
    # 断言版本 '1.9.0' 小于 '1.10.0'
    assert Version('1.9.0') < Version('1.10.0')
    # 断言版本 '1.11.0' 小于 '1.11.1'
    assert Version('1.11.0') < Version('1.11.1')
    # 断言版本 '1.11.0' 等于 '1.11.0'
    assert Version('1.11.0') == Version('1.11.0')
    # 断言版本 '1.99.11' 小于 '1.99.12'
    assert Version('1.99.11') < Version('1.99.12')


# 定义测试函数 test_alpha_beta_rc
def test_alpha_beta_rc():
    # 断言版本 '1.8.0rc1' 等于 '1.8.0rc1'
    assert Version('1.8.0rc1') == Version('1.8.0rc1')
    # 循环遍历版本列表，断言 '1.8.0rc1' 小于循环变量表示的版本
    for ver in ['1.8.0', '1.8.0rc2']:
        assert Version('1.8.0rc1') < Version(ver)

    # 循环遍历版本列表，断言 '1.8.0rc1' 大于循环变量表示的版本
    for ver in ['1.8.0a2', '1.8.0b3', '1.7.2rc4']:
        assert Version('1.8.0rc1') > Version(ver)

    # 断言版本 '1.8.0b1' 大于 '1.8.0a2'
    assert Version('1.8.0b1') > Version('1.8.0a2')


# 定义测试函数 test_dev_version
def test_dev_version():
    # 断言版本 '1.9.0.dev+Unknown' 小于 '1.9.0'
    assert Version('1.9.0.dev+Unknown') < Version('1.9.0')
    # 循环遍历版本列表，断言 '1.9.0.dev+f16acvda' 小于循环变量表示的版本
    for ver in ['1.9.0', '1.9.0a1', '1.9.0b2', '1.9.0b2.dev+ffffffff', '1.9.0.dev1']:
        assert Version('1.9.0.dev+f16acvda') < Version(ver)

    # 断言版本 '1.9.0.dev+f16acvda' 等于 '1.9.0.dev+f16acvda'
    assert Version('1.9.0.dev+f16acvda') == Version('1.9.0.dev+f16acvda')


# 定义测试函数 test_dev_a_b_rc_mixed
def test_dev_a_b_rc_mixed():
    # 断言版本 '1.9.0a2.dev+f16acvda' 等于 '1.9.0a2.dev+f16acvda'
    assert Version('1.9.0a2.dev+f16acvda') == Version('1.9.0a2.dev+f16acvda')
    # 断言版本 '1.9.0a2.dev+6acvda54' 小于 '1.9.0a2'
    assert Version('1.9.0a2.dev+6acvda54') < Version('1.9.0a2')


# 定义测试函数 test_dev0_version
def test_dev0_version():
    # 断言版本 '1.9.0.dev0+Unknown' 小于 '1.9.0'
    assert Version('1.9.0.dev0+Unknown') < Version('1.9.0')
    # 循环遍历版本列表，断言 '1.9.0.dev0+f16acvda' 小于循环变量表示的版本
    for ver in ['1.9.0', '1.9.0a1', '1.9.0b2', '1.9.0b2.dev0+ffffffff']:
        assert Version('1.9.0.dev0+f16acvda') < Version(ver)

    # 断言版本 '1.9.0.dev0+f16acvda' 等于 '1.9.0.dev0+f16acvda'
    assert Version('1.9.0.dev0+f16acvda') == Version('1.9.0.dev0+f16acvda')


# 定义测试函数 test_dev0_a_b_rc_mixed
def test_dev0_a_b_rc_mixed():
    # 断言版本 '1.9.0a2.dev0+f16acvda' 等于 '1.9.0a2.dev0+f16acvda'
    assert Version('1.9.0a2.dev0+f16acvda') == Version('1.9.0a2.dev0+f16acvda')
    # 断言版本 '1.9.0a2.dev0+6acvda54' 小于 '1.9.0a2'
    assert Version('1.9.0a2.dev0+6acvda54') < Version('1.9.0a2')


# 定义测试函数 test_raises
def test_raises():
    # 循环遍历版本列表，断言 ValueError 被触发，传入版本号 ver 作为参数
    for ver in ['1,9.0', '1.7.x']:
        assert_raises(ValueError, Version, ver)


# 定义测试函数 test_legacy_version
def test_legacy_version():
    # 非 PEP-440 版本标识符始终小于给定的 Version('0.0.0')
    # 对于 NumPy，这仅适用于不支持的开发构建版本小于 1.10.0
    assert parse('invalid') < Version('0.0.0')
    assert parse('1.9.0-f16acvda') < Version('1.0.0')
```