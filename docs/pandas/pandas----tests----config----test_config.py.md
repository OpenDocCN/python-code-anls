# `D:\src\scipysrc\pandas\pandas\tests\config\test_config.py`

```
import pytest  # 导入 pytest 模块

from pandas._config import config as cf  # 导入 pandas 库中的 _config/config 模块，并将其重命名为 cf
from pandas._config.config import OptionError  # 从 pandas 库中的 _config/config 模块导入 OptionError 类

import pandas as pd  # 导入 pandas 库并重命名为 pd
import pandas._testing as tm  # 导入 pandas 库中的 _testing 模块并重命名为 tm


class TestConfig:  # 定义测试类 TestConfig
    @pytest.fixture(autouse=True)  # 定义自动使用的测试夹具
    def clean_config(self, monkeypatch):  # 定义用于清理配置的方法 clean_config，并接收 monkeypatch 参数
        with monkeypatch.context() as m:  # 使用 monkeypatch 创建上下文 m
            m.setattr(cf, "_global_config", {})  # 设置 cf 模块的 _global_config 属性为空字典
            m.setattr(cf, "options", cf.DictWrapper(cf._global_config))  # 设置 cf 模块的 options 属性为 cf._global_config 的字典封装器
            m.setattr(cf, "_deprecated_options", {})  # 设置 cf 模块的 _deprecated_options 属性为空字典
            m.setattr(cf, "_registered_options", {})  # 设置 cf 模块的 _registered_options 属性为空字典

            # 我们在 conftest.py 中的测试夹具设置了 "chained_assignment" 为 "raise"
            # 但是在完成这些设置之后，不再存在 "chained_assignment" 选项，因此重新注册它。
            cf.register_option("chained_assignment", "raise")  # 注册 "chained_assignment" 选项为 "raise"
            yield  # 返回当前配置状态

    def test_api(self):  # 定义测试 API 的方法
        # pandas 对象暴露了用户 API
        assert hasattr(pd, "get_option")  # 断言 pd 对象有 get_option 方法
        assert hasattr(pd, "set_option")  # 断言 pd 对象有 set_option 方法
        assert hasattr(pd, "reset_option")  # 断言 pd 对象有 reset_option 方法
        assert hasattr(pd, "describe_option")  # 断言 pd 对象有 describe_option 方法

    def test_is_one_of_factory(self):  # 定义测试 is_one_of_factory 方法
        v = cf.is_one_of_factory([None, 12])  # 使用 cf.is_one_of_factory 创建 v 对象，可接受 None 和 12 作为参数

        v(12)  # 调用 v 对象，传入参数 12
        v(None)  # 调用 v 对象，传入参数 None
        msg = r"Value must be one of None\|12"  # 设置异常消息的正则表达式
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言捕获 ValueError 异常，并匹配指定消息
            v(1.1)  # 调用 v 对象，传入参数 1.1

    def test_register_option(self):  # 定义测试 register_option 方法
        cf.register_option("a", 1, "doc")  # 注册名为 "a" 的选项，初始值为 1，文档字符串为 "doc"

        # 不能注册已经注册过的选项
        msg = "Option 'a' has already been registered"  # 设置异常消息
        with pytest.raises(OptionError, match=msg):  # 使用 pytest 断言捕获 OptionError 异常，并匹配指定消息
            cf.register_option("a", 1, "doc")  # 再次尝试注册名为 "a" 的选项

        # 不能注册已经注册过的选项路径前缀
        msg = "Path prefix to option 'a' is already an option"  # 设置异常消息
        with pytest.raises(OptionError, match=msg):  # 使用 pytest 断言捕获 OptionError 异常，并匹配指定消息
            cf.register_option("a.b.c.d1", 1, "doc")  # 尝试注册包含已注册选项前缀的新选项
        with pytest.raises(OptionError, match=msg):  # 使用 pytest 断言捕获 OptionError 异常，并匹配指定消息
            cf.register_option("a.b.c.d2", 1, "doc")  # 再次尝试注册包含已注册选项前缀的新选项

        # 不可使用 Python 关键字作为选项名
        msg = "for is a python keyword"  # 设置异常消息
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言捕获 ValueError 异常，并匹配指定消息
            cf.register_option("for", 0)  # 尝试注册名为 "for" 的选项
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言捕获 ValueError 异常，并匹配指定消息
            cf.register_option("a.for.b", 0)  # 尝试注册包含已注册选项前缀的新选项
        # 必须是有效的标识符（确保属性访问正常工作）
        msg = "oh my goddess! is not a valid identifier"  # 设置异常消息
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言捕获 ValueError 异常，并匹配指定消息
            cf.register_option("Oh my Goddess!", 0)  # 尝试注册名为 "Oh my Goddess!" 的选项

        # 我们可以在多层次深度上注册选项，而无需预定义中间步骤
        # 并且可以在同一命名空间中定义不同命名的选项
        cf.register_option("k.b.c.d1", 1, "doc")  # 注册多层级选项 "k.b.c.d1"
        cf.register_option("k.b.c.d2", 1, "doc")  # 注册多层级选项 "k.b.c.d2"
    # 定义测试用例：描述选项功能测试
    def test_describe_option(self):
        # 注册选项 'a'，初始值为 1，文档描述为 "doc"
        cf.register_option("a", 1, "doc")
        # 注册选项 'b'，初始值为 1，文档描述为 "doc2"
        cf.register_option("b", 1, "doc2")
        # 废弃选项 'b'
        cf.deprecate_option("b")

        # 注册选项 'c.d.e1'，初始值为 1，文档描述为 "doc3"
        cf.register_option("c.d.e1", 1, "doc3")
        # 注册选项 'c.d.e2'，初始值为 1，文档描述为 "doc4"
        cf.register_option("c.d.e2", 1, "doc4")
        # 注册选项 'f'，初始值为 1，无文档描述
        cf.register_option("f", 1)
        # 注册选项 'g.h'，初始值为 1，无文档描述
        cf.register_option("g.h", 1)
        # 注册选项 'k'，初始值为 2，文档描述为空
        cf.register_option("k", 2)
        # 将选项 'g.h' 标记为废弃，关联键为 'k'
        cf.deprecate_option("g.h", rkey="k")
        # 注册选项 'l'，初始值为 "foo"，无文档描述
        cf.register_option("l", "foo")

        # 当试图描述不存在的键时，引发 OptionError 异常，异常信息包含 "No such key(s)"
        msg = r"No such keys\(s\)"
        with pytest.raises(OptionError, match=msg):
            cf.describe_option("no.such.key")

        # 可以获取任何已注册键的描述信息
        assert "doc" in cf.describe_option("a", _print_desc=False)
        assert "doc2" in cf.describe_option("b", _print_desc=False)
        assert "precated" in cf.describe_option("b", _print_desc=False)
        assert "doc3" in cf.describe_option("c.d.e1", _print_desc=False)
        assert "doc4" in cf.describe_option("c.d.e2", _print_desc=False)

        # 对于未指定文档描述的选项，返回默认消息 "description not available"
        assert "available" in cf.describe_option("f", _print_desc=False)
        assert "available" in cf.describe_option("g.h", _print_desc=False)
        assert "precated" in cf.describe_option("g.h", _print_desc=False)
        assert "k" in cf.describe_option("g.h", _print_desc=False)

        # 默认值被报告
        assert "foo" in cf.describe_option("l", _print_desc=False)
        # 当前值被报告
        assert "bar" not in cf.describe_option("l", _print_desc=False)
        # 设置选项 'l' 的当前值为 "bar"
        cf.set_option("l", "bar")
        assert "bar" in cf.describe_option("l", _print_desc=False)
    # 测试设置单个选项的方法
    def test_set_option(self):
        # 注册选项 "a"，初始值为 1，文档字符串为 "doc"
        cf.register_option("a", 1, "doc")
        # 注册选项 "b.c"，初始值为 "hullo"，文档字符串为 "doc2"
        cf.register_option("b.c", "hullo", "doc2")
        # 注册选项 "b.b"，初始值为 None，文档字符串为 "doc2"
        cf.register_option("b.b", None, "doc2")

        # 断言获取选项 "a" 的值为 1
        assert cf.get_option("a") == 1
        # 断言获取选项 "b.c" 的值为 "hullo"
        assert cf.get_option("b.c") == "hullo"
        # 断言获取选项 "b.b" 的值为 None
        assert cf.get_option("b.b") is None

        # 设置选项 "a" 的值为 2
        cf.set_option("a", 2)
        # 设置选项 "b.c" 的值为 "wurld"
        cf.set_option("b.c", "wurld")
        # 设置选项 "b.b" 的值为 1.1
        cf.set_option("b.b", 1.1)

        # 断言获取选项 "a" 的值为 2
        assert cf.get_option("a") == 2
        # 断言获取选项 "b.c" 的值为 "wurld"
        assert cf.get_option("b.c") == "wurld"
        # 断言获取选项 "b.b" 的值为 1.1
        assert cf.get_option("b.b") == 1.1

        # 定义匹配消息，用于断言抛出 OptionError 异常
        msg = r"No such keys\(s\): 'no.such.key'"
        # 断言设置不存在的选项时抛出 OptionError 异常，并匹配错误消息
        with pytest.raises(OptionError, match=msg):
            cf.set_option("no.such.key", None)

    # 测试设置选项时空参数的情况
    def test_set_option_empty_args(self):
        # 定义错误消息，用于断言抛出 ValueError 异常
        msg = "Must provide an even number of non-keyword arguments"
        # 断言调用 set_option() 时抛出 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            cf.set_option()

    # 测试设置选项时参数数量不匹配的情况
    def test_set_option_uneven_args(self):
        # 定义错误消息，用于断言抛出 ValueError 异常
        msg = "Must provide an even number of non-keyword arguments"
        # 断言调用 set_option() 时参数数量不匹配时抛出 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            cf.set_option("a.b", 2, "b.c")

    # 测试设置选项时使用非法单个参数类型的情况
    def test_set_option_invalid_single_argument_type(self):
        # 定义错误消息，用于断言抛出 ValueError 异常
        msg = "Must provide an even number of non-keyword arguments"
        # 断言调用 set_option() 时使用非法单个参数类型时抛出 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            cf.set_option(2)

    # 测试同时设置多个选项的方法
    def test_set_option_multiple(self):
        # 注册选项 "a"，初始值为 1，文档字符串为 "doc"
        cf.register_option("a", 1, "doc")
        # 注册选项 "b.c"，初始值为 "hullo"，文档字符串为 "doc2"
        cf.register_option("b.c", "hullo", "doc2")
        # 注册选项 "b.b"，初始值为 None，文档字符串为 "doc2"
        cf.register_option("b.b", None, "doc2")

        # 断言获取选项 "a" 的值为 1
        assert cf.get_option("a") == 1
        # 断言获取选项 "b.c" 的值为 "hullo"
        assert cf.get_option("b.c") == "hullo"
        # 断言获取选项 "b.b" 的值为 None
        assert cf.get_option("b.b") is None

        # 设置选项 "a" 的值为 "2"，选项 "b.c" 的值为 None，选项 "b.b" 的值为 10.0
        cf.set_option("a", "2", "b.c", None, "b.b", 10.0)

        # 断言获取选项 "a" 的值为 "2"
        assert cf.get_option("a") == "2"
        # 断言获取选项 "b.c" 的值为 None
        assert cf.get_option("b.c") is None
        # 断言获取选项 "b.b" 的值为 10.0
        assert cf.get_option("b.b") == 10.0
    # 定义一个测试方法，用于验证配置选项的注册和设置功能
    def test_validation(self):
        # 注册选项 "a"，默认值为 1，文档说明为 "doc"，验证器为 is_int
        cf.register_option("a", 1, "doc", validator=cf.is_int)
        # 注册选项 "d"，默认值为 1，文档说明为 "doc"，验证器为 is_nonnegative_int
        cf.register_option("d", 1, "doc", validator=cf.is_nonnegative_int)
        # 注册选项 "b.c"，默认值为 "hullo"，文档说明为 "doc2"，验证器为 is_text
        cf.register_option("b.c", "hullo", "doc2", validator=cf.is_text)

        # 定义错误消息
        msg = "Value must have type '<class 'int'>'"
        # 使用 pytest 的断言，验证注册时的异常情况，期望抛出 ValueError 异常，且异常消息符合预期
        with pytest.raises(ValueError, match=msg):
            cf.register_option("a.b.c.d2", "NO", "doc", validator=cf.is_int)

        # 设置选项 "a" 的值为 2，期望值为整数
        cf.set_option("a", 2)  # int is_int
        # 设置选项 "b.c" 的值为 "wurld"，期望值为字符串
        cf.set_option("b.c", "wurld")  # str is_str
        # 设置选项 "d" 的值为 2
        cf.set_option("d", 2)
        # 设置选项 "d" 的值为 None，非负整数可以是 None
        cf.set_option("d", None)  # non-negative int can be None

        # 定义错误消息
        msg = "Value must have type '<class 'int'>'"
        # 使用 pytest 的断言，验证设置时的异常情况，期望抛出 ValueError 异常，且异常消息符合预期
        with pytest.raises(ValueError, match=msg):
            cf.set_option("a", None)  # None not is_int
        with pytest.raises(ValueError, match=msg):
            cf.set_option("a", "ab")  # None not is_int

        # 定义错误消息
        msg = "Value must be a nonnegative integer or None"
        # 使用 pytest 的断言，验证注册时的异常情况，期望抛出 ValueError 异常，且异常消息符合预期
        with pytest.raises(ValueError, match=msg):
            cf.register_option("a.b.c.d3", "NO", "doc", validator=cf.is_nonnegative_int)
        with pytest.raises(ValueError, match=msg):
            cf.register_option("a.b.c.d3", -2, "doc", validator=cf.is_nonnegative_int)

        # 定义错误消息
        msg = r"Value must be an instance of <class 'str'>\|<class 'bytes'>"
        # 使用 pytest 的断言，验证设置时的异常情况，期望抛出 ValueError 异常，且异常消息符合预期
        with pytest.raises(ValueError, match=msg):
            cf.set_option("b.c", 1)

        # 创建验证器，验证值是否为 None 或者是 callable
        validator = cf.is_one_of_factory([None, cf.is_callable])
        # 注册选项 "b"，默认值为 lambda 函数返回 None，文档说明为 "doc"，验证器为 validator
        cf.register_option("b", lambda: None, "doc", validator=validator)
        # 设置选项 "b" 的值为格式化字符串 "%.1f"，期望值为可调用对象
        cf.set_option("b", "%.1f".format)  # Formatter is callable
        # 设置选项 "b" 的值为 None，期望值为 None（默认）
        cf.set_option("b", None)  # Formatter is none (default)
        # 使用 pytest 的断言，验证设置时的异常情况，期望抛出 ValueError 异常，且异常消息符合预期
        with pytest.raises(ValueError, match="Value must be a callable"):
            cf.set_option("b", "%.1f")

    # 定义一个测试方法，用于验证配置选项的重置功能
    def test_reset_option(self):
        # 注册选项 "a"，默认值为 1，文档说明为 "doc"，验证器为 is_int
        cf.register_option("a", 1, "doc", validator=cf.is_int)
        # 注册选项 "b.c"，默认值为 "hullo"，文档说明为 "doc2"，验证器为 is_str
        cf.register_option("b.c", "hullo", "doc2", validator=cf.is_str)
        
        # 使用断言，验证获取选项 "a" 的值为 1
        assert cf.get_option("a") == 1
        # 使用断言，验证获取选项 "b.c" 的值为 "hullo"
        assert cf.get_option("b.c") == "hullo"

        # 设置选项 "a" 的值为 2
        cf.set_option("a", 2)
        # 设置选项 "b.c" 的值为 "wurld"
        cf.set_option("b.c", "wurld")
        # 使用断言，验证获取选项 "a" 的值为 2
        assert cf.get_option("a") == 2
        # 使用断言，验证获取选项 "b.c" 的值为 "wurld"
        assert cf.get_option("b.c") == "wurld"

        # 重置选项 "a" 的值
        cf.reset_option("a")
        # 使用断言，验证获取选项 "a" 的值恢复为默认值 1
        assert cf.get_option("a") == 1
        # 使用断言，验证获取选项 "b.c" 的值仍然为 "wurld"
        assert cf.get_option("b.c") == "wurld"

        # 重置选项 "b.c" 的值
        cf.reset_option("b.c")
        # 使用断言，验证获取选项 "a" 的值仍然为默认值 1
        assert cf.get_option("a") == 1
        # 使用断言，验证获取选项 "b.c" 的值恢复为默认值 "hullo"
        assert cf.get_option("b.c") == "hullo"

    # 定义一个测试方法，用于验证配置选项的全部重置功能
    def test_reset_option_all(self):
        # 注册选项 "a"，默认值为 1，文档说明为 "doc"，验证器为 is_int
        cf.register_option("a", 1, "doc", validator=cf.is_int)
        # 注册选项 "b.c"，默认值为 "hullo"，文档说明为 "doc2"，验证器为 is_str
        cf.register_option("b.c", "hullo", "doc2", validator=cf.is_str)
        
        # 使用断言，验证获取选项 "a" 的值为 1
        assert cf.get_option("a") == 1
        # 使用断言，验证获取选项 "b.c" 的值为 "hullo"
        assert cf.get_option("b.c") == "hullo"

        # 设置选项 "a" 的值为 2
        cf.set_option("a", 2)
        # 设置选项 "b.c" 的值为 "wurld"
        cf.set_option("b.c", "wurld")
        # 使用断言，验证获取选项 "a" 的值为 2
        assert cf.get_option("a") == 2
        # 使用断言，验证获取选项 "b.c" 的值为 "wurld"
        assert cf.get_option("b.c") == "wurld"

        # 全部重置配置选项
        cf.reset_option("all")
        # 使用断言，验证获取选项 "a" 的值恢复为默认值 1
        assert cf.get_option("a") == 1
        # 使用断言，验证获取选项 "b.c" 的值恢复为默认值 "hullo"
        assert cf.get_option("b.c") == "hullo"
    # 定义测试函数，用于测试选项弃用功能
    def test_deprecate_option(self):
        # 可以弃用不存在的选项
        cf.deprecate_option("foo")

        # 断言产生 FutureWarning 警告，并检查警告消息中是否包含 "deprecated"
        with tm.assert_produces_warning(FutureWarning, match="deprecated"):
            # 断言引发 KeyError 异常，并检查异常消息中是否包含 "No such keys.s.: 'foo'"
            with pytest.raises(KeyError, match="No such keys.s.: 'foo'"):
                cf.get_option("foo")

        # 注册三个选项，分别包括带验证器的整数选项和无验证器的字符串选项
        cf.register_option("a", 1, "doc", validator=cf.is_int)
        cf.register_option("b.c", "hullo", "doc2")
        cf.register_option("foo", "hullo", "doc2")

        # 弃用选项 "a"，设置移除版本为 "nifty_ver"
        cf.deprecate_option("a", removal_ver="nifty_ver")
        # 断言产生 FutureWarning 警告，并检查警告消息中是否包含 "deprecated.*nifty_ver"
        with tm.assert_produces_warning(FutureWarning, match="eprecated.*nifty_ver"):
            # 获取选项 "a" 的值
            cf.get_option("a")

            # 准备错误消息 "Option 'a' has already been defined as deprecated"
            msg = "Option 'a' has already been defined as deprecated"
            # 断言引发 OptionError 异常，并检查异常消息是否与预期匹配
            with pytest.raises(OptionError, match=msg):
                cf.deprecate_option("a")

        # 弃用选项 "b.c"，提供自定义警告消息 "zounds!"
        cf.deprecate_option("b.c", "zounds!")
        # 断言产生 FutureWarning 警告，并检查警告消息中是否包含 "zounds!"
        with tm.assert_produces_warning(FutureWarning, match="zounds!"):
            # 获取选项 "b.c" 的值
            cf.get_option("b.c")

        # 测试重定向选项键的功能
        # 注册两个带有前缀 "d" 的选项，并分别获取它们的值
        cf.register_option("d.a", "foo", "doc2")
        cf.register_option("d.dep", "bar", "doc2")
        assert cf.get_option("d.a") == "foo"
        assert cf.get_option("d.dep") == "bar"

        # 弃用选项 "d.dep"，将其重定向到选项 "d.a"
        cf.deprecate_option("d.dep", rkey="d.a")
        # 断言产生 FutureWarning 警告，并检查警告消息中是否包含 "deprecated"
        with tm.assert_produces_warning(FutureWarning, match="eprecated"):
            # 获取选项 "d.dep" 的值，应该与选项 "d.a" 的值相同
            assert cf.get_option("d.dep") == "foo"

        # 使用警告功能设置选项 "d.dep" 的新值 "baz"，应该覆盖原来的值 "foo"
        with tm.assert_produces_warning(FutureWarning, match="eprecated"):
            cf.set_option("d.dep", "baz")

        # 断言获取选项 "d.dep" 的值，应该为新设置的值 "baz"
        with tm.assert_produces_warning(FutureWarning, match="eprecated"):
            assert cf.get_option("d.dep") == "baz"

    # 测试配置前缀功能
    def test_config_prefix(self):
        # 使用配置前缀 "base"，注册两个选项 "a" 和 "b"，并分别获取它们的值
        with cf.config_prefix("base"):
            cf.register_option("a", 1, "doc1")
            cf.register_option("b", 2, "doc2")
            assert cf.get_option("a") == 1
            assert cf.get_option("b") == 2

            # 设置选项 "a" 和 "b" 的新值，并再次获取它们的值
            cf.set_option("a", 3)
            cf.set_option("b", 4)
            assert cf.get_option("a") == 3
            assert cf.get_option("b") == 4

        # 断言获取带有前缀 "base" 的选项 "a" 和 "b" 的值
        assert cf.get_option("base.a") == 3
        assert cf.get_option("base.b") == 4

        # 断言选项 "base.a" 和 "base.b" 的描述中包含对应的文档描述
        assert "doc1" in cf.describe_option("base.a", _print_desc=False)
        assert "doc2" in cf.describe_option("base.b", _print_desc=False)

        # 重置选项 "base.a" 和 "base.b" 的值
        cf.reset_option("base.a")
        cf.reset_option("base.b")

        # 使用配置前缀 "base"，再次获取选项 "a" 和 "b" 的值，并进行断言
        with cf.config_prefix("base"):
            assert cf.get_option("a") == 1
            assert cf.get_option("b") == 2
    def test_callback(self):
        # 初始化空列表，用于存储回调函数中的键和值
        k = [None]
        v = [None]

        # 定义回调函数，将键添加到 k 列表，将键对应的选项值添加到 v 列表
        def callback(key):
            k.append(key)
            v.append(cf.get_option(key))

        # 注册选项 'd.a' 和 'd.b'，并指定回调函数为 callback
        cf.register_option("d.a", "foo", cb=callback)
        cf.register_option("d.b", "foo", cb=callback)

        # 删除 k 和 v 列表的最后一个元素（初始化时添加的 None 元素）
        del k[-1], v[-1]

        # 设置选项 'd.a' 的值为 "fooz"，断言回调函数中 k 的最后一个元素为 "d.a"，v 的最后一个元素为 "fooz"
        cf.set_option("d.a", "fooz")
        assert k[-1] == "d.a"
        assert v[-1] == "fooz"

        # 再次删除 k 和 v 列表的最后一个元素
        del k[-1], v[-1]

        # 设置选项 'd.b' 的值为 "boo"，断言回调函数中 k 的最后一个元素为 "d.b"，v 的最后一个元素为 "boo"
        cf.set_option("d.b", "boo")
        assert k[-1] == "d.b"
        assert v[-1] == "boo"

        # 再次删除 k 和 v 列表的最后一个元素
        del k[-1], v[-1]

        # 重置选项 'd.b'，断言回调函数中 k 的最后一个元素为 "d.b"
        cf.reset_option("d.b")
        assert k[-1] == "d.b"

    def test_set_ContextManager(self):
        # 定义函数 eq，用于断言选项 'a' 的值是否等于给定的 val
        def eq(val):
            assert cf.get_option("a") == val

        # 注册选项 'a'，并断言其初始值为 0
        cf.register_option("a", 0)
        eq(0)

        # 使用上下文管理器设置选项 'a' 的值为 15，断言值为 15
        with cf.option_context("a", 15):
            eq(15)

            # 嵌套上下文管理器，设置选项 'a' 的值为 25，断言值为 25
            with cf.option_context("a", 25):
                eq(25)

            # 再次断言选项 'a' 的值为最内层上下文管理器中设置的 15
            eq(15)

        # 最外层上下文管理器结束后，断言选项 'a' 的值回到初始值 0
        eq(0)

        # 设置选项 'a' 的值为 17，断言值为 17
        cf.set_option("a", 17)
        eq(17)

        # 使用选项上下文管理器作为装饰器测试
        @cf.option_context("a", 123)
        def f():
            eq(123)

        # 调用装饰器函数 f，断言选项 'a' 的值为 123
        f()

    def test_attribute_access(self):
        # 初始化列表 holder，用于存储回调函数触发的标志
        holder = []

        # 定义回调函数 f3，将 True 添加到 holder 列表中
        def f3(key):
            holder.append(True)

        # 注册选项 'a' 和 'c'，其中 'c' 的回调函数为 f3
        cf.register_option("a", 0)
        cf.register_option("c", 0, cb=f3)

        # 获取选项对象 options
        options = cf.options

        # 断言选项 'a' 的初始值为 0
        assert options.a == 0

        # 使用上下文管理器设置选项 'a' 的值为 15，断言值为 15
        with cf.option_context("a", 15):
            assert options.a == 15

        # 直接设置 options.a 的值为 500，断言选项 'a' 的值为 500
        options.a = 500
        assert cf.get_option("a") == 500

        # 重置选项 'a'，断言 options.a 的值与 cf.get_option("a") 相同
        cf.reset_option("a")
        assert options.a == cf.get_option("a")

        # 尝试设置不存在的选项 'b' 和 'display'，预期引发 OptionError 异常
        msg = "You can only set the value of existing options"
        with pytest.raises(OptionError, match=msg):
            options.b = 1
        with pytest.raises(OptionError, match=msg):
            options.display = 1

        # 设置选项 'c' 的值为 1，断言 holder 列表长度为 1，即回调函数 f3 被调用一次
        options.c = 1
        assert len(holder) == 1

    def test_option_context_scope(self):
        # 确保创建上下文不会影响当前环境，应该与 'with' 语句一起使用

        # 初始值和上下文值
        original_value = 60
        context_value = 10
        option_name = "a"

        # 注册选项 'a'，初始值为 original_value
        cf.register_option(option_name, original_value)

        # 创建上下文 ctx，设置选项 'a' 的值为 context_value
        ctx = cf.option_context(option_name, context_value)

        # 断言在创建上下文之前，选项 'a' 的值为 original_value
        assert cf.get_option(option_name) == original_value

        # 进入上下文 ctx 后，断言选项 'a' 的值为 context_value
        with ctx:
            assert cf.get_option(option_name) == context_value

        # 离开上下文 ctx 后，再次断言选项 'a' 的值恢复为 original_value
        assert cf.get_option(option_name) == original_value
    def test_dictwrapper_getattr(self):
        options = cf.options
        # 获取配置对象 cf 的 options 属性
        # GH 19789
        # 使用 pytest 断言检查是否引发 OptionError 异常，并匹配异常消息 "No such option"
        with pytest.raises(OptionError, match="No such option"):
            # 尝试访问 options 对象的属性 bananas，预期会引发 OptionError 异常
            options.bananas
        # 断言 options 对象不具有属性 "bananas"
        assert not hasattr(options, "bananas")
```