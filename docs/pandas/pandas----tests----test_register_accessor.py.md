# `D:\src\scipysrc\pandas\pandas\tests\test_register_accessor.py`

```
# 导入所需模块和类
from collections.abc import Generator
import contextlib
import weakref

import pytest  # 导入pytest测试框架

import pandas as pd  # 导入pandas库
import pandas._testing as tm  # 导入pandas测试工具模块
from pandas.core import accessor  # 导入pandas的accessor模块


def test_dirname_mixin() -> None:
    # GH37173
    # 定义一个类X，继承自accessor.DirNamesMixin，测试类属性和实例属性的获取
    class X(accessor.DirNamesMixin):
        x = 1  # 类属性x
        y: int

        def __init__(self) -> None:
            self.z = 3  # 实例属性z

    # 获取X类实例的所有非私有属性名
    result = [attr_name for attr_name in dir(X()) if not attr_name.startswith("_")]

    # 断言获取的属性名与预期结果相符
    assert result == ["x", "z"]


@contextlib.contextmanager
def ensure_removed(obj, attr) -> Generator[None, None, None]:
    """Ensure that an attribute added to 'obj' during the test is
    removed when we're done
    """
    try:
        yield  # 执行测试代码块
    finally:
        try:
            delattr(obj, attr)  # 删除obj对象的指定属性attr
        except AttributeError:
            pass
        obj._accessors.discard(attr)  # 从obj的_accessors集合中移除attr


class MyAccessor:
    def __init__(self, obj) -> None:
        self.obj = obj
        self.item = "item"  # 初始化属性item为字符串"item"

    @property
    def prop(self):
        return self.item  # 返回实例属性item的值作为属性prop的值

    def method(self):
        return self.item  # 返回实例属性item的值作为方法method的返回值


@pytest.mark.parametrize(
    "obj, registrar",
    [
        (pd.Series, pd.api.extensions.register_series_accessor),  # 参数化测试数据：注册Series访问器
        (pd.DataFrame, pd.api.extensions.register_dataframe_accessor),  # 参数化测试数据：注册DataFrame访问器
        (pd.Index, pd.api.extensions.register_index_accessor),  # 参数化测试数据：注册Index访问器
    ],
)
def test_register(obj, registrar):
    with ensure_removed(obj, "mine"):  # 使用ensure_removed上下文管理器确保测试期间添加的属性"mine"在测试结束时被移除
        before = set(dir(obj))  # 记录注册访问器前的对象属性集合
        registrar("mine")(MyAccessor)  # 注册名为"mine"的访问器类MyAccessor
        o = obj([]) if obj is not pd.Series else obj([], dtype=object)  # 创建相应的对象o
        assert o.mine.prop == "item"  # 断言访问器可以正常访问属性prop
        after = set(dir(obj))  # 记录注册访问器后的对象属性集合
        assert (before ^ after) == {"mine"}  # 断言注册访问器后，对象属性集合只多出属性"mine"
        assert "mine" in obj._accessors  # 断言"mine"在obj的_accessors集合中


def test_accessor_works():
    with ensure_removed(pd.Series, "mine"):  # 使用ensure_removed上下文管理器确保测试期间添加的属性"mine"在测试结束时被移除
        pd.api.extensions.register_series_accessor("mine")(MyAccessor)  # 注册名为"mine"的Series访问器类MyAccessor

        s = pd.Series([1, 2])  # 创建Series对象s
        assert s.mine.obj is s  # 断言访问器可以正常访问属性obj
        assert s.mine.prop == "item"  # 断言访问器可以正常访问属性prop
        assert s.mine.method() == "item"  # 断言访问器可以正常调用方法method


def test_overwrite_warns():
    match = r".*MyAccessor.*fake.*Series.*"
    with tm.assert_produces_warning(UserWarning, match=match):  # 使用tm.assert_produces_warning断言产生UserWarning且匹配指定的正则表达式match
        with ensure_removed(pd.Series, "fake"):  # 使用ensure_removed上下文管理器确保测试期间添加的属性"fake"在测试结束时被移除
            setattr(pd.Series, "fake", 123)  # 向pd.Series动态添加属性"fake"
            pd.api.extensions.register_series_accessor("fake")(MyAccessor)  # 注册名为"fake"的Series访问器类MyAccessor
            s = pd.Series([1, 2])  # 创建Series对象s
            assert s.fake.prop == "item"  # 断言访问器可以正常访问属性prop


def test_raises_attribute_error():
    with ensure_removed(pd.Series, "bad"):  # 使用ensure_removed上下文管理器确保测试期间添加的属性"bad"在测试结束时被移除

        @pd.api.extensions.register_series_accessor("bad")  # 注册名为"bad"的Series访问器
        class Bad:
            def __init__(self, data) -> None:
                raise AttributeError("whoops")  # 在初始化时抛出AttributeError异常

        with pytest.raises(AttributeError, match="whoops"):  # 使用pytest.raises断言抛出AttributeError异常且异常信息匹配"whoops"
            pd.Series([], dtype=object).bad  # 访问不存在的属性"bad"


@pytest.mark.parametrize(
    "klass, registrar",
    [
        (pd.Series, pd.api.extensions.register_series_accessor),  # 参数化测试数据：注册Series访问器
        (pd.DataFrame, pd.api.extensions.register_dataframe_accessor),  # 参数化测试数据：注册DataFrame访问器
        (pd.Index, pd.api.extensions.register_index_accessor),  # 参数化测试数据：注册Index访问器
    ],
)
# 测试函数，用于验证在没有循环引用的情况下的行为
def test_no_circular_reference(klass, registrar):
    # 标识：GH 41357，可能是关联的 GitHub issue 或其它问题编号
    # 使用 ensure_removed 上下文管理器，确保在完成操作后 "access" 被移除
    with ensure_removed(klass, "access"):
        # 向注册器注册 "access" 属性并关联 MyAccessor 类
        registrar("access")(MyAccessor)
        # 创建 klass 类的实例 obj，参数为包含单个元素 0 的列表
        obj = klass([0])
        # 创建 obj 的弱引用 ref
        ref = weakref.ref(obj)
        # 断言：obj.access.obj 应该是 obj 自身
        assert obj.access.obj is obj
        # 删除 obj，期望 ref() 返回 None 表示 obj 已被释放
        del obj
        # 断言：ref() 返回 None，表明 obj 已被正确释放
        assert ref() is None
```