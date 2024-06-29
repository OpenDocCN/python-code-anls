# `D:\src\scipysrc\pandas\pandas\tests\test_optional_dependency.py`

```
# 导入标准库模块
import sys
import types

# 导入 pytest 测试框架
import pytest

# 从 pandas.compat._optional 模块中导入必要的函数和常量
from pandas.compat._optional import (
    VERSIONS,
    import_optional_dependency,
)

# 导入 pandas 测试工具模块
import pandas._testing as tm


# 测试导入可选依赖项函数 import_optional_dependency
def test_import_optional():
    # 匹配错误信息的正则表达式
    match = "Missing .*notapackage.* pip .* conda .* notapackage"
    # 检查是否引发 ImportError 异常，并且异常信息要与 match 匹配
    with pytest.raises(ImportError, match=match) as exc_info:
        import_optional_dependency("notapackage")
    # 原始异常应该作为上下文存在：
    assert isinstance(exc_info.value.__context__, ImportError)

    # 测试 errors 参数为 "ignore" 时的返回结果
    result = import_optional_dependency("notapackage", errors="ignore")
    assert result is None


# 测试 xlrd 版本的降级兼容性
def test_xlrd_version_fallback():
    pytest.importorskip("xlrd")
    import_optional_dependency("xlrd")


# 测试不兼容版本的处理
def test_bad_version(monkeypatch):
    # 设置虚拟的模块和版本信息
    name = "fakemodule"
    module = types.ModuleType(name)
    module.__version__ = "0.9.0"
    sys.modules[name] = module
    monkeypatch.setitem(VERSIONS, name, "1.0.0")

    # 匹配错误信息的正则表达式
    match = "Pandas requires .*1.0.0.* of .fakemodule.*'0.9.0'"
    # 检查是否引发 ImportError 异常，并且异常信息要与 match 匹配
    with pytest.raises(ImportError, match=match):
        import_optional_dependency("fakemodule")

    # 测试 min_version 参数为 "0.8" 时的返回结果
    result = import_optional_dependency("fakemodule", min_version="0.8")
    assert result is module

    # 测试 errors 参数为 "warn" 时，是否产生 UserWarning 警告
    with tm.assert_produces_warning(UserWarning, match=match):
        result = import_optional_dependency("fakemodule", errors="warn")
    assert result is None

    # 修改模块版本号为 "1.0.0"，确保精确匹配可以通过
    module.__version__ = "1.0.0"
    result = import_optional_dependency("fakemodule")
    assert result is module

    # 测试 min_version 参数为 "1.1.0" 时，是否引发 ImportError 异常
    with pytest.raises(ImportError, match="Pandas requires version '1.1.0'"):
        import_optional_dependency("fakemodule", min_version="1.1.0")

    # 测试 errors 参数为 "warn"，min_version 为 "1.1.0" 时，是否产生 UserWarning 警告
    with tm.assert_produces_warning(UserWarning, match="Pandas requires version"):
        result = import_optional_dependency(
            "fakemodule", errors="warn", min_version="1.1.0"
        )
    assert result is None

    # 测试 errors 参数为 "ignore"，min_version 为 "1.1.0" 时，返回结果是否为 None
    result = import_optional_dependency(
        "fakemodule", errors="ignore", min_version="1.1.0"
    )
    assert result is None


# 测试子模块导入及版本匹配情况
def test_submodule(monkeypatch):
    # 创建一个带有子模块的虚拟模块
    name = "fakemodule"
    module = types.ModuleType(name)
    module.__version__ = "0.9.0"
    sys.modules[name] = module
    sub_name = "submodule"
    submodule = types.ModuleType(sub_name)
    setattr(module, sub_name, submodule)
    sys.modules[f"{name}.{sub_name}"] = submodule
    monkeypatch.setitem(VERSIONS, name, "1.0.0")

    # 匹配错误信息的正则表达式
    match = "Pandas requires .*1.0.0.* of .fakemodule.*'0.9.0'"
    # 检查是否引发 ImportError 异常，并且异常信息要与 match 匹配
    with pytest.raises(ImportError, match=match):
        import_optional_dependency("fakemodule.submodule")

    # 测试 errors 参数为 "warn" 时，是否产生 UserWarning 警告
    with tm.assert_produces_warning(UserWarning, match=match):
        result = import_optional_dependency("fakemodule.submodule", errors="warn")
    assert result is None

    # 修改模块版本号为 "1.0.0"，确保精确匹配可以通过
    module.__version__ = "1.0.0"
    result = import_optional_dependency("fakemodule.submodule")
    assert result is submodule


# 测试未指定版本时的异常处理
def test_no_version_raises(monkeypatch):
    name = "fakemodule"
    # 创建一个新的模块对象，使用指定的名称
    module = types.ModuleType(name)
    
    # 将新创建的模块对象添加到系统模块字典中，使其可被其他部分引用
    sys.modules[name] = module
    
    # 使用 monkeypatch 模块设置版本字典 VERSIONS 中的指定模块名称的版本号为 "1.0.0"
    monkeypatch.setitem(VERSIONS, name, "1.0.0")
    
    # 使用 pytest 的异常断言上下文，确保导入可选依赖时会触发 ImportError 异常，并且异常信息匹配指定的正则表达式
    with pytest.raises(ImportError, match="Can't determine .* fakemodule"):
        # 导入可选依赖的函数，这里期望它会抛出 ImportError 异常
        import_optional_dependency(name)
```