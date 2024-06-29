# `D:\src\scipysrc\pandas\scripts\tests\test_check_test_naming.py`

```
# 导入 pytest 模块
import pytest

# 从 scripts.check_test_naming 模块中导入 main 函数
from scripts.check_test_naming import main

# 使用 pytest.mark.parametrize 装饰器定义参数化测试
@pytest.mark.parametrize(
    "src, expected_out, expected_ret",
    [
        (
            "def foo(): pass\n",
            "t.py:1:0 found test function which does not start with 'test'\n",
            1,
        ),
        (
            "class Foo:\n    def test_foo(): pass\n",
            "t.py:1:0 found test class which does not start with 'Test'\n",
            1,
        ),
        ("def test_foo(): pass\n", "", 0),
        ("class TestFoo:\n    def test_foo(): pass\n", "", 0),
        (
            "def foo():\n    pass\ndef test_foo():\n    foo()\n",
            "",
            0,
        ),
        (
            "class Foo:  # not a test\n"
            "    pass\n"
            "def test_foo():\n"
            "    Class.foo()\n",
            "",
            0,
        ),
        ("@pytest.fixture\ndef foo(): pass\n", "", 0),
        ("@pytest.fixture()\ndef foo(): pass\n", "", 0),
        ("@register_extension_dtype\nclass Foo: pass\n", "", 0),
    ],
)
# 定义测试函数 test_main，接受参数 src, expected_out, expected_ret
def test_main(capsys, src, expected_out, expected_ret):
    # 调用 main 函数进行测试
    ret = main(src, "t.py")
    # 读取标准输出和标准错误
    out, _ = capsys.readouterr()
    # 断言标准输出符合预期
    assert out == expected_out
    # 断言返回值符合预期
    assert ret == expected_ret
```