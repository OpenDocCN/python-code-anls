# `MetaGPT\tests\metagpt\actions\test_write_docstring.py`

```py

# 导入 pytest 模块
import pytest

# 导入 WriteDocstring 类
from metagpt.actions.write_docstring import WriteDocstring

# 定义待测试的代码
code = """
def add_numbers(a: int, b: int):
    return a + b


class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def greet(self):
        return f"Hello, my name is {self.name} and I am {self.age} years old."
"""

# 使用 pytest.mark.asyncio 标记异步测试
@pytest.mark.asyncio
# 参数化测试用例
@pytest.mark.parametrize(
    ("style", "part"),
    [
        ("google", "Args:"),
        ("numpy", "Parameters"),
        ("sphinx", ":param name:"),
    ],
    ids=["google", "numpy", "sphinx"],
)
# 异步测试函数
async def test_write_docstring(style: str, part: str):
    # 运行 WriteDocstring 类的 run 方法，生成文档字符串
    ret = await WriteDocstring().run(code, style=style)
    # 断言文档字符串中包含特定部分
    assert part in ret

# 使用 pytest.mark.asyncio 标记异步测试
@pytest.mark.asyncio
# 异步测试函数
async def test_write():
    # 调用 WriteDocstring 类的 write_docstring 方法，生成当前文件的文档字符串
    code = await WriteDocstring.write_docstring(__file__)
    # 断言生成的文档字符串不为空
    assert code

# 如果当前文件被直接执行
if __name__ == "__main__":
    # 运行 pytest 测试
    pytest.main([__file__, "-s"])

```