# `MetaGPT\tests\metagpt\serialize_deserialize\test_write_docstring.py`

```

# 设置文件编码为 utf-8
# 描述：导入 pytest 模块和 WriteDocstring 类
import pytest
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

# 使用 pytest.mark.asyncio 标记为异步测试
# 使用 pytest.mark.parametrize 定义参数化测试
# 测试不同的风格和部分
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("style", "part"),
    [
        ("google", "Args:"),
        ("numpy", "Parameters"),
        ("sphinx", ":param name:"),
    ],
    ids=["google", "numpy", "sphinx"],
)
# 定义测试函数
async def test_action_deserialize(style: str, part: str):
    # 创建 WriteDocstring 实例
    action = WriteDocstring()
    # 序列化实例数据
    serialized_data = action.model_dump()

    # 断言序列化后的数据中包含 "name" 键
    assert "name" in serialized_data
    # 断言序列化后的数据中的描述为 "Write docstring for code."
    assert serialized_data["desc"] == "Write docstring for code."

    # 创建新的 WriteDocstring 实例，传入序列化后的数据
    new_action = WriteDocstring(**serialized_data)

    # 断言新实例的名称和描述
    assert new_action.name == "WriteDocstring"
    assert new_action.desc == "Write docstring for code."
    # 运行新实例的 run 方法，传入代码和风格参数
    ret = await new_action.run(code, style=style)
    # 断言返回结果中包含指定部分
    assert part in ret

```