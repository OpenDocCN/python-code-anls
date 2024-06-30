# `D:\src\scipysrc\seaborn\tests\test_docstrings.py`

```
from seaborn._docstrings import DocstringComponents  # 导入 DocstringComponents 类

# 示例字典，包含参数说明的文本
EXAMPLE_DICT = dict(
    param_a="""
a : str
    The first parameter.
    """,
)

# 定义一个示例类 ExampleClass
class ExampleClass:
    def example_method(self):
        """An example method.

        Parameters
        ----------
        a : str
           A method parameter.

        """

# 定义一个示例函数 example_func
def example_func():
    """An example function.

    Parameters
    ----------
    a : str
        A function parameter.

    """

# 测试类 TestDocstringComponents
class TestDocstringComponents:

    # 测试从字典创建 DocstringComponents 对象
    def test_from_dict(self):
        obj = DocstringComponents(EXAMPLE_DICT)
        assert obj.param_a == "a : str\n    The first parameter."

    # 测试从嵌套组件创建 DocstringComponents 对象
    def test_from_nested_components(self):
        obj_inner = DocstringComponents(EXAMPLE_DICT)
        obj_outer = DocstringComponents.from_nested_components(inner=obj_inner)
        assert obj_outer.inner.param_a == "a : str\n    The first parameter."

    # 测试从函数创建 DocstringComponents 对象
    def test_from_function(self):
        obj = DocstringComponents.from_function_params(example_func)
        assert obj.a == "a : str\n    A function parameter."

    # 测试从方法创建 DocstringComponents 对象
    def test_from_method(self):
        obj = DocstringComponents.from_function_params(
            ExampleClass.example_method
        )
        assert obj.a == "a : str\n    A method parameter."
```