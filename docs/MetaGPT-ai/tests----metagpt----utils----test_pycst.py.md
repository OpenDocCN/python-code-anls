# `MetaGPT\tests\metagpt\utils\test_pycst.py`

```

# 从metagpt.utils模块中导入pycst工具
from metagpt.utils import pycst

# 定义一个包含函数和类定义的代码字符串
code = """
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import overload

@overload
def add_numbers(a: int, b: int):
    ...

@overload
def add_numbers(a: float, b: float):
    ...

def add_numbers(a: int, b: int):
    return a + b


class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def greet(self):
        return f"Hello, my name is {self.name} and I am {self.age} years old."
"""

# 定义一个包含文档字符串的代码字符串
documented_code = '''
"""
This is an example module containing a function and a class definition.
"""

def add_numbers(a: int, b: int):
    """This function is used to add two numbers and return the result.

    Parameters:
        a: The first integer.
        b: The second integer.

    Returns:
        int: The sum of the two numbers.
    """
    return a + b

class Person:
    """This class represents a person's information, including name and age.

    Attributes:
        name: The person's name.
        age: The person's age.
    """

    def __init__(self, name: str, age: int):
        """Creates a new instance of the Person class.

        Parameters:
            name: The person's name.
            age: The person's age.
        """
        ...

    def greet(self):
        """
        Returns a greeting message including the name and age.

        Returns:
            str: The greeting message.
        """
        ...
'''

# 定义一个包含合并文档字符串的代码字符串
merged_code = '''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is an example module containing a function and a class definition.
"""

from typing import overload

@overload
def add_numbers(a: int, b: int):
    ...

@overload
def add_numbers(a: float, b: float):
    ...

def add_numbers(a: int, b: int):
    """This function is used to add two numbers and return the result.

    Parameters:
        a: The first integer.
        b: The second integer.

    Returns:
        int: The sum of the two numbers.
    """
    return a + b


class Person:
    """This class represents a person's information, including name and age.

    Attributes:
        name: The person's name.
        age: The person's age.
    """
    def __init__(self, name: str, age: int):
        """Creates a new instance of the Person class.

        Parameters:
            name: The person's name.
            age: The person's age.
        """
        self.name = name
        self.age = age

    def greet(self):
        """
        Returns a greeting message including the name and age.

        Returns:
            str: The greeting message.
        """
        return f"Hello, my name is {self.name} and I am {self.age} years old."
'''

# 定义一个测试函数，用于测试合并文档字符串的功能
def test_merge_docstring():
    # 调用pycst工具中的merge_docstring函数，将代码字符串和文档字符串合并
    data = pycst.merge_docstring(code, documented_code)
    # 打印合并后的结果
    print(data)
    # 断言合并后的结果与预期的合并代码字符串相等
    assert data == merged_code

```