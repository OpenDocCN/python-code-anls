# MetaGPT源码解析 17

# `tests/metagpt/utils/test_parse_html.py`

This HTML code is a web page with a heading, paragraph, and a list. It also has a table with some columns and rows. Additionally, it has an image, a form, and a div.

The `Random HTML Example` is a placeholder for displaying some example content that is not related to the HTML code itself.


```py
from metagpt.utils import parse_html

PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Random HTML Example</title>
</head>
<body>
    <h1>This is a Heading</h1>
    <p>This is a paragraph with <a href="test">a link</a> and some <em>emphasized</em> text.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
    </ul>
    <ol>
        <li>Numbered Item 1</li>
        <li>Numbered Item 2</li>
        <li>Numbered Item 3</li>
    </ol>
    <table>
        <tr>
            <th>Header 1</th>
            <th>Header 2</th>
        </tr>
        <tr>
            <td>Row 1, Cell 1</td>
            <td>Row 1, Cell 2</td>
        </tr>
        <tr>
            <td>Row 2, Cell 1</td>
            <td>Row 2, Cell 2</td>
        </tr>
    </table>
    <img src="image.jpg" alt="Sample Image">
    <form action="/submit" method="post">
        <label for="name">Name:</label>
        <input type="text" id="name" name="name" required>
        <label for="email">Email:</label>
        <input type="email" id="email" name="email" required>
        <button type="submit">Submit</button>
    </form>
    <div class="box">
        <p>This is a div with a class "box".</p>
        <p><a href="https://metagpt.com">a link</a></p>
        <p><a href="#section2"></a></p>
        <p><a href="ftp://192.168.1.1:8080"></a></p>
        <p><a href="javascript:alert('Hello');"></a></p>
    </div>
```

这段代码是一个 HTML 页面，其中包含一个带有标题和内容的 div 元素。这个 div 元素被赋予了 class="box"，并且在页面上被一个链接所包围。这个链接的 URL 是 "http://example.com"。

def test_web_page(): 是 defined 函数，用到了 parse_html.WebPage 类来测试这个 HTML 页面。这个函数接受一个 inner_text 参数，用来初始化 HTML 页面的内容。还接受一个 html 参数，用来初始化 HTML 页面的内容，但是这个参数的值是空字符串。最后，函数还有一个 url 参数，用来设置链接的 URL。

函数内部首先使用 parse_html.WebPage 类来解析 HTML 页面，然后使用 assert 语句来验证页面的标题和链接是否符合预期。


```py
</body>
</html>
"""

CONTENT = 'This is a HeadingThis is a paragraph witha linkand someemphasizedtext.Item 1Item 2Item 3Numbered Item 1Numbered '\
'Item 2Numbered Item 3Header 1Header 2Row 1, Cell 1Row 1, Cell 2Row 2, Cell 1Row 2, Cell 2Name:Email:SubmitThis is a div '\
'with a class "box".a link'


def test_web_page():
    page = parse_html.WebPage(inner_text=CONTENT, html=PAGE, url="http://example.com")
    assert page.title == "Random HTML Example"
    assert list(page.get_links()) == ["http://example.com/test", "https://metagpt.com"]


```

这段代码是一个函数 `test_get_page_content()`，用于测试 `parse_html.get_html_content()` 函数获取指定网页的内容并将其存储在 `ret` 变量中，然后使用 `assert` 断言函数检查 `ret` 是否等于 `CONTENT`。

具体来说，这段代码的作用是测试 `parse_html.get_html_content()` 函数是否能够正确地从指定的网页中提取出内容并将其存储在 `ret` 变量中。在这个测试中，我们使用 `assert` 断言函数来确保 `ret` 的值与 `CONTENT` 相等，如果 `ret` 的值与 `CONTENT` 不等，那么函数的行为就会被判定为不正确。


```py
def test_get_page_content():
    ret = parse_html.get_html_content(PAGE, "http://example.com")
    assert ret == CONTENT

```

# `tests/metagpt/utils/test_pycst.py`

这段代码使用了Python中的装饰器（met合金）来定义了两个函数 `add_numbers`，一个是使用 `@overload` 装饰器重载了 `add_numbers` 函数，另一个是定义了两个版本的函数 `add_numbers`。具体来说：

1. @overload 装饰器允许在函数定义中使用 @overload 装饰器，这样就可以在不使用 `@overload` 的前提下，该函数可以被当做普通函数来调用。而重载的装饰器 `@overload`，则允许对传入的函数参数使用 `overload` 装饰器，这样就可以在不支持该函数的情况下，通过装饰器来定义该函数的行为。
2. 在函数内部，没有对参数类型进行限制，因此函数可以接受任意类型的参数。而由于 `@overload` 装饰器的特殊作用，实际上 `add_numbers` 函数的唯一行为是在其定义中给出的，即对传入的参数做加法运算。


```py
from metagpt.utils import pycst

code = '''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import overload

@overload
def add_numbers(a: int, b: int):
    ...

@overload
def add_numbers(a: float, b: float):
    ...

```

这段代码定义了一个名为 `add_numbers` 的函数，它接收两个整数参数 `a` 和 `b`，并返回它们的和。

这段代码定义了一个名为 `Person` 的类，这个类有一个名为 `__init__` 的方法，它接受两个整数参数 `name` 和 `age`。这个方法用于初始化对象的属性。

这段代码还有一段测试代码，用于打印出对象的问候信息。这段测试代码创建了一个 `Person` 对象，然后调用它的 `greet` 方法，最后输出出对象的姓名和年龄。


```py
def add_numbers(a: int, b: int):
    return a + b


class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def greet(self):
        return f"Hello, my name is {self.name} and I am {self.age} years old."
'''

documented_code = '''
"""
```

这段代码定义了一个名为 `add_numbers` 的函数和一个名为 `add_numbers_class` 的类。

函数 `add_numbers` 接收两个整数参数 `a` 和 `b`，并返回它们的和。这个函数的功能是计算两个整数的和，并返回结果。

类 `add_numbers_class` 定义了一个名为 `add_numbers` 的函数，但是没有定义参数 `a` 和 `b`，因为这个函数是在函数内部定义的，不需要在类中定义参数。这个函数与函数 `add_numbers` 具有相同的名称和签名，因此可以认为它们是相同的函数。

函数和类都可以被使用，但必须在调用时提供参数。例如，可以在程序中创建一个 `add_numbers_class` 对象，并调用其 `add_numbers` 函数，以便将两个整数相加并输出结果。


```py
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

```

这段代码定义了一个名为"Person"的类，表示一个人的信息，包括姓名和年龄。这个类有两个主要的属性，分别是姓名和年龄，并且这两个属性都是整数类型(int)。

此外，这个类还定义了一个名为"__init__"的函数，这个函数用于创建一个新的Person实例。这个函数需要传入两个参数，一个是姓名(str类型)，另一个是年龄(int类型)，并且需要在两个参数之间添加一个空格。

另外，这个类还有一个名为"greet"的函数，这个函数用于输出一个问候消息，包括姓名和年龄。这个函数没有参数，并且返回一个字符串类型(str)。在函数内部，似乎没有做任何实际的工作，只是返回了一个字符串而已。


```py
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
```

这段代码定义了一个名为 "merged_code" 的模块，其中包含一个名为 "add\_numbers" 的函数和一个名为 "add\_numbers" 的类定义。

函数部分使用了一种特殊的语法糖，将普通函数的功能转换为类函数的形式。这种语法糖是 "is-implementation" 中的 "lambda" 函数，通过将一个函数引用作为参数，将其包装成一个类的实例，从而将函数转换为类函数。

类定义中包含一个名为 "add\_numbers" 的函数，该函数接收两个整数参数 "a" 和 "b"，没有返回值，只有函数体。函数体内部使用了 "is-implementation" 的语法，将其包装成一个类函数的形式，但是实际上仍然是普通函数的功能。

该代码的作用是提供一个简单的 "add\_numbers" 函数和相应的类定义，用于某些 Python 应用程序中的数学计算。


```py
'''


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

```



这段代码定义了一个名为 `add_numbers` 的函数，并使用了 Python 的类型提示 `@overload` 来重载该函数。重载的函数应该与原始函数具有相同的签名，但可以有不同的实现。

函数的实际实现包括两个参数 `a` 和 `b`，它们都是浮点数或整数。函数返回两个参数的和。

具体来说，`add_numbers` 函数接受两个参数 `a` 和 `b`，它们可以是任意实数类型(包括浮点数和整数)。函数首先检查 `a` 和 `b` 是否都是浮点数，如果是，则执行以下操作：

1. 将两个浮点数相加，并将结果保留为浮点数类型。
2. 如果 `a` 或 `b` 中有一个或多个整数，则将它们转换为浮点数并相加。
3. 返回两个浮点数的和。

如果 `a` 和 `b` 都是整数，则按照上述步骤执行，但结果将保留为整数类型。

函数的实现是使用 Python 的类型提示 `@overload` 来重载的，这意味着 Python 编译器将在编译时检查函数的签名是否与定义的签名完全匹配。如果签名不匹配，编译器将报告错误。


```py
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


```

这段代码定义了一个名为Person的类，表示一个人的信息，包括姓名和年龄。该类有两个方法：构造函数(__init__)和 greet方法，分别用于创建新的Person实例和输出问候消息。

在构造函数中，使用两个参数name和age，分别表示这个人的姓名和年龄，并使用self.name和self.age属性来访问它们。

在greet方法中，使用f-string格式化语法来输出问候消息，其中包含人的姓名和年龄。这种格式化语法将字符串和变量名相互绑定，使代码更易于阅读和理解。


```py
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
```

这段代码定义了一个名为 `test_merge_docstring` 的函数，该函数接受两个参数：`code` 和 `documented_code`。函数内部使用 `pycst.merge_docstring` 函数将 `code` 中的文档字符串与 `documented_code` 中的文档字符串进行合并，并将结果存储在 `data` 变量中。最后，函数打印出 `data` 变量，并使用 `assert` 语句检查 `data` 是否等于 `merged_code`，即两者是否相等。


```py
'''


def test_merge_docstring():
    data = pycst.merge_docstring(code, documented_code)
    print(data)
    assert data == merged_code

```

# `tests/metagpt/utils/test_read_docx.py`

这段代码是一个Python脚本，使用了`metagpt`库来处理特定的Python版本。以下是对脚本的解释：

1. `#!/usr/bin/env python`是脚本的元数据，告诉Python运行脚本时使用哪个Python版本。`-*- coding: utf-8 -*-`是脚本内部的编码提示，告诉Python使用utf-8编码。

2. `from metagpt.const import PROJECT_ROOT`导入`metagpt`库中的`PROJECT_ROOT`常量。

3. `from metagpt.utils.read_document import read_docx`导入`metagpt`库中的`read_document`函数，用于读取docx格式的文档。

4. `class TestReadDocx:`定义一个名为`TestReadDocx`的类。

5. `def test_read_docx(self):`定义了一个名为`test_read_docx`的函数，用于测试读取docx格式的文档。

6. `docx_sample = PROJECT_ROOT / "tests/data/docx_for_test.docx"`导入了`PROJECT_ROOT`和`tests`目录下的`docx_for_test.docx`文件，作为`docx_sample`变量。

7. `docx = read_docx(docx_sample)`使用`read_document`函数读取`docx_sample`中的docx文档，并将其存储在`docx`变量中。

8. `assert len(docx) == 6`使用`assert`语句对`docx`的长度进行断言，如果`docx`的长度等于6，则说明读取的docx文档没有问题。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 16:02
@Author  : alexanderwu
@File    : test_read_docx.py
"""

from metagpt.const import PROJECT_ROOT
from metagpt.utils.read_document import read_docx


class TestReadDocx:
    def test_read_docx(self):
        docx_sample = PROJECT_ROOT / "tests/data/docx_for_test.docx"
        docx = read_docx(docx_sample)
        assert len(docx) == 6

```

# `tests/metagpt/utils/test_serialize.py`

这段代码是一个Python脚本，使用了`unittest`模块的`describe`方法来定义测试用例。其作用是进行一个序列化的测试，以验证一个名为`WritePRD`的类的实现是否符合预期。

具体来说，这段代码的作用是：

1. 从`typing`模块中定义了一个包含两个元素的`List`类型变量`actions`和一个名为`Tuple`的元组类型变量`actionstub_output_schema`；
2. 从`metagpt.actions`模块中引入了`WritePRD`类和相关的`action_output`成员函数；
3. 从`metagpt.schema`模块中引入了`Message`类；
4. 从`metagpt.utils.serialize`模块中引入了`actionoutout_schema_to_mapping`函数，用于将`action_output`的schema映射到`Message`类的相应成员函数；
5. 在测试用例中，通过`describe`方法定义了以下三个测试函数：

  ```py
  def test_write_prd(self):
      #: 准备参数
      input_data = '{"actor_id": 1, "action": "write_prd", "message": {"type": "test_message", "data": "test_input"}}'
      expected_output = '{"actor_id": 1, "action": "write_prd", "message": {"type": "test_message", "data": "test_input"}}'
      actual_output = ser.deserialize_message(actionstub_output_schema, input_data)
      self.assertEqual(actual_output, expected_output)
  ```
  
  ```py
  def test_write_prd_with_pycodecode(self):
      #: 准备参数
      input_data = '{"actor_id": 1, "action": "write_prd", "message": {"type": "test_message", "data": "test_input"}}'
      expected_output = '{"actor_id": 1, "action": "write_prd", "message": {"type": "test_message", "data": "test_input"}}'
      actual_output = ser.deserialize_message(actionstub_output_schema, input_data)
      self.assertEqual(actual_output, expected_output)
  ```
  
  ```py
  def test_write_prd_with_pycodecode_no_decode(self):
      #: 准备参数
      input_data = '{"actor_id": 1, "action": "write_prd", "message": {"type": "test_message", "data": "test_input"}}'
      expected_output = '{"actor_id": 1, "action": "write_prd", "message": {"type": "test_message", "data": "test_input"}}'
      actual_output = ser.deserialize_message(actionstub_output_schema, input_data)
      self.assertEqual(actual_output, expected_output)
  ```

运行这段代码后，如果`WritePRD`类中的实现符合预期，那么这段代码会输出测试用例的名称，例如：

```py
unittest.main()
```

如果实现不符合预期，则会输出更详细的错误信息。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the unittest of serialize

from typing import List, Tuple

from metagpt.actions import WritePRD
from metagpt.actions.action_output import ActionOutput
from metagpt.schema import Message
from metagpt.utils.serialize import (
    actionoutout_schema_to_mapping,
    deserialize_message,
    serialize_message,
)


```

这段代码定义了一个函数 `test_actionoutout_schema_to_mapping()`，用于测试将ActionOutout Schema映射到Mapping对象的过程。

函数接受两个参数，一个是Schema对象，一个是Mapping对象。函数返回一个匿名函数，该函数将Schema对象传递给第二个参数，并返回一个Mapping对象。

函数内部首先创建了一个Schema对象，其中包含一个"title" property，其值为"test"，以及一个"type" property，其值为"object"，再一个"properties" property，其中包含一个名为"field"的属性，其包含一个"title" property，其值为"field"，以及一个"type" property，其值为"string"。然后，使用 `actionoutout_schema_to_mapping()` 函数将Schema对象传递给第二个参数，并将其转换为Mapping对象。

接下来，分别对两个不同的Schema对象进行测试。第一个Schema对象包含一个"field" property，其包含一个包含两个"type"相同为"string"的元素的"array" property。第二个Schema对象包含一个"field" property，其包含一个包含两个不同类型的元素的"array" property，其中元素的"type"属性为"minItems"和"maxItems"。最后，函数使用这些Schema对象分别调用 `actionoutout_schema_to_mapping()` 函数，并打印出返回的Mapping对象。

函数的最后一个语句使用断言来测试这些Mapping对象是否正确。如果返回的Mapping对象是正确的，则打印出"True"。否则，代码将引发一个异常并停止执行。


```py
def test_actionoutout_schema_to_mapping():
    schema = {"title": "test", "type": "object", "properties": {"field": {"title": "field", "type": "string"}}}
    mapping = actionoutout_schema_to_mapping(schema)
    assert mapping["field"] == (str, ...)

    schema = {
        "title": "test",
        "type": "object",
        "properties": {"field": {"title": "field", "type": "array", "items": {"type": "string"}}},
    }
    mapping = actionoutout_schema_to_mapping(schema)
    assert mapping["field"] == (List[str], ...)

    schema = {
        "title": "test",
        "type": "object",
        "properties": {
            "field": {
                "title": "field",
                "type": "array",
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": [{"type": "string"}, {"type": "string"}],
                },
            }
        },
    }
    mapping = actionoutout_schema_to_mapping(schema)
    assert mapping["field"] == (List[Tuple[str, str]], ...)

    assert True, True


```

这段代码的作用是测试一个名为“serialize_and_deserialize_message”的函数，该函数的主要目的是创建一个“ActionOutput”对象并将消息内容和动作作为参数传递给该函数。函数内部创建了一个名为“ic_obj”的“ActionOutput”对象，该对象根据传入的“out_mapping”参数创建了一个具有两个属性的模型类，分别标记为“field1”为字符串类型，“field2”为列表类型。然后，将生成的消息内容设置为“prd demand”并设置操作用户角色进行发送，消息发送的动作名为“WritePRD”。

接下来，代码将创建一个名为“message”的消息对象，该消息内容为“prd demand”并设置操作用户角色进行发送。然后，调用“serialize_message”函数对消息对象进行序列化，得到一个新的消息对象“message_ser”。最后，代码将使用“deserialize_message”函数对新的消息对象进行解码，比较解码后的消息内容和原始消息内容是否相同，同时检查消息发送的动作是否与设置的动作一致。


```py
def test_serialize_and_deserialize_message():
    out_mapping = {"field1": (str, ...), "field2": (List[str], ...)}
    out_data = {"field1": "field1 value", "field2": ["field2 value1", "field2 value2"]}
    ic_obj = ActionOutput.create_model_class("prd", out_mapping)

    message = Message(
        content="prd demand", instruct_content=ic_obj(**out_data), role="user", cause_by=WritePRD
    )  # WritePRD as test action

    message_ser = serialize_message(message)

    new_message = deserialize_message(message_ser)
    assert new_message.content == message.content
    assert new_message.cause_by == message.cause_by
    assert new_message.instruct_content.field1 == out_data["field1"]

```

# `tests/metagpt/utils/test_text.py`

这段代码使用pytest库导入了一个名为metagpt的第三方库，并定义了一个名为_msgs的函数。

metagpt是一个基于Python的人工智能库，可以用来生成文本。在这个函数中，使用了一系列的函数式编程技巧，如decode_unicode_escape用于解码特殊字符，generate_prompt_chunk用于生成提示信息，reduce_message_length用于截取消息的长度并将其作为函数返回，split_paragraph用于将段落级别的对话进行分割。

总的来说，这段代码的作用是定义了一些函数，用于生成文本和处理文本数据。这些函数可以被其他测试函数或用户使用，以进一步测试和优化metagpt库的功能。


```py
import pytest

from metagpt.utils.text import (
    decode_unicode_escape,
    generate_prompt_chunk,
    reduce_message_length,
    split_paragraph,
)


def _msgs():
    length = 20
    while length:
        yield "Hello," * 1000 * length
        length -= 1


```

这段代码定义了一个名为 `_paragraphs` 的函数，接受一个整数参数 `n`。函数的作用是生成 `n` 个字符串，每个字符串都包含 "Hello World" 这个词组。

接下来，代码使用一个带有参数的 `@pytest.mark.parametrize` 装饰器，该装饰器根据不同的参数值，返回一个或多个不同的样例。对于每个参数，函数都会生成一个字符串，该字符串由 `_msgs` 和一些参数（如 `model_name` 和 `system_text`）组成。

最后，生成的字符串中的每个字符串都是用 `".join"` 方法连接在一起的。最终的结果是一个由多个字符串连接而成的字符串，它们的顺序是按照参数列表中的顺序生成的。


```py
def _paragraphs(n):
    return " ".join("Hello World." for _ in range(n))


@pytest.mark.parametrize(
    "msgs, model_name, system_text, reserved, expected",
    [
        (_msgs(), "gpt-3.5-turbo", "System", 1500, 1),
        (_msgs(), "gpt-3.5-turbo-16k", "System", 3000, 6),
        (_msgs(), "gpt-3.5-turbo-16k", "Hello," * 1000, 3000, 5),
        (_msgs(), "gpt-4", "System", 2000, 3),
        (_msgs(), "gpt-4", "Hello," * 1000, 2000, 2),
        (_msgs(), "gpt-4-32k", "System", 4000, 14),
        (_msgs(), "gpt-4-32k", "Hello," * 2000, 4000, 12),
    ]
)
```

这段代码是一个测试用例，用于测试生成Prompt Chunk的能力。具体来说，这段代码定义了一个函数`test_generate_prompt_chunk`，该函数接受五个参数：`text`是测试数据，`prompt_template`是用于生成Prompt的模板，`model_name`是模型名称，`system_text`是系统Text，`reserved`是保留字，`expected`是预期的结果。

在函数内部，首先定义了一个指针变量`msgs`，该指针用于存储测试数据中的消息列表。然后，定义了一个字符串变量`system_text`，用于存储系统Text。接下来，定义了一个字符串变量`prompt_template`，用于存储生成Prompt的模板。

函数中调用了两个函数`generate_prompt_chunk`和`generate_prompt_chunk_with_model_name`，分别用于生成Prompt和生成带有模型的Prompt。最后，比较生成的Prompt Chunk的长度与预期长度的关系，如果两者长度不符，则输出错误信息。


```py
def test_reduce_message_length(msgs, model_name, system_text, reserved, expected):
    assert len(reduce_message_length(msgs, model_name, system_text, reserved)) / (len("Hello,")) / 1000 == expected


@pytest.mark.parametrize(
    "text, prompt_template, model_name, system_text, reserved, expected",
    [
        (" ".join("Hello World." for _ in range(1000)), "Prompt: {}", "gpt-3.5-turbo", "System", 1500, 2),
        (" ".join("Hello World." for _ in range(1000)), "Prompt: {}", "gpt-3.5-turbo-16k", "System", 3000, 1),
        (" ".join("Hello World." for _ in range(4000)), "Prompt: {}", "gpt-4", "System", 2000, 2),
        (" ".join("Hello World." for _ in range(8000)), "Prompt: {}", "gpt-4-32k", "System", 4000, 1),
    ]
)
def test_generate_prompt_chunk(text, prompt_template, model_name, system_text, reserved, expected):
    ret = list(generate_prompt_chunk(text, prompt_template, model_name, system_text, reserved))
    assert len(ret) == expected


```

这段代码使用了@pytest.mark.parametrize装饰来定义一个测试函数，该函数接收四个参数：paragraph、sep、count和expected。通过这个装饰，我们可以将参数列表标记为参数，而不是将参数直接传递给函数。

在函数体内部，我们使用parametrize装饰定义了一个新的参数组，这个新的参数组包含了我们之前定义的四个参数。在这个新的参数组中，每个参数都有一个默认值，这些默认值会在我们定义函数时使用，如果没有提供具体的参数值。

在测试函数内部，我们使用parametrize装饰定义了新的测试函数，这个新的函数使用了我们之前定义的新的参数组。通过这个新的函数，我们可以使用parametrize装饰来定义和提供测试函数的参数，这样就可以在不公开函数代码的情况下提供测试数据了。


```py
@pytest.mark.parametrize(
    "paragraph, sep, count, expected",
    [
        (_paragraphs(10), ".", 2, [_paragraphs(5), f" {_paragraphs(5)}"]),
        (_paragraphs(10), ".", 3, [_paragraphs(4), f" {_paragraphs(3)}", f" {_paragraphs(3)}"]),
        (f"{_paragraphs(5)}\n{_paragraphs(3)}", "\n.", 2, [f"{_paragraphs(5)}\n", _paragraphs(3)]),
        ("......", ".", 2, ["...", "..."]),
        ("......", ".", 3, ["..", "..", ".."]),
        (".......", ".", 2, ["....", "..."]),
    ]
)
def test_split_paragraph(paragraph, sep, count, expected):
    ret = split_paragraph(paragraph, sep, count)
    assert ret == expected


```

这段代码使用了@pytest.mark.parametrize装饰来定义一个测试函数，该函数有两个参数：text和expected。text参数表示需要测试的字符串，expected参数表示期望的输出字符串。

在该函数中，使用parametrize装饰器定义了一个包含四个参数的列表：text和expected分别被赋值为参数[0]和参数[1]。通过这种方式，每个测试用例的text和expected参数都会在运行时传递给函数。

函数内部使用assert语句来验证decode_unicode_escape函数的正确性，具体来说，assert decode_unicode_escape(text) == expected。这意味着，如果test_decode_unicode_escape函数能够正确地解码和解码test_case用提供的text参数，那么decode_unicode_escape函数的实现应该与expected输出相等。


```py
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello\\nWorld", "Hello\nWorld"),
        ("Hello\\tWorld", "Hello\tWorld"),
        ("Hello\\u0020World", "Hello World"),
    ]
)
def test_decode_unicode_escape(text, expected):
    assert decode_unicode_escape(text) == expected

```

# `tests/metagpt/utils/test_token_counter.py`

这段代码是一个Python脚本，用于测试一个名为`count_message_tokens`的函数。该函数使用`metagpt.utils.token_counter`模块来统计文本中的消息标记（如"Hello"和"Hi there!"）。以下是脚本的详细解释：

1. 首先，导入`pytest`模块，用于进行测试。
2. 使用`import metagpt.utils.token_counter`来导入`metagpt.utils.token_counter`模块，该模块可能包含与测试相关的函数和/或类。
3. 使用自己的定义的`count_message_tokens`函数，该函数接受一个列表`messages`，该列表包含两个字典，每个字典包含一个`role`和一个`content`字段，分别表示消息发送者和消息内容。
4. 编写测试函数`test_count_message_tokens`，该函数使用`count_message_tokens`函数统计给定消息列表中的消息标记数量。
5. 最后，通过调用`pytest.main`函数来运行测试，该函数将在控制台输出测试结果。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/24 17:54
@Author  : alexanderwu
@File    : test_token_counter.py
"""
import pytest

from metagpt.utils.token_counter import count_message_tokens, count_string_tokens


def test_count_message_tokens():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert count_message_tokens(messages) == 17


```

这段代码是一个测试用例，用于测试 Python 函数 `count_message_tokens` 的作用。函数接收一个列表 `messages`，包含两个对象，每个对象都包含 `role` 和 `content` 属性，以及一个 `name` 属性。函数使用 `assert` 语句来验证函数的正确性，其中第一个参数是一个函数字符串 `count_message_tokens`，第二个参数是一个包含 `messages` 的列表，第三个参数是一个字符串 `model`，表示要使用的模型。

具体来说，这段代码可以分为以下几个部分：

1. 定义一个测试函数 `test_count_message_tokens_with_name`，该函数使用 `assert` 语句来验证 `count_message_tokens` 函数的正确性。函数接收一个列表 `messages`，包含两个对象，每个对象都包含 `role`、`content` 和 `name` 属性。函数的作用是测试 `count_message_tokens` 函数在如何传递一个包含两个消息的对象时，它是否能够正确返回消息的数量。
2. 定义一个测试函数 `test_count_message_tokens_empty_input`，该函数使用 `assert` 语句来验证 `count_message_tokens` 函数在如何传递一个空列表时，它是否能够正确返回 3。函数的作用是测试 `count_message_tokens` 函数在如何传递一个空列表时，它是否能够正确返回消息的数量。
3. 定义一个测试函数 `test_count_message_tokens_invalid_model`，该函数使用 `assert` 语句来验证 `count_message_tokens` 函数在如何传递一个不存在的模型时，它是否能够正确 raise `NotImplementedError`。函数的作用是测试 `count_message_tokens` 函数在如何传递一个不存在的模型时，它是否能够正确 raise `NotImplementedError`。


```py
def test_count_message_tokens_with_name():
    messages = [
        {"role": "user", "content": "Hello", "name": "John"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert count_message_tokens(messages) == 17


def test_count_message_tokens_empty_input():
    """Empty input should return 3 tokens"""
    assert count_message_tokens([]) == 3


def test_count_message_tokens_invalid_model():
    """Invalid model should raise a KeyError"""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    with pytest.raises(NotImplementedError):
        count_message_tokens(messages, model="invalid_model")


```

这段代码是一个测试用例，用于分别测试 GPT-4 模型对文本消息和字符串的计数功能。

首先，代码定义了一个名为 `test_count_message_tokens_gpt_4()` 的函数，用于测试 GPT-4 对文本消息的计数功能。该函数接收一个消息列表，其中每个消息包含一个键（即消息类型）和内容。然后，函数调用 `count_message_tokens()` 函数，并传入消息列表和模型名称。最后，函数使用断言确保计数结果正确，即 `assert count_message_tokens(messages, model="gpt-4-0314") == 15`。

接着，代码定义了一个名为 `test_count_string_tokens()` 的函数，用于测试 GPT-3.5 对字符串的计数功能。该函数同样接收一个字符串参数，然后调用 `count_string_tokens()` 函数，并传入字符串和模型名称。最后，函数使用断言确保计数结果正确，即 `assert count_string_tokens(string, model_name="gpt-3.5-turbo-0301") == 4`。

这些测试函数旨在验证 GPT-4 和 GPT-3.5 模型的计数功能是否正确，可以在测试中根据需要进行修改。


```py
def test_count_message_tokens_gpt_4():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert count_message_tokens(messages, model="gpt-4-0314") == 15


def test_count_string_tokens():
    """Test that the string tokens are counted correctly."""

    string = "Hello, world!"
    assert count_string_tokens(string, model_name="gpt-3.5-turbo-0301") == 4


```



这两段代码是测试代码，用于测试 GPT模型的字符串处理能力。

具体来说，第一段代码 `test_count_count_string_tokens_empty_input` 是用于测试空字符串输入下，GPT模型是否能够正确地计数字符串中的字符。该测试中，使用了 `assert count_string_tokens("", model_name="gpt-3.5-turbo-0301") == 0`，表示期望GPT模型对空字符串的计数结果为0。如果GPT模型在空字符串输入下，也能够正确地计数字符，该代码将不会输出任何错误信息，直接返回。

第二段代码 `test_count_string_tokens_gpt_4` 是用于测试GPT模型字符串处理能力的一组测试，包括：

1. 如果输入字符串为空，GPT模型是否能够正确地计数字符串中的字符；
2. 如果输入字符串不为空，GPT模型是否能够正确地计数字符串中的字符。

该代码使用了两个测试函数 `test_count_string_tokens_empty_input` 和 `test_count_string_tokens_gpt_4`，第一个函数用于测试空字符串输入，第二个函数用于测试非空字符串输入。


```py
def test_count_string_tokens_empty_input():
    """Test that the string tokens are counted correctly."""

    assert count_string_tokens("", model_name="gpt-3.5-turbo-0301") == 0


def test_count_string_tokens_gpt_4():
    """Test that the string tokens are counted correctly."""

    string = "Hello, world!"
    assert count_string_tokens(string, model_name="gpt-4-0314") == 4

```

# `tests/metagpt/utils/__init__.py`

这段代码是一个Python脚本，使用了Python标准库中的`#!/usr/bin/env python`作为脚本解释器的行号前缀。这表示该脚本使用Python 3作为运行时环境，而不是Python 2。

该脚本包含一个`#`注释，表示该处有一些元数据，其中包括脚本的名称、作者、以及脚本所在的文件名。

该脚本使用了一个`@Time`格式，表示该处记录了脚本的创建时间。

该脚本使用了一个`@Author`格式，表示该处记录了脚本的作者。

该脚本使用了一个`@File`格式，表示该处记录了脚本所在的文件名。

该脚本中没有其他语句，因此该脚本的作用是：在当前目录下创建一个名为`__init__.py`的Python脚本，并在其中输出"Hello, World!"。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 16:01
@Author  : alexanderwu
@File    : __init__.py
"""

```