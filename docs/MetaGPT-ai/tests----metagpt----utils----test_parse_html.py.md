# `MetaGPT\tests\metagpt\utils\test_parse_html.py`

```

# 从metagpt.utils中导入parse_html模块
from metagpt.utils import parse_html

# 定义一个HTML页面的字符串
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
</body>
</html>
"""

# 定义一个字符串变量，包含HTML页面的内容
CONTENT = (
    "This is a HeadingThis is a paragraph witha linkand someemphasizedtext.Item 1Item 2Item 3Numbered Item 1Numbered "
    "Item 2Numbered Item 3Header 1Header 2Row 1, Cell 1Row 1, Cell 2Row 2, Cell 1Row 2, Cell 2Name:Email:SubmitThis is a div "
    'with a class "box".a link'
)

# 定义一个测试函数，用于测试解析HTML页面的功能
def test_web_page():
    # 创建一个WebPage对象，传入页面内容、HTML字符串和URL
    page = parse_html.WebPage(inner_text=CONTENT, html=PAGE, url="http://example.com")
    # 断言页面的标题为"Random HTML Example"
    assert page.title == "Random HTML Example"
    # 断言页面中的链接列表为["http://example.com/test", "https://metagpt.com"]
    assert list(page.get_links()) == ["http://example.com/test", "https://metagpt.com"]

# 定义一个测试函数，用于测试获取页面内容的功能
def test_get_page_content():
    # 调用parse_html模块的get_html_content函数，传入HTML字符串和URL
    ret = parse_html.get_html_content(PAGE, "http://example.com")
    # 断言返回的页面内容与预期的内容相等
    assert ret == CONTENT

```