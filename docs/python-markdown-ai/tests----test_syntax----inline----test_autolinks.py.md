# `markdown\tests\test_syntax\inline\test_autolinks.py`

```
"""
这段代码是一个测试用例，用于测试自动链接的功能。

从第 7 行开始，定义了一个名为 TestAutomaticLinks 的测试类，继承自 markdown.test_tools 中的 TestCase 类。

接下来的几个方法是测试用例，用于测试不同类型的电子邮件地址的渲染效果。

- test_email_address 方法测试普通的电子邮件地址的渲染效果。
- test_mailto_email_address 方法测试使用 mailto 标签的电子邮件地址的渲染效果。
- test_email_address_with_ampersand 方法测试包含特殊字符 & 的电子邮件地址的渲染效果。
- test_invalid_email_address_local_part 方法测试缺少本地部分的电子邮件地址的渲染效果。
- test_invalid_email_address_domain 方法测试缺少域名部分的电子邮件地址的渲染效果。

每个测试方法中，使用 assertMarkdownRenders 方法来断言 Markdown 渲染的结果是否符合预期。

这段代码是 Python Markdown 项目的一部分，用于测试自动链接的功能。

"""
```