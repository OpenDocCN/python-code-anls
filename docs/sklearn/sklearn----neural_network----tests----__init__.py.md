# `D:\src\scipysrc\scikit-learn\sklearn\neural_network\tests\__init__.py`

```
# 定义一个名为 create_html_element 的函数，用于生成 HTML 元素的字符串表示
def create_html_element(tag, text=None, **attributes):
    # 初始化一个空字符串，用于存储生成的 HTML 元素
    result = ''
    # 开始标签部分，根据标签名和属性生成对应的 HTML 开始标签
    result += f'<{tag}'
    # 遍历传入的属性字典，将属性名和属性值转换为 HTML 属性的形式添加到结果中
    for attr, value in attributes.items():
        result += f' {attr}="{value}"'
    # 如果没有传入文本内容，则生成一个自闭合标签并返回
    if text is None:
        result += ' />'
    else:
        # 否则生成一个包含文本内容的开始标签和结束标签，并将文本内容添加到其中
        result += f'>{text}</{tag}>'
    # 返回生成的 HTML 元素字符串
    return result
```