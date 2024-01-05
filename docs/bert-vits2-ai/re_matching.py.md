# `d:/src/tocomm/Bert-VITS2\re_matching.py`

```
import re  # 导入re模块，用于正则表达式匹配

def extract_language_and_text_updated(speaker, dialogue):
    pattern_language_text = r"<(\S+?)>([^<]+)"  # 定义正则表达式模式，匹配<语言>标签和其后的文本
    matches = re.findall(pattern_language_text, dialogue, re.DOTALL)  # 使用正则表达式匹配模式，返回匹配结果列表
    speaker = speaker[1:-1]  # 去除说话人字符串中的方括号
    matches_cleaned = [(lang.upper(), text.strip()) for lang, text in matches]  # 清理匹配结果，去除两边的空白字符，并将语言转换为大写
    matches_cleaned.append(speaker)  # 将清理后的说话人添加到匹配结果列表中
    return matches_cleaned  # 返回清理后的匹配结果列表

def validate_text(input_text):
    pattern_speaker = r"(\[\S+?\])((?:\s*<\S+?>[^<\[\]]+?)+)"  # 定义正则表达式模式，匹配说话人和其后的文本
    matches = re.findall(pattern_speaker, input_text, re.DOTALL)  # 使用正则表达式匹配模式，返回匹配结果列表
# 对每个匹配到的说话人内容进行进一步验证
for _, dialogue in matches:
    # 调用函数extract_language_and_text_updated提取语言和文本信息
    language_text_matches = extract_language_and_text_updated(_, dialogue)
    # 如果提取的语言和文本信息为空，则返回错误信息
    if not language_text_matches:
        return (
            False,
            "Error: Invalid format detected in dialogue content. Please check your input.",
        )

# 如果输入的文本中没有找到任何匹配项，则返回错误信息
if not matches:
    return (
        False,
        "Error: No valid speaker format detected. Please check your input.",
    )

# 返回验证结果为True和提示信息"Input is valid."
return True, "Input is valid."
def text_matching(text: str) -> list:
    # 定义匹配说话者和对话内容的正则表达式模式
    speaker_pattern = r"(\[\S+?\])(.+?)(?=\[\S+?\]|$)"
    # 在文本中查找所有匹配的说话者和对话内容，并返回结果列表
    matches = re.findall(speaker_pattern, text, re.DOTALL)
    # 创建一个空列表用于存储处理后的结果
    result = []
    # 遍历匹配结果，提取语言和文本，并调用函数进行进一步处理
    for speaker, dialogue in matches:
        result.append(extract_language_and_text_updated(speaker, dialogue))
    # 返回处理后的结果列表
    return result


def cut_para(text):
    # 按照换行符对文本进行分段
    splitted_para = re.split("[\n]", text)
    # 删除空字符串，并去除段落两端的空格
    splitted_para = [
        sentence.strip() for sentence in splitted_para if sentence.strip()
    ]
    # 返回分段后的结果列表
    return splitted_para


def cut_sent(para):
    # 使用正则表达式将文本中的单字符断句符替换为换行符
    para = re.sub("([。！;？\?])([^”’])", r"\1\n\2", para)
    # 使用正则表达式将文本中的英文省略号替换为换行符
    para = re.sub("(\.{6})([^”’])", r"\1\n\2", para)
    # 返回断句后的文本
    return para
para = re.sub("(\…{2})([^”’])", r"\1\n\2", para)  # 中文省略号
```
这行代码使用正则表达式将连续的中文省略号替换为省略号和换行符的组合。

```
para = re.sub("([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
```
这行代码使用正则表达式将标点符号后面的引号和后续内容替换为标点符号、换行符和后续内容的组合。

```
para = para.rstrip()  # 段尾如果有多余的\n就去掉它
```
这行代码去除段落末尾的多余换行符。

```
return para.split("\n")
```
这行代码将字符串按照换行符进行分割，返回一个列表。

```
if __name__ == "__main__":
```
这行代码判断当前模块是否为主模块，即直接运行的模块。

```
text = """
[说话人1]
[说话人2]<zh>你好吗？<jp>元気ですか？<jp>こんにちは，世界。<zh>你好吗？
[说话人3]<zh>谢谢。<jp>どういたしまして。
"""
```
这行代码定义了一个多行字符串，其中包含了一段对话文本。

```
text_matching(text)
```
这行代码调用了名为`text_matching`的函数，传入了`text`作为参数。

```
test_text = """
[说话人1]<zh>你好，こんにちは！<jp>こんにちは，世界。
[说话人2]<zh>你好吗？
"""
```
这行代码定义了一个多行字符串，其中包含了一段测试文本。

```
text_matching(test_text)
```
这行代码调用了名为`text_matching`的函数，传入了`test_text`作为参数。

```
res = validate_text(test_text)
```
这行代码调用了名为`validate_text`的函数，传入了`test_text`作为参数，并将返回值赋给变量`res`。
# 打印变量res的值
```