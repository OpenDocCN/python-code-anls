# `Bert-VITS2\re_matching.py`

```py
import re  # 导入正则表达式模块


def extract_language_and_text_updated(speaker, dialogue):
    # 使用正则表达式匹配<语言>标签和其后的文本
    pattern_language_text = r"<(\S+?)>([^<]+)"
    matches = re.findall(pattern_language_text, dialogue, re.DOTALL)
    speaker = speaker[1:-1]  # 去除说话人字符串两边的方括号
    # 清理文本：去除两边的空白字符，并将语言转换为大写
    matches_cleaned = [(lang.upper(), text.strip()) for lang, text in matches]
    matches_cleaned.append(speaker)  # 将说话人添加到清理后的匹配结果中
    return matches_cleaned  # 返回清理后的匹配结果


def validate_text(input_text):
    # 验证说话人的正则表达式
    pattern_speaker = r"(\[\S+?\])((?:\s*<\S+?>[^<\[\]]+?)+)"

    # 使用re.DOTALL标志使.匹配包括换行符在内的所有字符
    matches = re.findall(pattern_speaker, input_text, re.DOTALL)

    # 对每个匹配到的说话人内容进行进一步验证
    for _, dialogue in matches:
        language_text_matches = extract_language_and_text_updated(_, dialogue)
        if not language_text_matches:
            return (
                False,
                "Error: Invalid format detected in dialogue content. Please check your input.",
            )

    # 如果输入的文本中没有找到任何匹配项
    if not matches:
        return (
            False,
            "Error: No valid speaker format detected. Please check your input.",
        )

    return True, "Input is valid."


def text_matching(text: str) -> list:
    speaker_pattern = r"(\[\S+?\])(.+?)(?=\[\S+?\]|$)"
    matches = re.findall(speaker_pattern, text, re.DOTALL)
    result = []
    for speaker, dialogue in matches:
        result.append(extract_language_and_text_updated(speaker, dialogue))
    return result


def cut_para(text):
    splitted_para = re.split("[\n]", text)  # 按段分，将文本分割成段落
    splitted_para = [
        sentence.strip() for sentence in splitted_para if sentence.strip()
    ]  # 删除空字符串
    return splitted_para  # 返回分割后的段落列表


def cut_sent(para):
    para = re.sub("([。！;？\?])([^”’])", r"\1\n\2", para)  # 单字符断句符
    para = re.sub("(\.{6})([^”’])", r"\1\n\2", para)  # 英文省略号
    para = re.sub("(\…{2})([^”’])", r"\1\n\2", para)  # 中文省略号
    para = re.sub("([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)  # 根据标点符号断句
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 使用换行符"\n"将字符串para分割成列表，返回结果
    return para.split("\n")
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 定义包含多语言文本的字符串
    text = """
    [说话人1]
    [说话人2]<zh>你好吗？<jp>元気ですか？<jp>こんにちは，世界。<zh>你好吗？
    [说话人3]<zh>谢谢。<jp>どういたしまして。
    """
    # 调用text_matching函数处理文本
    text_matching(text)
    # 定义测试用的包含多语言文本的字符串
    test_text = """
    [说话人1]<zh>你好，こんにちは！<jp>こんにちは，世界。
    [说话人2]<zh>你好吗？
    """
    # 调用text_matching函数处理测试文本
    text_matching(test_text)
    # 调用validate_text函数验证测试文本
    res = validate_text(test_text)
    # 打印验证结果
    print(res)
```