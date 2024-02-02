# `Bert-VITS2\tools\classify_language.py`

```py
# 导入 regex 库，用于正则表达式操作
import regex as re

# 尝试从 config 模块中导入 config 对象
try:
    from config import config
    # 从 config 对象中获取 language_identification_library 属性值
    LANGUAGE_IDENTIFICATION_LIBRARY = (
        config.webui_config.language_identification_library
    )
# 如果导入失败，则设置 LANGUAGE_IDENTIFICATION_LIBRARY 为 "langid"
except:
    LANGUAGE_IDENTIFICATION_LIBRARY = "langid"

# 将 LANGUAGE_IDENTIFICATION_LIBRARY 转换为小写，并赋值给 module
module = LANGUAGE_IDENTIFICATION_LIBRARY.lower()

# 定义支持的语言列表
langid_languages = [
    # 列举了很多语言的缩写
]

# 定义 classify_language 函数，用于识别文本的语言
def classify_language(text: str, target_languages: list = None) -> str:
    # 如果 module 是 "fastlid" 或 "fasttext"
    if module == "fastlid" or module == "fasttext":
        # 从 fastlid 模块中导入 fastlid 和 supported_langs
        from fastlid import fastlid, supported_langs
        # 将 fastlid 赋值给 classifier
        classifier = fastlid
        # 如果 target_languages 不为 None
        if target_languages != None:
            # 将 target_languages 中支持的语言筛选出来
            target_languages = [
                lang for lang in target_languages if lang in supported_langs
            ]
            # 设置 fastlid 的语言为 target_languages
            fastlid.set_languages = target_languages
    # 如果 module 是 "langid"
    elif module == "langid":
        # 从 langid 模块中导入 classify
        import langid
        # 将 classify 赋值给 classifier
        classifier = langid.classify
        # 如果 target_languages 不为 None
        if target_languages != None:
            # 将 target_languages 中支持的语言筛选出来
            target_languages = [
                lang for lang in target_languages if lang in langid_languages
            ]
            # 设置 langid 的语言为 target_languages
            langid.set_languages(target_languages)
    # 如果模块不在已知的模块列表中，则抛出数值错误
    else:
        raise ValueError(f"Wrong module {module}")

    # 使用分类器对文本进行语言分类，并获取第一个语言
    lang = classifier(text)[0]

    # 返回语言分类结果
    return lang
def classify_zh_ja(text: str) -> str:
    # 遍历文本中的字符及其索引
    for idx, char in enumerate(text):
        # 获取字符的 Unicode 值
        unicode_val = ord(char)

        # 检测日语字符范围
        if 0x3040 <= unicode_val <= 0x309F or 0x30A0 <= unicode_val <= 0x30FF:
            return "ja"  # 如果是日语字符，则返回 "ja"

        # 检测汉字字符范围
        if 0x4E00 <= unicode_val <= 0x9FFF:
            # 检查周围的字符
            next_char = text[idx + 1] if idx + 1 < len(text) else None

            if next_char and (
                0x3040 <= ord(next_char) <= 0x309F or 0x30A0 <= ord(next_char) <= 0x30FF
            ):
                return "ja"  # 如果周围有日语字符，则返回 "ja"

    return "zh"  # 如果没有检测到日语字符，则返回 "zh"


def split_alpha_nonalpha(text, mode=1):
    if mode == 1:
        pattern = r"(?<=[\u4e00-\u9fff\u3040-\u30FF\d\s])(?=[\p{Latin}])|(?<=[\p{Latin}\s])(?=[\u4e00-\u9fff\u3040-\u30FF\d])"
    elif mode == 2:
        pattern = r"(?<=[\u4e00-\u9fff\u3040-\u30FF\s])(?=[\p{Latin}\d])|(?<=[\p{Latin}\d\s])(?=[\u4e00-\u9fff\u3040-\u30FF])"
    else:
        raise ValueError("Invalid mode. Supported modes are 1 and 2.")

    return re.split(pattern, text)  # 使用正则表达式模式分割文本


if __name__ == "__main__":
    text = "这是一个测试文本"
    print(classify_language(text))  # 调用不存在的函数，应该是 classify_zh_ja(text)
    print(classify_zh_ja(text))  # "zh"

    text = "これはテストテキストです"
    print(classify_language(text))  # 调用不存在的函数，应该是 classify_zh_ja(text)
    print(classify_zh_ja(text))  # "ja"

    text = "vits和Bert-VITS2是tts模型。花费3days.花费3天。Take 3 days"

    print(split_alpha_nonalpha(text, mode=1))
    # output: ['vits', '和', 'Bert-VITS', '2是', 'tts', '模型。花费3', 'days.花费3天。Take 3 days']

    print(split_alpha_nonalpha(text, mode=2))
    # output: ['vits', '和', 'Bert-VITS2', '是', 'tts', '模型。花费', '3days.花费', '3', '天。Take 3 days']

    text = "vits 和 Bert-VITS2 是 tts 模型。花费3days.花费3天。Take 3 days"
    print(split_alpha_nonalpha(text, mode=1))
    # output: ['vits ', '和 ', 'Bert-VITS', '2 ', '是 ', 'tts ', '模型。花费3', 'days.花费3天。Take ', '3 ', 'days']

    text = "vits 和 Bert-VITS2 是 tts 模型。花费3days.花费3天。Take 3 days"
    print(split_alpha_nonalpha(text, mode=2))
    # 定义一个列表，包含了一些字符串元素
    output = ['vits ', '和 ', 'Bert-VITS2 ', '是 ', 'tts ', '模型。花费', '3days.花费', '3', '天。Take ', '3 ', 'days']
```