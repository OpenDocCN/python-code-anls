# `Bert-VITS2\tools\classify_language.py`

```

# 导入 regex 库并重命名为 re
import regex as re

# 尝试从 config 模块中导入 config 对象
try:
    from config import config

    # 从 config 对象中获取 language_identification_library 属性值
    LANGUAGE_IDENTIFICATION_LIBRARY = (
        config.webui_config.language_identification_library
    )
# 如果导入失败，则将 LANGUAGE_IDENTIFICATION_LIBRARY 设置为 "langid"
except:
    LANGUAGE_IDENTIFICATION_LIBRARY = "langid"

# 将 LANGUAGE_IDENTIFICATION_LIBRARY 转换为小写
module = LANGUAGE_IDENTIFICATION_LIBRARY.lower()

# 支持的语言列表
langid_languages = [
    # 列出了很多语言的缩写
]

# 根据给定的文本和目标语言列表，返回文本所属的语言
def classify_language(text: str, target_languages: list = None) -> str:
    # 如果 module 是 "fastlid" 或 "fasttext"
    if module == "fastlid" or module == "fasttext":
        # 从 fastlid 模块中导入 fastlid 和 supported_langs
        from fastlid import fastlid, supported_langs

        # 将 classifier 设置为 fastlid
        classifier = fastlid
        # 如果目标语言列表不为空
        if target_languages != None:
            # 将目标语言列表中存在于 supported_langs 中的语言筛选出来
            target_languages = [
                lang for lang in target_languages if lang in supported_langs
            ]
            # 设置 fastlid 的语言列表
            fastlid.set_languages = target_languages
    # 如果 module 是 "langid"
    elif module == "langid":
        # 从 langid 模块中导入 classify
        import langid

        # 将 classifier 设置为 langid 的 classify 函数
        classifier = langid.classify
        # 如果目标语言列表不为空
        if target_languages != None:
            # 将目标语言列表中存在于 langid_languages 中的语言筛选出来
            target_languages = [
                lang for lang in target_languages if lang in langid_languages
            ]
            # 设置 langid 的语言列表
            langid.set_languages(target_languages)
    # 如果 module 不是 "fastlid" 也不是 "langid"，则抛出 ValueError
    else:
        raise ValueError(f"Wrong module {module}")

    # 使用 classifier 对文本进行语言分类，并返回语言
    lang = classifier(text)[0]

    return lang

# 根据文本内容判断其属于中文还是日文
def classify_zh_ja(text: str) -> str:
    for idx, char in enumerate(text):
        unicode_val = ord(char)

        # 检测日语字符
        if 0x3040 <= unicode_val <= 0x309F or 0x30A0 <= unicode_val <= 0x30FF:
            return "ja"

        # 检测汉字字符
        if 0x4E00 <= unicode_val <= 0x9FFF:
            # 检查周围的字符
            next_char = text[idx + 1] if idx + 1 < len(text) else None

            if next_char and (
                0x3040 <= ord(next_char) <= 0x309F or 0x30A0 <= ord(next_char) <= 0x30FF
            ):
                return "ja"

    return "zh"

# 根据指定模式将文本分割为中文和非中文部分
def split_alpha_nonalpha(text, mode=1):
    if mode == 1:
        pattern = r"(?<=[\u4e00-\u9fff\u3040-\u30FF\d\s])(?=[\p{Latin}])|(?<=[\p{Latin}\s])(?=[\u4e00-\u9fff\u3040-\u30FF\d])"
    elif mode == 2:
        pattern = r"(?<=[\u4e00-\u9fff\u3040-\u30FF\s])(?=[\p{Latin}\d])|(?<=[\p{Latin}\d\s])(?=[\u4e00-\u9fff\u3040-\u30FF])"
    else:
        raise ValueError("Invalid mode. Supported modes are 1 and 2.")

    return re.split(pattern, text)

# 如果当前脚本被直接执行
if __name__ == "__main__":
    text = "这是一个测试文本"
    print(classify_language(text))  # 输出文本所属的语言
    print(classify_zh_ja(text))  # 输出 "zh"

    text = "これはテストテキストです"
    print(classify_language(text))  # 输出文本所属的语言
    print(classify_zh_ja(text))  # 输出 "ja"

    text = "vits和Bert-VITS2是tts模型。花费3days.花费3天。Take 3 days"

    print(split_alpha_nonalpha(text, mode=1))  # 输出按照模式1分割后的结果
    # output: ['vits', '和', 'Bert-VITS', '2是', 'tts', '模型。花费3', 'days.花费3天。Take 3 days']

    print(split_alpha_nonalpha(text, mode=2))  # 输出按照模式2分割后的结果
    # output: ['vits', '和', 'Bert-VITS2', '是', 'tts', '模型。花费', '3days.花费', '3', '天。Take 3 days']

    text = "vits 和 Bert-VITS2 是 tts 模型。花费3days.花费3天。Take 3 days"
    print(split_alpha_nonalpha(text, mode=1))  # 输出按照模式1分割后的结果
    # output: ['vits ', '和 ', 'Bert-VITS', '2 ', '是 ', 'tts ', '模型。花费3', 'days.花费3天。Take ', '3 ', 'days']

    text = "vits 和 Bert-VITS2 是 tts 模型。花费3days.花费3天。Take 3 days"
    print(split_alpha_nonalpha(text, mode=2))  # 输出按照模式2分割后的结果
    # output: ['vits ', '和 ', 'Bert-VITS2 ', '是 ', 'tts ', '模型。花费', '3days.花费', '3', '天。Take ', '3 ', 'days']

```