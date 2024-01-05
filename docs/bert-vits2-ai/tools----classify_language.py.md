# `d:/src/tocomm/Bert-VITS2\tools\classify_language.py`

```
import regex as re  # 导入 regex 模块，用于正则表达式操作

try:
    from config import config  # 尝试从 config 模块中导入 config 对象

    LANGUAGE_IDENTIFICATION_LIBRARY = (
        config.webui_config.language_identification_library
    )  # 从 config 对象中获取 language_identification_library 属性，并赋值给 LANGUAGE_IDENTIFICATION_LIBRARY
except:
    LANGUAGE_IDENTIFICATION_LIBRARY = "langid"  # 如果导入失败，则将 LANGUAGE_IDENTIFICATION_LIBRARY 赋值为 "langid"

module = LANGUAGE_IDENTIFICATION_LIBRARY.lower()  # 将 LANGUAGE_IDENTIFICATION_LIBRARY 转换为小写，并赋值给 module

langid_languages = [  # 创建一个包含多个语言代码的列表
    "af",
    "am",
    "an",
    "ar",
    "as",
    "az",
    # 其他语言代码
    ...
    "be",  # 白俄罗斯语
    "bg",  # 保加利亚语
    "bn",  # 孟加拉语
    "br",  # 布列塔尼语
    "bs",  # 波斯尼亚语
    "ca",  # 加泰罗尼亚语
    "cs",  # 捷克语
    "cy",  # 威尔士语
    "da",  # 丹麦语
    "de",  # 德语
    "dz",  # 不丹语
    "el",  # 希腊语
    "en",  # 英语
    "eo",  # 世界语
    "es",  # 西班牙语
    "et",  # 爱沙尼亚语
    "eu",  # 巴斯克语
    "fa",  # 波斯语
    "fi",  # 芬兰语
    "fo",  # 法罗语
    "fr",  # 法语
    "ga",  # 爱尔兰语
    "gl",  # 加利西亚语
    "gu",  # 古吉拉特语
    "he",  # 希伯来语
    "hi",  # 印地语
    "hr",  # 克罗地亚语
    "ht",  # 海地克里奥尔语
    "hu",  # 匈牙利语
    "hy",  # 亚美尼亚语
    "id",  # 印度尼西亚语
    "is",  # 冰岛语
    "it",  # 意大利语
    "ja",  # 日语
    "jv",  # 爪哇语
    "ka",  # 格鲁吉亚语
    "kk",  # 哈萨克语
    "km",  # 高棉语
    "kn",  # 卡纳达语
    "ko",  # 朝鲜语
    "ku",  # Kurdish language
    "ky",  # Kyrgyz language
    "la",  # Latin language
    "lb",  # Luxembourgish language
    "lo",  # Lao language
    "lt",  # Lithuanian language
    "lv",  # Latvian language
    "mg",  # Malagasy language
    "mk",  # Macedonian language
    "ml",  # Malayalam language
    "mn",  # Mongolian language
    "mr",  # Marathi language
    "ms",  # Malay language
    "mt",  # Maltese language
    "nb",  # Norwegian Bokmål language
    "ne",  # Nepali language
    "nl",  # Dutch language
    "nn",  # Norwegian Nynorsk language
    "no",  # Norwegian language
    "oc",  # Occitan language
    "or",  # or 是 Python 的关键字，表示逻辑或，这里是字符串列表中的一个元素
    "pa",  # 字符串列表中的一个元素
    "pl",  # 字符串列表中的一个元素
    "ps",  # 字符串列表中的一个元素
    "pt",  # 字符串列表中的一个元素
    "qu",  # 字符串列表中的一个元素
    "ro",  # 字符串列表中的一个元素
    "ru",  # 字符串列表中的一个元素
    "rw",  # 字符串列表中的一个元素
    "se",  # 字符串列表中的一个元素
    "si",  # 字符串列表中的一个元素
    "sk",  # 字符串列表中的一个元素
    "sl",  # 字符串列表中的一个元素
    "sq",  # 字符串列表中的一个元素
    "sr",  # 字符串列表中的一个元素
    "sv",  # 字符串列表中的一个元素
    "sw",  # 字符串列表中的一个元素
    "ta",  # 字符串列表中的一个元素
    "te",  # 字符串列表中的一个元素
    "th",  # 字符串列表中的一个元素
    "tl",  # Tagalog language
    "tr",  # Turkish language
    "ug",  # Uighur language
    "uk",  # Ukrainian language
    "ur",  # Urdu language
    "vi",  # Vietnamese language
    "vo",  # Volapük language
    "wa",  # Walloon language
    "xh",  # Xhosa language
    "zh",  # Chinese language
    "zu",  # Zulu language
]


def classify_language(text: str, target_languages: list = None) -> str:
    if module == "fastlid" or module == "fasttext":
        from fastlid import fastlid, supported_langs  # Importing the fastlid module and its supported_langs

        classifier = fastlid  # Assigning the fastlid classifier to the variable 'classifier'
        if target_languages != None:  # Checking if the target_languages parameter is not None
            target_languages = [
                lang for lang in target_languages if lang in supported_langs
            ]
            # 从目标语言列表中筛选出在支持语言列表中的语言
            fastlid.set_languages = target_languages
    elif module == "langid":
        import langid

        classifier = langid.classify
        # 如果目标语言列表不为空
        if target_languages != None:
            target_languages = [
                lang for lang in target_languages if lang in langid_languages
            ]
            # 从目标语言列表中筛选出在 langid 支持的语言列表中的语言
            langid.set_languages(target_languages)
    else:
        # 如果模块不是 "langid"，抛出数值错误
        raise ValueError(f"Wrong module {module}")

    # 使用分类器对文本进行语言分类
    lang = classifier(text)[0]

    # 返回分类结果
    return lang
def classify_zh_ja(text: str) -> str:
    # 遍历文本中的每个字符及其索引
    for idx, char in enumerate(text):
        # 获取字符的 Unicode 值
        unicode_val = ord(char)

        # 检测日语字符范围
        if 0x3040 <= unicode_val <= 0x309F or 0x30A0 <= unicode_val <= 0x30FF:
            # 如果字符在日语字符范围内，则返回 "ja"
            return "ja"

        # 检测汉字字符范围
        if 0x4E00 <= unicode_val <= 0x9FFF:
            # 检查周围的字符是否为日语字符
            next_char = text[idx + 1] if idx + 1 < len(text) else None

            if next_char and (
                0x3040 <= ord(next_char) <= 0x309F or 0x30A0 <= ord(next_char) <= 0x30FF
            ):
                # 如果周围的字符为日语字符，则返回 "ja"
                return "ja"

    # 如果没有检测到日语字符，则返回 "zh"
    return "zh"
def split_alpha_nonalpha(text, mode=1):
    # 根据不同的模式选择不同的正则表达式模式
    if mode == 1:
        pattern = r"(?<=[\u4e00-\u9fff\u3040-\u30FF\d\s])(?=[\p{Latin}])|(?<=[\p{Latin}\s])(?=[\u4e00-\u9fff\u3040-\u30FF\d])"
    elif mode == 2:
        pattern = r"(?<=[\u4e00-\u9fff\u3040-\u30FF\s])(?=[\p{Latin}\d])|(?<=[\p{Latin}\d\s])(?=[\u4e00-\u9fff\u3040-\u30FF])"
    else:
        # 如果模式不是1或2，则抛出值错误
        raise ValueError("Invalid mode. Supported modes are 1 and 2.")

    # 使用正则表达式模式分割文本
    return re.split(pattern, text)


if __name__ == "__main__":
    text = "这是一个测试文本"
    print(classify_language(text))  # 打印文本的语言分类
    print(classify_zh_ja(text))  # 打印文本的中日分类，预期输出为"zh"

    text = "これはテストテキストです"
    print(classify_language(text))  # 打印文本的语言分类
```
    print(classify_zh_ja(text))  # 调用名为classify_zh_ja的函数，传入text参数，打印输出结果为"ja"

    text = "vits和Bert-VITS2是tts模型。花费3days.花费3天。Take 3 days"

    print(split_alpha_nonalpha(text, mode=1))
    # 调用名为split_alpha_nonalpha的函数，传入text和mode参数，打印输出结果为指定mode下的分割结果

    print(split_alpha_nonalpha(text, mode=2))
    # 调用名为split_alpha_nonalpha的函数，传入text和mode参数，打印输出结果为指定mode下的分割结果

    text = "vits 和 Bert-VITS2 是 tts 模型。花费3days.花费3天。Take 3 days"
    print(split_alpha_nonalpha(text, mode=1))
    # 调用名为split_alpha_nonalpha的函数，传入text和mode参数，打印输出结果为指定mode下的分割结果

    text = "vits 和 Bert-VITS2 是 tts 模型。花费3days.花费3天。Take 3 days"
    print(split_alpha_nonalpha(text, mode=2))
    # 调用名为split_alpha_nonalpha的函数，传入text和mode参数，打印输出结果为指定mode下的分割结果
```