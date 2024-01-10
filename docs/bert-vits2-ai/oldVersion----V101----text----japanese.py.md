# `Bert-VITS2\oldVersion\V101\text\japanese.py`

```
# 从指定 URL 修改而来，导入所需的模块
import re  # 导入正则表达式模块
import sys  # 导入系统模块

import pyopenjtalk  # 导入 pyopenjtalk 模块

from . import symbols  # 从当前目录导入 symbols 模块

# 匹配不带标点符号的日文的正则表达式：
_japanese_characters = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# 匹配非日文字符或标点符号的正则表达式：
_japanese_marks = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# 标点符号的（符号，日文）对列表：
_symbols_to_japanese = [(re.compile("%s" % x[0]), x[1]) for x in [("％", "パーセント")]]

# 辅音和促音的（辅音，促音）对列表：
_real_sokuon = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        (r"Q([↑↓]*[kg])", r"k#\1"),
        (r"Q([↑↓]*[tdjʧ])", r"t#\1"),
        (r"Q([↑↓]*[sʃ])", r"s\1"),
        (r"Q([↑↓]*[pb])", r"p#\1"),
    ]
]

# 辅音和拨音的（辅音，拨音）对列表：
_real_hatsuon = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        (r"N([↑↓]*[pbm])", r"m\1"),
        (r"N([↑↓]*[ʧʥj])", r"n^\1"),
        (r"N([↑↓]*[tdn])", r"n\1"),
        (r"N([↑↓]*[kg])", r"ŋ\1"),
    ]
]


def post_replace_ph(ph):
    # 替换映射表
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
        "v": "V",
    }
    # 如果 ph 在映射表中，则替换为对应的值
    if ph in rep_map.keys():
        ph = rep_map[ph]
    # 如果 ph 在 symbols 中，则返回 ph
    if ph in symbols:
        return ph
    # 如果 ph 不在 symbols 中，则将其替换为 "UNK"
    if ph not in symbols:
        ph = "UNK"
    return ph


def symbols_to_japanese(text):
    # 对文本中的符号进行替换
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text


def preprocess_jap(text):
    """参考 https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html"""
    # 对文本进行符号到日文的替换
    text = symbols_to_japanese(text)
    # 使用非日文字符或标点符号进行分割，得到句子列表
    sentences = re.split(_japanese_marks, text)
    # 使用正则表达式在文本中查找日语标点符号
    marks = re.findall(_japanese_marks, text)
    # 初始化一个空列表用于存储处理后的文本
    text = []
    # 遍历句子列表
    for i, sentence in enumerate(sentences):
        # 如果句子符合日语字符的正则表达式
        if re.match(_japanese_characters, sentence):
            # 使用 pyopenjtalk 库将句子转换为日语音素
            p = pyopenjtalk.g2p(sentence)
            # 将转换后的音素以空格为分隔符拆分，并添加到文本列表中
            text += p.split(" ")

        # 如果索引小于标点符号列表的长度
        if i < len(marks):
            # 将当前索引对应的标点符号去除空格后添加到文本列表中
            text += [marks[i].replace(" ", "")]
    # 返回处理后的文本列表
    return text
# 文本规范化函数，目前未实现具体功能
def text_normalize(text):
    # todo: jap text normalize
    return text


# 文本转音素函数
def g2p(norm_text):
    # 预处理日语文本，得到音素列表
    phones = preprocess_jap(norm_text)
    # 对音素列表进行后处理
    phones = [post_replace_ph(i) for i in phones]
    # todo: 实现音调和单词到音素的转换
    tones = [0 for i in phones]
    word2ph = [1 for i in phones]
    return phones, tones, word2ph


# 主函数
if __name__ == "__main__":
    # 逐行读取文本文件中的内容
    for line in open("../../../Downloads/transcript_utf8.txt").readlines():
        # 提取文本内容
        text = line.split(":")[1]
        # 调用文本转音素函数
        phones, tones, word2ph = g2p(text)
        # 遍历音素列表
        for p in phones:
            # 如果音素为"z"，则打印文本内容和音素列表，并退出程序
            if p == "z":
                print(text, phones)
                sys.exit(0)
```