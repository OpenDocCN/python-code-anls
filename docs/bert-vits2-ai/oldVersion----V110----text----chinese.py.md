# `Bert-VITS2\oldVersion\V110\text\chinese.py`

```
# 导入 os 模块
import os
# 导入 re 模块
import re
# 导入 cn2an 模块
import cn2an
# 从 pypinyin 模块中导入 lazy_pinyin 和 Style
from pypinyin import lazy_pinyin, Style
# 从当前目录下的 symbols 模块中导入 punctuation 符号
from .symbols import punctuation
# 从当前目录下的 tone_sandhi 模块中导入 ToneSandhi 类
from .tone_sandhi import ToneSandhi

# 获取当前文件所在目录路径
current_file_path = os.path.dirname(__file__)
# 创建拼音到符号的映射字典
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

# 导入 jieba.posseg 模块并重命名为 psg
import jieba.posseg as psg

# 创建替换映射字典
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
    "$": ".",
    "“": "'",
    "”": "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

# 创建 ToneSandhi 实例
tone_modifier = ToneSandhi()

# 定义替换标点符号的函数
def replace_punctuation(text):
    # 替换特定词语
    text = text.replace("嗯", "恩").replace("呣", "母")
    # 编译正则表达式模式
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    # 使用替换映射字典替换文本中的标点符号
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    # 使用正则表达式去除非中文字符和标点符号
    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
    )
    # 返回替换后的文本
    return replaced_text

# 定义将文本转换为拼音的函数
def g2p(text):
    # 定义正则表达式模式
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    # 根据正则表达式模式分割文本为句子列表
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    # 调用 _g2p 函数处理句子列表
    phones, tones, word2ph = _g2p(sentences)
    # 断言检查
    assert sum(word2ph) == len(phones)
    assert len(word2ph) == len(text)  # Sometimes it will crash,you can add a try-catch.
    # 对 phones, tones, word2ph 进行处理
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    # 返回处理后的结果
    return phones, tones, word2ph

# 定义获取声母和韵母的函数
def _get_initials_finals(word):
    # 初始化声母和韵母列表
    initials = []
    finals = []
    # 获取原始声母和韵母
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    # 遍历两个列表中对应位置的元素，分别添加到initials和finals列表中
    for c, v in zip(orig_initials, orig_finals):
        # 将元音字母c添加到initials列表中
        initials.append(c)
        # 将辅音字母v添加到finals列表中
        finals.append(v)
    # 返回initials和finals列表
    return initials, finals
# 定义一个函数_g2p，接受一个参数segments，返回三个空列表
def _g2p(segments):
    phones_list = []  # 存储音素的列表
    tones_list = []  # 存储音调的列表
    word2ph = []  # 存储单词到音素的映射关系
    return phones_list, tones_list, word2ph  # 返回三个空列表


# 定义一个函数text_normalize，接受一个参数text，对文本进行数字和标点的处理
def text_normalize(text):
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)  # 使用正则表达式找出文本中的数字
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)  # 将数字转换为中文数字
    text = replace_punctuation(text)  # 替换文本中的标点符号
    return text  # 返回处理后的文本


# 定义一个函数get_bert_feature，接受两个参数text和word2ph，调用chinese_bert模块的get_bert_feature函数
def get_bert_feature(text, word2ph):
    from text import chinese_bert  # 导入chinese_bert模块

    return chinese_bert.get_bert_feature(text, word2ph)  # 调用chinese_bert模块的get_bert_feature函数


# 如果该脚本作为主程序运行
if __name__ == "__main__":
    from text.chinese_bert import get_bert_feature  # 从text.chinese_bert模块导入get_bert_feature函数

    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)  # 对文本进行数字和标点的处理
    print(text)  # 打印处理后的文本
    phones, tones, word2ph = g2p(text)  # 调用g2p函数，获取音素、音调和单词到音素的映射关系
    bert = get_bert_feature(text, word2ph)  # 调用get_bert_feature函数，获取BERT特征
    print(phones, tones, word2ph, bert.shape)  # 打印音素、音调、单词到音素的映射关系和BERT特征的形状


# 示例用法
# text = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
```