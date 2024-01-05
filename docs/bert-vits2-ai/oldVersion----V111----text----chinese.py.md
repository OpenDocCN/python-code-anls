# `d:/src/tocomm/Bert-VITS2\oldVersion\V111\text\chinese.py`

```
import os  # 导入os模块，用于操作文件和目录
import re  # 导入re模块，用于正则表达式的匹配和替换

import cn2an  # 导入cn2an模块，用于中文数字和阿拉伯数字的转换
from pypinyin import lazy_pinyin, Style  # 导入pypinyin模块，用于汉字转拼音

from .symbols import punctuation  # 从当前目录下的symbols模块中导入punctuation变量，用于标点符号的处理
from .tone_sandhi import ToneSandhi  # 从当前目录下的tone_sandhi模块中导入ToneSandhi类，用于音调变化的处理

current_file_path = os.path.dirname(__file__)  # 获取当前文件所在目录的路径
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]  # 读取opencpop-strict.txt文件中的每一行，将拼音和对应的符号映射关系存储到字典中
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()  # 打开opencpop-strict.txt文件，逐行读取内容
}

import jieba.posseg as psg  # 导入jieba.posseg模块，用于中文分词和词性标注

rep_map = {
    "：": ",",  # 将中文冒号替换为英文逗号
# 将给定的字符替换为相应的字符，用于文本处理
replace_dict = {
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
}
    "【": "'",  # 将中文的【替换为英文的'
    "】": "'",  # 将中文的】替换为英文的'
    "[": "'",  # 将中文的[替换为英文的'
    "]": "'",  # 将中文的]替换为英文的'
    "—": "-",  # 将中文的—替换为英文的-
    "～": "-",  # 将中文的～替换为英文的-
    "~": "-",  # 将英文的~替换为-
    "「": "'",  # 将中文的「替换为英文的'
    "」": "'",  # 将中文的」替换为英文的'
}

tone_modifier = ToneSandhi()  # 创建一个ToneSandhi对象，用于处理声调变化


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")  # 将文本中的"嗯"替换为"恩"，将"呣"替换为"母"
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))  # 创建一个正则表达式模式，用于匹配rep_map中的所有键

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)  # 使用正则表达式模式替换文本中匹配到的内容，并将结果赋值给replaced_text变量
replaced_text = re.sub(
    r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
)
```
这段代码使用正则表达式将`replaced_text`中的非中文字符和标点符号替换为空字符串。

```
return replaced_text
```
返回替换后的文本。

```
def g2p(text):
```
定义了一个名为`g2p`的函数，接受一个参数`text`。

```
pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
```
根据标点符号生成一个正则表达式模式，用于将文本分割成句子。

```
sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
```
使用正则表达式模式将文本分割成句子，并将非空的句子保存在`sentence`列表中。

```
phones, tones, word2ph = _g2p(sentences)
```
调用`_g2p`函数处理句子，返回音素、音调和单词到音素的映射。

```
assert sum(word2ph) == len(phones)
assert len(word2ph) == len(text)  # Sometimes it will crash,you can add a try-catch.
```
断言`word2ph`列表中的元素总和等于音素列表`phones`的长度，并且`word2ph`列表的长度等于原始文本`text`的长度。

```
phones = ["_"] + phones + ["_"]
tones = [0] + tones + [0]
word2ph = [1] + word2ph + [1]
```
在音素、音调和单词到音素的映射列表前后分别添加了特殊的标记。

```
return phones, tones, word2ph
```
返回音素、音调和单词到音素的映射列表。


```
def _get_initials_finals(word):
```
定义了一个名为`_get_initials_finals`的函数，接受一个参数`word`。
    initials = []
    finals = []
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    # 遍历原始声母和韵母列表，将每个元素添加到对应的列表中
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    # 返回声母列表和韵母列表
    return initials, finals


def _g2p(segments):
    phones_list = []
    tones_list = []
    word2ph = []
    # 遍历分段列表中的每个分段
    for seg in segments:
        # 将分段中的所有英文单词替换为空字符串
        seg = re.sub("[a-zA-Z]+", "", seg)
        # 使用结巴分词对分段进行分词
        seg_cut = psg.lcut(seg)
        initials = []  # 初始化一个空列表，用于存储每个词语的声母
        finals = []  # 初始化一个空列表，用于存储每个词语的韵母
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)  # 调用 tone_modifier 模块的 pre_merge_for_modify 函数对 seg_cut 进行预处理
        for word, pos in seg_cut:  # 遍历 seg_cut 中的每个词语和词性
            if pos == "eng":  # 如果词性为 "eng"，则跳过当前循环，继续下一次循环
                continue
            sub_initials, sub_finals = _get_initials_finals(word)  # 调用 _get_initials_finals 函数获取当前词语的声母和韵母
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)  # 调用 tone_modifier 模块的 modified_tone 函数对当前词语的韵母进行调整
            initials.append(sub_initials)  # 将当前词语的声母添加到 initials 列表中
            finals.append(sub_finals)  # 将当前词语的韵母添加到 finals 列表中

            # assert len(sub_initials) == len(sub_finals) == len(word)  # 断言：当前词语的声母、韵母和词语本身的长度应该相等
        initials = sum(initials, [])  # 将 initials 列表中的所有子列表合并成一个列表
        finals = sum(finals, [])  # 将 finals 列表中的所有子列表合并成一个列表
        #
        for c, v in zip(initials, finals):  # 遍历 initials 和 finals 列表中的元素，分别赋值给 c 和 v
            raw_pinyin = c + v  # 将 c 和 v 拼接成 raw_pinyin
            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if c == v:  # 如果 c 和 v 相等
                assert c in punctuation
```
这行代码用于断言变量c是否在标点符号集合中，如果不在则会抛出异常。

```
                phone = [c]
```
这行代码用于将变量c作为单个元素的列表赋值给变量phone。

```
                tone = "0"
```
这行代码用于将字符串"0"赋值给变量tone。

```
                word2ph.append(1)
```
这行代码用于将整数1添加到列表word2ph的末尾。

```
                v_without_tone = v[:-1]
                tone = v[-1]
```
这两行代码用于将变量v的最后一个字符赋值给变量tone，并将除去最后一个字符的部分赋值给变量v_without_tone。

```
                pinyin = c + v_without_tone
                assert tone in "12345"
```
这两行代码用于将变量c和变量v_without_tone拼接成字符串，并断言变量tone是否在字符串"12345"中，如果不在则会抛出异常。

```
                if c:
```
这行代码用于判断变量c是否为真（非空、非零），如果为真则执行下面的代码块。

```
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
```
这段代码用于创建一个字典v_rep_map，然后判断变量v_without_tone是否在字典的键集合中，如果在则将字典中对应的值与变量c拼接成字符串赋值给变量pinyin。
                else:
                    # 单音节
                    # 定义一个字典，用于存储需要替换的拼音
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    # 如果当前拼音在替换字典的键中，则将其替换为对应的值
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        # 定义一个字典，用于存储需要替换的单音节拼音
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        # 如果当前拼音的首字母在单音节替换字典的键中，则将其替换为对应的值加上原拼音的剩余部分
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]
```

这段代码是一个条件语句的分支，用于处理单音节的拼音。首先，定义了一个字典`pinyin_rep_map`，用于存储需要替换的拼音。如果当前拼音在替换字典的键中，则将其替换为对应的值。如果不在替换字典中，则继续判断是否为单音节拼音。如果是单音节拼音，则定义了另一个字典`single_rep_map`，用于存储需要替换的单音节拼音。如果当前拼音的首字母在单音节替换字典的键中，则将其替换为对应的值加上原拼音的剩余部分。
# 根据拼音在拼音到音标映射表中查找对应的音标，并将其添加到word2ph列表中
assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
phone = pinyin_to_symbol_map[pinyin].split(" ")
word2ph.append(len(phone))

# 将phone列表中的元素添加到phones_list列表中
phones_list += phone
# 将tone转换为整数，并将其重复len(phone)次后添加到tones_list列表中
tones_list += [int(tone)] * len(phone)

# 返回phones_list、tones_list和word2ph列表作为结果
return phones_list, tones_list, word2ph


# 对文本进行规范化处理，将文本中的数字替换为中文数字，去除标点符号
def text_normalize(text):
    # 使用正则表达式找到文本中的数字
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    # 遍历找到的数字列表
    for number in numbers:
        # 将数字替换为中文数字
        text = text.replace(number, cn2an.an2cn(number), 1)
    # 替换文本中的标点符号
    text = replace_punctuation(text)
    # 返回处理后的文本
    return text


# 获取文本的BERT特征
def get_bert_feature(text, word2ph):
    # 导入chinese_bert模块
    from text import chinese_bert
    return chinese_bert.get_bert_feature(text, word2ph)
```
这行代码调用了`chinese_bert`模块中的`get_bert_feature`函数，并返回其结果。它接受两个参数`text`和`word2ph`。

```
if __name__ == "__main__":
    from text.chinese_bert import get_bert_feature

    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)
```
这部分代码是程序的入口点，当直接运行这个脚本时会执行这部分代码。它首先从`text.chinese_bert`模块中导入`get_bert_feature`函数。然后定义了一个字符串变量`text`，并对其进行了规范化处理。接下来调用了`g2p`函数，将`text`作为参数传入，并将返回的结果分别赋值给`phones`、`tones`和`word2ph`变量。然后调用`get_bert_feature`函数，将`text`和`word2ph`作为参数传入，并将返回的结果赋值给`bert`变量。最后打印了`phones`、`tones`、`word2ph`和`bert.shape`的值。

```
# # 示例用法
# text = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
```
这部分代码是一个示例用法的注释，它展示了如何使用`g2p_paddle`函数。首先定义了一个字符串变量`text`，然后调用`g2p_paddle`函数，将`text`作为参数传入，并将返回的结果打印出来。
```