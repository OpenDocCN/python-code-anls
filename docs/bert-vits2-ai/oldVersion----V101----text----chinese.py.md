# `d:/src/tocomm/Bert-VITS2\oldVersion\V101\text\chinese.py`

```
import os  # 导入os模块，用于操作文件和目录
import re  # 导入re模块，用于正则表达式的匹配和替换

import cn2an  # 导入cn2an模块，用于中文数字和阿拉伯数字的转换
from pypinyin import lazy_pinyin, Style  # 导入pypinyin模块，用于汉字转拼音

from .symbols import punctuation  # 从当前目录下的symbols模块中导入punctuation变量，用于标点符号的处理
from .tone_sandhi import ToneSandhi  # 从当前目录下的tone_sandhi模块中导入ToneSandhi类，用于音调变化的处理

current_file_path = os.path.dirname(__file__)  # 获取当前文件所在目录的路径
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]  # 读取opencpop-strict.txt文件的每一行，将拼音和对应的符号映射关系存储到字典中
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()  # 打开opencpop-strict.txt文件，逐行读取内容
}

import jieba.posseg as psg  # 导入jieba.posseg模块，用于中文分词和词性标注

rep_map = {  # 定义一个空字典，用于存储替换规则
# 将一些特殊字符替换为对应的标点符号
replace_dict = {
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
}
    "》": "'",  # 将中文标点符号“》”替换为英文单引号
    "【": "'",  # 将中文标点符号“【”替换为英文单引号
    "】": "'",  # 将中文标点符号“】”替换为英文单引号
    "[": "'",  # 将中文标点符号“[”替换为英文单引号
    "]": "'",  # 将中文标点符号“]”替换为英文单引号
    "—": "-",  # 将中文标点符号“—”替换为英文破折号
    "～": "-",  # 将中文标点符号“～”替换为英文破折号
    "~": "-",  # 将英文标点符号“~”替换为英文破折号
    "「": "'",  # 将中文标点符号“「”替换为英文单引号
    "」": "'",  # 将中文标点符号“」”替换为英文单引号
}

tone_modifier = ToneSandhi()  # 创建一个ToneSandhi对象，用于处理声调变化


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")  # 将文本中的“嗯”替换为“恩”，将文本中的“呣”替换为“母”
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))  # 创建一个正则表达式模式，用于匹配rep_map中的所有键

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)  # 使用正则表达式模式替换文本中匹配到的内容，并将结果赋值给replaced_text变量
# 使用正则表达式替换文本中的非中文字符和标点符号为空字符串
replaced_text = re.sub(
    r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
)

# 返回替换后的文本
return replaced_text


# 将文本分割成句子，并去除空白句子
def g2p(text):
    # 根据标点符号创建正则表达式模式
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    # 使用正则表达式模式将文本分割成句子，并去除空白句子
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    # 调用_g2p函数将句子转换为音素、音调和单词到音素的映射
    phones, tones, word2ph = _g2p(sentences)
    # 断言单词到音素的映射的总和等于音素的数量
    assert sum(word2ph) == len(phones)
    # 断言单词到音素的映射的长度等于文本的长度
    assert len(word2ph) == len(text)  # Sometimes it will crash,you can add a try-catch.
    # 在音素列表的开头和结尾添加占位符"_"
    phones = ["_"] + phones + ["_"]
    # 在音调列表的开头和结尾添加0
    tones = [0] + tones + [0]
    # 在单词到音素的映射列表的开头和结尾添加1
    word2ph = [1] + word2ph + [1]
    # 返回音素列表、音调列表和单词到音素的映射列表
    return phones, tones, word2ph
def _get_initials_finals(word):
    # 初始化一个空列表，用于存储拼音的声母
    initials = []
    # 初始化一个空列表，用于存储拼音的韵母
    finals = []
    # 使用lazy_pinyin函数将单词转换为拼音，设置参数neutral_tone_with_five为True，表示将轻声标记为5
    # 设置参数style为Style.INITIALS，表示只获取拼音的声母部分
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    # 使用lazy_pinyin函数将单词转换为拼音，设置参数neutral_tone_with_five为True，表示将轻声标记为5
    # 设置参数style为Style.FINALS_TONE3，表示只获取拼音的韵母部分，并且使用数字表示声调
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    # 使用zip函数将声母列表和韵母列表进行一一对应的组合
    for c, v in zip(orig_initials, orig_finals):
        # 将声母添加到声母列表中
        initials.append(c)
        # 将韵母添加到韵母列表中
        finals.append(v)
    # 返回声母列表和韵母列表
    return initials, finals


def _g2p(segments):
    # 初始化一个空列表，用于存储音素
    phones_list = []
    # 初始化一个空列表，用于存储音调
    tones_list = []
    # 初始化一个空列表，用于存储单词到音素的映射关系
    word2ph = []
    # 遍历segments列表中的每个元素
    for seg in segments:
        # 使用正则表达式将句子中的所有英文单词替换为空字符串
        seg = re.sub("[a-zA-Z]+", "", seg)
seg_cut = psg.lcut(seg)
```
将变量`seg`进行分词，得到分词结果，并赋值给变量`seg_cut`。

```
initials = []
finals = []
```
创建空列表`initials`和`finals`，用于存储拼音的声母和韵母。

```
seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
```
对`seg_cut`进行预处理，以便后续修改拼音的声调。

```
for word, pos in seg_cut:
```
遍历`seg_cut`中的每个词和词性。

```
if pos == "eng":
    continue
```
如果词性为"eng"（英文），则跳过当前循环，继续下一次循环。

```
sub_initials, sub_finals = _get_initials_finals(word)
```
调用函数`_get_initials_finals`，将当前词`word`作为参数传入，返回该词的声母和韵母，并分别赋值给变量`sub_initials`和`sub_finals`。

```
sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
```
调用函数`tone_modifier.modified_tone`，将当前词`word`、词性`pos`和韵母`sub_finals`作为参数传入，返回修改后的韵母，并赋值给变量`sub_finals`。

```
initials.append(sub_initials)
finals.append(sub_finals)
```
将`sub_initials`添加到`initials`列表末尾，将`sub_finals`添加到`finals`列表末尾。

```
initials = sum(initials, [])
finals = sum(finals, [])
```
将`initials`和`finals`列表中的所有元素合并为一个列表。

```
for c, v in zip(initials, finals):
```
使用`zip`函数将`initials`和`finals`中的元素一一配对。

```
raw_pinyin = c + v
```
将声母`c`和韵母`v`拼接成原始拼音`raw_pinyin`。

```
# NOTE: post process for pypinyin outputs
# we discriminate i, ii and iii
```
注释说明：对pypinyin输出进行后处理，区分i、ii和iii。
            if c == v:
                assert c in punctuation
                phone = [c]
                tone = "0"
                word2ph.append(1)
```
这段代码是一个条件语句，判断变量c是否等于变量v。如果相等，则断言变量c是标点符号，将变量c作为列表phone的元素，将字符串"0"赋值给变量tone，并将整数1添加到列表word2ph中。

```
            else:
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c + v_without_tone
                assert tone in "12345"

                if c:
                    # 多音节
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
```
这段代码是一个条件语句的else分支。首先，将变量v的最后一个字符赋值给变量tone，将变量v除去最后一个字符的部分赋值给变量v_without_tone。然后，将变量c和v_without_tone拼接成字符串pinyin。接下来，断言变量tone是字符串"1"、"2"、"3"、"4"或"5"中的一个。

如果变量c不为空，则进入下一个条件语句。在这个条件语句中，定义了一个字典v_rep_map，将一些多音节拼音的变体映射到标准拼音。然后，判断变量v_without_tone是否在v_rep_map的键中。
# 根据拼音转换规则替换拼音字符串中的部分音节
def replace_pinyin(pinyin):
    # 定义需要替换的拼音音节映射关系
    v_rep_map = {
        "a": "āáǎà",
        "o": "ōóǒò",
        "e": "ēéěè",
        "i": "īíǐì",
        "u": "ūúǔù",
        "v": "ǖǘǚǜ",
    }
    # 如果拼音字符串长度大于1，表示多音节
    if len(pinyin) > 1:
        # 获取拼音字符串中的第一个音节
        c = pinyin[0]
        # 获取拼音字符串中的第二个音节（去除音调）
        v_without_tone = pinyin[1:].replace("1", "").replace("2", "").replace("3", "").replace("4", "")
        # 如果第二个音节在映射关系中，替换为对应的音节
        if v_without_tone in v_rep_map.keys():
            pinyin = c + v_rep_map[v_without_tone]
    else:
        # 单音节
        # 定义需要替换的拼音映射关系
        pinyin_rep_map = {
            "ing": "ying",
            "i": "yi",
            "in": "yin",
            "u": "wu",
        }
        # 如果拼音在映射关系中，替换为对应的拼音
        if pinyin in pinyin_rep_map.keys():
            pinyin = pinyin_rep_map[pinyin]
        else:
            # 定义需要替换的单音节映射关系
            single_rep_map = {
                "v": "yu",
                "e": "e",
                "i": "y",
                "u": "w",
            }
            # 如果拼音的第一个音节在映射关系中，替换为对应的音节
            if pinyin[0] in single_rep_map.keys():
                pinyin = single_rep_map[pinyin[0]] + pinyin[1:]
```

这段代码是一个用于替换拼音字符串中的部分音节的函数。函数接受一个拼音字符串作为参数，并根据一定的规则对拼音字符串进行替换。

首先，定义了一个字典 `v_rep_map`，用于存储需要替换的拼音音节映射关系。其中，键是需要替换的音节，值是替换后的音节。

然后，判断拼音字符串的长度。如果长度大于1，表示拼音字符串是多音节的，需要进行替换。获取拼音字符串中的第一个音节 `c`，并获取拼音字符串中的第二个音节（去除音调）`v_without_tone`。如果第二个音节在映射关系中，将其替换为对应的音节，并将结果赋值给拼音字符串 `pinyin`。

如果拼音字符串的长度为1，表示拼音字符串是单音节的，同样需要进行替换。首先定义了一个字典 `pinyin_rep_map`，用于存储需要替换的拼音映射关系。如果拼音在映射关系中，将其替换为对应的拼音，并将结果赋值给拼音字符串 `pinyin`。如果拼音不在映射关系中，再定义一个字典 `single_rep_map`，用于存储需要替换的单音节映射关系。如果拼音的第一个音节在映射关系中，将其替换为对应的音节，并将结果赋值给拼音字符串 `pinyin`。

最后，函数没有返回值，只是对传入的拼音字符串进行了替换操作。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```

```
# 根据拼音找到对应的音标，并将音标转换为音素
def get_phones(pinyin_list):
    # 加载拼音到音标的映射字典
    pinyin_to_symbol_map = load_pinyin_to_symbol_map()
    # 初始化音素列表
    phones_list = []
    # 初始化音调列表
    tones_list = []
    # 初始化单词到音素数量的字典
    word2ph = []
    # 遍历拼音列表
    for pinyin in pinyin_list:
        # 检查拼音是否在映射字典中
        assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
        # 根据拼音找到对应的音标，并将音标按空格分割为音素列表
        phone = pinyin_to_symbol_map[pinyin].split(" ")
        # 将音素数量添加到单词到音素数量的字典中
        word2ph.append(len(phone))
        # 将音素列表添加到总音素列表中
        phones_list += phone
        # 将音调添加到总音调列表中，数量与音素列表中的音素数量相同
        tones_list += [int(tone)] * len(phone)
    # 返回音素列表、音调列表和单词到音素数量的字典
    return phones_list, tones_list, word2ph
```

```
# 对文本进行规范化处理
def text_normalize(text):
    # 使用正则表达式找到文本中的数字
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    # 遍历找到的数字列表
    for number in numbers:
        # 将文本中的数字替换为中文数字
        text = text.replace(number, cn2an.an2cn(number), 1)
    # 替换文本中的标点符号
    text = replace_punctuation(text)
    # 返回规范化后的文本
    return text
```

```
# 获取文本的 BERT 特征
def get_bert_feature(text, word2ph):
    # 导入中文 BERT 模型
    from text import chinese_bert
    # ...
    # 其他代码
    # ...
# 导入chinese_bert模块中的get_bert_feature函数
from text.chinese_bert import get_bert_feature

# 定义文本内容
text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
# 对文本进行规范化处理
text = text_normalize(text)
# 打印规范化后的文本
print(text)
# 调用g2p函数，将文本转换为音素、音调和word2ph字典
phones, tones, word2ph = g2p(text)
# 调用get_bert_feature函数，获取文本的BERT特征
bert = get_bert_feature(text, word2ph)

# 打印音素、音调、word2ph字典和BERT特征的形状
print(phones, tones, word2ph, bert.shape)


# 示例用法
# text = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
```

在给定的代码中，注释解释了每个语句的作用：

1. 导入chinese_bert模块中的get_bert_feature函数。
2. 定义文本内容。
3. 对文本进行规范化处理。
4. 打印规范化后的文本。
5. 调用g2p函数，将文本转换为音素、音调和word2ph字典。
6. 调用get_bert_feature函数，获取文本的BERT特征。
7. 打印音素、音调、word2ph字典和BERT特征的形状。
8. 示例用法的注释被注释掉，不会被执行。
```