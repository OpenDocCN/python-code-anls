# `d:/src/tocomm/Bert-VITS2\oldVersion\V220\text\chinese.py`

```
# 导入 os 模块，用于操作文件和目录
import os
# 导入 re 模块，用于正则表达式操作
import re
# 导入 cn2an 模块，用于中文数字和阿拉伯数字的转换
import cn2an
# 从 pypinyin 模块中导入 lazy_pinyin 和 Style，用于将汉字转换为拼音
from pypinyin import lazy_pinyin, Style
# 从当前目录下的 symbols 模块中导入 punctuation 符号
from .symbols import punctuation
# 从当前目录下的 tone_sandhi 模块中导入 ToneSandhi 类
from .tone_sandhi import ToneSandhi

# 获取当前文件所在目录的路径
current_file_path = os.path.dirname(__file__)
# 创建拼音到符号的映射字典
pinyin_to_symbol_map = {
    # 读取 opencpop-strict.txt 文件中每一行的拼音和符号，组成映射关系
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

# 导入 jieba 模块中的 posseg，用于中文分词和词性标注
import jieba.posseg as psg

# 创建替换映射字典，用于替换中文符号
rep_map = {
    "：": ",",  # 将中文冒号替换为英文逗号
    "；": ",",  # 将中文分号替换为英文逗号
    "，": ",",  # 将中文逗号替换为英文逗号
    "。": ".",  # 将中文句号替换为英文句号
    "！": "!",  # 将中文感叹号替换为英文感叹号
    "？": "?",  # 将中文问号替换为英文问号
    "\n": ".",  # 将换行符替换为英文句号
    "·": ",",  # 将中文间隔点替换为英文逗号
    "、": ",",  # 将中文顿号替换为英文逗号
    "...": "…",  # 将中文省略号替换为英文省略号
    "$": ".",  # 将美元符号替换为英文句号
    "“": "'",  # 将中文左双引号替换为英文单引号
    "”": "'",  # 将中文右双引号替换为英文单引号
    '"': "'",  # 将双引号替换为单引号
    "‘": "'",  # 将中文左单引号替换为英文单引号
    "’": "'",  # 将中文右单引号替换为英文单引号
    "（": "'",  # 将中文左括号替换为英文单引号
    "）": "'",  # 将中文右括号替换为英文单引号
    "(": "'",  # 将左括号替换为单引号
    ")": "'",  # 将右括号替换为单引号
    "《": "'",  # 将中文左书名号替换为英文单引号
    "》": "'",  # 将中文标点符号“》”替换为英文单引号
    "【": "'",  # 将中文标点符号“【”替换为英文单引号
    "】": "'",  # 将中文标点符号“】”替换为英文单引号
    "[": "'",   # 将中文标点符号“[”替换为英文单引号
    "]": "'",   # 将中文标点符号“]”替换为英文单引号
    "—": "-",   # 将中文标点符号“—”替换为英文破折号
    "～": "-",  # 将中文标点符号“～”替换为英文破折号
    "~": "-",   # 将中文标点符号“~”替换为英文破折号
    "「": "'",  # 将中文标点符号“「”替换为英文单引号
    "」": "'",  # 将中文标点符号“」”替换为英文单引号
}

tone_modifier = ToneSandhi()  # 创建一个ToneSandhi对象的实例，用于处理声调变化

def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")  # 将中文词语“嗯”替换为“恩”，将中文词语“呣”替换为“母”
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))  # 创建一个正则表达式模式，用于匹配需要替换的中文标点符号

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)  # 使用正则表达式模式替换文本中的中文标点符号为对应的英文标点符号
    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
    )
    # 使用正则表达式替换所有非中文字符和标点符号的内容为空，保留中文字符和标点符号

    return replaced_text
    # 返回替换后的文本

def g2p(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    # 构建正则表达式的模式，用于在标点符号后面匹配零个或多个空格
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    # 使用正则表达式模式分割文本，去除空白行，得到句子列表
    phones, tones, word2ph = _g2p(sentences)
    # 调用 _g2p 函数处理句子列表，得到音素、音调和单词到音素的映射
    assert sum(word2ph) == len(phones)
    # 断言单词到音素的映射的和等于音素列表的长度
    assert len(word2ph) == len(text)  # Sometimes it will crash,you can add a try-catch.
    # 断言单词到音素的映射的长度等于文本的长度，有时会崩溃，可以添加 try-catch
    phones = ["_"] + phones + ["_"]
    # 在音素列表的开头和结尾添加占位符"_"
    tones = [0] + tones + [0]
    # 在音调列表的开头和结尾添加0
    word2ph = [1] + word2ph + [1]
    # 在单词到音素的映射列表的开头和结尾添加1
    return phones, tones, word2ph
    # 返回处理后的音素列表、音调列表和单词到音素的映射列表
def _get_initials_finals(word):
    # 初始化声母和韵母列表
    initials = []
    finals = []
    # 使用lazy_pinyin函数获取带有声调的拼音的声母和韵母
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    # 遍历声母和韵母，将其添加到对应的列表中
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    # 返回声母和韵母列表
    return initials, finals


def _g2p(segments):
    # 初始化音素列表和声调列表
    phones_list = []
    tones_list = []
    word2ph = []
    # 遍历句子中的每个词
    for seg in segments:
        # 替换句子中的所有英文单词为空字符串
        seg = re.sub("[a-zA-Z]+", "", seg)
        # 对输入的文本进行分词
        seg_cut = psg.lcut(seg)
        # 初始化声母和韵母列表
        initials = []
        finals = []
        # 对分词结果进行预处理
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        # 遍历分词结果，获取每个词的声母和韵母
        for word, pos in seg_cut:
            # 如果词性为英文，则跳过
            if pos == "eng":
                continue
            # 获取词的声母和韵母
            sub_initials, sub_finals = _get_initials_finals(word)
            # 对韵母进行调音处理
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            # 将声母和韵母添加到对应的列表中
            initials.append(sub_initials)
            finals.append(sub_finals)

            # 确保每个词的声母和韵母长度相等
            # assert len(sub_initials) == len(sub_finals) == len(word)
        # 将声母和韵母列表展开成一维列表
        initials = sum(initials, [])
        finals = sum(finals, [])
        #
        # 遍历声母和韵母列表，组合成原始拼音
        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            # 注意：对 pypinyin 输出进行后处理
            # 区分 i、ii 和 iii
            # 如果当前字符等于声母，则断言当前字符在标点符号中
            if c == v:
                assert c in punctuation
                # 创建一个包含当前字符的列表，并设置声调为 "0"
                phone = [c]
                tone = "0"
                # 将 1 添加到 word2ph 列表中
                word2ph.append(1)
            else:
                # 获取去除声调的拼音
                v_without_tone = v[:-1]
                # 获取声调
                tone = v[-1]

                # 拼接声母和去除声调的拼音，断言声调在 "12345" 中
                pinyin = c + v_without_tone
                assert tone in "12345"

                if c:
                    # 多音节
                    # 定义多音节的替换映射
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    # 如果去除声调的拼音在替换映射的键中
                    if v_without_tone in v_rep_map.keys():
                pinyin = c + v_rep_map[v_without_tone]
```
这行代码将当前处理的音节 c 与去掉声调的元音 v_without_tone 组合起来，得到完整的拼音音节。

```python
                else:
```
如果当前处理的音节不是复音节，则执行下面的代码。

```python
                    # 单音节
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
```
定义了一个单音节拼音的替换映射表，将特定的拼音替换为另一个拼音。

```python
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
```
如果当前处理的拼音在替换映射表中，则将其替换为对应的值，否则执行下面的代码。

```python
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
```
定义了一个单音节拼音首字母的替换映射表，将特定的拼音首字母替换为另一个首字母。

```python
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]
```
如果当前处理的拼音的首字母在替换映射表中，则将其替换为对应的值，并保留原拼音的其他部分。
                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                # 检查拼音是否在拼音到符号映射的键中，如果不在则抛出异常
                phone = pinyin_to_symbol_map[pinyin].split(" ")
                # 从拼音到符号映射中获取对应的音素，并以空格分割成列表
                word2ph.append(len(phone)
                # 将音素列表的长度添加到word2ph列表中

            phones_list += phone
            # 将phone列表中的元素添加到phones_list列表中
            tones_list += [int(tone)] * len(phone)
            # 将tone转换为整数后，重复len(phone)次，然后添加到tones_list列表中
    return phones_list, tones_list, word2ph
    # 返回phones_list, tones_list, word2ph三个列表


def text_normalize(text):
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    # 从文本中找到所有数字
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
        # 将文本中的数字替换为中文数字
    text = replace_punctuation(text)
    # 调用replace_punctuation函数，替换文本中的标点符号
    return text
    # 返回处理后的文本


def get_bert_feature(text, word2ph):
    from text import chinese_bert
    # 从text模块中导入chinese_bert函数
    return chinese_bert.get_bert_feature(text, word2ph)
# 调用 chinese_bert 模块中的 get_bert_feature 函数，传入文本和 word2ph 参数，并返回结果

if __name__ == "__main__":
    from text.chinese_bert import get_bert_feature
# 如果当前脚本被直接执行，则从 text.chinese_bert 模块中导入 get_bert_feature 函数

    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)
# 对文本进行预处理，然后调用 g2p 函数获取 phones, tones, word2ph，最后调用 get_bert_feature 函数获取 bert

    print(phones, tones, word2ph, bert.shape)
# 打印 phones, tones, word2ph 和 bert 的形状

# # 示例用法
# text = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
# 示例用法的注释被注释掉了，不会被执行
```