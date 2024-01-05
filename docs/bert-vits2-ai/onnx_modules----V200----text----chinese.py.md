# `d:/src/tocomm/Bert-VITS2\onnx_modules\V200\text\chinese.py`

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
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
    "【": "'",  # 将中文方括号替换为单引号
    "】": "'",  # 将中文方括号替换为单引号
    "[": "'",   # 将中文方括号替换为单引号
    "]": "'",   # 将中文方括号替换为单引号
    "—": "-",   # 将中文破折号替换为英文破折号
    "～": "-",  # 将中文波浪线替换为英文破折号
    "~": "-",   # 将英文波浪线替换为英文破折号
    "「": "'",  # 将中文书名号替换为单引号
    "」": "'",  # 将中文书名号替换为单引号
}

tone_modifier = ToneSandhi()  # 创建一个ToneSandhi对象用于处理声调变化


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")  # 将特定中文词语替换为其他中文词语
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))  # 创建一个正则表达式模式，用于匹配需要替换的标点符号

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)  # 使用正则表达式模式替换文本中的标点符号
    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
    )
    # 使用正则表达式替换文本中除中文和标点符号外的所有字符为空字符串

    return replaced_text
    # 返回替换后的文本


def g2p(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    # 创建正则表达式模式，用于在文本中查找标点符号并去除前面的空格
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    # 使用正则表达式模式分割文本，去除空白行
    phones, tones, word2ph = _g2p(sentences)
    # 调用 _g2p 函数处理分割后的文本，获取音素、音调和单词到音素的映射
    assert sum(word2ph) == len(phones)
    assert len(word2ph) == len(text)  # Sometimes it will crash,you can add a try-catch.
    # 断言检查单词到音素的映射和音素列表的长度是否一致
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    # 在音素、音调和单词到音素的映射列表前后添加特殊标记
    return phones, tones, word2ph
    # 返回处理后的音素、音调和单词到音素的映射列表


def _get_initials_finals(word):
    # 定义一个函数用于获取单词的声母和韵母
    initials = []  # 创建一个空列表，用于存储声母
    finals = []  # 创建一个空列表，用于存储韵母
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)  # 使用lazy_pinyin函数获取带声调的拼音的声母
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )  # 使用lazy_pinyin函数获取带声调的拼音的韵母
    for c, v in zip(orig_initials, orig_finals):  # 遍历声母和韵母列表
        initials.append(c)  # 将声母添加到initials列表中
        finals.append(v)  # 将韵母添加到finals列表中
    return initials, finals  # 返回声母和韵母列表


def _g2p(segments):
    phones_list = []  # 创建一个空列表，用于存储音素
    tones_list = []  # 创建一个空列表，用于存储声调
    word2ph = []  # 创建一个空列表，用于存储单词的音素
    for seg in segments:  # 遍历输入的句子中的每个词
        seg = re.sub("[a-zA-Z]+", "", seg)  # 用空字符串替换句子中的所有英文单词
        seg_cut = psg.lcut(seg)  # 使用psg.lcut函数对句子进行分词
        # 创建空列表用于存储每个词语的声母和韵母
        initials = []
        finals = []
        # 对分词结果进行预处理，合并相邻的音调
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        # 遍历分词结果中的每个词语和词性
        for word, pos in seg_cut:
            # 如果词性为英文，则跳过当前循环，继续下一个词语
            if pos == "eng":
                continue
            # 获取当前词语的声母和韵母
            sub_initials, sub_finals = _get_initials_finals(word)
            # 根据词语和词性修正韵母的音调
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            # 将当前词语的声母和韵母添加到对应的列表中
            initials.append(sub_initials)
            finals.append(sub_finals)

            # assert len(sub_initials) == len(sub_finals) == len(word)
        # 将列表中的列表合并成一个列表
        initials = sum(initials, [])
        finals = sum(finals, [])
        #
        # 遍历声母和韵母列表，同时获取对应位置的元素
        for c, v in zip(initials, finals):
            # 将声母和韵母拼接成原始的拼音
            raw_pinyin = c + v
            # 注意：对 pypinyin 输出进行后处理
            # 区分 i, ii 和 iii
            if c == v:
                # 确保字符 c 在标点符号中
                assert c in punctuation
                # 创建一个包含字符 c 的列表
                phone = [c]
                # 设置音调为 "0"
                tone = "0"
                # 将 1 添加到 word2ph 列表中
                word2ph.append(1)
            else:
                # 从声母和韵母中分离出音调
                v_without_tone = v[:-1]
                tone = v[-1]

                # 将声母和韵母组合成拼音
                pinyin = c + v_without_tone
                # 确保音调在 "12345" 中
                assert tone in "12345"

                if c:
                    # 对于多音节的情况
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    # 如果韵母在替换映射中
                    if v_without_tone in v_rep_map.keys():
                        # 替换韵母
                        pinyin = c + v_rep_map[v_without_tone]
                    else:
                        # 单音节
                        # 定义需要替换的拼音与替换后的拼音的映射关系
                        pinyin_rep_map = {
                            "ing": "ying",
                            "i": "yi",
                            "in": "yin",
                            "u": "wu",
                        }
                        # 如果当前拼音在映射关系中，则替换为对应的拼音
                        if pinyin in pinyin_rep_map.keys():
                            pinyin = pinyin_rep_map[pinyin]
                        else:
                            # 定义需要替换的单音节拼音与替换后的拼音的映射关系
                            single_rep_map = {
                                "v": "yu",
                                "e": "e",
                                "i": "y",
                                "u": "w",
                            }
                            # 如果当前拼音的第一个字母在映射关系中，则替换为对应的拼音
                            if pinyin[0] in single_rep_map.keys():
                                pinyin = single_rep_map[pinyin[0]] + pinyin[1:]
                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                # 检查拼音是否在拼音到符号映射的键中，如果不在则抛出异常，同时打印拼音、分词和原始拼音
                phone = pinyin_to_symbol_map[pinyin].split(" ")
                # 从拼音到符号映射中获取对应的音素，并按空格分割成列表
                word2ph.append(len(phone))
                # 将音素列表的长度添加到word2ph列表中

            phones_list += phone
            # 将phone列表中的元素添加到phones_list列表中
            tones_list += [int(tone)] * len(phone)
            # 将tone转换为整数后，重复len(phone)次，然后添加到tones_list列表中
    return phones_list, tones_list, word2ph
    # 返回phones_list、tones_list和word2ph列表


def text_normalize(text):
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    # 使用正则表达式查找文本中的数字
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
        # 将文本中的数字替换为中文数字
    text = replace_punctuation(text)
    # 调用replace_punctuation函数，替换文本中的标点符号
    return text
    # 返回处理后的文本


def get_bert_feature(text, word2ph):
    from text import chinese_bert
    # 从text模块中导入chinese_bert模块
    return chinese_bert.get_bert_feature(text, word2ph)
# 调用 chinese_bert 模块中的 get_bert_feature 函数，传入文本和 word2ph 参数，并返回结果


if __name__ == "__main__":
    from text.chinese_bert import get_bert_feature
# 如果当前脚本被直接执行，则导入 text.chinese_bert 模块中的 get_bert_feature 函数


    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)
    print(text)
    # 对文本进行规范化处理
    phones, tones, word2ph = g2p(text)
    # 调用 g2p 函数，传入文本，获取 phones, tones, word2ph 变量
    bert = get_bert_feature(text, word2ph)
    # 调用 get_bert_feature 函数，传入文本和 word2ph 参数，获取 bert 变量

    print(phones, tones, word2ph, bert.shape)
    # 打印 phones, tones, word2ph 变量的值，以及 bert 变量的形状信息


# # 示例用法
# text = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
# 打印使用 g2p_paddle 函数处理文本后的结果
```