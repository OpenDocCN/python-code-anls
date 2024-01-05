# `d:/src/tocomm/Bert-VITS2\oldVersion\V101\text\english.py`

```
import pickle  # 导入pickle模块，用于序列化和反序列化对象
import os  # 导入os模块，用于操作文件和目录
import re  # 导入re模块，用于正则表达式匹配
from g2p_en import G2p  # 从g2p_en模块中导入G2p类

from text import symbols  # 从text模块中导入symbols变量

current_file_path = os.path.dirname(__file__)  # 获取当前文件所在目录的路径
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")  # 将当前文件所在目录路径与"cmudict.rep"拼接成完整路径
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")  # 将当前文件所在目录路径与"cmudict_cache.pickle"拼接成完整路径
_g2p = G2p()  # 创建一个G2p对象，用于将英文文本转换为音素表示

arpa = {  # 创建一个集合，包含一些音素
    "AH0",
    "S",
    "AH1",
    "EY2",
    "AE2",
    "EH0",
    "OW2",
这段代码是一个字符串列表，每个字符串代表一个单词的发音。这些字符串是用国际音标表示的，用于语音识别或语音合成等应用。
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

需要注释的代码：

```
    "IY2",  # 表示音标中的元音音素 /iy/ 的第二个发音
    "T",  # 表示音标中的辅音音素 /t/
    "AA1",  # 表示音标中的元音音素 /aa/ 的第一个发音
    "ER1",  # 表示音标中的元音音素 /er/ 的第一个发音
    "EH2",  # 表示音标中的元音音素 /eh/ 的第二个发音
    "OY0",  # 表示音标中的元音音素 /oy/ 的第零个发音
    "UH2",  # 表示音标中的元音音素 /uh/ 的第二个发音
    "UW1",  # 表示音标中的元音音素 /uw/ 的第一个发音
    "Z",  # 表示音标中的辅音音素 /z/
    "AW2",  # 表示音标中的元音音素 /aw/ 的第二个发音
    "AW1",  # 表示音标中的元音音素 /aw/ 的第一个发音
    "V",  # 表示音标中的辅音音素 /v/
    "UW2",  # 表示音标中的元音音素 /uw/ 的第二个发音
    "AA2",  # 表示音标中的元音音素 /aa/ 的第二个发音
    "ER",  # 表示音标中的元音音素 /er/
    "AW0",  # 表示音标中的元音音素 /aw/ 的第零个发音
    "UW0",  # 表示音标中的元音音素 /uw/ 的第零个发音
    "R",  # 表示音标中的辅音音素 /r/
    "OW1",  # 表示音标中的元音音素 /ow/ 的第一个发音
    "EH1",  # 表示音标中的元音音素 /eh/ 的第一个发音
# 定义一个字符串列表，包含了一些音标的表示
phonemes = [
    "ZH",
    "AE0",
    "IH2",
    "IH",
    "Y",
    "JH",
    "P",
    "AY1",
    "EY0",
    "OY2",
    "TH",
    "HH",
    "D",
    "ER0",
    "CH",
    "AO1",
    "AE1",
    "AO2",
    "OY1",
    "AY2",
]
    "IH1",
    "OW0",
    "L",
    "SH",
}
```
这部分代码是一个包含字符串的集合。它定义了一组音素，用于表示英语中的不同音节。

```
def post_replace_ph(ph):
```
这是一个函数定义，函数名为`post_replace_ph`，它接受一个参数`ph`。

```
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
```
这部分代码定义了一个字典`rep_map`，它将一些特定字符映射到其他字符。这个字典用于替换文本中的特殊字符。
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph
```

注释：
```
    }
    if ph in rep_map.keys():  # 如果ph在rep_map的键中
        ph = rep_map[ph]  # 将ph替换为rep_map中对应的值
    if ph in symbols:  # 如果ph在symbols中
        return ph  # 返回ph
    if ph not in symbols:  # 如果ph不在symbols中
        ph = "UNK"  # 将ph替换为"UNK"
    return ph  # 返回ph


def read_dict():
    g2p_dict = {}  # 创建一个空字典g2p_dict
    start_line = 49  # 设置起始行号为49
    with open(CMU_DICT_PATH) as f:  # 打开CMU_DICT_PATH文件，并将其赋值给变量f
        line = f.readline()  # 读取文件的一行，并将其赋值给变量line
        line_index = 1  # 设置行号为1
        while line:  # 当line不为空时，执行循环
            if line_index >= start_line:  # 如果行号大于等于起始行号
                line = line.strip()  # 去除行首行尾的空格和换行符
                word_split = line.split("  ")  # 将行按照两个空格进行分割，得到一个列表word_split
                word = word_split[0]
```
这行代码将`word_split`列表中的第一个元素赋值给变量`word`。

```
                syllable_split = word_split[1].split(" - ")
```
这行代码将`word_split`列表中的第二个元素按照" - "进行分割，并将分割后的结果赋值给变量`syllable_split`。

```
                g2p_dict[word] = []
```
这行代码创建一个空列表，并将其赋值给字典`g2p_dict`中以`word`为键的值。

```
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)
```
这段代码遍历`syllable_split`列表中的每个元素，并将每个元素按照空格进行分割，并将分割后的结果添加到字典`g2p_dict`中以`word`为键的值的列表中。

```
            line_index = line_index + 1
            line = f.readline()
```
这两行代码分别将`line_index`的值加1，并将文件`f`的下一行内容赋值给变量`line`。

```
    return g2p_dict
```
这行代码将字典`g2p_dict`作为函数的返回值。

```
def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)
```
这段代码定义了一个名为`cache_dict`的函数，该函数接受两个参数`g2p_dict`和`file_path`。在函数内部，它使用`open`函数打开一个文件，并将字典`g2p_dict`以二进制形式写入到文件中。

```
def get_dict():
```
这行代码定义了一个名为`get_dict`的函数，该函数没有参数。
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict
```

这段代码的作用是从缓存中获取一个字典对象。它首先检查是否存在缓存文件（通过`os.path.exists(CACHE_PATH)`判断），如果存在，则使用`pickle.load()`函数从缓存文件中加载字典对象。如果缓存文件不存在，则调用`read_dict()`函数获取字典对象，并将其缓存到文件中（通过`cache_dict(g2p_dict, CACHE_PATH)`）。最后，返回获取到的字典对象。

```
eng_dict = get_dict()
```

这段代码的作用是调用`get_dict()`函数获取一个英文字典对象，并将其赋值给变量`eng_dict`。

```
def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone
```

这段代码定义了一个函数`refine_ph()`，它接受一个参数`phn`。函数首先将变量`tone`初始化为0。然后，通过正则表达式`re.search(r"\d$", phn)`判断`phn`是否以数字结尾。如果是，则将最后一个数字转换为整数并加1，赋值给变量`tone`。同时，将`phn`去除最后一个字符。最后，返回`phn`的小写形式和`tone`的值作为一个元组。
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
def refine_syllables(syllables):
    # 创建空列表用于存储音调
    tones = []
    # 创建空列表用于存储音素
    phonemes = []
    # 遍历音节列表
    for phn_list in syllables:
        # 遍历每个音节
        for i in range(len(phn_list)):
            # 获取当前音节
            phn = phn_list[i]
            # 对当前音节进行处理，得到处理后的音节和音调
            phn, tone = refine_ph(phn)
            # 将处理后的音节添加到音素列表中
            phonemes.append(phn)
            # 将音调添加到音调列表中
            tones.append(tone)
    # 返回音素列表和音调列表
    return phonemes, tones
```

```
def text_normalize(text):
    # todo: eng text normalize
    # 返回原始文本
    return text
```

```
def g2p(text):
    # 创建空列表用于存储音素
    phones = []
    tones = []  # 初始化一个空列表，用于存储音调信息
    words = re.split(r"([,;.\-\?\!\s+])", text)  # 使用正则表达式将文本按照标点符号和空格进行分割，得到一个单词列表
    for w in words:  # 遍历单词列表
        if w.upper() in eng_dict:  # 如果单词的大写形式在英语字典中存在
            phns, tns = refine_syllables(eng_dict[w.upper()])  # 调用函数refine_syllables，将单词的音节信息和音调信息分别存储在phns和tns中
            phones += phns  # 将phns中的音节信息添加到phones列表中
            tones += tns  # 将tns中的音调信息添加到tones列表中
        else:  # 如果单词的大写形式在英语字典中不存在
            phone_list = list(filter(lambda p: p != " ", _g2p(w)))  # 调用函数_g2p将单词转换为音素列表，并过滤掉空格，将结果存储在phone_list中
            for ph in phone_list:  # 遍历音素列表
                if ph in arpa:  # 如果音素在arpa中存在
                    ph, tn = refine_ph(ph)  # 调用函数refine_ph，将音素和音调信息分别存储在ph和tn中
                    phones.append(ph)  # 将ph添加到phones列表中
                    tones.append(tn)  # 将tn添加到tones列表中
                else:  # 如果音素在arpa中不存在
                    phones.append(ph)  # 将音素添加到phones列表中
                    tones.append(0)  # 将音调设置为0，并添加到tones列表中
    # todo: implement word2ph
    word2ph = [1 for i in phones]  # 创建一个与phones列表长度相同的列表，每个元素都为1，并将其赋值给word2ph变量
phones = [post_replace_ph(i) for i in phones]
```
这行代码使用列表推导式，将列表`phones`中的每个元素`i`传入函数`post_replace_ph`进行处理，并将处理结果组成一个新的列表赋值给变量`phones`。

```
return phones, tones, word2ph
```
这行代码返回三个变量`phones`、`tones`和`word2ph`作为函数的结果。

```
if __name__ == "__main__":
```
这行代码判断当前模块是否作为主程序运行。

```
# print(get_dict())
# print(eng_word_to_phoneme("hello"))
print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))
```
这几行代码用于测试函数`g2p`的功能，分别传入不同的参数进行测试，并将结果打印输出。

```
# all_phones = set()
# for k, syllables in eng_dict.items():
#     for group in syllables:
#         for ph in group:
#             all_phones.add(ph)
# print(all_phones)
```
这几行代码用于统计英文词典`eng_dict`中所有音素的集合，并将结果打印输出。
```