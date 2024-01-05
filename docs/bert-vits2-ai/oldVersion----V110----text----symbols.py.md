# `d:/src/tocomm/Bert-VITS2\oldVersion\V110\text\symbols.py`

```
punctuation = ["!", "?", "…", ",", ".", "'", "-"]
# 定义一个包含标点符号的列表

pu_symbols = punctuation + ["SP", "UNK"]
# 将标点符号列表与["SP", "UNK"]列表合并，得到一个包含特殊符号的列表

pad = "_"
# 定义一个表示填充字符的变量

# chinese
zh_symbols = [
    "E",
    "En",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    ...
]
# 定义一个包含中文字符的列表，用于表示中文文本的各个音节或字母
# 创建一个包含字符串的列表，每个字符串代表一个拼音音节
pinyin_list = [
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l"
]
```

这段代码创建了一个包含拼音音节的列表。每个字符串代表一个拼音音节，例如"eng"代表"eng"音节，"er"代表"er"音节，依此类推。这个列表将在后续的代码中使用。
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
    "m",    # 字符串 "m"
    "n",    # 字符串 "n"
    "o",    # 字符串 "o"
    "ong",  # 字符串 "ong"
    "ou",   # 字符串 "ou"
    "p",    # 字符串 "p"
    "q",    # 字符串 "q"
    "r",    # 字符串 "r"
    "s",    # 字符串 "s"
    "sh",   # 字符串 "sh"
    "t",    # 字符串 "t"
    "u",    # 字符串 "u"
    "ua",   # 字符串 "ua"
    "uai",  # 字符串 "uai"
    "uan",  # 字符串 "uan"
    "uang", # 字符串 "uang"
    "ui",   # 字符串 "ui"
    "un",   # 字符串 "un"
    "uo",   # 字符串 "uo"
    "v",    # 字符串 "v"
```

这段代码是一个函数的注释，函数名为`read_zip`，接受一个参数`fname`，用于指定 ZIP 文件的文件名。函数的作用是读取 ZIP 文件中的内容，并将文件名与数据组成字典返回。

在函数内部，首先根据 ZIP 文件名读取其二进制内容，并将其封装成字节流对象`bio`。然后使用字节流对象创建 ZIP 对象`zip`，用于操作 ZIP 文件。接下来，通过遍历 ZIP 对象中包含的文件名，读取每个文件的数据，并将文件名与数据组成字典`fdict`。最后，关闭 ZIP 对象，并返回结果字典`fdict`。
    "van",  # 姓氏 "van"
    "ve",  # 姓氏 "ve"
    "vn",  # 姓氏 "vn"
    "w",  # 姓氏 "w"
    "x",  # 姓氏 "x"
    "y",  # 姓氏 "y"
    "z",  # 姓氏 "z"
    "zh",  # 姓氏 "zh"
    "AA",  # 姓氏 "AA"
    "EE",  # 姓氏 "EE"
    "OO",  # 姓氏 "OO"
]
num_zh_tones = 6  # 中文音调的数量

# 日语
ja_symbols = [
    "N",  # 日语中的 "ん"
    "a",  # 日语中的 "あ"
    "a:",  # 日语中的 "あ:"
    "b",  # 日语中的 "ぶ"
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
    "by",  # 用于表示年份的后两位
    "ch",  # 用于表示月份的后两位
    "d",   # 用于表示日期的后两位
    "dy",  # 用于表示日期的后两位
    "e",   # 用于表示文件扩展名的后两位
    "e:",  # 用于表示文件扩展名的后两位
    "f",   # 用于表示文件名的后两位
    "g",   # 用于表示文件名的后两位
    "gy",  # 用于表示年份的后两位
    "h",   # 用于表示小时的后两位
    "hy",  # 用于表示小时的后两位
    "i",   # 用于表示分钟的后两位
    "i:",  # 用于表示分钟的后两位
    "j",   # 用于表示天数的后两位
    "k",   # 用于表示小时的后两位
    "ky",  # 用于表示年份的后两位
    "m",   # 用于表示月份的后两位
    "my",  # 用于表示月份的后两位
    "n",   # 用于表示文件名的后两位
    "ny",  # 用于表示年份的后两位
```

这些代码是用于表示日期、时间和文件名的后两位的字符串。它们可能是用于命名文件或记录日期和时间的变量。
    "o",        # 字符串 "o"
    "o:",       # 字符串 "o:"
    "p",        # 字符串 "p"
    "py",       # 字符串 "py"
    "q",        # 字符串 "q"
    "r",        # 字符串 "r"
    "ry",       # 字符串 "ry"
    "s",        # 字符串 "s"
    "sh",       # 字符串 "sh"
    "t",        # 字符串 "t"
    "ts",       # 字符串 "ts"
    "ty",       # 字符串 "ty"
    "u",        # 字符串 "u"
    "u:",       # 字符串 "u:"
    "w",        # 字符串 "w"
    "y",        # 字符串 "y"
    "z",        # 字符串 "z"
    "zy",       # 字符串 "zy"
]
num_ja_tones = 1    # 整数变量 num_ja_tones 被赋值为 1，表示日语音调的数量为 1
# 创建一个包含英文音标的列表
en_symbols = [
    "aa",  # 音标 aa
    "ae",  # 音标 ae
    "ah",  # 音标 ah
    "ao",  # 音标 ao
    "aw",  # 音标 aw
    "ay",  # 音标 ay
    "b",   # 音标 b
    "ch",  # 音标 ch
    "d",   # 音标 d
    "dh",  # 音标 dh
    "eh",  # 音标 eh
    "er",  # 音标 er
    "ey",  # 音标 ey
    "f",   # 音标 f
    "g",   # 音标 g
    "hh",  # 音标 hh
    "ih",  # 音标 ih
    # ... 继续添加其他音标
]
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
    "iy",  # 表示元音音素 /iy/
    "jh",  # 表示辅音音素 /jh/
    "k",   # 表示辅音音素 /k/
    "l",   # 表示辅音音素 /l/
    "m",   # 表示辅音音素 /m/
    "n",   # 表示辅音音素 /n/
    "ng",  # 表示辅音音素 /ng/
    "ow",  # 表示元音音素 /ow/
    "oy",  # 表示元音音素 /oy/
    "p",   # 表示辅音音素 /p/
    "r",   # 表示辅音音素 /r/
    "s",   # 表示辅音音素 /s/
    "sh",  # 表示辅音音素 /sh/
    "t",   # 表示辅音音素 /t/
    "th",  # 表示辅音音素 /th/
    "uh",  # 表示元音音素 /uh/
    "uw",  # 表示元音音素 /uw/
    "V",   # 表示元音音素 /V/
    "w",   # 表示辅音音素 /w/
    "y",   # 表示辅音音素 /y/
```

这段代码是一个函数 `read_zip`，用于读取 ZIP 文件中的内容并返回一个字典，其中键是文件名，值是文件的数据。

在函数中，首先根据 ZIP 文件名创建一个字节流对象 `bio`，然后使用这个字节流对象创建一个 ZIP 对象 `zip`。接下来，通过遍历 ZIP 对象的文件名列表，读取每个文件的数据，并将文件名和数据组成一个字典 `fdict`。最后，关闭 ZIP 对象，并返回结果字典。

需要注释的代码是一些字符串，它们表示不同的音素。音素是语音中最小的语音单位，用于表示语音的基本音素。这些字符串是用来表示不同的音素，例如 "iy" 表示元音音素 /iy/，"jh" 表示辅音音素 /jh/，以此类推。这些注释提供了对这些字符串的解释和说明。
    "z",
    "zh",
]
num_en_tones = 4

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))
# 将中文、日文和英文的符号合并，并按照字母顺序排序

symbols = [pad] + normal_symbols + pu_symbols
# 将特殊符号、合并后的符号和标点符号合并成一个列表

sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]
# 获取标点符号在符号列表中的索引

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones
# 将中文、日文和英文的音调数量相加得到总音调数量

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}
# 定义语言到ID的映射关系

num_languages = len(language_id_map.keys())
# 获取语言数量

language_tone_start_map = {
    "ZH": 0,
    "JP": num_zh_tones,
```
# 定义每种语言的音调起始位置，中文音调起始位置为0，日文音调起始位置为中文音调数量
    "EN": num_zh_tones + num_ja_tones,
}
```
这段代码是一个字典的定义，键为"EN"，值为`num_zh_tones + num_ja_tones`的结果。

```
if __name__ == "__main__":
    a = set(zh_symbols)
    b = set(en_symbols)
    print(sorted(a & b))
```
这段代码是一个条件语句，判断当前模块是否作为主程序运行。如果是，则执行以下代码：
- 创建一个集合a，其中包含`zh_symbols`中的元素。
- 创建一个集合b，其中包含`en_symbols`中的元素。
- 打印集合a和集合b的交集，按照升序排序后的结果。
```