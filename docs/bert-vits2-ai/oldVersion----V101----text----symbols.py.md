# `Bert-VITS2\oldVersion\V101\text\symbols.py`

```

# 定义标点符号列表
punctuation = ["!", "?", "…", ",", ".", "'", "-"]
# 将标点符号列表与特殊符号列表合并
pu_symbols = punctuation + ["SP", "UNK"]
# 定义填充符号
pad = "_"

# 定义中文音素列表
zh_symbols = [
    # ...（省略部分内容）
]
num_zh_tones = 6

# 定义日文音素列表
ja_symbols = [
    # ...（省略部分内容）
]
num_ja_tones = 1

# 定义英文音素列表
en_symbols = [
    # ...（省略部分内容）
]
num_en_tones = 4

# 合并所有音素列表并去重排序
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))
# 将填充符号、普通音素和特殊符号合并成一个总的音素列表
symbols = [pad] + normal_symbols + pu_symbols
# 找出特殊符号在总音素列表中的索引
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]

# 计算总的音调数量
num_tones = num_zh_tones + num_ja_tones + num_en_tones

# 定义语言到编号的映射
language_id_map = {"ZH": 0, "JA": 1, "EN": 2}
# 计算语言数量
num_languages = len(language_id_map.keys())

# 定义每种语言音调的起始位置
language_tone_start_map = {
    "ZH": 0,
    "JA": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

# 如果作为主程序运行，则找出中英文音素列表的交集并打印
if __name__ == "__main__":
    a = set(zh_symbols)
    b = set(en_symbols)
    print(sorted(a & b))

```