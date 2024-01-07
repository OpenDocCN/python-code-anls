# `Bert-VITS2\oldVersion\V200\text\symbols.py`

```

# 定义标点符号列表
punctuation = ["!", "?", "…", ",", ".", "'", "-"]
# 将标点符号列表与额外的符号列表合并
pu_symbols = punctuation + ["SP", "UNK"]
# 定义填充符号
pad = "_"

# 定义中文音素列表
zh_symbols = [
    # ...（省略部分内容）
]
num_zh_tones = 6  # 定义中文音调数量

# 定义日文音素列表
ja_symbols = [
    # ...（省略部分内容）
]
num_ja_tones = 2  # 定义日文音调数量

# 定义英文音素列表
en_symbols = [
    # ...（省略部分内容）
]
num_en_tones = 4  # 定义英文音调数量

# 合并所有音素列表并排序
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))
# 将填充符号、普通音素和特殊符号合并成一个总的音素列表
symbols = [pad] + normal_symbols + pu_symbols
# 获取特殊音素的索引列表
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]

# 计算所有语言的音调总数
num_tones = num_zh_tones + num_ja_tones + num_en_tones

# 定义语言到编号的映射
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}
# 计算语言数量
num_languages = len(language_id_map.keys())

# 定义每种语言音调的起始位置
language_tone_start_map = {
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

# 如果作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 创建中文音素集合
    a = set(zh_symbols)
    # 创建英文音素集合
    b = set(en_symbols)
    # 打印中文音素集合和英文音素集合的交集
    print(sorted(a & b))

```