# `Bert-VITS2\onnx_modules\V230\text\symbols.py`

```

# 定义标点符号列表
punctuation = ["!", "?", "…", ",", ".", "'", "-"]
# 定义包含标点符号和特殊符号的列表
pu_symbols = punctuation + ["SP", "UNK"]
# 定义填充符号
pad = "_"

# 定义中文音素列表
zh_symbols = [
    # ...（省略部分内容）
]
# 定义中文音调数量
num_zh_tones = 6

# 定义日文音素列表
ja_symbols = [
    # ...（省略部分内容）
]
# 定义日文音调数量
num_ja_tones = 2

# 定义英文音素列表
en_symbols = [
    # ...（省略部分内容）
]
# 定义英文音调数量
num_en_tones = 4

# 合并所有音素列表并去重排序
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))
# 定义所有音素列表，包括填充符号和特殊符号
symbols = [pad] + normal_symbols + pu_symbols
# 定义特殊音素的索引列表
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]

# 计算所有音调的总数量
num_tones = num_zh_tones + num_ja_tones + num_en_tones

# 定义语言到编号的映射
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}
# 计算语言数量
num_languages = len(language_id_map.keys())

# 定义每种语言音调的起始位置映射
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
    # 打印中文音素和英文音素的交集
    print(sorted(a & b))

```