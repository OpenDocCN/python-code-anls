# `Bert-VITS2\onnx_modules\V230\text\symbols.py`

```
# 定义标点符号列表
punctuation = ["!", "?", "…", ",", ".", "'", "-"]
# 定义标点符号和特殊符号列表
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

# 合并所有音素列表并排序
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))
# 定义所有音素列表，包括填充符号和特殊符号
symbols = [pad] + normal_symbols + pu_symbols
# 定义特殊符号的索引列表
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]

# 计算所有音调的总数量
num_tones = num_zh_tones + num_ja_tones + num_en_tones

# 定义语言到编号的映射
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}
# 获取语言-编号映射字典的键的数量，即语言的数量
num_languages = len(language_id_map.keys())

# 创建语言-起始音调映射字典，用于记录每种语言音调的起始位置
language_tone_start_map = {
    "ZH": 0,  # 中文音调起始位置为0
    "JP": num_zh_tones,  # 日文音调起始位置为中文音调数量
    "EN": num_zh_tones + num_ja_tones,  # 英文音调起始位置为中文音调数量加上日文音调数量
}

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建集合a，包含中文符号
    a = set(zh_symbols)
    # 创建集合b，包含英文符号
    b = set(en_symbols)
    # 打印集合a和集合b的交集，并按照字母顺序排序
    print(sorted(a & b))
```