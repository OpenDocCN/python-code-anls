# `Bert-VITS2\onnx_modules\V220_novq_dev\text\symbols.py`

```py
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
# 获取特殊符号在音素列表中的索引
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]

# 计算所有语言的音调数量总和
num_tones = num_zh_tones + num_ja_tones + num_en_tones

# 定义语言到编号的映射
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}
# 获取语言-编号映射字典的键的数量
num_languages = len(language_id_map.keys())

# 创建语言-起始音调映射字典
language_tone_start_map = {
    "ZH": 0,  # 中文起始音调为0
    "JP": num_zh_tones,  # 日文起始音调为中文音调数量
    "EN": num_zh_tones + num_ja_tones,  # 英文起始音调为中文音调数量加上日文音调数量
}

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建中文符号集合
    a = set(zh_symbols)
    # 创建英文符号集合
    b = set(en_symbols)
    # 打印中文符号集合和英文符号集合的交集，并按顺序打印
    print(sorted(a & b))
```