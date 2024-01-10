# `Bert-VITS2\onnx_modules\V200\text\symbols.py`

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
# 将填充符号、普通音素和特殊符号合并成一个总的音素列表
symbols = [pad] + normal_symbols + pu_symbols
# 找出特殊符号在总音素列表中的索引
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]

# 计算所有语言的音调总数
num_tones = num_zh_tones + num_ja_tones + num_en_tones

# 定义语言到编号的映射
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}
# 获取语言-编号映射字典的键的数量，即语言的数量
num_languages = len(language_id_map.keys())

# 创建语言-起始音调映射字典，用于记录每种语言音调的起始位置
language_tone_start_map = {
    "ZH": 0,  # 中文音调起始位置为0
    "JP": num_zh_tones,  # 日语音调起始位置为中文音调数量
    "EN": num_zh_tones + num_ja_tones,  # 英语音调起始位置为中文音调数量加上日语音调数量
}

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建中文符号集合
    a = set(zh_symbols)
    # 创建英文符号集合
    b = set(en_symbols)
    # 打印中文符号集合和英文符号集合的交集，并按照字母顺序排序后输出
    print(sorted(a & b))
```