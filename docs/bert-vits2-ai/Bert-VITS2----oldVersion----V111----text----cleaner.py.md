# `Bert-VITS2\oldVersion\V111\text\cleaner.py`

```

# 从当前目录中导入 chinese, japanese, cleaned_text_to_sequence 模块
from . import chinese, japanese, cleaned_text_to_sequence
# 从 fix 模块中导入 japanese 模块
from .fix import japanese as japanese_fix

# 创建语言模块映射字典
language_module_map = {"ZH": chinese, "JP": japanese}
# 创建修复后的语言模块映射字典
language_module_map_fix = {"ZH": chinese, "JP": japanese_fix}

# 清洗文本的函数，根据语言选择对应的语言模块
def clean_text(text, language):
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取音素、音调和单词到音素的映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph

# 修复后的清洗文本函数，使用 dev 分支修复
def clean_text_fix(text, language):
    language_module = language_module_map_fix[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph

# 使用 BERT 模型进行文本清洗
def clean_text_bert(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 获取 BERT 特征
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert

# 将文本转换为序列
def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)

# 主函数入口
if __name__ == "__main__":
    pass

```