# `Bert-VITS2\oldVersion\V111\text\cleaner.py`

```py
# 从当前目录中导入 chinese, japanese, cleaned_text_to_sequence 模块
from . import chinese, japanese, cleaned_text_to_sequence
# 从 fix 模块中导入 japanese 模块并重命名为 japanese_fix
from .fix import japanese as japanese_fix

# 创建语言模块映射字典，将语言代码映射到对应的语言模块
language_module_map = {"ZH": chinese, "JP": japanese}
# 创建修复后的语言模块映射字典，将语言代码映射到对应的修复后的语言模块
language_module_map_fix = {"ZH": chinese, "JP": japanese_fix}

# 定义函数 clean_text，接受文本和语言作为参数
def clean_text(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和单词到音素的映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 返回规范化后的文本、音素、音调和单词到音素的映射
    return norm_text, phones, tones, word2ph

# 定义函数 clean_text_fix，接受文本和语言作为参数
def clean_text_fix(text, language):
    """使用dev分支修复"""
    # 根据语言选择对应的修复后的语言模块
    language_module = language_module_map_fix[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和单词到音素的映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 返回规范化后的文本、音素、音调和单词到音素的映射
    return norm_text, phones, tones, word2ph

# 定义函数 clean_text_bert，接受文本和语言作为参数
def clean_text_bert(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和单词到音素的映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 获取文本的 BERT 特征
    bert = language_module.get_bert_feature(norm_text, word2ph)
    # 返回音素、音调和 BERT 特征
    return phones, tones, bert

# 定义函数 text_to_sequence，接受文本和语言作为参数
def text_to_sequence(text, language):
    # 对文本进行清洗处理，获取规范化后的文本、音素、音调和单词到音素的映射
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 将音素和音调转换为序列
    return cleaned_text_to_sequence(phones, tones, language)

# 如果当前脚本被直接执行，则不执行任何操作
if __name__ == "__main__":
    pass
```