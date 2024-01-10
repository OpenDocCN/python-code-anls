# `Bert-VITS2\oldVersion\V210\text\cleaner.py`

```
# 导入中文、日文、英文和文本清洗到序列的模块
from . import chinese, japanese, english, cleaned_text_to_sequence

# 定义语言模块映射，将语言代码映射到对应的语言模块
language_module_map = {"ZH": chinese, "JP": japanese, "EN": english}

# 清洗文本，返回规范化后的文本、音素、音调和单词到音素的映射
def clean_text(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取音素、音调和单词到音素的映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph

# 清洗文本并获取BERT特征
def clean_text_bert(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取音素、音调和单词到音素的映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 获取BERT特征
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert

# 将文本转换为序列
def text_to_sequence(text, language):
    # 清洗文本，获取规范化后的文本、音素、音调和单词到音素的映射
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 将音素和音调转换为序列
    return cleaned_text_to_sequence(phones, tones, language)

# 主函数入口
if __name__ == "__main__":
    pass
```