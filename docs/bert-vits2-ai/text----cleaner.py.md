# `Bert-VITS2\text\cleaner.py`

```

# 从文本模块中导入中文、日文、英文和清理文本到序列的函数
from text import chinese, japanese, english, cleaned_text_to_sequence

# 创建语言模块映射字典，将语言代码映射到对应的语言模块
language_module_map = {"ZH": chinese, "JP": japanese, "EN": english}

# 清理文本函数，根据语言选择对应的语言模块进行文本清理和转换
def clean_text(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 将规范化后的文本转换成音素、音调和单词到音素的映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph

# 使用BERT模型进行文本清理函数，根据语言选择对应的语言模块进行文本清理和转换
def clean_text_bert(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 将规范化后的文本转换成音素、音调和单词到音素的映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 使用BERT模型获取文本特征
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert

# 将文本转换成序列的函数，根据语言选择对应的语言模块进行文本清理和转换
def text_to_sequence(text, language):
    # 对文本进行清理和转换
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 将清理后的文本转换成序列
    return cleaned_text_to_sequence(phones, tones, language)

# 主函数入口
if __name__ == "__main__":
    pass

```