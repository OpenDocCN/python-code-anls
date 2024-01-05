# `d:/src/tocomm/Bert-VITS2\onnx_modules\V200\text\cleaner.py`

```
# 导入所需的模块
from . import chinese, japanese, english, cleaned_text_to_sequence

# 创建语言模块映射字典
language_module_map = {"ZH": chinese, "JP": japanese, "EN": english}

# 定义函数，用于清洗文本
def clean_text(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和音素到拼音的映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 返回规范化后的文本、音素、音调和音素到拼音的映射
    return norm_text, phones, tones, word2ph

# 定义函数，用于使用BERT模型清洗文本
def clean_text_bert(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和音素到拼音的映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 使用BERT模型获取文本的特征表示
    bert = language_module.get_bert_feature(norm_text, word2ph)
    # 返回音素、音调和BERT特征
    return phones, tones, bert
# 将文本转换为序列
def text_to_sequence(text, language):
    # 对文本进行清洗和处理，得到规范化的文本、音素、音调和单词到音素的映射
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 调用函数将清洗后的文本转换为序列
    return cleaned_text_to_sequence(phones, tones, language)


# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    pass
```