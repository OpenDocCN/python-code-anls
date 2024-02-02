# `Bert-VITS2\oldVersion\V101\text\cleaner.py`

```py
# 从当前目录中导入 chinese 模块和 cleaned_text_to_sequence 函数
from . import chinese, cleaned_text_to_sequence

# 定义一个语言模块映射的字典，将语言代码映射到对应的语言模块
language_module_map = {"ZH": chinese}

# 定义一个函数，用于清洗文本数据
def clean_text(text, language):
    # 根据语言代码获取对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和单词到音素的映射关系
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 返回规范化后的文本、音素、音调和单词到音素的映射关系
    return norm_text, phones, tones, word2ph

# 定义一个函数，用于将文本转换为序列
def clean_text_bert(text, language):
    # 根据语言代码获取对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和单词到音素的映射关系
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 获取文本的 BERT 特征
    bert = language_module.get_bert_feature(norm_text, word2ph)
    # 返回音素、音调和 BERT 特征
    return phones, tones, bert

# 定义一个函数，用于将文本转换为序列
def text_to_sequence(text, language):
    # 调用 clean_text 函数，获取规范化后的文本、音素、音调和单词到音素的映射关系
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 调用 cleaned_text_to_sequence 函数，将音素和音调转换为序列
    return cleaned_text_to_sequence(phones, tones, language)

# 如果该脚本作为主程序运行，则执行 pass 语句
if __name__ == "__main__":
    pass
```