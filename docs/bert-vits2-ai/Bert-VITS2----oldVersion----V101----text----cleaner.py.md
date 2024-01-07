# `Bert-VITS2\oldVersion\V101\text\cleaner.py`

```

# 从当前目录中导入 chinese 模块和 cleaned_text_to_sequence 函数
from . import chinese, cleaned_text_to_sequence

# 定义一个语言模块映射字典，将语言代码映射到对应的语言模块
language_module_map = {"ZH": chinese}

# 清洗文本的函数，接受文本和语言作为参数
def clean_text(text, language):
    # 根据语言获取对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和单词到音素的映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph

# 使用 BERT 模型清洗文本的函数，接受文本和语言作为参数
def clean_text_bert(text, language):
    # 根据语言获取对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和单词到音素的映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 获取文本的 BERT 特征
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert

# 将文本转换为序列的函数，接受文本和语言作为参数
def text_to_sequence(text, language):
    # 清洗文本，获取规范化文本、音素、音调和单词到音素的映射
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 使用 cleaned_text_to_sequence 函数将音素和音调转换为序列
    return cleaned_text_to_sequence(phones, tones, language)

# 如果作为主程序运行，则不执行任何操作
if __name__ == "__main__":
    pass

```