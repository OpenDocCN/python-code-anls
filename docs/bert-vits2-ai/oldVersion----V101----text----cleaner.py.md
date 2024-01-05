# `d:/src/tocomm/Bert-VITS2\oldVersion\V101\text\cleaner.py`

```
# 导入模块 chinese 和 cleaned_text_to_sequence
from . import chinese, cleaned_text_to_sequence

# 定义一个字典，将语言和对应的模块进行映射
language_module_map = {"ZH": chinese}

# 定义函数 clean_text，接受文本和语言作为参数
def clean_text(text, language):
    # 根据语言获取对应的模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 使用模块中的函数将文本转换为音素序列
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 返回规范化后的文本、音素序列、音调和单词到音素的映射
    return norm_text, phones, tones, word2ph

# 定义函数 clean_text_bert，接受文本和语言作为参数
def clean_text_bert(text, language):
    # 根据语言获取对应的模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 使用模块中的函数将文本转换为音素序列
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 使用模块中的函数获取文本的 BERT 特征
    bert = language_module.get_bert_feature(norm_text, word2ph)
    # 返回音素序列、音调和 BERT 特征
    return phones, tones, bert
```

这段代码定义了两个函数 `clean_text` 和 `clean_text_bert`，用于对文本进行清洗和处理。其中使用了一个字典 `language_module_map`，将语言和对应的模块进行映射。在函数中，根据语言获取对应的模块，然后对文本进行规范化处理，再将文本转换为音素序列。最后，`clean_text_bert` 函数还使用了模块中的函数获取文本的 BERT 特征。函数返回了不同的处理结果，包括规范化后的文本、音素序列、音调和单词到音素的映射，以及音素序列、音调和 BERT 特征。
# 将文本转换为序列的函数
def text_to_sequence(text, language):
    # 清理文本，得到规范化的文本、音素、音调和单词到音素的映射字典
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 将清理后的文本转换为序列
    return cleaned_text_to_sequence(phones, tones, language)


# 主程序入口
if __name__ == "__main__":
    # 程序入口点，暂时不执行任何操作
    pass
```

注释解释了每个语句的作用，包括函数的功能和主程序的入口点。
```