# `d:/src/tocomm/Bert-VITS2\oldVersion\V111\text\cleaner.py`

```
from . import chinese, japanese, cleaned_text_to_sequence
from .fix import japanese as japanese_fix
```
导入所需的模块和函数。

```
language_module_map = {"ZH": chinese, "JP": japanese}
language_module_map_fix = {"ZH": chinese, "JP": japanese_fix}
```
创建一个字典，将语言代码映射到相应的语言模块。

```
def clean_text(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph
```
定义一个函数`clean_text`，接受一个文本和语言代码作为参数。根据语言代码选择相应的语言模块，对文本进行规范化处理，然后使用该语言模块的`g2p`函数将文本转换为音素序列，并返回规范化后的文本、音素序列、音调和单词到音素的映射。

```
def clean_text_fix(text, language):
    """使用dev分支修复"""
    language_module = language_module_map_fix[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
```
定义一个函数`clean_text_fix`，接受一个文本和语言代码作为参数。根据语言代码选择相应的修复后的语言模块，对文本进行规范化处理，然后使用该语言模块的`g2p`函数将文本转换为音素序列。
def clean_text_bert(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 使用语言模块进行音素转换
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 使用语言模块获取BERT特征
    bert = language_module.get_bert_feature(norm_text, word2ph)
    # 返回音素、音调和BERT特征
    return phones, tones, bert


def text_to_sequence(text, language):
    # 对文本进行清洗和BERT处理
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 将清洗后的文本转换为序列
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
```

注释已经添加在代码中，解释了每个语句的作用。
```