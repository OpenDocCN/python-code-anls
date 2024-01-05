# `d:/src/tocomm/Bert-VITS2\oldVersion\V110\text\cleaner.py`

```
from . import chinese, japanese, cleaned_text_to_sequence
```
导入模块`chinese`、`japanese`和`cleaned_text_to_sequence`。

```
language_module_map = {"ZH": chinese, "JP": japanese}
```
创建一个字典`language_module_map`，将字符串"ZH"映射到模块`chinese`，将字符串"JP"映射到模块`japanese`。

```
def clean_text(text, language):
```
定义一个函数`clean_text`，接受两个参数`text`和`language`。

```
language_module = language_module_map[language]
```
根据`language`参数从`language_module_map`字典中获取对应的模块。

```
norm_text = language_module.text_normalize(text)
```
调用`language_module`模块的`text_normalize`函数，对`text`进行规范化处理，将结果赋值给`norm_text`。

```
phones, tones, word2ph = language_module.g2p(norm_text)
```
调用`language_module`模块的`g2p`函数，将`norm_text`转换为音素序列，将结果分别赋值给`phones`、`tones`和`word2ph`。

```
return norm_text, phones, tones, word2ph
```
返回`norm_text`、`phones`、`tones`和`word2ph`。

```
def clean_text_bert(text, language):
```
定义一个函数`clean_text_bert`，接受两个参数`text`和`language`。

```
language_module = language_module_map[language]
```
根据`language`参数从`language_module_map`字典中获取对应的模块。

```
norm_text = language_module.text_normalize(text)
```
调用`language_module`模块的`text_normalize`函数，对`text`进行规范化处理，将结果赋值给`norm_text`。

```
phones, tones, word2ph = language_module.g2p(norm_text)
```
调用`language_module`模块的`g2p`函数，将`norm_text`转换为音素序列，将结果分别赋值给`phones`、`tones`和`word2ph`。

```
bert = language_module.get_bert_feature(norm_text, word2ph)
```
调用`language_module`模块的`get_bert_feature`函数，获取`norm_text`和`word2ph`的BERT特征，将结果赋值给`bert`。

```
return phones, tones, bert
```
返回`phones`、`tones`和`bert`。
# 将文本转换为序列的函数
def text_to_sequence(text, language):
    # 清理文本，得到规范化的文本、音素、音调和单词到音素的映射字典
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 将清理后的文本转换为序列
    return cleaned_text_to_sequence(phones, tones, language)


# 主程序入口
if __name__ == "__main__":
    pass
```

注释解释：

- `def text_to_sequence(text, language):`：定义了一个名为`text_to_sequence`的函数，该函数接受两个参数`text`和`language`，用于将文本转换为序列。
- `norm_text, phones, tones, word2ph = clean_text(text, language)`：调用`clean_text`函数，将文本进行清理，得到规范化的文本、音素、音调和单词到音素的映射字典，并将结果分别赋值给`norm_text`、`phones`、`tones`和`word2ph`变量。
- `return cleaned_text_to_sequence(phones, tones, language)`：调用`cleaned_text_to_sequence`函数，将音素、音调和语言作为参数，将清理后的文本转换为序列，并将结果返回。
- `if __name__ == "__main__":`：判断当前模块是否为主程序入口，如果是，则执行以下代码。
- `pass`：占位符，表示不执行任何操作。
```