# `D:\src\scipysrc\scikit-learn\sklearn\inspection\tests\__init__.py`

```
# 定义一个名为 analyze_text 的函数，接受一个参数 text
def analyze_text(text):
    # 将文本全部转换为小写，以便统一处理
    text = text.lower()
    # 统计文本中每个字符出现的次数，并存储在字典 char_counts 中
    char_counts = {char: text.count(char) for char in set(text)}
    # 返回统计结果的字典 char_counts
    return char_counts
```