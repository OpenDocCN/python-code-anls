# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\6424.b5a83902fa364ea8.js`

```py
# 定义一个名为count_words的函数，接受一个参数text
def count_words(text):
    # 使用split()方法将文本分割成单词列表，存储在变量words中
    words = text.split()
    # 创建一个空字典word_count，用于存储每个单词的出现次数
    word_count = {}
    # 遍历单词列表words中的每个单词word
    for word in words:
        # 如果单词word已经在字典word_count中，则将其计数加1；否则将其计数设为1
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    # 返回统计好的单词计数字典word_count
    return word_count
```