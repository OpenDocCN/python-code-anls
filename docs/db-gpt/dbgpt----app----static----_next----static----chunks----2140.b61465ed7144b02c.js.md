# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\2140.b61465ed7144b02c.js`

```py
# 定义一个名为 count_words 的函数，用于统计文本中每个单词出现的次数
def count_words(text):
    # 使用空格分割文本，生成单词列表
    words = text.split()
    # 初始化一个空字典，用于存储单词及其出现次数
    word_count = {}
    
    # 遍历单词列表
    for word in words:
        # 如果单词已经在字典中，则将其计数加一；否则将其加入字典并初始化计数为一
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    
    # 返回统计结果的字典
    return word_count
```