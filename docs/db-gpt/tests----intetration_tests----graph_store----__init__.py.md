# `.\DB-GPT-src\tests\intetration_tests\graph_store\__init__.py`

```py
# 定义一个名为 count_words 的函数，接收一个字符串参数 text
def count_words(text):
    # 使用 split() 方法将字符串 text 拆分为单词列表，基于空白字符进行分割
    words = text.split()
    # 初始化一个空字典 word_count 用于存储单词及其出现次数
    word_count = {}
    
    # 遍历单词列表 words 中的每个单词
    for word in words:
        # 如果单词 word 已经在 word_count 字典中，则将其计数加一；否则设置计数为1
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    
    # 返回统计好的 word_count 字典，包含每个单词及其出现的次数
    return word_count
```