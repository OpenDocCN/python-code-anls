# `D:\src\scipysrc\pandas\pandas\tests\arrays\__init__.py`

```
# 定义一个名为 count_words 的函数，接受一个字符串参数 s
def count_words(s):
    # 创建一个名为 counts 的空字典，用于存储单词计数
    counts = {}
    # 使用空格分割字符串 s，并遍历生成的单词列表
    for word in s.split():
        # 如果单词 word 已经在 counts 字典中，则将其计数加一；否则将其计数设置为 1
        counts[word] = counts.get(word, 0) + 1
    # 返回统计好的 counts 字典，其中包含每个单词及其出现的次数
    return counts
```