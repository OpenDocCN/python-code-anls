# `D:\src\scipysrc\scipy\scipy\integrate\tests\__init__.py`

```
# 导入所需的模块：Counter 是一个用来计数的数据结构
from collections import Counter

# 定义一个函数 count_words，接收一个字符串作为参数
def count_words(sentence):
    # 使用 split() 方法将句子分割成单词列表，并将所有单词转换为小写
    words = sentence.split()
    # 使用 Counter 类创建一个单词计数器，统计每个单词出现的次数
    word_count = Counter(words)
    # 返回统计结果，即单词计数器
    return word_count
```