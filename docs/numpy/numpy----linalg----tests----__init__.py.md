# `.\numpy\numpy\linalg\tests\__init__.py`

```
# 导入所需模块：Counter 是一个用于计数的工具，defaultdict 是一种带有默认值的字典
from collections import Counter, defaultdict

# 定义一个函数 count_words，接受一个字符串作为参数
def count_words(s):
    # 使用 Counter 类来计算字符串中每个单词出现的次数，并返回一个 Counter 对象
    return Counter(s.split())

# 定义一个函数 find_anagrams，接受一个字符串列表作为参数
def find_anagrams(words):
    # 使用 defaultdict 来创建一个字典，键为按单词中字母排序后的元组，值为对应的单词列表
    anagrams = defaultdict(list)
    
    # 遍历传入的字符串列表
    for word in words:
        # 将单词按字母排序，转换为元组作为字典的键，并将当前单词添加到对应键的列表中
        sorted_word = tuple(sorted(word))
        anagrams[sorted_word].append(word)
    
    # 返回包含所有变位词的字典
    return anagrams
```