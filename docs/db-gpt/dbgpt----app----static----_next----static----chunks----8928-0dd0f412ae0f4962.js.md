# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\8928-0dd0f412ae0f4962.js`

```py
# 定义一个名为 find_anagrams 的函数，用于查找给定字符串列表中的变位词
def find_anagrams(word_list):
    # 使用字典推导式，将每个单词转换成其字母排序后的元组作为键，原始单词作为值，构建变位词字典
    anagram_dict = {tuple(sorted(word)): word for word in word_list}
    # 返回构建好的变位词字典
    return anagram_dict
```