# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\9684.e87674104e78b256.js`

```py
# 定义一个名为 find_anagrams 的函数，接受一个字符串参数 word
def find_anagrams(word):
    # 使用 sorted() 函数对输入的单词进行排序，生成排序后的元组
    sorted_word = sorted(word)
    # 返回排序后的元组，以及单词本身作为元组中的第一个元素
    return [w for w in wordlist if sorted(w) == sorted_word]
```