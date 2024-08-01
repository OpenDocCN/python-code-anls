# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\5288.4c7463607fe34303.js`

```py
# 定义一个名为 count_vowels 的函数，接受一个字符串参数 s
def count_vowels(s):
    # 定义一个元音字母集合
    vowels = {'a', 'e', 'i', 'o', 'u'}
    # 使用生成器表达式计算字符串 s 中元音字母的数量
    count = sum(1 for char in s if char.lower() in vowels)
    # 返回计算结果
    return count
```