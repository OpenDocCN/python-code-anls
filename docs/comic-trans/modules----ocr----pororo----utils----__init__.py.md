# `.\comic-translate\modules\ocr\pororo\utils\__init__.py`

```py
# 定义一个名为 `filter_long_words` 的函数，接收两个参数：一个单词列表 `word_list` 和一个整数 `n`
def filter_long_words(word_list, n):
    # 使用列表推导式，遍历 `word_list` 中的每个单词，筛选出长度大于 `n` 的单词
    filtered_words = [word for word in word_list if len(word) > n]
    # 返回筛选后的单词列表
    return filtered_words
```