# `jieba\jieba\analyse\analyzer.py`

```
# 设置文件编码为 UTF-8
from __future__ import unicode_literals
# 导入所需的分词器、过滤器和词干处理器
from whoosh.analysis import RegexAnalyzer, LowercaseFilter, StopFilter, StemFilter
from whoosh.analysis import Tokenizer, Token
from whoosh.lang.porter import stem

# 导入结巴分词和正则表达式模块
import jieba
import re

# 定义停用词集合
STOP_WORDS = frozenset(('a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can',
                        'for', 'from', 'have', 'if', 'in', 'is', 'it', 'may',
                        'not', 'of', 'on', 'or', 'tbd', 'that', 'the', 'this',
                        'to', 'us', 'we', 'when', 'will', 'with', 'yet',
                        'you', 'your', '的', '了', '和'))

# 定义用于匹配中文字符的正则表达式
accepted_chars = re.compile(r"[\u4E00-\u9FD5]+")

# 定义一个自定义的中文分词器
class ChineseTokenizer(Tokenizer):

    def __call__(self, text, **kargs):
        # 使用结巴分词对文本进行分词
        words = jieba.tokenize(text, mode="search")
        token = Token()
        # 遍历分词结果
        for (w, start_pos, stop_pos) in words:
            # 如果分词结果不是中文字符或长度小于等于1，则跳过
            if not accepted_chars.match(w) and len(w) <= 1:
                continue
            # 设置 token 的原始文本、文本、位置等属性
            token.original = token.text = w
            token.pos = start_pos
            token.startchar = start_pos
            token.endchar = stop_pos
            yield token

# 定义一个中文分析器，包括分词器、小写过滤器、停用词过滤器和词干过滤器
def ChineseAnalyzer(stoplist=STOP_WORDS, minsize=1, stemfn=stem, cachesize=50000):
    return (ChineseTokenizer() | LowercaseFilter() |
            StopFilter(stoplist=stoplist, minsize=minsize) |
            StemFilter(stemfn=stemfn, ignore=None, cachesize=cachesize))
```