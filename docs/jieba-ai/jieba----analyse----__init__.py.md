# `jieba\jieba\analyse\__init__.py`

```py
# 导入绝对路径的未来特性
from __future__ import absolute_import
# 从当前目录下的 tfidf 模块中导入 TFIDF 类
from .tfidf import TFIDF
# 从当前目录下的 textrank 模块中导入 TextRank 类
from .textrank import TextRank
# 尝试从当前目录下的 analyzer 模块中导入 ChineseAnalyzer 类，如果导入失败则忽略
try:
    from .analyzer import ChineseAnalyzer
except ImportError:
    pass

# 创建默认的 TFIDF 对象
default_tfidf = TFIDF()
# 创建默认的 TextRank 对象
default_textrank = TextRank()

# 将 default_tfidf 对象的 extract_tags 方法赋值给 extract_tags 变量
extract_tags = tfidf = default_tfidf.extract_tags
# 将 default_tfidf 对象的 set_idf_path 方法赋值给 set_idf_path 变量
set_idf_path = default_tfidf.set_idf_path
# 将 default_textrank 对象的 extract_tags 方法赋值给 textrank 变量
textrank = default_textrank.extract_tags

# 定义设置停用词路径的函数，将停用词路径传递给 default_tfidf 和 default_textrank 对象
def set_stop_words(stop_words_path):
    default_tfidf.set_stop_words(stop_words_path)
    default_textrank.set_stop_words(stop_words_path)
```