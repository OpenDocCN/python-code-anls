# `jieba\test\test_whoosh_file_read.py`

```py
# 设置文件编码为 UTF-8
# 导入 unicode_literals 模块
from __future__ import unicode_literals
# 导入 sys 和 os 模块
import sys
import os
# 将上级目录添加到 sys.path 中
sys.path.append("../")
# 导入 whoosh 库中的 create_in、open_dir、Schema、TEXT、ID、QueryParser 模块
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
# 从 jieba.analyse 模块中导入 ChineseAnalyzer 类
from jieba.analyse import ChineseAnalyzer 

# 创建一个中文分析器对象
analyzer = ChineseAnalyzer()

# 定义索引结构，包括 title、path 和 content 三个字段
schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True, analyzer=analyzer))
# 如果 "tmp" 文件夹不存在，则创建该文件夹
if not os.path.exists("tmp"):
    os.mkdir("tmp")
# 打开名为 "tmp" 的索引
ix = open_dir("tmp")

# 创建一个搜索器对象
searcher = ix.searcher()
# 创建一个查询解析器对象，指定查询字段为 "content"
parser = QueryParser("content", schema=ix.schema)

# 遍历关键字列表，依次搜索并打印结果
for keyword in ("水果小姐", "你", "first", "中文", "交换机", "交换", "少林", "乔峰"):
    # 打印当前关键字的搜索结果
    print("result of ", keyword)
    # 解析当前关键字为查询对象
    q = parser.parse(keyword)
    # 在索引中搜索匹配当前关键字的结果
    results = searcher.search(q)
    # 遍历搜索结果，打印内容高亮部分
    for hit in results:  
        print(hit.highlights("content"))
    # 打印分隔线
    print("="*10)
```