# `jieba\test\test_whoosh_file.py`

```py
# 设置文件编码为 UTF-8
# 导入 unicode_literals 模块
from __future__ import unicode_literals
# 导入 sys、os 模块
import sys
import os
# 将上级目录添加到 sys.path 中
sys.path.append("../")
# 导入 whoosh 相关模块
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
# 导入中文分词模块
from jieba.analyse import ChineseAnalyzer

# 创建中文分词器
analyzer = ChineseAnalyzer()

# 定义索引结构
schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True, analyzer=analyzer))
# 如果 "tmp" 目录不存在，则创建
if not os.path.exists("tmp"):
    os.mkdir("tmp")
# 在 "tmp" 目录下创建索引
ix = create_in("tmp", schema)
# 获取索引写入对象
writer = ix.writer()

# 从命令行参数中获取文件名
file_name = sys.argv[1]

# 打开文件并逐行读取
with open(file_name,"rb") as inf:
    i=0
    for line in inf:
        i+=1
        # 将每行内容添加到索引中
        writer.add_document(
            title="line"+str(i),
            path="/a",
            content=line.decode('gbk','ignore')
        )
# 提交索引写入
writer.commit()

# 获取索引搜索器
searcher = ix.searcher()
# 创建查询解析器
parser = QueryParser("content", schema=ix.schema)

# 遍历关键词列表进行搜索
for keyword in ("水果小姐","你","first","中文","交换机","交换"):
    print("result of " + keyword)
    # 解析关键词
    q = parser.parse(keyword)
    # 执行搜索
    results = searcher.search(q)
    # 遍历搜索结果并打印高亮内容
    for hit in results:
        print(hit.highlights("content"))
    print("="*10)
```