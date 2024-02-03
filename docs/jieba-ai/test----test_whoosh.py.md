# `jieba\test\test_whoosh.py`

```
# 设置文件编码为 UTF-8
# 导入必要的模块
from __future__ import unicode_literals
import sys,os
# 将上级目录添加到系统路径中
sys.path.append("../")
# 导入创建索引和打开索引的函数
from whoosh.index import create_in,open_dir
# 导入字段类型和查询解析器
from whoosh.fields import *
from whoosh.qparser import QueryParser
# 导入中文分词器
from jieba.analyse.analyzer import ChineseAnalyzer

# 创建中文分词器对象
analyzer = ChineseAnalyzer()

# 定义索引结构
schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True, analyzer=analyzer))
# 如果临时文件夹不存在，则创建
if not os.path.exists("tmp"):
    os.mkdir("tmp")

# 创建或打开索引
ix = create_in("tmp", schema) # for create new index
#ix = open_dir("tmp") # for read only
# 获取写入索引的对象
writer = ix.writer()

# 向索引中添加文档
writer.add_document(
    title="document1",
    path="/a",
    content="This is the first document we’ve added!"
)

writer.add_document(
    title="document2",
    path="/b",
    content="The second one 你 中文测试中文 is even more interesting! 吃水果"
)

writer.add_document(
    title="document3",
    path="/c",
    content="买水果然后来世博园。"
)

writer.add_document(
    title="document4",
    path="/c",
    content="工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
)

writer.add_document(
    title="document4",
    path="/c",
    content="咱俩交换一下吧。"
)

# 提交写入的文档
writer.commit()
# 获取搜索器对象
searcher = ix.searcher()
# 创建查询解析器
parser = QueryParser("content", schema=ix.schema)

# 对关键词进行搜索
for keyword in ("水果世博园","你","first","中文","交换机","交换"):
    print("result of ",keyword)
    q = parser.parse(keyword)
    results = searcher.search(q)
    for hit in results:
        print(hit.highlights("content"))
    print("="*10)

# 对文本进行中文分词
for t in analyzer("我的好朋友是李明;我爱北京天安门;IBM和Microsoft; I have a dream. this is intetesting and interested me a lot"):
    print(t.text)
```