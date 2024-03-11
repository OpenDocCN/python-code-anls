# `.\Langchain-Chatchat\tests\custom_splitter\test_different_splitter.py`

```
# 导入必要的模块
import os
from transformers import AutoTokenizer
import sys

# 将上级目录添加到系统路径中
sys.path.append("../..")
# 从配置文件中导入CHUNK_SIZE和OVERLAP_SIZE
from configs import (
    CHUNK_SIZE,
    OVERLAP_SIZE
)
# 从server.knowledge_base.utils模块中导入make_text_splitter函数
from server.knowledge_base.utils import make_text_splitter

# 定义函数text，接受一个参数splitter_name
def text(splitter_name):
    # 从langchain模块中导入document_loaders
    from langchain import document_loaders

    # 指定文件路径
    filepath = "../../knowledge_base/samples/content/test.txt"
    # 使用UnstructuredFileLoader加载文件
    loader = document_loaders.UnstructuredFileLoader(filepath, autodetect_encoding=True)
    # 加载文档
    docs = loader.load()
    # 创建文本分割器
    text_splitter = make_text_splitter(splitter_name, CHUNK_SIZE, OVERLAP_SIZE)
    # 根据不同的分割器类型进行处理
    if splitter_name == "MarkdownHeaderTextSplitter":
        # 使用MarkdownHeaderTextSplitter分割文本
        docs = text_splitter.split_text(docs[0].page_content)
        # 为每个文档添加元数据
        for doc in docs:
            if doc.metadata:
                doc.metadata["source"] = os.path.basename(filepath)
    else:
        # 使用其他分割器分割文档
        docs = text_splitter.split_documents(docs)
    # 打印每个文档
    for doc in docs:
        print(doc)
    # 返回处理后的文档列表
    return docs

# 导入pytest模块
import pytest
# 从langchain.docstore.document模块中导入Document类

# 使用pytest.mark.parametrize装饰器，参数化测试用例
@pytest.mark.parametrize("splitter_name",
                         [
                             "ChineseRecursiveTextSplitter",
                             "SpacyTextSplitter",
                             "RecursiveCharacterTextSplitter",
                             "MarkdownHeaderTextSplitter"
                         ])
# 定义测试函数test_different_splitter，接受一个参数splitter_name
def test_different_splitter(splitter_name):
    try:
        # 调用text函数处理文档
        docs = text(splitter_name)
        # 断言docs是一个列表
        assert isinstance(docs, list)
        # 如果docs不为空，断言docs的第一个元素是Document类型
        if len(docs)>0:
            assert isinstance(docs[0], Document)
    except Exception as e:
        # 捕获异常并输出错误信息
        pytest.fail(f"test_different_splitter failed with {splitter_name}, error: {str(e)}")
```