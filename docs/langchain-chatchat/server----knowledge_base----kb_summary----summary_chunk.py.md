# `.\Langchain-Chatchat\server\knowledge_base\kb_summary\summary_chunk.py`

```
# 导入必要的模块
from typing import List, Optional
from langchain.schema.language_model import BaseLanguageModel
from server.knowledge_base.model.kb_document_model import DocumentWithVSId
from configs import (logger)
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.output_parsers.regex import RegexParser
from langchain.chains.combine_documents.map_reduce import ReduceDocumentsChain, MapReduceDocumentsChain
import sys
import asyncio

# 定义一个类 SummaryAdapter
class SummaryAdapter:
    # 定义类变量
    _OVERLAP_SIZE: int
    token_max: int
    _separator: str = "\n\n"
    chain: MapReduceDocumentsChain

    # 初始化方法，接收重叠大小、最大令牌数和链对象作为参数
    def __init__(self, overlap_size: int, token_max: int, chain: MapReduceDocumentsChain):
        self._OVERLAP_SIZE = overlap_size
        self.chain = chain
        self.token_max = token_max

    # 类方法，用于总结文档
    @classmethod
    def summarize(self, file_description: str, docs: List[DocumentWithVSId] = []) -> List[Document]:
        # 检查 Python 版本，选择合适的事件循环
        if sys.version_info < (3, 10):
            loop = asyncio.get_event_loop()
        else:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()

            asyncio.set_event_loop(loop)
        
        # 同步调用协程代码，返回总结结果
        return loop.run_until_complete(self.asummarize(file_description=file_description, docs=docs))
    def _drop_overlap(self, docs: List[DocumentWithVSId]) -> List[str]:
        """
         # 将文档中page_content句子叠加的部分去掉
        :param docs: 包含文档内容的列表
        :param separator: 分隔符
        :return: 返回去除叠加部分后的文档内容列表
        """
        merge_docs = []  # 初始化合并文档列表

        pre_doc = None  # 初始化前一个文档内容为空
        for doc in docs:  # 遍历文档列表
            # 第一个文档直接添加
            if len(merge_docs) == 0:  # 如果合并文档列表为空
                pre_doc = doc.page_content  # 将当前文档内容赋值给前一个文档内容
                merge_docs.append(doc.page_content)  # 将当前文档内容添加到合并文档列表
                continue

            # 列表中上一个结尾与下一个开头重叠的部分，删除下一个开头重叠的部分
            # 迭代递减pre_doc的长度，每次迭代删除前面的字符，
            # 查询重叠部分，直到pre_doc的长度小于 self._OVERLAP_SIZE // 2 - 2len(separator)
            for i in range(len(pre_doc), self._OVERLAP_SIZE // 2 - 2 * len(self._separator), -1):
                # 每次迭代删除前面的字符
                pre_doc = pre_doc[1:]  # 删除前一个文档内容的第一个字符
                if doc.page_content[:len(pre_doc)] == pre_doc:  # 如果当前文档内容的开头与前一个文档内容相同
                    # 删除下一个开头重叠的部分
                    merge_docs.append(doc.page_content[len(pre_doc):])  # 将当前文档内容中重叠部分后的内容添加到合并文档列表
                    break

            pre_doc = doc.page_content  # 更新前一个文档内容为当前文档内容

        return merge_docs  # 返回去除叠加部分后的文档内容列表

    def _join_docs(self, docs: List[str]) -> Optional[str]:
        text = self._separator.join(docs)  # 使用分隔符将文档内容列表连接成一个字符串
        text = text.strip()  # 去除字符串两端的空格
        if text == "":  # 如果字符串为空
            return None  # 返回空值
        else:
            return text  # 返回连接后的文档内容字符串
if __name__ == '__main__':
    # 如果当前脚本被直接执行，则执行以下代码块

    docs = [
        # 定义一个包含多个文档句子的列表
        '梦者有特别的作用，也就是说梦是在预卜未来。因此，梦内容的',
        '梦内容的多彩多姿以及对梦者本身所遗留的特殊印象，使他们很难想象',
        '使他们很难想象出一套系统划一的观念，而需要以其个别的价值与可靠性作各',
        '值与可靠性作各种不同的分化与聚合。因此，古代哲学家们对梦的评价也就完全'
    ]
    _OVERLAP_SIZE = 1
    separator: str = "\n\n"
    merge_docs = []
    # 初始化叠加文档列表

    pre_doc = None
    # 初始化前一个文档为空
    for doc in docs:
        # 遍历文档列表
        if len(merge_docs) == 0:
            # 如果叠加文档列表为空，直接添加第一个文档
            pre_doc = doc
            merge_docs.append(doc)
            continue

        for i in range(len(pre_doc), _OVERLAP_SIZE - 2 * len(separator), -1):
            # 递减pre_doc的长度，每次迭代删除前面的字符，直到长度小于_OVERLAP_SIZE-2*len(separator)
            pre_doc = pre_doc[1:]
            if doc[:len(pre_doc)] == pre_doc:
                # 如果当前文档与前一个文档有重叠部分，删除重叠部分后添加到叠加文档列表
                page_content = doc[len(pre_doc):]
                merge_docs.append(page_content)

                pre_doc = doc
                break

    text = separator.join(merge_docs)
    # 将叠加文档列表中的句子用分隔符连接成一个文档
    text = text.strip()
    # 去除文档首尾的空白字符

    print(text)
    # 打印合并后的文档
```