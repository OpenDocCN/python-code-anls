# `.\DB-GPT-src\dbgpt\app\scene\chat_knowledge\v1\chat.py`

```py
import json
import os
from functools import reduce
from typing import Dict, List

from dbgpt._private.config import Config
from dbgpt.app.knowledge.chunk_db import DocumentChunkDao, DocumentChunkEntity
from dbgpt.app.knowledge.document_db import (
    KnowledgeDocumentDao,
    KnowledgeDocumentEntity,
)
from dbgpt.app.knowledge.request.request import KnowledgeSpaceRequest
from dbgpt.app.knowledge.service import KnowledgeService
from dbgpt.app.scene import BaseChat, ChatScene
from dbgpt.configs.model_config import EMBEDDING_MODEL_CONFIG
from dbgpt.core import (
    ChatPromptTemplate,
    HumanPromptTemplate,
    MessagesPlaceholder,
    SystemPromptTemplate,
)
from dbgpt.rag.retriever.rerank import RerankEmbeddingsRanker
from dbgpt.rag.retriever.rewrite import QueryRewrite
from dbgpt.util.tracer import root_tracer, trace

CFG = Config()


class ChatKnowledge(BaseChat):
    chat_scene: str = ChatScene.ChatKnowledge.value()
    """KBQA Chat Module"""

    async def stream_call(self):
        last_output = None
        async for output in super().stream_call():
            last_output = output
            yield output

        if (
            CFG.KNOWLEDGE_CHAT_SHOW_RELATIONS
            and last_output
            and type(self.relations) == list
            and len(self.relations) > 0
            and hasattr(last_output, "text")
        ):
            # 将关系信息添加到输出文本中
            last_output.text = (
                last_output.text + "\n\nrelations:\n\n" + ",".join(self.relations)
            )
        # 构建参考文本并追加到最终输出中
        reference = f"\n\n{self.parse_source_view(self.chunks_with_score)}"
        last_output = last_output + reference
        yield last_output

    def stream_call_reinforce_fn(self, text):
        """返回包含参考文本的文本"""
        return text + f"\n\n{self.parse_source_view(self.chunks_with_score)}"

    @trace()
    # 异步方法，生成输入数值的字典
    async def generate_input_values(self) -> Dict:
        # 检查是否存在空间上下文，并且其中包含 "prompt" 键
        if self.space_context and self.space_context.get("prompt"):
            # 不使用模板定义
            # self.prompt_template.template_define = self.space_context["prompt"]["scene"]
            # self.prompt_template.template = self.space_context["prompt"]["template"]
            # 用提示模板替换模板
            self.prompt_template.prompt = ChatPromptTemplate(
                messages=[
                    SystemPromptTemplate.from_template(
                        self.space_context["prompt"]["template"]
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanPromptTemplate.from_template("{question}"),
                ]
            )
        
        # 导入异步任务运行函数
        from dbgpt.util.chat_util import run_async_tasks

        # 执行类似搜索任务
        tasks = [self.execute_similar_search(self.current_user_input)]
        # 并发运行任务，获取候选项及其分数
        candidates_with_scores = await run_async_tasks(tasks=tasks, concurrency_limit=1)
        # 将所有候选项及其分数列表合并为一个列表
        candidates_with_scores = reduce(lambda x, y: x + y, candidates_with_scores)
        
        # 初始化空列表用于存储带分数的数据块
        self.chunks_with_score = []
        # 如果候选项为空或长度为0，则打印无相关文档可检索的信息
        if not candidates_with_scores or len(candidates_with_scores) == 0:
            print("no relevant docs to retrieve")
            context = "no relevant docs to retrieve"
        else:
            # 清空带分数的数据块列表
            self.chunks_with_score = []
            # 遍历每个候选项
            for chunk in candidates_with_scores:
                # 获取文档块，并加入带分数的数据块列表
                chucks = self.chunk_dao.get_document_chunks(
                    query=DocumentChunkEntity(content=chunk.content),
                    document_ids=self.document_ids,
                )
                if len(chucks) > 0:
                    self.chunks_with_score.append((chucks[0], chunk.score))

            # 将候选项的内容连接成一个字符串，作为上下文
            context = "\n".join([doc.content for doc in candidates_with_scores])
        
        # 从候选项中提取源文件的基本名称，并去重
        self.relations = list(
            set(
                [
                    os.path.basename(str(d.metadata.get("source", "")))
                    for d in candidates_with_scores
                ]
            )
        )
        
        # 构建输入数值的字典
        input_values = {
            "context": context,  # 上下文内容
            "question": self.current_user_input,  # 当前用户输入的问题
            "relations": self.relations,  # 相关文档的基本名称列表
        }
        return input_values  # 返回构建的输入数值的字典
    def parse_source_view(self, chunks_with_score: List):
        """
        format knowledge reference view message to web
        <references title="'References'" references="'[{name:aa.pdf,chunks:[{10:text},{11:text}]},{name:bb.pdf,chunks:[{12,text}]}]'"> </references>
        """
        import xml.etree.ElementTree as ET  # 导入用于操作 XML 的 ElementTree 模块

        # 创建 XML 元素 <references>
        references_ele = ET.Element("references")
        title = "References"  # 设置元素的 title 属性为 "References"
        references_ele.set("title", title)
        references_dict = {}

        # 遍历 chunks_with_score 列表中的每个元素
        for chunk, score in chunks_with_score:
            doc_name = chunk.doc_name
            # 如果文档名不在 references_dict 中，则创建新条目
            if doc_name not in references_dict:
                references_dict[doc_name] = {
                    "name": doc_name,
                    "chunks": [
                        {
                            "id": chunk.id,
                            "content": chunk.content,
                            "meta_info": chunk.meta_info,
                            "recall_score": score,
                        }
                    ],
                }
            else:
                # 如果文档名已存在于 references_dict 中，则向其 chunks 列表中添加新条目
                references_dict[doc_name]["chunks"].append(
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "meta_info": chunk.meta_info,
                        "recall_score": score,
                    }
                )
        
        # 将 references_dict 转换为列表，并设置为 references 元素的属性
        references_list = list(references_dict.values())
        references_ele.set(
            "references", json.dumps(references_list, ensure_ascii=False)
        )
        
        # 将 XML 元素转换为 UTF-8 编码的 HTML 字符串
        html = ET.tostring(references_ele, encoding="utf-8")
        reference = html.decode("utf-8")
        
        # 返回 HTML 字符串，去除其中的换行符
        return reference.replace("\\n", "")

    @property
    def chat_type(self) -> str:
        return ChatScene.ChatKnowledge.value()  # 返回当前聊天场景的知识类型值

    def get_space_context(self, space_name):
        service = KnowledgeService()  # 创建知识服务对象
        return service.get_space_context(space_name)  # 获取指定空间名称的上下文信息

    def get_knowledge_search_top_size(self, space_name) -> int:
        service = KnowledgeService()  # 创建知识服务对象
        request = KnowledgeSpaceRequest(name=space_name)
        spaces = service.get_knowledge_space(request)  # 获取指定知识空间的信息列表
        if len(spaces) == 1:
            from dbgpt.storage import vector_store

            # 如果空间的向量类型在知识图谱存储中，则返回特定的搜索大小常量
            if spaces[0].vector_type in vector_store.__knowledge_graph__:
                return CFG.KNOWLEDGE_GRAPH_SEARCH_TOP_SIZE

        # 否则返回默认的搜索大小常量
        return CFG.KNOWLEDGE_SEARCH_TOP_SIZE

    async def execute_similar_search(self, query):
        """execute similarity search"""
        with root_tracer.start_span(
            "execute_similar_search", metadata={"query": query}
        ):
            # 使用异步方法执行相似性搜索，并返回结果
            return await self.embedding_retriever.aretrieve_with_scores(
                query, self.recall_score
            )
```