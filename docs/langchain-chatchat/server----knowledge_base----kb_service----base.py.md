# `.\Langchain-Chatchat\server\knowledge_base\kb_service\base.py`

```
# 导入 operator 模块，用于操作符函数
import operator
# 导入 ABC 抽象基类和 abstractmethod 装饰器
from abc import ABC, abstractmethod

# 导入 os 模块
import os
# 导入 Path 类
from pathlib import Path
# 导入 numpy 模块，并重命名为 np
import numpy as np
# 导入 Embeddings 类
from langchain.embeddings.base import Embeddings
# 导入 Document 类
from langchain.docstore.document import Document

# 导入知识库相关的数据库操作函数
from server.db.repository.knowledge_base_repository import (
    add_kb_to_db, delete_kb_from_db, list_kbs_from_db, kb_exists,
    load_kb_from_db, get_kb_detail,
)
# 导入知识文件相关的数据库操作函数
from server.db.repository.knowledge_file_repository import (
    add_file_to_db, delete_file_from_db, delete_files_from_db, file_exists_in_db,
    count_files_from_db, list_files_from_db, get_file_detail, delete_file_from_db,
    list_docs_from_db,
)

# 导入配置文件中的相关配置
from configs import (kbs_config, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD,
                     EMBEDDING_MODEL, KB_INFO)
# 导入知识库工具函数
from server.knowledge_base.utils import (
    get_kb_path, get_doc_path, KnowledgeFile,
    list_kbs_from_folder, list_files_from_folder,
)

# 导入 List、Union、Dict、Optional、Tuple 类型
from typing import List, Union, Dict, Optional, Tuple

# 导入 embeddings_api 模块中的函数
from server.embeddings_api import embed_texts, aembed_texts, embed_documents
# 导入 kb_document_model 模块中的 DocumentWithVSId 类
from server.knowledge_base.model.kb_document_model import DocumentWithVSId

# 定义 normalize 函数，用于对嵌入向量进行归一化处理
def normalize(embeddings: List[List[float]) -> np.ndarray:
    '''
    sklearn.preprocessing.normalize 的替代（使用 L2），避免安装 scipy, scikit-learn
    '''
    # 计算嵌入向量的 L2 范数
    norm = np.linalg.norm(embeddings, axis=1)
    # 将范数重塑为二维数组
    norm = np.reshape(norm, (norm.shape[0], 1))
    # 将范数复制为与 embeddings 相同形状的数组
    norm = np.tile(norm, (1, len(embeddings[0]))
    # 返回归一化后的嵌入向量
    return np.divide(embeddings, norm)

# 定义支持的向量搜索类型类
class SupportedVSType:
    FAISS = 'faiss'
    MILVUS = 'milvus'
    DEFAULT = 'default'
    ZILLIZ = 'zilliz'
    PG = 'pg'
    ES = 'es'
    CHROMADB = 'chromadb'

# 定义 KBService 抽象基类
class KBService(ABC):
    # 初始化函数，接受知识库名称和嵌入模型名称作为参数，设置实例属性
    def __init__(self,
                 knowledge_base_name: str,
                 embed_model: str = EMBEDDING_MODEL,
                 ):
        # 设置知识库名称
        self.kb_name = knowledge_base_name
        # 获取知识库信息，如果不存在则返回默认信息
        self.kb_info = KB_INFO.get(knowledge_base_name, f"关于{knowledge_base_name}的知识库")
        # 设置嵌入模型
        self.embed_model = embed_model
        # 获取知识库路径
        self.kb_path = get_kb_path(self.kb_name)
        # 获取文档路径
        self.doc_path = get_doc_path(self.kb_name)
        # 执行初始化操作
        self.do_init()

    # 返回实例的字符串表示形式
    def __repr__(self) -> str:
        return f"{self.kb_name} @ {self.embed_model}"

    # 保存向量库的方法，暂时未实现
    def save_vector_store(self):
        '''
        保存向量库:FAISS保存到磁盘，milvus保存到数据库。PGVector暂未支持
        '''
        pass

    # 创建知识库的方法
    def create_kb(self):
        """
        创建知识库
        """
        # 如果文档路径不存在，则创建
        if not os.path.exists(self.doc_path):
            os.makedirs(self.doc_path)
        # 执行创建知识库的操作
        self.do_create_kb()
        # 将知识库信息添加到数据库中
        status = add_kb_to_db(self.kb_name, self.kb_info, self.vs_type(), self.embed_model)
        return status

    # 清空向量库中所有内容的方法
    def clear_vs(self):
        """
        删除向量库中所有内容
        """
        # 执行清空向量库的操作
        self.do_clear_vs()
        # 从数据库中删除知识库相关文件
        status = delete_files_from_db(self.kb_name)
        return status

    # 删除知识库的方法
    def drop_kb(self):
        """
        删除知识库
        """
        # 执行删除知识库的操作
        self.do_drop_kb()
        # 从数据库中删除知识库
        status = delete_kb_from_db(self.kb_name)
        return status

    # 将文档列表转换为嵌入向量的方法
    def _docs_to_embeddings(self, docs: List[Document]) -> Dict:
        '''
        将 List[Document] 转化为 VectorStore.add_embeddings 可以接受的参数
        '''
        return embed_documents(docs=docs, embed_model=self.embed_model, to_query=False)
    # 向知识库添加文件，可以指定是否将文本向量化
    def add_doc(self, kb_file: KnowledgeFile, docs: List[Document] = [], **kwargs):
        """
        向知识库添加文件
        如果指定了docs，则不再将文本向量化，并将数据库对应条目标为custom_docs=True
        """
        # 如果指定了docs，则将custom_docs设置为True，并为每个文档设置元数据中的"source"为kb_file的文件名
        if docs:
            custom_docs = True
            for doc in docs:
                doc.metadata.setdefault("source", kb_file.filename)
        else:
            # 如果未指定docs，则将kb_file转换为文本，并将custom_docs设置为False
            docs = kb_file.file2text()
            custom_docs = False

        # 如果docs不为空
        if docs:
            # 将每个文档的元数据中的"source"改为相对路径
            for doc in docs:
                try:
                    source = doc.metadata.get("source", "")
                    if os.path.isabs(source):
                        rel_path = Path(source).relative_to(self.doc_path)
                        doc.metadata["source"] = str(rel_path.as_posix().strip("/"))
                except Exception as e:
                    print(f"cannot convert absolute path ({source}) to relative path. error is : {e}")
            # 删除已存在的kb_file，并将docs添加到知识库中
            self.delete_doc(kb_file)
            doc_infos = self.do_add_doc(docs, **kwargs)
            # 将文件信息添加到数据库中
            status = add_file_to_db(kb_file,
                                    custom_docs=custom_docs,
                                    docs_count=len(docs),
                                    doc_infos=doc_infos)
        else:
            status = False
        return status

    # 从知识库删除文件
    def delete_doc(self, kb_file: KnowledgeFile, delete_content: bool = False, **kwargs):
        """
        从知识库删除文件
        """
        # 删除知识库中的文件
        self.do_delete_doc(kb_file, **kwargs)
        # 从数据库中删除文件信息
        status = delete_file_from_db(kb_file)
        # 如果delete_content为True且文件存在，则删除文件
        if delete_content and os.path.exists(kb_file.filepath):
            os.remove(kb_file.filepath)
        return status

    # 更新知识库介绍
    def update_info(self, kb_info: str):
        """
        更新知识库介绍
        """
        # 更新知识库介绍，并将信息添加到数据库中
        self.kb_info = kb_info
        status = add_kb_to_db(self.kb_name, self.kb_info, self.vs_type(), self.embed_model)
        return status
    # 更新向量库中的文档，使用content中的文件更新
    def update_doc(self, kb_file: KnowledgeFile, docs: List[Document] = [], **kwargs):
        """
        使用content中的文件更新向量库
        如果指定了docs，则使用自定义docs，并将数据库对应条目标为custom_docs=True
        """
        # 如果指定的kb_file文件路径存在
        if os.path.exists(kb_file.filepath):
            # 删除向量库中的文档
            self.delete_doc(kb_file, **kwargs)
            # 添加新的文档到向量库中
            return self.add_doc(kb_file, docs=docs, **kwargs)

    # 检查指定文件名的文档是否存在于数据库中
    def exist_doc(self, file_name: str):
        return file_exists_in_db(KnowledgeFile(knowledge_base_name=self.kb_name,
                                               filename=file_name))

    # 列出数据库中的文件
    def list_files(self):
        return list_files_from_db(self.kb_name)

    # 统计数据库中的文件数量
    def count_files(self):
        return count_files_from_db(self.kb_name)

    # 根据查询内容搜索文档
    def search_docs(self,
                    query: str,
                    top_k: int = VECTOR_SEARCH_TOP_K,
                    score_threshold: float = SCORE_THRESHOLD,
                    ) ->List[Document]:
        # 执行搜索操作，返回文档列表
        docs = self.do_search(query, top_k, score_threshold)
        return docs

    # 根据文档id列表获取文档
    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        return []

    # 根据文档id列表删除文档
    def del_doc_by_ids(self, ids: List[str]) -> bool:
        # 抛出未实现的错误
        raise NotImplementedError

    # 根据文档id和文档内容更新文档
    def update_doc_by_ids(self, docs: Dict[str, Document]) -> bool:
        '''
        传入参数为： {doc_id: Document, ...}
        如果对应 doc_id 的值为 None，或其 page_content 为空，则删除该文档
        '''
        # 删除指定文档id的文档
        self.del_doc_by_ids(list(docs.keys()))
        docs = []
        ids = []
        # 遍历传入的文档字典
        for k, v in docs.items():
            # 如果文档为空或者文档内容为空，则跳过
            if not v or not v.page_content.strip():
                continue
            # 添加文档id和文档内容到对应列表中
            ids.append(k)
            docs.append(v)
        # 执行添加文档操作
        self.do_add_doc(docs=docs, ids=ids)
        return True
    # 定义一个方法，用于列出文档信息
    def list_docs(self, file_name: str = None, metadata: Dict = {}) -> List[DocumentWithVSId]:
        '''
        通过file_name或metadata检索Document
        '''
        # 从数据库中获取文档信息
        doc_infos = list_docs_from_db(kb_name=self.kb_name, file_name=file_name, metadata=metadata)
        docs = []
        # 遍历文档信息
        for x in doc_infos:
            # 根据文档ID获取文档信息
            doc_info = self.get_doc_by_ids([x["id"]])[0]
            if doc_info is not None:
                # 处理非空的情况
                doc_with_id = DocumentWithVSId(**doc_info.dict(), id=x["id"])
                docs.append(doc_with_id)
            else:
                # 处理空的情况
                # 可以选择跳过当前循环迭代或执行其他操作
                pass
        return docs

    @abstractmethod
    def do_create_kb(self):
        """
        创建知识库子类实自己逻辑
        """
        pass

    @staticmethod
    def list_kbs_type():
        return list(kbs_config.keys())

    @classmethod
    def list_kbs():
        return list_kbs_from_db()

    # 检查知识库是否存在
    def exists(self, kb_name: str = None):
        kb_name = kb_name or self.kb_name
        return kb_exists(kb_name)

    @abstractmethod
    def vs_type(self) -> str:
        pass

    @abstractmethod
    def do_init(self):
        pass

    @abstractmethod
    def do_drop_kb(self):
        """
        删除知识库子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_search(self,
                  query: str,
                  top_k: int,
                  score_threshold: float,
                  ) -> List[Tuple[Document, float]]:
        """
        搜索知识库子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_add_doc(self,
                   docs: List[Document],
                   **kwargs,
                   ) -> List[Dict]:
        """
        向知识库添加文档子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_delete_doc(self,
                      kb_file: KnowledgeFile):
        """
        从知识库删除文档子类实自己逻辑
        """
        pass

    @abstractmethod
    # 定义一个方法，用于从知识库中删除全部向量子类实例的逻辑
    def do_clear_vs(self):
        # 该方法暂时不包含任何具体逻辑，使用 pass 关键字占位
        pass
# 定义 KBServiceFactory 类
class KBServiceFactory:

    # 静态方法，根据知识库名称获取对应的 KBService 对象
    @staticmethod
    def get_service_by_name(kb_name: str) -> KBService:
        # 从数据库加载知识库信息
        _, vs_type, embed_model = load_kb_from_db(kb_name)
        # 如果数据库中没有该知识库，则返回 None
        if _ is None:  
            return None
        # 调用 get_service 方法获取 KBService 对象
        return KBServiceFactory.get_service(kb_name, vs_type, embed_model)

    # 静态方法，获取默认的 KBService 对象
    @staticmethod
    def get_default():
        return KBServiceFactory.get_service("default", SupportedVSType.DEFAULT)


# 获取知识库详情信息
def get_kb_details() -> List[Dict]:
    # 获取文件夹中的知识库列表
    kbs_in_folder = list_kbs_from_folder()
    # 获取数据库中的知识库列表
    kbs_in_db = KBService.list_kbs()
    # 初始化结果字典
    result = {}

    # 遍历文件夹中的知识库列表
    for kb in kbs_in_folder:
        # 初始化知识库详情信息
        result[kb] = {
            "kb_name": kb,
            "vs_type": "",
            "kb_info": "",
            "embed_model": "",
            "file_count": 0,
            "create_time": None,
            "in_folder": True,
            "in_db": False,
        }

    # 遍历数据库中的知识库列表
    for kb in kbs_in_db:
        # 获取知识库详情信息
        kb_detail = get_kb_detail(kb)
        # 如果知识库详情信息存在
        if kb_detail:
            kb_detail["in_db"] = True
            # 如果知识库在结果字典中存在，则更新信息，否则添加到结果字典中
            if kb in result:
                result[kb].update(kb_detail)
            else:
                kb_detail["in_folder"] = False
                result[kb] = kb_detail

    # 初始化数据列表
    data = []
    # 遍历结果字典的值，并添加编号信息
    for i, v in enumerate(result.values()):
        v['No'] = i + 1
        data.append(v)

    return data


# 获取知识库文件详情信息
def get_kb_file_details(kb_name: str) -> List[Dict]:
    # 根据知识库名称获取 KBService 对象
    kb = KBServiceFactory.get_service_by_name(kb_name)
    # 如果 KBService 对象不存在，则返回空列表
    if kb is None:
        return []

    # 获取文件夹中的文件列表
    files_in_folder = list_files_from_folder(kb_name)
    # 获取数据库中的文件列表
    files_in_db = kb.list_files()
    # 初始化结果字典
    result = {}

    # 遍历文件夹中的文件列表
    for doc in files_in_folder:
        # 初始化文件详情信息
        result[doc] = {
            "kb_name": kb_name,
            "file_name": doc,
            "file_ext": os.path.splitext(doc)[-1],
            "file_version": 0,
            "document_loader": "",
            "docs_count": 0,
            "text_splitter": "",
            "create_time": None,
            "in_folder": True,
            "in_db": False,
        }
    # 创建一个字典，将result中的每个元素转换为小写后作为key，原元素作为value
    lower_names = {x.lower(): x for x in result}
    # 遍历数据库中的文件
    for doc in files_in_db:
        # 获取文件的详细信息
        doc_detail = get_file_detail(kb_name, doc)
        # 如果文件详细信息存在
        if doc_detail:
            # 将"in_db"字段设置为True
            doc_detail["in_db"] = True
            # 如果文件名的小写形式在lower_names中存在
            if doc.lower() in lower_names:
                # 更新result中对应文件名的详细信息
                result[lower_names[doc.lower()]].update(doc_detail)
            else:
                # 将"in_folder"字段设置为False，并将文件详细信息添加到result中
                doc_detail["in_folder"] = False
                result[doc] = doc_detail

    # 创建一个空列表data
    data = []
    # 遍历result中的值
    for i, v in enumerate(result.values()):
        # 将"No"字段设置为索引加1
        v['No'] = i + 1
        # 将值添加到data列表中
        data.append(v)

    # 返回data列表
    return data
class EmbeddingsFunAdapter(Embeddings):
    # 初始化函数，接受一个嵌入模型的字符串参数，默认为全局变量EMBEDDING_MODEL
    def __init__(self, embed_model: str = EMBEDDING_MODEL):
        self.embed_model = embed_model

    # 将文档列表嵌入，返回嵌入向量的列表
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 调用embed_texts函数将文本列表嵌入，返回嵌入向量数据
        embeddings = embed_texts(texts=texts, embed_model=self.embed_model, to_query=False).data
        # 对嵌入向量进行归一化处理，并转换为列表返回
        return normalize(embeddings).tolist()

    # 将查询文本嵌入，返回嵌入向量的列表
    def embed_query(self, text: str) -> List[float]:
        # 调用embed_texts函数将文本嵌入，返回嵌入向量数据
        embeddings = embed_texts(texts=[text], embed_model=self.embed_model, to_query=True).data
        # 获取查询文本的嵌入向量
        query_embed = embeddings[0]
        # 将一维数组转换为二维数组
        query_embed_2d = np.reshape(query_embed, (1, -1))
        # 对嵌入向量进行归一化处理
        normalized_query_embed = normalize(query_embed_2d)
        # 将结果转换为一维数组并返回
        return normalized_query_embed[0].tolist()

    # 异步函数，将文档列表嵌入，返回嵌入向量的列表
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        # 调用aembed_texts异步函数将文本列表嵌入，返回嵌入向量数据
        embeddings = (await aembed_texts(texts=texts, embed_model=self.embed_model, to_query=False)).data
        # 对嵌入向量进行归一化处理，并转换为列表返回
        return normalize(embeddings).tolist()

    # 异步函数，将查询文本嵌入，返回嵌入向量的列表
    async def aembed_query(self, text: str) -> List[float]:
        # 调用aembed_texts异步函数将文本嵌入，返回嵌入向量数据
        embeddings = (await aembed_texts(texts=[text], embed_model=self.embed_model, to_query=True)).data
        # 获取查询文本的嵌入向量
        query_embed = embeddings[0]
        # 将一维数组转换为二维数组
        query_embed_2d = np.reshape(query_embed, (1, -1))
        # 对嵌入向量进行归一化处理
        normalized_query_embed = normalize(query_embed_2d)
        # 将结果转换为一维数组并返回
        return normalized_query_embed[0].tolist()


# 处理得分阈值的函数，根据得分阈值和文档数量k对文档进行筛选
def score_threshold_process(score_threshold, k, docs):
    # 如果得分阈值不为None
    if score_threshold is not None:
        # 定义比较函数为小于等于
        cmp = (
            operator.le
        )
        # 对文档列表进行筛选，只保留得分小于等于阈值的文档
        docs = [
            (doc, similarity)
            for doc, similarity in docs
            if cmp(similarity, score_threshold)
        ]
    # 返回前k个文档
    return docs[:k]
```