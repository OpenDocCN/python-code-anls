# `.\Langchain-Chatchat\server\db\repository\knowledge_file_repository.py`

```py
# 导入所需的模块和类
from server.db.models.knowledge_base_model import KnowledgeBaseModel
from server.db.models.knowledge_file_model import KnowledgeFileModel, FileDocModel
from server.db.session import with_session
from server.knowledge_base.utils import KnowledgeFile
from typing import List, Dict

# 使用装饰器，确保在数据库会话中执行函数
@with_session
def list_file_num_docs_id_by_kb_name_and_file_name(session,
                                                   kb_name: str,
                                                   file_name: str,
                                                   ) -> List[int]:
    '''
    列出某知识库某文件对应的所有Document的id。
    返回形式：[str, ...]
    '''
    # 查询指定知识库和文件名对应的所有Document的id
    doc_ids = session.query(FileDocModel.doc_id).filter_by(kb_name=kb_name, file_name=file_name).all()
    # 将查询结果转换为整数列表并返回
    return [int(_id[0]) for _id in doc_ids]

# 使用装饰器，确保在数据库会话中执行函数
@with_session
def list_docs_from_db(session,
                      kb_name: str,
                      file_name: str = None,
                      metadata: Dict = {},
                      ) -> List[Dict]:
    '''
    列出某知识库某文件对应的所有Document。
    返回形式：[{"id": str, "metadata": dict}, ...]
    '''
    # 查询指定知识库和文件名对应的所有Document
    docs = session.query(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name))
    # 如果有文件名参数，则进一步筛选
    if file_name:
        docs = docs.filter(FileDocModel.file_name.ilike(file_name))
    # 根据元数据筛选Document
    for k, v in metadata.items():
        docs = docs.filter(FileDocModel.meta_data[k].as_string() == str(v))

    # 将查询结果转换为包含id和metadata的字典列表并返回
    return [{"id": x.doc_id, "metadata": x.metadata} for x in docs.all()]

# 使用装饰器，确保在数据库会话中执行函数
@with_session
def delete_docs_from_db(session,
                        kb_name: str,
                        file_name: str = None,
                        ) -> List[Dict]:
    '''
    删除某知识库某文件对应的所有Document，并返回被删除的Document。
    返回形式：[{"id": str, "metadata": dict}, ...]
    '''
    # 获取要删除的Document列表
    docs = list_docs_from_db(kb_name=kb_name, file_name=file_name)
    # 创建查询对象
    query = session.query(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name))
    # 如果有文件名参数，则进一步筛选
    if file_name:
        query = query.filter(FileDocModel.file_name.ilike(file_name))
    # 删除符合条件的Document，不同步会话
    query.delete(synchronize_session=False)
    # 提交会话中的所有更改
    session.commit()
    # 返回文档列表
    return docs
# 将某知识库某文件对应的所有Document信息添加到数据库
# doc_infos形式：[{"id": str, "metadata": dict}, ...]
@with_session
def add_docs_to_db(session,
                   kb_name: str,
                   file_name: str,
                   doc_infos: List[Dict]):
    # 检查doc_infos是否为None，如果是则打印错误信息并返回False
    if doc_infos is None:
        print("输入的server.db.repository.knowledge_file_repository.add_docs_to_db的doc_infos参数为None")
        return False
    # 遍历doc_infos，创建FileDocModel对象并添加到数据库
    for d in doc_infos:
        obj = FileDocModel(
            kb_name=kb_name,
            file_name=file_name,
            doc_id=d["id"],
            meta_data=d["metadata"],
        )
        session.add(obj)
    return True


# 查询数据库中某知识库的文件数量
@with_session
def count_files_from_db(session, kb_name: str) -> int:
    return session.query(KnowledgeFileModel).filter(KnowledgeFileModel.kb_name.ilike(kb_name)).count()


# 查询数据库中某知识库的文件列表
@with_session
def list_files_from_db(session, kb_name):
    # 查询数据库中某知识库的文件列表
    files = session.query(KnowledgeFileModel).filter(KnowledgeFileModel.kb_name.ilike(kb_name)).all()
    # 提取文件名列表
    docs = [f.file_name for f in files]
    return docs


# 向数据库中添加文件
@with_session
def add_file_to_db(session,
                   kb_file: KnowledgeFile,
                   docs_count: int = 0,
                   custom_docs: bool = False,
                   doc_infos: List[Dict] = [],  # 形式：[{"id": str, "metadata": dict}, ...]
                   ):
    # 查询数据库中指定知识库的信息
    kb = session.query(KnowledgeBaseModel).filter_by(kb_name=kb_file.kb_name).first()
    # 如果 kb 为真，则执行以下操作
    if kb:
        # 查询数据库中是否已存在相同文件名和知识库名的文件
        existing_file: KnowledgeFileModel = (session.query(KnowledgeFileModel)
                                             .filter(KnowledgeFileModel.kb_name.ilike(kb_file.kb_name),
                                                     KnowledgeFileModel.file_name.ilike(kb_file.filename))
                                             .first())
        # 获取文件的修改时间和大小
        mtime = kb_file.get_mtime()
        size = kb_file.get_size()

        # 如果存在相同文件，则更新文件信息和版本号
        if existing_file:
            existing_file.file_mtime = mtime
            existing_file.file_size = size
            existing_file.docs_count = docs_count
            existing_file.custom_docs = custom_docs
            existing_file.file_version += 1
        # 否则，添加新文件到数据库
        else:
            new_file = KnowledgeFileModel(
                file_name=kb_file.filename,
                file_ext=kb_file.ext,
                kb_name=kb_file.kb_name,
                document_loader_name=kb_file.document_loader_name,
                text_splitter_name=kb_file.text_splitter_name or "SpacyTextSplitter",
                file_mtime=mtime,
                file_size=size,
                docs_count=docs_count,
                custom_docs=custom_docs,
            )
            # 增加知识库的文件数量
            kb.file_count += 1
            session.add(new_file)
        
        # 将文档信息添加到数据库
        add_docs_to_db(kb_name=kb_file.kb_name, file_name=kb_file.filename, doc_infos=doc_infos)
    
    # 返回 True
    return True
# 从数据库中删除指定文件
@with_session
def delete_file_from_db(session, kb_file: KnowledgeFile):
    # 查询数据库中是否存在指定文件
    existing_file = (session.query(KnowledgeFileModel)
                     .filter(KnowledgeFileModel.file_name.ilike(kb_file.filename),
                             KnowledgeFileModel.kb_name.ilike(kb_file.kb_name))
                     .first())
    # 如果存在指定文件
    if existing_file:
        # 删除文件
        session.delete(existing_file)
        # 从数据库中删除文件相关的文档
        delete_docs_from_db(kb_name=kb_file.kb_name, file_name=kb_file.filename)
        # 提交事务
        session.commit()

        # 查询知识库对象
        kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_file.kb_name)).first()
        # 如果知识库对象存在
        if kb:
            # 更新文件数量
            kb.file_count -= 1
            # 提交事务
            session.commit()
    return True


# 从数据库中删除指定知识库下的所有文件
@with_session
def delete_files_from_db(session, knowledge_base_name: str):
    # 删除指定知识库下的所有文件
    session.query(KnowledgeFileModel).filter(KnowledgeFileModel.kb_name.ilike(knowledge_base_name)).delete(
        synchronize_session=False)
    # 删除指定知识库下的所有文件相关的文档
    session.query(FileDocModel).filter(FileDocModel.kb_name.ilike(knowledge_base_name)).delete(
        synchronize_session=False)
    # 查询知识库对象
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(knowledge_base_name)).first()
    # 如果知识库对象存在
    if kb:
        # 将文件数量设置为0
        kb.file_count = 0

    # 提交事务
    session.commit()
    return True


# 检查数据库中是否存在指定文件
@with_session
def file_exists_in_db(session, kb_file: KnowledgeFile):
    # 查询数据库中是否存在指定文件
    existing_file = (session.query(KnowledgeFileModel)
                     .filter(KnowledgeFileModel.file_name.ilike(kb_file.filename),
                             KnowledgeFileModel.kb_name.ilike(kb_file.kb_name))
                     .first())
    # 如果存在指定文件，返回True；否则返回False
    return True if existing_file else False


# 获取指定知识库下指定文件的详细信息
@with_session
def get_file_detail(session, kb_name: str, filename: str) -> dict:
    # 查询指定知识库下指定文件的详细信息
    file: KnowledgeFileModel = (session.query(KnowledgeFileModel)
                                .filter(KnowledgeFileModel.file_name.ilike(filename),
                                        KnowledgeFileModel.kb_name.ilike(kb_name))
                                .first())
    # 如果文件对象存在，则返回包含文件信息的字典
    if file:
        return {
            "kb_name": file.kb_name,  # 文件知识库名称
            "file_name": file.file_name,  # 文件名
            "file_ext": file.file_ext,  # 文件扩展名
            "file_version": file.file_version,  # 文件版本
            "document_loader": file.document_loader_name,  # 文档加载器名称
            "text_splitter": file.text_splitter_name,  # 文本分割器名称
            "create_time": file.create_time,  # 创建时间
            "file_mtime": file.file_mtime,  # 文件修改时间
            "file_size": file.file_size,  # 文件大小
            "custom_docs": file.custom_docs,  # 自定义文档
            "docs_count": file.docs_count,  # 文档数量
        }
    # 如果文件对象不存在，则返回空字典
    else:
        return {}
```