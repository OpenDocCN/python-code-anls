# `.\Langchain-Chatchat\server\db\repository\knowledge_base_repository.py`

```py
# 导入所需的模块和类
from server.db.models.knowledge_base_model import KnowledgeBaseModel
from server.db.session import with_session

# 将函数装饰为数据库会话上下文
@with_session
# 向数据库中添加知识库
def add_kb_to_db(session, kb_name, kb_info, vs_type, embed_model):
    # 查询数据库中是否已存在同名知识库
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    # 如果不存在，则创建新的知识库实例并添加到数据库中
    if not kb:
        kb = KnowledgeBaseModel(kb_name=kb_name, kb_info=kb_info, vs_type=vs_type, embed_model=embed_model)
        session.add(kb)
    else:  # 如果存在同名知识库，则更新其vs_type和embed_model
        kb.kb_info = kb_info
        kb.vs_type = vs_type
        kb.embed_model = embed_model
    return True

# 将函数装饰为数据库会话上下文
@with_session
# 从数据库中列出文件数量大于指定值的知识库
def list_kbs_from_db(session, min_file_count: int = -1):
    # 查询数据库中文件数量大于指定值的知识库名称
    kbs = session.query(KnowledgeBaseModel.kb_name).filter(KnowledgeBaseModel.file_count > min_file_count).all()
    # 将查询结果转换为列表
    kbs = [kb[0] for kb in kbs]
    return kbs

# 将函数装饰为数据库会话上下文
@with_session
# 检查数据库中是否存在指定名称的知识库
def kb_exists(session, kb_name):
    # 查询数据库中是否存在指定名称的知识库
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    # 返回查询结果的布尔值
    status = True if kb else False
    return status

# 将函数装饰为数据库会话上下文
@with_session
# 从数据库中加载指定名称的知识库信息
def load_kb_from_db(session, kb_name):
    # 查询数据库中指定名称的知识库
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    # 如果存在该知识库，则获取其名称、vs_type和embed_model，否则返回None
    if kb:
        kb_name, vs_type, embed_model = kb.kb_name, kb.vs_type, kb.embed_model
    else:
        kb_name, vs_type, embed_model = None, None, None
    return kb_name, vs_type, embed_model

# 将函数装饰为数据库会话上下文
@with_session
# 从数据库中删除指定名称的知识库
def delete_kb_from_db(session, kb_name):
    # 查询数据库中指定名称的知识库
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    # 如果存在该知识库，则删除
    if kb:
        session.delete(kb)
    return True

# 将函数装饰为数据库会话上下文
@with_session
# 从数据库中获取指定名称的知识库详细信息
def get_kb_detail(session, kb_name: str) -> dict:
    # 查询数据库中指定名称的知识库
    kb: KnowledgeBaseModel = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    # 如果 kb 存在，则返回包含 kb 属性的字典
    if kb:
        return {
            "kb_name": kb.kb_name,  # 返回 kb 的名称
            "kb_info": kb.kb_info,  # 返回 kb 的信息
            "vs_type": kb.vs_type,  # 返回 kb 的类型
            "embed_model": kb.embed_model,  # 返回 kb 的嵌入模型
            "file_count": kb.file_count,  # 返回 kb 的文件数量
            "create_time": kb.create_time,  # 返回 kb 的创建时间
        }
    else:
        return {}  # 如果 kb 不存在，则返回空字典
```