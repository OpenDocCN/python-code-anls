# `.\Langchain-Chatchat\server\db\repository\knowledge_metadata_repository.py`

```
# 导入所需模块和类
from server.db.models.knowledge_metadata_model import SummaryChunkModel
from server.db.session import with_session
from typing import List, Dict

# 从数据库中列出某知识库chunk summary
@with_session
def list_summary_from_db(session,
                         kb_name: str,
                         metadata: Dict = {},
                         ) -> List[Dict]:
    '''
    列出某知识库chunk summary。
    返回形式：[{"id": str, "summary_context": str, "doc_ids": str}, ...]
    '''
    # 查询符合条件的SummaryChunkModel对象
    docs = session.query(SummaryChunkModel).filter(SummaryChunkModel.kb_name.ilike(kb_name))

    # 根据metadata筛选符合条件的对象
    for k, v in metadata.items():
        docs = docs.filter(SummaryChunkModel.meta_data[k].as_string() == str(v))

    # 将查询结果转换为字典列表返回
    return [{"id": x.id,
             "summary_context": x.summary_context,
             "summary_id": x.summary_id,
             "doc_ids": x.doc_ids,
             "metadata": x.metadata} for x in docs.all()]

# 从数据库中删除知识库chunk summary
@with_session
def delete_summary_from_db(session,
                           kb_name: str
                           ) -> List[Dict]:
    '''
    删除知识库chunk summary，并返回被删除的Dchunk summary。
    返回形式：[{"id": str, "summary_context": str, "doc_ids": str}, ...]
    '''
    # 获取要删除的summary信息
    docs = list_summary_from_db(kb_name=kb_name)
    # 创建查询对象并删除符合条件的数据
    query = session.query(SummaryChunkModel).filter(SummaryChunkModel.kb_name.ilike(kb_name))
    query.delete(synchronize_session=False)
    # 提交事务
    session.commit()
    # 返回被删除的summary信息
    return docs

# 向数据库中添加总结信息
@with_session
def add_summary_to_db(session,
                      kb_name: str,
                      summary_infos: List[Dict]):
    '''
    将总结信息添加到数据库。
    summary_infos形式：[{"summary_context": str, "doc_ids": str}, ...]
    '''
    # 遍历summary_infos列表，创建SummaryChunkModel对象并添加到数据库
    for summary in summary_infos:
        obj = SummaryChunkModel(
            kb_name=kb_name,
            summary_context=summary["summary_context"],
            summary_id=summary["summary_id"],
            doc_ids=summary["doc_ids"],
            meta_data=summary["metadata"],
        )
        session.add(obj)

    # 提交事务
    session.commit()
    return True

# 以下代码未提供完整，可能是因为字符限制，需要继续补充
# 从数据库中查询指定 kb_name 的 SummaryChunkModel 记录数量并返回
def count_summary_from_db(session, kb_name: str) -> int:
    # 使用 session 对象查询 SummaryChunkModel 表，过滤 kb_name 列值忽略大小写与给定的 kb_name 相匹配的记录，计算数量并返回
    return session.query(SummaryChunkModel).filter(SummaryChunkModel.kb_name.ilike(kb_name)).count()
```