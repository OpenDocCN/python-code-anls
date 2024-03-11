# `.\Langchain-Chatchat\server\knowledge_base\kb_api.py`

```py
# 导入 urllib 模块
import urllib
# 从 server.utils 模块中导入 BaseResponse, ListResponse 类
from server.utils import BaseResponse, ListResponse
# 从 server.knowledge_base.utils 模块中导入 validate_kb_name 函数
from server.knowledge_base.utils import validate_kb_name
# 从 server.knowledge_base.kb_service.base 模块中导入 KBServiceFactory 类
from server.knowledge_base.kb_service.base import KBServiceFactory
# 从 server.db.repository.knowledge_base_repository 模块中导入 list_kbs_from_db 函数
from server.db.repository.knowledge_base_repository import list_kbs_from_db
# 从 configs 模块中导入 EMBEDDING_MODEL, logger, log_verbose 变量
from configs import EMBEDDING_MODEL, logger, log_verbose
# 从 fastapi 模块中导入 Body 类
from fastapi import Body

# 定义函数 list_kbs，用于获取知识库列表
def list_kbs():
    # 返回知识库列表的 ListResponse 对象，数据来源于 list_kbs_from_db 函数
    return ListResponse(data=list_kbs_from_db())

# 定义函数 create_kb，用于创建知识库
def create_kb(knowledge_base_name: str = Body(..., examples=["samples"]),
              vector_store_type: str = Body("faiss"),
              embed_model: str = Body(EMBEDDING_MODEL),
              ) -> BaseResponse:
    # 创建选定的知识库
    # 如果知识库名称不合法，则返回 BaseResponse 对象，code 为 403，msg 为 "Don't attack me"
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    # 如果知识库名称为空或只包含空格，则返回 BaseResponse 对象，code 为 404，msg 为 "知识库名称不能为空，请重新填写知识库名称"
    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        return BaseResponse(code=404, msg="知识库名称不能为空，请重新填写知识库名称")

    # 根据知识库名称获取 KBService 对象
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    # 如果已存在同名知识库，则返回 BaseResponse 对象，code 为 404，msg 包含已存在知识库名称
    if kb is not None:
        return BaseResponse(code=404, msg=f"已存在同名知识库 {knowledge_base_name}")

    # 根据知识库名称、向量存储类型、嵌入模型获取 KBService 对象
    kb = KBServiceFactory.get_service(knowledge_base_name, vector_store_type, embed_model)
    try:
        # 创建知识库
        kb.create_kb()
    except Exception as e:
        # 如果创建知识库出错，则记录错误信息并返回 BaseResponse 对象，code 为 500，msg 包含错误信息
        msg = f"创建知识库出错： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    # 返回 BaseResponse 对象，code 为 200，msg 包含已新增知识库名称
    return BaseResponse(code=200, msg=f"已新增知识库 {knowledge_base_name}")

# 定义函数 delete_kb，用于删除知识库
def delete_kb(
        knowledge_base_name: str = Body(..., examples=["samples"])
) -> BaseResponse:
    # 删除选定的知识库
    # 如果知识库名称不合法，则返回 BaseResponse 对象，code 为 403，msg 为 "Don't attack me"
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    # 对知识库名称进行 URL 解码
    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)

    # 根据知识库名称获取 KBService 对象
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    # 如果知识库对象为空，则返回未找到知识库的响应
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    try:
        # 清空知识库中的所有知识
        status = kb.clear_vs()
        # 删除知识库
        status = kb.drop_kb()
        # 如果删除成功，则返回成功删除知识库的响应
        if status:
            return BaseResponse(code=200, msg=f"成功删除知识库 {knowledge_base_name}")
    except Exception as e:
        # 如果删除知识库时出现异常，则记录错误日志并返回删除知识库失败的响应
        msg = f"删除知识库时出现意外： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    # 如果删除知识库失败，则返回删除知识库失败的响应
    return BaseResponse(code=500, msg=f"删除知识库失败 {knowledge_base_name}")
```