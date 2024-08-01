# `.\DB-GPT-src\dbgpt\serve\conversation\models\models.py`

```py
"""
This is an auto-generated model file
You can define your own models and DAOs here
"""
# 导入所需模块和类
import json
from typing import Any, Dict, List, Optional, Union

# 导入核心模块和类
from dbgpt.core import MessageStorageItem
# 导入聊天历史相关数据库实体
from dbgpt.storage.chat_history.chat_history_db import ChatHistoryEntity as ServeEntity
from dbgpt.storage.chat_history.chat_history_db import ChatHistoryMessageEntity
# 导入元数据相关模块和类
from dbgpt.storage.metadata import BaseDao, Model, db
# 导入分页结果类
from dbgpt.util import PaginationResult

# 导入相关的 API schemas 和配置
from ..api.schemas import ServeRequest, ServerResponse
from ..config import SERVER_APP_TABLE_NAME, ServeConfig

# 定义 ServeDao 类，继承自 BaseDao 类，用于处理 Conversation 相关操作
class ServeDao(BaseDao[ServeEntity, ServeRequest, ServerResponse]):
    """The DAO class for Conversation"""

    def __init__(self, serve_config: ServeConfig):
        # 调用父类构造函数初始化对象
        super().__init__()
        # 存储 ServeConfig 实例的引用
        self._serve_config = serve_config

    def from_request(self, request: Union[ServeRequest, Dict[str, Any]]) -> ServeEntity:
        """Convert the request to an entity

        Args:
            request (Union[ServeRequest, Dict[str, Any]]): The request

        Returns:
            T: The entity
        """
        # 如果 request 是 ServeRequest 类型，则转换成字典
        request_dict = (
            request.to_dict() if isinstance(request, ServeRequest) else request
        )
        # 根据 request_dict 创建 ServeEntity 实例
        entity = ServeEntity(**request_dict)
        # TODO 实现自定义逻辑，将 request_dict 转换为 entity
        return entity

    def to_request(self, entity: ServeEntity) -> ServeRequest:
        """Convert the entity to a request

        Args:
            entity (T): The entity

        Returns:
            REQ: The request
        """
        # TODO 实现自定义逻辑，将 entity 转换为 ServeRequest 类型的对象
        return ServeRequest()

    def to_response(self, entity: ServeEntity) -> ServerResponse:
        """Convert the entity to a response

        Args:
            entity (T): The entity

        Returns:
            RES: The response
        """
        # TODO 实现自定义逻辑，将 entity 转换为 ServerResponse 类型的对象
        return ServerResponse(
            conv_uid=entity.conv_uid,
            user_input=entity.summary,
            chat_mode=entity.chat_mode,
            user_name=entity.user_name,
            sys_code=entity.sys_code,
        )
    def get_latest_message(self, conv_uid: str) -> Optional[MessageStorageItem]:
        """Get the latest message of a conversation
        
        Args:
            conv_uid (str): The conversation UID
        
        Returns:
            Optional[MessageStorageItem]: The latest message or None if no message found
        """
        # 使用会话对象打开一个数据库会话
        with self.session() as session:
            # 查询指定会话 UID 下的消息实体，按创建时间倒序排序，取第一个（即最新的）
            entity: ChatHistoryMessageEntity = (
                session.query(ChatHistoryMessageEntity)
                .filter(ChatHistoryMessageEntity.conv_uid == conv_uid)
                .order_by(ChatHistoryMessageEntity.gmt_created.desc())
                .first()
            )
            # 如果未找到实体，返回 None
            if not entity:
                return None
            # 尝试解析消息详细信息的 JSON 格式，如果为空则使用空字典
            message_detail = (
                json.loads(entity.message_detail) if entity.message_detail else {}
            )
            # 返回消息存储项对象，包含会话 UID、消息索引和消息详细信息
            return MessageStorageItem(entity.conv_uid, entity.index, message_detail)

    def _parse_old_messages(self, entity: ServeEntity) -> List[Dict[str, Any]]:
        """Parse the old messages
        
        Args:
            entity (ServeEntity): The entity
        
        Returns:
            List[Dict[str, Any]]: The list of old messages
        """
        # 解析实体中的消息 JSON 字符串，返回消息列表
        messages = json.loads(entity.messages)
        return messages

    def get_conv_by_page(
        self, req: ServeRequest, page: int, page_size: int
        ) -> Tuple[List[ServeEntity], int]:
        """Retrieve conversation entities by page
        
        Args:
            req (ServeRequest): The request object
            page (int): The page number
            page_size (int): The number of entities per page
        
        Returns:
            Tuple[List[ServeEntity], int]: A tuple containing a list of ServeEntity objects and total count
        """
    ) -> PaginationResult[ServerResponse]:
        """获取指定页的会话列表

        Args:
            req (ServeRequest): 请求对象
            page (int): 页码
            page_size (int): 每页数据量

        Returns:
            List[ChatHistoryEntity]: 会话列表
        """
        # 使用数据库会话对象，不自动提交事务
        with self.session(commit=False) as session:
            # 创建查询对象，并根据请求构建查询条件
            query = self._create_query_object(session, req)
            # 按创建时间倒序排序查询结果
            query = query.order_by(ServeEntity.gmt_created.desc())
            # 获取符合条件的总记录数
            total_count = query.count()
            # 根据页码和每页数据量计算偏移量，并限制返回结果条数
            items = query.offset((page - 1) * page_size).limit(page_size)
            # 计算总页数
            total_pages = (total_count + page_size - 1) // page_size
            # 初始化结果列表
            result_items = []
            # 遍历查询结果
            for item in items:
                select_param, model_name = "", None
                # 如果会话项包含消息历史
                if item.messages:
                    # 解析旧消息并找出最后一轮消息
                    messages = self._parse_old_messages(item)
                    last_round = max(messages, key=lambda x: x["chat_order"])
                    # 获取最后一轮消息的参数值
                    if "param_value" in last_round:
                        select_param = last_round["param_value"]
                    else:
                        select_param = ""
                else:
                    # 获取会话的最新消息
                    latest_message = self.get_latest_message(item.conv_uid)
                    if latest_message:
                        # 将最新消息转换为消息对象，并获取附加参数值和模型名称
                        message = latest_message.to_message()
                        select_param = message.additional_kwargs.get("param_value")
                        model_name = message.additional_kwargs.get("model_name")
                # 将会话项转换为响应对象，并设置附加的参数值和模型名称
                res_item = self.to_response(item)
                res_item.select_param = select_param
                res_item.model_name = model_name
                # 将处理后的结果项添加到结果列表
                result_items.append(res_item)

            # 构造分页结果对象
            result = PaginationResult(
                items=result_items,
                total_count=total_count,
                total_pages=total_pages,
                page=page,
                page_size=page_size,
            )

        # 返回分页结果对象
        return result
```