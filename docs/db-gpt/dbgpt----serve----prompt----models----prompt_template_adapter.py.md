# `.\DB-GPT-src\dbgpt\serve\prompt\models\prompt_template_adapter.py`

```py
from typing import Type

from sqlalchemy.orm import Session

from dbgpt.core.interface.prompt import PromptTemplateIdentifier, StoragePromptTemplate
from dbgpt.core.interface.storage import StorageItemAdapter

from .models import ServeEntity

class PromptTemplateAdapter(StorageItemAdapter[StoragePromptTemplate, ServeEntity]):
    # 将 StoragePromptTemplate 转换为 ServeEntity 格式
    def to_storage_format(self, item: StoragePromptTemplate) -> ServeEntity:
        return ServeEntity(
            chat_scene=item.chat_scene,  # 聊天场景
            sub_chat_scene=item.sub_chat_scene,  # 子聊天场景
            prompt_type=item.prompt_type,  # 提示类型
            prompt_name=item.prompt_name,  # 提示名称
            content=item.content,  # 内容
            input_variables=item.input_variables,  # 输入变量
            model=item.model,  # 模型
            prompt_language=item.prompt_language,  # 提示语言
            prompt_format=item.prompt_format,  # 提示格式
            user_name=item.user_name,  # 用户名称
            sys_code=item.sys_code,  # 系统代码
        )

    # 将 ServeEntity 格式转换为 StoragePromptTemplate
    def from_storage_format(self, model: ServeEntity) -> StoragePromptTemplate:
        return StoragePromptTemplate(
            chat_scene=model.chat_scene,  # 聊天场景
            sub_chat_scene=model.sub_chat_scene,  # 子聊天场景
            prompt_type=model.prompt_type,  # 提示类型
            prompt_name=model.prompt_name,  # 提示名称
            content=model.content,  # 内容
            input_variables=model.input_variables,  # 输入变量
            model=model.model,  # 模型
            prompt_language=model.prompt_language,  # 提示语言
            prompt_format=model.prompt_format,  # 提示格式
            user_name=model.user_name,  # 用户名称
            sys_code=model.sys_code,  # 系统代码
        )

    # 根据资源标识符获取查询对象
    def get_query_for_identifier(
        self,
        storage_format: Type[ServeEntity],
        resource_id: PromptTemplateIdentifier,
        **kwargs,
    ):
        session: Session = kwargs.get("session")
        if session is None:
            raise Exception("session is None")  # 抛出异常，如果会话对象为空
        query_obj = session.query(ServeEntity)
        for key, value in resource_id.to_dict().items():
            if value is None:
                continue
            # 根据每个资源标识符的键值对，过滤查询对象
            query_obj = query_obj.filter(getattr(ServeEntity, key) == value)
        return query_obj  # 返回过滤后的查询对象
```