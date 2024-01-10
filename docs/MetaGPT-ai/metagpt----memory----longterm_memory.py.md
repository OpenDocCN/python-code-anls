# `MetaGPT\metagpt\memory\longterm_memory.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Desc   : the implement of Long-term memory
@Modified By: mashenquan, 2023/8/20. Remove global configuration `CONFIG`, enable configuration support for business isolation.
"""

# 导入必要的模块
from typing import Optional
from pydantic import ConfigDict, Field
from metagpt.logs import logger
from metagpt.memory import Memory
from metagpt.memory.memory_storage import MemoryStorage
from metagpt.roles.role import RoleContext
from metagpt.schema import Message

# 定义 LongTermMemory 类，继承自 Memory 类
class LongTermMemory(Memory):
    """
    The Long-term memory for Roles
    - recover memory when it staruped
    - update memory when it changed
    """

    # 定义 model_config 属性
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 定义 memory_storage 属性，默认使用 MemoryStorage 工厂函数创建
    memory_storage: MemoryStorage = Field(default_factory=MemoryStorage)
    # 定义 rc 属性，默认为 None
    rc: Optional[RoleContext] = None
    # 定义 msg_from_recover 属性，默认为 False
    msg_from_recover: bool = False

    # 定义 recover_memory 方法，用于恢复内存
    def recover_memory(self, role_id: str, rc: RoleContext):
        # 从 memory_storage 中恢复内存
        messages = self.memory_storage.recover_memory(role_id)
        self.rc = rc
        # 如果 memory_storage 未初始化，则记录警告日志
        if not self.memory_storage.is_initialized:
            logger.warning(f"It may the first time to run Agent {role_id}, the long-term memory is empty")
        else:
            logger.warning(
                f"Agent {role_id} has existing memory storage with {len(messages)} messages " f"and has recovered them."
            )
        self.msg_from_recover = True
        # 将恢复的消息批量添加到内存中
        self.add_batch(messages)
        self.msg_from_recover = False

    # 定义 add 方法，用于添加消息到内存中
    def add(self, message: Message):
        super().add(message)
        for action in self.rc.watch:
            if message.cause_by == action and not self.msg_from_recover:
                # 当前，只将角色的观察消息添加到其 memory_storage 中，并忽略重复添加从恢复中的消息
                self.memory_storage.add(message)

    # 定义 find_news 方法，用于查找新消息
    def find_news(self, observed: list[Message], k=0) -> list[Message]:
        """
        find news (previously unseen messages) from the the most recent k memories, from all memories when k=0
            1. find the short-term memory(stm) news
            2. furthermore, filter out similar messages based on ltm(long-term memory), get the final news
        """
        # 查找短期内存（stm）中的新消息
        stm_news = super().find_news(observed, k=k)  # shot-term memory news
        if not self.memory_storage.is_initialized:
            # 如果 memory_storage 未初始化，则使用默认的 `find_news` 方法获取 stm_news
            return stm_news

        ltm_news: list[Message] = []
        for mem in stm_news:
            # 过滤掉在 ltm 中与先前看到的消息相似的消息，只保留新消息
            mem_searched = self.memory_storage.search_dissimilar(mem)
            if len(mem_searched) > 0:
                ltm_news.append(mem)
        return ltm_news[-k:]

    # 定义 delete 方法，用于删除消息
    def delete(self, message: Message):
        super().delete(message)
        # TODO delete message in memory_storage

    # 定义 clear 方法，用于清空内存
    def clear(self):
        super().clear()
        self.memory_storage.clean()

```