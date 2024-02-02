# `MetaGPT\metagpt\provider\zhipuai\async_sse_client.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : async_sse_client to make keep the use of Event to access response
#           refs to `https://github.com/zhipuai/zhipuai-sdk-python/blob/main/zhipuai/utils/sse_client.py`

# 导入所需的模块和类
from zhipuai.utils.sse_client import _FIELD_SEPARATOR, Event, SSEClient

# 创建一个继承自SSEClient的AsyncSSEClient类
class AsyncSSEClient(SSEClient):
    # 定义一个异步方法_aread，用于异步读取数据
    async def _aread(self):
        data = b""
        # 异步遍历事件源的数据块
        async for chunk in self._event_source:
            for line in chunk.splitlines(True):
                data += line
                if data.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
                    yield data
                    data = b""
        if data:
            yield data

    # 定义一个异步方法async_events，用于异步处理事件
    async def async_events(self):
        # 异步遍历_aread方法返回的数据块
        async for chunk in self._aread():
            event = Event()
            # 按行分割数据块，并解码
            for line in chunk.splitlines():
                line = line.decode(self._char_enc)

                # 忽略以分隔符开头的行和空行
                if not line.strip() or line.startswith(_FIELD_SEPARATOR):
                    continue

                data = line.split(_FIELD_SEPARATOR, 1)
                field = data[0]

                # 忽略未知字段
                if field not in event.__dict__:
                    self._logger.debug("Saw invalid field %s while parsing " "Server Side Event", field)
                    continue

                if len(data) > 1:
                    # 如果值以空格开头，则去除空格
                    if data[1].startswith(" "):
                        value = data[1][1:]
                    else:
                        value = data[1]
                else:
                    # 如果分隔符后没有值，则假定为空值
                    value = ""

                # 数据字段可能跨多行，将它们的值连接起来
                if field == "data":
                    event.__dict__[field] += value + "\n"
                else:
                    event.__dict__[field] = value

            # 不包含数据的事件不会被处理
            if not event.data:
                continue

            # 如果数据字段以换行符结尾，则移除它
            if event.data.endswith("\n"):
                event.data = event.data[0:-1]

            # 空的事件名称默认为'message'
            event.event = event.event or "message"

            # 分发事件
            self._logger.debug("Dispatching %s...", event)
            yield event

```