# `MetaGPT\metagpt\actions\rebuild_sequence_view.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4
@Author  : mashenquan
@File    : rebuild_sequence_view.py
@Desc    : Rebuild sequence view info
"""
# 导入必要的模块
from __future__ import annotations
from pathlib import Path
from typing import List
# 导入自定义模块
from metagpt.actions import Action
from metagpt.config import CONFIG
from metagpt.const import GRAPH_REPO_FILE_REPO
from metagpt.logs import logger
from metagpt.utils.common import aread, list_files
from metagpt.utils.di_graph_repository import DiGraphRepository
from metagpt.utils.graph_repository import GraphKeyword

# 定义一个类，继承自Action类
class RebuildSequenceView(Action):
    # 异步方法，用于运行重建序列视图
    async def run(self, with_messages=None, format=CONFIG.prompt_schema):
        # 获取图数据库路径
        graph_repo_pathname = CONFIG.git_repo.workdir / GRAPH_REPO_FILE_REPO / CONFIG.git_repo.workdir.name
        # 从路径加载图数据库
        graph_db = await DiGraphRepository.load_from(str(graph_repo_pathname.with_suffix(".json")))
        # 搜索主要条目
        entries = await RebuildSequenceView._search_main_entry(graph_db)
        # 遍历条目并重建序列视图
        for entry in entries:
            await self._rebuild_sequence_view(entry, graph_db)
        # 保存图数据库
        await graph_db.save()

    # 静态方法，用于搜索主要条目
    @staticmethod
    async def _search_main_entry(graph_db) -> List:
        # 选择具有页面信息的行
        rows = await graph_db.select(predicate=GraphKeyword.HAS_PAGE_INFO)
        tag = "__name__:__main__"
        entries = []
        # 遍历行，找到包含特定标签的条目
        for r in rows:
            if tag in r.subject or tag in r.object_:
                entries.append(r)
        return entries

    # 异步方法，用于重建序列视图
    async def _rebuild_sequence_view(self, entry, graph_db):
        # 获取文件名
        filename = entry.subject.split(":", 1)[0]
        # 获取源文件名
        src_filename = RebuildSequenceView._get_full_filename(root=self.context, pathname=filename)
        # 读取文件内容
        content = await aread(filename=src_filename, encoding="utf-8")
        # 将文件内容转换为Mermaid序列图
        content = f"```python\n{content}\n```\n\n---\nTranslate the code above into Mermaid Sequence Diagram."
        # 通过交互式对话框获取数据
        data = await self.llm.aask(
            msg=content, system_msgs=["You are a python code to Mermaid Sequence Diagram translator in function detail"]
        )
        # 将数据插入图数据库
        await graph_db.insert(subject=filename, predicate=GraphKeyword.HAS_SEQUENCE_VIEW, object_=data)
        logger.info(data)

    # 静态方法，用于获取完整文件名
    @staticmethod
    def _get_full_filename(root: str | Path, pathname: str | Path) -> Path | None:
        files = list_files(root=root)
        postfix = "/" + str(pathname)
        # 遍历文件列表，找到匹配的文件名
        for i in files:
            if str(i).endswith(postfix):
                return i
        return None

```