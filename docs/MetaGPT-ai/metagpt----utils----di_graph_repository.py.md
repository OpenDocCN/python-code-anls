# `MetaGPT\metagpt\utils\di_graph_repository.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19
@Author  : mashenquan
@File    : di_graph_repository.py
@Desc    : Graph repository based on DiGraph
"""
# 导入必要的模块
from __future__ import annotations
import json
from pathlib import Path
from typing import List
import networkx
from metagpt.utils.common import aread, awrite
from metagpt.utils.graph_repository import SPO, GraphRepository

# 创建一个基于有向图的图形仓库类
class DiGraphRepository(GraphRepository):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self._repo = networkx.DiGraph()

    # 插入一条有向边
    async def insert(self, subject: str, predicate: str, object_: str):
        self._repo.add_edge(subject, object_, predicate=predicate)

    # 更新或插入一条有向边
    async def upsert(self, subject: str, predicate: str, object_: str):
        pass

    # 更新一条有向边
    async def update(self, subject: str, predicate: str, object_: str):
        pass

    # 查询符合条件的有向边
    async def select(self, subject: str = None, predicate: str = None, object_: str = None) -> List[SPO]:
        result = []
        for s, o, p in self._repo.edges(data="predicate"):
            if subject and subject != s:
                continue
            if predicate and predicate != p:
                continue
            if object_ and object_ != o:
                continue
            result.append(SPO(subject=s, predicate=p, object_=o))
        return result

    # 将图形数据转换为 JSON 格式
    def json(self) -> str:
        m = networkx.node_link_data(self._repo)
        data = json.dumps(m)
        return data

    # 保存图形数据到指定路径
    async def save(self, path: str | Path = None):
        data = self.json()
        path = path or self._kwargs.get("root")
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        pathname = Path(path) / self.name
        await awrite(filename=pathname.with_suffix(".json"), data=data, encoding="utf-8")

    # 从指定路径加载图形数据
    async def load(self, pathname: str | Path):
        data = await aread(filename=pathname, encoding="utf-8")
        m = json.loads(data)
        self._repo = networkx.node_link_graph(m)

    # 从指定路径加载图形数据并创建图形仓库对象
    @staticmethod
    async def load_from(pathname: str | Path) -> GraphRepository:
        pathname = Path(pathname)
        name = pathname.with_suffix("").name
        root = pathname.parent
        graph = DiGraphRepository(name=name, root=root)
        if pathname.exists():
            await graph.load(pathname=pathname)
        return graph

    # 返回图形仓库的根路径
    @property
    def root(self) -> str:
        return self._kwargs.get("root")

    # 返回图形仓库的文件路径
    @property
    def pathname(self) -> Path:
        p = Path(self.root) / self.name
        return p.with_suffix(".json")

```