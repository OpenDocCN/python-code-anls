# `MetaGPT\tests\metagpt\utils\test_di_graph_repository.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19
@Author  : mashenquan
@File    : test_di_graph_repository.py
@Desc    : Unit tests for di_graph_repository.py
"""

from pathlib import Path  # 导入 Path 模块

import pytest  # 导入 pytest 模块
from pydantic import BaseModel  # 导入 pydantic 模块中的 BaseModel 类

from metagpt.const import DEFAULT_WORKSPACE_ROOT  # 从 metagpt.const 模块中导入 DEFAULT_WORKSPACE_ROOT 常量
from metagpt.repo_parser import RepoParser  # 从 metagpt.repo_parser 模块中导入 RepoParser 类
from metagpt.utils.di_graph_repository import DiGraphRepository  # 从 metagpt.utils.di_graph_repository 模块中导入 DiGraphRepository 类
from metagpt.utils.graph_repository import GraphRepository  # 从 metagpt.utils.graph_repository 模块中导入 GraphRepository 类

# 标记为异步测试
@pytest.mark.asyncio
async def test_di_graph_repository():
    class Input(BaseModel):  # 定义一个名为 Input 的子类，继承自 BaseModel
        s: str  # 字符串类型的属性 s
        p: str  # 字符串类型的属性 p
        o: str  # 字符串类型的属性 o

    inputs = [  # 定义一个列表 inputs
        {"s": "main.py:Game:draw", "p": "method:hasDescription", "o": "Draw image"},  # 列表中的字典元素
        {"s": "main.py:Game:draw", "p": "method:hasDescription", "o": "Show image"},
    ]
    path = Path(__file__).parent  # 获取当前文件的父目录路径
    graph = DiGraphRepository(name="test", root=path)  # 创建一个名为 graph 的 DiGraphRepository 对象
    for i in inputs:  # 遍历 inputs 列表
        data = Input(**i)  # 使用字典 i 创建 Input 类的实例
        await graph.insert(subject=data.s, predicate=data.p, object_=data.o)  # 调用 graph 对象的 insert 方法
        v = graph.json()  # 获取 graph 对象的 JSON 格式数据
        assert v  # 断言 v 的值为真
    await graph.save()  # 调用 graph 对象的 save 方法
    assert graph.pathname.exists()  # 断言 graph 对象的路径存在
    graph.pathname.unlink()  # 删除 graph 对象的路径


# 标记为异步测试
@pytest.mark.asyncio
async def test_js_parser():
    class Input(BaseModel):  # 定义一个名为 Input 的子类，继承自 BaseModel
        path: str  # 字符串类型的属性 path

    inputs = [  # 定义一个列表 inputs
        {"path": str(Path(__file__).parent / "../../data/code")},  # 列表中的字典元素
    ]
    path = Path(__file__).parent  # 获取当前文件的父目录路径
    graph = DiGraphRepository(name="test", root=path)  # 创建一个名为 graph 的 DiGraphRepository 对象
    for i in inputs:  # 遍历 inputs 列表
        data = Input(**i)  # 使用字典 i 创建 Input 类的实例
        repo_parser = RepoParser(base_directory=data.path)  # 创建一个名为 repo_parser 的 RepoParser 对象
        symbols = repo_parser.generate_symbols()  # 调用 repo_parser 对象的 generate_symbols 方法
        for s in symbols:  # 遍历 symbols 列表
            await GraphRepository.update_graph_db_with_file_info(graph_db=graph, file_info=s)  # 调用 GraphRepository 的 update_graph_db_with_file_info 方法
    data = graph.json()  # 获取 graph 对象的 JSON 格式数据
    assert data  # 断言 data 的值为真


# 标记为异步测试
@pytest.mark.asyncio
async def test_codes():
    path = DEFAULT_WORKSPACE_ROOT / "snake_game"  # 定义一个路径
    repo_parser = RepoParser(base_directory=path)  # 创建一个名为 repo_parser 的 RepoParser 对象

    graph = DiGraphRepository(name="test", root=path)  # 创建一个名为 graph 的 DiGraphRepository 对象
    symbols = repo_parser.generate_symbols()  # 调用 repo_parser 对象的 generate_symbols 方法
    for file_info in symbols:  # 遍历 symbols 列表
        for code_block in file_info.page_info:  # 遍历 file_info.page_info 列表
            try:
                val = code_block.model_dump_json()  # 调用 code_block 对象的 model_dump_json 方法
                assert val  # 断言 val 的值为真
            except TypeError as e:  # 捕获 TypeError 异常
                assert not e  # 断言异常 e 的值为假
        await GraphRepository.update_graph_db_with_file_info(graph_db=graph, file_info=file_info)  # 调用 GraphRepository 的 update_graph_db_with_file_info 方法
    data = graph.json()  # 获取 graph 对象的 JSON 格式数据
    assert data  # 断言 data 的值为真
    print(data)  # 打印 data


if __name__ == "__main__":
    pytest.main([__file__, "-s"])  # 执行测试并输出结果

```