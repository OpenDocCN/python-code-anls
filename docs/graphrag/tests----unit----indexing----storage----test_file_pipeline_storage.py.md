# `.\graphrag\tests\unit\indexing\storage\test_file_pipeline_storage.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Blob Storage Tests."""

# 导入必要的库和模块
import os
import re
from pathlib import Path

# 导入自定义的文件存储模块
from graphrag.index.storage.file_pipeline_storage import FilePipelineStorage

# 获取当前文件的目录路径
__dirname__ = os.path.dirname(__file__)


# 异步测试函数：测试查找功能
async def test_find():
    # 创建 FilePipelineStorage 对象
    storage = FilePipelineStorage()
    
    # 调用存储对象的 find 方法，查找指定目录下的所有符合条件的文件
    items = list(
        storage.find(
            base_dir="tests/fixtures/text",  # 指定基础目录
            file_pattern=re.compile(r".*\.txt$"),  # 文件名匹配模式
            progress=None,  # 进度条参数（此处为 None）
            file_filter=None,  # 文件过滤器（此处为 None）
        )
    )
    
    # 断言查找结果是否符合预期
    assert items == [(str(Path("tests/fixtures/text/input/dulce.txt")), {})]
    
    # 获取指定文件的内容
    output = await storage.get("tests/fixtures/text/input/dulce.txt")
    assert len(output) > 0  # 断言内容长度大于 0
    
    # 设置一个新的文件，并验证内容是否正确设置
    await storage.set("test.txt", "Hello, World!", encoding="utf-8")
    output = await storage.get("test.txt")
    assert output == "Hello, World!"  # 断言内容正确
    
    # 删除设置的测试文件，并验证是否删除成功
    await storage.delete("test.txt")
    output = await storage.get("test.txt")
    assert output is None  # 断言文件已删除


# 异步测试函数：测试子目录功能
async def test_child():
    # 创建 FilePipelineStorage 对象
    storage = FilePipelineStorage()
    
    # 切换到指定的子目录，并进行文件查找操作
    storage = storage.child("tests/fixtures/text")
    items = list(storage.find(re.compile(r".*\.txt$")))
    
    # 断言子目录查找结果是否符合预期
    assert items == [(str(Path("input/dulce.txt")), {})]
    
    # 获取子目录中指定文件的内容
    output = await storage.get("input/dulce.txt")
    assert len(output) > 0  # 断言内容长度大于 0
    
    # 设置一个新的文件，并验证内容是否正确设置
    await storage.set("test.txt", "Hello, World!", encoding="utf-8")
    output = await storage.get("test.txt")
    assert output == "Hello, World!"  # 断言内容正确
    
    # 删除设置的测试文件，并验证是否删除成功
    await storage.delete("test.txt")
    output = await storage.get("test.txt")
    assert output is None  # 断言文件已删除
```