# `.\graphrag\tests\unit\indexing\storage\test_blob_pipeline_storage.py`

```py
# 版权声明和许可证信息
# 此文件用于 Blob 存储测试
"""Blob Storage Tests."""

# 导入正则表达式模块
import re

# 导入自定义的 BlobPipelineStorage 类
from graphrag.index.storage.blob_pipeline_storage import BlobPipelineStorage

# 定义预设的 Blob 存储连接字符串常量
# cspell:disable-next-line well-known-key
WELL_KNOWN_BLOB_STORAGE_KEY = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"

# 异步测试函数：查找文件
async def test_find():
    # 创建 BlobPipelineStorage 对象，连接到指定的 Blob 存储和容器
    storage = BlobPipelineStorage(
        connection_string=WELL_KNOWN_BLOB_STORAGE_KEY,
        container_name="testfind",
    )
    try:
        try:
            # 查找指定目录下以 .txt 结尾的文件列表
            items = list(
                storage.find(base_dir="input", file_pattern=re.compile(r".*\.txt$"))
            )
            # 获取文件列表中的文件名
            items = [item[0] for item in items]
            # 断言文件列表为空
            assert items == []

            # 在 Blob 存储中设置一个新文件
            await storage.set(
                "input/christmas.txt", "Merry Christmas!", encoding="utf-8"
            )
            # 再次查找 .txt 结尾的文件列表
            items = list(
                storage.find(base_dir="input", file_pattern=re.compile(r".*\.txt$"))
            )
            # 获取文件列表中的文件名
            items = [item[0] for item in items]
            # 断言文件列表中包含指定文件
            assert items == ["input/christmas.txt"]

            # 在根目录下设置一个新文件
            await storage.set("test.txt", "Hello, World!", encoding="utf-8")
            # 再次查找 .txt 结尾的文件列表
            items = list(storage.find(file_pattern=re.compile(r".*\.txt$")))
            # 获取文件列表中的文件名
            items = [item[0] for item in items]
            # 断言文件列表中包含两个指定文件
            assert items == ["input/christmas.txt", "test.txt"]

            # 获取指定文件的内容
            output = await storage.get("test.txt")
            # 断言获取的内容正确
            assert output == "Hello, World!"
        finally:
            # 在测试完成后删除测试文件
            await storage.delete("test.txt")
            # 再次获取已删除文件的内容，断言为空
            output = await storage.get("test.txt")
            assert output is None
    finally:
        # 最终删除整个容器
        storage.delete_container()


# 异步测试函数：使用路径前缀为“.”的情况
async def test_dotprefix():
    # 创建 BlobPipelineStorage 对象，连接到指定的 Blob 存储和容器，并设置路径前缀为“.”
    storage = BlobPipelineStorage(
        connection_string=WELL_KNOWN_BLOB_STORAGE_KEY,
        container_name="testfind",
        path_prefix=".",
    )
    try:
        # 在指定路径下设置一个新文件
        await storage.set("input/christmas.txt", "Merry Christmas!", encoding="utf-8")
        # 查找以 .txt 结尾的文件列表
        items = list(storage.find(file_pattern=re.compile(r".*\.txt$")))
        # 获取文件列表中的文件名
        items = [item[0] for item in items]
        # 断言文件列表中包含指定文件
        assert items == ["input/christmas.txt"]
    finally:
        # 最终删除整个容器
        storage.delete_container()


# 异步测试函数：测试 BlobPipelineStorage 的子容器功能
async def test_child():
    # 创建 BlobPipelineStorage 对象，连接到指定的 Blob 存储和子容器
    parent = BlobPipelineStorage(
        connection_string=WELL_KNOWN_BLOB_STORAGE_KEY,
        container_name="testchild",
    )
    try:
        try:
            # 获取名为 "input" 的子容器对象
            storage = parent.child("input")
            # 在 "input" 容器中设置名为 "christmas.txt" 的文件，内容为 "Merry Christmas!"
            await storage.set("christmas.txt", "Merry Christmas!", encoding="utf-8")
            # 查找所有以 .txt 结尾的文件，并将结果存储在列表中
            items = list(storage.find(re.compile(r".*\.txt$")))
            # 提取列表中每个项目的第一个元素（文件名）
            items = [item[0] for item in items]
            # 断言查找到的文件名与预期的相符
            assert items == ["christmas.txt"]

            # 在 "input" 容器中设置名为 "test.txt" 的文件，内容为 "Hello, World!"
            await storage.set("test.txt", "Hello, World!", encoding="utf-8")
            # 再次查找所有以 .txt 结尾的文件，并将结果存储在列表中
            items = list(storage.find(re.compile(r".*\.txt$")))
            # 提取列表中每个项目的第一个元素（文件名）
            items = [item[0] for item in items]
            # 打印查找到的所有文件名
            print("FOUND", items)
            # 断言查找到的文件名与预期的相符
            assert items == ["christmas.txt", "test.txt"]

            # 从 "input" 容器中获取名为 "test.txt" 的文件内容
            output = await storage.get("test.txt")
            # 断言获取的文件内容与预期的相符
            assert output == "Hello, World!"

            # 在父容器中查找所有以 .txt 结尾的文件，并将结果存储在列表中
            items = list(parent.find(re.compile(r".*\.txt$")))
            # 提取列表中每个项目的第一个元素（文件路径）
            items = [item[0] for item in items]
            # 打印查找到的所有文件路径
            print("FOUND ITEMS", items)
            # 断言查找到的文件路径与预期的相符
            assert items == ["input/christmas.txt", "input/test.txt"]
        finally:
            # 最终操作：删除 "input" 容器中的 "test.txt" 文件
            await parent.delete("input/test.txt")
            # 检查 "input" 容器中是否仍然存在 "test.txt" 文件
            has_test = await parent.has("input/test.txt")
            # 断言 "input" 容器中不再存在 "test.txt" 文件
            assert not has_test
    finally:
        # 最终操作：删除父容器
        parent.delete_container()
```