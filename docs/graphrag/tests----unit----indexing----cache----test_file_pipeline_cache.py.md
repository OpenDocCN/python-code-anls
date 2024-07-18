# `.\graphrag\tests\unit\indexing\cache\test_file_pipeline_cache.py`

```py
# 导入必要的库和模块
import asyncio  # 异步编程库
import os  # 系统操作库
import unittest  # 单元测试库

# 从特定路径导入模块和类
from graphrag.index.cache import (
    JsonPipelineCache,  # 导入 JsonPipelineCache 类
)
from graphrag.index.storage.file_pipeline_storage import (
    FilePipelineStorage,  # 导入 FilePipelineStorage 类
)

# 设置临时目录路径
TEMP_DIR = "./.tmp"


# 创建缓存对象的函数
def create_cache():
    # 创建文件管道存储对象，指定当前工作目录下的临时目录路径
    storage = FilePipelineStorage(os.path.join(os.getcwd(), ".tmp"))
    # 返回一个基于文件管道存储的 JsonPipelineCache 对象
    return JsonPipelineCache(storage)


# 定义一个测试类，继承自 IsolatedAsyncioTestCase 类
class TestFilePipelineCache(unittest.IsolatedAsyncioTestCase):

    # 在每个测试方法运行之前执行的设置方法
    def setUp(self):
        # 创建并初始化缓存对象
        self.cache = create_cache()

    # 在每个测试方法运行之后执行的清理方法
    def tearDown(self):
        # 异步运行缓存对象的清理操作
        asyncio.run(self.cache.clear())

    # 测试清空缓存的异步方法
    async def test_cache_clear(self):
        # 如果临时目录不存在，则创建它
        if not os.path.exists(TEMP_DIR):
            os.mkdir(TEMP_DIR)
        
        # 创建测试文件 test1 和 test2
        with open(f"{TEMP_DIR}/test1", "w") as f:
            f.write("This is test1 file.")
        with open(f"{TEMP_DIR}/test2", "w") as f:
            f.write("This is test2 file.")

        # 调用缓存对象的清空方法
        await self.cache.clear()

        # 检查临时目录是否为空
        files = os.listdir(TEMP_DIR)
        assert len(files) == 0

    # 测试子缓存功能的异步方法
    async def test_child_cache(self):
        # 设置键为 "test1" 的缓存项
        await self.cache.set("test1", "test1")
        assert os.path.exists(f"{TEMP_DIR}/test1")

        # 创建名为 "test" 的子缓存对象
        child = self.cache.child("test")
        assert os.path.exists(f"{TEMP_DIR}/test")

        # 在子缓存对象中设置键为 "test2" 的缓存项
        await child.set("test2", "test2")
        assert os.path.exists(f"{TEMP_DIR}/test/test2")

        # 设置键为 "test1" 的缓存项，然后删除它
        await self.cache.set("test1", "test1")
        await self.cache.delete("test1")
        assert not os.path.exists(f"{TEMP_DIR}/test1")

    # 测试缓存项存在性检查的异步方法
    async def test_cache_has(self):
        # 设置键为 "test1" 的缓存项内容
        test1 = "this is a test file"
        await self.cache.set("test1", test1)

        # 检查缓存中是否存在键为 "test1" 的缓存项
        assert await self.cache.has("test1")
        # 检查缓存中是否存在键为 "NON_EXISTENT" 的缓存项
        assert not await self.cache.has("NON_EXISTENT")
        # 获取键为 "NON_EXISTENT" 的缓存项，应返回 None
        assert await self.cache.get("NON_EXISTENT") is None

    # 测试设置和获取缓存项的异步方法
    async def test_get_set(self):
        # 定义测试内容
        test1 = "this is a test file"
        test2 = "\\n test"
        test3 = "\\\\\\"

        # 设置键为 "test1", "test2", "test3" 的缓存项内容
        await self.cache.set("test1", test1)
        await self.cache.set("test2", test2)
        await self.cache.set("test3", test3)

        # 获取并断言键为 "test1", "test2", "test3" 的缓存项内容是否与预期一致
        assert await self.cache.get("test1") == test1
        assert await self.cache.get("test2") == test2
        assert await self.cache.get("test3") == test3
```