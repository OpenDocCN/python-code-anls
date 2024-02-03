# `.\AutoGPT\autogpts\autogpt\tests\integration\memory\_test_json_file_memory.py`

```py
# 导入所需的模块
import orjson
import pytest

# 导入自定义模块
from autogpt.config import Config
from autogpt.file_workspace import FileWorkspace
from autogpt.memory.vector import JSONFileMemory, MemoryItem

# 测试 JSONFileMemory 类的初始化，不使用后备文件
def test_json_memory_init_without_backing_file(
    config: Config, workspace: FileWorkspace
):
    # 获取内存索引文件路径
    index_file = workspace.root / f"{config.memory_index}.json"

    # 断言索引文件不存在
    assert not index_file.exists()
    # 初始化 JSONFileMemory 对象
    JSONFileMemory(config)
    # 断言索引文件存在
    assert index_file.exists()
    # 断言索引文件内容为 "[]"
    assert index_file.read_text() == "[]"

# 测试 JSONFileMemory 类的初始化，使用空的后备文件
def test_json_memory_init_with_backing_empty_file(
    config: Config, workspace: FileWorkspace
):
    # 获取内存索引文件路径
    index_file = workspace.root / f"{config.memory_index}.json"
    # 创建空的索引文件
    index_file.touch()

    # 断言索引文件存在
    assert index_file.exists()
    # 初始化 JSONFileMemory 对象
    JSONFileMemory(config)
    # 断言索引文件存在
    assert index_file.exists()
    # 断言索引文件内容为 "[]"
    assert index_file.read_text() == "[]"

# 测试 JSONFileMemory 类的初始化，使用无效的后备文件
def test_json_memory_init_with_backing_invalid_file(
    config: Config, workspace: FileWorkspace
):
    # 获取内存索引文件路径
    index_file = workspace.root / f"{config.memory_index}.json"
    # 创建空的索引文件
    index_file.touch()

    # 准备数据
    raw_data = {"texts": ["test"]}
    data = orjson.dumps(raw_data, option=JSONFileMemory.SAVE_OPTIONS)
    # 将数据写入索引文件
    with index_file.open("wb") as f:
        f.write(data)

    # 断言索引文件存在
    assert index_file.exists()
    # 初始化 JSONFileMemory 对象
    JSONFileMemory(config)
    # 断言索引文件存在
    assert index_file.exists()
    # 断言索引文件内容为 "[]"
    assert index_file.read_text() == "[]"

# 测试 JSONFileMemory 类的添加功能
def test_json_memory_add(config: Config, memory_item: MemoryItem):
    # 初始化 JSONFileMemory 对象
    index = JSONFileMemory(config)
    # 添加内存项
    index.add(memory_item)
    # 断言内存中的第一个项为 memory_item
    assert index.memories[0] == memory_item

# 测试 JSONFileMemory 类的清空功能
def test_json_memory_clear(config: Config, memory_item: MemoryItem):
    # 初始化 JSONFileMemory 对象
    index = JSONFileMemory(config)
    # 断言内存为空
    assert index.memories == []

    # 添加内存项
    index.add(memory_item)
    # 断言内存中的第一个项为 memory_item，如果 add() 失败则无法测试 clear()
    assert index.memories[0] == memory_item, "Cannot test clear() because add() fails"

    # 清空内存
    index.clear()
    # 断言内存为空
    assert index.memories == []

# 测试 JSONFileMemory 类的获取功能
def test_json_memory_get(config: Config, memory_item: MemoryItem, mock_get_embedding):
    # 初始化 JSONFileMemory 对象
    index = JSONFileMemory(config)
    # 断言检查：确保在初始索引不为空的情况下无法测试 get() 方法
    assert (
        index.get("test", config) is None
    ), "Cannot test get() because initial index is not empty"

    # 向索引中添加内存项
    index.add(memory_item)
    # 从索引中获取指定键的值
    retrieved = index.get("test", config)
    # 断言检查：确保获取的值不为空
    assert retrieved is not None
    # 断言检查：确保获取的内存项与添加的内存项相同
    assert retrieved.memory_item == memory_item
# 测试从 JSON 文件中加载索引的函数
def test_json_memory_load_index(config: Config, memory_item: MemoryItem):
    # 创建一个 JSON 文件内存对象
    index = JSONFileMemory(config)
    # 向索引中添加内存项
    index.add(memory_item)

    try:
        # 检查索引文件是否存在
        assert index.file_path.exists(), "index was not saved to file"
        # 检查索引中是否只包含一个项
        assert len(index) == 1, f"index contains {len(index)} items instead of 1"
        # 检查索引中的第一个项是否与添加的项相同
        assert index.memories[0] == memory_item, "item in index != added mock item"
    except AssertionError as e:
        raise ValueError(f"Setting up for load_index test failed: {e}")

    # 清空索引中的内存项列表
    index.memories = []
    # 加载索引
    index.load_index()

    # 检查索引中是否只包含一个项
    assert len(index) == 1
    # 检查索引中的第一个项是否与添加的项相同
    assert index.memories[0] == memory_item


# 测试从 JSON 文件中获取相关项的函数
@pytest.mark.vcr
@pytest.mark.requires_openai_api_key
def test_json_memory_get_relevant(config: Config, cached_openai_client: None) -> None:
    # 创建一个 JSON 文件内存对象
    index = JSONFileMemory(config)
    # 创建多个内存项
    mem1 = MemoryItem.from_text_file("Sample text", "sample.txt", config)
    mem2 = MemoryItem.from_text_file(
        "Grocery list:\n- Pancake mix", "groceries.txt", config
    )
    mem3 = MemoryItem.from_text_file(
        "What is your favorite color?", "color.txt", config
    )
    lipsum = "Lorem ipsum dolor sit amet"
    mem4 = MemoryItem.from_text_file(" ".join([lipsum] * 100), "lipsum.txt", config)
    # 向索引中添加内存项
    index.add(mem1)
    index.add(mem2)
    index.add(mem3)
    index.add(mem4)

    # 检查获取相关项函数是否返回正确的内存项
    assert index.get_relevant(mem1.raw_content, 1, config)[0].memory_item == mem1
    assert index.get_relevant(mem2.raw_content, 1, config)[0].memory_item == mem2
    assert index.get_relevant(mem3.raw_content, 1, config)[0].memory_item == mem3
    assert [mr.memory_item for mr in index.get_relevant(lipsum, 2, config)] == [
        mem4,
        mem1,
    ]


# 测试获取 JSON 文件内存对象统计信息的函数
def test_json_memory_get_stats(config: Config, memory_item: MemoryItem) -> None:
    # 创建一个 JSON 文件内存对象
    index = JSONFileMemory(config)
    # 向索引中添加内存项
    index.add(memory_item)
    # 获取索引的统计信息
    n_memories, n_chunks = index.get_stats()
    # 检查内存项数量是否为 1
    assert n_memories == 1
    # 检查块数量是否为 1
    assert n_chunks == 1
```