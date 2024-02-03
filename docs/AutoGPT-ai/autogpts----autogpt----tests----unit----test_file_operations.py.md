# `.\AutoGPT\autogpts\autogpt\tests\unit\test_file_operations.py`

```py
# 导入所需的模块
import hashlib
import os
import re
from io import TextIOWrapper
from pathlib import Path

# 导入 pytest 相关模块
import pytest
from pytest_mock import MockerFixture

# 导入自定义模块
import autogpt.commands.file_operations as file_ops
from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import DuplicateOperationError
from autogpt.config import Config
from autogpt.file_workspace import FileWorkspace
from autogpt.memory.vector.memory_item import MemoryItem
from autogpt.memory.vector.utils import Embedding

# 定义 fixture，返回测试文件内容
@pytest.fixture()
def file_content():
    return "This is a test file.\n"

# 定义 fixture，模拟 MemoryItem 的 from_text 方法
@pytest.fixture()
def mock_MemoryItem_from_text(
    mocker: MockerFixture, mock_embedding: Embedding, config: Config
):
    mocker.patch.object(
        file_ops.MemoryItemFactory,
        "from_text",
        new=lambda content, source_type, config, metadata: MemoryItem(
            raw_content=content,
            summary=f"Summary of content '{content}'",
            chunk_summaries=[f"Summary of content '{content}'"],
            chunks=[content],
            e_summary=mock_embedding,
            e_chunks=[mock_embedding],
            metadata=metadata | {"source_type": source_type},
        ),
    )

# 定义 fixture，返回测试文件名
@pytest.fixture()
def test_file_name():
    return Path("test_file.txt")

# 定义 fixture，返回测试文件路径
@pytest.fixture
def test_file_path(test_file_name: Path, workspace: FileWorkspace):
    return workspace.get_path(test_file_name)

# 定义 fixture，创建测试文件
@pytest.fixture()
def test_file(test_file_path: Path):
    file = open(test_file_path, "w")
    yield file
    if not file.closed:
        file.close()

# 定义 fixture，创建带内容的测试文件路径
@pytest.fixture()
def test_file_with_content_path(test_file: TextIOWrapper, file_content, agent: Agent):
    test_file.write(file_content)
    test_file.close()
    # 记录文件操作日志
    file_ops.log_operation(
        "write", Path(test_file.name), agent, file_ops.text_checksum(file_content)
    )
    return Path(test_file.name)

# 定义 fixture，返回测试目录路径
@pytest.fixture()
def test_directory(workspace: FileWorkspace):
    return workspace.get_path("test_directory")

# 定义 fixture
@pytest.fixture()
# 定义一个测试函数，用于获取嵌套文件的路径
def test_nested_file(workspace: FileWorkspace):
    return workspace.get_path("nested/test_file.txt")


# 定义一个测试函数，用于测试文件操作日志
def test_file_operations_log(test_file: TextIOWrapper):
    # 准备日志文件内容
    log_file_content = (
        "File Operation Logger\n"
        "write: path/to/file1.txt #checksum1\n"
        "write: path/to/file2.txt #checksum2\n"
        "write: path/to/file3.txt #checksum3\n"
        "append: path/to/file2.txt #checksum4\n"
        "delete: path/to/file3.txt\n"
    )
    # 将日志内容写入测试文件
    test_file.write(log_file_content)
    # 关闭测试文件
    test_file.close()

    # 预期的操作列表
    expected = [
        ("write", "path/to/file1.txt", "checksum1"),
        ("write", "path/to/file2.txt", "checksum2"),
        ("write", "path/to/file3.txt", "checksum3"),
        ("append", "path/to/file2.txt", "checksum4"),
        ("delete", "path/to/file3.txt", None),
    ]
    # 断言从日志文件中解析出的操作列表与预期的操作列表相同
    assert list(file_ops.operations_from_log(test_file.name)) == expected


# 定义一个测试函数，用于测试文件操作状态
def test_file_operations_state(test_file: TextIOWrapper):
    # 准备一个假的日志文件
    log_file_content = (
        "File Operation Logger\n"
        "write: path/to/file1.txt #checksum1\n"
        "write: path/to/file2.txt #checksum2\n"
        "write: path/to/file3.txt #checksum3\n"
        "append: path/to/file2.txt #checksum4\n"
        "delete: path/to/file3.txt\n"
    )
    # 将日志内容写入测试文件
    test_file.write(log_file_content)
    # 关闭测试文件
    test_file.close()

    # 调用函数并检查返回的字典
    expected_state = {
        "path/to/file1.txt": "checksum1",
        "path/to/file2.txt": "checksum4",
    }
    # 断言文件操作状态与预期状态相同
    assert file_ops.file_operations_state(test_file.name) == expected_state


# 定义一个测试函数，用于测试是否为重复操作
def test_is_duplicate_operation(agent: Agent, mocker: MockerFixture):
    # 准备一个假的状态字典供函数使用
    state = {
        "path/to/file1.txt": "checksum1",
        "path/to/file2.txt": "checksum2",
    }
    # 模拟 file_operations_state 函数返回假的状态字典
    mocker.patch.object(file_ops, "file_operations_state", lambda _: state)

    # 测试写操作的情况
    # 断言检查是否为写操作，文件路径为"path/to/file1.txt"，代理为agent，校验和为"checksum1"，返回True
    assert (
        file_ops.is_duplicate_operation(
            "write", Path("path/to/file1.txt"), agent, "checksum1"
        )
        is True
    )
    # 断言检查是否为写操作，文件路径为"path/to/file1.txt"，代理为agent，校验和为"checksum2"，返回False
    assert (
        file_ops.is_duplicate_operation(
            "write", Path("path/to/file1.txt"), agent, "checksum2"
        )
        is False
    )
    # 断言检查是否为写操作，文件路径为"path/to/file3.txt"，代理为agent，校验和为"checksum3"，返回False
    assert (
        file_ops.is_duplicate_operation(
            "write", Path("path/to/file3.txt"), agent, "checksum3"
        )
        is False
    )
    # 测试追加操作的情况
    assert (
        file_ops.is_duplicate_operation(
            "append", Path("path/to/file1.txt"), agent, "checksum1"
        )
        is False
    )
    # 测试删除操作的情况
    assert (
        file_ops.is_duplicate_operation("delete", Path("path/to/file1.txt"), agent)
        is False
    )
    # 断言检查是否为删除操作，文件路径为"path/to/file3.txt"，代理为agent，返回True
    assert (
        file_ops.is_duplicate_operation("delete", Path("path/to/file3.txt"), agent)
        is True
    )
# 测试记录文件操作
def test_log_operation(agent: Agent):
    # 记录文件操作到日志中
    file_ops.log_operation("log_test", Path("path/to/test"), agent=agent)
    # 读取文件操作日志内容
    with open(agent.file_manager.file_ops_log_path, "r", encoding="utf-8") as f:
        content = f.read()
    # 断言日志中包含特定内容
    assert "log_test: path/to/test\n" in content


# 测试文本校验和
def test_text_checksum(file_content: str):
    # 计算文件内容的校验和
    checksum = file_ops.text_checksum(file_content)
    # 计算不同内容的校验和
    different_checksum = file_ops.text_checksum("other content")
    # 断言校验和为十六进制字符串
    assert re.match(r"^[a-fA-F0-9]+$", checksum) is not None
    # 断言两个不同内容的校验和不相同
    assert checksum != different_checksum


# 测试记录文件操作并包含校验和
def test_log_operation_with_checksum(agent: Agent):
    # 记录文件操作到日志中，包含校验和
    file_ops.log_operation(
        "log_test", Path("path/to/test"), agent=agent, checksum="ABCDEF"
    )
    # 读取文件操作日志内容
    with open(agent.file_manager.file_ops_log_path, "r", encoding="utf-8") as f:
        content = f.read()
    # 断言日志中包含特定内容和校验和
    assert "log_test: path/to/test #ABCDEF\n" in content


# 测试读取文件内容
def test_read_file(
    mock_MemoryItem_from_text,
    test_file_with_content_path: Path,
    file_content,
    agent: Agent,
):
    # 读取文件内容
    content = file_ops.read_file(test_file_with_content_path, agent=agent)
    # 断言文件内容与预期内容相同
    assert content.replace("\r", "") == file_content


# 测试读取不存在的文件
def test_read_file_not_found(agent: Agent):
    # 指定不存在的文件名
    filename = "does_not_exist.txt"
    # 断言读取不存在的文件会引发 FileNotFoundError 异常
    with pytest.raises(FileNotFoundError):
        file_ops.read_file(filename, agent=agent)


# 异步测试写入文件（相对路径）
@pytest.mark.asyncio
async def test_write_to_file_relative_path(test_file_name: Path, agent: Agent):
    # 新内容
    new_content = "This is new content.\n"
    # 异步写入文件
    await file_ops.write_to_file(test_file_name, new_content, agent=agent)
    # 读取写入的文件内容
    with open(agent.workspace.get_path(test_file_name), "r", encoding="utf-8") as f:
        content = f.read()
    # 断言文件内容与新内容相同
    assert content == new_content


# 异步测试写入文件（绝对路径）
@pytest.mark.asyncio
async def test_write_to_file_absolute_path(test_file_path: Path, agent: Agent):
    # 新内容
    new_content = "This is new content.\n"
    # 异步写入文件
    await file_ops.write_to_file(test_file_path, new_content, agent=agent)
    # 读取写入的文件内容
    with open(test_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # 使用断言来检查变量 content 是否等于变量 new_content，如果不相等则会触发 AssertionError
    assert content == new_content
# 使用 pytest.mark.asyncio 标记异步测试函数
@pytest.mark.asyncio
async def test_write_file_logs_checksum(test_file_name: Path, agent: Agent):
    # 定义新内容和新内容的校验和
    new_content = "This is new content.\n"
    new_checksum = file_ops.text_checksum(new_content)
    # 异步写入新内容到文件，并记录操作日志
    await file_ops.write_to_file(test_file_name, new_content, agent=agent)
    # 打开文件操作日志文件，读取其中的内容
    with open(agent.file_manager.file_ops_log_path, "r", encoding="utf-8") as f:
        log_entry = f.read()
    # 断言操作日志中是否包含正确的写入信息
    assert log_entry == f"write: {test_file_name} #{new_checksum}\n"


# 使用 pytest.mark.asyncio 标记异步测试函数
@pytest.mark.asyncio
async def test_write_file_fails_if_content_exists(test_file_name: Path, agent: Agent):
    # 定义新内容和新内容的校验和
    new_content = "This is new content.\n"
    # 记录写入操作到文件操作日志
    file_ops.log_operation(
        "write",
        test_file_name,
        agent=agent,
        checksum=file_ops.text_checksum(new_content),
    )
    # 断言写入已存在内容时是否引发重复操作错误
    with pytest.raises(DuplicateOperationError):
        await file_ops.write_to_file(test_file_name, new_content, agent=agent)


# 使用 pytest.mark.asyncio 标记异步测试函数
@pytest.mark.asyncio
async def test_write_file_succeeds_if_content_different(
    test_file_with_content_path: Path, agent: Agent
):
    # 定义新内容
    new_content = "This is different content.\n"
    # 异步写入新内容到文件
    await file_ops.write_to_file(test_file_with_content_path, new_content, agent=agent)


# 使用 pytest.mark.asyncio 标记异步测试函数
@pytest.mark.asyncio
async def test_append_to_file(test_nested_file: Path, agent: Agent):
    # 定义追加的文本
    append_text = "This is appended text.\n"
    # 异步写入追加文本到文件
    await file_ops.write_to_file(test_nested_file, append_text, agent=agent)
    # 追加文本到文件
    file_ops.append_to_file(test_nested_file, append_text, agent=agent)
    # 打开文件，读取其中的内容
    with open(test_nested_file, "r") as f:
        content_after = f.read()
    # 断言文件内容是否正确追加
    assert content_after == append_text + append_text


# 测试追加文件时是否使用追加文件的校验和
def test_append_to_file_uses_checksum_from_appended_file(
    test_file_name: Path, agent: Agent
):
    # 定义追加的文本
    append_text = "This is appended text.\n"
    # 追加文本到文件
    file_ops.append_to_file(
        agent.workspace.get_path(test_file_name),
        append_text,
        agent=agent,
    )
    # 再次追加相同文本到文件
    file_ops.append_to_file(
        agent.workspace.get_path(test_file_name),
        append_text,
        agent=agent,
    )
    # 以只读模式打开文件管理器中的文件操作日志文件，使用 UTF-8 编码
    with open(agent.file_manager.file_ops_log_path, "r", encoding="utf-8") as f:
        # 读取文件内容
        log_contents = f.read()
    
    # 创建 MD5 散列对象
    digest = hashlib.md5()
    # 更新散列对象的内容，使用 UTF-8 编码
    digest.update(append_text.encode("utf-8"))
    # 计算散列值的十六进制表示
    checksum1 = digest.hexdigest()
    # 再次更新散列对象的内容，使用 UTF-8 编码
    digest.update(append_text.encode("utf-8"))
    # 计算第二个散列值的十六进制表示
    checksum2 = digest.hexdigest()
    # 断言文件内容与特定格式的字符串相等
    assert log_contents == (
        f"append: {test_file_name} #{checksum1}\n"
        f"append: {test_file_name} #{checksum2}\n"
    )
# 测试列出文件的函数，接受工作空间、测试目录和代理作为参数
def test_list_files(workspace: FileWorkspace, test_directory: Path, agent: Agent):
    # 案例1：创建文件A和B，搜索A，并确保我们不返回A和B
    file_a = workspace.get_path("file_a.txt")
    file_b = workspace.get_path("file_b.txt")

    # 写入文件A的内容
    with open(file_a, "w") as f:
        f.write("This is file A.")

    # 写入文件B的内容
    with open(file_b, "w") as f:
        f.write("This is file B.")

    # 创建一个子目录并将文件A的副本放入其中
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    with open(os.path.join(test_directory, file_a.name), "w") as f:
        f.write("This is file A in the subdirectory.")

    # 列出工作空间根目录下的所有文件
    files = file_ops.list_folder(str(workspace.root), agent=agent)
    # 断言文件A的名称在列表中
    assert file_a.name in files
    # 断言文件B的名称在列表中
    assert file_b.name in files
    # 断言子目录中文件A的完整路径在列表中
    assert os.path.join(Path(test_directory).name, file_a.name) in files

    # 清理文件
    os.remove(file_a)
    os.remove(file_b)
    os.remove(os.path.join(test_directory, file_a.name))
    os.rmdir(test_directory)

    # 案例2：搜索一个不存在的文件，并确保不会抛出异常
    non_existent_file = "non_existent_file.txt"
    # 列出空字符串路径下的所有文件
    files = file_ops.list_folder("", agent=agent)
    # 断言不存在的文件不在列表中
    assert non_existent_file not in files
```