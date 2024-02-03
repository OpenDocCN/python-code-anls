# `.\AutoGPT\autogpts\forge\forge\sdk\workspace_test.py`

```py
# 导入必要的模块
import os
import pytest

# 假设类定义在名为 workspace.py 的文件中
from .workspace import LocalWorkspace

# 定义常量
TEST_BASE_PATH = "/tmp/test_workspace"
TEST_FILE_CONTENT = b"Hello World"
TEST_TASK_ID = "1234"

# LocalWorkspace 的设置和清理

# 使用 pytest.fixture 装饰器定义设置函数
@pytest.fixture
def setup_local_workspace():
    # 创建测试工作空间目录，如果目录已存在则不报错
    os.makedirs(TEST_BASE_PATH, exist_ok=True)
    # 使用 yield 将控制权交给测试函数
    yield
    # 在测试结束后清理测试工作空间目录
    os.system(f"rm -rf {TEST_BASE_PATH}")

# 测试本地读写删除和存在性
def test_local_read_write_delete_exists(setup_local_workspace):
    # 创建本地工作空间对象
    workspace = LocalWorkspace(TEST_BASE_PATH)

    # 写入文件
    workspace.write(TEST_TASK_ID, "test_file.txt", TEST_FILE_CONTENT)

    # 检查文件是否存在
    assert workspace.exists(TEST_TASK_ID, "test_file.txt")

    # 读取文件内容并进行断言
    assert workspace.read(TEST_TASK_ID, "test_file.txt") == TEST_FILE_CONTENT

    # 删除文件并检查是否存在
    workspace.delete(TEST_TASK_ID, "test_file.txt")
    assert not workspace.exists(TEST_TASK_ID, "test_file.txt")

# 测试本地列表功能
def test_local_list(setup_local_workspace):
    # 创建本地工作空间对象
    workspace = LocalWorkspace(TEST_BASE_PATH)
    # 写入两个测试文件
    workspace.write(TEST_TASK_ID, "test1.txt", TEST_FILE_CONTENT)
    workspace.write(TEST_TASK_ID, "test2.txt", TEST_FILE_CONTENT)

    # 列出指定任务 ID 下的所有文件
    files = workspace.list(TEST_TASK_ID, ".")
    # 断言文件列表是否包含指定文件名
    assert set(files) == {"test1.txt", "test2.txt"}
```