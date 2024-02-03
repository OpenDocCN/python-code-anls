# `.\AutoGPT\autogpts\autogpt\tests\unit\test_local_file_workspace.py`

```py
# 导入所需模块
from pathlib import Path
import pytest
# 导入本地文件工作空间相关模块
from autogpt.file_workspace.local import FileWorkspaceConfiguration, LocalFileWorkspace

# 定义工作空间根目录路径
_WORKSPACE_ROOT = Path("home/users/monty/auto_gpt_workspace")

# 可访问路径列表
_ACCESSIBLE_PATHS = [
    Path("."),
    Path("test_file.txt"),
    Path("test_folder"),
    Path("test_folder/test_file.txt"),
    Path("test_folder/.."),
    Path("test_folder/../test_file.txt"),
    Path("test_folder/../test_folder"),
    Path("test_folder/../test_folder/test_file.txt"),
]

# 不可访问路径列表
_INACCESSIBLE_PATHS = (
    [
        # 将我们带出工作空间
        Path(".."),
        Path("../test_file.txt"),
        Path("../not_auto_gpt_workspace"),
        Path("../not_auto_gpt_workspace/test_file.txt"),
        Path("test_folder/../.."),
        Path("test_folder/../../test_file.txt"),
        Path("test_folder/../../not_auto_gpt_workspace"),
        Path("test_folder/../../not_auto_gpt_workspace/test_file.txt"),
    ]
    + [
        # 包含空字节的路径
        Path("\0"),
        Path("\0test_file.txt"),
        Path("test_folder/\0"),
        Path("test_folder/\0test_file.txt"),
    ]
    + [
        # 绝对路径
        Path("/"),
        Path("/test_file.txt"),
        Path("/home"),
    ]
)

# 定义 fixture 函数，返回工作空间根目录路径
@pytest.fixture()
def workspace_root(tmp_path):
    return tmp_path / _WORKSPACE_ROOT

# 定义 fixture 函数，返回可访问路径
@pytest.fixture(params=_ACCESSIBLE_PATHS)
def accessible_path(request):
    return request.param

# 定义 fixture 函数，返回不可访问路径
@pytest.fixture(params=_INACCESSIBLE_PATHS)
def inaccessible_path(request):
    return request.param

# 测试函数，测试对可访问路径的路径清理功能
def test_sanitize_path_accessible(accessible_path, workspace_root):
    # 调用 LocalFileWorkspace 类的 _sanitize_path 方法，对可访问路径进行路径清理
    full_path = LocalFileWorkspace._sanitize_path(
        accessible_path,
        root=workspace_root,
        restrict_to_root=True,
    )
    # 断言清理后的路径为绝对路径
    assert full_path.is_absolute()
    # 断言清理后的路径在工作空间根目录下
    assert full_path.is_relative_to(workspace_root)

# 测试函数，测试对不可访问路径的路径清理功能
def test_sanitize_path_inaccessible(inaccessible_path, workspace_root):
    # 使用 pytest 来测试是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        # 调用 LocalFileWorkspace 类的 _sanitize_path 方法，传入不可访问的路径、工作空间根目录、以及限制在根目录内的标志
        LocalFileWorkspace._sanitize_path(
            inaccessible_path,
            root=workspace_root,
            restrict_to_root=True,
        )
# 测试可访问路径是否正确获取
def test_get_path_accessible(accessible_path, workspace_root):
    # 创建本地文件工作空间对象
    workspace = LocalFileWorkspace(FileWorkspaceConfiguration(root=workspace_root))
    # 获取完整路径
    full_path = workspace.get_path(accessible_path)
    # 断言完整路径是绝对路径
    assert full_path.is_absolute()
    # 断言完整路径是相对于工作空间根目录的相对路径

# 测试不可访问路径是否会引发异常
def test_get_path_inaccessible(inaccessible_path, workspace_root):
    # 创建本地文件工作空间对象
    workspace = LocalFileWorkspace(FileWorkspaceConfiguration(root=workspace_root))
    # 使用 pytest 检查是否引发 ValueError 异常
    with pytest.raises(ValueError):
        # 尝试获取不可访问路径的完整路径
        workspace.get_path(inaccessible_path)
```