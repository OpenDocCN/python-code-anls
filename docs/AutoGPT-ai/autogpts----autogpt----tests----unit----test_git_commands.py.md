# `.\AutoGPT\autogpts\autogpt\tests\unit\test_git_commands.py`

```py
# 导入 pytest 模块
import pytest
# 导入 GitCommandError 异常类
from git.exc import GitCommandError
# 导入 Repo 类
from git.repo.base import Repo

# 导入 Agent 类
from autogpt.agents.agent import Agent
# 导入 CommandExecutionError 异常类
from autogpt.agents.utils.exceptions import CommandExecutionError
# 导入 clone_repository 函数
from autogpt.commands.git_operations import clone_repository

# 定义 mock_clone_from fixture
@pytest.fixture
def mock_clone_from(mocker):
    return mocker.patch.object(Repo, "clone_from")

# 测试克隆 Auto-GPT 仓库
def test_clone_auto_gpt_repository(workspace, mock_clone_from, agent: Agent):
    # 设置 mock_clone_from 的返回值为 None
    mock_clone_from.return_value = None

    # 定义仓库 URL 和协议
    repo = "github.com/Significant-Gravitas/Auto-GPT.git"
    scheme = "https://"
    url = scheme + repo
    # 获取克隆路径
    clone_path = workspace.get_path("auto-gpt-repo")

    # 期望的输出信息
    expected_output = f"Cloned {url} to {clone_path}"

    # 调用 clone_repository 函数进行克隆操作
    clone_result = clone_repository(url=url, clone_path=clone_path, agent=agent)

    # 断言克隆结果与期望输出一致
    assert clone_result == expected_output
    # 断言 mock_clone_from 被调用一次
    mock_clone_from.assert_called_once_with(
        url=f"{scheme}{agent.legacy_config.github_username}:{agent.legacy_config.github_api_key}@{repo}",  # noqa: E501
        to_path=clone_path,
    )

# 测试克隆仓库时出现错误
def test_clone_repository_error(workspace, mock_clone_from, agent: Agent):
    # 定义错误的仓库 URL 和克隆路径
    url = "https://github.com/this-repository/does-not-exist.git"
    clone_path = workspace.get_path("does-not-exist")

    # 设置 mock_clone_from 抛出 GitCommandError 异常
    mock_clone_from.side_effect = GitCommandError(
        "clone", "fatal: repository not found", ""
    )

    # 使用 pytest 断言捕获 CommandExecutionError 异常
    with pytest.raises(CommandExecutionError):
        clone_repository(url=url, clone_path=clone_path, agent=agent)
```