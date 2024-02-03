# `.\AutoGPT\autogpts\autogpt\tests\integration\test_execute_code.py`

```py
# 导入所需的模块
import random
import string
import tempfile
from pathlib import Path

import pytest

# 导入被测试的模块
import autogpt.commands.execute_code as sut  # system under testing
from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import (
    InvalidArgumentError,
    OperationNotAllowedError,
)

# 定义生成随机代码的 fixture
@pytest.fixture
def random_code(random_string) -> str:
    return f"print('Hello {random_string}!')"

# 定义生成 Python 测试文件的 fixture
@pytest.fixture
def python_test_file(agent: Agent, random_code: str):
    temp_file = tempfile.NamedTemporaryFile(dir=agent.workspace.root, suffix=".py")
    temp_file.write(str.encode(random_code))
    temp_file.flush()

    yield Path(temp_file.name)
    temp_file.close()

# 定义生成带参数的 Python 测试文件的 fixture
@pytest.fixture
def python_test_args_file(agent: Agent):
    temp_file = tempfile.NamedTemporaryFile(dir=agent.workspace.root, suffix=".py")
    temp_file.write(str.encode("import sys\nprint(sys.argv[1], sys.argv[2])"))
    temp_file.flush()

    yield Path(temp_file.name)
    temp_file.close()

# 定义生成随机字符串的 fixture
@pytest.fixture
def random_string():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(10))

# 测试执行 Python 文件的函数
def test_execute_python_file(python_test_file: Path, random_string: str, agent: Agent):
    result: str = sut.execute_python_file(python_test_file, agent=agent)
    assert result.replace("\r", "") == f"Hello {random_string}!\n"

# 测试执行带参数的 Python 文件的函数
def test_execute_python_file_args(
    python_test_args_file: Path, random_string: str, agent: Agent
):
    random_args = [random_string] * 2
    random_args_string = " ".join(random_args)
    result = sut.execute_python_file(
        python_test_args_file, args=random_args, agent=agent
    )
    assert result == f"{random_args_string}\n"

# 测试执行 Python 代码的函数
def test_execute_python_code(random_code: str, random_string: str, agent: Agent):
    result: str = sut.execute_python_code(random_code, agent=agent)
    assert result.replace("\r", "") == f"Hello {random_string}!\n"

# 测试执行无效 Python 文件的函数
def test_execute_python_file_invalid(agent: Agent):
    # 使用 pytest 来测试 sut.execute_python_file 方法是否会引发 InvalidArgumentError 异常
    with pytest.raises(InvalidArgumentError):
        # 调用 sut.execute_python_file 方法，传入一个非 Python 文件的路径和 agent 参数
        sut.execute_python_file(Path("not_python.txt"), agent)
# 测试执行 Python 文件不存在的情况，期望抛出 FileNotFoundError 异常
def test_execute_python_file_not_found(agent: Agent):
    # 使用 pytest 的断言检查是否抛出指定异常，并匹配异常信息
    with pytest.raises(
        FileNotFoundError,
        match=r"python: can't open file '([a-zA-Z]:)?[/\\\-\w]*notexist.py': "
        r"\[Errno 2\] No such file or directory",
    ):
        # 调用被测试的函数执行 Python 文件
        sut.execute_python_file(Path("notexist.py"), agent)


# 测试执行 shell 命令的情况
def test_execute_shell(random_string: str, agent: Agent):
    # 执行 shell 命令，返回结果
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    # 使用断言检查结果中是否包含特定字符串
    assert f"Hello {random_string}!" in result


# 测试执行不允许的本地 shell 命令的情况
def test_execute_shell_local_commands_not_allowed(random_string: str, agent: Agent):
    # 执行 shell 命令，返回结果
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    # 使用断言检查结果中是否包含特定字符串
    assert f"Hello {random_string}!" in result


# 测试执行被拒绝的 shell 命令的情况
def test_execute_shell_denylist_should_deny(agent: Agent, random_string: str):
    # 设置 agent 的 shell_denylist 属性
    agent.legacy_config.shell_denylist = ["echo"]

    # 使用 pytest 的断言检查是否抛出 OperationNotAllowedError 异常，并匹配异常信息
    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        # 调用被测试的函数执行 shell 命令
        sut.execute_shell(f"echo 'Hello {random_string}!'", agent)


# 测试执行被允许的 shell 命令的情况
def test_execute_shell_denylist_should_allow(agent: Agent, random_string: str):
    # 设置 agent 的 shell_denylist 属性
    agent.legacy_config.shell_denylist = ["cat"]

    # 执行 shell 命令，返回结果
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    # 使用断言检查结果中是否包含特定字符串
    assert "Hello" in result and random_string in result


# 测试执行被拒绝的 shell 命令的情况
def test_execute_shell_allowlist_should_deny(agent: Agent, random_string: str):
    # 设置 agent 的 shell_command_control 和 shell_allowlist 属性
    agent.legacy_config.shell_command_control = sut.ALLOWLIST_CONTROL
    agent.legacy_config.shell_allowlist = ["cat"]

    # 使用 pytest 的断言检查是否抛出 OperationNotAllowedError 异常，并匹配异常信息
    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        # 调用被测试的函数执行 shell 命令
        sut.execute_shell(f"echo 'Hello {random_string}!'", agent)


# 测试执行被允许的 shell 命令的情况
def test_execute_shell_allowlist_should_allow(agent: Agent, random_string: str):
    # 设置 agent 的 shell_command_control 和 shell_allowlist 属性
    agent.legacy_config.shell_command_control = sut.ALLOWLIST_CONTROL
    agent.legacy_config.shell_allowlist = ["echo"]

    # 执行 shell 命令，返回结果
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    # 使用断言检查结果中是否包含特定字符串
    assert "Hello" in result and random_string in result
```