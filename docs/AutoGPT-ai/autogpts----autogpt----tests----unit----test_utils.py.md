# `.\AutoGPT\autogpts\autogpt\tests\unit\test_utils.py`

```py
# 导入所需的模块
import json
import os
from unittest.mock import patch
import pytest
import requests
from git import InvalidGitRepositoryError

# 导入自定义模块中的函数
from autogpt.app.utils import (
    get_bulletin_from_web,
    get_current_git_branch,
    get_latest_bulletin,
)
from autogpt.json_utils.utilities import extract_dict_from_response
from autogpt.utils import validate_yaml_file
from tests.utils import skip_in_ci

# 定义一个 fixture，返回一个有效的 JSON 响应字典
@pytest.fixture
def valid_json_response() -> dict:
    return {
        "thoughts": {
            "text": "My task is complete. I will use the 'task_complete' command "
            "to shut down.",
            "reasoning": "I will use the 'task_complete' command because it allows me "
            "to shut down and signal that my task is complete.",
            "plan": "I will use the 'task_complete' command with the reason "
            "'Task complete: retrieved Tesla's revenue in 2022.' to shut down.",
            "criticism": "I need to ensure that I have completed all necessary tasks "
            "before shutting down.",
            "speak": "All done!",
        },
        "command": {
            "name": "task_complete",
            "args": {"reason": "Task complete: retrieved Tesla's revenue in 2022."},
        },
    }

# 定义一个 fixture，返回一个无效的 JSON 响应字典
@pytest.fixture
def invalid_json_response() -> dict:
    return {
        "thoughts": {
            "text": "My task is complete. I will use the 'task_complete' command "
            "to shut down.",
            "reasoning": "I will use the 'task_complete' command because it allows me "
            "to shut down and signal that my task is complete.",
            "plan": "I will use the 'task_complete' command with the reason "
            "'Task complete: retrieved Tesla's revenue in 2022.' to shut down.",
            "criticism": "I need to ensure that I have completed all necessary tasks "
            "before shutting down.",
            "speak": "",
        },
        "command": {"name": "", "args": {}},
    }

# 定义一个测试函数，用于测试验证 YAML 文件的有效性
def test_validate_yaml_file_valid():
    # 使用写入模式打开文件"valid_test_file.yaml"，写入内容"setting: value"
    with open("valid_test_file.yaml", "w") as f:
        f.write("setting: value")
    # 调用validate_yaml_file函数验证"valid_test_file.yaml"文件，获取返回的结果和消息
    result, message = validate_yaml_file("valid_test_file.yaml")
    # 删除文件"valid_test_file.yaml"
    os.remove("valid_test_file.yaml")
    
    # 断言结果为True
    assert result is True
    # 断言消息中包含"Successfully validated"
    assert "Successfully validated" in message
# 测试当 YAML 文件不存在时的验证函数
def test_validate_yaml_file_not_found():
    # 调用验证 YAML 文件函数，传入不存在的文件名
    result, message = validate_yaml_file("non_existent_file.yaml")

    # 断言验证结果为 False
    assert result is False
    # 断言消息中包含特定字符串
    assert "wasn't found" in message


# 测试当 YAML 文件内容无效时的验证函数
def test_validate_yaml_file_invalid():
    # 创建一个无效的测试 YAML 文件
    with open("invalid_test_file.yaml", "w") as f:
        f.write(
            "settings:\n"
            "  first_setting: value\n"
            "  second_setting: value\n"
            "    nested_setting: value\n"
            "  third_setting: value\n"
            "unindented_setting: value"
        )
    # 调用验证 YAML 文件函数，传入无效的文件名
    result, message = validate_yaml_file("invalid_test_file.yaml")
    # 删除测试文件
    os.remove("invalid_test_file.yaml")
    # 打印结果和消息
    print(result)
    print(message)
    # 断言验证结果为 False
    assert result is False
    # 断言消息中包含特定字符串
    assert "There was an issue while trying to read" in message


# 测试从网络获取公告成功的函数
@patch("requests.get")
def test_get_bulletin_from_web_success(mock_get):
    expected_content = "Test bulletin from web"

    # 设置模拟请求返回的状态码和内容
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = expected_content
    # 调用从网络获取公告函数
    bulletin = get_bulletin_from_web()

    # 断言公告内容在返回结果中
    assert expected_content in bulletin
    # 断言请求的 URL
    mock_get.assert_called_with(
        "https://raw.githubusercontent.com/Significant-Gravitas/AutoGPT/master/autogpts/autogpt/BULLETIN.md"  # noqa: E501
    )


# 测试从网络获取公告失败的函数
@patch("requests.get")
def test_get_bulletin_from_web_failure(mock_get):
    # 设置模拟请求返回的状态码
    mock_get.return_value.status_code = 404
    # 调用从网络获取公告函数
    bulletin = get_bulletin_from_web()

    # 断言返回结果为空字符串
    assert bulletin == ""


# 测试从网络获取公告发生异常的函数
@patch("requests.get")
def test_get_bulletin_from_web_exception(mock_get):
    # 设置模拟请求抛出异常
    mock_get.side_effect = requests.exceptions.RequestException()
    # 调用从网络获取公告函数
    bulletin = get_bulletin_from_web()

    # 断言返回结果为空字符串
    assert bulletin == ""


# 测试获取最新公告时文件不存在的情况
def test_get_latest_bulletin_no_file():
    # 如果当前公告文件存在，则删除
    if os.path.exists("data/CURRENT_BULLETIN.md"):
        os.remove("data/CURRENT_BULLETIN.md")

    # 调用获取最新公告函数
    bulletin, is_new = get_latest_bulletin()
    # 断言为新公告
    assert is_new


# 测试获取最新公告时文件存在的情况
def test_get_latest_bulletin_with_file():
    expected_content = "Test bulletin"
    # 创建一个包含内容的当前公告文件
    with open("data/CURRENT_BULLETIN.md", "w", encoding="utf-8") as f:
        f.write(expected_content)
    # 使用 patch 函数模拟 get_bulletin_from_web 函数的返回值为空字符串
    with patch("autogpt.app.utils.get_bulletin_from_web", return_value=""):
        # 调用 get_latest_bulletin 函数获取最新公告内容和是否为新公告的标志
        bulletin, is_new = get_latest_bulletin()
        # 断言预期内容在获取的公告内容中
        assert expected_content in bulletin
        # 断言 is_new 为 False
        assert is_new is False
    
    # 删除指定路径下的文件 CURRENT_BULLETIN.md
    os.remove("data/CURRENT_BULLETIN.md")
# 测试获取最新公告，当有新公告时
def test_get_latest_bulletin_with_new_bulletin():
    # 创建一个文件对象，写入旧公告内容
    with open("data/CURRENT_BULLETIN.md", "w", encoding="utf-8") as f:
        f.write("Old bulletin")

    # 设置期望的新公告内容
    expected_content = "New bulletin from web"
    # 使用 patch 临时替换函数的返回值为期望内容
    with patch(
        "autogpt.app.utils.get_bulletin_from_web", return_value=expected_content
    ):
        # 调用函数获取最新公告
        bulletin, is_new = get_latest_bulletin()
        # 断言新公告中包含特定标记
        assert "::NEW BULLETIN::" in bulletin
        # 断言新公告中包含期望内容
        assert expected_content in bulletin
        # 断言返回值中标记为新公告
        assert is_new

    # 删除临时创建的文件
    os.remove("data/CURRENT_BULLETIN.md")


# 测试获取最新公告，当新公告与旧公告相同
def test_get_latest_bulletin_new_bulletin_same_as_old_bulletin():
    # 设置期望的公告内容
    expected_content = "Current bulletin"
    # 创建一个文件对象，写入旧公告内容
    with open("data/CURRENT_BULLETIN.md", "w", encoding="utf-8") as f:
        f.write(expected_content)

    # 使用 patch 临时替换函数的返回值为期望内容
    with patch(
        "autogpt.app.utils.get_bulletin_from_web", return_value=expected_content
    ):
        # 调用函数获取最新公告
        bulletin, is_new = get_latest_bulletin()
        # 断言新公告中包含期望内容
        assert expected_content in bulletin
        # 断言返回值中标记为非新公告
        assert is_new is False

    # 删除临时创建的文件
    os.remove("data/CURRENT_BULLETIN.md")


# 跳过在 CI 环境中运行的测试
@skip_in_ci
def test_get_current_git_branch():
    # 获取当前 Git 分支名称
    branch_name = get_current_git_branch()
    # 断言分支名称不为空
    assert branch_name != ""


# 测试成功获取当前 Git 分支
@patch("autogpt.app.utils.Repo")
def test_get_current_git_branch_success(mock_repo):
    # 设置模拟的 Git 仓库分支名称
    mock_repo.return_value.active_branch.name = "test-branch"
    # 获取当前 Git 分支名称
    branch_name = get_current_git_branch()

    # 断言分支名称为模拟的名称
    assert branch_name == "test-branch"


# 测试获取当前 Git 分支失败
@patch("autogpt.app.utils.Repo")
def test_get_current_git_branch_failure(mock_repo):
    # 设置模拟的 Git 仓库抛出异常
    mock_repo.side_effect = InvalidGitRepositoryError()
    # 获取当前 Git 分支名称
    branch_name = get_current_git_branch()

    # 断言分支名称为空
    assert branch_name == ""


# 测试从响应中提取 JSON 数据
def test_extract_json_from_response(valid_json_response: dict):
    # 模拟来自 OpenAI 的响应数据
    emulated_response_from_openai = json.dumps(valid_json_response)
    # 断言提取的 JSON 数据与原始数据相同
    assert (
        extract_dict_from_response(emulated_response_from_openai) == valid_json_response
    )


# 测试从包含在代码块中的响应中提取 JSON 数据
def test_extract_json_from_response_wrapped_in_code_block(valid_json_response: dict):
    # 模拟包含在代码块中的响应数据
    emulated_response_from_openai = "```" + json.dumps(valid_json_response) + "```"
    # 使用断言来检查从 OpenAI 模拟响应中提取的字典是否与有效的 JSON 响应相匹配
    assert (
        extract_dict_from_response(emulated_response_from_openai) == valid_json_response
    )
# 测试从包含在代码块中的 JSON 响应中提取 JSON 数据
def test_extract_json_from_response_wrapped_in_code_block_with_language(
    valid_json_response: dict,
):
    # 模拟来自 OpenAI 的响应，将 JSON 数据包装在 JSON 代码块中
    emulated_response_from_openai = "```json" + json.dumps(valid_json_response) + "```"
    # 断言提取出的字典与有效的 JSON 响应相等
    assert (
        extract_dict_from_response(emulated_response_from_openai) == valid_json_response
    )

# 测试从包含在字符串中的 JSON 响应中提取 JSON 数据
def test_extract_json_from_response_json_contained_in_string(valid_json_response: dict):
    # 模拟来自 OpenAI 的响应，将 JSON 数据包含在字符串中
    emulated_response_from_openai = (
        "sentence1" + json.dumps(valid_json_response) + "sentence2"
    )
    # 断言提取出的字典与有效的 JSON 响应相等
    assert (
        extract_dict_from_response(emulated_response_from_openai) == valid_json_response
    )
```