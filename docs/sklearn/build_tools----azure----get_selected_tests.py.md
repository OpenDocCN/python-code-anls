# `D:\src\scipysrc\scikit-learn\build_tools\azure\get_selected_tests.py`

```
from get_commit_message import get_commit_message

# 导入函数 get_commit_message 用于获取提交信息

def get_selected_tests():
    """Parse the commit message to check if pytest should run only specific tests.
    
    If so, selected tests will be run with SKLEARN_TESTS_GLOBAL_RANDOM_SEED="all".
    
    The commit message must take the form:
        <title> [all random seeds]
        <test_name_1>
        <test_name_2>
        ...
    """
    # 调用 get_commit_message 函数获取提交信息
    commit_message = get_commit_message()

    # 检查提交信息中是否包含 "[all random seeds]"
    if "[all random seeds]" in commit_message:
        # 如果包含，提取出选择的测试用例部分，并进行格式处理
        selected_tests = commit_message.split("[all random seeds]")[1].strip()
        selected_tests = selected_tests.replace("\n", " or ")
    else:
        # 如果不包含，则没有选择的测试用例
        selected_tests = ""

    # 返回处理后的选择的测试用例字符串
    return selected_tests

if __name__ == "__main__":
    # 设置要传递到其他步骤的环境变量
    selected_tests = get_selected_tests()

    # 如果有选择的测试用例，则输出设置变量的命令和选择的测试用例信息（帮助调试）
    if selected_tests:
        print(f"##vso[task.setvariable variable=SELECTED_TESTS]'{selected_tests}'")
        print(f"selected tests: {selected_tests}")  # helps debugging
    else:
        # 如果没有选择的测试用例，则输出信息提示没有选择的测试用例
        print("no selected tests")
```