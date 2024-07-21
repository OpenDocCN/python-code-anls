# `.\pytorch\scripts\compile_tests\failures_histogram.py`

```py
import argparse  # 导入 argparse 模块，用于解析命令行参数
import re  # 导入 re 模块，用于正则表达式操作

from common import download_reports, get_testcases, key, open_test_results, skipped_test  # 导入自定义模块和函数

from passrate import compute_pass_rate  # 导入 compute_pass_rate 函数


"""
python failures_histogram.py commit_sha

Analyzes skip reasons for Dynamo tests and prints a histogram with repro
commands. You'll need to provide the commit_sha for a commit on the main branch,
from which we will pull CI test results.

This script requires the `gh` cli. You'll need to install it and then
authenticate with it via `gh auth login` before using this script.
https://docs.github.com/en/github-cli/github-cli/quickstart
"""


def skip_reason(testcase):
    for child in testcase.iter():  # 遍历测试用例的所有子元素
        if child.tag != "skipped":  # 如果子元素的标签不是 "skipped"，则跳过
            continue
        return child.attrib["message"]  # 返回 "skipped" 标签的 message 属性值
    raise AssertionError("no message?")  # 若未找到 "skipped" 标签，则抛出异常


def skip_reason_normalized(testcase):
    for child in testcase.iter():  # 遍历测试用例的所有子元素
        if child.tag != "skipped":  # 如果子元素的标签不是 "skipped"，则跳过
            continue
        result = child.attrib["message"].split("\n")[0]  # 获取 "skipped" 标签的 message 属性，并以换行符分割取第一部分
        result = result.split(">")[0]  # 以 ">" 分割字符串，并取第一部分
        result = re.sub(r"0x\w+", "0xDEADBEEF", result)  # 使用正则表达式替换十六进制数值为 "0xDEADBEEF"
        result = re.sub(r"MagicMock id='\d+'", "MagicMock id='0000000000'", result)  # 使用正则表达式替换 MagicMock 对象的 id
        result = re.sub(r"issues/\d+", "issues/XXX", result)  # 使用正则表达式替换 issues 编号为 "issues/XXX"
        result = re.sub(r"torch.Size\(\[.*\]\)", "torch.Size([...])", result)  # 使用正则表达式替换 torch.Size 的内容为 "[...]"
        result = re.sub(
            r"Could not get qualified name for class '.*'",
            "Could not get qualified name for class",
            result,
        )  # 使用正则表达式替换无法获取类名的消息
        return result  # 返回处理后的结果
    raise AssertionError("no message?")  # 若未找到 "skipped" 标签，则抛出异常


def get_failures(testcases):
    skipped = [t for t in testcases if skipped_test(t)]  # 筛选出被跳过的测试用例列表
    skipped_dict = {}  # 初始化空字典，用于存储跳过原因及其对应的测试用例列表
    for s in skipped:
        reason = skip_reason_normalized(s)  # 获取归一化后的跳过原因
        if reason not in skipped_dict:
            skipped_dict[reason] = []  # 若原因不在字典中，添加空列表
        skipped_dict[reason].append(s)  # 将当前测试用例加入对应原因的列表中
    result = []
    for s, v in skipped_dict.items():  # 遍历跳过原因字典
        result.append((len(v), s, v))  # 将原因及其对应的测试用例列表长度作为元组加入结果列表
    result.sort(reverse=True)  # 按照测试用例列表长度降序排序
    return result  # 返回结果列表


def repro(testcase):
    return f"PYTORCH_TEST_WITH_DYNAMO=1 pytest {testcase.attrib['file']} -v -k {testcase.attrib['name']}"
    # 构造用于复现测试的命令行字符串，包含文件路径和测试用例名称


def all_tests(testcase):
    return f"{testcase.attrib['file']}::{testcase.attrib['classname']}.{testcase.attrib['name']}"
    # 返回包含文件路径、类名和测试用例名称的字符串


# e.g. "17c5f69852/eager", "17c5f69852/dynamo"
def failures_histogram(eager_dir, dynamo_dir, verbose=False, format_issues=False):
    fail_keys = compute_pass_rate(eager_dir, dynamo_dir)  # 计算测试失败率的关键字列表
    xmls = open_test_results(dynamo_dir)  # 打开 Dynamo 测试结果的 XML 文件

    testcases = get_testcases(xmls)  # 从 XML 中获取测试用例列表
    testcases = [t for t in testcases if key(t) in fail_keys]  # 筛选出关键字在失败率列表中的测试用例
    dct = get_failures(testcases)  # 获取归类后的失败原因和对应的测试用例列表

    result = []
    for count, reason, testcases in dct:  # 遍历归类结果
        if verbose:
            row = (
                count,
                reason,
                repro(testcases[0]),  # 获取第一个测试用例的复现命令
                [all_tests(t) for t in testcases],  # 获取所有测试用例的详细描述列表
            )
        else:
            row = (count, reason, repro(testcases[0]))  # 仅获取第一个测试用例的复现命令
        result.append(row)  # 将行加入结果列表
    # 根据 verbose 变量的值选择合适的 header 格式
    header = (
        "(num_failed_tests, error_msg, sample_test, all_tests)"
        if verbose
        else "(num_failed_tests, error_msg, sample_test)"
    )
    # 打印选择的 header 标题
    print(header)
    # 计算所有测试结果中第一项的总和
    sum_counts = sum(r[0] for r in result)
    # 遍历结果集 result 中的每一行
    for row in result:
        # 如果 format_issues 为 True，则打印格式化的问题信息
        if format_issues:
            print(as_issue(*row))
        # 否则打印原始行数据
        else:
            print(row)
    # 打印测试结果的总和
    print("[counts]", sum_counts)
# 定义一个函数，用于生成一个包含问题详细信息的文本片段，并返回该文本片段
def as_issue(count, msg, repro, tests):
    # 将测试列表转换为以换行符分隔的字符串
    tests = "\n".join(tests)
    # 使用格式化字符串构建结果文本，包含问题计数、消息和重现命令
    result = f"""
{'-' * 50}
{count} Dynamo test are failing with \"{msg}\".

## Repro

`{repro}`

You will need to remove the skip or expectedFailure before running the repro command.
This may be just removing a sentinel file from in
[dynamo_expected_failures](https://github.com/pytorch/pytorch/blob/main/test/dynamo_expected_failures)
or [dynamo_skips](https://github.com/pytorch/pytorch/blob/main/test/dynamo_skips).


## Failing tests

Here's a comprehensive list of tests that fail (as of this issue) with the above message:
<details>
<summary>Click me</summary>

{tests}

</details>
"""
    # 返回构建的结果文本片段
    return result


if __name__ == "__main__":
    # 解析命令行参数的解析器
    parser = argparse.ArgumentParser(
        prog="failures_histogram",
        description="See statistics about skipped Dynamo tests",
    )
    # 添加位置参数：提交的 SHA 值，用于获取 CI 测试结果
    parser.add_argument(
        "commit",
        help=(
            "The commit sha for the latest commit on a PR from which we will "
            "pull CI test results, e.g. 7e5f597aeeba30c390c05f7d316829b3798064a5"
        ),
    )
    # 添加可选参数：是否输出详细信息
    parser.add_argument(
        "-v", "--verbose", help="Prints all failing test names", action="store_true"
    )
    # 添加可选参数：是否以 GitHub issues 的格式输出直方图
    parser.add_argument(
        "--format-issues",
        help="Prints histogram in a way that they can be copy-pasted as a github issues",
        action="store_true",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 如果设置了 --format-issues 参数，则强制 verbose=True
    verbose = args.verbose
    if args.format_issues:
        verbose = True

    # 下载指定提交的报告文件，包括 dynamo311 和 eager311 的报告
    dynamo311, eager311 = download_reports(args.commit, ("dynamo311", "eager311"))
    # 生成测试失败的直方图，并根据 verbose 和 args.format_issues 输出不同格式
    failures_histogram(eager311, dynamo311, verbose, args.format_issues)
```