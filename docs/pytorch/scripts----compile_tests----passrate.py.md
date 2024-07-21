# `.\pytorch\scripts\compile_tests\passrate.py`

```py
# 导入 argparse 库，用于解析命令行参数
import argparse

# 从 common 模块导入以下函数和变量
from common import (
    get_excluded_testcases,
    get_passed_testcases,
    get_testcases,
    key,
    open_test_results,
)

# 从 download_reports 模块导入 download_reports 函数
from download_reports import download_reports

"""
Usage: passrate.py commit_sha

Parses test reports to measure the passrate. The passrate is defined as:

A) Take the number of tests that pass under eager mode, excluding
CUDA, OpInfo, and ModuleInfo tests
B) Of those tests, count the number of tests that pass under Dynamo
C) Take B/A.

You'll need to provide the commit_sha for a commit on the main branch,
from which we will pull CI test results.

This script requires the `gh` cli. You'll need to install it and then
authenticate with it via `gh auth login` before using this script.
https://docs.github.com/en/github-cli/github-cli/quickstart
"""

# 定义函数 testcases_by_time，按测试用例执行时间降序排列返回
def testcases_by_time(xmls):
    # 调用 get_testcases 函数获取测试用例列表
    testcases = get_testcases(xmls)
    # 根据测试用例执行时间排序（降序）
    testcases.sort(reverse=True, key=lambda x: float(x.attrib["time"]))
    return testcases

# 定义函数 should_exclude，判断是否应该排除特定测试用例
def should_exclude(key):
    # 根据测试用例键获取测试文件名
    test_file = key.split("::")[0]
    # 排除 UNKNOWN 类型的测试用例
    if test_file == "UNKNOWN":
        return True
    # 策略：排除 inductor、export 和 dynamo 开头的测试用例
    if test_file.startswith("inductor/"):
        return True
    if test_file.startswith("export/"):
        return True
    if test_file.startswith("dynamo/"):
        return True
    return False

# 定义函数 compute_pass_rate，计算 Dynamo 单元测试的通过率
def compute_pass_rate(eager_dir, dynamo_dir):
    print("parsing xmls")
    # 打开并解析 Eager 模式测试结果 XML
    eager_xmls = open_test_results(eager_dir)
    # 打开并解析 Dynamo 模式测试结果 XML
    dynamo_xmls = open_test_results(dynamo_dir)

    print("computing pass rate")
    # 获取 Eager 模式下通过的测试用例列表
    eager_passed = get_passed_testcases(eager_xmls)
    # 获取 Dynamo 模式下通过的测试用例列表
    dynamo_passed = get_passed_testcases(dynamo_xmls)
    # 构建 Dynamo 模式下通过测试用例的键集合
    dynamo_pass_keys = {key(testcase) for testcase in dynamo_passed}
    # 过滤掉应该排除的 Dynamo 模式测试用例键
    dynamo_pass_keys = {key_ for key_ in dynamo_pass_keys if not should_exclude(key_)}
    # 构建 Eager 模式下通过测试用例的键集合
    tmp_eager_pass_keys = {key(testcase) for testcase in eager_passed}
    tmp_eager_pass_keys = {
        key_ for key_ in tmp_eager_pass_keys if not should_exclude(key_)
    }
    # 获取 Dynamo 模式下被排除的测试用例的键列表
    excluded = [key(t) for t in get_excluded_testcases(dynamo_xmls)]
    # 从 Eager 模式通过测试用例中排除在 Dynamo 模式中被排除的测试用例
    eager_pass_keys = tmp_eager_pass_keys - set(excluded)

    # 计算 Eager 模式和 Dynamo 模式共同通过的测试用例集合
    subset = eager_pass_keys.intersection(dynamo_pass_keys)
    total_subset = len(subset)
    total_tests = len(eager_pass_keys)
    # 计算并输出通过率、通过的测试用例数和总测试用例数
    print("pass rate", total_subset / total_tests, total_subset, total_tests)

    # 获取 Dynamo 模式下所有的测试用例列表
    dynamo_testcases = get_testcases(dynamo_xmls)
    # 构建测试用例键到测试用例对象的映射
    tc = {key(t): t for t in dynamo_testcases}

    # 用于调试的工具，查找在 Eager 模式下通过但在 Dynamo 模式下未找到的测试用例键集合
    not_there_keys = set()
    for key_ in eager_pass_keys:
        if key_ not in tc:
            not_there_keys.add(key_)

    # 计算并返回在 Eager 模式下通过但在 Dynamo 模式下未通过的测试用例键集合
    fail_keys = eager_pass_keys - subset
    return fail_keys

# 如果当前脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        prog="passrate", description="Computes the Dynamo unittest pass rate"
    )
    # 添加一个位置参数 'commit' 到命令行参数解析器，用于指定要拉取CI测试结果的PR最新提交的提交SHA
    parser.add_argument(
        "commit",
        help=(
            "The commit sha for the latest commit on a PR from which we will "
            "pull CI test results, e.g. 7e5f597aeeba30c390c05f7d316829b3798064a5"
        ),
    )
    # 解析命令行参数并将结果存储在args对象中
    args = parser.parse_args()
    # 调用download_reports函数，传入args.commit作为参数，下载'dynamo311'和'eager311'两种报告
    dynamo311, eager311 = download_reports(args.commit, ("dynamo311", "eager311"))
    # 调用compute_pass_rate函数，计算并输出eager311和dynamo311报告的通过率
    compute_pass_rate(eager311, dynamo311)
```