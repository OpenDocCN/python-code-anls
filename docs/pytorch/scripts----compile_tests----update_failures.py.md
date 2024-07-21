# `.\pytorch\scripts\compile_tests\update_failures.py`

```
    # 解析命令行参数并执行相应操作的脚本
    filename, test_dir, unexpected_successes, new_xfails, new_skips, unexpected_skips = (
        # 初始化输出文件的目录
        failures_directory = os.path.join(test_dir, "dynamo_expected_failures")
        # 初始化跳过文件的目录
        skips_directory = os.path.join(test_dir, "dynamo_skips")

        # 获取已有的 Dynamo 期望失败文件列表
        dynamo_expected_failures = set(os.listdir(failures_directory))
        # 获取已有的 Dynamo 跳过文件列表
        dynamo_skips = set(os.listdir(skips_directory))

        # 这些是手工编写的跳过文件
        extra_dynamo_skips = set()
        # 打开给定文件并逐行读取
        with open(filename) as f:
            start = False
            for text in f.readlines():
                text = text.strip()
                # 开始处理跳过文件的列表
                if start:
                    # 如果读取到结束标志
                    if text == "}":
                        break
                    # 添加处理过的跳过文件名到集合中
                    extra_dynamo_skips.add(text.strip(',"'))
                else:
                    # 如果读取到开始标志，则开始处理
                    if text == "extra_dynamo_skips = {":
                        start = True

    # 格式化测试用例名称以便后续使用
    formatted_unexpected_successes = {
        f"{format(test)}" for test in unexpected_successes.values()
    }
    # 格式化不期待跳过的测试用例名称以便后续使用
    formatted_unexpected_skips = {
        f"{format(test)}" for test in unexpected_skips.values()
    }
    # 格式化新的预期失败测试用例名称列表以便后续使用
    formatted_new_xfails = [f"{format(test)}" for test in new_xfails.values()]
    # 格式化新的跳过测试用例名称列表以便后续使用
    formatted_new_skips = [f"{format(test)}" for test in new_skips.values()]

    # 移除指定路径下的文件
    def remove_file(path, name):
        # 拼接文件路径
        file = os.path.join(path, name)
        # 构建移除文件的 Git 命令
        cmd = ["git", "rm", file]
        # 执行 Git 命令
        subprocess.run(cmd)

    # 在指定路径下创建文件
    def add_file(path, name):
        # 拼接文件路径
        file = os.path.join(path, name)
        # 创建空文件
        with open(file, "w") as fp:
            pass
        # 构建添加文件的 Git 命令
        cmd = ["git", "add", file]
        # 执行 Git 命令
        subprocess.run(cmd)

    # 已覆盖的不期待成功的测试用例集合
    covered_unexpected_successes = set()

    # dynamo_expected_failures
    # 对于每个预期失败的测试，检查是否在格式化后的意外成功集合中
    # 如果是，则将其添加到已覆盖的意外成功集合，并移除对应的文件
    for test in dynamo_expected_failures:
        if test in formatted_unexpected_successes:
            covered_unexpected_successes.add(test)
            remove_file(failures_directory, test)
    
    # 对于每个新的预期失败的测试，将其添加到失败目录中
    for test in formatted_new_xfails:
        add_file(failures_directory, test)

    # 计算剩余的未处理意外成功测试
    leftover_unexpected_successes = (
        formatted_unexpected_successes - covered_unexpected_successes
    )
    if len(leftover_unexpected_successes) > 0:
        # 输出警告信息，指出无法移除的预期失败测试数量
        print(
            "WARNING: we were unable to remove these "
            f"{len(leftover_unexpected_successes)} expectedFailures:"
        )
        # 逐个打印剩余的未处理意外成功测试
        for stuff in leftover_unexpected_successes:
            print(stuff)

    # 检查并处理 dynamo_skips
    for test in dynamo_skips:
        if test in formatted_unexpected_skips:
            # 如果在格式化后的意外跳过测试中，则移除对应的文件
            remove_file(skips_directory, test)
    
    # 处理额外的 dynamo_skips
    for test in extra_dynamo_skips:
        if test in formatted_unexpected_skips:
            # 输出警告信息，说明需要手动移除的测试
            print(
                f"WARNING: {test} in dynamo_test_failures.py needs to be removed manually"
            )
    
    # 处理新的跳过测试，将其添加到跳过目录中
    for test in formatted_new_skips:
        add_file(skips_directory, test)
# 定义一个函数，接收两个字典作为输入，返回它们的交集和差集组成的字典
def get_intersection_and_outside(a_dict, b_dict):
    # 将a_dict的所有键转换为集合a
    a = set(a_dict.keys())
    # 将b_dict的所有键转换为集合b
    b = set(b_dict.keys())
    # 计算a和b的交集，即两个字典共有的键的集合
    intersection = a.intersection(b)
    # 计算a和b的并集，并去除交集部分，得到两个字典各自独有的键的集合
    outside = (a.union(b)) - intersection

    # 定义一个内部函数，根据给定的键集合构建一个字典
    def build_dict(keys):
        result = {}
        for k in keys:
            # 如果键k存在于a_dict中，则将a_dict[k]加入结果字典中
            if k in a_dict:
                result[k] = a_dict[k]
            # 否则，将b_dict[k]加入结果字典中
            else:
                result[k] = b_dict[k]
        return result

    # 返回由交集和差集构建的两个字典
    return build_dict(intersection), build_dict(outside)


# 定义一个函数，更新给定的文件和相关目录中的测试结果
def update(filename, test_dir, py38_dir, py311_dir, also_remove_skips):
    # 定义一个内部函数，读取指定目录中的测试结果，并返回不同类型的测试结果字典
    def read_test_results(directory):
        # 调用open_test_results函数获取目录中的xmls对象
        xmls = open_test_results(directory)
        # 调用get_testcases函数从xmls中获取测试用例对象集合
        testcases = get_testcases(xmls)
        # 从testcases中过滤出意外成功的测试用例，并构建字典unexpected_successes
        unexpected_successes = {
            key(test): test for test in testcases if is_unexpected_success(test)
        }
        # 从testcases中过滤出失败的测试用例，并构建字典failures
        failures = {key(test): test for test in testcases if is_failure(test)}
        # 从testcases中过滤出通过但被跳过的测试用例，并构建字典passing_skipped_tests
        passing_skipped_tests = {
            key(test): test for test in testcases if is_passing_skipped_test(test)
        }
        # 返回三种不同类型的测试结果字典
        return unexpected_successes, failures, passing_skipped_tests

    # 分别读取py38_dir和py311_dir目录中的测试结果，并解包赋值给对应的变量
    (
        py38_unexpected_successes,
        py38_failures,
        py38_passing_skipped_tests,
    ) = read_test_results(py38_dir)
    (
        py311_unexpected_successes,
        py311_failures,
        py311_passing_skipped_tests,
    ) = read_test_results(py311_dir)

    # 将py38_unexpected_successes和py311_unexpected_successes合并为unexpected_successes字典
    unexpected_successes = {**py38_unexpected_successes, **py311_unexpected_successes}
    # 调用get_intersection_and_outside函数，计算py38_unexpected_successes和py311_unexpected_successes的交集和差集
    _, skips = get_intersection_and_outside(
        py38_unexpected_successes, py311_unexpected_successes
    )
    # 调用get_intersection_and_outside函数，计算py38_failures和py311_failures的交集和差集
    xfails, more_skips = get_intersection_and_outside(py38_failures, py311_failures)
    
    # 如果also_remove_skips为True，则计算py38_passing_skipped_tests和py311_passing_skipped_tests的交集
    if also_remove_skips:
        unexpected_skips, _ = get_intersection_and_outside(
            py38_passing_skipped_tests, py311_passing_skipped_tests
        )
    # 否则，将unexpected_skips置为空字典
    else:
        unexpected_skips = {}
    
    # 将所有跳过测试的字典合并为all_skips
    all_skips = {**skips, **more_skips}
    
    # 打印更新的结果信息，包括新的意外成功数、新的xfail数、新的跳过数和新的意外跳过数
    print(
        f"Discovered {len(unexpected_successes)} new unexpected successes, "
        f"{len(xfails)} new xfails, {len(all_skips)} new skips, {len(unexpected_skips)} new unexpected skips"
    )
    
    # 调用patch_file函数，更新filename文件，传入测试结果相关的字典参数
    return patch_file(
        filename, test_dir, unexpected_successes, xfails, all_skips, unexpected_skips
    )


# 如果当前脚本作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象parser，用于解析命令行参数
    parser = argparse.ArgumentParser(
        prog="update_dynamo_test_failures",
        description="Read from logs and update the dynamo_test_failures file",
    )
    # 添加命令行参数filename，表示dynamo_test_failures.py的路径
    parser.add_argument(
        "filename",
        nargs="?",
        default=str(
            Path(__file__).absolute().parent.parent.parent
            / "torch/testing/_internal/dynamo_test_failures.py"
        ),
        help="Optional path to dynamo_test_failures.py",
    )
    # 添加命令行参数test_dir，表示测试文件夹的路径
    parser.add_argument(
        "test_dir",
        nargs="?",
        default=str(Path(__file__).absolute().parent.parent.parent / "test"),
        help="Optional path to test folder",
    )
    parser.add_argument(
        "commit",
        help=(
            "The commit sha for the latest commit on a PR from which we will "
            "pull CI test results, e.g. 7e5f597aeeba30c390c05f7d316829b3798064a5"
        ),
    )
    # 添加一个位置参数 'commit' 到参数解析器，用于指定来自 PR 的最新提交的 SHA 值，用于提取 CI 测试结果

    parser.add_argument(
        "--also-remove-skips",
        help="Also attempt to remove skips. WARNING: does not guard against test flakiness",
        action="store_true",
    )
    # 添加一个可选参数 '--also-remove-skips' 到参数解析器，如果指定则尝试移除测试中的跳过标记，但不保证测试的稳定性

    args = parser.parse_args()
    # 解析命令行参数，并存储在 args 对象中

    assert Path(args.filename).exists(), args.filename
    # 断言检查 args.filename 指定的文件路径是否存在，如果不存在则抛出异常并显示文件名

    assert Path(args.test_dir).exists(), args.test_dir
    # 断言检查 args.test_dir 指定的目录路径是否存在，如果不存在则抛出异常并显示目录名

    dynamo38, dynamo311 = download_reports(args.commit, ("dynamo38", "dynamo311"))
    # 调用 download_reports 函数，传递 args.commit 作为参数之一，下载并返回两个报告数据，分别赋给 dynamo38 和 dynamo311

    update(args.filename, args.test_dir, dynamo38, dynamo311, args.also_remove_skips)
    # 调用 update 函数，传递 args.filename、args.test_dir、dynamo38、dynamo311 和 args.also_remove_skips 作为参数，执行更新操作
```