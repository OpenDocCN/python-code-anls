# `.\pytorch\torch\testing\_internal\dynamo_test_failures.py`

```py
# mypy: allow-untyped-defs
# 引入日志模块
import logging
# 引入操作系统相关的功能
import os
# 引入系统相关的功能
import sys

# NOTE: [dynamo_test_failures.py]
#
# We generate xFailIfTorchDynamo* for all tests in `dynamo_expected_failures`
# We generate skipIfTorchDynamo* for all tests in `dynamo_skips`
#
# For an easier-than-manual way of generating and updating these lists,
# see scripts/compile_tests/update_failures.py
#
# If you're adding a new test, and it's failing PYTORCH_TEST_WITH_DYNAMO=1,
# either add the appropriate decorators to your test or add skips for them
# via test/dynamo_skips and test/dynamo_expected_failures.
#
# *These are not exactly unittest.expectedFailure and unittest.skip. We'll
# always execute the test and then suppress the signal, if necessary.
# If your tests crashes, or is slow, please use @skipIfTorchDynamo instead.
#
# The expected failure and skip files are located in test/dynamo_skips and
# test/dynamo_expected_failures. They're individual files rather than a list so
# git will merge changes easier.

def find_test_dir():
    # Find the path to the dynamo expected failure and skip files.
    from os.path import abspath, basename, dirname, exists, join, normpath

    # 如果运行平台为 win32，则返回 None
    if sys.platform == "win32":
        return None

    # 检查相对于当前文件的路径（本地构建）
    test_dir = normpath(join(dirname(abspath(__file__)), "../../../test"))
    # 如果存在 dynamo_expected_failures 目录，则返回找到的测试目录路径
    if exists(join(test_dir, "dynamo_expected_failures")):
        return test_dir

    # 检查相对于 __main__ 的路径（安装构建相对于测试文件）
    main = sys.modules["__main__"]
    file = getattr(main, "__file__", None)
    if file is None:
        # 生成的文件没有模块.__file__
        return None
    test_dir = dirname(abspath(file))
    while dirname(test_dir) != test_dir:
        # 如果找到 "test" 目录并且存在 dynamo_expected_failures 目录，则返回测试目录路径
        if basename(test_dir) == "test" and exists(
            join(test_dir, "dynamo_expected_failures")
        ):
            return test_dir
        test_dir = dirname(test_dir)

    # 未找到目录，返回 None
    return None

# 调用 find_test_dir() 函数获取测试目录路径
test_dir = find_test_dir()
# 如果未找到测试目录路径
if not test_dir:
    # 获取当前模块的日志记录器
    logger = logging.getLogger(__name__)
    # 输出警告信息，指示 dynamo_expected_failures 目录未找到，已知 dynamo 错误将不会被跳过
    logger.warning(
        "test/dynamo_expected_failures directory not found - known dynamo errors won't be skipped."
    )

# Tests that run without strict mode in PYTORCH_TEST_WITH_INDUCTOR=1.
# Please don't add anything to this list.
# 没有严格模式运行的测试集合，用于 PYTORCH_TEST_WITH_INDUCTOR=1
FIXME_inductor_non_strict = {
    "test_modules",
    "test_ops",
    "test_ops_gradients",
    "test_torch",
}

# We generate unittest.expectedFailure for all of the following tests
# when run under PYTORCH_TEST_WITH_DYNAMO=1.
# see NOTE [dynamo_test_failures.py] for more details
#
# This lists exists so we can more easily add large numbers of failing tests,
# 当运行 PYTORCH_TEST_WITH_DYNAMO=1 时，为以下所有测试生成 unittest.expectedFailure
# 详细信息请参阅 NOTE [dynamo_test_failures.py]
#
# This lists exists so we can more easily add large numbers of failing tests,
# 这些列表存在是为了更容易地添加大量失败的测试用例
if test_dir is None:
    # 如果未找到测试目录路径，初始化 dynamo_expected_failures 和 dynamo_skips 为空集合
    dynamo_expected_failures = set()
    dynamo_skips = set()
else:
    # 否则，设置失败目录和跳过目录的路径
    failures_directory = os.path.join(test_dir, "dynamo_expected_failures")
    skips_directory = os.path.join(test_dir, "dynamo_skips")

    # 获取 failures 目录和 skips 目录下的文件列表并转化为集合
    dynamo_expected_failures = set(os.listdir(failures_directory))
    dynamo_skips = set(os.listdir(skips_directory))
# 添加额外的 Dynamo 跳过的测试用例，由于大小写敏感问题，目前手动列出这些文件
extra_dynamo_skips = {
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_t_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_t_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_t_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_t_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_t_cpu_float32",
}
# 将额外的 Dynamo 跳过的测试用例并入 dynamo_skips 集合中
dynamo_skips = dynamo_skips.union(extra_dynamo_skips)

# 验证一些不变量
# 检查 dynamo_expected_failures 和 dynamo_skips 集合中的每个测试名称是否由一个点分隔成两部分
for test in dynamo_expected_failures.union(dynamo_skips):
    if len(test.split(".")) != 2:
        raise AssertionError(f'Invalid test name: "{test}"')

# 检查 dynamo_expected_failures 和 dynamo_skips 两个集合是否有重叠
intersection = dynamo_expected_failures.intersection(dynamo_skips)
if len(intersection) > 0:
    raise AssertionError(
        "there should be no overlap between dynamo_expected_failures "
        "and dynamo_skips, got " + str(intersection)
    )
```