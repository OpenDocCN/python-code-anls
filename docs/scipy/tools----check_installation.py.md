# `D:\src\scipysrc\scipy\tools\check_installation.py`

```
# 导入必要的库：os（操作系统功能）、glob（文件名匹配）、sys（系统相关功能）
import os
import glob
import sys

# 获取当前脚本所在的绝对路径
CUR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
# 获取当前脚本的根目录路径
ROOT_DIR = os.path.dirname(CUR_DIR)
# 设置 SciPy 源码目录的路径
SCIPY_DIR = os.path.join(ROOT_DIR, 'scipy')

# 指定因安装路径不同而需变更的文件字典
changed_installed_path = {
    'scipy/_build_utils/tests/test_scipy_version.py':
        'scipy/_lib/tests/test_scipy_version.py'
}

# 需要排除的测试文件列表，这些文件不会被检查
exception_list_test_files = [
    "_lib/array_api_compat/tests/test_all.py",
    "_lib/array_api_compat/tests/test_array_namespace.py",
    "_lib/array_api_compat/tests/test_common.py",
    "_lib/array_api_compat/tests/test_isdtype.py",
    "_lib/array_api_compat/tests/test_vendoring.py",
    "_lib/array_api_compat/tests/test_array_namespace.py",
    "cobyqa/cobyqa/tests/test_main.py",
    "cobyqa/cobyqa/tests/test_models.py",
    "cobyqa/cobyqa/tests/test_problem.py",
    "cobyqa/utils/tests/test_exceptions.py",
    "cobyqa/utils/tests/test_math.py",
]

def main(install_dir, no_tests):
    # 构建安装目录的绝对路径
    INSTALLED_DIR = os.path.join(ROOT_DIR, install_dir)
    # 如果安装目录不存在，则抛出异常
    if not os.path.exists(INSTALLED_DIR):
        raise ValueError(f"Provided install dir {INSTALLED_DIR} does not exist")

    # 获取 SciPy 源码中的测试文件列表
    scipy_test_files = get_test_files(SCIPY_DIR)
    # 获取安装目录中的扩展模块的测试文件列表（以 .so 结尾的文件）
    scipy_test_extension_modules = get_test_files(INSTALLED_DIR, "so")
    # 获取安装目录中的所有测试文件列表
    installed_test_files = get_test_files(INSTALLED_DIR)

    # 如果 no_tests 参数为真，则检查是否有不应该安装的扩展模块测试文件
    if no_tests:
        if len(scipy_test_extension_modules) > 0:
            raise Exception(f"{scipy_test_extension_modules.values()} "
                            "should not be installed but "
                            "are found in the installation directory.")
    else:
        # 如果 no_tests 参数为假，则检查是否缺少扩展模块的测试文件
        if len(scipy_test_extension_modules) == 0:
            raise Exception("Test for extension modules should be "
                            "installed but are not found in the "
                            "installation directory.")

    # 检查检测到的测试文件是否在安装目录中
    # 遍历 scipy_test_files 字典中的每个测试文件名
    for test_file in scipy_test_files.keys():
        # 如果测试文件在异常列表中，则跳过当前循环，进行下一轮循环
        if test_file in exception_list_test_files:
            continue

        # 如果设置了 no_tests 标志
        if no_tests:
            # 如果测试文件在已安装的测试文件列表中，则抛出异常
            if test_file in installed_test_files:
                raise Exception(f"{test_file} should not be installed but "
                        "is found in the installation directory.")
            # 继续下一轮循环
            continue

        # 如果测试文件不在已安装的测试文件列表中，则抛出异常
        if test_file not in installed_test_files.keys():
            raise Exception(f"{scipy_test_files[test_file]} is not installed; "
                            f"either install it or add `{test_file}` to the "
                            "exception list in `tools/check_installation.py`")

    # 如果设置了 no_tests 标志，则输出相应信息
    if no_tests:
        print("----------- No test files were installed --------------")
    else:
        # 否则输出所有测试文件都已安装的信息
        print("----------- All the test files were installed --------------")

    # 获取 scipy 目录下的 .pyi 文件列表
    scipy_pyi_files = get_pyi_files(SCIPY_DIR)
    # 获取已安装目录下的 .pyi 文件列表
    installed_pyi_files = get_pyi_files(INSTALLED_DIR)

    # 检查在代码库中检测到的 *.pyi 文件是否在已安装目录中
    for pyi_file in scipy_pyi_files.keys():
        # 如果 *.pyi 文件不在已安装目录中
        if pyi_file not in installed_pyi_files.keys():
            # 如果设置了 no_tests，并且文件名中包含 "test" 字样，则跳过当前循环
            if no_tests and "test" in scipy_pyi_files[pyi_file]:
                continue
            # 否则抛出异常，指示该文件未安装
            raise Exception("%s is not installed" % scipy_pyi_files[pyi_file])

    # 输出所有必要的 .pyi 文件均已安装的信息
    print("----------- All the necessary .pyi files were installed --------------")
# 获取指定路径的上层路径，返回其相对路径
def get_suffix_path(current_path, levels=1):
    current_new = current_path
    # 循环获取上层路径，直到达到指定的层数（levels + 1）
    for i in range(levels + 1):
        current_new = os.path.dirname(current_new)

    # 返回当前路径与上层路径的相对路径
    return os.path.relpath(current_path, current_new)


# 获取指定目录下特定扩展名文件的路径，并返回以文件路径的后缀路径为键的字典
def get_test_files(dir, ext="py"):
    test_files = dict()
    # 根据扩展名构造下划线前缀
    underscore = "_" if ext == "so" else ""
    # 遍历指定目录及其子目录下符合条件的文件路径
    for path in glob.glob(f'{dir}/**/{underscore}test_*.{ext}', recursive=True):
        # 获取文件路径的后缀路径（向上三级）
        suffix_path = get_suffix_path(path, 3)
        # 如果后缀路径在变量 changed_installed_path 中有对应的映射，则替换为映射后的路径
        suffix_path = changed_installed_path.get(suffix_path, suffix_path)
        # 如果后缀路径不包含 "highspy"，则将其添加到测试文件字典中
        if "highspy" not in suffix_path:
            test_files[suffix_path] = path

    # 返回测试文件字典
    return test_files


# 获取指定目录及其子目录下所有.pyi文件的路径，并返回以文件路径的后缀路径为键的字典
def get_pyi_files(dir):
    pyi_files = dict()
    # 遍历指定目录及其子目录下所有.pyi文件的路径
    for path in glob.glob(f'{dir}/**/*.pyi', recursive=True):
        # 获取文件路径的后缀路径（向上两级）
        suffix_path = get_suffix_path(path, 2)
        # 将后缀路径与对应的.pyi文件路径添加到字典中
        pyi_files[suffix_path] = path

    # 返回.pyi文件字典
    return pyi_files


# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 如果输入参数少于两个，则抛出值错误异常
    if len(sys.argv) < 2:
        raise ValueError("Incorrect number of input arguments, need "
                         "check_installation.py relpath/to/installed/scipy")

    # 获取安装目录参数
    install_dir = sys.argv[1]
    # 默认不跳过测试文件
    no_tests = False
    # 如果输入参数为三个，并且第三个参数为 "--no-tests"，则跳过测试文件
    if len(sys.argv) == 3:
        no_tests = sys.argv[2] == "--no-tests"
    # 调用主函数，并传入安装目录和跳过测试文件标志
    main(install_dir, no_tests)
```