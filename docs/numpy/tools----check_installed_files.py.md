# `.\numpy\tools\check_installed_files.py`

```py
"""
Check if all the test and .pyi files are installed after building.

Examples::

    $ python check_installed_files.py install_dirname

        install_dirname:
            the relative path to the directory where NumPy is installed after
            building and running `meson install`.

Notes
=====

The script will stop on encountering the first missing file in the install dir,
it will not give a full listing. This should be okay, because the script is
meant for use in CI so it's not like many files will be missing at once.

"""

import os                   # 导入操作系统接口模块
import glob                 # 导入文件通配符模块
import sys                  # 导入系统相关模块
import json                 # 导入 JSON 解析模块


CUR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))  # 获取当前脚本所在目录的绝对路径
ROOT_DIR = os.path.dirname(CUR_DIR)  # 获取当前脚本所在目录的父目录
NUMPY_DIR = os.path.join(ROOT_DIR, 'numpy')  # 构建 NumPy 目录路径


# Files whose installation path will be different from original one
changed_installed_path = {
    #'numpy/_build_utils/some_file.py': 'numpy/lib/some_file.py'
}


def main(install_dir, tests_check):
    INSTALLED_DIR = os.path.join(ROOT_DIR, install_dir)  # 构建安装目录的绝对路径
    if not os.path.exists(INSTALLED_DIR):  # 如果安装目录不存在，则抛出异常
        raise ValueError(
            f"Provided install dir {INSTALLED_DIR} does not exist"
        )

    numpy_test_files = get_files(NUMPY_DIR, kind='test')  # 获取 NumPy 测试文件列表
    installed_test_files = get_files(INSTALLED_DIR, kind='test')  # 获取安装目录下的测试文件列表

    if tests_check == "--no-tests":  # 如果传入参数指定不检查测试文件
        if len(installed_test_files) > 0:  # 如果安装目录下存在测试文件，则抛出异常
            raise Exception("Test files aren't expected to be installed in %s"
                            ", found %s" % (INSTALLED_DIR, installed_test_files))
        print("----------- No test files were installed --------------")
    else:
        # Check test files detected in repo are installed
        for test_file in numpy_test_files.keys():
            if test_file not in installed_test_files.keys():  # 如果有测试文件未安装，则抛出异常
                raise Exception(
                    "%s is not installed" % numpy_test_files[test_file]
                )

        print("----------- All the test files were installed --------------")

    numpy_pyi_files = get_files(NUMPY_DIR, kind='stub')  # 获取 NumPy .pyi 文件列表
    installed_pyi_files = get_files(INSTALLED_DIR, kind='stub')  # 获取安装目录下的 .pyi 文件列表

    # Check *.pyi files detected in repo are installed
    for pyi_file in numpy_pyi_files.keys():
        if pyi_file not in installed_pyi_files.keys():  # 如果有 .pyi 文件未安装，则根据情况继续检查或抛出异常
            if (tests_check == "--no-tests" and
                    "tests" in numpy_pyi_files[pyi_file]):
                continue
            raise Exception("%s is not installed" % numpy_pyi_files[pyi_file])

    print("----------- All the necessary .pyi files "
          "were installed --------------")


def get_files(dir_to_check, kind='test'):
    files = dict()
    patterns = {
        'test': f'{dir_to_check}/**/test_*.py',  # 搜索测试文件的通配符模式
        'stub': f'{dir_to_check}/**/*.pyi',     # 搜索 .pyi 文件的通配符模式
    }
    for path in glob.glob(patterns[kind], recursive=True):  # 使用通配符模式搜索文件并遍历结果
        relpath = os.path.relpath(path, dir_to_check)  # 获取相对路径
        files[relpath] = path  # 将文件相对路径和绝对路径添加到字典中

    if sys.version_info >= (3, 12):  # 如果 Python 版本大于等于 3.12
        files = {
            k: v for k, v in files.items() if not k.startswith('distutils')  # 过滤掉 distutils 开头的键
        }

    return files  # 返回文件字典
    # 在文件字典中过滤掉来自于名为 'pythoncapi-compat' 的子模块的 Python 文件
    files = {
        k: v for k, v in files.items() if 'pythoncapi-compat' not in k
    }

    # 返回过滤后的文件字典
    return files
if __name__ == '__main__':
    # 检查是否在主程序中执行
    if len(sys.argv) < 2:
        # 如果输入参数少于2个，抛出数值错误异常
        raise ValueError("Incorrect number of input arguments, need "
                         "check_installation.py relpath/to/installed/numpy")

    # 获取安装目录
    install_dir = sys.argv[1]
    # 初始化测试检查字符串
    tests_check = ""
    # 如果输入参数大于等于3个，将第三个参数赋值给tests_check
    if len(sys.argv) >= 3:
        tests_check = sys.argv[2]
    # 调用主函数，传入安装目录和测试检查字符串
    main(install_dir, tests_check)

    # 初始化一个空集合用来存储所有标签
    all_tags = set()

    # 打开并读取intro-install_plan.json文件
    with open(os.path.join('build', 'meson-info',
                           'intro-install_plan.json'), 'r') as f:
        # 解析JSON文件内容并赋值给targets变量
        targets = json.load(f)

    # 遍历targets字典的键
    for key in targets.keys():
        # 遍历每个键对应的值（字典）
        for values in list(targets[key].values()):
            # 如果值字典中的'tag'键对应的值不在all_tags集合中
            if not values['tag'] in all_tags:
                # 将'tag'键对应的值添加到all_tags集合中
                all_tags.add(values['tag'])

    # 检查all_tags集合是否包含预期的标签集合
    if all_tags != set(['runtime', 'python-runtime', 'devel', 'tests']):
        # 如果不符合预期，抛出断言错误，显示实际找到的标签集合
        raise AssertionError(f"Found unexpected install tag: {all_tags}")
```