# `D:\src\scipysrc\sympy\bin\coverage_report.py`

```
#!/usr/bin/env python
"""
Script to generate test coverage reports.

Usage:

$ bin/coverage_report.py

This will create a directory covhtml with the coverage reports. To
restrict the analysis to a directory, you just need to pass its name as
argument. For example:


$ bin/coverage_report.py sympy/logic

runs only the tests in sympy/logic/ and reports only on the modules in
sympy/logic/. To also run slow tests use --slow option. You can also get a
report on the parts of the whole sympy code covered by the tests in
sympy/logic/ by following up the previous command with


$ bin/coverage_report.py -c

"""
from __future__ import print_function

import os
import re
import sys
from argparse import ArgumentParser

# 检查 Python 版本要求
minver = '3.4'
try:
    import coverage
    if coverage.__version__ < minver:
        raise ImportError
except ImportError:
    # 提示用户安装 coverage 模块
    print(
        "You need to install module coverage (version %s or newer required).\n"
        "See https://coverage.readthedocs.io/en/latest/ or \n"
        "https://launchpad.net/ubuntu/+source/python-coverage/" % minver)
    sys.exit(-1)


# 要排除的目录模式列表
omit_dir_patterns = ['benchmark', 'examples',
                     'pyglet', 'test_external']
omit_dir_re = re.compile(r'|'.join(omit_dir_patterns))
# 匹配 Python 源文件的正则表达式
source_re = re.compile(r'.*\.py$')


# 生成指定目录下所有 Python 源文件的生成器函数
def generate_covered_files(top_dir):
    for dirpath, dirnames, filenames in os.walk(top_dir):
        # 从目录列表中移除要排除的目录
        omit_dirs = [dirn for dirn in dirnames if omit_dir_re.match(dirn)]
        for x in omit_dirs:
            dirnames.remove(x)
        for filename in filenames:
            # 如果是 Python 源文件，则生成其完整路径
            if source_re.match(filename):
                yield os.path.join(dirpath, filename)


# 生成测试覆盖率报告的函数
def make_report(
    test_args, source_dir='sympy/', report_dir='covhtml', use_cache=False,
    slow=False
    ):
    # 从 get_sympy 导入 path_hack，用于设置 sympy 项目的路径
    from get_sympy import path_hack
    # 获取 sympy 项目的顶层路径
    sympy_top = path_hack()
    # 切换到 sympy 项目的顶层路径
    os.chdir(sympy_top)

    # 创建 coverage 对象
    cov = coverage.coverage()
    # 排除特定的代码行不计入覆盖率
    cov.exclude("raise NotImplementedError")
    cov.exclude("def canonize")  # this should be "@decorated"
    
    # 如果使用缓存数据，则加载之前的覆盖率数据
    if use_cache:
        cov.load()
    else:
        # 否则，清空之前的覆盖率数据，开始新的覆盖率统计
        cov.erase()
        cov.start()
        # 导入 sympy 模块，并运行测试（非子进程方式，不包含慢速测试）
        import sympy
        sympy.test(*test_args, subprocess=False, slow=slow)
        # 停止覆盖率统计
        cov.stop()
        # 尝试保存覆盖率数据，如果出现权限错误则发出警告
        try:
            cov.save()
        except PermissionError:
            import warnings
            warnings.warn(
                "PermissionError has been raised while saving the " \
                "coverage result.",
                RuntimeWarning
            )

    # 获取所有被覆盖的源文件列表
    covered_files = list(generate_covered_files(source_dir))
    # 生成 HTML 格式的覆盖率报告
    cov.html_report(morfs=covered_files, directory=report_dir)


# 创建命令行参数解析器
parser = ArgumentParser()
# 添加命令行参数选项 -c/--use-cache，表示使用缓存数据
parser.add_argument(
    '-c', '--use-cache', action='store_true', default=False,
    help='Use cached data.')
# 添加命令行参数选项 -d/--report-dir，表示指定报告输出目录
parser.add_argument(
    '-d', '--report-dir', default='covhtml',
    help='Directory to put the generated report in.')
# 添加命令行参数选项
    # 添加一个命令行选项 "--slow"，当存在时将其设置为 True，否则默认为 False
    "--slow", action="store_true", dest="slow", default=False,
    # 设置帮助信息，说明 "--slow" 选项的作用
    help="Run slow functions also.")
# 解析命令行参数，将选项解析为 options 对象，剩余的参数解析为 args 列表
options, args = parser.parse_known_args()

# 如果脚本作为主程序执行
if __name__ == '__main__':
    # 从 options 对象中获取报告目录
    report_dir = options.report_dir
    # 从 options 对象中获取是否使用缓存的选项
    use_cache = options.use_cache
    # 从 options 对象中获取是否执行慢速模式的选项
    slow = options.slow
    # 调用 make_report 函数生成报告，传入参数 args，报告目录、缓存使用选项和慢速模式选项
    make_report(
        args, report_dir=report_dir, use_cache=use_cache, slow=slow)

    # 打印生成的覆盖率报告在 covhtml 目录下
    print("The generated coverage report is in covhtml directory.")
    # 打印提示信息，指示用户在浏览器中打开报告
    print(
        "Open %s in your web browser to view the report" %
        os.sep.join([report_dir, 'index.html'])
    )
```