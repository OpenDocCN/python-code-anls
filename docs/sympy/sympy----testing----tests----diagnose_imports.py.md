# `D:\src\scipysrc\sympy\sympy\testing\tests\diagnose_imports.py`

```
#!/usr/bin/env python

"""
Import diagnostics. Run bin/diagnose_imports.py --help for details.
"""

# 引入未来版本的注解支持
from __future__ import annotations

if __name__ == "__main__":

    # 引入系统模块
    import sys
    # 提供检查函数、类、方法、模块等功能的访问
    import inspect
    # 提供内建函数和异常的访问
    import builtins

    # 引入命令行选项解析模块
    import optparse

    # 从os.path模块中引入几个特定的函数
    from os.path import abspath, dirname, join, normpath
    # 获取当前文件的绝对路径
    this_file = abspath(__file__)
    # 构造Sympy目录的路径
    sympy_dir = join(dirname(this_file), '..', '..', '..')
    # 规范化Sympy目录的路径
    sympy_dir = normpath(sympy_dir)
    # 将Sympy目录添加到系统路径中，使得Python解释器能够找到该目录下的模块
    sys.path.insert(0, sympy_dir)

    # 创建命令行选项解析器对象
    option_parser = optparse.OptionParser(
        usage=
            "Usage: %prog option [options]\n"
            "\n"
            "Import analysis for imports between SymPy modules.")
    
    # 创建命令行选项组对象，定义分析选项
    option_group = optparse.OptionGroup(
        option_parser,
        'Analysis options',
        'Options that define what to do. Exactly one of these must be given.')
    
    # 添加--problems选项，描述打印所有导入问题的功能
    option_group.add_option(
        '--problems',
        help=
            'Print all import problems, that is: '
            'If an import pulls in a package instead of a module '
            '(e.g. sympy.core instead of sympy.core.add); ' # see ##PACKAGE##
            'if it imports a symbol that is already present; ' # see ##DUPLICATE##
            'if it imports a symbol '
            'from somewhere other than the defining module.', # see ##ORIGIN##
        action='count')
    
    # 添加--origins选项，描述打印每个模块中每个导入符号的定义模块的功能
    option_group.add_option(
        '--origins',
        help=
            'For each imported symbol in each module, '
            'print the module that defined it. '
            '(This is useful for import refactoring.)',
        action='count')
    
    # 将分析选项组添加到选项解析器中
    option_parser.add_option_group(option_group)
    
    # 创建排序选项组对象，定义排序选项
    option_group = optparse.OptionGroup(
        option_parser,
        'Sort options',
        'These options define the sort order for output lines. '
        'At most one of these options is allowed. '
        'Unsorted output will reflect the order in which imports happened.')
    
    # 添加--by-importer选项，描述按导入模块名称排序输出的功能
    option_group.add_option(
        '--by-importer',
        help='Sort output lines by name of importing module.',
        action='count')
    
    # 添加--by-origin选项，描述按导入模块的定义模块名称排序输出的功能
    option_group.add_option(
        '--by-origin',
        help='Sort output lines by name of imported module.',
        action='count')
    
    # 将排序选项组添加到选项解析器中
    option_parser.add_option_group(option_group)
    
    # 解析命令行参数
    (options, args) = option_parser.parse_args()
    
    # 检查是否有未预期的位置参数，如果有则显示错误信息
    if args:
        option_parser.error(
            'Unexpected arguments %s (try %s --help)' % (args, sys.argv[0]))
    
    # 检查--problems选项是否出现多次，如果是则显示错误信息
    if options.problems > 1:
        option_parser.error('--problems must not be given more than once.')
    
    # 检查--origins选项是否出现多次，如果是则显示错误信息
    if options.origins > 1:
        option_parser.error('--origins must not be given more than once.')
    
    # 检查--by-importer选项是否出现多次，如果是则显示错误信息
    if options.by_importer > 1:
        option_parser.error('--by-importer must not be given more than once.')
    
    # 检查--by-origin选项是否出现多次，如果是则显示错误信息
    if options.by_origin > 1:
        option_parser.error('--by-origin must not be given more than once.')
    
    # 将选项转换为布尔值以便后续处理
    options.problems = options.problems == 1
    options.origins = options.origins == 1
    options.by_importer = options.by_importer == 1
    options.by_origin = options.by_origin == 1
    # 如果既没有指定 --problems 也没有指定 --origins，则抛出错误提示
    if not options.problems and not options.origins:
        option_parser.error(
            'At least one of --problems and --origins is required')
    
    # 如果同时指定了 --problems 和 --origins，则抛出错误提示
    if options.problems and options.origins:
        option_parser.error(
            'At most one of --problems and --origins is allowed')
    
    # 如果同时指定了 --by-importer 和 --by-origin，则抛出错误提示
    if options.by_importer and options.by_origin:
        option_parser.error(
            'At most one of --by-importer and --by-origin is allowed')
    
    # 根据 --by-importer 和 --by-origin 的值确定是否设置了 --by-process
    options.by_process = not options.by_importer and not options.by_origin

    # 保存内建的 __import__ 函数的引用
    builtin_import = builtins.__import__

    class Definition:
        """Information about a symbol's definition."""
        def __init__(self, name, value, definer):
            self.name = name
            self.value = value
            self.definer = definer
        def __hash__(self):
            return hash(self.name)
        def __eq__(self, other):
            return self.name == other.name and self.value == other.value
        def __ne__(self, other):
            return not (self == other)
        def __repr__(self):
            return 'Definition(%s, ..., %s)' % (
                repr(self.name), repr(self.definer))

    # 定义一个字典，将每个函数或变量的定义映射到定义它的模块的名称
    symbol_definers: dict[Definition, str] = {}

    def in_module(a, b):
        """Is a the same module as or a submodule of b?"""
        return a == b or a != None and b != None and a.startswith(b + '.')

    def relevant(module):
        """Is module relevant for import checking?

        Only imports between relevant modules will be checked."""
        # 判断模块是否与 'sympy' 相同或者是 'sympy' 的子模块
        return in_module(module, 'sympy')

    # 存储排序后的消息
    sorted_messages = []

    def msg(msg, *args):
        global options, sorted_messages
        # 如果使用 --by-process 标志，则直接打印消息
        if options.by_process:
            print(msg % args)
        else:
            # 否则将消息添加到排序消息列表中
            sorted_messages.append(msg % args)

    # 替换内建的 __import__ 函数为 tracking_import 函数
    builtins.__import__ = tracking_import
    
    # 动态导入 'sympy' 模块
    __import__('sympy')

    # 对排序后的消息进行排序
    sorted_messages.sort()
    
    # 打印每条排序后的消息
    for message in sorted_messages:
        print(message)
```