# `D:\src\scipysrc\scipy\tools\refguide_summaries.py`

```
#!/usr/bin/env python
"""Generate function summaries for the refguide. For example, if the
__init__ file of a submodule contains:

.. autosummary::
   :toctree: generated/

   foo
   foobar

Then it will modify the __init__ file to contain (*)

.. autosummary::
   :toctree: generated/

   foo    -- First line of the documentation of `foo`.
   foobar -- First line of the documentation of `foobar`.

If there is already text after the function definitions it will be
overwritten, i.e.

.. autosummary::
   :toctree: generated/

   foo    -- Blah blah blah.
   foobar -- Blabbity blabbity.

will also become (*).

"""

import os                   # 导入操作系统相关模块
import argparse             # 导入命令行参数解析模块
import importlib            # 导入模块动态加载模块
import re                   # 导入正则表达式模块


EXCEPTIONS = {              # 定义异常说明字典
    'jn': ('Bessel function of the first kind of real order and '
           'complex argument')
}


def main():
    parser = argparse.ArgumentParser()    # 创建参数解析器对象
    parser.add_argument("module",          # 添加位置参数，指定要添加摘要的模块
                        help="module to add summaries to")
    parser.add_argument("--dry-run",       # 添加可选参数，打印而不是覆盖 __init__ 文件
                        help="print __init__ file instead of overwriting",
                        action="store_true")
    args = parser.parse_args()             # 解析命令行参数

    # 构建要操作的文件路径，基于当前文件所在目录的相对路径
    filename = os.path.join(os.path.dirname(__file__), '..', 'scipy',
                            args.module, '__init__.py')

    # 动态加载要操作的模块
    module = importlib.import_module('scipy.' + args.module)

    fnew = []  # 初始化一个空列表用于存储处理后的文件内容
    # 使用文件名打开文件，并将文件对象赋值给变量 f
    with open(filename) as f:
        # 读取文件的第一行并赋值给变量 line
        line = f.readline()
        # 循环直到文件末尾
        while line:
            # 检查当前行是否包含 '.. autosummary::'
            if '.. autosummary::' in line:
                # 将当前行去除右侧空白字符后添加到列表 fnew 中
                fnew.append(line.rstrip())
                # 将下一行去除右侧空白字符后添加到列表 fnew 中，表示目录生成指令
                fnew.append(f.readline().rstrip())  # :toctree: generated/
                # 将接下来的空白行添加到列表 fnew 中
                fnew.append(f.readline().rstrip())  # blank line
                # 继续读取下一行内容
                line = f.readline()
                # 初始化空列表 summaries 和最大长度 maxlen
                summaries = []
                maxlen = 0
                # 循环直到遇到空行
                while line.strip():
                    # 提取函数名，并检查是否有 '[+]' 标记
                    func = line.split('--')[0].strip()
                    ufunc = '[+]' not in line
                    # 更新最大函数名长度
                    if len(func) > maxlen:
                        maxlen = len(func)

                    # 如果函数名在 EXCEPTIONS 字典中，则使用其注释
                    if func in EXCEPTIONS.keys():
                        summary = [EXCEPTIONS[func]]
                    else:
                        # 否则，获取函数对象的文档字符串，并提取相关注释内容
                        summary = []
                        doc = getattr(module, func).__doc__.split('\n')
                        i = 0 if doc[0].strip() else 1
                        while True:
                            # 如果文档字符串包含函数签名，则跳过
                            if re.match(func + r'\(.*\)', doc[i].strip()):
                                i += 2
                            else:
                                break
                        # 读取非空行的注释内容
                        while i < len(doc) and doc[i].strip():
                            summary.append(doc[i].lstrip())
                            i += 1

                    # 格式化注释内容，并根据 ufunc 标志决定是否加入 '[+]' 标记
                    summary = ' '.join([x.lstrip() for x in summary])
                    summary = '[+]' + summary if not ufunc else summary
                    # 将函数名及其注释添加到 summaries 列表中
                    summaries.append((func, summary))
                    # 继续读取下一行内容
                    line = f.readline()
                
                # 将每个函数及其注释按格式添加到列表 fnew 中
                for (func, summary) in summaries:
                    spaces = ' '*(maxlen - len(func) + 1)
                    fnew.append('   ' + func + spaces + '-- ' + summary)
                # 将最后读取的空行或非目录内容行添加到 fnew 中
                fnew.append(line.rstrip())
            else:
                # 如果当前行不包含 '.. autosummary::'，直接添加到 fnew 中
                fnew.append(line.rstrip())
            # 继续读取下一行内容
            line = f.readline()

    # 如果设置了 dry_run 参数，则打印处理后的文本内容
    if args.dry_run:
        print('\n'.join(fnew))
    else:
        # 否则，将处理后的文本内容写回到原文件中
        with open(filename, 'w') as f:
            f.write('\n'.join(fnew))
            f.write('\n')
# 如果当前脚本被直接执行（而不是被导入到其他模块中执行），那么执行以下代码块
if __name__ == "__main__":
    # 调用主函数 main()
    main()
```