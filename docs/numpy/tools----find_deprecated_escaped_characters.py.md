# `.\numpy\tools\find_deprecated_escaped_characters.py`

```
#!/usr/bin/env python3
r"""
Look for escape sequences deprecated in Python 3.6.

Python 3.6 deprecates a number of non-escape sequences starting with '\' that
were accepted before. For instance, '\(' was previously accepted but must now
be written as '\\(' or r'\('.

"""

# 主函数，用于查找已弃用的转义序列
def main(root):
    """Find deprecated escape sequences.

    Checks for deprecated escape sequences in ``*.py files``. If `root` is a
    file, that file is checked, if `root` is a directory all ``*.py`` files
    found in a recursive descent are checked.

    If a deprecated escape sequence is found, the file and line where found is
    printed. Note that for multiline strings the line where the string ends is
    printed and the error(s) are somewhere in the body of the string.

    Parameters
    ----------
    root : str
        File or directory to check.
    Returns
    -------
    None

    """
    import ast  # 导入抽象语法树模块
    import tokenize  # 导入 tokenize 模块
    import warnings  # 导入警告模块
    from pathlib import Path  # 从 pathlib 模块导入 Path 类

    count = 0  # 初始化计数器为 0
    base = Path(root)  # 将输入的根路径转换为 Path 对象
    paths = base.rglob("*.py") if base.is_dir() else [base]  # 获取所有以 .py 结尾的文件路径列表
    for path in paths:
        # 使用 tokenize 打开文件，自动检测编码
        with tokenize.open(str(path)) as f:
            # 使用警告模块捕获警告
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')  # 设置警告过滤器
                tree = ast.parse(f.read())  # 解析文件内容为抽象语法树
            if w:
                print("file: ", str(path))  # 打印文件路径
                for e in w:
                    print('line: ', e.lineno, ': ', e.message)  # 打印警告所在行及消息
                print()
                count += len(w)  # 计算警告数量
    print("Errors Found", count)  # 打印总共找到的错误数量


if __name__ == "__main__":
    from argparse import ArgumentParser  # 从 argparse 模块导入 ArgumentParser 类

    parser = ArgumentParser(description="Find deprecated escaped characters")
    parser.add_argument('root', help='directory or file to be checked')  # 添加命令行参数 root
    args = parser.parse_args()  # 解析命令行参数
    main(args.root)  # 调用主函数进行检查
```