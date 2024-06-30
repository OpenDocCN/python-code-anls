# `D:\src\scipysrc\scipy\tools\unicode-check.py`

```
#!/usr/bin/env python

# 导入正则表达式模块
import re
# 导入chain函数用于串联迭代器，导入iglob函数用于查找文件
from itertools import chain
from glob import iglob
# 导入sys模块和argparse模块
import sys
import argparse

# 允许在源代码中出现的大于127的Unicode码点集合
latin1_letters = set(chr(cp) for cp in range(192, 256))
# ASCII绘图字符集合
box_drawing_chars = set(chr(cp) for cp in range(0x2500, 0x2580))
# 其他特定符号集合
extra_symbols = set(['®', 'ő', 'λ', 'π', 'ω', '∫', '≠', '≥', '≤', 'μ'])
# 所有允许的字符集合
allowed = latin1_letters | box_drawing_chars | extra_symbols

# 检查Unicode字符的函数，如果showall为True，则显示所有非ASCII字符
def unicode_check(showall=False):
    """
    If showall is True, all non-ASCII characters are displayed.
    """
    # PEP-263定义的文件编码正则表达式
    encoding_pat = re.compile("^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)")

    # 非ASCII字符计数器
    nbad = 0
    # 遍历所有匹配的文件
    for name in chain(iglob('scipy/**/*.py', recursive=True),
                      iglob('scipy/**/*.pyx', recursive=True),
                      iglob('scipy/**/*.px[di]', recursive=True)):
        # 以字节形式读取文件内容，并检查是否有大于127的字节
        with open(name, 'rb') as f:
            content = f.read()
        # 如果内容为空则跳过
        if len(content) == 0:
            continue
        # 如果存在大于127的字节
        if max(content) > 127:
            # 文件中至少有一个非ASCII字符
            # 检查前两行是否有编码注释
            lines = content.splitlines()
            for line in lines[:2]:
                match = re.match(encoding_pat,
                                 line.decode(encoding='latin-1'))
                if match:
                    break

            # 如果发现了编码注释，则使用该编码解码内容；否则使用UTF-8
            if match:
                encoding = match[1]
                file_enc_msg = f"(explicit encoding '{encoding}')"
            else:
                encoding = 'utf-8'
                file_enc_msg = "(no explicit encoding; utf-8 assumed)"
            # 解码文件内容
            content = content.decode(encoding=encoding)

            # 输出结果列表
            out = []
            # 遍历内容的每一行
            for n, line in enumerate(content.splitlines()):
                for pos, char in enumerate(line):
                    cp = ord(char)
                    # 如果字符大于127
                    if cp > 127:
                        msg = (f"... line {n+1}, position {pos+1}: "
                               f"character '{char}', code point U+{cp:04X}")
                        # 如果showall为True，则添加所有非ASCII字符的信息
                        if showall:
                            out.append(msg)
                        else:
                            # 如果字符不在允许的字符集合中，则添加信息
                            if char not in allowed:
                                out.append(msg)
            # 如果有输出结果，则增加不良文件计数
            if len(out) > 0:
                nbad += 1
                print(f"{name} {file_enc_msg}")
                # 输出每一个不良信息
                for msg in out:
                    print(msg)
    # 返回不良文件计数
    return nbad

# 如果该脚本作为主程序运行
if __name__ == "__main__":
    descr = ('Check for disallowed Unicode characters in the SciPy Python and '
             ' Cython source code.')
    # 创建参数解析器
    parser = argparse.ArgumentParser(description=descr)
    # 添加一个命令行参数 '--showall'，当存在时将其值设为 True，用于指示是否显示所有文件中的非ASCII Unicode字符
    parser.add_argument('--showall', action='store_true',
                        help=('Show non-ASCII Unicode characters from all '
                              'files.'))
    # 解析命令行参数，并将结果保存在 args 变量中
    args = parser.parse_args()
    # 调用 unicode_check 函数，传入 showall 参数的值，并检查返回值是否大于 0
    # 如果 unicode_check 返回值大于 0，则以该值作为退出码退出程序
    sys.exit(unicode_check(args.showall) > 0)
```