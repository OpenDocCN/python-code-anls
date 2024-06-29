# `.\numpy\doc\postprocess.py`

```py
#!/usr/bin/env python3
"""
Post-processes HTML and Latex files output by Sphinx.
"""

# 主程序入口
def main():
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description=__doc__)
    # 添加参数：模式（html 或 tex）
    parser.add_argument('mode', help='file mode', choices=('html', 'tex'))
    # 添加参数：输入文件列表
    parser.add_argument('file', nargs='+', help='input file(s)')
    # 解析命令行参数
    args = parser.parse_args()

    # 获取模式参数
    mode = args.mode

    # 遍历每个输入文件
    for fn in args.file:
        # 打开文件并读取内容
        with open(fn, encoding="utf-8") as f:
            # 根据模式选择处理函数处理文件内容
            if mode == 'html':
                lines = process_html(fn, f.readlines())
            elif mode == 'tex':
                lines = process_tex(f.readlines())

        # 将处理后的内容写回文件
        with open(fn, 'w', encoding="utf-8") as f:
            f.write("".join(lines))

# 处理 HTML 文件内容的函数
def process_html(fn, lines):
    return lines

# 处理 LaTeX 文件内容的函数
def process_tex(lines):
    """
    Remove unnecessary section titles from the LaTeX file.
    移除 LaTeX 文件中不必要的章节标题。
    """
    new_lines = []
    # 遍历每一行内容
    for line in lines:
        # 如果是以特定的 numpy 相关标题开头，则跳过这一行
        if (line.startswith(r'\section{numpy.')
            or line.startswith(r'\subsection{numpy.')
            or line.startswith(r'\subsubsection{numpy.')
            or line.startswith(r'\paragraph{numpy.')
            or line.startswith(r'\subparagraph{numpy.')
            ):
            pass # 跳过这些行！
        else:
            # 否则将这一行添加到新的内容列表中
            new_lines.append(line)
    return new_lines

# 如果作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()
```