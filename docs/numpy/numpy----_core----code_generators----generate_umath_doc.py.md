# `.\numpy\numpy\_core\code_generators\generate_umath_doc.py`

```py
# 导入必要的模块
import sys  # 系统相关的操作
import os   # 系统路径操作
import textwrap  # 文本包装模块，用于处理文档字符串格式
import argparse  # 命令行参数解析模块

# 将当前文件所在目录加入到系统路径中
sys.path.insert(0, os.path.dirname(__file__))
import ufunc_docstrings as docstrings  # 导入自定义模块 ufunc_docstrings，并命名为 docstrings
sys.path.pop(0)  # 移除刚才添加的路径，保持环境干净

# 格式化文档字符串，使其符合特定格式
def normalize_doc(docstring):
    docstring = textwrap.dedent(docstring).strip()  # 去除文档字符串的缩进并去除首尾空白
    docstring = docstring.encode('unicode-escape').decode('ascii')  # 将非 ASCII 字符转义为 Unicode 转义序列
    docstring = docstring.replace(r'"', r'\"')  # 转义双引号
    docstring = docstring.replace(r"'", r"\'")  # 转义单引号
    # 将文档字符串按换行符分割，重新连接成多行字符串，以避免某些编译器不喜欢过长的 C 代码字符串字面量
    docstring = '\\n\"\"'.join(docstring.split(r"\n"))
    return docstring

# 写入生成的 C 代码到目标文件
def write_code(target):
    with open(target, 'w') as fid:
        fid.write(
            "#ifndef NUMPY_CORE_INCLUDE__UMATH_DOC_GENERATED_H_\n"
            "#define NUMPY_CORE_INCLUDE__UMATH_DOC_GENERATED_H_\n"
        )
        # 遍历文档字符串字典，生成 C 代码宏定义
        for place, string in docstrings.docdict.items():
            cdef_name = f"DOC_{place.upper().replace('.', '_')}"
            cdef_str = normalize_doc(string)
            fid.write(f"#define {cdef_name} \"{cdef_str}\"\n")
        fid.write("#endif //NUMPY_CORE_INCLUDE__UMATH_DOC_GENERATED_H\n")

# 主函数，用于命令行参数解析和调用写入函数
def main():
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="Path to the output directory"
    )
    args = parser.parse_args()  # 解析命令行参数

    outfile = os.path.join(os.getcwd(), args.outfile)  # 获取输出文件路径
    write_code(outfile)  # 调用写入代码函数

# 如果当前脚本作为主程序运行，则执行 main 函数
if __name__ == '__main__':
    main()
```