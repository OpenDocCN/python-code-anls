# `D:\src\scipysrc\scikit-learn\sklearn\_build_utils\tempita.py`

```
import argparse
import os

from Cython import Tempita as tempita

# XXX: 如果这个导入失败会怎样？请决定是否要提供 cython.tempita 或 numpy/npy_tempita.


def process_tempita(fromfile, outfile=None):
    """处理 tempita 模板文件并将结果写入到输出文件中。

    模板文件应以 `.c.tp` 或 `.pyx.tp` 结尾：
    例如，处理 `template.c.in` 将生成 `template.c`。

    """
    # 打开模板文件，读取内容
    with open(fromfile, "r", encoding="utf-8") as f:
        template_content = f.read()

    # 使用 Tempita 库加载模板内容
    template = tempita.Template(template_content)
    # 替换模板中的变量，获取最终生成的内容
    content = template.substitute()

    # 将生成的内容写入到输出文件
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加输入文件参数，类型为字符串
    parser.add_argument("infile", type=str, help="Path to the input file")
    # 添加输出目录参数
    parser.add_argument("-o", "--outdir", type=str, help="Path to the output directory")
    # 添加忽略参数，用于增加自定义目标之间的依赖性
    parser.add_argument(
        "-i",
        "--ignore",
        type=str,
        help=(
            "An ignored input - may be useful to add a "
            "dependency between custom targets"
        ),
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 检查输入文件是否以 `.tp` 结尾
    if not args.infile.endswith(".tp"):
        raise ValueError(f"Unexpected extension: {args.infile}")

    # 检查是否提供了输出目录参数
    if not args.outdir:
        raise ValueError("Missing `--outdir` argument to tempita.py")

    # 获取输出目录的绝对路径
    outdir_abs = os.path.join(os.getcwd(), args.outdir)
    # 构建输出文件的完整路径
    outfile = os.path.join(
        outdir_abs, os.path.splitext(os.path.split(args.infile)[1])[0]
    )

    # 调用处理函数，处理模板文件
    process_tempita(args.infile, outfile)


if __name__ == "__main__":
    # 执行主函数
    main()
```