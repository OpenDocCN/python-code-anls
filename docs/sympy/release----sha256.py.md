# `D:\src\scipysrc\sympy\release\sha256.py`

```
#!/usr/bin/env python3

import os  # 导入标准库 os，用于操作文件系统
from pathlib import Path  # 导入 pathlib 库中的 Path 类，用于处理文件路径
from subprocess import check_output  # 导入 subprocess 库中的 check_output 函数，用于执行外部命令并获取输出结果

def main(version, outdir):
    outdir = Path(outdir)  # 将输出目录路径转换为 Path 对象
    build_files = [
        outdir / f'sympy-{version}.tar.gz',  # 构建 sympy 版本的压缩包路径
        outdir / f'sympy-{version}-py3-none-any.whl',  # 构建 sympy 版本的 wheel 文件路径
        outdir / f'sympy-docs-html-{version}.zip',  # 构建 sympy 版本的 HTML 文档压缩包路径
        outdir / f'sympy-docs-pdf-{version}.pdf',  # 构建 sympy 版本的 PDF 文档路径
    ]
    out = check_output(['shasum', '-a', '256'] + build_files)  # 执行 shasum 命令计算指定文件的 SHA-256 值
    out = out.decode('ascii')  # 将字节流解码为 ASCII 字符串
    
    # 移除输出结果中文件路径的 release/ 部分，方便复制到发布说明中
    out = [i.split() for i in out.strip().split('\n')]
    out = '\n'.join(["%s\t%s" % (i, os.path.split(j)[1]) for i, j in out])

    # 将结果输出到文件和屏幕上
    with open(outdir / 'sha256.txt', 'w') as shafile:  # 打开 sha256.txt 文件以写入输出结果
        shafile.write(out)
    print(out)  # 打印输出结果到标准输出流


if __name__ == "__main__":
    import sys
    sys.exit(main(*sys.argv[1:]))  # 解析命令行参数并调用 main 函数处理
```