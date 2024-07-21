# `.\pytorch\tools\setup_helpers\generate_linker_script.py`

```py
import subprocess  # 导入subprocess模块，用于执行外部命令

def gen_linker_script(
    filein: str = "cmake/prioritized_text.txt", fout: str = "cmake/linker_script.ld"
) -> None:
    with open(filein) as f:
        prioritized_text = f.readlines()  # 读取指定文件中的所有行
        prioritized_text = [
            line.replace("\n", "") for line in prioritized_text if line != "\n"
        ]  # 去除每行末尾的换行符，并过滤空行

    # 执行 ld 命令，获取链接器脚本的详细输出，并将其分割成行列表
    linker_script_lines = subprocess.check_output(["ld", "-verbose"], text=True).split(
        "\n"
    )

    # 查找具有特定标志的行索引，用于截取链接器脚本的相关部分
    indices = [
        i
        for i, x in enumerate(linker_script_lines)
        if x == "=================================================="
    ]
    linker_script_lines = linker_script_lines[indices[0] + 1 : indices[1]]  # 截取感兴趣的部分

    # 查找链接器脚本中以 ".text           :" 开头的行的索引
    text_line_start = [
        i for i, line in enumerate(linker_script_lines) if ".text           :" in line
    ]
    assert len(text_line_start) == 1, "The linker script has multiple text sections!"  # 确保只有一个文本段
    text_line_start = text_line_start[0]

    # 将生成的链接器脚本写入指定的输出文件
    with open(fout, "w") as f:
        for lineid, line in enumerate(linker_script_lines):
            if lineid == text_line_start + 2:
                f.write("    *(\n")  # 写入开始符号
                for plines in prioritized_text:
                    f.write(f"      .text.{plines}\n")  # 按照优先级文本列表写入对应行
                f.write("    )\n")  # 写入结束符号
            f.write(f"{line}\n")  # 将链接器脚本的每一行写入输出文件
```