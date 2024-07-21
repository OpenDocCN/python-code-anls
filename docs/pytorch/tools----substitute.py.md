# `.\pytorch\tools\substitute.py`

```py
# 导入 argparse 模块，用于解析命令行参数
import argparse
# 导入 os 模块，提供与操作系统交互的功能
import os
# 导入 os.path 模块，用于处理文件路径相关操作
import os.path


# 定义主函数入口
def main() -> None:
    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加命令行参数 --input-file，用于指定输入文件路径
    parser.add_argument("--input-file")
    # 添加命令行参数 --output-file，用于指定输出文件名
    parser.add_argument("--output-file")
    # 添加命令行参数 --install-dir 或 --install_dir，用于指定安装目录路径
    parser.add_argument("--install-dir", "--install_dir")
    # 添加命令行参数 --replace，允许多次出现，每次包含两个参数
    parser.add_argument("--replace", action="append", nargs=2)
    # 解析命令行参数，将解析结果存储在 options 中
    options = parser.parse_args()

    # 打开输入文件 options.input_file，并读取其内容
    with open(options.input_file) as f:
        contents = f.read()

    # 构建输出文件的完整路径，路径为 options.install_dir 下的 options.output_file
    output_file = os.path.join(options.install_dir, options.output_file)
    # 确保输出文件所在的目录存在，如果不存在则创建
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 遍历替换规则 options.replace 中的每对 (old, new)
    for old, new in options.replace:
        # 在 contents 中执行字符串替换，将 old 替换为 new
        contents = contents.replace(old, new)

    # 将替换后的内容写入到输出文件 output_file 中
    with open(output_file, "w") as f:
        f.write(contents)


# 如果当前脚本作为主程序运行，则调用主函数 main()
if __name__ == "__main__":
    main()  # pragma: no cover
```