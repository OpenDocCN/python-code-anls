# `.\marker\chunk_convert.py`

```py
# 导入 argparse 模块，用于解析命令行参数
import argparse
# 导入 subprocess 模块，用于执行外部命令
import subprocess

# 定义主函数
def main():
    # 创建 ArgumentParser 对象，设置描述信息
    parser = argparse.ArgumentParser(description="Convert a folder of PDFs to a folder of markdown files in chunks.")
    # 添加命令行参数，指定输入文件夹路径
    parser.add_argument("in_folder", help="Input folder with pdfs.")
    # 添加命令行参数，指定输出文件夹路径
    parser.add_argument("out_folder", help="Output folder")
    # 解析命令行参数
    args = parser.parse_args()

    # 构造要执行的 shell 命令
    cmd = f"./chunk_convert.sh {args.in_folder} {args.out_folder}"

    # 执行 shell 脚本
    subprocess.run(cmd, shell=True, check=True)

# 如果当前脚本作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
```