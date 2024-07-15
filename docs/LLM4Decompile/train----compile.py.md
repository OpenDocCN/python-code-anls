# `.\LLM4Decompile\train\compile.py`

```py
import glob  # 导入 glob 模块，用于文件路径的匹配
import json  # 导入 json 模块，用于 JSON 数据的处理
import subprocess  # 导入 subprocess 模块，用于执行外部命令
import os  # 导入 os 模块，用于操作系统相关功能
import multiprocessing  # 导入 multiprocessing 模块，用于多进程处理
import re  # 导入 re 模块，用于正则表达式操作
import argparse  # 导入 argparse 模块，用于命令行参数解析

zeros_pattern = r"^0+\s"  # 定义匹配连续 0 开头的行的正则表达式模式
OPT = ["O0", "O1", "O2", "O3"]  # 优化级别列表

def compile_and_write(input_file, output_file):
    base_output_file = input_file.replace(".c", "")  # 根据输入文件生成基础输出文件名
    asm_all = {}  # 初始化汇编代码字典
    input_text = open(input_file).read()  # 读取输入文件内容

    # 处理输入文本，排除宏和类型声明
    if "/* Variables and functions */" in input_text:
        input_text = input_text.split("/* Variables and functions */")[-1]
        input_text = "\n\n".join(input_text.split("\n\n")[1:])  # 排除变量声明

        # 移除 __attribute__ 属性
        input_text = input_text.replace("__attribute__((used)) ", "")

    try:
        for opt_state in OPT:  # 遍历优化级别列表
            obj_output = base_output_file + "_" + opt_state + ".o"  # 定义目标文件名
            asm_output = base_output_file + "_" + opt_state + ".s"  # 定义汇编文件名

            # 编译 C 程序到目标文件
            subprocess.run(
                ["gcc", "-c", "-o", obj_output, input_file, "-" + opt_state],
                check=True,  # 如果编译失败则抛出异常
            )

            # 使用 objdump 生成目标文件的汇编代码
            subprocess.run(
                f"objdump -d {obj_output} > {asm_output}",
                shell=True,  # 使用 shell 进行重定向操作
                check=True,  # 如果生成失败则抛出异常
            )

            with open(asm_output) as f:
                asm = f.read()  # 读取汇编文件内容

                # 清理汇编代码，移除二进制代码和注释
                asm_clean = ""
                asm = asm.split("Disassembly of section .text:")[-1].strip()
                for tmp in asm.split("\n"):
                    tmp_asm = tmp.split("\t")[-1]  # 移除二进制代码
                    tmp_asm = tmp_asm.split("#")[0].strip()  # 移除注释
                    asm_clean += tmp_asm + "\n"
                if len(asm_clean.split("\n")) < 4:  # 如果汇编代码行数小于 4 行，则抛出异常
                    raise ValueError("compile fails")
                asm = asm_clean

                # 过滤连续 0 开头的行和 __attribute__ 属性
                asm = re.sub(zeros_pattern, "", asm)
                asm = asm.replace("__attribute__((used)) ", "")

                # 将汇编代码存入汇编代码字典中
                asm_all["opt-state-" + opt_state] = asm

            # 移除目标文件
            if os.path.exists(obj_output):
                os.remove(obj_output)

    except Exception as e:  # 捕获异常
        print(f"Error in file {input_file}: {e}")  # 打印错误信息
        return  # 返回

    finally:  # 无论是否发生异常，都会执行的代码块
        # 移除汇编输出文件
        for opt_state in OPT:
            asm_output = base_output_file + "_" + opt_state + ".s"
            if os.path.exists(asm_output):
                os.remove(asm_output)

    # 构建样本数据字典
    sample = {
        "name": input_file,  # 文件名
        "input": input_text,  # 使用处理后的输入文本
        "input_ori": open(input_file).read(),  # 原始输入文本
        "output": asm_all,  # 使用汇编代码字典
    }
    # 将数据写入文件
    write_to_file(output_file, sample)
# 定义一个函数，用于将数据写入文件
def write_to_file(file_path, data):
    # 使用 multiprocessing.Lock() 创建一个进程锁，确保多进程下的文件写入安全
    with multiprocessing.Lock():
        # 以追加模式打开文件，准备写入数据
        with open(file_path, "a") as f:
            # 将数据以 JSON 格式写入文件
            json.dump(data, f)
            # 在数据末尾写入换行符
            f.write("\n")


# 解析命令行参数的函数
def parse_args():
    # 创建参数解析器对象，设置描述信息
    parser = argparse.ArgumentParser(
        description="Compile C files and generate JSONL output."
    )
    # 添加命令行参数 --root，指定AnghaBench文件所在的根目录
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory where AnghaBench files are located.",
    )
    # 添加命令行参数 --output，指定生成的JSONL输出文件的路径
    parser.add_argument("--output", required=True, help="Path to JSONL output file.")
    # 解析命令行参数并返回解析结果
    args = parser.parse_args()
    return args


# 主函数
def main():
    # 解析命令行参数
    args = parse_args()
    # 获取 AnghaBench 文件所在的根目录
    root = args.root
    # 获取 JSONL 输出文件的路径
    jsonl_output_file = args.output
    # 使用 glob 模块查找根目录下所有子目录中的 .c 文件
    files = glob.glob(f"{root}/**/*.c", recursive=True)

    # 使用 multiprocessing.Pool 创建一个进程池，最大并发数为 32
    with multiprocessing.Pool(32) as pool:
        # 导入 functools 模块中的 partial 函数，创建一个带默认参数的函数
        compile_write_func = partial(compile_and_write, output_file=jsonl_output_file)
        # 对文件列表中的每个文件，调用 compile_write_func 函数进行编译并写入操作，使用进程池并发处理
        pool.map(compile_write_func, files)


# 如果该脚本作为主程序运行，则执行主函数
if __name__ == "__main__":
    main()
```