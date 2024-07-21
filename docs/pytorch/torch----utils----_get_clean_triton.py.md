# `.\pytorch\torch\utils\_get_clean_triton.py`

```py
# mypy: allow-untyped-defs
# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import os  # 提供操作系统相关的功能
import re  # 提供正则表达式操作支持
from pathlib import Path  # 提供处理路径的类和函数
from typing import Dict, List  # 提供静态类型检查所需的类型注解


def remove_triton_function_declaration(source_code: str) -> str:
    # 移除源代码中以三个单引号（'''）开头的函数声明
    remove_head = re.sub(r"(\n.+\s\'\'\'\n)", "\n", source_code)
    # 移除源代码中以三个单引号（'''）结尾的函数声明的剩余部分
    remove_tail = re.sub(r"(\'\'\'\,.+)", "\n", remove_head)
    return remove_tail


def remove_async_compile(source_code: str) -> str:
    # 从源代码中移除顶层的 async_compile = AsyncCompile() 声明
    remove_top_level = str.replace(source_code, "async_compile = AsyncCompile()", "")
    # 从源代码中移除 async_compile.wait(globals()) 的调用
    remove_compile = str.replace(remove_top_level, "async_compile.wait(globals())", "")
    # 从源代码中删除 async_compile 变量的删除语句
    remove_del = str.replace(remove_compile, "del async_compile", "")
    return remove_del


def rename_kernels(source_code: str) -> str:
    # 匹配 async_compile.triton('triton_', 后的内核声明，并捕获函数名
    pattern = r"(\w+)\s*=\s*async_compile\.triton\('triton_',\s"
    triton_kernel_decl = "def triton_"
    # 找到所有匹配的内核声明，并记录每个声明的结束位置和捕获的函数名
    matches = [
        (match.end(), match.group(1))
        for match in re.finditer(pattern, source_code, re.DOTALL)
    ]

    # 从最后一个匹配开始，避免替换后索引错位的问题
    for end_index, captured_string in reversed(matches):
        # 查找当前匹配之后的下一个 triton_kernel_decl 的位置
        index_of_B = source_code.find(triton_kernel_decl, end_index)
        if index_of_B != -1:
            # 将 triton_kernel_decl 替换为捕获的函数名
            source_code = (
                source_code[:index_of_B]
                + f"def {captured_string}"
                + source_code[index_of_B + len(triton_kernel_decl) :]
            )
        else:
            # 如果在当前匹配之后找不到 triton_kernel_decl，则继续下一个匹配
            continue

    return source_code


def merge_params(original_params: List[str], new_params: List[str]) -> List[str]:
    # 断言新参数列表的长度不小于原参数列表的长度
    assert len(new_params) >= len(original_params)
    # 遍历新参数列表，将 "T" 替换为对应原参数列表的值
    for idx in range(len(new_params)):
        if new_params[idx] == "T":
            new_params[idx] = original_params[idx]
    return new_params


def add_launch_params(original: str, kernel_to_params: Dict[str, str]) -> str:
    # 正则表达式匹配原始字符串中的函数调用
    pattern = r"(\w+)\.run\(([^)]*), grid=(.*\)), [^)]*\)"

    def replace(match) -> str:
        # 提取正则匹配的部分
        func_name = match.group(1)
        params = match.group(2)
        grid = match.group(3)
        new_params = kernel_to_params[func_name]
        new_params = merge_params(params.split(", "), new_params.split(", "))

        # 格式化新的函数调用字符串
        new_string = f"{func_name}[{grid}]({', '.join(new_params)})"
        return new_string

    # 替换原始字符串中的函数调用，并生成转换后的字符串
    transformed = re.sub(pattern, replace, original)

    # 移除源代码中的 @triton_heuristics 和与之匹配的 @triton.jit 之间的内容
    remove_inductor_wrappers = re.sub(
        r"@triton_heuristics[^@]*@triton.jit",
        r"@triton.jit",
        transformed,
        flags=re.DOTALL,
    )

    return remove_inductor_wrappers


def process_file(input_filename: str, output_filename: str) -> str:
    with open(input_filename) as file:
        source_code = file.read()
    # 复制源代码到新变量中
    transformed_code = source_code
    # 如果源代码包含特定字符串，则抛出运行时错误
    if "def triton_(" in source_code:
        raise RuntimeError(
            "Need to run original Pytorch code generating kernels with TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1"
        )
    
    # 移除代码中的 Triton 函数声明
    transformed_code = remove_triton_function_declaration(transformed_code)
    # 移除异步编译相关的内容
    transformed_code = remove_async_compile(transformed_code)

    # 构造启动参数文件名
    launch_params_filename = f"{input_filename}.launch_params"
    # 如果启动参数文件不存在，则抛出运行时错误
    if not os.path.exists(launch_params_filename):
        raise RuntimeError(
            f"Missing {launch_params_filename}. Run `TORCHINDUCTOR_DUMP_LAUNCH_PARAMS=1 python {input_filename}` first."
        )

    # 打开并读取启动参数文件的内容
    with open(launch_params_filename) as f:
        launch_params_meta = f.readlines()

    # 对启动参数进行处理：先按 '|' 分割每行内容，再去除每部分的首尾空格
    split_params = [i.split("|") for i in launch_params_meta]
    strip_params = [[a.strip(), b.strip()] for a, b in split_params]
    # 将处理后的启动参数列表转换成字典，键为内核名，值为参数
    kernel_to_args: Dict[str, str] = dict(strip_params)
    # 向代码中添加启动参数
    transformed_code = add_launch_params(transformed_code, kernel_to_args)

    # 将变换后的代码写入输出文件
    with open(output_filename, "w") as file:
        file.write(transformed_code)
    
    # 返回经过处理的代码
    return transformed_code
def get_clean_triton(
    input_path: Path, output_path: Path = Path("triton_only_repro.py")
):
    """Run experiments and output results to file

    Args:
        input_path (Optional[Path]): Path to inductor generated output codede
        output_path (Optional[Path]): Path to write out the new python file
    """
    # 调用 process_file 函数，传入输入路径和输出路径，并返回其结果
    return process_file(str(input_path), str(output_path))


if __name__ == "__main__":
    """Sample usage:
    # Running sweep
    python inputcode.py
    """
    # 创建 argparse.ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser(
        description="Clean Inductor generated code to remove Inductor dependencies"
    )

    # 添加命令行参数
    parser.add_argument(
        "input_path", type=Path, help="Path to inductor generated output code"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("triton_only_repro.py"),
        help="Path to write out the clean triton output",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用 get_clean_triton 函数，传入解析后的参数对象，获取处理结果
    result = get_clean_triton(args.input_path, args.output_path)
```