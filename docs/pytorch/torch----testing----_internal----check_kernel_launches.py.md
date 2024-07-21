# `.\pytorch\torch\testing\_internal\check_kernel_launches.py`

```py
# mypy: ignore-errors

# 导入必要的库
import os
import re
import sys
from typing import List

# 导出的符号列表，模块外可见的函数和变量
__all__ = [
    "check_code_for_cuda_kernel_launches",
    "check_cuda_kernel_launches",
]

# 要排除的文件列表（通过后缀匹配）
# 如果检查器有问题，可以暂时排除这些文件，修复检查器后再取消排除
exclude_files: List[str] = []

# 无法百分之百地检测 CUDA 核函数的启动，因此我们用模式 "<<<parameters>>>(arguments);" 进行建模
# 然后需要在下一个语句出现 `C10_CUDA_KERNEL_LAUNCH_CHECK`
#
# 我们将下一个语句建模为在下一个 `}` 或 `;` 结束
# 如果看到 `}` 则表示子句结束（不好），如果看到分号则期望在其之前有启动检查
#
# 由于核函数启动可能包含 lambda 表达式，正确找到核函数启动的结束括号是很重要的。
# 为了避免额外的依赖，我们构建一个前缀正则表达式来找到核函数启动的开始，
# 使用括号匹配算法找到启动的结束，然后使用另一个正则表达式确定是否存在启动检查。

# 查找可能的核函数启动的开始位置
kernel_launch_start = re.compile(
    r"^.*<<<[^>]+>>>\s*\(", flags=re.MULTILINE
)

# 这个模式应该从核函数启动的最后一个右括号后的字符开始。
# 如果下一个语句不是启动检查，则返回匹配。
has_check = re.compile(
    r"\s*;(?![^;}]*C10_CUDA_KERNEL_LAUNCH_CHECK\(\);)", flags=re.MULTILINE
)

def find_matching_paren(s: str, startpos: int) -> int:
    """给定字符串 "prefix (未知字符数) suffix" 和第一个 `(` 的位置，
    返回与之匹配的 `)` 的索引位置（后一个位置），考虑到括号的嵌套
    """
    opening = 0
    for i, c in enumerate(s[startpos:]):
        if c == '(':
            opening += 1
        elif c == ')':
            opening -= 1
            if opening == 0:
                return startpos + i + 1

    raise IndexError("未找到闭合括号！")

def should_exclude_file(filename) -> bool:
    """检查文件名是否应该被排除"""
    for exclude_suffix in exclude_files:
        if filename.endswith(exclude_suffix):
            return True
    return False

def check_code_for_cuda_kernel_launches(code, filename=None):
    """检查代码中是否存在未包含 CUDA 错误检查的核函数启动。

    Args:
        filename - 包含代码的文件名，仅用于显示目的
        code     - 要检查的代码

    Returns:
        代码中不安全的核函数启动数量
    """
    if filename is None:
        filename = "##Python Function Call##"

    # 我们将代码分解并重新组合以添加
    # 将代码按行拆分成列表，并为每行添加行号
    code = enumerate(code.split("\n"))                             # 按换行符拆分代码，同时给每行添加行号
    code = [f"{lineno}: {linecode}" for lineno, linecode in code]  # 格式化行号和代码内容
    code = '\n'.join(code)                                         # 将格式化后的代码列表重新连接成字符串

    # 初始化计数器，用于记录没有检查的 CUDA kernel 启动次数
    num_launches_without_checks = 0

    # 在代码中查找所有 CUDA kernel 启动的位置
    for m in kernel_launch_start.finditer(code):
        # 找到 CUDA kernel 启动语句的结束括号位置
        end_paren = find_matching_paren(code, m.end() - 1)

        # 检查结束括号之后是否有 C10_CUDA_KERNEL_LAUNCH_CHECK
        if has_check.match(code, end_paren):
            # 如果没有检查，增加未检查 CUDA kernel 启动的计数
            num_launches_without_checks += 1
            # 提取出问题代码的上下文信息
            context = code[m.start():end_paren + 1]
            # 输出错误信息，指出缺少 C10_CUDA_KERNEL_LAUNCH_CHECK 的位置和上下文
            print(f"Missing C10_CUDA_KERNEL_LAUNCH_CHECK in '{filename}'. Context:\n{context}", file=sys.stderr)

    # 返回没有检查的 CUDA kernel 启动次数
    return num_launches_without_checks
def check_file(filename):
    """检查文件是否存在 CUDA 内核启动但没有 CUDA 错误检查

    Args:
        filename - 要检查的文件名

    Returns:
        文件中存在的不安全内核启动次数
    """
    # 如果文件不是以 .cu 或者 .cuh 结尾，则直接返回 0
    if not (filename.endswith((".cu", ".cuh"))):
        return 0
    # 如果应该排除该文件，则返回 0
    if should_exclude_file(filename):
        return 0
    # 打开文件并读取其内容
    with open(filename) as fo:
        contents = fo.read()
        # 检查文件内容中 CUDA 内核启动但没有 CUDA 错误检查的次数
        unsafeCount = check_code_for_cuda_kernel_launches(contents, filename)
    return unsafeCount


def check_cuda_kernel_launches():
    """检查所有 PyTorch 代码中的 CUDA 内核启动但没有 CUDA 错误检查

    Returns:
        代码库中存在的不安全内核启动次数
    """
    # 获取当前文件的父目录的父目录，即 Torch 的根目录
    torch_dir = os.path.dirname(os.path.realpath(__file__))
    torch_dir = os.path.dirname(torch_dir)  # 上溯一级至 Torch 的父目录
    torch_dir = os.path.dirname(torch_dir)  # 再上溯一级至 Caffe2 的父目录

    # 记录未经检查的内核启动次数和文件列表
    kernels_without_checks = 0
    files_without_checks = []
    # 遍历 Torch 根目录及其子目录中的所有文件
    for root, dirnames, filenames in os.walk(torch_dir):
        # 如果是 `$BASE/build` 或 `$BASE/torch/include` 目录，则跳过
        if root == os.path.join(torch_dir, "build") or root == os.path.join(torch_dir, "torch/include"):
            # 修改 dirnames 和 filenames 来停止在这些目录中搜索
            # 这是正确的做法，参考 `help(os.walk)`
            dirnames[:] = []
            continue

        for x in filenames:
            filename = os.path.join(root, x)
            # 检查文件中 CUDA 内核启动但没有 CUDA 错误检查的次数
            file_result = check_file(filename)
            if file_result > 0:
                kernels_without_checks += file_result
                files_without_checks.append(filename)

    # 如果存在未经检查的内核启动次数，则打印警告信息
    if kernels_without_checks > 0:
        count_str = f"Found {kernels_without_checks} instances in " \
                    f"{len(files_without_checks)} files where kernel " \
                    "launches didn't have checks."
        print(count_str, file=sys.stderr)
        print("Files without checks:", file=sys.stderr)
        for x in files_without_checks:
            print(f"\t{x}", file=sys.stderr)
        print(count_str, file=sys.stderr)

    return kernels_without_checks


if __name__ == "__main__":
    # 检查 CUDA 内核启动但没有 CUDA 错误检查的情况，并根据结果退出程序
    unsafe_launches = check_cuda_kernel_launches()
    sys.exit(0 if unsafe_launches == 0 else 1)
```