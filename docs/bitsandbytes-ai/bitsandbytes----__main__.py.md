# `bitsandbytes\bitsandbytes\__main__.py`

```py
# 导入必要的库
import os
from os.path import isdir
import sys
from warnings import warn

# 导入 PyTorch 库
import torch

# 定义全局变量 HEADER_WIDTH
HEADER_WIDTH = 60

# 递归查找指定文件名的文件
def find_file_recursive(folder, filename):
    import glob
    outs = []
    try:
        # 遍历不同操作系统下的动态链接库文件扩展名
        for ext in ["so", "dll", "dylib"]:
            out = glob.glob(os.path.join(folder, "**", filename + ext))
            outs.extend(out)
    except Exception as e:
        # 抛出异常
        raise RuntimeError('Error: Something when wrong when trying to find file.') from e

    return outs

# 生成 Bug 报告信息
def generate_bug_report_information():
    # 打印 BUG 报告信息的标题
    print_header("")
    print_header("BUG REPORT INFORMATION")
    print_header("")
    print('')

    # 检查是否存在 Anaconda 环境变量 CONDA_PREFIX
    if 'CONDA_PREFIX' in os.environ:
        # 查找包含 *cuda* 的文件路径
        paths = find_file_recursive(os.environ['CONDA_PREFIX'], '*cuda*')
        print_header("ANACONDA CUDA PATHS")
        print(paths)
        print('')
    # 检查是否存在 /usr/local/ 目录
    if isdir('/usr/local/'):
        # 查找包含 *cuda* 的文件路径
        paths = find_file_recursive('/usr/local', '*cuda*')
        print_header("/usr/local CUDA PATHS")
        print(paths)
        print('')
    # 检查是否存在 CUDA_PATH 环境变量
    if 'CUDA_PATH' in os.environ and isdir(os.environ['CUDA_PATH']):
        # 查找包含 *cuda* 的文件路径
        paths = find_file_recursive(os.environ['CUDA_PATH'], '*cuda*')
        print_header("CUDA PATHS")
        print(paths)
        print('')

    # 检查当前工作目录是否存在
    if isdir(os.getcwd()):
        # 查找包含 *cuda* 的文件路径
        paths = find_file_recursive(os.getcwd(), '*cuda*')
        print_header("WORKING DIRECTORY CUDA PATHS")
        print(paths)
        print('')

    # 打印 LD_LIBRARY CUDA PATHS 的标题
    print_header("LD_LIBRARY CUDA PATHS")
    # 检查是否存在 LD_LIBRARY_PATH 环境变量
    if 'LD_LIBRARY_PATH' in os.environ:
        # 获取 LD_LIBRARY_PATH 环境变量的值
        lib_path = os.environ['LD_LIBRARY_PATH'].strip()
        # 遍历 LD_LIBRARY_PATH 中的路径
        for path in set(lib_path.split(os.pathsep)):
            try:
                if isdir(path):
                    # 打印当前路径下包含 *cuda* 的文件路径
                    print_header(f"{path} CUDA PATHS")
                    paths = find_file_recursive(path, '*cuda*')
                    print(paths)
            except Exception as e:
                # 打印异常信息
                print(f'Could not read LD_LIBRARY_PATH: {path} ({e})')
    print('')

# 打印带有指定宽度和填充字符的标题
def print_header(
    txt: str, width: int = HEADER_WIDTH, filler: str = "+"
) -> None:
    # 如果 txt 不为空，则在其前后添加空格，否则为空字符串
    txt = f" {txt} " if txt else ""
    # 将 txt 居中显示，并用指定的填充字符填充到指定的宽度
    print(txt.center(width, filler))
# 定义一个函数，用于打印调试信息
def print_debug_info() -> None:
    # 导入 PACKAGE_GITHUB_URL 变量
    from . import PACKAGE_GITHUB_URL
    # 打印调试信息，包括 PACKAGE_GITHUB_URL 变量
    print(
        "\nAbove we output some debug information. Please provide this info when "
        f"creating an issue via {PACKAGE_GITHUB_URL}/issues/new/choose ...\n"
    )


# 主函数
def main():
    # 生成错误报告信息
    generate_bug_report_information()

    # 导入 COMPILED_WITH_CUDA 变量和 get_compute_capabilities 函数
    from . import COMPILED_WITH_CUDA
    from .cuda_setup.main import get_compute_capabilities

    # 打印头部信息
    print_header("OTHER")
    # 打印 COMPILED_WITH_CUDA 变量的值
    print(f"COMPILED_WITH_CUDA = {COMPILED_WITH_CUDA}")
    # 打印 COMPUTE_CAPABILITIES_PER_GPU 变量的值
    print(f"COMPUTE_CAPABILITIES_PER_GPU = {get_compute_capabilities()}")
    # 打印空行
    print_header("")
    # 打印头部信息
    print_header("DEBUG INFO END")
    # 打印空行
    print_header("")
    # 打印信息
    print("Checking that the library is importable and CUDA is callable...")
    # 打印警告信息
    print("\nWARNING: Please be sure to sanitize sensitive info from any such env vars!\n")

    try:
        # 导入 Adam 类
        from bitsandbytes.optim import Adam

        # 创建一个 CUDA 张量
        p = torch.nn.Parameter(torch.rand(10, 10).cuda())
        a = torch.rand(10, 10).cuda()

        # 计算 p 的数据总和
        p1 = p.data.sum().item()

        # 创建 Adam 优化器
        adam = Adam([p])

        # 进行张量运算
        out = a * p
        loss = out.sum()
        loss.backward()
        adam.step()

        # 计算 p 的数据总和
        p2 = p.data.sum().item()

        # 断言 p1 不等于 p2
        assert p1 != p2
        # 打印成功信息
        print("SUCCESS!")
        print("Installation was successful!")
    except ImportError:
        # 打印警告信息
        print()
        warn(
            f"WARNING: {__package__} is currently running as CPU-only!\n"
            "Therefore, 8-bit optimizers and GPU quantization are unavailable.\n\n"
            f"If you think that this is so erroneously,\nplease report an issue!"
        )
        # 调用打印调试信息函数
        print_debug_info()
    except Exception as e:
        # 打印异常信息
        print(e)
        # 调用打印调试信息函数
        print_debug_info()
        # 退出程序
        sys.exit(1)


# 如果当前脚本作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
```