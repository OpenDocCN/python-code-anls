# `.\pytorch\tools\build_libtorch.py`

```
import argparse  # 导入用于解析命令行参数的模块
import sys  # 导入系统相关的模块
from os.path import abspath, dirname  # 导入用于处理文件路径的函数

# 通过将 pytorch_root 添加到 sys.path 中，使得即使作为独立脚本运行，
# 此模块也能导入其他 torch 模块。例如，可以执行 `python build_libtorch.py`
# 或者 `python -m tools.build_libtorch`。
pytorch_root = dirname(dirname(abspath(__file__)))
sys.path.append(pytorch_root)

from tools.build_pytorch_libs import build_caffe2  # 导入构建 caffe2 的函数
from tools.setup_helpers.cmake import CMake  # 导入 CMake 相关的模块

if __name__ == "__main__":
    # 为将来的接口预留的占位符。目前仅显示一个漂亮的帮助信息 `-h`。
    parser = argparse.ArgumentParser(description="Build libtorch")
    parser.add_argument("--rerun-cmake", action="store_true", help="重新运行 cmake")
    parser.add_argument(
        "--cmake-only",
        action="store_true",
        help="仅运行 cmake。留给用户调整构建选项的机会",
    )
    options = parser.parse_args()

    build_caffe2(
        version=None,  # 不指定版本号
        cmake_python_library=None,  # 不指定 cmake Python 库
        build_python=False,  # 不构建 Python
        rerun_cmake=options.rerun_cmake,  # 是否重新运行 cmake，根据命令行参数决定
        cmake_only=options.cmake_only,  # 是否仅运行 cmake，根据命令行参数决定
        cmake=CMake(),  # 使用默认的 CMake 实例
    )
```