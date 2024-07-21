# `.\pytorch\tools\code_coverage\package\tool\utils.py`

```py
# 导入 subprocess 模块，用于执行外部命令
import subprocess

# 从相对路径的 "..util.setting" 模块中导入 TestPlatform 枚举
from ..util.setting import TestPlatform
# 从相对路径的 "..util.utils" 模块中导入 print_error 函数
from ..util.utils import print_error


# 定义一个函数 run_cpp_test，用于运行 C++ 测试二进制文件
def run_cpp_test(binary_file: str) -> None:
    # 尝试执行给定的二进制文件
    try:
        subprocess.check_call(binary_file)
    except subprocess.CalledProcessError:
        # 如果执行出错，则打印错误信息并指出失败的二进制文件
        print_error(f"Binary failed to run: {binary_file}")


# 定义一个函数 get_tool_path_by_platform，根据测试平台返回工具路径
def get_tool_path_by_platform(platform: TestPlatform) -> str:
    # 如果平台是 TestPlatform.FBCODE
    if platform == TestPlatform.FBCODE:
        # 从 caffe2.fb.code_coverage.tool.package.fbcode.utils 中导入 get_llvm_tool_path 函数
        from caffe2.fb.code_coverage.tool.package.fbcode.utils import (
            get_llvm_tool_path,
        )

        # 调用获取 LLVM 工具路径的函数并返回结果
        return get_llvm_tool_path()  # type: ignore[no-any-return]
    # 如果平台不是 FBCODE
    else:
        # 从相对路径的 "..oss.utils" 模块中导入 get_llvm_tool_path 函数
        from ..oss.utils import get_llvm_tool_path  # type: ignore[no-redef]

        # 调用获取 LLVM 工具路径的函数并返回结果
        return get_llvm_tool_path()  # type: ignore[no-any-return]
```