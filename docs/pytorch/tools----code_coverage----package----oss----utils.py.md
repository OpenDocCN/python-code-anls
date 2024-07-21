# `.\pytorch\tools\code_coverage\package\oss\utils.py`

```py
# 导入必要的模块和包
from __future__ import annotations
import os
import subprocess
from ..util.setting import CompilerType, TestType, TOOLS_FOLDER
from ..util.utils import print_error, remove_file

# 返回指定测试类型的 OSS 二进制文件夹路径
def get_oss_binary_folder(test_type: TestType) -> str:
    assert test_type in {TestType.CPP, TestType.PY}
    # TODO: change the way we get binary file -- binary may not in build/bin ?
    return os.path.join(
        get_pytorch_folder(), "build/bin" if test_type == TestType.CPP else "test"
    )

# 返回 OSS 共享库文件路径列表
def get_oss_shared_library() -> list[str]:
    lib_dir = os.path.join(get_pytorch_folder(), "build", "lib")
    return [
        os.path.join(lib_dir, lib)
        for lib in os.listdir(lib_dir)
        if lib.endswith(".dylib")
    ]

# 返回指定测试名称和类型的 OSS 二进制文件路径
def get_oss_binary_file(test_name: str, test_type: TestType) -> str:
    assert test_type in {TestType.CPP, TestType.PY}
    binary_folder = get_oss_binary_folder(test_type)
    binary_file = os.path.join(binary_folder, test_name)
    if test_type == TestType.PY:
        # 将 "python" 添加到命令中，以便可以直接使用 binary_file 变量运行脚本
        binary_file = "python " + binary_file
    return binary_file

# 返回 LLVM 工具链路径
def get_llvm_tool_path() -> str:
    return os.environ.get(
        "LLVM_TOOL_PATH", "/usr/local/opt/llvm/bin"
    )  # set default as llvm path in dev server, on mac the default may be /usr/local/opt/llvm/bin

# 返回 PyTorch 根文件夹路径
def get_pytorch_folder() -> str:
    # TOOLS_FOLDER in oss: pytorch/tools/code_coverage
    return os.path.abspath(
        os.environ.get(
            "PYTORCH_FOLDER", os.path.join(TOOLS_FOLDER, os.path.pardir, os.path.pardir)
        )
    )

# 检测编译器类型并返回
def detect_compiler_type() -> CompilerType | None:
    # 检查用户是否指定了编译器类型
    user_specify = os.environ.get("CXX", None)
    if user_specify:
        if user_specify in ["clang", "clang++"]:
            return CompilerType.CLANG
        elif user_specify in ["gcc", "g++"]:
            return CompilerType.GCC

        raise RuntimeError(f"User specified compiler is not valid {user_specify}")

    # 自动检测编译器类型
    auto_detect_result = subprocess.check_output(
        ["cc", "-v"], stderr=subprocess.STDOUT
    ).decode("utf-8")
    if "clang" in auto_detect_result:
        return CompilerType.CLANG
    elif "gcc" in auto_detect_result:
        return CompilerType.GCC
    raise RuntimeError(f"Auto detected compiler is not valid {auto_detect_result}")

# 清理 .gcda 文件
def clean_up_gcda() -> None:
    gcda_files = get_gcda_files()
    for item in gcda_files:
        remove_file(item)

# 获取当前 PyTorch 构建文件夹中的所有 .gcda 文件列表
def get_gcda_files() -> list[str]:
    folder_has_gcda = os.path.join(get_pytorch_folder(), "build")
    if os.path.isdir(folder_has_gcda):
        # TODO use glob
        # output = glob.glob(f"{folder_has_gcda}/**/*.gcda")
        output = subprocess.check_output(["find", folder_has_gcda, "-iname", "*.gcda"])
        return output.decode("utf-8").split("\n")
    else:
        return []

# 运行 OSS Python 测试脚本
def run_oss_python_test(binary_file: str) -> None:
    # python test script
    pass  # Placeholder for future implementation
    # 尝试执行外部的可执行文件，使用 subprocess 模块来执行命令
    try:
        # 使用 subprocess.check_call 方法执行命令，这里的 binary_file 是要执行的可执行文件路径
        # shell=True 表示使用 shell 解释器来执行命令
        # cwd=get_oss_binary_folder(TestType.PY) 指定了命令执行的当前工作目录
        subprocess.check_call(
            binary_file, shell=True, cwd=get_oss_binary_folder(TestType.PY)
        )
    except subprocess.CalledProcessError:
        # 如果命令执行返回非零状态码，则捕获 CalledProcessError 异常
        # 打印错误信息，指示可执行文件运行失败
        print_error(f"Binary failed to run: {binary_file}")
```