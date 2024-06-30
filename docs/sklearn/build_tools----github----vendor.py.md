# `D:\src\scipysrc\scikit-learn\build_tools\github\vendor.py`

```
# 导入必要的库
import os
import os.path as op
import shutil
import sys
import textwrap

# 目标文件夹路径
TARGET_FOLDER = op.join("sklearn", ".libs")
# _distributor_init.py 文件路径
DISTRIBUTOR_INIT = op.join("sklearn", "_distributor_init.py")
# vcomp140.dll 在系统中的路径
VCOMP140_SRC_PATH = "C:\\Windows\\System32\\vcomp140.dll"
# msvcp140.dll 在系统中的路径
MSVCP140_SRC_PATH = "C:\\Windows\\System32\\msvcp140.dll"

# 创建 _distributor_init.py 文件用于 64 位架构
def make_distributor_init_64_bits(
    distributor_init,
    vcomp140_dll_filename,
    msvcp140_dll_filename,
):
    """Create a _distributor_init.py file for 64-bit architectures.

    This file is imported first when importing the sklearn package
    so as to pre-load the vendored vcomp140.dll and msvcp140.dll.
    """
    # 打开 _distributor_init.py 文件，写入预加载 vcomp140.dll 和 msvcp140.dll 的代码
    with open(distributor_init, "wt") as f:
        f.write(
            textwrap.dedent(
                """
            '''Helper to preload vcomp140.dll and msvcp140.dll to prevent
            "not found" errors.

            Once vcomp140.dll and msvcp140.dll are
            preloaded, the namespace is made available to any subsequent
            vcomp140.dll and msvcp140.dll. This is
            created as part of the scripts that build the wheel.
            '''


            import os
            import os.path as op
            from ctypes import WinDLL

            # 如果操作系统是 Windows
            if os.name == "nt":
                # 构建 .libs 文件夹路径
                libs_path = op.join(op.dirname(__file__), ".libs")
                # 构建 vcomp140.dll 和 msvcp140.dll 的目标路径
                vcomp140_dll_filename = op.join(libs_path, "{0}")
                msvcp140_dll_filename = op.join(libs_path, "{1}")
                # 使用 ctypes 加载 vcomp140.dll 和 msvcp140.dll
                WinDLL(op.abspath(vcomp140_dll_filename))
                WinDLL(op.abspath(msvcp140_dll_filename))
            """.format(
                    vcomp140_dll_filename,
                    msvcp140_dll_filename,
                )
            )
        )

# 主函数，嵌入 vcomp140.dll 和 msvcp140.dll
def main(wheel_dirname):
    """Embed vcomp140.dll and msvcp140.dll."""
    # 检查 vcomp140.dll 是否存在
    if not op.exists(VCOMP140_SRC_PATH):
        raise ValueError(f"Could not find {VCOMP140_SRC_PATH}.")

    # 检查 msvcp140.dll 是否存在
    if not op.exists(MSVCP140_SRC_PATH):
        raise ValueError(f"Could not find {MSVCP140_SRC_PATH}.")

    # 检查 wheel 目录是否存在
    if not op.isdir(wheel_dirname):
        raise RuntimeError(f"Could not find {wheel_dirname} file.")

    # 获取 vcomp140.dll 和 msvcp140.dll 的文件名
    vcomp140_dll_filename = op.basename(VCOMP140_SRC_PATH)
    msvcp140_dll_filename = op.basename(MSVCP140_SRC_PATH)

    # 构建目标文件夹路径
    target_folder = op.join(wheel_dirname, TARGET_FOLDER)
    # 构建 _distributor_init.py 文件路径
    distributor_init = op.join(wheel_dirname, DISTRIBUTOR_INIT)

    # 如果 "sklearn/.libs" 文件夹不存在，则创建它
    if not op.exists(target_folder):
        os.mkdir(target_folder)

    # 复制 vcomp140.dll 到目标文件夹
    print(f"Copying {VCOMP140_SRC_PATH} to {target_folder}.")
    shutil.copy2(VCOMP140_SRC_PATH, target_folder)

    # 复制 msvcp140.dll 到目标文件夹
    print(f"Copying {MSVCP140_SRC_PATH} to {target_folder}.")
    shutil.copy2(MSVCP140_SRC_PATH, target_folder)

    # 生成 _distributor_init.py 文件在源代码树中
    print("Generating the '_distributor_init.py' file.")
    make_distributor_init_64_bits(
        distributor_init,
        vcomp140_dll_filename,
        msvcp140_dll_filename,
    )

# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 从 sys 模块导入 argv 对象，用于获取命令行参数
    _, wheel_file = sys.argv
    # 调用 main 函数，传入命令行参数中的 wheel 文件名作为参数
    main(wheel_file)
```