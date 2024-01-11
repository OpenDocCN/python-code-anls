# `ZeroNet\src\util\OpensslFindPatch.py`

```
# 导入日志、操作系统和系统相关的模块
import logging
import os
import sys
import ctypes.util

# 从 Config 模块中导入 config 对象
from Config import config

# 保存 ctypes.util.find_library 的原始实现
find_library_original = ctypes.util.find_library

# 获取 OpenSSL 路径的函数
def getOpensslPath():
    # 如果配置中已经指定了 OpenSSL 库文件，则直接返回
    if config.openssl_lib_file:
        return config.openssl_lib_file

    # 根据不同的操作系统平台设置 OpenSSL 库文件的路径
    if sys.platform.startswith("win"):
        lib_paths = [
            os.path.join(os.getcwd(), "tools/openssl/libeay32.dll"),  # ZeroBundle Windows
            os.path.join(os.path.dirname(sys.executable), "DLLs/libcrypto-1_1-x64.dll"),
            os.path.join(os.path.dirname(sys.executable), "DLLs/libcrypto-1_1.dll")
        ]
    elif sys.platform == "cygwin":
        lib_paths = ["/bin/cygcrypto-1.0.0.dll"]
    else:
        lib_paths = [
            "../runtime/lib/libcrypto.so.1.1",  # ZeroBundle Linux
            "../../Frameworks/libcrypto.1.1.dylib",  # ZeroBundle macOS
            "/opt/lib/libcrypto.so.1.0.0",  # For optware and entware
            "/usr/local/ssl/lib/libcrypto.so"
        ]

    # 遍历不同平台的 OpenSSL 库文件路径，找到存在的文件并返回
    for lib_path in lib_paths:
        if os.path.isfile(lib_path):
            return lib_path

    # 如果是在安卓环境下，尝试从环境变量中获取 OpenSSL 库文件路径
    if "ANDROID_APP_PATH" in os.environ:
        try:
            lib_dir = os.environ["ANDROID_APP_PATH"] + "/../../lib"
            return [lib for lib in os.listdir(lib_dir) if "crypto" in lib][0]
        except Exception as err:
            logging.debug("OpenSSL lib not found in: %s (%s)" % (lib_dir, err))

    # 如果是在具有 LD_LIBRARY_PATH 环境变量的系统中，尝试从环境变量指定的路径中查找 OpenSSL 库文件
    if "LD_LIBRARY_PATH" in os.environ:
        lib_dir_paths = os.environ["LD_LIBRARY_PATH"].split(":")
        for path in lib_dir_paths:
            try:
                return [lib for lib in os.listdir(path) if "libcrypto.so" in lib][0]
            except Exception as err:
                logging.debug("OpenSSL lib not found in: %s (%s)" % (path, err))

    # 尝试使用 ctypes.util.find_library 来查找 OpenSSL 库文件
    lib_path = (
        find_library_original('ssl.so') or find_library_original('ssl') or
        find_library_original('crypto') or find_library_original('libcrypto') or 'libeay32'
    )

    return lib_path

# 用于修补 ctypes.util.find_library 的函数
def patchCtypesOpensslFindLibrary():
    # 定义一个名为findLibraryPatched的函数，用于查找库文件
    def findLibraryPatched(name):
        # 如果传入的库名是"ssl"、"crypto"或"libeay32"，则调用getOpensslPath()函数获取路径并返回
        if name in ("ssl", "crypto", "libeay32"):
            lib_path = getOpensslPath()
            return lib_path
        # 如果传入的库名不是上述三者，则调用原始的find_library_original()函数进行查找并返回结果
        else:
            return find_library_original(name)
    
    # 将ctypes.util.find_library函数重写为findLibraryPatched函数
    ctypes.util.find_library = findLibraryPatched
# 调用 patchCtypesOpensslFindLibrary() 函数，用于修补 ctypes 模块对 OpenSSL 库的查找
patchCtypesOpensslFindLibrary()
```