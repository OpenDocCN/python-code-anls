# `ZeroNet\src\lib\sslcrypto\openssl\library.py`

```
# 导入必要的模块
import os
import sys
import ctypes
import ctypes.util
from .discovery import discover as user_discover

# 禁用误报的 _MEIPASS
# pylint: disable=no-member,protected-access

# 发现 OpenSSL 库的路径
def discover_paths():
    # 首先搜索本地文件
    if "win" in sys.platform:
        # Windows
        # 定义 OpenSSL 库文件名
        names = [
            "libeay32.dll"
        ]
        # 获取绝对路径
        openssl_paths = [os.path.abspath(path) for path in names]
        # 如果存在 _MEIPASS 属性，添加到路径列表中
        if hasattr(sys, "_MEIPASS"):
            openssl_paths += [os.path.join(sys._MEIPASS, path) for path in openssl_paths]
        # 使用 ctypes.util.find_library 寻找库文件并添加到路径列表中
        openssl_paths.append(ctypes.util.find_library("libeay32"))
    elif "darwin" in sys.platform:
        # Mac OS
        # 定义 OpenSSL 库文件名
        names = [
            "libcrypto.dylib",
            "libcrypto.1.1.0.dylib",
            "libcrypto.1.0.2.dylib",
            "libcrypto.1.0.1.dylib",
            "libcrypto.1.0.0.dylib",
            "libcrypto.0.9.8.dylib"
        ]
        # 获取绝对路径
        openssl_paths = [os.path.abspath(path) for path in names]
        # 将所有文件名添加到路径列表中
        openssl_paths += names
        # 添加额外的路径
        openssl_paths += [
            "/usr/local/opt/openssl/lib/libcrypto.dylib"
        ]
        # 如果存在 _MEIPASS 属性和 RESOURCEPATH 环境变量，添加额外的路径
        if hasattr(sys, "_MEIPASS") and "RESOURCEPATH" in os.environ:
            openssl_paths += [
                os.path.join(os.environ["RESOURCEPATH"], "..", "Frameworks", name)
                for name in names
            ]
        # 使用 ctypes.util.find_library 寻找库文件并添加到路径列表中
        openssl_paths.append(ctypes.util.find_library("ssl"))
    else:
        # 对于 Linux、BSD 等系统，指定需要查找的 OpenSSL 库文件名
        names = [
            "libcrypto.so",
            "libssl.so",
            "libcrypto.so.1.1.0",
            "libssl.so.1.1.0",
            "libcrypto.so.1.0.2",
            "libssl.so.1.0.2",
            "libcrypto.so.1.0.1",
            "libssl.so.1.0.1",
            "libcrypto.so.1.0.0",
            "libssl.so.1.0.0",
            "libcrypto.so.0.9.8",
            "libssl.so.0.9.8"
        ]
        # 将文件名转换为绝对路径
        openssl_paths = [os.path.abspath(path) for path in names]
        # 将文件名添加到路径列表中
        openssl_paths += names
        # 如果存在特定的路径（PyInstaller 打包时使用），将文件名添加到路径列表中
        if hasattr(sys, "_MEIPASS"):
            openssl_paths += [os.path.join(sys._MEIPASS, path) for path in names]
        # 查找并添加 SSL 库的路径
        openssl_paths.append(ctypes.util.find_library("ssl"))
    # 调用 user_discover 函数，获取返回值
    lst = user_discover()
    # 如果返回值是字符串，将其转换为列表
    if isinstance(lst, str):
        lst = [lst]
    # 如果返回值为空，将其赋值为空列表
    elif not lst:
        lst = []
    # 返回列表和 OpenSSL 路径列表的组合
    return lst + openssl_paths
# 发现并加载 OpenSSL 库
def discover_library():
    # 遍历发现路径
    for path in discover_paths():
        # 如果路径存在
        if path:
            try:
                # 尝试加载库
                return ctypes.CDLL(path)
            except OSError:
                pass
    # 如果无法加载库，则抛出异常
    raise OSError("OpenSSL is unavailable")

# 调用发现库的函数，将结果赋值给变量 lib
lib = discover_library()

# 初始化内部状态
try:
    # 尝试调用库中的函数
    lib.OPENSSL_add_all_algorithms_conf()
except AttributeError:
    pass

try:
    # 设置返回类型为 c_char_p
    lib.OpenSSL_version.restype = ctypes.c_char_p
    # 调用 OpenSSL_version 函数并解码结果
    openssl_backend = lib.OpenSSL_version(0).decode()
except AttributeError:
    # 设置返回类型为 c_char_p
    lib.SSLeay_version.restype = ctypes.c_char_p
    # 调用 SSLeay_version 函数并解码结果
    openssl_backend = lib.SSLeay_version(0).decode()

# 将 OpenSSL 版本信息和库名称拼接起来
openssl_backend += " at " + lib._name
```