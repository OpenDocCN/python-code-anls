# `.\pytorch\scripts\diagnose_protobuf.py`

```
## @package diagnose_protobuf
# Module scripts.diagnose_protobuf
"""Diagnoses the current protobuf situation.

Protocol buffer needs to be properly installed for Caffe2 to work, and
sometimes it is rather tricky. Specifically, we will need to have a
consistent version between C++ and python simultaneously. This is a
convenience script for one to quickly check if this is so on one's local
machine.

Usage:
    [set your environmental variables like PATH and PYTHONPATH]
    python scripts/diagnose_protobuf.py
"""

import os
import re
from subprocess import PIPE, Popen

# 获取 Python 环境中的 protobuf 版本。
try:
    import google.protobuf

    python_version = google.protobuf.__version__
    python_protobuf_installed = True
except ImportError:
    # 如果找不到 python protobuf 安装，则输出调试信息。
    print("DEBUG: cannot find python protobuf install.")
    python_protobuf_installed = False

# 根据操作系统类型设置 protoc 的可执行文件名。
if os.name == "nt":
    protoc_name = "protoc.exe"
else:
    protoc_name = "protoc"

try:
    # 尝试执行 protoc 命令，并获取其输出和错误信息。
    p = Popen([protoc_name, "--version"], stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
except:
    # 如果无法执行 protoc 命令，则输出调试信息。
    print("DEBUG: did not find protoc binary.")
    print("DEBUG: out: " + out)
    print("DEBUG: err: " + err)
    native_protobuf_installed = False
else:
    if p.returncode:
        # 如果 protoc 返回非零返回码，则输出调试信息。
        print("DEBUG: protoc returned a non-zero return code.")
        print("DEBUG: out: " + out)
        print("DEBUG: err: " + err)
        native_protobuf_installed = False
    else:
        # 尝试从 protoc 输出中解析版本号。
        tmp = re.search(r"\d\.\d\.\d", out)
        if tmp:
            native_version = tmp.group(0)
            native_protobuf_installed = True
        else:
            # 如果无法解析 protoc 版本号，则输出调试信息。
            print("DEBUG: cannot parse protoc version string.")
            print("DEBUG: out: " + out)
            native_protobuf_installed = False

# 未安装 python protobuf 的警告信息。
PYTHON_PROTOBUF_NOT_INSTALLED = """
You have not installed python protobuf. Protobuf is needed to run caffe2. You
can install protobuf via pip or conda (if you are using anaconda python).
"""

# 未安装 native protoc 的警告信息。
NATIVE_PROTOBUF_NOT_INSTALLED = """
You have not installed the protoc binary. Protoc is needed to compile Caffe2
protobuf source files. Depending on the platform you are on, you can install
protobuf via:
    (1) Mac: using homebrew and do brew install protobuf.
    (2) Linux: use apt and do apt-get install libprotobuf-dev
    (3) Windows: install from source, or from the releases here:
        https://github.com/google/protobuf/releases/
"""

# Python protobuf 和 native protoc 版本不一致的警告信息。
VERSION_MISMATCH = f"""
Your python protobuf is of version {python_version} but your native protoc version is of
version {native_version}. This will cause the installation to produce incompatible
protobuf files. This is bad in general - consider installing the same version.
"""

# 现在开始给出实际的建议
if not python_protobuf_installed:
    # 如果未安装 python protobuf，则输出警告信息。
    print(PYTHON_PROTOBUF_NOT_INSTALLED)

if not native_protobuf_installed:
    # 如果未安装 native protoc，则输出警告信息。
    print(NATIVE_PROTOBUF_NOT_INSTALLED)

if python_protobuf_installed and native_protobuf_installed:
    if python_version != native_version:
        # 如果 python protobuf 和 native protoc 版本不一致，则输出警告信息。
        print(VERSION_MISMATCH)
    else:
        # 如果一切正常，则输出提示信息。
        print("All looks good.")
```