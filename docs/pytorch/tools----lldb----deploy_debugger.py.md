# `.\pytorch\tools\lldb\deploy_debugger.py`

```py
import lldb  # type: ignore[import]

# 创建一个新的断点对象，使用正则表达式作为断点条件
bp = target.BreakpointCreateByRegex("__deploy_register_code")

# 设置断点的脚本回调函数体，用于处理断点命中时的逻辑
bp.SetScriptCallbackBody(
    """\
# 获取当前线程的进程对象
process = frame.thread.GetProcess()
# 获取进程的目标对象
target = process.target
# 查找指定符号在当前模块中的地址
symbol_addr = frame.module.FindSymbol("__deploy_module_info").GetStartAddress()
# 获取符号的加载地址
info_addr = symbol_addr.GetLoadAddress(target)
# 创建一个 LLDB 错误对象
e = lldb.SBError()
# 设置指针大小为 8 字节
ptr_size = 8
# 从内存中读取指定地址处的指针值，解析为字符串地址
str_addr = process.ReadPointerFromMemory(info_addr, e)
# 从内存中读取指定地址处的指针值，解析为文件地址
file_addr = process.ReadPointerFromMemory(info_addr + ptr_size, e)
# 从内存中读取指定地址处的指针值，解析为文件大小
file_size = process.ReadPointerFromMemory(info_addr + 2*ptr_size, e)
# 从内存中读取指定地址处的指针值，解析为加载偏移量
load_bias = process.ReadPointerFromMemory(info_addr + 3*ptr_size, e)
# 从指定地址读取固定长度的 C 字符串，最大长度为 512 字节
name = process.ReadCStringFromMemory(str_addr, 512, e)
# 从指定地址读取指定长度的内存块数据
r = process.ReadMemory(file_addr, file_size, e)

# 导入临时文件模块和路径操作模块
from tempfile import NamedTemporaryFile
from pathlib import Path
# 获取文件名的主干部分
stem = Path(name).stem
# 创建一个以主干部分命名、扩展名为 '.so' 的临时文件
with NamedTemporaryFile(prefix=stem, suffix='.so', delete=False) as tf:
    # 向临时文件写入读取的内存数据
    tf.write(r)
    # 打印调试信息
    print("torch_deploy registering debug information for ", tf.name)
    # 构造并执行 LLDB 命令，将临时文件加载为调试目标模块
    cmd1 = f"target modules add {tf.name}"
    lldb.debugger.HandleCommand(cmd1)
    # 构造并执行 LLDB 命令，加载临时文件到指定加载偏移地址的调试模块
    cmd2 = f"target modules load -f {tf.name} -s {hex(load_bias)}"
    lldb.debugger.HandleCommand(cmd2)

# 返回 False，继续执行原有的 LLDB 断点操作
return False
"""
)
```