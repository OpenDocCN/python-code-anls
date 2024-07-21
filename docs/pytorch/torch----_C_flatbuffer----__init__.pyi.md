# `.\pytorch\torch\_C_flatbuffer\__init__.pyi`

```py
# 引入类型提示允许未定义的函数/方法
# 从 torch._C 模块导入 LiteScriptModule 和 ScriptModule 类
from torch._C import LiteScriptModule, ScriptModule

# 从文件中加载移动端模块，返回加载的模块对象
def _load_mobile_module_from_file(filename: str): ...

# 从字节流中加载移动端模块，返回加载的模块对象
def _load_mobile_module_from_bytes(bytes_: bytes): ...

# 从文件中加载 JIT 模块，返回加载的模块对象
def _load_jit_module_from_file(filename: str): ...

# 从字节流中加载 JIT 模块，返回加载的模块对象
def _load_jit_module_from_bytes(bytes_: bytes): ...

# 将 LiteScriptModule 对象保存到文件中
def _save_mobile_module(m: LiteScriptModule, filename: str): ...

# 将 ScriptModule 对象保存到文件中
def _save_jit_module(m: ScriptModule, filename: str): ...

# 将 LiteScriptModule 对象保存为字节流，并返回字节流对象
def _save_mobile_module_to_bytes(m: LiteScriptModule) -> bytes: ...

# 将 ScriptModule 对象保存为字节流，并返回字节流对象
def _save_jit_module_to_bytes(m: ScriptModule) -> bytes: ...
```