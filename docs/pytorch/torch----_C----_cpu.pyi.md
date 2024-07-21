# `.\pytorch\torch\_C\_cpu.pyi`

```
# 导入 torch 库中的 _bool 类型，该类型可能是用于定义布尔值的特定类型

# 在 torch/csrc/cpu/Module.cpp 文件中定义以下几个函数：

# 检查当前 CPU 是否支持 AVX2 指令集，返回一个 _bool 类型的结果
def _is_cpu_support_avx2() -> _bool: ...

# 检查当前 CPU 是否支持 AVX-512 指令集，返回一个 _bool 类型的结果
def _is_cpu_support_avx512() -> _bool: ...

# 检查当前 CPU 是否支持 AVX-512 VNNI 指令集，返回一个 _bool 类型的结果
def _is_cpu_support_avx512_vnni() -> _bool: ...

# 检查当前 CPU 是否支持 AMX Tile 指令集，返回一个 _bool 类型的结果
def _is_cpu_support_amx_tile() -> _bool: ...

# 在需要的时候初始化 AMX 相关的设置，并返回一个 _bool 类型的结果
def _init_amx() -> _bool: ...
```