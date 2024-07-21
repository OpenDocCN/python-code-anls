# `.\pytorch\torch\_C\_nvtx.pyi`

```
# mypy: allow-untyped-defs
# 声明在 torch/csrc/cuda/shared/nvtx.cpp 中的函数，这些函数通常用于与 NVIDIA Tools Extension (NVTX) 进行交互
def rangePushA(message: str) -> int:
    # 开始一个命名的范围，用给定的消息字符串
    ...

def rangePop() -> int:
    # 结束当前范围
    ...

def rangeStartA(message: str) -> int:
    # 开始一个命名的范围，用给定的消息字符串
    ...

def rangeEnd(int) -> None:
    # 结束当前范围
    ...

def markA(message: str) -> None:
    # 在当前范围中做一个标记，用给定的消息字符串
    ...
```