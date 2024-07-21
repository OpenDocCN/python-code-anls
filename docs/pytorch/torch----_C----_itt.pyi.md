# `.\pytorch\torch\_C\_itt.pyi`

```py
# 在 torch/csrc/itt.cpp 中定义的函数 is_available()，声明不返回任何值
def is_available() -> None: ...

# 在 torch/csrc/itt.cpp 中定义的函数 rangePush(message: str)，声明不返回任何值
def rangePush(message: str) -> None: ...

# 在 torch/csrc/itt.cpp 中定义的函数 rangePop()，声明不返回任何值
def rangePop() -> None: ...

# 在 torch/csrc/itt.cpp 中定义的函数 mark(message: str)，声明不返回任何值
def mark(message: str) -> None: ...
```