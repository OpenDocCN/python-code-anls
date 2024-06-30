# `D:\src\scipysrc\sympy\sympy\utilities\magic.py`

```
"""Functions that involve magic. """

# 定义一个函数，用于将 symbols 到 objects 的映射污染到全局命名空间中
def pollute(names, objects):
    # 导入 inspect 模块中的 currentframe 函数，用于获取当前帧
    from inspect import currentframe
    # 获取调用 pollute 函数的上一级和上上级帧（即调用 pollute 的函数的上一级帧）
    frame = currentframe().f_back.f_back

    try:
        # 遍历 names 和 objects 列表，将每个 name 对应的 obj 放入全局命名空间中
        for name, obj in zip(names, objects):
            frame.f_globals[name] = obj
    finally:
        # 清除 frame 变量，以打破在 inspect 文档中提到的循环依赖
        del frame  # break cyclic dependencies as stated in inspect docs
```