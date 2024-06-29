# `.\numpy\numpy\f2py\_backends\__init__.py`

```
# 定义一个函数，用于根据给定的 name 参数生成不同的构建后端生成器
def f2py_build_generator(name):
    # 如果 name 参数为 "meson"
    if name == "meson":
        # 导入 MesonBackend 类并返回
        from ._meson import MesonBackend
        return MesonBackend
    # 如果 name 参数为 "distutils"
    elif name == "distutils":
        # 导入 DistutilsBackend 类并返回
        from ._distutils import DistutilsBackend
        return DistutilsBackend
    else:
        # 如果 name 参数不是预期的值，则引发 ValueError 异常
        raise ValueError(f"Unknown backend: {name}")
```