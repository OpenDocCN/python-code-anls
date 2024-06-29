# `.\numpy\doc\source\f2py\code\setup_skbuild.py`

```
# 导入 skbuild 库中的 setup 函数，用于配置和安装 Python 包
from skbuild import setup

# 调用 setup 函数，配置和安装一个 Python 包
setup(
    # 包的名称为 "fibby"
    name="fibby",
    # 包的版本号为 "0.0.1"
    version="0.0.1",
    # 包的描述信息为 "a minimal example package (fortran version)"
    description="a minimal example package (fortran version)",
    # 包的许可证为 MIT 许可证
    license="MIT",
    # 指定要包含的包列表，这里包含名为 'fibby' 的包
    packages=['fibby'],
    # 指定 Python 的最低版本要求为 3.7 及以上
    python_requires=">=3.7",
)
```