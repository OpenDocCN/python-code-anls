# `.\segment-anything\setup.py`

```
# 导入必要的模块
from setuptools import find_packages, setup

# 设置包的基本信息
setup(
    # 包的名称
    name="segment_anything",
    # 版本号
    version="1.0",
    # 安装依赖为空列表
    install_requires=[],
    # 查找所有包，排除 notebooks 目录
    packages=find_packages(exclude="notebooks"),
    # 额外的依赖
    extras_require={
        # all 额外依赖
        "all": ["matplotlib", "pycocotools", "opencv-python", "onnx", "onnxruntime"],
        # dev 额外依赖
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)
```