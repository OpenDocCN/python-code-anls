# `bitsandbytes\setup.py`

```py
# 导入所需的模块
import glob
import os

from setuptools import Extension, find_packages, setup

# 查找指定路径下的动态链接库文件
libs = list(glob.glob("./bitsandbytes/libbitsandbytes*.so"))
libs += list(glob.glob("./bitsandbytes/libbitsandbytes*.dll"))
# 获取文件路径中的文件名
libs = [os.path.basename(p) for p in libs]
# 打印找到的动态链接库文件名
print("libs:", libs)

# 读取文件内容的函数
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# 设置包的信息和依赖
setup(
    name="bitsandbytes",
    version="0.43.0.dev0",
    author="Tim Dettmers",
    author_email="dettmers@cs.washington.edu",
    description="k-bit optimizers and matrix multiplication routines.",
    license="MIT",
    keywords="gpu optimizers optimization 8-bit quantization compression",
    url="https://github.com/TimDettmers/bitsandbytes",
    packages=find_packages(),
    # 包含的数据文件
    package_data={"": libs},
    install_requires=['torch', 'numpy'],
    extras_require={
        'benchmark': ['pandas', 'matplotlib'],
        'test': ['scipy'],
    },
    # 读取长描述文件
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    # 伪装一个空的本地扩展模块，以便正确标记平台标签
    ext_modules=[Extension("bitsandbytes", sources=[], language="c")],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
```