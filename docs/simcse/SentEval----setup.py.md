# `.\SentEval\setup.py`

```py
# 导入 io 模块，用于处理文件输入输出
import io
# 从 setuptools 模块导入 setup 和 find_packages 函数
from setuptools import setup, find_packages

# 打开 README.md 文件，使用 utf-8 编码读取其内容并赋值给变量 readme
with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

# 设置项目的元数据和配置信息
setup(
    # 设置项目名称为 'SentEval'
    name='SentEval',
    # 设置项目版本号为 '0.1.0'
    version='0.1.0',
    # 设置项目的 URL 地址为 GitHub 上的仓库地址
    url='https://github.com/facebookresearch/SentEval',
    # 查找并包含所有的子包，但排除 'examples' 目录
    packages=find_packages(exclude=['examples']),
    # 设置项目使用的许可证为 'Attribution-NonCommercial 4.0 International'
    license='Attribution-NonCommercial 4.0 International',
    # 将 README 文件的内容作为长描述
    long_description=readme,
)
```