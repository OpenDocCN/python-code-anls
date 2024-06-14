# `.\setup.py`

```
# 导入所需模块
import io
from setuptools import setup, find_packages

# 打开并读取 README.md 文件内容，以 UTF-8 编码方式
with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

# 设置安装配置
setup(
    # 包名
    name='simcse',
    # 包含的包
    packages=['simcse'],
    # 版本号
    version='0.4',
    # 许可证信息
    license='MIT',
    # 包描述
    description='A sentence embedding tool based on SimCSE',
    # 作者信息
    author='Tianyu Gao, Xingcheng Yao, Danqi Chen',
    # 作者邮箱
    author_email='tianyug@cs.princeton.edu',
    # 项目 URL
    url='https://github.com/princeton-nlp/SimCSE',
    # 下载 URL
    download_url='https://github.com/princeton-nlp/SimCSE/archive/refs/tags/0.4.tar.gz',
    # 关键字列表
    keywords=['sentence', 'embedding', 'simcse', 'nlp'],
    # 安装所需的依赖包列表
    install_requires=[
        "tqdm",
        "scikit-learn",
        "scipy>=1.5.4,<1.6",
        "transformers",
        "torch",
        "numpy>=1.19.5,<1.20",
        "setuptools"
    ]
)
```