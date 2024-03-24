# `.\lucidrains\triton-transformer\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'triton-transformer', # 包的名称
  packages = find_packages(), # 查找所有包
  version = '0.1.1', # 版本号
  license='MIT', # 许可证
  description = 'Transformer in Triton', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  url = 'https://github.com/lucidrains/triton-transformer', # 项目链接
  keywords = [
    'artificial intelligence', # 关键词
    'attention mechanism', # 关键词
    'transformers' # 关键词
  ],
  install_requires=[
    'einops', # 安装所需的依赖包
    'torch>=1.6', # 安装所需的依赖包
    'triton==1.0.1.dev20210924' # 安装所需的依赖包
  ],
  classifiers=[
    'Development Status :: 4 - Beta', # 分类器
    'Intended Audience :: Developers', # 分类器
    'Topic :: Scientific/Engineering :: Artificial Intelligence', # 分类器
    'License :: OSI Approved :: MIT License', # 分类器
    'Programming Language :: Python :: 3.6', # 分类器
  ],
)
```