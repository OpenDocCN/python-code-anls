# `.\lucidrains\reformer-pytorch\setup.py`

```
# 导入设置安装包和查找包的模块
from setuptools import setup, find_packages

# 设置安装包的信息
setup(
  # 包的名称
  name = 'reformer_pytorch',
  # 查找包，排除 examples 和 pretraining 文件夹
  packages = find_packages(exclude=['examples', 'pretraining']),
  # 版本号
  version = '1.4.4',
  # 许可证
  license='MIT',
  # 描述
  description = 'Reformer, the Efficient Transformer, Pytorch',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/reformer-pytorch',
  # 关键词
  keywords = ['transformers', 'attention', 'artificial intelligence'],
  # 安装依赖
  install_requires=[
    'axial-positional-embedding>=0.1.0',
    'einops',
    'local-attention',
    'product-key-memory',
    'torch'
  ],
  # 分类
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```