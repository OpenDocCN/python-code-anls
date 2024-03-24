# `.\lucidrains\metnet3-pytorch\setup.py`

```
# 导入设置安装包和查找包的函数
from setuptools import setup, find_packages

# 设置安装包的信息
setup(
  name = 'metnet3-pytorch',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找并包含所有包
  version = '0.0.12',  # 版本号
  license='MIT',  # 许可证
  description = 'MetNet 3 - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/metnet3-pytorch',  # 项目链接
  keywords = [  # 关键词列表
    'artificial intelligence',
    'deep learning',
    'vision transformers',
    'unet',
    'weather forecasting'
  ],
  install_requires=[  # 安装依赖
    'beartype',
    'einops>=0.7.0',
    'torch>=2.0',
  ],
  classifiers=[  # 分类器列表
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```