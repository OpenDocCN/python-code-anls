# `.\lucidrains\deep-daze\setup.py`

```py
# 导入 sys 模块
import sys
# 从 setuptools 模块中导入 setup 和 find_packages 函数
from setuptools import setup, find_packages

# 将 deep_daze 目录添加到 sys.path 中
sys.path[0:0] = ['deep_daze']
# 从 version 模块中导入 __version__ 变量
from version import __version__

# 设置包的元数据和配置信息
setup(
  # 包的名称
  name = 'deep-daze',
  # 查找并包含所有包
  packages = find_packages(),
  # 包含所有数据文件
  include_package_data = True,
  # 设置入口点，命令行脚本
  entry_points={
    'console_scripts': [
      'imagine = deep_daze.cli:main',
    ],
  },
  # 版本号
  version = __version__,
  # 许可证
  license='MIT',
  # 描述
  description = 'Deep Daze',
  # 作者
  author = 'Ryan Murdock, Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/deep-daze',
  # 关键词
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'implicit neural representations',
    'text to image'
  ],
  # 安装依赖
  install_requires=[
    'einops>=0.3',
    'fire',
    'ftfy',
    'imageio>=2.9.0',
    'siren-pytorch>=0.0.8',
    'torch>=1.10',
    'torch_optimizer',
    'torchvision>=0.8.2',
    'tqdm',
    'regex'
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