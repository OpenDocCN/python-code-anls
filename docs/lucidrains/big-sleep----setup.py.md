# `.\lucidrains\big-sleep\setup.py`

```py
# 导入 sys 模块
import sys
# 从 setuptools 模块中导入 setup 和 find_packages 函数
from setuptools import setup, find_packages

# 将 'big_sleep' 目录添加到 sys.path 的最前面
sys.path[0:0] = ['big_sleep']
# 从 version 模块中导入 __version__ 变量
from version import __version__

# 设置包的元数据
setup(
  # 包的名称
  name = 'big-sleep',
  # 查找并包含所有包
  packages = find_packages(),
  # 包含所有数据文件
  include_package_data = True,
  # 设置入口点，命令行脚本为 'dream'
  entry_points={
    'console_scripts': [
      'dream = big_sleep.cli:main',
    ],
  },
  # 版本号
  version = __version__,
  # 许可证
  license='MIT',
  # 描述
  description = 'Big Sleep',
  # 作者
  author = 'Ryan Murdock, Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/big-sleep',
  # 关键词
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'text to image',
    'generative adversarial networks'
  ],
  # 安装依赖
  install_requires=[
    'torch>=1.7.1',
    'einops>=0.3',
    'fire',
    'ftfy',
    'pytorch-pretrained-biggan',
    'regex',
    'torchvision>=0.8.2',
    'tqdm'
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