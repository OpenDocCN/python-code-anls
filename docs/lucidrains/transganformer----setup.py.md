# `.\lucidrains\transganformer\setup.py`

```
# 导入 sys 模块
import sys
# 从 setuptools 模块中导入 setup 和 find_packages 函数
from setuptools import setup, find_packages

# 将 'transganformer' 目录添加到 sys.path 的最前面
sys.path[0:0] = ['transganformer']
# 从 version 模块中导入 __version__ 变量
from version import __version__

# 设置包的元数据
setup(
  # 包名为 'transganformer'
  name = 'transganformer',
  # 查找所有包
  packages = find_packages(),
  # 设置入口点，命令行脚本为 'transganformer'
  entry_points={
    'console_scripts': [
      'transganformer = transganformer.cli:main',
    ],
  },
  # 设置版本号为导入的 __version__ 变量
  version = __version__,
  # 设置许可证为 MIT
  license='MIT',
  # 设置描述为 'TransGanFormer'
  description = 'TransGanFormer',
  # 设置作者为 'Phil Wang'
  author = 'Phil Wang',
  # 设置作者邮箱为 'lucidrains@gmail.com'
  author_email = 'lucidrains@gmail.com',
  # 设置项目 URL 为 'https://github.com/lucidrains/transganformer'
  url = 'https://github.com/lucidrains/transganformer',
  # 设置关键词列表
  keywords = [
    'artificial intelligence',
    'deep learning',
    'generative adversarial networks',
    'transformers',
    'attention-mechanism'
  ],
  # 设置依赖包列表
  install_requires=[
    'einops>=0.3',
    'fire',
    'kornia',
    'numpy',
    'pillow',
    'retry',
    'torch>=1.6',
    'torchvision',
    'tqdm'
  ],
  # 设置分类列表
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```