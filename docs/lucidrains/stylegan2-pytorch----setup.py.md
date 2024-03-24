# `.\lucidrains\stylegan2-pytorch\setup.py`

```
# 导入 sys 模块
import sys
# 从 setuptools 模块中导入 setup 和 find_packages 函数
from setuptools import setup, find_packages

# 将 stylegan2_pytorch 目录添加到 sys.path 中
sys.path[0:0] = ['stylegan2_pytorch']
# 从 version 模块中导入 __version__ 变量
from version import __version__

# 设置包的元数据
setup(
  # 包的名称
  name = 'stylegan2_pytorch',
  # 查找并包含所有包
  packages = find_packages(),
  # 设置入口点，命令行脚本为 stylegan2_pytorch
  entry_points={
      'console_scripts': [
          'stylegan2_pytorch = stylegan2_pytorch.cli:main',
      ],
  },
  # 设置版本号为导入的 __version__ 变量
  version = __version__,
  # 设置许可证为 GPLv3+
  license='GPLv3+',
  # 设置描述信息
  description = 'StyleGan2 in Pytorch',
  # 设置长描述内容类型为 markdown
  long_description_content_type = 'text/markdown',
  # 设置作者
  author = 'Phil Wang',
  # 设置作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 设置项目 URL
  url = 'https://github.com/lucidrains/stylegan2-pytorch',
  # 设置下载 URL
  download_url = 'https://github.com/lucidrains/stylegan2-pytorch/archive/v_036.tar.gz',
  # 设置关键词
  keywords = ['generative adversarial networks', 'artificial intelligence'],
  # 设置依赖的包
  install_requires=[
      'aim',
      'einops',
      'contrastive_learner>=0.1.0',
      'fire',
      'kornia>=0.5.4',
      'numpy',
      'retry',
      'tqdm',
      'torch',
      'torchvision',
      'pillow',
      'vector-quantize-pytorch==0.1.0'
  ],
  # 设置分类标签
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)
```