# `.\lucidrains\phenaki-pytorch\setup.py`

```py
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的信息
setup(
  name = 'phenaki-pytorch',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.4.2',  # 版本号
  license='MIT',  # 许可证
  description = 'Phenaki - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/phenaki-pytorch',  # URL
  keywords = [  # 关键词列表
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanisms',
    'text-to-video'
  ],
  install_requires = [  # 安装依赖列表
    'accelerate',
    'beartype',
    'einops>=0.7',
    'ema-pytorch>=0.2.2',
    'opencv-python',
    'pillow',
    'numpy',
    'sentencepiece',
    'torch>=1.6',
    'torchtyping',
    'torchvision',
    'transformers>=4.20.1',
    'tqdm',
    'vector-quantize-pytorch>=1.11.8'
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