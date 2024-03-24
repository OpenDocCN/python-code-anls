# `.\lucidrains\DALLE2-pytorch\setup.py`

```
# 导入所需的模块和函数
from setuptools import setup, find_packages
# 执行版本文件中的代码，将版本信息导入当前环境
exec(open('dalle2_pytorch/version.py').read())

# 设置包的信息和配置
setup(
  # 包名
  name = 'dalle2-pytorch',
  # 查找所有包，不排除任何包
  packages = find_packages(exclude=[]),
  # 包含所有数据文件
  include_package_data = True,
  # 设置命令行入口点
  entry_points={
    'console_scripts': [
      'dalle2_pytorch = dalle2_pytorch.cli:main',
      'dream = dalle2_pytorch.cli:dream'
    ],
  },
  # 版本号
  version = __version__,
  # 许可证信息
  license='MIT',
  # 描述信息
  description = 'DALL-E 2',
  # 作者信息
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 项目链接
  url = 'https://github.com/lucidrains/dalle2-pytorch',
  # 关键词
  keywords = [
    'artificial intelligence',
    'deep learning',
    'text to image'
  ],
  # 安装依赖
  install_requires=[
    'accelerate',
    'click',
    'open-clip-torch>=2.0.0,<3.0.0',
    'clip-anytorch>=2.5.2',
    'coca-pytorch>=0.0.5',
    'ema-pytorch>=0.0.7',
    'einops>=0.7.0',
    'embedding-reader',
    'kornia>=0.5.4',
    'numpy',
    'packaging',
    'pillow',
    'pydantic>=2',
    'pytorch-warmup',
    'resize-right>=0.0.2',
    'rotary-embedding-torch',
    'torch>=1.10',
    'torchvision',
    'tqdm',
    'vector-quantize-pytorch',
    'x-clip>=0.4.4',
    'webdataset>=0.2.5',
    'fsspec>=2022.1.0',
    'torchmetrics[image]>=0.8.0'
  ],
  # 分类信息
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```