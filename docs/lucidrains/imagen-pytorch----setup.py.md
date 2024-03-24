# `.\lucidrains\imagen-pytorch\setup.py`

```py
# 导入设置工具和查找包工具
from setuptools import setup, find_packages
# 执行版本文件中的代码，将版本信息导入当前环境
exec(open('imagen_pytorch/version.py').read())

# 设置包的信息
setup(
  # 包名
  name = 'imagen-pytorch',
  # 查找所有包，不排除任何包
  packages = find_packages(exclude=[]),
  # 包含所有数据文件
  include_package_data = True,
  # 设置入口点，定义命令行脚本
  entry_points={
    'console_scripts': [
      'imagen_pytorch = imagen_pytorch.cli:main',
      'imagen = imagen_pytorch.cli:imagen'
    ],
  },
  # 版本号
  version = __version__,
  # 许可证
  license='MIT',
  # 描述
  description = 'Imagen - unprecedented photorealism × deep level of language understanding',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 项目链接
  url = 'https://github.com/lucidrains/imagen-pytorch',
  # 关键词
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'text-to-image',
    'denoising-diffusion'
  ],
  # 安装依赖
  install_requires=[
    'accelerate>=0.23.0',
    'beartype',
    'click',
    'datasets',
    'einops>=0.7.0',
    'ema-pytorch>=0.0.3',
    'fsspec',
    'kornia',
    'numpy',
    'packaging',
    'pillow',
    'pydantic>=2',
    'pytorch-warmup',
    'sentencepiece',
    'torch>=1.6',
    'torchvision',
    'transformers',
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