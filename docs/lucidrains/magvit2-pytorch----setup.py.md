# `.\lucidrains\magvit2-pytorch\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 执行版本文件中的代码，将版本信息导入当前环境
exec(open('magvit2_pytorch/version.py').read())

# 设置包的元数据
setup(
  # 包名
  name = 'magvit2-pytorch',
  # 查找所有包
  packages = find_packages(),
  # 版本号
  version = __version__,
  # 许可证
  license='MIT',
  # 描述
  description = 'MagViT2 - Pytorch',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/magvit2-pytorch',
  # 关键词
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformer',
    'attention mechanisms',
    'generative video model'
  ],
  # 安装依赖
  install_requires=[
    'accelerate>=0.24.0',
    'beartype',
    'einops>=0.7.0',
    'ema-pytorch>=0.2.4',
    'pytorch-warmup',
    'gateloop-transformer>=0.2.2',
    'kornia',
    'opencv-python',
    'pillow',
    'pytorch-custom-utils>=0.0.9',
    'numpy',
    'vector-quantize-pytorch>=1.11.8',
    'taylor-series-linear-attention>=0.1.5',
    'torch',
    'torchvision',
    'x-transformers'
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