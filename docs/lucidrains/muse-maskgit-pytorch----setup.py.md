# `.\lucidrains\muse-maskgit-pytorch\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的信息
setup(
  # 包的名称
  name = 'muse-maskgit-pytorch',
  # 查找所有包，不排除任何包
  packages = find_packages(exclude=[]),
  # 版本号
  version = '0.3.5',
  # 许可证类型
  license='MIT',
  # 描述信息
  description = 'MUSE - Text-to-Image Generation via Masked Generative Transformers, in Pytorch',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 项目链接
  url = 'https://github.com/lucidrains/muse-maskgit-pytorch',
  # 关键词列表
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'text-to-image'
  ],
  # 安装依赖列表
  install_requires=[
    'accelerate',
    'beartype',
    'einops>=0.7',
    'ema-pytorch>=0.2.2',
    'memory-efficient-attention-pytorch>=0.1.4',
    'pillow',
    'sentencepiece',
    'torch>=1.6',
    'transformers',
    'torch>=1.6',
    'torchvision',
    'tqdm',
    'vector-quantize-pytorch>=1.11.8'
  ],
  # 分类列表
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```