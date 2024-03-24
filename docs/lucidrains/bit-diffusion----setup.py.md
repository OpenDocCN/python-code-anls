# `.\lucidrains\bit-diffusion\setup.py`

```py
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的信息
setup(
  # 包的名称
  name = 'bit-diffusion',
  # 查找所有包，不排除任何包
  packages = find_packages(exclude=[]),
  # 版本号
  version = '0.1.4',
  # 许可证
  license='MIT',
  # 描述
  description = 'Bit Diffusion - Pytorch',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 项目链接
  url = 'https://github.com/lucidrains/bit-diffusion',
  # 关键词
  keywords = [
    'artificial intelligence',
    'deep learning',
    'denoising diffusion'
  ],
  # 安装依赖
  install_requires=[
    'accelerate',
    'einops',
    'ema-pytorch',
    'pillow',
    'torch>=1.12.0',
    'torchvision',
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