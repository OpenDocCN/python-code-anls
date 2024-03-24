# `.\lucidrains\SAC-pytorch\setup.py`

```py
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'SAC-pytorch',
  # 查找所有包，不排除任何包
  packages = find_packages(exclude=[]),
  # 版本号
  version = '0.0.1',
  # 许可证类型
  license='MIT',
  # 描述
  description = 'Soft Actor Critic',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 项目链接
  url = 'https://github.com/lucidrains/SAC-pytorch',
  # 关键词列表
  keywords = [
    'artificial intelligence',
    'deep learning',
    'reinforcement learning',
    'soft actor critic'
  ],
  # 安装依赖项
  install_requires=[
    'beartype',
    'einops>=0.7.0',
    'einx[torch]>=0.1.3',
    'ema-pytorch',
    'pytorch-custom-utils>=0.0.18',
    'soft-moe-pytorch>=0.1.6',
    'torch>=2.0'
  ],
  # 分类器列表
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```