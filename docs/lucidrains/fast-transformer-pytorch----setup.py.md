# `.\lucidrains\fast-transformer-pytorch\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的信息
setup(
  # 包名
  name = 'fast-transformer-pytorch',
  # 查找所有包
  packages = find_packages(),
  # 版本号
  version = '0.0.4',
  # 许可证
  license='MIT',
  # 描述
  description = 'Fast Transformer - Pytorch',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/fast-transformer-pytorch',
  # 关键词
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers'
  ],
  # 安装依赖
  install_requires=[
    'einops>=0.3',
    'rotary-embedding-torch',
    'torch>=1.6'
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