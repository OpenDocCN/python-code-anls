# `.\lucidrains\remixer-pytorch\setup.py`

```py
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的元信息
setup(
  # 包名
  name = 'remixer-pytorch',
  # 查找所有包，不排除任何包
  packages = find_packages(exclude=[]),
  # 版本号
  version = '0.0.3',
  # 许可证
  license='MIT',
  # 描述
  description = 'Remixer - Pytorch',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/remixer-pytorch',
  # 关键词列表
  keywords = [
    'artificial intelligence',
    'transformer',
    'feedforward',
    'mlp-mixer'
  ],
  # 安装依赖
  install_requires=[
    'einops>=0.3',
    'torch>=1.6'
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