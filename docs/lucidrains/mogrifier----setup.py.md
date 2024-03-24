# `.\lucidrains\mogrifier\setup.py`

```py
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'mogrifier',
  # 查找并包含所有包
  packages = find_packages(),
  # 版本号
  version = '0.0.3',
  # 许可证信息
  license='MIT',
  # 描述信息
  description = 'Implementation of Mogrifier circuit from Deepmind',
  # 作者信息
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/mogrifier',
  # 关键词
  keywords = ['artificial intelligence', 'natural language processing'],
  # 安装依赖
  install_requires=[
      'torch'
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