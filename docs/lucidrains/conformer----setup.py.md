# `.\lucidrains\conformer\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'conformer',
  # 查找并包含所有包
  packages = find_packages(),
  # 版本号
  version = '0.3.2',
  # 许可证信息
  license='MIT',
  # 描述信息
  description = 'The convolutional module from the Conformer paper',
  # 作者信息
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/conformer',
  # 关键词列表
  keywords = [
      'artificial intelligence',
      'deep learning',
      'transformers',
      'audio'
  ],
  # 安装依赖列表
  install_requires=[
      'einops>=0.6.1',
      'torch'
  ],
  # 分类信息列表
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)
```