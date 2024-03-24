# `.\lucidrains\memory-transformer-xl\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的信息
setup(
  # 包的名称
  name = 'memory-transformer-xl',
  # 查找包，排除 examples 文件夹
  packages = find_packages(exclude=['examples']),
  # 版本号
  version = '0.1.0',
  # 许可证
  license='MIT',
  # 描述
  description = 'Memory Transformer-XL, a variant of Transformer-XL that uses linear attention update long term memory',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/memory-transformer-xl',
  # 关键词
  keywords = ['attention mechanism', 'artificial intelligence', 'transformer', 'deep learning'],
  # 安装依赖
  install_requires=[
      'torch',
      'mogrifier'
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