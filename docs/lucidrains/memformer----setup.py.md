# `.\lucidrains\memformer\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'memformer',  # 包的名称
  packages = find_packages(exclude=['examples']),  # 查找并包含除了 examples 之外的所有包
  version = '0.3.1',  # 版本号
  license='MIT',  # 许可证信息
  description = 'Memformer - Pytorch',  # 描述信息
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/memformer',  # 项目链接
  keywords = [  # 关键词列表
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'memory'
  ],
  install_requires=[  # 安装依赖列表
    'torch>=1.6',
    'einops>=0.3'
  ],
  classifiers=[  # 分类器列表
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```