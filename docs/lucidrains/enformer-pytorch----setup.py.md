# `.\lucidrains\enformer-pytorch\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'enformer-pytorch',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找并包含所有包
  include_package_data = True,  # 包含所有数据文件
  version = '0.8.8',  # 版本号
  license='MIT',  # 许可证
  description = 'Enformer - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/enformer-pytorch',  # URL
  keywords = [  # 关键词
    'artificial intelligence',
    'transformer',
    'gene-expression'
  ],
  install_requires=[  # 安装依赖
    'discrete-key-value-bottleneck-pytorch>=0.0.8',
    'einops>=0.3',
    'numpy',
    'torch>=1.6',
    'torchmetrics',
    'polars',
    'pyfaidx',
    'pyyaml',
    'transformers[torch]',
  ],
  classifiers=[  # 分类
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```