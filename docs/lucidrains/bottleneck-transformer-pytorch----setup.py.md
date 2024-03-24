# `.\lucidrains\bottleneck-transformer-pytorch\setup.py`

```py
# 导入设置工具和查找包的模块
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'bottleneck-transformer-pytorch',  # 包的名称
  packages = find_packages(),  # 查找并包含所有包
  version = '0.1.4',  # 版本号
  license='MIT',  # 许可证类型
  description = 'Bottleneck Transformer - Pytorch',  # 包的描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者的电子邮件
  url = 'https://github.com/lucidrains/bottleneck-transformer-pytorch',  # 包的URL
  keywords = [  # 关键字列表
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'image classification',
    'vision'
  ],
  install_requires=[  # 安装所需的依赖项
    'einops>=0.3',
    'torch>=1.6'
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