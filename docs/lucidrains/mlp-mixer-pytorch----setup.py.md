# `.\lucidrains\mlp-mixer-pytorch\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'mlp-mixer-pytorch', # 包的名称
  packages = find_packages(exclude=[]), # 查找所有包
  version = '0.1.1', # 版本号
  license='MIT', # 许可证
  description = 'MLP Mixer - Pytorch', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  url = 'https://github.com/lucidrains/mlp-mixer-pytorch', # 项目链接
  keywords = [ # 关键词
    'artificial intelligence',
    'deep learning',
    'image recognition'
  ],
  install_requires=[ # 安装依赖
    'einops>=0.3',
    'torch>=1.6'
  ],
  classifiers=[ # 分类
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```