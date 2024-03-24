# `.\lucidrains\se3-transformer-pytorch\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'se3-transformer-pytorch',  # 包的名称
  packages = find_packages(),  # 查找所有包
  include_package_data = True,  # 包含所有数据文件
  version = '0.9.0',  # 版本号
  license='MIT',  # 许可证
  description = 'SE3 Transformer - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/se3-transformer-pytorch',  # 项目链接
  keywords = [  # 关键词列表
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'equivariance',
    'SE3'
  ],
  install_requires=[  # 安装依赖
    'einops>=0.3',
    'filelock',
    'numpy',
    'torch>=1.6'
  ],
  setup_requires=[  # 设置需要的依赖
    'pytest-runner',
  ],
  tests_require=[  # 测试需要的依赖
    'pytest',
    'lie_learn',
    'numpy',
  ],
  classifiers=[  # 分类器
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```