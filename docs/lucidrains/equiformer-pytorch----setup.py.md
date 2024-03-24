# `.\lucidrains\equiformer-pytorch\setup.py`

```py
# 导入设置安装和查找包的函数
from setuptools import setup, find_packages

# 执行版本文件中的代码，将版本信息导入当前环境
exec(open('equiformer_pytorch/version.py').read())

# 设置包的元数据
setup(
  name = 'equiformer-pytorch',  # 包名
  packages = find_packages(exclude=[]),  # 查找包
  version = __version__,  # 版本号
  license='MIT',  # 许可证
  description = 'Equiformer - SE3/E3 Graph Attention Transformer for Molecules and Proteins',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/equiformer-pytorch',  # URL
  keywords = [  # 关键词
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'equivariance',
    'molecules',
    'proteins'
  ],
  install_requires=[  # 安装依赖
    'beartype',
    'einops>=0.6',
    'einx',
    'filelock',
    'opt-einsum',
    'taylor-series-linear-attention>=0.1.4',
    'torch>=1.6',
  ],
  setup_requires=[  # 设置需要的依赖
    'pytest-runner',
  ],
  tests_require=[  # 测试需要的依赖
    'pytest'
  ],
  include_package_data = True,  # 包含包数据
  classifiers=[  # 分类
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```