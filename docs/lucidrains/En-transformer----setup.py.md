# `.\lucidrains\En-transformer\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'En-transformer',  # 包的名称
  packages = find_packages(),  # 查找所有包
  version = '1.6.5',  # 版本号
  license='MIT',  # 许可证
  description = 'E(n)-Equivariant Transformer',  # 描述
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/En-transformer',  # 项目链接
  keywords = [  # 关键词列表
    'artificial intelligence',
    'deep learning',
    'equivariance',
    'transformer'
  ],
  install_requires=[  # 安装依赖
    'einops>=0.3',
    'einx',
    'taylor-series-linear-attention>=0.1.4',
    'torch>=1.7'
  ],
  setup_requires=[  # 设置需要的依赖
    'pytest-runner',
  ],
  tests_require=[  # 测试需要的依赖
    'pytest'
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