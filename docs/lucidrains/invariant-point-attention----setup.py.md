# `.\lucidrains\invariant-point-attention\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'invariant-point-attention',  # 包的名称
  packages = find_packages(),  # 查找所有包
  version = '0.2.2',  # 版本号
  license='MIT',  # 许可证
  description = 'Invariant Point Attention',  # 描述
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/invariant-point-attention',  # 项目链接
  keywords = [  # 关键词列表
    'artificial intelligence',
    'deep learning',
    'protein folding'
  ],
  install_requires=[  # 安装依赖
    'einops>=0.3',
    'torch>=1.7'
  ],
  setup_requires=[  # 设置依赖
    'pytest-runner',
  ],
  tests_require=[  # 测试依赖
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